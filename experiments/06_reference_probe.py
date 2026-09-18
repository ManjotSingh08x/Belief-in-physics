"""Why the external Mess-3 belief probe scores high and ours scores low.

Their script builds one probe row per HMM tick from **all four block outputs at
all ten physics positions in that tick** (4 * 128 * 10 = 5,120 features), then
fits and scores ridge on the same rows. Our `02_probe.py` reads a **single
layer at a single token position** (128 features) and averages the score over
every position in the tick, including the positions that have barely seen the
current letter's physical response.

Those are different measurements, so this experiment runs both feature
constructions on our own models against an architecture-matched
random-initialisation control.

Nothing here is scored on the rows it was fitted on. Every probe reports a
validation score, used for any selection, and a test score on sequences no fit
and no choice has touched. The external script's in-sample number is what this
experiment exists to replace, not to reproduce.

Two targets are scored on the same rows and the same fit: the exact predictive
belief, and the system's own physical metric (`omega` for the pendulum,
`(dx_dt, dy_dt)` for predator-prey, and so on). The metric is what the model
plainly needs for its actual job of predicting the next observation, so it is
the reference point for how much the belief number means.

A raw-observation-token window is scored on the same rows under the same split,
because a residual-stream number is only interpretable against how much of it a
window of recent observations already carries.

`N_EVAL` must keep rows-per-feature well above one. Their run had 15; the
earlier version of this file had 1.9, which is why it returned a negative R2.

Ridge is solved from accumulated moments rather than a dense design matrix, so
131k rows by 5,120 features costs a 5,120^2 Gram instead of 5 GB of float64.

Run:  uv run python experiments/06_reference_probe.py
Env:  OUTPUT_DIR, CONFIGS, N_EVAL, RIDGE_ALPHAS, N_DEPTHS, WINDOWS, TAG, REPORT_NAME.
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

from models.analysis import residual_streams_batched, tick_features
from models.probe_extra import sparse_token_window_features
from models.train import pick_device
from models.transformer import ModelConfig, TinyTransformer
from physics.messk_configs import MESSK_CONFIGS, make_process

N_EVAL = int(os.environ.get("N_EVAL", 6_144))  # 15.4 rows per feature, matching theirs
TRAIN_FRAC = 0.6
VALIDATION_FRAC = 0.2
WINDOWS = tuple(int(w) for w in os.environ.get("WINDOWS", "10,20,40,80").split(","))
TOKEN_ALPHAS = (1.0, 10.0, 100.0, 1_000.0)
EVAL_SEED = 20_260_829
SPLIT_SEED = 0
RIDGE_ALPHAS = tuple(float(a) for a in os.environ.get("RIDGE_ALPHAS", "1,10,100,1000,10000").split(","))
# Four blocks deep, so the readout is the same shape whatever the model's depth.
N_DEPTHS = int(os.environ.get("N_DEPTHS", 4))
MOMENT_CHUNK = 4_096

OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "experiments/outputs-messk"))
CONFIGS = os.environ.get("CONFIGS", ",".join(MESSK_CONFIGS)).split(",")
ANALYSIS_COMMIT = os.environ.get("ANALYSIS_COMMIT", "unknown")
# Runs of the same system at different sizes share one directory, so the tag
# selects which set of checkpoints and which training report to read.
TAG = os.environ.get("TAG", "")
REPORT_NAME = os.environ.get("REPORT_NAME", "messk_01_training.json")


def _moments(features: np.ndarray, targets: np.ndarray, rows: np.ndarray) -> dict:
    """Sufficient statistics for a standardised ridge fit, accumulated in chunks."""
    width, n_out = features.shape[1], targets.shape[1]
    acc = {
        "n": 0,
        "sx": np.zeros(width),
        "sxx": np.zeros(width),
        "gram": np.zeros((width, width)),
        "xty": np.zeros((width, n_out)),
        "sy": np.zeros(n_out),
        "syy": np.zeros(n_out),
    }
    for start in range(0, len(rows), MOMENT_CHUNK):
        index = np.sort(rows[start : start + MOMENT_CHUNK])
        x = features[index].astype(np.float64)
        y = targets[index].astype(np.float64)
        acc["n"] += len(index)
        acc["sx"] += x.sum(0)
        acc["sxx"] += (x * x).sum(0)
        acc["gram"] += x.T @ x
        acc["xty"] += x.T @ y
        acc["sy"] += y.sum(0)
        acc["syy"] += (y * y).sum(0)
    return acc


def _standardise(acc: dict, mean: np.ndarray, scale: np.ndarray) -> tuple:
    """Z^T Z, Z^T y and the column sums of Z, for Z = (X - mean) / scale."""
    inverse = 1.0 / scale
    centred = (
        acc["gram"]
        - np.outer(acc["sx"], mean)
        - np.outer(mean, acc["sx"])
        + acc["n"] * np.outer(mean, mean)
    )
    gram = centred * np.outer(inverse, inverse)
    cross = (acc["xty"] - np.outer(mean, acc["sy"])) * inverse[:, None]
    return gram, cross, (acc["sx"] - acc["n"] * mean) * inverse


def _r2(acc: dict, gram, cross, z_sum, weights, y_bar) -> np.ndarray:
    """Per-output R2 for the prediction `y_bar + weights^T z`.

    Per output rather than averaged, so one fit can carry several target blocks
    and each is scored over its own columns.
    """
    residual = (
        acc["syy"]
        - 2 * y_bar * acc["sy"]
        + acc["n"] * y_bar**2
        - 2 * np.einsum("ij,ij->j", weights, cross)
        + 2 * y_bar * (z_sum @ weights)
        + np.einsum("ij,ik,kj->j", weights, gram, weights)
    )
    return 1.0 - residual / (acc["syy"] - acc["sy"] ** 2 / acc["n"])


def ridge_scores(features, targets, groups, split, alphas=RIDGE_ALPHAS) -> dict:
    """Standardised ridge, penalty chosen on validation, scored once on test.

    All target blocks share one fit, because the design matrix and its Gram are
    the expensive part and do not depend on the target. Each block in `groups`
    is scored over its own columns and picks its own penalty, so the belief
    probe's regularisation is not dragged around by the physical metric's.

    A fixed alpha is not safe across model sizes: widening the residual stream
    multiplies the feature count, so the same penalty that suited 5,120 columns
    under-regularises 10,240 and the test score reports the fit rather than the
    representation.
    """
    train_rows, validation_rows, test_rows = split
    train = _moments(features, targets, train_rows)
    mean = train["sx"] / train["n"]
    variance = np.maximum(train["sxx"] / train["n"] - mean**2, 0.0)
    scale = np.where(variance > 1e-12, np.sqrt(variance), 1.0)
    gram, cross, z_sum = _standardise(train, mean, scale)
    y_bar = train["sy"] / train["n"]

    held = {
        "validation_r2": _moments(features, targets, validation_rows),
        "test_r2": _moments(features, targets, test_rows),
    }
    standardised = {k: _standardise(v, mean, scale) for k, v in held.items()}

    shared = {
        "feature_dim": int(features.shape[1]),
        "n_train_rows": int(train["n"]),
        "rows_per_feature": float(train["n"] / features.shape[1]),
    }
    best: dict[str, dict] = {}
    eye = np.eye(len(gram))
    for alpha in alphas:
        weights = np.linalg.solve(gram + alpha * eye, cross)
        scored = {
            label: _r2(held[label], *standardised[label], weights, y_bar) for label in held
        }
        for group, columns in groups.items():
            candidate = {
                label: float(np.mean(values[columns])) for label, values in scored.items()
            } | {"alpha": alpha}
            if group not in best or candidate["validation_r2"] > best[group]["validation_r2"]:
                best[group] = candidate | shared
    return best


def token_window_scores(tokens, targets, groups, tick_rows, proc, split) -> dict:
    """The myopic control, scored on the same rows under the same protocol.

    Without this at tick level the residual-stream number cannot be read: the
    question is always how much of it a window of raw observations already has.
    """
    widest = max(w for w in WINDOWS if w <= tokens.shape[1])
    kept = sparse_token_window_features(tokens, proc.n_obs, widest, proc.n_steps)[tick_rows]
    train_rows, validation_rows, test_rows = split
    best: dict[str, dict] = {}
    for window in (w for w in WINDOWS if w <= tokens.shape[1]):
        columns = np.concatenate(
            [
                np.arange(window * proc.n_obs),
                np.arange(widest * proc.n_obs, widest * proc.n_obs + proc.n_steps),
            ]
        )
        design = kept[:, columns]
        for alpha in TOKEN_ALPHAS:
            probe = Ridge(alpha=alpha, solver="lsqr", tol=1e-6).fit(
                design[train_rows], targets[train_rows]
            )
            scored = {
                label: r2_score(
                    targets[rows], probe.predict(design[rows]), multioutput="raw_values"
                )
                for label, rows in (
                    ("validation_r2", validation_rows),
                    ("test_r2", test_rows),
                )
            }
            for group, block in groups.items():
                candidate = {
                    label: float(np.mean(values[block])) for label, values in scored.items()
                } | {"window": window, "alpha": alpha, "feature_dim": int(design.shape[1])}
                if group not in best or candidate["validation_r2"] > best[group]["validation_r2"]:
                    best[group] = candidate
    return best


def _load(path: Path, config: ModelConfig, device: str) -> TinyTransformer:
    model = TinyTransformer(config)
    model.load_state_dict(torch.load(path, map_location=device))
    return model.to(device).eval()


def tick_targets(batch: dict, proc) -> tuple[np.ndarray, dict[str, slice]]:
    """Probe targets for one row per tick, and the column block for each.

    Both are read at the tick's final physics step, which is the step the
    `last_position` readout and the raw-token control also see. The belief is
    constant inside a tick, so only the metric is actually sensitive to that
    choice, and taking its final value keeps every target on one clock.
    """
    last = np.arange(proc.n_steps - 1, proc.seq_len, proc.n_steps)
    belief = batch["beliefs"][:, last, :]
    metric = batch["metric"][:, last, :]
    stacked = np.concatenate([belief, metric], axis=-1).reshape(-1, belief.shape[-1] + metric.shape[-1])
    return stacked.astype(np.float64), {
        "belief": slice(0, belief.shape[-1]),
        "metric": slice(belief.shape[-1], stacked.shape[-1]),
    }


def _probe(path, config, batch, proc, targets, groups, split, device) -> dict:
    model = _load(path, config, device)
    # The top N_DEPTHS blocks. Index 0 is the embedding, so block i is i+1.
    depths = list(range(config.n_layers + 1 - N_DEPTHS, config.n_layers + 1))
    streams = residual_streams_batched(model, batch["tokens"], device, depths=depths)
    scores = {}
    for name, whole_tick in (("whole_tick", True), ("last_position", False)):
        features = tick_features(streams, proc.n_steps, whole_tick)
        scores[name] = ridge_scores(features, targets, groups, split)
        del features
        gc.collect()
    del model, streams
    gc.collect()
    return scores


def _demo() -> None:
    streams = [np.arange(2 * 20 * 3, dtype=np.float32).reshape(2, 20, 3) for _ in range(4)]
    assert tick_features(streams, 10, whole_tick=True).shape == (4, 120)
    assert tick_features(streams, 10, whole_tick=False).shape == (4, 12)
    assert tick_features(streams[:2], 10, whole_tick=True).shape == (4, 60)
    rng = np.random.default_rng(0)
    x = rng.normal(size=(400, 5))
    y = x @ rng.normal(size=(5, 2)) + 0.1 * rng.normal(size=(400, 2))
    split = (np.arange(240), np.arange(240, 320), np.arange(320, 400))
    blocks = {"first": slice(0, 1), "second": slice(1, 2)}
    scored = ridge_scores(x, y, blocks, split, alphas=(1e-6,))
    assert set(scored) == set(blocks), scored
    for block in blocks:
        assert scored[block]["test_r2"] > 0.95, (block, scored)
        assert scored[block]["validation_r2"] > 0.95, (block, scored)


def main() -> None:
    _demo()
    device = pick_device()
    training = json.loads((OUTPUT_DIR / REPORT_NAME).read_text())
    results = {}
    print(f"device={device}  n_eval={N_EVAL}  depths={N_DEPTHS}  tag={TAG!r}  configs={CONFIGS}", flush=True)

    for name in CONFIGS:
        started = time.perf_counter()
        proc = make_process(name)
        batch = proc.sample_batch(np.random.default_rng(EVAL_SEED), N_EVAL)
        order = np.random.default_rng(SPLIT_SEED).permutation(N_EVAL)
        ticks = np.arange(proc.m)
        rows = lambda seqs: (seqs[:, None] * proc.m + ticks[None, :]).reshape(-1)  # noqa: E731
        train_end = int(TRAIN_FRAC * N_EVAL)
        validation_end = train_end + int(VALIDATION_FRAC * N_EVAL)
        split = (
            rows(order[:train_end]),
            rows(order[train_end:validation_end]),
            rows(order[validation_end:]),
        )
        targets, groups = tick_targets(batch, proc)
        # last position of each tick, in the same sequence-major order as `targets`
        tick_rows = (
            np.arange(N_EVAL)[:, None] * proc.seq_len
            + (ticks[None, :] * proc.n_steps + proc.n_steps - 1)
        ).reshape(-1)

        config = ModelConfig(**training[name]["model"])
        entry = {
            "protocol": "one row per tick at its final step; belief probabilities and physical metric",
            "metric_names": list(proc.system.metric_names),
            "ridge_alphas": list(RIDGE_ALPHAS),
            "n_depths": N_DEPTHS,
            "tag": TAG,
            "n_eval_sequences": N_EVAL,
            "model_seed": training[name]["seed"],
            "evaluation_seed": EVAL_SEED,
            "analysis_commit": ANALYSIS_COMMIT,
            "device": device,
            "trained": _probe(
                OUTPUT_DIR / f"{name}{TAG}_trained.pt", config, batch, proc, targets, groups, split, device
            ),
            "random_init": _probe(
                OUTPUT_DIR / f"{name}{TAG}_random_init.pt", config, batch, proc, targets, groups, split, device
            ),
            "raw_tokens": token_window_scores(batch["tokens"], targets, groups, tick_rows, proc, split),
        }
        entry["wall_seconds"] = time.perf_counter() - started
        results[name] = entry

        print(f"\n=== {name} ===  metric={list(proc.system.metric_names)}", flush=True)
        for group in groups:
            for features in ("whole_tick", "last_position"):
                trained = entry["trained"][features][group]
                random = entry["random_init"][features][group]
                print(
                    f"  {group:<7} {features:<14} dim={trained['feature_dim']:>5} "
                    f"rows/feat={trained['rows_per_feature']:>6.2f} | "
                    f"trained val {trained['validation_r2']:+.3f} test {trained['test_r2']:+.3f} | "
                    f"random val {random['validation_r2']:+.3f} test {random['test_r2']:+.3f}",
                    flush=True,
                )
            raw = entry["raw_tokens"][group]
            print(
                f"  {group:<7} {'raw_tokens':<14} dim={raw['feature_dim']:>5} "
                f"W={raw['window']:<3} alpha={raw['alpha']:<7g} | "
                f"val {raw['validation_r2']:+.3f} test {raw['test_r2']:+.3f}",
                flush=True,
            )

    path = OUTPUT_DIR / f"messk_06_reference_probe{TAG}.json"
    path.write_text(json.dumps(results, indent=2, default=float))
    print(f"\nwrote {path}", flush=True)


if __name__ == "__main__":
    main()
