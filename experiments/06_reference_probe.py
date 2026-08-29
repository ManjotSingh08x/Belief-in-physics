"""Why the external Mess-3 belief probe scores high and ours scores low.

Their script builds one probe row per HMM tick from **all four block outputs at
all ten physics positions in that tick** (4 * 128 * 10 = 5,120 features), then
fits and scores ridge on the same rows. Our `02_probe.py` reads a **single
layer at a single token position** (128 features) and averages the score over
every position in the tick, including the positions that have barely seen the
current letter's physical response.

Those are different measurements, so this experiment runs both feature
constructions on our own models, each scored in-sample and on untouched test
sequences, against an architecture-matched random-initialisation control. The
four numbers separate the causes:

* random-init in-sample      -> what the fit buys with no learning at all
* random-init held out       -> what the last tick of tokens carries by itself
* trained in-sample          -> the number their protocol would print for us
* trained held out           -> the honest number

`N_EVAL` must keep rows-per-feature well above one. Their run had 15; the
earlier version of this file had 1.9, which is why it returned a negative R2.

Ridge is solved from accumulated moments rather than a dense design matrix, so
131k rows by 5,120 features costs a 5,120^2 Gram instead of 5 GB of float64.

Run:  uv run python experiments/06_reference_probe.py
Env:  OUTPUT_DIR, CONFIGS, N_EVAL, RIDGE_ALPHA, ANALYSIS_COMMIT.
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

from models.analysis import residual_streams_batched
from models.train import pick_device
from models.transformer import ModelConfig, TinyTransformer
from physics.messk_configs import MESSK_CONFIGS, make_process

N_EVAL = int(os.environ.get("N_EVAL", 6_144))  # 15.4 rows per feature, matching theirs
TRAIN_FRAC = 0.8
EVAL_SEED = 20_260_829
SPLIT_SEED = 0
RIDGE_ALPHA = float(os.environ.get("RIDGE_ALPHA", 1.0))
MOMENT_CHUNK = 4_096

OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "experiments/outputs-messk"))
CONFIGS = os.environ.get("CONFIGS", ",".join(MESSK_CONFIGS)).split(",")
ANALYSIS_COMMIT = os.environ.get("ANALYSIS_COMMIT", "unknown")


def tick_features(streams: list[np.ndarray], steps_per_tick: int, whole_tick: bool) -> np.ndarray:
    """One row per HMM tick from the four block outputs.

    `whole_tick` reproduces the external construction: every physics position in
    the tick, concatenated. Otherwise only the tick's final position, which is
    the same information a per-position probe sees at its most informative step.
    """
    if len(streams) != 5:
        raise ValueError(f"expected embedding plus four block streams, got {len(streams)}")
    blocks = np.concatenate(streams[1:], axis=-1)
    n, length, width = blocks.shape
    if length % steps_per_tick:
        raise ValueError(f"sequence length {length} is not divisible by {steps_per_tick}")
    per_tick = blocks.reshape(n, length // steps_per_tick, steps_per_tick, width)
    chosen = per_tick if whole_tick else per_tick[:, :, -1:, :]
    return chosen.reshape(n * (length // steps_per_tick), -1)


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


def _r2(acc: dict, gram, cross, z_sum, weights, y_bar) -> float:
    """sklearn's uniform-average R2 for the prediction `y_bar + weights^T z`."""
    residual = (
        acc["syy"]
        - 2 * y_bar * acc["sy"]
        + acc["n"] * y_bar**2
        - 2 * np.einsum("ij,ij->j", weights, cross)
        + 2 * y_bar * (z_sum @ weights)
        + np.einsum("ij,ik,kj->j", weights, gram, weights)
    )
    total = acc["syy"] - acc["sy"] ** 2 / acc["n"]
    return float(np.mean(1.0 - residual / total))


def ridge_scores(features, targets, train_rows, test_rows, alpha=RIDGE_ALPHA) -> dict:
    """Standardised ridge, scored on the fitting rows and on held-out rows."""
    train = _moments(features, targets, train_rows)
    mean = train["sx"] / train["n"]
    variance = np.maximum(train["sxx"] / train["n"] - mean**2, 0.0)
    scale = np.where(variance > 1e-12, np.sqrt(variance), 1.0)
    gram, cross, z_sum = _standardise(train, mean, scale)
    y_bar = train["sy"] / train["n"]
    weights = np.linalg.solve(gram + alpha * np.eye(len(gram)), cross)

    test = _moments(features, targets, test_rows)
    gram_test, cross_test, z_sum_test = _standardise(test, mean, scale)
    return {
        "in_sample_r2": _r2(train, gram, cross, z_sum, weights, y_bar),
        "test_r2": _r2(test, gram_test, cross_test, z_sum_test, weights, y_bar),
        "feature_dim": int(features.shape[1]),
        "n_train_rows": int(train["n"]),
        "n_test_rows": int(test["n"]),
        "rows_per_feature": float(train["n"] / features.shape[1]),
    }


def _load(path: Path, config: ModelConfig, device: str) -> TinyTransformer:
    model = TinyTransformer(config)
    model.load_state_dict(torch.load(path, map_location=device))
    return model.to(device).eval()


def _probe(path, config, batch, proc, split, device) -> dict:
    model = _load(path, config, device)
    streams = residual_streams_batched(model, batch["tokens"], device)
    targets = batch["beliefs"][:, :: proc.n_steps, :].reshape(-1, batch["beliefs"].shape[-1])
    scores = {
        name: ridge_scores(
            tick_features(streams, proc.n_steps, whole_tick), targets, *split
        )
        for name, whole_tick in (("whole_tick", True), ("last_position", False))
    }
    del model, streams
    gc.collect()
    return scores


def _demo() -> None:
    streams = [np.arange(2 * 20 * 3, dtype=np.float32).reshape(2, 20, 3) for _ in range(5)]
    assert tick_features(streams, 10, whole_tick=True).shape == (4, 120)
    assert tick_features(streams, 10, whole_tick=False).shape == (4, 12)
    rng = np.random.default_rng(0)
    x = rng.normal(size=(400, 5))
    y = x @ rng.normal(size=(5, 2)) + 0.1 * rng.normal(size=(400, 2))
    scored = ridge_scores(x, y, np.arange(320), np.arange(320, 400), alpha=1e-6)
    assert scored["test_r2"] > 0.95, scored


def main() -> None:
    _demo()
    device = pick_device()
    training = json.loads((OUTPUT_DIR / "messk_01_training.json").read_text())
    results = {}
    print(f"device={device}  n_eval={N_EVAL}  alpha={RIDGE_ALPHA}  configs={CONFIGS}", flush=True)

    for name in CONFIGS:
        started = time.perf_counter()
        proc = make_process(name)
        batch = proc.sample_batch(np.random.default_rng(EVAL_SEED), N_EVAL)
        order = np.random.default_rng(SPLIT_SEED).permutation(N_EVAL)
        ticks = np.arange(proc.m)
        rows = lambda seqs: (seqs[:, None] * proc.m + ticks[None, :]).reshape(-1)  # noqa: E731
        split = (rows(order[: int(TRAIN_FRAC * N_EVAL)]), rows(order[int(TRAIN_FRAC * N_EVAL) :]))

        config = ModelConfig(**training[name]["model"])
        entry = {
            "protocol": "one row per HMM tick, four block outputs, raw belief probabilities",
            "ridge_alpha": RIDGE_ALPHA,
            "n_eval_sequences": N_EVAL,
            "model_seed": training[name]["seed"],
            "evaluation_seed": EVAL_SEED,
            "analysis_commit": ANALYSIS_COMMIT,
            "device": device,
            "trained": _probe(OUTPUT_DIR / f"{name}_trained.pt", config, batch, proc, split, device),
            "random_init": _probe(
                OUTPUT_DIR / f"{name}_random_init.pt", config, batch, proc, split, device
            ),
        }
        entry["wall_seconds"] = time.perf_counter() - started
        results[name] = entry

        print(f"\n=== {name} ===", flush=True)
        for features in ("whole_tick", "last_position"):
            trained, random = entry["trained"][features], entry["random_init"][features]
            print(
                f"  {features:<14} dim={trained['feature_dim']:>5} "
                f"rows/feat={trained['rows_per_feature']:>6.2f} | "
                f"trained in-sample {trained['in_sample_r2']:+.3f} test {trained['test_r2']:+.3f} | "
                f"random in-sample {random['in_sample_r2']:+.3f} test {random['test_r2']:+.3f}",
                flush=True,
            )

    path = OUTPUT_DIR / "messk_06_reference_probe.json"
    path.write_text(json.dumps(results, indent=2, default=float))
    print(f"\nwrote {path}", flush=True)


if __name__ == "__main__":
    main()
