"""Bridge probe matching the external Mess-3 feature construction, but held out.

The external protocol concatenates all four block outputs at all ten physics
positions in one HMM tick, producing 4 * 128 * 10 = 5,120 features for one
belief target. It then standardises features and fits multi-output ridge to the
raw probability vector.

This experiment preserves that construction and fixed ridge alpha, but splits
whole physical sequences into train, validation, and untouched test sets. It
runs trained and architecture-matched random-initialisation checkpoints.

Run:  uv run python experiments/06_reference_probe.py
Env:  OUTPUT_DIR, CONFIGS, N_EVAL, ANALYSIS_COMMIT.
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
from sklearn.preprocessing import StandardScaler

from models.analysis import residual_streams_batched
from models.train import pick_device
from models.transformer import ModelConfig, TinyTransformer
from physics.messk_configs import MESSK_CONFIGS, make_process

N_EVAL = int(os.environ.get("N_EVAL", 1_024))
TRAIN_FRAC = 0.6
VALIDATION_FRAC = 0.2
EVAL_SEED = 20_260_829
SPLIT_SEED = 0
RIDGE_ALPHA = 1.0

OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "experiments/outputs-messk"))
CONFIGS = os.environ.get("CONFIGS", ",".join(MESSK_CONFIGS)).split(",")
ANALYSIS_COMMIT = os.environ.get("ANALYSIS_COMMIT", "unknown")


def _tick_features(streams: list[np.ndarray], steps_per_tick: int) -> np.ndarray:
    """All block outputs and all positions in a tick: (n, ticks, 5120)."""
    if len(streams) != 5:
        raise ValueError(f"expected embedding plus four block streams, got {len(streams)}")
    all_blocks = np.concatenate(streams[1:], axis=-1)
    n, length, width = all_blocks.shape
    if length % steps_per_tick:
        raise ValueError(f"sequence length {length} is not divisible by {steps_per_tick}")
    return all_blocks.reshape(n, length // steps_per_tick, steps_per_tick * width)


def _split_sequences(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.random.default_rng(SPLIT_SEED).permutation(n)
    train_end = int(TRAIN_FRAC * n)
    validation_end = train_end + int(VALIDATION_FRAC * n)
    return order[:train_end], order[train_end:validation_end], order[validation_end:]


def _rows(array: np.ndarray, indices: np.ndarray) -> np.ndarray:
    return array[indices].reshape(-1, array.shape[-1])


def _score_split(
    features: np.ndarray,
    targets: np.ndarray,
    train_idx: np.ndarray,
    validation_idx: np.ndarray,
    test_idx: np.ndarray,
) -> dict:
    scaler = StandardScaler(copy=False)
    x_train = scaler.fit_transform(_rows(features, train_idx))
    x_validation = scaler.transform(_rows(features, validation_idx))
    x_test = scaler.transform(_rows(features, test_idx))
    y_train = _rows(targets, train_idx)
    y_validation = _rows(targets, validation_idx)
    y_test = _rows(targets, test_idx)

    probe = Ridge(alpha=RIDGE_ALPHA, solver="lsqr", tol=1e-6).fit(x_train, y_train)
    return {
        "validation_r2": float(probe.score(x_validation, y_validation)),
        "test_r2": float(probe.score(x_test, y_test)),
        "n_train_rows": int(len(x_train)),
        "n_validation_rows": int(len(x_validation)),
        "n_test_rows": int(len(x_test)),
    }


def _load(path: Path, config: ModelConfig, device: str) -> TinyTransformer:
    model = TinyTransformer(config)
    model.load_state_dict(torch.load(path, map_location=device))
    return model.to(device).eval()


def _probe_model(
    path: Path,
    config: ModelConfig,
    batch: dict,
    steps_per_tick: int,
    split: tuple[np.ndarray, np.ndarray, np.ndarray],
    device: str,
) -> dict:
    model = _load(path, config, device)
    streams = residual_streams_batched(model, batch["tokens"], device)
    features = _tick_features(streams, steps_per_tick)
    targets = batch["beliefs"][:, ::steps_per_tick, :]
    result = _score_split(features, targets, *split)
    result["feature_dim"] = int(features.shape[-1])
    del model, streams, features
    gc.collect()
    return result


def _demo() -> None:
    streams = [np.zeros((2, 20, 3), dtype=np.float32) for _ in range(5)]
    features = _tick_features(streams, steps_per_tick=10)
    assert features.shape == (2, 2, 120)


def main() -> None:
    _demo()
    device = pick_device()
    training = json.loads((OUTPUT_DIR / "messk_01_training.json").read_text())
    split = _split_sequences(N_EVAL)
    results = {}
    print(f"device={device}  n_eval={N_EVAL}  configs={CONFIGS}", flush=True)

    for name in CONFIGS:
        print(f"\n=== {name} ===", flush=True)
        started = time.perf_counter()
        proc = make_process(name)
        batch = proc.sample_batch(np.random.default_rng(EVAL_SEED), N_EVAL)
        config = ModelConfig(**training[name]["model"])
        trained = _probe_model(
            OUTPUT_DIR / f"{name}_trained.pt", config, batch,
            proc.n_steps, split, device,
        )
        random_init = _probe_model(
            OUTPUT_DIR / f"{name}_random_init.pt", config, batch,
            proc.n_steps, split, device,
        )
        results[name] = {
            "protocol": "four block outputs x ten tick positions -> raw belief probabilities",
            "ridge_alpha": RIDGE_ALPHA,
            "n_eval_sequences": N_EVAL,
            "model_seed": training[name]["seed"],
            "evaluation_seed": EVAL_SEED,
            "analysis_commit": ANALYSIS_COMMIT,
            "device": device,
            "trained": trained,
            "random_init": random_init,
            "trained_minus_random_test_r2": trained["test_r2"] - random_init["test_r2"],
            "wall_seconds": time.perf_counter() - started,
        }
        print(
            f"  trained val/test {trained['validation_r2']:+.3f}/{trained['test_r2']:+.3f}  "
            f"random val/test {random_init['validation_r2']:+.3f}/{random_init['test_r2']:+.3f}",
            flush=True,
        )

    path = OUTPUT_DIR / "messk_06_reference_probe.json"
    path.write_text(json.dumps(results, indent=2, default=float))
    print(f"\nwrote {path}", flush=True)


if __name__ == "__main__":
    main()
