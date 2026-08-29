"""The decisive control: is the belief just a token window?

The shared Mess-4 belief forgets its prior after about seven letters, so it is close to a
fixed function of recent input. If a ridge fit from the last few raw tokens
predicts it as well as one from the residual stream does, then "the model
encodes the belief" and "the model remembers what it just saw" are the same
statement, and the first claims nothing beyond the second.

Reported per observation-token window length W:

  belief_from_tokens   sparse ridge from ordered one-hots of the last W tokens
  belief_from_resid    ridge from the residual stream at its best depth
  excess               how much the residual adds over the raw window

The sweep reaches W=80, which covers eight ten-step chain ticks and therefore
the full seven-letter belief memory. Position inside the current tick is included
as a control feature. Sparse-ridge strength is selected on a sequence-level
validation split before the untouched test split is scored, so larger windows
are not punished by arbitrary under-regularisation. An excess near zero is the
damaging case.

Run:  uv run python experiments/03_myopic.py
Env:  OUTPUT_DIR, CONFIGS, N_EVAL, WINDOWS.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import os
import time

import numpy as np
import torch

from models.analysis import residual_streams_batched
from models.bootstrap import r2_columns
from models.probe import fit_probe
from models.probe_extra import sparse_token_window_features
from models.train import pick_device
from models.transformer import ModelConfig, TinyTransformer
from sklearn.linear_model import Ridge
from physics.messk_configs import MESSK_CONFIGS, make_process

N_EVAL = int(os.environ.get("N_EVAL", 384))
WINDOWS = [int(w) for w in os.environ.get("WINDOWS", "1,2,4,8,16,32,64,80").split(",")]
TRAIN_FRAC = 0.6
VALIDATION_FRAC = 0.2
SPARSE_ALPHAS = (0.1, 1.0, 10.0, 100.0, 1_000.0, 10_000.0)
EVAL_SEED = 20_260_829

OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "experiments/outputs-messk"))
CONFIGS = os.environ.get("CONFIGS", ",".join(MESSK_CONFIGS)).split(",")


def _fit(x_tr, y_tr, x_te, y_te) -> float:
    """Mean held-out R^2 across the target's columns."""
    return float(np.mean(r2_columns(fit_probe(x_tr, y_tr)(x_te), y_te)))


def _fit_sparse(x_sub, y_sub, x_val, y_val, x_tr, y_tr, x_te, y_te) -> tuple[float, float]:
    """Tune sparse-ridge strength on held-out sequences, then score the test set."""
    candidates = []
    for alpha in SPARSE_ALPHAS:
        probe = Ridge(alpha=alpha, solver="lsqr").fit(x_sub, y_sub)
        score = float(np.mean(r2_columns(probe.predict(x_val), y_val)))
        candidates.append((score, alpha))
    _, best_alpha = max(candidates)
    probe = Ridge(alpha=best_alpha, solver="lsqr").fit(x_tr, y_tr)
    score = float(np.mean(r2_columns(probe.predict(x_te), y_te)))
    return score, best_alpha


def main() -> None:
    device = pick_device()
    training = json.loads((OUTPUT_DIR / "messk_01_training.json").read_text())
    results = {}

    for name in CONFIGS:
        print(f"\n=== {name} ===", flush=True)
        t0 = time.perf_counter()
        proc = make_process(name)
        batch = proc.sample_batch(np.random.default_rng(EVAL_SEED), N_EVAL)
        features, groups = proc.features_and_groups(batch)
        belief = features[:, :, groups["belief"]]

        # Split by sequence, never by position: positions inside one sequence
        # share a belief trajectory and would leak across the split.
        rng = np.random.default_rng(0)
        order = rng.permutation(N_EVAL)
        train_end = int(TRAIN_FRAC * N_EVAL)
        validation_end = train_end + int(VALIDATION_FRAC * N_EVAL)
        tr, val, te = order[:train_end], order[train_end:validation_end], order[validation_end:]
        train_validation = np.concatenate([tr, val])
        flat = lambda a, idx: a[idx].reshape(-1, a.shape[-1]).astype(np.float64)
        y_tr, y_val = flat(belief, tr), flat(belief, val)
        y_train_validation = flat(belief, train_validation)
        y_te = flat(belief, te)

        model = TinyTransformer(ModelConfig(**training[name]["model"]))
        model.load_state_dict(torch.load(OUTPUT_DIR / f"{name}_trained.pt", map_location=device))
        streams = residual_streams_batched(model.to(device).eval(), batch["tokens"], device)
        validation_scores = [
            _fit(flat(a, tr), y_tr, flat(a, val), y_val) for a in streams
        ]
        best_depth = int(np.argmax(validation_scores))
        resid = _fit(
            flat(streams[best_depth], train_validation), y_train_validation,
            flat(streams[best_depth], te), y_te,
        )

        row_indices = lambda idx: (
            idx[:, None] * proc.seq_len + np.arange(proc.seq_len)
        ).reshape(-1)
        train_rows, val_rows = row_indices(tr), row_indices(val)
        train_validation_rows = row_indices(train_validation)
        test_rows = row_indices(te)
        rows = []
        for w in WINDOWS:
            tw = sparse_token_window_features(
                batch["tokens"], proc.n_obs, w, proc.n_steps
            )
            from_tokens, alpha = _fit_sparse(
                tw[train_rows], y_tr, tw[val_rows], y_val,
                tw[train_validation_rows], y_train_validation,
                tw[test_rows], y_te,
            )
            rows.append({
                "window": w,
                "ridge_alpha": alpha,
                "belief_from_tokens": from_tokens,
                "belief_from_resid": resid,
                "excess": resid - from_tokens,
            })
            print(f"  W={w:2d}  tokens {from_tokens:.3f} (alpha {alpha:g})  "
                  f"resid {resid:.3f}  excess {resid - from_tokens:+.3f}", flush=True)

        results[name] = {
            "n_eval": N_EVAL,
            "belief_memory_letters": proc.chain.memory_length(),
            "steps_per_tick": proc.n_steps,
            "residual_depth_selected_on_validation": best_depth,
            "residual_validation_r2": validation_scores[best_depth],
            "by_window": rows,
            "wall_seconds": time.perf_counter() - t0,
        }

    path = OUTPUT_DIR / "messk_03_myopic.json"
    path.write_text(json.dumps(results, indent=2, default=float))
    print(f"\nwrote {path}", flush=True)


if __name__ == "__main__":
    main()
