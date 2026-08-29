"""The decisive control: is the belief just a token window?

The belief forgets its prior after about six letters, so it is close to a
fixed function of recent input. If a ridge fit from the last few raw tokens
predicts it as well as one from the residual stream does, then "the model
encodes the belief" and "the model remembers what it just saw" are the same
statement, and the first claims nothing beyond the second.

Reported per window length W:

  belief_from_tokens   ridge from a one-hot of the last W tokens -> belief
  belief_from_resid    ridge from the residual stream -> belief (the best depth)
  excess               how much the residual adds over the window

An excess near zero is the damaging case.

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
from models.train import pick_device
from models.transformer import ModelConfig, TinyTransformer
from physics.messk import token_window_features
from physics.messk_configs import MESSK_CONFIGS, make_process

N_EVAL = int(os.environ.get("N_EVAL", 384))
WINDOWS = [int(w) for w in os.environ.get("WINDOWS", "1,2,4,8").split(",")]
TRAIN_FRAC = 0.7
EVAL_SEED = 20_260_829

OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "experiments/outputs-messk"))
CONFIGS = os.environ.get("CONFIGS", ",".join(MESSK_CONFIGS)).split(",")


def _fit(x_tr, y_tr, x_te, y_te) -> float:
    """Mean held-out R^2 across the target's columns."""
    return float(np.mean(r2_columns(fit_probe(x_tr, y_tr)(x_te), y_te)))


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
        cut = int(TRAIN_FRAC * N_EVAL)
        tr, te = order[:cut], order[cut:]
        flat = lambda a, idx: a[idx].reshape(-1, a.shape[-1]).astype(np.float64)
        y_tr, y_te = flat(belief, tr), flat(belief, te)

        model = TinyTransformer(ModelConfig(**training[name]["model"]))
        model.load_state_dict(torch.load(OUTPUT_DIR / f"{name}_trained.pt", map_location=device))
        streams = residual_streams_batched(model.to(device).eval(), batch["tokens"], device)
        resid = max(
            (_fit(flat(a, tr), y_tr, flat(a, te), y_te) for a in streams), default=float("nan")
        )

        rows = []
        for w in WINDOWS:
            tw = token_window_features(batch["tokens"], proc.n_obs, w)
            from_tokens = _fit(flat(tw, tr), y_tr, flat(tw, te), y_te)
            rows.append({
                "window": w,
                "belief_from_tokens": from_tokens,
                "belief_from_resid": resid,
                "excess": resid - from_tokens,
            })
            print(f"  W={w:2d}  tokens {from_tokens:.3f}   resid {resid:.3f}   "
                  f"excess {resid - from_tokens:+.3f}", flush=True)

        results[name] = {
            "n_eval": N_EVAL,
            "belief_memory_letters": proc.chain.memory_length(),
            "steps_per_tick": proc.n_steps,
            "by_window": rows,
            "wall_seconds": time.perf_counter() - t0,
        }

    path = OUTPUT_DIR / "messk_03_myopic.json"
    path.write_text(json.dumps(results, indent=2, default=float))
    print(f"\nwrote {path}", flush=True)


if __name__ == "__main__":
    main()
