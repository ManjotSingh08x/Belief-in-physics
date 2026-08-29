"""When does each target appear, as training proceeds.

One probe per checkpoint per target, giving an R^2 curve against tokens seen.
The reported crossing is the point where a target first reaches a fraction of
its OWN final learned gain, not an absolute R^2: an absolute threshold conflates
"learned sooner" with "learned more", because a target that ends higher crosses
any fixed line earlier almost mechanically.

`learned gain` is measured from the untrained model at the same seed, so a
target that is already readable at initialisation is not credited for it.

Run:  uv run python experiments/04_emergence.py
Env:  OUTPUT_DIR, CONFIGS, N_EVAL, FRACTIONS.
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

from models.analysis import probe_layers
from models.train import pick_device
from models.transformer import ModelConfig, TinyTransformer
from physics.messk_configs import MESSK_CONFIGS, make_process

N_EVAL = int(os.environ.get("N_EVAL", 384))
FRACTIONS = [float(f) for f in os.environ.get("FRACTIONS", "0.5,0.75").split(",")]
MIN_GAIN = 0.05  # below this a "crossing" is noise, not learning
EVAL_SEED = 20_260_829

OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "experiments/outputs-messk"))
CONFIGS = os.environ.get("CONFIGS", ",".join(MESSK_CONFIGS)).split(",")


def crossing(axis: list[int], curve: list[float], threshold: float) -> float | None:
    """Tokens at which the curve's gain first reaches `threshold`, log-interpolated."""
    gain = np.asarray(curve) - curve[0]
    hit = np.flatnonzero(gain >= threshold)
    if not hit.size:
        return None
    i = int(hit[0])
    if i == 0:
        return float(axis[0])
    x0, x1 = np.log(max(axis[i - 1], 1)), np.log(max(axis[i], 1))
    y0, y1 = gain[i - 1], gain[i]
    return float(np.exp(x0 + (threshold - y0) / (y1 - y0) * (x1 - x0)))


def main() -> None:
    device = pick_device()
    training = json.loads((OUTPUT_DIR / "messk_01_training.json").read_text())
    ckpt_dir = OUTPUT_DIR / "checkpoints"
    results = {}

    for name in CONFIGS:
        print(f"\n=== {name} ===", flush=True)
        t0 = time.perf_counter()
        proc = make_process(name)
        batch = proc.sample_batch(np.random.default_rng(EVAL_SEED), N_EVAL)
        features, groups = proc.features_and_groups(batch)
        config = ModelConfig(**training[name]["model"])

        axis = sorted(training[name]["checkpoint_tokens"])
        curves = {g: [] for g in groups}
        for seen in axis:
            path = ckpt_dir / f"{name}_{seen}.pt"
            if not path.exists():
                raise SystemExit(f"missing checkpoint {path}")
            model = TinyTransformer(config)
            model.load_state_dict(torch.load(path, map_location=device))
            records = probe_layers(model.to(device).eval(), batch["tokens"], features,
                                   groups, device)
            for g in groups:  # the depth that reads this target best, per checkpoint
                curves[g].append(max(r["r2_by_group"][g] for r in records))
            print(f"  {seen:>12,}  " + "  ".join(f"{g} {curves[g][-1]:+.3f}" for g in groups),
                  flush=True)

        targets = []
        for g, curve in curves.items():
            gain = curve[-1] - curve[0]
            entry = {"target": g, "curve": curve, "learned_gain": gain,
                     "final": curve[-1], "untrained": curve[0]}
            for f in FRACTIONS:
                entry[f"cross@{f}"] = (
                    crossing(axis, curve, f * gain) if gain > MIN_GAIN else None
                )
            targets.append(entry)

        results[name] = {"tokens_axis": axis, "n_eval": N_EVAL, "targets": targets,
                         "wall_seconds": time.perf_counter() - t0}
        print("  ordering at 75% of each target's own gain:", flush=True)
        for e in sorted(targets, key=lambda e: (e[f"cross@0.75"] is None, e[f"cross@0.75"] or 0)):
            c = e["cross@0.75"]
            print(f"    {e['target']:9s} gain {e['learned_gain']:+.3f}  "
                  f"cross {'never' if c is None else format(int(c), ',')}", flush=True)

    path = OUTPUT_DIR / "messk_04_emergence.json"
    path.write_text(json.dumps(results, indent=2, default=float))
    print(f"\nwrote {path}", flush=True)


if __name__ == "__main__":
    main()
