"""Phase 3 -- probe the residual stream for the belief state and the metrics.

Reads the checkpoints written by 01. For every system and every depth it fits
the same linear probe three ways:

  trained       the model that saw 100M tokens of next-token loss
  random_init   the *same* initialisation, never trained -- the baseline check
  shuffled      trained activations against permuted targets, which must collapse

If `random_init` scores near `trained`, the result means nothing: a fixed random
projection of the token history was already enough, and the probe, not the
model, is doing the work. That comparison is the reason this script exists, so
it is reported first for every system.

Run:  uv run python -m experiments.02_probe_beliefs
Env:  OUTPUT_DIR (default experiments/outputs-01), SYSTEMS, N_EVAL.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for models/ and physics/

import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from models.analysis import best_layer, probe_layers
from models.features import belief_features, build_feature_map
from models.train import pick_device
from models.transformer import ModelConfig, TinyTransformer
from physics.configs import DEFAULT_CONFIGS
from physics.factory import make_simulator

OBS_PER_SEGMENT = 8
NUM_PERTURBATIONS = 8
N_EVAL = int(os.environ.get("N_EVAL", 512))
TRAIN_FRAC = 0.7
EVAL_SEED = 20_260_828

OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "experiments/outputs-01"))
SYSTEMS = os.environ.get("SYSTEMS", ",".join(DEFAULT_CONFIGS)).split(",")


def _load(path: Path, config: ModelConfig, device: str) -> TinyTransformer:
    model = TinyTransformer(config)
    model.load_state_dict(torch.load(path, map_location=device))
    return model.to(device).eval()


def main() -> None:
    device = pick_device()
    training = json.loads((OUTPUT_DIR / "phase2_training.json").read_text())
    print(f"device={device}  systems={SYSTEMS}  n_eval={N_EVAL}", flush=True)

    results = {}
    for name in SYSTEMS:
        print(f"\n=== {name} ===", flush=True)
        t0 = time.perf_counter()
        sim = make_simulator(DEFAULT_CONFIGS[name])
        n_steps = sim.K * OBS_PER_SEGMENT

        # Held-out data: a seed disjoint from both the training stream and the
        # training-time eval batch, so nothing here was seen during Phase 2.
        episodes = sim.simulate_batch(
            N_EVAL, n_steps, NUM_PERTURBATIONS, rng=np.random.default_rng(EVAL_SEED)
        )
        fmap = build_feature_map(sim.hmm)
        print(f"computing exact beliefs -> {fmap.n_features} features", flush=True)
        features = belief_features(sim.hmm, episodes.tokens, episodes.is_action, fmap)

        config = ModelConfig(**training[name]["model"])
        record = {
            "n_eval": N_EVAL,
            "n_features": fmap.n_features,
            "feature_groups": {k: [v.start, v.stop] for k, v in fmap.groups.items()},
            "final_train_loss": training[name]["final"],
        }
        for tag, filename in (("trained", f"{name}_trained.pt"), ("random_init", f"{name}_random_init.pt")):
            model = _load(OUTPUT_DIR / filename, config, device)
            record[tag] = probe_layers(
                model, episodes.tokens, features, fmap.groups, device, train_frac=TRAIN_FRAC
            )

        trained_best = best_layer(record["trained"])
        random_best = best_layer(record["random_init"])
        record["verdict"] = {
            "metric_r2_trained": trained_best["r2_by_group"]["metric"],
            "metric_r2_random_init": random_best["r2_by_group"]["metric"],
            "best_depth_trained": trained_best["name"],
            "shuffled_control": trained_best["r2_shuffled_control"],
            "baseline_intact": bool(
                trained_best["r2_by_group"]["metric"] > random_best["r2_by_group"]["metric"] + 0.05
            ),
        }
        record["wall_seconds"] = time.perf_counter() - t0
        results[name] = record

        v = record["verdict"]
        print(
            f"{name}: metric R^2 trained={v['metric_r2_trained']:.4f} @ {v['best_depth_trained']} | "
            f"random-init={v['metric_r2_random_init']:.4f} | shuffled={v['shuffled_control']:.4f} | "
            f"baseline {'INTACT' if v['baseline_intact'] else 'SUSPECT'}",
            flush=True,
        )

    (OUTPUT_DIR / "phase3_probes.json").write_text(json.dumps(results, indent=2, default=str))
    print(f"\nwrote {OUTPUT_DIR / 'phase3_probes.json'}", flush=True)


if __name__ == "__main__":
    main()
