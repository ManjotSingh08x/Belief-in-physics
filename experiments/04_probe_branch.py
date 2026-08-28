"""Phase 3 (exact branch process) -- probe the residual stream for the belief simplex.

The headline target is `action_lag0`: the exact posterior over which perturbation
just happened, a point in the (n_actions - 1) simplex. That is the direct
analogue of the object Shai et al. recover for Mess3, and unlike the marginals
in 02 it carries no discretisation error -- the belief is exact by enumeration.

Also probed: `action_lag1`/`action_lag2` (how far back the model still resolves
perturbations), `z0` (the posterior over initial conditions) and `metric`.

Every number is compared against the same architecture left untrained. If the
random-init probe matches the trained one, the trained model has shown nothing.

Run:  uv run python experiments/04_probe_branch.py
Env:  OUTPUT_DIR (default experiments/outputs-03), SYSTEMS, N_EVAL.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root

import json
import os
import time

import numpy as np
import torch

from models.analysis import best_layer, probe_layers
from models.train import pick_device
from models.transformer import ModelConfig, TinyTransformer
from physics.branch import feature_groups, forward_features
from physics.branch_configs import BRANCH_CONFIGS, make_branch_process

N_EVAL = int(os.environ.get("N_EVAL", 512))
TRAIN_FRAC = 0.7
EVAL_SEED = 20_260_828
HEADLINE = "action_lag0"

OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "experiments/outputs-03"))
SYSTEMS = os.environ.get("SYSTEMS", ",".join(BRANCH_CONFIGS)).split(",")


def _load(path: Path, config: ModelConfig, device: str) -> TinyTransformer:
    model = TinyTransformer(config)
    model.load_state_dict(torch.load(path, map_location=device))
    return model.to(device).eval()


def main() -> None:
    device = pick_device()
    training = json.loads((OUTPUT_DIR / "phase2_branch_training.json").read_text())
    print(f"device={device}  systems={SYSTEMS}  n_eval={N_EVAL}", flush=True)

    results = {}
    for name in SYSTEMS:
        print(f"\n=== {name} ===", flush=True)
        t0 = time.perf_counter()
        process = make_branch_process(name)
        episodes = process.sample_batch(np.random.default_rng(EVAL_SEED), N_EVAL)

        groups = feature_groups(process)
        print(f"computing exact beliefs over {process.n_branches(process.M):,} branches", flush=True)
        features = forward_features(process, episodes.tokens)

        # How uncertain the belief actually is, so a strong R^2 cannot be
        # explained by the target having collapsed onto a vertex.
        simplex = features[:, :, groups[HEADLINE]].reshape(-1, process.n_actions)
        entropy = float(
            -(simplex * np.log(np.clip(simplex, 1e-12, None))).sum(1).mean() / np.log(process.n_actions)
        )
        recovery = float(
            (features[:, -1, groups[HEADLINE]].argmax(1) == episodes.actions[:, -1]).mean()
        )

        config = ModelConfig(**training[name]["model"])
        record = {
            "n_eval": N_EVAL,
            "n_actions": process.n_actions,
            "belief_entropy_normalised": entropy,
            "oracle_last_action_recovery": recovery,
            "chance": 1.0 / process.n_actions,
            "feature_groups": {k: [v.start, v.stop] for k, v in groups.items()},
            "final_train_loss": training[name]["final"],
        }
        for tag, filename in (("trained", f"{name}_trained.pt"), ("random_init", f"{name}_random_init.pt")):
            model = _load(OUTPUT_DIR / filename, config, device)
            record[tag] = probe_layers(
                model, episodes.tokens, features, groups, device, train_frac=TRAIN_FRAC
            )

        trained_best = best_layer(record["trained"], group=HEADLINE)
        random_best = best_layer(record["random_init"], group=HEADLINE)
        record["verdict"] = {
            "simplex_r2_trained": trained_best["r2_by_group"][HEADLINE],
            "simplex_r2_random_init": random_best["r2_by_group"][HEADLINE],
            "best_depth_trained": trained_best["name"],
            "shuffled_control": trained_best["r2_shuffled_control"],
            "metric_r2_trained": trained_best["r2_by_group"]["metric"],
            "z0_r2_trained": trained_best["r2_by_group"]["z0"],
            "baseline_intact": bool(
                trained_best["r2_by_group"][HEADLINE] > random_best["r2_by_group"][HEADLINE] + 0.05
            ),
        }
        record["wall_seconds"] = time.perf_counter() - t0
        results[name] = record

        v = record["verdict"]
        print(
            f"{name}: belief entropy={entropy:.3f} (oracle recovery {recovery:.3f} vs chance "
            f"{1/process.n_actions:.3f})\n"
            f"  simplex R^2 trained={v['simplex_r2_trained']:.4f} @ {v['best_depth_trained']} | "
            f"random-init={v['simplex_r2_random_init']:.4f} | shuffled={v['shuffled_control']:.4f} | "
            f"baseline {'INTACT' if v['baseline_intact'] else 'SUSPECT'}",
            flush=True,
        )

    (OUTPUT_DIR / "phase3_branch_probes.json").write_text(json.dumps(results, indent=2, default=str))
    print(f"\nwrote {OUTPUT_DIR / 'phase3_branch_probes.json'}", flush=True)


if __name__ == "__main__":
    main()
