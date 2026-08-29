"""Probe the residual stream for the belief, the true mood, and the velocity.

Three targets, because the difference between them is where the evidence is.
`belief` is P(next mood | letters), the headline object. `mood` is the
true state one-hot: a model that reads the belief well but the mood less well is
tracking the posterior rather than the answer, which is the interesting case.
`velocity` is the physical quantity the model plainly needs for its actual job.

Every number is measured against the same architecture left untrained. If the
random-init probe matches the trained one, the training showed nothing.

Run:  uv run python experiments/02_probe.py
Env:  OUTPUT_DIR, CONFIGS, N_EVAL.
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

from models.analysis import best_layer, probe_layers
from models.train import pick_device
from models.transformer import ModelConfig, TinyTransformer
from physics.messk_configs import MESSK_CONFIGS, make_process

N_EVAL = int(os.environ.get("N_EVAL", 512))
TRAIN_FRAC = 0.7
EVAL_SEED = 20_260_829
HEADLINE = "belief"

OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "experiments/outputs-messk"))
CONFIGS = os.environ.get("CONFIGS", ",".join(MESSK_CONFIGS)).split(",")


def _load(path: Path, config: ModelConfig, device: str) -> TinyTransformer:
    model = TinyTransformer(config)
    model.load_state_dict(torch.load(path, map_location=device))
    return model.to(device).eval()


def main() -> None:
    device = pick_device()
    training = json.loads((OUTPUT_DIR / "messk_01_training.json").read_text())
    print(f"device={device}  configs={CONFIGS}  n_eval={N_EVAL}", flush=True)

    results = {}
    for name in CONFIGS:
        print(f"\n=== {name} ===", flush=True)
        t0 = time.perf_counter()
        proc = make_process(name)
        batch = proc.sample_batch(np.random.default_rng(EVAL_SEED), N_EVAL)
        features, groups = proc.features_and_groups(batch)
        k = proc.chain.n_states

        # How uncertain the belief actually is, so a high R^2 cannot be explained
        # by the target having collapsed onto a corner of the simplex.
        b = features[:, :, groups["belief"]].reshape(-1, k)
        entropy = float(-(b * np.log(np.clip(b, 1e-12, None))).sum(1).mean() / np.log(k))
        recovery = float((b.argmax(1) == batch["moods"].reshape(-1)).mean())

        config = ModelConfig(**training[name]["model"])
        record = {
            "n_eval": N_EVAL,
            "n_states": k,
            "belief_entropy_normalised": entropy,
            "oracle_mood_recovery": recovery,
            "chance": 1.0 / k,
            "belief_memory_letters": proc.chain.memory_length(),
            "feature_groups": {g: [s.start, s.stop] for g, s in groups.items()},
            "final_train_loss": training[name]["final"],
        }
        for tag, filename in (("trained", f"{name}_trained.pt"),
                              ("random_init", f"{name}_random_init.pt")):
            model = _load(OUTPUT_DIR / filename, config, device)
            record[tag] = probe_layers(model, batch["tokens"], features, groups, device,
                                       train_frac=TRAIN_FRAC)

        tb = best_layer(record["trained"], group=HEADLINE)
        rb = best_layer(record["random_init"], group=HEADLINE)
        record["verdict"] = {
            "belief_r2_trained": tb["r2_by_group"]["belief"],
            "belief_r2_random_init": rb["r2_by_group"]["belief"],
            "mood_r2_trained": tb["r2_by_group"]["mood"],
            "velocity_r2_trained": tb["r2_by_group"]["velocity"],
            "best_depth_trained": tb["name"],
            "shuffled_control": tb["r2_shuffled_control"],
            "baseline_intact": bool(
                tb["r2_by_group"]["belief"] > rb["r2_by_group"]["belief"] + 0.05
            ),
        }
        record["wall_seconds"] = time.perf_counter() - t0
        results[name] = record
        v = record["verdict"]
        print(
            f"  belief R2 {v['belief_r2_trained']:.3f} (random-init {v['belief_r2_random_init']:.3f})  "
            f"mood {v['mood_r2_trained']:.3f}  velocity {v['velocity_r2_trained']:.3f}  "
            f"at {v['best_depth_trained']}  intact={v['baseline_intact']}",
            flush=True,
        )

    path = OUTPUT_DIR / "messk_02_probes.json"
    path.write_text(json.dumps(results, indent=2, default=float))
    print(f"\nwrote {path}", flush=True)


if __name__ == "__main__":
    main()
