"""Phase 2 -- train one small transformer per physical system on next-token loss.

The model sees tokens and nothing else: no belief, no latent bin, no metric
appears in the objective. That is the whole point; Phase 3 then asks what the
residual stream turned out to contain.

Run:  uv run python -m experiments.01_train_transformer
Env:  OUTPUT_DIR (default experiments/outputs-01), SYSTEMS (comma separated),
      TOTAL_TOKENS.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for models/ and physics/

import json
import os
import time
from pathlib import Path

import torch

from models.train import TrainConfig, pick_device, train
from models.transformer import ModelConfig, TinyTransformer
from physics.configs import DEFAULT_CONFIGS
from physics.factory import make_simulator

EMBED_DIM = 128
NUM_LAYERS = 4
NUM_HEADS = 1
D_MLP = 4 * EMBED_DIM
OBS_PER_SEGMENT = 8
NUM_PERTURBATIONS = 8
MAX_SEQ_LEN = NUM_PERTURBATIONS * (1 + OBS_PER_SEGMENT)
BATCH_SIZE = 128
TOTAL_TOKENS = int(os.environ.get("TOTAL_TOKENS", 100_000_000))
LEARNING_RATE = 1e-3
SEED = 0

OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "experiments/outputs-01"))
SYSTEMS = os.environ.get("SYSTEMS", ",".join(DEFAULT_CONFIGS)).split(",")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = pick_device()
    print(f"device={device}  systems={SYSTEMS}  total_tokens={TOTAL_TOKENS:,}", flush=True)

    summary = {}
    for name in SYSTEMS:
        print(f"\n=== {name} ===", flush=True)
        t0 = time.perf_counter()
        sim = make_simulator(DEFAULT_CONFIGS[name])
        n_steps = sim.K * OBS_PER_SEGMENT
        vocab_size = sim.hmm.n_obs + sim.hmm.n_actions

        model_cfg = ModelConfig(
            vocab_size=vocab_size,
            n_ctx=MAX_SEQ_LEN,
            n_layers=NUM_LAYERS,
            n_heads=NUM_HEADS,
            d_model=EMBED_DIM,
            d_mlp=D_MLP,
            seed=SEED,
        )
        # Saved before training so Phase 3 can probe the *same* random init as a
        # control, rather than a differently seeded one.
        random_init = TinyTransformer(model_cfg)
        torch.save(random_init.state_dict(), OUTPUT_DIR / f"{name}_random_init.pt")

        model = TinyTransformer(model_cfg)
        model.load_state_dict(random_init.state_dict())

        report = train(
            model,
            sim,
            N=n_steps,
            M=NUM_PERTURBATIONS,
            config=TrainConfig(
                total_tokens=TOTAL_TOKENS,
                batch_size=BATCH_SIZE,
                learning_rate=LEARNING_RATE,
                seed=SEED,
            ),
            device=device,
        )
        torch.save(model.state_dict(), OUTPUT_DIR / f"{name}_trained.pt")

        report |= {
            "system": name,
            "vocab_size": vocab_size,
            "n_latent": int(sim.hmm.n_latent),
            "n_obs": int(sim.hmm.n_obs),
            "n_actions": int(sim.hmm.n_actions),
            "K": int(sim.K),
            "N": int(n_steps),
            "M": NUM_PERTURBATIONS,
            "model": model_cfg.__dict__,
            "wall_seconds": time.perf_counter() - t0,
        }
        summary[name] = report
        final = report["final"]
        print(
            f"{name}: obs_loss={final['eval_obs_loss']:.4f}  "
            f"action_loss={final['eval_action_loss']:.4f} (floor {report['action_loss_floor']:.4f})  "
            f"{report['wall_seconds']:.0f}s",
            flush=True,
        )

    (OUTPUT_DIR / "phase2_training.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nwrote {OUTPUT_DIR / 'phase2_training.json'}", flush=True)


if __name__ == "__main__":
    main()
