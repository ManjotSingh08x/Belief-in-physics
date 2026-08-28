"""Phase 2 (exact branch process) -- train one transformer per system.

Differs from 01 in what the model is shown. Here the perturbations are hidden:
the vocabulary is observation bins only, and the model has to infer which
perturbations occurred from the observation stream alone. That inference is
what makes the belief a non-trivial point in a simplex rather than a delta.

Run:  uv run python experiments/03_train_branch.py
Env:  OUTPUT_DIR (default experiments/outputs-03), SYSTEMS, TOTAL_TOKENS.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root

import json
import os
import time

import torch

from models.train import TrainConfig, pick_device, train
from models.transformer import ModelConfig, TinyTransformer
from physics.branch_configs import BRANCH_CONFIGS, make_branch_process

EMBED_DIM = 128
NUM_LAYERS = 4
NUM_HEADS = 1
D_MLP = 4 * EMBED_DIM
BATCH_SIZE = 128
TOTAL_TOKENS = int(os.environ.get("TOTAL_TOKENS", 500_000_000))
LEARNING_RATE = 1e-3
SEED = 0

# Log-spaced snapshots so the probe curves have resolution early, where the
# interesting reorganisation happens. Fractions of TOTAL_TOKENS rather than
# absolute counts, so shrinking the budget for a smoke test keeps the shape.
CHECKPOINT_FRACTIONS = (0.004, 0.01, 0.024, 0.06, 0.12, 0.24, 0.5, 1.0)

OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "experiments/outputs-03"))
SYSTEMS = os.environ.get("SYSTEMS", ",".join(BRANCH_CONFIGS)).split(",")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = pick_device()
    print(f"device={device}  systems={SYSTEMS}  total_tokens={TOTAL_TOKENS:,}", flush=True)

    summary = {}
    for name in SYSTEMS:
        print(f"\n=== {name} ===", flush=True)
        t0 = time.perf_counter()
        process = make_branch_process(name)
        print(
            f"vocab={process.n_obs}  seq_len={process.seq_len}  "
            f"branches={process.n_branches(process.M):,}  n_actions={process.n_actions}",
            flush=True,
        )

        model_cfg = ModelConfig(
            vocab_size=process.n_obs,
            n_ctx=process.seq_len,
            n_layers=NUM_LAYERS,
            n_heads=NUM_HEADS,
            d_model=EMBED_DIM,
            d_mlp=D_MLP,
            seed=SEED,
        )
        # Saved before training so Phase 3 probes the same initialisation.
        random_init = TinyTransformer(model_cfg)
        torch.save(random_init.state_dict(), OUTPUT_DIR / f"{name}_random_init.pt")
        model = TinyTransformer(model_cfg)
        model.load_state_dict(random_init.state_dict())

        ckpt_dir = OUTPUT_DIR / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        def save(tokens_seen: int, m: TinyTransformer, _name=name, _dir=ckpt_dir) -> None:
            torch.save(m.state_dict(), _dir / f"{_name}_{tokens_seen}.pt")

        schedule = tuple(sorted({int(f * TOTAL_TOKENS) for f in CHECKPOINT_FRACTIONS}))
        save(0, random_init)  # the untrained control lives on the same axis

        report = train(
            model,
            lambda rng, n: process.sample_batch(rng, n).tokens,
            seq_len=process.seq_len,
            config=TrainConfig(
                total_tokens=TOTAL_TOKENS,
                batch_size=BATCH_SIZE,
                learning_rate=LEARNING_RATE,
                seed=SEED,
                checkpoint_at=schedule,
            ),
            device=device,
            on_checkpoint=save,
        )
        torch.save(model.state_dict(), OUTPUT_DIR / f"{name}_trained.pt")

        report |= {
            "system": name,
            "vocab_size": process.n_obs,
            "seq_len": process.seq_len,
            "n_actions": process.n_actions,
            "n_z0": process.n_z0,
            "M": process.M,
            "branches": int(process.n_branches(process.M)),
            "uniform_token_loss": float(__import__("numpy").log(process.n_obs)),
            "checkpoint_tokens": [0, *schedule],
            "model": model_cfg.__dict__,
            "wall_seconds": time.perf_counter() - t0,
        }
        summary[name] = report
        print(
            f"{name}: eval loss={report['final']['eval_obs_loss']:.4f} "
            f"(uniform baseline {report['uniform_token_loss']:.4f})  "
            f"{report['wall_seconds']:.0f}s",
            flush=True,
        )

    (OUTPUT_DIR / "phase2_branch_training.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nwrote {OUTPUT_DIR / 'phase2_branch_training.json'}", flush=True)


if __name__ == "__main__":
    main()
