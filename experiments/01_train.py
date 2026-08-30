"""Train one transformer per physical-system and Mess-K pair.

The model sees nothing but the system's binned scalar observable. It is never
shown the chain's moods, its letters, the impulses, or the belief. The whole
claim under test is that next-token prediction builds the belief anyway, because
predicting the next observation requires anticipating the next impulse, and that
requires inferring what mood the chain is in.

Run:  uv run python experiments/01_train.py
Env:  OUTPUT_DIR, CONFIGS, TOTAL_TOKENS, SEED, TAG, REPORT_NAME.
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

from models.train import TrainConfig, pick_device, train
from models.transformer import ModelConfig, TinyTransformer
from physics.messk_configs import MESSK_CONFIGS, make_process

# Architecture is env-overridable so a capacity sweep is a launch flag rather
# than a code fork, and the seed-0 4x128 runs stay reproducible from defaults.
EMBED_DIM = int(os.environ.get("EMBED_DIM", 128))
NUM_LAYERS = int(os.environ.get("NUM_LAYERS", 4))
NUM_HEADS = int(os.environ.get("NUM_HEADS", 1))
D_MLP = int(os.environ.get("D_MLP", 4 * EMBED_DIM))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 128))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", 1e-3))

TOTAL_TOKENS = int(os.environ.get("TOTAL_TOKENS", 500_000_000))
SEED = int(os.environ.get("SEED", 0))
TAG = os.environ.get("TAG", "")
REPORT_NAME = os.environ.get("REPORT_NAME", "messk_01_training.json")
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "experiments/outputs-messk"))
CONFIGS = os.environ.get("CONFIGS", ",".join(MESSK_CONFIGS)).split(",")

# Log-spaced, so the emergence curves have resolution early where the
# reorganisation happens. Fractions, so a smoke test keeps the shape.
CHECKPOINT_FRACTIONS = (0.004, 0.01, 0.024, 0.06, 0.24, 0.5, 1.0)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = pick_device()
    print(f"device={device}  configs={CONFIGS}  total_tokens={TOTAL_TOKENS:,}", flush=True)

    summary = {}
    for name in CONFIGS:
        print(f"\n=== {name} ===", flush=True)
        t0 = time.perf_counter()
        proc = make_process(name)
        chain = proc.chain
        print(
            f"vocab={proc.n_obs}  seq_len={proc.seq_len}  moods={chain.n_states}  "
            f"alpha={chain.alpha}  stay={chain.stay}  actions={np.round(proc.actions, 3).tolist()}  "
            f"belief memory={chain.memory_length()} letters",
            flush=True,
        )

        cfg = ModelConfig(
            vocab_size=proc.n_obs, n_ctx=proc.seq_len, n_layers=NUM_LAYERS,
            n_heads=NUM_HEADS, d_model=EMBED_DIM, d_mlp=D_MLP, seed=SEED,
        )
        random_init = TinyTransformer(cfg)
        torch.save(random_init.state_dict(), OUTPUT_DIR / f"{name}{TAG}_random_init.pt")
        model = TinyTransformer(cfg)
        model.load_state_dict(random_init.state_dict())

        ckpt_dir = OUTPUT_DIR / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        def save(seen: int, m: TinyTransformer, _n=name, _d=ckpt_dir) -> None:
            torch.save(m.state_dict(), _d / f"{_n}{TAG}_{seen}.pt")

        schedule = tuple(sorted({int(f * TOTAL_TOKENS) for f in CHECKPOINT_FRACTIONS}))
        save(0, random_init)  # the untrained control on the same axis

        report = train(
            model,
            lambda rng, n: proc.sample_batch(rng, n)["tokens"],
            seq_len=proc.seq_len,
            config=TrainConfig(
                total_tokens=TOTAL_TOKENS, batch_size=BATCH_SIZE,
                learning_rate=LEARNING_RATE, seed=SEED, checkpoint_at=schedule,
            ),
            device=device,
            on_checkpoint=save,
        )
        torch.save(model.state_dict(), OUTPUT_DIR / f"{name}{TAG}_trained.pt")

        report |= {
            "config": name,
            "system": name.rsplit("_mess", 1)[0],
            "metrics": list(proc.system.metric_names),
            "vocab_size": proc.n_obs,
            "seq_len": proc.seq_len,
            "n_states": chain.n_states,
            "alpha": chain.alpha,
            "stay": chain.stay,
            "actions": proc.actions.tolist(),
            "belief_memory_letters": chain.memory_length(),
            "uniform_token_loss": float(np.log(proc.n_obs)),
            "checkpoint_tokens": [0, *schedule],
            "model": cfg.__dict__,
            "wall_seconds": time.perf_counter() - t0,
            "seed": SEED,
            "tag": TAG,
        }
        summary[name] = report
        print(
            f"{name}: eval loss={report['final']['eval_loss']:.4f} "
            f"(uniform baseline {report['uniform_token_loss']:.4f})  "
            f"{report['wall_seconds']:.0f}s",
            flush=True,
        )

    path = OUTPUT_DIR / REPORT_NAME
    path.write_text(json.dumps(summary, indent=2, default=float))
    print(f"\nwrote {path}", flush=True)


if __name__ == "__main__":
    main()
