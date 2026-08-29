"""Next-observation-token training, with no auxiliary objective.

The generator is unlimited and every optimisation step sees a fresh batch, so
there is no finite training corpus to memorise. The model receives only the 181
observation bins and is never shown the chain letter, physical action, mood, or
belief used to produce them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from tqdm.auto import tqdm

from .transformer import TinyTransformer


@dataclass(frozen=True)
class TrainConfig:
    total_tokens: int = 40_000_000
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    warmup_frac: float = 0.02
    grad_clip: float = 1.0
    seed: int = 0
    log_every: int = 100
    checkpoint_at: tuple[int, ...] = ()  # token counts to hand to `on_checkpoint`


def checkpoint_paths(directory, name: str, tag: str = ""):
    """`(tokens_seen, path)` for every checkpoint of one run, sorted.

    `tag` separates runs of the same system that differ in seed or in kick
    magnitude, which all live in one directory.
    """
    from pathlib import Path

    prefix = f"{name}{tag}_"
    found = []
    for path in Path(directory).glob(f"{prefix}*.pt"):
        suffix = path.stem[len(prefix):]
        if suffix.isdigit():
            found.append((int(suffix), path))
    return sorted(found)


def pick_device() -> str:
    forced = __import__("os").environ.get("DEVICE")
    if forced:
        return forced
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def train(
    model: TinyTransformer,
    sampler,
    seq_len: int,
    config: TrainConfig,
    device: str | None = None,
    on_checkpoint=None,
) -> dict:
    """Stream batches from `sampler(rng, n) -> tokens` and fit next-token loss.

    `sampler` rather than a process object keeps the loop indifferent to which
    physical system generated the observation tokens.

    `on_checkpoint(tokens_seen, model)` fires the first time the token count
    passes each entry of `config.checkpoint_at`. The LR schedule is defined over
    `total_tokens`, so a checkpoint is a snapshot part-way along *this* run, not
    a model that was trained to that budget and annealed.
    """
    device = device or pick_device()
    model = model.to(device)

    steps = max(1, config.total_tokens // (config.batch_size * seq_len))
    warmup = max(1, int(steps * config.warmup_frac))

    optimiser = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    def lr_at(step: int) -> float:
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(1, steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    schedule = torch.optim.lr_scheduler.LambdaLR(optimiser, lr_at)

    stream = np.random.default_rng(config.seed + 1)
    eval_rng = np.random.default_rng(config.seed + 99_999)
    eval_np = sampler(eval_rng, 256)
    eval_tokens = torch.as_tensor(eval_np, dtype=torch.long, device=device)

    history: list[dict] = []
    tokens_per_step = config.batch_size * seq_len
    pending = sorted(config.checkpoint_at)
    saved: list[int] = []

    model.train()
    for step in tqdm(range(steps), desc="train", dynamic_ncols=True):
        seen = step * tokens_per_step
        while pending and seen >= pending[0]:
            target = pending.pop(0)
            if on_checkpoint is not None:
                model.eval()
                on_checkpoint(target, model)
                model.train()
            saved.append(target)

        tokens = torch.as_tensor(sampler(stream, config.batch_size), dtype=torch.long, device=device)

        loss = model.loss(tokens)
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimiser.step()
        schedule.step()

        if step % config.log_every == 0 or step == steps - 1:
            model.eval()
            with torch.no_grad():
                eval_loss = float(model.loss(eval_tokens))
            model.train()
            history.append(
                {"step": step, "train_loss": float(loss.item()), "eval_loss": eval_loss}
            )

    model.eval()
    for target in pending:  # schedule entries past the final step
        if on_checkpoint is not None:
            on_checkpoint(target, model)
        saved.append(target)

    return {
        "history": history,
        "checkpoints": saved,
        "steps": steps,
        "seq_len": seq_len,
        "tokens_seen": steps * config.batch_size * seq_len,
        "device": device,
        "final": history[-1] if history else None,
    }
