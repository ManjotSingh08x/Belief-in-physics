"""Phase 2 training: next-token cross entropy, nothing else.

Two things worth stating because they change how the loss curve should be read.

**Action tokens are unpredictable by construction.** `DiscreteHMM.sample_batch`
draws each perturbation uniformly at random, so no model can do better than
`log(n_actions)` on those positions. They are kept in the loss because the claim
is about ordinary next-token pretraining, but the number that carries signal is
the loss restricted to observation positions, so both are reported.

**Data is streamed, never reused.** The generator is an HMM, so training data is
unlimited and every step sees a fresh batch. There is no train/test gap to
manage and no memorisation to control for -- which is exactly why a probe result
here cannot be explained by the model having memorised a finite corpus.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
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


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _split_loss(logits: torch.Tensor, tokens: torch.Tensor, is_action: torch.Tensor) -> tuple[float, float]:
    """(observation-position loss, action-position loss) on the next-token targets."""
    flat_logits = logits[:, :-1].reshape(-1, logits.shape[-1])
    targets = tokens[:, 1:].reshape(-1)
    per_token = F.cross_entropy(flat_logits, targets, reduction="none")
    action_mask = is_action[:, 1:].reshape(-1)

    obs = per_token[~action_mask]
    act = per_token[action_mask]
    return (
        float(obs.mean()) if obs.numel() else float("nan"),
        float(act.mean()) if act.numel() else float("nan"),
    )


def train(
    model: TinyTransformer,
    sampler,
    seq_len: int,
    config: TrainConfig,
    device: str | None = None,
    action_mask_fn=None,
    action_loss_floor: float = float("nan"),
    on_checkpoint=None,
) -> dict:
    """Stream batches from `sampler(rng, n) -> tokens` and fit next-token loss.

    `sampler` rather than a simulator object so the same loop serves both the
    Ulam pipeline (where perturbations are observed tokens) and the exact branch
    pipeline (where they are hidden and the vocabulary is observations only).
    `action_mask_fn` is only meaningful for the former; without it the split
    loss is not reported because there are no action positions to split on.

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
    eval_actions = (
        torch.as_tensor(action_mask_fn(eval_np), dtype=torch.bool, device=device)
        if action_mask_fn is not None
        else None
    )

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
                logits = model(eval_tokens)
                if eval_actions is not None:
                    obs_loss, act_loss = _split_loss(logits, eval_tokens, eval_actions)
                else:
                    obs_loss = float(
                        F.cross_entropy(
                            logits[:, :-1].reshape(-1, logits.shape[-1]), eval_tokens[:, 1:].reshape(-1)
                        )
                    )
                    act_loss = float("nan")
            model.train()
            history.append(
                {"step": step, "train_loss": float(loss.item()), "eval_obs_loss": obs_loss, "eval_action_loss": act_loss}
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
        "action_loss_floor": action_loss_floor,
        "final": history[-1] if history else None,
    }
