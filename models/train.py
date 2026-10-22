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
    eval_batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    warmup_frac: float = 0.02
    grad_clip: float = 1.0
    seed: int = 0
    log_every: int = 100
    checkpoint_at: tuple[int, ...] = ()  # token counts to hand to `on_checkpoint`
    num_workers: int = 4


class StreamingTokenDataset(torch.utils.data.IterableDataset):
    def __init__(self, sampler, batch_size: int, steps: int, base_seed: int):
        super().__init__()
        self.sampler = sampler
        self.batch_size = batch_size
        self.steps = steps
        self.base_seed = base_seed

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        rng = np.random.default_rng(self.base_seed + 1000 * worker_id + 1)
        my_steps = (self.steps + num_workers - 1 - worker_id) // num_workers
        for _ in range(my_steps):
            tokens = self.sampler(rng, self.batch_size)
            yield torch.as_tensor(tokens, dtype=torch.long)


def checkpoint_paths(directory, name: str, tag: str = ""):
    """`(tokens_seen, path)` for every checkpoint of one run, sorted.

    `tag` separates runs of the same system that differ in seed or in kick
    magnitude, which all live in one directory.
    """
    from pathlib import Path

    prefix = f"{name}{tag}_"
    return sorted(
        (int(p.stem[len(prefix):]), p)
        for p in Path(directory).glob(f"{prefix}*.pt")
        if p.stem[len(prefix):].isdigit()
    )


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

    is_cuda = str(device).startswith("cuda")
    device_type = "cuda" if is_cuda else "cpu"
    use_amp = is_cuda
    amp_dtype = torch.bfloat16 if (is_cuda and torch.cuda.is_bf16_supported()) else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and amp_dtype == torch.float16))

    if is_cuda:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    steps = max(1, config.total_tokens // (config.batch_size * seq_len))
    warmup = max(1, int(steps * config.warmup_frac))

    optimiser = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    def lr_at(step: int) -> float:
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(1, steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    schedule = torch.optim.lr_scheduler.LambdaLR(optimiser, lr_at)

    eval_rng = np.random.default_rng(config.seed + 99_999)
    eval_np = sampler(eval_rng, config.eval_batch_size)
    eval_tokens = torch.as_tensor(eval_np, dtype=torch.long, device=device)

    if config.num_workers > 0:
        dataset = StreamingTokenDataset(sampler, config.batch_size, steps, config.seed)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=None,
            num_workers=config.num_workers,
            pin_memory=is_cuda,
            prefetch_factor=2,
            persistent_workers=True if steps > config.num_workers else False,
        )
        token_iter = iter(loader)
    else:
        def _sync_iter():
            stream = np.random.default_rng(config.seed + 1)
            for _ in range(steps):
                tokens = sampler(stream, config.batch_size)
                yield torch.as_tensor(tokens, dtype=torch.long)
        token_iter = _sync_iter()

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

        tokens = next(token_iter).to(device, non_blocking=True)

        optimiser.zero_grad(set_to_none=True)
        if use_amp:
            with torch.autocast(device_type=device_type, dtype=amp_dtype):
                loss = model.loss(tokens)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimiser)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                scaler.step(optimiser)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                optimiser.step()
        else:
            loss = model.loss(tokens)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimiser.step()
        schedule.step()

        if step % config.log_every == 0 or step == steps - 1:
            model.eval()
            with torch.no_grad():
                if use_amp:
                    with torch.autocast(device_type=device_type, dtype=amp_dtype):
                        eval_loss = float(model.loss(eval_tokens))
                else:
                    eval_loss = float(model.loss(eval_tokens))
            model.train()
            history.append(
                {"step": step, "train_loss": float(loss.item()), "eval_loss": eval_loss}
            )
            print_every = max(config.log_every, max(1, steps // 200))
            if step % print_every == 0 or step == steps - 1:
                lr_curr = schedule.get_last_lr()[0]
                print(
                    f"train [{step:5d}/{steps} ({step / steps:5.1%})] "
                    f"loss: {loss.item():.4f} | eval_loss: {eval_loss:.4f} | lr: {lr_curr:.2e}",
                    flush=True,
                )

    model.eval()
    total_seen = steps * tokens_per_step
    for target in pending:
        if target <= total_seen:
            if on_checkpoint is not None:
                on_checkpoint(target, model)
            saved.append(target)

    return {
        "history": history,
        "checkpoints": saved,
        "steps": steps,
        "seq_len": seq_len,
        "tokens_seen": steps * config.batch_size * seq_len,
        "device": str(device),
        "final": history[-1] if history else None,
    }
