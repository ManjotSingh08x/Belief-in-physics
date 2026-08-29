"""Small decoder-only transformer, deliberately unremarkable.

The claim under test is about what *ordinary* next-token pretraining produces,
so nothing here is clever: no auxiliary loss, no belief supervision, no physics
prior. The model never sees a belief, a latent bin, or a metric -- only tokens.

Plain torch rather than transformer_lens. The only thing the probes need from the
model is the residual stream after each block, which `residual_streams` returns
directly; transformer_lens would pull transformers/tokenizers/wandb (40+
packages) to provide hooks for a 1M-parameter model with a custom vocabulary
that never loads a pretrained checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    n_ctx: int
    n_layers: int = 4
    n_heads: int = 1
    d_model: int = 128
    d_mlp: int = 512
    seed: int = 0

    @property
    def n_params_non_embed(self) -> int:
        return self.n_layers * (4 * self.d_model**2 + 2 * self.d_model * self.d_mlp)


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model {d_model} not divisible by n_heads {n_heads}")
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        q, k, v = self.qkv(x).split(d, dim=2)
        shape = (b, t, self.n_heads, self.d_head)
        q = q.view(shape).transpose(1, 2)
        k = k.view(shape).transpose(1, 2)
        v = v.view(shape).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).reshape(b, t, d))


class Block(nn.Module):
    """Pre-LN residual block. `resid_post` is the output of this module."""

    def __init__(self, d_model: int, n_heads: int, d_mlp: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_mlp), nn.GELU(), nn.Linear(d_mlp, d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class TinyTransformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        torch.manual_seed(config.seed)
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.n_ctx, config.d_model)
        self.blocks = nn.ModuleList(
            Block(config.d_model, config.n_heads, config.d_mlp)
            for _ in range(config.n_layers)
        )
        self.ln_f = nn.LayerNorm(config.d_model)
        self.unembed = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def _embed(self, tokens: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        return self.tok_emb(tokens) + self.pos_emb(positions)[None]

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self._embed(tokens)
        for block in self.blocks:
            x = block(x)
        return self.unembed(self.ln_f(x))

    def loss(self, tokens: torch.Tensor) -> torch.Tensor:
        """Next-token cross entropy over every position."""
        logits = self(tokens)
        return F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]), tokens[:, 1:].reshape(-1)
        )

    @torch.no_grad()
    def residual_streams(self, tokens: torch.Tensor) -> list[np.ndarray]:
        """Residual stream at each depth, as numpy (batch, n_ctx, d_model).

        Index 0 is the embedding; index i+1 is `resid_post` of block i. Length
        is n_layers + 1.
        """
        x = self._embed(tokens)
        out = [x]
        for block in self.blocks:
            x = block(x)
            out.append(x)
        return [r.float().cpu().numpy() for r in out]


def _demo() -> None:
    cfg = ModelConfig(vocab_size=20, n_ctx=16, n_layers=2, d_model=32, d_mlp=64)
    model = TinyTransformer(cfg)
    tokens = torch.randint(0, 20, (4, 16))

    logits = model(tokens)
    assert logits.shape == (4, 16, 20), logits.shape
    assert len(model.residual_streams(tokens)) == 3
    assert model.residual_streams(tokens)[0].shape == (4, 16, 32)

    # Causality: changing a later token must not move an earlier position's logits.
    other = tokens.clone()
    other[:, -1] = (other[:, -1] + 1) % 20
    assert torch.allclose(model(other)[:, :-1], logits[:, :-1], atol=1e-5), "attention leaks future"

    loss = model.loss(tokens)
    assert loss.item() > 0 and np.isfinite(loss.item())
    print("transformer ok")


if __name__ == "__main__":
    _demo()
