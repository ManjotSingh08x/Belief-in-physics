"""Subspace ablation: delete what a probe reads, then measure what it cost.

The claim a probe supports is weak on its own -- a direction can be linearly
decodable without the model using it. Ablation is the causal follow-up: project
the residual stream onto the orthogonal complement of the probe's row space,
finish the forward pass, and read the increase in next-token loss.

Two choices that decide whether the number means anything.

**Mean-ablate, not zero-ablate.** Setting a direction to zero moves the stream
off the distribution the later blocks were trained on, so part of the damage is
distribution shift rather than lost information. Replacing the component with
its dataset mean removes the *variation* while leaving the stream where the
model expects it.

**Always against a random subspace of the same rank.** Deleting any r
directions from a d_model-dimensional stream costs something, and the blocks
differ in how much slack they have. `metric` has rank 1-2 and `z0` rank 3, so
their raw damage is not comparable to each other or across depths. The quantity
to report is the excess over a random-subspace control at matched rank.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .transformer import TinyTransformer


def probe_basis(weight: np.ndarray, columns: slice, tol: float = 1e-6) -> np.ndarray:
    """Orthonormal basis (d_model, r) for the directions a probe block reads.

    Rank is checked rather than assumed: a block of marginals sums to 1, so its
    columns are linearly dependent and the true rank is one less than the width.
    """
    block = weight[:, columns]
    u, s, _ = np.linalg.svd(block, full_matrices=False)
    return u[:, s > tol * s[0]] if s[0] > 0 else u[:, :0]


def random_basis(rng: np.random.Generator, d_model: int, rank: int) -> np.ndarray:
    q, _ = np.linalg.qr(rng.normal(size=(d_model, rank)))
    return q[:, :rank]


def _project_out(x: torch.Tensor, basis: torch.Tensor, mean: torch.Tensor) -> torch.Tensor:
    centred = x - mean
    return x - (centred @ basis) @ basis.T


@torch.no_grad()
def ablated_loss(
    model: TinyTransformer,
    tokens: torch.Tensor,
    depth: int,
    basis: np.ndarray | None,
    mean: np.ndarray | None = None,
) -> float:
    """Next-token cross entropy with `basis` mean-ablated from the stream at `depth`.

    `depth` indexes as `residual_streams` does: 0 is the embedding, i + 1 is the
    output of block i, so the intervention lands *before* block `depth`.
    """
    x = model._embed(tokens)
    if basis is not None:
        b = torch.as_tensor(basis, dtype=x.dtype, device=x.device)
        m = torch.as_tensor(
            np.zeros(x.shape[-1]) if mean is None else mean, dtype=x.dtype, device=x.device
        )

    for i, block in enumerate(model.blocks):
        if basis is not None and i == depth:
            x = _project_out(x, b, m)
        x = block(x)
    if basis is not None and depth == len(model.blocks):
        x = _project_out(x, b, m)

    logits = model.unembed(model.ln_f(x))
    return float(
        F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), tokens[:, 1:].reshape(-1))
    )


@torch.no_grad()
def ablated_stream(
    model: TinyTransformer, tokens: torch.Tensor, depth: int, basis: np.ndarray, mean: np.ndarray
) -> np.ndarray:
    """The stream at `depth` with `basis` removed -- the input for cross-probing."""
    x = model._embed(tokens)
    for i, block in enumerate(model.blocks):
        if i == depth:
            break
        x = block(x)
    b = torch.as_tensor(basis, dtype=x.dtype, device=x.device)
    m = torch.as_tensor(mean, dtype=x.dtype, device=x.device)
    return _project_out(x, b, m).float().cpu().numpy()


def principal_angles(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Principal angles in degrees between two orthonormal bases.

    0 degrees on every angle means one subspace is contained in the other; 90
    means orthogonal. This is how we ask whether the metric is read off the
    belief or computed beside it.
    """
    if a.shape[1] == 0 or b.shape[1] == 0:
        return np.zeros(0)
    s = np.linalg.svd(a.T @ b, compute_uv=False)
    return np.degrees(np.arccos(np.clip(s, -1.0, 1.0)))


def _demo() -> None:
    from .transformer import ModelConfig

    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=12, n_ctx=16, n_layers=2, n_heads=1, d_model=32, d_mlp=64)
    model = TinyTransformer(cfg).eval()
    tokens = torch.randint(0, cfg.vocab_size, (8, cfg.n_ctx))

    with torch.no_grad():
        reference = float(model.loss(tokens))
    base = ablated_loss(model, tokens, 0, None)
    assert abs(base - reference) < 1e-5, "no basis must be a no-op"

    # A basis of rank 0 cannot change anything, at any depth.
    empty = np.zeros((cfg.d_model, 0))
    for depth in range(cfg.n_layers + 1):
        assert abs(ablated_loss(model, tokens, depth, empty) - base) < 1e-5, depth

    # Ablating the whole space must collapse the stream onto the mean exactly --
    # this is the projection identity the causal claim rests on.
    mean = rng_mean = model.residual_streams(tokens)[1].reshape(-1, cfg.d_model).mean(axis=0)
    collapsed = ablated_stream(model, tokens, 1, np.eye(cfg.d_model), rng_mean)
    assert np.allclose(collapsed, mean, atol=1e-5), "full ablation must leave only the mean"

    # Rank detection: a block whose columns sum to a constant is rank-deficient.
    rng = np.random.default_rng(0)
    w = rng.normal(size=(cfg.d_model, 4))
    w[:, -1] = -w[:, :-1].sum(axis=1)  # columns now sum to zero
    assert probe_basis(w, slice(0, 4)).shape[1] == 3, "simplex block must lose one rank"

    q = random_basis(rng, cfg.d_model, 5)
    assert np.allclose(q.T @ q, np.eye(5), atol=1e-10), "control basis must be orthonormal"

    sub = q[:, :2]
    # arccos loses half its precision near 1, so a contained subspace reads as a
    # few 1e-6 degrees rather than exactly zero.
    assert principal_angles(sub, q).max() < 1e-3, "contained subspace has zero angles"
    assert principal_angles(q[:, :2], q[:, 2:]).min() > 89.9, "disjoint blocks are orthogonal"
    print("ablation ok")


if __name__ == "__main__":
    _demo()
