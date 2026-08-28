"""Demonstration figures per system: phase portrait, token raster, belief
heatmap, separability curves, energy/damping sanity check.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .factory import Simulator
from .separability import predictive_obs_curve, total_variation


def _marginalize(beliefs: np.ndarray, bin_counts: tuple[int, ...], keep_dim: int) -> np.ndarray:
    """(L, n_latent) -> (L, bin_counts[keep_dim]) by summing out every other dim.
    A full (n_latent,) belief renders as an invisible single-pixel streak in a
    heatmap when n_latent is in the tens of thousands; this projects onto the
    dimension the observable actually reads, which is legible at figure size.
    """
    L = beliefs.shape[0]
    reshaped = beliefs.reshape((L, *bin_counts))
    other_axes = tuple(1 + d for d in range(len(bin_counts)) if d != keep_dim)
    return reshaped.sum(axis=other_axes)


def make_figure(
    name: str, sim: Simulator, out_dir: Path, n_show: int = 1, N: int = 60, M: int = 6, marginal_dim: int = 0
) -> Path:
    hmm = sim.hmm
    ep = sim.simulate_batch(num_simulations=max(4, n_show), N=N, M=M, with_beliefs=True)

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle(f"{name} -- Phase 1 demonstration")

    # 1. metric trajectory with perturbation events marked
    ax = axes[0, 0]
    metric0 = ep.metrics[0]
    for d in range(metric0.shape[1]):
        ax.plot(metric0[:, d], label=f"metric[{d}]", linewidth=1.2)
    action_pos = np.flatnonzero(ep.is_action[0])
    for p in action_pos:
        ax.axvline(p, color="red", alpha=0.25, linewidth=1)
    ax.set_title("metric trajectory (red = perturbation)")
    ax.set_xlabel("token position")
    ax.legend(fontsize=8)

    # 2. token raster: observation tokens vs action tokens
    ax = axes[0, 1]
    tok = ep.tokens[0]
    is_act = ep.is_action[0]
    ax.scatter(np.flatnonzero(~is_act), tok[~is_act], s=8, label="observation", color="tab:blue")
    ax.scatter(np.flatnonzero(is_act), tok[is_act], s=20, marker="x", label="action", color="tab:red")
    ax.set_title("token raster")
    ax.set_xlabel("position")
    ax.set_ylabel("token id")
    ax.legend(fontsize=8)

    # 3. belief heatmap, marginalised onto the observable's own dimension,
    # vs time, true latent (same dimension) overlaid
    ax = axes[1, 0]
    beliefs = ep.beliefs[0]  # (L, n_latent)
    marginal = _marginalize(beliefs, hmm.grid.bin_counts, marginal_dim)  # (L, bin_counts[marginal_dim])
    true_per_dim = np.empty(ep.latent.shape[1], dtype=np.int64)
    rem = ep.latent[0].copy()
    for d in reversed(range(hmm.grid.ndim)):
        digit = rem % hmm.grid.bin_counts[d]
        rem = rem // hmm.grid.bin_counts[d]
        if d == marginal_dim:
            true_per_dim = digit
    im = ax.imshow(marginal.T, aspect="auto", origin="lower", cmap="magma")
    ax.plot(np.arange(marginal.shape[0]), true_per_dim, color="cyan", linewidth=1, label="true bin (this dim)")
    ax.set_title(f"posterior marginal on latent dim {marginal_dim} (true state overlaid)")
    ax.set_xlabel("position")
    ax.set_ylabel(f"latent bin (dim {marginal_dim})")
    ax.legend(fontsize=8, loc="upper right")
    fig.colorbar(im, ax=ax, fraction=0.046)

    # 4. separability: TV(action vs noop) at first post-kick observation, all actions.
    # Neither a uniform belief (invariant under a near-permutation action) nor
    # this one episode's realised belief (which can land on a domain-boundary
    # clip, e.g. velocity already saturated at omega_max, where a further kick
    # is a no-op) is a representative starting point -- average over several
    # random-bin-then-a-few-flow-steps beliefs instead, matching
    # scripts/validate_simulators.py's own sampling.
    ax = axes[1, 1]
    rng = np.random.default_rng(0)
    belief0s = []
    for _ in range(6):
        b = np.zeros(hmm.n_latent)
        b[rng.integers(0, hmm.n_latent)] = 1.0
        for _ in range(2):
            b = b @ hmm.T
        belief0s.append(b)
    noop_curves = [predictive_obs_curve(hmm, b, None, n_steps=5) for b in belief0s]
    for a, aname in enumerate(hmm.action_names):
        tv = np.mean(
            [total_variation(predictive_obs_curve(hmm, b, a, n_steps=5), nc) for b, nc in zip(belief0s, noop_curves)],
            axis=0,
        )
        ax.plot(tv, marker="o", label=aname)
    ax.axhline(0.15, color="gray", linestyle="--", linewidth=1, label="first-step threshold")
    ax.set_title("TV(action, no-op) over post-kick steps")
    ax.set_xlabel("steps after perturbation")
    ax.set_ylabel("total variation")
    ax.legend(fontsize=7)

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path
