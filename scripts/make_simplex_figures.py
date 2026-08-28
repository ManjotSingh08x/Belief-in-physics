"""Belief-simplex geometry, one figure per system.

The object plotted is `action_lag0`: the exact posterior over which perturbation
just happened. For the pendulum (3 actions) that is a point in the 2-simplex and
the triangle is faithful. The other systems have 4 or 5 actions, so the triangle
becomes a lossy projection -- panel 2 carries the honest view (PCA with its
explained variance stated) and no quantitative claim is read off panel 1.

Run:  uv run python scripts/make_simplex_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from physics.branch import feature_groups, forward_features, simplex_to_2d
from physics.branch_configs import BRANCH_CONFIGS, make_branch_process

N_SEQ = 256
SEED = 20_260_828
FIGDIR = Path(__file__).resolve().parent.parent / "figures"


def _simplex_points(process, tokens):
    features = forward_features(process, tokens)
    return features[:, :, feature_groups(process)["action_lag0"]]


def _entropy(p):
    return -(p * np.log(np.clip(p, 1e-12, None))).sum(-1) / np.log(p.shape[-1])


def _pca_2d(points):
    centred = points - points.mean(0)
    _, s, vt = np.linalg.svd(centred, full_matrices=False)
    var = s**2 / (s**2).sum()
    return centred @ vt[:2].T, var[:2].sum()


def make_figure(name: str) -> Path:
    process = make_branch_process(name)
    episodes = process.sample_batch(np.random.default_rng(SEED), N_SEQ)
    simplex = _simplex_points(process, episodes.tokens)  # (n, L, n_actions)
    a = process.n_actions

    flat = simplex.reshape(-1, a)
    # Colour by the perturbation that actually happened, so the figure shows
    # whether the geometry separates the hidden cause rather than just spreading.
    truth = np.repeat(episodes.actions, process.steps_per_segment, axis=1).reshape(-1)
    entropy = _entropy(flat)

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 10))
    fig.suptitle(
        f"{name}  -  belief simplex over the hidden perturbation\n"
        f"{a} actions, {process.n_z0} initial conditions, "
        f"{process.n_branches(process.M):,} branches, seq_len {process.seq_len}",
        fontsize=12,
    )
    cmap = plt.get_cmap("viridis", a)

    # 1. the simplex itself
    ax = axes[0, 0]
    xy = simplex_to_2d(flat)
    ax.scatter(xy[:, 0], xy[:, 1], c=truth, cmap=cmap, s=1.5, alpha=0.35, vmin=-0.5, vmax=a - 0.5)
    corners = simplex_to_2d(np.eye(a))
    ax.plot(*np.vstack([corners, corners[:1]]).T, color="0.3", lw=0.8)
    for vertex, label in zip(corners, process.action_names):
        ax.annotate(label, vertex * 1.12, ha="center", va="center", fontsize=9)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(
        "simplex projection" + ("" if a == 3 else f"  (LOSSY: {a - 1}-simplex to 2D)"), fontsize=10
    )

    # 2. the honest view for n > 3
    ax = axes[0, 1]
    proj, explained = _pca_2d(flat)
    ax.scatter(proj[:, 0], proj[:, 1], c=truth, cmap=cmap, s=1.5, alpha=0.35, vmin=-0.5, vmax=a - 0.5)
    ax.set_title(f"PCA of the same points  ({explained:.1%} of variance)", fontsize=10)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

    # 3. is the belief actually uncertain?
    ax = axes[1, 0]
    ax.hist(entropy, bins=60, color="#4c72b0")
    ax.axvline(entropy.mean(), color="crimson", lw=1.5, label=f"mean {entropy.mean():.3f}")
    ax.set_xlim(0, 1)
    ax.set_xlabel("normalised entropy of the belief  (0 = collapsed, 1 = uniform)")
    ax.set_ylabel("positions")
    ax.set_title("belief uncertainty -- a collapsed belief is a label, not a simplex", fontsize=10)
    ax.legend(fontsize=8)

    # 4. one sequence through time
    ax = axes[1, 1]
    for k in range(a):
        ax.plot(simplex[0, :, k], lw=1.4, color=cmap(k), label=process.action_names[k])
    for m in range(1, process.M):
        ax.axvline(m * process.steps_per_segment, color="0.75", lw=0.7, ls="--")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("token position  (dashed = a new hidden perturbation)")
    ax.set_ylabel("P(last perturbation)")
    ax.set_title("one sequence: the posterior resets and re-sharpens", fontsize=10)
    ax.legend(fontsize=8, ncol=min(a, 3))

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = FIGDIR / f"simplex_{name}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)

    recovery = (simplex[:, -1].argmax(1) == episodes.actions[:, -1]).mean()
    print(
        f"{name:16s} n_actions={a}  entropy={entropy.mean():.3f}  "
        f"recovery={recovery:.3f} (chance {1/a:.3f})  pca_var={explained:.3f}  -> {out.name}",
        flush=True,
    )
    return out


if __name__ == "__main__":
    FIGDIR.mkdir(exist_ok=True)
    for system in sys.argv[1:] or sorted(BRANCH_CONFIGS):
        make_figure(system)
