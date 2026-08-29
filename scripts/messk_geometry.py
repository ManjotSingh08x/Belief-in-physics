"""Belief geometry of the Mess-K chains, their way.

The belief here is P(mood | letters) from the chain's own forward algorithm, so
these pictures are properties of (n_states, alpha, stay) and owe nothing to the
pendulum. K=3 is included because it reproduces the reference triangle, which is
the only way to know the Mess-4 picture is being drawn correctly.

Run:  uv run python scripts/messk_geometry.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from physics.messk import MessKProcess, simplex_embedding  # noqa: E402

OUT = Path("experiments/outputs-messk")
N_SEQ, N_TICKS, BURN = 400, 520, 20  # burn-in so the cloud is the attractor, not the prior


def collect(k: int, alpha: float = 0.7, stay: float = 0.7, seed: int = 0) -> dict:
    chain = MessKProcess(n_states=k, alpha=alpha, stay=stay)
    rng = np.random.default_rng(seed)
    states, letters = chain.sample(rng, N_SEQ, N_TICKS)
    beliefs = chain.beliefs(letters)[:, BURN:].reshape(-1, k)
    truth = states[:, 1:][:, BURN:].reshape(-1)
    ent = -(beliefs * np.log(np.clip(beliefs, 1e-12, None))).sum(1) / np.log(k)
    return {
        "n_states": k, "alpha": alpha, "stay": stay, "x": chain.x,
        "memory_letters": chain.memory_length(),
        "mood_recovery": float((beliefs.argmax(1) == truth).mean()),
        "chance": 1.0 / k,
        "entropy_normalised": float(ent.mean()),
        "n_points": int(beliefs.shape[0]),
        "_b": beliefs, "_truth": truth,
    }


def _cloud(ax, d, view=None, dims=(0, 1)):
    xy = d["_b"] @ simplex_embedding(d["n_states"])
    colours = ["#E41A1C", "#4DAF4A", "#377EB8", "#FF8C00", "#984EA3"][: d["n_states"]]
    c = [colours[t] for t in d["_truth"][:: max(1, len(d["_truth"]) // 60000)]]
    step = max(1, len(d["_truth"]) // 60000)
    if view is not None:
        v = simplex_embedding(d["n_states"])
        for i in range(len(v)):
            for j in range(i + 1, len(v)):
                ax.plot(*zip(v[i], v[j]), color="0.85", lw=0.6, zorder=1)
        ax.scatter(xy[::step, 0], xy[::step, 1], xy[::step, 2], c=c, s=0.35, alpha=0.25, lw=0)
        ax.set_axis_off()
        ax.view_init(elev=view[0], azim=view[1])
    else:
        if d["n_states"] == 3:
            v = simplex_embedding(3)
            for i in range(3):
                ax.plot(*zip(v[i], v[(i + 1) % 3]), color="0.75", lw=1.0, zorder=1)
        ax.scatter(xy[::step, dims[0]], xy[::step, dims[1]], c=c, s=0.35, alpha=0.25, lw=0)
        ax.set_aspect("equal")
        ax.axis("off")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    m3, m4 = collect(3), collect(4)
    for d in (m3, m4):
        print(
            f"Mess-{d['n_states']}  alpha={d['alpha']} stay={d['stay']} x={d['x']:.3f}  "
            f"memory={d['memory_letters']} letters  "
            f"mood recovery={d['mood_recovery']:.3f} ({d['mood_recovery'] / d['chance']:.2f}x chance)  "
            f"entropy={d['entropy_normalised']:.3f}  points={d['n_points']:,}",
            flush=True,
        )

    fig = plt.figure(figsize=(14, 9))
    fig.suptitle(
        "Belief geometry, their way: P(mood | letters) for a K-mood chain\n"
        f"alpha={m3['alpha']} (a mood emits its own letter {m3['alpha']:.0%} of the time), "
        f"stay={m3['stay']} -- the physics does not enter",
        fontsize=11,
    )

    ax = fig.add_subplot(2, 3, 1)
    _cloud(ax, m3)
    ax.set_title(
        f"Mess-3, the reference triangle\n"
        f"memory {m3['memory_letters']} letters, recovery {m3['mood_recovery'] / m3['chance']:.1f}x chance",
        fontsize=9,
    )

    for i, view in enumerate(((18, 35), (18, 125))):
        ax = fig.add_subplot(2, 3, 2 + i, projection="3d")
        _cloud(ax, m4, view=view)
        ax.set_title(f"Mess-4 tetrahedron, view {i + 1}", fontsize=9)

    for i, dims in enumerate(((0, 1), (0, 2), (1, 2))):
        ax = fig.add_subplot(2, 3, 4 + i)
        _cloud(ax, m4, dims=dims)
        ax.set_title(f"Mess-4, flattened ({'xyz'[dims[0]]}{'xyz'[dims[1]]})", fontsize=9)

    fig.text(
        0.5, 0.02,
        f"Mess-4: memory {m4['memory_letters']} letters, "
        f"mood recovery {m4['mood_recovery']:.2f} vs {m4['chance']:.2f} chance "
        f"({m4['mood_recovery'] / m4['chance']:.1f}x), normalised entropy {m4['entropy_normalised']:.2f}",
        ha="center", fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.92))
    fig.savefig(OUT / "messk_belief_geometry.png", dpi=170)
    plt.close(fig)

    (OUT / "messk_geometry.json").write_text(
        json.dumps([{k: v for k, v in d.items() if not k.startswith("_")} for d in (m3, m4)],
                   indent=2, default=float)
    )
    print(f"wrote {OUT}/messk_belief_geometry.png")


if __name__ == "__main__":
    main()
