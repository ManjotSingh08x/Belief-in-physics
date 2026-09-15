"""Exact predictive-belief geometry of the shared Mess-4 chain.

All four physical experiments use this same chain, so there is one exact
geometric target: the reachable set of P(mood_{t+1} | letters_1..t) inside a
regular tetrahedron. The physics does not enter this calculation.

Run:  uv run python scripts/messk_geometry.py
"""

from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from physics.messk import (  # noqa: E402
    MessKProcess,
    simplex_embedding,
    square_projection_vertices,
)

RESULT_OUT = Path("experiments/outputs-messk")
FIGURE_OUT = Path("figures/messk")
N_SEQ, N_TICKS, BURN = 400, 520, 20
MAX_POINTS = 60_000
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#898781"
EDGE = "#c3c2b7"
STATE_COLOURS = ("#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7")


def collect(seed: int = 0) -> dict:
    chain = MessKProcess(n_states=4, alpha=0.7, stay=0.7)
    rng = np.random.default_rng(seed)
    states, letters = chain.sample(rng, N_SEQ, N_TICKS)
    beliefs = chain.beliefs(letters)[:, BURN:].reshape(-1, 4)
    truth = states[:, 1:][:, BURN:].reshape(-1)
    entropy = -(beliefs * np.log(np.clip(beliefs, 1e-12, None))).sum(1) / np.log(4)
    return {
        "n_states": 4,
        "alpha": chain.alpha,
        "stay": chain.stay,
        "x": chain.x,
        "memory_letters": chain.memory_length(),
        "mood_recovery": float((beliefs.argmax(1) == truth).mean()),
        "chance": 0.25,
        "entropy_normalised": float(entropy.mean()),
        "n_points": int(len(beliefs)),
        "_beliefs": beliefs,
        "_truth": truth,
    }


def _sample(data: dict) -> tuple[np.ndarray, np.ndarray]:
    step = max(1, data["n_points"] // MAX_POINTS)
    vertices = simplex_embedding(4)
    return data["_beliefs"][::step] @ vertices, data["_truth"][::step]


def _sample_planar(data: dict) -> tuple[np.ndarray, np.ndarray]:
    step = max(1, data["n_points"] // MAX_POINTS)
    points = data["_beliefs"][::step] @ square_projection_vertices()
    return points, data["_truth"][::step]


def _edges_3d(ax, vertices: np.ndarray) -> None:
    for i, j in combinations(range(4), 2):
        ax.plot(*zip(vertices[i], vertices[j]), color=EDGE, lw=0.7, zorder=1)
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()


def _cloud_3d(ax, points: np.ndarray, labels: np.ndarray, view: tuple[int, int]) -> None:
    vertices = simplex_embedding(4)
    for state in range(4):
        mask = labels == state
        ax.scatter(*points[mask].T, c=STATE_COLOURS[state], s=0.45, alpha=0.22,
                   linewidths=0, depthshade=False, rasterized=True)
    _edges_3d(ax, vertices)
    ax.view_init(elev=view[0], azim=view[1])


def _cloud_2d(ax, points: np.ndarray, labels: np.ndarray, dims: tuple[int, int]) -> None:
    vertices = simplex_embedding(4)
    for i, j in combinations(range(4), 2):
        ax.plot(*zip(vertices[i, list(dims)], vertices[j, list(dims)]), color=EDGE, lw=0.7)
    for state in range(4):
        mask = labels == state
        ax.scatter(points[mask, dims[0]], points[mask, dims[1]], c=STATE_COLOURS[state],
                   s=0.45, alpha=0.22, linewidths=0, rasterized=True)
    ax.set_aspect("equal")
    ax.axis("off")


def _cloud_square(ax, points: np.ndarray, labels: np.ndarray) -> None:
    vertices = square_projection_vertices()
    ring = np.vstack([vertices, vertices[0]])
    ax.plot(ring[:, 0], ring[:, 1], color=MUTED, lw=1.0)
    ax.plot([-1, 1], [-1, 1], color=EDGE, lw=0.55)
    ax.plot([-1, 1], [1, -1], color=EDGE, lw=0.55)
    for state in range(4):
        mask = labels == state
        ax.scatter(*points[mask].T, c=STATE_COLOURS[state], s=0.55, alpha=0.24,
                   linewidths=0, rasterized=True)
    align = (("right", "top"), ("left", "top"), ("left", "bottom"), ("right", "bottom"))
    for state, ((x, y), (ha, va)) in enumerate(zip(vertices, align)):
        ax.text(x + (0.04 if x > 0 else -0.04), y + (0.04 if y > 0 else -0.04),
                f"state {state}", ha=ha, va=va, fontsize=9, color=INK)
    ax.set(xlim=(-1.16, 1.16), ylim=(-1.16, 1.16))
    ax.set_aspect("equal")
    ax.axis("off")


def main() -> None:
    RESULT_OUT.mkdir(parents=True, exist_ok=True)
    FIGURE_OUT.mkdir(parents=True, exist_ok=True)
    data = collect()
    points, labels = _sample(data)

    print(
        f"Mess-4 alpha={data['alpha']} stay={data['stay']} x={data['x']:.3f}  "
        f"memory={data['memory_letters']} letters  mood recovery={data['mood_recovery']:.3f} "
        f"({data['mood_recovery'] / data['chance']:.2f}x chance)  "
        f"entropy={data['entropy_normalised']:.3f}  points={data['n_points']:,}",
        flush=True,
    )

    fig = plt.figure(figsize=(14, 8.6), facecolor=SURFACE)
    fig.suptitle("Exact Mess-4 predictive-belief geometry", y=0.985, fontsize=15, color=INK)
    fig.text(
        0.5, 0.955,
        "the same tetrahedral target is used for pendulum, predator-prey, sphere, and double pendulum",
        ha="center", va="top", fontsize=9, color=MUTED,
    )
    fig.legend(
        handles=[
            Line2D([], [], marker="o", linestyle="none", markersize=5,
                   color=STATE_COLOURS[state], label=f"true next mood {state}")
            for state in range(4)
        ],
        loc="upper center", bbox_to_anchor=(0.5, 0.935), ncol=4,
        frameon=False, fontsize=8,
    )

    for index, view in enumerate(((18, 35), (18, 125), (65, 45)), 1):
        ax = fig.add_subplot(2, 3, index, projection="3d", facecolor=SURFACE)
        _cloud_3d(ax, points, labels, view)
        ax.set_title(f"3D view {index}", fontsize=9, color=INK)

    for index, dims in enumerate(((0, 1), (0, 2), (1, 2)), 4):
        ax = fig.add_subplot(2, 3, index, facecolor=SURFACE)
        _cloud_2d(ax, points, labels, dims)
        ax.set_title(f"flattened {'xyz'[dims[0]]}{'xyz'[dims[1]]}", fontsize=9, color=INK)

    fig.text(
        0.5, 0.015,
        f"memory {data['memory_letters']} letters   |   mood recovery "
        f"{data['mood_recovery']:.3f} vs {data['chance']:.2f} chance   |   "
        f"normalised entropy {data['entropy_normalised']:.3f}",
        ha="center", fontsize=9, color=INK,
    )
    fig.tight_layout(rect=(0.01, 0.04, 0.99, 0.90))

    figure_path = FIGURE_OUT / "mess4_belief_geometry.png"
    fig.savefig(figure_path, dpi=180, facecolor=SURFACE)
    plt.close(fig)

    planar_points, planar_labels = _sample_planar(data)
    planar_fig, planar_ax = plt.subplots(figsize=(8.2, 8.2), facecolor=SURFACE)
    planar_ax.set_facecolor(SURFACE)
    _cloud_square(planar_ax, planar_points, planar_labels)
    planar_fig.suptitle("Exact Mess-4 belief geometry: planar square projection",
                        y=0.975, fontsize=15, color=INK)
    planar_fig.text(
        0.5, 0.94,
        "four pure states are square corners; this readable 2-D view collapses one belief dimension",
        ha="center", va="top", fontsize=9, color=MUTED,
    )
    planar_fig.legend(
        handles=[
            Line2D([], [], marker="o", linestyle="none", markersize=5,
                   color=STATE_COLOURS[state], label=f"true next mood {state}")
            for state in range(4)
        ],
        loc="lower center", bbox_to_anchor=(0.5, 0.02), ncol=4,
        frameon=False, fontsize=8,
    )
    planar_fig.tight_layout(rect=(0.02, 0.06, 0.98, 0.91))
    planar_path = FIGURE_OUT / "mess4_belief_geometry_planar.png"
    planar_fig.savefig(planar_path, dpi=180, facecolor=SURFACE)
    plt.close(planar_fig)

    record = {key: value for key, value in data.items() if not key.startswith("_")}
    (RESULT_OUT / "mess4_geometry.json").write_text(json.dumps(record, indent=2, default=float))
    print(f"wrote {figure_path} and {planar_path}")


if __name__ == "__main__":
    main()
