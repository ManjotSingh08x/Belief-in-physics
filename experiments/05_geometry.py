"""Transformer-predicted belief geometry over training time and model depth.

For every saved checkpoint, fit a linear map on held-out-sequence training data
from each residual-stream depth to the K-1 independent coordinates of the exact
belief simplex. Plot those predictions on a disjoint set of sequences, without
clipping or projecting them back into the simplex. A cloud outside the triangle
or tetrahedron is therefore a visible failure rather than something the plotting
code silently repairs.

Each output matrix has log-spaced training checkpoints down the rows and the
embedding plus all transformer blocks across the columns, so the figure itself
does not collapse layers. Any best-depth summary is selected on validation
sequences and reports the separate test score. The pale cloud is the
exact reachable belief set on the same held-out points. Predicted points are
coloured by the dominant state of the exact belief, not by a quantity inferred
from the transformer.

Run:  uv run python experiments/05_geometry.py
Env:  OUTPUT_DIR, FIGURE_DIR, CONFIGS, N_EVAL, MAX_POINTS, PROJECTION.
Set `PROJECTION=square` for a readable planar view; it is intentionally lossy.
"""

from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D

from models.analysis import residual_streams_batched
from models.bootstrap import r2_columns
from models.probe import fit_probe
from models.train import pick_device
from models.transformer import ModelConfig, TinyTransformer
from physics.messk import simplex_embedding, square_projection_vertices
from physics.messk_configs import MESSK_CONFIGS, make_process

N_EVAL = int(os.environ.get("N_EVAL", 256))
MAX_POINTS = int(os.environ.get("MAX_POINTS", 2_500))
TRAIN_FRAC = 0.6
VALIDATION_FRAC = 0.2
EVAL_SEED = 20_260_829
SPLIT_SEED = 0

OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "experiments/outputs-messk"))
FIGURE_DIR = Path(os.environ.get("FIGURE_DIR", "figures/transformer-belief"))
CONFIGS = os.environ.get("CONFIGS", ",".join(MESSK_CONFIGS)).split(",")
PROJECTION = os.environ.get("PROJECTION", "tetrahedron")

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
REFERENCE = "#c3c2b7"
STATE_COLOURS = ("#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7")


def _load(path: Path, config: ModelConfig, device: str) -> TinyTransformer:
    model = TinyTransformer(config)
    model.load_state_dict(torch.load(path, map_location=device))
    return model.to(device).eval()


def _flat(array: np.ndarray, indices: np.ndarray) -> np.ndarray:
    return array[indices].reshape(-1, array.shape[-1]).astype(np.float64)


def _format_tokens(tokens: int) -> str:
    if tokens == 0:
        return "untrained"
    if tokens >= 1_000_000:
        value = tokens / 1_000_000
        return f"{value:g}M tokens"
    return f"{tokens / 1_000:g}k tokens"


def _square_coords(coords: np.ndarray, simplex_vertices: np.ndarray) -> np.ndarray:
    barycentric = 1.0 / len(simplex_vertices) + coords @ np.linalg.pinv(simplex_vertices)
    return barycentric @ square_projection_vertices()


def _set_square_style(ax) -> None:
    vertices = square_projection_vertices()
    ring = np.vstack([vertices, vertices[0]])
    ax.plot(ring[:, 0], ring[:, 1], color=MUTED, lw=0.8, zorder=1)
    ax.plot([-1, 1], [-1, 1], color=GRID, lw=0.45, zorder=0)
    ax.plot([-1, 1], [1, -1], color=GRID, lw=0.45, zorder=0)
    ax.set_xlim(-1.12, 1.12)
    ax.set_ylim(-1.12, 1.12)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _set_2d_style(ax, vertices: np.ndarray) -> None:
    ring = np.vstack([vertices, vertices[0]])
    ax.plot(ring[:, 0], ring[:, 1], color=MUTED, lw=0.8, zorder=1)
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-0.85, 1.22)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _set_3d_style(ax, vertices: np.ndarray) -> None:
    for i, j in combinations(range(4), 2):
        ax.plot(*zip(vertices[i], vertices[j]), color=MUTED, lw=0.65, zorder=1)
    ax.set(xlim=(-0.82, 0.82), ylim=(-0.82, 0.82), zlim=(-0.82, 0.82))
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=19, azim=37)
    ax.set_axis_off()


def _draw_cloud(
    ax, exact: np.ndarray, predicted: np.ndarray, labels: np.ndarray, k: int,
    projection: str,
) -> None:
    vertices = simplex_embedding(k)
    if projection == "square":
        exact_square = _square_coords(exact, vertices)
        predicted_square = _square_coords(predicted, vertices)
        ax.scatter(*exact_square.T, s=2.0, c=REFERENCE, alpha=0.14,
                   linewidths=0, rasterized=True)
        for state in range(k):
            mask = labels == state
            ax.scatter(*predicted_square[mask].T, s=3.0, c=STATE_COLOURS[state],
                       alpha=0.30, linewidths=0, rasterized=True)
        _set_square_style(ax)
        return
    if k == 3:
        ax.scatter(exact[:, 0], exact[:, 1], s=2.0, c=REFERENCE, alpha=0.12, linewidths=0, rasterized=True)
        for state in range(k):
            mask = labels == state
            ax.scatter(*predicted[mask].T, s=3.0, c=STATE_COLOURS[state], alpha=0.28,
                       linewidths=0, rasterized=True)
        _set_2d_style(ax, vertices)
        return

    ax.scatter(*exact.T, s=1.4, c=REFERENCE, alpha=0.09, linewidths=0, depthshade=False, rasterized=True)
    for state in range(k):
        mask = labels == state
        ax.scatter(*predicted[mask].T, s=2.2, c=STATE_COLOURS[state], alpha=0.24,
                   linewidths=0, depthshade=False, rasterized=True)
    _set_3d_style(ax, vertices)


def _prediction_record(predicted: np.ndarray, target: np.ndarray, vertices: np.ndarray) -> dict:
    barycentric = 1.0 / vertices.shape[0] + predicted @ np.linalg.pinv(vertices)
    negative = np.minimum(barycentric, 0.0)
    return {
        "r2": float(np.mean(r2_columns(predicted, target))),
        "outside_simplex_fraction": float(np.any(barycentric < -1e-8, axis=1).mean()),
        "mean_negative_mass": float(-negative.sum(axis=1).mean()),
    }


def _legend(k: int) -> list[Line2D]:
    handles = [
        Line2D([], [], marker="o", linestyle="none", markersize=5, color=REFERENCE,
               label="exact reachable set")
    ]
    handles.extend(
        Line2D([], [], marker="o", linestyle="none", markersize=5,
               color=STATE_COLOURS[state], label=f"dominant belief state {state}")
        for state in range(k)
    )
    return handles


def _make_axes(n_rows: int, n_depths: int, k: int, projection_mode: str):
    fig = plt.figure(figsize=(3.05 * n_depths, 2.45 * n_rows + 1.3), facecolor=SURFACE)
    axes = []
    for row in range(n_rows):
        row_axes = []
        for col in range(n_depths):
            index = row * n_depths + col + 1
            axes_projection = "3d" if k == 4 and projection_mode == "tetrahedron" else None
            row_axes.append(
                fig.add_subplot(
                    n_rows, n_depths, index,
                    projection=axes_projection, facecolor=SURFACE,
                )
            )
        axes.append(row_axes)
    return fig, axes


def _plot_config(name: str, training: dict, device: str) -> dict:
    t0 = time.perf_counter()
    proc = make_process(name)
    batch = proc.sample_batch(np.random.default_rng(EVAL_SEED), N_EVAL)
    k = proc.chain.n_states
    vertices = simplex_embedding(k)
    target = batch["beliefs"] @ vertices

    order = np.random.default_rng(SPLIT_SEED).permutation(N_EVAL)
    train_end = int(TRAIN_FRAC * N_EVAL)
    validation_end = train_end + int(VALIDATION_FRAC * N_EVAL)
    train_idx = order[:train_end]
    validation_idx = order[train_end:validation_end]
    test_idx = order[validation_end:]
    y_train = _flat(target, train_idx)
    y_validation = _flat(target, validation_idx)
    y_test = _flat(target, test_idx)
    test_beliefs = _flat(batch["beliefs"], test_idx)

    rng = np.random.default_rng(EVAL_SEED + 1)
    chosen = np.sort(rng.choice(len(y_test), size=min(MAX_POINTS, len(y_test)), replace=False))
    exact_plot = y_test[chosen]
    labels = test_beliefs[chosen].argmax(axis=1)

    config = ModelConfig(**training[name]["model"])
    axis = sorted(training[name]["checkpoint_tokens"])
    n_depths = config.n_layers + 1
    fig, axes = _make_axes(len(axis), n_depths, k, PROJECTION)
    rows = []

    for row, seen in enumerate(axis):
        model = _load(OUTPUT_DIR / "checkpoints" / f"{name}_{seen}.pt", config, device)
        streams = residual_streams_batched(model, batch["tokens"], device)
        depth_records = []
        for depth, activations in enumerate(streams):
            probe = fit_probe(_flat(activations, train_idx), y_train)
            predicted_validation = probe(_flat(activations, validation_idx))
            predicted = probe(_flat(activations, test_idx))
            record = _prediction_record(predicted, y_test, vertices)
            record["r2_validation"] = float(
                np.mean(r2_columns(predicted_validation, y_validation))
            )
            record |= {"depth": depth, "name": "embedding" if depth == 0 else f"resid_post_{depth - 1}"}
            depth_records.append(record)

            ax = axes[row][depth]
            _draw_cloud(ax, exact_plot, predicted[chosen], labels, k, PROJECTION)
            annotation = f"R² {record['r2']:+.2f}  outside {record['outside_simplex_fraction']:.0%}"
            if PROJECTION == "tetrahedron" and k == 4:
                ax.text2D(0.02, 0.02, annotation, transform=ax.transAxes, fontsize=7, color=INK)
            else:
                ax.text(0.02, 0.02, annotation, transform=ax.transAxes, fontsize=7, color=INK)
            if row == 0:
                ax.set_title("embedding" if depth == 0 else f"residual after block {depth}",
                             fontsize=9, color=INK, pad=5)

        best = max(depth_records, key=lambda record: record["r2_validation"])
        rows.append({"tokens_seen": seen, "best_depth": best["name"], "depths": depth_records})
        print(
            f"  {seen:>12,}  validation-selected {best['name']:12s}  "
            f"test R2 {best['r2']:+.3f}", flush=True,
        )

    height = fig.get_figheight()
    top = 1.0 - 0.72 / height
    bottom = 0.42 / height
    fig.subplots_adjust(left=0.04, right=0.995, top=top, bottom=bottom, hspace=0.08, wspace=0.04)
    for row, seen in enumerate(axis):
        bounds = axes[row][0].get_position()
        fig.text(0.008, (bounds.y0 + bounds.y1) / 2, _format_tokens(seen),
                 ha="left", va="center", fontsize=8, color=INK, rotation=90)

    system_name = name.rsplit("_mess", 1)[0].replace("_", " ").title()
    chain_name = f"Mess-{k}"
    view_name = "planar square projection" if PROJECTION == "square" else "tetrahedral view"
    fig.suptitle(f"{system_name} × {chain_name}: {view_name}",
                 x=0.5, y=1.0 - 0.10 / height, fontsize=15, color=INK)
    subtitle = (
        "lossy 2-D display only; probe scores remain in the faithful 3-D simplex"
        if PROJECTION == "square"
        else "held-out linear-probe predictions; pale points are the exact reachable set"
    )
    fig.text(0.5, 1.0 - 0.40 / height, subtitle,
             ha="center", va="top", fontsize=9, color=MUTED)
    fig.legend(handles=_legend(k), loc="lower center", ncol=k + 1, frameon=False,
               bbox_to_anchor=(0.5, 0.005), fontsize=8)

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "_planar" if PROJECTION == "square" else ""
    figure_path = FIGURE_DIR / f"{name}_checkpoint_layers{suffix}.png"
    fig.savefig(figure_path, dpi=180, facecolor=SURFACE)
    plt.close(fig)
    return {
        "n_eval": N_EVAL,
        "n_train_sequences": int(len(train_idx)),
        "n_validation_sequences": int(len(validation_idx)),
        "n_test_sequences": int(len(test_idx)),
        "n_plot_points": int(len(chosen)),
        "projection": PROJECTION,
        "figure": str(figure_path),
        "checkpoints": rows,
        "wall_seconds": time.perf_counter() - t0,
    }


def _demo() -> None:
    vertices = simplex_embedding(4)
    target = np.array([
        [0.7, 0.1, 0.1, 0.1],
        [0.1, 0.7, 0.1, 0.1],
        [0.1, 0.1, 0.7, 0.1],
        [0.1, 0.1, 0.1, 0.7],
    ]) @ vertices
    inside = _prediction_record(target, target, vertices)
    assert inside["r2"] > 0.99 and inside["outside_simplex_fraction"] == 0.0
    pure_states = np.eye(4) @ vertices
    assert np.allclose(_square_coords(pure_states, vertices), square_projection_vertices())
    assert np.allclose(_square_coords(np.zeros((1, 3)), vertices), 0.0)
    outside = _prediction_record(np.full_like(target, 2.0), target, vertices)
    assert outside["outside_simplex_fraction"] == 1.0


def main() -> None:
    _demo()
    if PROJECTION not in {"tetrahedron", "square"}:
        raise SystemExit("PROJECTION must be 'tetrahedron' or 'square'")
    device = pick_device()
    training = json.loads((OUTPUT_DIR / "messk_01_training.json").read_text())
    results = {}
    print(
        f"device={device}  projection={PROJECTION}  configs={CONFIGS}  "
        f"n_eval={N_EVAL}  max_points={MAX_POINTS}", flush=True,
    )
    for name in CONFIGS:
        print(f"\n=== {name} ===", flush=True)
        results[name] = _plot_config(name, training, device)

    result_name = "messk_05_geometry_planar.json" if PROJECTION == "square" else "messk_05_geometry.json"
    path = OUTPUT_DIR / result_name
    path.write_text(json.dumps(results, indent=2, default=float))
    print(f"\nwrote {path}", flush=True)


if __name__ == "__main__":
    main()
