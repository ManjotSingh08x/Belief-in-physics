"""Figures for the checkpoint analysis written by experiments/06.

Run:  uv run python scripts/make_analysis_figures.py [results.json]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FIGDIR = ROOT / "figures" / "analysis"
DEFAULT = ROOT / "experiments" / "results-branch" / "phase4_checkpoint_analysis.json"

GROUPS = ("action_lag0", "z0", "metric")
LABEL = {"action_lag0": "belief (simplex)", "z0": "initial condition", "metric": "metric"}
COLOUR = {"action_lag0": "#c44e52", "z0": "#4c72b0", "metric": "#55a868"}


def _best(entry: dict) -> dict:
    return next(d for d in entry["by_depth"] if d["name"] == entry["best_depth"])


def _x(curve) -> np.ndarray:
    """Token counts, with the untrained checkpoint nudged onto a log axis."""
    tokens = np.array([c["tokens"] for c in curve], dtype=float)
    positive = tokens[tokens > 0]
    tokens[tokens == 0] = positive.min() / 3 if positive.size else 1.0
    return tokens


def r2_vs_tokens(results: dict) -> Path:
    fig, axes = plt.subplots(1, len(results), figsize=(4.2 * len(results), 4.0), squeeze=False)
    fig.suptitle("R² of each belief quantity vs training tokens (probe at the best depth)", fontsize=12)

    for ax, (name, r) in zip(axes[0], results.items()):
        x = _x(r["curve"])
        for group in GROUPS:
            y = [_best(c)["r2_by_group"][group] for c in r["curve"]]
            ax.plot(x, y, "o-", ms=3.5, lw=1.6, color=COLOUR[group], label=LABEL[group])
        shuffled = [_best(c)["r2_shuffled_control"] for c in r["curve"]]
        ax.plot(x, shuffled, "--", lw=1.0, color="0.55", label="shuffled control")
        ax.axvline(x[0], color="0.85", lw=6, zorder=0)
        ax.set_xscale("log")
        ax.set_xlabel("tokens seen  (leftmost = untrained)")
        ax.set_title(name, fontsize=10)
        ax.set_ylim(-0.05, 1.0)
        ax.grid(alpha=0.25)
    axes[0][0].set_ylabel("held-out R²")
    axes[0][-1].legend(fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = FIGDIR / "r2_vs_tokens.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def r2_vs_depth(results: dict) -> Path:
    fig, axes = plt.subplots(
        len(GROUPS), len(results), figsize=(3.6 * len(results), 3.1 * len(GROUPS)), squeeze=False
    )
    fig.suptitle("R² across residual-stream depth, one line per checkpoint", fontsize=12)
    cmap = plt.get_cmap("plasma")

    for col, (name, r) in enumerate(results.items()):
        x = _x(r["curve"])
        shade = np.log10(x)
        shade = (shade - shade.min()) / max(np.ptp(shade), 1e-9)
        for row, group in enumerate(GROUPS):
            ax = axes[row][col]
            for c, tone in zip(r["curve"], shade):
                depths = [d["name"] for d in c["by_depth"]]
                ax.plot(
                    range(len(depths)),
                    [d["r2_by_group"][group] for d in c["by_depth"]],
                    "o-", ms=3, lw=1.3, color=cmap(0.15 + 0.8 * tone),
                )
            ax.set_xticks(range(len(depths)))
            ax.set_xticklabels(["emb", *[f"L{i}" for i in range(len(depths) - 1)]], fontsize=8)
            ax.set_ylim(-0.05, 1.0)
            ax.grid(alpha=0.25)
            if row == 0:
                ax.set_title(name, fontsize=10)
            if col == 0:
                ax.set_ylabel(f"{LABEL[group]}\nR²", fontsize=9)

    fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap), ax=axes, fraction=0.02, pad=0.01,
        label="training progress (dark = untrained)",
    )
    out = FIGDIR / "r2_vs_depth.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def ablation_vs_tokens(results: dict) -> Path:
    fig, axes = plt.subplots(1, len(results), figsize=(4.2 * len(results), 4.0), squeeze=False)
    fig.suptitle(
        "Cost of deleting each quantity: Δ next-token loss above a random subspace of equal rank",
        fontsize=12,
    )

    for ax, (name, r) in zip(axes[0], results.items()):
        x = _x(r["curve"])
        for group in GROUPS:
            y, rank = [], None
            for c in r["curve"]:
                entry = next(a for a in c["ablation"] if a["name"] == c["best_depth"])
                cell = entry["groups"].get(group)
                y.append(np.nan if cell is None else cell["excess"])
                rank = rank if cell is None else cell["rank"]
            ax.plot(x, y, "o-", ms=3.5, lw=1.6, color=COLOUR[group],
                    label=f"{LABEL[group]}  (rank {rank})")
        ax.axhline(0, color="0.4", lw=1.0, ls="--")
        ax.set_xscale("log")
        ax.set_xlabel("tokens seen")
        ax.set_title(name, fontsize=10)
        ax.grid(alpha=0.25)
    axes[0][0].set_ylabel("excess Δ loss (nats)\n0 = no worse than deleting random directions")
    for ax in axes[0]:
        ax.legend(fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = FIGDIR / "ablation_vs_tokens.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def geometry(results: dict) -> Path:
    fig, axes = plt.subplots(2, len(results), figsize=(4.0 * len(results), 7.4), squeeze=False)
    fig.suptitle(
        "Subspace overlap\ntop: principal angles over training   "
        "bottom: R² retained after an ablation, final checkpoint",
        fontsize=11,
    )
    pairs = [f"{a}|{b}" for i, a in enumerate(GROUPS) for b in GROUPS[i + 1 :]]

    for col, (name, r) in enumerate(results.items()):
        ax = axes[0][col]
        x = _x(r["curve"])
        for pair, style in zip(pairs, ["-", "--", ":"]):
            y = [np.mean(c["geometry"]["principal_angles_deg"].get(pair, [np.nan])) for c in r["curve"]]
            ax.plot(x, y, style, lw=1.6, marker="o", ms=3,
                    label=pair.replace("action_lag0", "belief").replace("|", " vs "))
        ax.axhline(90, color="0.4", lw=1.0, ls="--")
        ax.set_xscale("log")
        ax.set_ylim(0, 95)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("tokens seen")
        ax.grid(alpha=0.25)
        if col == 0:
            ax.set_ylabel("mean principal angle (deg)\n0 = nested, 90 = independent")
        ax.legend(fontsize=7)

        ax = axes[1][col]
        cross = r["curve"][-1]["geometry"]["cross_r2"]
        intact = {g: _best(r["curve"][-1])["r2_by_group"][g] for g in GROUPS}
        matrix = np.array(
            [[cross[a][b] / intact[b] if intact[b] > 1e-6 and a in cross else np.nan for b in GROUPS]
             for a in GROUPS]
        )
        im = ax.imshow(matrix, vmin=0, vmax=1, cmap="RdYlGn")
        ax.set_xticks(range(3), [LABEL[g] for g in GROUPS], rotation=30, ha="right", fontsize=8)
        ax.set_yticks(range(3), [LABEL[g] for g in GROUPS], fontsize=8)
        for i in range(3):
            for j in range(3):
                if not np.isnan(matrix[i, j]):
                    ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=8)
        if col == 0:
            ax.set_ylabel("ablated")
        ax.set_xlabel("fraction of R² retained")
        fig.colorbar(im, ax=ax, fraction=0.046)

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = FIGDIR / "geometry.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def emergence_table(results: dict) -> str:
    """Tokens at which each group first reaches half its final R^2."""
    lines = ["| system | " + " | ".join(LABEL[g] for g in GROUPS) + " |",
             "|---|" + "---|" * len(GROUPS)]
    for name, r in results.items():
        cells = []
        for group in GROUPS:
            y = np.array([_best(c)["r2_by_group"][group] for c in r["curve"]])
            x = np.array([c["tokens"] for c in r["curve"]])
            target = y[-1] / 2
            hit = np.flatnonzero(y >= target)
            cells.append(f"{x[hit[0]]:,}" if hit.size and y[-1] > 0.02 else "n/a")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
    results = json.loads(path.read_text())
    FIGDIR.mkdir(parents=True, exist_ok=True)
    for fn in (r2_vs_tokens, r2_vs_depth, ablation_vs_tokens, geometry):
        print(f"wrote {fn(results).relative_to(ROOT)}", flush=True)
    print("\nhalf-of-final-R² crossing:\n" + emergence_table(results))


if __name__ == "__main__":
    main()
