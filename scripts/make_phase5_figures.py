"""Figures for phase 5. One per question, nothing decorative.

Run:  uv run python scripts/make_phase5_figures.py
Env:  OUTPUT_DIR.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", ROOT / "experiments/outputs-03"))
FIGDIR = ROOT / "figures" / "phase5"
COLOUR = {"action_lag0": "#c44e52", "z0": "#4c72b0", "metric": "#55a868"}
LABEL = {"action_lag0": "belief", "z0": "initial condition", "metric": "metric"}


def _load(name):
    path = OUTPUT_DIR / name
    return json.loads(path.read_text()) if path.exists() else None


def _logx(ax, tokens):
    t = np.array(tokens, dtype=float)
    pos = t[t > 0]
    t[t == 0] = pos.min() / 3 if pos.size else 1.0
    ax.set_xscale("log")
    return t


def fig_excess_loss(validity) -> Path:
    """V4/F3: the loss did or did not converge, measured against the Bayes floor."""
    systems = list(validity["systems"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    fig.suptitle("Excess loss over the Bayes floor. Raw cross entropy hides this: "
                 "its irreducible offset is most of the number.", fontsize=11)
    for name in systems:
        v = validity["systems"][name]
        curve = v["loss_curve"]
        t = np.array([c["tokens"] for c in curve], dtype=float)
        t[t == 0] = t[t > 0].min() / 3
        axes[0].plot(t, [c["loss"] for c in curve], "o-", ms=3, label=name)
        axes[1].plot(t, np.maximum([c["excess_loss"] for c in curve], 1e-4), "o-", ms=3, label=name)
        axes[0].axhline(v["bayes_floor"]["plugin"], ls=":", lw=0.9, color="0.5")
    axes[0].set_xscale("log"); axes[0].set_ylabel("next-token loss (nats)")
    axes[0].set_title("raw loss, dotted = each system's Bayes floor", fontsize=9)
    axes[1].set_xscale("log"); axes[1].set_yscale("log")
    axes[1].set_ylabel("loss - Bayes floor (nats)")
    axes[1].set_title("excess loss", fontsize=9)
    for ax in axes:
        ax.set_xlabel("tokens seen"); ax.grid(alpha=0.25); ax.legend(fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out = FIGDIR / "excess_loss.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    return out


def fig_recovery_phase(validity) -> Path:
    """C21: when the information about a kick actually arrives."""
    systems = list(validity["systems"])
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for name in systems:
        c = validity["systems"][name]["c21_recovery_by_phase"]
        ax.plot(c["observations_since_kick"], np.array(c["oracle_recovery"]) - c["chance"],
                "o-", ms=4, label=f"{name} (chance {c['chance']:.2f})")
    ax.axhline(0, color="0.4", lw=1.0, ls="--")
    ax.set_xlabel("observations available since the kick")
    ax.set_ylabel("oracle recovery above chance")
    ax.set_title("When the hidden perturbation becomes identifiable at all\n"
                 "(the exact posterior, no model involved)", fontsize=10)
    ax.grid(alpha=0.25); ax.legend(fontsize=8)
    fig.tight_layout()
    out = FIGDIR / "recovery_vs_phase.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    return out


def fig_ablation_grid(grid) -> Path:
    """F1/R1: excess damage as a (depth x tokens) surface, per quantity."""
    systems = list(grid)
    groups = ("action_lag0", "z0", "metric")
    fig, axes = plt.subplots(len(groups), len(systems),
                             figsize=(3.4 * len(systems), 2.9 * len(groups)), squeeze=False)
    fig.suptitle("Excess Δ loss over a matched-rank random subspace, at every depth.\n"
                 "Phase 4 read one cell per column, chosen by an argmax that moved during training.",
                 fontsize=11)
    for col, name in enumerate(systems):
        curve = grid[name]["curve"]
        tokens = [c["tokens"] for c in curve]
        depths = [d["name"] for d in curve[0]["by_depth"]]
        for row, g in enumerate(groups):
            m = np.full((len(depths), len(tokens)), np.nan)
            for j, c in enumerate(curve):
                for i, d in enumerate(c["by_depth"]):
                    if g in d["groups"]:
                        m[i, j] = d["groups"][g]["excess"]
            ax = axes[row][col]
            lim = np.nanmax(np.abs(m)) if np.isfinite(m).any() else 1.0
            im = ax.imshow(m, aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim)
            ax.set_yticks(range(len(depths)), ["emb", *[f"L{i}" for i in range(len(depths) - 1)]], fontsize=7)
            ax.set_xticks(range(len(tokens)), [f"{t/1e6:.0f}M" if t else "0" for t in tokens],
                          rotation=60, fontsize=6)
            if row == 0:
                ax.set_title(name, fontsize=9)
            if col == 0:
                ax.set_ylabel(f"{LABEL[g]}\ndepth", fontsize=8)
            fig.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = FIGDIR / "ablation_depth_grid.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    return out


def fig_controls(grid) -> Path:
    """R2/R3: damage against three nulls, as a function of rank, final checkpoint."""
    systems = list(grid)
    fig, axes = plt.subplots(1, len(systems), figsize=(3.6 * len(systems), 4.0), squeeze=False)
    fig.suptitle("Damage vs rank at each system's belief depth, against three nulls.\n"
                 "A fitted subspace sits in high-variance directions, so the isotropic "
                 "control is the easiest of the three.", fontsize=10)
    for ax, name in zip(axes[0], systems):
        final = grid[name]["curve"][-1]
        best = max(final["by_depth"],
                   key=lambda d: d["groups"].get("action_lag0", {}).get("r2_before", -1))
        for g in ("action_lag0", "metric"):
            if g not in best["groups"]:
                continue
            lad = best["groups"][g]["ladder"]
            r = [x["rank"] for x in lad]
            ax.plot(r, [x["delta_loss"] for x in lad], "o-", ms=3.5, color=COLOUR[g], label=LABEL[g])
        ctrl = best["controls"]
        rr = sorted(int(k) for k in ctrl)
        ax.plot(rr, [ctrl[str(k)]["random_delta"] for k in rr], "--", lw=1.2, color="0.45",
                label="random subspace")
        ax.plot(rr, [ctrl[str(k)]["pca_delta"] for k in rr], ":", lw=1.4, color="0.15",
                label="top-r principal dirs")
        ax.set_yscale("symlog", linthresh=1e-3)
        ax.set_xlabel("rank deleted"); ax.set_title(f"{name} @{best['name']}", fontsize=9)
        ax.grid(alpha=0.25); ax.legend(fontsize=7)
    axes[0][0].set_ylabel("Δ next-token loss (nats)")
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    out = FIGDIR / "ablation_controls.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    return out


def fig_myopic(myopic) -> Path:
    """E1a/E1b/E1c in one panel each."""
    systems = list(myopic)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    for name in systems:
        h = myopic[name]["e1c_horizon"]
        axes[0].plot([c["k"] for c in h], [c["belief_from_p1_to_pk"] for c in h],
                     "o-", ms=3.5, label=name)
        axes[0].plot([c["k"] for c in h], [c["metric_from_p1_to_pk"] for c in h],
                     "s--", ms=3, alpha=0.45)
    axes[0].set_xlabel("horizon K"); axes[0].set_ylabel("R² of target from p^(1..K)")
    axes[0].set_title("E1c: how fast the k-step law determines the target\n"
                      "solid = belief, dashed = metric", fontsize=9)
    axes[0].set_ylim(0, 1.02); axes[0].grid(alpha=0.25); axes[0].legend(fontsize=8)

    for name in systems:
        sweep = myopic[name]["e1a"]["radius_sweep"]
        axes[1].plot([s["radius"] for s in sweep], [s["ceiling"] for s in sweep],
                     "o-", ms=3.5, label=f"{name}")
        axes[1].scatter([sweep[0]["radius"]], [myopic[name]["e1a"]["belief_from_p_linear"]],
                        marker="_", s=180, color="0.3")
    axes[1].set_xscale("log"); axes[1].set_xlabel("clustering radius (TV)")
    axes[1].set_ylabel("ceiling: 1 - E[Var(b|p)]/Var(b)")
    axes[1].set_title("E1a: best possible prediction of the belief from p,\n"
                      "any functional form. Biased up at small radius.", fontsize=9)
    axes[1].grid(alpha=0.25); axes[1].legend(fontsize=8)

    names, paired, pooled, shuf = [], [], [], []
    for name in systems:
        e = myopic[name]["e1b"]
        if not e.get("runnable"):
            continue
        names.append(f"{name}\n({e['n_pairs_used']:,} pairs)")
        paired.append(e["paired_r2"])
        shuf.append(e["paired_r2_shuffled_control"])
        pooled.append(myopic[name]["e1a"]["dimension_controls"]["belief_from_full_stream"])
    x = np.arange(len(names))
    axes[2].bar(x - 0.26, pooled, 0.25, label="pooled R² (all positions)", color="0.6")
    axes[2].bar(x, paired, 0.25, label="paired R² (matched p)", color=COLOUR["action_lag0"])
    axes[2].bar(x + 0.26, shuf, 0.25, label="shuffled control", color="0.25")
    axes[2].set_xticks(x, names, fontsize=7)
    axes[2].set_ylabel("R² of belief from residual stream")
    axes[2].set_title("E1b: positions with the same next-token distribution\n"
                      "but different beliefs are still separated", fontsize=9)
    axes[2].axhline(0, color="0.3", lw=0.8); axes[2].grid(alpha=0.25, axis="y")
    axes[2].legend(fontsize=7)

    fig.tight_layout()
    out = FIGDIR / "myopic.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    return out


def fig_causal(causal) -> Path:
    """E2b and E2c: the two tests that are about the model rather than the process."""
    systems = list(causal)
    fig, axes = plt.subplots(2, len(systems), figsize=(3.5 * len(systems), 7.0), squeeze=False)
    fig.suptitle("Top: corrupting the belief at one position, read downstream.  "
                 "Bottom: patching it from a donor.", fontsize=11)
    for col, name in enumerate(systems):
        depths = [d for d in causal[name]["by_depth"] if "e2b_position_restricted" in d]
        ax = axes[0][col]
        for d in depths:
            b = d["e2b_position_restricted"]
            if not b.get("runnable"):
                continue
            rows = [r for r in b["by_offset"] if r["block_len"] == 4]
            ax.errorbar([r["offset"] for r in rows], [r["excess"] for r in rows],
                        yerr=[r["control_sd"] for r in rows], marker="o", ms=3, lw=1.2,
                        capsize=2, label=d["name"])
        ax.axhline(0, color="0.4", ls="--", lw=1.0)
        ax.set_xlabel("positions after the corrupted block")
        ax.set_title(name, fontsize=9); ax.grid(alpha=0.25); ax.legend(fontsize=7)
        if col == 0:
            ax.set_ylabel("excess Δ loss over random\n(E2b, block of 4)")

        ax = axes[1][col]
        for d in depths:
            p = d.get("e2c_patching")
            if not p:
                continue
            r = [q["rank"] for q in p["by_rank"]]
            ax.plot(r, [q["belief"]["transplant_cosine"] for q in p["by_rank"]],
                    "o-", ms=3, lw=1.3, label=f"{d['name']} belief")
            ax.plot(r, [q["random_mean_transplant_cosine"] for q in p["by_rank"]],
                    "--", lw=1.0, alpha=0.6, label=f"{d['name']} random")
        ax.set_xlabel("rank patched"); ax.grid(alpha=0.25); ax.legend(fontsize=6)
        if col == 0:
            ax.set_ylabel("alignment with a full transplant\n(E2c, cosine)")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = FIGDIR / "causal.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    return out


def fig_emergence(emergence) -> Path:
    """E5: emergence time against measured coupling, within one model."""
    systems = list(emergence)
    fig, axes = plt.subplots(1, len(systems), figsize=(3.6 * len(systems), 4.0), squeeze=False)
    fig.suptitle("E5: does coupling to the next-token distribution predict emergence?\n"
                 "One model, targets matched on width and scale, alpha sweeping the coupling.",
                 fontsize=10)
    for ax, name in zip(axes[0], systems):
        rows = emergence[name]["targets"]
        thr = sorted(rows[0]["crossings"])[0]
        synth = [r for r in rows if r.get("alpha") is not None]
        real = [r for r in rows if r.get("alpha") is None]
        xs = [r["coupling_to_p"] for r in synth]
        ys = [r["crossings"][thr] for r in synth]
        ok = [(x, y) for x, y in zip(xs, ys) if y is not None]
        if ok:
            ax.scatter(*zip(*ok), c=[r["alpha"] for r, y in zip(synth, ys) if y is not None],
                       cmap="viridis", s=36, label="matched targets")
        for r in real:
            y = r["crossings"][thr]
            if y is not None:
                ax.scatter([r["coupling_to_p"]], [y], marker="*", s=140,
                           color=COLOUR.get(r["target"].replace("real_", ""), "k"),
                           label=r["target"].replace("real_", ""))
        rho = emergence[name]["spearman"].get(thr, {}).get("spearman_coupling_vs_log_tokens")
        ax.set_yscale("log"); ax.set_xlabel("coupling to p (measured R²)")
        ax.set_title(f"{name}   Spearman={rho if rho is None else round(rho, 2)}", fontsize=9)
        ax.grid(alpha=0.25); ax.legend(fontsize=6)
    axes[0][0].set_ylabel(f"tokens to reach R² = {sorted(emergence[list(emergence)[0]]['targets'][0]['crossings'])[0]}")
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    out = FIGDIR / "emergence_coupling.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    return out


def main() -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    jobs = [
        ("phase5_00_validity.json", (fig_excess_loss, fig_recovery_phase)),
        ("phase5_01_ablation_grid.json", (fig_ablation_grid, fig_controls)),
        ("phase5_03_myopic.json", (fig_myopic,)),
        ("phase5_05_causal.json", (fig_causal,)),
        ("phase5_07_emergence.json", (fig_emergence,)),
    ]
    for filename, fns in jobs:
        data = _load(filename)
        if data is None:
            print(f"skip {filename} (not present yet)", flush=True)
            continue
        for fn in fns:
            try:
                print(f"wrote {fn(data).relative_to(ROOT)}", flush=True)
            except Exception as exc:  # a partial JSON should not kill the rest
                print(f"!! {fn.__name__} failed: {type(exc).__name__}: {exc}", flush=True)


if __name__ == "__main__":
    main()
