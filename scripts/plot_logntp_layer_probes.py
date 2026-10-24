#!/usr/bin/env python3
"""Visualization script for Layer-Wise Residual Stream Probing to logNTP.

Plots a 4-subplot diagnostic panel:
1A: Layer Emergence Curve (Trained vs Random Control R^2 across depth)
1B: Temporal Trajectory Overlay (True vs Probed logNTP across sequence timestamps)
1C: Calibration Scatter Plot (True vs Probed on held-out test tokens with y=x reference)
1D: Intra-Cycle Physical Breakdown (Mean R^2 across physical steps within cycle)

Usage:
  uv run python scripts/plot_logntp_layer_probes.py
  uv run python scripts/plot_logntp_layer_probes.py --checkpoint path/to/model.pt --out_dir ./figures
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.logntp_probing import ProbeEvaluationReport, evaluate_layer_probes
from models.transformer import ModelConfig, TinyTransformer
from physics.messk_configs import make_process


def plot_logntp_probing_panel(
    report: ProbeEvaluationReport,
    out_path: Path | str | None = None,
    title_suffix: str = "",
) -> plt.Figure:
    """Renders a publication-grade 4-subplot diagnostic panel for layer logNTP probing."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle(
        f"Layer-Wise Residual Stream Probing to logNTP {title_suffix}".strip(),
        fontsize=16,
        fontweight="bold",
    )

    depths = report.depths
    layer_labels = [name.replace("_", " ").title() for name in report.layer_names]

    # --------------------------------------------------------------------------
    # Subplot 1A: Layer Emergence Curve
    # --------------------------------------------------------------------------
    ax_a = axes[0, 0]
    ax_a.plot(
        depths,
        report.trained_r2,
        marker="o",
        linewidth=2.5,
        color="#1f77b4",
        label="Trained Model (Residual Stream)",
    )
    ax_a.plot(
        depths,
        report.random_r2,
        marker="s",
        linestyle="--",
        linewidth=2.0,
        color="#d62728",
        label="Random Untrained Control (Shai et al.)",
    )
    ax_a.set_xticks(depths)
    ax_a.set_xticklabels(layer_labels, rotation=15, ha="right")
    ax_a.set_ylabel(r"Probed $R^2$ (Test Variance Explained)", fontsize=12)
    ax_a.set_title(
        r"$\mathbf{1A}$: Linear Emergence of logNTP across Depth", fontsize=13
    )
    ax_a.grid(True, linestyle=":", alpha=0.6)
    ax_a.legend(loc="lower right", frameon=True, framealpha=0.9)

    best_d = report.best_layer_depth
    best_r2 = report.trained_r2[best_d]
    rand_r2 = report.random_r2[best_d]
    gain = best_r2 - rand_r2
    ax_a.annotate(
        f"Peak: {report.layer_names[best_d]}\n$R^2 = {best_r2:.3f}$\n($\\Delta = +{gain:.3f}$)",
        xy=(best_d, best_r2),
        xytext=(best_d - 0.5 if best_d > 1 else best_d + 0.3, best_r2 - 0.15),
        arrowprops=dict(facecolor="#1f77b4", shrink=0.08, width=1.5, headwidth=6),
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#e8f4f8", edgecolor="#1f77b4"),
        fontsize=10,
    )

    # --------------------------------------------------------------------------
    # Subplot 1B: Temporal Trajectory Tracking Overlay
    # --------------------------------------------------------------------------
    ax_b = axes[0, 1]
    sample_true = report.sample_true_trajectory
    t_steps = np.arange(len(sample_true))

    ax_b.plot(
        t_steps,
        sample_true,
        color="black",
        linewidth=2.2,
        label=r"True Model logNTP ($\log P(x_{t+1})$)",
        zorder=5,
    )

    # Plot sample probed predictions from embedding and last 2 blocks
    colors = ["#2ca02c", "#ff7f0e", "#1f77b4", "#9467bd", "#8c564b"]
    for d in depths:
        if d in report.sample_pred_trajectories:
            pred_y = report.sample_pred_trajectories[d]
            lbl = f"Probed {report.layer_names[d]}"
            ax_b.plot(
                t_steps,
                pred_y,
                linestyle="--",
                alpha=0.85,
                color=colors[d % len(colors)],
                linewidth=1.5,
                label=lbl,
            )

    # Mark physical cycle boundaries (every n_steps)
    if report.n_steps > 0:
        for boundary in range(0, len(sample_true), report.n_steps):
            ax_b.axvline(
                boundary, color="gray", linestyle=":", alpha=0.5, linewidth=1.0
            )

    ax_b.set_xlabel("Sequence Timestamp $t$ (Tokens)", fontsize=12)
    ax_b.set_ylabel(r"$\log P(x_{t+1})$", fontsize=12)
    ax_b.set_title(
        r"$\mathbf{1B}$: Temporal Trajectory Tracking (Held-Out Sequence)",
        fontsize=13,
    )
    ax_b.grid(True, linestyle=":", alpha=0.6)
    ax_b.legend(loc="lower left", fontsize=9, framealpha=0.9, ncol=2)

    # --------------------------------------------------------------------------
    # Subplot 1C: Calibration Scatter Plot (Best Layer)
    # --------------------------------------------------------------------------
    ax_c = axes[1, 0]
    ax_c.scatter(
        report.scatter_true,
        report.scatter_pred,
        alpha=0.4,
        color="#1f77b4",
        edgecolors="none",
        s=25,
        label=f"Test Tokens ({report.layer_names[best_d]})",
    )

    # Reference diagonal y = x
    all_vals = np.concatenate([report.scatter_true, report.scatter_pred])
    val_min, val_max = np.percentile(all_vals, 1), np.percentile(all_vals, 99)
    pad = (val_max - val_min) * 0.05
    ref_x = np.linspace(val_min - pad, val_max + pad, 100)
    ax_c.plot(
        ref_x,
        ref_x,
        linestyle="--",
        color="#d62728",
        linewidth=1.8,
        label="Ideal Calibration ($y = x$)",
    )

    ax_c.set_xlim(val_min - pad, val_max + pad)
    ax_c.set_ylim(val_min - pad, val_max + pad)
    ax_c.set_xlabel(r"True Model logNTP ($\log P(x_{t+1})$)", fontsize=12)
    ax_c.set_ylabel(r"Probed $\widehat{\log P}(x_{t+1})$", fontsize=12)
    ax_c.set_title(
        rf"$\mathbf{{1C}}$: Calibration Scatter ({report.layer_names[best_d]})",
        fontsize=13,
    )
    ax_c.grid(True, linestyle=":", alpha=0.6)
    ax_c.legend(loc="upper left", framealpha=0.9)

    # Inset metrics
    corr_val = report.trained_corr[best_d]
    mse_val = report.trained_mse[best_d]
    stats_text = (
        f"Pearson $r = {corr_val:.3f}$\n"
        f"Test $R^2 = {best_r2:.3f}$\n"
        f"Test MSE $= {mse_val:.3f}$"
    )
    ax_c.text(
        0.95,
        0.05,
        stats_text,
        transform=ax_c.transAxes,
        fontsize=10,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9, edgecolor="gray"),
    )

    # --------------------------------------------------------------------------
    # Subplot 1D: Intra-Cycle Physical Breakdown
    # --------------------------------------------------------------------------
    ax_d = axes[1, 1]
    n_steps = report.n_steps
    step_axis = np.arange(n_steps)

    for d in depths:
        r2_by_step = report.intra_cycle_r2[d]
        ax_d.plot(
            step_axis,
            r2_by_step,
            marker="^",
            linewidth=1.8,
            alpha=0.85,
            color=colors[d % len(colors)],
            label=report.layer_names[d],
        )

    ax_d.set_xticks(step_axis)
    ax_d.set_xlabel("Step within Physical Tick s = t % n (0 = Post-Kick)", fontsize=12)
    ax_d.set_ylabel(r"Substep $R^2$", fontsize=12)
    ax_d.set_title(
        r"$\mathbf{1D}$: Intra-Cycle Physical Emergence (Lagrangian Flow vs Kick)",
        fontsize=13,
    )
    ax_d.grid(True, linestyle=":", alpha=0.6)
    ax_d.legend(loc="lower right", fontsize=9, framealpha=0.9, ncol=2)

    fig.subplots_adjust(top=0.92, bottom=0.08, left=0.08, right=0.95, hspace=0.3, wspace=0.25)
    if out_path is not None:
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=300, bbox_inches="tight")

    return fig


def find_default_checkpoint() -> Path | None:
    """Finds a trained model checkpoint in experiments/ directory."""
    results_dir = REPO_ROOT / "experiments"
    ckpts = list(results_dir.glob("**/*_final.pt"))
    if not ckpts:
        ckpts = list(results_dir.glob("**/*.pt"))
    return ckpts[0] if ckpts else None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and plot layer-wise probes from residual stream to logNTP."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to trained TinyTransformer checkpoint (.pt). Defaults to auto-discovery.",
    )
    parser.add_argument(
        "--process",
        type=str,
        default="double_pendulum_mess4",
        help="Physical process name (default: double_pendulum_mess4).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Number of sequences for probe evaluation (default: 64).",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=160,
        help="Sequence context length (default: 160).",
    )
    parser.add_argument(
        "--target_type",
        type=str,
        default="scalar_logp",
        choices=["scalar_logp", "full_vocab", "action_logp"],
        help="logNTP target formulation (default: scalar_logp).",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./experiments/results/logntp_probes",
        help="Output directory for plots and JSON report.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed (default: 0).",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Quick smoke run with synthetic model if no checkpoint exists.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Process
    proc = make_process(args.process)
    print(f"Loaded process: {args.process} (vocab={proc.n_obs}, n_steps={proc.n_steps})")

    # 2. Load Checkpoint or fallback
    ckpt_path = Path(args.checkpoint) if args.checkpoint else find_default_checkpoint()

    if ckpt_path and ckpt_path.is_file():
        print(f"Loading checkpoint: {ckpt_path}")
        state = torch.load(ckpt_path, map_location="cpu")
        # Handle dict format
        if isinstance(state, dict) and "config" in state and "model_state_dict" in state:
            cfg = state["config"]
            model = TinyTransformer(cfg)
            model.load_state_dict(state["model_state_dict"])
        elif isinstance(state, dict) and "tok_emb.weight" in state:
            # Reconstruct ModelConfig from state dict shapes
            vocab_size = state["tok_emb.weight"].shape[0]
            d_model = state["tok_emb.weight"].shape[1]
            n_ctx = state["pos_emb.weight"].shape[0]
            n_layers = max([int(k.split(".")[1]) for k in state.keys() if k.startswith("blocks.")]) + 1
            cfg = ModelConfig(vocab_size=vocab_size, n_ctx=n_ctx, n_layers=n_layers, d_model=d_model)
            model = TinyTransformer(cfg)
            model.load_state_dict(state)
        else:
            raise ValueError(f"Unrecognized checkpoint format at {ckpt_path}")
    else:
        print("No checkpoint found; instantiating architecture for smoke evaluation...")
        cfg = ModelConfig(
            vocab_size=proc.n_obs,
            n_ctx=args.seq_len,
            n_layers=4,
            d_model=64,
            d_mlp=256,
            seed=args.seed,
        )
        model = TinyTransformer(cfg)

    model.eval()

    # 3. Simulate Evaluation Sequences
    print(f"Simulating {args.batch_size} sequences of length {args.seq_len}...")
    rng = np.random.default_rng(args.seed)
    batch = proc.sample_batch(rng, args.batch_size)
    tokens = torch.as_tensor(batch["tokens"], dtype=torch.long)

    # 4. Evaluate Layer Probes
    print(f"Evaluating layer probes to target '{args.target_type}'...")
    report = evaluate_layer_probes(
        model=model,
        tokens=tokens,
        target_type=args.target_type,
        k=1,
        n_steps=proc.n_steps,
        seed=args.seed,
    )

    # Print Summary Table
    print("\nLayer Probing Results Summary:")
    print(f"{'Depth':<6} {'Layer Name':<15} {'Trained R^2':<12} {'Random R^2':<12} {'Gain':<10} {'Corr':<8}")
    print("-" * 65)
    for d in report.depths:
        t_r2 = report.trained_r2[d]
        r_r2 = report.random_r2[d]
        gain = t_r2 - r_r2
        corr = report.trained_corr[d]
        print(f"{d:<6} {report.layer_names[d]:<15} {t_r2:<12.4f} {r_r2:<12.4f} {gain:<+10.4f} {corr:<8.4f}")

    # 5. Render Panel
    plot_file = out_dir / "plot_logntp_layer_probes.png"
    print(f"\nRendering diagnostic panel to {plot_file}...")
    plot_logntp_probing_panel(report, out_path=plot_file, title_suffix=f"({args.process})")

    # 6. Save JSON Metrics
    metrics_file = out_dir / "logntp_probe_report.json"
    metrics_data = {
        "process": args.process,
        "checkpoint": str(ckpt_path) if ckpt_path else "synthetic",
        "target_type": args.target_type,
        "depths": report.depths,
        "layer_names": report.layer_names,
        "trained_r2": report.trained_r2,
        "random_r2": report.random_r2,
        "trained_mse": report.trained_mse,
        "trained_corr": report.trained_corr,
        "best_layer_depth": report.best_layer_depth,
        "best_layer_name": report.layer_names[report.best_layer_depth],
        "best_layer_r2": report.trained_r2[report.best_layer_depth],
    }
    with open(metrics_file, "w") as f:
        json.dump(metrics_data, f, indent=2)
    print(f"Saved metrics report to {metrics_file}")
    print("Done!")


if __name__ == "__main__":
    main()
