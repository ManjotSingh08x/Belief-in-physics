#!/usr/bin/env python3
"""Empirical Loss Saturation Profiler for Double Pendulum Next-Token & Lookahead Models.

Trains a representative suite of 10-15 models up to 1 Billion (1B) tokens to locate
the empirical loss plateau/knee before committing to the full 270-model training run.

Usage:
    # Full 1B run on 15 models (auto-detects Kaggle 2x T4 or local GPUs):
    uv run python scripts/loss_saturation_check.py

    # Rapid dry-run to verify pipeline plumbing in seconds:
    uv run python scripts/loss_saturation_check.py --smoke

    # Custom token target & output directory:
    uv run python scripts/loss_saturation_check.py --tokens 500000000 --batch-size 1024 --out-dir ./outputs
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import shutil
import sys
import time
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.transformer import ModelConfig, TinyTransformer
from physics.messk_configs import make_process


# ==============================================================================
# 1. Model Architecture & Lookahead Wrapper
# ==============================================================================

class LookaheadTransformer(TinyTransformer):
    """Causal decoder-only transformer trained to predict observation k steps ahead.
    
    At position t, logits predict token t + k. k=1 is standard next-token prediction.
    """
    def __init__(self, config: ModelConfig, k: int):
        super().__init__(config)
        self.k = k

    def loss(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.k < 1 or self.k >= tokens.shape[1]:
            raise ValueError(f"lookahead k={self.k} must satisfy 1 <= k < {tokens.shape[1]}")
        logits = self(tokens)
        return F.cross_entropy(
            logits[:, :-self.k].reshape(-1, logits.shape[-1]),
            tokens[:, self.k:].reshape(-1),
        )


# ==============================================================================
# 2. Experiment Configurations & Grid Definition
# ==============================================================================

@dataclass(frozen=True)
class SaturationModelSpec:
    name: str
    n_steps: int
    m: int
    delta_v: float
    gamma: float
    d_model: int
    d_mlp: int
    n_layers: int
    n_heads: int
    k: int
    k_suffix: str
    seed: int


def get_pilot_matrix(smoke: bool = False) -> List[SaturationModelSpec]:
    """Builds a diverse, representative 15-model matrix (or 2 models if smoke)."""
    if smoke:
        return [
            SaturationModelSpec(
                name="dp_pilot_smoke_w32_k1",
                n_steps=10, m=50, delta_v=1.2, gamma=2.5,
                d_model=32, d_mlp=128, n_layers=4, n_heads=2,
                k=1, k_suffix="_k1", seed=0,
            ),
            SaturationModelSpec(
                name="dp_pilot_smoke_w64_kfull",
                n_steps=10, m=50, delta_v=1.2, gamma=2.5,
                d_model=64, d_mlp=256, n_layers=4, n_heads=2,
                k=10, k_suffix="_k_full", seed=0,
            ),
        ]

    # Representative 15-model matrix:
    # Tuple 1 (Fast kick dynamics: n=10, m=50 -> 500 tokens):
    #   3 widths (32, 64, 128) x 3 lookaheads (k=1, k=5, k=10) = 9 models
    # Tuple 2 (Slow kick dynamics: n=20, m=25 -> 500 tokens):
    #   3 widths (32, 64, 128) x 2 lookaheads (k=1, k=20) = 6 models
    # Total: 15 models
    specs = []
    
    # --- Tuple 1: n=10, m=50 (delta_v=1.2, gamma=2.5) ---
    widths = [(32, 128), (64, 256), (128, 512)]
    horizons_n10 = [(1, "_k1"), (5, "_k_mid"), (10, "_k_full")]
    for d_model, d_mlp in widths:
        for k, k_suf in horizons_n10:
            specs.append(
                SaturationModelSpec(
                    name=f"dp_dv1.2_g2.5_n10_dmodel_{d_model}{k_suf}_seed0",
                    n_steps=10, m=50, delta_v=1.2, gamma=2.5,
                    d_model=d_model, d_mlp=d_mlp, n_layers=4, n_heads=2,
                    k=k, k_suffix=k_suf, seed=0,
                )
            )

    # --- Tuple 2: n=20, m=25 (delta_v=1.2, gamma=2.5) ---
    horizons_n20 = [(1, "_k1"), (20, "_k_full")]
    for d_model, d_mlp in widths:
        for k, k_suf in horizons_n20:
            specs.append(
                SaturationModelSpec(
                    name=f"dp_dv1.2_g2.5_n20_dmodel_{d_model}{k_suf}_seed0",
                    n_steps=20, m=25, delta_v=1.2, gamma=2.5,
                    d_model=d_model, d_mlp=d_mlp, n_layers=4, n_heads=2,
                    k=k, k_suffix=k_suf, seed=0,
                )
            )

    assert len(specs) == 15, f"Expected 15 specs, got {len(specs)}"
    return specs


# ==============================================================================
# 3. High-Throughput Streaming Training Loop
# ==============================================================================

def train_single_model(
    spec: SaturationModelSpec,
    total_tokens: int,
    effective_batch_size: int,
    device: str,
    log_every_steps: int,
    out_dir: Path,
    smoke: bool = False,
) -> pd.DataFrame:
    """Trains a single model, recording periodic train and eval loss to disk."""
    print(f"[{spec.name}] Starting training on {device} (Target: {total_tokens:,} tokens)...", flush=True)
    
    # 1. Physics simulator setup
    proc = make_process(
        "double_pendulum_mess4",
        m=spec.m,
        n_steps=spec.n_steps,
        delta_v=spec.delta_v,
        system={"gamma1": spec.gamma, "gamma2": spec.gamma},
    )
    seq_len = spec.m * spec.n_steps
    assert seq_len == 500, f"Invariant violated: seq_len={seq_len} != 500"

    # Micro-batching with gradient accumulation to guarantee zero OOM risk
    if smoke or device == "cpu":
        micro_batch_size = min(16, effective_batch_size)
    else:
        micro_batch_size = min(256, effective_batch_size)
    grad_accum_steps = max(1, effective_batch_size // micro_batch_size)
    actual_batch_size = micro_batch_size * grad_accum_steps

    tokens_per_step = actual_batch_size * seq_len
    total_steps = max(1, total_tokens // tokens_per_step)
    warmup_steps = max(1, int(total_steps * 0.02))

    # 2. Model & Optimizer initialization
    model_cfg = ModelConfig(
        vocab_size=proc.n_obs,
        n_ctx=seq_len,
        n_layers=spec.n_layers,
        n_heads=spec.n_heads,
        d_model=spec.d_model,
        d_mlp=spec.d_mlp,
        seed=spec.seed,
    )
    model = LookaheadTransformer(model_cfg, k=spec.k).to(device)

    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0)

    def lr_schedule(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(warmup_steps)
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, lr_schedule)

    # 3. Dedicated evaluation set (held-out seed, 64 trajectories for smoke / 256 for prod)
    eval_n = 32 if smoke else 256
    eval_rng = np.random.default_rng(spec.seed + 999_999)
    eval_tokens_np = proc.sample_batch(eval_rng, eval_n)["tokens"]
    eval_tokens = torch.as_tensor(eval_tokens_np, dtype=torch.long, device=device)

    # Streaming training generator
    stream_rng = np.random.default_rng(spec.seed + 42)

    history = []
    t_start = time.perf_counter()

    pbar = tqdm(range(total_steps), desc=spec.name, leave=False, dynamic_ncols=True)
    for step in pbar:
        model.train()
        optimiser.zero_grad(set_to_none=True)
        step_loss_accum = 0.0

        for _ in range(grad_accum_steps):
            batch_np = proc.sample_batch(stream_rng, micro_batch_size)["tokens"]
            batch = torch.as_tensor(batch_np, dtype=torch.long, device=device)
            micro_loss = model.loss(batch) / grad_accum_steps
            micro_loss.backward()
            step_loss_accum += float(micro_loss.item())

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()
        scheduler.step()

        tokens_seen = (step + 1) * tokens_per_step

        if step % log_every_steps == 0 or step == total_steps - 1:
            model.eval()
            with torch.no_grad():
                eval_loss = float(model.loss(eval_tokens).item())

            train_loss = step_loss_accum
            lr_curr = scheduler.get_last_lr()[0]
            elapsed = time.perf_counter() - t_start
            tok_per_sec = tokens_seen / max(1e-3, elapsed)

            record = {
                "model_name": spec.name,
                "step": step,
                "tokens_seen": tokens_seen,
                "tokens_seen_m": tokens_seen / 1_000_000,
                "train_loss": train_loss,
                "eval_loss": eval_loss,
                "lr": lr_curr,
                "tokens_per_sec": tok_per_sec,
                "d_model": spec.d_model,
                "k": spec.k,
                "k_suffix": spec.k_suffix,
                "n_steps": spec.n_steps,
            }
            history.append(record)

            pbar.set_postfix({
                "loss": f"{train_loss:.3f}",
                "val": f"{eval_loss:.3f}",
                "tok/s": f"{tok_per_sec:,.0f}",
            })

    # Save final model state and loss history JSON
    model_dir = out_dir / spec.name
    model_dir.mkdir(parents=True, exist_ok=True)
    
    torch.save(model.state_dict(), model_dir / f"{spec.name}_final.pt")
    
    with open(model_dir / f"{spec.name}_loss_history.json", "w") as f:
        json.dump(history, f, indent=2)

    df_hist = pd.DataFrame(history)
    df_hist.to_csv(model_dir / f"{spec.name}_loss_history.csv", index=False)

    print(f"[{spec.name}] Done. Final Eval Loss: {history[-1]['eval_loss']:.4f}", flush=True)
    return df_hist


# ==============================================================================
# 4. Saturation Analysis & Knee Detection
# ==============================================================================

def analyze_saturation(df_all: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """Calculates saturation milestones (90%, 95%, 98%, 99% of final loss reduction)."""
    summary_rows = []
    
    for name, group in df_all.groupby("model_name"):
        group = group.sort_values("tokens_seen").reset_index(drop=True)
        initial_loss = float(group.iloc[0]["eval_loss"])
        final_loss = float(group.iloc[-1]["eval_loss"])
        total_drop = initial_loss - final_loss
        max_tokens_m = float(group["tokens_seen_m"].max())

        # Find tokens where loss reduction reaches X% of total drop
        def tokens_at_pct(pct: float) -> float:
            if total_drop <= 1e-4:
                return max_tokens_m
            target_loss = initial_loss - (pct * total_drop)
            sub = group[group["eval_loss"] <= target_loss]
            if len(sub) > 0:
                return float(sub.iloc[0]["tokens_seen_m"])
            return max_tokens_m

        tok_90 = tokens_at_pct(0.90)
        tok_95 = tokens_at_pct(0.95)
        tok_98 = tokens_at_pct(0.98)
        tok_99 = tokens_at_pct(0.99)

        # Loss at standardized checkpoints (None if not yet reached)
        def loss_at(tokens_m: float) -> Optional[float]:
            if tokens_m > max_tokens_m:
                return None
            sub = group[group["tokens_seen_m"] >= tokens_m]
            if len(sub) > 0:
                return float(sub.iloc[0]["eval_loss"])
            return None

        summary_rows.append({
            "model_name": name,
            "d_model": int(group.iloc[0]["d_model"]),
            "k_suffix": str(group.iloc[0]["k_suffix"]),
            "n_steps": int(group.iloc[0]["n_steps"]),
            "initial_loss": initial_loss,
            "final_loss": final_loss,
            "loss_drop": total_drop,
            "loss_at_100M": loss_at(100),
            "loss_at_200M": loss_at(200),
            "loss_at_500M": loss_at(500),
            "loss_at_800M": loss_at(800),
            "tokens_m_to_90pct_drop": tok_90,
            "tokens_m_to_95pct_drop": tok_95,
            "tokens_m_to_98pct_drop": tok_98,
            "tokens_m_to_99pct_drop": tok_99,
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "loss_saturation_summary.csv", index=False)
    return summary_df


# ==============================================================================
# 5. Publication-Quality 4-Panel Visualization
# ==============================================================================

def plot_saturation_curves(df_all: pd.DataFrame, summary_df: pd.DataFrame, out_dir: Path) -> None:
    """Generates comprehensive 4-panel publication-quality saturation figures."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 11), dpi=150)
    plt.subplots_adjust(hspace=0.28, wspace=0.22)

    colors = {32: "#1f77b4", 64: "#ff7f0e", 128: "#2ca02c"}
    linestyles = {"_k1": "-", "_k_mid": "--", "_k_full": ":"}

    # --------------------------------------------------------------------------
    # Panel 1 (Top-Left): All Individual Evaluation Loss Curves
    # --------------------------------------------------------------------------
    ax1 = axes[0, 0]
    for name, group in df_all.groupby("model_name"):
        d_m = int(group.iloc[0]["d_model"])
        k_s = str(group.iloc[0]["k_suffix"])
        ax1.plot(
            group["tokens_seen_m"],
            group["eval_loss"],
            color=colors.get(d_m, "#333333"),
            linestyle=linestyles.get(k_s, "-"),
            alpha=0.75,
            linewidth=1.6,
        )

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="#1f77b4", lw=2, label="d_model=32"),
        Line2D([0], [0], color="#ff7f0e", lw=2, label="d_model=64"),
        Line2D([0], [0], color="#2ca02c", lw=2, label="d_model=128"),
        Line2D([0], [0], color="black", linestyle="-", lw=1.5, label="k=1 (_k1)"),
        Line2D([0], [0], color="black", linestyle="--", lw=1.5, label="k=mid (_k_mid)"),
        Line2D([0], [0], color="black", linestyle=":", lw=1.5, label="k=full (_k_full)"),
    ]
    ax1.legend(handles=legend_elements, loc="upper right", fontsize=8, framealpha=0.9)
    ax1.set_title("A. Individual Model Loss Trajectories", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Tokens Trained (Millions)", fontsize=10)
    ax1.set_ylabel("Evaluation Cross-Entropy Loss", fontsize=10)
    ax1.grid(True, alpha=0.3)

    # --------------------------------------------------------------------------
    # Panel 2 (Top-Right): Mean Loss by Model Width (Capacity Scaling)
    # --------------------------------------------------------------------------
    ax2 = axes[0, 1]
    for d_m in [32, 64, 128]:
        sub = df_all[df_all["d_model"] == d_m]
        if len(sub) == 0:
            continue
        # Group by step across seeds / configs
        step_grouped = sub.groupby("step").agg({
            "tokens_seen_m": "mean",
            "eval_loss": ["mean", "std"],
        })
        x_tok = step_grouped["tokens_seen_m"]["mean"]
        y_mean = step_grouped["eval_loss"]["mean"]
        y_std = step_grouped["eval_loss"]["std"].fillna(0.0)

        ax2.plot(x_tok, y_mean, label=f"d_model={d_m}", color=colors[d_m], lw=2.2)
        ax2.fill_between(x_tok, y_mean - y_std, y_mean + y_std, color=colors[d_m], alpha=0.15)

    ax2.set_title("B. Capacity Scaling: Mean Loss by Model Width", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Tokens Trained (Millions)", fontsize=10)
    ax2.set_ylabel("Mean Evaluation Loss ± 1σ", fontsize=10)
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(True, alpha=0.3)

    # --------------------------------------------------------------------------
    # Panel 3 (Bottom-Left): Mean Loss by Forecasting Horizon (k)
    # --------------------------------------------------------------------------
    ax3 = axes[1, 0]
    horizon_colors = {"_k1": "#d62728", "_k_mid": "#9467bd", "_k_full": "#8c564b"}
    for k_suf in ["_k1", "_k_mid", "_k_full"]:
        sub = df_all[df_all["k_suffix"] == k_suf]
        if len(sub) == 0:
            continue
        step_grouped = sub.groupby("step").agg({
            "tokens_seen_m": "mean",
            "eval_loss": ["mean", "std"],
        })
        x_tok = step_grouped["tokens_seen_m"]["mean"]
        y_mean = step_grouped["eval_loss"]["mean"]
        y_std = step_grouped["eval_loss"]["std"].fillna(0.0)

        ax3.plot(x_tok, y_mean, label=f"Lookahead {k_suf}", color=horizon_colors[k_suf], lw=2.0)
        ax3.fill_between(x_tok, y_mean - y_std, y_mean + y_std, color=horizon_colors[k_suf], alpha=0.15)

    ax3.set_title("C. Horizon Scaling: Mean Loss by Lookahead Target (k)", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Tokens Trained (Millions)", fontsize=10)
    ax3.set_ylabel("Mean Evaluation Loss ± 1σ", fontsize=10)
    ax3.legend(loc="upper right", fontsize=9)
    ax3.grid(True, alpha=0.3)

    # --------------------------------------------------------------------------
    # Panel 4 (Bottom-Right): Saturation Milestones Bar Chart
    # --------------------------------------------------------------------------
    ax4 = axes[1, 1]
    if not summary_df.empty:
        mean_90 = summary_df["tokens_m_to_90pct_drop"].mean()
        mean_95 = summary_df["tokens_m_to_95pct_drop"].mean()
        mean_98 = summary_df["tokens_m_to_98pct_drop"].mean()
        mean_99 = summary_df["tokens_m_to_99pct_drop"].mean()

        labels = ["90% Drop", "95% Drop", "98% Drop", "99% Drop"]
        vals = [mean_90, mean_95, mean_98, mean_99]
        bar_colors = ["#c6dbef", "#9ecae1", "#4292c6", "#08519c"]

        bars = ax4.bar(labels, vals, color=bar_colors, edgecolor="black", width=0.55)
        for bar in bars:
            yval = bar.get_height()
            ax4.text(
                bar.get_x() + bar.get_width() / 2,
                yval + (max(vals) * 0.02 if max(vals) > 0 else 0.01),
                f"{yval:.1f}M",
                ha="center", va="bottom", fontsize=10, fontweight="bold",
            )

        ax4.set_ylim(0, max(1.0, max(vals) * 1.25))
        ax4.set_title("D. Empirical Saturation Milestones (Tokens Needed)", fontsize=11, fontweight="bold")
        ax4.set_ylabel("Tokens Required (Millions)", fontsize=10)
        ax4.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    out_fig = out_dir / "loss_saturation_analysis.png"
    plt.savefig(out_fig, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saturation analysis figure saved -> {out_fig}")


# ==============================================================================
# 6. Main Orchestrator
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Pilot 1B Loss Saturation Profiler for Double Pendulum")
    parser.add_argument("--tokens", type=int, default=1_000_000_000, help="Tokens per model (default: 1,000,000,000)")
    parser.add_argument("--batch-size", type=int, default=1024, help="Batch size (default: 1024)")
    parser.add_argument("--log-every", type=int, default=20, help="Log eval loss every N steps (default: 20)")
    parser.add_argument("--smoke", action="store_true", help="Rapid dry-run with 2 models, batch 32, and 2M tokens")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    # Determine execution platform and output path
    is_kaggle = Path("/kaggle").exists()
    if args.out_dir:
        out_dir = Path(args.out_dir)
    elif is_kaggle:
        out_dir = Path("/kaggle/working/results_saturation")
    else:
        out_dir = REPO_ROOT / "experiments" / "results_saturation"

    out_dir.mkdir(parents=True, exist_ok=True)

    # In smoke mode: use smaller batch and token count to verify realistic curve across ~125 steps
    if args.smoke:
        total_tokens = 2_000_000
        effective_batch_size = 32
        log_every = 5
    else:
        total_tokens = args.tokens
        effective_batch_size = args.batch_size
        log_every = args.log_every

    specs = get_pilot_matrix(smoke=args.smoke)

    print("=" * 78)
    print(" DOUBLE PENDULUM: EMPIRICAL LOSS SATURATION PROFILER (1B PILOT RUN)")
    print("=" * 78)
    print(f"Platform:      {'Kaggle' if is_kaggle else 'Local'}")
    print(f"Output Dir:    {out_dir}")
    print(f"Models Count:  {len(specs)}")
    print(f"Tokens/Model:  {total_tokens:,}")
    print(f"Batch Size:    {effective_batch_size}")
    print(f"Log Every:     {log_every} steps")
    print(f"Smoke Mode:    {args.smoke}")

    # Accelerator detection
    device_count = torch.cuda.device_count()
    devices = [f"cuda:{i}" for i in range(device_count)] if device_count > 0 else ["cpu"]
    print(f"CUDA Devices:  {device_count} -> {devices}")
    print("=" * 78)

    all_dfs = []

    # Parallel or sequential execution across available accelerators
    if len(devices) > 1 and not args.smoke:
        print(f"Spawning ThreadPoolExecutor across {len(devices)} GPUs...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(devices)) as executor:
            futures = []
            for i, spec in enumerate(specs):
                dev = devices[i % len(devices)]
                fut = executor.submit(
                    train_single_model,
                    spec=spec,
                    total_tokens=total_tokens,
                    effective_batch_size=effective_batch_size,
                    device=dev,
                    log_every_steps=log_every,
                    out_dir=out_dir,
                    smoke=args.smoke,
                )
                futures.append(fut)

            for fut in concurrent.futures.as_completed(futures):
                all_dfs.append(fut.result())
    else:
        dev = devices[0]
        for spec in specs:
            df = train_single_model(
                spec=spec,
                total_tokens=total_tokens,
                effective_batch_size=effective_batch_size,
                device=dev,
                log_every_steps=log_every,
                out_dir=out_dir,
                smoke=args.smoke,
            )
            all_dfs.append(df)

    # Consolidate all history into master CSV
    df_all = pd.concat(all_dfs, ignore_index=True)
    master_csv = out_dir / "saturation_loss_curves.csv"
    df_all.to_csv(master_csv, index=False)
    print(f"\nMaster loss curve CSV saved -> {master_csv} ({len(df_all)} data points)")

    # Analyze knee and saturation points
    summary_df = analyze_saturation(df_all, out_dir)
    print("\n" + "=" * 78)
    print(" LOSS SATURATION SUMMARY & KNEE ANALYSIS")
    print("=" * 78)
    cols_to_print = ["model_name", "d_model", "k_suffix", "initial_loss", "final_loss", "tokens_m_to_95pct_drop"]
    if not args.smoke:
        cols_to_print.extend(["loss_at_200M", "loss_at_500M", "loss_at_800M"])
    print(summary_df[[c for c in cols_to_print if c in summary_df.columns]].to_string(index=False))

    # Generate analytical plots
    plot_saturation_curves(df_all, summary_df, out_dir)

    # Package into zip archive
    zip_path = out_dir.parent / "loss_saturation_results.zip"
    print(f"\nPackaging results into {zip_path}...")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for file in out_dir.rglob("*"):
            if file.is_file():
                arcname = file.relative_to(out_dir)
                zipf.write(file, arcname)
    print(f"Archive ready: {zip_path} ({zip_path.stat().st_size / 1024:.1f} KB)")
    print("=" * 78)
    print("PROFILING COMPLETE. Inspect 'loss_saturation_analysis.png' to set optimal token budget.")
    print("=" * 78)


if __name__ == "__main__":
    main()
