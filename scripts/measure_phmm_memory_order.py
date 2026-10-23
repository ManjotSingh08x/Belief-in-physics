#!/usr/bin/env python3
"""Measure p-HMM Markov Memory Depth and Model Intersection on Logarithmic Scale.

This script determines the effective Markov memory order p* (or k*) that the trained
transformer's internal residual stream has learned to represent:
1. Theoretical p-HMM Curve: Computes the asymptotic KL divergence
     D_KL(b_inf || b_{p-HMM})
   for order-p Markov approximations p in [0, 20]. On a logarithmic Y-axis,
   this decay forms an exact downward-sloping linear curve.
2. Baselines:
   - 0-HMM: Memoryless stationary prior (p=0).
   - 1-HMM: Myopic single-letter lookback (p=1).
   - Trained Model: Probed belief state from the transformer residual stream.
3. Intersection: Computes the exact log-linear crossing point p* where
     D_KL(b_{p*-HMM}) = D_KL(model)
4. Visualization: Generates the publication-grade visualization with:
   - Logarithmic scale on the Y-axis (D_KL)
   - Thick black frame and spines
   - Multicolored gradient line for the p-HMM curve
   - Horizontal black line for the trained model baseline
   - Vertical blue line at the intersection with bold 'k = {p*:.1f}' annotation

Usage:
  uv run python scripts/measure_phmm_memory_order.py --checkpoint path/to/model_final.pt
  uv run python scripts/measure_phmm_memory_order.py --dir ./experiments/results
  uv run python scripts/measure_phmm_memory_order.py --smoke
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.transformer import ModelConfig, TinyTransformer
from physics.messk_configs import make_process
from scripts.measure_optimal_observer_kl import (
    LookaheadTransformer,
    extract_residual_streams_batched,
    fit_softmax_linear_probe,
    load_model_and_metadata,
    sequence_train_test_split,
)


# ==============================================================================
# 1. Theoretical p-HMM Belief Computation
# ==============================================================================

def compute_phmm_beliefs_batch(
    letters: np.ndarray,
    p: int,
    joint: np.ndarray,
    pi: np.ndarray,
) -> np.ndarray:
    """Computes exact order-p HMM predictive belief states across all sequences and cycles.
    
    Args:
        letters: Array of emitted letters, shape (N, m)
        p: Memory order (p=0 is memoryless prior pi; p=1 is 1-letter lookback; etc.)
        joint: Transition-emission tensor, shape (K, K, K)
        pi: Stationary prior distribution, shape (K,)
        
    Returns:
        Array of belief vectors, shape (N, m, K)
    """
    N, m = letters.shape
    K = len(pi)
    out = np.empty((N, m, K), dtype=np.float64)

    if p == 0:
        return np.tile(pi, (N, m, 1))

    for i in range(N):
        for c in range(m):
            start = max(0, c - p + 1)
            b = pi.copy()
            for t in range(start, c + 1):
                b = b @ joint[letters[i, t]]
                sum_b = b.sum()
                b = b / sum_b if sum_b > 0 else pi.copy()
            out[i, c] = b

    return out


def evaluate_phmm_decay_curve(
    proc: Any,
    p_range: List[int],
    n_trajs: int = 64,
    burn_in: int = 20,
    seed: int = 20260829,
) -> Dict[str, Any]:
    """Evaluates theoretical KL divergence of p-HMM relative to optimal observer on steady-state cycles."""
    rng = np.random.default_rng(seed)
    batch = proc.sample_batch(rng, n_trajs)
    letters = batch["letters"]  # (N, m)
    all_beliefs = batch["beliefs"]  # (N, seq_len, K)

    n_steps = proc.n_steps
    m_cycles = proc.m
    K = proc.chain.n_states
    joint = proc.chain.joint
    pi = np.full(K, 1.0 / K)

    # Extract boundary belief states: b_c^(inf)
    boundary_indices = np.array([c * n_steps + n_steps - 1 for c in range(m_cycles)])
    beliefs_opt = all_beliefs[:, boundary_indices, :]  # (N, m, K)

    # Ensure evaluation happens on stationary cycles (c >= burn_in)
    eval_burn_in = min(burn_in, m_cycles // 3)
    eval_slice = slice(eval_burn_in, None)
    b_opt_eval = np.clip(beliefs_opt[:, eval_slice, :], 1e-15, 1.0)
    b_opt_eval = b_opt_eval / b_opt_eval.sum(axis=-1, keepdims=True)

    results = []
    p_kl_dict: Dict[int, float] = {}

    for p in p_range:
        b_p = compute_phmm_beliefs_batch(letters, p, joint, pi)
        b_p_eval = np.clip(b_p[:, eval_slice, :], 1e-15, 1.0)
        b_p_eval = b_p_eval / b_p_eval.sum(axis=-1, keepdims=True)

        # D_KL(b_opt || b_p)
        kl_matrix = np.sum(b_opt_eval * np.log(b_opt_eval / b_p_eval), axis=-1)
        mean_kl = float(np.mean(kl_matrix))
        p_kl_dict[p] = mean_kl
        results.append({
            "p": p,
            "kl_nats": mean_kl,
            "kl_bits": mean_kl / math.log(2.0),
        })

    return {
        "p_kl_dict": p_kl_dict,
        "curve_df": pd.DataFrame(results),
        "0hmm_kl": p_kl_dict[0],
        "1hmm_kl": p_kl_dict.get(1, p_kl_dict[0]),
    }


# ==============================================================================
# 2. Model Belief Probing Baseline Evaluation
# ==============================================================================

def evaluate_model_probed_kl(
    trained_model: TinyTransformer,
    random_model: TinyTransformer,
    proc: Any,
    n_trajs: int = 64,
    burn_in: int = 20,
    train_frac: float = 0.7,
    device: str = "cpu",
    seed: int = 20260829,
) -> Dict[str, Any]:
    """Extracts residual streams, fits softmax linear probe, and scores test KL divergence."""
    rng = np.random.default_rng(seed)
    batch = proc.sample_batch(rng, n_trajs)
    tokens = batch["tokens"]
    all_beliefs = batch["beliefs"]

    N, seq_len = tokens.shape
    n_steps = proc.n_steps
    m_cycles = proc.m
    K = proc.chain.n_states

    boundary_indices = np.array([c * n_steps + n_steps - 1 for c in range(m_cycles)])
    beliefs_opt = all_beliefs[:, boundary_indices, :]

    eval_burn_in = min(burn_in, m_cycles // 3)
    eval_boundary_indices = boundary_indices[eval_burn_in:]
    n_eval_cycles = len(eval_boundary_indices)

    train_seqs, test_seqs = sequence_train_test_split(N, train_frac=train_frac, seed=seed)
    n_test_seqs = len(test_seqs)

    trained_streams = extract_residual_streams_batched(trained_model, tokens, device=device)
    random_streams = extract_residual_streams_batched(random_model, tokens, device=device)
    n_depths = len(trained_streams)

    B_train_t = torch.as_tensor(beliefs_opt[train_seqs, eval_burn_in:, :].reshape(-1, K), dtype=torch.float32)
    B_test_np = beliefs_opt[test_seqs, eval_burn_in:, :].reshape(-1, K)
    B_test_np = np.clip(B_test_np, 1e-15, 1.0)
    B_test_np = B_test_np / B_test_np.sum(axis=-1, keepdims=True)

    best_layer_idx = 0
    best_test_kl = float("inf")
    best_random_kl = float("inf")

    for d_idx in range(n_depths):
        s_tr_d = trained_streams[d_idx][:, eval_boundary_indices, :]
        s_rnd_d = random_streams[d_idx][:, eval_boundary_indices, :]

        X_tr_train = torch.as_tensor(s_tr_d[train_seqs].reshape(-1, s_tr_d.shape[-1]), dtype=torch.float32)
        X_tr_test = torch.as_tensor(s_tr_d[test_seqs].reshape(-1, s_tr_d.shape[-1]), dtype=torch.float32)

        X_rnd_train = torch.as_tensor(s_rnd_d[train_seqs].reshape(-1, s_rnd_d.shape[-1]), dtype=torch.float32)
        X_rnd_test = torch.as_tensor(s_rnd_d[test_seqs].reshape(-1, s_rnd_d.shape[-1]), dtype=torch.float32)

        probe_tr = fit_softmax_linear_probe(X_tr_train, B_train_t, device=device)
        probe_rnd = fit_softmax_linear_probe(X_rnd_train, B_train_t, device=device)

        with torch.no_grad():
            tr_test_pred = probe_tr.predict_probs(X_tr_test.to(device)).cpu().numpy()
            rnd_test_pred = probe_rnd.predict_probs(X_rnd_test.to(device)).cpu().numpy()

        tr_test_pred = np.clip(tr_test_pred, 1e-15, 1.0)
        tr_test_pred = tr_test_pred / tr_test_pred.sum(axis=-1, keepdims=True)
        kl_tr = float(np.mean(np.sum(B_test_np * np.log(B_test_np / tr_test_pred), axis=-1)))

        rnd_test_pred = np.clip(rnd_test_pred, 1e-15, 1.0)
        rnd_test_pred = rnd_test_pred / rnd_test_pred.sum(axis=-1, keepdims=True)
        kl_rnd = float(np.mean(np.sum(B_test_np * np.log(B_test_np / rnd_test_pred), axis=-1)))

        if kl_tr < best_test_kl:
            best_test_kl = kl_tr
            best_layer_idx = d_idx
            best_random_kl = kl_rnd

    return {
        "best_layer_idx": best_layer_idx,
        "best_layer_name": "Embedding" if best_layer_idx == 0 else f"Layer {best_layer_idx - 1}",
        "trained_model_kl": best_test_kl,
        "random_control_kl": best_random_kl,
    }


# ==============================================================================
# 3. Log-Linear Intersection Calculation (Effective Memory Depth p*)
# ==============================================================================

def find_effective_memory_depth(
    p_vals: List[int],
    kl_vals: List[float],
    target_kl: float,
) -> Tuple[float, str]:
    """Computes exact fractional Markov order p* where D_KL(p*) == target_kl via log-linear interpolation."""
    p_arr = np.array(p_vals, dtype=np.float64)
    kl_arr = np.array(kl_vals, dtype=np.float64)

    # Edge Case: Model is worse than 0-HMM (memoryless prior)
    if target_kl >= kl_arr[0]:
        return 0.0, "<= 0 (at or above 0-HMM prior)"

    # Edge Case: Model is more accurate than the highest evaluated order
    if target_kl <= kl_arr[-1]:
        # Log-linear extrapolation from the last 3 points
        log_y = np.log(np.maximum(kl_arr[-3:], 1e-30))
        slope, intercept = np.polyfit(p_arr[-3:], log_y, 1)
        extrapolated_p = float((np.log(max(target_kl, 1e-30)) - intercept) / slope)
        return extrapolated_p, f"> {p_vals[-1]} (extrapolated: {extrapolated_p:.2f})"

    # Find the bounding interval [p_low, p_high]
    for i in range(len(kl_arr) - 1):
        if kl_arr[i] >= target_kl >= kl_arr[i + 1]:
            p0, p1 = p_arr[i], p_arr[i + 1]
            log_k0, log_k1 = math.log(kl_arr[i]), math.log(kl_arr[i + 1])
            log_target = math.log(target_kl)
            # Linear interpolation on log(D_KL)
            frac = (log_target - log_k0) / (log_k1 - log_k0)
            p_star = float(p0 + frac * (p1 - p0))
            return p_star, "interpolated"

    return 0.0, "undefined"


# ==============================================================================
# 4. Publication-Grade Visualization (Matching User Reference Image)
# ==============================================================================

def render_phmm_reference_figure(
    p_vals: List[int],
    kl_vals: List[float],
    model_kl: float,
    effective_p: float,
    out_path: Path,
    title_label: str = "k",
    display_p_max: int = 8,
) -> None:
    """Renders the publication-grade visualization matching the exact uploaded reference style.
    
    Visual features:
    - Logarithmic scale on the Y-axis (D_KL)
    - Thick black spines and bold ticks
    - Multicolored segmented gradient line (dark green -> olive -> brown -> orange -> red)
    - Solid thick horizontal black line for the trained model
    - Solid thick vertical blue line at the intersection
    - Bold high-contrast blue callout: 'k = {effective_p:.1f}'
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(4.2, 4.2), dpi=200)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    p_arr = np.array(p_vals, dtype=np.float64)
    kl_arr = np.array(kl_vals, dtype=np.float64)

    # Filter to display window (e.g. 0 to display_p_max, or dynamically covering intersection)
    max_x = max(display_p_max, int(math.ceil(effective_p + 2.0)))
    mask = p_arr <= max_x
    x_sub = p_arr[mask]
    y_sub = kl_arr[mask]

    # Construct multicolored gradient line using LineCollection
    points = np.array([x_sub, y_sub]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Colormap matching reference: dark green -> yellow-green/olive -> brown -> orange -> red
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "phmm_gradient",
        ["#0b6138", "#8ea954", "#bf8243", "#de5d3b", "#bd2e2e"],
    )
    norm = plt.Normalize(x_sub.min(), x_sub.max())
    lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=5.5, capstyle="round", zorder=3)
    lc.set_array(x_sub[:-1])
    ax.add_collection(lc)

    # 1. Horizontal black line: Trained Model KL
    ax.axhline(model_kl, color="black", linewidth=3.8, zorder=4)

    # 2. Vertical blue line: Effective Memory Depth Intersection
    ax.axvline(effective_p, color="#0000ff", linewidth=5.0, zorder=5)

    # 3. Bold blue text annotation: 'k = {effective_p:.1f}'
    # Positioned at top-left of the vertical line as in reference image
    annot_text = f"{title_label}={effective_p:.1f}"
    ax.text(
        effective_p - 0.45,
        max(kl_arr[0] * 0.45, model_kl * 5.0),
        annot_text,
        color="#0000ff",
        fontsize=20,
        fontweight="bold",
        ha="right",
        va="top",
        zorder=6,
    )

    # Logarithmic scale on the Y-axis
    ax.set_yscale("log")

    # Set axes limits
    y_min = min(y_sub.min() * 0.5, model_kl * 0.3)
    y_max = y_sub.max() * 2.2
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(-0.3, max_x + 0.3)

    # X-axis ticks: 0, 2, 4, 6, 8, ...
    tick_step = 2
    x_ticks = np.arange(0, max_x + 1, tick_step)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([str(int(t)) for t in x_ticks], fontsize=15, fontweight="bold")

    # Y-axis ticks formatting
    ax.tick_params(axis="both", which="major", direction="out", length=7, width=2.8, labelsize=14)
    ax.tick_params(axis="y", which="minor", direction="out", length=4, width=1.8)
    # Hide y-axis tick text labels for minimalist publication style matching reference
    ax.set_yticklabels([])

    # Thick black border spines matching reference
    for spine in ax.spines.values():
        spine.set_linewidth(2.8)
        spine.set_color("black")

    # Faint subtle grid lines
    ax.grid(True, which="major", color="#e8e8e8", linewidth=0.8, alpha=0.7, zorder=1)

    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved publication-grade figure -> {out_path}")


def render_phmm_diagnostics_figure(
    p_vals: List[int],
    kl_vals: List[float],
    model_kl: float,
    random_kl: float,
    effective_p: float,
    job_id: str,
    out_path: Path,
) -> None:
    """Renders a companion diagnostic figure showing labeled baselines and numerical axes."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("white")

    p_arr = np.array(p_vals)
    kl_arr = np.array(kl_vals)

    ax.plot(p_arr, kl_arr, marker="o", markersize=6, color="#1d3557", lw=2.0, label="p-HMM Theoretical Decay Curve", zorder=3)
    ax.axhline(model_kl, color="#e63946", lw=2.2, ls="--", label=f"Trained Model KL ({model_kl:.4e} nats)", zorder=4)
    ax.axhline(random_kl, color="#9a8c98", lw=1.5, ls=":", label=f"Random Control KL ({random_kl:.4e} nats)", zorder=4)

    # Baselines markers
    ax.scatter([0], [kl_arr[0]], color="#e76f51", s=80, zorder=5, label=f"0-HMM Baseline ({kl_arr[0]:.4f} nats)")
    if len(kl_arr) > 1:
        ax.scatter([1], [kl_arr[1]], color="#2a9d8f", s=80, zorder=5, label=f"1-HMM Baseline ({kl_arr[1]:.4f} nats)")

    # Intersection marker
    ax.axvline(effective_p, color="#0000ff", lw=2.0, ls="-.", label=f"Effective Memory p* = {effective_p:.2f}")
    ax.scatter([effective_p], [model_kl], color="#0000ff", s=110, zorder=6)

    ax.set_yscale("log")
    ax.set_xlabel("Markov Memory Order (p)", fontsize=11, fontweight="bold")
    ax.set_ylabel("D_KL(b_opt || b_p) [nats] (Log Scale)", fontsize=11, fontweight="bold")
    ax.set_title(f"Markov Memory Depth Estimation: {job_id} (p* = {effective_p:.2f})", fontsize=11.5, fontweight="bold", pad=12)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=True, facecolor="white", edgecolor="#ddd", fontsize=8.5, loc="upper right")

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved diagnostic overview figure -> {out_path}")


# ==============================================================================
# 5. End-to-End Orchestrator Pipeline
# ==============================================================================

def run_phmm_memory_analysis(
    ckpt_path: Path,
    out_dir: Path,
    p_max: int = 15,
    n_trajs: int = 64,
    device: str = "cpu",
    smoke: bool = False,
    burn_in: int = 20,
) -> Dict[str, Any]:
    """Runs full p-HMM memory order evaluation and generates reference visualization."""
    t0 = time.perf_counter()

    trained_m, random_m, info = load_model_and_metadata(ckpt_path=ckpt_path, device=device)
    job_id = info["job_id"]
    n_steps = info["n_steps"]
    n_ctx = info["n_ctx"]

    # Trajectory length: at least 60 cycles to have steady burn-in + memory depth
    target_m = max(50, burn_in + p_max + 15)
    actual_m = min(target_m, n_ctx // n_steps)
    effective_seq_len = actual_m * n_steps

    print(f"\n[{job_id}] Running p-HMM Markov Memory Order Analysis (p in 0..{p_max})...")
    print(f"  Configuration: dv={info['delta_v']}, gamma={info['gamma']}, n_steps={n_steps}")
    print(f"  Sequence Length: {effective_seq_len} tokens ({actual_m} cycles, burn-in={burn_in} cycles)")

    proc = make_process(
        "double_pendulum_mess4",
        m=actual_m,
        n_steps=n_steps,
        delta_v=info["delta_v"],
        dt=info["dt"],
        system={"gamma1": info["gamma"], "gamma2": info["gamma"]},
    )

    # 1. Theoretical p-HMM decay curve
    p_range = list(range(0, p_max + 1))
    phmm_res = evaluate_phmm_decay_curve(
        proc=proc,
        p_range=p_range,
        n_trajs=16 if smoke else n_trajs,
        burn_in=burn_in,
    )

    # 2. Probed model baseline
    model_res = evaluate_model_probed_kl(
        trained_model=trained_m,
        random_model=random_m,
        proc=proc,
        n_trajs=16 if smoke else n_trajs,
        burn_in=burn_in,
        device=device,
    )

    model_kl = model_res["trained_model_kl"]
    random_kl = model_res["random_control_kl"]
    p_vals = p_range
    kl_vals = [phmm_res["p_kl_dict"][p] for p in p_range]

    # 3. Log-linear intersection (effective memory order p*)
    effective_p, interp_status = find_effective_memory_depth(p_vals, kl_vals, model_kl)

    elapsed = time.perf_counter() - t0

    print("=" * 82)
    print(f" MARKOV MEMORY ORDER ANALYSIS RESULTS: {job_id}")
    print("=" * 82)
    print(f"  * 0-HMM Baseline KL:       {phmm_res['0hmm_kl']:.6f} nats")
    print(f"  * 1-HMM Baseline KL:       {phmm_res['1hmm_kl']:.6f} nats")
    print(f"  * Trained Model Probed KL: {model_kl:.6f} nats ({model_res['best_layer_name']})")
    print(f"  * Random Control KL:       {random_kl:.6f} nats")
    print(f"  * Effective Markov Depth:  k = {effective_p:.2f} ({interp_status})")
    print("=" * 82)

    # Save outputs
    model_out_dir = out_dir / job_id
    model_out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Reference publication-style figure (matching uploaded image)
    ref_fig_path = model_out_dir / f"{job_id}_phmm_memory_order.png"
    render_phmm_reference_figure(
        p_vals=p_vals,
        kl_vals=kl_vals,
        model_kl=model_kl,
        effective_p=effective_p,
        out_path=ref_fig_path,
        title_label="k",
        display_p_max=8,
    )

    # 2. Overview diagnostic figure with full legend and axes
    diag_fig_path = model_out_dir / f"{job_id}_phmm_diagnostics.png"
    render_phmm_diagnostics_figure(
        p_vals=p_vals,
        kl_vals=kl_vals,
        model_kl=model_kl,
        random_kl=random_kl,
        effective_p=effective_p,
        job_id=job_id,
        out_path=diag_fig_path,
    )

    # 3. CSV Table
    df = phmm_res["curve_df"].copy()
    df["model_kl_nats"] = model_kl
    df["effective_p"] = effective_p
    csv_path = model_out_dir / f"{job_id}_phmm_memory_order.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV table -> {csv_path}")

    # 4. JSON Payload
    json_path = model_out_dir / f"{job_id}_phmm_memory_order.json"
    payload = {
        "job_id": job_id,
        "evaluated_at": time.time(),
        "elapsed_seconds": elapsed,
        "effective_p": effective_p,
        "effective_k": effective_p,
        "interpolation_status": interp_status,
        "model_kl_nats": model_kl,
        "random_kl_nats": random_kl,
        "0hmm_kl_nats": phmm_res["0hmm_kl"],
        "1hmm_kl_nats": phmm_res["1hmm_kl"],
        "best_layer": model_res["best_layer_name"],
        "p_values": p_vals,
        "phmm_kl_nats": kl_vals,
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved JSON payload -> {json_path}")

    return payload


# ==============================================================================
# 6. Main CLI Orchestrator
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Estimate Effective Markov Memory Depth p* of Transformer via p-HMM Decay Curve Intersection"
    )
    parser.add_argument("--checkpoint", "-c", type=str, default=None,
                        help="Path to specific *_final.pt model checkpoint to evaluate")
    parser.add_argument("--dir", "-d", "--results-dir", dest="results_dir", type=str, default=None,
                        help="Directory to scan for *_final.pt checkpoints (default: ./experiments/results)")
    parser.add_argument("--models", "-m-filter", type=str, default=None,
                        help="Optional substring filter for model IDs when scanning directory")
    parser.add_argument("--p-max", type=int, default=15,
                        help="Maximum Markov order p to compute (default: 15)")
    parser.add_argument("--n-trajs", type=int, default=64,
                        help="Number of evaluation trajectories to simulate (default: 64)")
    parser.add_argument("--smoke", action="store_true",
                        help="Rapid smoke testing mode (16 trajectories, p_max=6)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to run inference on (default: cuda:0 if available, else cpu)")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Output directory to save evaluations and plots (default: ./experiments/phmm_results)")
    args = parser.parse_args()

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir) if args.out_dir else REPO_ROOT / "experiments" / "phmm_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    p_max = 6 if args.smoke else args.p_max

    ckpts: List[Path] = []
    if args.checkpoint:
        c_path = Path(args.checkpoint).resolve()
        if not c_path.exists():
            print(f"Error: Specified checkpoint does not exist: {c_path}", file=sys.stderr)
            sys.exit(1)
        ckpts.append(c_path)
    else:
        search_dir = Path(args.results_dir) if args.results_dir else REPO_ROOT / "experiments" / "results"
        if not search_dir.exists():
            alt_search = REPO_ROOT / "experiments"
            print(f"Notice: {search_dir} not found; falling back to {alt_search}...")
            search_dir = alt_search

        found = sorted(search_dir.rglob("*_final.pt"))
        if args.models:
            patterns = [p.strip() for p in args.models.split(",")]
            found = [c for c in found if any(pat in c.name or pat in c.parent.name for pat in patterns)]
        ckpts.extend(found)

    if not ckpts:
        print(f"No *_final.pt checkpoints found in {args.results_dir or search_dir}. Provide a --checkpoint path.")
        sys.exit(1)

    print("=" * 82)
    print(" p-HMM MARKOV MEMORY ORDER & EFFECTIVE DEPTH ESTIMATION")
    print("=" * 82)
    print(f"Device:           {device}")
    print(f"Max Order p:      {p_max}")
    print(f"Trajectories:     {16 if args.smoke else args.n_trajs}")
    print(f"Checkpoints:      {len(ckpts)} checkpoint(s) queued for evaluation")
    print(f"Output Directory: {out_dir}")
    print("=" * 82)

    summary_rows = []
    for idx, ckpt in enumerate(ckpts, 1):
        print(f"\nProcessing [{idx}/{len(ckpts)}]: {ckpt.name}")
        try:
            res = run_phmm_memory_analysis(
                ckpt_path=ckpt,
                out_dir=out_dir,
                p_max=p_max,
                n_trajs=args.n_trajs,
                device=device,
                smoke=args.smoke,
            )
            summary_rows.append({
                "job_id": res["job_id"],
                "effective_k": res["effective_k"],
                "model_kl": res["model_kl_nats"],
                "0hmm_kl": res["0hmm_kl_nats"],
                "1hmm_kl": res["1hmm_kl_nats"],
                "random_kl": res["random_kl_nats"],
                "status": res["interpolation_status"],
            })
        except Exception as e:
            print(f"Error evaluating {ckpt}: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()

    if summary_rows:
        sum_df = pd.DataFrame(summary_rows)
        master_csv = out_dir / "phmm_memory_order_summary.csv"
        sum_df.to_csv(master_csv, index=False)
        print("\n" + "=" * 82)
        print(" MASTER MEMORY DEPTH SUMMARY")
        print("=" * 82)
        print(sum_df.to_string(index=False))
        print(f"\nMaster summary saved -> {master_csv}")


if __name__ == "__main__":
    main()
