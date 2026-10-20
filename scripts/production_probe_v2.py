#!/usr/bin/env python3
"""Double Pendulum Production Probing & Metrics Evaluation Engine (v2).

Decoupled, standalone probing worker and metrics evaluator for trained double-pendulum
LookaheadTransformer checkpoints.

Evaluates the complete representation & predictive metric suite:
1. Hidden Markovian Belief States (Simplex R² via GroupKFold Ridge)
2. Physical Velocity Dynamics (omega1, omega2 R²)
3. Discrete Underlying Mood State (One-Hot Mood R²)
4. Matched Untrained Controls & Net Emergence Gain (Trained - Random R²)
5. Lookahead Language Modeling Metrics (Mean NLL, Perplexity, Top-1 Accuracy)
6. Representation Decoupling Metrics (Belief R² vs. Physics R² by layer/mode)
7. Belief Simplex 3D Orientations & 2D Orthogonal Projections
8. Lookahead Predictive Distribution Heatmaps vs. Continuous Trajectory
9. Phase 3 Emergence Synthesis Dashboard (5 publication-grade figures)
10. Atomic Per-Model and Master Distribution Archives (.zip)

Usage:
    # Run probing across all unprobed models in default results directory
    python scripts/production_probe_v2.py

    # Probe models in a specific results directory
    python scripts/production_probe_v2.py --results-dir ./experiments/outputs-smoke-v2-5k/results

    # Watch mode: continuously poll and probe checkpoints as training completes
    python scripts/production_probe_v2.py --watch 15

    # Quick smoke test with fewer eval trajectories
    python scripts/production_probe_v2.py --smoke --results-dir ./experiments/outputs-smoke-v2-5k/results

    # Force re-probing even if probe CSV exists
    python scripts/production_probe_v2.py --force

    # Only regenerate Phase 3 synthesis dashboard from existing probe CSVs
    python scripts/production_probe_v2.py --dashboard-only
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
import traceback
import zipfile
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure repository root and scripts directory are in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import matplotlib
matplotlib.use("Agg")  # Headless server safe
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import GroupKFold

from models.transformer import ModelConfig, TinyTransformer
from physics.messk import simplex_embedding, square_projection_vertices
from physics.messk_configs import DOUBLE_PENDULUM_BINS, make_process, set_double_pendulum_bins

# Enforce canonical 50x50 observation grid resolution (vocabulary = 2500)
set_double_pendulum_bins((50, 50))


# ==============================================================================
# 1. Model Architecture & Data Structures
# ==============================================================================

class LookaheadTransformer(TinyTransformer):
    """Causal decoder-only transformer trained to predict token k steps ahead."""
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


@dataclass
class DoublePendulumConfig:
    delta_v: float = 1.2
    gamma1: float = 0.65
    gamma2: float = 0.65
    dt: float = 0.2
    obs_bins: Tuple[int, int] = (50, 50)
    omega_max: float = 10.0
    joint1_action_gain: float = 2.0
    angle_mode: str = "jitter"
    delta_theta: float = 0.5


@dataclass
class HMMConfig:
    n_states: int = 4
    alpha: float = 0.7
    stay: float = 0.7


@dataclass
class ExperimentConfig:
    name: str
    physics: DoublePendulumConfig
    hmm: HMMConfig
    m: int
    n: int
    seed: int = 20260925


@dataclass
class TrainingConfig:
    name: str
    d_model: int = 128
    d_mlp: int = 512
    n_layers: int = 4
    n_heads: int = 2
    k: int = 1
    k_suffix: str = "_k1"


# ==============================================================================
# 2. Closed-Form GPU Ridge Regression Engine
# ==============================================================================

def gpu_ridge_cv(
    X_t: torch.Tensor,
    Y_t: torch.Tensor,
    groups_t: torch.Tensor,
    alphas: List[float] = [0.1, 1.0, 10.0, 50.0, 100.0],
    n_splits: int = 5,
) -> Tuple[float, float, torch.Tensor]:
    """Runs 5-fold GroupKFold Ridge regression with StandardScaler entirely on device."""
    device = X_t.device
    D = X_t.shape[1]

    # Pre-compute exact GroupKFold splits matching Scikit-Learn
    groups_np = groups_t.cpu().numpy() if isinstance(groups_t, torch.Tensor) else np.asarray(groups_t)
    cv = GroupKFold(n_splits=n_splits)
    splits = [
        (torch.as_tensor(tr, device=device), torch.as_tensor(va, device=device))
        for tr, va in cv.split(groups_np, groups=groups_np)
    ]

    alpha_mean_r2s = []
    for alpha in alphas:
        fold_r2s = []
        for tr_idx, val_idx in splits:
            X_tr, Y_tr = X_t[tr_idx], Y_t[tr_idx]
            X_va, Y_va = X_t[val_idx], Y_t[val_idx]

            # StandardScaler on train, apply to val
            mean_X = X_tr.mean(dim=0, keepdim=True)
            std_X = X_tr.std(dim=0, unbiased=False, keepdim=True)
            std_X[std_X == 0] = 1.0

            X_tr_s = (X_tr - mean_X) / std_X
            X_va_s = (X_va - mean_X) / std_X

            # Center Y
            mean_Y = Y_tr.mean(dim=0, keepdim=True)
            Y_tr_c = Y_tr - mean_Y

            # Ridge closed-form solve: (X^T X + alpha*I)^(-1) X^T Y
            reg = alpha * torch.eye(D, device=device)
            A = X_tr_s.T @ X_tr_s + reg
            B = X_tr_s.T @ Y_tr_c
            W = torch.linalg.solve(A, B)

            # Predict and calculate R^2
            Y_pred = X_va_s @ W + mean_Y
            u = ((Y_va - Y_pred) ** 2).sum(dim=0)
            v = ((Y_va - Y_va.mean(dim=0, keepdim=True)) ** 2).sum(dim=0)
            v[v == 0] = 1e-8
            r2_cols = 1.0 - (u / v)
            fold_r2s.append(r2_cols.mean().item())

        alpha_mean_r2s.append(float(np.mean(fold_r2s)))

    best_idx = int(np.argmax(alpha_mean_r2s))
    best_alpha = alphas[best_idx]
    best_cv_r2 = alpha_mean_r2s[best_idx]

    # Fit final model on 100% of data with best alpha
    mean_X_all = X_t.mean(dim=0, keepdim=True)
    std_X_all = X_t.std(dim=0, unbiased=False, keepdim=True)
    std_X_all[std_X_all == 0] = 1.0
    X_all_s = (X_t - mean_X_all) / std_X_all

    mean_Y_all = Y_t.mean(dim=0, keepdim=True)
    Y_all_c = Y_t - mean_Y_all

    reg = best_alpha * torch.eye(D, device=device)
    A_all = X_all_s.T @ X_all_s + reg
    B_all = X_all_s.T @ Y_all_c
    W_final = torch.linalg.solve(A_all, B_all)
    Y_pred_all = X_all_s @ W_final + mean_Y_all

    return best_cv_r2, best_alpha, Y_pred_all


# ==============================================================================
# 3. Synchronous Probing & Language Modeling Evaluation
# ==============================================================================

def evaluate_lookahead_metrics(
    model: LookaheadTransformer,
    proc: Any,
    device: str,
    eval_seed: int = 20260829,
    n_seqs: int = 64,
) -> Dict[str, float]:
    """Evaluates next-token cross-entropy, perplexity, and top-1 accuracy on held-out trajectories."""
    model.eval()
    k = model.k
    rng = np.random.default_rng(eval_seed + 123)
    batch = proc.sample_batch(rng, n_seqs)
    tokens_np = batch["tokens"]  # (N, seq_len)
    tokens_t = torch.as_tensor(tokens_np, dtype=torch.long, device=device)

    with torch.no_grad():
        logits = model(tokens_t)  # (N, seq_len, vocab_size)
        pred_logits = logits[:, :-k].reshape(-1, logits.shape[-1])
        targets = tokens_t[:, k:].reshape(-1)
        ce_loss = F.cross_entropy(pred_logits, targets).item()
        perplexity = float(np.exp(min(ce_loss, 50.0)))
        top1_acc = (pred_logits.argmax(dim=-1) == targets).float().mean().item()

    return {
        "lookahead_nll": ce_loss,
        "perplexity": perplexity,
        "lookahead_top1_acc": top1_acc,
    }


def run_full_probing_suite(
    trained_model: LookaheadTransformer,
    random_model: LookaheadTransformer,
    proc: Any,
    exp_name: str,
    training_name: str,
    device: str,
    n_eval_trajs: int = 512,
    eval_seed: int = 20260829,
) -> Tuple[pd.DataFrame, Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]], Dict[str, float]]:
    """Runs synchronous multi-target Ridge regression probing on device across all layers and modes."""
    eval_rng = np.random.default_rng(eval_seed)
    eval_batch = proc.sample_batch(eval_rng, n_eval_trajs)
    tokens = eval_batch["tokens"]
    eval_tokens = torch.as_tensor(tokens, dtype=torch.long, device=device)

    # Compute baseline information-theoretic metrics on the sampled beliefs
    k_states = proc.chain.n_states
    b_np = eval_batch["beliefs"].reshape(-1, k_states)
    belief_entropy_norm = float(-(b_np * np.log(np.clip(b_np, 1e-12, None))).sum(1).mean() / np.log(k_states))
    oracle_mood_recovery = float((b_np.argmax(1) == eval_batch["moods"].reshape(-1)).mean())

    N, seq_len = tokens.shape
    n = proc.n_steps
    m = proc.m
    k = trained_model.k
    assert seq_len == m * n, f"Expected seq_len == m * n, got {seq_len} != {m} * {n}"

    # Extract residual streams directly on device in small chunks to prevent VRAM spikes
    def extract_streams(model: LookaheadTransformer) -> List[torch.Tensor]:
        streams = [[] for _ in range(len(model.blocks))]
        chunk_size = max(1, min(64, 32_768 // max(1, seq_len))) if device != "cpu" else 4
        with torch.no_grad():
            for i in range(0, eval_tokens.shape[0], chunk_size):
                tok_chunk = eval_tokens[i : i + chunk_size]
                x = model._embed(tok_chunk)
                for b_idx, block in enumerate(model.blocks):
                    x = block(x)
                    streams[b_idx].append(x.cpu())
        return [torch.cat(s, dim=0) for s in streams]

    trained_streams = extract_streams(trained_model)
    random_streams = extract_streams(random_model)

    # Targets on device
    beliefs_np = eval_batch["beliefs"] @ simplex_embedding(proc.chain.n_states)
    metric_np = eval_batch["metric"]
    moods_np = eval_batch["moods"]  # (N, seq_len) discrete moods

    embedded_beliefs_t = torch.as_tensor(beliefs_np, dtype=torch.float32, device=device)
    physics_metric_t = torch.as_tensor(metric_np, dtype=torch.float32, device=device)
    moods_t = torch.as_tensor(moods_np, dtype=torch.long, device=device)
    mood_one_hot_t = F.one_hot(moods_t, num_classes=k_states).float()

    # Pre-calculate usable tokens indices: the k tokens right before each action kick
    usable_indices = np.concatenate([
        np.arange(c * n + n - k, c * n + n)
        for c in range(m)
    ])
    usable_indices_t = torch.as_tensor(usable_indices, dtype=torch.long, device=device)

    probe_modes = ["all_tokens_probing", "usable_tokens_probing"]
    target_domains = ["belief", "physics", "mood"]

    results = []
    simplex_payload = None
    model_id = f"{exp_name}_{training_name}"
    n_layers = len(trained_streams)

    for layer_idx in range(n_layers):
        s_tr = trained_streams[layer_idx].to(device)
        s_rnd = random_streams[layer_idx].to(device)

        for mode in probe_modes:
            if mode == "all_tokens_probing":
                X_tr = s_tr.reshape(N * seq_len, -1)
                X_rnd = s_rnd.reshape(N * seq_len, -1)
                groups = torch.repeat_interleave(torch.arange(N, device=device), seq_len)
            elif mode == "usable_tokens_probing":
                X_tr = s_tr[:, usable_indices_t, :].reshape(N * len(usable_indices), -1)
                X_rnd = s_rnd[:, usable_indices_t, :].reshape(N * len(usable_indices), -1)
                groups = torch.repeat_interleave(torch.arange(N, device=device), len(usable_indices))
            else:
                raise ValueError(f"Unknown mode: {mode}")

            for domain in target_domains:
                if domain == "belief":
                    data = embedded_beliefs_t
                elif domain == "physics":
                    data = physics_metric_t
                elif domain == "mood":
                    data = mood_one_hot_t
                else:
                    raise ValueError(f"Unknown domain: {domain}")

                if mode == "all_tokens_probing":
                    y = data.reshape(N * seq_len, -1)
                elif mode == "usable_tokens_probing":
                    y = data[:, usable_indices_t, :].reshape(N * len(usable_indices), -1)
                else:
                    raise ValueError(f"Unknown mode: {mode}")

                # Fast GPU closed-form Ridge regression CV
                r2_trained, best_alpha, pred_y = gpu_ridge_cv(X_tr, y, groups)
                r2_random, _, _ = gpu_ridge_cv(X_rnd, y, groups)

                # Capture simplex payload for 3D/2D projection plot (last layer under usable_tokens_probing)
                if domain == "belief" and mode == "usable_tokens_probing" and layer_idx == n_layers - 1:
                    exact_y_np = y.cpu().numpy()
                    pred_y_np = pred_y.cpu().numpy()
                    mood_labels = eval_batch["moods"][:, usable_indices].reshape(-1)
                    simplex_payload = (exact_y_np, pred_y_np, mood_labels)

                results.append({
                    "model_id": model_id,
                    "exp_name": exp_name,
                    "training_name": training_name,
                    "layer": layer_idx,
                    "probe_mode": mode,
                    "target": domain,
                    "trained_r2": r2_trained,
                    "random_r2": r2_random,
                    "trained_minus_random": r2_trained - r2_random,
                    "best_alpha": best_alpha,
                })

    meta_info = {
        "belief_entropy_normalised": belief_entropy_norm,
        "oracle_mood_recovery": oracle_mood_recovery,
        "n_states": k_states,
    }

    return pd.DataFrame(results), simplex_payload, meta_info


# ==============================================================================
# 4. Standardized Diagnostic Plots
# ==============================================================================

def plot_belief_simplex_projections(
    exact_points: np.ndarray,
    pred_points: np.ndarray,
    labels: np.ndarray,
    save_path: Path,
    title: str = "Probed Belief Simplex Geometry",
) -> None:
    """Renders reconstructed belief simplex in 3D orientations and 2D projections."""
    vertices_3d = simplex_embedding(4)
    vertices_sq = square_projection_vertices()
    bary_exact = 0.25 + exact_points @ np.linalg.pinv(vertices_3d)
    bary_pred = 0.25 + pred_points @ np.linalg.pinv(vertices_3d)
    pred_sq = bary_pred @ vertices_sq
    exact_sq = bary_exact @ vertices_sq

    # Subsample points if large for snappy rendering and crisp vector graphics
    max_pts = 2000
    if len(labels) > max_pts:
        step = max(1, len(labels) // max_pts)
        exact_points = exact_points[::step]
        pred_points = pred_points[::step]
        labels = labels[::step]
        exact_sq = exact_sq[::step]
        pred_sq = pred_sq[::step]

    fig = plt.figure(figsize=(16, 8), dpi=150, facecolor="#fcfcfb")
    views_3d = [(18, 35), (18, 125), (65, 45), (5, 70)]
    colors = ("#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7")

    # 1. 3D Orientations
    for idx, (elev, azim) in enumerate(views_3d, 1):
        ax = fig.add_subplot(2, 4, idx, projection="3d", facecolor="#fcfcfb")
        for i, j in combinations(range(4), 2):
            ax.plot(*zip(vertices_3d[i], vertices_3d[j]), color="#898781", lw=0.8, alpha=0.7)
        ax.scatter(*exact_points.T, c="#c3c2b7", s=2, alpha=0.15, linewidths=0, depthshade=False, rasterized=True)
        for s in range(4):
            mask = (labels == s)
            ax.scatter(*pred_points[mask].T, c=colors[s], s=4, alpha=0.35, linewidths=0, depthshade=False, rasterized=True)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=elev, azim=azim)
        ax.set_axis_off()
        ax.set_title(f"3D View (elev={elev}°, azim={azim}°)", fontsize=9)

    # 2. 2D Orthogonal Projections
    dims_list = [(0, 1), (0, 2), (1, 2)]
    for idx, dims in enumerate(dims_list, 5):
        ax = fig.add_subplot(2, 4, idx, facecolor="#fcfcfb")
        for i, j in combinations(range(4), 2):
            ax.plot(*zip(vertices_3d[i, list(dims)], vertices_3d[j, list(dims)]), color="#898781", lw=0.8, alpha=0.7)
        ax.scatter(exact_points[:, dims[0]], exact_points[:, dims[1]], c="#c3c2b7", s=2, alpha=0.15, linewidths=0, rasterized=True)
        for s in range(4):
            mask = (labels == s)
            ax.scatter(pred_points[mask, dims[0]], pred_points[mask, dims[1]], c=colors[s], s=4, alpha=0.35, linewidths=0, rasterized=True)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(f"2D Projection {'XYZ'[dims[0]]}{'XYZ'[dims[1]]}", fontsize=9)

    # 3. 2D Planar Square View
    ax_sq = fig.add_subplot(2, 4, 8, facecolor="#fcfcfb")
    ring = np.vstack([vertices_sq, vertices_sq[0]])
    ax_sq.plot(ring[:, 0], ring[:, 1], color="#898781", lw=1.0)
    ax_sq.plot([-1, 1], [-1, 1], color="#c3c2b7", lw=0.5)
    ax_sq.plot([-1, 1], [1, -1], color="#c3c2b7", lw=0.5)
    ax_sq.scatter(exact_sq[:, 0], exact_sq[:, 1], c="#c3c2b7", s=2, alpha=0.15, linewidths=0, rasterized=True)
    for s in range(4):
        mask = (labels == s)
        ax_sq.scatter(pred_sq[mask, 0], pred_sq[mask, 1], c=colors[s], s=4, alpha=0.35, linewidths=0, rasterized=True)
    ax_sq.set_aspect("equal")
    ax_sq.axis("off")
    ax_sq.set_title("2D Planar Square View", fontsize=9)

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#c3c2b7", markersize=6, label="Exact Target Set"),
    ] + [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=colors[s], markersize=6, label=f"Mood {s}")
        for s in range(4)
    ]
    fig.legend(handles=legend_elements, loc="upper center", bbox_to_anchor=(0.5, 0.96), ncol=5, frameon=False, fontsize=9)
    fig.suptitle(title, fontsize=12, y=0.99, fontweight="bold")
    plt.tight_layout(rect=(0.01, 0.02, 0.99, 0.94))
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_lookahead_probability_map(
    model: LookaheadTransformer,
    proc: Any,
    save_path: Path,
    k_suffix: str = "",
    fixed_eval_seed: int = 20260829,
    device: str = "cpu",
) -> None:
    """Generates standardized lookahead predictive distribution heatmap on fixed held-out sequence."""
    model.eval()
    k = model.k
    rng = np.random.default_rng(fixed_eval_seed)
    seq = proc.sample_batch(rng, 1)["tokens"][0]  # shape (seq_len,)
    tokens_tensor = torch.as_tensor(seq[np.newaxis, :], dtype=torch.long, device=device)

    with torch.no_grad():
        logits = model(tokens_tensor)[0]  # (seq_len, vocab_size)
        probs = F.softmax(logits, dim=-1).cpu().numpy()

    if k < len(seq):
        pred_probs = probs[:-k]
        actual = seq[k:]
        fig, ax = plt.subplots(figsize=(12, 5), dpi=150)
        im = ax.imshow(pred_probs.T, origin="lower", aspect="auto", cmap="magma")
        ax.plot(np.arange(len(actual)), actual, color="cyan", lw=1.0, label="Actual")
        nll = -np.log(np.maximum(pred_probs[np.arange(len(actual)), actual], 1e-12)).mean()
        suf = f" ({k_suffix})" if k_suffix else ""
        ax.set_title(f"Horizon k={k}{suf} | Mean NLL: {nll:.3f}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Time step t")
        ax.set_ylabel(f"Token bin (0..{proc.n_obs - 1})")
        ax.legend(loc="upper right")
        plt.colorbar(im, ax=ax, label=f"P(token_{{t+{k}}} | tokens_{{<=t}})")
        plt.tight_layout()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()


def plot_lookahead_comparison_heatmaps(
    exp_cfg: ExperimentConfig,
    proc: Any,
    results_dir: Path,
    save_path: Path,
    device: str = "cpu",
    fixed_eval_seed: int = 20260829,
) -> None:
    """Generates 3 side-by-side heatmaps (k=1, k=n//2, k=n)."""
    n = exp_cfg.n
    horizons = [
        (1, "_k1"),
        (max(1, n // 2), "_k_mid"),
        (n, "_k_full"),
    ]
    rng = np.random.default_rng(fixed_eval_seed)
    tokens_single = proc.sample_batch(rng, 1)["tokens"][0]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=150, sharey=True)
    width = 128

    for col_idx, (k, k_suffix) in enumerate(horizons):
        ax = axes[col_idx]
        train_name = f"dmodel_{width}_dmlp_{4 * width}{k_suffix}"
        model_id = f"{exp_cfg.name}_{train_name}"
        model_dir = results_dir / model_id
        pt_path = model_dir / f"{model_id}_final.pt"
        json_path = model_dir / f"{model_id}_final.json"

        if pt_path.exists():
            layer_count = 4
            if json_path.exists():
                try:
                    meta = json.loads(json_path.read_text())
                    layer_count = meta.get("n_layers", 4)
                except Exception:
                    pass
            cfg = ModelConfig(
                vocab_size=proc.n_obs,
                n_ctx=proc.seq_len,
                n_layers=layer_count,
                n_heads=2,
                d_model=width,
                d_mlp=4 * width,
                seed=exp_cfg.seed,
            )
            model = LookaheadTransformer(cfg, k=k)
            model.load_state_dict(torch.load(pt_path, map_location=device, weights_only=True))
            model.to(device).eval()

            tokens_tensor = torch.as_tensor(tokens_single[np.newaxis, :], dtype=torch.long, device=device)
            with torch.no_grad():
                logits = model(tokens_tensor)[0]
                probs = F.softmax(logits, dim=-1).cpu().numpy()

            if k < len(tokens_single):
                pred_probs = probs[:-k]
                actual = tokens_single[k:]
                im = ax.imshow(pred_probs.T, origin="lower", aspect="auto", cmap="magma")
                ax.plot(np.arange(len(actual)), actual, color="cyan", lw=1.2, label="Actual")
                nll = -np.log(np.maximum(pred_probs[np.arange(len(actual)), actual], 1e-12)).mean()
                ax.set_title(f"Horizon {k_suffix} (k={k}) | Mean NLL: {nll:.3f}")
                ax.legend(loc="upper right")
            else:
                ax.text(0.5, 0.5, f"k={k} >= seq_len", ha="center", va="center")
        else:
            ax.text(0.5, 0.5, f"Model not trained:\n{k_suffix} (k={k})", ha="center", va="center")

        ax.set_xlabel("Time step t")
        if col_idx == 0:
            ax.set_ylabel(f"Token bin (0..{proc.n_obs - 1})")

    plt.suptitle(f"Lookahead Predictive Distributions: {exp_cfg.name}", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ==============================================================================
# 5. Phase 3 Emergence Synthesis Dashboard (5-Panel)
# ==============================================================================

def generate_synthesis_dashboard(
    probe_df_all: pd.DataFrame,
    out_dir: Path,
    configs: Optional[List[ExperimentConfig]] = None,
    device: str = "cpu",
) -> None:
    """Generates the high-resolution synthesis figures and consolidated master results."""
    print("\n" + "=" * 78)
    print(" PHASE 3: EMERGENCE SYNTHESIS & 5-PANEL DASHBOARD")
    print("=" * 78)
    out_dir.mkdir(parents=True, exist_ok=True)

    probe_df_all = probe_df_all.copy()
    probe_df_all["d_model"] = 128
    probe_df_all["lookahead"] = probe_df_all["training_name"].apply(lambda s: s.split("_k")[-1] if "_k" in s else "k1")

    # 1. scaling_curves.png -> Config Performance Comparison across configurations
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    cfg_grouped = probe_df_all.groupby(["exp_name", "target"])["trained_r2"].mean().unstack()
    if not cfg_grouped.empty:
        cfg_grouped.plot(kind="bar", ax=ax, colormap="tab10", width=0.6)
    ax.set(xlabel="Physics Configuration", ylabel="Mean Test R²", title="1. Physics Configurations Performance (d_model=128)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha="right")
    ax.grid(True, alpha=0.3)
    fig.savefig(out_dir / "scaling_curves.png", bbox_inches="tight")
    plt.close()

    # 2. lookahead_comparison.png & lookahead_heatmaps.png
    if configs:
        for idx, exp_cfg in enumerate(configs):
            proc_temp = make_process(
                "double_pendulum_mess4",
                m=exp_cfg.m,
                n_steps=exp_cfg.n,
                delta_v=exp_cfg.physics.delta_v,
                system={
                    "gamma1": exp_cfg.physics.gamma1,
                    "gamma2": exp_cfg.physics.gamma2,
                    "angle_mode": getattr(exp_cfg.physics, "angle_mode", "jitter"),
                    "delta_theta": getattr(exp_cfg.physics, "delta_theta", 0.5),
                },
            )
            cfg_heatmap_path = out_dir / f"{exp_cfg.name}_lookahead_heatmaps.png"
            plot_lookahead_comparison_heatmaps(exp_cfg, proc_temp, out_dir, cfg_heatmap_path, device=device)
            if idx == 0:
                plot_lookahead_comparison_heatmaps(exp_cfg, proc_temp, out_dir, out_dir / "lookahead_comparison.png", device=device)
                plot_lookahead_comparison_heatmaps(exp_cfg, proc_temp, out_dir, out_dir / "lookahead_heatmaps.png", device=device)
    else:
        fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
        lh_grouped = probe_df_all.groupby(["lookahead", "target"])["trained_r2"].mean().unstack()
        if not lh_grouped.empty:
            lh_grouped.plot(kind="bar", ax=ax, colormap="tab10", width=0.6)
        ax.set(xlabel="Lookahead Horizon", ylabel="Mean Test R²", title="2. Forecasting Horizon Comparison")
        ax.grid(True, alpha=0.3)
        fig.savefig(out_dir / "lookahead_comparison.png", bbox_inches="tight")
        plt.close()

    # 3. probe_views_breakdown.png -> Layer-Wise Breakdown across Probe Modes
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    pv_grouped = probe_df_all.groupby(["layer", "probe_mode", "target"])["trained_r2"].mean().unstack()
    if not pv_grouped.empty:
        pv_grouped.plot(kind="bar", ax=ax, colormap="Set2", width=0.7)
    ax.set(xlabel="(Layer, Probe Mode)", ylabel="Mean Test R²", title="3. Layer-Wise Probing Performance Breakdown")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha="right")
    ax.grid(True, alpha=0.3)
    fig.savefig(out_dir / "probe_views_breakdown.png", bbox_inches="tight")
    plt.close()

    # 4. physics_vs_belief_scatter.png -> Representation Decoupling
    fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
    p_b = probe_df_all.pivot_table(index=["model_id", "layer", "probe_mode"], columns="target", values="trained_r2").reset_index()
    if "belief" in p_b.columns and "physics" in p_b.columns:
        layers = sorted(p_b["layer"].unique()) if "layer" in p_b.columns else [0]
        layer_colors = plt.cm.viridis(np.linspace(0, 1, max(1, len(layers))))
        color_map = {l: layer_colors[i] for i, l in enumerate(layers)}
        marker_map = {
            "all_tokens_probing": "o",
            "usable_tokens_probing": "s",
        }
        for _, row in p_b.iterrows():
            layer = row["layer"] if "layer" in row else 0
            mode = row["probe_mode"]
            color = color_map.get(layer, "#1f77b4")
            marker = marker_map.get(mode, "o")
            ax.scatter(row["physics"], row["belief"], color=color, marker=marker, s=80, alpha=0.85, edgecolors="k", linewidths=0.5)

        ax.axhline(0.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.axvline(0.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.set(xlabel="Physics Test R² (Angular Velocities)", ylabel="Belief Test R² (Simplex Coordinates)",
               title="4. Representation Decoupling: Belief vs. Physics Readability (by Layer & Mode)")
        ax.grid(True, alpha=0.3)
        fig.savefig(out_dir / "physics_vs_belief_scatter.png", bbox_inches="tight")
        plt.close()

    # 5. trained_vs_random_gain.png
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    gain_grouped = probe_df_all.groupby(["layer", "probe_mode"])["trained_minus_random"].mean().unstack()
    if not gain_grouped.empty:
        gain_grouped.plot(kind="bar", ax=ax, colormap="coolwarm", width=0.6)
    ax.set(xlabel="Layer", ylabel="Mean R² Gain (Trained - Random)", title="5. Net Representation Emergence Gain by Layer")
    if not gain_grouped.empty:
        ax.set_xticklabels([f"Layer {l}" for l in gain_grouped.index], rotation=0)
    ax.grid(True, alpha=0.3)
    fig.savefig(out_dir / "trained_vs_random_gain.png", bbox_inches="tight")
    plt.close()

    print(f"5-Panel Synthesis Dashboard successfully compiled in {out_dir}:")
    for f in ["scaling_curves.png", "lookahead_comparison.png", "lookahead_heatmaps.png", "probe_views_breakdown.png", "physics_vs_belief_scatter.png", "trained_vs_random_gain.png"]:
        if (out_dir / f).exists():
            print(f"  -> {out_dir / f}")


# ==============================================================================
# 6. Model Discovery & Single-Model Probe Orchestrator
# ==============================================================================

def parse_model_metadata(model_dir: Path, job_id: str) -> Tuple[ExperimentConfig, TrainingConfig]:
    """Extracts ExperimentConfig and TrainingConfig from sidecar JSON or job_id fallback."""
    sidecar_path = model_dir / f"{job_id}_final.json"
    if sidecar_path.exists():
        try:
            with open(sidecar_path, "r") as f:
                sidecar = json.load(f)
            exp_data = sidecar["exp_config"]
            phys_data = exp_data["physics"]
            phys_cfg = DoublePendulumConfig(
                delta_v=float(phys_data.get("delta_v", 1.2)),
                gamma1=float(phys_data.get("gamma1", 0.65)),
                gamma2=float(phys_data.get("gamma2", 0.65)),
                dt=float(phys_data.get("dt", 0.2)),
                obs_bins=tuple(phys_data.get("obs_bins", (50, 50))),
                omega_max=float(phys_data.get("omega_max", 10.0)),
                joint1_action_gain=float(phys_data.get("joint1_action_gain", 2.0)),
            )
            hmm_cfg = HMMConfig(
                n_states=int(exp_data.get("hmm", {}).get("n_states", 4)),
                alpha=float(exp_data.get("hmm", {}).get("alpha", 0.7)),
                stay=float(exp_data.get("hmm", {}).get("stay", 0.7)),
            )
            exp_cfg = ExperimentConfig(
                name=exp_data.get("name", "tuple"),
                physics=phys_cfg,
                hmm=hmm_cfg,
                m=int(exp_data.get("m", 200)),
                n=int(exp_data.get("n", 10)),
                seed=int(exp_data.get("seed", 20260925)),
            )
            train_data = sidecar.get("train_config", {})
            train_cfg = TrainingConfig(
                name=train_data.get("name", "dmodel_128_dmlp_512"),
                d_model=int(sidecar.get("d_model", train_data.get("d_model", 128))),
                d_mlp=int(sidecar.get("d_mlp", train_data.get("d_mlp", 512))),
                n_layers=int(train_data.get("n_layers", 4)),
                n_heads=int(train_data.get("n_heads", 2)),
                k=int(sidecar.get("k", train_data.get("k", 1))),
                k_suffix=sidecar.get("k_suffix", train_data.get("k_suffix", "_k1")),
            )
            return exp_cfg, train_cfg
        except Exception as e:
            print(f"[{job_id}] Notice: Error reading sidecar JSON ({e}), parsing from ID...", flush=True)

    # Fallback parse from job_id: e.g. tuple01_dv0.80_g0.10_n10_dmodel_128_dmlp_512_k1
    m_dv = re.search(r"dv([0-9.]+)", job_id)
    m_g = re.search(r"g([0-9.]+)", job_id)
    m_n = re.search(r"n([0-9]+)", job_id)
    m_dm = re.search(r"dmodel_([0-9]+)", job_id)
    m_dmlp = re.search(r"dmlp_([0-9]+)", job_id)

    dv = float(m_dv.group(1)) if m_dv else 1.2
    gamma = float(m_g.group(1)) if m_g else 0.65
    n = int(m_n.group(1)) if m_n else 10
    m = 2000 // n
    d_model = int(m_dm.group(1)) if m_dm else 128
    d_mlp = int(m_dmlp.group(1)) if m_dmlp else 4 * d_model

    k = 1
    k_suffix = "_k1"
    if "_k_mid" in job_id:
        k = max(1, n // 2)
        k_suffix = "_k_mid"
    elif "_k_full" in job_id:
        k = n
        k_suffix = "_k_full"
    elif "_k" in job_id:
        m_k = re.search(r"_k([0-9]+)", job_id)
        if m_k:
            k = int(m_k.group(1))
            k_suffix = f"_k{k}"

    tuple_name = f"tuple_dv{dv:.2f}_g{gamma:.2f}_n{n}"
    phys_cfg = DoublePendulumConfig(delta_v=dv, gamma1=gamma, gamma2=gamma, dt=0.2, obs_bins=(50, 50))
    exp_cfg = ExperimentConfig(name=tuple_name, physics=phys_cfg, hmm=HMMConfig(), m=m, n=n)
    train_cfg = TrainingConfig(name=f"dmodel_{d_model}_dmlp_{d_mlp}{k_suffix}", d_model=d_model, d_mlp=d_mlp, k=k, k_suffix=k_suffix)
    return exp_cfg, train_cfg


def probe_single_checkpoint(
    model_dir: Path,
    results_dir: Path,
    device: str = "cuda:0",
    smoke: bool = False,
    force: bool = False,
    seq_len_override: Optional[int] = None,
) -> Optional[Tuple[str, pd.DataFrame, Dict[str, Any]]]:
    """Evaluates all metrics for a single model checkpoint."""
    job_id = model_dir.name
    ckpt_path = model_dir / f"{job_id}_final.pt"
    probe_csv_path = model_dir / f"{job_id}_probes.csv"
    probe_json_path = model_dir / f"{job_id}_probes.json"
    prob_map_path = model_dir / f"{job_id}_lookahead_prob_map.png"
    simplex_path = model_dir / f"{job_id}_belief_simplex.png"
    summary_json_path = model_dir / f"{job_id}_metrics_summary.json"
    zip_path = results_dir / f"{job_id}.zip"

    if not ckpt_path.exists():
        return None

    # Check auto-resume
    if not force and probe_csv_path.exists() and summary_json_path.exists() and zip_path.exists():
        try:
            probe_df = pd.read_csv(probe_csv_path)
            with open(summary_json_path, "r") as f:
                summary = json.load(f)
            return job_id, probe_df, summary
        except Exception:
            pass

    t0 = time.perf_counter()
    exp_cfg, train_cfg = parse_model_metadata(model_dir, job_id)

    m_val = (seq_len_override // exp_cfg.n) if seq_len_override is not None else exp_cfg.m

    proc = make_process(
        "double_pendulum_mess4",
        m=m_val,
        n_steps=exp_cfg.n,
        delta_v=exp_cfg.physics.delta_v,
        system={
            "gamma1": exp_cfg.physics.gamma1,
            "gamma2": exp_cfg.physics.gamma2,
            "angle_mode": getattr(exp_cfg.physics, "angle_mode", "jitter"),
            "delta_theta": getattr(exp_cfg.physics, "delta_theta", 0.5),
        },
    )

    cfg = ModelConfig(
        vocab_size=proc.n_obs,
        n_ctx=proc.seq_len,
        n_layers=train_cfg.n_layers,
        n_heads=train_cfg.n_heads,
        d_model=train_cfg.d_model,
        d_mlp=train_cfg.d_mlp,
        seed=exp_cfg.seed,
    )

    target_device = device if (torch.cuda.is_available() and "cuda" in device) else ("cuda:0" if torch.cuda.is_available() else "cpu")
    trained_model = None
    random_model = None

    try:
        trained_model = LookaheadTransformer(cfg, k=train_cfg.k).to(target_device)
        trained_model.load_state_dict(torch.load(ckpt_path, map_location=target_device, weights_only=True))
        random_model = LookaheadTransformer(cfg, k=train_cfg.k).to(target_device)  # Untrained control

        # 1. Full Multi-Target GPU Probing Suite
        n_eval_trajs = 8 if smoke else (128 if proc.seq_len >= 2000 else 512)
        probe_df, simplex_data, meta_info = run_full_probing_suite(
            trained_model=trained_model,
            random_model=random_model,
            proc=proc,
            exp_name=exp_cfg.name,
            training_name=train_cfg.name,
            device=target_device,
            n_eval_trajs=n_eval_trajs,
        )

        # 2. Lookahead Predictive Language Modeling Metrics
        lm_metrics = evaluate_lookahead_metrics(
            model=trained_model,
            proc=proc,
            device=target_device,
            n_seqs=8 if smoke else 64,
        )

        # Save probe evaluations
        probe_df.to_csv(probe_csv_path, index=False)
        probe_df.to_json(probe_json_path, orient="records", indent=2)

        # 3. Standardized Diagnostic Plots
        plot_lookahead_probability_map(
            model=trained_model,
            proc=proc,
            save_path=prob_map_path,
            k_suffix=train_cfg.k_suffix,
            device=target_device,
        )

        if simplex_data is not None:
            plot_belief_simplex_projections(
                *simplex_data,
                save_path=simplex_path,
                title=f"Probed Belief Simplex Geometry (k={train_cfg.k}) — {job_id}",
            )

        # Compile comprehensive metrics summary
        b_df = probe_df[probe_df["target"] == "belief"]
        p_df = probe_df[probe_df["target"] == "physics"]
        m_df = probe_df[probe_df["target"] == "mood"]

        peak_belief_row = b_df.loc[b_df["trained_r2"].idxmax()] if not b_df.empty else None
        peak_physics_row = p_df.loc[p_df["trained_r2"].idxmax()] if not p_df.empty else None
        peak_mood_row = m_df.loc[m_df["trained_r2"].idxmax()] if not m_df.empty else None

        summary: Dict[str, Any] = {
            "job_id": job_id,
            "exp_name": exp_cfg.name,
            "training_name": train_cfg.name,
            "k": train_cfg.k,
            "k_suffix": train_cfg.k_suffix,
            "runtime_seconds": time.perf_counter() - t0,
            "lookahead_nll": lm_metrics["lookahead_nll"],
            "perplexity": lm_metrics["perplexity"],
            "lookahead_top1_acc": lm_metrics["lookahead_top1_acc"],
            "belief_entropy_normalised": meta_info["belief_entropy_normalised"],
            "oracle_mood_recovery": meta_info["oracle_mood_recovery"],
            "peak_belief_r2": float(peak_belief_row["trained_r2"]) if peak_belief_row is not None else 0.0,
            "peak_belief_layer": int(peak_belief_row["layer"]) if peak_belief_row is not None else -1,
            "peak_belief_mode": str(peak_belief_row["probe_mode"]) if peak_belief_row is not None else "",
            "peak_belief_gain": float(peak_belief_row["trained_minus_random"]) if peak_belief_row is not None else 0.0,
            "peak_physics_r2": float(peak_physics_row["trained_r2"]) if peak_physics_row is not None else 0.0,
            "peak_physics_layer": int(peak_physics_row["layer"]) if peak_physics_row is not None else -1,
            "peak_physics_gain": float(peak_physics_row["trained_minus_random"]) if peak_physics_row is not None else 0.0,
            "peak_mood_r2": float(peak_mood_row["trained_r2"]) if peak_mood_row is not None else 0.0,
            "peak_mood_layer": int(peak_mood_row["layer"]) if peak_mood_row is not None else -1,
            "peak_mood_gain": float(peak_mood_row["trained_minus_random"]) if peak_mood_row is not None else 0.0,
            "decoupling_score_belief_minus_physics": (
                (float(peak_belief_row["trained_r2"]) if peak_belief_row is not None else 0.0) -
                (float(peak_physics_row["trained_r2"]) if peak_physics_row is not None else 0.0)
            ),
        }

        with open(summary_json_path, "w") as f:
            json.dump(summary, f, indent=2)

        # 4. Instant atomic per-model zip packaging
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in model_dir.rglob("*"):
                if f.is_file():
                    zf.write(f, arcname=f.relative_to(model_dir))

        elapsed = time.perf_counter() - t0
        print(
            f"[{job_id}] Probed in {elapsed:4.1f}s | "
            f"Belief R²: {summary['peak_belief_r2']:+.3f} (L{summary['peak_belief_layer']}, Gain: {summary['peak_belief_gain']:+.3f}) | "
            f"Physics R²: {summary['peak_physics_r2']:+.3f} | NLL: {summary['lookahead_nll']:.3f} | PPL: {summary['perplexity']:.2f} -> {zip_path.name}",
            flush=True,
        )
        return job_id, probe_df, summary
    finally:
        del trained_model
        del random_model
        if "cuda" in str(target_device) and torch.cuda.is_available():
            torch.cuda.empty_cache()


# ==============================================================================
# 7. Main Probing Runner CLI
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Double Pendulum Production Probing & Metrics Evaluation Engine (v2)")
    parser.add_argument("--dir", "-d", "--results-dir", dest="results_dir", type=str, default=None,
                        help="Root experiments or results directory containing trained models")
    parser.add_argument("--device", type=str, default=None,
                        help="Compute device for probing (default: cuda:0 if available, else cpu)")
    parser.add_argument("--models", "-m", type=str, default=None,
                        help="Optional comma-separated list of job IDs or substring filter to probe")
    parser.add_argument("--force", "-f", action="store_true",
                        help="Force recomputation of probes even if {job_id}_probes.csv exists")
    parser.add_argument("--smoke", action="store_true",
                        help="Run rapid smoke probing (~8 eval trajectories)")
    parser.add_argument("--watch", "-w", type=int, default=None, metavar="SECONDS",
                        help="Watch mode: continuously poll and probe newly trained models every N seconds")
    parser.add_argument("--dashboard-only", action="store_true",
                        help="Skip model probing; only regenerate Phase 3 synthesis dashboard from existing probe CSVs")
    parser.add_argument("--seq-len", type=int, default=None,
                        help="Override evaluation sequence length m * n (default: auto from metadata or 2000)")
    args = parser.parse_args()

    # Determine directory paths
    if args.results_dir:
        target_path = Path(args.results_dir).expanduser().resolve()
    elif (REPO_ROOT / "experiments" / "results").exists():
        target_path = (REPO_ROOT / "experiments" / "results").resolve()
    elif (REPO_ROOT / "experiments").exists():
        target_path = (REPO_ROOT / "experiments").resolve()
    elif Path("/workspace/Belief-in-physics/experiments/results").exists():
        target_path = Path("/workspace/Belief-in-physics/experiments/results").resolve()
    elif Path("/workspace/Belief-in-physics/experiments").exists():
        target_path = Path("/workspace/Belief-in-physics/experiments").resolve()
    else:
        target_path = (REPO_ROOT / "experiments").resolve()

    if target_path.name == "results":
        results_dir = target_path
        output_base = target_path.parent
    elif (target_path / "results").exists():
        results_dir = target_path / "results"
        output_base = target_path
    else:
        results_dir = target_path
        output_base = target_path

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    print("=" * 78)
    print(" DOUBLE PENDULUM PRODUCTION PROBING & METRICS ENGINE (v2)")
    print("=" * 78)
    print(f"Results Directory:  {results_dir}")
    print(f"Output Base:        {output_base}")
    print(f"Compute Device:     {device}")
    print(f"Execution Mode:     {'Dashboard-Only' if args.dashboard_only else ('Watch Mode' if args.watch else 'Full Probing Pass')}")
    print(f"Smoke Mode:         {args.smoke}")
    print(f"Force Recompute:    {args.force}")
    print("=" * 78)

    master_csv_path = output_base / "master_probe_results.csv"
    master_summary_csv = output_base / "master_metrics_summary.csv"
    master_zip_path = output_base / "double_pendulum_results.zip"

    def execute_probing_cycle() -> Tuple[List[pd.DataFrame], List[Dict[str, Any]]]:
        if not results_dir.exists():
            print(f"Notice: Results directory {results_dir} does not exist yet.", flush=True)
            return [], []

        # Find all trained checkpoints
        all_ckpts = sorted(results_dir.glob("*/*_final.pt"))
        if not all_ckpts:
            print(f"Notice: No *_final.pt checkpoints found in {results_dir}.", flush=True)
            return [], []

        # Filter by --models argument if specified
        if args.models:
            patterns = [p.strip() for p in args.models.split(",")]
            all_ckpts = [c for c in all_ckpts if any(p in c.parent.name for p in patterns)]

        # Determine queue of models to probe
        queue = []
        for ckpt in all_ckpts:
            m_dir = ckpt.parent
            probe_csv = m_dir / f"{m_dir.name}_probes.csv"
            sum_json = m_dir / f"{m_dir.name}_metrics_summary.json"
            zip_p = results_dir / f"{m_dir.name}.zip"
            if args.force or not (probe_csv.exists() and sum_json.exists() and zip_p.exists()):
                queue.append(m_dir)

        print(f"Found {len(all_ckpts)} total checkpoint(s) in {results_dir.name} ({len(queue)} pending probing).", flush=True)

        probed_dfs: List[pd.DataFrame] = []
        probed_summaries: List[Dict[str, Any]] = []

        # Probe pending queue
        for idx, m_dir in enumerate(queue, 1):
            print(f"\n[{idx}/{len(queue)}] Probing {m_dir.name}...", flush=True)
            try:
                res = probe_single_checkpoint(
                    model_dir=m_dir,
                    results_dir=results_dir,
                    device=device,
                    smoke=args.smoke,
                    force=args.force,
                    seq_len_override=args.seq_len,
                )
                if res is not None:
                    _, df, summary = res
                    probed_dfs.append(df)
                    probed_summaries.append(summary)
            except Exception as e:
                print(f"[{m_dir.name}] Error during probing: {e}", flush=True)
                traceback.print_exc()

        # Always gather all available probe CSVs and summary JSONs from results_dir for dashboard
        all_dfs = []
        all_sums = []
        for ckpt in all_ckpts:
            m_dir = ckpt.parent
            p_csv = m_dir / f"{m_dir.name}_probes.csv"
            s_json = m_dir / f"{m_dir.name}_metrics_summary.json"
            if p_csv.exists():
                try:
                    all_dfs.append(pd.read_csv(p_csv))
                except Exception:
                    pass
            if s_json.exists():
                try:
                    with open(s_json, "r") as f:
                        all_sums.append(json.load(f))
                except Exception:
                    pass

        # Update Master Probe CSV
        if all_dfs:
            combined_df = pd.concat(all_dfs, ignore_index=True).drop_duplicates()
            combined_df.to_csv(master_csv_path, index=False)
            print(f"\nMaster probe evaluations updated -> {master_csv_path} ({len(combined_df)} rows)")

        # Update Master Metrics Summary Table
        if all_sums:
            sum_df = pd.DataFrame(all_sums).drop_duplicates(subset=["job_id"])
            sum_df.to_csv(master_summary_csv, index=False)
            print(f"Master metrics summary updated -> {master_summary_csv} ({len(sum_df)} models)")

            # Display Console Report Table
            print("\n" + "=" * 105)
            print(f"{'MODEL JOB ID':<48} | {'BELIEF R²':<10} | {'PHYS R²':<10} | {'GAIN':<8} | {'NLL':<6} | {'PPL':<6}")
            print("-" * 105)
            for _, r in sum_df.iterrows():
                print(
                    f"{r['job_id']:<48} | {r['peak_belief_r2']:+9.3f}  | {r['peak_physics_r2']:+9.3f}  | "
                    f"{r['peak_belief_gain']:+7.3f} | {r['lookahead_nll']:6.3f} | {r['perplexity']:6.2f}"
                )
            print("=" * 105)

            # Generate Phase 3 Emergence Synthesis Dashboard
            if all_dfs:
                combined_df = pd.concat(all_dfs, ignore_index=True).drop_duplicates()
                generate_synthesis_dashboard(combined_df, results_dir, device=device)

            # Package Master Distribution Archive
            print(f"\nPackaging master distribution archive -> {master_zip_path}...")
            with zipfile.ZipFile(master_zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                if master_csv_path.exists():
                    zf.write(master_csv_path, arcname=master_csv_path.name)
                if master_summary_csv.exists():
                    zf.write(master_summary_csv, arcname=master_summary_csv.name)
                for f in results_dir.rglob("*"):
                    if f.is_file() and f.suffix != ".zip":
                        zf.write(f, arcname=f"results/{f.relative_to(results_dir)}")

            print(f"Master distribution package complete: {master_zip_path} ({master_zip_path.stat().st_size / 1024:.1f} KB)")

        return probed_dfs, probed_summaries

    if args.dashboard_only:
        # Load existing probe CSVs and regenerate dashboard
        all_dfs = []
        for p_csv in results_dir.glob("*/*_probes.csv"):
            try:
                all_dfs.append(pd.read_csv(p_csv))
            except Exception:
                pass
        if all_dfs:
            combined_df = pd.concat(all_dfs, ignore_index=True).drop_duplicates()
            generate_synthesis_dashboard(combined_df, results_dir, device=device)
        else:
            print("No existing probe CSVs found to compile dashboard.")
        return

    if args.watch:
        print(f"\nStarting watch mode: checking for new models every {args.watch}s (Ctrl+C to stop)...")
        try:
            while True:
                execute_probing_cycle()
                time.sleep(args.watch)
        except KeyboardInterrupt:
            print("\n[!] Watch loop stopped by user.")
    else:
        execute_probing_cycle()


if __name__ == "__main__":
    main()
