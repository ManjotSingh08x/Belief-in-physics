#!/usr/bin/env python3
"""Measure KL Divergence between Transformer Representations and the Optimal Observer Belief State.

Primary Benchmark: Option B (Internal Residual Stream Belief State Probing)
- Optimal Observer (Full-HMM / infinity-HMM): Exact recursive Bayesian belief state
    b_c = P(mood_{c+1} | letters_{0..c}) on the 3-simplex.
- 1-HMM Baseline: Myopic belief state conditioning only on the single preceding letter L_c.
- 0-HMM Baseline: Memoryless stationary prior over moods pi = [0.25, 0.25, 0.25, 0.25].
- Architecture-matched Random Control Model: Identically probed untrained network.
- Trained Model Probed Belief: Softmax linear probe fitted on residual stream activations
    \\hat{b}_c = softmax(W h_c + c) minimizing cross-entropy (KL divergence).

Configuration:
- Uses m = 20 cycles by default (e.g. 20 cycles x 10 steps = 200 tokens per sequence for n=10).
- Sequence-level train/test split (70% train, 30% test) to prevent sequence leakage.
- Generates 4-panel publication-grade diagnostic plots (cycle convergence, layer profile,
  suboptimality ladder, and 2D belief simplex square projection).

Usage:
  uv run python scripts/measure_optimal_observer_kl.py --checkpoint path/to/model_final.pt --m 20
  uv run python scripts/measure_optimal_observer_kl.py --dir ./experiments/results --m 20
  uv run python scripts/measure_optimal_observer_kl.py --smoke --m 20
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
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
from physics.messk import square_projection_vertices
from physics.messk_configs import make_process


# ==============================================================================
# 1. Lookahead / Checkpoint Compatibility Helpers
# ==============================================================================

class LookaheadTransformer(TinyTransformer):
    """Transformer trained with k-step lookahead loss (logits at t predict t+k)."""
    def __init__(self, config: ModelConfig, k: int = 1):
        super().__init__(config)
        self.k = k

    def loss(self, tokens: torch.Tensor) -> torch.Tensor:
        logits = self(tokens)
        if self.k >= tokens.shape[1]:
            raise ValueError(f"lookahead k={self.k} must be < sequence length {tokens.shape[1]}")
        return F.cross_entropy(logits[:, :-self.k].reshape(-1, logits.shape[-1]), tokens[:, self.k:].reshape(-1))


def load_model_and_process(
    ckpt_path: Path,
    device: str = "cpu",
    m_cycles: int = 20,
    override_seq_len: Optional[int] = None,
) -> Tuple[TinyTransformer, TinyTransformer, Any, Dict[str, Any]]:
    """Loads trained checkpoint, matched random control, and reconstructed process with m cycles."""
    model_dir = ckpt_path.parent
    job_id = model_dir.name
    sidecar_path = model_dir / f"{job_id}_final.json"

    meta: Dict[str, Any] = {}
    if sidecar_path.exists():
        try:
            meta = json.loads(sidecar_path.read_text())
        except Exception:
            meta = {}

    exp_data = meta.get("experiment", meta.get("exp_config", {}))
    train_data = meta.get("training", meta.get("train_config", {}))

    # Parse physical parameters
    phys_data = exp_data.get("physics", {})
    delta_v = float(phys_data.get("delta_v", 1.2))
    gamma1 = float(phys_data.get("gamma1", 0.65))
    gamma2 = float(phys_data.get("gamma2", 0.65))
    dt = float(phys_data.get("dt", 0.2))
    n_steps = int(exp_data.get("n", 10))

    # Parse architecture parameters
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    n_ctx = int(state_dict["pos_emb.weight"].shape[0]) if "pos_emb.weight" in state_dict else 2000
    vocab_size = int(state_dict["tok_emb.weight"].shape[0]) if "tok_emb.weight" in state_dict else 2500
    d_model = int(state_dict["pos_emb.weight"].shape[1]) if "pos_emb.weight" in state_dict else 128
    d_mlp = int(train_data.get("d_mlp", 4 * d_model))
    n_layers = int(train_data.get("n_layers", 4))
    n_heads = int(train_data.get("n_heads", 2))
    k_lookahead = int(train_data.get("k", meta.get("k", 1)))
    seed = int(exp_data.get("seed", 20260925))

    # Determine sequence length and cycles m (default m = 20 cycles)
    if override_seq_len is not None:
        target_seq_len = min(override_seq_len, n_ctx)
        m = max(1, target_seq_len // n_steps)
    else:
        m = min(m_cycles, n_ctx // n_steps)
    actual_seq_len = m * n_steps

    proc = make_process(
        "double_pendulum_mess4",
        m=m,
        n_steps=n_steps,
        delta_v=delta_v,
        dt=dt,
        system={"gamma1": gamma1, "gamma2": gamma2},
    )

    model_cfg = ModelConfig(
        vocab_size=vocab_size,
        n_ctx=n_ctx,
        n_layers=n_layers,
        n_heads=n_heads,
        d_model=d_model,
        d_mlp=d_mlp,
        seed=seed,
    )

    if k_lookahead > 1:
        trained_model = LookaheadTransformer(model_cfg, k=k_lookahead)
        random_model = LookaheadTransformer(model_cfg, k=k_lookahead)
    else:
        trained_model = TinyTransformer(model_cfg)
        random_model = TinyTransformer(model_cfg)

    trained_model.load_state_dict(state_dict)
    trained_model.to(device).eval()
    random_model.to(device).eval()

    info = {
        "job_id": job_id,
        "delta_v": delta_v,
        "gamma": gamma1,
        "n_steps": n_steps,
        "m": m,
        "seq_len": actual_seq_len,
        "vocab_size": vocab_size,
        "n_ctx": n_ctx,
        "d_model": d_model,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "k": k_lookahead,
    }
    return trained_model, random_model, proc, info


# ==============================================================================
# 2. Residual Stream Extraction & Sequence Splitting
# ==============================================================================

def extract_residual_streams_batched(
    model: TinyTransformer,
    tokens: np.ndarray,
    device: str = "cpu",
    chunk_size: int = 32,
) -> List[np.ndarray]:
    """Extracts residual streams for all depths [0..n_layers] in chunks.
    
    Index 0: embedding output
    Index 1..L: block 0..L-1 post-residual output
    """
    model.eval()
    n_trajs, seq_len = tokens.shape
    depth_lists: Optional[List[List[np.ndarray]]] = None

    with torch.no_grad():
        for start in range(0, n_trajs, chunk_size):
            end = min(start + chunk_size, n_trajs)
            tok_chunk = torch.as_tensor(tokens[start:end], dtype=torch.long, device=device)
            streams = model.residual_streams(tok_chunk)
            if depth_lists is None:
                depth_lists = [[] for _ in range(len(streams))]
            for d_idx, s in enumerate(streams):
                depth_lists[d_idx].append(s)

    assert depth_lists is not None
    return [np.concatenate(chunks, axis=0) for chunks in depth_lists]


def sequence_train_test_split(
    n_total: int,
    train_frac: float = 0.7,
    seed: int = 20260829,
) -> Tuple[np.ndarray, np.ndarray]:
    """Partitions whole sequences to guarantee zero cross-sequence history leakage."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_total)
    n_train = max(1, int(round(n_total * train_frac)))
    return np.sort(perm[:n_train]), np.sort(perm[n_train:])


# ==============================================================================
# 3. Softmax Linear Probe (Multinomial Logistic Regression with Soft Belief Targets)
# ==============================================================================

class SoftmaxLinearProbe(nn.Module):
    """Affine probe z = x W + c with softmax outputs for belief state prediction."""
    def __init__(self, d_in: int, n_classes: int = 4):
        super().__init__()
        self.linear = nn.Linear(d_in, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    def predict_probs(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.linear(x), dim=-1)


def fit_softmax_linear_probe(
    X_train: torch.Tensor,
    B_train: torch.Tensor,
    weight_decay: float = 1e-4,
    max_iter: int = 100,
    device: str = "cpu",
) -> SoftmaxLinearProbe:
    """Fits an affine softmax probe minimizing soft cross-entropy (KL divergence) via L-BFGS."""
    d_in = X_train.shape[-1]
    n_classes = B_train.shape[-1]
    probe = SoftmaxLinearProbe(d_in, n_classes).to(device)

    X_t = X_train.to(device)
    B_t = B_train.to(device)

    optimizer = torch.optim.LBFGS(
        probe.parameters(),
        lr=1.0,
        max_iter=max_iter,
        tolerance_grad=1e-7,
        tolerance_change=1e-9,
        line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer.zero_grad()
        logits = probe(X_t)
        loss = F.cross_entropy(logits, B_t)
        if weight_decay > 0:
            loss = loss + 0.5 * weight_decay * torch.norm(probe.linear.weight) ** 2
        loss.backward()
        return loss

    optimizer.step(closure)
    return probe


# ==============================================================================
# 4. Information-Theoretic Divergence Evaluators
# ==============================================================================

def compute_belief_divergences(
    P: np.ndarray,
    Q: np.ndarray,
    eps: float = 1e-15,
) -> Dict[str, np.ndarray]:
    """Computes exact information divergences between true belief P and prediction Q.
    
    Args:
        P: True belief distribution, shape (..., K)
        Q: Predicted distribution, shape (..., K)
    Returns:
        dict containing 'kl', 'cross_entropy', 'entropy', 'jsd', 'argmax_match'
    """
    P_c = np.clip(P, eps, 1.0)
    P_c = P_c / P_c.sum(axis=-1, keepdims=True)
    Q_c = np.clip(Q, eps, 1.0)
    Q_c = Q_c / Q_c.sum(axis=-1, keepdims=True)

    kl = np.sum(P_c * np.log(P_c / Q_c), axis=-1)
    kl = np.maximum(0.0, kl)

    entropy = -np.sum(P_c * np.log(P_c), axis=-1)
    ce = entropy + kl

    # Exact JSD
    M = 0.5 * (P_c + Q_c)
    kl_pm = np.sum(P_c * np.log(P_c / M), axis=-1)
    kl_qm = np.sum(Q_c * np.log(Q_c / M), axis=-1)
    jsd = np.clip(0.5 * (kl_pm + kl_qm), 0.0, math.log(2.0))

    argmax_match = (np.argmax(P_c, axis=-1) == np.argmax(Q_c, axis=-1)).astype(np.float64)

    return {
        "kl": kl,
        "cross_entropy": ce,
        "entropy": entropy,
        "jsd": jsd,
        "argmax_match": argmax_match,
    }


# ==============================================================================
# 5. Core Option B Belief Probing Pipeline (with 0-HMM & 1-HMM Baselines)
# ==============================================================================

def evaluate_probed_belief_kl(
    trained_model: TinyTransformer,
    random_model: TinyTransformer,
    proc: Any,
    n_trajs: int = 64,
    device: str = "cpu",
    seed: int = 20260829,
    smoke: bool = False,
    train_frac: float = 0.7,
) -> Dict[str, Any]:
    """Fits linear softmax probes on residual streams and evaluates KL divergence vs Optimal Observer."""
    actual_trajs = 16 if smoke else n_trajs
    rng = np.random.default_rng(seed)

    # 1. Sample complete physical rollouts with exact beliefs and moods
    batch = proc.sample_batch(rng, actual_trajs)
    tokens = batch["tokens"]              # (N, seq_len)
    letters = batch["letters"]            # (N, m)
    moods = batch["moods"]                # (N, seq_len)
    all_beliefs = batch["beliefs"]        # (N, seq_len, K)

    N, seq_len = tokens.shape
    m_cycles = proc.m
    n_steps = proc.n_steps
    K = proc.chain.n_states
    joint = proc.chain.joint
    pi = np.full(K, 1.0 / K)

    # 2. Extract boundary positions (usable tokens: final step of each cycle right before the kick)
    # At this step, the pendulum has completed flowing through cycle c, and the model must predict next kick
    boundary_indices = np.array([c * n_steps + n_steps - 1 for c in range(m_cycles)])

    # Target beliefs at boundary positions: b_c = P(mood_{c+1} | letters_{0..c})
    beliefs_opt = all_beliefs[:, boundary_indices, :]  # Shape: (N, m_cycles, K)
    target_moods = moods[:, boundary_indices]          # Shape: (N, m_cycles)

    # 3. Compute HMM Baselines across all sequences
    # 0-HMM: Memoryless stationary prior for all cycles
    beliefs_0hmm = np.tile(pi, (N, m_cycles, 1))  # (N, m_cycles, K)

    # 1-HMM: 1-letter memory observer conditioning only on letter L_c
    beliefs_1hmm = np.empty((N, m_cycles, K), dtype=np.float64)
    for i in range(N):
        for c in range(m_cycles):
            l = letters[i, c]
            b1 = pi @ joint[l]
            beliefs_1hmm[i, c] = b1 / b1.sum()

    # 4. Sequence-level train/test split
    train_seqs, test_seqs = sequence_train_test_split(N, train_frac=train_frac, seed=seed)
    n_train_seqs = len(train_seqs)
    n_test_seqs = len(test_seqs)

    # 5. Extract residual streams for all depths
    trained_streams = extract_residual_streams_batched(trained_model, tokens, device=device)
    random_streams = extract_residual_streams_batched(random_model, tokens, device=device)
    n_depths = len(trained_streams)

    # Flatten train/test targets at boundary tokens
    B_train_np = beliefs_opt[train_seqs].reshape(-1, K)
    B_test_np = beliefs_opt[test_seqs].reshape(-1, K)
    B_train_t = torch.as_tensor(B_train_np, dtype=torch.float32)
    B_test_t = torch.as_tensor(B_test_np, dtype=torch.float32)

    # Test baseline predictions
    B_0hmm_test = beliefs_0hmm[test_seqs]  # (n_test, m_cycles, K)
    B_1hmm_test = beliefs_1hmm[test_seqs]  # (n_test, m_cycles, K)
    B_opt_test = beliefs_opt[test_seqs]    # (n_test, m_cycles, K)

    # Evaluate HMM Baselines on test sequences
    div_0hmm = compute_belief_divergences(B_opt_test, B_0hmm_test)
    div_1hmm = compute_belief_divergences(B_opt_test, B_1hmm_test)

    # 6. Fit probes across all layers and score on held-out test sequences
    layer_results: List[Dict[str, Any]] = []
    best_layer_idx = 0
    best_test_kl = float("inf")
    best_trained_preds_test: Optional[np.ndarray] = None
    best_random_preds_test: Optional[np.ndarray] = None

    for d_idx in range(n_depths):
        layer_name = "Embedding" if d_idx == 0 else f"Layer {d_idx - 1}"

        # Extract features at boundary tokens
        s_tr_d = trained_streams[d_idx][:, boundary_indices, :]  # (N, m_cycles, d_model)
        s_rnd_d = random_streams[d_idx][:, boundary_indices, :]  # (N, m_cycles, d_model)

        X_tr_train = torch.as_tensor(s_tr_d[train_seqs].reshape(-1, s_tr_d.shape[-1]), dtype=torch.float32)
        X_tr_test = torch.as_tensor(s_tr_d[test_seqs].reshape(-1, s_tr_d.shape[-1]), dtype=torch.float32)

        X_rnd_train = torch.as_tensor(s_rnd_d[train_seqs].reshape(-1, s_rnd_d.shape[-1]), dtype=torch.float32)
        X_rnd_test = torch.as_tensor(s_rnd_d[test_seqs].reshape(-1, s_rnd_d.shape[-1]), dtype=torch.float32)

        # Fit softmax probe on trained model
        probe_tr = fit_softmax_linear_probe(X_tr_train, B_train_t, device=device)
        # Fit softmax probe on random control model
        probe_rnd = fit_softmax_linear_probe(X_rnd_train, B_train_t, device=device)

        # Predict on train and test
        with torch.no_grad():
            tr_train_pred = probe_tr.predict_probs(X_tr_train.to(device)).cpu().numpy()
            tr_test_pred = probe_tr.predict_probs(X_tr_test.to(device)).cpu().numpy()

            rnd_train_pred = probe_rnd.predict_probs(X_rnd_train.to(device)).cpu().numpy()
            rnd_test_pred = probe_rnd.predict_probs(X_rnd_test.to(device)).cpu().numpy()

        div_tr_train = compute_belief_divergences(B_train_np, tr_train_pred)
        div_tr_test = compute_belief_divergences(B_test_np, tr_test_pred)

        div_rnd_train = compute_belief_divergences(B_train_np, rnd_train_pred)
        div_rnd_test = compute_belief_divergences(B_test_np, rnd_test_pred)

        test_kl_val = float(np.mean(div_tr_test["kl"]))
        if test_kl_val < best_test_kl:
            best_test_kl = test_kl_val
            best_layer_idx = d_idx
            best_trained_preds_test = tr_test_pred.reshape(n_test_seqs, m_cycles, K)
            best_random_preds_test = rnd_test_pred.reshape(n_test_seqs, m_cycles, K)

        layer_results.append({
            "depth_idx": d_idx,
            "layer_name": layer_name,
            "trained_train_kl": float(np.mean(div_tr_train["kl"])),
            "trained_test_kl": test_kl_val,
            "trained_test_ce": float(np.mean(div_tr_test["cross_entropy"])),
            "trained_test_jsd": float(np.mean(div_tr_test["jsd"])),
            "trained_test_acc": float(np.mean(div_tr_test["argmax_match"])),
            "random_train_kl": float(np.mean(div_rnd_train["kl"])),
            "random_test_kl": float(np.mean(div_rnd_test["kl"])),
            "random_test_ce": float(np.mean(div_rnd_test["cross_entropy"])),
            "random_test_acc": float(np.mean(div_rnd_test["argmax_match"])),
            "net_emergence_gain": float(np.mean(div_rnd_test["kl"])) - test_kl_val,
        })

    assert best_trained_preds_test is not None
    assert best_random_preds_test is not None

    # 7. Cycle-by-cycle metrics for the best layer
    div_tr_by_cycle = compute_belief_divergences(B_opt_test, best_trained_preds_test)
    div_rnd_by_cycle = compute_belief_divergences(B_opt_test, best_random_preds_test)

    cycle_records: List[Dict[str, Any]] = []
    for c in range(m_cycles):
        cycle_records.append({
            "cycle": c,
            "trained_kl": float(np.mean(div_tr_by_cycle["kl"][:, c])),
            "1hmm_kl": float(np.mean(div_1hmm["kl"][:, c])),
            "0hmm_kl": float(np.mean(div_0hmm["kl"][:, c])),
            "random_kl": float(np.mean(div_rnd_by_cycle["kl"][:, c])),
            "oracle_entropy": float(np.mean(div_0hmm["entropy"][:, c])),
            "trained_accuracy": float(np.mean(div_tr_by_cycle["argmax_match"][:, c])),
            "random_accuracy": float(np.mean(div_rnd_by_cycle["argmax_match"][:, c])),
            "1hmm_accuracy": float(np.mean(div_1hmm["argmax_match"][:, c])),
            "0hmm_accuracy": float(np.mean(div_0hmm["argmax_match"][:, c])),
        })

    # Summary aggregations
    mean_opt_entropy = float(np.mean(div_0hmm["entropy"]))
    mean_0hmm_kl = float(np.mean(div_0hmm["kl"]))
    mean_1hmm_kl = float(np.mean(div_1hmm["kl"]))
    best_layer_info = layer_results[best_layer_idx]

    return {
        "n_total_trajs": actual_trajs,
        "n_train_seqs": n_train_seqs,
        "n_test_seqs": n_test_seqs,
        "m_cycles": m_cycles,
        "n_steps": n_steps,
        "best_layer_idx": best_layer_idx,
        "best_layer_name": best_layer_info["layer_name"],
        "mean_optimal_entropy": mean_opt_entropy,
        "uniform_floor_kl": math.log(K) - mean_opt_entropy,
        # Primary Benchmark Scores (on held-out test split)
        "best_trained_kl": best_layer_info["trained_test_kl"],
        "best_trained_ce": best_layer_info["trained_test_ce"],
        "best_trained_jsd": best_layer_info["trained_test_jsd"],
        "best_trained_acc": best_layer_info["trained_test_acc"],
        "best_random_kl": best_layer_info["random_test_kl"],
        "best_random_ce": best_layer_info["random_test_ce"],
        "best_random_acc": best_layer_info["random_test_acc"],
        "net_emergence_gain": best_layer_info["net_emergence_gain"],
        "1hmm_baseline_kl": mean_1hmm_kl,
        "1hmm_baseline_ce": mean_opt_entropy + mean_1hmm_kl,
        "1hmm_baseline_acc": float(np.mean(div_1hmm["argmax_match"])),
        "0hmm_baseline_kl": mean_0hmm_kl,
        "0hmm_baseline_ce": mean_opt_entropy + mean_0hmm_kl,
        "0hmm_baseline_acc": float(np.mean(div_0hmm["argmax_match"])),
        # Structured tables
        "layer_results": layer_results,
        "cycle_results": cycle_records,
        # Visualization raw points
        "_vis_data": {
            "B_opt_test": B_opt_test,
            "B_pred_test": best_trained_preds_test,
            "B_1hmm_test": B_1hmm_test,
        },
    }


# ==============================================================================
# 6. Publication-Grade Diagnostic Visualizations (4 Panels)
# ==============================================================================

def save_belief_kl_diagnostic_plots(
    results: Dict[str, Any],
    job_id: str,
    out_path: Path,
) -> None:
    """Renders a 4-panel publication-grade diagnostic figure for belief state probing."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(22, 5.2), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")

    cycles = [r["cycle"] for r in results["cycle_results"]]
    tr_kl_c = [r["trained_kl"] for r in results["cycle_results"]]
    one_kl_c = [r["1hmm_kl"] for r in results["cycle_results"]]
    zero_kl_c = [r["0hmm_kl"] for r in results["cycle_results"]]
    rnd_kl_c = [r["random_kl"] for r in results["cycle_results"]]

    # --------------------------------------------------------------------------
    # Panel 1: Cycle-by-Cycle Convergence vs HMM Baselines
    # --------------------------------------------------------------------------
    ax0 = axes[0]
    ax0.set_facecolor("white")
    ax0.plot(cycles, tr_kl_c, marker="o", markersize=4.5, color="#1d3557", lw=1.8, label=f"Trained ({results['best_layer_name']})")
    ax0.plot(cycles, one_kl_c, marker="^", markersize=4, color="#2a9d8f", lw=1.3, ls="--", label="1-HMM Baseline")
    ax0.plot(cycles, zero_kl_c, marker="s", markersize=4, color="#e76f51", lw=1.3, ls="-.", label="0-HMM Baseline")
    ax0.plot(cycles, rnd_kl_c, color="#9a8c98", lw=1.1, ls=":", label="Random Control")
    ax0.axhline(0.0, color="#2b78d6", lw=1.0, ls="-", alpha=0.8, label="Optimal Floor (0.00)")

    ax0.set(
        title="1. Belief State KL vs Cycle",
        xlabel="Cycle Index (m)",
        ylabel="D_KL(b_c || \\hat{b}_c) [nats]",
    )
    ax0.grid(True, alpha=0.25)
    ax0.legend(frameon=True, facecolor="white", edgecolor="#ddd", fontsize=8)

    # --------------------------------------------------------------------------
    # Panel 2: Layer-by-Layer Probing Profile
    # --------------------------------------------------------------------------
    ax1 = axes[1]
    ax1.set_facecolor("white")
    layer_names = [lr["layer_name"] for lr in results["layer_results"]]
    tr_layers_kl = [lr["trained_test_kl"] for lr in results["layer_results"]]
    rnd_layers_kl = [lr["random_test_kl"] for lr in results["layer_results"]]
    x_layers = np.arange(len(layer_names))

    ax1.plot(x_layers, tr_layers_kl, marker="o", markersize=5, color="#1d3557", lw=1.8, label="Trained Test KL")
    ax1.plot(x_layers, rnd_layers_kl, marker="x", markersize=5, color="#9a8c98", lw=1.2, ls="--", label="Random Control")
    ax1.axhline(results["1hmm_baseline_kl"], color="#2a9d8f", lw=1.2, ls=":", label="1-HMM Baseline")
    ax1.axhline(results["0hmm_baseline_kl"], color="#e76f51", lw=1.2, ls="-.", label="0-HMM Baseline")

    # Highlight best layer
    best_idx = results["best_layer_idx"]
    ax1.scatter([best_idx], [tr_layers_kl[best_idx]], color="#e63946", s=60, zorder=6, label=f"Best: {results['best_layer_name']}")

    ax1.set(
        title="2. Layer-by-Layer Belief Representation",
        xlabel="Network Depth",
        ylabel="D_KL(b || \\hat{b}) [nats]",
    )
    ax1.set_xticks(x_layers)
    ax1.set_xticklabels(layer_names, rotation=25, fontsize=8)
    ax1.grid(True, alpha=0.25)
    ax1.legend(frameon=True, facecolor="white", edgecolor="#ddd", fontsize=8)

    # --------------------------------------------------------------------------
    # Panel 3: Suboptimality Benchmark Ladder
    # --------------------------------------------------------------------------
    ax2 = axes[2]
    ax2.set_facecolor("white")
    ladder_labels = ["Opt Floor\n(0-loss)", "1-HMM\n(1-step)", "0-HMM\n(0-step)", "Trained\n(Best Layer)", "Random\nControl"]
    ladder_vals = [
        0.0,
        results["1hmm_baseline_kl"],
        results["0hmm_baseline_kl"],
        results["best_trained_kl"],
        results["best_random_kl"],
    ]
    colors = ["#2b78d6", "#2a9d8f", "#e76f51", "#1d3557", "#adb5bd"]
    x_pos = np.arange(len(ladder_labels))

    ax2.bar(x_pos, ladder_vals, width=0.55, color=colors)
    ax2.set(
        title="3. Belief Suboptimality Ladder",
        ylabel="D_KL(b_opt || b_pred) [nats]",
    )
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(ladder_labels, fontsize=8)
    ax2.grid(True, alpha=0.25)

    max_v = max(ladder_vals) if max(ladder_vals) > 0 else 1.0
    for i_b, val in enumerate(ladder_vals):
        ax2.text(i_b, val + 0.02 * max_v, f"{val:.4f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    # --------------------------------------------------------------------------
    # Panel 4: 2D Belief Simplex Square Projection
    # --------------------------------------------------------------------------
    ax3 = axes[3]
    ax3.set_facecolor("white")
    sq_v = square_projection_vertices()  # (4, 2) corners in [-1, 1]^2

    # Draw simplex boundary polygon
    square_poly = np.array([sq_v[0], sq_v[1], sq_v[2], sq_v[3], sq_v[0]])
    ax3.plot(square_poly[:, 0], square_poly[:, 1], color="#bbb", lw=1.2, ls="--", zorder=1)

    vis = results.get("_vis_data", {})
    if "B_opt_test" in vis and "B_pred_test" in vis:
        # Flatten across sequences and cycles
        pts_opt = (vis["B_opt_test"].reshape(-1, 4)) @ sq_v
        pts_pred = (vis["B_pred_test"].reshape(-1, 4)) @ sq_v

        # Subsample for clear visual density if large
        n_pts = min(400, pts_opt.shape[0])
        sub_idx = np.random.default_rng(0).choice(pts_opt.shape[0], size=n_pts, replace=False)

        ax3.scatter(pts_opt[sub_idx, 0], pts_opt[sub_idx, 1], s=12, color="#0077b6", alpha=0.55, label="Optimal b_t (Fractal)", zorder=3)
        ax3.scatter(pts_pred[sub_idx, 0], pts_pred[sub_idx, 1], s=12, color="#e63946", alpha=0.55, label="Probed \\hat{b}_t", zorder=4)

    # Label mood vertices
    mood_names = ["Mood 0", "Mood 1", "Mood 2", "Mood 3"]
    offsets = [(-0.16, -0.12), (-0.16, 0.06), (0.04, 0.06), (0.04, -0.12)]
    for m_idx in range(4):
        ax3.scatter(sq_v[m_idx, 0], sq_v[m_idx, 1], color="#333", s=35, zorder=5)
        dx, dy = offsets[m_idx]
        ax3.text(sq_v[m_idx, 0] + dx, sq_v[m_idx, 1] + dy, mood_names[m_idx], fontsize=8, fontweight="bold")

    ax3.set(
        title="4. Belief Simplex Projection (Mess-4)",
        xlabel="Simplex X",
        ylabel="Simplex Y",
    )
    ax3.set_xlim(-1.3, 1.3)
    ax3.set_ylim(-1.3, 1.3)
    ax3.grid(True, alpha=0.25)
    ax3.legend(frameon=True, facecolor="white", edgecolor="#ddd", fontsize=8, loc="upper right")

    fig.suptitle(f"Optimal Observer Belief State Probing (Option B): {job_id} (m={len(cycles)} cycles)", fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved publication-grade diagnostic plots -> {out_path}")


# ==============================================================================
# 7. Single Checkpoint Complete Evaluation Pipeline
# ==============================================================================

def evaluate_checkpoint_kl(
    ckpt_path: Path,
    out_dir: Path,
    n_trajs: int = 64,
    device: str = "cpu",
    seed: int = 20260829,
    smoke: bool = False,
    m_cycles: int = 20,
    override_seq_len: Optional[int] = None,
    save_plot: bool = True,
) -> Dict[str, Any]:
    """Runs complete Option B Belief State probing evaluation against Optimal Observer, 1-HMM, and 0-HMM."""
    t0 = time.perf_counter()

    trained_m, random_m, proc, info = load_model_and_process(
        ckpt_path=ckpt_path,
        device=device,
        m_cycles=m_cycles,
        override_seq_len=override_seq_len,
    )

    print(f"\n[{info['job_id']}] Evaluating Probed Belief State KL against Optimal Observer & HMM Baselines...")
    print(f"  Configuration: dv={info['delta_v']}, gamma={info['gamma']}, n={info['n_steps']}, m={info['m']} cycles")
    print(f"  Sequence Len:  {info['seq_len']} tokens ({info['m']} HMM cycles)")
    print(f"  Trajectories:  {16 if smoke else n_trajs} sequences (70% train probe, 30% held-out test)")

    # Execute Option B Evaluation
    belief_results = evaluate_probed_belief_kl(
        trained_model=trained_m,
        random_model=random_m,
        proc=proc,
        n_trajs=n_trajs,
        device=device,
        seed=seed,
        smoke=smoke,
    )

    elapsed = time.perf_counter() - t0

    result_payload = {
        "job_id": info["job_id"],
        "checkpoint": str(ckpt_path),
        "evaluated_at": time.time(),
        "elapsed_seconds": elapsed,
        "m_cycles": info["m"],
        "seq_len": info["seq_len"],
        "info": info,
        "belief_probing_metrics": {
            k: v for k, v in belief_results.items() if not k.startswith("_") and k not in ("layer_results", "cycle_results")
        },
        "layer_results": belief_results["layer_results"],
        "cycle_results": belief_results["cycle_results"],
    }

    # Console Report
    print("=" * 84)
    print(f" OPTIMAL OBSERVER BELIEF STATE PROBING (OPTION B): {info['job_id']}")
    print("=" * 84)
    print(f" Cycles: m = {info['m']} cycles | Sequence Length = {info['seq_len']} tokens | Split: {belief_results['n_train_seqs']} train / {belief_results['n_test_seqs']} test")
    print(f" Selected Representation: {belief_results['best_layer_name']} (Minimal Held-Out Test KL)")
    print("-" * 84)
    print(" SUBOPTIMALITY BENCHMARK LADDER: D_KL(b_opt || b_pred) on Held-Out Test Sequences:")
    print("-" * 84)
    print(f"  * Optimal Observer (Full-HMM):    D_KL =  0.0000 nats  ( 0.0000 bits) | H(b) = {belief_results['mean_optimal_entropy']:.4f} nats")
    print(f"  * 1-HMM Baseline (1-step memory): D_KL = {belief_results['1hmm_baseline_kl']:7.4f} nats  ({belief_results['1hmm_baseline_kl']/math.log(2):7.4f} bits) | Acc = {belief_results['1hmm_baseline_acc']:.2%}")
    print(f"  * 0-HMM Baseline (0-step memory): D_KL = {belief_results['0hmm_baseline_kl']:7.4f} nats  ({belief_results['0hmm_baseline_kl']/math.log(2):7.4f} bits) | Acc = {belief_results['0hmm_baseline_acc']:.2%}")
    print(f"  * Trained Transformer Probed:     D_KL = {belief_results['best_trained_kl']:7.4f} nats  ({belief_results['best_trained_kl']/math.log(2):7.4f} bits) | Acc = {belief_results['best_trained_acc']:.2%}")
    print(f"  * Untrained Random Control:       D_KL = {belief_results['best_random_kl']:7.4f} nats  ({belief_results['best_random_kl']/math.log(2):7.4f} bits) | Acc = {belief_results['best_random_acc']:.2%}")
    print(f"  * Uniform Prior (ln 4 - H_opt):   D_KL = {belief_results['uniform_floor_kl']:7.4f} nats  ({belief_results['uniform_floor_kl']/math.log(2):7.4f} bits)")
    print("-" * 84)
    print(f"  * Net Emergence Gain (Random KL - Trained KL): {belief_results['net_emergence_gain']:+.4f} nats")
    print("=" * 84)

    # Save summary artifacts
    model_out_dir = out_dir / info["job_id"]
    model_out_dir.mkdir(parents=True, exist_ok=True)

    summary_json_path = model_out_dir / f"{info['job_id']}_optimal_observer_kl.json"
    with open(summary_json_path, "w") as f:
        json.dump(result_payload, f, indent=2)
    print(f"Saved JSON metrics -> {summary_json_path}")

    # Save CSV of per-cycle dynamics
    cycle_df = pd.DataFrame(belief_results["cycle_results"])
    cycle_csv_path = model_out_dir / f"{info['job_id']}_belief_kl_by_cycle.csv"
    cycle_df.to_csv(cycle_csv_path, index=False)
    print(f"Saved per-cycle CSV -> {cycle_csv_path}")

    # Save CSV of layer-by-layer profile
    layer_df = pd.DataFrame(belief_results["layer_results"])
    layer_csv_path = model_out_dir / f"{info['job_id']}_belief_kl_by_layer.csv"
    layer_df.to_csv(layer_csv_path, index=False)
    print(f"Saved layer profile CSV -> {layer_csv_path}")

    if save_plot:
        plot_path = model_out_dir / f"{info['job_id']}_kl_analysis.png"
        save_belief_kl_diagnostic_plots(belief_results, info["job_id"], plot_path)

    return result_payload


# ==============================================================================
# 8. Main Orchestrator Flow
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Measure KL Divergence between Probed Transformer Belief States and Optimal Observer / HMM Baselines (Option B)"
    )
    parser.add_argument("--checkpoint", "-c", type=str, default=None,
                        help="Path to specific *_final.pt model checkpoint to evaluate")
    parser.add_argument("--dir", "-d", "--results-dir", dest="results_dir", type=str, default=None,
                        help="Directory to scan for *_final.pt checkpoints (default: ./experiments/results)")
    parser.add_argument("--models", "-m-filter", type=str, default=None,
                        help="Optional substring filter for model IDs when scanning directory")
    parser.add_argument("--m", type=int, default=20,
                        help="Number of HMM cycles m to evaluate (default: 20 cycles)")
    parser.add_argument("--n-trajs", type=int, default=64,
                        help="Number of evaluation trajectories to simulate (default: 64)")
    parser.add_argument("--smoke", action="store_true",
                        help="Rapid smoke testing mode (16 trajectories)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to run inference on (default: cuda:0 if available, else cpu)")
    parser.add_argument("--seq-len", type=int, default=None,
                        help="Override evaluation sequence length (default: m * n)")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Output directory to save KL evaluations and plots (default: ./experiments/kl_results)")
    parser.add_argument("--no-plot", action="store_true",
                        help="Disable PNG diagnostic plot rendering")
    args = parser.parse_args()

    # Determine compute device
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"

    # Determine output directory
    out_dir = Path(args.out_dir) if args.out_dir else REPO_ROOT / "experiments" / "kl_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Locate checkpoints to evaluate
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

    print("=" * 84)
    print(" OPTIMAL OBSERVER & HMM BASELINES BELIEF STATE KL DIVERGENCE (OPTION B)")
    print("=" * 84)
    print(f"Device:           {device}")
    print(f"Cycles (m):       {args.m} cycles")
    print(f"Trajectories:     {16 if args.smoke else args.n_trajs} sequences")
    print(f"Checkpoints:      {len(ckpts)} checkpoint(s) queued for evaluation")
    print(f"Output Directory: {out_dir}")
    print("=" * 84)

    all_results: List[Dict[str, Any]] = []
    for idx, ckpt in enumerate(ckpts, 1):
        print(f"\nProcessing [{idx}/{len(ckpts)}]: {ckpt.name}")
        try:
            res = evaluate_checkpoint_kl(
                ckpt_path=ckpt,
                out_dir=out_dir,
                n_trajs=args.n_trajs,
                device=device,
                smoke=args.smoke,
                m_cycles=args.m,
                override_seq_len=args.seq_len,
                save_plot=not args.no_plot,
            )
            all_results.append(res)
        except Exception as e:
            print(f"Error evaluating {ckpt}: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()

    # Master summary table across all evaluated checkpoints
    if all_results:
        summary_rows = []
        for r in all_results:
            bm = r["belief_probing_metrics"]
            info = r["info"]
            summary_rows.append({
                "job_id": r["job_id"],
                "m": info["m"],
                "best_layer": bm["best_layer_name"],
                "trained_belief_kl": bm["best_trained_kl"],
                "1hmm_belief_kl": bm["1hmm_baseline_kl"],
                "0hmm_belief_kl": bm["0hmm_baseline_kl"],
                "random_belief_kl": bm["best_random_kl"],
                "net_gain": bm["net_emergence_gain"],
                "trained_acc": bm["best_trained_acc"],
                "1hmm_acc": bm["1hmm_baseline_acc"],
                "0hmm_acc": bm["0hmm_baseline_acc"],
            })
        summary_df = pd.DataFrame(summary_rows)
        master_csv_path = out_dir / "optimal_observer_kl_summary.csv"
        summary_df.to_csv(master_csv_path, index=False)

        print("\n" + "=" * 84)
        print(" MASTER EVALUATION SUMMARY TABLE")
        print("=" * 84)
        cols_to_show = ["job_id", "m", "best_layer", "trained_belief_kl", "1hmm_belief_kl", "0hmm_belief_kl", "random_belief_kl", "net_gain"]
        print(summary_df[cols_to_show].to_string(index=False))
        print("=" * 84)
        print(f"Master summary CSV saved -> {master_csv_path}")


if __name__ == "__main__":
    main()
