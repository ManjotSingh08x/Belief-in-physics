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

Multi-Regime Context Lengths:
- Evaluates across 3 context lengths:
    * Short context: 500 tokens (50 cycles for n=10)
    * Mid context:   1500 tokens (150 cycles for n=10)
    * Long context:  2500 tokens (250 cycles for n=10)
- Sequence-level train/test split (70% train, 30% held-out test) to prevent sequence leakage.
- Tracks model cross-entropy loss vs sequence position and training loss vs epochs/steps.
- Generates 4-panel publication-grade diagnostic plots (multi-regime convergence, loss curves,
  context-length scaling ladder, and 2D belief simplex square projection).

Usage:
  uv run python scripts/measure_optimal_observer_kl.py --checkpoint path/to/model_final.pt
  uv run python scripts/measure_optimal_observer_kl.py --dir ./experiments/results
  uv run python scripts/measure_optimal_observer_kl.py --smoke
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


def load_model_and_metadata(
    ckpt_path: Path,
    device: str = "cpu",
) -> Tuple[TinyTransformer, TinyTransformer, Dict[str, Any]]:
    """Loads trained checkpoint, matched random control, and experiment configuration metadata."""
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

    # Check for training loss history CSV
    loss_history_df: Optional[pd.DataFrame] = None
    possible_loss_csvs = [
        model_dir / f"{job_id}_loss_history.csv",
        model_dir.parent / f"{job_id}_loss_history.csv",
    ]
    for p_csv in possible_loss_csvs:
        if p_csv.exists():
            try:
                loss_history_df = pd.read_csv(p_csv)
                break
            except Exception:
                pass

    info = {
        "job_id": job_id,
        "delta_v": delta_v,
        "gamma": gamma1,
        "dt": dt,
        "n_steps": n_steps,
        "vocab_size": vocab_size,
        "n_ctx": n_ctx,
        "d_model": d_model,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "k": k_lookahead,
        "loss_history_df": loss_history_df,
    }
    return trained_model, random_model, info


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
    """Computes exact information divergences between true belief P and prediction Q."""
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
# 5. Core Single-Regime Probing Pipeline
# ==============================================================================

def evaluate_single_regime_belief_kl(
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

    # 1. Sample complete physical rollouts
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

    # 2. Extract boundary positions (final step of each cycle right before kick)
    boundary_indices = np.array([c * n_steps + n_steps - 1 for c in range(m_cycles)])
    beliefs_opt = all_beliefs[:, boundary_indices, :]  # Shape: (N, m_cycles, K)

    # 3. Compute HMM Baselines across all sequences
    beliefs_0hmm = np.tile(pi, (N, m_cycles, 1))  # (N, m_cycles, K)
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

    # 5. Extract residual streams
    trained_streams = extract_residual_streams_batched(trained_model, tokens, device=device)
    random_streams = extract_residual_streams_batched(random_model, tokens, device=device)
    n_depths = len(trained_streams)

    # Flatten train/test targets at boundary tokens
    B_train_np = beliefs_opt[train_seqs].reshape(-1, K)
    B_test_np = beliefs_opt[test_seqs].reshape(-1, K)
    B_train_t = torch.as_tensor(B_train_np, dtype=torch.float32)

    # Test baseline predictions
    B_0hmm_test = beliefs_0hmm[test_seqs]  # (n_test, m_cycles, K)
    B_1hmm_test = beliefs_1hmm[test_seqs]  # (n_test, m_cycles, K)
    B_opt_test = beliefs_opt[test_seqs]    # (n_test, m_cycles, K)

    # Evaluate HMM Baselines on test sequences
    div_0hmm = compute_belief_divergences(B_opt_test, B_0hmm_test)
    div_1hmm = compute_belief_divergences(B_opt_test, B_1hmm_test)

    # 6. Compute in-context language modeling cross-entropy loss by sequence position
    with torch.no_grad():
        test_tok_t = torch.as_tensor(tokens[test_seqs], dtype=torch.long, device=device)
        logits = trained_model(test_tok_t)
        loss_by_pos = F.cross_entropy(
            logits[:, :-1, :].reshape(-1, logits.shape[-1]),
            test_tok_t[:, 1:].reshape(-1),
            reduction="none",
        ).reshape(n_test_seqs, seq_len - 1).mean(dim=0).cpu().numpy()

    # 7. Fit probes across all layers and score on held-out test sequences
    layer_results: List[Dict[str, Any]] = []
    best_layer_idx = 0
    best_test_kl = float("inf")
    best_trained_preds_test: Optional[np.ndarray] = None
    best_random_preds_test: Optional[np.ndarray] = None

    for d_idx in range(n_depths):
        layer_name = "Embedding" if d_idx == 0 else f"Layer {d_idx - 1}"

        s_tr_d = trained_streams[d_idx][:, boundary_indices, :]
        s_rnd_d = random_streams[d_idx][:, boundary_indices, :]

        X_tr_train = torch.as_tensor(s_tr_d[train_seqs].reshape(-1, s_tr_d.shape[-1]), dtype=torch.float32)
        X_tr_test = torch.as_tensor(s_tr_d[test_seqs].reshape(-1, s_tr_d.shape[-1]), dtype=torch.float32)

        X_rnd_train = torch.as_tensor(s_rnd_d[train_seqs].reshape(-1, s_rnd_d.shape[-1]), dtype=torch.float32)
        X_rnd_test = torch.as_tensor(s_rnd_d[test_seqs].reshape(-1, s_rnd_d.shape[-1]), dtype=torch.float32)

        # Fit softmax probe on trained model
        probe_tr = fit_softmax_linear_probe(X_tr_train, B_train_t, device=device)
        # Fit softmax probe on random control model
        probe_rnd = fit_softmax_linear_probe(X_rnd_train, B_train_t, device=device)

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

    # 8. Cycle-by-cycle metrics for the best layer
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

    mean_opt_entropy = float(np.mean(div_0hmm["entropy"]))
    mean_0hmm_kl = float(np.mean(div_0hmm["kl"]))
    mean_1hmm_kl = float(np.mean(div_1hmm["kl"]))
    best_layer_info = layer_results[best_layer_idx]

    return {
        "seq_len": seq_len,
        "m_cycles": m_cycles,
        "n_steps": n_steps,
        "best_layer_idx": best_layer_idx,
        "best_layer_name": best_layer_info["layer_name"],
        "mean_optimal_entropy": mean_opt_entropy,
        "best_trained_kl": best_layer_info["trained_test_kl"],
        "best_trained_ce": best_layer_info["trained_test_ce"],
        "best_trained_acc": best_layer_info["trained_test_acc"],
        "best_random_kl": best_layer_info["random_test_kl"],
        "best_random_acc": best_layer_info["random_test_acc"],
        "net_emergence_gain": best_layer_info["net_emergence_gain"],
        "1hmm_baseline_kl": mean_1hmm_kl,
        "1hmm_baseline_acc": float(np.mean(div_1hmm["argmax_match"])),
        "0hmm_baseline_kl": mean_0hmm_kl,
        "0hmm_baseline_acc": float(np.mean(div_0hmm["argmax_match"])),
        "mean_eval_loss": float(np.mean(loss_by_pos)),
        "loss_by_pos": loss_by_pos.tolist(),
        "layer_results": layer_results,
        "cycle_results": cycle_records,
        "_vis_data": {
            "B_opt_test": B_opt_test,
            "B_pred_test": best_trained_preds_test,
        },
    }


# Backwards-compatibility alias
evaluate_probed_belief_kl = evaluate_single_regime_belief_kl


# ==============================================================================
# 6. Publication-Grade Multi-Regime Diagnostic Visualizations (4 Panels)
# ==============================================================================

def save_multiregime_kl_diagnostic_plots(
    regimes_data: Dict[str, Dict[str, Any]],
    training_loss_df: Optional[pd.DataFrame],
    job_id: str,
    out_path: Path,
) -> None:
    """Renders a 4-panel publication-grade diagnostic figure comparing Short, Mid, and Long regimes."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(22, 5.2), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")

    # Order of regimes
    regime_order = ["short", "mid", "long"]
    active_regimes = [r for r in regime_order if r in regimes_data]

    # --------------------------------------------------------------------------
    # Panel 1: Cycle-by-Cycle Belief KL Across Context Regimes
    # --------------------------------------------------------------------------
    ax0 = axes[0]
    ax0.set_facecolor("white")

    regime_styles = {
        "short": {"color": "#457b9d", "lw": 1.4, "ls": "-.", "label": "Short (500 tok)"},
        "mid":   {"color": "#1d3557", "lw": 1.6, "ls": "--", "label": "Mid (1500 tok)"},
        "long":  {"color": "#e63946", "lw": 1.8, "ls": "-",  "label": "Long (2500 tok)"},
    }

    for r_key in active_regimes:
        r_res = regimes_data[r_key]
        cycles = [row["cycle"] for row in r_res["cycle_results"]]
        tr_kl = [row["trained_kl"] for row in r_res["cycle_results"]]
        style = regime_styles.get(r_key, {"color": "black", "lw": 1.2, "ls": "-", "label": r_key})
        ax0.plot(cycles, tr_kl, color=style["color"], lw=style["lw"], ls=style["ls"], label=f"Trained: {style['label']}")

    # Baseline references from the longest available regime
    longest_key = active_regimes[-1]
    long_res = regimes_data[longest_key]
    long_cycles = [row["cycle"] for row in long_res["cycle_results"]]
    one_kl = [row["1hmm_kl"] for row in long_res["cycle_results"]]
    zero_kl = [row["0hmm_kl"] for row in long_res["cycle_results"]]

    ax0.plot(long_cycles, one_kl, color="#2a9d8f", lw=1.2, ls=":", label="1-HMM Baseline")
    ax0.plot(long_cycles, zero_kl, color="#e76f51", lw=1.2, ls=":", label="0-HMM Baseline")
    ax0.axhline(0.0, color="#2b78d6", lw=0.9, ls="-", alpha=0.7, label="Optimal Floor (0.00)")

    ax0.set(
        title="1. Belief State KL vs Cycle across Regimes",
        xlabel="Cycle Index (m)",
        ylabel="D_KL(b_c || \\hat{b}_c) [nats]",
    )
    ax0.grid(True, alpha=0.25)
    ax0.legend(frameon=True, facecolor="white", edgecolor="#ddd", fontsize=7.5)

    # --------------------------------------------------------------------------
    # Panel 2: Model Cross-Entropy Loss vs Token Position / Training Loss vs Steps
    # --------------------------------------------------------------------------
    ax1 = axes[1]
    ax1.set_facecolor("white")

    if training_loss_df is not None and not training_loss_df.empty:
        # Plot training loss history
        steps = training_loss_df["step"].values
        t_loss = training_loss_df["loss"].values
        ax1.plot(steps, t_loss, color="#1d3557", alpha=0.35, lw=1.0, label="Step Loss")
        if len(t_loss) >= 5:
            win = max(3, len(t_loss) // 15)
            s_loss = pd.Series(t_loss).rolling(win, min_periods=1).mean().values
            ax1.plot(steps, s_loss, color="#e63946", lw=2.0, label=f"Smoothed Loss (win={win})")
        ax1.set(
            title="2. Training Loss Curve (Phase 2)",
            xlabel="Optimization Step",
            ylabel="Cross-Entropy Loss [nats]",
        )
        final_l = float(t_loss[-1])
        ax1.axhline(final_l, color="#2a9d8f", lw=1.0, ls="--", alpha=0.7)
        ax1.text(steps[-1], final_l, f" Final: {final_l:.4f}", va="center", fontsize=8, fontweight="bold", color="#2a9d8f")
    else:
        # Plot evaluation in-context loss curve across sequence position
        loss_curve = np.array(long_res["loss_by_pos"])
        tok_steps = np.arange(len(loss_curve))
        ax1.plot(tok_steps, loss_curve, color="#1d3557", alpha=0.35, lw=0.8, label="Token Loss")
        if len(loss_curve) >= 10:
            win = max(5, len(loss_curve) // 30)
            smooth_curve = pd.Series(loss_curve).rolling(win, min_periods=1).mean().values
            ax1.plot(tok_steps, smooth_curve, color="#e63946", lw=1.8, label=f"Smoothed (win={win})")
        ax1.set(
            title="2. In-Context Eval Loss vs Position",
            xlabel="Context Position (t)",
            ylabel="Cross-Entropy Loss [nats]",
        )

    ax1.grid(True, alpha=0.25)
    ax1.legend(frameon=True, facecolor="white", edgecolor="#ddd", fontsize=8)

    # --------------------------------------------------------------------------
    # Panel 3: Context-Length Scaling Profile (Short vs Mid vs Long)
    # --------------------------------------------------------------------------
    ax2 = axes[2]
    ax2.set_facecolor("white")

    x_indices = np.arange(len(active_regimes))
    bar_width = 0.22

    trained_vals = [regimes_data[r]["best_trained_kl"] for r in active_regimes]
    one_vals = [regimes_data[r]["1hmm_baseline_kl"] for r in active_regimes]
    zero_vals = [regimes_data[r]["0hmm_baseline_kl"] for r in active_regimes]
    rnd_vals = [regimes_data[r]["best_random_kl"] for r in active_regimes]

    ax2.bar(x_indices - 1.5 * bar_width, trained_vals, width=bar_width, color="#1d3557", label="Trained Model")
    ax2.bar(x_indices - 0.5 * bar_width, one_vals, width=bar_width, color="#2a9d8f", label="1-HMM Baseline")
    ax2.bar(x_indices + 0.5 * bar_width, zero_vals, width=bar_width, color="#e76f51", label="0-HMM Baseline")
    ax2.bar(x_indices + 1.5 * bar_width, rnd_vals, width=bar_width, color="#adb5bd", label="Random Control")

    regime_labels = [f"{r.capitalize()}\n({regimes_data[r]['seq_len']} tok)" for r in active_regimes]
    ax2.set_xticks(x_indices)
    ax2.set_xticklabels(regime_labels, fontsize=8.5)
    ax2.set(
        title="3. Context-Length Scaling Ladder",
        ylabel="D_KL(b_opt || b_pred) [nats]",
    )
    ax2.grid(True, alpha=0.25)
    ax2.legend(frameon=True, facecolor="white", edgecolor="#ddd", fontsize=7.5)

    # --------------------------------------------------------------------------
    # Panel 4: 2D Belief Simplex Square Projection
    # --------------------------------------------------------------------------
    ax3 = axes[3]
    ax3.set_facecolor("white")
    sq_v = square_projection_vertices()

    square_poly = np.array([sq_v[0], sq_v[1], sq_v[2], sq_v[3], sq_v[0]])
    ax3.plot(square_poly[:, 0], square_poly[:, 1], color="#bbb", lw=1.2, ls="--", zorder=1)

    vis = long_res.get("_vis_data", {})
    if "B_opt_test" in vis and "B_pred_test" in vis:
        pts_opt = (vis["B_opt_test"].reshape(-1, 4)) @ sq_v
        pts_pred = (vis["B_pred_test"].reshape(-1, 4)) @ sq_v

        n_pts = min(400, pts_opt.shape[0])
        sub_idx = np.random.default_rng(0).choice(pts_opt.shape[0], size=n_pts, replace=False)

        ax3.scatter(pts_opt[sub_idx, 0], pts_opt[sub_idx, 1], s=12, color="#0077b6", alpha=0.55, label="Optimal b_t (Fractal)", zorder=3)
        ax3.scatter(pts_pred[sub_idx, 0], pts_pred[sub_idx, 1], s=12, color="#e63946", alpha=0.55, label="Probed \\hat{b}_t", zorder=4)

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

    fig.suptitle(f"Multi-Regime Belief State KL & Loss Analysis: {job_id}", fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved publication-grade diagnostic plots -> {out_path}")


# ==============================================================================
# 7. Checkpoint Multi-Regime Evaluation Pipeline
# ==============================================================================

def evaluate_checkpoint_multiregime_kl(
    ckpt_path: Path,
    out_dir: Path,
    target_seq_lens: List[int] = [500, 1500, 2500],
    n_trajs: int = 64,
    device: str = "cpu",
    seed: int = 20260829,
    smoke: bool = False,
    save_plot: bool = True,
) -> Dict[str, Any]:
    """Runs complete Option B Belief State probing evaluation across 3 context lengths (Short, Mid, Long)."""
    t0 = time.perf_counter()

    trained_m, random_m, info = load_model_and_metadata(ckpt_path=ckpt_path, device=device)
    n_ctx = info["n_ctx"]
    n_steps = info["n_steps"]
    job_id = info["job_id"]

    print(f"\n[{job_id}] Evaluating Probed Belief State KL across Context Lengths {target_seq_lens}...")
    print(f"  Architecture: n_ctx={n_ctx}, d_model={info['d_model']}, n_layers={info['n_layers']}, n_steps={n_steps}")

    regimes_data: Dict[str, Dict[str, Any]] = {}
    regime_names = ["short", "mid", "long"]

    for idx, target_len in enumerate(target_seq_lens):
        regime_label = regime_names[idx] if idx < len(regime_names) else f"regime_{target_len}"
        actual_len = min(target_len, n_ctx)
        m_cycles = max(1, actual_len // n_steps)
        effective_len = m_cycles * n_steps

        if actual_len < target_len:
            print(f"  Notice: Target context length {target_len} clamped to model context limit {actual_len} tokens ({m_cycles} cycles).")

        proc = make_process(
            "double_pendulum_mess4",
            m=m_cycles,
            n_steps=n_steps,
            delta_v=info["delta_v"],
            dt=info["dt"],
            system={"gamma1": info["gamma"], "gamma2": info["gamma"]},
        )

        print(f"  Running [{regime_label.upper()}]: {effective_len} tokens ({m_cycles} cycles)...", flush=True)
        res = evaluate_single_regime_belief_kl(
            trained_model=trained_m,
            random_model=random_m,
            proc=proc,
            n_trajs=n_trajs,
            device=device,
            seed=seed,
            smoke=smoke,
        )
        res["target_len"] = target_len
        res["regime"] = regime_label
        regimes_data[regime_label] = res

    elapsed = time.perf_counter() - t0

    # Console Summary Ladder across context lengths
    print("=" * 86)
    print(f" MULTI-REGIME BELIEF STATE KL EVALUATION (OPTION B): {job_id}")
    print("=" * 86)
    print(f"{'Regime':<10} | {'Tokens':<8} | {'Cycles':<7} | {'Trained KL':<11} | {'1-HMM KL':<10} | {'0-HMM KL':<10} | {'Random KL':<10} | {'Net Gain':<10}")
    print("-" * 86)
    for r_key in ["short", "mid", "long"]:
        if r_key in regimes_data:
            rd = regimes_data[r_key]
            print(
                f"{r_key.capitalize():<10} | {rd['seq_len']:<8} | {rd['m_cycles']:<7} | "
                f"{rd['best_trained_kl']:7.4f} nats | {rd['1hmm_baseline_kl']:7.4f} nats | "
                f"{rd['0hmm_baseline_kl']:7.4f} nats | {rd['best_random_kl']:7.4f} nats | "
                f"{rd['net_emergence_gain']:+7.4f} nats"
            )
    print("=" * 86)

    # Save summary artifacts
    model_out_dir = out_dir / job_id
    model_out_dir.mkdir(parents=True, exist_ok=True)

    result_payload = {
        "job_id": job_id,
        "checkpoint": str(ckpt_path),
        "evaluated_at": time.time(),
        "elapsed_seconds": elapsed,
        "info": {k: v for k, v in info.items() if k != "loss_history_df"},
        "regimes": {
            r: {k: v for k, v in rd.items() if not k.startswith("_") and k not in ("loss_by_pos",)}
            for r, rd in regimes_data.items()
        },
    }

    summary_json_path = model_out_dir / f"{job_id}_optimal_observer_kl.json"
    with open(summary_json_path, "w") as f:
        json.dump(result_payload, f, indent=2)
    print(f"Saved multi-regime JSON metrics -> {summary_json_path}")

    # Save multi-regime summary CSV
    regime_rows = []
    for r_key, rd in regimes_data.items():
        regime_rows.append({
            "regime": r_key,
            "target_len": rd["target_len"],
            "seq_len": rd["seq_len"],
            "m_cycles": rd["m_cycles"],
            "best_layer": rd["best_layer_name"],
            "trained_kl": rd["best_trained_kl"],
            "1hmm_kl": rd["1hmm_baseline_kl"],
            "0hmm_kl": rd["0hmm_baseline_kl"],
            "random_kl": rd["best_random_kl"],
            "net_gain": rd["net_emergence_gain"],
            "trained_acc": rd["best_trained_acc"],
            "1hmm_acc": rd["1hmm_baseline_acc"],
            "0hmm_acc": rd["0hmm_baseline_acc"],
            "mean_loss": rd["mean_eval_loss"],
        })
    pd.DataFrame(regime_rows).to_csv(model_out_dir / f"{job_id}_belief_kl_summary_by_regime.csv", index=False)

    # Save per-cycle CSV for the longest available regime
    longest_key = "long" if "long" in regimes_data else list(regimes_data.keys())[-1]
    pd.DataFrame(regimes_data[longest_key]["cycle_results"]).to_csv(
        model_out_dir / f"{job_id}_belief_kl_by_cycle.csv", index=False
    )

    if save_plot:
        plot_path = model_out_dir / f"{job_id}_kl_analysis.png"
        save_multiregime_kl_diagnostic_plots(
            regimes_data=regimes_data,
            training_loss_df=info.get("loss_history_df"),
            job_id=job_id,
            out_path=plot_path,
        )

    return result_payload


# ==============================================================================
# 8. Main Orchestrator Flow
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Measure KL Divergence between Probed Transformer Belief States and Optimal Observer across Context Lengths (Option B)"
    )
    parser.add_argument("--checkpoint", "-c", type=str, default=None,
                        help="Path to specific *_final.pt model checkpoint to evaluate")
    parser.add_argument("--dir", "-d", "--results-dir", dest="results_dir", type=str, default=None,
                        help="Directory to scan for *_final.pt checkpoints (default: ./experiments/results)")
    parser.add_argument("--models", "-m-filter", type=str, default=None,
                        help="Optional substring filter for model IDs when scanning directory")
    parser.add_argument("--seq-lens", nargs="+", type=int, default=[500, 1500, 2500],
                        help="Context lengths to evaluate (default: 500 1500 2500 tokens)")
    parser.add_argument("--m", type=int, default=None,
                        help="Single HMM cycle count override (e.g. --m 20)")
    parser.add_argument("--n-trajs", type=int, default=64,
                        help="Number of evaluation trajectories to simulate (default: 64)")
    parser.add_argument("--smoke", action="store_true",
                        help="Rapid smoke testing mode (16 trajectories)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to run inference on (default: cuda:0 if available, else cpu)")
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

    # Determine target sequence lengths
    if args.m is not None:
        target_seq_lens = [args.m * 10]
    else:
        target_seq_lens = args.seq_lens

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

    print("=" * 86)
    print(" OPTIMAL OBSERVER & HMM BASELINES MULTI-REGIME BELIEF STATE KL (OPTION B)")
    print("=" * 86)
    print(f"Device:           {device}")
    print(f"Sequence Lengths: {target_seq_lens} tokens (Short / Mid / Long)")
    print(f"Trajectories:     {16 if args.smoke else args.n_trajs} sequences")
    print(f"Checkpoints:      {len(ckpts)} checkpoint(s) queued for evaluation")
    print(f"Output Directory: {out_dir}")
    print("=" * 86)

    all_results: List[Dict[str, Any]] = []
    for idx, ckpt in enumerate(ckpts, 1):
        print(f"\nProcessing [{idx}/{len(ckpts)}]: {ckpt.name}")
        try:
            res = evaluate_checkpoint_multiregime_kl(
                ckpt_path=ckpt,
                out_dir=out_dir,
                target_seq_lens=target_seq_lens,
                n_trajs=args.n_trajs,
                device=device,
                smoke=args.smoke,
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
            j_id = r["job_id"]
            row = {"job_id": j_id}
            for reg in ["short", "mid", "long"]:
                if reg in r["regimes"]:
                    rg_d = r["regimes"][reg]
                    row[f"{reg}_trained_kl"] = rg_d["best_trained_kl"]
                    row[f"{reg}_1hmm_kl"] = rg_d["1hmm_baseline_kl"]
                    row[f"{reg}_0hmm_kl"] = rg_d["0hmm_baseline_kl"]
                    row[f"{reg}_random_kl"] = rg_d["best_random_kl"]
                    row[f"{reg}_net_gain"] = rg_d["net_emergence_gain"]
            summary_rows.append(row)

        summary_df = pd.DataFrame(summary_rows)
        master_csv_path = out_dir / "optimal_observer_kl_summary.csv"
        summary_df.to_csv(master_csv_path, index=False)

        print("\n" + "=" * 86)
        print(" MASTER MULTI-REGIME EVALUATION SUMMARY TABLE")
        print("=" * 86)
        print(summary_df.to_string(index=False))
        print("=" * 86)
        print(f"Master summary CSV saved -> {master_csv_path}")


if __name__ == "__main__":
    main()
