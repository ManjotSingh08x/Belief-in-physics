#!/usr/bin/env python3
"""Measure KL Divergence between Transformer Next-Token Probabilities and the Optimal Observer.

This script evaluates how closely a trained transformer's next-token predictive
distribution P_model matches the true physical/Markovian optimal observer P_opt:

1. Physical / Markovian Ground-Truth (Optimal Observer P_opt):
   - Interior ODE Steps: Deterministic Lagrangian integration (H = 0 nats).
     P_opt places 100% probability mass on the exact discrete token produced by RK4 flow.
   - Cycle Boundary Transitions: An HMM kick occurs with letter probability (b_t @ E)_l.
     P_opt branches over the K=4 physical kicks, producing a categorical distribution
     over the resulting tokens with entropy H(P_opt) > 0.

2. Information-Theoretic Metrics Measured:
   - Forward KL Divergence: D_KL(P_opt || P_model) = H(P_opt, P_model) - H(P_opt)
     (in both nats and bits; measures excess surprise / suboptimality of the model).
   - Reverse KL Divergence: D_KL(P_model || P_opt) (with epsilon smoothing).
   - Jensen-Shannon Divergence: JSD(P_opt || P_model) in [0, ln 2].
   - Model Cross-Entropy: H(P_opt, P_model) (nats).
   - Optimal Observer Entropy Floor: H(P_opt) (nats).
   - Categorical breakdown: Boundary steps (HMM belief learning) vs Interior steps (physics flow).
   - Positional evolution: D_KL(t) over context length (tracks belief synchronization).
   - Matched controls: Untrained random initialization baseline and uniform baseline.

Usage:
  uv run python scripts/measure_optimal_observer_kl.py --checkpoint path/to/model_final.pt
  uv run python scripts/measure_optimal_observer_kl.py --dir ./experiments/results --smoke
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

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.transformer import ModelConfig, TinyTransformer
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
    override_seq_len: Optional[int] = None,
) -> Tuple[TinyTransformer, TinyTransformer, Any, Dict[str, Any]]:
    """Loads trained checkpoint, matched random control, and reconstructed process."""
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

    # Determine sequence length and cycles m
    target_seq_len = override_seq_len or n_ctx
    target_seq_len = min(target_seq_len, n_ctx)
    m = max(1, target_seq_len // n_steps)
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
# 2. Trajectory Rollout with Physical States
# ==============================================================================

def rollout_trajectories_with_states(
    proc: Any,
    n_trajs: int,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    """Generates n trajectories recording continuous physical states, beliefs, and tokens."""
    states, letters = proc.chain.sample(rng, n_trajs, proc.m)
    tick_beliefs = proc.chain.beliefs(letters)

    try:
        z0 = proc.system.initial_state(n_trajs, rng=rng, delta_v=proc.delta_v)
    except TypeError:
        try:
            z0 = proc.system.initial_state(n_trajs, rng=rng)
        except TypeError:
            z0 = proc.system.initial_state(n_trajs)

    z = np.asarray(z0, dtype=np.float64).copy()
    seq_len = proc.seq_len
    state_dim = z.shape[1]

    continuous_states = np.empty((n_trajs, seq_len, state_dim), dtype=np.float64)
    obs = np.empty((n_trajs, seq_len, len(proc.obs_bins)), dtype=np.float64)

    for t_idx in range(proc.m):
        z = proc.system.kick(z, proc.actions[letters[:, t_idx]])
        for s_idx in range(proc.n_steps):
            z = proc.system.flow(z, proc.dt)
            curr_pos = t_idx * proc.n_steps + s_idx
            continuous_states[:, curr_pos] = z
            obs[:, curr_pos] = proc.channels(z)

    tokens = proc.discretise(obs)
    return {
        "tokens": tokens,
        "continuous_states": continuous_states,
        "letters": letters,
        "tick_beliefs": tick_beliefs,
        "states": states,
    }


# ==============================================================================
# 3. Core Information-Theoretic Evaluator
# ==============================================================================

def compute_optimal_observer_kl(
    model: TinyTransformer,
    proc: Any,
    rollout: Dict[str, np.ndarray],
    device: str = "cpu",
    chunk_size: int = 16,
) -> Dict[str, Any]:
    """Computes exact next-token KL divergence and JSD against the Optimal Observer."""
    tokens_np = rollout["tokens"]
    continuous_states = rollout["continuous_states"]
    tick_beliefs = rollout["tick_beliefs"]

    n_trajs, seq_len = tokens_np.shape
    n_steps = proc.n_steps
    m_cycles = proc.m
    K = proc.chain.n_states
    E = proc.chain.E  # (K, K) emission matrix
    vocab_size = proc.n_obs

    # 1. Forward model on device to extract log_softmax probabilities
    log_probs_list: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, n_trajs, chunk_size):
            end = min(start + chunk_size, n_trajs)
            tok_batch = torch.as_tensor(tokens_np[start:end], dtype=torch.long, device=device)
            # Logits shape: (chunk, seq_len, vocab_size)
            logits = model(tok_batch)[:, :-1, :]  # Predict next tokens t+1
            log_p = F.log_softmax(logits, dim=-1).cpu().numpy()
            log_probs_list.append(log_p)

    log_probs = np.concatenate(log_probs_list, axis=0)  # Shape: (N, seq_len - 1, vocab_size)

    # 2. Iterate over all transitions and compute information divergences
    n_transitions = seq_len - 1

    kl_all = np.empty((n_trajs, n_transitions), dtype=np.float64)
    ce_all = np.empty((n_trajs, n_transitions), dtype=np.float64)
    h_opt_all = np.empty((n_trajs, n_transitions), dtype=np.float64)
    jsd_all = np.empty((n_trajs, n_transitions), dtype=np.float64)
    is_boundary = np.zeros(n_transitions, dtype=bool)
    top1_opt_match = np.empty((n_trajs, n_transitions), dtype=bool)
    top1_target_match = np.empty((n_trajs, n_transitions), dtype=bool)

    ln2 = math.log(2.0)

    for i in range(n_trajs):
        for t in range(n_transitions):
            c = t // n_steps
            s = t % n_steps
            log_q = log_probs[i, t]  # (vocab_size,)
            actual_next_tok = tokens_np[i, t + 1]

            if s == n_steps - 1:
                # -------------------------------------------------------------
                # Case B: Boundary Step (Incoming kick with HMM letter uncertainty)
                # -------------------------------------------------------------
                is_boundary[t] = True
                b = tick_beliefs[i, c]  # Predictive belief for tick c+1, shape (K,)
                p_letters = b @ E       # Probability distribution over letter kicks, shape (K,)
                zt = continuous_states[i, t]  # Continuous state at boundary

                # Evaluate all K=4 physical kicks and ODE flow
                z_branches = np.repeat(zt[np.newaxis, :], K, axis=0)
                z_kicked = proc.system.kick(z_branches, proc.actions)
                z_flowed = proc.system.flow(z_kicked, proc.dt)
                branch_tokens = proc.discretise(proc.channels(z_flowed))

                # Aggregate mass on discrete token grid (handles any bin collisions)
                mass: Dict[int, float] = defaultdict(float)
                for l_idx, tok in enumerate(branch_tokens):
                    mass[tok] += float(p_letters[l_idx])

                # Optimal Entropy H(P_opt) & Cross-Entropy H(P_opt, P_model)
                ce = -sum(p * log_q[tok] for tok, p in mass.items())
                h_opt = -sum(p * math.log(max(p, 1e-15)) for p in mass.values())
                kl = max(0.0, ce - h_opt)

                # Fast Exact JSD: only sums over support of P_opt
                q_probs = np.exp(log_q)
                kl_pm = sum(p * math.log(2.0 * p / (p + max(q_probs[tok], 1e-15))) for tok, p in mass.items())
                sum_q_supp = sum(q_probs[tok] for tok in mass.keys())
                kl_qm = sum(
                    q_probs[tok] * math.log(max(2.0 * q_probs[tok] / (p + q_probs[tok]), 1e-15))
                    for tok, p in mass.items()
                ) + (1.0 - sum_q_supp) * ln2
                jsd = max(0.0, min(ln2, 0.5 * (kl_pm + kl_qm)))

                # Top-1 agreement
                best_opt_tok = max(mass.items(), key=lambda kv: kv[1])[0]
                pred_tok = int(np.argmax(log_q))
                top1_opt_match[i, t] = (pred_tok == best_opt_tok)
                top1_target_match[i, t] = (pred_tok == actual_next_tok)

            else:
                # -------------------------------------------------------------
                # Case A: Interior Step (Deterministic ODE flow, H(P_opt) = 0)
                # -------------------------------------------------------------
                # P_opt is a delta mass on actual_next_tok
                ce = -float(log_q[actual_next_tok])
                h_opt = 0.0
                kl = ce

                # Fast Exact JSD for delta mass
                q_star = max(float(np.exp(log_q[actual_next_tok])), 1e-15)
                kl_pm = math.log(2.0 / (1.0 + q_star))
                kl_qm = q_star * math.log(2.0 * q_star / (1.0 + q_star)) + (1.0 - q_star) * ln2
                jsd = max(0.0, min(ln2, 0.5 * (kl_pm + kl_qm)))

                pred_tok = int(np.argmax(log_q))
                top1_opt_match[i, t] = (pred_tok == actual_next_tok)
                top1_target_match[i, t] = (pred_tok == actual_next_tok)

            kl_all[i, t] = kl
            ce_all[i, t] = ce
            h_opt_all[i, t] = h_opt
            jsd_all[i, t] = jsd

    # Aggregations
    boundary_mask = is_boundary
    interior_mask = ~is_boundary

    kl_mean_nats = float(np.mean(kl_all))
    kl_mean_bits = float(kl_mean_nats / ln2)
    ce_mean = float(np.mean(ce_all))
    h_opt_mean = float(np.mean(h_opt_all))
    jsd_mean = float(np.mean(jsd_all))

    kl_boundary_nats = float(np.mean(kl_all[:, boundary_mask])) if np.any(boundary_mask) else 0.0
    ce_boundary = float(np.mean(ce_all[:, boundary_mask])) if np.any(boundary_mask) else 0.0
    h_boundary = float(np.mean(h_opt_all[:, boundary_mask])) if np.any(boundary_mask) else 0.0

    kl_interior_nats = float(np.mean(kl_all[:, interior_mask])) if np.any(interior_mask) else 0.0
    ce_interior = float(np.mean(ce_all[:, interior_mask])) if np.any(interior_mask) else 0.0

    # Per-step and per-cycle statistics
    kl_by_step = np.mean(kl_all, axis=0)
    ce_by_step = np.mean(ce_all, axis=0)
    h_by_step = np.mean(h_opt_all, axis=0)
    jsd_by_step = np.mean(jsd_all, axis=0)

    # Per-cycle aggregation (m_cycles)
    cycle_records = []
    for c in range(m_cycles):
        start_step = c * n_steps
        end_step = min((c + 1) * n_steps, n_transitions)
        if start_step >= n_transitions:
            break
        c_kl = np.mean(kl_all[:, start_step:end_step])
        c_ce = np.mean(ce_all[:, start_step:end_step])
        c_b_step = (c + 1) * n_steps - 1
        c_b_kl = np.mean(kl_all[:, c_b_step]) if c_b_step < n_transitions else np.nan
        cycle_records.append({
            "cycle": c,
            "kl_mean": float(c_kl),
            "cross_entropy": float(c_ce),
            "boundary_kl": float(c_b_kl) if np.isfinite(c_b_kl) else None,
        })

    return {
        "kl_mean_nats": kl_mean_nats,
        "kl_mean_bits": kl_mean_bits,
        "cross_entropy_mean": ce_mean,
        "optimal_entropy_mean": h_opt_mean,
        "jsd_mean": jsd_mean,
        "kl_boundary_nats": kl_boundary_nats,
        "kl_boundary_bits": kl_boundary_nats / ln2,
        "ce_boundary": ce_boundary,
        "optimal_entropy_boundary": h_boundary,
        "kl_interior_nats": kl_interior_nats,
        "kl_interior_bits": kl_interior_nats / ln2,
        "ce_interior": ce_interior,
        "top1_accuracy_vs_optimal": float(np.mean(top1_opt_match)),
        "top1_accuracy_vs_target": float(np.mean(top1_target_match)),
        "uniform_kl_nats": float(math.log(vocab_size) - h_opt_mean),
        "kl_by_step": kl_by_step.tolist(),
        "ce_by_step": ce_by_step.tolist(),
        "h_by_step": h_by_step.tolist(),
        "jsd_by_step": jsd_by_step.tolist(),
        "is_boundary_step": is_boundary.tolist(),
        "cycle_summary": cycle_records,
    }


# ==============================================================================
# 4. Diagnostic Plotting
# ==============================================================================

def save_kl_diagnostic_plots(
    trained_metrics: Dict[str, Any],
    random_metrics: Optional[Dict[str, Any]],
    job_id: str,
    out_path: Path,
) -> None:
    """Renders a 3-panel publication-grade diagnostic figure of KL divergence."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("Notice: matplotlib not installed, skipping diagnostic plot generation.")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")

    steps = np.arange(len(trained_metrics["kl_by_step"]))
    trained_kl_step = np.array(trained_metrics["kl_by_step"])
    is_b = np.array(trained_metrics["is_boundary_step"])

    # Panel 1: Step-by-Step KL Divergence Curve
    ax0 = axes[0]
    ax0.set_facecolor("white")
    ax0.plot(steps, trained_kl_step, color="#1b4965", lw=1.2, label="Trained Model KL")
    if random_metrics:
        ax0.plot(steps, random_metrics["kl_by_step"], color="#9a8c98", lw=0.9, ls="--", alpha=0.7, label="Random Control KL")

    # Highlight boundary transitions (kicks)
    b_steps = steps[is_b]
    ax0.scatter(b_steps, trained_kl_step[is_b], color="#e63946", s=18, zorder=5, label="Boundary Kick Transition")
    ax0.axhline(0.0, color="gray", lw=0.6, ls=":")
    ax0.set(
        title="1. Next-Token KL Divergence vs Step",
        xlabel="Step in Sequence (t)",
        ylabel="D_KL(P_opt || P_model) [nats]",
    )
    ax0.grid(True, alpha=0.25)
    ax0.legend(frameon=True, facecolor="white", edgecolor="#ddd", fontsize=8)

    # Panel 2: Cycle-by-Cycle Synchronization
    ax1 = axes[1]
    ax1.set_facecolor("white")
    cycles = [r["cycle"] for r in trained_metrics["cycle_summary"]]
    c_kl = [r["kl_mean"] for r in trained_metrics["cycle_summary"]]
    c_b_kl = [r["boundary_kl"] if r["boundary_kl"] is not None else 0.0 for r in trained_metrics["cycle_summary"]]

    ax1.plot(cycles, c_kl, marker="o", markersize=4, color="#2b78d6", lw=1.5, label="Mean Cycle KL")
    ax1.plot(cycles, c_b_kl, marker="s", markersize=4, color="#e63946", lw=1.2, ls="--", label="Boundary-Only KL")
    ax1.set(
        title="2. KL Divergence Across Cycles (Belief Synchronization)",
        xlabel="Cycle Index (m)",
        ylabel="Mean D_KL [nats]",
    )
    ax1.grid(True, alpha=0.25)
    ax1.legend(frameon=True, facecolor="white", edgecolor="#ddd", fontsize=8)

    # Panel 3: Information Budget Breakdown Bar Chart
    ax2 = axes[2]
    ax2.set_facecolor("white")
    categories = ["Total", "Boundary (HMM)", "Interior (ODE)"]
    trained_vals = [
        trained_metrics["kl_mean_nats"],
        trained_metrics["kl_boundary_nats"],
        trained_metrics["kl_interior_nats"],
    ]
    x_pos = np.arange(len(categories))
    bar_width = 0.35

    ax2.bar(x_pos - bar_width / 2, trained_vals, width=bar_width, color="#2b78d6", label="Trained Model")
    if random_metrics:
        random_vals = [
            random_metrics["kl_mean_nats"],
            random_metrics["kl_boundary_nats"],
            random_metrics["kl_interior_nats"],
        ]
        ax2.bar(x_pos + bar_width / 2, random_vals, width=bar_width, color="#adb5bd", label="Random Control")

    ax2.axhline(trained_metrics["uniform_kl_nats"], color="#e76f51", ls="--", lw=1.0, label="Uniform Floor (ln V)")
    ax2.set(
        title="3. KL Divergence Regime Breakdown",
        ylabel="D_KL [nats]",
    )
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(categories)
    ax2.grid(True, alpha=0.25)
    ax2.legend(frameon=True, facecolor="white", edgecolor="#ddd", fontsize=8)

    fig.suptitle(f"Optimal Observer KL Divergence Analysis: {job_id}", fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved diagnostic plots -> {out_path}")


# ==============================================================================
# 5. Single Checkpoint Evaluation Pipeline
# ==============================================================================

def evaluate_checkpoint_kl(
    ckpt_path: Path,
    out_dir: Path,
    n_trajs: int = 64,
    device: str = "cpu",
    seed: int = 20260829,
    smoke: bool = False,
    override_seq_len: Optional[int] = None,
    save_plot: bool = True,
) -> Dict[str, Any]:
    """Runs full KL evaluation for one model checkpoint against the Optimal Observer."""
    actual_trajs = 8 if smoke else n_trajs
    t0 = time.perf_counter()

    trained_m, random_m, proc, info = load_model_and_process(
        ckpt_path=ckpt_path,
        device=device,
        override_seq_len=override_seq_len,
    )

    print(f"\n[{info['job_id']}] Evaluating against Optimal Observer...")
    print(f"  Configuration: dv={info['delta_v']}, gamma={info['gamma']}, n={info['n_steps']}, m={info['m']}")
    print(f"  Vocabulary:    {info['vocab_size']} bins, Evaluation Seq Length: {info['seq_len']} tokens")
    print(f"  Rollout:       {actual_trajs} test trajectories ({actual_trajs * (info['seq_len'] - 1):,} total transitions)")

    # Generate test rollouts
    eval_rng = np.random.default_rng(seed)
    rollout = rollout_trajectories_with_states(proc, actual_trajs, eval_rng)

    # Evaluate trained model
    trained_metrics = compute_optimal_observer_kl(trained_m, proc, rollout, device=device)

    # Evaluate random control model
    random_metrics = compute_optimal_observer_kl(random_m, proc, rollout, device=device)

    elapsed = time.perf_counter() - t0

    result_payload = {
        "job_id": info["job_id"],
        "checkpoint": str(ckpt_path),
        "evaluated_at": time.time(),
        "elapsed_seconds": elapsed,
        "n_trajs": actual_trajs,
        "seq_len": info["seq_len"],
        "info": info,
        "trained_model": trained_metrics,
        "random_control": random_metrics,
    }

    # Console Report
    print("-" * 78)
    print(f" OPTIMAL OBSERVER KL EVALUATION RESULTS: {info['job_id']}")
    print("-" * 78)
    print(f" Forward KL D_KL(P_opt || P_model):  {trained_metrics['kl_mean_nats']:8.4f} nats  ({trained_metrics['kl_mean_bits']:8.4f} bits)")
    print(f" Random Control Forward KL:          {random_metrics['kl_mean_nats']:8.4f} nats  ({random_metrics['kl_mean_bits']:8.4f} bits)")
    print(f" Uniform Predictor Baseline KL:      {trained_metrics['uniform_kl_nats']:8.4f} nats")
    print(f" Model Cross-Entropy Loss:           {trained_metrics['cross_entropy_mean']:8.4f} nats")
    print(f" Optimal Observer Entropy Floor:     {trained_metrics['optimal_entropy_mean']:8.4f} nats")
    print(f" Jensen-Shannon Divergence (JSD):    {trained_metrics['jsd_mean']:8.4f} / ln(2) = {trained_metrics['jsd_mean']/math.log(2):.4f}")
    print(f" Top-1 Match vs Optimal Argmax:      {trained_metrics['top1_accuracy_vs_optimal']:8.2%}")
    print(f" Top-1 Match vs Actual Next Token:   {trained_metrics['top1_accuracy_vs_target']:8.2%}")
    print(" Regime Breakdown:")
    print(f"   * Boundary Transitions (HMM Kick): D_KL = {trained_metrics['kl_boundary_nats']:.4f} nats (H_opt = {trained_metrics['optimal_entropy_boundary']:.4f})")
    print(f"   * Interior Steps (ODE Physics Flow): D_KL = {trained_metrics['kl_interior_nats']:.4f} nats (H_opt = 0.0000)")
    print("-" * 78)

    # Save summary artifacts
    model_out_dir = out_dir / info["job_id"]
    model_out_dir.mkdir(parents=True, exist_ok=True)

    summary_json_path = model_out_dir / f"{info['job_id']}_optimal_observer_kl.json"
    with open(summary_json_path, "w") as f:
        json.dump(result_payload, f, indent=2)
    print(f"Saved JSON metrics -> {summary_json_path}")

    # Save CSV of per-step KL divergence
    csv_path = model_out_dir / f"{info['job_id']}_kl_by_step.csv"
    step_df = pd.DataFrame({
        "step": np.arange(len(trained_metrics["kl_by_step"])),
        "is_boundary": trained_metrics["is_boundary_step"],
        "trained_kl": trained_metrics["kl_by_step"],
        "random_kl": random_metrics["kl_by_step"],
        "trained_ce": trained_metrics["ce_by_step"],
        "optimal_entropy": trained_metrics["h_by_step"],
        "jsd": trained_metrics["jsd_by_step"],
    })
    step_df.to_csv(csv_path, index=False)
    print(f"Saved per-step CSV -> {csv_path}")

    if save_plot:
        plot_path = model_out_dir / f"{info['job_id']}_kl_analysis.png"
        save_kl_diagnostic_plots(trained_metrics, random_metrics, info["job_id"], plot_path)

    return result_payload


# ==============================================================================
# 6. Main Orchestrator Flow
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Measure KL Divergence between Transformer Next-Token Probabilities and Optimal Observer"
    )
    parser.add_argument("--checkpoint", "-c", type=str, default=None,
                        help="Path to specific *_final.pt model checkpoint to evaluate")
    parser.add_argument("--dir", "-d", "--results-dir", dest="results_dir", type=str, default=None,
                        help="Directory to scan for *_final.pt checkpoints (default: ./experiments/results)")
    parser.add_argument("--models", "-m", type=str, default=None,
                        help="Optional substring filter for model IDs when scanning directory")
    parser.add_argument("--n-trajs", type=int, default=64,
                        help="Number of evaluation trajectories to simulate (default: 64)")
    parser.add_argument("--smoke", action="store_true",
                        help="Rapid smoke testing mode (8 trajectories)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to run inference on (default: cuda:0 if available, else cpu)")
    parser.add_argument("--seq-len", type=int, default=None,
                        help="Override evaluation sequence length (default: model n_ctx)")
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

    print("=" * 78)
    print(" OPTIMAL OBSERVER NEXT-TOKEN KL DIVERGENCE MEASUREMENT")
    print("=" * 78)
    print(f"Device:           {device}")
    print(f"Trajectories:     {8 if args.smoke else args.n_trajs}")
    print(f"Checkpoints:      {len(ckpts)} checkpoint(s) queued for evaluation")
    print(f"Output Directory: {out_dir}")
    print("=" * 78)

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
            tm = r["trained_model"]
            rm = r["random_control"]
            info = r["info"]
            summary_rows.append({
                "job_id": r["job_id"],
                "k": info["k"],
                "seq_len": info["seq_len"],
                "trained_kl_nats": tm["kl_mean_nats"],
                "trained_kl_bits": tm["kl_mean_bits"],
                "random_kl_nats": rm["kl_mean_nats"],
                "kl_gain_over_random": rm["kl_mean_nats"] - tm["kl_mean_nats"],
                "cross_entropy": tm["cross_entropy_mean"],
                "oracle_entropy_floor": tm["optimal_entropy_mean"],
                "boundary_kl": tm["kl_boundary_nats"],
                "interior_kl": tm["kl_interior_nats"],
                "jsd": tm["jsd_mean"],
                "top1_acc_vs_opt": tm["top1_accuracy_vs_optimal"],
            })
        summary_df = pd.DataFrame(summary_rows)
        master_csv_path = out_dir / "optimal_observer_kl_summary.csv"
        summary_df.to_csv(master_csv_path, index=False)

        print("\n" + "=" * 78)
        print(" MASTER EVALUATION SUMMARY TABLE")
        print("=" * 78)
        cols_to_show = ["job_id", "k", "trained_kl_nats", "random_kl_nats", "oracle_entropy_floor", "top1_acc_vs_opt"]
        print(summary_df[cols_to_show].to_string(index=False))
        print("=" * 78)
        print(f"Master summary CSV saved -> {master_csv_path}")


if __name__ == "__main__":
    main()
