#!/usr/bin/env python3
"""Double Pendulum Production Training & Multi-Platform Queue Orchestrator (v2: 15 Models).

Key Specifications:
- 1 model trained per physics configuration x lookahead horizon (no multi-seed sweep).
- 5 distilled physics configurations x 3 lookahead horizons (_k1, _k_mid, _k_full) x 1 fixed width (d_model=128) = 15 models total.
- Trajectory starting angular velocities randomized across [-delta_v, delta_v] via truncated normal distribution.
- Clean model naming: {exp_name}_{train_name} without seed suffixes.
- Seed column omitted from probe results and manifest.
- Multi-Platform Modes: 'rtx5090' (concurrent workers on cuda:0), 'kaggle' (2 workers on cuda:0/1), 'local_cpu' (1 worker).
- Smoke mode: 10,000 tokens for rapid ~2-second sanity testing.
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
from dataclasses import asdict, dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Headless server safe
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import GroupKFold

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.transformer import ModelConfig, TinyTransformer
from physics.controls import compute_optimal_gamma
from physics.messk import simplex_embedding, square_projection_vertices
from physics.messk_configs import DOUBLE_PENDULUM_BINS, make_process, set_double_pendulum_bins
from physics.visualise import stability, trace

# Enforce 50x50 observation grid resolution (vocabulary = 2500)
set_double_pendulum_bins((50, 50))


# ==============================================================================
# 1. Platform Detection & Hardware Configuration
# ==============================================================================

def detect_platform_mode() -> str:
    env_mode = os.environ.get("BELIEF_PLATFORM", "").lower().strip()
    if env_mode in ("rtx5090", "kaggle", "local_cpu"):
        return env_mode
    if Path("/kaggle").exists():
        return "kaggle"
    if torch.cuda.is_available():
        dev_name = torch.cuda.get_device_name(0).lower()
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if "5090" in dev_name or vram_gb >= 28.0:
            return "rtx5090"
        if torch.cuda.device_count() >= 2:
            return "kaggle"
        return "rtx5090"
    return "local_cpu"


# ==============================================================================
# 2. Lookahead Transformer Architecture
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


# ==============================================================================
# 3. Data Configurations & Level 1 Distillation Data Structures
# ==============================================================================

@dataclass
class DoublePendulumConfig:
    delta_v: float = 1.2
    gamma1: float = 0.65
    gamma2: float = 0.65
    dt: float = 0.2
    obs_bins: Tuple[int, int] = (50, 50)
    omega_max: float = 10.0
    joint1_action_gain: float = 2.0


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


@dataclass
class ModelJob:
    job_id: str
    exp_config: ExperimentConfig
    train_config: TrainingConfig
    k: int
    k_suffix: str
    tokens_seen: int
    target_tokens: int
    checkpoint_path: Path
    sidecar_path: Path
    probe_csv_path: Path
    prob_map_path: Path
    status: str
    seed: int = 20260925


# ==============================================================================
# 4. Phase 1: Physics Screening & 5-Tuple Distillation
# ==============================================================================

def run_phase1_distillation(
    results_dir: Path,
    smoke: bool = False,
    seed: int = 20260925,
) -> List[ExperimentConfig]:
    """Executes Sweep 1A and Sweep 1B, selecting the top 5 physics configurations."""
    print("\n" + "=" * 78)
    print(" PHASE 1: PHYSICS SCREENING & 5-TUPLE DISTILLATION")
    print("=" * 78)

    tuples_csv_path = results_dir / "distilled_physics_tuples.csv"
    tuples_json_path = results_dir / "distilled_physics_tuples.json"

    # Auto-resume check: reload previously distilled tuples if present
    if tuples_csv_path.exists() and tuples_json_path.exists():
        print(f"Loading existing distilled physics tuples from {tuples_csv_path.name}...")
        df_top = pd.read_csv(tuples_csv_path)
        configs = []
        for idx, row in df_top.iterrows():
            tuple_name = str(row["name"]) if "name" in row else f"tuple{idx+1:02d}_dv{row['delta_v']:.2f}_g{row['gamma']:.2f}_n{int(row['n_steps'])}"
            configs.append(ExperimentConfig(
                name=tuple_name,
                physics=DoublePendulumConfig(
                    delta_v=float(row["delta_v"]),
                    gamma1=float(row["gamma"]),
                    gamma2=float(row["gamma"]),
                    dt=0.2,
                    obs_bins=(50, 50),
                ),
                hmm=HMMConfig(n_states=4, alpha=0.7, stay=0.7),
                m=int(row["m"]),
                n=int(row["n_steps"]),
                seed=seed,
            ))
        print(f"Resumed {len(configs)} physics tuples from Level 1 export.")
        return configs

    # Sweep 1A: (dv x n) Grid Screening
    delta_vs = [0.8, 1.2] if smoke else [0.6, 0.8, 1.0, 1.2, 1.5]
    n_candidates = [10, 25] if smoke else [5, 10, 20, 25]

    sweep_1a_records = []
    print(f"Executing Sweep 1A: {len(delta_vs)} dv values x {len(n_candidates)} n values under baseline gamma=0.65...")
    pair_id = 0
    for dv in delta_vs:
        for n in n_candidates:
            pair_id += 1
            m = 5000 // n
            proc = make_process(
                "double_pendulum_mess4",
                m=m, n_steps=n, delta_v=dv, dt=0.2,
                system={"gamma1": 0.65, "gamma2": 0.65},
            )
            tr = trace(proc, seed=42)
            stab = stability(tr)

            used_bins = int(stab["used_bins"])
            used_bins_frac = float(used_bins) / float(stab["n_obs"])
            gap_free = float(stab.get("gap_free_mean", 0.0))
            score = (
                float(stab["bayes_gap"])
                - 0.5 * min(0.0, float(stab["lyapunov"]))
                - 10.0 * float(stab["clipped"])
                - 5.0 * max(0.0, 0.10 - used_bins_frac)
            )
            is_go = (
                float(stab["lyapunov"]) < -0.30
                and float(stab["clipped"]) < 0.01
                and used_bins_frac >= 0.10
                and float(stab["bayes_gap"]) >= 0.15
                and gap_free > 1e-3
            )
            sweep_1a_records.append({
                "pair_id": pair_id,
                "delta_v": dv,
                "n_steps": n,
                "m": m,
                "lyapunov": stab["lyapunov"],
                "clipped": stab["clipped"],
                "used_bins": stab["used_bins"],
                "used_bins_frac": used_bins_frac,
                "gap_free_mean": gap_free,
                "bayes_gap": stab["bayes_gap"],
                "score": score,
                "is_go": is_go,
            })

    df_1a = pd.DataFrame(sweep_1a_records).sort_values("score", ascending=False).reset_index(drop=True)
    print(f"Sweep 1A complete. Evaluated {len(df_1a)} candidate (dv, n) pairs.")

    # Distillation Checkpoint 1A: Filter is_go == True before taking top candidates
    n_pairs_target = min(len(df_1a), 2 if smoke else 5)
    passing_1a = df_1a[df_1a["is_go"] == True]
    if len(passing_1a) == 0:
        print("Warning: 0 pairs passed is_go in Sweep 1A; falling back to top non-chaotic pairs by score.")
        candidate_pool_1a = df_1a[df_1a["lyapunov"] < 0.0]
        if len(candidate_pool_1a) == 0:
            candidate_pool_1a = df_1a
        top_pairs = candidate_pool_1a.head(n_pairs_target).copy()
    elif len(passing_1a) < n_pairs_target:
        print(f"Warning: Only {len(passing_1a)} pairs passed is_go in Sweep 1A (target: {n_pairs_target}). Taking all passing pairs.")
        top_pairs = passing_1a.copy()
    else:
        top_pairs = passing_1a.head(n_pairs_target).copy()

    # Intermediate Step: Energy-Balance Optimal Gamma Calculation
    k_factors = [0.03, 0.07] if smoke else [0.03, 0.07, 0.10]
    d_gamma = 0.05

    gamma_candidates_by_pair = {}
    for _, row in top_pairs.iterrows():
        p_id = int(row["pair_id"])
        dv_val = float(row["delta_v"])
        n_val = int(row["n_steps"])
        m_val = int(row["m"])

        proc_temp = make_process(
            "double_pendulum_mess4",
            m=m_val, n_steps=n_val, delta_v=dv_val, dt=0.2,
            system={"gamma1": 0.65, "gamma2": 0.65},
        )
        try:
            opt_res = compute_optimal_gamma(proc_temp, n_trajs=16 if smoke else 64)
            gamma_star = float(opt_res.get("optimal_gamma", 0.65))
        except Exception:
            gamma_star = 0.65

        gammas = []
        for k in k_factors:
            base_g = k * gamma_star
            for g_cand in [base_g - d_gamma, base_g, base_g + d_gamma]:
                if g_cand > 0.05:
                    gammas.append(round(g_cand, 4))
        gammas = sorted(list(set(gammas)))
        gamma_candidates_by_pair[p_id] = gammas

    # Sweep 1B: Damping Sweep on Chosen Pairs
    sweep_1b_records = []
    print("Executing Sweep 1B: Damping sweeps over candidate pairs...")
    for _, row in top_pairs.iterrows():
        p_id = int(row["pair_id"])
        dv_val = float(row["delta_v"])
        n_val = int(row["n_steps"])
        m_val = int(row["m"])
        for g_val in gamma_candidates_by_pair[p_id]:
            proc = make_process(
                "double_pendulum_mess4",
                m=m_val, n_steps=n_val, delta_v=dv_val,
                system={"gamma1": g_val, "gamma2": g_val},
            )
            tr = trace(proc, seed=42)
            stab = stability(tr)
            used_bins = int(stab["used_bins"])
            used_bins_frac = float(used_bins) / float(stab["n_obs"])
            gap_free = float(stab.get("gap_free_mean", 0.0))
            score_1b = (
                float(stab["bayes_gap"])
                - 0.5 * min(0.0, float(stab["lyapunov"]))
                - 10.0 * float(stab["clipped"])
                - 5.0 * max(0.0, 0.10 - used_bins_frac)
            )
            is_go_1b = (
                float(stab["lyapunov"]) < -0.30
                and float(stab["clipped"]) < 0.01
                and used_bins_frac >= 0.10
                and float(stab["bayes_gap"]) >= 0.15
                and gap_free > 1e-3
            )
            sweep_1b_records.append({
                "pair_id": p_id,
                "delta_v": dv_val,
                "gamma": g_val,
                "n_steps": n_val,
                "m": m_val,
                "lyapunov": stab["lyapunov"],
                "clipped": stab["clipped"],
                "used_bins": stab["used_bins"],
                "used_bins_frac": used_bins_frac,
                "gap_free_mean": gap_free,
                "bayes_gap": stab["bayes_gap"],
                "score": score_1b,
                "is_go": is_go_1b,
            })

    df_1b = pd.DataFrame(sweep_1b_records).sort_values("score", ascending=False).reset_index(drop=True)
    print(f"Sweep 1B complete. Evaluated {len(df_1b)} candidate physics tuples.")

    # Distillation Checkpoint 2: Select Top 5 Physics Tuples (Diversity Constrained)
    n_tuples_target = min(len(df_1b), 2 if smoke else 5)
    passing_1b = df_1b[df_1b["is_go"] == True]

    if len(passing_1b) == 0:
        print("Warning: 0 tuples passed is_go in Sweep 1B; falling back to non-chaotic tuples by composite score.")
        candidate_pool = df_1b[df_1b["lyapunov"] < 0.0]
        if len(candidate_pool) == 0:
            candidate_pool = df_1b
    elif len(passing_1b) < n_tuples_target:
        print(f"Warning: Only {len(passing_1b)} tuples passed is_go in Sweep 1B (target: {n_tuples_target}). Taking available passing tuples.")
        candidate_pool = passing_1b
    else:
        candidate_pool = passing_1b

    max_per_pair = 2
    selected_indices = []
    pair_counts: Dict[int, int] = {}
    remaining_indices = []

    for idx, row in candidate_pool.iterrows():
        p_id = int(row["pair_id"])
        if pair_counts.get(p_id, 0) < max_per_pair and len(selected_indices) < n_tuples_target:
            selected_indices.append(idx)
            pair_counts[p_id] = pair_counts.get(p_id, 0) + 1
        else:
            remaining_indices.append(idx)

    for idx in remaining_indices:
        if len(selected_indices) >= n_tuples_target:
            break
        selected_indices.append(idx)

    top_tuples_df = candidate_pool.loc[selected_indices].copy().reset_index(drop=True)

    chosen_configs: List[ExperimentConfig] = []
    for idx, row in top_tuples_df.iterrows():
        tuple_name = f"tuple{idx+1:02d}_dv{row['delta_v']:.2f}_g{row['gamma']:.2f}_n{int(row['n_steps'])}"
        cfg = ExperimentConfig(
            name=tuple_name,
            physics=DoublePendulumConfig(
                delta_v=float(row["delta_v"]),
                gamma1=float(row["gamma"]),
                gamma2=float(row["gamma"]),
                dt=0.2,
                obs_bins=(50, 50),
            ),
            hmm=HMMConfig(n_states=4, alpha=0.7, stay=0.7),
            m=int(row["m"]),
            n=int(row["n_steps"]),
            seed=seed,
        )
        chosen_configs.append(cfg)

    # Export Level 1 Artifacts
    top_tuples_df["name"] = [cfg.name for cfg in chosen_configs]
    top_tuples_df.to_csv(tuples_csv_path, index=False)
    with open(tuples_json_path, "w") as f:
        json.dump([asdict(c) for c in chosen_configs], f, indent=2)

    print(f"\nDistillation Checkpoint 2 Complete -> Selected {len(chosen_configs)} Top Physics Tuples:")
    for cfg in chosen_configs:
        print(f"  * {cfg.name} (dv={cfg.physics.delta_v}, gamma={cfg.physics.gamma1}, n={cfg.n}, m={cfg.m})")
    print(f"Exported Level 1 Distillations -> {tuples_csv_path.name} & {tuples_json_path.name}")
    return chosen_configs


# ==============================================================================
# 5. GPU-Accelerated Ridge Regression & Probing Engine
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


def run_synchronous_probing_gpu(
    trained_model: LookaheadTransformer,
    random_model: LookaheadTransformer,
    proc: Any,
    exp_name: str,
    training_name: str,
    device: str,
    n_eval_trajs: int = 512,
    eval_seed: int = 20260829,
) -> Tuple[pd.DataFrame, Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    """Runs 4-way Ridge regression probing directly on device using PyTorch linear algebra."""
    eval_rng = np.random.default_rng(eval_seed)
    eval_batch = proc.sample_batch(eval_rng, n_eval_trajs)
    tokens = eval_batch["tokens"]
    eval_tokens = torch.as_tensor(tokens, dtype=torch.long, device=device)

    # Extract residual streams directly on device (chunked to prevent peak VRAM/memory spikes)
    def extract_streams(model: LookaheadTransformer) -> List[torch.Tensor]:
        streams = [[] for _ in range(len(model.blocks))]
        chunk_size = 4 if device == "cpu" else 64
        with torch.no_grad():
            for i in range(0, eval_tokens.shape[0], chunk_size):
                tok_chunk = eval_tokens[i : i + chunk_size]
                x = model._embed(tok_chunk)
                for b_idx, block in enumerate(model.blocks):
                    x = block(x)
                    streams[b_idx].append(x)
        return [torch.cat(s, dim=0) for s in streams]

    trained_streams = extract_streams(trained_model)
    random_streams = extract_streams(random_model)

    N, seq_len = tokens.shape
    n = proc.n_steps
    m = proc.m
    k = trained_model.k
    assert seq_len == m * n == 5000

    # Targets on device
    beliefs_np = eval_batch["beliefs"] @ simplex_embedding(proc.chain.n_states)
    metric_np = eval_batch["metric"]

    embedded_beliefs_t = torch.as_tensor(beliefs_np, dtype=torch.float32, device=device)
    physics_metric_t = torch.as_tensor(metric_np, dtype=torch.float32, device=device)

    # Compute usable tokens indices: range of k tokens before each perturbation
    # For cycle c, perturbation occurs at (c+1)*n. Usable tokens are (c+1)*n - k to (c+1)*n - 1.
    usable_indices = np.concatenate([
        np.arange(c * n + n - k, c * n + n)
        for c in range(m)
    ])
    usable_indices_t = torch.as_tensor(usable_indices, dtype=torch.long, device=device)

    probe_modes = [
        "all_tokens_probing",
        "usable_tokens_probing",
    ]
    target_domains = ["belief", "physics"]

    results = []
    simplex_payload = None
    model_id = f"{exp_name}_{training_name}"
    n_layers = len(trained_streams)

    for layer_idx in range(n_layers):
        s_tr = trained_streams[layer_idx]
        s_rnd = random_streams[layer_idx]

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
                data = embedded_beliefs_t if domain == "belief" else physics_metric_t
                if mode == "all_tokens_probing":
                    y = data.reshape(N * seq_len, -1)
                elif mode == "usable_tokens_probing":
                    y = data[:, usable_indices_t, :].reshape(N * len(usable_indices), -1)
                else:
                    raise ValueError(f"Unknown mode: {mode}")

                # Fast GPU closed-form Ridge regression CV
                r2_trained, best_alpha, pred_y = gpu_ridge_cv(X_tr, y, groups)
                r2_random, _, _ = gpu_ridge_cv(X_rnd, y, groups)

                # Capture simplex payload for visualization (last layer under usable_tokens_probing)
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

    return pd.DataFrame(results), simplex_payload


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

    # Subsample if large for fast rendering and clean visualization
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
    """Generates 3 side-by-side heatmaps (k=1, k=n//2, k=n) matching user specification."""
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
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ==============================================================================
# 6. Worker Execution Engine (Phase 2 Training & GPU In-Worker Probing)
# ==============================================================================

def train_only_worker(
    job: ModelJob,
    total_tokens: int,
    batch_size: int,
    device: str,
    results_dir: Path,
    smoke: bool = False,
    micro_batch_size: Optional[int] = None,
) -> Tuple[str, Path, Path, pd.DataFrame]:
    """Trains single model to total_tokens, saves checkpoint and sidecar JSON, returns history."""
    model_dir = results_dir / job.job_id
    model_dir.mkdir(parents=True, exist_ok=True)
    zip_path = results_dir / f"{job.job_id}.zip"
    probe_csv_path = model_dir / f"{job.job_id}_probes.csv"

    # Auto-resume check: skip training if checkpoint or completed package already exists
    if (zip_path.exists() and probe_csv_path.exists()) or (job.checkpoint_path.exists() and job.sidecar_path.exists()):
        print(f"[{job.job_id}] Existing checkpoint found -> skipping training.", flush=True)
        return job.job_id, job.checkpoint_path, job.sidecar_path, pd.DataFrame()

    print(f"[{job.job_id}] Starting training on {device} (Target: {total_tokens:,} tokens)...", flush=True)

    # 1. Instantiate Process
    exp_cfg = job.exp_config
    proc = make_process(
        "double_pendulum_mess4",
        m=exp_cfg.m,
        n_steps=exp_cfg.n,
        delta_v=exp_cfg.physics.delta_v,
        system={"gamma1": exp_cfg.physics.gamma1, "gamma2": exp_cfg.physics.gamma2},
    )

    default_mb = 1 if device == "cpu" else (16 if smoke else int(os.environ.get("BELIEF_MICRO_BATCH", "64")))
    chosen_mb = micro_batch_size or default_mb
    micro_batch = min(chosen_mb, batch_size)
    eff_batch = micro_batch if smoke else batch_size
    grad_accum_steps = max(1, eff_batch // micro_batch)
    tokens_per_step = micro_batch * grad_accum_steps * proc.seq_len
    total_steps = max(1, total_tokens // tokens_per_step)
    warmup_steps = max(1, int(total_steps * 0.02))

    # 2. Model Initialization
    train_cfg = job.train_config
    cfg = ModelConfig(
        vocab_size=proc.n_obs,
        n_ctx=proc.seq_len,
        n_layers=train_cfg.n_layers,
        n_heads=train_cfg.n_heads,
        d_model=train_cfg.d_model,
        d_mlp=train_cfg.d_mlp,
        seed=job.seed,
    )
    trained_model = None
    try:
        trained_model = LookaheadTransformer(cfg, k=job.k).to(device)
        optimiser = torch.optim.Adam(trained_model.parameters(), lr=1e-3, weight_decay=0.0)

        def lr_schedule(step: int) -> float:
            if step < warmup_steps:
                return float(step) / float(warmup_steps)
            prog = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * prog))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, lr_schedule)

        # 3. Streaming Training Loop (Uses randomized initial velocities per batch)
        stream_rng = np.random.default_rng(job.seed + 101)
        history = []
        t_start = time.perf_counter()

        for step in range(total_steps):
            trained_model.train()
            optimiser.zero_grad(set_to_none=True)
            loss_accum = 0.0

            for _ in range(grad_accum_steps):
                batch_tokens = proc.sample_batch(stream_rng, micro_batch)["tokens"]
                batch_tensor = torch.as_tensor(batch_tokens, dtype=torch.long, device=device)
                micro_loss = trained_model.loss(batch_tensor) / grad_accum_steps
                micro_loss.backward()
                loss_accum += float(micro_loss.item())

            torch.nn.utils.clip_grad_norm_(trained_model.parameters(), max_norm=1.0)
            optimiser.step()
            scheduler.step()

            log_interval = 1 if smoke else 50
            if step % log_interval == 0 or step == total_steps - 1:
                cur_tokens = (step + 1) * tokens_per_step
                lr_curr = scheduler.get_last_lr()[0]
                history.append({"step": step, "loss": loss_accum, "lr": lr_curr, "tokens_seen": cur_tokens})
                print(
                    f"[{job.job_id}] Step {step+1:4d}/{total_steps} ({(step+1)/total_steps:5.1%}) | "
                    f"Loss: {loss_accum:.4f} | LR: {lr_curr:.2e} | Tokens: {cur_tokens:,}",
                    flush=True,
                )

        # Save final model weights and JSON sidecar
        job.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(trained_model.state_dict(), job.checkpoint_path)
        with open(job.sidecar_path, "w") as f:
            json.dump({
                "job_id": job.job_id,
                "exp_config": asdict(job.exp_config),
                "train_config": asdict(job.train_config),
                "d_model": train_cfg.d_model,
                "d_mlp": train_cfg.d_mlp,
                "k": job.k,
                "k_suffix": job.k_suffix,
                "final_loss": history[-1]["loss"] if history else loss_accum,
                "tokens_trained": total_tokens,
                "runtime_seconds": time.perf_counter() - t_start,
            }, f, indent=2)

        # Package per-model zip archive containing checkpoint and metadata
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in model_dir.rglob("*"):
                if f.is_file():
                    zf.write(f, arcname=f.relative_to(model_dir))

        print(f"[{job.job_id}] Training completed in {time.perf_counter()-t_start:.1f}s. Saved -> {job.checkpoint_path.name} & {zip_path.name}", flush=True)
        return job.job_id, job.checkpoint_path, job.sidecar_path, pd.DataFrame(history)
    finally:
        del trained_model
        if device != "cpu" and torch.cuda.is_available():
            torch.cuda.empty_cache()


def probe_only_worker_gpu(
    job: ModelJob,
    checkpoint_path: Path,
    results_dir: Path,
    device: str = "cuda:0",
    smoke: bool = False,
) -> Tuple[str, pd.DataFrame]:
    """Loads checkpoint, runs 4-way GPU Ridge probing via PyTorch, saves figures, and creates atomic zip."""
    model_dir = results_dir / job.job_id
    model_dir.mkdir(parents=True, exist_ok=True)
    zip_path = results_dir / f"{job.job_id}.zip"
    probe_csv_path = job.probe_csv_path
    probe_json_path = model_dir / f"{job.job_id}_probes.json"

    # Auto-resume check: skip if already completed and packaged
    if zip_path.exists() and probe_csv_path.exists():
        print(f"[{job.job_id}] Existing completed package found -> skipping probing.", flush=True)
        try:
            probe_df = pd.read_csv(probe_csv_path)
            return job.job_id, probe_df
        except Exception as e:
            print(f"[{job.job_id}] Notice: Could not read existing probe CSV ({e}); re-probing.", flush=True)

    target_device = device if (torch.cuda.is_available() and "cuda" in device) else ("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[{job.job_id}] Starting GPU-accelerated probing on {target_device}...", flush=True)
    t_start = time.perf_counter()

    exp_cfg = job.exp_config
    proc = make_process(
        "double_pendulum_mess4",
        m=exp_cfg.m,
        n_steps=exp_cfg.n,
        delta_v=exp_cfg.physics.delta_v,
        system={"gamma1": exp_cfg.physics.gamma1, "gamma2": exp_cfg.physics.gamma2},
    )

    train_cfg = job.train_config
    cfg = ModelConfig(
        vocab_size=proc.n_obs,
        n_ctx=proc.seq_len,
        n_layers=train_cfg.n_layers,
        n_heads=train_cfg.n_heads,
        d_model=train_cfg.d_model,
        d_mlp=train_cfg.d_mlp,
        seed=job.seed,
    )

    trained_model = None
    random_model = None
    try:
        trained_model = LookaheadTransformer(cfg, k=job.k).to(target_device)
        trained_model.load_state_dict(torch.load(checkpoint_path, map_location=target_device, weights_only=True))
        random_model = LookaheadTransformer(cfg, k=job.k).to(target_device)  # untrained matched control

        # 4-way Synchronous GPU Ridge Probing via PyTorch linear algebra
        probe_df, simplex_data = run_synchronous_probing_gpu(
            trained_model=trained_model,
            random_model=random_model,
            proc=proc,
            exp_name=exp_cfg.name,
            training_name=train_cfg.name,
            device=target_device,
            n_eval_trajs=8 if smoke else 512,
            eval_seed=20260829,
        )
        probe_df.to_csv(probe_csv_path, index=False)
        probe_df.to_json(probe_json_path, orient="records", indent=2)

        belief_r2 = probe_df[probe_df["target"] == "belief"]["trained_r2"].max()
        phys_r2 = probe_df[probe_df["target"] == "physics"]["trained_r2"].max()
        gain = probe_df[probe_df["target"] == "belief"]["trained_minus_random"].max()
        print(
            f"[{job.job_id}] GPU Probe Complete ({time.perf_counter()-t_start:.2f}s) -> Max Belief R²: {belief_r2:+.3f} | "
            f"Max Physics R²: {phys_r2:+.3f} | Gain over Random: {gain:+.3f}",
            flush=True,
        )

        # Standardized lookahead probability map
        plot_lookahead_probability_map(trained_model, proc, job.prob_map_path, k_suffix=train_cfg.k_suffix, device=target_device)

        # Standardized belief simplex geometry
        simplex_path = model_dir / f"{job.job_id}_belief_simplex.png"
        if simplex_data is not None:
            plot_belief_simplex_projections(*simplex_data, save_path=simplex_path, title=f"Probed Belief Simplex Geometry (k={job.k}) — {job.job_id}")

        # Instant atomic per-model zip packaging
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in model_dir.rglob("*"):
                if f.is_file():
                    zf.write(f, arcname=f.relative_to(model_dir))

        print(f"[{job.job_id}] Probing finished and packaged -> {zip_path.name}", flush=True)
        return job.job_id, probe_df
    finally:
        del trained_model
        del random_model
        if target_device != "cpu" and torch.cuda.is_available():
            torch.cuda.empty_cache()


def train_and_probe_worker(
    job: ModelJob,
    total_tokens: int,
    batch_size: int,
    device: str,
    results_dir: Path,
    smoke: bool = False,
    micro_batch_size: Optional[int] = None,
) -> Tuple[str, pd.DataFrame]:
    """Integrated worker: trains model and immediately executes ultra-fast GPU probing on the same device."""
    _, ckpt_path, _, _ = train_only_worker(
        job=job,
        total_tokens=total_tokens,
        batch_size=batch_size,
        device=device,
        results_dir=results_dir,
        smoke=smoke,
        micro_batch_size=micro_batch_size,
    )
    return probe_only_worker_gpu(
        job=job,
        checkpoint_path=ckpt_path,
        results_dir=results_dir,
        device=device,
        smoke=smoke,
    )


# ==============================================================================
# 7. Phase 3: Emergence Synthesis & 5-Panel Dashboard
# ==============================================================================

def generate_synthesis_dashboard(
    probe_df_all: pd.DataFrame,
    out_dir: Path,
    configs: Optional[List[ExperimentConfig]] = None,
    device: str = "cpu",
) -> None:
    """Generates the high-resolution synthesis figures and master zip package."""
    print("\n" + "=" * 78)
    print(" PHASE 3: EMERGENCE SYNTHESIS & 5-PANEL DASHBOARD")
    print("=" * 78)
    out_dir.mkdir(parents=True, exist_ok=True)

    probe_df_all["d_model"] = 128
    probe_df_all["lookahead"] = probe_df_all["training_name"].apply(lambda s: s.split("_k")[-1])

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

    # 2. lookahead_comparison.png & lookahead_heatmaps.png -> Row 0: 3 Heatmaps (k=1, k=n//2, k=n)
    if configs:
        for idx, exp_cfg in enumerate(configs):
            proc_temp = make_process(
                "double_pendulum_mess4",
                m=exp_cfg.m,
                n_steps=exp_cfg.n,
                delta_v=exp_cfg.physics.delta_v,
                system={"gamma1": exp_cfg.physics.gamma1, "gamma2": exp_cfg.physics.gamma2},
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

    # 4. physics_vs_belief_scatter.png
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

    print(f"5-Panel Synthesis Dashboard generated in {out_dir}:")
    for f in ["scaling_curves.png", "lookahead_comparison.png", "lookahead_heatmaps.png", "probe_views_breakdown.png", "physics_vs_belief_scatter.png", "trained_vs_random_gain.png"]:
        if (out_dir / f).exists():
            print(f"  -> {out_dir / f}")


# ==============================================================================
# 8. Kaggle Auto-Resume Sync
# ==============================================================================

def sync_from_kaggle_input(results_dir: Path) -> None:
    """Scans /kaggle/input for prior run zips/folders and copies completed artifacts."""
    input_root = Path("/kaggle/input")
    if not input_root.exists():
        return

    print("Scanning /kaggle/input for prior run artifacts to auto-resume...")
    for prior_zip in input_root.rglob("*.zip"):
        if prior_zip.name == "double_pendulum_results.zip":
            continue
        target = results_dir / prior_zip.name
        if not target.exists():
            print(f"Syncing prior model zip: {prior_zip.name} -> {target}", flush=True)
            shutil.copy2(prior_zip, target)
            m_dir = results_dir / prior_zip.stem
            if not m_dir.exists():
                try:
                    with zipfile.ZipFile(target, "r") as zf:
                        zf.extractall(m_dir)
                except Exception as e:
                    print(f"Notice: Could not extract {target.name}: {e}", flush=True)

    for prior_dir in input_root.rglob("results"):
        if prior_dir.is_dir() and prior_dir != results_dir:
            for item in prior_dir.iterdir():
                dest = results_dir / item.name
                if not dest.exists():
                    print(f"Syncing prior result item: {item.name} -> {dest}", flush=True)
                    if item.is_dir():
                        shutil.copytree(item, dest)
                    else:
                        shutil.copy2(item, dest)


# ==============================================================================
# 9. Main Orchestration Flow
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Double Pendulum Production Training v2 (15 Models, GPU Probing)")
    parser.add_argument("--smoke", action="store_true", help="Run rapid smoke validation (~2 models, 10K tokens)")
    parser.add_argument("--platform", type=str, choices=["rtx5090", "kaggle", "local_cpu"], default=None)
    parser.add_argument("--out-dir", type=str, default=None, help="Root directory for outputs")
    parser.add_argument("--tokens", type=int, default=None, help="Token budget per model (default: 800M, smoke: 10K)")
    parser.add_argument("--batch-size", type=int, default=1024, help="Locked batch size invariant (default: 1024)")
    parser.add_argument("--workers", type=int, default=None, help="Number of concurrent worker processes (default: 5 for rtx5090, min(2, dev_count) for kaggle, 1 for local_cpu)")
    parser.add_argument("--micro-batch", type=int, default=None, help="Micro-batch size for gradient accumulation (default: 64)")
    args = parser.parse_args()

    platform_mode = args.platform or detect_platform_mode()
    smoke = args.smoke or (os.environ.get("BELIEF_SMOKE", "False").lower() in ("true", "1", "yes"))

    if args.out_dir:
        output_base = Path(args.out_dir)
    elif platform_mode == "kaggle":
        output_base = Path("/kaggle/working")
    else:
        output_base = Path("./experiments")

    results_dir = output_base / "results"
    output_base.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # 10k tokens in smoke mode per user request; 800M tokens in production
    tokens_per_model = args.tokens or (10_000 if smoke else 800_000_000)
    batch_size = 1024  # strictly locked invariant

    # Worker pool and hardware mapping
    if args.workers is not None:
        num_workers = args.workers
        worker_devices = ["cuda:0" if torch.cuda.is_available() else "cpu"] * num_workers
    elif platform_mode == "rtx5090":
        default_workers = 5
        num_workers = int(os.environ.get("BELIEF_NUM_WORKERS", str(default_workers)))
        worker_devices = ["cuda:0"] * num_workers
    elif platform_mode == "kaggle":
        dev_count = torch.cuda.device_count()
        default_workers = max(1, min(2, dev_count))
        num_workers = int(os.environ.get("BELIEF_NUM_WORKERS", str(default_workers)))
        worker_devices = [f"cuda:{i}" for i in range(num_workers)] if dev_count > 0 else ["cpu"] * num_workers
    else:  # local_cpu
        num_workers = 1
        worker_devices = ["cuda:0"] if torch.cuda.is_available() else ["cpu"]

    print("=" * 78)
    print(" DOUBLE PENDULUM PRODUCTION TRAINING ORCHESTRATOR (v2: 15 Models)")
    print("=" * 78)
    print(f"Platform Mode:    {platform_mode.upper()}")
    print(f"Smoke Mode:       {smoke}")
    print(f"Workers:          {num_workers} worker(s) on {worker_devices}")
    print(f"Batch Size:       {batch_size} (strictly locked invariant)")
    print(f"Token Budget:     {tokens_per_model:,} tokens/model")
    print(f"Execution Mode:   Pure Model Training (Probing launch decoupled)")
    print(f"Output Base:      {output_base}")
    print(f"Results Dir:      {results_dir}")
    print("=" * 78)

    # Phase 1: Physics Screening & 5-Tuple Distillation
    chosen_configs = run_phase1_distillation(results_dir, smoke=smoke)

    # Phase 2: Build the Production Queue (15 Models Total: 5 tuples x 3 lookaheads)
    w = 128
    d_mlp = 4 * w

    jobs: List[ModelJob] = []
    for exp_cfg in chosen_configs:
        n = exp_cfg.n
        lookahead_specs = [
            (1, "_k1"),
            (max(1, n // 2), "_k_mid"),
            (n, "_k_full"),
        ]
        for k_val, k_suf in lookahead_specs:
            train_name = f"dmodel_{w}_dmlp_{d_mlp}{k_suf}"
            train_cfg = TrainingConfig(
                name=train_name,
                d_model=w,
                d_mlp=d_mlp,
                n_layers=4,
                n_heads=2,
                k=k_val,
                k_suffix=k_suf,
            )
            j_id = f"{exp_cfg.name}_{train_name}"
            job = ModelJob(
                job_id=j_id,
                exp_config=exp_cfg,
                train_config=train_cfg,
                k=k_val,
                k_suffix=k_suf,
                tokens_seen=0,
                target_tokens=tokens_per_model,
                checkpoint_path=results_dir / j_id / f"{j_id}_final.pt",
                sidecar_path=results_dir / j_id / f"{j_id}_final.json",
                probe_csv_path=results_dir / j_id / f"{j_id}_probes.csv",
                prob_map_path=results_dir / j_id / f"{j_id}_lookahead_prob_map.png",
                status="QUEUED",
                seed=20260925,
            )
            jobs.append(job)

    if smoke:
        jobs = jobs[:2]

    print("\n" + "=" * 78)
    print(f" PHASE 2: PRODUCTION MODEL QUEUE ({len(jobs)} TOTAL MODELS)")
    print("=" * 78)
    print(f"Queue Size:       {len(jobs)} models across {len(chosen_configs)} tuples, 1 width ({w}), 3 lookaheads, 1 model/config.")

    manifest_path = output_base / "pipeline_state.json"
    master_csv_path = output_base / "master_probe_results.csv"

    manifest_data = {
        "platform_mode": platform_mode,
        "completed_models": [],
        "job_statuses": {},
        "total_models": len(jobs),
        "physics_tuples": [cfg.name for cfg in chosen_configs],
        "total_tokens_per_model": tokens_per_model,
        "timestamp": time.time(),
    }

    all_probe_dfs: List[pd.DataFrame] = []

    if platform_mode == "kaggle":
        sync_from_kaggle_input(results_dir)

    # Check already completed models from disk (Auto-Resume)
    if master_csv_path.exists():
        try:
            existing_master_df = pd.read_csv(master_csv_path)
            all_probe_dfs.append(existing_master_df)
            completed_set = set(existing_master_df["model_id"].unique())
            print(f"Auto-Resume: Loaded {len(completed_set)} previously completed models from {master_csv_path.name}.")
        except Exception as e:
            print(f"Notice: Could not parse existing master CSV ({e}); starting fresh.")
            completed_set = set()
    else:
        completed_set = set()

    for job in jobs:
        zip_path = results_dir / f"{job.job_id}.zip"
        if job.job_id in completed_set or (job.checkpoint_path.exists() and job.sidecar_path.exists()) or (zip_path.exists() and job.probe_csv_path.exists()):
            job.status = "COMPLETED"
            if job.job_id not in manifest_data["completed_models"]:
                manifest_data["completed_models"].append(job.job_id)
            if job.probe_csv_path.exists() and (not completed_set or job.job_id not in completed_set):
                try:
                    all_probe_dfs.append(pd.read_csv(job.probe_csv_path))
                except Exception:
                    pass
        manifest_data["job_statuses"][job.job_id] = job.status

    with open(manifest_path, "w") as f:
        json.dump(manifest_data, f, indent=2)

    # Multi-worker dispatch: each worker executes training only
    if num_workers > 1:
        print(f"\nDispatching {len(jobs)} training jobs across {num_workers} concurrent GPU workers ({worker_devices})...")
        ctx = torch.multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as executor:
            active_futs: Dict[concurrent.futures.Future, ModelJob] = {}
            for i, job in enumerate(jobs):
                if job.status == "COMPLETED":
                    continue
                dev = worker_devices[i % len(worker_devices)]
                job.status = "TRAINING"
                manifest_data["job_statuses"][job.job_id] = job.status
                fut = executor.submit(
                    train_only_worker,
                    job=job,
                    total_tokens=tokens_per_model,
                    batch_size=batch_size,
                    device=dev,
                    results_dir=results_dir,
                    smoke=smoke,
                    micro_batch_size=args.micro_batch,
                )
                active_futs[fut] = job

            manifest_data["timestamp"] = time.time()
            with open(manifest_path, "w") as f:
                json.dump(manifest_data, f, indent=2)

            for fut in concurrent.futures.as_completed(active_futs):
                job = active_futs[fut]
                job_id, ckpt_path, sidecar_path, history_df = fut.result()
                job.status = "COMPLETED"
                if job.job_id not in manifest_data["completed_models"]:
                    manifest_data["completed_models"].append(job.job_id)
                manifest_data["job_statuses"][job.job_id] = job.status
                manifest_data["timestamp"] = time.time()
                with open(manifest_path, "w") as f:
                    json.dump(manifest_data, f, indent=2)
                print(f"-> Progress saved: {len(manifest_data['completed_models'])}/{len(jobs)} models trained ({job.job_id})", flush=True)

    else:
        # Sequential path (single GPU or local_cpu)
        dev = worker_devices[0]
        for job in jobs:
            if job.status == "COMPLETED":
                continue
            job.status = "TRAINING"
            manifest_data["job_statuses"][job.job_id] = job.status
            manifest_data["timestamp"] = time.time()
            with open(manifest_path, "w") as f:
                json.dump(manifest_data, f, indent=2)

            train_only_worker(
                job=job,
                total_tokens=tokens_per_model,
                batch_size=batch_size,
                device=dev,
                results_dir=results_dir,
                smoke=smoke,
                micro_batch_size=args.micro_batch,
            )

            job.status = "COMPLETED"
            if job.job_id not in manifest_data["completed_models"]:
                manifest_data["completed_models"].append(job.job_id)
            manifest_data["job_statuses"][job.job_id] = job.status
            manifest_data["timestamp"] = time.time()
            with open(manifest_path, "w") as f:
                json.dump(manifest_data, f, indent=2)
            print(f"-> Progress saved: {len(manifest_data['completed_models'])}/{len(jobs)} models trained ({job.job_id})", flush=True)

    print(f"\nState Manifest finalized -> {manifest_path}")

    # Phase 3: Emergence Synthesis Dashboard & Distribution Archive
    if all_probe_dfs:
        master_probe_df = pd.concat(all_probe_dfs, ignore_index=True).drop_duplicates()
        master_probe_df.to_csv(master_csv_path, index=False)
        print(f"Master probe evaluations consolidated -> {master_csv_path} ({len(master_probe_df)} rows)")
        generate_synthesis_dashboard(master_probe_df, results_dir, configs=chosen_configs, device=worker_devices[0])
    else:
        print(f"Notice: No probe evaluations generated.")

    master_zip_path = output_base / "double_pendulum_results.zip"
    print(f"\nPackaging master distribution archive -> {master_zip_path}...")
    with zipfile.ZipFile(master_zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(manifest_path, arcname=manifest_path.name)
        if master_csv_path.exists():
            zf.write(master_csv_path, arcname=master_csv_path.name)
        for f in results_dir.rglob("*"):
            if f.is_file() and f.suffix != ".zip":
                zf.write(f, arcname=f"results/{f.relative_to(results_dir)}")

    print(f"Master distribution package complete: {master_zip_path} ({master_zip_path.stat().st_size / 1024:.1f} KB)")
    print("\n" + "=" * 78)
    print("PRODUCTION PIPELINE EXECUTION COMPLETE!")
    print("=" * 78)


if __name__ == "__main__":
    main()
