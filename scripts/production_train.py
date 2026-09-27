#!/usr/bin/env python3
"""Double Pendulum Production Training & Multi-Platform Queue Orchestrator (270 Models).

Authoritative implementation conforming strictly to prompts/kaggle_train.md:
- Multi-Platform Modes: 'rtx5090' (5 workers on cuda:0), 'kaggle' (2 workers on cuda:0/1), 'local_cpu' (1 worker)
- Invariant m * n = 500 tokens across all configurations
- Phase 1: Physics Screening & 10-Tuple Distillation:
  * Sweep 1A: (dv x n) grid screening (baseline gamma=0.65, dt=0.2)
  * Distillation 1A: Top candidate (dv, n) pairs
  * Energy-balance optimal gamma* calculation via compute_optimal_gamma
  * 9 candidate gammas per pair: k*gamma* and k*gamma* +/- d_gamma (k in [0.03, 0.07, 0.10], d_gamma=0.05)
  * Sweep 1B: Damping sweep on chosen pairs
  * Distillation 1B: Top 10 ExperimentConfig physics tuples
  * Level 1 export: distilled_physics_tuples.csv & .json
- Phase 2: 270-Model Production Queue:
  * 10 physics tuples x 3 widths (32, 64, 128) x 3 lookaheads (_k1, _k_mid, _k_full) x 3 seeds (0, 1, 2)
  * Locked BATCH_SIZE = 1024 with micro-batching and gradient accumulation
  * 400M tokens/model (781 steps)
  * Real-time streaming step loss logs every 50 steps
  * Synchronous in-worker 4-way linear Ridge probing against matched random_init controls (Cycle 0 excluded for last-token probe)
  * Headline probe verdicts printed immediately upon completion
  * Standardized lookahead probability map (_lookahead_prob_map.png)
  * Instant atomic per-model packaging ({model_id}.zip)
  * Incremental central persistence of pipeline_state.json and master_probe_results.csv
  * Dual-path auto-resume & recovery
- Phase 3: Emergence Synthesis & 5-Panel Dashboard:
  * scaling_curves.png, lookahead_comparison.png, probe_views_breakdown.png, physics_vs_belief_scatter.png, trained_vs_random_gain.png
  * Master distribution package: double_pendulum_results.zip
- Phase 4: Re-entrant Continuation Framework (dormant by default, requeue_model / requeue_top_models).
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

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

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
    d_model: int
    d_mlp: int
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
    seed: int
    tokens_seen: int
    target_tokens: int
    checkpoint_path: Path
    sidecar_path: Path
    probe_csv_path: Path
    prob_map_path: Path
    status: str


# ==============================================================================
# 4. Phase 1: Physics Screening & 10-Tuple Distillation
# ==============================================================================

def run_phase1_distillation(
    out_dir: Path,
    chosen_pair_ids: Optional[List[int]] = None,
    interactive_configs: Optional[List[ExperimentConfig]] = None,
    smoke: bool = False,
) -> List[ExperimentConfig]:
    """Runs Sweep 1A, optimal gamma* calculation, Sweep 1B, and exports Level 1 artifacts."""
    print("\n" + "=" * 78)
    print(" PHASE 1: PHYSICS SCREENING & 10-TUPLE DISTILLATION")
    print("=" * 78)

    if interactive_configs is not None and len(interactive_configs) > 0:
        print(f"Using {len(interactive_configs)} user-specified interactive ExperimentConfig objects.")
        return interactive_configs

    if smoke:
        dv_candidates = [0.9, 1.2]
        n_candidates = [10, 20]
    else:
        dv_candidates = [0.6, 0.9, 1.2, 1.5, 2.0]
        n_candidates = [5, 10, 20, 25]

    # --- Sweep 1A: (dv x n) Grid Screening ---
    print(f"\nSweep 1A: Screening {len(dv_candidates) * len(n_candidates)} (delta_v x n) pairs at baseline gamma=0.65 (dt=0.2)...")
    records_1a = []
    pair_id = 0
    for dv in dv_candidates:
        for n in n_candidates:
            m = 500 // n
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
            # Composite quality score: high bayes gap + negative/contractive lyapunov - penalty on chaos/clipping/under-coverage
            score = (
                float(stab["bayes_gap"])
                - 0.5 * min(0.0, float(stab["lyapunov"]))
                - 10.0 * max(0.0, float(stab["lyapunov"]))
                - 10.0 * float(stab["clipped"])
                - 1.0 * max(0.0, 0.015 - used_bins_frac)
            )
            is_go = bool(
                float(stab["lyapunov"]) < 0.0
                and float(stab["clipped"]) < 0.01
                and used_bins >= 15
                and float(stab["bayes_gap"]) >= 0.05
                and gap_free > 1e-3
            )
            records_1a.append({
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
            pair_id += 1

    df_1a = pd.DataFrame(records_1a).sort_values("score", ascending=False).reset_index(drop=True)
    print(f"Sweep 1A complete. Ranked {len(df_1a)} candidate pairs.")

    # --- Distillation Checkpoint 1A: Select Pairs for Gamma Sweeps ---
    n_top = min(len(df_1a), 2 if smoke else 10)
    if chosen_pair_ids is not None and len(chosen_pair_ids) > 0:
        selected_pairs_df = df_1a[df_1a["pair_id"].isin(chosen_pair_ids)].copy()
        if len(selected_pairs_df) == 0:
            print("Warning: chosen_pair_ids matched 0 pairs; falling back to top ranked pairs.")
            selected_pairs_df = df_1a.head(n_top).copy()
    else:
        # Gate on is_go == True before taking head(n_top)
        passing_1a = df_1a[df_1a["is_go"] == True]
        if len(passing_1a) >= n_top:
            selected_pairs_df = passing_1a.head(n_top).copy()
        elif len(passing_1a) > 0:
            print(f"Warning: Only {len(passing_1a)} pairs passed is_go in Sweep 1A (target: {n_top}). Taking what is available.")
            selected_pairs_df = passing_1a.copy()
        else:
            print(f"Warning: 0 pairs passed is_go in Sweep 1A; falling back to non-chaotic pairs by composite score.")
            candidate_pool_1a = df_1a[df_1a["lyapunov"] < 0.0]
            selected_pairs_df = (candidate_pool_1a if len(candidate_pool_1a) > 0 else df_1a).head(n_top).copy()

    print(f"Distillation Checkpoint 1A: Selected {len(selected_pairs_df)} candidate pairs for optimal damping calculation.")

    # --- Intermediate Step: Energy-Balance Optimal Gamma* and 9 Gammas per Pair ---
    sweep_1b_records = []
    print("\nIntermediate Step: Steady-State Energy Balance (gamma*) & 9-Point Damping Generation...")
    for _, row in selected_pairs_df.iterrows():
        p_id = int(row["pair_id"])
        dv_val = float(row["delta_v"])
        n_val = int(row["n_steps"])
        m_val = int(row["m"])

        opt_res = compute_optimal_gamma(
            system_name="double_pendulum_mess4",
            m=m_val, n_steps=n_val, dt=0.2,
            delta_v=dv_val, stay=0.7, alpha=0.7,
            damping_field="gamma1",
            n_trajs=16 if smoke else 32,
        )
        gamma_star = float(opt_res["gamma_opt"])

        # 9 candidate gammas: k*gamma* and k*gamma* +/- 0.05 for k in {0.03, 0.07, 0.10}
        cand_gammas = []
        for k_factor in [0.03, 0.07, 0.10]:
            base_g = k_factor * gamma_star
            for offset in [0.0, -0.05, 0.05]:
                val = max(0.05, round(base_g + offset, 4))
                if val not in cand_gammas:
                    cand_gammas.append(val)
        cand_gammas = sorted(cand_gammas)

        # --- Sweep 1B: Damping Screening on Chosen Pair ---
        for g_val in cand_gammas:
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
                - 10.0 * max(0.0, float(stab["lyapunov"]))
                - 10.0 * float(stab["clipped"])
                - 1.0 * max(0.0, 0.015 - used_bins_frac)
            )
            is_go_1b = bool(
                float(stab["lyapunov"]) < 0.0
                and float(stab["clipped"]) < 0.01
                and used_bins >= 15
                and float(stab["bayes_gap"]) >= 0.05
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

    # --- Distillation Checkpoint 2: Select Top 10 Physics Tuples ---
    n_tuples_target = min(len(df_1b), 2 if smoke else 10)
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

    # Add per-pair_id cap (2 gammas max per pair) when selecting top-10 tuples for diversity
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

    # Backfill remaining slots globally after cap applied
    for idx in remaining_indices:
        if len(selected_indices) >= n_tuples_target:
            break
        selected_indices.append(idx)

    top_tuples_df = candidate_pool.loc[selected_indices].copy().reset_index(drop=True)

    chosen_configs: List[ExperimentConfig] = []
    for idx, row in top_tuples_df.iterrows():
        tuple_name = f"dp_dv{row['delta_v']}_g{row['gamma']}_n{int(row['n_steps'])}"
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
            seed=20260925 + idx,
        )
        chosen_configs.append(cfg)

    # Export Level 1 artifacts
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "distilled_physics_tuples.csv"
    json_path = out_dir / "distilled_physics_tuples.json"
    top_tuples_df.to_csv(csv_path, index=False)

    exported_json_records = []
    for cfg in chosen_configs:
        exported_json_records.append({
            "name": cfg.name,
            "delta_v": cfg.physics.delta_v,
            "gamma1": cfg.physics.gamma1,
            "gamma2": cfg.physics.gamma2,
            "dt": cfg.physics.dt,
            "n_steps": cfg.n,
            "m": cfg.m,
            "seq_len": cfg.m * cfg.n,
        })
    with open(json_path, "w") as f:
        json.dump(exported_json_records, f, indent=2)

    print(f"\nDistillation 1B: Exported {len(chosen_configs)} Level 1 Distilled Physics Tuples:")
    print(f"  -> {csv_path}")
    print(f"  -> {json_path}")
    for i, cfg in enumerate(chosen_configs):
        print(f"  Tuple {i+1:2d}: {cfg.name} (m={cfg.m}, n={cfg.n}, seq_len={cfg.m*cfg.n})")

    return chosen_configs


# ==============================================================================
# 5. Probing Protocol & Probability Map Generation
# ==============================================================================

def run_synchronous_probing(
    trained_model: LookaheadTransformer,
    random_model: LookaheadTransformer,
    proc: Any,
    exp_name: str,
    training_name: str,
    seed: int,
    device: str,
    n_eval_trajs: int = 512,
) -> pd.DataFrame:
    """Grouped 5-Fold Cross-Validated Ridge Probing across 4 modes x 2 targets."""
    model_id = f"{exp_name}_{training_name}_seed{seed}"
    trained_model.eval()
    random_model.eval()

    eval_rng = np.random.default_rng(20260829 + seed)
    eval_batch = proc.sample_batch(eval_rng, n_eval_trajs)
    tokens = torch.as_tensor(eval_batch["tokens"], dtype=torch.long, device=device)

    def extract_streams(model: LookaheadTransformer) -> List[np.ndarray]:
        streams = []
        with torch.no_grad():
            x = model._embed(tokens)
            for block in model.blocks:
                x = block(x)
                streams.append(x.cpu().numpy())
        return streams

    trained_streams = extract_streams(trained_model)
    random_streams = extract_streams(random_model)

    N, seq_len = tokens.shape
    n = proc.n_steps
    m = proc.m
    assert seq_len == m * n == 500

    embedded_beliefs = eval_batch["beliefs"] @ simplex_embedding(proc.chain.n_states)
    physics_metric = eval_batch["metric"]  # (N, seq_len, 2)

    # Boundary token indices: c*n - 1
    cycle_indices = np.arange(n - 1, seq_len, n)
    # Strictly exclude Cycle 0 for single_layer_last_token probe (from Cycle 1 onwards: 2n-1, 3n-1, ...)
    post_sync_indices = np.arange(2 * n - 1, seq_len, n)

    probe_modes = [
        "all_layers_single_token",
        "single_layer_cycle",
        "single_layer_last_token",
        "single_layer_all_token",
    ]
    target_domains = ["belief", "physics"]

    results = []

    def get_features(streams: List[np.ndarray], mode: str) -> np.ndarray:
        if mode == "all_layers_single_token":
            # Concat all layers at cycle boundaries
            stacked = np.stack([s[:, cycle_indices] for s in streams], axis=-1)
            return stacked.reshape(N * m, -1)
        elif mode == "single_layer_cycle":
            # Last layer across all n tokens per cycle
            last_layer = streams[-1].reshape(N, m, n, -1)
            return last_layer.reshape(N * m, -1)
        elif mode == "single_layer_last_token":
            # Last layer at cycle boundaries EXCLUDING Cycle 0
            return streams[-1][:, post_sync_indices].reshape(N * (m - 1), -1)
        elif mode == "single_layer_all_token":
            # Last layer across all individual sequence tokens
            return streams[-1].reshape(N * seq_len, -1)
        raise ValueError(f"Unknown mode: {mode}")

    def get_target(domain: str, mode: str) -> Tuple[np.ndarray, np.ndarray]:
        data = embedded_beliefs if domain == "belief" else physics_metric
        if mode in ("all_layers_single_token", "single_layer_cycle"):
            y = data[:, cycle_indices].reshape(N * m, -1)
            groups = np.repeat(np.arange(N), m)
        elif mode == "single_layer_last_token":
            y = data[:, post_sync_indices].reshape(N * (m - 1), -1)
            groups = np.repeat(np.arange(N), m - 1)
        elif mode == "single_layer_all_token":
            y = data.reshape(N * seq_len, -1)
            groups = np.repeat(np.arange(N), seq_len)
        return y, groups

    simplex_payload = None

    for mode in probe_modes:
        X_tr = get_features(trained_streams, mode)
        X_rnd = get_features(random_streams, mode)

        for domain in target_domains:
            y, groups = get_target(domain, mode)

            pipe = make_pipeline(StandardScaler(), Ridge(solver="lsqr"))
            cv = GroupKFold(n_splits=5)
            param_grid = {"ridge__alpha": [0.1, 1.0, 10.0, 50.0, 100.0]}

            grid_tr = GridSearchCV(pipe, param_grid, cv=cv, scoring="r2", n_jobs=1)
            grid_tr.fit(X_tr, y, groups=groups)
            r2_trained = float(grid_tr.best_score_)
            best_alpha = float(grid_tr.best_params_["ridge__alpha"])

            if domain == "belief" and mode == "single_layer_last_token":
                pred_y = grid_tr.best_estimator_.predict(X_tr)
                mood_labels = eval_batch["moods"][:, post_sync_indices].reshape(-1)
                simplex_payload = (y, pred_y, mood_labels)

            grid_rnd = GridSearchCV(pipe, param_grid, cv=cv, scoring="r2", n_jobs=1)
            grid_rnd.fit(X_rnd, y, groups=groups)
            r2_random = float(grid_rnd.best_score_)

            results.append({
                "model_id": model_id,
                "exp_name": exp_name,
                "training_name": training_name,
                "seed": seed,
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
    device: str,
    fixed_eval_seed: int = 2026,
) -> None:
    """Generates standardized lookahead probability map on fixed held-out sequence."""
    model.eval()
    k = model.k
    rng = np.random.default_rng(fixed_eval_seed)
    seq = proc.sample_batch(rng, 1)["tokens"][0]  # shape (seq_len,)
    tokens_tensor = torch.as_tensor(seq[np.newaxis, :], dtype=torch.long, device=device)

    with torch.no_grad():
        logits = model(tokens_tensor)[0]  # (seq_len, vocab_size)
        probs = F.softmax(logits, dim=-1).cpu().numpy()

    source_len = len(seq) - k
    actual = seq[k:]

    fig, ax = plt.subplots(figsize=(12, 5), dpi=150)
    im = ax.imshow(probs[:source_len].T, origin="lower", aspect="auto", cmap="magma")
    ax.plot(np.arange(source_len), actual, color="cyan", lw=1.2, label=f"Actual token (t+{k})")
    ax.set_xlabel("Source Position t", fontsize=10)
    ax.set_ylabel(f"Vocabulary Token ID (0 to {proc.n_obs-1})", fontsize=10)
    ax.set_title(f"Standardized Lookahead Probability Map (k={k})", fontsize=11, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    plt.colorbar(im, ax=ax, label="Softmax Probability")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ==============================================================================
# 6. Worker Execution Engine (Phase 2 Training & In-Worker Probing)
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

    # Micro-batching with gradient accumulation (default: 64 to prevent VRAM spikes/OOM)
    default_mb = 16 if (smoke or device == "cpu") else int(os.environ.get("BELIEF_MICRO_BATCH", "64"))
    chosen_mb = micro_batch_size or default_mb
    micro_batch = min(chosen_mb, batch_size)
    grad_accum_steps = max(1, batch_size // micro_batch)
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

        # 3. Streaming Training Loop
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

            torch.nn.utils.clip_grad_norm_(trained_model.parameters(), 1.0)
            optimiser.step()
            scheduler.step()

            if step % (5 if smoke else 50) == 0 or step == total_steps - 1:
                tokens_seen = (step + 1) * tokens_per_step
                history.append({
                    "model_id": job.job_id,
                    "step": step,
                    "tokens_seen": tokens_seen,
                    "loss": loss_accum,
                })
                lr_now = scheduler.get_last_lr()[0]
                pct = 100.0 * (step + 1) / total_steps
                print(
                    f"[{job.job_id}] Step {step+1:4d}/{total_steps} ({pct:5.1f}%) | "
                    f"Tokens: {tokens_seen/1e6:6.1f}M / {total_tokens/1e6:5.1f}M | "
                    f"Loss: {loss_accum:6.4f} | LR: {lr_now:.2e}",
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
                "seed": job.seed,
                "final_loss": history[-1]["loss"] if history else loss_accum,
                "tokens_trained": total_tokens,
                "runtime_seconds": time.perf_counter() - t_start,
            }, f, indent=2)

        print(f"[{job.job_id}] Training completed in {time.perf_counter()-t_start:.1f}s. Saved -> {job.checkpoint_path.name}", flush=True)
        return job.job_id, job.checkpoint_path, job.sidecar_path, pd.DataFrame(history)
    finally:
        del trained_model
        if device != "cpu" and torch.cuda.is_available():
            torch.cuda.empty_cache()


def probe_only_worker(
    job: ModelJob,
    checkpoint_path: Path,
    results_dir: Path,
    device: str = "cpu",
    smoke: bool = False,
) -> Tuple[str, pd.DataFrame]:
    """Loads checkpoint, runs 4-way Ridge probing, saves probability map, and creates atomic zip."""
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

    print(f"[{job.job_id}] Starting probing on {device}...", flush=True)
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
        trained_model = LookaheadTransformer(cfg, k=job.k).to(device)
        trained_model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
        random_model = LookaheadTransformer(cfg, k=job.k).to(device)  # untrained matched control

        # 4-way Synchronous Ridge Probing
        probe_df, simplex_data = run_synchronous_probing(
            trained_model=trained_model,
            random_model=random_model,
            proc=proc,
            exp_name=exp_cfg.name,
            training_name=train_cfg.name,
            seed=job.seed,
            device=device,
            n_eval_trajs=64 if smoke else 512,
        )
        probe_df.to_csv(probe_csv_path, index=False)
        probe_df.to_json(probe_json_path, orient="records", indent=2)

        belief_r2 = probe_df[probe_df["target"] == "belief"]["trained_r2"].max()
        phys_r2 = probe_df[probe_df["target"] == "physics"]["trained_r2"].max()
        gain = probe_df[probe_df["target"] == "belief"]["trained_minus_random"].max()
        print(
            f"[{job.job_id}] Probe Results -> Max Belief R²: {belief_r2:+.3f} | "
            f"Max Physics R²: {phys_r2:+.3f} | Gain over Random: {gain:+.3f}",
            flush=True,
        )

        # Standardized lookahead probability map
        plot_lookahead_probability_map(trained_model, proc, job.prob_map_path, device=device)

        # Standardized belief simplex geometry in 3D orientations and 2D projections
        simplex_path = model_dir / f"{job.job_id}_belief_simplex.png"
        if simplex_data is not None:
            plot_belief_simplex_projections(*simplex_data, save_path=simplex_path, title=f"Probed Belief Simplex Geometry (k={job.k}) — {job.job_id}")

        # Instant atomic per-model zip packaging
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in model_dir.rglob("*"):
                if f.is_file():
                    zf.write(f, arcname=f.relative_to(model_dir))

        print(f"[{job.job_id}] Probing finished in {time.perf_counter()-t_start:.1f}s. Packaged -> {zip_path.name}", flush=True)
        return job.job_id, probe_df
    finally:
        del trained_model
        del random_model
        if device != "cpu" and torch.cuda.is_available():
            torch.cuda.empty_cache()


def train_and_probe_worker(
    job: ModelJob,
    total_tokens: int,
    batch_size: int,
    device: str,
    results_dir: Path,
    smoke: bool = False,
    micro_batch_size: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Sequential wrapper for single worker or local_cpu execution."""
    _, ckpt_path, _, hist_df = train_only_worker(
        job=job,
        total_tokens=total_tokens,
        batch_size=batch_size,
        device=device,
        results_dir=results_dir,
        smoke=smoke,
        micro_batch_size=micro_batch_size,
    )
    _, probe_df = probe_only_worker(
        job=job,
        checkpoint_path=ckpt_path,
        results_dir=results_dir,
        device="cpu",
        smoke=smoke,
    )
    return hist_df, probe_df


# ==============================================================================
# 7. Phase 3: Emergence Synthesis & 5-Panel Dashboard
# ==============================================================================

def generate_synthesis_dashboard(probe_df_all: pd.DataFrame, out_dir: Path) -> None:
    """Generates the 5 high-resolution synthesis figures and master zip package."""
    print("\n" + "=" * 78)
    print(" PHASE 3: EMERGENCE SYNTHESIS & 5-PANEL DASHBOARD")
    print("=" * 78)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Parse width and lookahead
    probe_df_all["d_model"] = probe_df_all["training_name"].apply(
        lambda s: int(s.split("dmodel_")[1].split("_")[0]) if "dmodel_" in s else 128
    )
    probe_df_all["lookahead"] = probe_df_all["training_name"].apply(lambda s: s.split("_k")[-1])

    # 1. scaling_curves.png
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    for target, color in [("belief", "#1f77b4"), ("physics", "#2ca02c")]:
        sub = probe_df_all[probe_df_all["target"] == target]
        grouped = sub.groupby("d_model")["trained_r2"].mean().reset_index()
        ax.plot(grouped["d_model"], grouped["trained_r2"], marker="o", lw=2, color=color, label=f"Trained {target.capitalize()} R²")
        grouped_rnd = sub.groupby("d_model")["random_r2"].mean().reset_index()
        ax.plot(grouped_rnd["d_model"], grouped_rnd["random_r2"], marker="s", ls="--", color=color, alpha=0.5, label=f"Random {target.capitalize()} R²")
    ax.set(xlabel="Model Width (d_model)", ylabel="Test R² Score", title="1. Model Width Scaling Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(out_dir / "scaling_curves.png", bbox_inches="tight")
    plt.close()

    # 2. lookahead_comparison.png
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    lh_grouped = probe_df_all.groupby(["lookahead", "target"])["trained_r2"].mean().unstack()
    lh_grouped.plot(kind="bar", ax=ax, colormap="tab10", width=0.6)
    ax.set(xlabel="Lookahead Horizon", ylabel="Mean Test R²", title="2. Forecasting Horizon Comparison")
    ax.grid(True, alpha=0.3)
    fig.savefig(out_dir / "lookahead_comparison.png", bbox_inches="tight")
    plt.close()

    # 3. probe_views_breakdown.png
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    pv_grouped = probe_df_all.groupby(["probe_mode", "target"])["trained_r2"].mean().unstack()
    pv_grouped.plot(kind="bar", ax=ax, colormap="Set2", width=0.6)
    ax.set(xlabel="Representation View", ylabel="Mean Test R²", title="3. Probe Views Breakdown")
    ax.set_xticklabels(pv_grouped.index, rotation=20, ha="right")
    ax.grid(True, alpha=0.3)
    fig.savefig(out_dir / "probe_views_breakdown.png", bbox_inches="tight")
    plt.close()

    # 4. physics_vs_belief_scatter.png (Enhanced Decoupling Scatter)
    fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
    p_b = probe_df_all.pivot_table(index=["model_id", "probe_mode"], columns="target", values="trained_r2").reset_index()
    if "belief" in p_b.columns and "physics" in p_b.columns:
        models = p_b["model_id"].unique()
        model_colors = plt.cm.tab20(np.linspace(0, 1, max(1, len(models))))
        color_map = {m: model_colors[i] for i, m in enumerate(models)}
        marker_map = {
            "all_layers_single_token": "o",
            "single_layer_cycle": "s",
            "single_layer_last_token": "^",
            "single_layer_all_token": "D",
        }
        for _, row in p_b.iterrows():
            m_id = row["model_id"]
            mode = row["probe_mode"]
            color = color_map.get(m_id, "#1f77b4")
            marker = marker_map.get(mode, "o")
            ax.scatter(row["physics"], row["belief"], color=color, marker=marker, s=80, alpha=0.85, edgecolors="k", linewidths=0.5)

        ax.axhline(0.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.axvline(0.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.set(xlabel="Physics Test R² (Angular Velocities)", ylabel="Belief Test R² (Simplex Coordinates)",
               title="4. Representation Decoupling: Belief vs. Physics Readability")
        ax.grid(True, alpha=0.3)
        fig.savefig(out_dir / "physics_vs_belief_scatter.png", bbox_inches="tight")
        plt.close()

    # 5. trained_vs_random_gain.png
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    gain_grouped = probe_df_all.groupby(["probe_mode", "target"])["trained_minus_random"].mean().unstack()
    gain_grouped.plot(kind="bar", ax=ax, colormap="coolwarm", width=0.6)
    ax.set(xlabel="Representation View", ylabel="Mean R² Gain (Trained - Random)", title="5. Net Representation Emergence Gain")
    ax.set_xticklabels(gain_grouped.index, rotation=20, ha="right")
    ax.grid(True, alpha=0.3)
    fig.savefig(out_dir / "trained_vs_random_gain.png", bbox_inches="tight")
    plt.close()

    print(f"5-Panel Synthesis Dashboard generated in {out_dir}:")
    for f in ["scaling_curves.png", "lookahead_comparison.png", "probe_views_breakdown.png", "physics_vs_belief_scatter.png", "trained_vs_random_gain.png"]:
        print(f"  -> {out_dir / f}")


# ==============================================================================
# 8. Phase 4: Re-entrant Continuation Interface
# ==============================================================================

REQUEUE_JOBS: List[Dict[str, Any]] = []

def requeue_model(job_id: str, additional_tokens: int = 500_000_000) -> None:
    """Enqueues a specific completed model for further extended training."""
    REQUEUE_JOBS.append({"job_id": job_id, "additional_tokens": additional_tokens})
    print(f"Enqueued {job_id} for +{additional_tokens:,} additional tokens.")


def requeue_top_models(
    master_probe_df: pd.DataFrame,
    metric: str = "trained_r2",
    target: str = "belief",
    top_n: int = 5,
    additional_tokens: int = 500_000_000,
) -> None:
    """Identifies and enqueues top-N performing models for further training."""
    sub = master_probe_df[master_probe_df["target"] == target]
    top_ids = sub.groupby("model_id")[metric].max().nlargest(top_n).index.tolist()
    for m_id in top_ids:
        requeue_model(m_id, additional_tokens)


# ==============================================================================
# 9. Kaggle Auto-Resume Sync & Main Orchestrator Execution
# ==============================================================================

def sync_from_kaggle_input(results_dir: Path) -> None:
    """Scans /kaggle/input for prior run artifacts and copies them into results_dir."""
    input_root = Path("/kaggle/input")
    if not input_root.exists():
        return
    print(f"Scanning {input_root} for prior run artifacts to auto-resume...", flush=True)
    parent_dir = results_dir.parent
    master_csv_target = parent_dir / "master_probe_results.csv"
    if not master_csv_target.exists():
        for prior_csv in input_root.rglob("master_probe_results.csv"):
            if prior_csv.is_file():
                print(f"Found prior master CSV: {prior_csv} -> copying to {master_csv_target}", flush=True)
                shutil.copy2(prior_csv, master_csv_target)
                break

    manifest_target = parent_dir / "pipeline_state.json"
    if not manifest_target.exists():
        for prior_manifest in input_root.rglob("pipeline_state.json"):
            if prior_manifest.is_file():
                print(f"Found prior pipeline manifest: {prior_manifest} -> copying to {manifest_target}", flush=True)
                shutil.copy2(prior_manifest, manifest_target)
                break

    for prior_zip in input_root.rglob("*.zip"):
        target = results_dir / prior_zip.name
        if not target.exists():
            print(f"Syncing prior archive: {prior_zip.name} -> {target}", flush=True)
            shutil.copy2(prior_zip, target)
        if prior_zip.name != "double_pendulum_results.zip":
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


def main():
    parser = argparse.ArgumentParser(description="Double Pendulum Production Training (270 Models)")
    parser.add_argument("--smoke", action="store_true", help="Run rapid smoke validation (~2 models, 5M tokens)")
    parser.add_argument("--platform", type=str, choices=["rtx5090", "kaggle", "local_cpu"], default=None)
    parser.add_argument("--out-dir", type=str, default=None, help="Root directory for outputs")
    parser.add_argument("--tokens", type=int, default=None, help="Token budget per model (default: 400M, smoke: 5M)")
    parser.add_argument("--batch-size", type=int, default=1024, help="Locked batch size invariant (default: 1024)")
    parser.add_argument("--workers", type=int, default=None, help="Number of concurrent train worker processes (default: 5 for rtx5090, min(2, dev_count) for kaggle, 1 for local_cpu)")
    parser.add_argument("--probe-workers", type=int, default=None, help="Number of concurrent probe worker processes (default: 2 for rtx5090, 1 for kaggle, 0 for local_cpu)")
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

    tokens_per_model = args.tokens or (5_000_000 if smoke else 400_000_000)
    batch_size = 1024  # strictly locked invariant

    # Worker pool and hardware mapping
    if args.workers is not None:
        num_train_workers = args.workers
        worker_devices = ["cuda:0" if torch.cuda.is_available() else "cpu"] * num_train_workers
        default_probe = 2 if platform_mode == "rtx5090" else (1 if platform_mode == "kaggle" else 0)
        num_probe_workers = args.probe_workers if args.probe_workers is not None else int(os.environ.get("BELIEF_NUM_PROBE_WORKERS", str(default_probe)))
    elif platform_mode == "rtx5090":
        default_train_workers = 5
        num_train_workers = int(os.environ.get("BELIEF_NUM_WORKERS", str(default_train_workers)))
        worker_devices = ["cuda:0"] * num_train_workers
        default_probe_workers = 2
        num_probe_workers = args.probe_workers if args.probe_workers is not None else int(os.environ.get("BELIEF_NUM_PROBE_WORKERS", str(default_probe_workers)))
    elif platform_mode == "kaggle":
        dev_count = torch.cuda.device_count()
        default_train_workers = max(1, min(2, dev_count))
        num_train_workers = int(os.environ.get("BELIEF_NUM_WORKERS", str(default_train_workers)))
        worker_devices = [f"cuda:{i}" for i in range(num_train_workers)] if dev_count > 0 else ["cpu"] * num_train_workers
        default_probe_workers = 1
        num_probe_workers = args.probe_workers if args.probe_workers is not None else int(os.environ.get("BELIEF_NUM_PROBE_WORKERS", str(default_probe_workers)))
    else:  # local_cpu
        num_train_workers = 1
        worker_devices = ["cuda:0"] if torch.cuda.is_available() else ["cpu"]
        num_probe_workers = args.probe_workers if args.probe_workers is not None else 0

    print("=" * 78)
    print(" DOUBLE PENDULUM PRODUCTION TRAINING ORCHESTRATOR")
    print("=" * 78)
    print(f"Platform Mode:    {platform_mode.upper()}")
    print(f"Smoke Mode:       {smoke}")
    print(f"Train Workers:    {num_train_workers} worker(s) on {worker_devices}")
    print(f"Probe Workers:    {num_probe_workers} worker(s) (cpu)")
    print(f"Batch Size:       {batch_size} (strictly locked invariant)")
    print(f"Token Budget:     {tokens_per_model:,} tokens/model")
    print(f"Output Base:      {output_base}")
    print(f"Results Dir:      {results_dir}")
    print("=" * 78)

    # --------------------------------------------------------------------------
    # Phase 1: Physics Screening & 10-Tuple Distillation
    # --------------------------------------------------------------------------
    chosen_configs = run_phase1_distillation(results_dir, smoke=smoke)

    # --------------------------------------------------------------------------
    # Phase 2: Build the Production Queue (270 Models)
    # --------------------------------------------------------------------------
    widths = [32, 64] if smoke else [32, 64, 128]
    seeds = [0] if smoke else [0, 1, 2]

    jobs: List[ModelJob] = []
    for exp_cfg in chosen_configs:
        n = exp_cfg.n
        # 3 lookahead horizons per n
        lookahead_specs = [
            (1, "_k1"),
            (max(1, n // 2), "_k_mid"),
            (n, "_k_full"),
        ]
        for w in widths:
            d_mlp = 4 * w
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
                for s in seeds:
                    j_id = f"{exp_cfg.name}_{train_name}_seed{s}"
                    job = ModelJob(
                        job_id=j_id,
                        exp_config=exp_cfg,
                        train_config=train_cfg,
                        k=k_val,
                        k_suffix=k_suf,
                        seed=s,
                        tokens_seen=0,
                        target_tokens=tokens_per_model,
                        checkpoint_path=results_dir / j_id / f"{j_id}_final.pt",
                        sidecar_path=results_dir / j_id / f"{j_id}_final.json",
                        probe_csv_path=results_dir / j_id / f"{j_id}_probes.csv",
                        prob_map_path=results_dir / j_id / f"{j_id}_lookahead_prob_map.png",
                        status="QUEUED",
                    )
                    jobs.append(job)

    if smoke:
        jobs = jobs[:2]

    print("\n" + "=" * 78)
    print(f" PHASE 2: PRODUCTION MODEL QUEUE ({len(jobs)} TOTAL MODELS)")
    print("=" * 78)
    print(f"Queue Size:       {len(jobs)} models across {len(chosen_configs)} tuples, {len(widths)} widths, 3 lookaheads, {len(seeds)} seeds.")

    # Manifest and cumulative CSV paths
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

    # Auto-resume from Kaggle dataset input if running on Kaggle
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
        if job.job_id in completed_set or (zip_path.exists() and job.probe_csv_path.exists()):
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

    # Concurrency Execution Dispatch
    if num_probe_workers > 0:
        print(f"\nDispatching {len(jobs)} jobs across {num_train_workers} train workers ({worker_devices}) and {num_probe_workers} probe workers (cpu)...")
        ctx = torch.multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_train_workers, mp_context=ctx) as train_executor, \
             concurrent.futures.ProcessPoolExecutor(max_workers=num_probe_workers, mp_context=ctx) as probe_executor:

            active_train: Dict[concurrent.futures.Future, Tuple[ModelJob, str]] = {}
            active_probe: Dict[concurrent.futures.Future, ModelJob] = {}

            # Submit training jobs that are not completed
            for i, job in enumerate(jobs):
                if job.status == "COMPLETED":
                    continue
                dev = worker_devices[i % len(worker_devices)]
                job.status = "TRAINING"
                manifest_data["job_statuses"][job.job_id] = job.status
                fut = train_executor.submit(
                    train_only_worker,
                    job=job,
                    total_tokens=tokens_per_model,
                    batch_size=batch_size,
                    device=dev,
                    results_dir=results_dir,
                    smoke=smoke,
                    micro_batch_size=args.micro_batch,
                )
                active_train[fut] = (job, dev)

            manifest_data["timestamp"] = time.time()
            with open(manifest_path, "w") as f:
                json.dump(manifest_data, f, indent=2)

            while active_train or active_probe:
                all_active = list(active_train.keys()) + list(active_probe.keys())
                done, _ = concurrent.futures.wait(all_active, return_when=concurrent.futures.FIRST_COMPLETED)
                for fut in done:
                    if fut in active_train:
                        job, dev = active_train.pop(fut)
                        job_id, ckpt_path, sidecar_path, hist_df = fut.result()
                        # Bridge immediately to probe executor
                        job.status = "PROBING"
                        manifest_data["job_statuses"][job.job_id] = job.status
                        manifest_data["timestamp"] = time.time()
                        with open(manifest_path, "w") as f:
                            json.dump(manifest_data, f, indent=2)

                        p_fut = probe_executor.submit(
                            probe_only_worker,
                            job=job,
                            checkpoint_path=ckpt_path,
                            results_dir=results_dir,
                            device="cpu",
                            smoke=smoke,
                        )
                        active_probe[p_fut] = job

                    elif fut in active_probe:
                        job = active_probe.pop(fut)
                        job_id, probe_df = fut.result()
                        job.status = "COMPLETED"
                        if not probe_df.empty:
                            all_probe_dfs.append(probe_df)
                            cur_master_df = pd.concat(all_probe_dfs, ignore_index=True)
                            cur_master_df.to_csv(master_csv_path, index=False)
                        if job.job_id not in manifest_data["completed_models"]:
                            manifest_data["completed_models"].append(job.job_id)
                        manifest_data["job_statuses"][job.job_id] = job.status
                        manifest_data["timestamp"] = time.time()
                        with open(manifest_path, "w") as f:
                            json.dump(manifest_data, f, indent=2)
                        print(f"-> Progress saved: {len(manifest_data['completed_models'])}/{len(jobs)} models completed in {master_csv_path.name}", flush=True)

    else:
        # Sequential execution path (e.g. local_cpu mode)
        dev = worker_devices[0]
        for job in jobs:
            if job.status == "COMPLETED":
                continue
            job.status = "TRAINING"
            manifest_data["job_statuses"][job.job_id] = job.status
            manifest_data["timestamp"] = time.time()
            with open(manifest_path, "w") as f:
                json.dump(manifest_data, f, indent=2)

            _, ckpt_path, _, hist_df = train_only_worker(
                job=job,
                total_tokens=tokens_per_model,
                batch_size=batch_size,
                device=dev,
                results_dir=results_dir,
                smoke=smoke,
                micro_batch_size=args.micro_batch,
            )

            job.status = "PROBING"
            manifest_data["job_statuses"][job.job_id] = job.status
            manifest_data["timestamp"] = time.time()
            with open(manifest_path, "w") as f:
                json.dump(manifest_data, f, indent=2)

            _, probe_df = probe_only_worker(
                job=job,
                checkpoint_path=ckpt_path,
                results_dir=results_dir,
                device="cpu",
                smoke=smoke,
            )

            job.status = "COMPLETED"
            if not probe_df.empty:
                all_probe_dfs.append(probe_df)
                cur_master_df = pd.concat(all_probe_dfs, ignore_index=True)
                cur_master_df.to_csv(master_csv_path, index=False)
            if job.job_id not in manifest_data["completed_models"]:
                manifest_data["completed_models"].append(job.job_id)
            manifest_data["job_statuses"][job.job_id] = job.status
            manifest_data["timestamp"] = time.time()
            with open(manifest_path, "w") as f:
                json.dump(manifest_data, f, indent=2)
            print(f"-> Progress saved: {len(manifest_data['completed_models'])}/{len(jobs)} models completed in {master_csv_path.name}", flush=True)

    print(f"\nState Manifest finalized -> {manifest_path}")

    # --------------------------------------------------------------------------
    # Phase 3: Emergence Synthesis Dashboard & Distribution Archive
    # --------------------------------------------------------------------------
    if all_probe_dfs:
        master_probe_df = pd.concat(all_probe_dfs, ignore_index=True).drop_duplicates()
        master_probe_df.to_csv(master_csv_path, index=False)
        print(f"Master probe evaluations consolidated -> {master_csv_path} ({len(master_probe_df)} rows)")
        generate_synthesis_dashboard(master_probe_df, results_dir)
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
