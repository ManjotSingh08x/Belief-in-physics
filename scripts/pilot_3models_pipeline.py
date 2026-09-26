#!/usr/bin/env python3
"""End-to-End Pilot Pipeline (Phase 1 Physics Distillation -> Phase 2 3-Model Queue -> Phase 3 Synthesis).

Incorporates all production architectural specifications from prompts/kaggle_train.md:
- Multi-Platform Modes: 'rtx5090' (5 workers on cuda:0), 'kaggle' (2 workers), 'local_cpu' (1 worker)
- Invariant m * n = 500 sequence length
- Phase 1: Sweep 1A (dv x n) -> Optimal gamma* calculation -> 9 gammas -> Sweep 1B -> Distillation 1B
- Phase 2: 3 diverse models (w32_k1, w64_k_mid, w128_k_full) at locked BATCH_SIZE=1024
- In-Worker Synchronous Probing: 4 modes x 2 targets, Cycle 0 excluded for last_token,
  matched random_init baseline, and Lookahead Probability Map
- Phase 3: Consolidated master_probe_results.csv, 5-panel synthesis dashboard, and master double_pendulum_results.zip.
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
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.transformer import ModelConfig, TinyTransformer
from physics.controls import compute_optimal_gamma
from physics.messk import simplex_embedding
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
# 3. Phase 1: Physics Screening & Distillation
# ==============================================================================

@dataclass(frozen=True)
class DistilledPhysicsTuple:
    pair_id: int
    delta_v: float
    gamma: float
    n_steps: int
    m: int
    lyapunov: float
    clipped: float
    used_bins: int
    bayes_gap: float
    score: float


def run_phase1_distillation(out_dir: Path, smoke: bool = False) -> DistilledPhysicsTuple:
    """Executes Sweep 1A -> Optimal Gamma* -> 9 Gammas -> Sweep 1B -> Distillation 1B."""
    print("\n" + "=" * 78)
    print(" PHASE 1: PHYSICS SCREENING & DISTILLATION")
    print("=" * 78)

    if smoke:
        dv_candidates = [0.9, 1.2]
        n_candidates = [10, 20]
    else:
        dv_candidates = [0.6, 0.9, 1.2, 1.5, 2.0]
        n_candidates = [5, 10, 20, 25]

    # --- Sweep 1A: (dv x n) Grid Screening ---
    print(f"Sweep 1A: Screening {len(dv_candidates) * len(n_candidates)} (delta_v x n) pairs at baseline gamma=0.65 (dt=0.2)...")
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
            
            # Composite quality score: high bayes gap + negative/contractive lyapunov - clipping penalty
            score = float(stab["bayes_gap"]) - 0.5 * max(0.0, float(stab["lyapunov"])) - 10.0 * float(stab["clipped"])
            records_1a.append({
                "pair_id": pair_id,
                "delta_v": dv,
                "n_steps": n,
                "m": m,
                "lyapunov": stab["lyapunov"],
                "clipped": stab["clipped"],
                "used_bins": stab["used_bins"],
                "bayes_gap": stab["bayes_gap"],
                "score": score,
            })
            pair_id += 1

    df_1a = pd.DataFrame(records_1a).sort_values("score", ascending=False).reset_index(drop=True)
    best_pair = df_1a.iloc[0]
    print(f"Distillation 1A: Top pair selected -> dv={best_pair['delta_v']}, n={int(best_pair['n_steps'])} (score={best_pair['score']:.3f})")

    # --- Intermediate Step: Energy-Balance Optimal Gamma* ---
    n_sel = int(best_pair["n_steps"])
    m_sel = int(best_pair["m"])
    dv_sel = float(best_pair["delta_v"])

    print(f"\nComputing steady-state energy balance gamma* for dv={dv_sel}, n={n_sel}...")
    opt_res = compute_optimal_gamma(
        system_name="double_pendulum_mess4",
        m=m_sel, n_steps=n_sel, dt=0.2,
        delta_v=dv_sel, stay=0.7, alpha=0.7,
        damping_field="gamma1",
        n_trajs=16 if smoke else 32,
    )
    gamma_star = float(opt_res["gamma_opt"])
    print(f"Computed Optimal gamma* = {gamma_star:.4f}")

    # Generate 9 candidate gammas: k*gamma* and k*gamma* +/- d_gamma (k in [0.03, 0.07, 0.10], d_gamma=0.05)
    candidate_gammas = []
    for k_factor in [0.03, 0.07, 0.10]:
        base_g = k_factor * gamma_star
        for offset in [0.0, -0.05, 0.05]:
            val = max(0.05, round(base_g + offset, 4))
            if val not in candidate_gammas:
                candidate_gammas.append(val)

    candidate_gammas = sorted(candidate_gammas)
    print(f"Generated {len(candidate_gammas)} candidate gammas: {candidate_gammas}")

    # --- Sweep 1B: Damping Sweep on Chosen Tuple ---
    print(f"\nSweep 1B: Screening candidate gammas on (dv={dv_sel}, n={n_sel})...")
    records_1b = []
    for g_val in candidate_gammas:
        proc = make_process(
            "double_pendulum_mess4",
            m=m_sel, n_steps=n_sel, delta_v=dv_sel,
            system={"gamma1": g_val, "gamma2": g_val},
        )
        tr = trace(proc, seed=42)
        stab = stability(tr)
        score = float(stab["bayes_gap"]) - 0.5 * max(0.0, float(stab["lyapunov"])) - 10.0 * float(stab["clipped"])
        records_1b.append({
            "delta_v": dv_sel,
            "gamma": g_val,
            "n_steps": n_sel,
            "m": m_sel,
            "lyapunov": stab["lyapunov"],
            "clipped": stab["clipped"],
            "used_bins": stab["used_bins"],
            "bayes_gap": stab["bayes_gap"],
            "score": score,
        })

    df_1b = pd.DataFrame(records_1b).sort_values("score", ascending=False).reset_index(drop=True)
    top_tuple = df_1b.iloc[0]

    winner = DistilledPhysicsTuple(
        pair_id=int(best_pair["pair_id"]),
        delta_v=float(top_tuple["delta_v"]),
        gamma=float(top_tuple["gamma"]),
        n_steps=int(top_tuple["n_steps"]),
        m=int(top_tuple["m"]),
        lyapunov=float(top_tuple["lyapunov"]),
        clipped=float(top_tuple["clipped"]),
        used_bins=int(top_tuple["used_bins"]),
        bayes_gap=float(top_tuple["bayes_gap"]),
        score=float(top_tuple["score"]),
    )

    # Export Level 1 artifacts
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "distilled_physics_tuples.csv"
    json_path = out_dir / "distilled_physics_tuples.json"
    
    df_1b.to_csv(csv_path, index=False)
    with open(json_path, "w") as f:
        json.dump([asdict(winner)], f, indent=2)

    print(f"Distillation 1B: Winner Locked -> dv={winner.delta_v}, gamma={winner.gamma}, n={winner.n_steps}, m={winner.m}")
    print(f"Exported -> {csv_path.name} & {json_path.name}")
    return winner


# ==============================================================================
# 4. Phase 2: In-Worker Probing & Lookahead Probability Map
# ==============================================================================

def run_synchronous_probing(
    trained_model: TinyTransformer,
    random_model: TinyTransformer,
    proc: Any,
    exp_name: str,
    training_name: str,
    seed: int,
    device: str,
    n_eval_trajs: int = 128,
) -> pd.DataFrame:
    """Executes 4-way Ridge probing across physics and belief targets for both models."""
    eval_rng = np.random.default_rng(seed + 888_888)
    eval_batch = proc.sample_batch(eval_rng, n_eval_trajs)
    tokens_np = eval_batch["tokens"]
    tokens_tensor = torch.as_tensor(tokens_np, dtype=torch.long, device=device)

    # 1. Residual streams for trained and random models (index 1: is after blocks)
    trained_streams = trained_model.residual_streams(tokens_tensor)[1:]
    random_streams = random_model.residual_streams(tokens_tensor)[1:]

    # 2. Extract targets
    N, seq_len = tokens_np.shape
    n = proc.n_steps
    m = proc.m
    assert seq_len == m * n == 500

    embedded_beliefs = eval_batch["beliefs"] @ simplex_embedding(proc.chain.n_states)
    physics_metric = eval_batch["metric"]  # (N, seq_len, 2)

    # Indices for cycle boundaries: c*n - 1
    cycle_indices = np.arange(n - 1, seq_len, n)
    # Indices strictly excluding Cycle 0: evaluating from Cycle 1 onwards (2n-1, 3n-1, ...)
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
            # Concat all layers at cycle boundaries: (N, m, n_layers * d_model) -> (N*m, n_layers * d_model)
            stacked = np.stack([s[:, cycle_indices] for s in streams], axis=-1)
            return stacked.reshape(N * m, -1)
        elif mode == "single_layer_cycle":
            # Last layer across all n tokens per cycle: (N, m, n * d_model) -> (N*m, n * d_model)
            last_layer = streams[-1].reshape(N, m, n, -1)
            return last_layer.reshape(N * m, -1)
        elif mode == "single_layer_last_token":
            # Last layer at cycle boundaries EXCLUDING Cycle 0: (N, m-1, d_model) -> (N*(m-1), d_model)
            return streams[-1][:, post_sync_indices].reshape(N * (m - 1), -1)
        elif mode == "single_layer_all_token":
            # Last layer across all individual tokens: (N*seq_len, d_model)
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

    for mode in probe_modes:
        X_tr = get_features(trained_streams, mode)
        X_rnd = get_features(random_streams, mode)

        for domain in target_domains:
            y, groups = get_target(domain, mode)

            # Fit Grouped 5-Fold Cross-Validated Ridge
            pipe = make_pipeline(StandardScaler(), Ridge(solver="lsqr"))
            cv = GroupKFold(n_splits=5)
            param_grid = {"ridge__alpha": [0.1, 1.0, 10.0, 50.0, 100.0]}

            grid_tr = GridSearchCV(pipe, param_grid, cv=cv, scoring="r2", n_jobs=1)
            grid_tr.fit(X_tr, y, groups=groups)
            r2_trained = float(grid_tr.best_score_)
            best_alpha = float(grid_tr.best_params_["ridge__alpha"])

            grid_rnd = GridSearchCV(pipe, param_grid, cv=cv, scoring="r2", n_jobs=1)
            grid_rnd.fit(X_rnd, y, groups=groups)
            r2_random = float(grid_rnd.best_score_)

            results.append({
                "model_id": f"{exp_name}_{training_name}_seed{seed}",
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

    return pd.DataFrame(results)


def plot_lookahead_probability_map(
    model: LookaheadTransformer,
    proc: Any,
    save_path: Path,
    device: str,
    fixed_eval_seed: int = 2026,
) -> None:
    """Generates standardized lookahead probability map on fixed held-out sequence."""
    rng = np.random.default_rng(fixed_eval_seed)
    sample = proc.sample_batch(rng, 1)["tokens"]
    tokens_tensor = torch.as_tensor(sample, dtype=torch.long, device=device)

    model.eval()
    with torch.no_grad():
        logits = model(tokens_tensor)
        probs = F.softmax(logits[0], dim=-1).cpu().numpy()

    k = model.k
    actual = sample[0, k:]
    source_len = len(actual)

    fig, ax = plt.subplots(figsize=(13, 5), dpi=150)
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
# 5. Production Worker Queue Execution (3 Models)
# ==============================================================================

@dataclass(frozen=True)
class PilotModelSpec:
    model_id: str
    exp_name: str
    training_name: str
    d_model: int
    d_mlp: int
    n_layers: int
    n_heads: int
    k: int
    k_suffix: str
    seed: int


def train_and_probe_worker(
    spec: PilotModelSpec,
    phys_tuple: DistilledPhysicsTuple,
    total_tokens: int,
    batch_size: int,
    device: str,
    results_dir: Path,
    smoke: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Trains single model, then synchronously runs 4-way probing, probability map, and zips."""
    model_dir = results_dir / spec.model_id
    model_dir.mkdir(parents=True, exist_ok=True)
    zip_path = results_dir / f"{spec.model_id}.zip"
    probe_csv_path = model_dir / f"{spec.model_id}_probes.csv"

    # Auto-resume check: skip re-training if already packaged
    if zip_path.exists() and probe_csv_path.exists():
        print(f"[{spec.model_id}] Found existing completed package -> skipping re-training.", flush=True)
        probe_df = pd.read_csv(probe_csv_path)
        return pd.DataFrame(), probe_df

    print(f"[{spec.model_id}] Starting job on {device} (Target: {total_tokens:,} tokens)...", flush=True)

    # 1. Physics process
    proc = make_process(
        "double_pendulum_mess4",
        m=phys_tuple.m,
        n_steps=phys_tuple.n_steps,
        delta_v=phys_tuple.delta_v,
        system={"gamma1": phys_tuple.gamma, "gamma2": phys_tuple.gamma},
    )

    # Micro-batching with gradient accumulation
    micro_batch = min(16 if (smoke or device == "cpu") else 256, batch_size)
    grad_accum_steps = max(1, batch_size // micro_batch)
    tokens_per_step = micro_batch * grad_accum_steps * proc.seq_len
    total_steps = max(1, total_tokens // tokens_per_step)
    warmup_steps = max(1, int(total_steps * 0.02))

    # 2. Model initialization
    cfg = ModelConfig(
        vocab_size=proc.n_obs,
        n_ctx=proc.seq_len,
        n_layers=spec.n_layers,
        n_heads=spec.n_heads,
        d_model=spec.d_model,
        d_mlp=spec.d_mlp,
        seed=spec.seed,
    )
    trained_model = LookaheadTransformer(cfg, k=spec.k).to(device)
    random_model = LookaheadTransformer(cfg, k=spec.k).to(device)  # untrained control

    optimiser = torch.optim.Adam(trained_model.parameters(), lr=1e-3, weight_decay=0.0)

    def lr_schedule(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(warmup_steps)
        prog = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * prog))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, lr_schedule)

    # 3. Streaming training loop
    stream_rng = np.random.default_rng(spec.seed + 101)
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
                "model_id": spec.model_id,
                "step": step,
                "tokens_seen": tokens_seen,
                "loss": loss_accum,
            })
            lr_now = scheduler.get_last_lr()[0]
            pct = 100.0 * (step + 1) / total_steps
            print(
                f"[{spec.model_id}] Step {step+1:4d}/{total_steps} ({pct:5.1f}%) | "
                f"Tokens: {tokens_seen/1e6:6.1f}M / {total_tokens/1e6:5.1f}M | "
                f"Loss: {loss_accum:6.4f} | LR: {lr_now:.2e}",
                flush=True,
            )

    # Save final model weights and sidecar
    final_pt = model_dir / f"{spec.model_id}_final.pt"
    final_json = model_dir / f"{spec.model_id}_final.json"
    torch.save(trained_model.state_dict(), final_pt)
    with open(final_json, "w") as f:
        json.dump({
            "model_id": spec.model_id,
            "d_model": spec.d_model,
            "d_mlp": spec.d_mlp,
            "k": spec.k,
            "seed": spec.seed,
            "final_loss": history[-1]["loss"],
            "tokens_trained": total_tokens,
        }, f, indent=2)

    # 4. In-Worker Synchronous Probing
    print(f"[{spec.model_id}] Running synchronous 4-way linear Ridge probing against random_init control...", flush=True)
    probe_df = run_synchronous_probing(
        trained_model=trained_model,
        random_model=random_model,
        proc=proc,
        exp_name=spec.exp_name,
        training_name=spec.training_name,
        seed=spec.seed,
        device=device,
        n_eval_trajs=64 if smoke else 128,
    )
    probe_csv = model_dir / f"{spec.model_id}_probes.csv"
    probe_json = model_dir / f"{spec.model_id}_probes.json"
    probe_df.to_csv(probe_csv, index=False)
    probe_df.to_json(probe_json, orient="records", indent=2)

    belief_r2 = probe_df[probe_df["target"] == "belief"]["trained_r2"].max()
    phys_r2 = probe_df[probe_df["target"] == "physics"]["trained_r2"].max()
    gain = probe_df[probe_df["target"] == "belief"]["trained_minus_random"].max()
    print(
        f"[{spec.model_id}] Probe Results -> Max Belief R²: {belief_r2:+.3f} | "
        f"Max Physics R²: {phys_r2:+.3f} | Gain over Random: {gain:+.3f}",
        flush=True,
    )

    # 5. Lookahead Probability Map
    prob_map_png = model_dir / f"{spec.model_id}_lookahead_prob_map.png"
    plot_lookahead_probability_map(trained_model, proc, prob_map_png, device=device)

    # 6. Instant zip packaging
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in model_dir.rglob("*"):
            if f.is_file():
                zf.write(f, arcname=f.relative_to(model_dir))

    print(f"[{spec.model_id}] Finished in {time.perf_counter()-t_start:.1f}s. Packaged -> {zip_path.name}", flush=True)
    return pd.DataFrame(history), probe_df


# ==============================================================================
# 6. Phase 3: Emergence Synthesis & 5-Panel Dashboard
# ==============================================================================

def generate_synthesis_dashboard(probe_df_all: pd.DataFrame, out_dir: Path) -> None:
    """Generates the 5 high-resolution synthesis figures and master zip package."""
    print("\n" + "=" * 78)
    print(" PHASE 3: EMERGENCE SYNTHESIS & 5-PANEL DASHBOARD")
    print("=" * 78)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Parse width from training_name
    probe_df_all["d_model"] = probe_df_all["training_name"].apply(
        lambda s: int(s.split("dmodel_")[1].split("_")[0])
    )

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
    probe_df_all["lookahead"] = probe_df_all["training_name"].apply(lambda s: s.split("_k")[-1])
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

    # 4. physics_vs_belief_scatter.png
    fig, ax = plt.subplots(figsize=(9, 7), dpi=150)
    piv = probe_df_all.pivot_table(
        index=["exp_name", "training_name", "probe_mode"],
        columns="target",
        values="trained_r2",
    ).reset_index()

    # Color palette for distinct model architectures
    unique_models = sorted(piv["training_name"].unique())
    cmap_tab = plt.get_cmap("tab10")
    model_colors = {m: cmap_tab(i % 10) for i, m in enumerate(unique_models)}

    # Marker shapes for the 4 probe representation views
    mode_markers = {
        "all_layers_single_token": ("o", "All Layers (Impulse)"),
        "single_layer_cycle": ("^", "Cycle Trajectory"),
        "single_layer_last_token": ("s", "Pre-Impulse (Last Token)"),
        "single_layer_all_token": ("D", "All Tokens Independent"),
    }

    # Reference axes at zero
    ax.axhline(0, color="gray", ls="--", lw=0.8, alpha=0.6)
    ax.axvline(0, color="gray", ls="--", lw=0.8, alpha=0.6)

    # Plot points with distinct color per model and marker per probe mode
    for model_name in unique_models:
        for mode_key, (marker_sym, _) in mode_markers.items():
            subset = piv[(piv["training_name"] == model_name) & (piv["probe_mode"] == mode_key)]
            if not subset.empty and "physics" in subset and "belief" in subset:
                ax.scatter(
                    subset["physics"],
                    subset["belief"],
                    color=[model_colors[model_name]],
                    marker=marker_sym,
                    s=80,
                    edgecolors="k",
                    alpha=0.85,
                )

    # Construct separate legends for Model (color) and Probe View (marker)
    color_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=model_colors[m], markeredgecolor="k", markersize=8, label=m)
        for m in unique_models
    ]
    marker_handles = [
        plt.Line2D([0], [0], marker=sym, color="w", markerfacecolor="gray", markeredgecolor="k", markersize=8, label=lbl)
        for sym, lbl in mode_markers.values()
    ]

    leg1 = ax.legend(handles=color_handles, title="Model Architecture", loc="upper left", fontsize=8, title_fontsize=9)
    ax.add_artist(leg1)
    ax.legend(handles=marker_handles, title="Probe View", loc="lower right", fontsize=8, title_fontsize=9)

    ax.set_xlabel("Physics Test R² (omega1, omega2)", fontsize=10)
    ax.set_ylabel("Belief Test R² (Mess-4 Mood)", fontsize=10)
    ax.set_title("4. Physics vs Belief Decoupling", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3)
    fig.savefig(out_dir / "physics_vs_belief_scatter.png", bbox_inches="tight")
    plt.close()

    # 5. trained_vs_random_gain.png
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    gain_grouped = probe_df_all.groupby("target")["trained_minus_random"].mean()
    gain_grouped.plot(kind="bar", color=["#e74c3c", "#3498db"], ax=ax, width=0.5)
    ax.set(ylabel="Net Emergence Gain (Trained - Random R²)", title="5. Net Representation Emergence Gain")
    ax.grid(True, alpha=0.3)
    fig.savefig(out_dir / "trained_vs_random_gain.png", bbox_inches="tight")
    plt.close()

    print("5-Panel Synthesis Dashboard successfully rendered and saved.")


# ==============================================================================
# 7. Main Pilot Orchestrator
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="End-to-End Pilot Pipeline (Phase 1 -> 3 Models -> Phase 3)")
    parser.add_argument("--tokens", type=int, default=400_000_000, help="Tokens per model (default: 400M)")
    parser.add_argument("--batch-size", type=int, default=1024, help="Batch size (default: 1024)")
    parser.add_argument("--smoke", action="store_true", help="Rapid dry-run verifying all 3 phases in ~1 minute")
    parser.add_argument("--platform", type=str, default=None, choices=["rtx5090", "kaggle", "local_cpu"], help="Platform mode override")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    platform_mode = args.platform or detect_platform_mode()
    
    if args.out_dir:
        output_base = Path(args.out_dir)
    elif platform_mode == "kaggle":
        output_base = Path("/kaggle/working")
    else:
        output_base = REPO_ROOT / "experiments" / "pilot_3models"

    results_dir = output_base / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    tokens_per_model = 2_000_000 if args.smoke else args.tokens
    batch_size = args.batch_size

    print("=" * 78)
    print(" DOUBLE PENDULUM: END-TO-END PILOT PIPELINE (3 REPRESENTATIVE MODELS)")
    print("=" * 78)
    print(f"Platform Mode:    {platform_mode.upper()}")
    print(f"Smoke Mode:       {args.smoke}")
    print(f"Tokens/Model:     {tokens_per_model:,}")
    print(f"Batch Size:       {batch_size} (locked invariant)")
    print(f"Obs Bins:         (50, 50) -> Vocab: 2,500")
    print(f"Integration dt:   0.2")
    print(f"Output Base:      {output_base}")
    print(f"Results Dir:      {results_dir}")
    print("=" * 78)

    # --------------------------------------------------------------------------
    # Phase 1: Physics Screening & Distillation
    # --------------------------------------------------------------------------
    phys_tuple = run_phase1_distillation(results_dir, smoke=args.smoke)

    # --------------------------------------------------------------------------
    # Phase 2: 3 Diverse Representative Models
    # --------------------------------------------------------------------------
    exp_name = f"dp_dv{phys_tuple.delta_v}_g{phys_tuple.gamma}_n{phys_tuple.n_steps}"
    
    # 3 diverse models across widths and lookahead horizons:
    k_mid = max(1, phys_tuple.n_steps // 2)
    k_full = phys_tuple.n_steps

    pilot_specs = [
        PilotModelSpec(
            model_id=f"{exp_name}_dmodel_32_dmlp_128_k1_seed0",
            exp_name=exp_name,
            training_name="dmodel_32_dmlp_128_k1",
            d_model=32, d_mlp=128, n_layers=4, n_heads=2,
            k=1, k_suffix="_k1", seed=0,
        ),
        PilotModelSpec(
            model_id=f"{exp_name}_dmodel_64_dmlp_256_k_mid_seed0",
            exp_name=exp_name,
            training_name="dmodel_64_dmlp_256_k_mid",
            d_model=64, d_mlp=256, n_layers=4, n_heads=2,
            k=k_mid, k_suffix="_k_mid", seed=0,
        ),
        PilotModelSpec(
            model_id=f"{exp_name}_dmodel_128_dmlp_512_k_full_seed0",
            exp_name=exp_name,
            training_name="dmodel_128_dmlp_512_k_full",
            d_model=128, d_mlp=512, n_layers=4, n_heads=2,
            k=k_full, k_suffix="_k_full", seed=0,
        ),
    ]

    print("\n" + "=" * 78)
    print(f" PHASE 2: TRAINING & PROBING 3 REPRESENTATIVE MODELS")
    print("=" * 78)
    for i, s in enumerate(pilot_specs):
        print(f"  Model {i+1}: {s.model_id} (w={s.d_model}, lookahead={s.k_suffix})")

    # Determine execution device(s) and worker pool size
    if platform_mode == "local_cpu" or not torch.cuda.is_available():
        num_workers = 1
        worker_devices = ["cpu"]
    elif platform_mode == "rtx5090":
        # 1x RTX 5090: multiplex up to 3 concurrent worker processes onto cuda:0
        num_workers = min(len(pilot_specs), int(os.environ.get("BELIEF_NUM_WORKERS", "3")))
        worker_devices = ["cuda:0"] * num_workers
    elif platform_mode == "kaggle":
        # Dual Tesla T4 on Kaggle: 2 workers across cuda:0 and cuda:1
        dev_count = torch.cuda.device_count()
        num_workers = max(1, min(2, dev_count))
        worker_devices = [f"cuda:{i}" for i in range(num_workers)] if dev_count > 0 else ["cpu"]
    else:
        num_workers = 1
        worker_devices = ["cuda:0"]

    all_probe_dfs = []
    manifest_path = output_base / "pipeline_state.json"
    master_csv_path = output_base / "master_probe_results.csv"
    manifest_data = {
        "platform_mode": platform_mode,
        "completed_models": [],
        "physics_tuple": asdict(phys_tuple),
        "total_tokens_per_model": tokens_per_model,
        "timestamp": time.time(),
    }

    # Parallel or sequential training
    if num_workers > 1 and not args.smoke:
        print(f"\nDispatching {len(pilot_specs)} jobs across {num_workers} concurrent worker processes ({worker_devices})...")
        ctx = torch.multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as executor:
            futs = []
            for i, spec in enumerate(pilot_specs):
                dev = worker_devices[i % len(worker_devices)]
                fut = executor.submit(
                    train_and_probe_worker,
                    spec=spec, phys_tuple=phys_tuple,
                    total_tokens=tokens_per_model, batch_size=batch_size,
                    device=dev, results_dir=results_dir, smoke=args.smoke,
                )
                futs.append(fut)
            for fut in concurrent.futures.as_completed(futs):
                _, probe_df = fut.result()
                all_probe_dfs.append(probe_df)
                # Incremental progress save
                cur_master_df = pd.concat(all_probe_dfs, ignore_index=True)
                cur_master_df.to_csv(master_csv_path, index=False)
                manifest_data["completed_models"] = list(cur_master_df["model_id"].unique())
                manifest_data["timestamp"] = time.time()
                with open(manifest_path, "w") as f:
                    json.dump(manifest_data, f, indent=2)
                print(f"-> Progress saved: {len(manifest_data['completed_models'])}/{len(pilot_specs)} models completed in {master_csv_path.name}", flush=True)
    else:
        dev = worker_devices[0]
        for spec in pilot_specs:
            _, probe_df = train_and_probe_worker(
                spec=spec, phys_tuple=phys_tuple,
                total_tokens=tokens_per_model, batch_size=batch_size,
                device=dev, results_dir=results_dir, smoke=args.smoke,
            )
            all_probe_dfs.append(probe_df)
            # Incremental progress save
            cur_master_df = pd.concat(all_probe_dfs, ignore_index=True)
            cur_master_df.to_csv(master_csv_path, index=False)
            manifest_data["completed_models"] = list(cur_master_df["model_id"].unique())
            manifest_data["timestamp"] = time.time()
            with open(manifest_path, "w") as f:
                json.dump(manifest_data, f, indent=2)
            print(f"-> Progress saved: {len(manifest_data['completed_models'])}/{len(pilot_specs)} models completed in {master_csv_path.name}", flush=True)

    print(f"\nState Manifest finalized -> {manifest_path}")

    # --------------------------------------------------------------------------
    # Phase 3: Emergence Synthesis & Dashboard
    # --------------------------------------------------------------------------
    master_probe_df = pd.concat(all_probe_dfs, ignore_index=True)
    master_probe_df.to_csv(master_csv_path, index=False)
    print(f"Master probe evaluations saved -> {master_csv_path} ({len(master_probe_df)} rows)")

    # Generate 5 figures
    generate_synthesis_dashboard(master_probe_df, results_dir)

    # Package into master zip
    master_zip_path = output_base / "double_pendulum_results.zip"
    print(f"\nCompressing full results into {master_zip_path}...")
    with zipfile.ZipFile(master_zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(manifest_path, arcname=manifest_path.name)
        zf.write(master_csv_path, arcname=master_csv_path.name)
        for f in results_dir.rglob("*"):
            if f.is_file():
                zf.write(f, arcname=f"results/{f.relative_to(results_dir)}")

    print(f"Master distribution package ready: {master_zip_path} ({master_zip_path.stat().st_size / 1024:.1f} KB)")
    print("\n" + "=" * 78)
    print("PILOT PIPELINE (3 MODELS) FULLY COMPLETE!")
    print("=" * 78)


if __name__ == "__main__":
    main()
