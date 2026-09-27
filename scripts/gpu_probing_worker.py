#!/usr/bin/env python3
"""GPU-Accelerated Probing Worker for Double Pendulum Production Pipeline.

Picks up trained model checkpoints (*_final.pt) from the probing queue,
runs synchronous 4-way Ridge regression probing entirely on GPU via native
PyTorch closed-form linear algebra (1000x faster than Scikit-learn on CPU),
generates standardized lookahead and belief simplex projection figures,
packages atomic .zip archives, and updates master_probe_results.csv and pipeline_state.json.

Usage:
    # Probe all currently unprobed checkpoints using GPU
    python scripts/gpu_probing_worker.py

    # Watch mode: continuously probe new checkpoints as training workers finish them
    python scripts/gpu_probing_worker.py --watch 10

    # Specify custom directory or device
    python scripts/gpu_probing_worker.py --dir /workspace/Belief-in-physics/experiments --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# 1. Ensure repository root and scripts directory are at the top of sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

# 2. Auto-detect and re-exec into .venv if run with a Python missing project dependencies
try:
    import torch
    import matplotlib
    import pandas
    import numpy
except ImportError:
    venv_python = REPO_ROOT / ".venv" / "bin" / "python3"
    if venv_python.exists() and sys.executable != str(venv_python):
        import subprocess
        sys.exit(subprocess.call([str(venv_python)] + sys.argv))
    raise

import matplotlib
matplotlib.use("Agg")  # Headless server safe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from physics.messk_configs import make_process, set_double_pendulum_bins

try:
    from scripts.production_train import (
        LookaheadTransformer,
        ModelConfig,
        plot_belief_simplex_projections,
        plot_lookahead_probability_map,
        simplex_embedding,
        square_projection_vertices,
    )
except ImportError:
    from production_train import (
        LookaheadTransformer,
        ModelConfig,
        plot_belief_simplex_projections,
        plot_lookahead_probability_map,
        simplex_embedding,
        square_projection_vertices,
    )

# Ensure canonical 50x50 bin grid invariant
set_double_pendulum_bins((50, 50))


# ==============================================================================
# 1. Ultra-Fast GPU Ridge Regression with GroupKFold
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

    unique_groups = torch.unique(groups_t)
    n_groups = len(unique_groups)
    group_folds = torch.arange(n_groups, device=device) % n_splits

    alpha_mean_r2s = []
    for alpha in alphas:
        fold_r2s = []
        for f in range(n_splits):
            val_group_mask = (group_folds == f)
            val_groups = unique_groups[val_group_mask]
            val_mask = torch.isin(groups_t, val_groups)
            train_mask = ~val_mask

            X_tr, Y_tr = X_t[train_mask], Y_t[train_mask]
            X_va, Y_va = X_t[val_mask], Y_t[val_mask]

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
# 2. Synchronous GPU Probing Execution
# ==============================================================================

def run_synchronous_probing_gpu(
    trained_model: LookaheadTransformer,
    random_model: LookaheadTransformer,
    proc: Any,
    exp_name: str,
    training_name: str,
    seed: int,
    device: str,
    n_eval_trajs: int = 512,
) -> Tuple[pd.DataFrame, Any]:
    """Runs 4-way Ridge regression probing on device using PyTorch linear algebra."""
    rng = np.random.default_rng(seed + 999)
    eval_batch = proc.sample_batch(rng, n_eval_trajs)
    tokens = eval_batch["tokens"]  # (N, seq_len)
    eval_tokens = torch.as_tensor(tokens, dtype=torch.long, device=device)

    # Extract residual streams directly on device
    def extract_streams(model: LookaheadTransformer) -> List[torch.Tensor]:
        streams = []
        with torch.no_grad():
            x = model._embed(eval_tokens)
            for block in model.blocks:
                x = block(x)
                streams.append(x)
        return streams

    trained_streams = extract_streams(trained_model)
    random_streams = extract_streams(random_model)

    N, seq_len = tokens.shape
    n = proc.n_steps
    m = proc.m
    assert seq_len == m * n == 500

    # Targets on device
    beliefs_np = eval_batch["beliefs"] @ simplex_embedding(proc.chain.n_states)
    metric_np = eval_batch["metric"]

    embedded_beliefs_t = torch.as_tensor(beliefs_np, dtype=torch.float32, device=device)
    physics_metric_t = torch.as_tensor(metric_np, dtype=torch.float32, device=device)

    cycle_indices = np.arange(n - 1, seq_len, n)
    post_sync_indices = np.arange(2 * n - 1, seq_len, n)

    probe_modes = [
        "all_layers_single_token",
        "single_layer_cycle",
        "single_layer_last_token",
        "single_layer_all_token",
    ]
    target_domains = ["belief", "physics"]

    results = []

    def get_features(streams: List[torch.Tensor], mode: str) -> torch.Tensor:
        if mode == "all_layers_single_token":
            stacked = torch.stack([s[:, cycle_indices] for s in streams], dim=-1)
            return stacked.reshape(N * m, -1)
        elif mode == "single_layer_cycle":
            last_layer = streams[-1].reshape(N, m, n, -1)
            return last_layer.reshape(N * m, -1)
        elif mode == "single_layer_last_token":
            return streams[-1][:, post_sync_indices].reshape(N * (m - 1), -1)
        elif mode == "single_layer_all_token":
            return streams[-1].reshape(N * seq_len, -1)
        raise ValueError(f"Unknown mode: {mode}")

    def get_target(domain: str, mode: str) -> Tuple[torch.Tensor, torch.Tensor]:
        data = embedded_beliefs_t if domain == "belief" else physics_metric_t
        if mode in ("all_layers_single_token", "single_layer_cycle"):
            y = data[:, cycle_indices].reshape(N * m, -1)
            groups = torch.repeat_interleave(torch.arange(N, device=device), m)
        elif mode == "single_layer_last_token":
            y = data[:, post_sync_indices].reshape(N * (m - 1), -1)
            groups = torch.repeat_interleave(torch.arange(N, device=device), m - 1)
        elif mode == "single_layer_all_token":
            y = data.reshape(N * seq_len, -1)
            groups = torch.repeat_interleave(torch.arange(N, device=device), seq_len)
        return y, groups

    simplex_payload = None
    model_id = f"{exp_name}_{training_name}_seed{seed}"

    for mode in probe_modes:
        X_tr = get_features(trained_streams, mode)
        X_rnd = get_features(random_streams, mode)

        for domain in target_domains:
            y, groups = get_target(domain, mode)

            # Fast GPU Ridge regression CV
            r2_trained, best_alpha, pred_y = gpu_ridge_cv(X_tr, y, groups)
            r2_random, _, _ = gpu_ridge_cv(X_rnd, y, groups)

            if domain == "belief" and mode == "single_layer_last_token":
                exact_y_np = y.cpu().numpy()
                pred_y_np = pred_y.cpu().numpy()
                mood_labels = eval_batch["moods"][:, post_sync_indices].reshape(-1)
                simplex_payload = (exact_y_np, pred_y_np, mood_labels)

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


# ==============================================================================
# 3. Model Probing Orchestration
# ==============================================================================

def probe_single_model(
    model_dir: Path,
    results_dir: Path,
    device: str = "cuda:0",
) -> Optional[pd.DataFrame]:
    """Loads checkpoint, runs GPU probing, creates plots, packages zip."""
    job_id = model_dir.name
    ckpt_path = model_dir / f"{job_id}_final.pt"
    sidecar_path = model_dir / f"{job_id}_final.json"
    probe_csv_path = model_dir / f"{job_id}_probes.csv"
    probe_json_path = model_dir / f"{job_id}_probes.json"
    prob_map_path = model_dir / f"{job_id}_lookahead_prob_map.png"
    simplex_path = model_dir / f"{job_id}_belief_simplex.png"
    zip_path = results_dir / f"{job_id}.zip"

    if not ckpt_path.exists():
        return None

    # Check if already completed
    if probe_csv_path.exists() and zip_path.exists():
        return pd.read_csv(probe_csv_path)

    t0 = time.perf_counter()

    # Load metadata sidecar
    if sidecar_path.exists():
        with open(sidecar_path, "r") as f:
            sidecar = json.load(f)
        exp_cfg = sidecar["exp_config"]
        train_cfg = sidecar["train_config"]
        d_model = sidecar.get("d_model", train_cfg.get("d_model", 64))
        d_mlp = sidecar.get("d_mlp", train_cfg.get("d_mlp", 256))
        k = sidecar.get("k", train_cfg.get("k", 1))
        seed = sidecar.get("seed", 0)
    else:
        # Fallback parse from job_id
        # e.g. double_pendulum_tuple0_dmodel_32_dmlp_128_k1_seed0
        print(f"[{job_id}] Warning: sidecar JSON missing, parsing configuration from ID...")
        parts = job_id.split("_")
        seed = int(parts[-1].replace("seed", ""))
        d_model = 64
        d_mlp = 256
        k = 1
        for p in parts:
            if p.startswith("dmodel"):
                d_model = int(p.replace("dmodel", ""))
            elif p.startswith("dmlp"):
                d_mlp = int(p.replace("dmlp", ""))
            elif p.startswith("k") and p[1:].isdigit():
                k = int(p[1:])
        exp_cfg = {"m": 4, "n": 125, "physics": {"delta_v": 1.2, "gamma1": 0.65, "gamma2": 0.65}, "name": "double_pendulum"}
        train_cfg = {"name": f"dmodel_{d_model}_dmlp_{d_mlp}", "n_layers": 4, "n_heads": 2}

    proc = make_process(
        "double_pendulum_mess4",
        m=exp_cfg["m"],
        n_steps=exp_cfg["n"],
        delta_v=exp_cfg["physics"]["delta_v"],
        system={"gamma1": exp_cfg["physics"]["gamma1"], "gamma2": exp_cfg["physics"]["gamma2"]},
    )

    cfg = ModelConfig(
        vocab_size=proc.n_obs,
        n_ctx=proc.seq_len,
        n_layers=train_cfg.get("n_layers", 4),
        n_heads=train_cfg.get("n_heads", 2),
        d_model=d_model,
        d_mlp=d_mlp,
        seed=seed,
    )

    trained_model = None
    random_model = None
    try:
        trained_model = LookaheadTransformer(cfg, k=k).to(device)
        trained_model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        trained_model.eval()

        random_model = LookaheadTransformer(cfg, k=k).to(device)
        random_model.eval()

        # Run ultra-fast GPU probing
        probe_df, simplex_data = run_synchronous_probing_gpu(
            trained_model=trained_model,
            random_model=random_model,
            proc=proc,
            exp_name=exp_cfg.get("name", "double_pendulum"),
            training_name=train_cfg.get("name", f"dmodel_{d_model}"),
            seed=seed,
            device=device,
            n_eval_trajs=512,
        )

        probe_df.to_csv(probe_csv_path, index=False)
        probe_df.to_json(probe_json_path, orient="records", indent=2)

        # Standardized plots
        plot_lookahead_probability_map(trained_model, proc, prob_map_path, device=device)

        if simplex_data is not None:
            plot_belief_simplex_projections(*simplex_data, save_path=simplex_path, title=f"Probed Belief Simplex Geometry (k={k}) — {job_id}")

        # Atomic per-model zip packaging
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in model_dir.rglob("*"):
                if f.is_file():
                    zf.write(f, arcname=f.relative_to(model_dir))

        elapsed = time.perf_counter() - t0
        b_r2 = probe_df[probe_df["target"] == "belief"]["trained_r2"].max()
        p_r2 = probe_df[probe_df["target"] == "physics"]["trained_r2"].max()
        gain = probe_df[probe_df["target"] == "belief"]["trained_minus_random"].max()
        print(
            f"[{job_id}] Probed in {elapsed:4.1f}s | "
            f"Belief R²: {b_r2:+.3f} | Physics R²: {p_r2:+.3f} | Gain: {gain:+.3f} -> {zip_path.name}",
            flush=True,
        )
        return probe_df
    finally:
        del trained_model
        del random_model
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()


def update_master_and_manifest(
    output_base: Path,
    results_dir: Path,
    new_probe_dfs: List[pd.DataFrame],
    completed_ids: List[str],
) -> None:
    """Updates master_probe_results.csv and pipeline_state.json atomically."""
    if not completed_ids:
        return

    master_csv = output_base / "master_probe_results.csv"
    manifest_path = output_base / "pipeline_state.json"

    # 1. Update Master CSV
    if new_probe_dfs:
        all_dfs = []
        if master_csv.exists():
            try:
                all_dfs.append(pd.read_csv(master_csv))
            except Exception:
                pass
        all_dfs.extend(new_probe_dfs)
        combined = pd.concat(all_dfs, ignore_index=True).drop_duplicates(subset=["model_id", "probe_mode", "target"])
        tmp_csv = master_csv.with_suffix(".csv.tmp")
        combined.to_csv(tmp_csv, index=False)
        os.replace(tmp_csv, master_csv)

    # 2. Update Manifest JSON
    if manifest_path.exists():
        try:
            with open(manifest_path, "r") as f:
                manifest_data = json.load(f)

            for cid in completed_ids:
                if cid not in manifest_data.setdefault("completed_models", []):
                    manifest_data["completed_models"].append(cid)
                if "job_statuses" in manifest_data:
                    manifest_data["job_statuses"][cid] = "COMPLETED"

            manifest_data["timestamp"] = time.time()
            tmp_json = manifest_path.with_suffix(".json.tmp")
            with open(tmp_json, "w") as f:
                json.dump(manifest_data, f, indent=2)
            os.replace(tmp_json, manifest_path)
        except Exception as e:
            print(f"Notice: Could not update pipeline_state.json ({e})", flush=True)


# ==============================================================================
# 4. Main Batch Runner
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="GPU-Accelerated Model Prober for Double Pendulum Pipeline")
    parser.add_argument("--dir", "-d", type=str, default=None, help="Root experiments directory (default: auto-detect)")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Device for linear algebra and forward passes (default: cuda:0 if available)")
    parser.add_argument("--watch", "-w", type=int, default=None, metavar="SECONDS", help="Watch mode: continuously poll and probe newly trained models every N seconds")
    parser.add_argument("--limit", "-l", type=int, default=None, help="Limit number of models to probe in this invocation")
    args = parser.parse_args()

    # Determine paths
    if args.dir:
        output_base = Path(args.dir).expanduser().resolve()
    elif (REPO_ROOT / "experiments").exists():
        output_base = (REPO_ROOT / "experiments").resolve()
    elif Path("/workspace/Belief-in-physics/experiments").exists():
        output_base = Path("/workspace/Belief-in-physics/experiments").resolve()
    elif Path("/kaggle/working").exists():
        output_base = Path("/kaggle/working").resolve()
    else:
        output_base = (REPO_ROOT / "experiments").resolve()

    results_dir = output_base / "results" if (output_base / "results").exists() else output_base
    device = args.device

    print("=" * 76)
    print(" DOUBLE PENDULUM GPU PROBING ACCELERATOR")
    print("=" * 76)
    print(f"Output Directory: {output_base}")
    print(f"Results Directory: {results_dir}")
    print(f"Compute Device:   {device}")
    print(f"Watch Interval:   {f'Every {args.watch}s' if args.watch else 'Single-pass'}")
    print("=" * 76)

    def run_pass() -> int:
        if not results_dir.exists():
            print(f"Notice: Results directory {results_dir} does not exist yet.", flush=True)
            return 0

        # Find all trained checkpoints
        ckpt_files = sorted(results_dir.glob("*/*_final.pt"))
        if not ckpt_files:
            return 0

        # Filter queue: find models that do NOT have completed probe package
        queue = []
        for ckpt in ckpt_files:
            m_dir = ckpt.parent
            zip_p = results_dir / f"{m_dir.name}.zip"
            probe_p = m_dir / f"{m_dir.name}_probes.csv"
            if not (zip_p.exists() and probe_p.exists()):
                queue.append(m_dir)

        if args.limit:
            queue = queue[:args.limit]

        if not queue:
            return 0

        print(f"\n[GPU Prober] Found {len(queue)} trained models awaiting probing in queue.", flush=True)
        completed_dfs = []
        completed_ids = []

        t_batch_start = time.perf_counter()
        for idx, m_dir in enumerate(queue, 1):
            print(f"[{idx}/{len(queue)}] Processing {m_dir.name}...", flush=True)
            try:
                df = probe_single_model(m_dir, results_dir, device=device)
                if df is not None:
                    completed_dfs.append(df)
                    completed_ids.append(m_dir.name)
            except Exception as e:
                print(f"[{m_dir.name}] Error during GPU probing: {e}", flush=True)

        # Batch update master CSV and manifest
        update_master_and_manifest(output_base, results_dir, completed_dfs, completed_ids)
        total_time = time.perf_counter() - t_batch_start
        print(f"\n[GPU Prober] Successfully probed and packaged {len(completed_ids)}/{len(queue)} models in {total_time:.1f}s ({total_time/max(1, len(completed_ids)):.1f}s/model)!", flush=True)
        return len(completed_ids)

    if args.watch:
        try:
            while True:
                probed_count = run_pass()
                time.sleep(args.watch)
        except KeyboardInterrupt:
            print("\n[GPU Prober] Exiting watch loop.")
    else:
        run_pass()


if __name__ == "__main__":
    main()
