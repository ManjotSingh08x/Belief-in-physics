#!/usr/bin/env python3
"""Standalone worker script for parallel_train.ipynb.

Trains LookaheadTransformers across 3 horizons (k=1, k=n//2, k=n),
evaluates 6 linear probe modes against exact Bayesian belief and angular velocity,
saves sidecars, CSV results, and generates 8-panel summary plots.
"""

from __future__ import annotations

import argparse
import atexit
import fcntl
import gc
import json
import os
from pathlib import Path
import signal
import sys
import time

# Ensure repo root is on sys.path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
import torch
from itertools import combinations


from models.analysis import residual_streams_batched
from models.explore import next_token_probs
from models.train import TrainConfig, train
from models.transformer import ModelConfig, TinyTransformer
from physics.messk import MessDriven, MessKProcess, simplex_embedding
from physics.systems.pendulum import Pendulum
from lookahead_viz import plot_enhanced_summary

class SampledPhysicsSystem:
    """Wraps pendulum flow to execute fixed-step substeps for numerical accuracy."""

    def __init__(self, base, integration_dt: float = 0.01):
        self.base = base
        self.integration_dt = integration_dt

    def __getattr__(self, name: str):
        return getattr(self.base, name)

    def flow(self, z, dt: float, substeps: int = 1):
        internal_steps = round(dt / self.integration_dt)
        return self.base.flow(z, dt, substeps=internal_steps * substeps)


class LookaheadTransformer(TinyTransformer):
    """Causal transformer trained to predict token t+k ahead."""

    def __init__(self, config: ModelConfig, k: int):
        super().__init__(config)
        self.k = k

    def loss(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.k < 1 or self.k >= tokens.shape[1]:
            raise ValueError(f"lookahead k must satisfy 1 <= k < {tokens.shape[1]}, got {self.k}")
        logits = self(tokens)
        return torch.nn.functional.cross_entropy(
            logits[:, :-self.k].reshape(-1, logits.shape[-1]),
            tokens[:, self.k:].reshape(-1),
        )


def build_process(delta_v: float, gamma: float, dt: float, n_steps: int, m: int) -> MessDriven:
    pendulum = Pendulum(gamma=gamma)
    sampled = SampledPhysicsSystem(base=pendulum, integration_dt=0.01)
    chain = MessKProcess(n_states=4, alpha=0.7, stay=0.7)
    return MessDriven(
        chain=chain,
        system=sampled,
        delta_v=delta_v,
        m=m,
        n_steps=n_steps,
        dt=dt,
        obs_bins=181,
    )


def make_sampler(proc: MessDriven):
    def sampler(rng: np.random.Generator, batch_size: int):
        return proc.sample_batch(rng, batch_size)["tokens"]
    return sampler


def is_pid_alive(pid: int) -> bool:
    """Checks if a process with given PID is alive."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def claim_next_job(queue_path: Path, state_path: Path, worker_id: int | str = 0) -> dict | None:
    """Atomically claims the next pending job from the queue, recording in_progress."""
    lock_path = state_path.with_suffix(".lock")
    with open(lock_path, "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            state = {"completed": [], "in_progress": {}}
            if state_path.exists():
                try:
                    loaded = json.loads(state_path.read_text())
                    if isinstance(loaded, dict):
                        state["completed"] = loaded.get("completed", [])
                        state["in_progress"] = loaded.get("in_progress", {})
                except Exception:
                    pass

            completed = set(state["completed"])
            in_prog = state["in_progress"]

            # Clean up claims held by dead PIDs
            cleaned_in_prog = {}
            for j_id, claim_info in in_prog.items():
                pid = claim_info.get("pid") if isinstance(claim_info, dict) else None
                if pid is not None and not is_pid_alive(pid):
                    print(f"  [Queue] Reclaiming abandoned job {j_id} from dead PID {pid}")
                    continue
                cleaned_in_prog[j_id] = claim_info
            state["in_progress"] = cleaned_in_prog

            if not queue_path.exists():
                return None

            jobs = json.loads(queue_path.read_text())
            for job in jobs:
                j_id = job["job_id"]
                if j_id not in completed and j_id not in state["in_progress"]:
                    state["in_progress"][j_id] = {
                        "worker_id": str(worker_id),
                        "pid": os.getpid(),
                        "claimed_at": time.time(),
                    }
                    state_path.write_text(json.dumps(state, indent=2))
                    return job

            state_path.write_text(json.dumps(state, indent=2))
            return None
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)


def release_job(state_path: Path, job_id: str) -> None:
    """Atomically releases a claimed job from in_progress back to the queue."""
    lock_path = state_path.with_suffix(".lock")
    with open(lock_path, "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            if not state_path.exists():
                return
            try:
                state = json.loads(state_path.read_text())
            except Exception:
                return
            in_prog = state.get("in_progress", {})
            if job_id in in_prog:
                del in_prog[job_id]
                state["in_progress"] = in_prog
                state_path.write_text(json.dumps(state, indent=2))
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)


def mark_completed(state_path: Path, job_id: str) -> None:
    """Atomically moves a job from in_progress to completed."""
    lock_path = state_path.with_suffix(".lock")
    with open(lock_path, "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            state = {"completed": [], "in_progress": {}}
            if state_path.exists():
                try:
                    loaded = json.loads(state_path.read_text())
                    if isinstance(loaded, dict):
                        state["completed"] = loaded.get("completed", [])
                        state["in_progress"] = loaded.get("in_progress", {})
                except Exception:
                    pass

            if job_id not in state["completed"]:
                state["completed"].append(job_id)
            if job_id in state["in_progress"]:
                del state["in_progress"][job_id]

            state_path.write_text(json.dumps(state, indent=2))
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)

PROBE_MODES = [
    "single_layer_single_token",
    "all_layers_single_token",
    "single_layer_cycle",
    "all_layers_cycle",
    "single_layer_last_token",
    "all_layers_last_token",
]


def extract_raw_targets(batch: dict, proc: MessDriven) -> dict:
    seq_len = proc.seq_len
    n_steps = proc.n_steps
    last_positions = np.arange(n_steps - 1, seq_len, n_steps)
    physics_token = batch["metric"]  # (N, seq_len, 1)
    belief_raw = batch["beliefs"]     # (N, seq_len, 4)
    belief_token = belief_raw @ simplex_embedding(proc.chain.n_states)  # (N, seq_len, 3)
    physics_cycle = physics_token[:, last_positions, :]  # (N, m, 1)
    belief_cycle = belief_token[:, last_positions, :]    # (N, m, 3)
    return {
        "physics_token": physics_token,
        "belief_token": belief_token,
        "physics_cycle": physics_cycle,
        "belief_cycle": belief_cycle,
    }


def extract_probe_features_and_targets(
    streams: list[np.ndarray], raw_targets: dict, mode: str, n_steps: int, m: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    N, seq_len, d_model = streams[0].shape
    L = len(streams)
    if mode == "single_layer_single_token":
        stacked = np.stack(streams, axis=2)  # (N, seq_len, L, d_model)
        X = stacked.reshape(N * seq_len * L, d_model)
        y_phys = np.repeat(raw_targets["physics_token"][:, :, np.newaxis, :], L, axis=2).reshape(N * seq_len * L, -1)
        y_bel = np.repeat(raw_targets["belief_token"][:, :, np.newaxis, :], L, axis=2).reshape(N * seq_len * L, -1)
        groups = np.repeat(np.arange(N), seq_len * L)
    elif mode == "all_layers_single_token":
        concat = np.concatenate(streams, axis=-1)  # (N, seq_len, L * d_model)
        X = concat.reshape(N * seq_len, L * d_model)
        y_phys = raw_targets["physics_token"].reshape(N * seq_len, -1)
        y_bel = raw_targets["belief_token"].reshape(N * seq_len, -1)
        groups = np.repeat(np.arange(N), seq_len)
    elif mode == "single_layer_cycle":
        stacked = np.stack([s.reshape(N, m, n_steps * d_model) for s in streams], axis=2)
        X = stacked.reshape(N * m * L, n_steps * d_model)
        y_phys = np.repeat(raw_targets["physics_cycle"][:, :, np.newaxis, :], L, axis=2).reshape(N * m * L, -1)
        y_bel = np.repeat(raw_targets["belief_cycle"][:, :, np.newaxis, :], L, axis=2).reshape(N * m * L, -1)
        groups = np.repeat(np.arange(N), m * L)
    elif mode == "all_layers_cycle":
        concat = np.concatenate(streams, axis=-1)
        X = concat.reshape(N * m, n_steps * L * d_model)
        y_phys = raw_targets["physics_cycle"].reshape(N * m, -1)
        y_bel = raw_targets["belief_cycle"].reshape(N * m, -1)
        groups = np.repeat(np.arange(N), m)
    elif mode == "single_layer_last_token":
        last_positions = np.arange(n_steps - 1, seq_len, n_steps)
        X = streams[-1][:, last_positions, :].reshape(N * m, d_model)
        y_phys = raw_targets["physics_cycle"].reshape(N * m, -1)
        y_bel = raw_targets["belief_cycle"].reshape(N * m, -1)
        groups = np.repeat(np.arange(N), m)
    elif mode == "all_layers_last_token":
        last_positions = np.arange(n_steps - 1, seq_len, n_steps)
        concat = np.concatenate(streams, axis=-1)
        X = concat[:, last_positions, :].reshape(N * m, L * d_model)
        y_phys = raw_targets["physics_cycle"].reshape(N * m, -1)
        y_bel = raw_targets["belief_cycle"].reshape(N * m, -1)
        groups = np.repeat(np.arange(N), m)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return X.astype(np.float32), y_phys.astype(np.float32), y_bel.astype(np.float32), groups


def _ridge_r2_gpu(X_tr, y_tr, X_val, y_val, alpha: float, device: str) -> float:
    """Closed-form ridge on GPU: (XᵀX + αI)⁻¹ Xᵀy via torch.linalg.solve."""
    Xtr = torch.tensor(X_tr, dtype=torch.float32, device=device)
    ytr = torch.tensor(y_tr, dtype=torch.float32, device=device)
    Xv  = torch.tensor(X_val, dtype=torch.float32, device=device)
    yv  = torch.tensor(y_val, dtype=torch.float32, device=device)
    A = Xtr.T @ Xtr
    A.diagonal().add_(alpha)
    W = torch.linalg.solve(A, Xtr.T @ ytr)   # (features, targets)
    pred = Xv @ W
    ss_res = ((yv - pred) ** 2).sum()
    ss_tot = ((yv - yv.mean(0, keepdim=True)) ** 2).sum()
    return float(1.0 - ss_res / ss_tot.clamp(min=1e-10))


def cross_validated_ridge(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    dev_mask: np.ndarray,
    test_mask: np.ndarray,
    ridge_alphas: tuple = (0.1, 1.0, 10.0, 50.0),
    cv_folds: int = 5,
    device: str = "cpu",
) -> dict:
    """
    StandardScaler + Ridge with alpha CV on GPU (torch.linalg.solve),
    parallelised across folds via joblib (n_jobs=-1).
    Falls back to CPU if CUDA unavailable.
    """
    from joblib import Parallel, delayed

    X_dev, y_dev, g_dev = X[dev_mask], y[dev_mask], groups[dev_mask]

    # StandardScale on dev set; apply same transform to test
    scaler = StandardScaler().fit(X_dev)
    X_dev_s = scaler.transform(X_dev).astype(np.float32)
    X_test_s = scaler.transform(X[test_mask]).astype(np.float32)
    y_dev_f  = y_dev.astype(np.float32)

    ridge_device = device  # GPU if available, else cpu

    # Warm-up torch.linalg.solve on device to avoid multi-thread lazy loader race condition
    if ridge_device != "cpu":
        _ = torch.linalg.solve(torch.eye(2, device=ridge_device), torch.ones(2, 1, device=ridge_device))

    # Build (train_idx, val_idx) splits for GroupKFold
    splits = list(GroupKFold(n_splits=cv_folds).split(X_dev_s, y_dev_f, g_dev))

    def _fold_alpha(alpha, tr_idx, val_idx):
        return _ridge_r2_gpu(
            X_dev_s[tr_idx], y_dev_f[tr_idx],
            X_dev_s[val_idx], y_dev_f[val_idx],
            alpha, ridge_device,
        )

    # Parallel over (alpha × fold) — n_jobs=-1 uses all CPU threads
    results = Parallel(n_jobs=-1, prefer="threads")(
        delayed(_fold_alpha)(alpha, tr, val)
        for alpha in ridge_alphas
        for tr, val in splits
    )

    n_folds = len(splits)
    # Mean CV R² per alpha
    alpha_scores = {
        alpha: float(np.mean(results[i * n_folds : (i + 1) * n_folds]))
        for i, alpha in enumerate(ridge_alphas)
    }
    best_alpha = max(alpha_scores, key=alpha_scores.__getitem__)
    best_cv_r2 = alpha_scores[best_alpha]

    # Refit on full dev set with best alpha and score on test
    test_r2 = _ridge_r2_gpu(X_dev_s, y_dev_f, X_test_s,
                             y[test_mask].astype(np.float32), best_alpha, ridge_device)
    return {
        "alpha": float(best_alpha),
        "cv_r2": float(best_cv_r2),
        "test_r2": float(test_r2),
        "feature_dim": int(X.shape[1]),
    }

def generate_summary_plot(
    config_dir: Path,
    job: dict,
    width: int,
    seed: int,
    horizons: list[tuple[int, str]],
    loss_histories: dict[str, list[dict]],
    probe_df: pd.DataFrame,
    proc: MessDriven,
    device: str,
    n_layers: int = 4,
    n_heads: int = 1,
    belief_gt: np.ndarray | None = None,
    belief_pred: np.ndarray | None = None,
    hmm_states: np.ndarray | None = None,
):
    fig = plt.figure(figsize=(18, 14), constrained_layout=True)
    gs = fig.add_gridspec(3, 3)

    eval_batch = proc.sample_batch(np.random.default_rng(seed + 9999), 1)
    tokens_single = eval_batch["tokens"][0]

    # Row 0: 3 Heatmaps (k=1, k=n//2, k=n)
    for col_idx, (k, k_suffix) in enumerate(horizons):
        ax = fig.add_subplot(gs[0, col_idx])
        model_id = f"d{width}_{k_suffix}_seed{seed}"
        pt_path = config_dir / f"{model_id}_final.pt"
        json_path = config_dir / f"{model_id}_final.json"
        if pt_path.exists():
            layer_count = n_layers
            if json_path.exists():
                try:
                    meta = json.loads(json_path.read_text())
                    layer_count = meta.get("n_layers", n_layers)
                except Exception:
                    pass
            cfg = ModelConfig(
                vocab_size=181,
                n_ctx=proc.seq_len,
                n_layers=layer_count,
                n_heads=n_heads,
                d_model=width,
                d_mlp=4 * width,
                seed=seed,
            )
            model = LookaheadTransformer(cfg, k=k)
            model.load_state_dict(torch.load(pt_path, map_location=device))
            model.to(device).eval()
            probs = next_token_probs(model, tokens_single, device=device)

            if k < len(tokens_single):
                pred_probs = probs[:-k]
                actual = tokens_single[k:]
                ax.imshow(pred_probs.T, origin="lower", aspect="auto", cmap="magma")
                ax.plot(np.arange(len(actual)), actual, color="cyan", lw=1.2, label="Actual")
                nll = -np.log(np.maximum(pred_probs[np.arange(len(actual)), actual], 1e-12)).mean()
                ax.set_title(f"Horizon {k_suffix} (k={k}) | Mean NLL: {nll:.3f}")
            else:
                ax.text(0.5, 0.5, f"k={k} >= seq_len", ha="center", va="center")
        ax.set_xlabel("Time step t")
        ax.set_ylabel("Token bin (0..180)")

    # Row 1: Loss curves across horizons (colspan=3)
    ax_loss = fig.add_subplot(gs[1, :])
    colors = {"k1": "#2ecc71", "kn2": "#3498db", "kn": "#9b59b6"}
    for k, k_suffix in horizons:
        hist = loss_histories.get(k_suffix, [])
        if hist:
            steps = [h["step"] for h in hist]
            eval_losses = [h["eval_loss"] for h in hist]
            ax_loss.plot(
                steps,
                eval_losses,
                label=f"Eval Loss {k_suffix} (k={k})",
                color=colors.get(k_suffix, "black"),
                lw=1.8,
            )
    ax_loss.axhline(np.log(181), ls=":", color="gray", lw=1.5, label="Random ln(181)")
    ax_loss.set_title(f"Loss Curves — {job['config_name']} (d={width}, seed={seed})")
    ax_loss.set_xlabel("Training Steps")
    ax_loss.set_ylabel("Cross Entropy Loss")
    ax_loss.legend(loc="upper right")
    ax_loss.grid(True, alpha=0.3)

    # Row 2: Probe R² bars (Belief on col 0, Physics on col 1, Info on col 2)
    for col_idx, target_name in enumerate(["belief", "physics"]):
        ax_probe = fig.add_subplot(gs[2, col_idx])
        sub_df = probe_df[probe_df["target"] == target_name]
        modes = PROBE_MODES
        x = np.arange(len(modes))
        bar_w = 0.35

        trained_r2 = []
        random_r2 = []
        for m_name in modes:
            m_trained = sub_df[(sub_df["mode"] == m_name) & (sub_df["model_type"] == "trained")]["test_r2"]
            m_random = sub_df[(sub_df["mode"] == m_name) & (sub_df["model_type"] == "random_init")]["test_r2"]
            trained_r2.append(float(m_trained.max()) if len(m_trained) > 0 else 0.0)
            random_r2.append(float(m_random.max()) if len(m_random) > 0 else 0.0)

        ax_probe.bar(x - bar_w / 2, trained_r2, bar_w, label="Trained (best k)", color="#2980b9")
        ax_probe.bar(x + bar_w / 2, random_r2, bar_w, label="Random init", color="#e74c3c")
        ax_probe.set_xticks(x)
        ax_probe.set_xticklabels([m.replace("_", "\n") for m in modes], rotation=45, ha="right", fontsize=8)
        ax_probe.set_ylabel("Test R²")
        ax_probe.set_ylim(-0.1, 1.05)
        ax_probe.set_title(f"{target_name.capitalize()} Probe R² by Mode")
        ax_probe.legend(loc="upper left")
        ax_probe.grid(True, axis="y", alpha=0.3)

    # Row 2, Col 2: 3D tetrahedron belief projection
    ax3d = fig.add_subplot(gs[2, 2], projection="3d")

    tet_verts = simplex_embedding(4)  # (4, 3) — tetrahedron vertices
    # Draw tetrahedron wireframe edges
    for i, j in combinations(range(4), 2):
        v = tet_verts[[i, j]]
        ax3d.plot(v[:, 0], v[:, 1], v[:, 2], color="#555555", lw=0.8, alpha=0.5)

    STATE_COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]  # 4 HMM states

    if belief_gt is not None and hmm_states is not None:
        # Flatten for scatter: pick first N_vis trajectories, all cycles
        bg = belief_gt.reshape(-1, 3)      # (N_vis*m, 3)
        st = hmm_states.reshape(-1)        # (N_vis*m,)
        for s in range(4):
            mask = st == s
            ax3d.scatter(
                bg[mask, 0], bg[mask, 1], bg[mask, 2],
                s=6, c=STATE_COLORS[s], alpha=0.35, lw=0, label=f"State {s}",
                rasterized=True,
            )

    if belief_pred is not None:
        bp = belief_pred.reshape(-1, 3)
        ax3d.scatter(
            bp[:, 0], bp[:, 1], bp[:, 2],
            s=8, c="#111111", alpha=0.7, lw=0.8, marker="x", label="Predicted",
            rasterized=True,
        )

    # Tetrahedron vertex labels
    state_labels = ["S0", "S1", "S2", "S3"]
    for i, (v, lab) in enumerate(zip(tet_verts, state_labels)):
        ax3d.text(v[0]*1.12, v[1]*1.12, v[2]*1.12, lab,
                  color=STATE_COLORS[i], fontsize=8, fontweight="bold")

    ax3d.set_title("Belief 3D (tetrahedron)\nColored by HMM state", fontsize=9)
    ax3d.set_xlim(-1.1, 1.1); ax3d.set_ylim(-1.1, 1.1); ax3d.set_zlim(-1.1, 1.1)
    ax3d.set_box_aspect([1, 1, 1])
    ax3d.tick_params(labelsize=6)
    ax3d.grid(False)
    ax3d.set_xlabel(""); ax3d.set_ylabel(""); ax3d.set_zlabel("")
    handles, labels = ax3d.get_legend_handles_labels()
    if handles:
        ax3d.legend(handles, labels, loc="upper left", fontsize=6, markerscale=1.5)

    plot_path = config_dir / f"d{width}_seed{seed}_summary.png"
    plt.savefig(plot_path, dpi=120)
    plt.close(fig)


# --- 5. Main Worker Loop ---

def worker_main():
    parser = argparse.ArgumentParser(description="BeliefPhysics single-pendulum sweep worker")
    parser.add_argument("--queue-file", type=str, required=True, help="Path to _job_queue.json")
    parser.add_argument("--output-dir", type=str, required=True, help="Base output directory")
    parser.add_argument("--worker-id", type=int, default=0, help="Worker ID for logging")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--total-tokens", type=int, default=600_000_000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--n-probe-traj", type=int, default=1024)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--ridge-alphas", type=str, default="0.1,1.0,10.0,50.0")
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-heads", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=50)

    args = parser.parse_args()
    queue_path = Path(args.queue_file)
    output_dir = Path(args.output_dir)
    state_path = output_dir / "pipeline_state.json"
    ridge_alphas = tuple(float(a.strip()) for a in args.ridge_alphas.split(",") if a.strip())
    device = args.device

    current_job_id = [None]

    def cleanup_handler(*_):
        if current_job_id[0] is not None:
            print(f"\n[Worker {args.worker_id}] Terminating: releasing job {current_job_id[0]}...")
            release_job(state_path, current_job_id[0])
            current_job_id[0] = None
        sys.exit(0)

    try:
        signal.signal(signal.SIGINT, cleanup_handler)
        signal.signal(signal.SIGTERM, cleanup_handler)
    except (ValueError, AttributeError):
        pass

    atexit.register(lambda: release_job(state_path, current_job_id[0]) if current_job_id[0] else None)

    print(f"[Worker {args.worker_id}] Started on device: {device} (PID {os.getpid()})")
    oom_consecutive = 0

    while True:
        job = claim_next_job(queue_path, state_path, worker_id=args.worker_id)
        if job is None:
            has_in_progress = False
            if state_path.exists():
                try:
                    s_data = json.loads(state_path.read_text())
                    has_in_progress = bool(s_data.get("in_progress"))
                except Exception:
                    pass
            if has_in_progress:
                time.sleep(5)
                continue
            print(f"[Worker {args.worker_id}] Queue empty and no active jobs. Exiting.")
            break

        job_id = job["job_id"]
        current_job_id[0] = job_id
        print(f"\n[Worker {args.worker_id}] Claimed job: {job_id}")
        t0 = time.time()

        try:
            config_dir = output_dir / job["config_name"]
            config_dir.mkdir(parents=True, exist_ok=True)

            # 1. Build MessDriven process
            proc = build_process(
                delta_v=job["delta_v"],
                gamma=job["gamma"],
                dt=job["dt"],
                n_steps=job["n_steps"],
                m=job["m"],
            )
            seq_len = proc.seq_len
            sampler = make_sampler(proc)

            width = job["width"]
            seed = job["seed"]
            n_steps = job["n_steps"]
            m = job["m"]
            horizons = [(1, "k1"), (max(1, n_steps // 2), "kn2"), (n_steps, "kn")]
            loss_histories: dict[str, list[dict]] = {}

            # 2. Train 3 horizons
            for k, k_suffix in horizons:
                model_id = f"d{width}_{k_suffix}_seed{seed}"
                final_pt = config_dir / f"{model_id}_final.pt"
                final_json = config_dir / f"{model_id}_final.json"

                if final_pt.exists() and final_json.exists():
                    print(f"  [{model_id}] Checkpoint exists. Skipping training.")
                    loss_histories[k_suffix] = []
                    continue

                model_config = ModelConfig(
                    vocab_size=181,
                    n_ctx=seq_len,
                    n_layers=args.n_layers,
                    n_heads=args.n_heads,
                    d_model=width,
                    d_mlp=4 * width,
                    seed=seed,
                )
                model = LookaheadTransformer(model_config, k=k)

                train_cfg = TrainConfig(
                    total_tokens=args.total_tokens,
                    batch_size=args.batch_size,
                    learning_rate=1e-3,
                    weight_decay=0.0,
                    warmup_frac=0.02,
                    grad_clip=1.0,
                    seed=seed,
                    log_every=args.log_every,
                    checkpoint_at=(),
                )

                t_train = time.time()
                report = train(model, sampler, seq_len, train_cfg, device=device)
                elapsed = time.time() - t_train

                # Save weights
                torch.save({k_w: v_w.detach().cpu() for k_w, v_w in model.state_dict().items()}, final_pt)

                # Save metadata sidecar
                meta = {
                    "config_name": job["config_name"],
                    "delta_v": job["delta_v"],
                    "gamma": job["gamma"],
                    "dt": job["dt"],
                    "n_steps": n_steps,
                    "m": m,
                    "seq_len": seq_len,
                    "d_model": width,
                    "d_mlp": 4 * width,
                    "n_layers": args.n_layers,
                    "n_heads": args.n_heads,
                    "vocab_size": 181,
                    "k": k,
                    "k_suffix": k_suffix,
                    "seed": seed,
                    "total_tokens": args.total_tokens,
                    "tokens_seen": report["tokens_seen"],
                    "batch_size": args.batch_size,
                    "gradient_steps": report["steps"],
                    "final_train_loss": report["final"]["train_loss"] if report["final"] else None,
                    "final_eval_loss": report["final"]["eval_loss"] if report["final"] else None,
                    "device": device,
                    "training_time_seconds": round(elapsed, 1),
                }
                final_json.write_text(json.dumps(meta, indent=2))
                loss_histories[k_suffix] = report["history"]
                print(f"  [{model_id}] Finished in {elapsed:.1f}s | Eval loss: {meta['final_eval_loss']}")

                del model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # 3. Probing: 3 horizons × 2 models (trained, random) × 6 modes × 2 targets
            print(f"  [Probing] Generating {args.n_probe_traj} evaluation trajectories...")
            probe_rng = np.random.default_rng(seed + 1000)
            analysis_batch = proc.sample_batch(probe_rng, args.n_probe_traj)
            raw_targets = extract_raw_targets(analysis_batch, proc)

            N = args.n_probe_traj
            order = np.random.default_rng(seed + 4).permutation(N)
            n_test = max(1, int(N * args.test_frac))
            test_seqs = set(order[:n_test].tolist())

            probe_rows = []
            for k, k_suffix in horizons:
                model_id = f"d{width}_{k_suffix}_seed{seed}"
                final_pt = config_dir / f"{model_id}_final.pt"

                model_config = ModelConfig(
                    vocab_size=181,
                    n_ctx=seq_len,
                    n_layers=args.n_layers,
                    n_heads=args.n_heads,
                    d_model=width,
                    d_mlp=4 * width,
                    seed=seed,
                )

                # Load trained model
                trained_model = LookaheadTransformer(model_config, k=k)
                trained_model.load_state_dict(torch.load(final_pt, map_location=device))
                trained_model.to(device).eval()

                # Randomly initialized baseline model
                random_model = LookaheadTransformer(model_config, k=k).to(device).eval()

                for model_tag, model_obj in [("trained", trained_model), ("random_init", random_model)]:
                    streams = residual_streams_batched(model_obj, analysis_batch["tokens"], device)
                    for mode in PROBE_MODES:
                        X, y_phys, y_bel, groups = extract_probe_features_and_targets(
                            streams, raw_targets, mode, n_steps, m
                        )
                        t_mask = np.isin(groups, list(test_seqs))
                        d_mask = ~t_mask

                        for target_name, y in [("physics", y_phys), ("belief", y_bel)]:
                            score = cross_validated_ridge(
                                X, y, groups, d_mask, t_mask,
                                ridge_alphas=ridge_alphas, cv_folds=args.cv_folds,
                                device=device,
                            )
                            probe_rows.append({
                                "config_name": job["config_name"],
                                "delta_v": job["delta_v"],
                                "gamma": job["gamma"],
                                "dt": job["dt"],
                                "n_steps": n_steps,
                                "m": m,
                                "d_model": width,
                                "seed": seed,
                                "horizon": k_suffix,
                                "k_value": k,
                                "model_type": model_tag,
                                "mode": mode,
                                "target": target_name,
                                **score,
                            })

            # 4. Save per-job probes CSV
            probe_df = pd.DataFrame(probe_rows)
            probe_csv = config_dir / f"d{width}_seed{seed}_probes.csv"
            probe_df.to_csv(probe_csv, index=False)
            print(f"  [Probing] Saved {len(probe_df)} rows to {probe_csv.name}")

            # Compute 3D belief arrays for visualization (use first 32 trajectories max)
            N_VIS = min(32, args.n_probe_traj)
            vis_beliefs = analysis_batch["beliefs"][:N_VIS]            # (N_VIS, seq_len, 4)
            # Extract at cycle-end positions
            last_positions = np.arange(n_steps - 1, proc.seq_len, n_steps)  # (m,)
            vis_belief_gt = (vis_beliefs[:, last_positions, :]          # (N_VIS, m, 4)
                             @ simplex_embedding(4))                    # (N_VIS, m, 3)
            vis_hmm_states = vis_beliefs[:, last_positions, :].argmax(-1)  # (N_VIS, m)

            # Probe-predicted belief from best model (trained, all_layers_last_token, best horizon by cv_r2)
            best_row = (
                probe_df[(probe_df["target"] == "belief")
                         & (probe_df["model_type"] == "trained")
                         & (probe_df["mode"] == "all_layers_last_token")]
                .sort_values("cv_r2", ascending=False)
            )
            vis_belief_pred = None
            if len(best_row) > 0:
                best_k_suffix = best_row.iloc[0]["horizon"]
                best_k = best_row.iloc[0]["k_value"]
                best_alpha = best_row.iloc[0]["alpha"]
                best_model_id = f"d{width}_{best_k_suffix}_seed{seed}"
                best_pt = config_dir / f"{best_model_id}_final.pt"
                if best_pt.exists():
                    vis_cfg = ModelConfig(
                        vocab_size=181, n_ctx=proc.seq_len, n_layers=args.n_layers,
                        n_heads=args.n_heads, d_model=width, d_mlp=4 * width, seed=seed,
                    )
                    vis_model = LookaheadTransformer(vis_cfg, k=int(best_k))
                    vis_model.load_state_dict(torch.load(best_pt, map_location=device))
                    vis_model.to(device).eval()
                    vis_streams = residual_streams_batched(
                        vis_model, analysis_batch["tokens"][:N_VIS], device
                    )
                    # all_layers_last_token: concat streams, pick last_positions
                    vis_X_raw = np.concatenate(vis_streams, axis=-1)[:, last_positions, :]  # (N_VIS, m, L*d)
                    vis_X = StandardScaler().fit_transform(
                        vis_X_raw.reshape(N_VIS * len(last_positions), -1)
                    )
                    vis_y_bel = vis_belief_gt.reshape(N_VIS * len(last_positions), 3)
                    # Fit ridge with best alpha (no CV needed — just for viz)
                    _Xt = torch.tensor(vis_X, dtype=torch.float32, device=device)
                    _yt = torch.tensor(vis_y_bel.astype(np.float32), dtype=torch.float32, device=device)
                    _A = _Xt.T @ _Xt
                    _A.diagonal().add_(float(best_alpha))
                    _W = torch.linalg.solve(_A, _Xt.T @ _yt)
                    vis_belief_pred = (_Xt @ _W).cpu().numpy().reshape(N_VIS, len(last_positions), 3)

            # 5. Generate summary plot
            print(f"  [Summary] Generating summary plot...")
            generate_summary_plot(
                config_dir, job, width, seed, horizons, loss_histories, probe_df, proc, device,
                n_layers=args.n_layers, n_heads=args.n_heads,
                belief_gt=vis_belief_gt,
                belief_pred=vis_belief_pred,
                hmm_states=vis_hmm_states,
            )

            # 6. Mark job complete atomically
            mark_completed(state_path, job_id)
            current_job_id[0] = None
            oom_consecutive = 0
            elapsed_total = time.time() - t0
            print(f"[Worker {args.worker_id}] Job {job_id} complete in {elapsed_total:.1f}s")

        except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
            is_oom = isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()
            if not is_oom:
                release_job(state_path, job_id)
                current_job_id[0] = None
                raise

            oom_consecutive += 1
            print(f"\n[Worker {args.worker_id}] CAUGHT CUDA OOM on job {job_id} (count={oom_consecutive}): {exc}")

            # Clean up all GPU memory
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Atomically release job back to queue
            release_job(state_path, job_id)
            current_job_id[0] = None
            print(f"[Worker {args.worker_id}] Released job {job_id} back to queue.")

            # Back off to allow concurrent GPU processes to release memory (batch size is preserved!)
            backoff_secs = min(120.0, 15.0 * oom_consecutive + float(np.random.uniform(3.0, 10.0)))
            print(f"[Worker {args.worker_id}] Backing off for {backoff_secs:.1f}s before attempting next job...")
            time.sleep(backoff_secs)
            continue
        except Exception:
            release_job(state_path, job_id)
            current_job_id[0] = None
            raise


if __name__ == "__main__":
    worker_main()
