#!/usr/bin/env python3
"""Standalone worker script for parallel_train.ipynb.

Trains LookaheadTransformers across 3 horizons (k=1, k=n//2, k=n) with randomized initial conditions,
evaluates linear probe modes (including per-layer probes reporting mean and max R² and k-aware usable_token)
against exact Bayesian belief and angular velocity, saves sidecars, CSV results, and generates 8-panel summary plots.
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
        if name in ("base", "integration_dt"):
            raise AttributeError(name)
        return getattr(self.base, name)

    def __getstate__(self):
        return {"base": self.base, "integration_dt": self.integration_dt}

    def __setstate__(self, state):
        self.base = state["base"]
        self.integration_dt = state["integration_dt"]

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


def build_process(
    delta_v: float, gamma: float, dt: float, n_steps: int, m: int, random_init: bool = True
) -> MessDriven:
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
        random_init=random_init,
        init_theta_std=0.3,
    )


class ProcessSampler:
    """Picklable callable for DataLoader worker processes that uses fast sample_tokens."""

    def __init__(self, proc: MessDriven):
        self.proc = proc

    def __call__(self, rng: np.random.Generator, batch_size: int):
        if hasattr(self.proc, "sample_tokens"):
            return self.proc.sample_tokens(rng, batch_size)
        return self.proc.sample_batch(rng, batch_size)["tokens"]


def make_sampler(proc: MessDriven):
    return ProcessSampler(proc)


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


def get_pending_jobs(
    jobs: list[dict],
    output_dir: Path | str,
    state_path: Path | str | None = None,
) -> list[dict]:
    """Returns list of pending jobs, reconciling pipeline_state.json with on-disk probe CSVs."""
    output_dir = Path(output_dir)
    state_p = Path(state_path) if state_path is not None else output_dir / "pipeline_state.json"
    raw_completed = set()
    state_data = {"completed": [], "in_progress": {}}
    if state_p.exists():
        try:
            state_data = json.loads(state_p.read_text())
            raw_completed = set(state_data.get("completed", []))
        except Exception:
            pass

    verified_completed = set()
    for j in jobs:
        jid = j["job_id"]
        probe_csv = output_dir / j["config_name"] / f"d{j['width']}_seed{j['seed']}_probes.csv"
        if jid in raw_completed and probe_csv.exists() and probe_csv.stat().st_size > 0:
            verified_completed.add(jid)

    if state_p.exists() and raw_completed != verified_completed:
        state_data["completed"] = [c for c in state_data.get("completed", []) if c in verified_completed]
        state_p.write_text(json.dumps(state_data, indent=2))

    return [j for j in jobs if j["job_id"] not in verified_completed]


PROBE_MODES = [
    "single_layer_single_token",
    "all_layers_single_token",
    "single_layer_cycle",
    "all_layers_cycle",
    "single_layer_usable_token",
    "all_layers_usable_token",
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
    streams: list[np.ndarray],
    raw_targets: dict,
    mode: str,
    n_steps: int,
    m: int,
    k: int = 1,
    layer: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    N, seq_len, d_model = streams[0].shape
    L = len(streams)

    # Aliases for backward compatibility
    if mode == "single_layer_last_token":
        mode = "single_layer_usable_token"
    elif mode == "all_layers_last_token":
        mode = "all_layers_usable_token"

    if mode == "single_layer_single_token":
        l_idx = -1 if layer is None else layer
        X = streams[l_idx].reshape(N * seq_len, d_model)
        y_phys = raw_targets["physics_token"].reshape(N * seq_len, -1)
        y_bel = raw_targets["belief_token"].reshape(N * seq_len, -1)
        groups = np.repeat(np.arange(N), seq_len)
    elif mode == "all_layers_single_token":
        concat = np.concatenate(streams, axis=-1)  # (N, seq_len, L * d_model)
        X = concat.reshape(N * seq_len, L * d_model)
        y_phys = raw_targets["physics_token"].reshape(N * seq_len, -1)
        y_bel = raw_targets["belief_token"].reshape(N * seq_len, -1)
        groups = np.repeat(np.arange(N), seq_len)
    elif mode == "single_layer_cycle":
        l_idx = -1 if layer is None else layer
        X = streams[l_idx].reshape(N, m, n_steps * d_model).reshape(N * m, n_steps * d_model)
        y_phys = raw_targets["physics_cycle"].reshape(N * m, -1)
        y_bel = raw_targets["belief_cycle"].reshape(N * m, -1)
        groups = np.repeat(np.arange(N), m)
    elif mode == "all_layers_cycle":
        concat = np.concatenate(streams, axis=-1)
        X = concat.reshape(N * m, n_steps * L * d_model)
        y_phys = raw_targets["physics_cycle"].reshape(N * m, -1)
        y_bel = raw_targets["belief_cycle"].reshape(N * m, -1)
        groups = np.repeat(np.arange(N), m)
    elif mode == "single_layer_usable_token":
        l_idx = -1 if layer is None else layer
        p = np.arange(seq_len - k)
        usable_positions = p[(p % n_steps) + k >= n_steps]
        n_u = len(usable_positions)
        X = streams[l_idx][:, usable_positions, :].reshape(N * n_u, d_model)
        y_phys = raw_targets["physics_token"][:, usable_positions, :].reshape(N * n_u, -1)
        y_bel = raw_targets["belief_token"][:, usable_positions, :].reshape(N * n_u, -1)
        groups = np.repeat(np.arange(N), n_u)
    elif mode == "all_layers_usable_token":
        p = np.arange(seq_len - k)
        usable_positions = p[(p % n_steps) + k >= n_steps]
        n_u = len(usable_positions)
        concat = np.concatenate(streams, axis=-1)
        X = concat[:, usable_positions, :].reshape(N * n_u, L * d_model)
        y_phys = raw_targets["physics_token"][:, usable_positions, :].reshape(N * n_u, -1)
        y_bel = raw_targets["belief_token"][:, usable_positions, :].reshape(N * n_u, -1)
        groups = np.repeat(np.arange(N), n_u)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return X.astype(np.float32), y_phys.astype(np.float32), y_bel.astype(np.float32), groups


def _compute_xtx_xty(
    X: np.ndarray, y: np.ndarray, device: str, chunk_size: int = 50_000
) -> tuple[torch.Tensor, torch.Tensor]:
    D = X.shape[1]
    if y.ndim == 1:
        y = y[:, None]
    K = y.shape[1]
    N = X.shape[0]

    if N <= chunk_size:
        Xt = torch.tensor(X, dtype=torch.float32, device=device)
        yt = torch.tensor(y, dtype=torch.float32, device=device)
        XtX = Xt.T @ Xt
        Xty = Xt.T @ yt
        del Xt, yt
        return XtX, Xty

    XtX = torch.zeros((D, D), dtype=torch.float32, device=device)
    Xty = torch.zeros((D, K), dtype=torch.float32, device=device)
    for i in range(0, N, chunk_size):
        xc = torch.tensor(X[i : i + chunk_size], dtype=torch.float32, device=device)
        yc = torch.tensor(y[i : i + chunk_size], dtype=torch.float32, device=device)
        XtX.addmm_(xc.T, xc)
        Xty.addmm_(xc.T, yc)
        del xc, yc
    return XtX, Xty


def _evaluate_r2(
    X_val: np.ndarray,
    y_val: np.ndarray,
    W: torch.Tensor,
    device: str,
    chunk_size: int = 50_000,
) -> float:
    if y_val.ndim == 1:
        y_val = y_val[:, None]
    N = X_val.shape[0]

    if N <= chunk_size:
        Xv = torch.tensor(X_val, dtype=torch.float32, device=device)
        yv = torch.tensor(y_val, dtype=torch.float32, device=device)
        pred = Xv @ W
        ss_res = ((yv - pred) ** 2).sum()
        ss_tot = ((yv - yv.mean(0, keepdim=True)) ** 2).sum()
        r2 = float(1.0 - ss_res / ss_tot.clamp(min=1e-10))
        del Xv, yv, pred
        return r2

    yv_cpu = torch.from_numpy(y_val)
    yv_mean = yv_cpu.mean(0, keepdim=True)
    ss_tot = float(((yv_cpu - yv_mean) ** 2).sum())

    ss_res = 0.0
    for i in range(0, N, chunk_size):
        xc = torch.tensor(X_val[i : i + chunk_size], dtype=torch.float32, device=device)
        yc = torch.tensor(y_val[i : i + chunk_size], dtype=torch.float32, device=device)
        pred = xc @ W
        ss_res += float(((yc - pred) ** 2).sum().item())
        del xc, yc, pred

    return float(1.0 - ss_res / max(ss_tot, 1e-10))


def _ridge_r2_gpu(X_tr, y_tr, X_val, y_val, alpha: float, device: str) -> float:
    """Closed-form ridge on GPU: (XᵀX + αI)⁻¹ Xᵀy via torch.linalg.solve."""
    XtX, Xty = _compute_xtx_xty(X_tr, y_tr, device)
    eye = torch.eye(XtX.shape[0], dtype=torch.float32, device=device)
    A = XtX.add(eye, alpha=float(alpha))
    W = torch.linalg.solve(A, Xty)
    del XtX, Xty, eye, A
    r2 = _evaluate_r2(X_val, y_val, W, device)
    del W
    return r2


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
    StandardScaler + Ridge with alpha CV on GPU (torch.linalg.solve).
    Accumulates XᵀX and Xᵀy once per fold in memory-bounded chunks,
    eliminating multi-threaded VRAM blowups.
    """
    X_dev, y_dev, g_dev = X[dev_mask], y[dev_mask], groups[dev_mask]

    scaler = StandardScaler().fit(X_dev)
    X_dev_s = scaler.transform(X_dev).astype(np.float32)
    X_test_s = scaler.transform(X[test_mask]).astype(np.float32)
    y_dev_f  = y_dev.astype(np.float32)
    y_test_f = y[test_mask].astype(np.float32)

    ridge_device = device

    splits = list(GroupKFold(n_splits=cv_folds).split(X_dev_s, y_dev_f, g_dev))
    alpha_fold_scores: dict[float, list[float]] = {alpha: [] for alpha in ridge_alphas}

    d_feat = X_dev_s.shape[1]
    eye_d = torch.eye(d_feat, dtype=torch.float32, device=ridge_device)

    for tr_idx, val_idx in splits:
        XtX, Xty = _compute_xtx_xty(X_dev_s[tr_idx], y_dev_f[tr_idx], device=ridge_device)
        for alpha in ridge_alphas:
            A = XtX.add(eye_d, alpha=float(alpha))
            W = torch.linalg.solve(A, Xty)
            del A
            r2 = _evaluate_r2(X_dev_s[val_idx], y_dev_f[val_idx], W, device=ridge_device)
            del W
            alpha_fold_scores[alpha].append(r2)
        del XtX, Xty

    del eye_d
    if str(ridge_device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()

    alpha_scores = {
        alpha: float(np.mean(alpha_fold_scores[alpha]))
        for alpha in ridge_alphas
    }
    best_alpha = max(alpha_scores, key=alpha_scores.__getitem__)
    best_cv_r2 = alpha_scores[best_alpha]

    XtX_full, Xty_full = _compute_xtx_xty(X_dev_s, y_dev_f, device=ridge_device)
    eye_full = torch.eye(d_feat, dtype=torch.float32, device=ridge_device)
    A_full = XtX_full.add(eye_full, alpha=float(best_alpha))
    W_best = torch.linalg.solve(A_full, Xty_full)
    del XtX_full, Xty_full, eye_full, A_full

    test_r2 = _evaluate_r2(X_test_s, y_test_f, W_best, device=ridge_device)
    del W_best
    if str(ridge_device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()

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
    heatmap_window: int = 20,
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
                nll = -np.log(np.maximum(pred_probs[np.arange(len(actual)), actual], 1e-12)).mean()

                n_steps_total = len(actual)
                if n_steps_total > heatmap_window:
                    start_t = (n_steps_total - heatmap_window) // 2
                    end_t = start_t + heatmap_window
                else:
                    start_t = 0
                    end_t = n_steps_total

                slice_pred = pred_probs[start_t:end_t]
                slice_actual = actual[start_t:end_t]
                time_steps = np.arange(start_t, end_t)

                ax.imshow(
                    slice_pred.T,
                    origin="lower",
                    aspect="auto",
                    cmap="magma",
                    extent=[start_t - 0.5, end_t - 0.5, -0.5, 180.5],
                )
                ax.plot(time_steps, slice_actual, color="cyan", lw=1.5, marker="o", markersize=3, label="Actual")
                ax.set_xlim(start_t - 0.5, end_t - 0.5)
                ax.set_ylim(-0.5, 180.5)
                if n_steps_total > heatmap_window:
                    ax.set_title(f"Horizon {k_suffix} (k={k}) | Mean NLL: {nll:.3f}\n[t={start_t}..{end_t-1}]")
                else:
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
    plot_modes = [
        ("all_layers_single_token", "all_layers\nsingle_token", False),
        ("single_layer_single_token", "single_layer\nsingle_token", True),
        ("all_layers_cycle", "all_layers\ncycle", False),
        ("single_layer_cycle", "single_layer\ncycle", True),
        ("all_layers_usable_token", "all_layers\nusable_token", False),
        ("single_layer_usable_token", "single_layer\nusable_token", True),
    ]
    for col_idx, target_name in enumerate(["belief", "physics"]):
        ax_probe = fig.add_subplot(gs[2, col_idx])
        sub_df = probe_df[probe_df["target"] == target_name]
        x = np.arange(len(plot_modes))
        bar_w = 0.35

        trained_mean, trained_max = [], []
        random_mean, random_max = [], []

        for m_name, _, is_single in plot_modes:
            if is_single:
                if "layer" in sub_df.columns:
                    tr_m = sub_df[(sub_df["mode"] == m_name) & (sub_df["layer"] == "mean") & (sub_df["model_type"] == "trained")]["test_r2"]
                    tr_mx = sub_df[(sub_df["mode"] == m_name) & (sub_df["layer"] == "max") & (sub_df["model_type"] == "trained")]["test_r2"]
                    rnd_m = sub_df[(sub_df["mode"] == m_name) & (sub_df["layer"] == "mean") & (sub_df["model_type"] == "random_init")]["test_r2"]
                    rnd_mx = sub_df[(sub_df["mode"] == m_name) & (sub_df["layer"] == "max") & (sub_df["model_type"] == "random_init")]["test_r2"]
                else:
                    tr_m = sub_df[(sub_df["mode"] == m_name) & (sub_df["model_type"] == "trained")]["test_r2"]
                    tr_mx = tr_m
                    rnd_m = sub_df[(sub_df["mode"] == m_name) & (sub_df["model_type"] == "random_init")]["test_r2"]
                    rnd_mx = rnd_m
                tm = float(tr_m.max()) if len(tr_m) > 0 else 0.0
                trained_mean.append(tm)
                trained_max.append(float(tr_mx.max()) if len(tr_mx) > 0 else tm)

                rm = float(rnd_m.max()) if len(rnd_m) > 0 else 0.0
                random_mean.append(rm)
                random_max.append(float(rnd_mx.max()) if len(rnd_mx) > 0 else rm)
            else:
                if "layer" in sub_df.columns:
                    tr = sub_df[(sub_df["mode"] == m_name) & (sub_df["layer"] == "all") & (sub_df["model_type"] == "trained")]["test_r2"]
                    rnd = sub_df[(sub_df["mode"] == m_name) & (sub_df["layer"] == "all") & (sub_df["model_type"] == "random_init")]["test_r2"]
                else:
                    tr = sub_df[(sub_df["mode"] == m_name) & (sub_df["model_type"] == "trained")]["test_r2"]
                    rnd = sub_df[(sub_df["mode"] == m_name) & (sub_df["model_type"] == "random_init")]["test_r2"]
                trained_mean.append(float(tr.max()) if len(tr) > 0 else 0.0)
                trained_max.append(None)
                random_mean.append(float(rnd.max()) if len(rnd) > 0 else 0.0)
                random_max.append(None)

        ax_probe.bar(x - bar_w / 2, trained_mean, bar_w, label="Trained (mean/all)", color="#2980b9", alpha=0.85)
        ax_probe.bar(x + bar_w / 2, random_mean, bar_w, label="Random init", color="#e74c3c", alpha=0.85)

        for i, (_, _, is_single) in enumerate(plot_modes):
            if is_single:
                x_tr = x[i] - bar_w / 2
                tm_val, tmx_val = trained_mean[i], trained_max[i]
                ax_probe.hlines(tmx_val, x_tr - bar_w * 0.45, x_tr + bar_w * 0.45, colors="#1b4f72", linewidth=2.5, zorder=5)
                if tmx_val > tm_val:
                    ax_probe.vlines(x_tr, tm_val, tmx_val, colors="#1b4f72", linestyle=":", linewidth=1.5, zorder=4)

                x_rnd = x[i] + bar_w / 2
                rm_val, rmx_val = random_mean[i], random_max[i]
                ax_probe.hlines(rmx_val, x_rnd - bar_w * 0.45, x_rnd + bar_w * 0.45, colors="#78281f", linewidth=2.5, zorder=5)
                if rmx_val > rm_val:
                    ax_probe.vlines(x_rnd, rm_val, rmx_val, colors="#78281f", linestyle=":", linewidth=1.5, zorder=4)

        ax_probe.plot([], [], color="black", marker="_", markersize=10, markeredgewidth=2.5, linestyle="None", label="Max Layer Ticker")
        ax_probe.set_xticks(x)
        ax_probe.set_xticklabels([lbl for _, lbl, _ in plot_modes], rotation=35, ha="right", fontsize=7.5)
        ax_probe.set_ylabel("Test R²")
        ax_probe.set_ylim(-0.1, 1.05)
        ax_probe.set_title(f"{target_name.capitalize()} Probe R² by Mode")
        ax_probe.legend(loc="upper left", fontsize=7.5)
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


def run_job(
    job: dict,
    output_dir: Path | str,
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu",
    total_tokens: int = 600_000_000,
    batch_size: int = 1024,
    eval_batch_size: int = 32,
    num_workers: int = 4,
    n_probe_traj: int = 1024,
    cv_folds: int = 5,
    ridge_alphas: tuple[float, ...] = (0.1, 1.0, 10.0, 50.0),
    test_frac: float = 0.2,
    n_layers: int = 4,
    n_heads: int = 8,
    log_every: int = 100,
    state_path: Path | str | None = None,
) -> pd.DataFrame:
    output_dir = Path(output_dir)
    state_path = Path(state_path) if state_path is not None else output_dir / "pipeline_state.json"
    job_id = job["job_id"]
    t0 = time.time()
    config_dir = output_dir / job["config_name"]
    config_dir.mkdir(parents=True, exist_ok=True)

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
            n_layers=n_layers,
            n_heads=n_heads,
            d_model=width,
            d_mlp=4 * width,
            seed=seed,
        )
        model = LookaheadTransformer(model_config, k=k)

        train_cfg = TrainConfig(
            total_tokens=total_tokens,
            batch_size=batch_size,
            eval_batch_size=eval_batch_size,
            learning_rate=1e-3,
            weight_decay=0.0,
            warmup_frac=0.02,
            grad_clip=1.0,
            seed=seed,
            log_every=log_every,
            checkpoint_at=(),
            num_workers=num_workers,
        )

        t_train = time.time()
        report = train(model, sampler, seq_len, train_cfg, device=device)
        elapsed = time.time() - t_train

        torch.save({k_w: v_w.detach().cpu() for k_w, v_w in model.state_dict().items()}, final_pt)

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
            "n_layers": n_layers,
            "n_heads": n_heads,
            "vocab_size": 181,
            "k": k,
            "k_suffix": k_suffix,
            "seed": seed,
            "total_tokens": total_tokens,
            "tokens_seen": report["tokens_seen"],
            "batch_size": batch_size,
            "gradient_steps": report["steps"],
            "final_train_loss": report["final"]["train_loss"] if report["final"] else None,
            "final_eval_loss": report["final"]["eval_loss"] if report["final"] else None,
            "device": device,
            "training_time_seconds": round(elapsed, 1),
            "random_init": True,
            "init_theta_std": 0.3,
            "init_omega_std": job["delta_v"],
        }
        final_json.write_text(json.dumps(meta, indent=2))
        loss_histories[k_suffix] = report["history"]
        print(f"  [{model_id}] Finished in {elapsed:.1f}s | Eval loss: {meta['final_eval_loss']}")

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    probe_csv = config_dir / f"d{width}_seed{seed}_probes.csv"
    if probe_csv.exists() and probe_csv.stat().st_size > 0:
        print(f"  [Probing] Probe file {probe_csv.name} already exists. Skipping probing.")
        if state_path is not None:
            mark_completed(state_path, job_id)
        return pd.read_csv(probe_csv)

    print(f"  [Probing] Generating {n_probe_traj} evaluation trajectories...")
    probe_rng = np.random.default_rng(seed + 1000)
    analysis_batch = proc.sample_batch(probe_rng, n_probe_traj)
    raw_targets = extract_raw_targets(analysis_batch, proc)

    N = n_probe_traj
    order = np.random.default_rng(seed + 4).permutation(N)
    n_test = max(1, int(N * test_frac))
    test_seqs = set(order[:n_test].tolist())

    probe_rows = []
    for k, k_suffix in horizons:
        model_id = f"d{width}_{k_suffix}_seed{seed}"
        final_pt = config_dir / f"{model_id}_final.pt"

        model_config = ModelConfig(
            vocab_size=181,
            n_ctx=seq_len,
            n_layers=n_layers,
            n_heads=n_heads,
            d_model=width,
            d_mlp=4 * width,
            seed=seed,
        )

        for model_tag in ["trained", "random_init"]:
            if model_tag == "trained":
                model_obj = LookaheadTransformer(model_config, k=k)
                model_obj.load_state_dict(torch.load(final_pt, map_location=device))
                model_obj.to(device).eval()
            else:
                model_obj = LookaheadTransformer(model_config, k=k).to(device).eval()

            streams = residual_streams_batched(model_obj, analysis_batch["tokens"], device)
            del model_obj
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            L = len(streams)
            for mode in PROBE_MODES:
                if mode.startswith("single_layer_"):
                    layer_scores = []
                    for layer_idx in range(L):
                        X, y_phys, y_bel, groups = extract_probe_features_and_targets(
                            streams, raw_targets, mode, n_steps, m, k=k, layer=layer_idx
                        )
                        t_mask = np.isin(groups, list(test_seqs))
                        d_mask = ~t_mask

                        for target_name, y in [("physics", y_phys), ("belief", y_bel)]:
                            score = cross_validated_ridge(
                                X, y, groups, d_mask, t_mask,
                                ridge_alphas=ridge_alphas, cv_folds=cv_folds,
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
                                "layer": str(layer_idx),
                                **score,
                            })
                            layer_scores.append((target_name, score))

                    for target_name in ["physics", "belief"]:
                        t_scores = [sc for tn, sc in layer_scores if tn == target_name]
                        cv_r2s = [s["cv_r2"] for s in t_scores]
                        test_r2s = [s["test_r2"] for s in t_scores]
                        feat_dim = t_scores[0]["feature_dim"] if t_scores else width

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
                            "layer": "mean",
                            "alpha": float("nan"),
                            "cv_r2": float(np.mean(cv_r2s)),
                            "test_r2": float(np.mean(test_r2s)),
                            "feature_dim": feat_dim,
                        })
                        max_idx = int(np.argmax(test_r2s)) if test_r2s else 0
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
                            "layer": "max",
                            "alpha": t_scores[max_idx]["alpha"] if t_scores else float("nan"),
                            "cv_r2": float(np.max(cv_r2s)) if cv_r2s else float("nan"),
                            "test_r2": float(np.max(test_r2s)) if test_r2s else float("nan"),
                            "feature_dim": feat_dim,
                        })
                else:
                    X, y_phys, y_bel, groups = extract_probe_features_and_targets(
                        streams, raw_targets, mode, n_steps, m, k=k
                    )
                    t_mask = np.isin(groups, list(test_seqs))
                    d_mask = ~t_mask

                    for target_name, y in [("physics", y_phys), ("belief", y_bel)]:
                        score = cross_validated_ridge(
                            X, y, groups, d_mask, t_mask,
                            ridge_alphas=ridge_alphas, cv_folds=cv_folds,
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
                            "layer": "all",
                            **score,
                        })

            del streams
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    probe_df = pd.DataFrame(probe_rows)
    probe_csv = config_dir / f"d{width}_seed{seed}_probes.csv"
    probe_df.to_csv(probe_csv, index=False)
    print(f"  [Probing] Saved {len(probe_df)} rows to {probe_csv.name}")

    N_VIS = min(32, n_probe_traj)
    vis_beliefs = analysis_batch["beliefs"][:N_VIS]

    best_row = (
        probe_df[(probe_df["target"] == "belief")
                 & (probe_df["model_type"] == "trained")
                 & (probe_df["mode"] == "all_layers_usable_token")]
        .sort_values("cv_r2", ascending=False)
    )
    vis_belief_pred = None
    vis_pos = None
    if len(best_row) > 0:
        best_k_suffix = best_row.iloc[0]["horizon"]
        best_k = int(best_row.iloc[0]["k_value"])
        best_alpha = best_row.iloc[0]["alpha"]
        best_model_id = f"d{width}_{best_k_suffix}_seed{seed}"
        best_pt = config_dir / f"{best_model_id}_final.pt"
        vis_pos = proc.usable_positions(best_k)
        if best_pt.exists():
            vis_cfg = ModelConfig(
                vocab_size=181, n_ctx=proc.seq_len, n_layers=n_layers,
                n_heads=n_heads, d_model=width, d_mlp=4 * width, seed=seed,
            )
            vis_model = LookaheadTransformer(vis_cfg, k=best_k)
            vis_model.load_state_dict(torch.load(best_pt, map_location=device))
            vis_model.to(device).eval()
            vis_streams = residual_streams_batched(
                vis_model, analysis_batch["tokens"][:N_VIS], device
            )
            vis_X_raw = np.concatenate(vis_streams, axis=-1)[:, vis_pos, :]
            vis_X = StandardScaler().fit_transform(
                vis_X_raw.reshape(N_VIS * len(vis_pos), -1)
            )
            vis_y_bel = (vis_beliefs[:, vis_pos, :] @ simplex_embedding(4)).reshape(N_VIS * len(vis_pos), 3)
            _Xt = torch.tensor(vis_X, dtype=torch.float32, device=device)
            _yt = torch.tensor(vis_y_bel.astype(np.float32), dtype=torch.float32, device=device)
            _A = _Xt.T @ _Xt
            _A.diagonal().add_(float(best_alpha))
            _W = torch.linalg.solve(_A, _Xt.T @ _yt)
            vis_belief_pred = (_Xt @ _W).cpu().numpy().reshape(N_VIS, len(vis_pos), 3)
            del vis_model, vis_streams
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if vis_pos is None:
        vis_pos = np.arange(n_steps - 1, proc.seq_len, n_steps)
    vis_belief_gt = (vis_beliefs[:, vis_pos, :] @ simplex_embedding(4))
    vis_hmm_states = vis_beliefs[:, vis_pos, :].argmax(-1)

    print(f"  [Summary] Generating summary plot...")
    generate_summary_plot(
        config_dir, job, width, seed, horizons, loss_histories, probe_df, proc, device,
        n_layers=n_layers, n_heads=n_heads,
        belief_gt=vis_belief_gt,
        belief_pred=vis_belief_pred,
        hmm_states=vis_hmm_states,
    )

    if state_path is not None:
        mark_completed(state_path, job_id)

    elapsed_total = time.time() - t0
    print(f"Job {job_id} complete in {elapsed_total:.1f}s")
    return probe_df


def worker_main():
    parser = argparse.ArgumentParser(description="BeliefPhysics single-pendulum sweep worker")
    parser.add_argument("--queue-file", type=str, required=True, help="Path to _job_queue.json")
    parser.add_argument("--output-dir", type=str, required=True, help="Base output directory")
    parser.add_argument("--worker-id", type=int, default=0, help="Worker ID for logging")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--total-tokens", type=int, default=600_000_000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--n-probe-traj", type=int, default=1024)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--ridge-alphas", type=str, default="0.1,1.0,10.0,50.0")
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=100)

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

        try:
            run_job(
                job=job,
                output_dir=output_dir,
                device=device,
                total_tokens=args.total_tokens,
                batch_size=args.batch_size,
                eval_batch_size=args.eval_batch_size,
                num_workers=args.num_workers,
                n_probe_traj=args.n_probe_traj,
                cv_folds=args.cv_folds,
                ridge_alphas=ridge_alphas,
                test_frac=args.test_frac,
                n_layers=args.n_layers,
                n_heads=args.n_heads,
                log_every=args.log_every,
                state_path=state_path,
            )
            current_job_id[0] = None
            oom_consecutive = 0
            print(f"[Worker {args.worker_id}] Job {job_id} complete")
        except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
            is_oom = isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()
            if not is_oom:
                release_job(state_path, job_id)
                current_job_id[0] = None
                raise

            oom_consecutive += 1
            print(f"\n[Worker {args.worker_id}] CAUGHT CUDA OOM on job {job_id} (count={oom_consecutive}): {exc}")

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            release_job(state_path, job_id)
            current_job_id[0] = None
            print(f"[Worker {args.worker_id}] Released job {job_id} back to queue.")

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
