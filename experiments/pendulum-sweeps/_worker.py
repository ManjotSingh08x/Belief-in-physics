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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import torch

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


def cross_validated_ridge(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    dev_mask: np.ndarray,
    test_mask: np.ndarray,
    ridge_alphas: tuple = (0.1, 1.0, 10.0, 50.0),
    cv_folds: int = 5,
) -> dict:
    estimator = make_pipeline(StandardScaler(), Ridge(solver="auto"))
    search = GridSearchCV(
        estimator,
        {"ridge__alpha": list(ridge_alphas)},
        cv=GroupKFold(n_splits=cv_folds),
        scoring="r2",
        n_jobs=1,
    )
    search.fit(X[dev_mask], y[dev_mask], groups=groups[dev_mask])
    prediction = search.best_estimator_.predict(X[test_mask])
    return {
        "alpha": float(search.best_params_["ridge__alpha"]),
        "cv_r2": float(search.best_score_),
        "test_r2": float(r2_score(y[test_mask], prediction, multioutput="uniform_average")),
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
):
    plot_enhanced_summary(
        config_dir=config_dir,
        job=job,
        width=width,
        seed=seed,
        horizons=horizons,
        loss_histories=loss_histories,
        probe_df=probe_df,
        proc=proc,
        device=device,
        n_layers=n_layers,
        n_heads=n_heads,
    )

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
                                X, y, groups, d_mask, t_mask, ridge_alphas=ridge_alphas, cv_folds=args.cv_folds
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

                del trained_model, random_model, streams
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # 4. Save per-job probes CSV
            probe_df = pd.DataFrame(probe_rows)
            probe_csv = config_dir / f"d{width}_seed{seed}_probes.csv"
            probe_df.to_csv(probe_csv, index=False)
            print(f"  [Probing] Saved {len(probe_df)} rows to {probe_csv.name}")

            # 5. Generate summary plot
            print(f"  [Summary] Generating summary plot...")
            generate_summary_plot(
                config_dir, job, width, seed, horizons, loss_histories, probe_df, proc, device,
                n_layers=args.n_layers, n_heads=args.n_heads
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
