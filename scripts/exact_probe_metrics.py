#!/usr/bin/env python3
"""Dedicated Layerwise Probing & Metrics Evaluation Pipeline.

Implements cross-validated Ridge regression probing across individual model layers
for two canonical modes:
1. 'usable_tokens_probing': Individual probe per layer on the k pre-perturbation
   tokens immediately preceding each action kick across cycles.
2. 'all_tokens_probing': Individual probe per layer on all sequence tokens
   across all integration steps.

Targets:
- 'belief': Posterior belief states projected onto the simplex via simplex_embedding(K)
- 'physics': Physical metric observables (e.g. double pendulum angular velocities)

Evaluation:
- 5-fold GroupKFold cross-validation grouped strictly by sequence ID (no temporal leakage).
- Evaluated against matched random_init controls to compute Net Gain (Trained - Random R²).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = EXPERIMENT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from IPython.display import HTML, clear_output, display
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from models.analysis import residual_streams_batched
from models.transformer import ModelConfig, TinyTransformer
from physics.messk import MessDriven, MessKProcess, simplex_embedding
from physics.systems.double_pendulum import DoublePendulum
from physics.systems.pendulum import Pendulum

OUTPUT_DIR = ROOT_DIR / "experiments" / "results"
ANALYSIS_CSV = OUTPUT_DIR / "analysis.csv"
RANGE_CSV = ROOT_DIR / "experiments" / "pendulum-sweeps" / "pendulum_range.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_CACHED_DF: pd.DataFrame | None = None


# ==============================================================================
# 1. Model & Physics Integration Wrappers
# ==============================================================================

class LookaheadTransformer(TinyTransformer):
    def __init__(self, config: ModelConfig, k: int):
        super().__init__(config)
        self.k = k

    def loss(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.k < 1 or self.k >= tokens.shape[1]:
            raise ValueError(f"lookahead k must satisfy 1 <= k < {tokens.shape[1]}")
        logits = self(tokens)
        return F.cross_entropy(
            logits[:, :-self.k].reshape(-1, logits.shape[-1]),
            tokens[:, self.k:].reshape(-1),
        )


@dataclass(frozen=True)
class SampledPhysicsSystem:
    base: object
    integration_dt: float

    def __getattr__(self, name: str):
        return getattr(self.base, name)

    def flow(self, z: np.ndarray, dt: float, substeps: int = 1) -> np.ndarray:
        internal_steps = round(dt / self.integration_dt)
        if internal_steps < 1 or not np.isclose(dt, internal_steps * self.integration_dt):
            raise ValueError("physics sampling gap dt must be an integer multiple of integration_dt")
        return self.base.flow(z, dt, substeps=internal_steps * substeps)


def build_process_from_meta(exp_meta: dict, hmm_meta: dict | None = None) -> MessDriven:
    phys_meta = exp_meta.get("physics", {})
    system_name = exp_meta.get("system", exp_meta.get("name", ""))

    # Support both double pendulum and single pendulum
    if "gamma1" in phys_meta or "double_pendulum" in system_name:
        dp_fields = {"gamma1", "gamma2", "omega_max", "joint1_action_gain"}
        sim_kwargs = {k: v for k, v in phys_meta.items() if k in dp_fields}
        sim = DoublePendulum(**sim_kwargs)
        delta_v = float(phys_meta.get("delta_v", 1.2))
        obs_bins = tuple(phys_meta.get("obs_bins", (50, 50)))
    else:
        pendulum_fields = {"g", "length", "gamma", "omega_max", "theta0", "omega0"}
        sim_kwargs = {k: v for k, v in phys_meta.items() if k in pendulum_fields}
        sim = Pendulum(**sim_kwargs)
        delta_v = float(phys_meta.get("delta_v", 0.55))
        obs_bins = phys_meta.get("obs_bins", 181)

    integration_dt = float(phys_meta.get("integration_dt", phys_meta.get("dt", 0.01)))
    sampled = SampledPhysicsSystem(base=sim, integration_dt=integration_dt)

    hmm_d = hmm_meta or exp_meta.get("hmm", {})
    chain = MessKProcess(
        n_states=int(hmm_d.get("n_states", 4)),
        alpha=float(hmm_d.get("alpha", 0.7)),
        stay=float(hmm_d.get("stay", 0.7)),
    )
    return MessDriven(
        chain=chain,
        system=sampled,
        delta_v=delta_v,
        m=int(exp_meta.get("m", 16)),
        n_steps=int(exp_meta.get("n", 10)),
        dt=float(phys_meta.get("dt", 0.2)),
        obs_bins=obs_bins,
    )


def sample_analysis_batch(
    proc: MessDriven,
    rng: np.random.Generator,
    n_trajectories: int,
    init_cfg: dict | None = None,
) -> dict:
    z0 = proc.system.initial_state(n_trajectories).astype(np.float64, copy=True)
    if init_cfg and init_cfg.get("mode") == "random":
        jitter = float(init_cfg.get("jitter_std", 0.1))
        z0 += rng.normal(0.0, jitter, size=z0.shape)
        if hasattr(proc.system, "omega_max"):
            z0[:, 1] = np.clip(z0[:, 1], -proc.system.omega_max, proc.system.omega_max)
    raw = proc.sample_batch(rng, n_trajectories, initial_state=z0)
    return {
        "tokens": raw["tokens"],
        "observable": raw["observable"],
        "metric": raw["metric"],
        "letters": raw["letters"],
        "beliefs": raw["beliefs"],
        "states": raw["states"],
    }


# ==============================================================================
# 2. Checkpoint Loading & Model Setup
# ==============================================================================

def discover_checkpoints(search_dirs: list[Path] | None = None) -> dict[str, dict]:
    dirs = search_dirs or [
        OUTPUT_DIR,
        ROOT_DIR / "experiments" / "results",
        ROOT_DIR / "experiments" / "outputs-smoke-v2-5k" / "results",
        ROOT_DIR / "experiments" / "outputs-three-phase",
        ROOT_DIR / "experiments",
    ]
    checkpoints = {}
    for d in dirs:
        if not d.exists():
            continue
        # Search both flat directory and subdirectories
        for json_file in sorted(list(d.glob("*_final.json")) + list(d.glob("*/*_final.json"))):
            base_name = json_file.stem.replace("_final", "")
            pt_file = json_file.with_suffix(".pt")
            if pt_file.exists() and base_name not in checkpoints:
                with open(json_file) as f:
                    meta = json.load(f)
                checkpoints[base_name] = {
                    "name": base_name,
                    "json_path": json_file,
                    "pt_path": pt_file,
                    "meta": meta,
                }
    return checkpoints


def load_models_for_run(ckpt_info: dict, device: str = DEVICE):
    meta = ckpt_info["meta"]
    cfg_d = meta.get("model", {})
    train_d = meta.get("training", meta.get("train_config", {}))
    k = meta.get("k", train_d.get("k", 1))

    # Support both schema conventions
    d_model = meta.get("d_model", train_d.get("d_model", cfg_d.get("d_model", 128)))
    d_mlp = meta.get("d_mlp", train_d.get("d_mlp", cfg_d.get("d_mlp", 512)))
    n_layers = train_d.get("n_layers", cfg_d.get("n_layers", 4))
    n_heads = train_d.get("n_heads", cfg_d.get("n_heads", 2))
    state_dict = torch.load(ckpt_info["pt_path"], map_location=device, weights_only=True)
    if "pos_emb.weight" in state_dict:
        n_ctx = int(state_dict["pos_emb.weight"].shape[0])
    else:
        exp_d = meta.get("experiment", meta.get("exp_config", {}))
        n_ctx = int(cfg_d.get("n_ctx", exp_d.get("m", 200) * exp_d.get("n", 10)))

    if "tok_emb.weight" in state_dict:
        vocab_size = int(state_dict["tok_emb.weight"].shape[0])
        d_model = int(state_dict["tok_emb.weight"].shape[1])

    seed = cfg_d.get("seed", 0)

    model_cfg = ModelConfig(
        vocab_size=vocab_size,
        n_ctx=n_ctx,
        n_layers=n_layers,
        n_heads=n_heads,
        d_model=d_model,
        d_mlp=d_mlp,
        seed=seed,
    )
    if k > 1:
        trained_model = LookaheadTransformer(model_cfg, k)
        random_model = LookaheadTransformer(model_cfg, k)
    else:
        trained_model = TinyTransformer(model_cfg)
        random_model = TinyTransformer(model_cfg)

    trained_model.load_state_dict(state_dict)
    trained_model.to(device).eval()
    random_model.to(device).eval()
    return trained_model, random_model, model_cfg, meta


# ==============================================================================
# 3. Exact Target & Feature Extraction Functions
# ==============================================================================

def extract_probe_targets_and_groups(
    batch: dict,
    proc: MessDriven,
    mode: str,
    n_steps: int = 10,
    k: int = 1,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Computes target arrays and sequence groups strictly aligned with feature extraction mode.

    Supported Modes:
    1. 'usable_tokens_probing': k pre-perturbation tokens preceding each action kick across cycles.
    2. 'all_tokens_probing': all sequence tokens across all integration steps.
    """
    N = batch["tokens"].shape[0]
    seq_len = batch["tokens"].shape[1]
    m_cycles = proc.m if hasattr(proc, "m") else (seq_len // n_steps)

    embedded_beliefs = batch["beliefs"] @ simplex_embedding(proc.chain.n_states)
    physics_metric = batch["metric"]

    if mode in ("usable_tokens_probing", "usable_tokens"):
        usable_indices = np.concatenate([
            np.arange(c * n_steps + n_steps - k, c * n_steps + n_steps)
            for c in range(m_cycles)
        ])
        targets = {
            "physics": physics_metric[:, usable_indices].reshape(N * len(usable_indices), -1),
            "belief": embedded_beliefs[:, usable_indices].reshape(N * len(usable_indices), -1),
        }
        groups = np.repeat(np.arange(N), len(usable_indices))
        return targets, groups

    elif mode in ("all_tokens_probing", "all_tokens", "single_layer_all_token"):
        targets = {
            "physics": physics_metric.reshape(N * seq_len, -1),
            "belief": embedded_beliefs.reshape(N * seq_len, -1),
        }
        groups = np.repeat(np.arange(N), seq_len)
        return targets, groups

    else:
        raise ValueError(
            f"Unsupported probe mode '{mode}'. Only 'usable_tokens_probing' and 'all_tokens_probing' are supported."
        )


def extract_probe_features(
    streams: list[np.ndarray],
    layer_idx: int,
    mode: str,
    n_steps: int = 10,
    m: int | None = None,
    k: int = 1,
) -> np.ndarray:
    """Extracts features from an individual layer stream for the specified probe mode.

    streams[layer_idx] has shape (N, seq_len, d_model).
    """
    s = streams[layer_idx]
    N, seq_len, _ = s.shape
    m_cycles = m or (seq_len // n_steps)

    if mode in ("usable_tokens_probing", "usable_tokens"):
        usable_indices = np.concatenate([
            np.arange(c * n_steps + n_steps - k, c * n_steps + n_steps)
            for c in range(m_cycles)
        ])
        return s[:, usable_indices, :].reshape(N * len(usable_indices), -1).astype(np.float32)

    elif mode in ("all_tokens_probing", "all_tokens", "single_layer_all_token"):
        return s.reshape(N * seq_len, -1).astype(np.float32)

    else:
        raise ValueError(
            f"Unsupported probe mode '{mode}'. Only 'usable_tokens_probing' and 'all_tokens_probing' are supported."
        )


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
        "dev_rows_per_feature": float(dev_mask.sum() / X.shape[1]),
    }


# ==============================================================================
# 4. Storage & CSV Helpers
# ==============================================================================

def load_range_csv(path: Path | str | None = None) -> pd.DataFrame:
    p = Path(path) if path else RANGE_CSV
    if not p.exists():
        p = ROOT_DIR / "experiments" / "pendulum-sweeps" / "pendulum_range.csv"
    if not p.exists():
        raise FileNotFoundError(f"Cannot find pendulum_range.csv at {p}")
    return pd.read_csv(p)


def make_experiment_name(dt: float, gamma: float, dv: float, n: int) -> str:
    return f"dt{dt:g}_gamma{gamma:g}_dv{dv:g}_n{n}"


def parse_experiment_name(name: str) -> dict[str, float | int]:
    m = re.search(r"(?:.*_)?dt(?P<dt>[\d\.]+)_gamma(?P<gamma>[\d\.]+)_dv(?P<dv>[\d\.]+)_n(?P<n>\d+)", name)
    if not m:
        return {}
    return {
        "dt": float(m.group("dt")),
        "gamma": float(m.group("gamma")),
        "dv": float(m.group("dv")),
        "n": int(m.group("n")),
    }


def fetch_probe_csvs(
    experiment: str | None = None,
    output_dir: Path | str = OUTPUT_DIR,
) -> pd.DataFrame | None:
    out = Path(output_dir)
    pattern = f"{experiment}*_probes.csv" if experiment else "*_probes.csv"
    files = sorted(out.glob(pattern))
    if not files:
        return None
    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_csv(f))
        except Exception:
            pass
    if not dfs:
        return None
    return pd.concat(dfs, ignore_index=True)


def load_analysis_csv(
    experiment: str | None = None,
    candidate_file: Path | str | None = None,
) -> pd.DataFrame | None:
    global _CACHED_DF
    if candidate_file is not None:
        p = Path(candidate_file)
        if p.exists():
            df = pd.read_csv(p)
            if experiment is not None and "experiment" in df.columns:
                df = df[df["experiment"] == experiment]
            _CACHED_DF = df
            return _CACHED_DF
        return None

    for candidate in [ANALYSIS_CSV, OUTPUT_DIR / "probe_results_all.csv"]:
        if candidate.exists():
            try:
                df = pd.read_csv(candidate)
                if len(df) > 0:
                    if experiment is not None and "experiment" in df.columns:
                        df = df[df["experiment"] == experiment]
                    _CACHED_DF = df
                    return _CACHED_DF
            except Exception:
                pass

    probes_df = fetch_probe_csvs(experiment=experiment)
    if probes_df is not None and len(probes_df) > 0:
        _CACHED_DF = probes_df
        return _CACHED_DF

    return None


def save_analysis_csv(df: pd.DataFrame) -> None:
    ANALYSIS_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(ANALYSIS_CSV, index=False)


# ==============================================================================
# 5. Precomputation Engine
# ==============================================================================

def run_precomputation_job(
    target_models: list[str],
    probe_usable_tokens: bool = True,
    probe_all_tokens: bool = True,
    targets: list[str] | tuple = ("belief", "physics"),
    n_traj: int = 512,
    cv_folds: int = 5,
    alphas: tuple = (0.1, 1.0, 10.0, 50.0),
    progress_callback=None,
    log_callback=None,
    **kwargs,
) -> pd.DataFrame:
    global _CACHED_DF
    checkpoints = discover_checkpoints()
    new_rows = []

    active_modes = []
    if probe_usable_tokens:
        active_modes.append("usable_tokens_probing")
    if probe_all_tokens:
        active_modes.append("all_tokens_probing")

    for model_name in target_models:
        if model_name not in checkpoints:
            continue
        ckpt_info = checkpoints[model_name]
        if log_callback:
            log_callback(f"\nProcessing model: {model_name}")

        trained_m, random_m, m_cfg, meta = load_models_for_run(ckpt_info, device=DEVICE)
        exp_meta = meta.get("experiment", meta.get("exp_config", {}))
        phys_meta = exp_meta.get("physics", {})
        proc = build_process_from_meta(exp_meta, meta.get("hmm", exp_meta.get("hmm")))
        n_steps = exp_meta.get("n", 10)

        eval_rng = np.random.default_rng(exp_meta.get("seed", 20260917) + 3)
        init_cfg = exp_meta.get("initial_state")
        batch = sample_analysis_batch(proc, eval_rng, n_traj, init_cfg=init_cfg)
        order = np.random.default_rng(exp_meta.get("seed", 20260917) + 4).permutation(n_traj)
        n_test = max(1, int(n_traj * 0.2))
        test_seqs = set(order[:n_test].tolist())

        k_lookahead = meta.get("training", meta.get("train_config", {})).get("k", 1)

        for model_tag, model_obj in (("trained", trained_m), ("random_init", random_m)):
            streams = residual_streams_batched(model_obj.to(DEVICE).eval(), batch["tokens"], DEVICE)
            n_stream_layers = len(streams)

            for mode in active_modes:
                curr_targets, curr_groups = extract_probe_targets_and_groups(
                    batch, proc, mode, n_steps=n_steps, k=k_lookahead
                )
                curr_test_mask = np.isin(curr_groups, list(test_seqs))
                curr_dev_mask = ~curr_test_mask

                for layer_idx in range(n_stream_layers):
                    depth_name = "emb" if layer_idx == 0 else f"L{layer_idx}"
                    X = extract_probe_features(
                        streams, layer_idx=layer_idx, mode=mode, n_steps=n_steps, m=proc.m, k=k_lookahead
                    )

                    for t_name in targets:
                        score = cross_validated_ridge(
                            X,
                            curr_targets[t_name],
                            curr_groups,
                            curr_dev_mask,
                            curr_test_mask,
                            ridge_alphas=alphas,
                            cv_folds=cv_folds,
                        )
                        row = {
                            "experiment": exp_meta.get("name", "standard"),
                            "system": meta.get("process", "double_pendulum_mess4" if "gamma1" in phys_meta else "pendulum_mess4"),
                            "training": meta.get("training", meta.get("train_config", {})).get("name", model_name),
                            "seed": meta.get("training", meta.get("train_config", {})).get("seed", 0),
                            "k_lookahead": k_lookahead,
                            "d_model": m_cfg.d_model,
                            "n_layers": m_cfg.n_layers,
                            "layer": layer_idx,
                            "depth_name": depth_name,
                            "model_type": model_tag,
                            "probe_mode": mode,
                            "mode": f"layer_{layer_idx}_{mode}",
                            "target": t_name,
                            **score,
                        }
                        new_rows.append(row)
                        if progress_callback:
                            progress_callback(1)
                        if log_callback:
                            log_callback(
                                f"  [{model_tag:11s}] {depth_name:>4s} | {mode:20s} | {t_name:7s} -> "
                                f"Test R²={score['test_r2']:.4f} (CV={score['cv_r2']:.4f})"
                            )

    new_df = pd.DataFrame(new_rows)
    existing_df = load_analysis_csv(candidate_file=ANALYSIS_CSV)
    if existing_df is not None and len(existing_df) > 0:
        combined = pd.concat([existing_df, new_df]).drop_duplicates(
            subset=["experiment", "training", "model_type", "mode", "target"], keep="last"
        ).reset_index(drop=True)
    else:
        combined = new_df

    _CACHED_DF = combined
    save_analysis_csv(combined)
    return combined


def run_precomputation(
    target_models: list[str] | str = "all",
    probe_usable_tokens: bool = True,
    probe_all_tokens: bool = True,
    targets: list[str] | tuple = ("belief", "physics"),
    n_traj: int = 512,
    cv_folds: int = 5,
    alphas: tuple = (0.1, 1.0, 10.0, 50.0),
    verbose: bool = True,
    **kwargs,
) -> pd.DataFrame:
    checkpoints = discover_checkpoints()
    if target_models == "all" or target_models == ["all"]:
        models = list(checkpoints.keys())
    elif isinstance(target_models, str):
        models = [target_models]
    else:
        models = list(target_models)

    return run_precomputation_job(
        target_models=models,
        probe_usable_tokens=probe_usable_tokens,
        probe_all_tokens=probe_all_tokens,
        targets=list(targets),
        n_traj=n_traj,
        cv_folds=cv_folds,
        alphas=alphas,
        log_callback=print if verbose else None,
        **kwargs,
    )


# ==============================================================================
# 6. Plotting & Reporting Functions
# ==============================================================================

# ==============================================================================
# 6. Plotting & Reporting Functions
# ==============================================================================

def plot_layerwise_emergence(
    run_name: str = "next-token",
    df: pd.DataFrame | None = None,
    targets: list[str] | tuple = ("belief", "physics"),
    probe_mode: str = "usable_tokens_probing",
    save_path: Path | str | None = None,
    experiment: str | None = None,
) -> plt.Figure | None:
    """Plots layerwise R² trajectory across residual stream depths (emb, L1..Ln) for a given probe mode."""
    current_df = df if df is not None else load_analysis_csv(experiment=experiment)
    if current_df is None or len(current_df) == 0:
        print("No analysis data available.")
        return None

    sub = current_df[
        (current_df["training"] == run_name)
        & (
            (current_df["probe_mode"] == probe_mode)
            if "probe_mode" in current_df.columns
            else current_df["mode"].str.contains(probe_mode)
        )
    ].copy()

    if experiment is not None and "experiment" in sub.columns:
        sub = sub[sub["experiment"] == experiment]
    elif "experiment" in sub.columns and sub["experiment"].nunique() > 1:
        chosen_exp = sub["experiment"].iloc[0]
        sub = sub[sub["experiment"] == chosen_exp]

    if len(sub) == 0:
        print(f"No {probe_mode} data for '{run_name}'.")
        return None

    if "layer" not in sub.columns:
        sub["layer"] = sub["mode"].apply(lambda m: int(m.split("_")[1]) if m.startswith("layer_") else 0)
    if "depth_name" not in sub.columns:
        sub["depth_name"] = sub["layer"].apply(lambda d: "emb" if d == 0 else f"L{d}")

    sub = sub.sort_values("layer")

    fig, axes = plt.subplots(1, len(targets), figsize=(7.0 * len(targets), 4.2), squeeze=False)
    model_name = f"{sub['experiment'].iloc[0]}_{run_name}" if "experiment" in sub.columns else run_name
    mode_title = "Usable Pre-Perturbation Tokens" if "usable" in probe_mode else "All Sequence Tokens"

    for c_idx, tgt in enumerate(targets):
        ax = axes[0, c_idx]
        t_sub = sub[sub["target"] == tgt]
        for m_type, style, col in [("trained", "-o", "#2980b9"), ("random_init", "--s", "#e74c3c")]:
            m_sub = t_sub[t_sub["model_type"] == m_type].sort_values("layer")
            if len(m_sub) > 0:
                ax.plot(m_sub["depth_name"], m_sub["test_r2"], style, color=col, lw=2.2, ms=6, label=m_type)
        ax.set_title(f"{tgt.capitalize()} Layerwise Emergence\n({mode_title})", fontweight="bold", fontsize=11)
        ax.set_xlabel("Residual Stream Depth")
        ax.set_ylabel("Test R²")
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(0, color="k", lw=0.8, ls="--")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(loc="lower right" if tgt == "belief" else "upper left")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=100)
    plt.show()
    return fig


def plot_modes_comparison(
    run_name: str = "next-token",
    df: pd.DataFrame | None = None,
    targets: list[str] | tuple = ("belief", "physics"),
    save_path: Path | str | None = None,
    experiment: str | None = None,
) -> plt.Figure | None:
    """Compares layerwise emergence between usable_tokens_probing and all_tokens_probing."""
    current_df = df if df is not None else load_analysis_csv(experiment=experiment)
    if current_df is None or len(current_df) == 0:
        print("No analysis data available.")
        return None

    sub = current_df[current_df["training"] == run_name].copy()
    if experiment is not None and "experiment" in sub.columns:
        sub = sub[sub["experiment"] == experiment]
    elif "experiment" in sub.columns and sub["experiment"].nunique() > 1:
        chosen_exp = sub["experiment"].iloc[0]
        sub = sub[sub["experiment"] == chosen_exp]

    if "probe_mode" not in sub.columns:
        sub["probe_mode"] = sub["mode"].apply(lambda m: "usable_tokens_probing" if "usable" in m else "all_tokens_probing")
    if "layer" not in sub.columns:
        sub["layer"] = sub["mode"].apply(lambda m: int(m.split("_")[1]) if m.startswith("layer_") else 0)
    if "depth_name" not in sub.columns:
        sub["depth_name"] = sub["layer"].apply(lambda d: "emb" if d == 0 else f"L{d}")

    fig, axes = plt.subplots(len(targets), 2, figsize=(14, 3.8 * len(targets)), squeeze=False)
    model_name = f"{sub['experiment'].iloc[0]}_{run_name}" if "experiment" in sub.columns else run_name

    mode_styles = {
        "usable_tokens_probing": ("Usable Tokens", "-o", "#27ae60", "#95a5a6"),
        "all_tokens_probing": ("All Tokens", "-s", "#2980b9", "#bdc3c7"),
    }

    for r_idx, tgt in enumerate(targets):
        t_sub = sub[sub["target"] == tgt]
        ax1, ax2 = axes[r_idx, 0], axes[r_idx, 1]

        # Left panel: Trained vs Random Init curves
        for p_mode in ["usable_tokens_probing", "all_tokens_probing"]:
            label_prefix, sym, col_tr, col_rnd = mode_styles[p_mode]
            m_sub = t_sub[t_sub["probe_mode"] == p_mode]
            m_tr = m_sub[m_sub["model_type"] == "trained"].sort_values("layer")
            m_rnd = m_sub[m_sub["model_type"] == "random_init"].sort_values("layer")
            if len(m_tr) > 0:
                ax1.plot(m_tr["depth_name"], m_tr["test_r2"], sym, color=col_tr, lw=2, ms=6, label=f"{label_prefix} (Trained)")
            if len(m_rnd) > 0:
                ax1.plot(m_rnd["depth_name"], m_rnd["test_r2"], "--" + sym[-1], color=col_rnd, lw=1.5, ms=5, label=f"{label_prefix} (Random)")

        ax1.set_title(f"Model: {model_name} | {tgt.capitalize()} Raw Test R²", fontweight="bold")
        ax1.set_xlabel("Residual Stream Depth")
        ax1.set_ylabel("Test R²")
        ax1.set_ylim(-0.05, 1.05)
        ax1.axhline(0, color="k", lw=0.8, ls="--")
        ax1.grid(axis="y", alpha=0.3)
        ax1.legend(loc="lower right" if tgt == "belief" else "upper left", fontsize=8)

        # Right panel: Net Gain (Trained - Random Init)
        for p_mode in ["usable_tokens_probing", "all_tokens_probing"]:
            label_prefix, sym, col_tr, _ = mode_styles[p_mode]
            m_sub = t_sub[t_sub["probe_mode"] == p_mode]
            piv = m_sub.pivot(index="layer", columns="model_type", values="test_r2")
            if "trained" in piv.columns and "random_init" in piv.columns:
                piv["net_gain"] = piv["trained"] - piv["random_init"]
                depth_labels = [m_sub[m_sub["layer"] == l]["depth_name"].iloc[0] for l in piv.index]
                ax2.plot(depth_labels, piv["net_gain"], sym, color=col_tr, lw=2, ms=6, label=f"{label_prefix} Net Gain")

        ax2.set_title(f"Model: {model_name} | {tgt.capitalize()} Net Gain (Trained − Random)", fontweight="bold")
        ax2.set_xlabel("Residual Stream Depth")
        ax2.set_ylabel("Δ Test R²")
        ax2.axhline(0, color="k", lw=0.8, ls="--")
        ax2.grid(axis="y", alpha=0.3)
        ax2.legend(loc="upper left", fontsize=9)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=100)
    plt.show()
    return fig


def plot_macro_overview(*args, **kwargs) -> plt.Figure | None:
    """Backwards-compatibility alias pointing to plot_modes_comparison."""
    return plot_modes_comparison(*args, **kwargs)


def plot_sweep_comparison(
    df: pd.DataFrame | None = None,
    targets: list[str] | tuple = ("belief", "physics"),
    probe_mode: str = "usable_tokens_probing",
    save_path: Path | str | None = None,
    experiment: str | None = None,
) -> plt.Figure | None:
    """Compares final layer test R² across different runs in the sweep."""
    current_df = df if df is not None else load_analysis_csv(experiment=experiment)
    if current_df is None or len(current_df) == 0:
        print("No analysis data available.")
        return None

    if experiment is not None and "experiment" in current_df.columns:
        current_df = current_df[current_df["experiment"] == experiment]

    sub = current_df[
        (
            (current_df["probe_mode"] == probe_mode)
            if "probe_mode" in current_df.columns
            else current_df["mode"].str.contains(probe_mode)
        )
    ].copy()

    if len(sub) == 0:
        print(f"No {probe_mode} sweep data available.")
        return None

    # Filter to final layer for each run
    if "layer" not in sub.columns:
        sub["layer"] = sub["mode"].apply(lambda m: int(m.split("_")[1]) if m.startswith("layer_") else 0)
    max_layer = sub["layer"].max()
    sub_final = sub[sub["layer"] == max_layer].copy()

    fig, axes = plt.subplots(len(targets), 1, figsize=(11, 4.2 * len(targets)), squeeze=False)
    index_col = ["experiment", "training"] if ("experiment" in sub_final.columns and sub_final["experiment"].nunique() > 1) else "training"
    for r_idx, tgt in enumerate(targets):
        ax = axes[r_idx, 0]
        piv = sub_final[sub_final["target"] == tgt].pivot_table(index=index_col, columns="model_type", values="test_r2")
        piv.plot.bar(ax=ax, color=["#e74c3c", "#2980b9"], width=0.6)
        ax.set_title(f"{tgt.capitalize()} Comparison Across Sweep (Final Layer: L{max_layer}, {probe_mode})", fontweight="bold")
        ax.set_ylabel("Test R²")
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(0, color="k", lw=0.8, ls="--")
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis="x", rotation=15 if index_col != "training" else 0)
        ax.legend(["Random Init", "Trained"], bbox_to_anchor=(1.02, 1), loc="upper left")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=100)
    plt.show()
    return fig


def get_metrics_table(
    run_name: str | None = None,
    df: pd.DataFrame | None = None,
    experiment: str | None = None,
) -> pd.DataFrame:
    current_df = df if df is not None else load_analysis_csv(experiment=experiment)
    if current_df is None or len(current_df) == 0:
        return pd.DataFrame()

    cols = ["experiment", "training", "model_type", "depth_name", "probe_mode", "target", "test_r2", "cv_r2", "alpha", "feature_dim"]
    sub = current_df[[c for c in cols if c in current_df.columns]]
    if experiment is not None and "experiment" in sub.columns:
        sub = sub[sub["experiment"] == experiment]
    if run_name is not None:
        sub = sub[sub["training"] == run_name]
    return sub.sort_values(["target", "probe_mode" if "probe_mode" in sub.columns else "mode", "model_type"])


# ==============================================================================
# 7. Interactive Jupyter Widgets
# ==============================================================================

def build_precomputation_widget() -> widgets.VBox:
    discovered = discover_checkpoints()
    names = list(discovered.keys())
    model_options = [("All Models", "all")] + [(n, n) for n in names]

    model_sel = widgets.SelectMultiple(
        options=model_options,
        value=("all",),
        description="Models:",
        rows=max(3, min(len(model_options), 6)),
        layout=widgets.Layout(width="420px"),
    )

    chk_usable = widgets.Checkbox(value=True, description="Usable Tokens Probing (Per Layer: emb, L1..Ln)", indent=False)
    chk_all_tokens = widgets.Checkbox(value=True, description="All Tokens Probing (Per Layer: emb, L1..Ln)", indent=False)

    chk_belief = widgets.Checkbox(value=True, description="Belief State (Simplex)", indent=False)
    chk_physics = widgets.Checkbox(value=True, description="Physical Observables", indent=False)

    slider_n_traj = widgets.IntSlider(value=512, min=128, max=2048, step=128, description="Trajectories:", layout=widgets.Layout(width="380px"))
    slider_cv_folds = widgets.IntSlider(value=5, min=2, max=10, step=1, description="CV Folds:", layout=widgets.Layout(width="380px"))
    text_alphas = widgets.Text(value="0.1, 1.0, 10.0, 50.0", description="Alphas:", layout=widgets.Layout(width="380px"))

    btn_run = widgets.Button(description="Run Precomputation", button_style="primary", icon="play", layout=widgets.Layout(width="200px", height="36px"))
    btn_load = widgets.Button(description="Load analysis.csv", button_style="info", icon="folder-open", layout=widgets.Layout(width="180px", height="36px"))

    prog = widgets.IntProgress(value=0, min=0, max=100, description="Progress:", bar_style="info", layout=widgets.Layout(width="600px"))
    status_lbl = widgets.Label(value="Ready.")
    console = widgets.Output(layout=widgets.Layout(height="200px", overflow="auto", border="1px solid #ddd", padding="6px"))

    def on_load(b):
        with console:
            clear_output()
            df = load_analysis_csv()
            if df is not None:
                status_lbl.value = f"Loaded {len(df)} records from {ANALYSIS_CSV.name}."
                display(df.head(10))
            else:
                status_lbl.value = "No existing cache found."
                print("No analysis.csv or probe_results_all.csv found in outputs/.")

    btn_load.on_click(on_load)

    def on_run(b):
        btn_run.disabled = True
        btn_load.disabled = True
        status_lbl.value = "Running cross-validated probing..."

        with console:
            clear_output()
            sel = list(model_sel.value)
            models = list(discovered.keys()) if "all" in sel else sel
            if not models:
                print("No models selected.")
                btn_run.disabled = False
                btn_load.disabled = False
                return

            targets = []
            if chk_physics.value:
                targets.append("physics")
            if chk_belief.value:
                targets.append("belief")
            if not targets:
                print("Select at least one target.")
                btn_run.disabled = False
                btn_load.disabled = False
                return

            if not chk_usable.value and not chk_all_tokens.value:
                print("Select at least one probe mode (Usable Tokens or All Tokens).")
                btn_run.disabled = False
                btn_load.disabled = False
                return

            try:
                alphas = tuple(float(x.strip()) for x in text_alphas.value.split(",") if x.strip())
            except ValueError:
                alphas = (0.1, 1.0, 10.0, 50.0)

            total_jobs = 0
            for m in models:
                meta = discovered[m]["meta"]
                cfg_d = meta.get("model", {})
                train_d = meta.get("training", meta.get("train_config", {}))
                n_layers = train_d.get("n_layers", cfg_d.get("n_layers", 4))
                layers_count = n_layers + 1
                modes_count = (1 if chk_usable.value else 0) + (1 if chk_all_tokens.value else 0)
                total_jobs += layers_count * modes_count * len(targets) * 2

            prog.max = max(1, total_jobs)
            prog.value = 0

            df = run_precomputation_job(
                target_models=models,
                probe_usable_tokens=chk_usable.value,
                probe_all_tokens=chk_all_tokens.value,
                targets=targets,
                n_traj=slider_n_traj.value,
                cv_folds=slider_cv_folds.value,
                alphas=alphas,
                progress_callback=lambda step: setattr(prog, "value", prog.value + step),
                log_callback=print,
            )
            status_lbl.value = f"Finished! Total {len(df)} records saved to analysis.csv."
            print(f"\nCompleted. Cached to {ANALYSIS_CSV.resolve()}")

        btn_run.disabled = False
        btn_load.disabled = False

    btn_run.on_click(on_run)

    return widgets.VBox([
        widgets.HTML("<h3>Layerwise Probing Controller</h3>"),
        widgets.HBox([
            model_sel,
            widgets.VBox([
                widgets.HTML("<b>Layerwise Probe Modes:</b>"),
                chk_usable,
                chk_all_tokens,
                widgets.HTML("<b>Targets:</b>"),
                chk_belief,
                chk_physics,
            ]),
        ]),
        widgets.HBox([slider_n_traj, slider_cv_folds]),
        text_alphas,
        widgets.HBox([btn_run, btn_load]),
        prog,
        status_lbl,
        widgets.HTML("<b>Log Console:</b>"),
        console,
    ])


def build_viewer_widget() -> widgets.VBox:
    df = load_analysis_csv()
    if df is None or len(df) == 0:
        return widgets.VBox([
            widgets.HTML("<h3>Results Viewer</h3>"),
            widgets.HTML("<p>No precomputed results found. Use the controller above to compute or load <code>analysis.csv</code>.</p>"),
        ])

    experiments = sorted(df["experiment"].unique().tolist()) if "experiment" in df.columns else []
    exp_dropdown = (
        widgets.Dropdown(options=experiments, value=experiments[0], description="Experiment:", layout=widgets.Layout(width="360px"))
        if len(experiments) > 1
        else None
    )

    runs = sorted(df["training"].unique().tolist())
    run_dropdown = widgets.Dropdown(options=runs, value=runs[0], description="Run:", layout=widgets.Layout(width="320px"))
    target_toggle = widgets.ToggleButtons(options=["belief", "physics", "both"], value="belief", description="Target:", button_style="info")
    btn_refresh = widgets.Button(description="Refresh Data", icon="refresh", layout=widgets.Layout(width="140px"))

    tab = widgets.Tab()
    out_usable = widgets.Output()
    out_all_tokens = widgets.Output()
    out_compare = widgets.Output()
    out_sweep = widgets.Output()
    out_table = widgets.Output()

    tab.children = [out_usable, out_all_tokens, out_compare, out_sweep, out_table]
    tab.set_title(0, "Layerwise (Usable Tokens)")
    tab.set_title(1, "Layerwise (All Tokens)")
    tab.set_title(2, "Modes Comparison")
    tab.set_title(3, "Sweep Comparison")
    tab.set_title(4, "Metrics Table")

    def render():
        current_df = _CACHED_DF if _CACHED_DF is not None else load_analysis_csv()
        if current_df is None:
            return
        run_name = run_dropdown.value
        exp_name = exp_dropdown.value if exp_dropdown is not None else None
        tgt_choice = target_toggle.value
        targets = ["belief", "physics"] if tgt_choice == "both" else [tgt_choice]

        with out_usable:
            clear_output(wait=True)
            plot_layerwise_emergence(run_name, df=current_df, targets=targets, probe_mode="usable_tokens_probing", experiment=exp_name)

        with out_all_tokens:
            clear_output(wait=True)
            plot_layerwise_emergence(run_name, df=current_df, targets=targets, probe_mode="all_tokens_probing", experiment=exp_name)

        with out_compare:
            clear_output(wait=True)
            plot_modes_comparison(run_name, df=current_df, targets=targets, experiment=exp_name)

        with out_sweep:
            clear_output(wait=True)
            plot_sweep_comparison(df=current_df, targets=targets, probe_mode="usable_tokens_probing", experiment=exp_name)

        with out_table:
            clear_output(wait=True)
            tbl = get_metrics_table(run_name, df=current_df, experiment=exp_name)
            title = f"<b>{exp_name} | {run_name}</b>" if exp_name else f"<b>{run_name}</b>"
            display(HTML(f"{title} ({len(tbl)} rows):"))
            display(tbl)

    def on_update(change=None):
        render()

    def on_refresh_clicked(b):
        load_analysis_csv()
        if _CACHED_DF is not None:
            if exp_dropdown is not None and "experiment" in _CACHED_DF.columns:
                exps_updated = sorted(_CACHED_DF["experiment"].unique().tolist())
                if exps_updated:
                    exp_dropdown.options = exps_updated
            runs_updated = sorted(_CACHED_DF["training"].unique().tolist())
            if runs_updated:
                run_dropdown.options = runs_updated
        render()

    if exp_dropdown is not None:
        exp_dropdown.observe(on_update, names="value")
    run_dropdown.observe(on_update, names="value")
    target_toggle.observe(on_update, names="value")
    btn_refresh.on_click(on_refresh_clicked)

    render()

    top_controls = [exp_dropdown, run_dropdown, target_toggle, btn_refresh] if exp_dropdown is not None else [run_dropdown, target_toggle, btn_refresh]
    return widgets.VBox([
        widgets.HTML("<h3>Interactive Results Dashboard</h3>"),
        widgets.HBox(top_controls),
        tab,
    ])


# ==============================================================================
# 8. Command-Line Execution Entrypoint
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run layerwise probe metrics extraction")
    parser.add_argument("--models", type=str, default="all", help="Models to probe ('all' or comma-separated names)")
    parser.add_argument("--n-traj", type=int, default=512, help="Number of evaluation trajectories")
    parser.add_argument("--cv-folds", type=int, default=5, help="Number of GroupKFold CV splits")
    parser.add_argument("--usable-only", action="store_true", help="Probe only usable tokens (k tokens before each kick)")
    parser.add_argument("--all-tokens-only", action="store_true", help="Probe only all sequence tokens across all steps")
    args = parser.parse_args()

    probe_usable = not args.all_tokens_only
    probe_all = not args.usable_only

    models = "all" if args.models == "all" else [m.strip() for m in args.models.split(",")]
    print(f"Discovering checkpoints from {OUTPUT_DIR}...")
    df_result = run_precomputation(
        target_models=models,
        probe_usable_tokens=probe_usable,
        probe_all_tokens=probe_all,
        n_traj=args.n_traj,
        cv_folds=args.cv_folds,
    )
    print(f"Finished extracting metrics. Results shape: {df_result.shape}")
