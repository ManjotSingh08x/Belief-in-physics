"""Belief State Existence Tests: Implementation and Visualization.

Provides empirical tests for the existence of belief representations
in the residual stream of transformers trained on partially observed
physical systems (MessDriven Pendulum).

Tests implemented:
  - Test A: Null-Space Probing ("Belief Beyond Physics")
  - Test B: Layer-wise Emergence Curves
  - Test C: PCA Alignment ("Does Training Concentrate Belief?")
  - Test D: Temporal Selectivity (Within-Cycle Gradient)
  - Test F: Matched-Physics Divergence Analysis
  - Test G: UMAP Geometry and Simplex Visualization

No hardcoded constants: all hyperparameters are passed as arguments.
"""

from __future__ import annotations

import gc
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
from sklearn.utils.extmath import randomized_svd
import torch

# Ensure repository root is on sys.path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SWEEP_DIR = Path(__file__).resolve().parent

from models.analysis import residual_streams_batched
from models.transformer import ModelConfig, TinyTransformer
from physics.messk import MessDriven, MessKProcess, simplex_embedding
from physics.systems.pendulum import Pendulum


# =============================================================================
# Section 1: Infrastructure
# =============================================================================

class SampledPhysicsWrapper:
    """Wraps pendulum flow to execute fixed-step substeps for numerical accuracy.
    
    Matches _worker.py's SampledPhysicsSystem behavior exactly.
    """
    def __init__(self, base: Pendulum, integration_dt: float = 0.01):
        self.base = base
        self.integration_dt = integration_dt

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base, name)

    def flow(self, z: np.ndarray, dt: float, substeps: int = 1) -> np.ndarray:
        internal_steps = round(dt / self.integration_dt)
        return self.base.flow(z, dt, substeps=internal_steps * substeps)


def build_process_from_config(config_name: str, output_dir: Path | str) -> MessDriven:
    """Build MessDriven from the JSON sidecar of any checkpoint in config_dir.

    Args:
        config_name: Subdirectory name, e.g. "dt0.2_gamma1_dv1.2_n10"
        output_dir: Path to parallel_output directory containing config subdirectories

    Returns:
        Configured MessDriven instance matching training conditions.
    """
    config_dir = Path(output_dir) / config_name
    meta_files = list(config_dir.glob("*.json"))
    if not meta_files:
        raise FileNotFoundError(f"No JSON metadata found in {config_dir}")
    # Choose first valid metadata file
    meta = json.loads(meta_files[0].read_text())

    pendulum = Pendulum(gamma=float(meta["gamma"]))
    sampled = SampledPhysicsWrapper(base=pendulum, integration_dt=0.01)
    chain = MessKProcess(n_states=4, alpha=0.7, stay=0.7)
    return MessDriven(
        chain=chain,
        system=sampled,
        delta_v=float(meta["delta_v"]),
        m=int(meta["m"]),
        n_steps=int(meta["n_steps"]),
        dt=float(meta["dt"]),
        obs_bins=int(meta.get("vocab_size", 181)),
    )


def load_model(
    config_dir: Path | str,
    d_model: int,
    k_suffix: str,
    seed: int,
    device: str,
) -> TinyTransformer:
    """Load trained TinyTransformer checkpoint from disk.

    Args:
        config_dir: Directory containing model checkpoint and json metadata.
        d_model: Width of model (e.g. 128).
        k_suffix: Horizon suffix (e.g. "kn2", "k1", "kn").
        seed: Training random seed.
        device: Torch device string (e.g. "cuda", "cpu").

    Returns:
        TinyTransformer in eval mode on target device.
    """
    c_dir = Path(config_dir)
    json_path = c_dir / f"d{d_model}_{k_suffix}_seed{seed}_final.json"
    pt_path = c_dir / f"d{d_model}_{k_suffix}_seed{seed}_final.pt"

    if not json_path.exists():
        # Fallback to search for matching file if exact name differs
        cand_json = list(c_dir.glob(f"d{d_model}_*seed{seed}_final.json"))
        if not cand_json:
            raise FileNotFoundError(f"Cannot find json metadata for d{d_model}_{k_suffix}_seed{seed} in {c_dir}")
        json_path = cand_json[0]
        pt_path = json_path.with_suffix(".pt")

    meta = json.loads(json_path.read_text())
    config = ModelConfig(
        vocab_size=meta["vocab_size"],
        n_ctx=meta["seq_len"],
        n_layers=meta["n_layers"],
        n_heads=meta.get("n_heads", 1),
        d_model=d_model,
        d_mlp=meta.get("d_mlp", 4 * d_model),
        seed=seed,
    )

    model = TinyTransformer(config)
    state = torch.load(pt_path, map_location=device)
    model.load_state_dict(state)
    return model.to(device).eval()


def load_trained_and_random(
    config_dir: Path | str,
    d_model: int,
    k_suffix: str,
    seed: int,
    device: str,
    random_seed_offset: int = 99999,
) -> tuple[TinyTransformer, TinyTransformer]:
    """Load trained checkpoint and construct an untrained random baseline model.

    Args:
        config_dir: Directory containing checkpoint files.
        d_model: Model width.
        k_suffix: Horizon suffix.
        seed: Training seed.
        device: Torch device.
        random_seed_offset: Offset added to seed for random model initialization.

    Returns:
        (trained_model, random_model) both in eval mode on device.
    """
    trained = load_model(config_dir, d_model, k_suffix, seed, device)
    
    # Construct identical architecture with un-trained weights
    cfg_dict = dict(trained.config.__dict__)
    cfg_dict["seed"] = seed + random_seed_offset
    random_config = ModelConfig(**cfg_dict)
    random_model = TinyTransformer(random_config).to(device).eval()
    return trained, random_model


def generate_data(
    proc: MessDriven,
    n: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Sample trajectories and compute derived belief and physics targets.

    Returns dict with keys:
      - "tokens": (n, seq_len) int64
      - "beliefs": (n, seq_len, 4) float64 - full 4-state probability distributions
      - "belief_coords": (n, seq_len, 3) float64 - Helmert coordinates on 3-simplex
      - "physics_state": (n, seq_len, 2) float64 - (theta_undiscretised, omega)
      - "moods": (n, seq_len) int64 - discrete hidden state
      - "metric": (n, seq_len, 1) float64 - omega values
    """
    rng = np.random.default_rng(seed)
    batch = proc.sample_batch(rng, n)
    
    H = simplex_embedding(4)  # (4, 3)
    belief_coords = batch["beliefs"] @ H  # (n, seq_len, 3)
    
    # Physical angle reconstructed from tokens
    theta = proc.undiscretise(batch["tokens"])[..., 0]  # (n, seq_len)
    omega = batch["metric"][..., 0]  # (n, seq_len)
    physics_state = np.stack([theta, omega], axis=-1)  # (n, seq_len, 2)

    return {
        **batch,
        "belief_coords": belief_coords.astype(np.float32),
        "physics_state": physics_state.astype(np.float32),
    }


def sequence_split(
    n: int,
    train_frac: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sequence-level train/test split. Returns (train_seq_idx, test_seq_idx)."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    n_train = int(n * train_frac)
    return order[:n_train], order[n_train:]


def gpu_ridge_fit(
    X_train: np.ndarray,
    y_train: np.ndarray,
    alpha: float,
    device: str,
    fit_intercept: bool = True,
) -> np.ndarray:
    """Closed-form Ridge regression weights on GPU: (X^T X + alpha * I)^(-1) X^T y.

    Args:
        X_train: (N_samples, d_features) numpy array
        y_train: (N_samples, d_targets) numpy array
        alpha: L2 regularization strength
        device: Torch device string
        fit_intercept: If True, centers X and y before fitting

    Returns:
        Weight matrix W as (d_features, d_targets) numpy array.
    """
    if fit_intercept:
        X_mean = X_train.mean(axis=0, keepdims=True)
        y_mean = y_train.mean(axis=0, keepdims=True)
        X_c = X_train - X_mean
        y_c = y_train - y_mean
    else:
        X_c = X_train
        y_c = y_train

    Xt = torch.tensor(X_c, dtype=torch.float32, device=device)
    yt = torch.tensor(y_c, dtype=torch.float32, device=device)
    
    A = Xt.T @ Xt
    A.diagonal().add_(alpha)
    W = torch.linalg.solve(A, Xt.T @ yt)
    result = W.cpu().numpy()
    
    del Xt, yt, A, W
    return result


def gpu_ridge_r2(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    alpha: float,
    device: str,
    fit_intercept: bool = True,
) -> float:
    """Fit Ridge on train, predict on test, compute mean per-column R² on GPU.

    Args:
        X_train, y_train: Training data
        X_test, y_test: Evaluation data
        alpha: L2 penalty
        device: Torch device
        fit_intercept: If True, fits unpenalized intercept via centering

    Returns:
        Mean test R² across active target dimensions.
    """
    if fit_intercept:
        X_mean = X_train.mean(axis=0, keepdims=True)
        y_mean = y_train.mean(axis=0, keepdims=True)
        X_tr = X_train - X_mean
        y_tr = y_train - y_mean
        X_te = X_test - X_mean
        y_te = y_test - y_mean
    else:
        X_tr = X_train
        y_tr = y_train
        X_te = X_test
        y_te = y_test

    Xt = torch.tensor(X_tr, dtype=torch.float32, device=device)
    yt = torch.tensor(y_tr, dtype=torch.float32, device=device)
    Xv = torch.tensor(X_te, dtype=torch.float32, device=device)
    yv = torch.tensor(y_te, dtype=torch.float32, device=device)

    A = Xt.T @ Xt
    A.diagonal().add_(alpha)
    W = torch.linalg.solve(A, Xt.T @ yt)
    pred = Xv @ W

    ss_res = ((yv - pred) ** 2).sum(dim=0)
    ss_tot = ((yv - yv.mean(dim=0, keepdim=True)) ** 2).sum(dim=0)
    live = ss_tot > 1e-10
    
    if live.any():
        r2_per_col = 1.0 - ss_res[live] / ss_tot[live]
        result = float(r2_per_col.mean())
    else:
        result = float("nan")

    del Xt, yt, Xv, yv, A, W, pred
    return result


def extract_features(
    model: TinyTransformer,
    tokens: np.ndarray,
    proc: MessDriven,
    mode: str,
    device: str,
    batch_size: int = 64,
    layer: int | None = None,
) -> np.ndarray:
    """Extract representation features from model residual stream.

    Modes:
      - "all_layers_single_token": Tick-end positions (last token before perturbation).
      - "all_layers_cycle": All tokens in each cycle concatenated horizontally.
      - "single_layer_all_tokens": Same as cycle concatenation for a single layer.
      - "all_tokens_flat": Every token position flattened independently.

    Args:
        model: Transformer model
        tokens: (N, seq_len) token IDs
        proc: MessDriven physics process
        mode: Feature extraction mode
        device: Torch device
        batch_size: Batch size for forward pass
        layer: If specified, extracts only this layer index (0=emb, 1..L).
               If None, concatenates all layers.

    Returns:
        2D numpy array of shape (n_samples, feature_dim).
    """
    depths = [layer] if layer is not None else None
    streams = residual_streams_batched(model, tokens, device, batch_size=batch_size, depths=depths)

    n_steps = proc.n_steps
    m = proc.m
    seq_len = proc.seq_len
    last_positions = np.arange(n_steps - 1, seq_len, n_steps)

    # streams is list of arrays: each (N, seq_len, d_model)
    concat = np.concatenate(streams, axis=-1)  # (N, seq_len, total_features)
    N, _, feat_dim = concat.shape

    if mode == "all_layers_single_token":
        selected = concat[:, last_positions, :]  # (N, m, feat_dim)
        return selected.reshape(N * m, feat_dim).astype(np.float32)

    elif mode in ("all_layers_cycle", "single_layer_all_tokens"):
        per_cycle = concat.reshape(N, m, n_steps, feat_dim)  # (N, m, n_steps, feat_dim)
        return per_cycle.reshape(N * m, n_steps * feat_dim).astype(np.float32)

    elif mode == "all_tokens_flat":
        return concat.reshape(N * seq_len, feat_dim).astype(np.float32)

    else:
        raise ValueError(f"Unknown extraction mode: {mode}")


def extract_targets(
    data: dict[str, np.ndarray],
    proc: MessDriven,
    mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract matched belief and physics targets for the given feature mode.

    Returns:
        (belief_targets, physics_targets):
          - For cycle/tick-level modes: (N * m, 3) and (N * m, 2)
          - For all_tokens_flat: (N * seq_len, 3) and (N * seq_len, 2)
    """
    last_positions = np.arange(proc.n_steps - 1, proc.seq_len, proc.n_steps)

    if mode in ("all_layers_single_token", "all_layers_cycle", "single_layer_all_tokens"):
        belief = data["belief_coords"][:, last_positions, :].reshape(-1, 3)
        physics = data["physics_state"][:, last_positions, :].reshape(-1, 2)
    elif mode == "all_tokens_flat":
        belief = data["belief_coords"].reshape(-1, 3)
        physics = data["physics_state"].reshape(-1, 2)
    else:
        raise ValueError(f"Unknown target mode: {mode}")

    return belief.astype(np.float32), physics.astype(np.float32)


# =============================================================================
# Section 2: Test Functions
# =============================================================================

def test_nullspace(
    model: TinyTransformer,
    data: dict[str, np.ndarray],
    proc: MessDriven,
    mode: str,
    alpha: float,
    train_frac: float,
    device: str,
    split_seed: int,
    batch_size: int = 64,
) -> dict[str, Any]:
    """Test A: Null-Space Probing — Belief Beyond Physics.

    Fits physics probe W_phys on residual features X. Projects X onto the null
    space of W_phys (removing the 2D physics subspace). Then probes the null
    space for belief. If null_belief_r2 > 0, belief is encoded orthogonally.
    """
    X = extract_features(model, data["tokens"], proc, mode, device, batch_size=batch_size)
    y_bel, y_phys = extract_targets(data, proc, mode)

    N_seq = data["tokens"].shape[0]
    train_seq, test_seq = sequence_split(N_seq, train_frac, split_seed)

    if mode in ("all_layers_single_token", "all_layers_cycle", "single_layer_all_tokens"):
        m = proc.m
        tr = (train_seq[:, None] * m + np.arange(m)).reshape(-1)
        te = (test_seq[:, None] * m + np.arange(m)).reshape(-1)
    else:
        seq_len = proc.seq_len
        tr = (train_seq[:, None] * seq_len + np.arange(seq_len)).reshape(-1)
        te = (test_seq[:, None] * seq_len + np.arange(seq_len)).reshape(-1)

    # 1. Full probes
    full_belief_r2 = gpu_ridge_r2(X[tr], y_bel[tr], X[te], y_bel[te], alpha, device)
    physics_r2 = gpu_ridge_r2(X[tr], y_phys[tr], X[te], y_phys[te], alpha, device)

    # 2. Fit physics subspace on training set
    W_phys = gpu_ridge_fit(X[tr], y_phys[tr], alpha, device, fit_intercept=True)
    # Orthonormal basis Q for physics subspace (dim d_features x 2)
    Q, _ = np.linalg.qr(W_phys)

    # 3. Project X onto the null space: X_null = X - X @ Q @ Q.T
    X_null = X - (X @ Q) @ Q.T

    # 4. Probe null space for belief
    null_belief_r2 = gpu_ridge_r2(X_null[tr], y_bel[tr], X_null[te], y_bel[te], alpha, device)

    return {
        "full_belief_r2": full_belief_r2,
        "null_belief_r2": null_belief_r2,
        "physics_r2": physics_r2,
        "feature_dim": X.shape[1],
        "null_feature_dim": X.shape[1] - Q.shape[1],
        "n_train": len(tr),
        "n_test": len(te),
        "mode": mode,
    }


def test_layer_emergence(
    model: TinyTransformer,
    data: dict[str, np.ndarray],
    proc: MessDriven,
    variation: str,
    alpha: float,
    train_frac: float,
    device: str,
    split_seed: int,
    batch_size: int = 64,
) -> list[dict[str, Any]]:
    """Test B: Layer-wise Emergence Curves.

    Evaluates R² separately for each layer (embedding + layers 1..n_layers)
    plus all-layers concatenated, across 3 variations:
      - "all_tokens"
      - "cycle_concat"
      - "last_token"
    """
    n_layers = model.config.n_layers
    layers_to_test = list(range(n_layers + 1))  # 0=emb, 1..n_layers

    # Map variation to feature extraction mode
    if variation == "all_tokens":
        mode_for_layer = "all_tokens_flat"
        target_mode = "all_tokens_flat"
    elif variation == "cycle_concat":
        mode_for_layer = "all_layers_cycle"
        target_mode = "all_layers_cycle"
    elif variation == "last_token":
        mode_for_layer = "all_layers_single_token"
        target_mode = "all_layers_single_token"
    else:
        raise ValueError(f"Unknown variation: {variation}")

    y_bel, y_phys = extract_targets(data, proc, target_mode)

    N_seq = data["tokens"].shape[0]
    train_seq, test_seq = sequence_split(N_seq, train_frac, split_seed)
    if target_mode == "all_tokens_flat":
        seq_len = proc.seq_len
        tr = (train_seq[:, None] * seq_len + np.arange(seq_len)).reshape(-1)
        te = (test_seq[:, None] * seq_len + np.arange(seq_len)).reshape(-1)
    else:
        m = proc.m
        tr = (train_seq[:, None] * m + np.arange(m)).reshape(-1)
        te = (test_seq[:, None] * m + np.arange(m)).reshape(-1)

    results = []

    # Individual layers
    for lyr in layers_to_test:
        X_lyr = extract_features(
            model, data["tokens"], proc, mode_for_layer, device,
            batch_size=batch_size, layer=lyr,
        )
        b_r2 = gpu_ridge_r2(X_lyr[tr], y_bel[tr], X_lyr[te], y_bel[te], alpha, device)
        p_r2 = gpu_ridge_r2(X_lyr[tr], y_phys[tr], X_lyr[te], y_phys[te], alpha, device)
        name = "emb" if lyr == 0 else f"L{lyr}"
        results.append({
            "layer": name,
            "layer_idx": lyr,
            "belief_r2": b_r2,
            "physics_r2": p_r2,
            "variation": variation,
        })
        del X_lyr

    # All layers concatenated
    X_all = extract_features(
        model, data["tokens"], proc, mode_for_layer, device,
        batch_size=batch_size, layer=None,
    )
    b_r2_all = gpu_ridge_r2(X_all[tr], y_bel[tr], X_all[te], y_bel[te], alpha, device)
    p_r2_all = gpu_ridge_r2(X_all[tr], y_phys[tr], X_all[te], y_phys[te], alpha, device)
    results.append({
        "layer": "all",
        "layer_idx": n_layers + 1,
        "belief_r2": b_r2_all,
        "physics_r2": p_r2_all,
        "variation": variation,
    })
    del X_all
    gc.collect()

    return results


def test_pca_alignment(
    model: TinyTransformer,
    data: dict[str, np.ndarray],
    proc: MessDriven,
    feature_config: str,
    alpha: float,
    train_frac: float,
    device: str,
    split_seed: int,
    max_pcs: int = 50,
    pc_steps: list[int] | None = None,
    batch_size: int = 64,
) -> dict[str, Any]:
    """Test C: PCA Alignment — Does Training Concentrate Belief?

    Projects residual representations onto the top k principal components
    and measures cumulative R² as a function of k.
    """
    layer_arg = None
    mode = feature_config
    if feature_config == "single_layer_all_tokens":
        layer_arg = model.config.n_layers  # default to last layer

    X = extract_features(
        model, data["tokens"], proc, mode, device,
        batch_size=batch_size, layer=layer_arg,
    )
    y_bel, y_phys = extract_targets(data, proc, mode)

    N_seq = data["tokens"].shape[0]
    m = proc.m
    train_seq, test_seq = sequence_split(N_seq, train_frac, split_seed)
    tr = (train_seq[:, None] * m + np.arange(m)).reshape(-1)
    te = (test_seq[:, None] * m + np.arange(m)).reshape(-1)

    # Center training data
    X_mean = X[tr].mean(axis=0, keepdims=True)
    X_c_tr = X[tr] - X_mean
    X_c_te = X[te] - X_mean

    d_feat = X.shape[1]
    n_comp = min(max_pcs, d_feat, len(tr))

    # Randomized SVD for speed and memory efficiency
    _, S, Vt = randomized_svd(
        X_c_tr,
        n_components=n_comp,
        n_iter=4,
        random_state=split_seed,
    )

    if pc_steps is None:
        raw_steps = [1, 2, 3, 5, 10, 15, 20, 30, 40, 50, 75, 100]
    else:
        raw_steps = pc_steps
    valid_steps = [k for k in raw_steps if k <= n_comp]
    if n_comp not in valid_steps:
        valid_steps.append(n_comp)
    valid_steps = sorted(list(set(valid_steps)))

    total_var = float(np.sum(X_c_tr ** 2))
    cum_var = np.cumsum(S ** 2) / (total_var + 1e-10)

    bel_r2_list = []
    phys_r2_list = []
    var_exp_list = []

    for k in valid_steps:
        Vk = Vt[:k].T  # (d_feat, k)
        X_tr_proj = X_c_tr @ Vk
        X_te_proj = X_c_te @ Vk

        b_r2 = gpu_ridge_r2(X_tr_proj, y_bel[tr], X_te_proj, y_bel[te], alpha, device)
        p_r2 = gpu_ridge_r2(X_tr_proj, y_phys[tr], X_te_proj, y_phys[te], alpha, device)
        bel_r2_list.append(b_r2)
        phys_r2_list.append(p_r2)
        var_exp_list.append(float(cum_var[k - 1]))

    return {
        "n_pcs": valid_steps,
        "belief_r2": bel_r2_list,
        "physics_r2": phys_r2_list,
        "explained_variance_ratio": var_exp_list,
        "total_feature_dim": d_feat,
        "feature_config": feature_config,
    }


def test_temporal_selectivity(
    model: TinyTransformer,
    data: dict[str, np.ndarray],
    proc: MessDriven,
    alpha: float,
    train_frac: float,
    device: str,
    split_seed: int,
    batch_size: int = 64,
) -> dict[str, Any]:
    """Test D: Temporal Selectivity (Within-Cycle Gradient).

    Evaluates belief and physics decodability across token positions
    t = 0, 1, ..., n_steps - 1 within each cycle.
    """
    streams = residual_streams_batched(model, data["tokens"], device, batch_size=batch_size)
    concat = np.concatenate(streams, axis=-1)  # (N, seq_len, L*d_model)
    N_seq, seq_len, d_feat = concat.shape
    n_steps, m = proc.n_steps, proc.m

    train_seq, test_seq = sequence_split(N_seq, train_frac, split_seed)
    tr = (train_seq[:, None] * m + np.arange(m)).reshape(-1)
    te = (test_seq[:, None] * m + np.arange(m)).reshape(-1)

    positions = list(range(n_steps))
    belief_r2s = []
    physics_r2s = []

    for t in positions:
        # Step t across all m cycles
        pos_t = np.arange(t, seq_len, n_steps)
        X_t = concat[:, pos_t, :].reshape(N_seq * m, d_feat).astype(np.float32)
        y_bel_t = data["belief_coords"][:, pos_t, :].reshape(N_seq * m, 3).astype(np.float32)
        y_phys_t = data["physics_state"][:, pos_t, :].reshape(N_seq * m, 2).astype(np.float32)

        b_r2 = gpu_ridge_r2(X_t[tr], y_bel_t[tr], X_t[te], y_bel_t[te], alpha, device)
        p_r2 = gpu_ridge_r2(X_t[tr], y_phys_t[tr], X_t[te], y_phys_t[te], alpha, device)
        belief_r2s.append(b_r2)
        physics_r2s.append(p_r2)

    del streams, concat
    gc.collect()

    return {
        "positions": positions,
        "belief_r2": belief_r2s,
        "physics_r2": physics_r2s,
    }


def test_matched_physics(
    model: TinyTransformer,
    data: dict[str, np.ndarray],
    proc: MessDriven,
    cycle_range: tuple[int, int] = (30, 45),
    epsilon: float = 0.05,
    belief_threshold: float = 0.3,
    device: str = "cpu",
    batch_size: int = 64,
    random_seed: int = 42,
) -> dict[str, Any]:
    """Test F: Matched-Physics Divergence Analysis.

    Identifies pairs of timesteps where physical state (theta, omega) is identical
    within epsilon, but hidden belief state differs by > belief_threshold.
    Compares residual stream L2 divergence for matched vs random unmatched pairs.
    """
    streams = residual_streams_batched(model, data["tokens"], device, batch_size=batch_size)
    concat = np.concatenate(streams, axis=-1)  # (N, seq_len, total_features)
    N_seq = concat.shape[0]
    n_steps = proc.n_steps

    # Timestep indices: last token before perturbation for cycles in cycle_range
    c_start, c_end = cycle_range
    selected_cycles = np.arange(c_start, min(c_end, proc.m))
    last_tokens = selected_cycles * n_steps + (n_steps - 1)

    X_sub = concat[:, last_tokens, :].reshape(-1, concat.shape[-1]).astype(np.float32)
    bel_sub = data["belief_coords"][:, last_tokens, :].reshape(-1, 3).astype(np.float32)
    phys_sub = data["physics_state"][:, last_tokens, :].reshape(-1, 2).astype(np.float32)

    # Normalize physics to [0, 1] for balanced distance query
    theta_min, theta_max = proc.obs_ranges[0]
    omega_max = proc.system.omega_max
    phys_norm = np.column_stack([
        (phys_sub[:, 0] - theta_min) / (theta_max - theta_min + 1e-10),
        (phys_sub[:, 1] + omega_max) / (2 * omega_max + 1e-10),
    ])

    tree = cKDTree(phys_norm)
    # L_infinity ball search
    raw_pairs = list(tree.query_pairs(r=epsilon, p=np.inf))

    if not raw_pairs:
        return {
            "n_matched_pairs": 0,
            "matched_resid_l2": np.array([], dtype=np.float32),
            "control_resid_l2": np.array([], dtype=np.float32),
            "matched_belief_l1": np.array([], dtype=np.float32),
            "matched_physics_linf": np.array([], dtype=np.float32),
            "mean_matched_l2": float("nan"),
            "mean_control_l2": float("nan"),
            "cycle_range": cycle_range,
            "epsilon": epsilon,
            "belief_threshold": belief_threshold,
        }

    pairs_arr = np.array(raw_pairs, dtype=np.int64)
    # Subsample candidate pairs if massive
    max_pairs = 100000
    if len(pairs_arr) > max_pairs:
        rng_pairs = np.random.default_rng(random_seed)
        pairs_arr = pairs_arr[rng_pairs.choice(len(pairs_arr), max_pairs, replace=False)]

    idx_i = pairs_arr[:, 0]
    idx_j = pairs_arr[:, 1]

    # Vectorized belief L1 distance
    b_dists = np.sum(np.abs(bel_sub[idx_i] - bel_sub[idx_j]), axis=1)
    mask = b_dists >= belief_threshold

    matched_i = idx_i[mask]
    matched_j = idx_j[mask]
    n_matched = len(matched_i)

    if n_matched > 0:
        matched_belief_l1 = b_dists[mask]
        matched_physics_linf = np.max(np.abs(phys_norm[matched_i] - phys_norm[matched_j]), axis=1)
        matched_resid_l2 = np.linalg.norm(X_sub[matched_i] - X_sub[matched_j], axis=1)

        # Control: random unmatched pairs
        rng_ctrl = np.random.default_rng(random_seed + 1)
        ctrl_i = rng_ctrl.integers(0, len(X_sub), size=n_matched)
        ctrl_j = rng_ctrl.integers(0, len(X_sub), size=n_matched)
        control_resid_l2 = np.linalg.norm(X_sub[ctrl_i] - X_sub[ctrl_j], axis=1)

        mean_matched = float(np.mean(matched_resid_l2))
        mean_control = float(np.mean(control_resid_l2))
    else:
        matched_belief_l1 = np.array([], dtype=np.float32)
        matched_physics_linf = np.array([], dtype=np.float32)
        matched_resid_l2 = np.array([], dtype=np.float32)
        control_resid_l2 = np.array([], dtype=np.float32)
        mean_matched = float("nan")
        mean_control = float("nan")

    del streams, concat
    gc.collect()

    return {
        "n_matched_pairs": n_matched,
        "matched_resid_l2": matched_resid_l2.astype(np.float32),
        "control_resid_l2": control_resid_l2.astype(np.float32),
        "matched_belief_l1": matched_belief_l1.astype(np.float32),
        "matched_physics_linf": matched_physics_linf.astype(np.float32),
        "mean_matched_l2": mean_matched,
        "mean_control_l2": mean_control,
        "cycle_range": cycle_range,
        "epsilon": epsilon,
        "belief_threshold": belief_threshold,
    }


def test_umap_geometry(
    model: TinyTransformer,
    data: dict[str, np.ndarray],
    proc: MessDriven,
    feature_config: str,
    n_epochs: int = 20,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    seed: int = 42,
    max_points: int = 5000,
    device: str = "cpu",
    batch_size: int = 64,
) -> dict[str, Any]:
    """Test G: UMAP Geometry and Simplex Representation.

    Reduces residual stream features to 2D via UMAP, preserving local geometry.
    Ground-truth belief probabilities are paired for multi-color simplex visualization.
    """
    import umap  # Lazy import

    layer_arg = None
    if feature_config == "single_layer_all_tokens":
        layer_arg = model.config.n_layers

    X = extract_features(
        model, data["tokens"], proc, feature_config, device,
        batch_size=batch_size, layer=layer_arg,
    )

    last_positions = np.arange(proc.n_steps - 1, proc.seq_len, proc.n_steps)
    if feature_config in ("all_layers_single_token", "all_layers_cycle", "single_layer_all_tokens"):
        beliefs = data["beliefs"][:, last_positions, :].reshape(-1, 4)
    else:
        beliefs = data["beliefs"].reshape(-1, 4)

    # Subsample if necessary
    N_pts = len(X)
    if N_pts > max_points:
        rng = np.random.default_rng(seed)
        sub_idx = rng.choice(N_pts, size=max_points, replace=False)
        X = X[sub_idx]
        beliefs = beliefs[sub_idx]

    reducer = umap.UMAP(
        n_components=2,
        n_epochs=n_epochs,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=seed,
    )
    embedding = reducer.fit_transform(X.astype(np.float32))

    return {
        "embedding": embedding,
        "beliefs": beliefs.astype(np.float32),
        "moods": beliefs.argmax(axis=1),
        "feature_config": feature_config,
        "seed": seed,
    }


# =============================================================================
# Section 3: Multi-Seed Runner
# =============================================================================

def run_test_multiseed(
    test_fn: Callable[..., Any],
    test_kwargs: dict[str, Any],
    data_seeds: list[int],
    proc: MessDriven,
    n_data: int,
) -> list[Any]:
    """Execute a test function across multiple data generation seeds.

    Args:
        test_fn: The test function (e.g. test_nullspace, test_layer_emergence)
        test_kwargs: Keyword arguments for the test (excluding data and proc)
        data_seeds: List of seeds to evaluate
        proc: MessDriven process
        n_data: Number of sample trajectories per seed

    Returns:
        List of results returned by test_fn, one per seed.
    """
    results = []
    for s in data_seeds:
        data = generate_data(proc, n_data, s)
        res = test_fn(data=data, proc=proc, **test_kwargs)
        results.append(res)
        del data
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return results


# =============================================================================
# Section 4: Plotting & Visualization
# =============================================================================

# Distinct, vibrant colors for the 4 hidden belief states
# State 0: Coral Red, State 1: Sky Blue, State 2: Emerald Green, State 3: Amber Orange
STATE_COLORS = np.array([
    [0.906, 0.298, 0.235, 1.0],  # #e74c3c
    [0.204, 0.596, 0.859, 1.0],  # #3498db
    [0.180, 0.800, 0.443, 1.0],  # #2ecc71
    [0.953, 0.612, 0.071, 1.0],  # #f39c12
], dtype=np.float32)


def belief_colors(beliefs: np.ndarray) -> np.ndarray:
    """Map (N, 4) belief probabilities to smooth RGBA colors via convex combination."""
    # Normalize beliefs to ensure valid probability simplex
    b_norm = beliefs / np.maximum(beliefs.sum(axis=-1, keepdims=True), 1e-10)
    cols = b_norm @ STATE_COLORS
    return np.clip(cols, 0.0, 1.0).astype(np.float32)


def plot_multi_config(
    results: dict[str, Any],
    plot_fn: Callable[[np.ndarray, str, Any, int], None],
    n_plots_per_config: int,
    max_cols: int = 4,
    figsize_per_subplot: tuple[float, float] = (4.5, 3.5),
    title: str = "",
) -> tuple[plt.Figure, np.ndarray]:
    """Generic layout manager: configs vertically stacked, tests arranged horizontally.

    Each config is allocated ceil(n_plots_per_config / max_cols) rows.
    """
    n_configs = len(results)
    n_cols = min(n_plots_per_config, max_cols)
    rows_per_cfg = -(-n_plots_per_config // n_cols)
    total_rows = n_configs * rows_per_cfg

    fig, axes_grid = plt.subplots(
        total_rows,
        n_cols,
        figsize=(figsize_per_subplot[0] * n_cols, figsize_per_subplot[1] * total_rows),
        squeeze=False,
    )

    for i, (cfg_name, res) in enumerate(results.items()):
        row_start = i * rows_per_cfg
        cfg_axes = axes_grid[row_start : row_start + rows_per_cfg].flatten()
        plot_fn(cfg_axes, cfg_name, res, i)

        # Hide extra unused subplot axes if any
        for j in range(n_plots_per_config, len(cfg_axes)):
            cfg_axes[j].set_visible(False)

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig, axes_grid


def plot_nullspace(
    axes: np.ndarray,
    config_name: str,
    results_by_seed: dict[str, list[dict[str, Any]]],
    config_idx: int,
) -> None:
    """Grouped bar chart for Test A (Trained vs Random model null-space probing).

    results_by_seed should contain:
      {"trained": [result_dict_seed1, ...], "random": [result_dict_seed1, ...]}
    """
    trained_res = results_by_seed["trained"]
    random_res = results_by_seed["random"]

    categories = ["Full Belief R²", "Null Belief R²\n(Orthogonal)", "Physics R²"]
    metrics = ["full_belief_r2", "null_belief_r2", "physics_r2"]
    bar_colors = ["#3498db", "#9b59b6", "#e67e22"]

    for ax_idx, (model_type, r_list) in enumerate([("Trained Model", trained_res), ("Random Baseline", random_res)]):
        ax = axes[ax_idx]
        means = [np.mean([r[m] for r in r_list]) for m in metrics]
        stds = [np.std([r[m] for r in r_list]) for m in metrics]

        x = np.arange(len(categories))
        bars = ax.bar(x, means, yerr=stds, capsize=5, color=bar_colors, alpha=0.85, edgecolor="black", width=0.55)

        for bar, mean_val in zip(bars, means):
            y_pos = max(mean_val, 0) + 0.03
            ax.text(bar.get_x() + bar.get_width() / 2, y_pos, f"{mean_val:.2f}", ha="center", va="bottom", fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(categories, fontsize=8)
        ax.set_ylim(-0.1, 1.05)
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_ylabel("Probe $R^2$")
        ax.set_title(f"{config_name}\n[{model_type}]", fontsize=10, fontweight="semibold")
        ax.grid(axis="y", linestyle=":", alpha=0.6)


def plot_layer_emergence(
    axes: np.ndarray,
    config_name: str,
    results_by_seed: dict[str, dict[str, list[list[dict[str, Any]]]]],
    config_idx: int,
) -> None:
    """Line plots for Test B: Emergence across layers for 3 variations (Trained vs Random).

    results_by_seed contains:
      {"trained": {var: [seeds...]}, "random": {var: [seeds...]}}
    """
    variations = ["last_token", "cycle_concat", "all_tokens"]
    var_labels = {"last_token": "Last Token", "cycle_concat": "Cycle Concat", "all_tokens": "All Tokens"}
    var_colors = {"last_token": "#2980b9", "cycle_concat": "#27ae60", "all_tokens": "#8e44ad"}

    for ax_idx, model_type in enumerate(["trained", "random"]):
        ax = axes[ax_idx]
        var_data = results_by_seed[model_type]

        for var in variations:
            if var not in var_data:
                continue
            seed_runs = var_data[var]  # list of lists
            # Layer names from first run
            layer_names = [d["layer"] for d in seed_runs[0]]
            x = np.arange(len(layer_names))

            # Matrix of shape (n_seeds, n_layers)
            b_r2s = np.array([[d["belief_r2"] for d in run] for run in seed_runs])
            mean_b = b_r2s.mean(axis=0)
            std_b = b_r2s.std(axis=0)

            c = var_colors[var]
            ax.plot(x, mean_b, marker="o", label=f"Belief ({var_labels[var]})", color=c, linewidth=1.8)
            ax.fill_between(x, mean_b - std_b, mean_b + std_b, color=c, alpha=0.18)

        ax.set_xticks(x)
        ax.set_xticklabels(layer_names, fontsize=9)
        ax.set_ylabel("Probe $R^2$")
        ax.set_ylim(-0.1, 1.05)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.set_title(f"{config_name}\n[{model_type.capitalize()} Model]", fontsize=10, fontweight="semibold")
        if ax_idx == 0:
            ax.legend(fontsize=7, loc="upper left")


def plot_pca_alignment(
    axes: np.ndarray,
    config_name: str,
    results_by_seed: dict[str, dict[str, list[dict[str, Any]]]],
    config_idx: int,
) -> None:
    """Line plots for Test C: Cumulative R² vs # PCs across feature configs."""
    configs = ["all_layers_cycle", "all_layers_single_token", "single_layer_all_tokens"]
    titles = {
        "all_layers_cycle": "All Layers + Cycle Concat",
        "all_layers_single_token": "All Layers + Single Token",
        "single_layer_all_tokens": "Single Layer + Cycle Concat",
    }

    present = [c for c in configs if c in results_by_seed.get("trained", {})]
    target_configs = present if present else configs

    for ax_idx, feat_cfg in enumerate(target_configs[:len(axes)]):
        ax = axes[ax_idx]
        tr_runs = results_by_seed.get("trained", {}).get(feat_cfg, [])
        rnd_runs = results_by_seed.get("random", {}).get(feat_cfg, [])

        if not tr_runs:
            continue

        pcs = tr_runs[0]["n_pcs"]

        # Trained model
        tr_bel = np.array([r["belief_r2"] for r in tr_runs])
        tr_mean, tr_std = tr_bel.mean(axis=0), tr_bel.std(axis=0)
        ax.plot(pcs, tr_mean, marker="o", label="Trained Belief", color="#2980b9", linewidth=1.8)
        ax.fill_between(pcs, tr_mean - tr_std, tr_mean + tr_std, color="#2980b9", alpha=0.2)

        # Random model
        if rnd_runs:
            rnd_bel = np.array([r["belief_r2"] for r in rnd_runs])
            rnd_mean, rnd_std = rnd_bel.mean(axis=0), rnd_bel.std(axis=0)
            ax.plot(pcs, rnd_mean, marker="s", label="Random Belief", color="#7f8c8d", linestyle="--", linewidth=1.5)
            ax.fill_between(pcs, rnd_mean - rnd_std, rnd_mean + rnd_std, color="#7f8c8d", alpha=0.15)

        ax.set_xlabel("Number of PCs", fontsize=8)
        ax.set_ylabel("Belief Probe $R^2$", fontsize=8)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(f"{config_name}\n{titles.get(feat_cfg, feat_cfg)}", fontsize=9, fontweight="semibold")
        ax.grid(True, linestyle=":", alpha=0.6)
        if ax_idx == 0:
            ax.legend(fontsize=7, loc="lower right")


def plot_temporal_selectivity(
    axes: np.ndarray,
    config_name: str,
    results_by_seed: list[dict[str, Any]],
    config_idx: int,
) -> None:
    """Plots for Test D: Decodability within cycle (Belief on Left, Physics on Right)."""
    positions = results_by_seed[0]["positions"]
    bel_r2s = np.array([r["belief_r2"] for r in results_by_seed])
    phys_r2s = np.array([r["physics_r2"] for r in results_by_seed])

    # Belief
    ax0 = axes[0]
    b_mean, b_std = bel_r2s.mean(axis=0), bel_r2s.std(axis=0)
    ax0.plot(positions, b_mean, marker="o", color="#8e44ad", linewidth=2.0, label="Belief $R^2$")
    ax0.fill_between(positions, b_mean - b_std, b_mean + b_std, color="#8e44ad", alpha=0.2)
    ax0.set_xlabel("Position within Cycle (0=Kick, 9=Perturb)", fontsize=8)
    ax0.set_ylabel("Belief $R^2$")
    ax0.set_ylim(-0.05, 1.05)
    ax0.set_title(f"{config_name}\nBelief Decodability", fontsize=10, fontweight="semibold")
    ax0.grid(True, linestyle=":", alpha=0.6)

    # Physics
    if len(axes) > 1:
        ax1 = axes[1]
        p_mean, p_std = phys_r2s.mean(axis=0), phys_r2s.std(axis=0)
        ax1.plot(positions, p_mean, marker="s", color="#d35400", linewidth=2.0, label="Physics $R^2$")
        ax1.fill_between(positions, p_mean - p_std, p_mean + p_std, color="#d35400", alpha=0.2)
        ax1.set_xlabel("Position within Cycle (0=Kick, 9=Perturb)", fontsize=8)
        ax1.set_ylabel("Physics $R^2$")
        ax1.set_ylim(-0.05, 1.05)
        ax1.set_title(f"{config_name}\nPhysics Decodability", fontsize=10, fontweight="semibold")
        ax1.grid(True, linestyle=":", alpha=0.6)


def plot_matched_physics(
    axes: np.ndarray,
    config_name: str,
    results_by_seed: list[dict[str, Any]],
    config_idx: int,
) -> None:
    """Plots for Test F: Matched vs Random residual stream divergence (Histogram & Summary)."""
    # Combine distributions across seeds
    all_matched = np.concatenate([r["matched_resid_l2"] for r in results_by_seed if len(r["matched_resid_l2"]) > 0])
    all_control = np.concatenate([r["control_resid_l2"] for r in results_by_seed if len(r["control_resid_l2"]) > 0])

    # Left: Histogram
    ax0 = axes[0]
    if len(all_matched) > 0 and len(all_control) > 0:
        bins = np.linspace(0, max(all_matched.max(), all_control.max()), 35)
        ax0.hist(all_matched, bins=bins, alpha=0.6, color="#e74c3c", label=f"Matched Physics (N={len(all_matched)})", density=True)
        ax0.hist(all_control, bins=bins, alpha=0.5, color="#34495e", label=f"Random Control (N={len(all_control)})", density=True)
        ax0.set_xlabel("Residual Stream Distance ($L_2$)", fontsize=8)
        ax0.set_ylabel("Density", fontsize=8)
        ax0.legend(fontsize=7)
    else:
        ax0.text(0.5, 0.5, "No matched pairs found\n(increase epsilon/n_trajectories)", ha="center", va="center")
    ax0.set_title(f"{config_name}\nDivergence Distribution", fontsize=10, fontweight="semibold")
    ax0.grid(True, linestyle=":", alpha=0.6)

    # Right: Bar summary
    if len(axes) > 1:
        ax1 = axes[1]
        matched_means = [r["mean_matched_l2"] for r in results_by_seed if not np.isnan(r["mean_matched_l2"])]
        control_means = [r["mean_control_l2"] for r in results_by_seed if not np.isnan(r["mean_control_l2"])]

        if matched_means and control_means:
            m_mean, m_std = np.mean(matched_means), np.std(matched_means)
            c_mean, c_std = np.mean(control_means), np.std(control_means)
            bars = ax1.bar(["Matched\nPhysics", "Random\nControl"], [m_mean, c_mean], yerr=[m_std, c_std], capsize=5, color=["#e74c3c", "#34495e"], alpha=0.85, width=0.5)
            for b, v in zip(bars, [m_mean, c_mean]):
                ax1.text(b.get_x() + b.get_width() / 2, v * 0.95, f"{v:.2f}", ha="center", va="top", color="white", fontweight="bold", fontsize=9)
            ax1.set_ylabel("Mean $L_2$ Distance")
        ax1.set_title(f"{config_name}\nMean Separation", fontsize=10, fontweight="semibold")
        ax1.grid(axis="y", linestyle=":", alpha=0.6)


def plot_umap_geometry(
    axes: np.ndarray,
    config_name: str,
    results_dict: dict[str, list[dict[str, Any]]],
    config_idx: int,
) -> None:
    """Scatter plots for Test G: 2D UMAP colored by ground-truth belief distribution."""
    configs = ["all_layers_cycle", "all_layers_single_token", "single_layer_all_tokens"]
    titles = {
        "all_layers_cycle": "All Layers + Cycle Concat",
        "all_layers_single_token": "All Layers + Single Token",
        "single_layer_all_tokens": "Single Layer + Cycle Concat",
    }

    present = [c for c in configs if c in results_dict]
    target_configs = present if present else configs

    for ax_idx, feat_cfg in enumerate(target_configs[:len(axes)]):
        ax = axes[ax_idx]
        runs = results_dict.get(feat_cfg, [])
        if not runs:
            ax.set_visible(False)
            continue

        # Plot first seed run
        res = runs[0]
        emb = res["embedding"]
        beliefs = res["beliefs"]
        colors = belief_colors(beliefs)

        ax.scatter(emb[:, 0], emb[:, 1], c=colors, s=4, alpha=0.65, edgecolors="none")
        ax.set_title(f"{config_name}\n{titles.get(feat_cfg, feat_cfg)}", fontsize=9, fontweight="semibold")
        ax.set_xticks([])
        ax.set_yticks([])

        # Simplex color legend on the first subplot
        if ax_idx == 0:
            patches = [
                mpatches.Patch(color=STATE_COLORS[0], label="State 0 (Red)"),
                mpatches.Patch(color=STATE_COLORS[1], label="State 1 (Blue)"),
                mpatches.Patch(color=STATE_COLORS[2], label="State 2 (Green)"),
                mpatches.Patch(color=STATE_COLORS[3], label="State 3 (Amber)"),
            ]
            ax.legend(handles=patches, loc="upper right", fontsize=6, framealpha=0.7)


# =============================================================================
# Section 5: Future Stubs
# =============================================================================

def test_cross_config_transfer(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Test E (Future TODO): Cross-configuration probe transfer."""
    raise NotImplementedError("Cross-configuration transfer planned for Phase 3.")


def test_mutual_information(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Test H (Future TODO): Mutual Information estimation between belief and residual stream."""
    raise NotImplementedError("Mutual Information estimation planned for Phase 3.")
