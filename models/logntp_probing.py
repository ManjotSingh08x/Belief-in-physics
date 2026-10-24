"""Layer-wise Residual Stream Probing to log Next-Token Probability (logNTP).

Investigates where and how linearly next-token prediction emerges across transformer layers.
Provides extraction, ridge probe fitting, evaluation against random untrained controls,
and intra-cycle decomposition.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F

from .transformer import ModelConfig, TinyTransformer


@dataclass
class LogNTPProbe:
    """Linear probe mapping layer activations to logNTP targets: y_hat = a @ W + b."""

    weight: np.ndarray  # shape [d_model, d_out]
    bias: np.ndarray  # shape [d_out]
    depth: int
    name: str

    def predict(self, activations: np.ndarray) -> np.ndarray:
        """Applies linear map to activations [..., d_model] -> [..., d_out]."""
        orig_shape = activations.shape
        flat_a = activations.reshape(-1, orig_shape[-1])
        pred = flat_a @ self.weight + self.bias
        out_shape = orig_shape[:-1] + (self.weight.shape[-1],)
        return pred.reshape(out_shape)


@dataclass
class ProbeEvaluationReport:
    """Structured report holding layer-wise evaluation metrics and diagnostic trajectory data."""

    layer_names: list[str]
    depths: list[int]
    trained_r2: list[float]
    random_r2: list[float]
    trained_mse: list[float]
    trained_corr: list[float]
    best_layer_depth: int
    target_type: str
    n_steps: int  # steps per cycle
    intra_cycle_r2: np.ndarray  # shape [n_layers + 1, n_steps]
    sample_true_trajectory: np.ndarray  # [N_sample]
    sample_pred_trajectories: dict[int, np.ndarray]  # depth -> [N_sample]
    scatter_true: np.ndarray  # [N_scatter]
    scatter_pred: np.ndarray  # [N_scatter]


def sequence_split(
    n_sequences: int,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Partitions sequence indices into train, validation, and test splits.

    Split is sequence-level to eliminate any intra-sequence temporal leakage.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    perm = rng.permutation(n_sequences)
    n_train = int(n_sequences * train_frac)
    n_val = int(n_sequences * val_frac)

    train_idx = perm[:n_train]
    val_idx = perm[n_train : n_train + n_val]
    test_idx = perm[n_train + n_val :]
    return train_idx, val_idx, test_idx


def compute_logntp_targets(
    model: TinyTransformer,
    tokens: torch.Tensor,
    target_type: Literal["scalar_logp", "full_vocab", "action_logp"] = "scalar_logp",
    k: int = 1,
    action_tokens: list[int] | None = None,
    device: torch.device | str | None = None,
) -> tuple[np.ndarray, slice]:
    """Computes ground-truth transformer logNTP target values.

    Args:
        model: TinyTransformer (or LookaheadTransformer)
        tokens: Token tensor [B, N]
        target_type:
            - 'scalar_logp': scalar log-probability of true next token log P(x_{t+k}) [B, N-k]
            - 'full_vocab': log-softmax probability distribution across all V tokens [B, N-k, V]
            - 'action_logp': log-probabilities of discrete action tokens [B, N-k, 4]
        k: Lookahead horizon (default 1)
        action_tokens: list of action token IDs (if target_type == 'action_logp')
        device: Target execution device

    Returns:
        tuple (targets, valid_slice) where valid_slice indicates positions 0..N-k.
    """
    target_device = (
        device if device is not None else next(model.parameters()).device
    )
    tokens_dev = tokens.to(target_device)
    b, n = tokens_dev.shape
    if k >= n:
        raise ValueError(f"Lookahead k={k} must be < sequence length {n}")

    was_training = model.training
    model.eval()
    with torch.no_grad():
        logits = model(tokens_dev)  # [B, N, V]
        log_probs = F.log_softmax(logits, dim=-1)  # [B, N, V]

        # In next-token prediction, logits at position t predict token at t+k
        # Valid positions are 0 .. N-k-1
        valid_log_probs = log_probs[:, :-k, :]  # [B, N-k, V]
        target_tokens = tokens_dev[:, k:]  # [B, N-k]

        if target_type == "scalar_logp":
            # Extract log P(actual token) = log_probs[b, t, target_tokens[b, t]]
            scalar_logp = torch.gather(
                valid_log_probs, dim=-1, index=target_tokens.unsqueeze(-1)
            ).squeeze(-1)  # [B, N-k]
            targets = scalar_logp.cpu().numpy().astype(np.float64)
        elif target_type == "full_vocab":
            targets = valid_log_probs.cpu().numpy().astype(np.float64)
        elif target_type == "action_logp":
            if action_tokens is None:
                # Default to first 4 tokens if not specified
                action_idx = torch.tensor([0, 1, 2, 3], device=target_device)
            else:
                action_idx = torch.tensor(action_tokens, device=target_device)
            action_logp = valid_log_probs[:, :, action_idx]  # [B, N-k, len(action_tokens)]
            targets = action_logp.cpu().numpy().astype(np.float64)
        else:
            raise ValueError(f"Unknown target_type: {target_type}")

    if was_training:
        model.train()

    valid_slice = slice(0, n - k)
    return targets, valid_slice


def extract_layer_activations(
    model: TinyTransformer,
    tokens: torch.Tensor,
    valid_slice: slice | None = None,
    device: torch.device | str | None = None,
) -> list[np.ndarray]:
    """Extracts residual stream activations at each layer depth [0 .. n_layers].

    Depth 0 is embedding output; depth l+1 is output of block l.
    Optionally slices sequence positions to match valid_slice (0 .. N-k).
    """
    target_device = (
        device if device is not None else next(model.parameters()).device
    )
    tokens_dev = tokens.to(target_device)

    was_training = model.training
    model.eval()
    with torch.no_grad():
        streams = model.residual_streams(tokens_dev)
    if was_training:
        model.train()

    out = []
    for s in streams:
        if valid_slice is not None:
            s = s[:, valid_slice, :]
        out.append(s.astype(np.float64))
    return out


def fit_layer_logntp_probe(
    activations: np.ndarray,
    targets: np.ndarray,
    depth: int,
    name: str = "",
    ridge: float = 1e-3,
) -> LogNTPProbe:
    """Fits multivariate ridge regression from activations to logNTP targets.

    Args:
        activations: [N_samples, d_model]
        targets: [N_samples, d_out] or [N_samples]
        depth: Layer depth index
        name: Human-readable layer name
        ridge: Ridge penalty factor matching models/probe.py
    """
    a = activations.reshape(-1, activations.shape[-1]).astype(np.float64)
    y = targets.reshape(-1, 1 if targets.ndim == 1 or targets.ndim == 2 else targets.shape[-1]).astype(np.float64)
    if targets.ndim == 2 and a.shape[0] != targets.shape[0]:
        y = targets.reshape(-1, 1)

    a_mean = a.mean(axis=0)
    y_mean = y.mean(axis=0)

    centered_a = a - a_mean
    centered_y = y - y_mean

    gram = centered_a.T @ centered_a
    tr = np.trace(gram)
    d = gram.shape[0]
    scale = tr / d if tr > 0 else 1.0
    gram_reg = gram + ridge * scale * np.eye(d)

    xty = centered_a.T @ centered_y
    weight = np.linalg.solve(gram_reg, xty)  # [d_model, d_out]
    bias = y_mean - a_mean @ weight

    probe_name = name if name else ("embedding" if depth == 0 else f"block_{depth - 1}")
    return LogNTPProbe(weight=weight, bias=bias, depth=depth, name=probe_name)


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> tuple[float, float, float]:
    """Computes R^2 (explained variance), MSE, and Pearson correlation."""
    y_t = y_true.reshape(-1).astype(np.float64)
    y_p = y_pred.reshape(-1).astype(np.float64)

    ss_res = np.sum((y_t - y_p) ** 2)
    ss_tot = np.sum((y_t - y_t.mean()) ** 2)
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else float("nan")
    mse = float(np.mean((y_t - y_p) ** 2))

    # Pearson correlation
    var_t = np.var(y_t)
    var_p = np.var(y_p)
    if var_t > 1e-12 and var_p > 1e-12:
        corr = float(np.corrcoef(y_t, y_p)[0, 1])
    else:
        corr = 0.0

    return r2, mse, corr


def evaluate_layer_probes(
    model: TinyTransformer,
    tokens: torch.Tensor,
    target_type: Literal["scalar_logp", "full_vocab", "action_logp"] = "scalar_logp",
    k: int = 1,
    n_steps: int = 10,
    ridge: float = 1e-3,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    seed: int = 0,
    device: torch.device | str | None = None,
) -> ProbeEvaluationReport:
    """Profiles linear decodability of logNTP across all layer depths vs random control.

    1. Computes ground-truth logNTP targets and layer residual activations.
    2. Splits data at sequence level (train/val/test).
    3. Fits probes on train split for each layer.
    4. Fits identical probes on an untrained architecture-matched random control model.
    5. Evaluates R^2, MSE, correlation, and intra-cycle breakdown on test split.
    """
    rng = np.random.default_rng(seed)
    n_seq = tokens.shape[0]
    train_idx, val_idx, test_idx = sequence_split(n_seq, train_frac, val_frac, rng)

    # 1. Compute targets
    targets, valid_slice = compute_logntp_targets(
        model=model,
        tokens=tokens,
        target_type=target_type,
        k=k,
        device=device,
    )
    # Shape of targets: [B, N_valid] or [B, N_valid, D]

    # 2. Extract activations for trained model
    streams = extract_layer_activations(model, tokens, valid_slice=valid_slice, device=device)

    # 3. Create and extract activations for random untrained control model
    random_cfg = ModelConfig(
        vocab_size=model.config.vocab_size,
        n_ctx=model.config.n_ctx,
        n_layers=model.config.n_layers,
        n_heads=model.config.n_heads,
        d_model=model.config.d_model,
        d_mlp=model.config.d_mlp,
        seed=seed + 9999,
    )
    random_model = TinyTransformer(random_cfg).eval()
    if device is not None:
        random_model = random_model.to(device)
    random_streams = extract_layer_activations(random_model, tokens, valid_slice=valid_slice, device=device)

    n_layers = len(streams)
    layer_names = ["embedding"] + [f"block_{i}" for i in range(n_layers - 1)]
    depths = list(range(n_layers))

    trained_r2 = []
    random_r2 = []
    trained_mse = []
    trained_corr = []
    test_preds_by_depth: dict[int, np.ndarray] = {}

    seq_len_valid = targets.shape[1]
    intra_cycle_r2 = np.zeros((n_layers, n_steps), dtype=np.float64)

    # Flatten targets for train and test splits
    y_train = targets[train_idx]
    y_test = targets[test_idx]

    best_r2 = -float("inf")
    best_depth = 0

    for depth in range(n_layers):
        a_trained = streams[depth]
        a_random = random_streams[depth]

        # Train split
        a_train = a_trained[train_idx]
        a_test = a_trained[test_idx]

        # Fit probe on trained model activations
        probe_trained = fit_layer_logntp_probe(
            activations=a_train,
            targets=y_train,
            depth=depth,
            name=layer_names[depth],
            ridge=ridge,
        )

        # Fit probe on random control activations
        a_rand_train = a_random[train_idx]
        a_rand_test = a_random[test_idx]
        probe_random = fit_layer_logntp_probe(
            activations=a_rand_train,
            targets=y_train,
            depth=depth,
            name=f"rand_{layer_names[depth]}",
            ridge=ridge,
        )

        # Predict on test split
        pred_test = probe_trained.predict(a_test)
        pred_rand = probe_random.predict(a_rand_test)

        r2_t, mse_t, corr_t = compute_metrics(y_test, pred_test)
        r2_r, _, _ = compute_metrics(y_test, pred_rand)

        trained_r2.append(r2_t)
        random_r2.append(r2_r)
        trained_mse.append(mse_t)
        trained_corr.append(corr_t)
        test_preds_by_depth[depth] = pred_test

        if r2_t > best_r2:
            best_r2 = r2_t
            best_depth = depth

        # Intra-cycle breakdown: compute test R^2 at each step within physical tick s in [0 .. n_steps-1]
        for s in range(n_steps):
            step_indices = np.arange(s, seq_len_valid, n_steps)
            if len(step_indices) > 0:
                y_sub = y_test[:, step_indices]
                p_sub = pred_test[:, step_indices]
                r2_s, _, _ = compute_metrics(y_sub, p_sub)
                intra_cycle_r2[depth, s] = r2_s

    # Diagnostic sample trajectory (first test sequence, up to 100 steps)
    sample_len = min(100, seq_len_valid)
    sample_true = y_test[0, :sample_len]
    sample_preds = {
        d: test_preds_by_depth[d][0, :sample_len] for d in range(n_layers)
    }

    # Calibration scatter points (subsample 1000 points from test set)
    flat_true = y_test.reshape(-1)
    flat_pred_best = test_preds_by_depth[best_depth].reshape(-1)
    n_scatter = min(1000, len(flat_true))
    scatter_idx = rng.choice(len(flat_true), size=n_scatter, replace=False)
    scatter_true = flat_true[scatter_idx]
    scatter_pred = flat_pred_best[scatter_idx]

    return ProbeEvaluationReport(
        layer_names=layer_names,
        depths=depths,
        trained_r2=trained_r2,
        random_r2=random_r2,
        trained_mse=trained_mse,
        trained_corr=trained_corr,
        best_layer_depth=best_depth,
        target_type=target_type,
        n_steps=n_steps,
        intra_cycle_r2=intra_cycle_r2,
        sample_true_trajectory=sample_true,
        sample_pred_trajectories=sample_preds,
        scatter_true=scatter_true,
        scatter_pred=scatter_pred,
    )
