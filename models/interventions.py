"""Mechanistic Model Interventions via Trajectory Steering and Activation Patching.

Provides causal mediation tools to intervene on TinyTransformer residual streams:
- `TrajectoryData` and `TrajectoryPair`: Container and simulation factory for paired physical trajectories.
- `PositionAwareInverseProbe`: Fits 4D regularized ridge regression per sequence position t in [0, N-1]
  preserving full-sequence positional encodings.
- `InterventionHarness`: Non-invasive PyTorch forward hook context manager for single-layer steering/patching.
- `intervene_past_inconsistent` & `intervene_past_consistent`: Dual historical context execution engines.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import torch
from torch import nn

from .transformer import TinyTransformer


@dataclass
class TrajectoryData:
    """Container holding physical simulation and residual stream activations for a single trajectory."""

    tokens: torch.Tensor  # shape [B, N] (integer token indices)
    beliefs: np.ndarray  # shape [B, N, 4] (Markovian mood beliefs in simplex)
    states: np.ndarray  # shape [B, N, 4] (continuous physical coordinates: theta1, theta2, omega1, omega2)
    residual_stream: list[np.ndarray] | None = None  # length n_layers + 1, each [B, N, d_model]

    def __post_init__(se                                                                                        lf) -> None:
        if self.tokens.ndim == 1:
            self.tokens = self.tokens.unsqueeze(0)
        if self.beliefs.ndim == 2:
            self.beliefs = np.expand_dims(self.beliefs, axis=0)
        if self.states.ndim == 2:
            self.states = np.expand_dims(self.states, axis=0)

    @property
    def batch_size(self) -> int:
        return int(self.tokens.shape[0])

    @property
    def seq_len(self) -> int:
        return int(self.tokens.shape[1])

    def get_block_activation(self, layer: int, device: torch.device | None = None) -> torch.Tensor:
        """Extracts residual stream activation after block `layer` (index layer + 1)."""
        if self.residual_stream is None:
            raise ValueError("residual_stream has not been computed for this TrajectoryData")
        stream_idx = layer + 1
        if stream_idx >= len(self.residual_stream):
            raise IndexError(
                f"Block {layer} (stream index {stream_idx}) out of range for stream length {len(self.residual_stream)}"
            )
        act = torch.as_tensor(self.residual_stream[stream_idx], dtype=torch.float32)
        return act if device is None else act.to(device)


@dataclass
class TrajectoryPair:
    """Paired base (X) and counterfactual donor (Y) trajectories simulated from double pendulum dynamics."""

    x: TrajectoryData
    y: TrajectoryData

    @property
    def seq_len(self) -> int:
        return self.x.seq_len

    @property
    def batch_size(self) -> int:
        return self.x.batch_size

    @classmethod
    def from_process(
        cls,
        proc: Any,
        batch_size: int = 1,
        seq_len: int = 1600,
        seed_x: int = 0,
        seed_y: int = 1,
        model: TinyTransformer | None = None,
        device: torch.device | str | None = None,
    ) -> "TrajectoryPair":
        """Simulates paired physical trajectories (X, Y) from a MessDriven process."""
        # Adjust ticks m if needed
        active_proc = proc
        if proc.seq_len != seq_len:
            m = seq_len // proc.n_steps
            active_proc = dataclasses.replace(proc, m=m)

        rng_x = np.random.default_rng(seed_x)
        rng_y = np.random.default_rng(seed_y)

        batch_x = active_proc.sample_batch(rng_x, batch_size)
        batch_y = active_proc.sample_batch(rng_y, batch_size)

        # Assemble continuous states: observable (angles) + metric (velocities)
        states_x = np.concatenate([batch_x["observable"], batch_x["metric"]], axis=-1)
        states_y = np.concatenate([batch_y["observable"], batch_y["metric"]], axis=-1)

        tokens_x = torch.as_tensor(batch_x["tokens"], dtype=torch.long)
        tokens_y = torch.as_tensor(batch_y["tokens"], dtype=torch.long)

        data_x = TrajectoryData(tokens=tokens_x, beliefs=batch_x["beliefs"], states=states_x)
        data_y = TrajectoryData(tokens=tokens_y, beliefs=batch_y["beliefs"], states=states_y)

        pair = cls(x=data_x, y=data_y)
        if model is not None:
            pair.compute_residual_streams(model, device=device)
        return pair

    def compute_residual_streams(
        self, model: TinyTransformer, device: torch.device | str | None = None
    ) -> None:
        """Precomputes clean residual streams for both X and Y trajectories."""
        target_device = (
            device if device is not None else next(model.parameters()).device
        )
        was_training = model.training
        model.eval()
        with torch.no_grad():
            self.x.residual_stream = model.residual_streams(self.x.tokens.to(target_device))
            self.y.residual_stream = model.residual_streams(self.y.tokens.to(target_device))
        if was_training:
            model.train()


@dataclass
class PositionAwareInverseProbe:
    """Position-aware inverse belief mapping emb_l(eta, t) = mu_{l, t} + (eta - bar_eta_t) W_{l, t}^{inv}.

    Fits 4D ridge regression independently for each timestamp t in [0, N-1], strictly preserving
    the full-sequence positional encoding baseline mu_{l, t}.
    """

    weights: np.ndarray  # shape [N, 4, d_model]
    biases: np.ndarray  # shape [N, d_model]
    mu: np.ndarray  # shape [N, d_model] (baseline mean activation)
    bar_eta: np.ndarray  # shape [N, 4] (baseline mean belief)
    ridge: float = 1e-3

    @property
    def seq_len(self) -> int:
        return int(self.weights.shape[0])

    @property
    def d_model(self) -> int:
        return int(self.weights.shape[2])

    def predict(
        self, beliefs: np.ndarray | torch.Tensor, device: torch.device | None = None
    ) -> torch.Tensor:
        """Maps belief tensor [B, N, 4] to residual stream activations [B, N, d_model]."""
        if isinstance(beliefs, torch.Tensor):
            b_arr = beliefs.detach().cpu().numpy()
        else:
            b_arr = np.asarray(beliefs)

        if b_arr.ndim == 2:
            b_arr = np.expand_dims(b_arr, axis=0)

        b, n, k = b_arr.shape
        if n != self.seq_len:
            raise ValueError(
                f"Belief sequence length {n} does not match probe fitted sequence length {self.seq_len}"
            )
        if k != 4:
            raise ValueError(f"Expected 4 belief dimensions, got {k}")

        # Centered belief: [B, N, 4] - [N, 4]
        centered_b = b_arr - self.bar_eta[None, :, :]
        # einsum: [B, N, 4] x [N, 4, d_model] -> [B, N, d_model]
        recon = np.einsum("bnk,nkd->bnd", centered_b, self.weights) + self.mu[None, :, :]

        out = torch.as_tensor(recon, dtype=torch.float32)
        return out if device is None else out.to(device)


def fit_position_aware_inverse_probe(
    beliefs: np.ndarray | torch.Tensor,
    activations: np.ndarray | torch.Tensor,
    ridge: float = 1e-3,
) -> PositionAwareInverseProbe:
    """Fits multivariate ridge regression from belief states to residual activations per timestamp.

    Args:
        beliefs: [B, N, 4] Markovian mood beliefs.
        activations: [B, N, d_model] clean residual stream activations at a chosen block.
        ridge: regularizer multiplier matching models/probe.py.
    """
    if isinstance(beliefs, torch.Tensor):
        b_arr = beliefs.detach().cpu().numpy()
    else:
        b_arr = np.asarray(beliefs, dtype=np.float64)

    if isinstance(activations, torch.Tensor):
        a_arr = activations.detach().cpu().numpy()
    else:
        a_arr = np.asarray(activations, dtype=np.float64)

    if b_arr.ndim == 2:
        b_arr = np.expand_dims(b_arr, axis=0)
    if a_arr.ndim == 2:
        a_arr = np.expand_dims(a_arr, axis=0)

    b, n, k = b_arr.shape
    _, _, d = a_arr.shape

    if b_arr.shape[:2] != a_arr.shape[:2]:
        raise ValueError(
            f"Shape mismatch: beliefs {b_arr.shape} vs activations {a_arr.shape}"
        )
    if k != 4:
        raise ValueError(f"Expected 4 belief features, got {k}")

    # Compute sequence baselines
    bar_eta = b_arr.mean(axis=0)  # [N, 4]
    mu = a_arr.mean(axis=0)  # [N, d_model]

    centered_b = b_arr - bar_eta[None, :, :]  # [B, N, 4]
    centered_a = a_arr - mu[None, :, :]  # [B, N, d]

    weights = np.empty((n, 4, d), dtype=np.float64)
    biases = np.empty((n, d), dtype=np.float64)

    eye_4 = np.eye(4, dtype=np.float64)
    for t in range(n):
        bt = centered_b[:, t, :]  # [B, 4]
        at = centered_a[:, t, :]  # [B, d]

        gram = bt.T @ bt  # [4, 4]
        tr = np.trace(gram)
        # Regularization matching models/probe.py: gram + ridge * tr / 4 * I
        scale = tr / 4.0 if tr > 0 else 1.0
        gram_reg = gram + ridge * scale * eye_4

        xty = bt.T @ at  # [4, d]
        wt = np.linalg.solve(gram_reg, xty)  # [4, d]
        weights[t] = wt
        biases[t] = mu[t] - bar_eta[t] @ wt

    return PositionAwareInverseProbe(
        weights=weights, biases=biases, mu=mu, bar_eta=bar_eta, ridge=ridge
    )


def extract_trajectory_representations(
    pair: TrajectoryPair,
    layer: int,
    source: Literal["inverse_probe", "activation"] = "inverse_probe",
    probe: PositionAwareInverseProbe | None = None,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extracts base (X) and counterfactual donor (Y) representations at block `layer`.

    Returns:
        tuple (repr_x, repr_y) both of shape [B, N, d_model] on `device`.
    """
    if source == "inverse_probe":
        if probe is None:
            if pair.x.residual_stream is None:
                raise ValueError("Cannot fit inverse probe: pair.x.residual_stream is None")
            act_x = pair.x.residual_stream[layer + 1]
            probe = fit_position_aware_inverse_probe(pair.x.beliefs, act_x)
        repr_x = probe.predict(pair.x.beliefs, device=device)
        repr_y = probe.predict(pair.y.beliefs, device=device)
    elif source == "activation":
        repr_x = pair.x.get_block_activation(layer, device=device)
        repr_y = pair.y.get_block_activation(layer, device=device)
    else:
        raise ValueError(f"Unknown representation source: {source}")

    return repr_x, repr_y


class InterventionHarness:
    """Non-invasive PyTorch forward hook context manager for single-layer intervention.

    Intervenes on `model.blocks[layer]` in steering or patching mode across all sequence positions.
    Deterministically cleans up hooks on exit.
    """

    def __init__(
        self,
        model: TinyTransformer,
        layer: int,
        donor_repr: torch.Tensor,
        base_repr: torch.Tensor | None = None,
        mode: Literal["steering", "patching"] = "steering",
        alpha: float = 1.0,
    ) -> None:
        if not (0 <= layer < len(model.blocks)):
            raise IndexError(
                f"layer {layer} out of bounds for model with {len(model.blocks)} blocks"
            )
        if mode == "steering" and base_repr is None:
            raise ValueError("base_repr must be provided when mode='steering'")

        self.model = model
        self.layer = layer
        self.donor_repr = donor_repr
        self.base_repr = base_repr
        self.mode = mode
        self.alpha = float(alpha)
        self.handle: Any | None = None

    def _hook_fn(self, module: nn.Module, input: Any, output: torch.Tensor) -> torch.Tensor:
        # output is [B, N, d_model] residual stream after block `layer`
        donor = self.donor_repr.to(device=output.device, dtype=output.dtype)
        if self.mode == "patching":
            return donor
        elif self.mode == "steering":
            base = self.base_repr.to(device=output.device, dtype=output.dtype)
            delta = donor - base
            return output + self.alpha * delta
        else:
            raise ValueError(f"Unknown intervention mode: {self.mode}")

    def __enter__(self) -> "InterventionHarness":
        self.handle = self.model.blocks[self.layer].register_forward_hook(self._hook_fn)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


@dataclass
class InterventionResult:
    """Container holding plain intervention outputs (tensors and metadata, zero metrics)."""

    logits: torch.Tensor  # shape [B, N, vocab_size]
    intervened_residual_stream: list[np.ndarray]  # residual stream under intervention
    clean_residual_stream: list[np.ndarray]  # clean un-intervened residual stream
    layer: int
    mode: str
    alpha: float
    regime: str
    source: str


def intervene_past_inconsistent(
    model: TinyTransformer,
    pair: TrajectoryPair,
    layer: int,
    mode: Literal["steering", "patching"] = "patching",
    alpha: float = 1.0,
    source: Literal["inverse_probe", "activation"] = "inverse_probe",
    probe: PositionAwareInverseProbe | None = None,
    device: torch.device | str | None = None,
) -> InterventionResult:
    """Executes past-inconsistent intervention.

    Feeds prompt tokens belonging to base trajectory X. Intervenes at block `layer` across
    all sequence positions using counterfactual donor trajectory Y representations.
    """
    target_device = (
        device if device is not None else next(model.parameters()).device
    )
    tokens_x = pair.x.tokens.to(target_device)

    # Ensure clean stream is available
    if pair.x.residual_stream is None:
        pair.compute_residual_streams(model, device=target_device)

    repr_x, repr_y = extract_trajectory_representations(
        pair, layer=layer, source=source, probe=probe, device=target_device
    )

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            clean_logits = model(tokens_x)
            clean_stream = model.residual_streams(tokens_x)

            with InterventionHarness(
                model=model,
                layer=layer,
                donor_repr=repr_y,
                base_repr=repr_x,
                mode=mode,
                alpha=alpha,
            ):
                intervened_logits = model(tokens_x)
                intervened_stream = model.residual_streams(tokens_x)
    finally:
        if was_training:
            model.train()

    return InterventionResult(
        logits=intervened_logits,
        intervened_residual_stream=intervened_stream,
        clean_residual_stream=clean_stream,
        layer=layer,
        mode=mode,
        alpha=alpha,
        regime="past_inconsistent",
        source=source,
    )


def intervene_past_consistent(
    model: TinyTransformer,
    pair: TrajectoryPair,
    layer: int,
    k: int = 600,
    mode: Literal["steering", "patching"] = "patching",
    alpha: float = 1.0,
    source: Literal["inverse_probe", "activation"] = "inverse_probe",
    probe: PositionAwareInverseProbe | None = None,
    device: torch.device | str | None = None,
) -> InterventionResult:
    """Executes past-consistent intervention.

    Slices base tokens x_{1:N-k} and donor tokens y_{1:k} to form spliced context of fixed length N.
    Intervenes at block `layer` across all sequence positions using counterfactual donor trajectory Y representation.
    """
    target_device = (
        device if device is not None else next(model.parameters()).device
    )
    n = pair.seq_len
    if not (0 < k < n):
        raise ValueError(f"Split point k={k} must satisfy 0 < k < {n}")

    # Spliced tokens: x[1:N-k] concatenated with y[1:k] -> length N
    tokens_x = pair.x.tokens.to(target_device)
    tokens_y = pair.y.tokens.to(target_device)
    spliced_tokens = torch.cat([tokens_x[:, : n - k], tokens_y[:, :k]], dim=1)

    # Spliced beliefs for base representation in steering mode if probe is used
    spliced_beliefs = np.concatenate(
        [pair.x.beliefs[:, : n - k], pair.y.beliefs[:, :k]], axis=1
    )

    # Precompute streams if needed
    if pair.x.residual_stream is None:
        pair.compute_residual_streams(model, device=target_device)

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            clean_spliced_stream = model.residual_streams(spliced_tokens)

            # Representation extraction
            if source == "inverse_probe":
                if probe is None:
                    act_x = pair.x.residual_stream[layer + 1]
                    probe = fit_position_aware_inverse_probe(pair.x.beliefs, act_x)
                donor_repr = probe.predict(pair.y.beliefs, device=target_device)
                base_repr = probe.predict(spliced_beliefs, device=target_device)
            elif source == "activation":
                donor_repr = pair.y.get_block_activation(layer, device=target_device)
                base_repr = torch.as_tensor(
                    clean_spliced_stream[layer + 1], dtype=torch.float32, device=target_device
                )
            else:
                raise ValueError(f"Unknown representation source: {source}")

            with InterventionHarness(
                model=model,
                layer=layer,
                donor_repr=donor_repr,
                base_repr=base_repr,
                mode=mode,
                alpha=alpha,
            ):
                intervened_logits = model(spliced_tokens)
                intervened_stream = model.residual_streams(spliced_tokens)
    finally:
        if was_training:
            model.train()

    return InterventionResult(
        logits=intervened_logits,
        intervened_residual_stream=intervened_stream,
        clean_residual_stream=clean_spliced_stream,
        layer=layer,
        mode=mode,
        alpha=alpha,
        regime="past_consistent",
        source=source,
    )
