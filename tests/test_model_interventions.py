"""Automated regression and unit test suite for mechanistic model interventions.

Tests trajectory pairing, 4D position-aware inverse ridge probes, single-layer hook isolation,
alpha scaling, identity checks, and dual historical regimes (past-consistent with N=1600, k=600).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from models.interventions import (
    InterventionHarness,
    PositionAwareInverseProbe,
    TrajectoryData,
    TrajectoryPair,
    extract_trajectory_representations,
    fit_position_aware_inverse_probe,
    intervene_past_consistent,
    intervene_past_inconsistent,
)
from models.transformer import ModelConfig, TinyTransformer
from physics.messk_configs import make_process


def _create_test_model(
    vocab_size: int = 50,
    n_ctx: int = 40,
    n_layers: int = 3,
    d_model: int = 32,
    d_mlp: int = 64,
    seed: int = 42,
) -> TinyTransformer:
    cfg = ModelConfig(
        vocab_size=vocab_size,
        n_ctx=n_ctx,
        n_layers=n_layers,
        n_heads=1,
        d_model=d_model,
        d_mlp=d_mlp,
        seed=seed,
    )
    return TinyTransformer(cfg).eval()


def _create_synthetic_pair(
    batch_size: int = 4,
    seq_len: int = 40,
    vocab_size: int = 50,
    d_model: int = 32,
    seed: int = 0,
) -> TrajectoryPair:
    rng = np.random.default_rng(seed)

    tokens_x = torch.randint(0, vocab_size, (batch_size, seq_len))
    tokens_y = torch.randint(0, vocab_size, (batch_size, seq_len))

    # Markovian beliefs in 4D simplex (sum to 1)
    raw_b_x = rng.exponential(scale=1.0, size=(batch_size, seq_len, 4))
    beliefs_x = raw_b_x / raw_b_x.sum(axis=-1, keepdims=True)

    raw_b_y = rng.exponential(scale=1.0, size=(batch_size, seq_len, 4))
    beliefs_y = raw_b_y / raw_b_y.sum(axis=-1, keepdims=True)

    states_x = rng.normal(size=(batch_size, seq_len, 4))
    states_y = rng.normal(size=(batch_size, seq_len, 4))

    data_x = TrajectoryData(tokens=tokens_x, beliefs=beliefs_x, states=states_x)
    data_y = TrajectoryData(tokens=tokens_y, beliefs=beliefs_y, states=states_y)
    return TrajectoryPair(x=data_x, y=data_y)


def test_trajectory_pair_shapes() -> None:
    pair = _create_synthetic_pair(batch_size=3, seq_len=20, vocab_size=30)
    assert pair.x.tokens.shape == (3, 20)
    assert pair.x.beliefs.shape == (3, 20, 4)
    assert pair.x.states.shape == (3, 20, 4)
    assert np.allclose(pair.x.beliefs.sum(axis=-1), 1.0)

    model = _create_test_model(vocab_size=30, n_ctx=20, n_layers=2, d_model=16)
    pair.compute_residual_streams(model)
    assert pair.x.residual_stream is not None
    assert len(pair.x.residual_stream) == 3  # embedding + 2 blocks
    assert pair.x.residual_stream[0].shape == (3, 20, 16)
    assert pair.x.residual_stream[1].shape == (3, 20, 16)


def test_inverse_probe_fit_and_predict() -> None:
    batch_size, seq_len, d_model = 64, 16, 32
    rng = np.random.default_rng(10)

    raw_b = rng.exponential(scale=1.0, size=(batch_size, seq_len, 4))
    beliefs = raw_b / raw_b.sum(axis=-1, keepdims=True)

    # True linear generation per position with distinct positional baseline
    pos_baseline = rng.normal(size=(seq_len, d_model))
    true_w = rng.normal(size=(seq_len, 4, d_model))
    activations = np.zeros((batch_size, seq_len, d_model))
    for t in range(seq_len):
        activations[:, t, :] = pos_baseline[t] + (beliefs[:, t, :] - beliefs[:, t, :].mean(axis=0)) @ true_w[t]

    probe = fit_position_aware_inverse_probe(beliefs, activations, ridge=1e-5)
    assert probe.weights.shape == (seq_len, 4, d_model)
    assert probe.biases.shape == (seq_len, d_model)

    pred = probe.predict(beliefs)
    assert pred.shape == (batch_size, seq_len, d_model)

    # Mean squared error on reconstruction must be near zero
    mse = np.mean((pred.numpy() - activations) ** 2)
    assert mse < 1e-4, f"Expected near-perfect fit on noiseless linear data, got MSE={mse}"


def test_single_layer_hook_isolation() -> None:
    model = _create_test_model(vocab_size=40, n_ctx=20, n_layers=3, d_model=32)
    pair = _create_synthetic_pair(batch_size=2, seq_len=20, vocab_size=40, d_model=32)
    pair.compute_residual_streams(model)

    layer_to_intervene = 1  # block 1
    donor_act = pair.y.get_block_activation(layer_to_intervene)
    clean_stream = model.residual_streams(pair.x.tokens)

    with InterventionHarness(
        model=model,
        layer=layer_to_intervene,
        donor_repr=donor_act,
        mode="patching",
    ):
        hooked_stream = model.residual_streams(pair.x.tokens)

    # Isolation Invariant:
    # Index 0 is embedding, Index 1 is block 0: must be bitwise identical
    assert np.allclose(hooked_stream[0], clean_stream[0]), "Embedding was modified by block hook"
    assert np.allclose(hooked_stream[1], clean_stream[1]), "Block 0 was modified by block 1 hook"

    # Index 2 is block 1 output: must match donor activation
    assert np.allclose(hooked_stream[2], donor_act.cpu().numpy(), atol=1e-5)

    # Clean detach invariant: subsequent forward pass outside context must match clean stream exactly
    post_stream = model.residual_streams(pair.x.tokens)
    assert np.allclose(post_stream[2], clean_stream[2]), "Hook was not removed after context exit"


def test_identity_steering() -> None:
    model = _create_test_model(vocab_size=40, n_ctx=20, n_layers=2, d_model=32)
    pair = _create_synthetic_pair(batch_size=2, seq_len=20, vocab_size=40, d_model=32)
    pair.compute_residual_streams(model)

    clean_logits = model(pair.x.tokens)
    base_act = pair.x.get_block_activation(0)

    # Identity Case 1: alpha = 0.0 with different donor
    donor_act = pair.y.get_block_activation(0)
    with InterventionHarness(
        model=model,
        layer=0,
        donor_repr=donor_act,
        base_repr=base_act,
        mode="steering",
        alpha=0.0,
    ):
        logits_alpha_zero = model(pair.x.tokens)
    assert torch.allclose(logits_alpha_zero, clean_logits, atol=1e-5), "alpha=0 must match clean logits"

    # Identity Case 2: donor equals base (Y = X) with alpha = 1.0
    with InterventionHarness(
        model=model,
        layer=0,
        donor_repr=base_act,
        base_repr=base_act,
        mode="steering",
        alpha=1.0,
    ):
        logits_identical_donor = model(pair.x.tokens)
    assert torch.allclose(logits_identical_donor, clean_logits, atol=1e-5), "donor=base must match clean logits"


def test_steering_alpha_scaling() -> None:
    model = _create_test_model(vocab_size=40, n_ctx=20, n_layers=2, d_model=32)
    pair = _create_synthetic_pair(batch_size=2, seq_len=20, vocab_size=40, d_model=32)
    pair.compute_residual_streams(model)

    base_act = pair.x.get_block_activation(0)
    donor_act = pair.y.get_block_activation(0)

    # Test perturbation scaling directly at the intervened block
    diffs = []
    for alpha in [0.5, 1.0, 1.5, 2.0]:
        with InterventionHarness(
            model=model,
            layer=0,
            donor_repr=donor_act,
            base_repr=base_act,
            mode="steering",
            alpha=alpha,
        ):
            stream = model.residual_streams(pair.x.tokens)
            diff = np.linalg.norm(stream[1] - pair.x.residual_stream[1])
            diffs.append(diff)

    # Differences at layer 0 output must scale linearly with alpha: diff / alpha = const
    ratios = [d / a for d, a in zip(diffs, [0.5, 1.0, 1.5, 2.0])]
    assert np.allclose(ratios, ratios[0], rtol=1e-4), f"Perturbation did not scale linearly with alpha: {diffs}"


def test_past_inconsistent_execution() -> None:
    model = _create_test_model(vocab_size=50, n_ctx=30, n_layers=2, d_model=32)
    pair = _create_synthetic_pair(batch_size=2, seq_len=30, vocab_size=50, d_model=32)
    pair.compute_residual_streams(model)

    # Test with source="inverse_probe"
    res_probe = intervene_past_inconsistent(
        model=model,
        pair=pair,
        layer=0,
        mode="patching",
        source="inverse_probe",
    )
    assert res_probe.logits.shape == (2, 30, 50)
    assert len(res_probe.intervened_residual_stream) == 3
    assert res_probe.regime == "past_inconsistent"

    # Test with source="activation"
    res_act = intervene_past_inconsistent(
        model=model,
        pair=pair,
        layer=1,
        mode="steering",
        alpha=1.5,
        source="activation",
    )
    assert res_act.logits.shape == (2, 30, 50)
    assert res_act.alpha == 1.5


def test_past_consistent_execution() -> None:
    model = _create_test_model(vocab_size=50, n_ctx=40, n_layers=2, d_model=32)
    pair = _create_synthetic_pair(batch_size=2, seq_len=40, vocab_size=50, d_model=32)
    pair.compute_residual_streams(model)

    # Split with k = 15 (40 - 15 = 25 tokens from X, 15 from Y)
    res = intervene_past_consistent(
        model=model,
        pair=pair,
        layer=0,
        k=15,
        mode="patching",
        source="inverse_probe",
    )
    assert res.logits.shape == (2, 40, 50)
    assert len(res.intervened_residual_stream) == 3
    assert res.regime == "past_consistent"

    # Verify invalid k raises ValueError
    with pytest.raises(ValueError):
        intervene_past_consistent(model=model, pair=pair, layer=0, k=40)
    with pytest.raises(ValueError):
        intervene_past_consistent(model=model, pair=pair, layer=0, k=0)


def test_full_scale_double_pendulum_integration() -> None:
    """Full production configuration integration test (N=1600, k=600) on double_pendulum_mess4."""
    proc = make_process("double_pendulum_mess4")
    vocab_size = proc.n_obs

    # TinyTransformer with context length 1600
    model = _create_test_model(
        vocab_size=vocab_size,
        n_ctx=1600,
        n_layers=2,
        d_model=32,
        d_mlp=64,
        seed=100,
    )

    # Simulate paired trajectories from double pendulum process
    pair = TrajectoryPair.from_process(
        proc=proc,
        batch_size=1,
        seq_len=1600,
        seed_x=10,
        seed_y=20,
        model=model,
    )
    assert pair.seq_len == 1600
    assert pair.x.tokens.shape == (1, 1600)
    assert pair.x.beliefs.shape == (1, 1600, 4)

    # Run past-consistent intervention with canonical k = 600
    res_consistent = intervene_past_consistent(
        model=model,
        pair=pair,
        layer=0,
        k=600,
        mode="steering",
        alpha=1.0,
        source="inverse_probe",
    )
    assert res_consistent.logits.shape == (1, 1600, vocab_size)
    assert len(res_consistent.intervened_residual_stream) == 3

    # Run past-inconsistent intervention
    res_inconsistent = intervene_past_inconsistent(
        model=model,
        pair=pair,
        layer=1,
        mode="patching",
        source="activation",
    )
    assert res_inconsistent.logits.shape == (1, 1600, vocab_size)
