"""Automated test suite for Layer-Wise Residual Stream Probing to logNTP.

Tests target computation, activation extraction, ridge regression convergence,
data partitioning, multi-layer evaluation, and publication panel plotting.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest
import torch

from models.logntp_probing import (
    compute_logntp_targets,
    compute_metrics,
    evaluate_layer_probes,
    extract_layer_activations,
    fit_layer_logntp_probe,
    sequence_split,
)
from models.transformer import ModelConfig, TinyTransformer
from scripts.plot_logntp_layer_probes import plot_logntp_probing_panel


def _create_synthetic_model(
    vocab_size: int = 30,
    n_ctx: int = 32,
    n_layers: int = 2,
    d_model: int = 16,
    d_mlp: int = 32,
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


def test_sequence_split_no_leakage() -> None:
    n_seq = 100
    train_idx, val_idx, test_idx = sequence_split(n_seq, train_frac=0.7, val_frac=0.15)
    assert len(train_idx) == 70
    assert len(val_idx) == 15
    assert len(test_idx) == 15

    # Mutual exclusion check
    assert len(set(train_idx).intersection(set(val_idx))) == 0
    assert len(set(train_idx).intersection(set(test_idx))) == 0
    assert len(set(val_idx).intersection(set(test_idx))) == 0

    # Completeness check
    union = set(train_idx).union(set(val_idx)).union(set(test_idx))
    assert union == set(range(n_seq))


def test_compute_logntp_targets_shapes() -> None:
    model = _create_synthetic_model(vocab_size=25, n_ctx=20, n_layers=2, d_model=16)
    tokens = torch.randint(0, 25, (4, 20))

    # 1. Scalar log-probability target
    targets_scalar, v_slice = compute_logntp_targets(model, tokens, target_type="scalar_logp", k=1)
    assert targets_scalar.shape == (4, 19)
    assert (targets_scalar <= 0.0).all(), "Log-probabilities must be <= 0"

    # 2. Full vocabulary distribution
    targets_full, _ = compute_logntp_targets(model, tokens, target_type="full_vocab", k=1)
    assert targets_full.shape == (4, 19, 25)
    probs_full = np.exp(targets_full)
    assert np.allclose(probs_full.sum(axis=-1), 1.0, atol=1e-5), "Softmax probabilities must sum to 1"

    # 3. Action log-probabilities
    targets_action, _ = compute_logntp_targets(
        model, tokens, target_type="action_logp", k=1, action_tokens=[0, 1, 2, 3]
    )
    assert targets_action.shape == (4, 19, 4)


def test_extract_layer_activations_shapes() -> None:
    model = _create_synthetic_model(vocab_size=25, n_ctx=20, n_layers=3, d_model=16)
    tokens = torch.randint(0, 25, (4, 20))
    streams = extract_layer_activations(model, tokens, valid_slice=slice(0, 19))

    assert len(streams) == 4  # embedding + 3 blocks
    for s in streams:
        assert s.shape == (4, 19, 16)


def test_fit_layer_logntp_probe_noiseless_recovery() -> None:
    rng = np.random.default_rng(123)
    n_samples, d_model = 500, 16
    activations = rng.normal(size=(n_samples, d_model))
    true_w = rng.normal(size=(d_model, 1))
    true_b = 0.75
    targets = activations @ true_w + true_b

    probe = fit_layer_logntp_probe(activations, targets, depth=1, name="test_block", ridge=1e-5)
    pred = probe.predict(activations)

    r2, mse, corr = compute_metrics(targets, pred)
    assert r2 > 0.999, f"Expected near-perfect R^2 on noiseless linear data, got {r2}"
    assert mse < 1e-4, f"Expected negligible MSE, got {mse}"
    assert corr > 0.999


def test_evaluate_layer_probes_pipeline() -> None:
    model = _create_synthetic_model(vocab_size=30, n_ctx=24, n_layers=2, d_model=16)
    tokens = torch.randint(0, 30, (16, 24))

    report = evaluate_layer_probes(
        model=model,
        tokens=tokens,
        target_type="scalar_logp",
        k=1,
        n_steps=4,
        seed=1,
    )

    assert len(report.depths) == 3  # embedding + 2 blocks
    assert len(report.trained_r2) == 3
    assert len(report.random_r2) == 3
    assert len(report.layer_names) == 3
    assert report.intra_cycle_r2.shape == (3, 4)
    assert len(report.sample_true_trajectory) > 0
    assert len(report.scatter_true) == len(report.scatter_pred)


def test_plot_panel_smoke(tmp_path: Path) -> None:
    model = _create_synthetic_model(vocab_size=30, n_ctx=20, n_layers=2, d_model=16)
    tokens = torch.randint(0, 30, (8, 20))
    report = evaluate_layer_probes(
        model=model,
        tokens=tokens,
        target_type="scalar_logp",
        k=1,
        n_steps=5,
        seed=2,
    )

    out_file = tmp_path / "test_panel.png"
    fig = plot_logntp_probing_panel(report, out_path=out_file, title_suffix="(Test)")
    assert out_file.is_file()
    assert out_file.stat().st_size > 1000  # valid image file generated
