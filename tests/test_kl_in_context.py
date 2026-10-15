from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
SWEEPS_DIR = ROOT / "experiments" / "pendulum-sweeps"
if str(SWEEPS_DIR) not in sys.path:
    sys.path.insert(0, str(SWEEPS_DIR))

import numpy as np
import pytest
import torch

from _worker import LookaheadTransformer, build_process
from kl_in_context import (
    batched_branch_tokens,
    compute_collision_merged_kl,
    find_crossover_k,
    khmm_letter_probs,
    plot_kl_vs_k,
    plot_kl_vs_position,
    run_kl_experiment,
    sample_eval_batch,
    true_letter_probs,
)
from lookahead_viz import compute_counterfactual_branches
from models.transformer import ModelConfig



@pytest.fixture
def test_proc():
    """Small MessDriven process for unit testing."""
    return build_process(
        delta_v=1.2,
        gamma=1.0,
        dt=0.2,
        n_steps=10,
        m=8,
        random_init=True,
    )


def test_batched_branch_tokens_matches_single_trajectory(test_proc):
    """Verify batched_branch_tokens exactly reproduces compute_counterfactual_branches."""
    batch = sample_eval_batch(test_proc, n_seq=4, seed=42)
    b_tokens = batched_branch_tokens(test_proc, batch)

    assert b_tokens.shape == (4, test_proc.m, 4, test_proc.n_steps)

    for i in range(4):
        single = compute_counterfactual_branches(
            test_proc, batch["letters"][i], initial_state=batch["initial_state"][i]
        )  # (4, seq_len)
        single_reshaped = single.reshape(4, test_proc.m, test_proc.n_steps).transpose(1, 0, 2)
        np.testing.assert_array_equal(b_tokens[i], single_reshaped)


def test_true_letter_probs_properties(test_proc):
    """Verify true_letter_probs produces valid probability distributions."""
    batch = sample_eval_batch(test_proc, n_seq=8, seed=123)
    p_true = true_letter_probs(test_proc, batch)

    assert p_true.shape == (8, test_proc.m, 4)
    assert np.all(p_true >= 0.0)
    assert np.all(p_true <= 1.0)
    np.testing.assert_allclose(p_true.sum(axis=-1), 1.0, atol=1e-12)


def test_khmm_letter_probs_bounds_and_monotonicity(test_proc):
    """Verify k-HMM letter probabilities: k=0 is uniform, k>=m matches true belief, and KL decreases with k."""
    batch = sample_eval_batch(test_proc, n_seq=32, seed=777)
    letters = batch["letters"]
    p_true = true_letter_probs(test_proc, batch)

    # 1. k=0 is uniform
    q0 = khmm_letter_probs(test_proc, letters, k=0)
    np.testing.assert_allclose(q0, 0.25, atol=1e-12)

    # 2. k >= m matches p_true
    qm = khmm_letter_probs(test_proc, letters, k=test_proc.m)
    np.testing.assert_allclose(qm, p_true, atol=1e-12)

    # 3. Monotonicity: compute average KL across sequences for k in [0, 1, 2, 4, 8]
    mean_kls = []
    for k_val in [0, 1, 2, 4, 8]:
        qk = khmm_letter_probs(test_proc, letters, k=k_val)
        kl = np.sum(p_true * np.log(np.maximum(p_true, 1e-12) / np.maximum(qk, 1e-12)), axis=-1)
        mean_kls.append(float(np.mean(kl)))

    for i in range(len(mean_kls) - 1):
        assert mean_kls[i] >= mean_kls[i + 1] - 1e-6, f"KL failed to decrease: {mean_kls}"


def test_collision_merge_and_leakage_decomposition():
    """Verify exact token collision merging and algebraic identity total_kl == belief_kl + leakage."""
    p_letters = np.array([0.4, 0.3, 0.2, 0.1])

    # 1. Synthetic full vocab model distribution
    vocab_size = 181
    q_vocab = np.zeros(vocab_size)
    # Put 80% mass on branch tokens, 20% mass elsewhere (leakage)
    branch_tokens = np.array([10, 20, 20, 40])  # tokens 20 collide!
    q_vocab[10] = 0.3
    q_vocab[20] = 0.4  # shared by letter 1 and 2
    q_vocab[40] = 0.1
    q_vocab[99] = 0.2  # off-branch leakage

    tot_kl, bel_kl, lkg = compute_collision_merged_kl(
        p_letters, q_vocab, branch_tokens, is_model=True
    )

    # Hand calculation:
    # unique tokens: 10, 20, 40
    # p_merged: [0.4, 0.3 + 0.2 = 0.5, 0.1]
    # q_merged: [0.3, 0.4, 0.1]
    # z = 0.3 + 0.4 + 0.1 = 0.8
    # q_norm = [0.3/0.8, 0.4/0.8, 0.1/0.8] = [0.375, 0.5, 0.125]
    expected_lkg = -np.log(0.8)
    expected_bel_kl = 0.4 * np.log(0.4 / 0.375) + 0.5 * np.log(0.5 / 0.5) + 0.1 * np.log(0.1 / 0.125)
    expected_tot_kl = expected_bel_kl + expected_lkg

    assert abs(lkg - expected_lkg) < 1e-9
    assert abs(bel_kl - expected_bel_kl) < 1e-9
    assert abs(tot_kl - expected_tot_kl) < 1e-9


def test_find_crossover_k():
    """Verify linear interpolation crossover order identification."""
    ks = [0, 1, 2, 4, 8]
    khmm_vals = [0.5, 0.3, 0.2, 0.1, 0.0]

    # Model KL = 0.25 lies halfway between k=1 (0.3) and k=2 (0.2)
    k_star = find_crossover_k(ks, khmm_vals, model_kl=0.25)
    assert abs(k_star - 1.5) < 1e-9

    # Model KL = 0.05 lies halfway between k=4 (0.1) and k=8 (0.0)
    k_star2 = find_crossover_k(ks, khmm_vals, model_kl=0.05)
    assert abs(k_star2 - 6.0) < 1e-9


def test_end_to_end_kl_experiment_and_plotting(tmp_path):
    """End-to-end smoke test of run_kl_experiment and plot generators."""
    proc = build_process(
        delta_v=1.2,
        gamma=1.0,
        dt=0.2,
        n_steps=10,
        m=4,
        random_init=True,
    )
    cfg = ModelConfig(vocab_size=181, n_ctx=proc.seq_len, n_layers=2, n_heads=1, d_model=32, d_mlp=64)
    model_k1 = LookaheadTransformer(cfg, k=1).eval()
    model_k1.k = 1
    model_k1.k_suffix = "k1"

    model_kn = LookaheadTransformer(cfg, k=10).eval()
    model_kn.k = 10
    model_kn.k_suffix = "kn"

    run_data = {
        "proc": proc,
        "models": {"k1": model_k1, "kn": model_kn},
        "config_name": "test_cfg",
        "width": 32,
        "seed": 0,
        "m_eval": proc.m,
        "n_steps": proc.n_steps,
    }

    results = run_kl_experiment(run_data, n_seq=8, ks=(0, 1, 2), device="cpu", batch_size=4)

    assert "horizons" in results
    assert "k1" in results["horizons"]
    assert "kn" in results["horizons"]
    assert len(results["horizons"]["k1"]["usable_positions"]) > 0

    # Test plotting functions
    fig1 = plot_kl_vs_position(results, yscale="both", decompose=True, min_pos=10, save_path=tmp_path / "plot1.png")
    assert (tmp_path / "plot1.png").exists()
    assert (tmp_path / "plot1.png").stat().st_size > 1000

    fig2 = plot_kl_vs_k(results, burn_in_tokens=10, save_path=tmp_path / "plot2.png")
    assert (tmp_path / "plot2.png").exists()
    assert (tmp_path / "plot2.png").stat().st_size > 1000

    from kl_in_context import plot_lookahead_probabilities
    fig3 = plot_lookahead_probabilities(run_data, window_tokens=20, save_path=tmp_path / "plot3.png")
    assert (tmp_path / "plot3.png").exists()
    assert (tmp_path / "plot3.png").stat().st_size > 1000
