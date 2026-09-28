"""Unit tests for lookahead_viz.py.

Covers:
- Physical counterfactual branch computation across impulse horizons n.
- Verification that the actual letter branch matches the physical rollout at each tick.
- Verification of horizon-aware active branching rates (k=1: 1-step, k=n//2: half-cycle, k=n: full cycle).
- Verification of Bayes-optimal predictive distribution and NLL calculations.
- Preparation of run prediction data, likelihoods, and physical alignments.
- Multi-row stacked horizon plotting (3 rows stretched across breadth).
- Single horizon detailed lookahead branching plot with time cursor and probability slice.
- Full enhanced summary plot generation with stacked horizons, loss curves, and probe metrics.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

SWEEPS_DIR = Path(__file__).resolve().parent.parent / "experiments" / "pendulum-sweeps"
if str(SWEEPS_DIR) not in sys.path:
    sys.path.insert(0, str(SWEEPS_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import torch

from _worker import LookaheadTransformer, build_process
from lookahead_viz import (
    compute_counterfactual_branches,
    compute_horizon_branches,
    plot_enhanced_summary,
    plot_lookahead_branching,
    plot_stacked_horizons,
    prepare_run_data,
)
from models.transformer import ModelConfig


@pytest.mark.parametrize("n_steps", [5, 10, 15, 20])
def test_compute_counterfactual_branches(n_steps: int):
    m = 6
    proc = build_process(delta_v=1.2, gamma=0.8, dt=0.2, n_steps=n_steps, m=m)
    rng = np.random.default_rng(123)
    batch = proc.sample_batch(rng, 1)
    tokens = batch["tokens"][0]
    letters = batch["letters"][0]

    branches = compute_counterfactual_branches(proc, letters)
    assert branches.shape == (4, proc.seq_len)

    for j in range(m):
        actual_letter = letters[j]
        start = j * n_steps
        end = min((j + 1) * n_steps, proc.seq_len)
        assert np.array_equal(
            branches[actual_letter, start:end],
            tokens[start:end],
        ), f"Mismatch on tick {j} for n_steps={n_steps}"


@pytest.mark.parametrize("n_steps", [10, 20])
def test_compute_horizon_branches_branching_rates(n_steps: int):
    m = 8
    proc = build_process(delta_v=1.0, gamma=0.7, dt=0.2, n_steps=n_steps, m=m)
    rng = np.random.default_rng(42)
    batch = proc.sample_batch(rng, 1)
    tokens = batch["tokens"][0]
    letters = batch["letters"][0]
    beliefs = batch["beliefs"][0]
    seq_len = proc.seq_len

    # Test k = 1: Exactly 1 step per tick has an unobserved impulse
    h_k1 = compute_horizon_branches(proc, tokens, letters, k=1, beliefs=beliefs)
    is_br_k1 = h_k1["is_branching"]
    for tau in range(1, seq_len):
        if tau % n_steps == 0:
            assert is_br_k1[tau] is np.True_ or is_br_k1[tau] == True, f"tau={tau} at tick boundary should branch for k=1"
        else:
            assert is_br_k1[tau] is np.False_ or is_br_k1[tau] == False, f"tau={tau} within tick should be deterministic for k=1"

    # Test k = n_steps // 2: First half of each tick branches, second half deterministic
    k_half = n_steps // 2
    h_kh = compute_horizon_branches(proc, tokens, letters, k=k_half, beliefs=beliefs)
    is_br_kh = h_kh["is_branching"]
    for tau in range(k_half, seq_len):
        step_in_tick = tau % n_steps
        if step_in_tick < k_half:
            assert is_br_kh[tau] == True, f"tau={tau} (step {step_in_tick}) should branch for k={k_half}"
        else:
            assert is_br_kh[tau] == False, f"tau={tau} (step {step_in_tick}) should be deterministic for k={k_half}"

    # Test k = n_steps: 100% of forecast range branches
    h_kn = compute_horizon_branches(proc, tokens, letters, k=n_steps, beliefs=beliefs)
    assert np.all(h_kn["is_branching"][n_steps:]), "k=n should branch for all forecast steps"

    # Test deterministic properties: non-actual branches masked with NaN, Bayes optimal prob=1.0, NLL=0.0
    for tau in range(1, seq_len):
        if not is_br_k1[tau]:
            actual_l = letters[tau // n_steps]
            actual_tok = tokens[tau]
            assert h_k1["branch_tokens"][actual_l, tau] == actual_tok
            for l in range(4):
                if l != actual_l:
                    assert np.isnan(h_k1["branch_tokens"][l, tau]), f"Non-actual branch {l} must be NaN at tau={tau}"
            assert np.isclose(h_k1["bayes_probs"][tau, actual_tok], 1.0)
            assert np.isclose(h_k1["bayes_nll"][tau], 0.0)

    # Test stochastic properties: all 4 branches finite integers, Bayes optimal NLL > 0
    for tau in range(1, seq_len):
        if is_br_k1[tau]:
            actual_tok = tokens[tau]
            assert np.all(np.isfinite(h_k1["branch_tokens"][:, tau]))
            assert h_k1["bayes_probs"][tau].sum() > 0.99
            assert h_k1["bayes_nll"][tau] >= 0.0


def test_prepare_run_data_and_shapes():
    n_steps = 10
    m = 10
    proc = build_process(delta_v=1.0, gamma=0.7, dt=0.2, n_steps=n_steps, m=m)
    rng = np.random.default_rng(42)
    batch = proc.sample_batch(rng, 1)

    cfg = ModelConfig(
        vocab_size=proc.n_obs,
        n_ctx=proc.seq_len,
        n_layers=2,
        n_heads=1,
        d_model=32,
        d_mlp=128,
        seed=0,
    )
    k = 5
    model = LookaheadTransformer(cfg, k=k)
    model.eval()

    run = {
        "model": model,
        "k": k,
        "k_suffix": "kn2",
        "proc": proc,
        "run_name": "TestModel_kn2",
        "config_name": "cfg_test",
    }

    data = prepare_run_data(run, batch, device="cpu")

    assert data["k"] == k
    assert data["seq_len"] == proc.seq_len
    assert data["branch_tokens"].shape == (4, proc.seq_len)
    assert data["all_branches"].shape == (4, proc.seq_len)
    assert data["is_branching"].shape == (proc.seq_len,)
    assert data["bayes_probs"].shape == (proc.seq_len, proc.n_obs)
    assert data["bayes_nll"].shape == (proc.seq_len,)
    assert data["probs_k"].shape == (proc.seq_len - k, proc.n_obs)
    assert data["actual_k"].shape == (proc.seq_len - k,)
    assert data["nll"].shape == (proc.seq_len - k,)
    assert np.all(data["nll"] >= 0)
    assert "mean_bayes_nll" in data
    assert "suboptimality_gap" in data


def test_plot_stacked_horizons(tmp_path: Path):
    n_steps = 10
    m = 10
    proc = build_process(delta_v=1.0, gamma=0.7, dt=0.2, n_steps=n_steps, m=m)
    rng = np.random.default_rng(99)
    batch = proc.sample_batch(rng, 1)

    cfg = ModelConfig(
        vocab_size=proc.n_obs,
        n_ctx=proc.seq_len,
        n_layers=2,
        n_heads=1,
        d_model=32,
        d_mlp=128,
        seed=0,
    )

    data_list = []
    for k, suffix in [(1, "k1"), (5, "kn2"), (10, "kn")]:
        model = LookaheadTransformer(cfg, k=k)
        model.eval()
        run = {
            "model": model,
            "k": k,
            "k_suffix": suffix,
            "proc": proc,
            "run_name": f"Model_{suffix}",
            "config_name": "test_cfg",
        }
        data_list.append(prepare_run_data(run, batch, device="cpu"))

    save_file = tmp_path / "stacked_horizons_test.png"
    fig = plot_stacked_horizons(data_list, save_path=save_file, title="Test Stacked Horizons")
    assert fig is not None
    assert len(fig.axes) == 3
    assert save_file.exists()
    plt.close(fig)


def test_plot_lookahead_branching(tmp_path: Path):
    n_steps = 10
    m = 10
    proc = build_process(delta_v=1.0, gamma=0.7, dt=0.2, n_steps=n_steps, m=m)
    rng = np.random.default_rng(77)
    batch = proc.sample_batch(rng, 1)

    cfg = ModelConfig(
        vocab_size=proc.n_obs,
        n_ctx=proc.seq_len,
        n_layers=2,
        n_heads=1,
        d_model=32,
        d_mlp=128,
        seed=0,
    )
    model = LookaheadTransformer(cfg, k=3)
    model.eval()
    run = {
        "model": model,
        "k": 3,
        "k_suffix": "k3",
        "proc": proc,
        "run_name": "Model_k3",
        "config_name": "test_branching",
    }
    data = prepare_run_data(run, batch, device="cpu")

    # Test stochastic cursor
    save_file_stoch = tmp_path / "branching_stoch_test.png"
    fig_stoch = plot_lookahead_branching(data, tau=11, save_path=save_file_stoch)
    assert fig_stoch is not None
    assert len(fig_stoch.axes) == 2
    assert save_file_stoch.exists()
    plt.close(fig_stoch)

    # Test deterministic cursor
    save_file_det = tmp_path / "branching_det_test.png"
    fig_det = plot_lookahead_branching(data, tau=15, save_path=save_file_det)
    assert fig_det is not None
    assert len(fig_det.axes) == 2
    assert save_file_det.exists()
    plt.close(fig_det)


def test_plot_enhanced_summary(tmp_path: Path):
    n_steps = 10
    m = 10
    proc = build_process(delta_v=1.0, gamma=0.7, dt=0.2, n_steps=n_steps, m=m)
    width = 32
    seed = 0

    cfg = ModelConfig(
        vocab_size=proc.n_obs,
        n_ctx=proc.seq_len,
        n_layers=2,
        n_heads=1,
        d_model=width,
        d_mlp=128,
        seed=seed,
    )

    horizons = [(1, "k1"), (5, "kn2"), (10, "kn")]
    for k, suffix in horizons:
        m_obj = LookaheadTransformer(cfg, k=k)
        torch.save(m_obj.state_dict(), tmp_path / f"d{width}_{suffix}_seed{seed}_final.pt")
        meta = {"n_layers": 2}
        (tmp_path / f"d{width}_{suffix}_seed{seed}_final.json").write_text(json.dumps(meta))

    probe_rows = []
    for mode in ["single_layer_single_token", "all_layers_single_token", "single_layer_cycle", "all_layers_cycle", "single_layer_last_token", "all_layers_last_token"]:
        for tgt in ["belief", "physics"]:
            for m_type in ["trained", "random_init"]:
                probe_rows.append({
                    "mode": mode,
                    "target": tgt,
                    "model_type": m_type,
                    "test_r2": 0.75 if m_type == "trained" else 0.1,
                })
    probe_df = pd.DataFrame(probe_rows)

    job = {
        "config_name": "test_cfg_n10",
        "delta_v": 1.0,
        "gamma": 0.7,
        "dt": 0.2,
        "n_steps": n_steps,
        "m": m,
    }
    loss_hist = {
        "k1": [{"step": 0, "eval_loss": 5.0}, {"step": 50, "eval_loss": 4.5}],
        "kn2": [{"step": 0, "eval_loss": 5.1}, {"step": 50, "eval_loss": 4.6}],
        "kn": [{"step": 0, "eval_loss": 5.2}, {"step": 50, "eval_loss": 4.7}],
    }

    summary_file = tmp_path / "summary_test.png"
    fig = plot_enhanced_summary(
        config_dir=tmp_path,
        job=job,
        width=width,
        seed=seed,
        horizons=horizons,
        loss_histories=loss_hist,
        probe_df=probe_df,
        proc=proc,
        device="cpu",
        n_layers=2,
        n_heads=1,
        save_path=summary_file,
    )
    assert fig is not None
    assert summary_file.exists()
    plt.close(fig)
