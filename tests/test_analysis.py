"""Unit tests for experiments/pendulum-sweeps/analysis.py.

Covers:
- Checkpoint discovery in outputs/
- Model loading for TinyTransformer and LookaheadTransformer
- Physical system initialization and batch sampling
- Probe target extraction (physics and belief)
- Feature extraction across macro and layerwise modes
- Cross-validated Ridge regression with StandardScaler and GroupKFold
- Mini precomputation job execution and CSV caching
- Standalone plotting functions (macro, layerwise, sweep)
- Precomputation and Results Viewer widget construction
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

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

import analysis
from analysis import (
    build_precomputation_widget,
    build_process_from_meta,
    build_viewer_widget,
    cross_validated_ridge,
    discover_checkpoints,
    extract_probe_features,
    extract_probe_targets,
    fetch_probe_csvs,
    get_metrics_table,
    load_analysis_csv,
    load_models_for_run,
    load_range_csv,
    make_experiment_name,
    parse_experiment_name,
    plot_last_token_comparison,
    plot_last_token_layerwise,
    plot_layerwise_emergence,
    plot_macro_overview,
    plot_sweep_comparison,
    run_precomputation_job,
    sample_analysis_batch,
)
from models.transformer import ModelConfig, TinyTransformer


@pytest.fixture(scope="module", autouse=True)
def ensure_mock_checkpoints():
    analysis.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = ModelConfig(
        vocab_size=181,
        n_ctx=160,
        n_layers=4,
        n_heads=1,
        d_model=64,
        d_mlp=256,
        seed=0,
    )
    model = TinyTransformer(cfg)
    models = ["standard_next-token_seed0", "standard_next-token_seed1", "standard_lookahead-k5_seed0"]
    created_files = []
    for name in models:
        pt_path = analysis.OUTPUT_DIR / f"{name}_final.pt"
        json_path = analysis.OUTPUT_DIR / f"{name}_final.json"
        if not pt_path.exists():
            torch.save(model.state_dict(), pt_path)
            created_files.append(pt_path)
        if not json_path.exists():
            k_val = 5 if "k5" in name else 1
            meta = {
                "experiment": {
                    "name": "standard",
                    "seed": 20260917,
                    "n": 10,
                    "m": 16,
                    "physics": {"delta_v": 0.55, "damping_value": 0.3, "dt": 0.2, "random_init": True, "init_theta_std": 0.3},
                },
                "model": {
                    "vocab_size": 181,
                    "n_ctx": 160,
                    "n_layers": 4,
                    "n_heads": 1,
                    "d_model": 64,
                    "d_mlp": 256,
                    "seed": 0,
                },
                "training": {
                    "name": "next-token" if "next-token" in name else "lookahead-k5",
                    "seed": 0 if "seed0" in name else 1,
                    "k": k_val,
                },
                "hmm": {"n_states": 4, "alpha": 0.7, "stay": 0.7},
            }
            json_path.write_text(json.dumps(meta))
            created_files.append(json_path)

    analysis_csv = analysis.OUTPUT_DIR / "analysis.csv"
    if not analysis_csv.exists():
        rows = []
        for mod in ["single_layer_single_token", "all_layers_cycle", "usable_token_all_layers", "last_token_all_layers", "layer_0_single_token"]:
            for tgt in ["belief", "physics"]:
                rows.append({
                    "experiment": "standard",
                    "training": "next-token",
                    "model_type": "trained",
                    "mode": mod,
                    "target": tgt,
                    "test_r2": 0.85,
                    "cv_r2": 0.82,
                    "alpha": 1.0,
                    "feature_dim": 64,
                })
        pd.DataFrame(rows).to_csv(analysis_csv, index=False)
        created_files.append(analysis_csv)

    probe_csv = analysis.OUTPUT_DIR / "standard_next-token_seed0_probes.csv"
    if not probe_csv.exists():
        p_df = pd.DataFrame([{
            "experiment": "standard",
            "training": "next-token",
            "model_type": "trained",
            "mode": "all_layers_cycle",
            "target": "belief",
            "test_r2": 0.85,
            "cv_r2": 0.82,
            "feature_dim": 64,
        }])
        p_df.to_csv(probe_csv, index=False)
        created_files.append(probe_csv)

    yield


def test_discover_checkpoints():
    ckpts = discover_checkpoints()
    assert len(ckpts) >= 3, f"Expected at least 3 models in outputs, found {len(ckpts)}"
    for name, info in ckpts.items():
        assert "json_path" in info and info["json_path"].exists()
        assert "pt_path" in info and info["pt_path"].exists()
        assert "meta" in info and "model" in info["meta"]


def test_load_models_and_meta():
    ckpts = discover_checkpoints()
    name = "standard_next-token_seed0"
    assert name in ckpts, f"Missing {name}"
    trained_m, random_m, m_cfg, meta = load_models_for_run(ckpts[name], device="cpu")
    assert trained_m is not None
    assert random_m is not None
    assert m_cfg.n_layers == 4
    assert m_cfg.d_model == 64


def test_process_building_and_sampling():
    ckpts = discover_checkpoints()
    name = "standard_next-token_seed0"
    meta = ckpts[name]["meta"]
    proc = build_process_from_meta(meta["experiment"], meta.get("hmm"))
    assert proc is not None
    assert proc.seq_len == 160

    rng = np.random.default_rng(42)
    batch = sample_analysis_batch(proc, rng, n_trajectories=4)
    assert batch["tokens"].shape == (4, 160)
    assert batch["observable"].shape == (4, 160, 1)
    assert batch["metric"].shape == (4, 160, 1)
    assert batch["beliefs"].shape == (4, 160, 4)


def test_target_and_feature_extraction():
    ckpts = discover_checkpoints()
    name = "standard_next-token_seed0"
    meta = ckpts[name]["meta"]
    proc = build_process_from_meta(meta["experiment"], meta.get("hmm"))

    rng = np.random.default_rng(42)
    batch = sample_analysis_batch(proc, rng, n_trajectories=4)
    targets = extract_probe_targets(batch, proc, n_steps=10)

    assert "physics" in targets and targets["physics"].shape == (4 * 16, 1)
    assert "belief" in targets and targets["belief"].shape == (4 * 16, 3)

    trained_m, _, _, _ = load_models_for_run(ckpts[name], device="cpu")
    from models.analysis import residual_streams_batched

    streams = residual_streams_batched(trained_m, batch["tokens"], "cpu")
    assert len(streams) == 5  # emb + 4 layers

    macro_single = extract_probe_features(streams, "single_layer_single_token", n_steps=10)
    assert macro_single.shape == (4 * 16, 64)

    macro_all = extract_probe_features(streams, "all_layers_single_token", n_steps=10)
    assert macro_all.shape == (4 * 16, 320)

    layer_token = extract_probe_features(streams, "layer_2_single_token", n_steps=10)
    assert layer_token.shape == (4 * 16, 64)


def test_cross_validated_ridge():
    rng = np.random.default_rng(0)
    n_traj = 10
    m = 16
    n_rows = n_traj * m
    d_feat = 32

    X = rng.standard_normal((n_rows, d_feat))
    W = rng.standard_normal((d_feat, 3))
    y = X @ W + 0.1 * rng.standard_normal((n_rows, 3))

    groups = np.repeat(np.arange(n_traj), m)
    test_mask = np.array([g in {0, 1} for g in groups])
    dev_mask = ~test_mask

    res = cross_validated_ridge(
        X=X,
        y=y,
        groups=groups,
        dev_mask=dev_mask,
        test_mask=test_mask,
        ridge_alphas=(0.1, 1.0, 10.0),
        cv_folds=3,
    )

    assert "test_r2" in res
    assert "cv_r2" in res
    assert "alpha" in res
    assert res["test_r2"] > 0.8
    assert res["feature_dim"] == d_feat


def test_mini_precomputation_job(tmp_path: Path):
    original_csv = analysis.ANALYSIS_CSV
    test_csv = tmp_path / "test_analysis.csv"
    analysis.ANALYSIS_CSV = test_csv

    try:
        df = run_precomputation_job(
            target_models=["standard_next-token_seed0"],
            macro_modes=True,
            layerwise_token=True,
            layerwise_cycle=False,
            targets=["physics", "belief"],
            n_traj=6,
            cv_folds=2,
            alphas=(1.0,),
        )
        assert df is not None
        assert len(df) > 0
        assert test_csv.exists()
        assert "test_r2" in df.columns
    finally:
        analysis.ANALYSIS_CSV = original_csv


def test_standalone_plotting(tmp_path: Path):
    df = load_analysis_csv()
    assert df is not None and len(df) > 0

    fig_macro = plot_macro_overview("next-token", df=df, save_path=tmp_path / "macro.png")
    assert fig_macro is not None
    plt.close("all")

    fig_layer = plot_layerwise_emergence("next-token", df=df, save_path=tmp_path / "layer.png")
    plt.close("all")

    fig_sweep = plot_sweep_comparison(df=df, save_path=tmp_path / "sweep.png")
    assert fig_sweep is not None
    plt.close("all")

    tbl = get_metrics_table("next-token", df=df)
    assert isinstance(tbl, pd.DataFrame)
    assert len(tbl) > 0


def test_widget_builders():
    w_prep = build_precomputation_widget()
    assert w_prep is not None

    w_view = build_viewer_widget()
    assert w_view is not None


def test_range_csv_and_naming():
    df = load_range_csv()
    assert df is not None
    assert len(df) == 25
    for col in ("dt", "damping_value", "delta_v", "n_steps"):
        assert col in df.columns

    name = make_experiment_name(0.2, 0.3, 0.7, 10)
    assert name == "dt0.2_gamma0.3_dv0.7_n10"

    parsed = parse_experiment_name(name)
    assert parsed == {"dt": 0.2, "gamma": 0.3, "dv": 0.7, "n": 10}

    parsed_prefix = parse_experiment_name("pendulum_dt0.2_gamma0.8_dv1.2_n20")
    assert parsed_prefix == {"dt": 0.2, "gamma": 0.8, "dv": 1.2, "n": 20}


def test_fetch_probe_csvs_and_filtering():
    probes_all = fetch_probe_csvs()
    assert probes_all is not None
    assert len(probes_all) > 0
    assert "test_r2" in probes_all.columns

    probes_standard = fetch_probe_csvs("standard")
    assert probes_standard is not None
    assert len(probes_standard) > 0
    assert (probes_standard["experiment"] == "standard").all()

    df_filtered = load_analysis_csv(experiment="standard")
    assert df_filtered is not None
    assert (df_filtered["experiment"] == "standard").all()

    fig = plot_macro_overview("next-token", experiment="standard")
    assert fig is not None
    plt.close("all")

    tbl = get_metrics_table("next-token", experiment="standard")
    assert len(tbl) > 0
    assert (tbl["experiment"] == "standard").all()


@pytest.mark.parametrize("n", [5, 10, 15, 20])
@pytest.mark.parametrize("m", [6, 10, 16])
def test_last_token_preperturbation_indices(n: int, m: int):
    seq_len = m * n
    # For k=1, usable tokens are (p % n) + 1 >= n and p + 1 < seq_len
    token_indices = np.array([p for p in range(seq_len) if ((p % n) + 1 >= n and p + 1 < seq_len)], dtype=np.int64)

    assert len(token_indices) == m - 1

    for idx in token_indices:
        assert (idx + 1) % n == 0
        assert idx + 1 < seq_len


@pytest.mark.parametrize("n", [5, 10, 15, 20])
def test_last_token_feature_extraction_accuracy_and_shapes(n: int):
    m = 16
    n_traj = 4
    d_model = 32
    n_layers = 4
    seq_len = m * n

    streams = [
        np.zeros((n_traj, seq_len, d_model), dtype=np.float32)
        for _ in range(n_layers + 1)
    ]
    for l_idx in range(n_layers + 1):
        for t_idx in range(seq_len):
            streams[l_idx][:, t_idx, :] = l_idx * 10000 + t_idx

    expected_indices = np.array([p for p in range(seq_len) if ((p % n) + 1 >= n and p + 1 < seq_len)], dtype=np.int64)
    n_usable = len(expected_indices)

    all_feats = extract_probe_features(streams, "usable_token_all_layers", n_steps=n, m=m, k=1)
    assert all_feats.shape == (n_traj * n_usable, (n_layers + 1) * d_model)

    # Test backward compatibility alias
    all_feats_alias = extract_probe_features(streams, "last_token_all_layers", n_steps=n, m=m, k=1)
    assert all_feats_alias.shape == (n_traj * n_usable, (n_layers + 1) * d_model)

    unflattened_all = all_feats.reshape(n_traj, n_usable, (n_layers + 1) * d_model)
    for c_i, t_idx in enumerate(expected_indices):
        for l in range(n_layers + 1):
            expected_val = l * 10000 + t_idx
            assert np.allclose(
                unflattened_all[:, c_i, l * d_model : (l + 1) * d_model],
                expected_val,
            )

    for l in range(n_layers + 1):
        layer_feats = extract_probe_features(streams, f"layer_{l}_usable_token", n_steps=n, m=m, k=1)
        assert layer_feats.shape == (n_traj * n_usable, d_model)
        unflattened_layer = layer_feats.reshape(n_traj, n_usable, d_model)
        for c_i, t_idx in enumerate(expected_indices):
            assert np.allclose(unflattened_layer[:, c_i, :], l * 10000 + t_idx)


@pytest.mark.parametrize("n", [5, 10, 15, 20])
def test_last_token_target_extraction(n: int):
    m = 16
    n_traj = 4
    seq_len = m * n

    class MockChain:
        n_states = 4

    class MockProc:
        def __init__(self, m_val, seq_len_val, n_val):
            self.m = m_val
            self.n_steps = n_val
            self.seq_len = seq_len_val
            self.chain = MockChain()

        def usable_positions(self, k: int = 1):
            p = np.arange(self.seq_len - k)
            return p[(p % self.n_steps) + k >= self.n_steps]

    proc = MockProc(m_val=m, seq_len_val=seq_len, n_val=n)

    metric = np.arange(n_traj * seq_len, dtype=np.float32).reshape(n_traj, seq_len, 1)
    beliefs = np.ones((n_traj, seq_len, 4), dtype=np.float32) / 4.0
    batch = {"metric": metric, "beliefs": beliefs}

    targets = extract_probe_targets(batch, proc, n_steps=n, pre_perturbation_only=True, k=1)
    expected_indices = proc.usable_positions(k=1)
    n_usable = len(expected_indices)
    assert targets["physics"].shape == (n_traj * n_usable, 1)
    assert targets["belief"].shape == (n_traj * n_usable, 3)

    expected_physics = metric[:, expected_indices, :].reshape(-1, 1)
    assert np.allclose(targets["physics"], expected_physics)


def test_last_token_group_disjointness_and_cv():
    n_traj = 12
    m = 16
    n_test = 3
    n_usable = m - 1

    order = np.random.default_rng(42).permutation(n_traj)
    test_seqs = set(order[:n_test].tolist())

    groups_lt = np.repeat(np.arange(n_traj), n_usable)
    test_mask_lt = np.array([g in test_seqs for g in groups_lt])
    dev_mask_lt = ~test_mask_lt

    dev_trajs = set(groups_lt[dev_mask_lt])
    test_trajs = set(groups_lt[test_mask_lt])
    assert dev_trajs.isdisjoint(test_trajs)
    assert len(test_trajs) == n_test
    assert len(dev_trajs) == n_traj - n_test

    rng = np.random.default_rng(42)
    X = rng.standard_normal((len(groups_lt), 64))
    y = rng.standard_normal((len(groups_lt), 1))
    res = cross_validated_ridge(
        X, y, groups_lt, dev_mask_lt, test_mask_lt, ridge_alphas=(1.0, 10.0), cv_folds=3
    )
    assert "test_r2" in res
    assert "cv_r2" in res
    assert res["feature_dim"] == 64


def test_precomputation_job_last_token(tmp_path: Path):
    original_csv = analysis.ANALYSIS_CSV
    test_csv = tmp_path / "test_last_token_analysis.csv"
    analysis.ANALYSIS_CSV = test_csv

    try:
        df = run_precomputation_job(
            target_models=["standard_next-token_seed0"],
            macro_modes=False,
            layerwise_token=False,
            layerwise_cycle=False,
            last_token_modes=True,
            layerwise_last_token=True,
            targets=["physics", "belief"],
            n_traj=6,
            cv_folds=2,
            alphas=(1.0,),
        )
        assert df is not None
        assert len(df) > 0
        modes = set(df["mode"].unique())
        assert ("usable_token_all_layers" in modes) or ("last_token_all_layers" in modes)
        for l in range(5):
            assert (f"layer_{l}_usable_token" in modes) or (f"layer_{l}_last_token" in modes)

        assert "all_layers_cycle" not in modes
        assert "single_layer_cycle" not in modes

        fig_layer = plot_last_token_layerwise("next-token", df=df, save_path=tmp_path / "lt_layer.png")
        assert fig_layer is not None
        plt.close("all")

        fig_comp = plot_last_token_comparison(df=df, save_path=tmp_path / "lt_comp.png")
        assert fig_comp is not None
        plt.close("all")
    finally:
        analysis.ANALYSIS_CSV = original_csv


