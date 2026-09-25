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

import os
from pathlib import Path

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

import analysis
from analysis import (
    build_precomputation_widget,
    build_process_from_meta,
    build_viewer_widget,
    cross_validated_ridge,
    discover_checkpoints,
    extract_probe_features,
    extract_probe_targets,
    get_metrics_table,
    load_analysis_csv,
    load_models_for_run,
    plot_layerwise_emergence,
    plot_macro_overview,
    plot_sweep_comparison,
    run_precomputation_job,
    sample_analysis_batch,
)


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
