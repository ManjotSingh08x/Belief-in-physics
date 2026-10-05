"""Unit tests for belief existence tests checklist."""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

SWEEP_DIR = Path(__file__).resolve().parents[1] / "experiments" / "pendulum-sweeps"
if str(SWEEP_DIR) not in sys.path:
    sys.path.insert(0, str(SWEEP_DIR))

import existence_tests as et

OUTPUT_DIR = SWEEP_DIR / "parallel_output"
CFG = "dt0.2_gamma1_dv1.2_n10"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def setup_env():
    proc = et.build_process_from_config(CFG, OUTPUT_DIR)
    tr_m, rnd_m = et.load_trained_and_random(OUTPUT_DIR / CFG, 128, "kn2", 0, DEVICE)
    data = et.generate_data(proc, n=32, seed=42)
    return {"proc": proc, "tr_m": tr_m, "rnd_m": rnd_m, "data": data}


def test_c1_process_and_model_loading(setup_env):
    proc = setup_env["proc"]
    tr_m = setup_env["tr_m"]
    rnd_m = setup_env["rnd_m"]
    assert proc.seq_len == 500
    assert proc.n_steps == 10
    assert proc.m == 50
    assert tr_m.config.d_model == 128
    assert rnd_m.config.d_model == 128


def test_c2_data_and_sequence_split(setup_env):
    proc = setup_env["proc"]
    data = setup_env["data"]
    assert data["tokens"].shape == (32, 500)
    assert data["belief_coords"].shape == (32, 500, 3)
    assert data["physics_state"].shape == (32, 500, 2)
    assert data["beliefs"].shape == (32, 500, 4)

    tr, te = et.sequence_split(32, 0.75, 42)
    assert len(tr) == 24 and len(te) == 8
    assert len(set(tr).intersection(set(te))) == 0


def test_c3_gpu_ridge():
    X_tr = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]], dtype=np.float32)
    y_tr = np.array([[2.0], [4.0], [6.0]], dtype=np.float32)
    X_te = np.array([[4.0, 8.0], [5.0, 10.0], [6.0, 12.0]], dtype=np.float32)
    y_te = np.array([[8.0], [10.0], [12.0]], dtype=np.float32)

    W = et.gpu_ridge_fit(X_tr, y_tr, alpha=1e-3, device=DEVICE)
    assert W.shape == (2, 1)

    r2 = et.gpu_ridge_r2(X_tr, y_tr, X_te, y_te, alpha=1e-3, device=DEVICE)
    assert r2 > 0.99


def test_c4_feature_and_target_extraction(setup_env):
    proc = setup_env["proc"]
    m = setup_env["tr_m"]
    data = setup_env["data"]

    for mode, expected_cols in [
        ("all_layers_single_token", 5 * 128),
        ("all_layers_cycle", 10 * 5 * 128),
        ("all_tokens_flat", 5 * 128),
    ]:
        X = et.extract_features(m, data["tokens"], proc, mode, DEVICE)
        b, p = et.extract_targets(data, proc, mode)
        assert X.shape[0] == b.shape[0] == p.shape[0]
        assert X.shape[1] == expected_cols


def test_c5_test_nullspace(setup_env):
    proc, tr_m, rnd_m, data = (setup_env[k] for k in ("proc", "tr_m", "rnd_m", "data"))

    res = et.test_nullspace(tr_m, data, proc, "all_layers_single_token", 1.0, 0.8, DEVICE, 42,
                            random_model=rnd_m, var_threshold=0.9)
    assert res["null_belief_r2"] > res["null_belief_r2_random"]
    assert res["null_feature_dim"] == res["feature_dim"] - 2
    assert 1 <= res["k"] <= res["feature_dim"] and res["var_explained"] >= 0.9
    assert not np.isnan(res["full_belief_r2_random"])


def test_c6_test_layer_emergence(setup_env):
    proc, tr_m, rnd_m, data = (setup_env[k] for k in ("proc", "tr_m", "rnd_m", "data"))

    res = et.test_layer_emergence(tr_m, data, proc, "last_token", 1.0, 0.8, DEVICE, 42, random_model=rnd_m)
    assert len(res) == 6
    assert [r["layer"] for r in res] == ["emb", "L1", "L2", "L3", "L4", "all"]
    assert all(r["k"] >= 1 and not np.isnan(r["belief_r2_random"]) for r in res)


def test_c7_test_pca_alignment(setup_env):
    proc, tr_m, rnd_m, data = (setup_env[k] for k in ("proc", "tr_m", "rnd_m", "data"))

    res = et.test_pca_alignment(tr_m, data, proc, "all_layers_single_token", 1.0, 0.8, DEVICE, 42,
                                max_pcs=10, random_model=rnd_m)
    assert len(res["n_pcs"]) == len(res["belief_r2"]) == len(res["belief_r2_random"])
    assert res["k_threshold"] in res["n_pcs"]
    assert res["belief_r2"][-1] >= res["belief_r2"][0]


def test_c8_test_temporal_selectivity(setup_env):
    proc, tr_m, rnd_m, data = (setup_env[k] for k in ("proc", "tr_m", "rnd_m", "data"))

    res = et.test_temporal_selectivity(tr_m, data, proc, 1.0, 0.8, DEVICE, 42, random_model=rnd_m)
    assert len(res["positions"]) == len(res["belief_r2"]) == len(res["physics_r2"]) == 10
    assert len(res["k"]) == len(res["belief_r2_random"]) == 10


def test_c12_pca_ridge_probe():
    rng = np.random.default_rng(0)
    z = rng.normal(size=(600, 3)).astype(np.float32)
    mix = rng.normal(size=(3, 40)).astype(np.float32)
    X = z @ mix + 0.01 * rng.normal(size=(600, 40)).astype(np.float32)  # rank-3 signal
    Xr = rng.normal(size=(600, 40)).astype(np.float32)                  # no structure
    y = z[:, :2] @ np.array([[1.0, 2.0], [-1.0, 0.5]], dtype=np.float32)

    res = et.pca_ridge_probe(X[:450], X[450:], {"t": (y[:450], y[450:])}, 1.0, DEVICE, 0.99,
                             X_rand_tr=Xr[:450], X_rand_te=Xr[450:])
    assert res["k"] == 3 and res["var_explained"] >= 0.99      # k from trained data only
    assert res["r2"]["t"] > 0.99
    assert res["random"]["k"] == 3                              # same k applied to baseline
    assert res["random"]["r2"]["t"] < 0.2
    assert res["probe"]["V"].shape == (40, 3) and res["probe"]["W"]["t"].shape == (3, 2)
    # returned probe reproduces the reported R²
    pred = ((X[450:] - res["probe"]["mean"]) @ res["probe"]["V"]) @ res["probe"]["W"]["t"] + res["probe"]["y_mean"]["t"]
    assert 1 - ((y[450:] - pred) ** 2).sum() / ((y[450:] - y[450:].mean(0)) ** 2).sum() > 0.99
    assert et.pca_ridge_probe(X[:450], X[450:], {"t": (y[:450], y[450:])}, 1.0, DEVICE)["random"] is None


def test_c9_test_matched_physics(setup_env):
    proc = setup_env["proc"]
    tr_m = setup_env["tr_m"]
    data = setup_env["data"]

    res = et.test_matched_physics(tr_m, data, proc, cycle_range=(10, 45), epsilon=0.2, belief_threshold=0.1, device=DEVICE)
    assert "n_matched_pairs" in res
    assert "mean_matched_l2" in res


def test_c10_test_umap_geometry(setup_env):
    proc = setup_env["proc"]
    tr_m = setup_env["tr_m"]
    data = setup_env["data"]

    res = et.test_umap_geometry(tr_m, data, proc, "all_layers_single_token", n_epochs=5, max_points=100, device=DEVICE)
    assert res["embedding"].shape == (100, 2)
    assert res["beliefs"].shape == (100, 4)


def test_c11_plotting_and_multiconfig_layout():
    b = np.array([[1.0, 0.0, 0.0, 0.0], [0.25, 0.25, 0.25, 0.25]])
    cols = et.belief_colors(b)
    assert cols.shape == (2, 4)

    fig, axes = et.plot_multi_config(
        {"cfg1": None, "cfg2": None},
        lambda ax, c, r, i: [ax[0].plot([0, 1]), ax[1].plot([1, 0])],
        n_plots_per_config=2,
    )
    assert len(axes) == 2
    plt.close(fig)
