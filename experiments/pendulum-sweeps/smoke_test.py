"""Smoke test runner for existence tests.

Executes the full suite of existence tests on lightweight parameters
and verifies that all visualization artifacts are produced.
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from existence_tests import (
    build_process_from_config,
    load_trained_and_random,
    generate_data,
    test_nullspace,
    test_layer_emergence,
    test_pca_alignment,
    test_temporal_selectivity,
    test_matched_physics,
    test_umap_geometry,
    plot_multi_config,
    plot_nullspace,
    plot_layer_emergence,
    plot_pca_alignment,
    plot_temporal_selectivity,
    plot_matched_physics,
    plot_umap_geometry,
)

SWEEP_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SWEEP_DIR / "parallel_output"
SMOKE_OUT_DIR = SWEEP_DIR / "smoke_test_outputs"
SMOKE_OUT_DIR.mkdir(parents=True, exist_ok=True)

CONFIGS = ["dt0.2_gamma1_dv1.2_n10"]
DATA_SEEDS = [42, 137]
N_DATA = 64
ALPHA = 1.0
VAR_THRESHOLD = 0.9
TRAIN_FRAC = 0.8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Starting smoke test on device: {DEVICE}...")

# 1. Init
cfg = CONFIGS[0]
proc = build_process_from_config(cfg, OUTPUT_DIR)
tr_m, rnd_m = load_trained_and_random(OUTPUT_DIR / cfg, 128, "kn2", 0, DEVICE)
datasets = {s: generate_data(proc, n=N_DATA, seed=s) for s in DATA_SEEDS}

# 2. Test A  (PCA+ridge, k from trained model; random baseline in the same call)
null_res = {cfg: [test_nullspace(tr_m, datasets[s], proc, "all_layers_single_token", ALPHA, TRAIN_FRAC,
                                 DEVICE, s, random_model=rnd_m, var_threshold=VAR_THRESHOLD)
                  for s in DATA_SEEDS]}
print("Test A k / k_null:", [(r["k"], r["k_null"]) for r in null_res[cfg]])
fig, _ = plot_multi_config(null_res, plot_nullspace, 2, max_cols=2, title="Smoke Test A")
fig.savefig(SMOKE_OUT_DIR / "test_a_nullspace.png", dpi=120, bbox_inches="tight")
plt.close(fig)

# 3. Test B
layer_res = {cfg: {"last_token": [test_layer_emergence(tr_m, datasets[s], proc, "last_token", ALPHA, TRAIN_FRAC,
                                                       DEVICE, s, random_model=rnd_m, var_threshold=VAR_THRESHOLD)
                                  for s in DATA_SEEDS]}}
print("Test B k per layer:", [d["k"] for d in layer_res[cfg]["last_token"][0]])
fig, _ = plot_multi_config(layer_res, plot_layer_emergence, 2, max_cols=2, title="Smoke Test B")
fig.savefig(SMOKE_OUT_DIR / "test_b_layer_emergence.png", dpi=120, bbox_inches="tight")
plt.close(fig)

# 4. Test C
pca_res = {cfg: {"all_layers_single_token": [test_pca_alignment(tr_m, datasets[s], proc, "all_layers_single_token",
                                                                ALPHA, TRAIN_FRAC, DEVICE, s, max_pcs=10,
                                                                random_model=rnd_m, var_threshold=VAR_THRESHOLD)
                                             for s in DATA_SEEDS]}}
print("Test C k_threshold:", [r["k_threshold"] for r in pca_res[cfg]["all_layers_single_token"]])
fig, _ = plot_multi_config(pca_res, plot_pca_alignment, 1, max_cols=1, title="Smoke Test C")
fig.savefig(SMOKE_OUT_DIR / "test_c_pca_alignment.png", dpi=120, bbox_inches="tight")
plt.close(fig)

# 5. Test D
temp_res = {cfg: [test_temporal_selectivity(tr_m, datasets[s], proc, ALPHA, TRAIN_FRAC, DEVICE, s,
                                            random_model=rnd_m, var_threshold=VAR_THRESHOLD)
                  for s in DATA_SEEDS]}
print("Test D k per position:", temp_res[cfg][0]["k"])
fig, _ = plot_multi_config(temp_res, plot_temporal_selectivity, 2, max_cols=2, title="Smoke Test D")
fig.savefig(SMOKE_OUT_DIR / "test_d_temporal_selectivity.png", dpi=120, bbox_inches="tight")
plt.close(fig)

# 6. Test G
umap_res = {cfg: {"all_layers_single_token": []}}
res_u = test_umap_geometry(tr_m, datasets[DATA_SEEDS[0]], proc, "all_layers_single_token", n_epochs=5, max_points=100, device=DEVICE)
umap_res[cfg]["all_layers_single_token"].append(res_u)

fig, _ = plot_multi_config(umap_res, plot_umap_geometry, 1, max_cols=1, title="Smoke Test G")
fig.savefig(SMOKE_OUT_DIR / "test_g_umap_geometry.png", dpi=120, bbox_inches="tight")
plt.close(fig)

# 7. Test F
matched_res = {cfg: []}
res_f = test_matched_physics(tr_m, datasets[DATA_SEEDS[0]], proc, cycle_range=(10, 45), epsilon=0.15, belief_threshold=0.15, device=DEVICE)
matched_res[cfg].append(res_f)

fig, _ = plot_multi_config(matched_res, plot_matched_physics, 2, max_cols=2, title="Smoke Test F")
fig.savefig(SMOKE_OUT_DIR / "test_f_matched_physics.png", dpi=120, bbox_inches="tight")
plt.close(fig)

print(f"Smoke test complete! Plots saved in: {SMOKE_OUT_DIR}")
for p in SMOKE_OUT_DIR.glob("*.png"):
    assert p.stat().st_size > 0, f"Empty file: {p}"
    print(f"  Verified plot: {p.name} ({p.stat().st_size:,} bytes)")
