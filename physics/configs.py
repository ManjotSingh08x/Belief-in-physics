"""Default configs for the four systems. Bin counts are scaled down from the
plan's target resolution (see docs/phase1-simulator-plan.md, "Grid sizing")
to keep Ulam-matrix build time tractable on a laptop CPU; scripts/build_hmms.py
reports nnz/build time/memory so these can be scaled back up deliberately.
"""
from __future__ import annotations

DEFAULT_CONFIGS: dict[str, dict] = {
    "pendulum": {
        "system": {"name": "pendulum", "g": 9.81, "length": 1.0, "gamma": 0.15, "kick": 1.5, "dt": 0.02},
        "hmm": {
            "latent_bins": (128, 96),
            "emission_bins": 64,
            "observable_periodic": True,
            "observable_dim": 0,
            "samples_per_bin": 200,
            "prune_below": 1e-4,
        },
        "perturbation": {"action_names": ["-dv", "noop", "+dv"]},
        "K": 8,
        "seed": 0,
    },
    "predator_prey": {
        "system": {
            "name": "predator_prey",
            "a": 1.0,
            "b": 0.6,
            "c": 0.8,
            "d": 0.4,
            "kappa": 5.0,
            "kick": 0.8,
            "dt": 0.02,
        },
        "hmm": {
            "latent_bins": (96, 96),
            "emission_bins": 64,
            "observable_periodic": False,
            "samples_per_bin": 200,
            "prune_below": 1e-4,
        },
        "perturbation": {"action_names": ["-prey", "+prey", "-pred", "+pred", "noop"]},
        "K": 8,
        "seed": 0,
    },
    "sphere": {
        "system": {"name": "sphere", "mu": 0.25, "kick": 1.2, "dt": 0.02},
        "hmm": {
            "latent_bins": (24, 24, 12, 12),
            "n_lat_cells": 16,
            "n_lon_cells_equator": 32,
            "noise_std": 0.08,
            "samples_per_bin": 120,
            "prune_below": 5e-4,
        },
        "perturbation": {"action_names": ["kick0", "kick120", "kick240", "noop"]},
        "K": 6,
        "seed": 0,
    },
    "double_pendulum": {
        "system": {
            "name": "double_pendulum",
            "g": 9.81,
            "l1": 1.0,
            "l2": 1.0,
            "m1": 1.0,
            "m2": 1.0,
            "gamma1": 0.08,
            "gamma2": 0.08,
            "kick": 8.0,
            "dt": 0.01,
        },
        "hmm": {
            "latent_bins": (32, 32, 14, 14),
            "emission_bins": 22,
            "observable_periodic": True,
            "observable_dim": 1,
            "samples_per_bin": 80,
            "prune_below": 1e-3,
        },
        "perturbation": {"action_names": ["kick1", "kick2", "kick_both", "noop"]},
        # K=10 (not 5): theta2's short-term future is driven directly by
        # omega2 but only indirectly, weakly by omega1 -- kicking omega1
        # alone is invisible in theta2 within a 5-step window at any kick
        # magnitude (checked up to kick=omega_max); K=10 gives that coupling
        # time to propagate. See docs/phase1-simulator-plan.md deviations.
        "K": 10,
        "seed": 0,
    },
}

NOOP_ACTION = {"pendulum": 1, "predator_prey": 4, "sphere": 3, "double_pendulum": 3}
