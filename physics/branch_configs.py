"""Per-system settings for the exact branch process, and the factory for it.

Two constraints set every number here.

**Branch count is n_z0 * n_actions^M.** Exact enumeration buys us a belief with
no discretisation error, and the price is sequence length: predator-prey has 5
actions, so M=8 would mean 1.56M branches. M is therefore chosen per system to
keep the emission tables and the belief array bounded, which is why the four
systems do not share a sequence length.

**The belief has to stay genuinely uncertain.** If the observation is too
informative the posterior collapses onto a vertex of the simplex and "the
residual stream encodes the belief" degenerates into "it encodes a discrete
label" -- the opposite of the Mess3 phenomenon, whose fractal exists precisely
because the process stays ambiguous. Too coarse and the belief is uniform, with
no information to probe at all. `emission_bins`, `noise_std` and `kick` were
swept jointly against normalised belief entropy; the values below sit in the
band where entropy is 0.50-0.76 of maximum while the posterior still identifies
the hidden perturbation at 1.8-2.9x chance. `scripts/tune_branch.py` reproduces
the sweep.
"""

from __future__ import annotations

import numpy as np

from .branch import build_branch_process
from .channels import equal_area_sphere_cells, n_sphere_cells, sphere_cell_index
from .systems.double_pendulum import DoublePendulum
from .systems.pendulum import Pendulum
from .systems.predator_prey import PredatorPrey
from .systems.sphere import SphereBall

STEPS_PER_SEGMENT = 8

BRANCH_CONFIGS: dict[str, dict] = {
    "pendulum": {
        "system": Pendulum, "system_kwargs": {"gamma": 0.15, "kick": 1.5},
        "z0": [[0.4, 0.0], [-0.3, 0.5], [1.0, -0.4], [-0.8, 0.2]],
        "K": 8, "dt": 0.02, "M": 8,
        "emission_bins": 16, "noise_std": 0.5,
        "obs_range": (-np.pi, np.pi), "periodic": True,
        "action_names": ("-dv", "noop", "+dv"),
    },
    "predator_prey": {
        "system": PredatorPrey, "system_kwargs": {"kick": 0.4},
        "z0": np.log([[3.0, 2.0], [1.5, 0.8], [2.5, 1.5], [1.2, 1.2]]).tolist(),
        "K": 8, "dt": 0.02, "M": 6,  # 5 actions, so M is the tightest here
        "emission_bins": 8, "noise_std": 0.09,
        "obs_range": (0.0, 1.0), "periodic": False,
        "action_names": ("-prey", "+prey", "-pred", "+pred", "noop"),
    },
    "sphere": {
        "system": SphereBall, "system_kwargs": {"mu": 0.25, "kick": 0.40},
        "z0": [[0.2, 0.0, 0.5, 0.3], [-0.3, 1.0, -0.4, 0.5], [0.5, -1.0, 0.3, -0.4], [0.0, 2.0, 0.6, 0.1]],
        "K": 6, "dt": 0.02, "M": 7,
        "n_lat_cells": 4, "n_lon_cells_equator": 8, "noise_std": 0.45,
        "action_names": ("kick0", "kick120", "kick240", "noop"),
    },
    # --- no-no-op Mess-4 variants ------------------------------------------
    # Both drop the no-op, so every action displaces the state, and both are
    # variance-matched to the parent config: the per-segment variance of the
    # kick sets how fast the belief spreads, so holding it fixed keeps the
    # dynamical regime comparable and makes the comparison about action-set
    # GEOMETRY rather than about a harder or easier process.
    #
    # They fill the 3-simplex in the two different ways available:
    #   pendulum       4 graded kicks on one axis  -> metric image is COLLINEAR
    #   predator_prey  4 unit kicks on two axes    -> metric image is a CROSS
    # so together they separate "encodes the simplex" from "encodes the metric".
    "pendulum_mess4": {
        # {-2,-1,+1,+2}*s with 2.5*s^2 == (2/3)*1.5^2  ->  s = sqrt(0.6)
        "system": Pendulum,
        "system_kwargs": {"gamma": 0.15, "kicks": tuple(m * np.sqrt(0.6) for m in (-2.0, -1.0, 1.0, 2.0))},
        "z0": [[0.4, 0.0], [-0.3, 0.5], [1.0, -0.4], [-0.8, 0.2]],
        "K": 8, "dt": 0.02, "M": 7,  # 4^7 * 4 z0 = 65,536 branches, the proven size
        "emission_bins": 16, "noise_std": 0.5,
        "obs_range": (-np.pi, np.pi), "periodic": True,
        "action_names": ("-2dv", "-dv", "+dv", "+2dv"),
    },
    # --- the all-positive 3-action set, no no-op --------------------------
    # {+1,+2,+3}*dv has variance (2/3)dv^2, which at dv = 1.5 is exactly the
    # parent's (2/3)*1.5^2. So this is a PURE mean shift against `pendulum`:
    # identical spread, identical M and branch count, mean +3.0 per segment
    # instead of 0. Whatever differs in the belief geometry is the drift.
    "pendulum_mess3_plus": {
        "system": Pendulum, "system_kwargs": {"gamma": 0.15, "kicks": (1.5, 3.0, 4.5)},
        "z0": [[0.4, 0.0], [-0.3, 0.5], [1.0, -0.4], [-0.8, 0.2]],
        "K": 8, "dt": 0.02, "M": 8,
        "emission_bins": 16, "noise_std": 0.5,
        "obs_range": (-np.pi, np.pi), "periodic": True,
        "action_names": ("+dv", "+2dv", "+3dv"),
    },
    # The same set shrunk until the drift no longer runs into omega_max. Mean
    # kick is 2*dv per segment against per-segment damping retention
    # exp(-gamma*K*dt*steps_per_segment) = 0.825, so omega settles at 2*dv/0.175;
    # dv = 0.35 puts that at 4.0, half the clip. The price is the whole point:
    # variance falls to 0.082, eighteen times less than the parent's 1.5.
    "pendulum_mess3_plus_small": {
        "system": Pendulum, "system_kwargs": {"gamma": 0.15, "kicks": (0.35, 0.70, 1.05)},
        "z0": [[0.4, 0.0], [-0.3, 0.5], [1.0, -0.4], [-0.8, 0.2]],
        "K": 8, "dt": 0.02, "M": 8,
        "emission_bins": 16, "noise_std": 0.5,
        "obs_range": (-np.pi, np.pi), "periodic": True,
        "action_names": ("+dv", "+2dv", "+3dv"),
    },
    "predator_prey_mess4": {
        # the legacy cross with the no-op deleted; per-coordinate variance goes
        # (2/5)k^2 -> (1/2)k'^2, so k' = k*sqrt(4/5)
        "system": PredatorPrey,
        "system_kwargs": {"kicks": tuple(
            (dx * 0.4 * np.sqrt(0.8), dy * 0.4 * np.sqrt(0.8))
            for dx, dy in ((-1.0, 0.0), (1.0, 0.0), (0.0, -1.0), (0.0, 1.0))
        )},
        "z0": np.log([[3.0, 2.0], [1.5, 0.8], [2.5, 1.5], [1.2, 1.2]]).tolist(),
        "K": 8, "dt": 0.02, "M": 7,
        "emission_bins": 8, "noise_std": 0.09,
        "obs_range": (0.0, 1.0), "periodic": False,
        "action_names": ("-prey", "+prey", "-pred", "+pred"),
    },
    "double_pendulum": {
        "system": DoublePendulum, "system_kwargs": {"kick": 5.0},
        "z0": [[0.9, -0.4, 0.0, 0.0], [-0.6, 0.8, 0.2, -0.1], [1.2, 0.3, -0.3, 0.2], [-1.0, -0.9, 0.1, 0.3]],
        "K": 10, "dt": 0.01, "M": 7,
        "emission_bins": 12, "noise_std": 0.6,
        "obs_range": (-np.pi, np.pi), "periodic": True,
        "action_names": ("kick1", "kick2", "kick_both", "noop"),
    },
}


def sphere_emission_fn(n_lat: int, n_lon: int, noise_std: float, seed: int = 0, n_mc: int = 64):
    """Categorical over equal-area sphere cells, by jittering (lat, lon).

    The sphere's observable is two-dimensional, so it cannot use the scalar
    Gaussian channel; this is the same construction `factory._build_sphere_
    emission` uses, vectorised over rows because the branch tables have far more
    of them than the latent grid did.
    """
    lat_edges, lon_edges = equal_area_sphere_cells(n_lat, n_lon)
    n_obs = n_sphere_cells(lon_edges)
    rng = np.random.default_rng(seed)

    def emission(obs: np.ndarray) -> np.ndarray:
        n = obs.shape[0]
        jitter = rng.normal(0.0, noise_std, size=(n, n_mc, 2))
        lat = np.clip(obs[:, None, 0] + jitter[..., 0], lat_edges[0], lat_edges[-1])
        lon = obs[:, None, 1] + jitter[..., 1]
        cells = sphere_cell_index(lat.ravel(), lon.ravel(), lat_edges, lon_edges).reshape(n, n_mc)
        flat = (np.arange(n)[:, None] * n_obs + cells).ravel()
        counts = np.bincount(flat, minlength=n * n_obs).reshape(n, n_obs).astype(np.float64)
        E = np.clip(counts / counts.sum(axis=1, keepdims=True), 1e-6, None)
        return E / E.sum(axis=1, keepdims=True)

    return emission


def make_branch_process(name: str, **overrides):
    """Build the exact branch process for one system."""
    if name not in BRANCH_CONFIGS:
        raise ValueError(f"unknown system {name!r}; have {sorted(BRANCH_CONFIGS)}")
    cfg = {**BRANCH_CONFIGS[name], **overrides}
    system = cfg["system"](**cfg["system_kwargs"])

    if name == "sphere":
        emission_fn = sphere_emission_fn(
            cfg["n_lat_cells"], cfg["n_lon_cells_equator"], cfg["noise_std"]
        )
        bin_edges, periodic, period = np.zeros(2), False, None
    else:
        lo, hi = cfg["obs_range"]
        emission_fn = None
        bin_edges = np.linspace(lo, hi, cfg["emission_bins"] + 1)
        periodic, period = cfg["periodic"], (lo, hi)

    return build_branch_process(
        system,
        np.asarray(cfg["z0"], dtype=np.float64),
        K=cfg["K"],
        dt=cfg["dt"],
        steps_per_segment=cfg.get("steps_per_segment", STEPS_PER_SEGMENT),
        M=cfg["M"],
        bin_edges=bin_edges,
        noise_std=cfg["noise_std"],
        action_names=cfg["action_names"],
        periodic=periodic,
        period=period,
        emission_fn=emission_fn,
    )


def scaled_kick(name: str, scale: float) -> dict:
    """Overrides that multiply a system's perturbation magnitude.

    Phase 1 chose each `kick` as the *smallest* value clearing its separability
    threshold, which means the coupling between the belief and the next-token
    distribution -- the quantity claim 1 reports -- was set by that choice. This
    is the knob for testing whether the claim is a property of the model or a
    restatement of the threshold.
    """
    if scale == 1.0:
        return {}
    base = BRANCH_CONFIGS[name]["system_kwargs"]
    if "kicks" in base:  # explicit ladder: scale every entry, scalar or vector
        scaled = tuple(
            tuple(v * scale for v in k) if hasattr(k, "__len__") else k * scale
            for k in base["kicks"]
        )
        return {"system_kwargs": {**base, "kicks": scaled}}
    return {"system_kwargs": {**base, "kick": base["kick"] * scale}}
