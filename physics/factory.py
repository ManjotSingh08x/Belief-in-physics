"""make_simulator(config: dict) -> Simulator: the single entry point.

config = {
    "system": {"name": "pendulum", "g": 9.81, "length": 1.0, "gamma": 0.15, "dt": 0.01, ...},
    "hmm": {"latent_bins": (128, 96), "emission_bins": 64, "samples_per_bin": 200},
    "perturbation": {"action_names": [...]},   # optional, purely descriptive
    "K": 10,
    "seed": 0,
}
sim = make_simulator(config)
episodes = sim.simulate_batch(num_simulations=1024, N=50, M=8)
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from . import cache
from .builder import build_action_map, build_transition_matrix
from .channels import equal_area_sphere_cells, gaussian_channel, n_sphere_cells, sphere_cell_index
from .grid import LatentGrid
from .hmm import DiscreteHMM, Episodes
from .systems.double_pendulum import DoublePendulum
from .systems.pendulum import Pendulum
from .systems.predator_prey import PredatorPrey
from .systems.sphere import SphereBall

SYSTEM_REGISTRY = {
    "pendulum": Pendulum,
    "predator_prey": PredatorPrey,
    "sphere": SphereBall,
    "double_pendulum": DoublePendulum,
}


def _build_sphere_emission(system: SphereBall, grid: LatentGrid, hmm_cfg: dict) -> np.ndarray:
    centers = grid.centers()
    obs_vals = system.observable(centers)  # (n_latent, 2) = (lat, lon)
    n_lat = hmm_cfg.get("n_lat_cells", 16)
    n_lon = hmm_cfg.get("n_lon_cells_equator", 32)
    lat_edges, lon_edges = equal_area_sphere_cells(n_lat, n_lon)
    n_obs = n_sphere_cells(lon_edges)
    std = hmm_cfg.get("noise_std", 0.05)
    n_mc = hmm_cfg.get("emission_mc_samples", 64)
    rng = np.random.default_rng(hmm_cfg.get("emission_seed", 0))
    jitter = rng.normal(0.0, std, size=(obs_vals.shape[0], n_mc, 2))
    jittered_lat = np.clip(obs_vals[:, None, 0] + jitter[..., 0], lat_edges[0], lat_edges[-1])
    jittered_lon = obs_vals[:, None, 1] + jitter[..., 1]
    cell_idx = sphere_cell_index(jittered_lat.ravel(), jittered_lon.ravel(), lat_edges, lon_edges)
    cell_idx = cell_idx.reshape(obs_vals.shape[0], n_mc)
    E = np.empty((obs_vals.shape[0], n_obs))
    for i in range(obs_vals.shape[0]):
        counts = np.bincount(cell_idx[i], minlength=n_obs).astype(np.float64)
        E[i] = counts / counts.sum()
    E = np.clip(E, 1e-6, None)
    E /= E.sum(axis=1, keepdims=True)
    return E


def _build_emission(system, name: str, grid: LatentGrid, hmm_cfg: dict) -> np.ndarray:
    if name == "sphere":
        return _build_sphere_emission(system, grid, hmm_cfg)

    centers = grid.centers()
    obs_vals = system.observable(centers)  # (n_latent,) scalar observable
    n_obs = hmm_cfg["emission_bins"]
    is_periodic = hmm_cfg.get("observable_periodic", False)
    if is_periodic:
        dim = hmm_cfg.get("observable_dim", 0)
        lo, hi = system.domain[dim]
    else:
        lo, hi = float(obs_vals.min()), float(obs_vals.max())
        pad = (hi - lo) * 0.01 or 1.0
        lo, hi = lo - pad, hi + pad
    edges = np.linspace(lo, hi, n_obs + 1)
    std = hmm_cfg.get("noise_std", (hi - lo) / n_obs * 0.75)
    E = gaussian_channel(obs_vals, edges, std, periodic=is_periodic, period=(lo, hi))
    return E


@dataclass(frozen=True)
class Simulator:
    hmm: DiscreteHMM
    system: object
    K: int
    seed: int

    def simulate_batch(
        self,
        num_simulations: int,
        N: int,
        M: int,
        with_beliefs: bool = False,
        rng: np.random.Generator | None = None,
    ) -> Episodes:
        """Pass `rng` to stream distinct batches; the default reseeds from
        `self.seed` every call and so returns the *same* batch each time.
        """
        if rng is None:
            rng = np.random.default_rng(self.seed)
        episodes = self.hmm.sample_batch(rng, num_simulations, N, M, self.K)
        if with_beliefs:
            beliefs = self.hmm.forward(episodes.tokens, episodes.is_action)
            episodes = replace(episodes, beliefs=beliefs)
        return episodes


def make_simulator(config: dict) -> Simulator:
    sys_cfg = dict(config["system"])
    name = sys_cfg.pop("name")
    if name not in SYSTEM_REGISTRY:
        raise ValueError(f"unknown system {name!r}; choices: {sorted(SYSTEM_REGISTRY)}")
    dt = sys_cfg.pop("dt")
    system = SYSTEM_REGISTRY[name](**sys_cfg)

    hmm_cfg = config["hmm"]
    K = config["K"]
    grid = LatentGrid(
        bin_counts=tuple(hmm_cfg["latent_bins"]),
        domain=system.domain,
        periodic=system.dims_periodic,
    )

    hmm = cache.load(config)
    if hmm is None:
        rng = np.random.default_rng(config.get("seed", 0))
        T = build_transition_matrix(
            system,
            grid,
            K,
            dt,
            samples_per_bin=hmm_cfg.get("samples_per_bin", 200),
            rng=rng,
            prune_below=hmm_cfg.get("prune_below", 1e-4),
        )
        n_actions = system.n_actions
        P_actions = tuple(build_action_map(system, grid, a) for a in range(n_actions))
        E = _build_emission(system, name, grid, hmm_cfg)
        metric_table = system.metrics(grid.centers())
        if metric_table.ndim == 1:
            metric_table = metric_table[:, None]
        pert_cfg = config.get("perturbation", {})
        action_names = tuple(pert_cfg.get("action_names", [str(i) for i in range(n_actions)]))
        hmm = DiscreteHMM(T=T, P_actions=P_actions, E=E, metric_table=metric_table, grid=grid, action_names=action_names)
        cache.save(config, hmm)

    return Simulator(hmm=hmm, system=system, K=K, seed=config.get("seed", 0))


def _demo() -> None:
    config = {
        "system": {"name": "pendulum", "g": 9.81, "length": 1.0, "gamma": 0.15, "dt": 0.02},
        "hmm": {"latent_bins": (16, 12), "emission_bins": 8, "samples_per_bin": 32},
        "perturbation": {"action_names": ["-dv", "noop", "+dv"]},
        "K": 3,
        "seed": 0,
    }
    sim = make_simulator(config)
    ep = sim.simulate_batch(num_simulations=8, N=6, M=4)
    assert ep.tokens.shape == (8, 4 * (1 + 2))
    assert ep.tokens.min() >= 0 and ep.tokens.max() < sim.hmm.n_obs + sim.hmm.n_actions
    ep2 = sim.simulate_batch(num_simulations=4, N=6, M=2, with_beliefs=True)
    assert ep2.beliefs is not None and ep2.beliefs.shape == (4, 2 * 3, sim.hmm.n_latent)
    assert np.allclose(ep2.beliefs.sum(axis=-1), 1.0, atol=1e-8)
    print("factory ok")


if __name__ == "__main__":
    _demo()
