import numpy as np
import pytest

from physics import cache as cache_module
from physics.builder import build_action_map, build_transition_matrix
from physics.grid import LatentGrid
from physics.hmm import DiscreteHMM
from physics.systems.pendulum import Pendulum


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(cache_module, "CACHE_DIR", tmp_path)
    yield


def _small_hmm():
    system = Pendulum(gamma=0.0)
    grid = LatentGrid(bin_counts=(8, 8), domain=system.domain, periodic=system.dims_periodic)
    rng = np.random.default_rng(0)
    T = build_transition_matrix(system, grid, K=3, dt=0.02, samples_per_bin=32, rng=rng)
    P = (build_action_map(system, grid, 0), build_action_map(system, grid, 1))
    E = np.full((grid.n_bins, 4), 0.25)
    return DiscreteHMM(T=T, P_actions=P, E=E, metric_table=np.zeros((grid.n_bins, 1)), grid=grid, action_names=("a", "b"))


def test_round_trip_preserves_matrices_and_grid():
    hmm = _small_hmm()
    cfg = {"system": {"name": "pendulum"}, "seed": 0}
    cache_module.save(cfg, hmm)
    loaded = cache_module.load(cfg)
    assert loaded is not None
    assert (loaded.T != hmm.T).nnz == 0
    assert np.array_equal(loaded.E, hmm.E)
    assert loaded.grid.bin_counts == hmm.grid.bin_counts


def test_load_returns_none_for_an_unknown_config():
    assert cache_module.load({"system": {"name": "never_saved"}}) is None
