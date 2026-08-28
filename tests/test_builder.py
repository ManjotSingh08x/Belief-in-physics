import numpy as np

from physics.builder import build_action_map, build_transition_matrix
from physics.grid import LatentGrid
from physics.systems.pendulum import Pendulum


def _grid():
    system = Pendulum(gamma=0.0)
    return system, LatentGrid(bin_counts=(24, 24), domain=system.domain, periodic=system.dims_periodic)


def test_transition_matrix_rows_sum_to_one():
    system, grid = _grid()
    rng = np.random.default_rng(0)
    T = build_transition_matrix(system, grid, K=5, dt=0.02, samples_per_bin=64, rng=rng)
    assert T.shape == (grid.n_bins, grid.n_bins)
    row_sums = np.asarray(T.sum(axis=1)).ravel()
    assert np.allclose(row_sums, 1.0, atol=1e-6)


def test_transition_matrix_concentrates_near_the_deterministic_flow():
    """Ulam-method sanity: at high samples_per_bin and a fine grid, the bulk
    of a row's mass should land near the bin containing flow(bin centre) --
    i.e. T really is a transfer-operator estimate, not noise.
    """
    system, grid = _grid()
    rng = np.random.default_rng(0)
    T = build_transition_matrix(system, grid, K=5, dt=0.02, samples_per_bin=256, rng=rng, prune_below=1e-5)
    centres = grid.centers()
    expected_dest = grid.index(system.flow(centres, 0.02, 5))
    row = 200
    row_probs = T.getrow(row).toarray().ravel()
    top_bin = row_probs.argmax()
    # deterministic destination should be within 1 latent bin of the mode
    assert abs(top_bin - expected_dest[row]) <= grid.bin_counts[1] + 1


def test_action_map_rows_sum_to_one():
    system, grid = _grid()
    P0 = build_action_map(system, grid, action=0)
    assert np.allclose(np.asarray(P0.sum(axis=1)).ravel(), 1.0)
