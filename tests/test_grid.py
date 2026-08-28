import numpy as np

from physics.grid import LatentGrid


def test_bin_centre_maps_back_to_its_own_index():
    g = LatentGrid(bin_counts=(4, 3), domain=((-np.pi, np.pi), (-1.0, 1.0)), periodic=(True, False))
    centres = g.centers()
    assert centres.shape == (12, 2)
    assert np.array_equal(g.index(centres), np.arange(12))


def test_samples_within_a_bin_map_back_to_that_bin():
    g = LatentGrid(bin_counts=(4, 3), domain=((-np.pi, np.pi), (-1.0, 1.0)), periodic=(True, False))
    rng = np.random.default_rng(0)
    samples = g.sample_in_bin(rng, np.array([0, 5, 11]), 50)
    idx = g.index(samples)
    assert np.all(idx[0] == 0) and np.all(idx[1] == 5) and np.all(idx[2] == 11)


def test_periodic_dim_wraps_across_the_domain_boundary():
    g = LatentGrid(bin_counts=(4, 3), domain=((-np.pi, np.pi), (-1.0, 1.0)), periodic=(True, False))
    just_above = g.index(np.array([np.pi + 0.01, 0.0]))
    just_below = g.index(np.array([-np.pi + 0.01, 0.0]))
    assert just_above == just_below


def test_nonperiodic_dim_clips_at_the_domain_boundary():
    g = LatentGrid(bin_counts=(4, 3), domain=((-np.pi, np.pi), (-1.0, 1.0)), periodic=(True, False))
    idx_over = g.index(np.array([0.0, 5.0]))
    idx_at_edge = g.index(np.array([0.0, 0.999]))
    assert idx_over == idx_at_edge
