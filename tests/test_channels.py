import numpy as np

from physics.channels import equal_area_sphere_cells, gaussian_channel, n_sphere_cells, sphere_cell_index


def test_gaussian_channel_rows_sum_to_one():
    vals = np.array([-1.5, -0.5, 0.5, 1.5])
    edges = np.linspace(-2.0, 2.0, 5)
    E = gaussian_channel(vals, edges, noise_std=0.3)
    assert np.allclose(E.sum(axis=1), 1.0)


def test_gaussian_channel_peaks_at_the_bin_containing_the_true_value():
    vals = np.array([-1.5, -0.5, 0.5, 1.5])
    edges = np.linspace(-2.0, 2.0, 5)
    E = gaussian_channel(vals, edges, noise_std=0.3)
    assert list(E.argmax(axis=1)) == [0, 1, 2, 3]


def test_sphere_cells_partition_into_a_positive_cell_count():
    lat_edges, lon_edges = equal_area_sphere_cells(8, 16)
    n = n_sphere_cells(lon_edges)
    assert n > 0
    idx = sphere_cell_index(np.array([0.0, 0.5, -0.5]), np.array([0.0, 3.0, -3.0]), lat_edges, lon_edges)
    assert idx.shape == (3,)
    assert np.all(idx < n) and np.all(idx >= 0)
