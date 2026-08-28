"""Emission construction: turn a system's `observable(z)` into a stochastic
(n_latent, n_obs) matrix E, where E[i, o] = P(observation bin o | latent bin i).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm


def gaussian_channel(
    obs_values: np.ndarray,
    bin_edges: np.ndarray,
    noise_std: float,
    periodic: bool = False,
    period: tuple[float, float] | None = None,
) -> np.ndarray:
    """1-D emission: the true observable is Gaussian-blurred by `noise_std`,
    then binned by `bin_edges`. `obs_values` is (n_latent,), evaluated at
    latent bin centres.
    """
    lo_arr, hi_arr = bin_edges[:-1], bin_edges[1:]
    if periodic:
        assert period is not None
        lo, hi = period
        span = hi - lo
        E = np.zeros((obs_values.shape[0], len(bin_edges) - 1))
        for k in (-1, 0, 1):
            mu = obs_values[:, None] + k * span
            E += norm.cdf(hi_arr[None, :], mu, noise_std) - norm.cdf(lo_arr[None, :], mu, noise_std)
    else:
        mu = obs_values[:, None]
        E = norm.cdf(hi_arr[None, :], mu, noise_std) - norm.cdf(lo_arr[None, :], mu, noise_std)
    E = np.clip(E, 1e-12, None)
    E /= E.sum(axis=1, keepdims=True)
    return E


def equal_area_sphere_cells(n_lat: int, n_lon_at_equator: int):
    """Latitude bands of equal sin(lat) measure; each split into a
    longitude count proportional to cos(lat) so cell area stays roughly
    constant. Returns (lat_edges (n_lat+1,), lon_edges: list[np.ndarray]).
    """
    sin_edges = np.linspace(-1.0, 1.0, n_lat + 1)
    lat_edges = np.arcsin(np.clip(sin_edges, -1.0, 1.0))
    centers = (lat_edges[:-1] + lat_edges[1:]) / 2
    lon_counts = np.maximum(1, np.round(n_lon_at_equator * np.cos(centers)).astype(int))
    lon_edges = [np.linspace(-np.pi, np.pi, c + 1) for c in lon_counts]
    return lat_edges, lon_edges


def sphere_cell_index(lat: np.ndarray, lon: np.ndarray, lat_edges: np.ndarray, lon_edges: list[np.ndarray]) -> np.ndarray:
    """Flat cell index for (lat, lon) arrays under the equal-area scheme."""
    n_lat = len(lat_edges) - 1
    lat_idx = np.clip(np.searchsorted(lat_edges, lat, side="right") - 1, 0, n_lat - 1)
    lon_wrapped = np.mod(lon + np.pi, 2 * np.pi) - np.pi
    offsets = np.concatenate([[0], np.cumsum([len(e) - 1 for e in lon_edges])])
    out = np.empty_like(lat_idx)
    for b in range(n_lat):
        mask = lat_idx == b
        if not np.any(mask):
            continue
        edges = lon_edges[b]
        lon_idx = np.clip(np.searchsorted(edges, lon_wrapped[mask], side="right") - 1, 0, len(edges) - 2)
        out[mask] = offsets[b] + lon_idx
    return out


def n_sphere_cells(lon_edges: list[np.ndarray]) -> int:
    return int(sum(len(e) - 1 for e in lon_edges))


def _demo() -> None:
    vals = np.array([-1.5, -0.5, 0.5, 1.5])  # bin centres, not edges
    edges = np.linspace(-2.0, 2.0, 5)
    E = gaussian_channel(vals, edges, noise_std=0.3)
    assert E.shape == (4, 4)
    assert np.allclose(E.sum(axis=1), 1.0)
    assert E[0].argmax() == 0 and E[3].argmax() == 3, "likelihood should peak at the bin containing the true value"

    lat_edges, lon_edges = equal_area_sphere_cells(8, 16)
    n = n_sphere_cells(lon_edges)
    assert n > 0
    idx = sphere_cell_index(np.array([0.0, 0.5, -0.5]), np.array([0.0, 3.0, -3.0]), lat_edges, lon_edges)
    assert idx.shape == (3,) and np.all(idx < n)
    print("channels ok")


if __name__ == "__main__":
    _demo()
