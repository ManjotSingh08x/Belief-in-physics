"""Ulam-method transfer-operator estimate of the K-step transition kernel,
plus the (near-)deterministic action remaps.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .dynamics import PhysicalSystem
from .grid import LatentGrid


def build_transition_matrix(
    system: PhysicalSystem,
    grid: LatentGrid,
    K: int,
    dt: float,
    samples_per_bin: int,
    rng: np.random.Generator,
    prune_below: float = 1e-4,
    chunk_bins: int = 2048,
) -> sp.csr_matrix:
    n = grid.n_bins
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    vals: list[np.ndarray] = []
    all_idx = np.arange(n)
    for start in range(0, n, chunk_bins):
        chunk = all_idx[start : start + chunk_bins]
        pts = grid.sample_in_bin(rng, chunk, samples_per_bin)  # (c, s, d)
        c, s, d = pts.shape
        flat = pts.reshape(c * s, d)
        dest = system.flow(flat, dt, K)
        dest_idx = grid.index(dest).reshape(c, s)
        for i, bin_i in enumerate(chunk):
            counts = np.bincount(dest_idx[i], minlength=n).astype(np.float64)
            counts /= counts.sum()
            keep = np.flatnonzero(counts >= prune_below)
            if keep.size == 0:
                keep = np.array([counts.argmax()])
            w = counts[keep]
            w /= w.sum()
            rows.append(np.full(keep.size, bin_i))
            cols.append(keep)
            vals.append(w)
    T = sp.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n, n),
    )
    return T


def build_action_map(system: PhysicalSystem, grid: LatentGrid, action: int) -> sp.csr_matrix:
    """Deterministic remap: push each bin centre through apply_action, record destination bin."""
    n = grid.n_bins
    centers = grid.centers()
    dest = system.apply_action(centers, action)
    dest_idx = grid.index(dest)
    data = np.ones(n)
    rows = np.arange(n)
    return sp.csr_matrix((data, (rows, dest_idx)), shape=(n, n))


def _demo() -> None:
    from .systems.pendulum import Pendulum

    system = Pendulum(gamma=0.0)
    grid = LatentGrid(bin_counts=(24, 24), domain=system.domain, periodic=system.dims_periodic)
    rng = np.random.default_rng(0)
    T = build_transition_matrix(system, grid, K=5, dt=0.02, samples_per_bin=64, rng=rng)
    assert T.shape == (grid.n_bins, grid.n_bins)
    row_sums = np.asarray(T.sum(axis=1)).ravel()
    assert np.allclose(row_sums, 1.0, atol=1e-6), "every row of T must sum to 1"

    P0 = build_action_map(system, grid, action=0)
    assert np.allclose(np.asarray(P0.sum(axis=1)).ravel(), 1.0)
    print("builder ok")


if __name__ == "__main__":
    _demo()
