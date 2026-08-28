from __future__ import annotations

from dataclasses import dataclass
from math import prod

import numpy as np


@dataclass(frozen=True)
class LatentGrid:
    """A rectangular binning of the latent state space. Periodic dims wrap,
    non-periodic dims clip. Flat index is C-order over `bin_counts`.
    """

    bin_counts: tuple[int, ...]
    domain: tuple[tuple[float, float], ...]
    periodic: tuple[bool, ...]

    def __post_init__(self) -> None:
        if not (len(self.bin_counts) == len(self.domain) == len(self.periodic)):
            raise ValueError("bin_counts, domain, periodic must have equal length")

    @property
    def ndim(self) -> int:
        return len(self.bin_counts)

    @property
    def n_bins(self) -> int:
        return prod(self.bin_counts)

    def _edges(self, dim: int) -> np.ndarray:
        lo, hi = self.domain[dim]
        return np.linspace(lo, hi, self.bin_counts[dim] + 1)

    def _wrap_or_clip(self, z: np.ndarray) -> np.ndarray:
        z = np.array(z, dtype=np.float64, copy=True)
        for d, (lo, hi) in enumerate(self.domain):
            if self.periodic[d]:
                span = hi - lo
                z[..., d] = lo + np.mod(z[..., d] - lo, span)
            else:
                z[..., d] = np.clip(z[..., d], lo, hi - 1e-9)
        return z

    def per_dim_index(self, z: np.ndarray) -> np.ndarray:
        """(..., ndim) integer per-axis bin index."""
        z = self._wrap_or_clip(z)
        idx = np.empty(z.shape, dtype=np.int64)
        for d, (lo, hi) in enumerate(self.domain):
            width = (hi - lo) / self.bin_counts[d]
            idx[..., d] = np.clip(((z[..., d] - lo) / width).astype(np.int64), 0, self.bin_counts[d] - 1)
        return idx

    def index(self, z: np.ndarray) -> np.ndarray:
        """(...,) flat bin index."""
        per_dim = self.per_dim_index(z)
        flat = np.zeros(per_dim.shape[:-1], dtype=np.int64)
        for d in range(self.ndim):
            flat = flat * self.bin_counts[d] + per_dim[..., d]
        return flat

    def centers(self) -> np.ndarray:
        """(n_bins, ndim) bin centres, ordering matching `index`."""
        axes = [(self._edges(d)[:-1] + self._edges(d)[1:]) / 2 for d in range(self.ndim)]
        mesh = np.meshgrid(*axes, indexing="ij")
        return np.stack([m.ravel() for m in mesh], axis=-1)

    def sample_in_bin(self, rng: np.random.Generator, flat_idx: np.ndarray, n_per: int) -> np.ndarray:
        """(len(flat_idx), n_per, ndim) uniform samples inside each named bin."""
        flat_idx = np.asarray(flat_idx)
        per_dim = np.empty(flat_idx.shape + (self.ndim,), dtype=np.int64)
        rem = flat_idx.copy()
        for d in reversed(range(self.ndim)):
            per_dim[..., d] = rem % self.bin_counts[d]
            rem = rem // self.bin_counts[d]
        out = np.empty(flat_idx.shape + (n_per, self.ndim), dtype=np.float64)
        for d, (lo, hi) in enumerate(self.domain):
            width = (hi - lo) / self.bin_counts[d]
            lo_d = lo + per_dim[..., d] * width
            u = rng.random(flat_idx.shape + (n_per,))
            out[..., d] = lo_d[..., None] + u * width
        return out


def _demo() -> None:
    g = LatentGrid(bin_counts=(4, 3), domain=((-np.pi, np.pi), (-1.0, 1.0)), periodic=(True, False))
    assert g.n_bins == 12
    c = g.centers()
    assert c.shape == (12, 2)
    idx = g.index(c)
    assert np.array_equal(idx, np.arange(12)), "centre of bin i must map back to i"
    rng = np.random.default_rng(0)
    s = g.sample_in_bin(rng, np.array([0, 5, 11]), 50)
    assert s.shape == (3, 50, 2)
    back = g.index(s)
    assert np.all(back[0] == 0) and np.all(back[1] == 5) and np.all(back[2] == 11)
    wrapped = g.index(np.array([np.pi + 0.01, 0.0]))
    assert wrapped == g.index(np.array([-np.pi + 0.01, 0.0])), "periodic dim must wrap"
    print("grid ok")


if __name__ == "__main__":
    _demo()
