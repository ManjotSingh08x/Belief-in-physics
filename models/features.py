"""Probe targets: low-dimensional features that are *exactly linear* in the belief.

The whole reason a linear probe is the right instrument is that expectation
linearises. A quantity that is nonlinear in the world state is a linear
functional of the belief over the world state -- this is the same argument
`belief-geometry/dictionary.py` makes for the Kepler invariants, and in a
discrete HMM it holds exactly rather than approximately.

The full belief lives on the simplex over `n_latent` bins, which is up to
200704 for the double pendulum: a (n, L, n_latent) float64 array for a few
hundred eval sequences is tens of gigabytes. So we never materialise it. We
project it, in chunks, onto a feature map Phi that keeps only what the analysis
needs:

    features = b @ Phi,      Phi = [ M_0 | M_1 | ... | metric_table ]

`M_d` is the 0/1 indicator that marginalises the belief onto latent dimension
d, so `b @ M_d` is the exact marginal posterior over that dimension. Both
blocks are linear in b by construction, so "the residual stream linearly
encodes these features" is a faithful weakening of "it linearly encodes the
belief" -- not a different claim dressed up as one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

# Cap on the transient (chunk, L, n_latent) float64 belief block. 1 GiB keeps
# the double pendulum (n_latent=200704) at a chunk of ~10 sequences.
CHUNK_BYTES = 1 << 30


@dataclass(frozen=True)
class BeliefFeatures:
    matrix: sp.csr_matrix  # (n_latent, n_features)
    names: tuple[str, ...]
    groups: dict[str, slice]  # "marginal_0" ... "metric" -> column slice

    @property
    def n_features(self) -> int:
        return self.matrix.shape[1]


def _demo() -> None:
    rng = np.random.default_rng(0)
    fmap = BeliefFeatures(
        sp.csr_matrix(rng.random((6, 3))), ("a", "b", "c"), {"g": slice(0, 3)}
    )
    assert fmap.n_features == 3
    print("features ok")
