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


def build_feature_map(hmm) -> BeliefFeatures:
    """Phi for one HMM: per-dimension marginals, then the metric expectations."""
    bin_counts = hmm.grid.bin_counts
    n_latent = hmm.n_latent
    per_dim = np.unravel_index(np.arange(n_latent), bin_counts)  # grid is C-order

    blocks: list[sp.spmatrix] = []
    names: list[str] = []
    groups: dict[str, slice] = {}
    col = 0

    ones = np.ones(n_latent)
    rows = np.arange(n_latent)
    for d, n_bins in enumerate(bin_counts):
        blocks.append(sp.csr_matrix((ones, (rows, per_dim[d])), shape=(n_latent, n_bins)))
        names += [f"marg{d}_{k}" for k in range(n_bins)]
        groups[f"marginal_{d}"] = slice(col, col + n_bins)
        col += n_bins

    d_metric = hmm.metric_table.shape[1]
    blocks.append(sp.csr_matrix(hmm.metric_table))
    names += [f"metric_{j}" for j in range(d_metric)]
    groups["metric"] = slice(col, col + d_metric)

    return BeliefFeatures(sp.hstack(blocks).tocsr(), tuple(names), groups)


def belief_features(
    hmm, tokens: np.ndarray, is_action: np.ndarray, fmap: BeliefFeatures, chunk_bytes: int = CHUNK_BYTES
) -> np.ndarray:
    """Exact filtered belief, projected onto `fmap`. Returns (n, L, n_features).

    Chunked over sequences so the full belief array is never resident.
    """
    n, seq_len = tokens.shape
    per_seq = seq_len * hmm.n_latent * 8
    chunk = max(1, int(chunk_bytes // max(per_seq, 1)))

    out = np.empty((n, seq_len, fmap.n_features), dtype=np.float32)
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        beliefs = hmm.forward(tokens[start:stop], is_action[start:stop])
        flat = beliefs.reshape(-1, hmm.n_latent)
        out[start:stop] = (flat @ fmap.matrix).reshape(stop - start, seq_len, -1).astype(np.float32)
        del beliefs, flat
    return out


def _demo() -> None:
    import scipy.sparse as sps

    from physics.grid import LatentGrid
    from physics.hmm import DiscreteHMM

    rng = np.random.default_rng(0)
    counts = (3, 4)
    n = 12
    grid = LatentGrid(bin_counts=counts, domain=((0.0, 1.0), (0.0, 1.0)), periodic=(False, False))
    hmm = DiscreteHMM(
        T=sps.csr_matrix(rng.dirichlet(np.ones(n), size=n)),
        P_actions=(sps.eye(n, format="csr"),),
        E=rng.dirichlet(np.ones(3), size=n),
        metric_table=rng.normal(size=(n, 2)),
        grid=grid,
        action_names=("noop",),
    )
    fmap = build_feature_map(hmm)
    assert fmap.n_features == 3 + 4 + 2

    b = rng.dirichlet(np.ones(n))
    feats = b @ fmap.matrix

    # Each marginal block must be a genuine marginal: normalised, and equal to
    # the belief summed over the other axis.
    grid_b = b.reshape(counts)
    assert np.allclose(feats[fmap.groups["marginal_0"]], grid_b.sum(axis=1))
    assert np.allclose(feats[fmap.groups["marginal_1"]], grid_b.sum(axis=0))
    assert np.allclose(feats[fmap.groups["metric"]], b @ hmm.metric_table)

    # Linearity in b is the property the whole probe argument rests on.
    b2 = rng.dirichlet(np.ones(n))
    assert np.allclose((0.3 * b + 0.7 * b2) @ fmap.matrix, 0.3 * (b @ fmap.matrix) + 0.7 * (b2 @ fmap.matrix))

    tokens = rng.integers(0, 3, size=(5, 6))
    got = belief_features(hmm, tokens, np.zeros_like(tokens, dtype=bool), fmap, chunk_bytes=1)
    assert got.shape == (5, 6, fmap.n_features)
    ref = hmm.forward(tokens, np.zeros_like(tokens, dtype=bool)).reshape(-1, n) @ fmap.matrix
    assert np.allclose(got.reshape(-1, fmap.n_features), ref, atol=1e-5), "chunking changed the answer"
    print("features ok")


if __name__ == "__main__":
    _demo()
