"""The discrete HMM is the generator, not an approximation of it: tokens are
sampled directly from T / P_actions / E, so the exact forward-algorithm
belief is the true posterior of the process that produced them.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from .grid import LatentGrid


@dataclass(frozen=True)
class Episodes:
    tokens: np.ndarray          # (n, L) int, vocab = [0, n_obs) obs bins, [n_obs, n_obs+n_actions) actions
    latent: np.ndarray          # (n, L) int, true latent bin index
    is_action: np.ndarray       # (n, L) bool
    is_prediction: np.ndarray   # (n, L) bool, true on pure-pushforward (observation) steps
    metrics: np.ndarray         # (n, L, d_metric) float, ground truth at the true latent bin
    beliefs: np.ndarray | None = None  # (n, L, n_latent) float32, exact posterior; None unless requested


def _sample_sparse_categorical(mat: sp.csr_matrix, states: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = np.empty_like(states)
    uniq, inv = np.unique(states, return_inverse=True)
    for u_i, s in enumerate(uniq):
        cols = mat.indices[mat.indptr[s] : mat.indptr[s + 1]]
        probs = mat.data[mat.indptr[s] : mat.indptr[s + 1]]
        probs = probs / probs.sum()
        group = np.flatnonzero(inv == u_i)
        out[group] = rng.choice(cols, size=group.size, p=probs)
    return out


def _sample_dense_categorical(mat: np.ndarray, states: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = np.empty_like(states)
    uniq, inv = np.unique(states, return_inverse=True)
    for u_i, s in enumerate(uniq):
        probs = mat[s]
        group = np.flatnonzero(inv == u_i)
        out[group] = rng.choice(mat.shape[1], size=group.size, p=probs)
    return out


@dataclass(frozen=True)
class DiscreteHMM:
    T: sp.csr_matrix
    P_actions: tuple[sp.csr_matrix, ...]
    E: np.ndarray
    metric_table: np.ndarray
    grid: LatentGrid
    action_names: tuple[str, ...]

    @property
    def n_latent(self) -> int:
        return self.T.shape[0]

    @property
    def n_obs(self) -> int:
        return self.E.shape[1]

    @property
    def n_actions(self) -> int:
        return len(self.P_actions)

    def bayes_update(self, belief: np.ndarray, obs: np.ndarray) -> np.ndarray:
        lik = self.E[:, obs].T  # (n, n_latent)
        post = belief * lik
        post_sum = post.sum(axis=-1, keepdims=True)
        return post / np.clip(post_sum, 1e-300, None)

    def sample_batch(
        self,
        rng: np.random.Generator,
        n: int,
        N: int,
        M: int,
        K: int,
        prior: np.ndarray | None = None,
    ) -> Episodes:
        if N % K != 0:
            raise ValueError(f"N={N} must be a multiple of K={K}")
        steps_per_segment = N // K
        seq_len = M * (1 + steps_per_segment)

        latent = np.empty((n, seq_len), dtype=np.int64)
        tokens = np.empty((n, seq_len), dtype=np.int64)
        is_action = np.zeros((n, seq_len), dtype=bool)
        is_prediction = np.zeros((n, seq_len), dtype=bool)

        if prior is None:
            prior = np.full(self.n_latent, 1.0 / self.n_latent)
        state = rng.choice(self.n_latent, size=n, p=prior)
        actions = rng.integers(0, self.n_actions, size=(n, M))

        pos = 0
        for m in range(M):
            a = actions[:, m]
            new_state = np.empty_like(state)
            for act in range(self.n_actions):
                mask = a == act
                if np.any(mask):
                    new_state[mask] = _sample_sparse_categorical(self.P_actions[act], state[mask], rng)
            state = new_state
            latent[:, pos] = state
            tokens[:, pos] = self.n_obs + a
            is_action[:, pos] = True
            pos += 1
            for _ in range(steps_per_segment):
                state = _sample_sparse_categorical(self.T, state, rng)
                obs = _sample_dense_categorical(self.E, state, rng)
                latent[:, pos] = state
                tokens[:, pos] = obs
                is_prediction[:, pos] = True
                pos += 1

        return Episodes(
            tokens=tokens,
            latent=latent,
            is_action=is_action,
            is_prediction=is_prediction,
            metrics=self.metric_table[latent],
            beliefs=None,
        )

    def forward(self, tokens: np.ndarray, is_action: np.ndarray, prior: np.ndarray | None = None) -> np.ndarray:
        """Exact posterior belief at every position, given realised tokens.
        Assumes the action/observation pattern is identical across the batch
        at each position (true for any Episodes this module produces).
        """
        n, seq_len = tokens.shape
        if prior is None:
            prior = np.full(self.n_latent, 1.0 / self.n_latent)
        belief = np.tile(prior, (n, 1))
        out = np.empty((n, seq_len, self.n_latent), dtype=np.float64)
        for t in range(seq_len):
            if is_action[0, t]:
                a_ids = tokens[:, t] - self.n_obs
                new_belief = np.empty_like(belief)
                for a in range(self.n_actions):
                    mask = a_ids == a
                    if np.any(mask):
                        new_belief[mask] = belief[mask] @ self.P_actions[a]
                belief = new_belief
            else:
                belief = belief @ self.T
                belief = self.bayes_update(belief, tokens[:, t])
            out[:, t, :] = belief
        return out


def _brute_force_forward(T: np.ndarray, E: np.ndarray, tokens: np.ndarray, prior: np.ndarray) -> np.ndarray:
    """Reference forward algorithm over dense matrices, for a toy exactness check."""
    n_states = T.shape[0]
    belief = prior.copy()
    out = np.empty((len(tokens), n_states))
    for t, o in enumerate(tokens):
        belief = belief @ T
        belief = belief * E[:, o]
        belief = belief / belief.sum()
        out[t] = belief
    return out


def _demo() -> None:
    # Toy 5-state / 3-observation HMM: verify sparse forward matches dense brute force.
    rng = np.random.default_rng(0)
    n_states, n_obs = 5, 3
    T_dense = rng.dirichlet(np.ones(n_states), size=n_states)
    E_dense = rng.dirichlet(np.ones(n_obs), size=n_states)
    T = sp.csr_matrix(T_dense)
    noop = sp.eye(n_states, format="csr")
    grid = LatentGrid(bin_counts=(n_states,), domain=((0.0, 1.0),), periodic=(False,))
    hmm = DiscreteHMM(
        T=T, P_actions=(noop,), E=E_dense, metric_table=np.zeros((n_states, 1)), grid=grid, action_names=("noop",)
    )
    prior = np.full(n_states, 1.0 / n_states)
    tokens_1d = rng.integers(0, n_obs, size=6)
    ref = _brute_force_forward(T_dense, E_dense, tokens_1d, prior)

    tokens = tokens_1d[None, :]
    is_action = np.zeros_like(tokens, dtype=bool)
    got = hmm.forward(tokens, is_action, prior=prior)[0]
    assert np.allclose(got, ref, atol=1e-10), "sparse forward must match brute-force dense forward exactly"

    assert np.allclose(hmm.E.sum(axis=1), 1.0)
    assert np.allclose(np.asarray(hmm.T.sum(axis=1)).ravel(), 1.0)

    ep = hmm.sample_batch(rng, n=16, N=6, M=3, K=2)
    assert ep.tokens.shape == (16, 3 * (1 + 3))
    assert ep.is_action.sum(axis=1).min() == 3 and ep.is_action.sum(axis=1).max() == 3
    assert np.allclose(ep.metrics, hmm.metric_table[ep.latent])

    try:
        hmm.sample_batch(rng, n=4, N=5, M=2, K=2)
        raise AssertionError("N % K != 0 must raise")
    except ValueError:
        pass

    print("hmm ok")


if __name__ == "__main__":
    _demo()
