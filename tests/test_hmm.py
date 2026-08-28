import numpy as np
import pytest
import scipy.sparse as sp

from physics.grid import LatentGrid
from physics.hmm import DiscreteHMM, _sample_dense_categorical, _sample_sparse_categorical


def _brute_force_forward(T: np.ndarray, E: np.ndarray, tokens: np.ndarray, prior: np.ndarray) -> np.ndarray:
    belief = prior.copy()
    out = np.empty((len(tokens), len(prior)))
    for t, o in enumerate(tokens):
        belief = belief @ T
        belief = belief * E[:, o]
        belief = belief / belief.sum()
        out[t] = belief
    return out


def _toy_hmm(seed=0):
    rng = np.random.default_rng(seed)
    n_states, n_obs = 5, 3
    T_dense = rng.dirichlet(np.ones(n_states), size=n_states)
    E_dense = rng.dirichlet(np.ones(n_obs), size=n_states)
    T = sp.csr_matrix(T_dense)
    noop = sp.eye(n_states, format="csr")
    grid = LatentGrid(bin_counts=(n_states,), domain=((0.0, 1.0),), periodic=(False,))
    hmm = DiscreteHMM(
        T=T, P_actions=(noop,), E=E_dense, metric_table=np.zeros((n_states, 1)), grid=grid, action_names=("noop",)
    )
    return hmm, T_dense, E_dense


def test_forward_matches_brute_force_enumeration():
    hmm, T_dense, E_dense = _toy_hmm()
    n_states = T_dense.shape[0]
    prior = np.full(n_states, 1.0 / n_states)
    rng = np.random.default_rng(1)
    tokens_1d = rng.integers(0, E_dense.shape[1], size=6)
    ref = _brute_force_forward(T_dense, E_dense, tokens_1d, prior)

    tokens = tokens_1d[None, :]
    is_action = np.zeros_like(tokens, dtype=bool)
    got = hmm.forward(tokens, is_action, prior=prior)[0]
    assert np.allclose(got, ref, atol=1e-10)


def test_belief_normalises_to_one_at_every_step():
    hmm, _, _ = _toy_hmm()
    rng = np.random.default_rng(2)
    tokens = rng.integers(0, hmm.n_obs, size=(3, 10))
    is_action = np.zeros_like(tokens, dtype=bool)
    beliefs = hmm.forward(tokens, is_action)
    assert np.allclose(beliefs.sum(axis=-1), 1.0, atol=1e-8)


def test_sample_batch_shape_and_action_count():
    hmm, _, _ = _toy_hmm()
    rng = np.random.default_rng(3)
    ep = hmm.sample_batch(rng, n=16, N=6, M=3, K=2)
    assert ep.tokens.shape == (16, 3 * (1 + 3))
    assert (ep.is_action.sum(axis=1) == 3).all()
    assert np.array_equal(ep.metrics, hmm.metric_table[ep.latent])


def test_sample_batch_rejects_n_not_a_multiple_of_k():
    hmm, _, _ = _toy_hmm()
    rng = np.random.default_rng(4)
    with pytest.raises(ValueError):
        hmm.sample_batch(rng, n=4, N=5, M=2, K=2)


def test_belief_mean_tracks_truth_better_than_the_stationary_prior():
    """The filter must actually be doing work: its posterior mean should be
    closer to the true latent value, on average, than just guessing the prior.
    """
    hmm, T_dense, E_dense = _toy_hmm(seed=7)
    rng = np.random.default_rng(5)
    ep = hmm.sample_batch(rng, n=200, N=8, M=1, K=2)
    beliefs = hmm.forward(ep.tokens, ep.is_action)
    values = np.arange(hmm.n_latent, dtype=np.float64)
    posterior_mean = beliefs[:, -1, :] @ values
    prior = np.full(hmm.n_latent, 1.0 / hmm.n_latent)
    prior_guess = prior @ values

    truth = ep.latent[:, -1].astype(np.float64)
    filter_err = np.mean((posterior_mean - truth) ** 2)
    prior_err = np.mean((prior_guess - truth) ** 2)
    assert filter_err < prior_err


def test_sparse_sampler_matches_the_row_distribution():
    """The vectorised inverse-CDF sampler must reproduce each CSR row exactly;
    a bug here is silent, since wrong-but-plausible tokens still train a model.
    """
    rng = np.random.default_rng(0)
    dense = rng.dirichlet(np.ones(8), size=6)
    mat = sp.csr_matrix(dense)
    for state in range(6):
        drawn = _sample_sparse_categorical(mat, np.full(200_000, state), rng)
        empirical = np.bincount(drawn, minlength=8) / 200_000
        assert np.abs(empirical - dense[state]).max() < 0.01


def test_dense_sampler_matches_the_row_distribution():
    rng = np.random.default_rng(1)
    dense = rng.dirichlet(np.ones(8), size=6)
    for state in range(6):
        drawn = _sample_dense_categorical(dense, np.full(200_000, state), rng)
        empirical = np.bincount(drawn, minlength=8) / 200_000
        assert np.abs(empirical - dense[state]).max() < 0.01


def test_samplers_handle_a_mixed_state_batch_and_an_empty_one():
    rng = np.random.default_rng(2)
    dense = rng.dirichlet(np.ones(5), size=4)
    mat = sp.csr_matrix(dense)
    states = rng.integers(0, 4, size=60_000)
    drawn = _sample_sparse_categorical(mat, states, rng)
    for state in range(4):
        selected = drawn[states == state]
        empirical = np.bincount(selected, minlength=5) / selected.size
        assert np.abs(empirical - dense[state]).max() < 0.02
    assert _sample_sparse_categorical(mat, np.empty(0, dtype=np.int64), rng).size == 0
    assert _sample_dense_categorical(dense, np.empty(0, dtype=np.int64), rng).size == 0
