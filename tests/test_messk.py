"""The Mess-K process: the belief must be exact, and the controls must bite."""

from __future__ import annotations

import numpy as np

from physics.messk import MessKProcess, MessPendulum, simplex_embedding, token_window_features
from physics.messk_configs import make_process


def _brute_force_belief(chain: MessKProcess, letters: np.ndarray) -> np.ndarray:
    """P(mood_{t+1} | letters) by enumerating every mood path. Exponential, so tiny only."""
    k, m = chain.n_states, letters.shape[0]
    T, E = chain.T, chain.E
    # paths over moods 0..m: m of them emit a letter, the last is the one the
    # predictive belief is about.
    paths = np.array(np.meshgrid(*[np.arange(k)] * (m + 1), indexing="ij")).reshape(m + 1, -1).T
    out = np.zeros(k)
    for path in paths:
        p = 1.0 / k
        for t in range(m):
            p *= E[path[t], letters[t]] * T[path[t], path[t + 1]]
        out[path[-1]] += p
    return out / out.sum()


def test_forward_algorithm_matches_brute_force():
    chain = MessKProcess(n_states=3, alpha=0.7, stay=0.7)
    letters = np.array([[0, 2, 1, 1, 0]])
    got = chain.beliefs(letters)[0, -1]
    want = _brute_force_belief(chain, letters[0])
    assert np.allclose(got, want, atol=1e-9), f"{got} != {want}"


def test_mess3_reproduces_their_parameterisation():
    chain = MessKProcess(n_states=3, alpha=0.7, stay=0.7)
    assert abs(chain.x - 0.15) < 1e-12
    assert np.allclose(np.diag(chain.T), 0.7) and np.allclose(np.diag(chain.E), 0.7)


def test_beliefs_are_distributions_and_beat_chance():
    chain = MessKProcess(n_states=4)
    rng = np.random.default_rng(0)
    states, letters = chain.sample(rng, 128, 40)
    b = chain.beliefs(letters)
    assert np.allclose(b.sum(-1), 1.0)
    assert (b.argmax(-1) == states[:, 1:]).mean() > 1 / 4


def test_belief_forgets_its_prior():
    """The property that makes the myopic control necessary rather than optional."""
    for k in (3, 4):
        n = MessKProcess(n_states=k).memory_length()
        assert 2 <= n <= 30, f"K={k} memory {n} letters is implausible"


def test_belief_does_not_depend_on_the_physics():
    """Their belief is computed before the pendulum, so changing it must not move."""
    rng_a, rng_b = np.random.default_rng(3), np.random.default_rng(3)
    a = make_process("mess4").sample_batch(rng_a, 16)
    b = make_process("mess4", gamma=2.0, delta_v=1.7).sample_batch(rng_b, 16)
    assert np.allclose(a["beliefs"], b["beliefs"])
    assert not np.allclose(a["tokens"], b["tokens"]), "the physics must have changed"


def test_tokens_stay_in_vocabulary():
    proc = make_process("mess4")
    batch = proc.sample_batch(np.random.default_rng(1), 32)
    assert batch["tokens"].shape == (32, proc.seq_len)
    assert batch["tokens"].min() >= 0 and batch["tokens"].max() < proc.n_obs


def test_token_window_is_causal():
    """A window feature must never see a token from the future."""
    tokens = np.arange(12).reshape(1, 12) % 5
    tw = token_window_features(tokens, n_obs=5, window=3)
    other = tokens.copy()
    other[0, 7:] = (other[0, 7:] + 1) % 5
    assert np.allclose(tw[:, :7], token_window_features(other, 5, 3)[:, :7])


def test_simplex_embedding_shapes():
    assert simplex_embedding(3).shape == (3, 2)
    assert simplex_embedding(4).shape == (4, 3)
    v = simplex_embedding(4)
    d = [np.linalg.norm(v[i] - v[j]) for i in range(4) for j in range(i + 1, 4)]
    assert np.allclose(d, d[0]), "a regular simplex has equal edges"


def test_features_line_up_with_their_groups():
    proc = MessPendulum(chain=MessKProcess(n_states=4))
    batch = proc.sample_batch(np.random.default_rng(0), 8)
    feats, groups = proc.features_and_groups(batch)
    assert set(groups) == {"belief", "mood", "velocity"}
    assert np.allclose(feats[:, :, groups["belief"]].sum(-1), 1.0)
    assert np.allclose(feats[:, :, groups["mood"]].sum(-1), 1.0)
