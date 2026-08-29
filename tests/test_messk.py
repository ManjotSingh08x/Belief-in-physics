"""The Mess-K process: the belief must be exact, and the controls must bite."""

from __future__ import annotations

from itertools import combinations

import numpy as np

from physics.messk import (
    MessDriven,
    MessKProcess,
    simplex_embedding,
    square_projection_vertices,
    token_window_features,
)
from physics.messk_configs import MESSK_CONFIGS, make_process


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


def test_three_state_chain_derives_per_neighbour_transition_probability():
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
    """The belief is computed before the physics, so changing it must not move."""
    rng_a, rng_b = np.random.default_rng(3), np.random.default_rng(3)
    a = make_process("pendulum_mess4").sample_batch(rng_a, 16)
    b = make_process(
        "pendulum_mess4", system={"gamma": 2.0}, delta_v=1.7
    ).sample_batch(rng_b, 16)
    assert np.allclose(a["beliefs"], b["beliefs"])
    assert not np.allclose(a["tokens"], b["tokens"]), "the physics must have changed"


def test_tokens_stay_in_vocabulary():
    for name in MESSK_CONFIGS:
        proc = make_process(name)
        batch = proc.sample_batch(np.random.default_rng(1), 32)
        assert batch["tokens"].shape == (32, proc.seq_len), name
        assert batch["tokens"].min() >= 0 and batch["tokens"].max() < proc.n_obs, name


def test_all_four_systems_use_mess4_without_noop_or_repeated_actions():
    assert set(MESSK_CONFIGS) == {
        "pendulum_mess4",
        "predator_prey_mess4",
        "sphere_mess4",
        "double_pendulum_mess4",
    }
    for name in MESSK_CONFIGS:
        proc = make_process(name)
        actions = proc.actions
        assert proc.chain.n_states == 4, name
        assert actions.shape[0] == 4, name
        assert np.all(np.linalg.norm(actions, axis=1) > 0), f"{name} contains a no-op"
        assert len(np.unique(actions, axis=0)) == 4, f"{name} repeats an action"
        assert np.allclose(actions.sum(axis=0), 0.0), f"{name} actions are directionally biased"


def test_system_action_semantics():
    pendulum = make_process("pendulum_mess4").actions[:, 0]
    assert np.allclose(pendulum, [-np.sqrt(2.7), -np.sqrt(0.3), np.sqrt(0.3), np.sqrt(2.7)])

    cardinal = np.array([[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]])
    for name in ("predator_prey_mess4", "sphere_mess4", "double_pendulum_mess4"):
        actions = make_process(name).actions
        unit = actions / np.linalg.norm(actions, axis=1, keepdims=True)
        assert np.allclose(unit, cardinal), name


def test_every_action_produces_a_distinct_observation_sequence():
    for name in MESSK_CONFIGS:
        proc = make_process(name)
        system = proc.system
        state = np.repeat(system.initial_state(1), 4, axis=0)
        state = system.kick(state, proc.actions)
        baseline = system.initial_state(1)
        action_tokens, baseline_tokens = [], []
        for _ in range(proc.n_steps):
            state = system.flow(state, proc.dt)
            baseline = system.flow(baseline, proc.dt)
            action_tokens.append(proc.discretise(system.observable(state)))
            baseline_tokens.append(proc.discretise(system.observable(baseline))[0])

        trajectories = np.stack(action_tokens, axis=1)
        baseline_trajectory = np.asarray(baseline_tokens)
        assert len(np.unique(trajectories, axis=0)) == 4, name
        assert np.all(np.any(trajectories != baseline_trajectory, axis=1)), name


def test_actions_remain_observably_distinct_on_typical_states():
    rng = np.random.default_rng(7)
    n = 64
    for name in MESSK_CONFIGS:
        proc = make_process(name)
        system = proc.system
        state = system.initial_state(n)
        for _ in range(8):
            state = system.kick(state, proc.actions[rng.integers(0, 4, size=n)])
            for _ in range(proc.n_steps):
                state = system.flow(state, proc.dt)

        forked = np.repeat(state, 4, axis=0)
        forked = system.kick(forked, np.tile(proc.actions, (n, 1)))
        tokens = []
        for _ in range(proc.n_steps):
            forked = system.flow(forked, proc.dt)
            tokens.append(proc.discretise(system.observable(forked)).reshape(n, 4))
        trajectories = np.stack(tokens, axis=-1)

        for left, right in combinations(range(4), 2):
            identical = np.all(trajectories[:, left] == trajectories[:, right], axis=1)
            hamming = np.mean(trajectories[:, left] != trajectories[:, right])
            assert identical.mean() < 0.25, (name, left, right, identical.mean())
            assert hamming > 0.20, (name, left, right, hamming)


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


def test_square_projection_places_four_states_at_distinct_corners():
    vertices = square_projection_vertices()
    assert vertices.shape == (4, 2)
    assert len(np.unique(vertices, axis=0)) == 4
    assert np.allclose(vertices.mean(axis=0), 0.0)
    assert np.allclose(np.eye(4) @ vertices, vertices)


def test_features_line_up_with_their_groups():
    proc = make_process("pendulum_mess4")
    batch = proc.sample_batch(np.random.default_rng(0), 8)
    feats, groups = proc.features_and_groups(batch)
    assert set(groups) == {"belief", "mood", "metric"}
    want_belief = batch["beliefs"] @ simplex_embedding(4)
    assert np.allclose(feats[:, :, groups["belief"]], want_belief)
    assert groups["belief"].stop - groups["belief"].start == 3
    assert np.allclose(feats[:, :, groups["mood"]].sum(-1), 1.0)
