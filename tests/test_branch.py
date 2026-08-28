import numpy as np
import pytest

from physics import branch as branch_module
from physics.branch import (
    branch_features,
    feature_groups,
    forward_features,
    simplex_to_2d,
)
from physics.branch_configs import BRANCH_CONFIGS, make_branch_process


def test_module_self_check_includes_the_brute_force_exactness_proof():
    branch_module._demo()


@pytest.fixture(scope="module")
def pendulum():
    return make_branch_process("pendulum")


def test_vocabulary_contains_no_action_tokens(pendulum):
    """The perturbations must be hidden -- if they were emitted the belief over
    them would collapse to a delta and there would be no simplex to probe.
    """
    episodes = pendulum.sample_batch(np.random.default_rng(0), 32)
    assert episodes.tokens.max() < pendulum.n_obs
    assert episodes.tokens.min() >= 0


def test_streamed_features_match_the_materialised_ones(pendulum):
    episodes = pendulum.sample_batch(np.random.default_rng(1), 24)
    materialised = branch_features(pendulum, pendulum.forward(episodes.tokens))
    streamed = forward_features(pendulum, episodes.tokens, chunk=5)
    assert np.allclose(materialised, streamed, atol=1e-5)


def test_every_feature_block_is_a_simplex_point(pendulum):
    episodes = pendulum.sample_batch(np.random.default_rng(2), 16)
    features = forward_features(pendulum, episodes.tokens)
    groups = feature_groups(pendulum)
    for name in ("action_lag0", "action_lag1", "action_lag2", "z0"):
        block = features[:, -1, groups[name]]
        assert np.allclose(block.sum(axis=1), 1.0, atol=1e-5), name
        assert (block >= -1e-6).all(), name


def test_belief_identifies_the_hidden_perturbation_better_than_chance(pendulum):
    episodes = pendulum.sample_batch(np.random.default_rng(3), 200)
    features = forward_features(pendulum, episodes.tokens)
    guessed = features[:, -1, feature_groups(pendulum)["action_lag0"]].argmax(axis=1)
    assert (guessed == episodes.actions[:, -1]).mean() > 0.5  # chance is 1/3


def test_belief_stays_genuinely_uncertain(pendulum):
    """A collapsed belief turns 'encodes the belief' into 'encodes a label', so
    the configs are tuned to keep entropy well away from both 0 and 1.
    """
    episodes = pendulum.sample_batch(np.random.default_rng(4), 128)
    features = forward_features(pendulum, episodes.tokens)
    p = features[:, :, feature_groups(pendulum)["action_lag0"]].reshape(-1, pendulum.n_actions)
    entropy = -(p * np.log(np.clip(p, 1e-12, None))).sum(1).mean() / np.log(pendulum.n_actions)
    assert 0.3 < entropy < 0.9, entropy


def test_simplex_projection_centres_the_uniform_belief():
    for n in (3, 4, 5):
        assert np.allclose(simplex_to_2d(np.full((1, n), 1.0 / n)), 0.0, atol=1e-12)


@pytest.mark.parametrize("name", sorted(BRANCH_CONFIGS))
def test_each_system_builds_and_stays_within_its_branch_budget(name):
    process = make_branch_process(name)
    assert process.n_branches(process.M) <= 300_000, name
    episodes = process.sample_batch(np.random.default_rng(0), 8)
    assert episodes.tokens.shape == (8, process.seq_len)
    assert episodes.tokens.max() < process.n_obs
