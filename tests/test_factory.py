import shutil

import numpy as np
import pytest

from physics import cache as cache_module
from physics.factory import make_simulator


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(cache_module, "CACHE_DIR", tmp_path)
    yield
    shutil.rmtree(tmp_path, ignore_errors=True)


def _small_pendulum_config():
    return {
        "system": {"name": "pendulum", "g": 9.81, "length": 1.0, "gamma": 0.15, "dt": 0.02},
        "hmm": {"latent_bins": (16, 12), "emission_bins": 8, "observable_periodic": True, "samples_per_bin": 32},
        "perturbation": {"action_names": ["-dv", "noop", "+dv"]},
        "K": 3,
        "seed": 0,
    }


def test_simulate_batch_shape_and_vocab_range():
    sim = make_simulator(_small_pendulum_config())
    ep = sim.simulate_batch(num_simulations=8, N=6, M=4)
    assert ep.tokens.shape == (8, 4 * (1 + 2))
    assert ep.tokens.min() >= 0
    assert ep.tokens.max() < sim.hmm.n_obs + sim.hmm.n_actions


def test_simulate_batch_with_beliefs_is_a_normalised_distribution():
    sim = make_simulator(_small_pendulum_config())
    ep = sim.simulate_batch(num_simulations=4, N=6, M=2, with_beliefs=True)
    assert ep.beliefs is not None
    assert ep.beliefs.shape == (4, 2 * 3, sim.hmm.n_latent)
    assert np.allclose(ep.beliefs.sum(axis=-1), 1.0, atol=1e-8)


def test_unknown_system_name_raises():
    config = _small_pendulum_config()
    config["system"]["name"] = "not_a_system"
    with pytest.raises(ValueError):
        make_simulator(config)


def test_make_simulator_is_cached_across_calls():
    config = _small_pendulum_config()
    sim1 = make_simulator(config)
    sim2 = make_simulator(config)
    assert (sim1.hmm.T != sim2.hmm.T).nnz == 0


def test_streaming_rng_yields_distinct_batches_including_the_perturbations():
    """Without an explicit rng, simulate_batch reseeds from self.seed and returns
    the *same* batch every call -- which would have trained every step on one
    repeated batch while the loss curve still looked like it was converging.
    """
    sim = make_simulator(_small_pendulum_config())
    repeated = [sim.simulate_batch(8, 6, 4) for _ in range(2)]
    assert np.array_equal(repeated[0].tokens, repeated[1].tokens), "documents the default"

    stream = np.random.default_rng(1)
    streamed = [sim.simulate_batch(8, 6, 4, rng=stream) for _ in range(3)]
    for earlier, later in zip(streamed, streamed[1:]):
        assert not np.array_equal(earlier.tokens, later.tokens)
        assert not np.array_equal(
            earlier.tokens[earlier.is_action], later.tokens[later.is_action]
        ), "perturbations must be redrawn too, not just the observations"


def test_perturbations_are_drawn_uniformly():
    sim = make_simulator(_small_pendulum_config())
    episodes = sim.simulate_batch(4000, 6, 4, rng=np.random.default_rng(2))
    actions = episodes.tokens[episodes.is_action] - sim.hmm.n_obs
    fractions = np.bincount(actions, minlength=sim.hmm.n_actions) / actions.size
    assert np.abs(fractions - 1.0 / sim.hmm.n_actions).max() < 0.02
