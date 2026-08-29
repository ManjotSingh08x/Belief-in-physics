"""Checks for the phase-5 machinery.

Each module carries its own `_demo` self-check; those run here as well, so a
regression shows up under pytest rather than only when the file is executed
directly. The tests beyond the self-checks are the properties the phase-5 claims
actually lean on.
"""

import numpy as np
import torch

from models import bootstrap as bootstrap_module
from models import probe_extra as probe_extra_module
from models.ablation import (
    _demo_interventions,
    complement_basis,
    intervened_loss,
    mean_ablate,
    pca_basis,
    principal_angles,
    random_basis,
    restrict_to_positions,
    variance_fraction,
)
from models.probe_extra import simplex_coords, token_window_features
from models.transformer import ModelConfig, TinyTransformer
from physics import myopic as myopic_module
from physics.branch_configs import BRANCH_CONFIGS, make_branch_process, scaled_kick
from physics.myopic import bayes_floor, predictive_stack


def test_module_self_checks():
    myopic_module._demo()
    bootstrap_module._demo()
    probe_extra_module._demo()
    _demo_interventions()


def _tiny_process():
    """A four-step pendulum: small enough to enumerate, real enough to be the
    same code path the experiments use."""
    return make_branch_process("pendulum", M=2, steps_per_segment=3)


def test_bayes_floor_is_below_uniform_and_above_zero():
    """The floor has to be a real bound: better than guessing uniformly, and
    positive, or every excess-loss number computed against it is meaningless."""
    process = _tiny_process()
    tokens = process.sample_batch(np.random.default_rng(0), 32).tokens
    floor = bayes_floor(process, tokens)
    assert 0.0 < floor["plugin"] < floor["uniform"]
    # Plug-in and realised estimate the same quantity; a large gap means the
    # predictive distributions are not the ones that generated the tokens.
    assert abs(floor["plugin"] - floor["realised"]) < 0.2, floor


def test_predictive_stack_matches_a_direct_pushforward():
    """`p^(1)` must equal the belief pushed one step and hit with the emission.

    Checked against the filter itself rather than against a second copy of the
    same code: `iter_beliefs` is the object the whole project trusts.
    """
    process = _tiny_process()
    tokens = process.sample_batch(np.random.default_rng(1), 8).tokens
    stack = predictive_stack(process, tokens, horizon=1)

    beliefs = list(process.iter_beliefs(tokens))
    for pos in range(process.seq_len - 1):
        m, s = divmod(pos, process.steps_per_segment)
        b = beliefs[pos]
        if s + 1 < process.steps_per_segment:
            expect = b @ process.emissions[m][:, s + 1]
        else:
            spread = np.repeat(b, process.n_actions, axis=1) / process.n_actions
            expect = spread @ process.emissions[m + 1][:, 0]
        assert np.allclose(stack[:, pos, 0], expect, atol=1e-5), pos


def test_position_restricted_edit_touches_only_its_positions():
    """E2b is only meaningful if the corruption really is confined to position t;
    everything downstream is supposed to be intact and merely attending to it."""
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=10, n_ctx=12, n_layers=2, n_heads=1, d_model=32, d_mlp=64)
    model = TinyTransformer(cfg).eval()
    tokens = torch.randint(0, cfg.vocab_size, (8, cfg.n_ctx))
    stream = model.residual_streams(tokens)[1]
    mean = stream.reshape(-1, cfg.d_model).mean(0)
    basis = random_basis(np.random.default_rng(0), cfg.d_model, 6)

    edit = restrict_to_positions(mean_ablate(basis, mean), np.array([3]))
    x = torch.as_tensor(stream)
    y = edit(x).numpy()
    assert not np.allclose(y[:, 3], stream[:, 3]), "the target position must change"
    untouched = [p for p in range(cfg.n_ctx) if p != 3]
    assert np.allclose(y[:, untouched], stream[:, untouched], atol=1e-6)


def test_variance_matched_control_ranks_the_way_the_argument_needs():
    """Top-r principal directions must capture more variance than an isotropic
    subspace of the same rank. The whole R2 repair rests on that ordering."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(2000, 32)) @ np.diag(np.linspace(5.0, 0.1, 32))
    for rank in (2, 8, 16):
        top = variance_fraction(pca_basis(x, rank), x)
        iso = np.mean([variance_fraction(random_basis(rng, 32, rank), x) for _ in range(8)])
        assert top > iso, (rank, top, iso)
        assert abs(iso - rank / 32) < 0.08, (rank, iso)


def test_complement_erasure_separates_overlapping_subspaces():
    """R4: when two bases share directions, the complements are what is left of
    each after the shared part is removed, and they must be orthogonal to it."""
    rng = np.random.default_rng(0)
    q = random_basis(rng, 32, 10)
    a, b = q[:, :6], q[:, 3:9]  # three shared directions
    a_only = complement_basis(a, b)
    assert a_only.shape[1] == 3
    assert principal_angles(a_only, b).min() > 89.0, "the complement must be clear of b"
    assert np.allclose(a_only.T @ a_only, np.eye(3), atol=1e-8)


def test_simplex_coords_preserve_what_r2_should_see():
    """C17: scoring A dependent columns is not the same as scoring A-1 free ones.
    The contrasts must be an isometry on the simplex's tangent space."""
    rng = np.random.default_rng(0)
    p = rng.dirichlet(np.ones(5), size=300)
    q = rng.dirichlet(np.ones(5), size=300)
    lhs = np.linalg.norm(p - q, axis=1)
    rhs = np.linalg.norm(simplex_coords(p) - simplex_coords(q), axis=1)
    assert np.allclose(lhs, rhs, atol=1e-9), "contrasts must preserve distances"


def test_token_window_baseline_cannot_see_the_future():
    """V3 is only a fair baseline if it is causal, like the model it competes with."""
    rng = np.random.default_rng(0)
    tokens = rng.integers(0, 6, size=(4, 10))
    feats = token_window_features(tokens, 6, 3, 5)
    other = tokens.copy()
    other[:, 7:] = (other[:, 7:] + 1) % 6
    assert np.allclose(feats[:, :7], token_window_features(other, 6, 3, 5)[:, :7])


def test_scaled_kick_changes_only_the_magnitude():
    for name in BRANCH_CONFIGS:
        assert scaled_kick(name, 1.0) == {}
        doubled = scaled_kick(name, 2.0)["system_kwargs"]
        base = BRANCH_CONFIGS[name]["system_kwargs"]
        assert doubled["kick"] == base["kick"] * 2
        assert {k: v for k, v in doubled.items() if k != "kick"} == {
            k: v for k, v in base.items() if k != "kick"
        }


def test_intervened_loss_per_position_reduces_to_the_scalar():
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=8, n_ctx=10, n_layers=2, n_heads=1, d_model=16, d_mlp=32)
    model = TinyTransformer(cfg).eval()
    tokens = torch.randint(0, cfg.vocab_size, (6, cfg.n_ctx))
    per_pos = intervened_loss(model, tokens, 1, None, reduce=False)
    assert per_pos.shape == (6, cfg.n_ctx - 1)
    assert abs(float(per_pos.mean()) - intervened_loss(model, tokens, 1, None)) < 1e-5
