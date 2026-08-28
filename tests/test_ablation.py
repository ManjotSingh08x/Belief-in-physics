import numpy as np
import torch

from models import ablation as ablation_module
from models.ablation import ablated_loss, principal_angles, probe_basis, random_basis
from models.transformer import ModelConfig, TinyTransformer


def test_module_self_check():
    ablation_module._demo()


def _model():
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=10, n_ctx=12, n_layers=2, n_heads=1, d_model=32, d_mlp=64)
    return TinyTransformer(cfg).eval(), cfg


def test_effect_grows_with_the_rank_removed():
    """The reason every ablation needs a matched-rank control: removing more
    directions moves the loss more, whatever those directions mean.

    Magnitude, not sign. On an *untrained* model the loss goes DOWN under
    ablation -- its logits are near-uniform with noise on top, and deleting
    variance removes the noise. So a negative delta at the first checkpoint is
    the expected reading, not a bug, and only the excess over the matched-rank
    control is interpretable.
    """
    model, cfg = _model()
    tokens = torch.randint(0, cfg.vocab_size, (16, cfg.n_ctx))
    mean = model.residual_streams(tokens)[1].reshape(-1, cfg.d_model).mean(axis=0)
    rng = np.random.default_rng(0)

    base = ablated_loss(model, tokens, 1, None)
    effect = [
        abs(ablated_loss(model, tokens, 1, random_basis(rng, cfg.d_model, r), mean) - base)
        for r in (1, 8, 32)
    ]
    assert effect[0] < effect[1] < effect[2], effect


def test_probe_basis_ignores_the_dependent_column_of_a_simplex_block():
    rng = np.random.default_rng(1)
    weight = rng.normal(size=(32, 9))
    weight[:, 2] = -weight[:, :2].sum(axis=1)  # columns 0..2 are a rank-2 simplex block
    assert probe_basis(weight, slice(0, 3)).shape[1] == 2
    assert probe_basis(weight, slice(3, 6)).shape[1] == 3


def test_principal_angles_are_symmetric_and_bounded():
    rng = np.random.default_rng(2)
    a, b = random_basis(rng, 16, 3), random_basis(rng, 16, 4)
    angles = principal_angles(a, b)
    assert angles.shape == (3,)
    assert np.allclose(np.sort(angles), np.sort(principal_angles(b, a)))
    assert (angles >= 0).all() and (angles <= 90 + 1e-9).all()
