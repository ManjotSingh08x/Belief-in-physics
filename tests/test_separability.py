import numpy as np
import scipy.sparse as sp

from physics.grid import LatentGrid
from physics.hmm import DiscreteHMM
from physics.separability import evaluate_separability


def _toy_hmm():
    rng = np.random.default_rng(0)
    n = 6
    grid = LatentGrid(bin_counts=(n,), domain=((0.0, 1.0),), periodic=(False,))
    T = sp.csr_matrix(rng.dirichlet(np.ones(n) * 3, size=n))
    E = rng.dirichlet(np.ones(4), size=n)
    noop = sp.eye(n, format="csr")
    flip = sp.csr_matrix(np.eye(n)[::-1])
    return DiscreteHMM(T=T, P_actions=(noop, flip), E=E, metric_table=np.zeros((n, 1)), grid=grid, action_names=("noop", "flip"))


def test_noop_has_zero_tv_against_itself():
    hmm = _toy_hmm()
    b0 = np.array([0.5, 0.2, 0.1, 0.1, 0.05, 0.05])
    report = evaluate_separability(hmm, [b0], n_steps=3, noop_action=0)
    assert report.tv_vs_noop_first[0] == 0.0


def test_a_real_permutation_shifts_a_nonuniform_belief():
    hmm = _toy_hmm()
    b0 = np.array([0.5, 0.2, 0.1, 0.1, 0.05, 0.05])
    report = evaluate_separability(hmm, [b0], n_steps=3, noop_action=0)
    assert report.tv_vs_noop_first[1] > 0.0


def test_a_permutation_does_not_move_a_uniform_belief():
    """Degenerate case that separability tuning must not be fooled by: a
    uniform belief is invariant under any permutation, so TV is zero even
    though the action genuinely does something to a non-uniform belief.
    """
    hmm = _toy_hmm()
    b0 = np.full(6, 1.0 / 6)
    report = evaluate_separability(hmm, [b0], n_steps=3, noop_action=0)
    assert report.tv_vs_noop_first[1] == 0.0


def test_passes_is_json_serialisable_python_bool():
    hmm = _toy_hmm()
    b0 = np.array([0.5, 0.2, 0.1, 0.1, 0.05, 0.05])
    report = evaluate_separability(hmm, [b0], n_steps=3, noop_action=0)
    assert isinstance(report.passes(), bool)
