"""Closed-form (no Monte-Carlo) check that a perturbation's effect on the
observation stream is actually detectable -- computed directly from the HMM
matrices via predictive observation distributions.

    p_a(t) = b0 @ P_a @ T^t @ E     predictive obs distribution t steps after action a
    p_0(t) = b0 @ T^t @ E           same, no perturbation at all
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .hmm import DiscreteHMM

TV_VS_NOOP_FIRST_STEP = 0.15
TV_VS_NOOP_MEAN = 0.10
TV_PAIRWISE_ACTIONS = 0.10


def total_variation(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    return 0.5 * np.abs(p - q).sum(axis=-1)


def predictive_obs_curve(hmm: DiscreteHMM, belief0: np.ndarray, action: int | None, n_steps: int) -> np.ndarray:
    """(n_steps, n_obs) predictive P(observation) at each of the next n_steps
    free-flow steps, optionally preceded by `action`'s instantaneous push.
    """
    b = belief0 @ hmm.P_actions[action] if action is not None else belief0
    out = np.empty((n_steps, hmm.n_obs))
    for t in range(n_steps):
        b = b @ hmm.T
        out[t] = b @ hmm.E
    return out


@dataclass(frozen=True)
class SeparabilityReport:
    action_names: tuple[str, ...]
    tv_vs_noop_first: np.ndarray  # (n_actions,)
    tv_vs_noop_mean: np.ndarray   # (n_actions,)
    tv_pairwise: np.ndarray       # (n_actions, n_actions)
    noop_action: int | None

    def passes(self) -> bool:
        non_noop = [a for a in range(len(self.action_names)) if a != self.noop_action]
        if not non_noop:
            return True
        first_ok = all(self.tv_vs_noop_first[a] >= TV_VS_NOOP_FIRST_STEP for a in non_noop)
        mean_ok = all(self.tv_vs_noop_mean[a] >= TV_VS_NOOP_MEAN for a in non_noop)
        pair_ok = True
        for i in non_noop:
            for j in non_noop:
                if i < j:
                    pair_ok = pair_ok and self.tv_pairwise[i, j] >= TV_PAIRWISE_ACTIONS
        return bool(first_ok and mean_ok and pair_ok)

    def as_dict(self) -> dict:
        return {
            "action_names": list(self.action_names),
            "tv_vs_noop_first": self.tv_vs_noop_first.tolist(),
            "tv_vs_noop_mean": self.tv_vs_noop_mean.tolist(),
            "tv_pairwise": self.tv_pairwise.tolist(),
            "passes": self.passes(),
        }


def evaluate_separability(
    hmm: DiscreteHMM, belief0s: list[np.ndarray], n_steps: int, noop_action: int | None = None
) -> SeparabilityReport:
    n_actions = hmm.n_actions
    curves = np.zeros((len(belief0s), n_actions, n_steps, hmm.n_obs))
    curves_noop = np.zeros((len(belief0s), n_steps, hmm.n_obs))
    for bi, b0 in enumerate(belief0s):
        curves_noop[bi] = predictive_obs_curve(hmm, b0, None, n_steps)
        for a in range(n_actions):
            curves[bi, a] = predictive_obs_curve(hmm, b0, a, n_steps)

    tv_vs_noop = total_variation(curves, curves_noop[:, None, :, :])  # (n_b0, n_actions, n_steps)
    tv_first = tv_vs_noop[:, :, 0].mean(axis=0)
    tv_mean = tv_vs_noop.mean(axis=(0, 2))

    tv_pairwise = np.zeros((n_actions, n_actions))
    for i in range(n_actions):
        for j in range(n_actions):
            if i == j:
                continue
            tv_pairwise[i, j] = np.mean([total_variation(curves[bi, i], curves[bi, j]).mean() for bi in range(len(belief0s))])

    return SeparabilityReport(
        action_names=hmm.action_names,
        tv_vs_noop_first=tv_first,
        tv_vs_noop_mean=tv_mean,
        tv_pairwise=tv_pairwise,
        noop_action=noop_action,
    )


def choose_magnitude(
    build_hmm_fn, magnitudes: list[float], belief0s: list[np.ndarray], n_steps: int, noop_action: int | None
) -> tuple[float, list[SeparabilityReport]]:
    """Sweep `magnitudes` (each fed to `build_hmm_fn(mag) -> DiscreteHMM`) and
    return the smallest magnitude whose report clears all thresholds.
    """
    reports = []
    for mag in magnitudes:
        hmm = build_hmm_fn(mag)
        report = evaluate_separability(hmm, belief0s, n_steps, noop_action)
        reports.append(report)
        if report.passes():
            return mag, reports
    raise ValueError("no magnitude in the sweep cleared the separability thresholds")


def _demo() -> None:
    from .grid import LatentGrid
    import scipy.sparse as sp

    rng = np.random.default_rng(0)
    n = 6
    grid = LatentGrid(bin_counts=(n,), domain=((0.0, 1.0),), periodic=(False,))
    T = sp.csr_matrix(rng.dirichlet(np.ones(n) * 3, size=n))
    E = rng.dirichlet(np.ones(4), size=n)
    noop = sp.eye(n, format="csr")
    # a "strong" action: reverse the state via a permutation
    perm = np.eye(n)[::-1]
    strong = sp.csr_matrix(perm)
    hmm = DiscreteHMM(
        T=T, P_actions=(noop, strong), E=E, metric_table=np.zeros((n, 1)), grid=grid, action_names=("noop", "flip")
    )
    b0 = np.array([0.5, 0.2, 0.1, 0.1, 0.05, 0.05])  # non-uniform: a permutation actually moves it
    report = evaluate_separability(hmm, [b0], n_steps=3, noop_action=0)
    assert report.tv_vs_noop_first[0] == 0.0, "noop vs itself must have zero TV"
    assert report.tv_vs_noop_first[1] > 0.0, "a real permutation must shift the predictive distribution"
    print("separability ok")


if __name__ == "__main__":
    _demo()
