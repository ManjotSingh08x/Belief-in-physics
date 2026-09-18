"""Linear probes from residual stream to belief features.

`AffineProbe`, `fit_probe`, `probe_quality` and `shuffled_control` are vendored
verbatim from the sibling project `belief-geometry`
(`src/beliefgeom/probe.py`) rather than reimplemented: they are pure numpy,
already generic over `(activations, targets)`, and already carry that project's
controls. Vendored rather than imported because the two repos are separate
git projects with separate environments and different owners.

Added here: `grouped_r2`. The upstream `probe_quality` pools the residual over
every target column, which is fine for one homogeneous dictionary but wrong for
our feature map -- belief probabilities live in [0, 1] while a physical quantity
like angular velocity ranges over several units, so a pooled R^2 is dominated by
whichever block has the largest scale and can hide a total failure on the other.
`grouped_r2` scores each column on its own variance and averages within a group.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AffineProbe:
    """b ~= a W + c, fitted by ridge regression."""

    weight: np.ndarray  # (d_model, n_features)
    bias: np.ndarray  # (n_features,)

    def __call__(self, activations: np.ndarray) -> np.ndarray:
        return activations @ self.weight + self.bias


def fit_probe(activations: np.ndarray, targets: np.ndarray, ridge: float = 1e-3) -> AffineProbe:
    """Least-squares affine map from residual stream to belief features."""
    a_mean, t_mean = activations.mean(axis=0), targets.mean(axis=0)
    centered_a, centered_t = activations - a_mean, targets - t_mean

    gram = centered_a.T @ centered_a
    gram = gram + ridge * np.trace(gram) / gram.shape[0] * np.eye(gram.shape[0])
    weight = np.linalg.solve(gram, centered_a.T @ centered_t)
    return AffineProbe(weight=weight, bias=t_mean - a_mean @ weight)


def probe_quality(probe: AffineProbe, activations: np.ndarray, targets: np.ndarray) -> float:
    """Pooled fraction of belief-feature variance explained on held-out contexts."""
    residual = targets - probe(activations)
    return float(1.0 - np.sum(residual**2) / np.sum((targets - targets.mean(axis=0)) ** 2))


def grouped_r2(
    probe: AffineProbe,
    activations: np.ndarray,
    targets: np.ndarray,
    groups: dict[str, slice],
    min_variance: float = 1e-12,
) -> dict[str, float]:
    """Mean per-column R^2 within each feature group, so scale cannot dominate.

    Columns with no variance on the eval set (a latent bin the trajectories
    never visit) carry no signal to explain and are dropped rather than scored
    as a perfect or a failed fit.
    """
    predicted = probe(activations)
    ss_res = np.sum((targets - predicted) ** 2, axis=0)
    ss_tot = np.sum((targets - targets.mean(axis=0)) ** 2, axis=0)

    out: dict[str, float] = {}
    for name, columns in groups.items():
        live = ss_tot[columns] > min_variance
        if not np.any(live):
            out[name] = float("nan")
            continue
        out[name] = float(np.mean(1.0 - ss_res[columns][live] / ss_tot[columns][live]))
    return out


def shuffled_control(
    rng: np.random.Generator, activations: np.ndarray, targets: np.ndarray
) -> float:
    """Shai et al.'s control: permute the context-to-belief correspondence.

    The regression must collapse. If it does not, the probe is doing the
    modelling work rather than reading it off.
    """
    permutation = rng.permutation(targets.shape[0])
    probe = fit_probe(activations, targets[permutation])
    return probe_quality(probe, activations, targets[permutation])


def _demo() -> None:
    rng = np.random.default_rng(0)
    n, d, k = 800, 16, 5
    activations = rng.normal(size=(n, d))
    weight = rng.normal(size=(d, k))
    targets = activations @ weight + 0.3

    probe = fit_probe(activations, targets)
    assert probe_quality(probe, activations, targets) > 0.99, "exactly linear target must be recovered"

    # Scale blindness is the reason grouped_r2 exists: inflate one column and
    # the pooled score stops seeing a total failure on the others.
    mixed = np.concatenate([targets, rng.normal(size=(n, 1)) * 500.0], axis=1)
    mixed_probe = fit_probe(activations, mixed)
    groups = {"linear": slice(0, k), "noise": slice(k, k + 1)}
    scores = grouped_r2(mixed_probe, activations, mixed, groups)
    assert scores["linear"] > 0.99, scores
    assert scores["noise"] < 0.5, scores

    assert shuffled_control(rng, activations, targets) < 0.1, "shuffled labels must collapse the fit"

    dead = np.concatenate([targets, np.ones((n, 1))], axis=1)
    dead_scores = grouped_r2(fit_probe(activations, dead), activations, dead, {"dead": slice(k, k + 1)})
    assert np.isnan(dead_scores["dead"]), "a constant column has no variance to explain"
    print("probe ok")


if __name__ == "__main__":
    _demo()
