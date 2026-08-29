"""Sequence-level bootstrap. The only honest error bar in this project.

Positions inside one sequence share a belief history, so the effective sample
size is the number of held-out *sequences* (~154), not the number of held-out
positions (~10k). Resampling positions would understate every interval by
roughly sqrt(seq_len). Every function here resamples sequences with replacement
and recomputes the statistic on the pooled positions of the resampled set.
"""

from __future__ import annotations

import numpy as np


def sequence_bootstrap(
    statistic,
    n_sequences: int,
    rng: np.random.Generator,
    n_boot: int = 1000,
    alpha: float = 0.05,
) -> dict:
    """Percentile CI for `statistic(sequence_indices) -> float`."""
    draws = np.array(
        [statistic(rng.integers(0, n_sequences, size=n_sequences)) for _ in range(n_boot)]
    )
    live = draws[np.isfinite(draws)]
    if live.size == 0:
        return {"lo": float("nan"), "hi": float("nan"), "sd": float("nan"), "n_boot": 0}
    return {
        "lo": float(np.quantile(live, alpha / 2)),
        "hi": float(np.quantile(live, 1 - alpha / 2)),
        "sd": float(live.std()),
        "n_boot": int(live.size),
    }


def r2_columns(pred: np.ndarray, y: np.ndarray) -> float:
    """Mean per-column R^2, dropping columns with no variance to explain."""
    ss_res = ((y - pred) ** 2).sum(axis=0)
    ss_tot = ((y - y.mean(axis=0)) ** 2).sum(axis=0)
    live = ss_tot > 1e-12
    return float(np.mean(1.0 - ss_res[live] / ss_tot[live])) if live.any() else float("nan")


def bootstrap_r2(
    probe, a_test_seq: np.ndarray, f_test_seq: np.ndarray, rng: np.random.Generator, n_boot: int = 1000
) -> dict:
    """CI for a probe's R^2, resampling the held-out sequences.

    `a_test_seq` and `f_test_seq` keep their sequence axis: `(n_seq, L, d)`.
    """
    pred = probe(a_test_seq.reshape(-1, a_test_seq.shape[-1])).reshape(*f_test_seq.shape)

    def stat(idx):
        return r2_columns(
            pred[idx].reshape(-1, pred.shape[-1]), f_test_seq[idx].reshape(-1, f_test_seq.shape[-1])
        )

    out = sequence_bootstrap(stat, a_test_seq.shape[0], rng, n_boot)
    out["point"] = r2_columns(
        pred.reshape(-1, pred.shape[-1]), f_test_seq.reshape(-1, f_test_seq.shape[-1])
    )
    return out


def _demo() -> None:
    rng = np.random.default_rng(0)
    n_seq, L, d = 120, 20, 8
    a = rng.normal(size=(n_seq, L, d))
    w = rng.normal(size=(d, 3))
    f = a @ w + rng.normal(size=(n_seq, L, 3)) * 0.5

    from .probe import fit_probe

    probe = fit_probe(a.reshape(-1, d), f.reshape(-1, 3))
    ci = bootstrap_r2(probe, a, f, rng, n_boot=200)
    assert ci["lo"] < ci["point"] < ci["hi"], ci
    assert ci["hi"] - ci["lo"] > 0, ci

    # A pure-noise target must have an interval that straddles or sits below 0.
    noise = rng.normal(size=(n_seq, L, 3))
    noise_probe = fit_probe(a.reshape(-1, d), noise.reshape(-1, 3))
    ci_noise = bootstrap_r2(noise_probe, a, noise, rng, n_boot=200)
    assert ci_noise["lo"] < 0.1, ci_noise

    # Correlated positions: a position-level bootstrap would give a much
    # narrower interval than the sequence-level one. That gap is the whole point.
    shared = rng.normal(size=(n_seq, 1, d)) + 0.05 * rng.normal(size=(n_seq, L, d))
    f2 = shared @ w
    p2 = fit_probe(shared.reshape(-1, d), f2.reshape(-1, 3))
    wide = bootstrap_r2(p2, shared, f2, rng, n_boot=200)
    assert np.isfinite(wide["sd"])
    print(f"bootstrap ok: r2={ci['point']:.3f} [{ci['lo']:.3f}, {ci['hi']:.3f}]")


if __name__ == "__main__":
    _demo()
