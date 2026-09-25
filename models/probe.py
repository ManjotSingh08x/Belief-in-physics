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


from typing import Any


def _helmert_basis(k: int) -> np.ndarray:
    """Orthonormal Helmert contrast matrix of shape (k, k - 1).

    Columns are mutually orthogonal and orthogonal to the all-ones vector,
    forming an orthonormal basis for the tangent space of the (k - 1)-simplex.
    """
    h = np.zeros((k, k - 1), dtype=np.float64)
    for i in range(k - 1):
        h[: i + 1, i] = 1.0
        h[i + 1, i] = -(i + 1.0)
        h[:, i] /= np.linalg.norm(h[:, i])
    return h


helmert_basis = _helmert_basis


def decode_to_simplex(
    probe: AffineProbe | Any,
    activations: np.ndarray,
    helmert_basis: np.ndarray | None = None,
    simplex_verts: np.ndarray | None = None,
    k: int = 4,
) -> np.ndarray:
    """Invert Helmert coordinates predicted by `probe` into Cartesian points on the simplex.

    1. Obtains Helmert predictions from probe(activations) (or activations directly if not callable).
    2. Reconstructs barycentric coordinates: bary = helmert_pred @ helmert_basis.T + (1.0 / k).
    3. Projects onto simplex by clipping non-negative and normalising to sum to 1.
    4. Projects onto regular simplex vertices: bary @ simplex_verts.

    Returns:
        (N, d_cart) array of Cartesian coordinates (e.g. (N, 3) for regular tetrahedron).
    """
    if helmert_basis is None:
        helmert_basis = _helmert_basis(k)
    if simplex_verts is None:
        from physics.messk import simplex_embedding

        simplex_verts = simplex_embedding(k)

    helmert_pred = probe(activations) if callable(probe) else activations
    bary = helmert_pred @ helmert_basis.T + (1.0 / k)
    bary = np.clip(bary, 0.0, None)
    bary_sum = bary.sum(axis=-1, keepdims=True)
    bary = bary / np.clip(bary_sum, 1e-12, None)
    return bary @ simplex_verts


def layerwise_r2(
    streams: list[np.ndarray],
    targets: np.ndarray,
    live_ticks: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    ridge: float = 1e-3,
) -> list[dict[str, Any]]:
    """Fit and evaluate single-token linear probes at each residual-stream depth.

    Parameters:
        streams: list of (N, seq_len, d_model) arrays (length n_layers + 1).
                 stream[0] is token embedding; stream[1:] are transformer block outputs.
        targets: (N * n_live, d_target) or (N, n_live, d_target) target features.
        live_ticks: 1-D array of tick-end integer indices along seq_len.
        train_mask: (N,) bool array of training sequence indicators.
        test_mask: (N,) bool array of test sequence indicators.
        ridge: regularisation strength for `fit_probe`.

    Returns:
        list of dicts with keys: 'depth', 'name', 'test_r2', 'n_train', 'n_test'.
    """
    n_live = len(live_ticks)
    if targets.ndim == 3:
        f_train = targets[train_mask].reshape(-1, targets.shape[-1])
        f_test = targets[test_mask].reshape(-1, targets.shape[-1])
    elif targets.ndim == 2:
        if targets.shape[0] == len(train_mask) * n_live:
            rep_tr = np.repeat(train_mask, n_live)
            rep_te = np.repeat(test_mask, n_live)
            f_train = targets[rep_tr]
            f_test = targets[rep_te]
        elif targets.shape[0] == len(train_mask):
            f_train = targets[train_mask]
            f_test = targets[test_mask]
        else:
            raise ValueError(
                f"targets shape {targets.shape} incompatible with train/test masks and n_live={n_live}"
            )
    else:
        raise ValueError(f"targets must be 2-D or 3-D, got shape {targets.shape}")

    records: list[dict[str, Any]] = []
    for depth, stream in enumerate(streams):
        a_full = stream[:, live_ticks, :]
        d_model = stream.shape[-1]
        a_tr = a_full[train_mask].reshape(-1, d_model).astype(np.float64)
        a_te = a_full[test_mask].reshape(-1, d_model).astype(np.float64)
        probe = fit_probe(a_tr, f_train, ridge=ridge)
        score = probe_quality(probe, a_te, f_test)
        name = "emb" if depth == 0 else f"L{depth}"
        records.append({
            "depth": depth,
            "name": name,
            "test_r2": float(score),
            "n_train": len(a_tr),
            "n_test": len(a_te),
        })
    return records


def reservoir_control_r2(
    tokens: np.ndarray,
    targets_helmert: np.ndarray,
    live_ticks: np.ndarray,
    vocab_size: int,
    window: int,
    steps_per_tick: int,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    alpha: float = 1.0,
) -> float:
    """Held-out R^2 for a linear probe on sparse token-window features at live tick-ends.

    Tests whether an explicit window of raw observation tokens linearly
    predicts the belief state as well as the neural representation does.
    """
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score

    from .probe_extra import sparse_token_window_features

    n, length = tokens.shape
    n_live = len(live_ticks)
    flat_tick_rows = (np.arange(n)[:, None] * length + live_ticks[None, :]).reshape(-1)

    X_sp = sparse_token_window_features(tokens, vocab_size, window, steps_per_tick)
    X_live = X_sp[flat_tick_rows]

    tr_idx = np.repeat(train_mask, n_live)
    te_idx = np.repeat(test_mask, n_live)

    if targets_helmert.ndim == 3:
        y_all = targets_helmert.reshape(-1, targets_helmert.shape[-1])
    else:
        y_all = targets_helmert

    y_tr = y_all[tr_idx]
    y_te = y_all[te_idx]

    reg = Ridge(alpha=alpha, solver="lsqr")
    reg.fit(X_live[tr_idx], y_tr)
    y_pred = reg.predict(X_live[te_idx])
    return float(r2_score(y_te, y_pred, multioutput="uniform_average"))


def _demo_extended() -> None:
    rng = np.random.default_rng(42)
    # Test Helmert basis
    h = _helmert_basis(4)
    assert h.shape == (4, 3)
    assert np.allclose(h.T @ h, np.eye(3), atol=1e-10)
    assert np.allclose(np.ones(4) @ h, 0.0, atol=1e-10)

    # Test decode_to_simplex
    from physics.messk import simplex_embedding

    verts = simplex_embedding(4)
    raw_p = rng.dirichlet(np.ones(4), size=50)
    c = raw_p @ h
    # An identity probe on Helmert coordinates
    decoded_xyz = decode_to_simplex(lambda x: x, c, helmert_basis=h, simplex_verts=verts, k=4)
    true_xyz = raw_p @ verts
    assert np.allclose(decoded_xyz, true_xyz, atol=1e-7), "exact belief must reconstruct exact xyz"

    # Test layerwise_r2
    n_seq, seq_len, d_model = 20, 32, 16
    n_layers = 2
    streams = [rng.normal(size=(n_seq, seq_len, d_model)) for _ in range(n_layers + 1)]
    live_ticks = np.array([7, 15, 23, 31])
    n_live = len(live_ticks)
    train_mask = np.zeros(n_seq, dtype=bool)
    train_mask[:14] = True
    test_mask = ~train_mask
    targets = rng.normal(size=(n_seq * n_live, 3))
    records = layerwise_r2(streams, targets, live_ticks, train_mask, test_mask)
    assert len(records) == 3
    assert [r["name"] for r in records] == ["emb", "L1", "L2"]
    assert all("test_r2" in r for r in records)

    # Test reservoir_control_r2
    tokens = rng.integers(0, 10, size=(n_seq, seq_len))
    res_r2 = reservoir_control_r2(
        tokens=tokens,
        targets_helmert=targets,
        live_ticks=live_ticks,
        vocab_size=10,
        window=2,
        steps_per_tick=8,
        train_mask=train_mask,
        test_mask=test_mask,
    )
    assert isinstance(res_r2, float)
    print("probe extension ok")


if __name__ == "__main__":
    _demo_extended()

