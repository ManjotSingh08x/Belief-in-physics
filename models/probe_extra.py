"""Probes that the linear one cannot stand in for, and the coordinates it should
have been scored in.

Four instruments, each answering a specific objection to a linear probe's number.

**`mlp_probe`** -- "not linearly decodable" and "not represented" are different
statements, and only a nonlinear probe separates them. Two uses: a ceiling for a
target whose flat linear R^2 would otherwise be unfalsifiable, and a check on
erasure, because iterative nullspace projection guarantees linear
non-decodability and nothing more. If an MLP recovers a target after INLP,
"erased" is the wrong word.

**`token_window_features`** -- a bag of the last W observation tokens plus the
position in the tick. If that reaches the transformer's R^2 then the
transformer's number is not evidence of a learned belief, it is evidence that
the belief is a simple function of recent tokens.

**`simplex_coords`** -- a block of A marginals has A - 1 degrees of freedom, so
averaging A per-column R^2 scores a dependent coordinate and understates or
overstates depending on the geometry. Helmert contrasts give an orthonormal
basis of the centred simplex, which is where R^2 should be read.

**`stratified_r2`** -- the belief is not equally uncertain at every position in a
tick. Pooling over positions lets the sharpest end, where the target is close to
a discrete label, carry the headline.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .bootstrap import r2_columns
from .probe import fit_probe


def simplex_coords(block: np.ndarray) -> np.ndarray:
    """Helmert contrasts: an orthonormal basis of the (A-1)-simplex's tangent.

    Columns of the returned array are independent, so a mean per-column R^2 over
    them scores A - 1 real degrees of freedom instead of A dependent ones.
    """
    a = block.shape[-1]
    h = np.zeros((a, a - 1))
    for k in range(a - 1):
        h[: k + 1, k] = 1.0
        h[k + 1, k] = -(k + 1.0)
        h[:, k] /= np.linalg.norm(h[:, k])
    return block @ h


def token_window_features(
    tokens: np.ndarray, vocab_size: int, window: int, steps_per_segment: int
) -> np.ndarray:
    """One-hot bag of the last `window` tokens, plus one-hot position in segment.

    Deliberately the dumbest sufficient-looking baseline. Positions before the
    start of the sequence contribute nothing rather than a pad symbol, so the
    early positions are honestly under-informed rather than given a extra feature
    the transformer does not have.
    """
    n, L = tokens.shape
    out = np.zeros((n, L, window * vocab_size + steps_per_segment), dtype=np.float32)
    for lag in range(window):
        src = tokens[:, : L - lag] if lag else tokens
        rows = np.arange(n)[:, None]
        cols = np.arange(lag, L)[None]
        out[rows, cols, lag * vocab_size + src] = 1.0
    phase = np.arange(L) % steps_per_segment
    out[:, np.arange(L), window * vocab_size + phase] = 1.0
    return out


class _MLP(nn.Module):
    def __init__(self, d_in: int, d_out: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, d_out),
        )

    def forward(self, x):
        return self.net(x)


def mlp_probe(
    a_train: np.ndarray,
    f_train: np.ndarray,
    a_test: np.ndarray,
    f_test: np.ndarray,
    hidden: int = 256,
    epochs: int = 300,
    lr: float = 3e-3,
    val_frac: float = 0.15,
    device: str = "cpu",
    seed: int = 0,
) -> dict:
    """Held-out R^2 for a two-hidden-layer MLP, with early stopping on a
    validation slice carved out of the training rows.

    Inputs are standardised because the residual-stream scale grows during
    training and an unstandardised MLP would read that growth as signal.
    """
    torch.manual_seed(seed)
    mu, sd = a_train.mean(0), a_train.std(0) + 1e-8
    tm, ts = f_train.mean(0), f_train.std(0) + 1e-8

    def prep(a, f):
        return (
            torch.as_tensor((a - mu) / sd, dtype=torch.float32, device=device),
            torch.as_tensor((f - tm) / ts, dtype=torch.float32, device=device),
        )

    n_val = max(1, int(a_train.shape[0] * val_frac))
    perm = np.random.default_rng(seed).permutation(a_train.shape[0])
    xtr, ytr = prep(a_train[perm[n_val:]], f_train[perm[n_val:]])
    xva, yva = prep(a_train[perm[:n_val]], f_train[perm[:n_val]])
    xte, _ = prep(a_test, f_test)

    model = _MLP(a_train.shape[1], f_train.shape[1], hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    best, best_state, patience = float("inf"), None, 0

    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        nn.functional.mse_loss(model(xtr), ytr).backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            v = float(nn.functional.mse_loss(model(xva), yva))
        if v < best - 1e-5:
            best, best_state, patience = v, {k: p.clone() for k, p in model.state_dict().items()}, 0
        else:
            patience += 1
            if patience > 40:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(xte).cpu().numpy() * ts + tm
    return {"r2": r2_columns(pred, f_test), "val_mse": best, "hidden": hidden}


def stratified_r2(probe, activations: np.ndarray, targets: np.ndarray, strata: np.ndarray, n_bins: int = 4) -> list[dict]:
    """R^2 within bins of `strata`, e.g. normalised belief entropy.

    A high pooled R^2 driven entirely by the low-entropy bin means the probe is
    reading a nearly-discrete label, not a simplex point.
    """
    pred = probe(activations)
    edges = np.quantile(strata, np.linspace(0, 1, n_bins + 1))
    edges[-1] += 1e-9
    out = []
    for i in range(n_bins):
        mask = (strata >= edges[i]) & (strata < edges[i + 1])
        if mask.sum() < 20:
            continue
        out.append(
            {
                "bin": i,
                "entropy_range": [float(edges[i]), float(edges[i + 1])],
                "n": int(mask.sum()),
                "r2": r2_columns(pred[mask], targets[mask]),
            }
        )
    return out


def _demo() -> None:
    rng = np.random.default_rng(0)

    # Helmert contrasts must be orthonormal and kill the constant direction.
    block = rng.dirichlet(np.ones(4), size=200)
    h = simplex_coords(np.eye(4))
    assert np.allclose(h.T @ h, np.eye(3), atol=1e-10), "contrasts must be orthonormal"
    assert np.allclose(np.ones(4) @ h, 0, atol=1e-10), "constant direction must vanish"
    assert simplex_coords(block).shape == (200, 3)

    # The token window must one-hot the right lags and never see the future.
    tokens = rng.integers(0, 5, size=(3, 6))
    feats = token_window_features(tokens, 5, 2, 3)
    assert feats.shape == (3, 6, 2 * 5 + 3)
    assert feats[0, 4, tokens[0, 4]] == 1.0, "lag 0 must be the current token"
    assert feats[0, 4, 5 + tokens[0, 3]] == 1.0, "lag 1 must be the previous token"
    assert feats[0, 0, 5:10].sum() == 0.0, "no lag-1 token exists at position 0"

    # The MLP must beat a linear probe on a target that is nonlinear in the input.
    n, d = 3000, 12
    x = rng.normal(size=(n, d))
    y = np.stack([np.sin(2 * x[:, 0]) * x[:, 1], (x[:, 2] ** 2 - 1.0)], axis=1)
    tr, te = slice(0, 2400), slice(2400, n)
    linear = r2_columns(fit_probe(x[tr], y[tr])(x[te]), y[te])
    nonlinear = mlp_probe(x[tr], y[tr], x[te], y[te], hidden=128, epochs=600)["r2"]
    assert nonlinear > linear + 0.2, (linear, nonlinear)

    probe = fit_probe(x[tr], y[tr])
    strata = np.abs(x[te, 0])
    bins = stratified_r2(probe, x[te], y[te], strata, n_bins=3)
    assert len(bins) == 3 and all(b["n"] > 20 for b in bins)
    print(f"probe_extra ok: linear={linear:.3f} mlp={nonlinear:.3f}")


if __name__ == "__main__":
    _demo()
