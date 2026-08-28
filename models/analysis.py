"""Phase 3: fit linear probes from the residual stream to belief features.

The control ordering is the point of this file, so it is worth stating plainly.
A linear probe with d_model=128 free parameters per target, fitted on
correlated data, can manufacture a correlation out of nothing. Three things
guard against reading that as a result:

1. **A randomly initialised model of the same shape**, probed identically. Its
   residual stream is a fixed random projection of the token history, which is
   genuinely informative -- a random network is not a zero baseline, and the
   trained model has to beat it, not merely beat chance.
2. **Sequence-level train/test splits.** Positions inside one sequence share a
   belief history, so splitting positions at random leaks the test set into the
   training set and inflates every number.
3. **Shuffled targets**, which must collapse the fit.

Reported per feature group rather than pooled, because the marginals and the
metric live on different scales -- see `probe.grouped_r2`.
"""

from __future__ import annotations

import numpy as np
import torch

from .features import BeliefFeatures
from .probe import fit_probe, grouped_r2, probe_quality, shuffled_control
from .transformer import TinyTransformer


def _sequence_split(n: int, train_frac: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    order = rng.permutation(n)
    cut = int(n * train_frac)
    return order[:cut], order[cut:]


def residual_streams_batched(
    model: TinyTransformer, tokens: np.ndarray, device: str, batch_size: int = 64
) -> list[np.ndarray]:
    """`model.residual_streams` over a large token array, one depth per entry."""
    chunks: list[list[np.ndarray]] = []
    model.eval()
    for start in range(0, tokens.shape[0], batch_size):
        block = torch.as_tensor(tokens[start : start + batch_size], dtype=torch.long, device=device)
        chunks.append(model.residual_streams(block))
    return [np.concatenate([c[depth] for c in chunks], axis=0) for depth in range(len(chunks[0]))]


def probe_layers(
    model: TinyTransformer,
    tokens: np.ndarray,
    features: np.ndarray,
    fmap: BeliefFeatures,
    device: str,
    train_frac: float = 0.7,
    seed: int = 0,
) -> list[dict]:
    """One record per depth: held-out R^2 overall, per group, and the shuffled control."""
    rng = np.random.default_rng(seed)
    train_idx, test_idx = _sequence_split(tokens.shape[0], train_frac, rng)
    streams = residual_streams_batched(model, tokens, device)

    records = []
    for depth, activations in enumerate(streams):
        d_model = activations.shape[-1]
        a_train = activations[train_idx].reshape(-1, d_model).astype(np.float64)
        a_test = activations[test_idx].reshape(-1, d_model).astype(np.float64)
        f_train = features[train_idx].reshape(-1, features.shape[-1]).astype(np.float64)
        f_test = features[test_idx].reshape(-1, features.shape[-1]).astype(np.float64)

        probe = fit_probe(a_train, f_train)
        records.append(
            {
                "depth": depth,
                "name": "embedding" if depth == 0 else f"resid_post_{depth - 1}",
                "r2_pooled": probe_quality(probe, a_test, f_test),
                "r2_by_group": grouped_r2(probe, a_test, f_test, fmap.groups),
                "r2_shuffled_control": shuffled_control(rng, a_train, f_train),
                "n_train": int(a_train.shape[0]),
                "n_test": int(a_test.shape[0]),
            }
        )
    return records


def best_layer(records: list[dict], group: str = "metric") -> dict:
    live = [r for r in records if not np.isnan(r["r2_by_group"].get(group, np.nan))]
    return max(live, key=lambda r: r["r2_by_group"][group]) if live else records[-1]


def _demo() -> None:
    """A probe must recover a target that is genuinely a linear function of the
    residual stream, and must fail on one that is independent of it.
    """
    from .features import BeliefFeatures
    import scipy.sparse as sp

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    n, seq_len, d_model = 120, 12, 32
    tokens = rng.integers(0, 20, size=(n, seq_len))

    from .transformer import ModelConfig

    model = TinyTransformer(ModelConfig(vocab_size=20, n_ctx=seq_len, n_layers=2, d_model=d_model, d_mlp=64))
    streams = residual_streams_batched(model, tokens, "cpu")

    signal = streams[-1] @ rng.normal(size=(d_model, 3))
    noise = rng.normal(size=(n, seq_len, 3))
    features = np.concatenate([signal, noise], axis=-1)
    fmap = BeliefFeatures(sp.csr_matrix((1, 6)), tuple("abcdef"), {"signal": slice(0, 3), "noise": slice(3, 6)})

    records = probe_layers(model, tokens, features, fmap, "cpu")
    assert len(records) == 3
    last = records[-1]
    assert last["r2_by_group"]["signal"] > 0.95, last
    assert last["r2_by_group"]["noise"] < 0.3, last
    assert last["r2_shuffled_control"] < 0.2, last
    print("analysis ok")


if __name__ == "__main__":
    _demo()
