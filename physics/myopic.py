"""Exact k-step predictive distributions, and everything that follows from them.

The branch HMM gives the *optimal* next-token predictor in closed form, which is
what makes this project able to quantify "beyond next-token prediction" instead
of asserting it. Three quantities come from here.

**The Bayes floor** `H_floor = E[H(p_t)]`. Next-token cross entropy has a large
irreducible offset, so "the loss moved 1%" is not a statement about convergence
until the floor is subtracted. Excess loss `L - H_floor` is the only comparable
quantity.

**`p^(k)`, the k-step-ahead predictive distribution.** Pushing the belief forward
without conditioning, splitting `n_actions` ways at each segment boundary. The
belief is the sufficient statistic for the whole future, so `R^2(b | p^(1..K))`
must rise to 1 in K; the question the project is really asking is how fast. If
K=3 already reaches 0.9, then ordinary next-token training at positions t+1..t+3
is a complete explanation for why `b_t` is present at t, and "the model
represents more than its objective requires" is false in the way that matters.

**Alignment between the two prediction indices.** The model's loss is
`logits[:, :-1]` against `tokens[:, 1:]`, so the prediction made *at* position t
is about position t+1. `predictive_stack` follows that convention exactly:
entry `[:, t, k-1]` is the law of the token at position `t + k` given the prefix
through position `t`.
"""

from __future__ import annotations

import numpy as np

from .branch import BranchProcess


def _advance(process: BranchProcess, belief: np.ndarray, m: int, s: int):
    """One step of the unconditioned pushforward. Returns (belief, m, s, emission_row).

    Inside a segment the dynamics are deterministic, so the belief does not move
    and only the emission row changes. At a segment boundary the hidden
    perturbation splits every branch `n_actions` ways with equal prior mass,
    which is the only place the belief spreads.
    """
    if s + 1 < process.steps_per_segment:
        return belief, m, s + 1
    if m + 1 >= process.M:
        return None, m, s
    return np.repeat(belief, process.n_actions, axis=1) / process.n_actions, m + 1, 0


def predictive_stack(
    process: BranchProcess, tokens: np.ndarray, horizon: int = 1, chunk: int = 64
) -> np.ndarray:
    """`(n, L, horizon, n_obs)`: law of token `t+k` given the prefix through `t`.

    Positions where `t + k` runs past the sequence are filled with NaN, so a
    caller that forgets to mask them gets a loud failure rather than a quiet
    bias.
    """
    n, L = tokens.shape
    out = np.full((n, L, horizon, process.n_obs), np.nan, dtype=np.float32)

    for start in range(0, n, chunk):
        block = tokens[start : start + chunk]
        nb = block.shape[0]
        for pos, belief in enumerate(process.iter_beliefs(block)):
            m, s = divmod(pos, process.steps_per_segment)
            b, mm, ss = belief, m, s
            for k in range(1, horizon + 1):
                b, mm, ss = _advance(process, b, mm, ss)
                if b is None:
                    break
                out[start : start + nb, pos, k - 1] = b @ process.emissions[mm][:, ss]
    return out


def bayes_floor(process: BranchProcess, tokens: np.ndarray, chunk: int = 64) -> dict:
    """Optimal next-token cross entropy for this process, two ways.

    `plugin` is `E[H(p_t)]`, the expected entropy of the optimal predictive
    distribution. `realised` is `E[-log p_t(x_{t+1})]`, the cross entropy the
    optimal predictor would actually record on this sample. They agree in
    expectation; the gap is finite-sample noise and is reported so it can be
    checked rather than assumed.
    """
    p = predictive_stack(process, tokens, horizon=1)[:, :-1, 0].astype(np.float64)
    nxt = tokens[:, 1:]
    safe = np.clip(p, 1e-12, None)
    plugin = float(-(p * np.log(safe)).sum(axis=-1).mean())
    realised = float(-np.log(np.take_along_axis(safe, nxt[..., None], axis=-1)).mean())
    uniform = float(np.log(process.n_obs))
    return {
        "plugin": plugin,
        "realised": realised,
        "uniform": uniform,
        "n_positions": int(p.shape[0] * p.shape[1]),
    }


def conditional_variance_ceiling(
    p: np.ndarray, target: np.ndarray, radius: float, rng: np.random.Generator, max_clusters: int = 4000
) -> dict:
    """`1 - E[Var(target | p)] / Var(target)`, the best possible predictor of
    `target` from `p`, linear or not.

    Estimated by greedy leader clustering in total-variation distance: pick an
    unassigned point, absorb everything within `radius` TV of it, repeat. Points
    sharing a cluster have (near-)identical predictive distributions, so the
    within-cluster variance of the target is variance that *no* function of `p`
    could explain.

    Biased upward as `radius` grows -- a fat cluster mixes genuinely different
    `p` -- and upward as it shrinks, because singleton clusters have zero
    within-variance. Sweep it and read the plateau, never a single value.
    """
    n = p.shape[0]
    order = rng.permutation(n)
    assigned = np.full(n, -1)
    leaders: list[int] = []

    for i in order:
        if assigned[i] >= 0:
            continue
        if len(leaders) >= max_clusters:
            break
        free = np.flatnonzero(assigned < 0)
        tv = 0.5 * np.abs(p[free] - p[i]).sum(axis=1)
        members = free[tv <= radius]
        assigned[members] = len(leaders)
        leaders.append(i)

    live = assigned >= 0
    labels, y = assigned[live], target[live]
    total = y.var(axis=0)

    within = np.zeros_like(total)
    sizes = np.bincount(labels)
    for c in np.flatnonzero(sizes > 1):
        within += sizes[c] * y[labels == c].var(axis=0)
    within /= sizes[sizes > 1].sum() if (sizes > 1).any() else 1.0

    keep = total > 1e-12
    return {
        "radius": radius,
        "ceiling": float(np.mean(1.0 - within[keep] / total[keep])) if keep.any() else float("nan"),
        "n_clusters": len(leaders),
        "n_used": int(live.sum()),
        "mean_cluster_size": float(sizes.mean()) if sizes.size else float("nan"),
        "singleton_fraction": float((sizes == 1).mean()) if sizes.size else float("nan"),
    }


def _demo() -> None:
    from .branch_configs import make_branch_process

    process = make_branch_process("pendulum")
    rng = np.random.default_rng(0)
    tokens = process.sample_batch(rng, 24).tokens

    stack = predictive_stack(process, tokens, horizon=3)
    assert stack.shape == (24, process.seq_len, 3, process.n_obs), stack.shape
    live = ~np.isnan(stack[..., 0])
    assert np.allclose(stack[live].sum(axis=-1), 1.0, atol=1e-6), "predictive rows must normalise"
    # k steps past the end must be masked, never silently zero.
    assert np.isnan(stack[:, -1, 0]).all(), "no successor exists for the final position"
    assert np.isnan(stack[:, -2, 1]).all()

    floor = bayes_floor(process, tokens)
    assert 0 < floor["plugin"] < floor["uniform"], floor
    assert abs(floor["plugin"] - floor["realised"]) < 0.15, floor

    # The ceiling of a target that IS a function of p must be ~1; of an
    # independent target, ~0.
    p = stack[:, :-1, 0].reshape(-1, process.n_obs).astype(np.float64)
    keep = rng.choice(p.shape[0], min(3000, p.shape[0]), replace=False)
    p = p[keep]
    dependent = p[:, :2] * 3.0
    independent = rng.normal(size=(p.shape[0], 2))
    hi = conditional_variance_ceiling(p, dependent, 0.02, np.random.default_rng(1))
    lo = conditional_variance_ceiling(p, independent, 0.02, np.random.default_rng(1))
    assert hi["ceiling"] > 0.9, hi
    assert lo["ceiling"] < hi["ceiling"], (lo, hi)
    print(f"myopic ok: floor={floor['plugin']:.4f} (uniform {floor['uniform']:.4f}), "
          f"ceiling dependent={hi['ceiling']:.3f} independent={lo['ceiling']:.3f}")


if __name__ == "__main__":
    _demo()
