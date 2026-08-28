"""Exact branch HMM: deterministic physics driven by a hidden 3-way switch.

This replaces the Ulam construction in `builder.py` for the belief-geometry
question, and the reason is worth stating precisely.

Ulam's method bins the phase space and estimates a transition kernel by
sampling inside each box. That makes the resulting HMM *stochastic* -- a box
spreads over several boxes -- but a pendulum is not stochastic. That spread is
a discretisation artefact, so the belief computed from it is the belief in an
artificially noisy surrogate rather than in the real process.

Here the only random events are the ones that are genuinely random:

    z0        drawn from a small finite set
    w_1..w_M  the perturbations, drawn uniformly and NOT shown to the model

Everything between perturbations is the true ODE, integrated exactly. So a
hidden state is a *branch* -- a pair (z0, action word) -- and given a branch the
trajectory is determined. There is no discretisation anywhere in the dynamics;
binning survives only in the emission, where it belongs, because the observation
genuinely is coarse.

The belief is therefore a posterior over "which perturbations happened", which
is what makes this a Mess3 analogue: the marginal over the most recent action is
a point in the (n_actions - 1) simplex, exactly the object Shai et al. plot.

Branch count is n_z0 * n_actions^m after m perturbations, so the state space is
finite and the forward algorithm is exact by enumeration rather than by
approximation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .channels import gaussian_channel
from .hmm import _sample_dense_categorical


@dataclass(frozen=True)
class BranchEpisodes:
    tokens: np.ndarray  # (n, L) int, vocabulary is observation bins ONLY
    branch: np.ndarray  # (n, L) int, true branch index at each position
    z0_index: np.ndarray  # (n,) int
    actions: np.ndarray  # (n, M) int, the hidden perturbations
    metrics: np.ndarray  # (n, L, d_metric) float, ground truth at the true branch
    segment: np.ndarray  # (n_positions,) int, which perturbation segment each position is in


@dataclass(frozen=True)
class BranchProcess:
    """Precomputed exact trajectories for every reachable branch."""

    z0: np.ndarray  # (n_z0, state_dim)
    n_actions: int
    M: int
    steps_per_segment: int
    bin_edges: np.ndarray
    action_names: tuple[str, ...]
    obs_tables: tuple[np.ndarray, ...]  # per segment: (n_branch, steps)
    metric_tables: tuple[np.ndarray, ...]  # per segment: (n_branch, steps, d_metric)
    emissions: tuple[np.ndarray, ...]  # per segment: (n_branch, steps, n_obs)

    @property
    def n_z0(self) -> int:
        return self.z0.shape[0]

    @property
    def n_obs(self) -> int:
        return self.emissions[0].shape[-1]  # derived, so a custom emission_fn works too

    @property
    def seq_len(self) -> int:
        return self.M * self.steps_per_segment

    def n_branches(self, level: int) -> int:
        """Reachable branches after `level` perturbations."""
        return self.n_z0 * self.n_actions**level

    @property
    def segment_of_position(self) -> np.ndarray:
        return np.repeat(np.arange(self.M), self.steps_per_segment)

    def sample_batch(self, rng: np.random.Generator, n: int) -> BranchEpisodes:
        z0_index = rng.integers(0, self.n_z0, size=n)
        actions = rng.integers(0, self.n_actions, size=(n, self.M))

        tokens = np.empty((n, self.seq_len), dtype=np.int64)
        branch_at = np.empty((n, self.seq_len), dtype=np.int64)
        metrics = np.empty((n, self.seq_len, self.metric_tables[0].shape[-1]))

        branch = z0_index.copy()
        for m in range(self.M):
            branch = branch * self.n_actions + actions[:, m]
            emission, metric = self.emissions[m], self.metric_tables[m]
            for s in range(self.steps_per_segment):
                pos = m * self.steps_per_segment + s
                tokens[:, pos] = _sample_dense_categorical(emission[:, s], branch, rng)
                branch_at[:, pos] = branch
                metrics[:, pos] = metric[branch, s]

        return BranchEpisodes(
            tokens=tokens,
            branch=branch_at,
            z0_index=z0_index,
            actions=actions,
            metrics=metrics,
            segment=self.segment_of_position,
        )

    def forward(self, tokens: np.ndarray) -> list[np.ndarray]:
        """Exact posterior over branches at every position, materialised.

        Convenient for small batches. The belief simplex grows with each
        perturbation, so retaining every position costs ~2.5 MB per sequence --
        use `iter_beliefs` (or `forward_features`) for anything batch-sized.
        """
        return list(self.iter_beliefs(tokens))

    def iter_beliefs(self, tokens: np.ndarray):
        """Yield the exact posterior at each position, holding only the current one."""
        n = tokens.shape[0]
        belief = np.full((n, self.n_z0), 1.0 / self.n_z0)

        for m in range(self.M):
            # A perturbation is drawn uniformly and is unobserved, so it splits
            # every branch into n_actions children of equal prior mass.
            belief = np.repeat(belief, self.n_actions, axis=1) / self.n_actions
            emission = self.emissions[m]
            for s in range(self.steps_per_segment):
                pos = m * self.steps_per_segment + s
                likelihood = emission[:, s][:, tokens[:, pos]].T  # (n, n_branch)
                belief = belief * likelihood
                belief /= belief.sum(axis=1, keepdims=True)
                yield belief


def build_branch_process(
    system,
    z0: np.ndarray,
    K: int,
    dt: float,
    steps_per_segment: int,
    M: int,
    bin_edges: np.ndarray,
    noise_std: float,
    action_names: tuple[str, ...],
    periodic: bool = False,
    period: tuple[float, float] | None = None,
    emission_fn=None,
) -> BranchProcess:
    """Integrate every reachable branch exactly and tabulate its observations."""
    obs_tables, metric_tables, emissions = [], [], []
    states = np.asarray(z0, dtype=np.float64)
    n_actions = system.n_actions

    for _ in range(M):
        # Branch first: child index is parent * n_actions + action, which makes
        # the flat index the mixed-radix action word and lets `forward` recover
        # any lag's marginal by a reshape.
        children = np.stack([system.apply_action(states, a) for a in range(n_actions)], axis=1)
        z = children.reshape(-1, states.shape[-1])

        # The observable may be scalar (theta, a population ratio) or vector
        # (the sphere's lat/lon), so keep a trailing axis in both cases.
        obs_dim = np.atleast_1d(system.observable(z[:1])[0]).size
        obs = np.empty((z.shape[0], steps_per_segment, obs_dim))
        met = np.empty((z.shape[0], steps_per_segment, system.metrics(z[:1]).shape[-1]))
        for s in range(steps_per_segment):
            z = system.flow(z, dt, K)  # the deterministic part, integrated exactly
            obs[:, s] = np.asarray(system.observable(z)).reshape(z.shape[0], obs_dim)
            met[:, s] = system.metrics(z)

        flat = obs.reshape(-1, obs_dim)
        channel = (
            emission_fn(flat)
            if emission_fn is not None
            else gaussian_channel(flat[:, 0], bin_edges, noise_std, periodic, period)
        )
        emissions.append(channel.reshape(obs.shape[0], steps_per_segment, -1).astype(np.float64))
        obs_tables.append(obs)
        metric_tables.append(met)
        states = z

    return BranchProcess(
        z0=np.asarray(z0, dtype=np.float64),
        n_actions=n_actions,
        M=M,
        steps_per_segment=steps_per_segment,
        bin_edges=np.asarray(bin_edges),
        action_names=tuple(action_names),
        obs_tables=tuple(obs_tables),
        metric_tables=tuple(metric_tables),
        emissions=tuple(emissions),
    )


N_ACTION_LAGS = 3


def feature_groups(process: BranchProcess) -> dict[str, slice]:
    a, col = process.n_actions, 0
    groups: dict[str, slice] = {}
    for lag in range(N_ACTION_LAGS):
        groups[f"action_lag{lag}"] = slice(col, col + a)
        col += a
    groups["z0"] = slice(col, col + process.n_z0)
    col += process.n_z0
    groups["metric"] = slice(col, col + process.metric_tables[0].shape[-1])
    return groups


def branch_features(process: BranchProcess, beliefs: list[np.ndarray]) -> np.ndarray:
    """Fixed-size probe targets from the growing branch belief: (n, L, n_features).

    Every column is a marginal of the belief, so all of them are *exactly*
    linear in it -- `action_lag0` is the Mess3 simplex point, `z0` is the
    posterior over initial conditions, `metric` is E_b[metric].
    """
    a, n_z0 = process.n_actions, process.n_z0
    groups = feature_groups(process)
    n_features = max(s.stop for s in groups.values())
    n = beliefs[0].shape[0]

    out = np.empty((n, len(beliefs), n_features), dtype=np.float64)
    for pos, belief in enumerate(beliefs):
        m = pos // process.steps_per_segment
        s = pos % process.steps_per_segment
        level = m + 1
        grid = belief.reshape(n, n_z0, *([a] * level))  # axes 2..level+1 are w_1..w_level

        for lag in range(N_ACTION_LAGS):
            column = groups[f"action_lag{lag}"]
            if lag < level:
                axis = level + 1 - lag  # w_{level-lag}
                keep = tuple(ax for ax in range(1, level + 2) if ax != axis)
                out[:, pos, column] = grid.sum(axis=keep)
            else:
                out[:, pos, column] = 1.0 / a  # that perturbation has not happened yet

        out[:, pos, groups["z0"]] = grid.sum(axis=tuple(range(2, level + 2)))
        out[:, pos, groups["metric"]] = belief @ process.metric_tables[m][:, s]
    return out


def _brute_force_posterior(process: BranchProcess, tokens: np.ndarray) -> np.ndarray:
    """Posterior over full branches by explicit enumeration of every (z0, word)."""
    n = tokens.shape[0]
    n_full = process.n_branches(process.M)
    log_p = np.zeros((n, n_full))
    for full in range(n_full):
        for m in range(process.M):
            # the level-(m+1) prefix of this full branch
            prefix = full // (process.n_actions ** (process.M - m - 1))
            for s in range(process.steps_per_segment):
                pos = m * process.steps_per_segment + s
                log_p[:, full] += np.log(process.emissions[m][prefix, s, tokens[:, pos]])
    log_p -= log_p.max(axis=1, keepdims=True)
    p = np.exp(log_p)
    return p / p.sum(axis=1, keepdims=True)


def _demo() -> None:
    from .systems.pendulum import Pendulum

    system = Pendulum(gamma=0.15, kick=1.5)
    z0 = np.array([[0.4, 0.0], [-0.3, 0.5]])
    edges = np.linspace(-np.pi, np.pi, 17)
    process = build_branch_process(
        system, z0, K=8, dt=0.02, steps_per_segment=3, M=3,
        bin_edges=edges, noise_std=0.25, action_names=("-dv", "noop", "+dv"),
        periodic=True, period=(-np.pi, np.pi),
    )
    assert process.n_branches(3) == 2 * 3**3 == 54
    assert process.seq_len == 9

    rng = np.random.default_rng(0)
    episodes = process.sample_batch(rng, n=16)
    assert episodes.tokens.shape == (16, 9)
    assert episodes.tokens.max() < process.n_obs, "vocabulary is observations only, no action tokens"

    beliefs = process.forward(episodes.tokens)
    assert len(beliefs) == 9
    for pos, b in enumerate(beliefs):
        level = pos // process.steps_per_segment + 1
        assert b.shape == (16, process.n_branches(level)), (pos, b.shape)
        assert np.allclose(b.sum(axis=1), 1.0), pos

    # The exactness proof: sequential filtering must equal brute-force
    # enumeration over every (z0, action word).
    assert np.allclose(beliefs[-1], _brute_force_posterior(process, episodes.tokens), atol=1e-9)

    features = branch_features(process, beliefs)
    groups = feature_groups(process)
    n_features = max(s.stop for s in groups.values())
    assert features.shape == (16, 9, n_features), features.shape
    assert n_features == 3 * process.n_actions + process.n_z0 + 1, n_features
    for name in ("action_lag0", "action_lag1", "action_lag2", "z0"):
        block = features[:, -1, groups[name]]
        assert np.allclose(block.sum(axis=1), 1.0), name  # every block is a simplex point
        assert (block >= -1e-12).all(), name

    # The belief must actually identify the hidden perturbations better than chance.
    guessed = features[:, -1, groups["action_lag0"]].argmax(axis=1)
    accuracy = (guessed == episodes.actions[:, -1]).mean()
    assert accuracy > 1.0 / 3, f"last-action posterior no better than chance: {accuracy}"
    print(f"branch ok (last-action recovery {accuracy:.2f} vs chance {1/3:.2f})")


if __name__ == "__main__":
    _demo()


def simplex_to_2d(points: np.ndarray) -> np.ndarray:
    """Project points on the (n-1)-simplex to the plane for plotting.

    Vertices are placed on a circle, so n=3 reproduces the usual triangle and
    larger n degrades gracefully. For n > 3 this is a genuine projection and is
    lossy -- distinct beliefs can overlap -- so it is for visualisation only,
    never for any quantitative claim.
    """
    n = points.shape[-1]
    angles = 2 * np.pi * np.arange(n) / n + np.pi / 2
    vertices = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    return points @ vertices


def forward_features(process: BranchProcess, tokens: np.ndarray, chunk: int = 256) -> np.ndarray:
    """Exact beliefs projected straight onto the fixed-size feature map.

    Never retains the belief history: only the current position's simplex is
    resident, so memory is set by the largest single level rather than by the
    sum over positions.
    """
    a, n_z0 = process.n_actions, process.n_z0
    groups = feature_groups(process)
    n_features = max(s.stop for s in groups.values())
    out = np.empty((tokens.shape[0], process.seq_len, n_features), dtype=np.float32)

    for start in range(0, tokens.shape[0], chunk):
        block = tokens[start : start + chunk]
        n = block.shape[0]
        for pos, belief in enumerate(process.iter_beliefs(block)):
            m, s = divmod(pos, process.steps_per_segment)
            level = m + 1
            grid = belief.reshape(n, n_z0, *([a] * level))
            for lag in range(N_ACTION_LAGS):
                col = groups[f"action_lag{lag}"]
                if lag < level:
                    axis = level + 1 - lag
                    keep = tuple(ax for ax in range(1, level + 2) if ax != axis)
                    out[start : start + n, pos, col] = grid.sum(axis=keep)
                else:
                    out[start : start + n, pos, col] = 1.0 / a
            out[start : start + n, pos, groups["z0"]] = grid.sum(axis=tuple(range(2, level + 2)))
            out[start : start + n, pos, groups["metric"]] = belief @ process.metric_tables[m][:, s]
    return out
