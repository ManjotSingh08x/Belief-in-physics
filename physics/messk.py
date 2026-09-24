"""Mess-K: a K-state chain driving a physical system.

The K-state chain sits in front of a physical system, and the exact belief is
conditioned on the chain's letters:

    mood_1 -> mood_2 -> ...          hidden, persists and drifts
      |          |
    letter     letter                hidden from the model
      |          |
    action     action
               physical system
                     |
              observation bins       the only thing the model sees

The exact belief does not depend on the physics, so all four systems share one
geometric target. Because the chain is stationary and mixing, the belief is a
contractive function of recent letters and forgets its prior at a fixed rate;
`memory_length` reports that rate.

The four experiments use K=4 with alpha=0.7 and stay=0.7, so their common exact
belief geometry is a tetrahedron.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .systems.pendulum import Pendulum


def simplex_embedding(k: int) -> np.ndarray:
    """Vertices of a regular (k-1)-simplex, so a belief plots by barycentric coords.

    k=3 gives the upright triangle the reference picture uses; k=4 a tetrahedron
    in 3-D; beyond that the simplex needs more than three dimensions and only the
    first three are returned, which is a projection rather than a faithful shape.
    """
    if k == 3:
        ang = np.array([90.0, 210.0, 330.0]) * np.pi / 180.0
        return np.stack([np.cos(ang), np.sin(ang)], axis=1)
    if k == 4:
        v = np.array([[1.0, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]])
        return v / np.sqrt(3.0)
    centred = np.eye(k) - 1.0 / k
    return (centred @ np.linalg.svd(centred)[2][: k - 1].T)[:, :3]


def square_projection_vertices() -> np.ndarray:
    """Four state vertices for a readable, lossy planar view of Mess-4.

    A tetrahedron is three-dimensional, so no 2-D map can preserve all belief
    distances. This projection places the four pure states at square corners and
    is for visual inspection only; every probe score remains in the faithful
    three-dimensional simplex coordinates.
    """
    return np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])


@dataclass(frozen=True)
class MessKProcess:
    """K moods, K letters. A mood emits its own letter with probability `alpha`.

    `x` is the chance of moving to each specific other state, derived from the
    total stay probability as `(1-stay)/(K-1)`.
    """

    n_states: int = 4
    alpha: float = 0.7
    stay: float = 0.7

    @property
    def x(self) -> float:
        return (1.0 - self.stay) / (self.n_states - 1)

    @property
    def T(self) -> np.ndarray:
        T = np.full((self.n_states, self.n_states), self.x)
        np.fill_diagonal(T, self.stay)
        return T

    @property
    def E(self) -> np.ndarray:
        noise = (1.0 - self.alpha) / (self.n_states - 1)
        E = np.full((self.n_states, self.n_states), noise)
        np.fill_diagonal(E, self.alpha)
        return E

    @property
    def joint(self) -> np.ndarray:
        """joint[l] = T * E[:, l], the operator the belief applies on letter l."""
        E = self.E
        return np.stack([self.T * E[:, l].reshape(-1, 1) for l in range(self.n_states)])

    def sample(self, rng: np.random.Generator, n: int, m: int) -> tuple[np.ndarray, np.ndarray]:
        """(states, letters) of shapes (n, m+1) and (n, m).

        States carries one extra column: `states[:, t]` is the mood that emitted
        `letters[:, t]`, and `states[:, t+1]` is the mood the belief at step t is
        about. See `beliefs` for why those differ.
        """
        T_cdf, E_cdf = np.cumsum(self.T, axis=1), np.cumsum(self.E, axis=1)
        states = np.empty((n, m + 1), dtype=np.int64)
        letters = np.empty((n, m), dtype=np.int64)
        s = rng.integers(0, self.n_states, size=n)
        for t in range(m):
            states[:, t] = s
            letters[:, t] = np.clip((rng.random((n, 1)) > E_cdf[s]).sum(axis=1), 0, self.n_states - 1)
            s = np.clip((rng.random((n, 1)) > T_cdf[s]).sum(axis=1), 0, self.n_states - 1)
        states[:, m] = s
        return states, letters

    def beliefs(self, letters: np.ndarray) -> np.ndarray:
        """Exact P(mood_{t+1} | letters_1..t), shape (n, m, K).

        This is the predictive belief, one transition ahead of the mood that
        emitted the last letter. The operator is `T * E[:, letter]`, with the
        emission indexed on the source
        state, so each step emits from the current mood and then moves on.

        The distinction is not pedantry. The predictive belief is the thing that
        determines the next letter, hence the next impulse, hence the next angle
        -- so it is the sufficient statistic for the model's actual task, and it
        is the object whose reachable set is the fractal. The filtered belief
        P(mood_t | letters_1..t) is a different quantity.
        """
        n, m = letters.shape
        joint, out = self.joint, np.empty((n, m, self.n_states))
        b = np.full((n, self.n_states), 1.0 / self.n_states)
        for t in range(m):
            b = np.einsum("ni,nij->nj", b, joint[letters[:, t]])
            total = b.sum(axis=1, keepdims=True)
            b = np.where(total > 0, b / np.maximum(total, 1e-300), 1.0 / self.n_states)
            out[:, t] = b
        return out

    def memory_length(self, tol: float = 0.01, seed: int = 0, n_priors: int = 200) -> int:
        """How many letters until the belief forgets which prior it started from.

        Run the same letter stream from many disagreeing priors and report when
        they agree to within `tol` total-variation. This is the number that says
        whether "the model encodes the belief" is separable from "the model
        remembers its recent input".
        """
        rng = np.random.default_rng(seed)
        b = rng.dirichlet(np.ones(self.n_states), size=n_priors)
        _, letters = self.sample(rng, 1, 2000)
        joint = self.joint
        for t, l in enumerate(letters[0], 1):
            b = b @ joint[l]
            b /= b.sum(axis=1, keepdims=True)
            if 0.5 * np.abs(b - b.mean(axis=0)).sum(axis=1).max() < tol:
                return t
        return -1


@dataclass(frozen=True)
class MessDriven:
    """The chain's letters become impulses on a physical system.

    Each system defines four balanced, non-zero action vectors: weak/strong
    negative and positive angular kicks for the pendulum, and opposite directions
    along each of two physical axes for the other systems. `delta_v` sets their
    scale. The observation is the system's scalar observable, binned uniformly
    over its own range into `n_obs` levels with no added noise.

    `system` exposes `actions`, `initial_state`, `flow`, `kick`, `observable`,
    `metric`, `metric_names` and `obs_range`. The chain, belief and binning are identical
    across systems, so a difference in the results is a difference in the
    physics rather than in the pipeline.
    """

    chain: MessKProcess = field(default_factory=MessKProcess)
    system: object = field(default_factory=Pendulum)
    delta_v: float = 0.3
    m: int = 16  # chain ticks
    n_steps: int = 10  # physics steps per tick
    dt: float = 0.2
    integration_dt: float | None = None
    #: Bins per observation channel. An int or a one-tuple is the single-channel
    #: case every committed run used. `(181, 181)` bins the system's first two
    #: channels and combines them into one token, so the vocabulary is the
    #: product -- the double pendulum, whose theta2 alone hides the driven joint,
    #: is the case this exists for.
    obs_bins: tuple[int, ...] | int = 181

    def __post_init__(self) -> None:
        bins = (self.obs_bins,) if isinstance(self.obs_bins, int) else tuple(self.obs_bins)
        if not bins or any(b < 2 for b in bins):
            raise ValueError(f"every channel needs at least 2 bins, got {self.obs_bins}")
        available = len(getattr(self.system, "obs_ranges", (self.system.obs_range,)))
        if len(bins) > available:
            raise ValueError(
                f"{type(self.system).__name__} exposes {available} observation channels, "
                f"but {len(bins)} bin counts were given"
            )
        object.__setattr__(self, "obs_bins", bins)
        integration_dt = self.dt if self.integration_dt is None else self.integration_dt
        ratio = self.dt / integration_dt
        if integration_dt <= 0 or not np.isclose(ratio, round(ratio)):
            raise ValueError("dt must be a positive integer multiple of integration_dt")
        object.__setattr__(self, "integration_dt", float(integration_dt))

    def flow(self, z: np.ndarray) -> np.ndarray:
        """Advance one physics sampling gap using the configured internal step."""
        return self.system.flow(z, self.dt, substeps=round(self.dt / self.integration_dt))

    @property
    def actions(self) -> np.ndarray:
        actions = np.asarray(self.system.actions(self.delta_v), dtype=np.float64)
        if actions.ndim != 2 or actions.shape[0] != self.chain.n_states:
            raise ValueError(
                f"system returned actions with shape {actions.shape}; expected "
                f"({self.chain.n_states}, action_dim)"
            )
        if np.any(np.linalg.norm(actions, axis=1) == 0.0):
            raise ValueError("every HMM letter must have a non-zero physical action")
        if len(np.unique(actions, axis=0)) != len(actions):
            raise ValueError("every HMM letter must have a distinct physical action")
        return actions

    @property
    def seq_len(self) -> int:
        return self.m * self.n_steps

    @property
    def n_obs(self) -> int:
        """Vocabulary size: the product over channels, since one token names a cell."""
        return int(np.prod(self.obs_bins))

    @property
    def obs_ranges(self) -> tuple[tuple[float, float], ...]:
        ranges = getattr(self.system, "obs_ranges", None) or (self.system.obs_range,)
        return tuple(ranges[: len(self.obs_bins)])

    @property
    def channel_names(self) -> tuple[str, ...]:
        names = getattr(self.system, "observable_names", None) or ("observable",)
        return tuple(names[: len(self.obs_bins)])

    def channels(self, z: np.ndarray) -> np.ndarray:
        """The observed channels of a state, shape (..., len(obs_bins))."""
        if len(self.obs_bins) == 1:
            return self.system.observable(z)[..., None]
        return self.system.observables(z)[..., : len(self.obs_bins)]

    def bin_indices(self, values: np.ndarray) -> np.ndarray:
        """Per-channel bin index, shape (..., len(obs_bins))."""
        out = np.empty(values.shape, dtype=np.int64)
        for channel, ((lo, hi), bins) in enumerate(zip(self.obs_ranges, self.obs_bins)):
            scaled = (values[..., channel] - lo) / (hi - lo) * (bins - 1)
            out[..., channel] = np.clip(np.rint(scaled), 0, bins - 1)
        return out

    def observe(self, z: np.ndarray) -> np.ndarray:
        """State to token, in one call, so every caller bins identically."""
        return self.discretise(self.channels(z))

    def discretise(self, value: np.ndarray) -> np.ndarray:
        """Channel values to one token, uniform bins, rounded and clipped.

        Rounding rather than flooring means the pendulum with range +-pi/2 and
        181 bins maps theta to the nearest whole degree and clamps it to +-90.

        Several channels are combined in mixed radix, so one token names one cell
        of the product grid and the vocabulary is the product of the bin counts.
        A trailing channel axis is optional: a bare scalar array is read as the
        single-channel case, which is what every committed run passes.
        """
        values = np.asarray(value, dtype=np.float64)
        channels = len(self.obs_bins)
        if channels == 1:
            # A bare array of scalars is the ordinary call. `channels()` returns
            # the same values with an explicit trailing 1-axis, and a length-1
            # trailing axis cannot mean anything else here, so both are accepted.
            if values.ndim < 2 or values.shape[-1] != 1:
                values = values[..., None]
        elif values.shape[-1:] != (channels,):
            raise ValueError(
                f"expected a trailing axis of {channels} channels, got shape {values.shape}"
            )
        index = self.bin_indices(values)
        token = np.zeros(values.shape[:-1], dtype=np.int64)
        for channel, bins in enumerate(self.obs_bins):
            token = token * bins + index[..., channel]
        return token

    def tokens_to_indices(self, token: np.ndarray) -> np.ndarray:
        """Unpack integer tokens back to per-channel bin indices, shape (..., len(obs_bins))."""
        tokens = np.asarray(token, dtype=np.int64)
        indices = np.empty((*tokens.shape, len(self.obs_bins)), dtype=np.int64)
        rem = tokens.copy()
        for channel in reversed(range(len(self.obs_bins))):
            bins = self.obs_bins[channel]
            indices[..., channel] = rem % bins
            rem = rem // bins
        return indices

    def undiscretise(self, token: np.ndarray) -> np.ndarray:
        """Reconstruct continuous physical values from integer tokens.

        Returns an array of shape (..., len(obs_bins)) in the physical units of each channel.
        """
        indices = self.tokens_to_indices(token)
        out = np.empty(indices.shape, dtype=np.float64)
        for channel, ((lo, hi), bins) in enumerate(zip(self.obs_ranges, self.obs_bins)):
            denom = max(bins - 1, 1)
            out[..., channel] = lo + indices[..., channel] / denom * (hi - lo)
        return out

    def sample_batch(
        self,
        rng: np.random.Generator,
        n: int,
        initial_state: np.ndarray | None = None,
    ) -> dict:
        """tokens (n, L), beliefs (n, L, K), metric (n, L), moods (n, L)."""
        states, letters = self.chain.sample(rng, n, self.m)
        tick_beliefs = self.chain.beliefs(letters)

        sys_, actions = self.system, self.actions
        expected = sys_.initial_state(n)
        z = expected if initial_state is None else np.asarray(initial_state, dtype=np.float64)
        if z.shape != expected.shape or not np.isfinite(z).all():
            raise ValueError(
                f"initial_state must be finite with shape {expected.shape}, got {z.shape}"
            )
        z = z.copy()
        obs = np.empty((n, self.seq_len, len(self.obs_bins)))
        metric = np.empty((n, self.seq_len, len(sys_.metric_names)))
        for t in range(self.m):
            # The impulse lands once per tick, before that tick's steps.
            z = sys_.kick(z, actions[letters[:, t]])
            for s in range(self.n_steps):
                z = self.flow(z)
                obs[:, t * self.n_steps + s] = self.channels(z)
                metric[:, t * self.n_steps + s] = sys_.metric(z)

        return {
            "tokens": self.discretise(obs),
            "observable": obs,
            "metric": metric,
            "letters": letters,
            "states": states,
            # the chain advances once per tick, so its belief is constant inside that tick
            "beliefs": np.repeat(tick_beliefs, self.n_steps, axis=1),
            # the mood the belief is about, so probe target and belief line up
            "moods": np.repeat(states[:, 1:], self.n_steps, axis=1),
            "emitting_moods": np.repeat(states[:, :-1], self.n_steps, axis=1),
        }

    def rollout(self, letters: np.ndarray) -> dict:
        """Replay an explicit letter sequence, with no chain sampling.

        Every system here is deterministic given its letters: fixed initial
        state, no process noise. So a letter sequence names exactly one
        trajectory, which is what makes a counterfactual well posed -- change one
        letter, replay, and the difference is that letter's whole causal effect
        rather than a difference of two random draws.
        """
        letters = np.atleast_2d(np.asarray(letters, dtype=np.int64))
        if letters.shape[1] != self.m:
            raise ValueError(f"expected {self.m} letters per row, got {letters.shape[1]}")
        n = len(letters)
        z = self.system.initial_state(n)
        obs = np.empty((n, self.seq_len, len(self.obs_bins)))
        metric = np.empty((n, self.seq_len, len(self.system.metric_names)))
        energy = np.empty((n, self.seq_len))
        for t in range(self.m):
            z = self.system.kick(z, self.actions[letters[:, t]])
            for s in range(self.n_steps):
                z = self.flow(z)
                at = t * self.n_steps + s
                obs[:, at] = self.channels(z)
                metric[:, at] = self.system.metric(z)
                energy[:, at] = self.system.energy(z)
        return {
            "tokens": self.discretise(obs),
            "observable": obs,
            "metric": metric,
            "energy": energy,
            "letters": letters,
        }

    def bin_report(self, observable: np.ndarray) -> dict:
        """How much of the vocabulary these observations actually use.

        `clipped` is the fraction of samples outside the channel's range, which
        land on the first or last bin and lose their value. A run with a high
        clipped fraction is not measuring the physics any more, it is measuring
        the range, so this is checked rather than assumed.

        With several channels, `used_bins` counts occupied cells of the product
        grid. That number falls off a cliff as channels are added -- two
        correlated channels visit a curve through the grid, not the whole of it,
        which is the cost of a product vocabulary and is worth seeing.
        """
        values = np.asarray(observable, dtype=np.float64)
        if values.shape[-1:] != (len(self.obs_bins),):
            values = values[..., None]
        counts = np.bincount(self.discretise(values).reshape(-1), minlength=self.n_obs)
        clipped = np.zeros(values.shape[:-1], dtype=bool)
        for channel, (lo, hi) in enumerate(self.obs_ranges):
            clipped |= (values[..., channel] < lo) | (values[..., channel] > hi)
        return {
            "counts": counts,
            "used_bins": int((counts > 0).sum()),
            "n_obs": self.n_obs,
            "obs_bins": tuple(self.obs_bins),
            "clipped": float(clipped.mean()),
            "channels": self.channel_names,
            "obs_range": tuple(self.obs_ranges[0]),
            "obs_ranges": tuple(self.obs_ranges),
            "observed_range": (float(values[..., 0].min()), float(values[..., 0].max())),
            "per_channel_observed_range": tuple(
                (float(values[..., c].min()), float(values[..., c].max()))
                for c in range(len(self.obs_bins))
            ),
            "per_channel_clipped": tuple(
                float(np.mean((values[..., c] < lo) | (values[..., c] > hi)))
                for c, (lo, hi) in enumerate(self.obs_ranges)
            ),
        }

    def suggested_obs_range(self, observable: np.ndarray, quantile: float = 0.001) -> tuple[float, float]:
        """The range spreading channel 0 across its bins, ignoring the tails."""
        values = np.asarray(observable, dtype=np.float64)
        if values.shape[-1:] == (len(self.obs_bins),):
            values = values[..., 0]
        lo, hi = np.quantile(values, [quantile, 1.0 - quantile])
        pad = 0.02 * (hi - lo)
        return float(lo - pad), float(hi + pad)

    def features_and_groups(self, batch: dict) -> tuple[np.ndarray, dict[str, slice]]:
        """Probe targets, in one matrix, with the column block for each.

        Three of them, and the separation matters. `belief` is represented in
        the K-1 independent coordinates of its regular simplex, rather than K
        redundant probabilities. `mood` is the true state, so a probe that reads
        the belief but not the mood is tracking the posterior rather than the answer.
        `metric` is the system's own physical quantity, kept because it is the
        thing the model plainly needs for its actual job of predicting the next
        observation.
        """
        k = self.chain.n_states
        belief_dim = k - 1
        belief_coords = batch["beliefs"] @ simplex_embedding(k)
        feats = np.concatenate(
            [
                belief_coords,
                np.eye(k, dtype=np.float64)[batch["moods"]],
                batch["metric"],
            ],
            axis=-1,
        ).astype(np.float32)
        metric_dim = batch["metric"].shape[-1]
        groups = {
            "belief": slice(0, belief_dim),
            "mood": slice(belief_dim, belief_dim + k),
            "metric": slice(belief_dim + k, belief_dim + k + metric_dim),
        }
        return feats, groups


def token_window_features(
    tokens: np.ndarray, n_obs: int, window: int, steps_per_segment: int | None = None
) -> np.ndarray:
    """One-hot of the last `window` tokens at each position, plus optional segment phase."""
    n, L = tokens.shape
    extra = steps_per_segment or 0
    out = np.zeros((n, L, window * n_obs + extra), dtype=np.float32)
    rows, cols = np.arange(n)[:, None], np.arange(L)[None, :]
    for w in range(window):
        past = np.clip(cols - w, 0, None)
        out[rows, cols, w * n_obs + tokens[rows, past]] = 1.0
        out[:, :w, w * n_obs : (w + 1) * n_obs] = 0.0
    if steps_per_segment:
        phase = np.arange(L) % steps_per_segment
        out[:, np.arange(L), window * n_obs + phase] = 1.0
    return out


def _demo() -> None:
    m3 = MessKProcess(n_states=3, alpha=0.7, stay=0.7)
    assert np.allclose(m3.T.sum(1), 1) and np.allclose(m3.E.sum(1), 1)
    assert abs(m3.x - 0.15) < 1e-12, "K=3 and stay=0.7 imply x=0.15"

    m4 = MessKProcess(n_states=4)
    assert np.allclose(m4.T.sum(1), 1) and np.allclose(m4.E.sum(1), 1)

    rng = np.random.default_rng(0)
    _, letters = m4.sample(rng, 8, 40)
    b = m4.beliefs(letters)
    assert b.shape == (8, 40, 4)
    assert np.allclose(b.sum(-1), 1), "beliefs must be distributions at every step"

    # Recovering the mood from the belief must beat guessing.
    states, letters = m4.sample(rng, 256, 60)
    acc = (m4.beliefs(letters).argmax(-1) == states[:, 1:]).mean()
    assert acc > 1 / 4, f"belief no better than chance at recovering the mood: {acc}"

    proc = MessDriven(chain=m4)
    batch = proc.sample_batch(rng, 4)
    assert batch["tokens"].shape == (4, proc.seq_len)
    assert batch["tokens"].max() < proc.n_obs and batch["tokens"].min() >= 0
    assert batch["beliefs"].shape == (4, proc.seq_len, 4)
    assert simplex_embedding(3).shape == (3, 2) and simplex_embedding(4).shape == (4, 3)

    feats, groups = proc.features_and_groups(batch)
    assert feats.shape == (4, proc.seq_len, (4 - 1) + 4 + 1)
    assert np.allclose(feats[:, :, groups["belief"]], batch["beliefs"] @ simplex_embedding(4))
    assert np.allclose(feats[:, :, groups["mood"]].sum(-1), 1)

    tw = token_window_features(batch["tokens"], proc.n_obs, window=3)
    assert tw.shape == (4, proc.seq_len, 3 * proc.n_obs)
    assert tw[0, 5].sum() == 3, "three one-hots once the window is full"
    assert tw[0, 0].sum() == 1, "only the current token exists at position 0"
    print(f"messk ok (mood recovery {acc:.2f} vs chance {1/4:.2f}, "
          f"memory {m4.memory_length()} letters)")


if __name__ == "__main__":
    _demo()
