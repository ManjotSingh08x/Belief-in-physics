"""Mess-K: a K-state chain driving a pendulum.

The hidden thing is a K-state chain sitting in front of the pendulum, and the
belief is conditioned on the chain's OWN letters:

    mood_1 -> mood_2 -> ...          hidden, persists and drifts
      |          |
    letter     letter                hidden from the model; the belief conditions on THESE
      |          |
    shove      shove
              pendulum
                |
             angle bins               the only thing the model ever sees

Two consequences follow, and both are properties of the approach rather than
bugs. The belief does not depend on the physics at all -- delete the pendulum
and it is unchanged. And because the chain is stationary and mixing, the belief
is a contractive function of recent letters, so it forgets its past at a fixed
rate; `memory_length` reports that rate.

K = 3 with alpha=0.7, stay=0.7 is the Mess3 case, whose belief simplex is the
familiar triangle and therefore the reference picture to check a K=4 run against.
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


@dataclass(frozen=True)
class MessKProcess:
    """K moods, K letters. A mood emits its own letter with probability `alpha`.

    Their code parameterises the transition by `x`, the chance of moving to each
    specific other state, so the stay probability is `1 - (K-1)x` and changes
    with K. Parameterising by `stay` instead holds persistence fixed as K grows,
    which is what makes Mess-3 and Mess-4 comparable; `x` is derived.
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
            letters[:, t] = (rng.random((n, 1)) > E_cdf[s]).sum(axis=1)
            s = (rng.random((n, 1)) > T_cdf[s]).sum(axis=1)
        states[:, m] = s
        return states, letters

    def beliefs(self, letters: np.ndarray) -> np.ndarray:
        """Exact P(mood_{t+1} | letters_1..t), shape (n, m, K).

        This is the PREDICTIVE belief, one transition ahead of the mood that
        emitted the last letter, and it is what their `optimal` computes: the
        operator is `T * E[:, letter]` with the emission indexed on the SOURCE
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
class MessPendulum:
    """The chain's letters become velocity impulses on a pendulum.

    Their mapping is all-positive and graded -- letter l becomes (l+1)*delta_v --
    and the observation is theta rounded to whole degrees and clamped to +-90,
    giving 181 bins with no added noise. Both are kept.
    """

    chain: MessKProcess = field(default_factory=MessKProcess)
    delta_v: float = 0.3
    m: int = 16  # chain ticks
    n_steps: int = 10  # physics steps per tick
    dt: float = 0.02
    gamma: float = 0.5
    g: float = 9.8
    length: float = 1.0
    initial_theta: float = 0.0
    initial_omega: float = 1.0
    theta_limit_deg: int = 90

    @property
    def kicks(self) -> tuple[float, ...]:
        return tuple((l + 1) * self.delta_v for l in range(self.chain.n_states))

    @property
    def n_obs(self) -> int:
        return 2 * self.theta_limit_deg + 1

    @property
    def seq_len(self) -> int:
        return self.m * self.n_steps

    @property
    def n_actions(self) -> int:
        return self.chain.n_states

    def _pendulum(self) -> Pendulum:
        return Pendulum(g=self.g, length=self.length, gamma=self.gamma, kicks=self.kicks)

    def discretise(self, theta: np.ndarray) -> np.ndarray:
        deg = np.rint(np.degrees(theta))
        return np.clip(deg, -self.theta_limit_deg, self.theta_limit_deg).astype(np.int64) + self.theta_limit_deg

    def sample_batch(self, rng: np.random.Generator, n: int) -> dict:
        """tokens (n, L), beliefs (n, L, K), letters/states (n, m), omega (n, L)."""
        states, letters = self.chain.sample(rng, n, self.m)
        tick_beliefs = self.chain.beliefs(letters)

        pend = self._pendulum()
        z = np.stack(
            [np.full(n, self.initial_theta), np.full(n, self.initial_omega)], axis=-1
        )
        theta = np.empty((n, self.seq_len))
        omega = np.empty((n, self.seq_len))
        for t in range(self.m):
            # The impulse lands once per tick, before that tick's steps.
            dv = np.array(self.kicks)[letters[:, t]]
            z = np.stack([z[:, 0], z[:, 1] + dv], axis=-1)
            for s in range(self.n_steps):
                z = pend.flow(z, self.dt, 1)
                theta[:, t * self.n_steps + s] = z[:, 0]
                omega[:, t * self.n_steps + s] = z[:, 1]

        return {
            "tokens": self.discretise(theta),
            "theta": theta,
            "omega": omega,
            "letters": letters,
            "states": states,
            # the belief is constant across a tick's steps, as in their pipeline
            "beliefs": np.repeat(tick_beliefs, self.n_steps, axis=1),
            # the mood the belief is about, so probe target and belief line up
            "moods": np.repeat(states[:, 1:], self.n_steps, axis=1),
            "emitting_moods": np.repeat(states[:, :-1], self.n_steps, axis=1),
        }

    def features_and_groups(self, batch: dict) -> tuple[np.ndarray, dict[str, slice]]:
        """Probe targets, in one matrix, with the column block for each.

        Three of them, and the separation matters. `belief` is what their
        approach is about. `mood` is the true state, so a probe that reads the
        belief but not the mood is tracking the posterior rather than the answer.
        `velocity` is the physical quantity, kept because it is the thing the
        model plainly needs for its actual job of predicting the next angle.
        """
        k = self.chain.n_states
        feats = np.concatenate(
            [
                batch["beliefs"],
                np.eye(k, dtype=np.float64)[batch["moods"]],
                batch["omega"][..., None],
            ],
            axis=-1,
        ).astype(np.float32)
        groups = {"belief": slice(0, k), "mood": slice(k, 2 * k), "velocity": slice(2 * k, 2 * k + 1)}
        return feats, groups


def token_window_features(tokens: np.ndarray, n_obs: int, window: int) -> np.ndarray:
    """One-hot of the last `window` tokens at each position, (n, L, window*n_obs).

    The myopic control. Their belief is a contractive function of recent letters,
    so if this predicts the belief as well as the residual stream does, then
    "the model encodes the belief" and "the model remembers its recent input"
    are the same statement and the first one claims nothing.
    """
    n, L = tokens.shape
    out = np.zeros((n, L, window * n_obs), dtype=np.float32)
    rows, cols = np.arange(n)[:, None], np.arange(L)[None, :]
    for w in range(window):
        past = np.clip(cols - w, 0, None)
        out[rows, cols, w * n_obs + tokens[rows, past]] = 1.0
        out[:, :w, w * n_obs : (w + 1) * n_obs] = 0.0  # nothing that far back yet
    return out


def _demo() -> None:
    m3 = MessKProcess(n_states=3, alpha=0.7, stay=0.7)
    assert np.allclose(m3.T.sum(1), 1) and np.allclose(m3.E.sum(1), 1)
    assert abs(m3.x - 0.15) < 1e-12, "K=3, stay=0.7 must reproduce their x=0.15"

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

    proc = MessPendulum(chain=m4)
    batch = proc.sample_batch(rng, 4)
    assert batch["tokens"].shape == (4, proc.seq_len)
    assert batch["tokens"].max() < proc.n_obs and batch["tokens"].min() >= 0
    assert batch["beliefs"].shape == (4, proc.seq_len, 4)
    assert simplex_embedding(3).shape == (3, 2) and simplex_embedding(4).shape == (4, 3)

    feats, groups = proc.features_and_groups(batch)
    assert feats.shape == (4, proc.seq_len, 2 * 4 + 1)
    assert np.allclose(feats[:, :, groups["belief"]].sum(-1), 1)
    assert np.allclose(feats[:, :, groups["mood"]].sum(-1), 1)

    tw = token_window_features(batch["tokens"], proc.n_obs, window=3)
    assert tw.shape == (4, proc.seq_len, 3 * proc.n_obs)
    assert tw[0, 5].sum() == 3, "three one-hots once the window is full"
    assert tw[0, 0].sum() == 1, "only the current token exists at position 0"
    print(f"messk ok (mood recovery {acc:.2f} vs chance {1/4:.2f}, "
          f"memory {m4.memory_length()} letters)")


if __name__ == "__main__":
    _demo()
