"""Mess-K: a K-state chain driving a pendulum, in the style of ManjotSingh08x/Belief-Physics.

This is a deliberate change of approach. In the branch process the hidden thing
was the pendulum's own history and the belief was conditioned on the pendulum's
(blurred) angles. Here the hidden thing is a separate K-state chain sitting in
front of the pendulum, and the belief is conditioned on the chain's OWN letters:

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

K = 3 with alpha=0.7, stay=0.7 reproduces their Mess3 exactly, which is what
makes the classic triangle the reference picture to check against.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .systems.pendulum import Pendulum


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
        """(states, letters), each (n, m). Vectorised over the batch."""
        T_cdf, E_cdf = np.cumsum(self.T, axis=1), np.cumsum(self.E, axis=1)
        states = np.empty((n, m), dtype=np.int64)
        letters = np.empty((n, m), dtype=np.int64)
        s = rng.integers(0, self.n_states, size=n)
        for t in range(m):
            states[:, t] = s
            letters[:, t] = (rng.random((n, 1)) > E_cdf[s]).sum(axis=1)
            s = (rng.random((n, 1)) > T_cdf[s]).sum(axis=1)
        return states, letters

    def beliefs(self, letters: np.ndarray) -> np.ndarray:
        """Exact P(mood_t | letters_1..t), shape (n, m, K). The forward algorithm."""
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
        }


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
    acc = (m4.beliefs(letters).argmax(-1) == states).mean()
    assert acc > 1 / 4, f"belief no better than chance at recovering the mood: {acc}"

    proc = MessPendulum(chain=m4)
    batch = proc.sample_batch(rng, 4)
    assert batch["tokens"].shape == (4, proc.seq_len)
    assert batch["tokens"].max() < proc.n_obs and batch["tokens"].min() >= 0
    assert batch["beliefs"].shape == (4, proc.seq_len, 4)
    print(f"messk ok (mood recovery {acc:.2f} vs chance {1/4:.2f}, "
          f"memory {m4.memory_length()} letters)")


if __name__ == "__main__":
    _demo()
