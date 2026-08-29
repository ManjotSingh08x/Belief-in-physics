"""Double pendulum with viscous joint friction. State z = (th1, th2, w1, w2).

Standard Lagrangian double-pendulum accelerations with -gamma_i * w_i added to
each joint's angular acceleration. Chaotic, so it is the case where the angle
sequence carries the least usable trace of the chain -- which is exactly why it
is worth running: if the belief is still recoverable here, it is not an artefact
of an easy observable.

The chain's letter arrives as an impulse on the first joint's angular velocity.
Every letter's impulse is different and non-zero.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DoublePendulum:
    g: float = 9.8
    l1: float = 1.0
    l2: float = 1.0
    m1: float = 1.0
    m2: float = 1.0
    gamma1: float = 0.5
    gamma2: float = 0.5
    omega_max: float = 10.0

    obs_range: tuple[float, float] = (-np.pi, np.pi)
    metric_name: str = "omega1"

    def initial_state(self, n: int) -> np.ndarray:
        return np.stack(
            [np.zeros(n), np.zeros(n), np.ones(n), np.zeros(n)], axis=-1
        )

    def _accel(self, th1, th2, w1, w2):
        g, l1, l2, m1, m2 = self.g, self.l1, self.l2, self.m1, self.m2
        delta = th1 - th2
        den = 2 * m1 + m2 - m2 * np.cos(2 * delta)
        num1 = (
            -g * (2 * m1 + m2) * np.sin(th1)
            - m2 * g * np.sin(th1 - 2 * th2)
            - 2 * np.sin(delta) * m2 * (w2**2 * l2 + w1**2 * l1 * np.cos(delta))
        )
        num2 = 2 * np.sin(delta) * (
            w1**2 * l1 * (m1 + m2) + g * (m1 + m2) * np.cos(th1) + w2**2 * l2 * m2 * np.cos(delta)
        )
        return num1 / (l1 * den) - self.gamma1 * w1, num2 / (l2 * den) - self.gamma2 * w2

    def flow(self, z: np.ndarray, dt: float, substeps: int = 1) -> np.ndarray:
        th1, th2, w1, w2 = (z[..., i].copy() for i in range(4))
        h = dt
        for _ in range(substeps):
            k1a, k1b = self._accel(th1, th2, w1, w2)
            k2t1, k2t2 = w1 + h / 2 * k1a, w2 + h / 2 * k1b
            k2a, k2b = self._accel(th1 + h / 2 * w1, th2 + h / 2 * w2, k2t1, k2t2)
            k3t1, k3t2 = w1 + h / 2 * k2a, w2 + h / 2 * k2b
            k3a, k3b = self._accel(th1 + h / 2 * k2t1, th2 + h / 2 * k2t2, k3t1, k3t2)
            k4t1, k4t2 = w1 + h * k3a, w2 + h * k3b
            k4a, k4b = self._accel(th1 + h * k3t1, th2 + h * k3t2, k4t1, k4t2)
            th1 = th1 + h / 6 * (w1 + 2 * k2t1 + 2 * k3t1 + k4t1)
            th2 = th2 + h / 6 * (w2 + 2 * k2t2 + 2 * k3t2 + k4t2)
            w1 = np.clip(w1 + h / 6 * (k1a + 2 * k2a + 2 * k3a + k4a), -self.omega_max, self.omega_max)
            w2 = np.clip(w2 + h / 6 * (k1b + 2 * k2b + 2 * k3b + k4b), -self.omega_max, self.omega_max)
        return np.stack([th1, th2, w1, w2], axis=-1)

    def kick(self, z: np.ndarray, dv: np.ndarray) -> np.ndarray:
        """Impulse on the first joint, one magnitude per row."""
        w1 = np.clip(z[..., 2] + dv, -self.omega_max, self.omega_max)
        return np.stack([z[..., 0], z[..., 1], w1, z[..., 3]], axis=-1)

    def observable(self, z: np.ndarray) -> np.ndarray:
        """theta2, wrapped -- the far joint, which is the chaotic one."""
        return (z[..., 1] + np.pi) % (2 * np.pi) - np.pi

    def metric(self, z: np.ndarray) -> np.ndarray:
        return z[..., 2]

    def energy(self, z: np.ndarray) -> np.ndarray:
        th1, th2, w1, w2 = (z[..., i] for i in range(4))
        m1, m2, l1, l2, g = self.m1, self.m2, self.l1, self.l2, self.g
        kinetic = 0.5 * m1 * l1**2 * w1**2 + 0.5 * m2 * (
            l1**2 * w1**2 + l2**2 * w2**2 + 2 * l1 * l2 * w1 * w2 * np.cos(th1 - th2)
        )
        potential = -(m1 + m2) * g * l1 * np.cos(th1) - m2 * g * l2 * np.cos(th2)
        return kinetic + potential


def _demo() -> None:
    z0 = np.array([[0.9, -0.4, 0.0, 0.0]])
    undamped = DoublePendulum(gamma1=0.0, gamma2=0.0)
    z, e0 = z0.copy(), undamped.energy(z0)[0]
    for _ in range(2000):
        z = undamped.flow(z, 0.002)
    assert abs(undamped.energy(z)[0] - e0) < 5e-2, "undamped energy must not drift"

    dp = DoublePendulum()
    z, e0 = z0.copy(), dp.energy(z0)[0]
    for _ in range(2000):
        z = dp.flow(z, 0.002)
    assert dp.energy(z)[0] < e0, "damped energy must decrease"

    kicked = dp.kick(np.zeros((1, 4)), np.array([1.2]))
    assert abs(kicked[0, 2] - 1.2) < 1e-12 and kicked[0, 3] == 0.0
    lo, hi = dp.obs_range
    assert lo <= dp.observable(np.array([[0.0, 7.0, 0.0, 0.0]]))[0] <= hi, "theta2 must wrap"
    print("double_pendulum ok")


if __name__ == "__main__":
    _demo()
