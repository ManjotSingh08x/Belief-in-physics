"""Double pendulum with viscous joint friction. State z = (theta1, theta2, omega1, omega2).
Standard Lagrangian double-pendulum accelerations plus -gamma_i*omega_i damping
added directly to each joint's angular acceleration.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class DoublePendulum:
    g: float = 9.81
    l1: float = 1.0
    l2: float = 1.0
    m1: float = 1.0
    m2: float = 1.0
    gamma1: float = 0.08
    gamma2: float = 0.08
    kick: float = 1.2
    omega_max: float = 10.0

    state_dim: int = field(default=4, init=False)
    n_actions: int = field(default=4, init=False)
    dims_periodic: tuple[bool, bool, bool, bool] = field(default=(True, True, False, False), init=False)

    @property
    def domain(self):
        return (
            (-np.pi, np.pi),
            (-np.pi, np.pi),
            (-self.omega_max, self.omega_max),
            (-self.omega_max, self.omega_max),
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
        a1 = num1 / (l1 * den) - self.gamma1 * w1
        num2 = 2 * np.sin(delta) * (
            w1**2 * l1 * (m1 + m2) + g * (m1 + m2) * np.cos(th1) + w2**2 * l2 * m2 * np.cos(delta)
        )
        a2 = num2 / (l2 * den) - self.gamma2 * w2
        return a1, a2

    def flow(self, z: np.ndarray, dt: float, substeps: int) -> np.ndarray:
        th1, th2, w1, w2 = (z[..., i].copy() for i in range(4))
        h = dt
        for _ in range(substeps):
            k1a, k1b = self._accel(th1, th2, w1, w2)
            k2t1, k2t2 = w1 + h / 2 * k1a, w2 + h / 2 * k1b
            k2a, k2b = self._accel(th1 + h / 2 * w1, th2 + h / 2 * w2, w1 + h / 2 * k1a, w2 + h / 2 * k1b)
            k3t1, k3t2 = w1 + h / 2 * k2a, w2 + h / 2 * k2b
            k3a, k3b = self._accel(th1 + h / 2 * k2t1, th2 + h / 2 * k2t2, w1 + h / 2 * k2a, w2 + h / 2 * k2b)
            k4t1, k4t2 = w1 + h * k3a, w2 + h * k3b
            k4a, k4b = self._accel(th1 + h * k3t1, th2 + h * k3t2, w1 + h * k3a, w2 + h * k3b)
            th1_new = th1 + h / 6 * (w1 + 2 * k2t1 + 2 * k3t1 + k4t1)
            th2_new = th2 + h / 6 * (w2 + 2 * k2t2 + 2 * k3t2 + k4t2)
            w1_new = w1 + h / 6 * (k1a + 2 * k2a + 2 * k3a + k4a)
            w2_new = w2 + h / 6 * (k1b + 2 * k2b + 2 * k3b + k4b)
            th1, th2 = th1_new, th2_new
            w1 = np.clip(w1_new, -self.omega_max, self.omega_max)
            w2 = np.clip(w2_new, -self.omega_max, self.omega_max)
        return np.stack([th1, th2, w1, w2], axis=-1)

    def apply_action(self, z: np.ndarray, action: int) -> np.ndarray:
        th1, th2, w1, w2 = z[..., 0], z[..., 1], z[..., 2], z[..., 3]
        d1, d2 = {0: (self.kick, 0.0), 1: (0.0, self.kick), 2: (self.kick, self.kick), 3: (0.0, 0.0)}[action]
        new_w1 = np.clip(w1 + d1, -self.omega_max, self.omega_max)
        new_w2 = np.clip(w2 + d2, -self.omega_max, self.omega_max)
        return np.stack([th1, th2, new_w1, new_w2], axis=-1)

    def energy(self, z: np.ndarray) -> np.ndarray:
        th1, th2, w1, w2 = z[..., 0], z[..., 1], z[..., 2], z[..., 3]
        m1, m2, l1, l2, g = self.m1, self.m2, self.l1, self.l2, self.g
        T = 0.5 * m1 * l1**2 * w1**2 + 0.5 * m2 * (
            l1**2 * w1**2 + l2**2 * w2**2 + 2 * l1 * l2 * w1 * w2 * np.cos(th1 - th2)
        )
        V = -(m1 + m2) * g * l1 * np.cos(th1) - m2 * g * l2 * np.cos(th2)
        return T + V

    def observable(self, z: np.ndarray) -> np.ndarray:
        return z[..., 1]  # theta2

    def metrics(self, z: np.ndarray) -> np.ndarray:
        return z[..., 2:4]  # (omega1, omega2)


def _demo() -> None:
    dp_undamped = DoublePendulum(gamma1=0.0, gamma2=0.0)
    z = np.array([[0.9, -0.4, 0.0, 0.0]])
    e0 = dp_undamped.energy(z)[0]
    for _ in range(2000):
        z = dp_undamped.flow(z, 0.002, 1)
    e1 = dp_undamped.energy(z)[0]
    assert abs(e1 - e0) < 5e-2, f"undamped double pendulum energy drifted: {e0} -> {e1}"

    dp = DoublePendulum()
    z = np.array([[0.9, -0.4, 0.0, 0.0]])
    e0 = dp.energy(z)[0]
    for _ in range(2000):
        z = dp.flow(z, 0.002, 1)
    e1 = dp.energy(z)[0]
    assert e1 < e0, "damped double pendulum energy must decrease"

    kicked = dp.apply_action(np.zeros((1, 4)), 2)
    assert kicked[0, 2] == dp.kick and kicked[0, 3] == dp.kick
    print("double_pendulum ok")


if __name__ == "__main__":
    _demo()
