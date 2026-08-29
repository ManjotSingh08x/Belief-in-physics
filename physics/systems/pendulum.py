"""Damped pendulum. State z = (theta, omega). theta periodic, omega bounded.

theta'' = -(g/L) sin(theta) - gamma * omega     (viscous damping)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Pendulum:
    g: float = 9.81
    length: float = 1.0
    gamma: float = 0.15
    kick: float = 1.5
    omega_max: float = 8.0
    kicks: tuple[float, ...] | None = None  # None -> the legacy (-kick, 0, +kick) set

    state_dim: int = field(default=2, init=False)
    dims_periodic: tuple[bool, bool] = field(default=(True, False), init=False)

    @property
    def dv_table(self) -> tuple[float, ...]:
        """Velocity increment per action.

        The legacy set carries a no-op, which is the exact midpoint of the other
        two and therefore invisible to `metrics` (= omega): E[omega] moves with
        p(+dv) - p(-dv) and does not depend on p(noop) at all. Passing `kicks`
        replaces it with an arbitrary ladder so that every action displaces the
        state.
        """
        return (-self.kick, 0.0, self.kick) if self.kicks is None else self.kicks

    @property
    def n_actions(self) -> int:
        return len(self.dv_table)

    @property
    def domain(self) -> tuple[tuple[float, float], tuple[float, float]]:
        return ((-np.pi, np.pi), (-self.omega_max, self.omega_max))

    def _accel(self, theta: np.ndarray, omega: np.ndarray) -> np.ndarray:
        return -(self.g / self.length) * np.sin(theta) - self.gamma * omega

    def flow(self, z: np.ndarray, dt: float, substeps: int) -> np.ndarray:
        theta, omega = z[..., 0].copy(), z[..., 1].copy()
        h = dt
        for _ in range(substeps):
            k1_th, k1_om = omega, self._accel(theta, omega)
            k2_th, k2_om = omega + h / 2 * k1_om, self._accel(theta + h / 2 * k1_th, omega + h / 2 * k1_om)
            k3_th, k3_om = omega + h / 2 * k2_om, self._accel(theta + h / 2 * k2_th, omega + h / 2 * k2_om)
            k4_th, k4_om = omega + h * k3_om, self._accel(theta + h * k3_th, omega + h * k3_om)
            theta = theta + h / 6 * (k1_th + 2 * k2_th + 2 * k3_th + k4_th)
            omega = omega + h / 6 * (k1_om + 2 * k2_om + 2 * k3_om + k4_om)
            omega = np.clip(omega, -self.omega_max, self.omega_max)
        return np.stack([theta, omega], axis=-1)

    def apply_action(self, z: np.ndarray, action: int) -> np.ndarray:
        theta, omega = z[..., 0], z[..., 1]
        dv = self.dv_table[action]
        new_omega = np.clip(omega + dv, -self.omega_max, self.omega_max)
        return np.stack([theta, new_omega], axis=-1)

    def energy(self, z: np.ndarray) -> np.ndarray:
        theta, omega = z[..., 0], z[..., 1]
        return 0.5 * omega**2 - (self.g / self.length) * np.cos(theta)

    def observable(self, z: np.ndarray) -> np.ndarray:
        return z[..., 0]

    def metrics(self, z: np.ndarray) -> np.ndarray:
        return z[..., 1:2]


def _demo() -> None:
    p = Pendulum()
    z0 = np.array([[0.3, 0.0]])
    # undamped check: zero out gamma via a fresh instance
    p_undamped = Pendulum(gamma=0.0)
    z = z0.copy()
    e0 = p_undamped.energy(z)[0]
    for _ in range(2000):
        z = p_undamped.flow(z, 0.01, 1)
    e1 = p_undamped.energy(z)[0]
    assert abs(e1 - e0) < 1e-3, f"undamped energy drifted: {e0} -> {e1}"

    z = z0.copy()
    e0 = p.energy(z)[0]
    for _ in range(2000):
        z = p.flow(z, 0.01, 1)
    e1 = p.energy(z)[0]
    assert e1 < e0, "damped energy must decrease"

    kicked = p.apply_action(z0, 2)
    assert kicked[0, 1] == p.kick
    print("pendulum ok")


if __name__ == "__main__":
    _demo()
