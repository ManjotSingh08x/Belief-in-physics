"""Damped pendulum. State z = (theta, omega).

theta'' = -(g/L) sin(theta) - gamma * omega     (viscous damping)

The chain's letter arrives as a velocity impulse, so `kick` adds to omega. Every
letter carries a different, non-zero impulse, so there is no action that leaves
the state untouched.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Pendulum:
    g: float = 9.8
    length: float = 1.0
    gamma: float = 1.2  # viscous damping on omega
    omega_max: float = 8.0

    #: Release condition. Every sequence starts here, so this is the state the
    #: whole dataset is a function of, together with the letters.
    theta0: float = 0.0
    omega0: float = 1.0

    #: (lo, hi) of the binned observable, and the name of the metric column.
    obs_range: tuple[float, float] = (-np.pi / 2, np.pi / 2)
    metric_names: tuple[str, ...] = ("omega",)
    state_names: tuple[str, ...] = ("theta", "omega")

    def actions(self, scale: float) -> np.ndarray:
        """Balanced weak/strong angular-velocity kicks, with no zero action."""
        return scale * np.array([[-3.0], [-1.0], [1.0], [3.0]])

    def initial_state(self, n: int) -> np.ndarray:
        return np.stack([np.full(n, self.theta0), np.full(n, self.omega0)], axis=-1)

    def _accel(self, theta: np.ndarray, omega: np.ndarray) -> np.ndarray:
        return -(self.g / self.length) * np.sin(theta) - self.gamma * omega

    def flow(self, z: np.ndarray, dt: float, substeps: int = 1) -> np.ndarray:
        theta, omega = z[..., 0].copy(), z[..., 1].copy()
        h = dt / substeps
        for _ in range(substeps):
            k1_th, k1_om = omega, self._accel(theta, omega)
            k2_th, k2_om = omega + h / 2 * k1_om, self._accel(theta + h / 2 * k1_th, omega + h / 2 * k1_om)
            k3_th, k3_om = omega + h / 2 * k2_om, self._accel(theta + h / 2 * k2_th, omega + h / 2 * k2_om)
            k4_th, k4_om = omega + h * k3_om, self._accel(theta + h * k3_th, omega + h * k3_om)
            theta = theta + h / 6 * (k1_th + 2 * k2_th + 2 * k3_th + k4_th)
            omega = omega + h / 6 * (k1_om + 2 * k2_om + 2 * k3_om + k4_om)
            omega = np.clip(omega, -self.omega_max, self.omega_max)
        return np.stack([theta, omega], axis=-1)

    def kick(self, z: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Angular-velocity impulse, one one-dimensional action per row."""
        omega = np.clip(z[..., 1] + action[..., 0], -self.omega_max, self.omega_max)
        return np.stack([z[..., 0], omega], axis=-1)

    #: Channel 0 is the one every committed run used. Later channels are opt-in:
    #: `MessDriven.obs_bins` decides how many are actually binned into the token.
    observable_names: tuple[str, ...] = ("theta", "omega")

    @property
    def obs_ranges(self) -> tuple[tuple[float, float], ...]:
        return (self.obs_range, (-self.omega_max, self.omega_max))

    def observable(self, z: np.ndarray) -> np.ndarray:
        return z[..., 0]

    def observables(self, z: np.ndarray) -> np.ndarray:
        return np.stack([z[..., 0], z[..., 1]], axis=-1)

    def metric(self, z: np.ndarray) -> np.ndarray:
        return z[..., 1:2]

    def energy(self, z: np.ndarray) -> np.ndarray:
        return 0.5 * z[..., 1] ** 2 - (self.g / self.length) * np.cos(z[..., 0])


def _demo() -> None:
    z0 = np.array([[0.3, 0.0]])
    undamped = Pendulum(gamma=0.0)
    z, e0 = z0.copy(), undamped.energy(z0)[0]
    for _ in range(2000):
        z = undamped.flow(z, 0.01)
    assert abs(undamped.energy(z)[0] - e0) < 1e-3, "undamped energy must not drift"

    p = Pendulum()
    z, e0 = z0.copy(), p.energy(z0)[0]
    for _ in range(2000):
        z = p.flow(z, 0.01)
    assert p.energy(z)[0] < e0, "damped energy must decrease"

    kicked = p.kick(z0, np.array([[0.6]]))
    assert abs(kicked[0, 1] - 0.6) < 1e-12 and kicked[0, 0] == z0[0, 0]
    assert p.initial_state(5).shape == (5, 2)
    print("pendulum ok")


if __name__ == "__main__":
    _demo()
