"""Ball constrained to a sphere under gravity -- a spherical pendulum.
State z = (theta, psi, theta_dot, psi_dot), theta measured from the downward
vertical and psi the azimuth around it.

    theta'' = sin(theta) cos(theta) psi'^2 - (g/L) sin(theta) - gamma theta'
    psi''   = -2 cot(theta) theta' psi' - gamma psi'

Gravity makes the constrained ball an oscillator instead of a freely drifting
geodesic. Theta therefore stays inside physical turning points rather than being
held against a coordinate clip. The first term of theta'' is the centrifugal
support that keeps a rotating ball from falling to the bottom, and it is why
psi' matters to the observable at all.

The four letters respectively kick the local tangent velocity north, south,
west, and east. Every action is non-zero, and opposite directions are paired so
the action set has no directional drift.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

THETA_MIN = 0.05  # rad; cot(theta) is singular at the bottom, and angular
THETA_MAX = np.pi - 0.15  # momentum keeps a real trajectory well clear of it


@dataclass(frozen=True)
class SphereBall:
    g: float = 9.8
    length: float = 1.0
    gamma: float = 0.35  # isotropic sliding friction on the sphere
    rate_max: float = 6.0

    #: The conical release: polar angle off the bottom, and the azimuthal rate
    #: that supplies the angular momentum keeping it off both poles.
    theta0: float = 0.6
    psi_dot0: float = 2.0

    #: The full 0 to 90 degree lower hemisphere mapped to [-1.0, 1.0] in Cartesian coordinates.
    obs_range: tuple[float, float] = (-1.0, 1.0)
    metric_names: tuple[str, ...] = ("v_meridional", "v_azimuthal")
    state_names: tuple[str, ...] = ("theta", "psi", "dtheta", "dpsi")

    def actions(self, scale: float) -> np.ndarray:
        """North/south and west/east impulses in the local tangent plane."""
        return scale * np.array([[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]])

    def initial_state(self, n: int) -> np.ndarray:
        """A conical swing: off the bottom, already going round."""
        return np.stack(
            [np.full(n, self.theta0), np.zeros(n), np.zeros(n), np.full(n, self.psi_dot0)],
            axis=-1,
        )

    def _accel(self, th, dth, dpsi):
        th = np.clip(th, THETA_MIN, THETA_MAX)
        a_th = (
            np.sin(th) * np.cos(th) * dpsi**2
            - (self.g / self.length) * np.sin(th)
            - self.gamma * dth
        )
        # Isotropic physical friction: drag opposes velocity in both meridional
        # and azimuthal directions in the tangent plane.
        a_psi = -2.0 * dth * dpsi / np.tan(th) - self.gamma * dpsi
        return a_th, a_psi

    def flow(self, z: np.ndarray, dt: float, substeps: int = 1) -> np.ndarray:
        th, psi, dth, dpsi = (z[..., i].copy() for i in range(4))
        h = dt / substeps
        for _ in range(substeps):
            k1a, k1b = self._accel(th, dth, dpsi)
            k2a, k2b = self._accel(th + h / 2 * dth, dth + h / 2 * k1a, dpsi + h / 2 * k1b)
            k3a, k3b = self._accel(th + h / 2 * (dth + h / 2 * k1a), dth + h / 2 * k2a, dpsi + h / 2 * k2b)
            k4a, k4b = self._accel(th + h * (dth + h / 2 * k2a), dth + h * k3a, dpsi + h * k3b)
            th_new = th + h * (dth + h / 6 * (k1a + k2a + k3a))
            psi = psi + h * (dpsi + h / 6 * (k1b + k2b + k3b))
            dth = np.clip(dth + h / 6 * (k1a + 2 * k2a + 2 * k3a + k4a), -self.rate_max, self.rate_max)
            dpsi = np.clip(dpsi + h / 6 * (k1b + 2 * k2b + 2 * k3b + k4b), -self.rate_max, self.rate_max)
            th = np.clip(th_new, THETA_MIN, THETA_MAX)
        return np.stack([th, (psi + np.pi) % (2 * np.pi) - np.pi, dth, dpsi], axis=-1)

    def kick(self, z: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Apply a physical two-dimensional tangent impulse to each row."""
        th = z[..., 0]
        dth = np.clip(z[..., 2] + action[..., 0] / self.length, -self.rate_max, self.rate_max)
        radius = self.length * np.clip(np.sin(th), 1e-3, None)
        dpsi = np.clip(z[..., 3] + action[..., 1] / radius, -self.rate_max, self.rate_max)
        return np.stack([th, z[..., 1], dth, dpsi], axis=-1)

    observable_names: tuple[str, ...] = ("x", "y")

    @property
    def obs_ranges(self) -> tuple[tuple[float, float], ...]:
        return (self.obs_range, self.obs_range)

    def observable(self, z: np.ndarray) -> np.ndarray:
        th, psi = z[..., 0], z[..., 1]
        return np.sin(th) * np.cos(psi)

    def observables(self, z: np.ndarray) -> np.ndarray:
        """Project spherical coordinates to 2D Cartesian (x, y) on the horizontal plane."""
        th, psi = z[..., 0], z[..., 1]
        return np.stack([np.sin(th) * np.cos(psi), np.sin(th) * np.sin(psi)], axis=-1)

    def metric(self, z: np.ndarray) -> np.ndarray:
        th, dth, dpsi = z[..., 0], z[..., 2], z[..., 3]
        return np.stack(
            [self.length * dth, self.length * np.sin(th) * dpsi], axis=-1
        )

    def energy(self, z: np.ndarray) -> np.ndarray:
        th, dth, dpsi = z[..., 0], z[..., 2], z[..., 3]
        kinetic = 0.5 * self.length**2 * (dth**2 + (np.sin(th) * dpsi) ** 2)
        return kinetic - self.g * self.length * np.cos(th)

    def state_gap(self, z1: np.ndarray, z2: np.ndarray) -> np.ndarray:
        """Physical distance in the sphere's tangent bundle phase space T(S^2).

        Using physical Cartesian positions and velocities avoids coordinate
        artifacts where azimuthal angular velocity dpsi inflates near the bottom pole.
        """
        th1, psi1, dth1, dpsi1 = z1[..., 0], z1[..., 1], z1[..., 2], z1[..., 3]
        th2, psi2, dth2, dpsi2 = z2[..., 0], z2[..., 1], z2[..., 2], z2[..., 3]
        r1 = self.length * np.stack([np.sin(th1) * np.cos(psi1), np.sin(th1) * np.sin(psi1), -np.cos(th1)], axis=-1)
        r2 = self.length * np.stack([np.sin(th2) * np.cos(psi2), np.sin(th2) * np.sin(psi2), -np.cos(th2)], axis=-1)
        v1 = self.length * np.stack([
            dth1 * np.cos(th1) * np.cos(psi1) - dpsi1 * np.sin(th1) * np.sin(psi1),
            dth1 * np.cos(th1) * np.sin(psi1) + dpsi1 * np.sin(th1) * np.cos(psi1),
            dth1 * np.sin(th1),
        ], axis=-1)
        v2 = self.length * np.stack([
            dth2 * np.cos(th2) * np.cos(psi2) - dpsi2 * np.sin(th2) * np.sin(psi2),
            dth2 * np.cos(th2) * np.sin(psi2) + dpsi2 * np.sin(th2) * np.cos(psi2),
            dth2 * np.sin(th2),
        ], axis=-1)
        return np.sqrt(np.sum((r1 - r2) ** 2, axis=-1) + np.sum((v1 - v2) ** 2, axis=-1))



def _demo() -> None:
    s = SphereBall()
    z = s.initial_state(1)
    e0 = s.energy(z)[0]
    for _ in range(2000):
        z = s.flow(z, 0.004)
    assert s.energy(z)[0] < e0, "friction must remove energy"

    undamped = SphereBall(gamma=0.0)
    z, e0 = undamped.initial_state(1), undamped.energy(undamped.initial_state(1))[0]
    for _ in range(2000):
        z = undamped.flow(z, 0.002)
        assert THETA_MIN < z[0, 0] < THETA_MAX, "angular momentum must keep it off both poles"
    assert abs(undamped.energy(z)[0] - e0) < 5e-2, "undamped energy must not drift"

    # A conical pendulum: at the right psi' the centrifugal term exactly cancels
    # gravity and theta holds steady, which is the check that the two terms of
    # a_th carry the right relative sign and scale.
    th0 = 0.6
    conical = SphereBall(gamma=0.0).flow(
        np.array([[th0, 0.0, 0.0, np.sqrt(9.8 / (1.0 * np.cos(th0)))]]), 0.002, 500
    )
    assert abs(conical[0, 0] - th0) < 1e-3, f"conical swing drifted: {conical[0, 0]} vs {th0}"

    start = s.initial_state(1)
    north = s.kick(start, np.array([[-0.7, 0.0]]))
    east = s.kick(start, np.array([[0.0, 0.7]]))
    assert abs(north[0, 2] + 0.7) < 1e-12
    assert east[0, 3] > start[0, 3]

    obs = s.observables(start)
    assert obs.shape == (1, 2)
    assert abs(obs[0, 0] - np.sin(s.theta0)) < 1e-12
    assert abs(obs[0, 1]) < 1e-12
    assert s.obs_ranges == ((-1.0, 1.0), (-1.0, 1.0))
    print("sphere ok")


if __name__ == "__main__":
    _demo()
