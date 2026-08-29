"""Ball constrained to a sphere under gravity -- a spherical pendulum.
State z = (theta, psi, theta_dot, psi_dot), theta measured from the downward
vertical and psi the azimuth around it.

    theta'' = sin(theta) cos(theta) psi'^2 - (g/L) sin(theta) - gamma theta'
    psi''   = -2 cot(theta) theta' psi'                       - gamma psi'

Gravity is the point. A ball on a *gravity-free* sphere has no restoring force,
so the graded all-positive impulses accumulate and drive it into the pole; with
gravity it is an oscillator, the impulses pump the swing instead of displacing
it permanently, and theta stays inside its turning points on its own rather than
against a clip. The second term of theta'' is the centrifugal support that keeps
a rotating ball from falling to the bottom, and it is why psi' matters to the
observable at all.

The chain's letter arrives as an impulse on theta_dot, the swing rate. Every
letter's impulse is different and non-zero.
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
    gamma: float = 0.35
    rate_max: float = 6.0

    #: The band theta actually visits under the driving impulses, not the full
    #: coordinate range -- otherwise most of the 181 bins would never be used.
    obs_range: tuple[float, float] = (0.05, 1.25)
    metric_name: str = "theta_dot"

    def initial_state(self, n: int) -> np.ndarray:
        """A conical swing: off the bottom, already going round."""
        return np.stack(
            [np.full(n, 0.6), np.zeros(n), np.zeros(n), np.full(n, 2.0)], axis=-1
        )

    def _accel(self, th, dth, dpsi):
        th = np.clip(th, THETA_MIN, THETA_MAX)
        a_th = (
            np.sin(th) * np.cos(th) * dpsi**2
            - (self.g / self.length) * np.sin(th)
            - self.gamma * dth
        )
        # No damping term on psi: friction acts on the swing, while the angular
        # momentum about the vertical is conserved. That is what keeps the
        # centrifugal barrier alive, so theta has a strict lower turning point
        # and never reaches the coordinate singularity at the bottom.
        return a_th, -2.0 * dth * dpsi / np.tan(th)

    def flow(self, z: np.ndarray, dt: float, substeps: int = 1) -> np.ndarray:
        th, psi, dth, dpsi = (z[..., i].copy() for i in range(4))
        h = dt
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

    def kick(self, z: np.ndarray, dv: np.ndarray) -> np.ndarray:
        """Impulse on the swing rate, one magnitude per row."""
        dth = np.clip(z[..., 2] + dv, -self.rate_max, self.rate_max)
        return np.stack([z[..., 0], z[..., 1], dth, z[..., 3]], axis=-1)

    def observable(self, z: np.ndarray) -> np.ndarray:
        return z[..., 0]  # polar angle from the bottom

    def metric(self, z: np.ndarray) -> np.ndarray:
        return z[..., 2]  # swing rate

    def energy(self, z: np.ndarray) -> np.ndarray:
        th, dth, dpsi = z[..., 0], z[..., 2], z[..., 3]
        kinetic = 0.5 * self.length**2 * (dth**2 + (np.sin(th) * dpsi) ** 2)
        return kinetic - self.g * self.length * np.cos(th)


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

    kicked = s.kick(s.initial_state(1), np.array([0.7]))
    assert abs(kicked[0, 2] - 0.7) < 1e-12
    print("sphere ok")


if __name__ == "__main__":
    _demo()
