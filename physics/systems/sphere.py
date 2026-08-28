"""Ball sliding on a unit sphere. State z = (lat, lon, u, w) with u=dlat/dt,
w=dlon/dt. Physical tangential speed components are v1=u (northward) and
v2=cos(lat)*w (eastward) -- these are what `metrics` reports as "velocity
vector of the ball".

Geodesic equations (Christoffel symbols of S^2) plus sliding friction
opposing physical velocity, -mu*|v|*v, derived so that it reduces cleanly
to two extra terms on the coordinate accelerations:

    u' = -cos(lat) sin(lat) w^2 - mu*|v|*u
    w' =  2 tan(lat) u w        - mu*|v|*w
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

POLE_MARGIN = 0.12  # rad kept clear of +-pi/2, avoids the cos(lat)=0 singularity


@dataclass(frozen=True)
class SphereBall:
    mu: float = 0.25
    kick: float = 1.2
    rate_max: float = 6.0

    state_dim: int = field(default=4, init=False)
    n_actions: int = field(default=4, init=False)
    dims_periodic: tuple[bool, bool, bool, bool] = field(default=(False, True, False, False), init=False)

    @property
    def lat_max(self) -> float:
        return np.pi / 2 - POLE_MARGIN

    @property
    def domain(self):
        return (
            (-self.lat_max, self.lat_max),
            (-np.pi, np.pi),
            (-self.rate_max, self.rate_max),
            (-self.rate_max, self.rate_max),
        )

    def _accel(self, phi, lam, u, w):
        v1, v2 = u, np.cos(phi) * w
        speed = np.sqrt(v1**2 + v2**2)
        u_dot = -np.cos(phi) * np.sin(phi) * w**2 - self.mu * speed * u
        w_dot = 2 * np.tan(phi) * u * w - self.mu * speed * w
        return u_dot, w_dot

    def flow(self, z: np.ndarray, dt: float, substeps: int) -> np.ndarray:
        phi, lam, u, w = (z[..., i].copy() for i in range(4))
        h = dt
        for _ in range(substeps):
            k1u_d, k1w_d = self._accel(phi, lam, u, w)
            k2phi, k2lam = u + h / 2 * k1u_d, w + h / 2 * k1w_d
            k2u_d, k2w_d = self._accel(phi + h / 2 * u, lam + h / 2 * w, u + h / 2 * k1u_d, w + h / 2 * k1w_d)
            k3phi, k3lam = u + h / 2 * k2u_d, w + h / 2 * k2w_d
            k3u_d, k3w_d = self._accel(
                phi + h / 2 * k2phi, lam + h / 2 * k2lam, u + h / 2 * k2u_d, w + h / 2 * k2w_d
            )
            k4phi, k4lam = u + h * k3u_d, w + h * k3w_d
            k4u_d, k4w_d = self._accel(phi + h * k3phi, lam + h * k3lam, u + h * k3u_d, w + h * k3w_d)
            phi_new = phi + h / 6 * (u + 2 * k2phi + 2 * k3phi + k4phi)
            lam_new = lam + h / 6 * (w + 2 * k2lam + 2 * k3lam + k4lam)
            u_new = u + h / 6 * (k1u_d + 2 * k2u_d + 2 * k3u_d + k4u_d)
            w_new = w + h / 6 * (k1w_d + 2 * k2w_d + 2 * k3w_d + k4w_d)
            phi = np.clip(phi_new, -self.lat_max, self.lat_max)
            lam = lam_new
            u = np.clip(u_new, -self.rate_max, self.rate_max)
            w = np.clip(w_new, -self.rate_max, self.rate_max)
        return np.stack([phi, lam, u, w], axis=-1)

    def apply_action(self, z: np.ndarray, action: int) -> np.ndarray:
        phi, lam, u, w = z[..., 0], z[..., 1], z[..., 2], z[..., 3]
        if action == 3:
            return np.stack([phi, lam, u, w], axis=-1)
        theta_k = action * (2 * np.pi / 3)
        v1 = u + self.kick * np.cos(theta_k)
        v2 = np.cos(phi) * w + self.kick * np.sin(theta_k)
        new_u = np.clip(v1, -self.rate_max, self.rate_max)
        new_w = np.clip(v2 / np.clip(np.cos(phi), 1e-3, None), -self.rate_max, self.rate_max)
        return np.stack([phi, lam, new_u, new_w], axis=-1)

    def energy(self, z: np.ndarray) -> np.ndarray:
        phi, u, w = z[..., 0], z[..., 2], z[..., 3]
        v1, v2 = u, np.cos(phi) * w
        return 0.5 * (v1**2 + v2**2)

    def observable(self, z: np.ndarray) -> np.ndarray:
        return z[..., :2]  # (lat, lon)

    def metrics(self, z: np.ndarray) -> np.ndarray:
        phi, u, w = z[..., 0], z[..., 2], z[..., 3]
        return np.stack([u, np.cos(phi) * w], axis=-1)


def _demo() -> None:
    s = SphereBall()
    z = np.array([[0.2, 0.0, 1.0, 0.8]])
    e0 = s.energy(z)[0]
    for _ in range(500):
        z = s.flow(z, 0.01, 1)
    e1 = s.energy(z)[0]
    assert e1 < e0, f"sliding friction must decrease speed: {e0} -> {e1}"

    # Short, slow trajectory that stays well clear of the pole-margin clip
    # (a real geodesic's speed is conserved only while it isn't hitting that
    # artificial boundary of this (lat,lon) chart).
    s0 = SphereBall(mu=0.0)
    z0 = np.array([[0.1, 0.0, 0.3, 0.3]])
    e00 = s0.energy(z0)[0]
    for _ in range(200):
        z0 = s0.flow(z0, 0.005, 1)
        assert abs(z0[0, 0]) < s0.lat_max - 1e-6, "trajectory hit the pole clip; shrink test speed/duration"
    e01 = s0.energy(z0)[0]
    assert abs(e01 - e00) < 1e-3, f"frictionless geodesic speed must be conserved: {e00} -> {e01}"

    kicked = s.apply_action(np.array([[0.0, 0.0, 0.0, 0.0]]), 0)
    assert abs(kicked[0, 2] - s.kick) < 1e-9
    print("sphere ok")


if __name__ == "__main__":
    _demo()
