"""Predator-prey with logistic self-limitation on prey. State z = (log x, log y).

dx/dt = a*x - b*x*y - a*x^2/kappa    (the -a*x^2/kappa term is the damping:
                                       implicit, no explicit friction term --
                                       it makes the spiral converge to a fixed
                                       point instead of the neutral closed
                                       orbits of plain Lotka-Volterra)
dy/dt = d*x*y - c*y
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class PredatorPrey:
    a: float = 1.0
    b: float = 0.6
    c: float = 0.8
    d: float = 0.4
    kappa: float = 5.0
    kick: float = 0.35
    log_bound: float = 3.5

    state_dim: int = field(default=2, init=False)
    n_actions: int = field(default=5, init=False)
    dims_periodic: tuple[bool, bool] = field(default=(False, False), init=False)

    @property
    def domain(self) -> tuple[tuple[float, float], tuple[float, float]]:
        return ((-self.log_bound, self.log_bound), (-self.log_bound, self.log_bound))

    def _rhs(self, lx: np.ndarray, ly: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """d(log x)/dt, d(log y)/dt."""
        x, y = np.exp(lx), np.exp(ly)
        dlx = self.a - self.b * y - self.a * x / self.kappa
        dly = self.d * x - self.c
        return dlx, dly

    def flow(self, z: np.ndarray, dt: float, substeps: int) -> np.ndarray:
        lx, ly = z[..., 0].copy(), z[..., 1].copy()
        h = dt
        for _ in range(substeps):
            k1x, k1y = self._rhs(lx, ly)
            k2x, k2y = self._rhs(lx + h / 2 * k1x, ly + h / 2 * k1y)
            k3x, k3y = self._rhs(lx + h / 2 * k2x, ly + h / 2 * k2y)
            k4x, k4y = self._rhs(lx + h * k3x, ly + h * k3y)
            lx = lx + h / 6 * (k1x + 2 * k2x + 2 * k3x + k4x)
            ly = ly + h / 6 * (k1y + 2 * k2y + 2 * k3y + k4y)
            lx = np.clip(lx, -self.log_bound, self.log_bound)
            ly = np.clip(ly, -self.log_bound, self.log_bound)
        return np.stack([lx, ly], axis=-1)

    def apply_action(self, z: np.ndarray, action: int) -> np.ndarray:
        lx, ly = z[..., 0], z[..., 1]
        d_lx, d_ly = {
            0: (-self.kick, 0.0), 1: (self.kick, 0.0),
            2: (0.0, -self.kick), 3: (0.0, self.kick),
            4: (0.0, 0.0),
        }[action]
        new_lx = np.clip(lx + d_lx, -self.log_bound, self.log_bound)
        new_ly = np.clip(ly + d_ly, -self.log_bound, self.log_bound)
        return np.stack([new_lx, new_ly], axis=-1)

    def energy(self, z: np.ndarray) -> np.ndarray:
        """Lotka-Volterra Lyapunov function; strictly conserved only as kappa -> inf.
        Used here only as a damping sanity check (must decrease when kappa is finite).
        """
        x, y = np.exp(z[..., 0]), np.exp(z[..., 1])
        return self.d * x - self.c * np.log(np.clip(x, 1e-9, None)) + self.b * y - self.a * np.log(np.clip(y, 1e-9, None))

    def observable(self, z: np.ndarray) -> np.ndarray:
        x, y = np.exp(z[..., 0]), np.exp(z[..., 1])
        return x / (x + y)

    def metrics(self, z: np.ndarray) -> np.ndarray:
        dlx_dt, dly_dt = self._rhs(z[..., 0], z[..., 1])
        x, y = np.exp(z[..., 0]), np.exp(z[..., 1])
        dx_dt = dlx_dt * x  # d(log x)/dt -> dx/dt
        dy_dt = dly_dt * y
        change_in_prey = dx_dt
        # ratio has a genuine singularity whenever the predator population is
        # at a local extremum (dy_dt = 0); clip so a probe regressing on it
        # doesn't get blown up by outliers at those crossings.
        dy_dt_safe = np.where(np.abs(dy_dt) > 1e-6, dy_dt, np.sign(dy_dt + 1e-30) * 1e-6)
        ratio = np.clip(dx_dt / dy_dt_safe, -20.0, 20.0)
        return np.stack([change_in_prey, ratio], axis=-1)


def _demo() -> None:
    pp = PredatorPrey()
    # Fixed point where dlx=0, dly=0: x* = c/d, y* = (a - a*x*/kappa)/b.
    x_star = pp.c / pp.d
    y_star = (pp.a - pp.a * x_star / pp.kappa) / pp.b
    z = np.array([[np.log(3.0), np.log(2.0)]])
    for _ in range(20000):
        z = pp.flow(z, 0.01, 1)
    x_final, y_final = np.exp(z[0, 0]), np.exp(z[0, 1])
    assert abs(x_final - x_star) < 0.05 and abs(y_final - y_star) < 0.05, (
        f"implicit damping must spiral into the fixed point ({x_star:.3f},{y_star:.3f}), "
        f"got ({x_final:.3f},{y_final:.3f})"
    )
    obs = pp.observable(z)
    assert 0.0 < obs[0] < 1.0
    kicked = pp.apply_action(z, 1)
    assert kicked[0, 0] > z[0, 0]
    print("predator_prey ok")


if __name__ == "__main__":
    _demo()
