"""Predator-prey with logistic self-limitation on prey. State z = (log x, log y).

dx/dt = a*x - b*x*y - a*x^2/kappa     (the -a*x^2/kappa term is the damping:
dy/dt = d*x*y - c*y                    no explicit friction, but it turns the
                                       neutral closed orbits of plain
                                       Lotka-Volterra into a spiral converging
                                       on a fixed point)

The chain's letter arrives as a boost to the prey population, in log space so a
fixed impulse is a fixed multiplicative factor regardless of the current level.
Every letter's boost is different and non-zero.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PredatorPrey:
    a: float = 1.0
    b: float = 0.6
    c: float = 0.8
    d: float = 0.4
    kappa: float = 5.0
    log_bound: float = 3.5

    obs_range: tuple[float, float] = (0.0, 1.0)
    metric_name: str = "dx_dt"

    def initial_state(self, n: int) -> np.ndarray:
        return np.stack([np.full(n, np.log(3.0)), np.full(n, np.log(2.0))], axis=-1)

    def _rhs(self, lx: np.ndarray, ly: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x, y = np.exp(lx), np.exp(ly)
        return self.a - self.b * y - self.a * x / self.kappa, self.d * x - self.c

    def flow(self, z: np.ndarray, dt: float, substeps: int = 1) -> np.ndarray:
        lx, ly = z[..., 0].copy(), z[..., 1].copy()
        h = dt
        for _ in range(substeps):
            k1x, k1y = self._rhs(lx, ly)
            k2x, k2y = self._rhs(lx + h / 2 * k1x, ly + h / 2 * k1y)
            k3x, k3y = self._rhs(lx + h / 2 * k2x, ly + h / 2 * k2y)
            k4x, k4y = self._rhs(lx + h * k3x, ly + h * k3y)
            lx = np.clip(lx + h / 6 * (k1x + 2 * k2x + 2 * k3x + k4x), -self.log_bound, self.log_bound)
            ly = np.clip(ly + h / 6 * (k1y + 2 * k2y + 2 * k3y + k4y), -self.log_bound, self.log_bound)
        return np.stack([lx, ly], axis=-1)

    def kick(self, z: np.ndarray, dv: np.ndarray) -> np.ndarray:
        """Restock the prey, one log-magnitude per row."""
        lx = np.clip(z[..., 0] + dv, -self.log_bound, self.log_bound)
        return np.stack([lx, z[..., 1]], axis=-1)

    def observable(self, z: np.ndarray) -> np.ndarray:
        """Prey share x/(x+y), which is bounded and so bins without a clip."""
        x, y = np.exp(z[..., 0]), np.exp(z[..., 1])
        return x / (x + y)

    def metric(self, z: np.ndarray) -> np.ndarray:
        dlx, _ = self._rhs(z[..., 0], z[..., 1])
        return dlx * np.exp(z[..., 0])

    def energy(self, z: np.ndarray) -> np.ndarray:
        """Lotka-Volterra Lyapunov function, strictly conserved only as kappa -> inf.
        Used as a damping sanity check: it must decrease when kappa is finite.
        """
        x, y = np.exp(z[..., 0]), np.exp(z[..., 1])
        return (
            self.d * x - self.c * np.log(np.clip(x, 1e-9, None))
            + self.b * y - self.a * np.log(np.clip(y, 1e-9, None))
        )


def _demo() -> None:
    pp = PredatorPrey()
    x_star = pp.c / pp.d
    y_star = (pp.a - pp.a * x_star / pp.kappa) / pp.b

    z = pp.initial_state(1)
    for _ in range(20000):
        z = pp.flow(z, 0.01)
    x_final, y_final = np.exp(z[0, 0]), np.exp(z[0, 1])
    assert abs(x_final - x_star) < 0.05 and abs(y_final - y_star) < 0.05, (
        f"must spiral into ({x_star:.3f},{y_star:.3f}), got ({x_final:.3f},{y_final:.3f})"
    )

    obs = pp.observable(z)
    assert 0.0 < obs[0] < 1.0, "prey share must stay in the unit interval"
    assert pp.kick(z, np.array([0.4]))[0, 0] > z[0, 0], "a boost must raise the prey"
    print("predator_prey ok")


if __name__ == "__main__":
    _demo()
