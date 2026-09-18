"""Test suite for Predator-Prey physics, Lyapunov function, and stability.

Covers:
- Monotonic decrease of the shifted Lyapunov function V(x,y)
- Spiral convergence into the damped coexistence fixed point (x*, y*)
- Prey share observable boundedness in (0, 1) under driven dynamics
- Multi-seed stability across 10 random seeds
"""

from __future__ import annotations

import numpy as np
import pytest

from physics.messk_configs import make_process
from physics.systems.predator_prey import PredatorPrey
from physics.visualise import stability, trace


def test_lyapunov_decreases_monotonically():
    """REASON: The shifted Lyapunov function centered at (x*, y*) = (2.0, 1.0)
    has dV/dt = -(a/kappa)*(x - x*)^2 <= 0.
    In contrast to the old undamped Hamiltonian (which oscillated sign under damping),
    V must decrease on virtually every step (allowing for tiny floating-point noise).
    """
    pp = PredatorPrey()
    z = pp.initial_state(1)
    energies = [float(pp.energy(z)[0])]

    for _ in range(10000):
        z = pp.flow(z, dt=0.01)
        energies.append(float(pp.energy(z)[0]))

    e = np.array(energies)
    diffs = np.diff(e)
    decreasing_frac = float((diffs <= 1e-12).mean())
    assert decreasing_frac > 0.99, (
        f"Lyapunov function must decrease monotonically; only {decreasing_frac:.2%} of steps decreased"
    )
    assert e[-1] < e[0] * 0.1, f"Lyapunov should decay significantly; e0={e[0]:.4f}, e_final={e[-1]:.4f}"


def test_spiral_converges_to_fixed_point():
    """REASON: Under carrying-capacity damping kappa < inf, the interior fixed point
    x* = c/d = 2.0, y* = (a - a*x*/kappa)/b = 1.0 is globally asymptotically stable.
    The trajectory must spiral into this point.
    """
    pp = PredatorPrey()
    x_star = pp.c / pp.d
    y_star = (pp.a - pp.a * x_star / pp.kappa) / pp.b

    z = pp.initial_state(1)
    for _ in range(20000):
        z = pp.flow(z, dt=0.01)

    x_final = float(np.exp(z[0, 0]))
    y_final = float(np.exp(z[0, 1]))

    assert abs(x_final - x_star) < 0.05, f"Prey did not converge: {x_final} vs {x_star}"
    assert abs(y_final - y_star) < 0.05, f"Predator did not converge: {y_final} vs {y_star}"


def test_prey_share_bounded_in_unit_interval():
    """REASON: The observable is the prey population share x / (x + y).
    For any non-negative population sizes, this must strictly remain in (0, 1).
    """
    proc = make_process("predator_prey_mess4", m=100)
    tr = trace(proc, seed=42)

    obs = tr["obs_driven"]
    assert np.all(obs > 0.0) and np.all(obs < 1.0), (
        f"Prey share went out of unit bounds: min={obs.min()}, max={obs.max()}"
    )


def test_predator_prey_stability_10_seeds():
    """REASON: Transformer training requires consistent stability without divergence
    or runaway clipping across multiple trajectory realizations.
    """
    proc = make_process("predator_prey_mess4", m=40)
    for s in range(10):
        tr = trace(proc, seed=s)
        rep = stability(tr)
        assert rep["stable"], f"Seed {s} unstable: {rep['reasons']}"
        assert rep["clipped"] < 0.01, f"Seed {s} clipped too high: {rep['clipped']:.2%}"
        assert rep["lyapunov"] < 0.05, f"Seed {s} has non-contractive lyapunov: {rep['lyapunov']}"
