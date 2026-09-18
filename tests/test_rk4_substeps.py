"""Integration tests for RK4 substep integration across all physics systems.

Covers:
- Verification that substeps > 1 integrates for exactly dt (step size h = dt / substeps)
- Undamped pendulum energy conservation under substeps
- Predator-Prey numerical convergence with substeps
"""

from __future__ import annotations

import numpy as np
import pytest

from physics.systems.pendulum import Pendulum
from physics.systems.predator_prey import PredatorPrey
from physics.systems.double_pendulum import DoublePendulum
from physics.systems.sphere import SphereBall


def test_substeps_converge_to_same_result_for_all_systems():
    """REASON: Verifies that h = dt / substeps fix works across all 4 systems.

    Prior to the fix, h = dt inside the substep loop meant substeps=10 integrated
    10*dt in time instead of dt. For a small time step dt=0.01, running substeps=1
    and substeps=5 should yield very close trajectories (within O(dt^4) RK4 truncation error).
    """
    systems = [
        ("Pendulum", Pendulum(), np.array([[0.5, 0.2]])),
        ("PredatorPrey", PredatorPrey(), np.array([[np.log(2.0), np.log(1.0)]])),
        ("DoublePendulum", DoublePendulum(), np.array([[0.5, -0.3, 0.1, -0.1]])),
        ("SphereBall", SphereBall(), np.array([[0.4, 0.2, 0.1, -0.1]])),
    ]

    dt = 0.01
    for name, sys, z0 in systems:
        z1 = sys.flow(z0.copy(), dt=dt, substeps=1)
        z5 = sys.flow(z0.copy(), dt=dt, substeps=5)
        diff = np.max(np.abs(z1 - z5))
        assert diff < 1e-3, (
            f"{name} substeps=1 vs substeps=5 diverged by {diff:.6e}; "
            f"indicates substeps step size h is incorrect (e.g. h = dt instead of dt / substeps)"
        )


def test_pendulum_undamped_energy_conserved_with_substeps():
    """REASON: In the undamped Hamiltonian system (gamma=0), RK4 with substeps
    must preserve total energy over thousands of integration steps.
    """
    p = Pendulum(gamma=0.0)
    z = np.array([[0.8, 0.0]])
    e0 = p.energy(z)[0]

    for _ in range(5000):
        z = p.flow(z, dt=0.01, substeps=4)

    e_final = p.energy(z)[0]
    drift = abs(e_final - e0)
    assert drift < 1e-4, f"Undamped pendulum energy drifted by {drift:.6e} with substeps=4"


def test_predator_prey_substeps_match_single_step():
    """REASON: Predator-Prey is non-linear and sensitive to time scale.
    Flowing dt=0.02 with substeps=1 vs substeps=4 must stay close on multiple steps.
    """
    pp = PredatorPrey()
    z1 = pp.initial_state(1)
    z4 = z1.copy()

    for _ in range(50):
        z1 = pp.flow(z1, dt=0.02, substeps=1)
        z4 = pp.flow(z4, dt=0.02, substeps=4)

    err = np.max(np.abs(z1 - z4))
    assert err < 1e-3, f"PredatorPrey multi-step substep discrepancy: {err:.6e}"
