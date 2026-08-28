"""Energy/damping sanity checks per system, mirroring each module's own
_demo() -- see the module docstrings for the physics derivation notes.
"""
import numpy as np

from physics.systems.double_pendulum import DoublePendulum
from physics.systems.pendulum import Pendulum
from physics.systems.predator_prey import PredatorPrey
from physics.systems.sphere import SphereBall


def test_pendulum_undamped_conserves_energy():
    p = Pendulum(gamma=0.0)
    z = np.array([[0.3, 0.0]])
    e0 = p.energy(z)[0]
    for _ in range(2000):
        z = p.flow(z, 0.01, 1)
    assert abs(p.energy(z)[0] - e0) < 1e-3


def test_pendulum_damped_energy_decreases():
    p = Pendulum(gamma=0.15)
    z = np.array([[0.3, 0.0]])
    e0 = p.energy(z)[0]
    for _ in range(2000):
        z = p.flow(z, 0.01, 1)
    assert p.energy(z)[0] < e0


def test_pendulum_kick_action_sets_omega():
    p = Pendulum(kick=1.5)
    kicked = p.apply_action(np.array([[0.3, 0.0]]), 2)
    assert kicked[0, 1] == p.kick


def test_predator_prey_implicit_damping_spirals_to_the_fixed_point():
    pp = PredatorPrey()
    x_star = pp.c / pp.d
    y_star = (pp.a - pp.a * x_star / pp.kappa) / pp.b
    z = np.array([[np.log(3.0), np.log(2.0)]])
    for _ in range(20000):
        z = pp.flow(z, 0.01, 1)
    x_final, y_final = np.exp(z[0, 0]), np.exp(z[0, 1])
    assert abs(x_final - x_star) < 0.05
    assert abs(y_final - y_star) < 0.05


def test_sphere_frictionless_geodesic_conserves_speed():
    s0 = SphereBall(mu=0.0)
    z = np.array([[0.1, 0.0, 0.3, 0.3]])
    e0 = s0.energy(z)[0]
    for _ in range(200):
        z = s0.flow(z, 0.005, 1)
        assert abs(z[0, 0]) < s0.lat_max - 1e-6
    assert abs(s0.energy(z)[0] - e0) < 1e-3


def test_sphere_sliding_friction_decreases_speed():
    s = SphereBall(mu=0.25)
    z = np.array([[0.2, 0.0, 1.0, 0.8]])
    e0 = s.energy(z)[0]
    for _ in range(500):
        z = s.flow(z, 0.01, 1)
    assert s.energy(z)[0] < e0


def test_double_pendulum_undamped_conserves_energy():
    dp = DoublePendulum(gamma1=0.0, gamma2=0.0)
    z = np.array([[0.9, -0.4, 0.0, 0.0]])
    e0 = dp.energy(z)[0]
    for _ in range(2000):
        z = dp.flow(z, 0.002, 1)
    assert abs(dp.energy(z)[0] - e0) < 5e-2


def test_double_pendulum_damped_energy_decreases():
    dp = DoublePendulum(gamma1=0.08, gamma2=0.08)
    z = np.array([[0.9, -0.4, 0.0, 0.0]])
    e0 = dp.energy(z)[0]
    for _ in range(2000):
        z = dp.flow(z, 0.002, 1)
    assert dp.energy(z)[0] < e0
