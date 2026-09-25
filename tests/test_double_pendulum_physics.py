"""Test suite for Double Pendulum physics, stabilization, and observable wrapping.

Covers:
- Undamped energy conservation (Lagrangian dynamics correctness)
- Energy dissipation under viscous damping
- Multi-seed stability under the stabilized default damping (gamma1=gamma2=2.5)
- Lyapunov exponent transition (contractive at high damping vs chaotic at low damping)
- Angle wrapping of observable theta2 within [-pi, pi]
"""

from __future__ import annotations

import numpy as np
import pytest

from physics.messk_configs import (
    DOUBLE_PENDULUM_BINS,
    make_process,
    set_double_pendulum_bins,
)
from physics.systems.double_pendulum import DoublePendulum
from physics.visualise import lyapunov, stability, trace


def test_undamped_energy_conserved():
    """REASON: When gamma1=gamma2=0, total mechanical energy (kinetic + potential)
    must be conserved under Lagrangian dynamics.
    """
    z0 = np.array([[0.9, -0.4, 0.0, 0.0]])
    undamped = DoublePendulum(gamma1=0.0, gamma2=0.0)
    z = z0.copy()
    e0 = float(undamped.energy(z0)[0])

    for _ in range(2000):
        z = undamped.flow(z, dt=0.002)

    e_final = float(undamped.energy(z)[0])
    drift = abs(e_final - e0)
    assert drift < 5e-2, f"Undamped double pendulum energy drifted by {drift:.4e}"


def test_damped_energy_decreases():
    """REASON: Viscous damping on the joints must continuously dissipate energy
    during unforced motion.
    """
    dp = DoublePendulum()
    z0 = np.array([[0.9, -0.4, 0.0, 0.0]])
    e0 = float(dp.energy(z0)[0])
    z = z0.copy()

    for _ in range(2000):
        z = dp.flow(z, dt=0.002)

    e_final = float(dp.energy(z)[0])
    assert e_final < e0, f"Damped energy did not decrease: e0={e0:.4f}, e_final={e_final:.4f}"


def test_default_damping_is_stable_10_seeds():
    """REASON: Changing default damping from gamma=0.5 to gamma=2.5 brings the system
    from 0/10 passing and chaotic divergence (mean lambda = +1.18/s) into the contractive
    regime with mean lambda < -0.30/s and 0% clipping across all seeds.
    """
    proc = make_process("double_pendulum_mess4", m=40)
    results = [stability(trace(proc, seed=s)) for s in range(10)]
    lams = [r["lyapunov"] for r in results]
    clipped = [r["clipped"] for r in results]

    assert np.mean(lams) < -0.30, f"Mean Lyapunov must be strongly contractive (< -0.30), got {np.mean(lams):+.3f}"
    assert np.max(clipped) == 0.0, f"Clipping detected: {clipped}"
    assert sum(r["stable"] for r in results) >= 8, f"Pass rate too low: {[r['reasons'] for r in results]}"


def test_lyapunov_negative_at_high_damping():
    """REASON: Verifies the phase transition from chaotic to contractive.
    At gamma=2.5, lyapunov should be negative. At gamma=0.1, lyapunov should be positive (chaotic).
    """
    proc_stable = make_process("double_pendulum_mess4", m=20, system={"gamma1": 2.5, "gamma2": 2.5})
    proc_chaotic = make_process("double_pendulum_mess4", m=20, system={"gamma1": 0.1, "gamma2": 0.1})

    tr_stable = trace(proc_stable, seed=1)
    tr_chaotic = trace(proc_chaotic, seed=1)

    lam_stable = lyapunov(tr_stable)
    lam_chaotic = lyapunov(tr_chaotic)

    assert lam_stable < 0.05, f"Expected contractive lyapunov at gamma=2.5, got {lam_stable:+.3f}"
    assert lam_chaotic > 0.10, f"Expected chaotic lyapunov at gamma=0.1, got {lam_chaotic:+.3f}"


def test_angles_wrap_stay_in_range():
    """REASON: Observables represent angles theta1 and theta2, wrapped to [-pi, pi].
    Even after large velocity impulses and full 360-degree rotations,
    the observables returned by the system must strictly stay within [-pi, pi].
    """
    dp = DoublePendulum()
    # Test large angles
    angles = np.array([[-10.0], [-np.pi], [0.0], [np.pi], [10.0], [100.0]])
    z = np.zeros((len(angles), 4))
    z[:, 0] = angles[:, 0]
    z[:, 1] = -angles[:, 0]

    obs1 = dp.observable(z)
    obs_both = dp.observables(z)
    lo, hi = dp.obs_range
    assert np.all(obs1 >= lo) and np.all(obs1 <= hi), f"theta1 out of wrap bounds: {obs1}"
    assert np.all(obs_both >= lo) and np.all(obs_both <= hi), f"observables out of wrap bounds: {obs_both}"
    assert dp.observable_names == ("theta1", "theta2")


def test_double_pendulum_token_encoding_order():
    """REASON: Verifies that token encoding follows the mixed-radix formula:
    token = index(theta1) * B2 + index(theta2).
    """
    proc = make_process("double_pendulum_mess4")
    assert len(proc.obs_bins) == 2, f"Expected 2-channel obs_bins, got {proc.obs_bins}"
    b1, b2 = proc.obs_bins
    assert proc.channel_names == ("theta1", "theta2")
    assert proc.n_obs == b1 * b2

    # Test explicit angle pairs using dynamic grid bounds
    i1, i2 = min(10, b1 - 1), min(20, b2 - 1)
    test_indices = np.array([[i1, i2], [0, 0], [b1 - 1, b2 - 1], [b1 - 1, 0], [0, b2 - 1]])
    expected_tokens = test_indices[:, 0] * b2 + test_indices[:, 1]

    # Continuous values corresponding to these bin centers:
    (lo1, hi1), (lo2, hi2) = proc.obs_ranges
    vals = np.stack([
        lo1 + test_indices[:, 0] / max(b1 - 1, 1) * (hi1 - lo1),
        lo2 + test_indices[:, 1] / max(b2 - 1, 1) * (hi2 - lo2),
    ], axis=-1)
    tokens = proc.discretise(vals)
    assert np.array_equal(tokens, expected_tokens), f"Tokens {tokens} != expected {expected_tokens}"

    # Reconstruct back
    recon = proc.undiscretise(tokens)
    assert np.allclose(recon, vals, atol=1e-4), f"Reconstruction {recon} != original {vals}"


def test_double_pendulum_single_line_bin_config():
    """REASON: Verifies that changing double pendulum bins in a single line
    works seamlessly via:
    1. make_process("double_pendulum_mess4", obs_bins=20) -> (20, 20) with 400 vocab
    2. set_double_pendulum_bins(25) -> updates system default to (25, 25) with 625 vocab
    3. set_double_pendulum_bins((15, 25)) -> updates system default to (15, 25) with 375 vocab
    """
    # 1. Direct int override in make_process
    proc_override = make_process("double_pendulum_mess4", obs_bins=20)
    assert proc_override.obs_bins == (20, 20)
    assert proc_override.n_obs == 400

    # 2. Global one-line helper function
    orig = DOUBLE_PENDULUM_BINS
    try:
        set_double_pendulum_bins(25)
        p25 = make_process("double_pendulum_mess4")
        assert p25.obs_bins == (25, 25)
        assert p25.n_obs == 625

        set_double_pendulum_bins((15, 25))
        p_rect = make_process("double_pendulum_mess4")
        assert p_rect.obs_bins == (15, 25)
        assert p_rect.n_obs == 375
    finally:
        set_double_pendulum_bins(orig)

    # Restored correctly
    restored = make_process("double_pendulum_mess4")
    assert restored.obs_bins == orig if isinstance(orig, tuple) else (orig, orig)

