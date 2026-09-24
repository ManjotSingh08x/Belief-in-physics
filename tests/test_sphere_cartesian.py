"""Test suite for the spherical pendulum Cartesian (x, y) observable refactor.

Covers:
- Correct Cartesian projection from spherical state (theta, psi) to (x, y)
- Preservation of the full 0 to 90 degree lower hemisphere (obs_ranges = [-1, 1]^2)
- Discretisation & undiscretisation roundtrip accuracy on the 30x30 token grid
- Elimination of azimuthal wrap jumps (local diffusive token transitions)
- Dynamic vocabulary size (n_obs == 900)
- Multi-seed stability protocol (10 seeds, mean lambda <= +0.02/s, max lambda < +0.05/s, 0% clipping)
- Action distinctness on typical trajectories
"""

from __future__ import annotations

import numpy as np
import pytest

from physics.messk_configs import make_process
from physics.systems.sphere import SphereBall
from physics.visualise import stability, trace


def test_sphere_observable_is_cartesian():
    """REASON: The core architectural shift maps internal (theta, psi) to Cartesian (x, y).

    Observables must return [sin(theta)*cos(psi), sin(theta)*sin(psi)] on the horizontal plane.
    For a conical swing state with theta=0.6, psi=pi/4, observable coordinates must match
    the physical projection within floating point tolerance.
    """
    s = SphereBall()
    z = np.array([[0.6, np.pi / 4, 0.0, 0.0]])
    obs = s.observables(z)
    expected_x = np.sin(0.6) * np.cos(np.pi / 4)
    expected_y = np.sin(0.6) * np.sin(np.pi / 4)

    assert obs.shape == (1, 2), f"Expected shape (1, 2), got {obs.shape}"
    assert abs(obs[0, 0] - expected_x) < 1e-9, f"x mismatch: {obs[0, 0]} vs {expected_x}"
    assert abs(obs[0, 1] - expected_y) < 1e-9, f"y mismatch: {obs[0, 1]} vs {expected_y}"
    assert s.observable_names == ("x", "y"), f"Observable names must be ('x', 'y'), got {s.observable_names}"


def test_sphere_obs_ranges_cover_full_hemisphere():
    """REASON: The user specification strictly requires the observation range to cover 0-90 degrees.

    The range must not be narrowed to observed empirical bounds. In Cartesian coordinates,
    theta in [0, pi/2] corresponds exactly to x, y in [-1.0, 1.0], bounding the unit disc.
    """
    s = SphereBall()
    assert s.obs_ranges == ((-1.0, 1.0), (-1.0, 1.0)), (
        f"obs_ranges must be ((-1.0, 1.0), (-1.0, 1.0)), got {s.obs_ranges}"
    )
    assert s.obs_range == (-1.0, 1.0), f"obs_range scalar default must be (-1.0, 1.0), got {s.obs_range}"


def test_sphere_token_roundtrip():
    """REASON: Token discretization and reconstruction must be mutually consistent.

    On a 30x30 grid spanning [-1, 1]^2, bin width is 2.0 / 29 ~= 0.069.
    Undiscretizing discretised coordinates must recover the original positions within half a bin width.
    """
    proc = make_process("sphere_mess4", m=2)
    rng = np.random.default_rng(42)
    xy = rng.uniform(-0.9, 0.9, (100, 2))
    tokens = proc.discretise(xy)
    recovered = proc.undiscretise(tokens)

    half_bin = (2.0 / 29) / 2.0 + 1e-6
    max_err = np.abs(recovered - xy).max()
    assert max_err <= half_bin, f"Roundtrip error {max_err:.4f} exceeds half-bin tolerance {half_bin:.4f}"


def test_no_psi_wrap_jumps_in_cartesian():
    """REASON: Shifting to Cartesian eliminates the coordinate singularity at theta=0 and wrap at +-pi.

    When passing through the origin region, trajectories in spherical coordinates suffered
    violent 15+ bin teleportations in token space. In Cartesian (x, y), transitions
    must be smooth and local (max step <= 4 bins).
    """
    proc = make_process("sphere_mess4", m=20, system={"theta0": 0.1, "psi_dot0": 3.0})
    tr = trace(proc, seed=0)
    tokens = tr["tokens_driven"]
    indices = proc.tokens_to_indices(tokens)  # shape (L, 2)
    diffs = np.abs(np.diff(indices, axis=0))  # step size per coordinate
    max_jump = diffs.max()

    assert max_jump <= 4, (
        f"Token jump of {max_jump} bins detected; Cartesian representation must be locally smooth (<= 4 bins)"
    )


def test_sphere_vocab_size():
    """REASON: Vocabulary must dynamically reflect the 30x30 Cartesian product grid (900 cells).

    Downstream transformers must use proc.n_obs rather than hardcoding 181.
    """
    proc = make_process("sphere_mess4")
    assert proc.n_obs == 900, f"Expected n_obs == 900, got {proc.n_obs}"
    assert proc.obs_bins == (30, 30), f"Expected obs_bins == (30, 30), got {proc.obs_bins}"


def test_10seed_stability():
    """REASON: Stability must be evaluated across 10 random seeds to guarantee robustness.

    Requirements:
    - 10/10 seeds pass stability
    - Mean finite-time Lyapunov exponent <= +0.02/s (actually strictly negative)
    - Max Lyapunov exponent < +0.05/s across all seeds
    - 0% clipping across all seeds (the ball never exceeds the unit bowl boundary)
    """
    proc = make_process("sphere_mess4")
    results = [stability(trace(proc, seed=s)) for s in range(10)]
    lams = [r["lyapunov"] for r in results]
    clipped = [r["clipped"] for r in results]
    passes = [r["stable"] for r in results]

    assert sum(passes) == 10, f"Only {sum(passes)}/10 seeds stable. Reasons: {[r['reasons'] for r in results]}"
    assert np.mean(lams) <= 0.02, f"Mean Lyapunov {np.mean(lams):+.3f}/s exceeds threshold +0.02/s"
    assert np.max(lams) < 0.05, f"Max Lyapunov {np.max(lams):+.3f}/s exceeds threshold +0.05/s"
    assert np.max(clipped) == 0.0, f"Clipping detected: {clipped}"


def test_sphere_actions_distinct_in_cartesian():
    """REASON: The four tangent kicks (north, south, west, east) must remain observably distinct.

    After projecting to Cartesian (x, y), each of the 4 actions must produce a distinct
    token sequence from the initial state, ensuring the HMM letters carry physical information.
    """
    proc = make_process("sphere_mess4")
    system = proc.system
    state = np.repeat(system.initial_state(1), 4, axis=0)
    state = system.kick(state, proc.actions)
    baseline = system.initial_state(1)
    action_tokens, baseline_tokens = [], []
    for _ in range(proc.n_steps):
        state = proc.flow(state)
        baseline = proc.flow(baseline)
        action_tokens.append(proc.observe(state))
        baseline_tokens.append(proc.observe(baseline)[0])

    trajectories = np.stack(action_tokens, axis=1)
    baseline_trajectory = np.asarray(baseline_tokens)
    assert len(np.unique(trajectories, axis=0)) == 4, "Every action must produce a unique trajectory"
    assert np.all(np.any(trajectories != baseline_trajectory, axis=1)), "Every action must differ from baseline"
