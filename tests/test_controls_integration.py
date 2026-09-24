"""Integration tests for controls, sweep UI, bin parsing, and damping calibration.

Covers:
- compute_optimal_gamma runs successfully on all 4 physical systems without crashing
- parse_bins accepts valid single and multi-channel specs and rejects invalid formats
- explorer widget constructs properly for all 4 systems in headless mode
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pytest

from physics.controls import compute_optimal_gamma, explorer, parse_bins
from physics.messk_configs import MESSK_CONFIGS


def test_compute_optimal_gamma_does_not_crash_on_any_system():
    """REASON: In prior versions, compute_optimal_gamma hardcoded sys_kw['gamma'] = gamma,
    raising TypeError / KeyError on systems using different parameter names (e.g. kappa in
    PredatorPrey, or gamma1/gamma2 in DoublePendulum).
    This test verifies that all 4 systems can compute optimal damping without crashing.
    """
    for system_name in sorted(MESSK_CONFIGS):
        res = compute_optimal_gamma(
            system_name=system_name,
            m=10,
            n_steps=5,
            n_trajs=4,
            max_iter=2,
        )
        assert "gamma_opt" in res, f"{system_name} missing gamma_opt"
        assert res["gamma_opt"] > 0.0, f"{system_name} returned non-positive gamma_opt: {res['gamma_opt']}"
        assert "balance_ratio" in res
        assert len(res["cum_injected"]) > 0
        assert len(res["cum_dissipated"]) > 0


def test_parse_bins_valid_and_invalid():
    """REASON: Tokenizer vocabulary parsing is critical for both 1D and 2D discretization.
    Valid single or cross-product specifications must parse into integer tuples,
    while empty strings, single-bin grids, or non-numeric tokens must be rejected with ValueError.
    """
    assert parse_bins("181") == (181,)
    assert parse_bins("30x30") == (30, 30)
    assert parse_bins("64, 32") == (64, 32)
    assert parse_bins(" 10 x 20 ") == (10, 20)

    for invalid in ("", "1", "0", "0x5", "abc", "1x1", "-5"):
        with pytest.raises(ValueError):
            parse_bins(invalid)


def test_explorer_builds_for_all_systems():
    """REASON: The explorer dashboard brings together make_process, trace, stability,
    and plotting. Building it headless for each system ensures that widget wiring,
    parameter introspection, and initial trace computation run without exception.
    """
    for system_name in sorted(MESSK_CONFIGS):
        w = explorer(default=system_name)
        assert w is not None, f"Failed to build explorer for {system_name}"


def test_verdict_multi_seed_and_regimes():
    """REASON: Verifies that _verdict formats multi-seed stability audits,
    correctly classifies dynamical regimes (contractive, marginal, chaotic),
    and generates informative transformer guidance without crashing.
    """
    from physics.controls import _verdict, classify_regime
    from physics.messk_configs import make_process
    from physics.visualise import stability, trace

    # Test regime classification
    assert classify_regime(-0.5)[0] == "CONTRACTIVE"
    assert classify_regime(0.0)[0] == "MARGINAL"
    assert classify_regime(0.5)[0] == "CHAOTIC"

    # Test multi-seed report formatting
    proc = make_process("pendulum_mess4", m=16)
    active_rep = stability(trace(proc, seed=0))
    multi = [(s, stability(trace(proc, seed=s))) for s in range(5)]

    html = _verdict(active_rep, multi_reports=multi, active_seed=0)
    assert "Multi-Seed Stability Audit" in html
    assert "Transformer Guidance" in html
    assert "Per-Seed Breakdown" in html
    assert "CONTRACTIVE" in html or "MARGINAL" in html or "CHAOTIC" in html
    assert "Seed 0" in html and "active" in html
    assert "Vocab Coverage" in html


def test_param_row_retune_range_inversion():
    """REASON: Shifting a slider's lower bound above its current upper bound
    must not trigger traitlets.TraitError: min > max.
    """
    from physics.controls import param_row

    slider, hbox = param_row("test_param", 5.0)
    lo_box, hi_box = hbox.children[2], hbox.children[3]
    # Current range is [0.0, 10.0]. Shift range upwards to [15.0, 25.0]
    lo_box.value = 15.0
    hi_box.value = 25.0
    assert slider.min == 15.0 and slider.max == 25.0


def test_lyapunov_short_and_degenerate_sequences():
    """REASON: Very short sequences (len <= 2) must return 0.0 safely without crashing."""
    from physics.visualise import lyapunov

    # Length 2 gap
    tr_short = {
        "gap_twin": [1e-8, 1e-8],
        "states_driven": [[0.0, 0.0], [0.1, 0.1]],
        "t_state": [0.0, 0.01],
    }
    assert lyapunov(tr_short) == 0.0


def test_predator_prey_low_kappa_finite_energy():
    """REASON: Carrying capacity kappa <= c/d (2.0) must not produce NaN or log-negative errors."""
    import numpy as np
    from physics.systems.predator_prey import PredatorPrey

    pp = PredatorPrey(kappa=1.0)
    z = pp.initial_state(2)
    e = pp.energy(z)
    assert np.all(np.isfinite(e))


def test_best_layer_empty_records_guard():
    """REASON: best_layer([]) must return empty dict instead of raising IndexError."""
    from models.analysis import best_layer

    assert best_layer([]) == {}


def test_cascade_screen_builds_five_n_damping_conditions():
    from physics.controls import cascade_screen

    result = cascade_screen(
        "pendulum_mess4",
        delta_v_values=[0.55],
        damping_values=[0.8],
        base_n=2,
        n_down=1,
        n_up=4,
        m=3,
        n_seeds=1,
        keep_stage1=1,
        keep_final=5,
    )
    by_condition = {row["condition"]: row for row in result["expanded"]}
    assert set(by_condition) == {
        "base", "n_up_gamma_down", "n_down_gamma_up",
        "n_up_gamma_constant", "n_down_gamma_constant",
    }
    baseline_product = by_condition["base"]["n_times_damping"]
    assert np.isclose(by_condition["n_up_gamma_down"]["n_times_damping"], baseline_product)
    assert np.isclose(by_condition["n_down_gamma_up"]["n_times_damping"], baseline_product)
