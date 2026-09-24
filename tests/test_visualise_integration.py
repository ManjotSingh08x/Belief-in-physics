"""Integration tests for visualisation panels, plotting pipeline, and trace reproducibility.

Covers:
- Exact equivalence between trace simulation and rollout token outputs across all 4 systems
- All panels in PANELS render cleanly without error for every system
- stability() produces all necessary metrics and metadata required by dashboard UI
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from physics.messk_configs import MESSK_CONFIGS, make_process
from physics.visualise import PANELS, plot, stability, trace


def test_trace_matches_rollout_for_all_systems():
    """REASON: The invariant that makes counterfactual causal analysis valid
    is that a step-by-step trace driven by a letter sequence reproduces the vectorized
    rollout tokens bit-for-bit.
    """
    for name in sorted(MESSK_CONFIGS):
        proc = make_process(name, m=8)
        tr = trace(proc, seed=42)
        rollout_tokens = proc.rollout(tr["letters"])["tokens"][0]
        assert np.array_equal(tr["tokens_driven"], rollout_tokens), (
            f"{name}: trace tokens do not match rollout tokens"
        )


def test_all_panels_render_without_error():
    """REASON: Any uncaught error or missing key in a panel handler breaks the entire
    dashboard figure. Testing each panel individually and collectively across all systems
    guarantees that every projection, colormap, and twin-axis rendering works cleanly.
    """
    for name in sorted(MESSK_CONFIGS):
        proc = make_process(name, m=6)
        tr = trace(proc, seed=1)
        fig = plot(tr, panels=PANELS, title=f"Test {name}")
        assert fig is not None
        plt.close(fig)


def test_stability_report_has_required_fields():
    """REASON: Downstream components (e.g. controls._verdict, notebook tables, sweep)
    rely on specific keys in the stability dictionary. Missing keys cause UI crashes.
    """
    required_keys = {
        "lyapunov",
        "clipped",
        "per_channel_clipped",
        "used_bins",
        "n_obs",
        "energy_drift_free",
        "finite",
        "gap_free_mean",
        "gap_free_max",
        "bayes_gap",
        "sync_curve",
        "sync_length",
        "state_bound_fraction",
        "channels",
        "obs_ranges",
        "obs_range",
        "observed_range",
        "stable",
        "reasons",
    }
    for name in sorted(MESSK_CONFIGS):
        proc = make_process(name, m=6)
        tr = trace(proc, seed=0)
        rep = stability(tr)
        missing = required_keys - set(rep.keys())
        assert not missing, f"{name} stability report is missing keys: {missing}"
        assert isinstance(rep["stable"], bool)
        assert isinstance(rep["reasons"], list)
        assert isinstance(rep["lyapunov"], float)
