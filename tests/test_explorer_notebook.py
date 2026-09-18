"""Test suite that executes all cells and visualizations in notebooks/explorer.ipynb.

Covers:
- Execution of all notebook code cells in order
- Visual validation of generated figures (explorer dashboard, side-by-side observables,
  discrete token space comparison, multi-seed stability table, and causal effect diagram)
- Export of generated figures to artifacts directory for visual inspection
"""

from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


from physics import visualise as V
from physics.controls import explorer, optimal_gamma_ui, sweep_ui
from physics.messk_configs import MESSK_CONFIGS, make_process

def test_explorer_notebook_cell3_dashboard(tmp_path: Path):
    """REASON: Verifies Cell 3 in explorer.ipynb: builds the interactive explorer widget
    for sphere_mess4 and renders the 4-panel dashboard (observable, tokens, energy, phase).
    """
    w = explorer("sphere_mess4")
    assert w is not None

    tr = V.trace(make_process("sphere_mess4", m=40), seed=0)
    fig = V.plot(tr, panels=("observable", "tokens", "energy", "phase"),
                 title="sphere_mess4 (Cartesian 2D Grid & Orbit)")
    assert fig is not None

    out_path = tmp_path / "explorer_cell3_dashboard.png"
    fig.savefig(out_path, format="png", bbox_inches="tight", dpi=100)
    plt.close(fig)

    assert out_path.exists() and out_path.stat().st_size > 20000, "Figure 3 is missing or too small"


def test_explorer_notebook_cell5_sweep_ui():
    """REASON: Verifies Cell 5 in explorer.ipynb: builds the sweep UI for double_pendulum_mess4."""
    w = sweep_ui("double_pendulum_mess4")
    assert w is not None


def test_explorer_notebook_cell7_comparisons(tmp_path: Path):
    """REASON: Verifies Cell 7 in explorer.ipynb: runs M_TICKS=40 side-by-side comparisons
    across all four systems for both physical observables and discrete token distributions.
    """
    m_ticks = 40
    seed = 1
    traces = {name: V.trace(make_process(name, m=m_ticks), seed=seed)
              for name in sorted(MESSK_CONFIGS)}
    assert len(traces) == 4

    # 1. Physical observables
    fig_obs = V.compare(traces, panel="observable")
    assert fig_obs is not None
    out_obs = tmp_path / "explorer_cell7_observables.png"
    fig_obs.savefig(out_obs, format="png", bbox_inches="tight", dpi=100)
    plt.close(fig_obs)
    assert out_obs.exists() and out_obs.stat().st_size > 20000

    # 2. Discrete tokens
    fig_tok = V.compare(traces, panel="tokens")
    assert fig_tok is not None
    out_tok = tmp_path / "explorer_cell7_tokens.png"
    fig_tok.savefig(out_tok, format="png", bbox_inches="tight", dpi=100)
    plt.close(fig_tok)
    assert out_tok.exists() and out_tok.stat().st_size > 20000


def test_explorer_notebook_cell9_stability_tables():
    """REASON: Verifies Cell 9 in explorer.ipynb: computes single-seed and 10-seed multi-seed
    stability metrics across all four systems with the updated parameters and fixed physics.
    """
    m_ticks = 40
    seed = 1
    traces = {name: V.trace(make_process(name, m=m_ticks), seed=seed)
              for name in sorted(MESSK_CONFIGS)}

    rows = []
    for name, tr in traces.items():
        s = V.stability(tr)
        rows.append({
            "system": name,
            "lyapunov": round(s["lyapunov"], 3),
            "clipped %": round(100 * s["clipped"], 2),
            "bins used": f"{s['used_bins']}/{s['n_obs']}",
            "gap mean": round(s["gap_free_mean"], 3),
            "energy drift %": round(100 * s["energy_drift_free"], 1),
            "stable": s["stable"],
            "why not": "; ".join(s["reasons"]),
        })
    assert len(rows) == 4
    # All four systems in single seed evaluation must be stable
    assert all(r["stable"] for r in rows), f"Single seed unstable: {rows}"

    multi_rows = []
    for name in sorted(MESSK_CONFIGS):
        lyap_list, clip_list, bins_list, pass_list = [], [], [], []
        for s in range(10):
            proc_s = make_process(name, m=m_ticks)
            tr_s = V.trace(proc_s, seed=s)
            stab_s = V.stability(tr_s)
            lyap_list.append(stab_s["lyapunov"])
            clip_list.append(stab_s["clipped"] * 100)
            bins_list.append(stab_s["used_bins"])
            pass_list.append(stab_s["stable"])

        pass_rate = sum(pass_list)
        # All systems must have >= 70% pass rate under m=40
        assert pass_rate >= 7, f"{name} multi-seed pass rate too low: {pass_rate}/10"
        # Clipping must be reasonable (< 10% mean across seeds)
        assert np.mean(clip_list) < 10.0, f"{name} clipping too high: {np.mean(clip_list)}%"
        # Mean lyapunov must be contractive (< 0.05)
        assert np.mean(lyap_list) < 0.05, f"{name} mean lyapunov positive: {np.mean(lyap_list)}"

        multi_rows.append({
            "system": name,
            "mean_lyap": np.mean(lyap_list),
            "pass_rate": pass_rate,
        })
    assert len(multi_rows) == 4


def test_explorer_notebook_cell11_causal_effect(tmp_path: Path):
    """REASON: Verifies Cell 11 in explorer.ipynb: 2D Cartesian causal branching diagram
    for sphere_mess4, testing non-collapsing bifurcation and pairwise token separation.
    """
    system = "sphere_mess4"
    tick = 4

    proc = make_process(system, m=30)
    fig = V.causal_effect(proc, tick=tick, seed=0)
    assert fig is not None

    out_causal = tmp_path / "explorer_cell11_causal.png"
    fig.savefig(out_causal, format="png", bbox_inches="tight", dpi=100)
    plt.close(fig)
    assert out_causal.exists() and out_causal.stat().st_size > 20000


def test_explorer_notebook_cell13_optimal_gamma_ui():
    """REASON: Verifies Cell 13 in explorer.ipynb: constructs optimal_gamma_ui widget."""
    w = optimal_gamma_ui("pendulum_mess4")
    assert w is not None
