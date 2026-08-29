"""Seed replication report.

The Kaggle seeds kernel predates two changes to `12_emergence.py`: the
belief-anchored target family, and scoring emergence at a fraction of each
target's own final learned gain rather than at an absolute threshold. It stores
the per-target curves, so both are recomputed here rather than rerunning 12
GPU-hours of training.

Expects /tmp/seedout/phase5_07_emergence_s{1,2,3}.json from
`kaggle kernels output chayanaggarwal45/belief-in-physics-phase-5-seeds`.

Run:  uv run python scripts/phase5_seeds.py
"""
import json, sys, numpy as np
from pathlib import Path
sys.path.insert(0, "scripts")
from phase5_summary import MIN_GAIN, _fractional_rho, _gain_crossing

def normalise(run):
    """The seeds kernel predates `learned_gain` and `family`; derive both."""
    for entry in run.values():
        for t in entry["targets"]:
            t.setdefault("learned_gain", t["curve"][-1] - t["curve"][0])
            t.setdefault("family", "random")
    return run

runs = {"seed0 (main)": json.load(open("experiments/outputs-03/phase5_07_emergence.json"))}
for i in (1, 2, 3):
    runs[f"seed{i}"] = normalise(json.load(open(f"experiments/results-branch/seeds/phase5_07_emergence_s{i}.json")))

SYS = list(runs["seed0 (main)"])
print("E5 Spearman(coupling, log emergence) at 75% of each target's own final gain")
print(f"{'system':<17}" + "".join(f"{k:>16}" for k in runs))
for s in SYS:
    cells = []
    for name, r in runs.items():
        if s not in r:
            cells.append("--"); continue
        rho, n = _fractional_rho(r[s], 0.75, None)
        cells.append("--" if rho is None else f"{rho:+.2f} (n={n})")
    print(f"{s:<17}" + "".join(f"{c:>16}" for c in cells))

print("\nReal-target final R2 across seeds (matched-target run, not the phase-4 probe)")
for tgt in ("real_action_lag0", "real_metric", "real_z0"):
    print(f"  {tgt}")
    for s in SYS:
        vals = []
        for name, r in runs.items():
            if s not in r: vals.append(None); continue
            row = next((t for t in r[s]["targets"] if t["target"] == tgt), None)
            vals.append(None if row is None else row["final_r2"])
        got = [v for v in vals if v is not None]
        rng = f"[{min(got):.3f}, {max(got):.3f}]" if got else "--"
        print(f"    {s:<17}" + "".join(f"{('--' if v is None else f'{v:.3f}'):>9}" for v in vals) + f"   range {rng}")

print("\nEmergence ORDER, metric before belief? (75%-of-own-gain crossing, real targets)")
print(f"{'system':<17}{'run':<14}{'metric':>12}{'belief':>12}{'order':>18}")
for s in SYS:
    for name, r in runs.items():
        if s not in r: continue
        axis = r[s]["tokens_axis"]
        got = {}
        for tgt in ("real_metric", "real_action_lag0"):
            row = next((t for t in r[s]["targets"] if t["target"] == tgt), None)
            got[tgt] = (_gain_crossing(axis, row["curve"], 0.75 * row["learned_gain"])
                        if row and row["learned_gain"] > MIN_GAIN else None)
        m, b = got["real_metric"], got["real_action_lag0"]
        order = ("metric first" if m and b and m < b else
                 "belief first" if m and b else
                 "belief never" if m and not b else
                 "metric never" if b else "neither learned")
        f = lambda x: "never" if x is None else f"{x/1e6:.1f}M"
        print(f"{s:<17}{name:<14}{f(m):>12}{f(b):>12}{order:>18}")
