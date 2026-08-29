"""The all-positive 3-action pendulum: {+dv, +2dv, +3dv}, no no-op.

At dv = 1.5 this set has variance (2/3)*1.5^2 = 1.5, exactly the parent
{-dv, 0, +dv}'s, with the same M and the same 26,244 branches. The only thing
that differs is the mean: +3.0 of omega per segment against 0. So the pair is a
clean single-variable test of what a deterministic drift does to the belief
geometry, and both plot as a Mess3-style 2-simplex.

The prediction under test is that the drift runs omega into the +-8 clip, where
distinct action words map to identical states: branches MERGE, so the posterior
cannot separate them, entropy rises toward uniform and last-action recovery
falls toward 1/3. That prediction is arithmetic from the config, not a
measurement, which is why the saturated fraction is measured here directly off
the branch tables rather than argued.

`pendulum_mess3_plus_small` shrinks dv until the drift equilibrium sits at half
the clip. It is the other side of the trade: no saturation, but the kick
variance falls to 0.082, eighteen times less than the parent's.

Run:  uv run python scripts/mess3_positive_geometry.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mess4_geometry import _scatter_simplex, collect  # noqa: E402
from physics.branch_configs import BRANCH_CONFIGS, make_branch_process  # noqa: E402

NAMES = ["pendulum", "pendulum_mess3_plus", "pendulum_mess3_plus_small"]
OUT = Path("experiments/outputs-mess4")


def saturation(name: str) -> dict:
    """Fraction of (branch, step) states pinned at the omega clip, by segment.

    This is the mechanism the drift argument rests on: a clipped state is one
    that several distinct action words share, so saturation is exactly the rate
    at which the process stops being identifiable.
    """
    proc = make_branch_process(name)
    cfg = BRANCH_CONFIGS[name]
    omega_max = cfg["system"](**cfg["system_kwargs"]).omega_max
    by_segment, prior_mean = [], []
    for m, table in enumerate(proc.metric_tables):
        omega = table[..., 0]  # pendulum's metric IS omega
        by_segment.append(float((np.abs(omega) >= omega_max - 1e-9).mean()))
        prior_mean.append(float(omega.mean()))  # branches are equiprobable a priori
    return {
        "omega_max": float(omega_max),
        "saturated_fraction_by_segment": by_segment,
        "mean_omega_by_segment": prior_mean,
        "saturated_fraction_overall": float(np.mean(by_segment)),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    records = []
    for name in NAMES:
        d = collect(name)
        d["saturation"] = saturation(name)
        kicks = BRANCH_CONFIGS[name]["system"](
            **BRANCH_CONFIGS[name]["system_kwargs"]
        ).dv_table
        d["kicks"] = [float(k) for k in kicks]
        records.append(d)
        sat = d["saturation"]
        print(
            f"{name:28s} kicks={[round(k, 2) for k in d['kicks']]} "
            f"entropy={d['entropy_mean']:.3f} recovery={d['oracle_recovery_mean']:.3f} "
            f"({d['recovery_over_chance']:.2f}x chance)  "
            f"saturated={sat['saturated_fraction_overall']:.1%} "
            f"omega by segment={[round(v, 1) for v in sat['mean_omega_by_segment']]}",
            flush=True,
        )

    fig = plt.figure(figsize=(13.5, 10.5))
    fig.suptitle(
        "Pendulum belief simplex: mean-zero vs all-positive action sets\n"
        "left and centre have IDENTICAL kick variance (1.5) and branch count; only the mean differs",
        fontsize=11,
    )
    for col, d in enumerate(records):
        ax = fig.add_subplot(3, 3, 1 + col)
        sc = _scatter_simplex(
            ax, d, d["_since"], "viridis",
            f"{d['name']}\nkicks {[round(k, 2) for k in d['kicks']]}",
        )
        if col == 2:
            fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02, label="obs since kick")

        ax = fig.add_subplot(3, 3, 4 + col)
        _scatter_simplex(ax, d, d["_truth"], "tab10", "coloured by TRUE last action")

    ax = fig.add_subplot(3, 3, 7)
    for d in records:
        ax.plot(np.arange(len(d["saturation"]["saturated_fraction_by_segment"])),
                d["saturation"]["saturated_fraction_by_segment"], "o-", ms=3, label=d["name"])
    ax.set_xlabel("segment")
    ax.set_ylabel(r"fraction of states at the $\omega$ clip")

    ax = fig.add_subplot(3, 3, 8)
    for d in records:
        ax.plot([r["obs_since_kick"] for r in d["by_obs_since_kick"]],
                [r["entropy_normalised"] for r in d["by_obs_since_kick"]], "o-", ms=3, label=d["name"])
    ax.axhspan(0.50, 0.76, color="0.9", zorder=0)
    ax.set_xlabel("observations since kick")
    ax.set_ylabel("normalised belief entropy")

    ax = fig.add_subplot(3, 3, 9)
    for d in records:
        ax.plot([r["obs_since_kick"] for r in d["by_obs_since_kick"]],
                [r["oracle_recovery"] for r in d["by_obs_since_kick"]], "o-", ms=3, label=d["name"])
    ax.axhline(1 / 3, color="0.7", lw=0.8, ls=":", zorder=0)
    ax.set_xlabel("observations since kick")
    ax.set_ylabel("oracle recovery (chance = 1/3)")

    for ax in fig.axes[-3:]:
        ax.legend(fontsize=6)
        ax.tick_params(labelsize=7)

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT / "mess3_positive_geometry.png", dpi=150)
    plt.close(fig)

    (OUT / "mess3_positive.json").write_text(
        json.dumps([{k: v for k, v in d.items() if not k.startswith("_")} for d in records],
                   indent=2, default=float)
    )
    print(f"wrote {OUT}/mess3_positive_geometry.png")


if __name__ == "__main__":
    main()
