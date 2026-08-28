"""Figures for Phase 2 and 3 from the JSON the Kaggle run writes.

One row per system: the training curve, then probe R^2 against depth for the
trained model and for the same initialisation left untrained. The second panel
is the one that carries the claim -- if the two curves sit on top of each other,
a random projection of the token history was already sufficient and the trained
model has shown nothing.

Run:  uv run python -m scripts.make_phase23_figures [results_dir]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = Path(sys.argv[1] if len(sys.argv) > 1 else "experiments/results-01")
FIG_DIR = Path("figures")


def main() -> None:
    training = json.loads((RESULTS_DIR / "phase2_training.json").read_text())
    probes = json.loads((RESULTS_DIR / "phase3_probes.json").read_text())
    systems = [s for s in training if s in probes]

    FIG_DIR.mkdir(exist_ok=True)
    fig, axes = plt.subplots(len(systems), 2, figsize=(11, 3.1 * len(systems)), squeeze=False)

    for row, name in enumerate(systems):
        history = training[name]["history"]
        steps = [h["step"] for h in history]

        ax = axes[row][0]
        ax.plot(steps, [h["eval_obs_loss"] for h in history], label="observation tokens")
        ax.plot(steps, [h["eval_action_loss"] for h in history], label="action tokens", alpha=0.7)
        ax.axhline(
            training[name]["action_loss_floor"],
            ls="--", c="k", lw=0.8,
            label=f"action floor log({training[name]['n_actions']})",
        )
        ax.set(xlabel="step", ylabel="held-out cross entropy (nats)", title=f"{name} - Phase 2")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

        ax = axes[row][1]
        for tag, style in (("trained", "-o"), ("random_init", "--s")):
            records = probes[name][tag]
            ax.plot(
                [r["depth"] for r in records],
                [r["r2_by_group"]["metric"] for r in records],
                style, label=f"{tag} (metric)", ms=4,
            )
        shuffled = [r["r2_shuffled_control"] for r in probes[name]["trained"]]
        ax.plot(range(len(shuffled)), shuffled, ":", c="grey", label="shuffled control")
        ax.axhline(0.0, c="k", lw=0.6)
        verdict = probes[name]["verdict"]
        ax.set(
            xlabel="depth (0 = embedding)",
            ylabel="held-out $R^2$",
            title=f"{name} - Phase 3  [baseline {'intact' if verdict['baseline_intact'] else 'SUSPECT'}]",
        )
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    fig.tight_layout()
    out = FIG_DIR / "phase23_summary.png"
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
