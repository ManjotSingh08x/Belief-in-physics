"""Separability report for all four systems -> results/phase1.json.

# ponytail: skips the plan's rank_metrics() diagnostic -- the metrics used
# are exactly the ones the user specified per system (already physically
# motivated), so there is no candidate to rank against; add rank_metrics()
# if a future system's target metric becomes ambiguous.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from physics.configs import DEFAULT_CONFIGS, NOOP_ACTION
from physics.factory import make_simulator
from physics.separability import evaluate_separability

OUT_PATH = Path(__file__).resolve().parent.parent / "results" / "phase1.json"


def _sample_belief0s(hmm, rng, n: int) -> list[np.ndarray]:
    """A handful of non-uniform, physically plausible starting beliefs:
    a delta smoothed by a few free-flow steps, from several random latent bins.
    """
    out = []
    for _ in range(n):
        b = np.zeros(hmm.n_latent)
        b[rng.integers(0, hmm.n_latent)] = 1.0
        for _ in range(2):
            b = b @ hmm.T
        out.append(b)
    return out


def main() -> None:
    rng = np.random.default_rng(1)
    report = {}
    all_pass = True
    for name, config in DEFAULT_CONFIGS.items():
        sim = make_simulator(config)
        belief0s = _sample_belief0s(sim.hmm, rng, n=12)
        sep = evaluate_separability(sim.hmm, belief0s, n_steps=5, noop_action=NOOP_ACTION[name])
        report[name] = sep.as_dict()
        all_pass = all_pass and sep.passes()
        status = "PASS" if sep.passes() else "FAIL"
        non_noop = [a for a in range(len(sep.action_names)) if a != NOOP_ACTION[name]]
        worst_first = min(sep.tv_vs_noop_first[a] for a in non_noop)
        print(f"{name:16s} {status}  tv_first(worst non-noop action)={worst_first:.3f}")

    report["_all_pass"] = all_pass
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=2))
    print(f"wrote {OUT_PATH}")
    if not all_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
