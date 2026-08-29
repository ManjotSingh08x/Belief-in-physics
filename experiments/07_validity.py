"""Stage 5.0 -- the checks that gate everything else.

None of these produce a headline. They decide whether the existing headlines
can be read at all, and three of them can fail in a way that retracts a claim.

**V1, the impossible probe.** Fit the residual stream to `y - b`: the true
one-hot last action minus the exact posterior over it. The model sees exactly
the tokens the filter sees, so it cannot know the filter's own error. Held-out
R^2 must be 0. Anything above 0 means leakage -- train/test contamination,
target misalignment, or a bug in the branch indexing -- and invalidates every
probe number in the project. This is the cheapest possible test of the whole
pipeline and it should have been the first thing written.

**V4, the Bayes floor.** `L - H_floor` is the only comparable loss quantity.
Cross entropy has a large irreducible offset; percentage moves in the raw loss
are meaningless next to percentage moves in an R^2 that floors at zero.

**C12, delta-R^2 per layer.** The residual stream is additive, so R^2 is
monotone non-decreasing in depth by construction and "present at layer L" says
nothing. Only the increment identifies where a quantity is *computed*.

**C19, the metric at its own best depth.** Phase 4 reported the metric at the
depth chosen for `action_lag0`, which can understate it and therefore corrupt
the emergence ordering.

**C21, oracle recovery versus position in segment.** The phase-3 field is
measured at the final position only, i.e. after a full segment of observation.
Resolving whether sphere's information arrives late needs it as a function of
steps since the kick.

Run:  uv run python experiments/07_validity.py
Env:  OUTPUT_DIR, SYSTEMS, N_EVAL, N_BOOT.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import os
import time

import numpy as np
import torch

from models.analysis import _sequence_split, residual_streams_batched
from models.bootstrap import bootstrap_r2, r2_columns, sequence_bootstrap
from models.probe import fit_probe
from models.train import pick_device
from models.transformer import ModelConfig, TinyTransformer
from physics.branch import feature_groups, forward_features
from physics.branch_configs import BRANCH_CONFIGS, make_branch_process
from physics.myopic import bayes_floor

N_EVAL = int(os.environ.get("N_EVAL", 512))
N_BOOT = int(os.environ.get("N_BOOT", 1000))
TRAIN_FRAC = 0.7
EVAL_SEED = 20_260_828
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "experiments/outputs-03"))
SYSTEMS = os.environ.get("SYSTEMS", ",".join(BRANCH_CONFIGS)).split(",")


def _depth_name(depth: int) -> str:
    return "embedding" if depth == 0 else f"resid_post_{depth - 1}"


def v1_impossible_probe(streams, features, groups, episodes, process, train_idx, test_idx, rng):
    """R^2 for a target the model provably cannot know: the filter's own error.

    `y` is the one-hot action that opened the segment each position sits in;
    `b` is the exact posterior over it. `y - b` is orthogonal to every function
    of the token history in expectation, so a probe must score 0.
    """
    seg = process.segment_of_position
    y = np.zeros((episodes.tokens.shape[0], process.seq_len, process.n_actions))
    rows = np.arange(y.shape[0])[:, None]
    y[rows, np.arange(process.seq_len)[None], episodes.actions[:, seg]] = 1.0
    residual = y - features[:, :, groups["action_lag0"]]

    out = []
    for depth, activations in enumerate(streams):
        d = activations.shape[-1]
        probe = fit_probe(
            activations[train_idx].reshape(-1, d).astype(np.float64),
            residual[train_idx].reshape(-1, process.n_actions),
        )
        ci = bootstrap_r2(
            probe, activations[test_idx].astype(np.float64), residual[test_idx], rng, N_BOOT
        )
        # The same probe against the real target, as a positive control: the
        # pipeline must be able to find something when something is there.
        real = fit_probe(
            activations[train_idx].reshape(-1, d).astype(np.float64),
            features[train_idx][:, :, groups["action_lag0"]].reshape(-1, process.n_actions).astype(np.float64),
        )
        ci_real = bootstrap_r2(
            real,
            activations[test_idx].astype(np.float64),
            features[test_idx][:, :, groups["action_lag0"]].astype(np.float64),
            rng,
            200,
        )
        out.append(
            {
                "depth": depth, "name": _depth_name(depth),
                "r2_impossible": ci["point"], "ci": [ci["lo"], ci["hi"]],
                "r2_real_target": ci_real["point"],
            }
        )
    return out


def c21_recovery_by_phase(process, features, episodes, groups) -> dict:
    """Oracle last-action recovery and belief entropy as a function of how many
    observations have arrived since the kick.

    Phase-3 reports this at the final position only. A system whose kick is not
    yet visible in the observable resolves it late, and only this curve can tell
    that apart from never resolving it.
    """
    seg = process.segment_of_position
    truth = episodes.actions[:, seg]  # (n, L)
    simplex = features[:, :, groups["action_lag0"]]
    guess = simplex.argmax(axis=-1)
    ent = -(simplex * np.log(np.clip(simplex, 1e-12, None))).sum(-1) / np.log(process.n_actions)

    steps, recovery, entropy = [], [], []
    for s in range(process.steps_per_segment):
        # Skip segment 0 at s where the prior has not yet split, and pool the rest.
        mask = np.arange(process.seq_len) % process.steps_per_segment == s
        steps.append(s + 1)  # observations available about this segment's kick
        recovery.append(float((guess[:, mask] == truth[:, mask]).mean()))
        entropy.append(float(ent[:, mask].mean()))
    return {
        "observations_since_kick": steps,
        "oracle_recovery": recovery,
        "belief_entropy_normalised": entropy,
        "chance": 1.0 / process.n_actions,
        "final_position_recovery": float((guess[:, -1] == truth[:, -1]).mean()),
    }


def from_phase4(phase4: dict) -> dict:
    """C12 and C19: both are re-readings of data already on disk."""
    out = {}
    for name, d in phase4.items():
        final = d["curve"][-1]
        by_depth = final["by_depth"]
        chosen = final["best_depth"]
        groups = ("action_lag0", "z0", "metric")

        delta = {}
        for g in groups:
            series = [r["r2_by_group"][g] for r in by_depth]
            delta[g] = {
                "profile": series,
                "increments": [series[0]] + list(np.diff(series)),
                "argmax_depth": by_depth[int(np.argmax(series))]["name"],
                "max": float(np.max(series)),
                "at_chosen_depth": next(r for r in by_depth if r["name"] == chosen)["r2_by_group"][g],
            }
        out[name] = {
            "depths": [r["name"] for r in by_depth],
            "chosen_depth": chosen,
            "per_group": delta,
            # The margin the argmax was won by. A depth chosen by 0.002 is not
            # a choice, it is a coin flip that the ablation then inherits.
            "chosen_margin": float(
                np.diff(sorted([r["r2_by_group"]["action_lag0"] for r in by_depth]))[-1]
            ),
            "ablation_depth_by_checkpoint": [
                {"tokens": c["tokens"], "depth": c["best_depth"],
                 "belief_excess_sd": c["ablation"][0]["groups"].get("action_lag0", {}).get("excess_in_control_sds")}
                for c in d["curve"]
            ],
        }
    return out


def main() -> None:
    device = pick_device()
    training = json.loads((OUTPUT_DIR / "phase2_branch_training.json").read_text())
    phase4 = json.loads((OUTPUT_DIR / "phase4_checkpoint_analysis.json").read_text())
    print(f"device={device}  systems={SYSTEMS}  n_eval={N_EVAL}", flush=True)

    results = {"reread_from_phase4": from_phase4(phase4), "systems": {}}

    for name in SYSTEMS:
        t0 = time.perf_counter()
        print(f"\n=== {name} ===", flush=True)
        process = make_branch_process(name)
        episodes = process.sample_batch(np.random.default_rng(EVAL_SEED), N_EVAL)
        features = forward_features(process, episodes.tokens)
        groups = feature_groups(process)
        rng = np.random.default_rng(0)
        train_idx, test_idx = _sequence_split(N_EVAL, TRAIN_FRAC, rng)

        floor = bayes_floor(process, episodes.tokens)
        curve = [
            {
                "tokens": c["tokens"],
                "loss": c["loss"],
                "excess_loss": c["loss"] - floor["plugin"],
            }
            for c in phase4[name]["curve"]
        ]
        print(f"  bayes floor plugin={floor['plugin']:.4f} realised={floor['realised']:.4f} "
              f"uniform={floor['uniform']:.4f}", flush=True)
        for c in curve:
            print(f"    {c['tokens']:>11,}  loss={c['loss']:.4f}  excess={c['excess_loss']:+.4f}", flush=True)

        config = ModelConfig(**training[name]["model"])
        model = TinyTransformer(config)
        model.load_state_dict(torch.load(OUTPUT_DIR / f"{name}_trained.pt", map_location=device))
        model = model.to(device).eval()
        streams = residual_streams_batched(model, episodes.tokens, device)

        v1 = v1_impossible_probe(streams, features, groups, episodes, process, train_idx, test_idx, rng)
        worst = max(v1, key=lambda r: r["r2_impossible"])
        print(f"  V1 impossible probe: worst R^2 = {worst['r2_impossible']:+.4f} "
              f"CI {worst['ci'][0]:+.4f}..{worst['ci'][1]:+.4f} @{worst['name']}  "
              f"(real target at that depth {worst['r2_real_target']:.3f})  "
              f"{'PASS' if worst['ci'][1] < 0.05 else '*** FAIL ***'}", flush=True)

        c21 = c21_recovery_by_phase(process, features, episodes, groups)
        print("  C21 recovery vs observations since kick: "
              + "  ".join(f"{s}:{r:.2f}" for s, r in
                          zip(c21["observations_since_kick"], c21["oracle_recovery"]))
              + f"   (chance {c21['chance']:.2f})", flush=True)

        results["systems"][name] = {
            "bayes_floor": floor,
            "loss_curve": curve,
            "v1_impossible_probe": v1,
            "v1_pass": bool(worst["ci"][1] < 0.05),
            "c21_recovery_by_phase": c21,
            "wall_seconds": time.perf_counter() - t0,
        }

    out = OUTPUT_DIR / "phase5_00_validity.json"
    out.write_text(json.dumps(results, indent=2, default=float))
    print(f"\nwrote {out}", flush=True)

    failed = [n for n, v in results["systems"].items() if not v["v1_pass"]]
    print(f"\nV1 GATE: {'PASS (all systems)' if not failed else 'FAIL on ' + ', '.join(failed)}")


if __name__ == "__main__":
    main()
