"""Stage 5.2 -- what the R^2 numbers actually mean.

Every check here is about interpretation, not about a new effect. They run on the
final checkpoint and the untrained control only, because they are about levels
rather than trajectories.

**V2, belief or label?** The interesting claim is that the stream carries the
*posterior* over the hidden kick, not a partial decoding of the kick itself. On
the positions where the posterior's mode disagrees with the truth, those two
targets pull apart, and whichever one the stream follows there is the one it
represents. If it follows the truth, the simplex framing is decoration.

**V3, the token-window baseline.** A bag of the last `steps_per_segment`
observations plus the position in segment has about as many features as
`d_model`. If it matches the transformer, the transformer's R^2 is not evidence
of a learned belief.

**V5/C14, the nonlinear ceiling.** A flat linear R^2 is unfalsifiable on its own:
"not represented" and "not linearly represented" look identical. The MLP settles
which one `sphere` is.

**R6, erasure under a nonlinear probe.** INLP guarantees linear non-decodability
and nothing more.

**C16, entropy stratification.** The belief is near-uniform just after a kick and
near-collapsed at the end of a segment; a pooled R^2 lets the collapsed end,
where the target is nearly a discrete label, carry the headline.

**C17, simplex coordinates.** A block of A marginals has A-1 degrees of freedom.

**V6, sequence bootstrap** on all of it, because n is ~154 held-out sequences.

Run:  uv run python experiments/09_probe_semantics.py
Env:  OUTPUT_DIR, SYSTEMS, N_EVAL, N_BOOT, MLP_EPOCHS.
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

from models.ablation import erasure_basis, variance_fraction
from models.analysis import _sequence_split, residual_streams_batched
from models.bootstrap import bootstrap_r2, r2_columns
from models.probe import fit_probe
from models.probe_extra import mlp_probe, simplex_coords, stratified_r2, token_window_features
from models.train import pick_device
from models.transformer import ModelConfig, TinyTransformer
from physics.branch import feature_groups, forward_features
from physics.branch_configs import BRANCH_CONFIGS, make_branch_process

N_EVAL = int(os.environ.get("N_EVAL", 512))
N_BOOT = int(os.environ.get("N_BOOT", 1000))
MLP_EPOCHS = int(os.environ.get("MLP_EPOCHS", 400))
TRAIN_FRAC = 0.7
EVAL_SEED = 20_260_828
TAG = os.environ.get("TAG", "")
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "experiments/outputs-03"))
SYSTEMS = os.environ.get("SYSTEMS", ",".join(BRANCH_CONFIGS)).split(",")
HEADLINE = "action_lag0"


def _depth_name(d: int) -> str:
    return "embedding" if d == 0 else f"resid_post_{d - 1}"


def _flat(x, idx):
    return x[idx].reshape(-1, x.shape[-1]).astype(np.float64)


def v2_belief_or_label(streams, belief, onehot, train_idx, test_idx, rng):
    """Does the stream track the posterior or the truth, where they disagree?

    Both probes are fitted on all training positions and then read only on the
    held-out positions where `argmax(belief) != argmax(onehot)`. Two readings:
    R^2 against each target, and which of the two the probe's own mode agrees
    with. The second is the discriminative version and is the one to trust,
    because R^2 against a one-hot target is scale-sensitive in a way R^2 against
    a simplex point is not.
    """
    out = []
    for depth, act in enumerate(streams):
        d = act.shape[-1]
        pb = fit_probe(_flat(act, train_idx), _flat(belief, train_idx))
        py = fit_probe(_flat(act, train_idx), _flat(onehot, train_idx))

        b_te, y_te, a_te = belief[test_idx], onehot[test_idx], act[test_idx]
        disagree = b_te.argmax(-1) != y_te.argmax(-1)
        if disagree.sum() < 50:
            continue
        a_d = a_te[disagree].astype(np.float64)
        pred_b, pred_y = pb(a_d), py(a_d)
        out.append(
            {
                "depth": depth, "name": _depth_name(depth),
                "n_disagree": int(disagree.sum()),
                "disagree_fraction": float(disagree.mean()),
                "r2_belief_on_disagreements": r2_columns(pred_b, b_te[disagree]),
                "r2_truth_on_disagreements": r2_columns(pred_y, y_te[disagree]),
                # Fitted to the belief, then asked which mode it lands on.
                "probe_agrees_with_belief": float((pred_b.argmax(-1) == b_te[disagree].argmax(-1)).mean()),
                "probe_agrees_with_truth": float((pred_b.argmax(-1) == y_te[disagree].argmax(-1)).mean()),
                "chance": 1.0 / belief.shape[-1],
            }
        )
    return out


def main() -> None:
    device = pick_device()
    training = json.loads((OUTPUT_DIR / f"phase2_branch_training{TAG}.json").read_text())
    print(f"device={device} systems={SYSTEMS}", flush=True)
    results = {}

    for name in SYSTEMS:
        t0 = time.perf_counter()
        print(f"\n=== {name} ===", flush=True)
        process = make_branch_process(name)
        episodes = process.sample_batch(np.random.default_rng(EVAL_SEED), N_EVAL)
        features = forward_features(process, episodes.tokens).astype(np.float64)
        groups = feature_groups(process)
        rng = np.random.default_rng(0)
        train_idx, test_idx = _sequence_split(N_EVAL, TRAIN_FRAC, rng)
        config = ModelConfig(**training[name]["model"])

        belief = features[:, :, groups[HEADLINE]]
        seg = process.segment_of_position
        onehot = np.zeros_like(belief)
        rows = np.arange(N_EVAL)[:, None]
        onehot[rows, np.arange(process.seq_len)[None], episodes.actions[:, seg]] = 1.0
        entropy = -(belief * np.log(np.clip(belief, 1e-12, None))).sum(-1) / np.log(process.n_actions)

        record = {"n_actions": process.n_actions, "vocab_size": config.vocab_size}

        # V3 -- the dumb baseline, with no model in it at all.
        window = process.steps_per_segment
        bag = token_window_features(
            episodes.tokens, config.vocab_size, window, process.steps_per_segment
        ).astype(np.float64)
        probe = fit_probe(_flat(bag, train_idx), _flat(features, train_idx))
        pred = probe(_flat(bag, test_idx))
        record["v3_token_window"] = {
            "window": window,
            "n_features": bag.shape[-1],
            "d_model": config.d_model,
            "r2_by_group": {
                g: r2_columns(pred[:, c], _flat(features, test_idx)[:, c]) for g, c in groups.items()
            },
        }
        v3 = record["v3_token_window"]["r2_by_group"]
        print(f"  V3 token-window baseline ({bag.shape[-1]} features vs d_model={config.d_model}): "
              f"belief={v3[HEADLINE]:.3f} metric={v3['metric']:.3f} z0={v3['z0']:.3f}", flush=True)

        for tag, filename in (("trained", f"{name}{TAG}_trained.pt"), ("untrained", f"{name}{TAG}_random_init.pt")):
            model = TinyTransformer(config)
            model.load_state_dict(torch.load(OUTPUT_DIR / filename, map_location=device))
            model = model.to(device).eval()
            streams = residual_streams_batched(model, episodes.tokens, device)

            per_depth = []
            for depth, act in enumerate(streams):
                a_tr, a_te = _flat(act, train_idx), _flat(act, test_idx)
                f_tr, f_te = _flat(features, train_idx), _flat(features, test_idx)
                probe = fit_probe(a_tr, f_tr)
                pred = probe(a_te)

                # C17: the same block, scored on independent coordinates.
                helmert = {}
                for g in (HEADLINE, "z0"):
                    c = groups[g]
                    helmert[g] = r2_columns(simplex_coords(pred[:, c]), simplex_coords(f_te[:, c]))

                cell = {
                    "depth": depth, "name": _depth_name(depth),
                    "r2_by_group": {g: r2_columns(pred[:, c], f_te[:, c]) for g, c in groups.items()},
                    "c17_r2_simplex_coords": helmert,
                    "v6_ci_belief": bootstrap_r2(
                        fit_probe(a_tr, _flat(belief, train_idx)),
                        act[test_idx].astype(np.float64), belief[test_idx], rng, N_BOOT,
                    ),
                    "c16_stratified_belief": stratified_r2(
                        fit_probe(a_tr, _flat(belief, train_idx)),
                        a_te, _flat(belief, test_idx), entropy[test_idx].reshape(-1),
                    ),
                    "v5_mlp": {
                        g: mlp_probe(a_tr, f_tr[:, c], a_te, f_te[:, c],
                                     epochs=MLP_EPOCHS, device=device)["r2"]
                        for g, c in ((HEADLINE, groups[HEADLINE]), ("metric", groups["metric"]))
                    },
                }
                per_depth.append(cell)

            best = max(per_depth, key=lambda r: r["r2_by_group"][HEADLINE])
            record[tag] = {
                "by_depth": per_depth,
                "best_depth": best["name"],
                "v2_belief_or_label": v2_belief_or_label(
                    streams, belief, onehot, train_idx, test_idx, rng
                ),
            }

            if tag == "trained":
                # R6: is the target really gone, or only linearly gone?
                depth = best["depth"]
                act = streams[depth]
                a_tr, a_te = _flat(act, train_idx), _flat(act, test_idx)
                b_tr, b_te = _flat(belief, train_idx), _flat(belief, test_idx)
                intact = r2_columns(fit_probe(a_tr, b_tr)(a_te), b_te)
                basis, history = erasure_basis(
                    a_tr, b_tr, a_te, b_te, floor=max(0.02, 0.1 * intact), max_rank=64
                )
                mu = act.reshape(-1, act.shape[-1]).mean(0)
                strip = lambda x: x - ((x - mu) @ basis) @ basis.T
                mlp_before = mlp_probe(a_tr, b_tr, a_te, b_te, epochs=MLP_EPOCHS, device=device)["r2"]
                record["r6_nonlinear_after_erasure"] = {
                    "depth": best["name"], "rank": int(basis.shape[1]),
                    "linear_before": intact, "linear_after": history[-1],
                    "mlp_before": mlp_before,
                    "mlp_after": mlp_probe(strip(a_tr), b_tr, strip(a_te), b_te,
                                           epochs=MLP_EPOCHS, device=device)["r2"],
                }

                # How far the rank has to go before the feature is actually gone,
                # rather than merely unreadable by a line. If the MLP holds up to
                # a rank where the random control is already destroying the model,
                # then no rank-limited linear erasure can isolate this feature and
                # the whole ablation methodology is inapplicable to it -- which is
                # a stronger statement than any individual ablation null.
                ladder = []
                for r in [x for x in (2, 4, 8, 16, 32, 48, 64, 96, 128) if x <= basis.shape[1]]:
                    sub = basis[:, :r]
                    cut = lambda x: x - ((x - mu) @ sub) @ sub.T
                    ladder.append(
                        {
                            "rank": r,
                            "linear": r2_columns(fit_probe(cut(a_tr), b_tr)(cut(a_te)), b_te),
                            "mlp": mlp_probe(cut(a_tr), b_tr, cut(a_te), b_te,
                                             epochs=MLP_EPOCHS, device=device)["r2"],
                            "variance_fraction": variance_fraction(sub, act.reshape(-1, act.shape[-1])),
                        }
                    )
                record["r6_rank_ladder"] = ladder
                print("  R6 rank ladder (linear / MLP): "
                      + "  ".join(f"r{c['rank']}:{c['linear']:.2f}/{c['mlp']:.2f}" for c in ladder),
                      flush=True)
                r6 = record["r6_nonlinear_after_erasure"]
                print(f"  R6 erasure @{r6['depth']} rank {r6['rank']}: linear "
                      f"{r6['linear_before']:.3f}->{r6['linear_after']:.3f}   "
                      f"MLP {r6['mlp_before']:.3f}->{r6['mlp_after']:.3f}", flush=True)

            b = best
            print(f"  {tag:9s} @{b['name']}: linear={b['r2_by_group'][HEADLINE]:.3f} "
                  f"CI[{b['v6_ci_belief']['lo']:.3f},{b['v6_ci_belief']['hi']:.3f}]  "
                  f"simplex-coords={b['c17_r2_simplex_coords'][HEADLINE]:.3f}  "
                  f"MLP={b['v5_mlp'][HEADLINE]:.3f}", flush=True)
            if tag == "trained":
                strat = "  ".join(f"H{s['bin']}:{s['r2']:.2f}" for s in b["c16_stratified_belief"])
                print(f"            C16 by belief entropy (low->high): {strat}", flush=True)
                v2 = max(record[tag]["v2_belief_or_label"], key=lambda r: r["r2_belief_on_disagreements"])
                print(f"            V2 @{v2['name']} on {v2['disagree_fraction']:.0%} disagreements: "
                      f"probe mode agrees with belief {v2['probe_agrees_with_belief']:.3f} vs "
                      f"truth {v2['probe_agrees_with_truth']:.3f} (chance {v2['chance']:.3f})", flush=True)

        record["wall_seconds"] = time.perf_counter() - t0
        results[name] = record
        (OUTPUT_DIR / f"phase5_02_probe_semantics{TAG}.json").write_text(
            json.dumps(results, indent=2, default=float)
        )

    print(f"\nwrote {OUTPUT_DIR / 'phase5_02_probe_semantics.json'}", flush=True)


if __name__ == "__main__":
    main()
