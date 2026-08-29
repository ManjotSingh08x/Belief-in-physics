"""Stage 5.7 -- does coupling to the next-token distribution predict emergence?

The phase-4 version of this claim compares two targets across four systems: four
binary outcomes, with the two targets differing in width (1-2 columns versus
3-5), in smoothness, and in how directly the observation reveals them. Coupling
is confounded with all of that, and four ordinal comparisons cannot fail in an
interesting way.

The fix is to make the comparison *within* one model, across a family of targets
that are matched on everything except coupling.

Construction. Every target is an exact linear functional of the belief, of
width 3, of the form

    f_alpha = alpha * (three columns of the emission table) + (1 - alpha) * (a random functional)

The alpha = 1 end *is* the next-token distribution, restricted to three bins, so
its coupling to `p` is 1 by construction. The alpha = 0 end is a random linear
functional of the belief, whose coupling is near 0. Sweeping alpha traces the
axis continuously, at fixed width, fixed scale and fixed smoothness, inside one
model. Coupling is then measured rather than assumed, and emergence time is
regressed on it across ~14 targets instead of compared across 4 runs.

Emergence time is read at a fixed absolute R^2 threshold, not at half of the
final value: the belief has not plateaued at 500M, so its final value is not an
asymptote and "half of it" is not a fixed point on the curve.

Run:  uv run python experiments/12_emergence.py
Env:  OUTPUT_DIR, SYSTEMS, N_EVAL.
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
from models.bootstrap import r2_columns
from models.probe import fit_probe
from models.train import checkpoint_paths, pick_device
from models.transformer import ModelConfig, TinyTransformer
from physics.branch import feature_groups, forward_features
from physics.branch_configs import BRANCH_CONFIGS, make_branch_process, scaled_kick
from physics.myopic import predictive_stack

N_EVAL = int(os.environ.get("N_EVAL", 384))
WIDTH = 3
ALPHAS = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
N_SEEDS = 2
# Thresholds on the *learned gain* R^2(t) - R^2(untrained), not on R^2 itself.
# A random projection of the residual stream already predicts some targets, so an
# absolute threshold reports "emerged at 0 tokens" for anything whose untrained
# baseline clears it, which is not emergence at all.
THRESHOLDS = (0.05, 0.10, 0.20)
MIN_LEARNED_GAIN = 0.10  # below this the target is never learned and has no emergence time
TRAIN_FRAC = 0.7
EVAL_SEED = 20_260_828
KICK_SCALE = float(os.environ.get("KICK_SCALE", 1.0))
TAG = os.environ.get("TAG", "")
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "experiments/outputs-03"))
SYSTEMS = os.environ.get("SYSTEMS", ",".join(BRANCH_CONFIGS)).split(",")


def _checkpoints(name: str):
    return checkpoint_paths(OUTPUT_DIR / "checkpoints", name, TAG)


def matched_targets(process, rng) -> tuple[np.ndarray, list[dict]]:
    """`(n_targets * WIDTH, ...)` tables per segment, and their labels.

    The random half is drawn once on the *full* branch space and marginalised
    down to each level by averaging over completions, so a given target is the
    same functional at every position rather than a new random one each segment.
    """
    n_full = process.n_branches(process.M)
    bins = rng.choice(process.n_obs, WIDTH, replace=False)
    specs, tables = [], []

    for seed in range(N_SEEDS):
        gen = np.random.default_rng(1000 + seed)
        full = gen.normal(size=(n_full, WIDTH))
        full = (full - full.mean(0)) / full.std(0)
        for alpha in ALPHAS:
            specs.append({"alpha": alpha, "seed": seed, "bins": bins.tolist()})
            per_segment = []
            for m in range(process.M):
                level = m + 1
                n_level = process.n_branches(level)
                # average the full-branch functional over its completions
                rand = full.reshape(n_level, -1, WIDTH).mean(axis=1)
                emis = process.emissions[m][:, :, bins]  # (n_level, steps, WIDTH)
                # emission is already scaled like a probability; standardise it so
                # alpha mixes two comparable quantities rather than two scales.
                e = (emis - emis.mean(axis=(0, 1))) / (emis.std(axis=(0, 1)) + 1e-9)
                per_segment.append(alpha * e + (1 - alpha) * rand[:, None, :])
            tables.append(per_segment)
    return tables, specs


def target_values(process, tokens, tables, chunk=64) -> np.ndarray:
    """`E_b[f]` for every target at every position: `(n, L, n_targets * WIDTH)`."""
    n_t = len(tables)
    out = np.empty((tokens.shape[0], process.seq_len, n_t * WIDTH), dtype=np.float64)
    for start in range(0, tokens.shape[0], chunk):
        block = tokens[start : start + chunk]
        nb = block.shape[0]
        for pos, belief in enumerate(process.iter_beliefs(block)):
            m, s = divmod(pos, process.steps_per_segment)
            for j, per_segment in enumerate(tables):
                out[start : start + nb, pos, j * WIDTH : (j + 1) * WIDTH] = (
                    belief @ per_segment[m][:, s]
                )
    return out


def _crossing(tokens_axis, series, threshold):
    """Tokens at which the learned gain first reaches `threshold`, interpolated.

    The gain is measured from the untrained model at the same seed, which is the
    first point on the axis, so a target the random initialisation already
    predicts does not register as emerging instantly.
    """
    y = np.asarray(series) - series[0]
    hit = np.flatnonzero(y >= threshold)
    if hit.size == 0:
        return None
    i = int(hit[0])
    if i == 0:
        return float(tokens_axis[0])
    x0, x1, y0, y1 = tokens_axis[i - 1], tokens_axis[i], y[i - 1], y[i]
    return float(x0 + (x1 - x0) * (threshold - y0) / max(y1 - y0, 1e-12))


def main() -> None:
    device = pick_device()
    training = json.loads((OUTPUT_DIR / f"phase2_branch_training{TAG}.json").read_text())
    print(f"device={device} systems={SYSTEMS} n_eval={N_EVAL} width={WIDTH}", flush=True)
    results = {}

    for name in SYSTEMS:
        checkpoints = _checkpoints(name)
        if not checkpoints:
            continue
        t0 = time.perf_counter()
        print(f"\n=== {name} ===", flush=True)
        process = make_branch_process(name, **scaled_kick(name, KICK_SCALE))
        episodes = process.sample_batch(np.random.default_rng(EVAL_SEED), N_EVAL)
        rng = np.random.default_rng(0)
        train_idx, test_idx = _sequence_split(N_EVAL, TRAIN_FRAC, rng)

        tables, specs = matched_targets(process, rng)
        values = target_values(process, episodes.tokens, tables)
        groups = {f"t{j}": slice(j * WIDTH, (j + 1) * WIDTH) for j in range(len(specs))}

        # The real belief and metric blocks, on the same axis for reference.
        real = forward_features(process, episodes.tokens).astype(np.float64)
        real_groups = feature_groups(process)
        values = np.concatenate([values, real], axis=-1)
        off = len(specs) * WIDTH
        for g, c in real_groups.items():
            groups[f"real_{g}"] = slice(off + c.start, off + c.stop)

        # Coupling: how much of each target the optimal next-token distribution
        # explains. Measured, not assumed, and by the same estimator throughout.
        p = predictive_stack(process, episodes.tokens, horizon=1)[:, :-1, 0].astype(np.float64)
        ptr = p[train_idx].reshape(-1, process.n_obs)
        pte = p[test_idx].reshape(-1, process.n_obs)
        vtr = values[train_idx][:, :-1].reshape(-1, values.shape[-1])
        vte = values[test_idx][:, :-1].reshape(-1, values.shape[-1])
        pred = fit_probe(ptr, vtr)(pte)
        coupling = {g: r2_columns(pred[:, c], vte[:, c]) for g, c in groups.items()}

        curves = {g: [] for g in groups}
        axis = []
        for tokens_seen, path in checkpoints:
            model = TinyTransformer(ModelConfig(**training[name]["model"]))
            model.load_state_dict(torch.load(path, map_location=device))
            model = model.to(device).eval()
            streams = residual_streams_batched(model, episodes.tokens, device)
            best = {g: -np.inf for g in groups}
            for act in streams:
                d = act.shape[-1]
                a_tr = act[train_idx].reshape(-1, d).astype(np.float64)
                a_te = act[test_idx].reshape(-1, d).astype(np.float64)
                f_tr = values[train_idx].reshape(-1, values.shape[-1])
                f_te = values[test_idx].reshape(-1, values.shape[-1])
                pr = fit_probe(a_tr, f_tr)(a_te)
                for g, c in groups.items():
                    best[g] = max(best[g], r2_columns(pr[:, c], f_te[:, c]))
            axis.append(tokens_seen)
            for g in groups:
                curves[g].append(best[g])
            print(f"  {tokens_seen:>11,} done", flush=True)

        rows = []
        for j, spec in enumerate(specs):
            g = f"t{j}"
            rows.append({
                "target": g, **spec, "coupling_to_p": coupling[g],
                "final_r2": curves[g][-1], "untrained_r2": curves[g][0],
                "learned_gain": curves[g][-1] - curves[g][0],
                "crossings": {str(t): _crossing(axis, curves[g], t) for t in THRESHOLDS},
                "curve": curves[g],
            })
        for g in ("real_action_lag0", "real_metric", "real_z0"):
            rows.append({
                "target": g, "alpha": None, "seed": None,
                "coupling_to_p": coupling[g], "final_r2": curves[g][-1],
                "untrained_r2": curves[g][0],
                "learned_gain": curves[g][-1] - curves[g][0],
                "crossings": {str(t): _crossing(axis, curves[g], t) for t in THRESHOLDS},
                "curve": curves[g],
            })

        # The test: across matched targets, does higher coupling mean earlier
        # emergence? Spearman on log-tokens, so a single slow target cannot
        # dominate the fit.
        report = {}
        for t in THRESHOLDS:
            # A target that never gets learned has no emergence time, and
            # including it as a censored point would let the threshold choice
            # decide the correlation.
            pts = [(r["coupling_to_p"], r["crossings"][str(t)])
                   for r in rows
                   if r["alpha"] is not None
                   and r["learned_gain"] > MIN_LEARNED_GAIN
                   and r["crossings"][str(t)] is not None]
            if len(pts) >= 5:
                x = np.array([a for a, _ in pts])
                y = np.log10(np.maximum([b for _, b in pts], 1.0))
                rx, ry = x.argsort().argsort(), y.argsort().argsort()
                rho = float(np.corrcoef(rx, ry)[0, 1])
                report[str(t)] = {
                    "n": len(pts),
                    "n_targets_learned": sum(
                        1 for r in rows if r["alpha"] is not None and r["learned_gain"] > MIN_LEARNED_GAIN
                    ),
                    "spearman_coupling_vs_log_tokens": rho,
                    "note": "positive rho means higher coupling emerges LATER, "
                            "which is the opposite of the phase-4 claim",
                }
            else:
                report[str(t)] = {"n": len(pts), "spearman_coupling_vs_log_tokens": None}

        for r in sorted(rows, key=lambda r: -r["coupling_to_p"]):
            c = r["crossings"][str(THRESHOLDS[0])]
            print(f"  {r['target']:<18} alpha={str(r['alpha']):>5}  coupling={r['coupling_to_p']:+.3f}  "
                  f"final={r['final_r2']:+.3f}  untr={r['untrained_r2']:+.3f}  "
                  f"gain={r['learned_gain']:+.3f}  "
                  f"cross@{THRESHOLDS[0]}={'never' if c is None else f'{c:,.0f}'}", flush=True)
        print(f"  Spearman(coupling, log tokens to reach threshold): "
              + "  ".join(f"{t}:{v['spearman_coupling_vs_log_tokens']}" for t, v in report.items()), flush=True)

        results[name] = {
            "tokens_axis": axis, "targets": rows, "spearman": report,
            "wall_seconds": time.perf_counter() - t0,
        }
        (OUTPUT_DIR / f"phase5_07_emergence{TAG}.json").write_text(json.dumps(results, indent=2, default=float))

    print(f"\nwrote {OUTPUT_DIR / 'phase5_07_emergence.json'}", flush=True)


if __name__ == "__main__":
    main()
