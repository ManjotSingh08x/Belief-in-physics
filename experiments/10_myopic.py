"""Stages 5.3 and 5.4 -- is the belief more than the next-token distribution?

Claim 1 currently rests on one number: a linear regression of the belief on the
optimal next-token distribution scores R^2 = 0.02 to 0.13 while the residual
stream scores up to 0.59. Three things are wrong with reading that as "the
belief is beyond next-token prediction", and each gets an experiment.

**E1a. It is a linear number, and the residual stream is not a linear function
of the history.** `conditional_variance_ceiling` computes the best possible
predictor of the belief from `p`, of any functional form, by clustering
positions with near-identical predictive distributions and measuring the
within-cluster variance of the belief. Variance inside a cluster is variance no
function of `p` can explain. Swept over the clustering radius, because the
estimate is biased upward at both ends and only the plateau means anything.
Also here: `log p` rather than `p`, since the stream is linearly related to
logits, and dimension-matched controls, since the current table compares a
16-feature predictor against a 128-feature one.

**E1b. The categorical version, which is what Shai et al. actually claim.**
Match pairs of positions whose predictive distributions are within `eps` in
total variation but whose beliefs differ by more than `delta`, and regress the
*difference* in residual stream on the *difference* in belief. Holding `p` fixed
within a pair differences out every myopic component non-parametrically, so a
positive result is immune to both the linearity and the dimensionality
objection. Falsifier: paired R^2 near zero while pooled R^2 is high, which would
mean the pooled number was driven by variation in `p` itself.

**E1c. The deflationary reading that has to be ruled out.** The belief is the
sufficient statistic for the whole future, so `R^2(b | p^(1..K))` rises to 1 in
K; the question is how fast. If K = 3 already reaches 0.9, then next-token
training at positions t+1..t+3 -- which attend to position t through the KV
cache -- is a complete explanation for the belief being present at t, and
"represents more than the objective requires" is false in the way that matters.

Run:  uv run python experiments/10_myopic.py
Env:  OUTPUT_DIR, SYSTEMS, N_EVAL, HORIZON, PAIR_EPS, PAIR_DELTA.
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
from scipy.spatial import cKDTree

from models.analysis import _sequence_split, residual_streams_batched
from models.bootstrap import r2_columns, sequence_bootstrap
from models.probe import fit_probe
from models.train import pick_device
from models.transformer import ModelConfig, TinyTransformer
from physics.branch import feature_groups, forward_features
from physics.branch_configs import BRANCH_CONFIGS, make_branch_process, scaled_kick
from physics.myopic import conditional_variance_ceiling, predictive_stack

N_EVAL = int(os.environ.get("N_EVAL", 512))
HORIZON = int(os.environ.get("HORIZON", 8))
PAIR_EPS = float(os.environ.get("PAIR_EPS", 0.02))
PAIR_DELTA = float(os.environ.get("PAIR_DELTA", 0.20))
MAX_PAIRS = int(os.environ.get("MAX_PAIRS", 200_000))
RADII = (0.005, 0.01, 0.02, 0.04, 0.08)
TRAIN_FRAC = 0.7
EVAL_SEED = 20_260_828
KICK_SCALE = float(os.environ.get("KICK_SCALE", 1.0))
TAG = os.environ.get("TAG", "")
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "experiments/outputs-03"))
SYSTEMS = os.environ.get("SYSTEMS", ",".join(BRANCH_CONFIGS)).split(",")
HEADLINE = "action_lag0"


def _fit_r2(x_tr, y_tr, x_te, y_te) -> float:
    return r2_columns(fit_probe(x_tr, y_tr)(x_te), y_te)


def e1a_ceiling(p, target, rng, n_sample=8000) -> dict:
    """Linear, log-linear, and functional-form-free predictability of the belief
    from the optimal next-token distribution."""
    idx = rng.choice(p.shape[0], min(n_sample, p.shape[0]), replace=False)
    return {
        "radius_sweep": [
            conditional_variance_ceiling(p[idx], target[idx], r, np.random.default_rng(1))
            for r in RADII
        ]
    }


def e1b_matched_pairs(p, belief, stream, seq_id, rng) -> dict:
    """Regress the residual-stream difference on the belief difference, over
    pairs of positions whose predictive distributions coincide.

    Pairs are formed inside a sequence-level split so no sequence contributes to
    both fitting and scoring. Power is reported before the result: if the pairs
    that satisfy `TV < eps` all have `||db|| ~ 0` there is nothing to detect and
    the design is not runnable, which is a fact about the process, not a null.
    """
    # TV = half the L1 distance, so a Chebyshev-free L1 tree is the right index.
    tree = cKDTree(p)
    raw = tree.query_pairs(2.0 * PAIR_EPS, p=1, output_type="ndarray")
    if raw.shape[0] == 0:
        return {"n_candidate_pairs": 0, "runnable": False}
    if raw.shape[0] > MAX_PAIRS:
        raw = raw[rng.choice(raw.shape[0], MAX_PAIRS, replace=False)]

    tv = 0.5 * np.abs(p[raw[:, 0]] - p[raw[:, 1]]).sum(1)
    db = belief[raw[:, 0]] - belief[raw[:, 1]]
    norm = np.linalg.norm(db, axis=1)

    power = {
        "n_candidate_pairs": int(raw.shape[0]),
        "tv_mean": float(tv.mean()),
        "belief_gap_mean": float(norm.mean()),
        "belief_gap_p90": float(np.quantile(norm, 0.9)),
        "fraction_above_delta": float((norm > PAIR_DELTA).mean()),
    }
    keep = norm > PAIR_DELTA
    if keep.sum() < 200:
        return {**power, "runnable": False}

    pairs = raw[keep]
    same = seq_id[pairs[:, 0]] == seq_id[pairs[:, 1]]
    pairs = pairs[~same]  # a within-sequence pair shares its whole history
    if pairs.shape[0] < 200:
        return {**power, "runnable": False}

    # Sign is arbitrary in a difference, so symmetrise: every pair enters twice.
    dx = np.concatenate([stream[pairs[:, 0]] - stream[pairs[:, 1]],
                         stream[pairs[:, 1]] - stream[pairs[:, 0]]])
    dy = np.concatenate([belief[pairs[:, 0]] - belief[pairs[:, 1]],
                         belief[pairs[:, 1]] - belief[pairs[:, 0]]])
    owner = np.concatenate([seq_id[pairs[:, 0]], seq_id[pairs[:, 1]]])

    held = rng.permutation(np.unique(owner))
    cut = int(0.7 * held.size)
    tr = np.isin(owner, held[:cut])
    te = ~tr
    if tr.sum() < 100 or te.sum() < 100:
        return {**power, "runnable": False}

    r2 = _fit_r2(dx[tr], dy[tr], dx[te], dy[te])
    # A shuffled control on the same pairs: if the pairing itself manufactures
    # structure, this will find it too.
    shuffled = _fit_r2(dx[tr], dy[tr][rng.permutation(tr.sum())], dx[te], dy[te])
    return {
        **power, "runnable": True,
        "n_pairs_used": int(pairs.shape[0]),
        "paired_r2": r2,
        "paired_r2_shuffled_control": shuffled,
        "mean_tv_of_used_pairs": float(0.5 * np.abs(p[pairs[:, 0]] - p[pairs[:, 1]]).sum(1).mean()),
        "mean_belief_gap_of_used_pairs": float(np.linalg.norm(belief[pairs[:, 0]] - belief[pairs[:, 1]], axis=1).mean()),
    }


def main() -> None:
    device = pick_device()
    training = json.loads((OUTPUT_DIR / f"phase2_branch_training{TAG}.json").read_text())
    print(f"device={device} systems={SYSTEMS} horizon={HORIZON}", flush=True)
    results = {}

    for name in SYSTEMS:
        t0 = time.perf_counter()
        print(f"\n=== {name} ===", flush=True)
        process = make_branch_process(name, **scaled_kick(name, KICK_SCALE))
        episodes = process.sample_batch(np.random.default_rng(EVAL_SEED), N_EVAL)
        features = forward_features(process, episodes.tokens).astype(np.float64)
        groups = feature_groups(process)
        rng = np.random.default_rng(0)
        train_idx, test_idx = _sequence_split(N_EVAL, TRAIN_FRAC, rng)
        config = ModelConfig(**training[name]["model"])

        stack = predictive_stack(process, episodes.tokens, horizon=HORIZON).astype(np.float64)
        live = ~np.isnan(stack[:, :, -1, 0])  # positions where the whole horizon exists
        belief = features[:, :, groups[HEADLINE]]
        metric = features[:, :, groups["metric"]]

        tr_mask = np.zeros(N_EVAL, bool); tr_mask[train_idx] = True
        seq_of = np.repeat(np.arange(N_EVAL)[:, None], process.seq_len, axis=1)
        sel_tr = live & tr_mask[:, None]
        sel_te = live & ~tr_mask[:, None]

        # --- E1c: how fast does the k-step law determine the belief? ---
        horizon_curve = []
        for k in range(1, HORIZON + 1):
            joint_tr = stack[sel_tr][:, :k].reshape(sel_tr.sum(), -1)
            joint_te = stack[sel_te][:, :k].reshape(sel_te.sum(), -1)
            single_tr = stack[sel_tr][:, k - 1]
            single_te = stack[sel_te][:, k - 1]
            joint2_tr = joint2_te = None
            if k == 2 and process.n_obs ** 2 <= 1024:
                # The belief is a sufficient statistic for the *joint* law of the
                # future, and marginals are not the joint. The pairwise joint is
                # the largest one that fits in n_obs^2 features, and it is the
                # honest version of this test at k = 2.
                outer = lambda a, b: (a[:, :, None] * b[:, None, :]).reshape(a.shape[0], -1)
                joint2_tr = outer(stack[sel_tr][:, 0], stack[sel_tr][:, 1])
                joint2_te = outer(stack[sel_te][:, 0], stack[sel_te][:, 1])
            horizon_curve.append(
                {
                    "k": k,
                    "belief_from_joint_pair": (
                        _fit_r2(joint2_tr, belief[sel_tr], joint2_te, belief[sel_te])
                        if joint2_tr is not None else None
                    ),
                    "belief_from_p1_to_pk": _fit_r2(joint_tr, belief[sel_tr], joint_te, belief[sel_te]),
                    "belief_from_pk_alone": _fit_r2(single_tr, belief[sel_tr], single_te, belief[sel_te]),
                    "metric_from_p1_to_pk": _fit_r2(joint_tr, metric[sel_tr], joint_te, metric[sel_te]),
                    "n_features": int(joint_tr.shape[1]),
                }
            )
        print("  E1c  R^2(belief | p^(1..k)):  "
              + "  ".join(f"k={c['k']}:{c['belief_from_p1_to_pk']:.3f}" for c in horizon_curve), flush=True)
        print("       R^2(metric | p^(1..k)):  "
              + "  ".join(f"k={c['k']}:{c['metric_from_p1_to_pk']:.3f}" for c in horizon_curve), flush=True)

        # --- E1a: the ceiling, plus log p and dimension-matched controls ---
        p1_tr, p1_te = stack[sel_tr][:, 0], stack[sel_te][:, 0]
        logp = lambda x: np.log(np.clip(x, 1e-9, None))
        ceiling = e1a_ceiling(
            np.concatenate([p1_tr, p1_te]), np.concatenate([belief[sel_tr], belief[sel_te]]), rng
        )
        e1a = {
            "belief_from_p_linear": _fit_r2(p1_tr, belief[sel_tr], p1_te, belief[sel_te]),
            "belief_from_log_p_linear": _fit_r2(logp(p1_tr), belief[sel_tr], logp(p1_te), belief[sel_te]),
            "metric_from_p_linear": _fit_r2(p1_tr, metric[sel_tr], p1_te, metric[sel_te]),
            "metric_from_log_p_linear": _fit_r2(logp(p1_tr), metric[sel_tr], logp(p1_te), metric[sel_te]),
            "n_obs": process.n_obs,
            **ceiling,
        }
        print(f"  E1a  belief from p: linear={e1a['belief_from_p_linear']:.3f} "
              f"log-linear={e1a['belief_from_log_p_linear']:.3f}", flush=True)
        print("       ceiling by radius: "
              + "  ".join(f"{c['radius']}:{c['ceiling']:.3f}(k={c['n_clusters']},sing={c['singleton_fraction']:.2f})"
                          for c in ceiling["radius_sweep"]), flush=True)

        # --- the residual stream, at its own best depth, for comparison ---
        model = TinyTransformer(config)
        model.load_state_dict(torch.load(OUTPUT_DIR / f"{name}{TAG}_trained.pt", map_location=device))
        model = model.to(device).eval()
        streams = residual_streams_batched(model, episodes.tokens, device)
        scores = [
            _fit_r2(s[sel_tr].astype(np.float64), belief[sel_tr], s[sel_te].astype(np.float64), belief[sel_te])
            for s in streams
        ]
        depth = int(np.argmax(scores))
        stream = streams[depth].astype(np.float64)

        # Dimension-matched controls: the honest comparison for n_obs features.
        d_model = stream.shape[-1]
        proj = rng.normal(size=(d_model, process.n_obs)) / np.sqrt(d_model)
        centred = stream[sel_tr] - stream[sel_tr].mean(0)
        _, _, vt = np.linalg.svd(centred, full_matrices=False)
        pcs = vt[: process.n_obs].T
        e1a["dimension_controls"] = {
            "belief_from_full_stream": scores[depth],
            "belief_from_random_projection_of_stream": _fit_r2(
                stream[sel_tr] @ proj, belief[sel_tr], stream[sel_te] @ proj, belief[sel_te]
            ),
            "belief_from_top_pcs_of_stream": _fit_r2(
                stream[sel_tr] @ pcs, belief[sel_tr], stream[sel_te] @ pcs, belief[sel_te]
            ),
            "n_components": process.n_obs,
            "depth": "embedding" if depth == 0 else f"resid_post_{depth - 1}",
        }
        dc = e1a["dimension_controls"]
        print(f"  E1a  stream @{dc['depth']}: full={dc['belief_from_full_stream']:.3f}  "
              f"top-{process.n_obs}-PCs={dc['belief_from_top_pcs_of_stream']:.3f}  "
              f"random-{process.n_obs}-proj={dc['belief_from_random_projection_of_stream']:.3f}", flush=True)

        # --- E1b: matched pairs ---
        flat_live = live.reshape(-1)
        e1b = e1b_matched_pairs(
            stack.reshape(-1, HORIZON, process.n_obs)[flat_live][:, 0],
            belief.reshape(-1, process.n_actions)[flat_live],
            stream.reshape(-1, d_model)[flat_live],
            seq_of.reshape(-1)[flat_live],
            rng,
        )
        if e1b["runnable"]:
            print(f"  E1b  {e1b['n_pairs_used']:,} pairs at TV<{PAIR_EPS} with |db|>{PAIR_DELTA}: "
                  f"paired R^2={e1b['paired_r2']:.3f} (shuffled {e1b['paired_r2_shuffled_control']:+.3f}); "
                  f"mean TV={e1b['mean_tv_of_used_pairs']:.4f}, mean |db|={e1b['mean_belief_gap_of_used_pairs']:.3f}",
                  flush=True)
        else:
            print(f"  E1b  NOT RUNNABLE: {e1b['n_candidate_pairs']:,} candidate pairs, "
                  f"{e1b.get('fraction_above_delta', 0):.3f} above delta", flush=True)

        results[name] = {
            "e1a": e1a, "e1b": e1b, "e1c_horizon": horizon_curve,
            "n_positions_full_horizon": int(live.sum()),
            "wall_seconds": time.perf_counter() - t0,
        }
        (OUTPUT_DIR / "phase5_03_myopic.json").write_text(json.dumps(results, indent=2, default=float))

    print(f"\nwrote {OUTPUT_DIR / 'phase5_03_myopic.json'}", flush=True)


if __name__ == "__main__":
    main()
