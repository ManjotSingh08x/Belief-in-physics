"""Stage 5.5/5.6 -- causal tests that are actually causal.

The phase-4 ablation asks "does deleting this raise the next-token loss". Two
things make that the wrong question here. The belief is nearly orthogonal to the
next-token distribution, so the dependent variable is almost blind to it by
construction; and at the final residual stream the only route to the loss is
`W_U LN(x)` with a vocabulary of 8-24, so most directions are provably inert.
Four experiments, in increasing strength.

**E2a, the decodability horizon (relabelled, not causal).** Fit a linear head
from the stream to the token at t+k, erase the belief, measure the damage as a
function of k. Worth having as a description of the representation, but the
belief is by construction the sufficient statistic for t+k, so damage growing
with k is entailed by the generative model and says nothing about the model. It
is reported as a property, never as evidence of use.

**E2b, position-restricted ablation.** Corrupt the belief subspace at position t
only and read the loss at t+1..t+k, which are themselves untouched but attend to
t. This separates "stored at t for later use" from "recomputed at every
position", which is the actual question. A transformer with full causal
attention can recompute the belief everywhere, so damage confined to position t
is a real and available negative.

**E2c, subspace patching against the exact counterfactual.** Overwrite the belief
component at position t' with the one from position t. The HMM says what should
happen: the model's next-token distribution should move toward `b_t E`. Measure
the alignment between the observed logit change and the predicted one. Ablation
asks whether removing it hurts; patching asks whether changing it moves the
model where the interpretation says it should, and only the second tests that
the subspace has the role attributed to it.

**R4, complement erasure.** With first principal angles of a few degrees the
belief and metric bases are not separate objects, so `metric - belief` and
`belief - metric` at matched rank is the only contrast that separates them.

Run:  uv run python experiments/11_causal.py
Env:  OUTPUT_DIR, SYSTEMS, N_EVAL, K_MAX, N_CONTROL, DEPTHS.
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

from models.ablation import (
    complement_basis,
    erasure_basis,
    intervened_loss,
    logits_under,
    mean_ablate,
    patch_subspace,
    probe_basis,
    principal_angles,
    random_basis,
    restrict_to_positions,
)
from models.analysis import _sequence_split, residual_streams_batched
from models.bootstrap import r2_columns
from models.probe import fit_probe
from models.train import pick_device
from models.transformer import ModelConfig, TinyTransformer
from physics.branch import feature_groups, forward_features
from physics.branch_configs import BRANCH_CONFIGS, make_branch_process
from physics.myopic import predictive_stack

N_EVAL = int(os.environ.get("N_EVAL", 512))
K_MAX = int(os.environ.get("K_MAX", 6))
N_CONTROL = int(os.environ.get("N_CONTROL", 20))
TRAIN_FRAC = 0.7
EVAL_SEED = 20_260_828
GROUPS = ("action_lag0", "metric")
TAG = os.environ.get("TAG", "")
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "experiments/outputs-03"))
SYSTEMS = os.environ.get("SYSTEMS", ",".join(BRANCH_CONFIGS)).split(",")
DEPTHS = os.environ.get("DEPTHS")


def _depth_name(d: int) -> str:
    return "embedding" if d == 0 else f"resid_post_{d - 1}"


def _flat(x, idx):
    return x[idx].reshape(-1, x.shape[-1]).astype(np.float64)


def e2a_decodability_horizon(stream, tokens, basis, mean, train_idx, test_idx):
    """R^2 of a linear head onto the token at t+k, before and after erasure.

    A property of the representation. The one-hot next token is the target so
    that "damage" is on the same footing as the loss-based test.
    """
    d = stream.shape[-1]
    n, L = tokens.shape
    strip = lambda x: x - ((x - mean) @ basis) @ basis.T
    out = []
    for k in range(1, K_MAX + 1):
        if L - k <= 1:
            break
        future = np.zeros((n, L - k, tokens.max() + 1))
        rows, cols = np.arange(n)[:, None], np.arange(L - k)[None]
        future[rows, cols, tokens[:, k:]] = 1.0
        a = stream[:, : L - k]
        fit = lambda x: r2_columns(
            fit_probe(_flat(x, train_idx), _flat(future, train_idx))(_flat(x, test_idx)),
            _flat(future, test_idx),
        )
        out.append({"k": k, "r2_intact": fit(a), "r2_after_erasure": fit(strip(a))})
    return out


def e2b_position_restricted(model, tokens_t, depth, basis, mean, rng, steps_per_segment):
    """Corrupt position t only; read the loss at t+1..t+k.

    Positions are chosen at a fixed phase in the segment so that "how far past a
    kick" is held constant, and the block variant corrupts a run of positions
    because a single one is a small perturbation that can sit under the noise.
    """
    seq_len = tokens_t.shape[1]
    base = intervened_loss(model, tokens_t, depth, None, reduce=False).cpu().numpy()
    d_model = basis.shape[0]
    phase = steps_per_segment - 1  # the most-informed position in each segment
    targets = np.arange(phase, seq_len - K_MAX - 1, steps_per_segment)
    if targets.size == 0:
        return {"runnable": False}

    def sweep(edit_basis, label):
        rows = []
        for block_len in (1, 4):
            positions = np.unique(
                np.concatenate([targets - o for o in range(block_len)])
            )
            positions = positions[positions >= 0]
            edit = restrict_to_positions(mean_ablate(edit_basis, mean), positions)
            per_pos = intervened_loss(model, tokens_t, depth, edit, reduce=False).cpu().numpy()
            delta = per_pos - base
            # Loss index j is the prediction of token j+1 made at position j.
            by_offset = []
            for k in range(0, K_MAX + 1):
                idx = targets + k
                idx = idx[idx < delta.shape[1]]
                by_offset.append({"offset": k, "delta_loss": float(delta[:, idx].mean())})
            rows.append({"block_len": block_len, "label": label, "by_offset": by_offset})
        return rows

    real = sweep(basis, "belief")
    controls = []
    for _ in range(max(3, N_CONTROL // 5)):
        controls.extend(sweep(random_basis(rng, d_model, basis.shape[1]), "random"))

    summary = []
    for block_len in (1, 4):
        r = next(x for x in real if x["block_len"] == block_len)
        c = [x for x in controls if x["block_len"] == block_len]
        for k in range(len(r["by_offset"])):
            draws = np.array([x["by_offset"][k]["delta_loss"] for x in c])
            summary.append(
                {
                    "block_len": block_len, "offset": k,
                    "delta_loss": r["by_offset"][k]["delta_loss"],
                    "control_mean": float(draws.mean()), "control_sd": float(draws.std()),
                    "excess": r["by_offset"][k]["delta_loss"] - float(draws.mean()),
                }
            )
    return {"runnable": True, "n_target_positions": int(targets.size), "by_offset": summary}


def e2c_patching(model, tokens_t, depth, basis, readout, stream, predictive, rng, n_pairs=4096):
    """Patch the belief component from a donor position and check the logits move
    where the HMM says they should -- as a function of how much is patched.

    Donor and recipient share the position index, so the positional embedding and
    the phase in segment are identical and only the history differs. The
    predicted change is `log p_donor - log p_recipient`, both exact.

    Rank has to be swept rather than fixed at the erasure rank. Erasure needs
    ~60 of 128 directions because the feature is stored redundantly, but patching
    60 directions from a donor replaces half the stream, at which point a *random*
    subspace also transplants the donor's computation wholesale and scores a high
    alignment for an uninteresting reason. The informative comparison is at small
    rank, where a random subspace moves the model nowhere in particular: the
    readout basis is `n_actions - 1` directions, the minimum that can carry the
    simplex point at all.
    """
    n, L, d = stream.shape
    donor_idx = rng.permutation(n)
    source = stream[donor_idx]

    before = logits_under(model, tokens_t, depth, None)
    # Two references. The HMM one asks whether the model moves toward the donor's
    # *optimal* predictive distribution; the transplant one asks how much of the
    # model's own donor-vs-recipient logit difference this subspace reproduces,
    # which does not depend on the model matching the optimum and is the standard
    # activation-patching quantity.
    full_donor = logits_under(
        model, tokens_t[torch.as_tensor(donor_idx, device=tokens_t.device)], depth, None
    )
    transplant = (full_donor - before)[:, :-1]
    pred_p = predictive[:, :-1, 0]
    want = np.log(np.clip(pred_p[donor_idx], 1e-9, None)) - np.log(np.clip(pred_p, 1e-9, None))
    live = np.isfinite(want).all(-1)
    idx = np.flatnonzero(live.reshape(-1))
    if idx.size > n_pairs:
        idx = rng.choice(idx, n_pairs, replace=False)
    centre = lambda v: v - v.mean(-1, keepdims=True)
    x = centre(want.reshape(-1, want.shape[-1])[idx])
    xt = centre(transplant.reshape(-1, transplant.shape[-1])[idx])

    def align(b):
        after = logits_under(model, tokens_t, depth, patch_subspace(b, source))
        o = (after - before)[:, :-1]
        y = centre(o.reshape(-1, o.shape[-1])[idx])
        pair = lambda ref: {
            "slope": float((ref * y).sum() / max((ref * ref).sum(), 1e-12)),
            "cosine": float((ref * y).sum() / max(np.linalg.norm(ref) * np.linalg.norm(y), 1e-12)),
        }
        return {
            **pair(x),
            "transplant_slope": pair(xt)["slope"],
            "transplant_cosine": pair(xt)["cosine"],
            "mean_abs_logit_change": float(np.abs(y).mean()),
        }

    rows = []
    ranks = sorted({readout.shape[1], 4, 8, 16, 32, basis.shape[1]} & set(range(1, basis.shape[1] + 1)))
    for r in ranks:
        controls = [align(random_basis(rng, d, r)) for _ in range(3)]
        rows.append(
            {
                "rank": r,
                "belief": align(basis[:, :r]),
                "random_mean_cosine": float(np.mean([c["cosine"] for c in controls])),
                "random_sd_cosine": float(np.std([c["cosine"] for c in controls])),
                "random_mean_transplant_cosine": float(np.mean([c["transplant_cosine"] for c in controls])),
                "random_mean_abs_logit_change": float(np.mean([c["mean_abs_logit_change"] for c in controls])),
            }
        )
    readout_row = {
        "rank": int(readout.shape[1]), "label": "probe_readout",
        "belief": align(readout),
        "random_mean_cosine": float(np.mean([
            align(random_basis(rng, d, readout.shape[1]))["cosine"] for _ in range(3)
        ])),
    }
    return {"by_rank": rows, "readout": readout_row, "n": int(idx.size)}


def main() -> None:
    device = pick_device()
    training = json.loads((OUTPUT_DIR / f"phase2_branch_training{TAG}.json").read_text())
    print(f"device={device} systems={SYSTEMS} k_max={K_MAX}", flush=True)
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
        tokens_t = torch.as_tensor(episodes.tokens, dtype=torch.long, device=device)
        predictive = predictive_stack(process, episodes.tokens, horizon=1).astype(np.float64)

        model = TinyTransformer(config)
        model.load_state_dict(torch.load(OUTPUT_DIR / f"{name}{TAG}_trained.pt", map_location=device))
        model = model.to(device).eval()
        streams = residual_streams_batched(model, episodes.tokens, device)

        # Every depth except the last, where the test has no power: the only
        # path from the stream to the loss is the unembedding, so at most
        # vocab_size - 1 of d_model directions can matter at all.
        depths = (
            [int(x) for x in DEPTHS.split(",")] if DEPTHS else list(range(len(streams) - 1))
        )
        record = {"depths_run": [_depth_name(d) for d in depths],
                  "final_depth_excluded": _depth_name(len(streams) - 1),
                  "vocab_size": config.vocab_size, "by_depth": []}

        for depth in depths:
            act = streams[depth].astype(np.float64)
            mean = act.reshape(-1, act.shape[-1]).mean(0)
            a_tr, a_te = _flat(act, train_idx), _flat(act, test_idx)
            f_tr, f_te = _flat(features, train_idx), _flat(features, test_idx)

            bases, cell = {}, {"depth": depth, "name": _depth_name(depth), "groups": {}}
            for g in GROUPS:
                c = groups[g]
                intact = r2_columns(fit_probe(a_tr, f_tr[:, c])(a_te), f_te[:, c])
                b, hist = erasure_basis(
                    a_tr, f_tr[:, c], a_te, f_te[:, c], floor=max(0.02, 0.1 * intact), max_rank=64
                )
                if b.shape[1] == 0:
                    continue
                bases[g] = b
                cell["groups"][g] = {"rank": int(b.shape[1]), "r2_before": hist[0], "r2_after": hist[-1]}

            if "action_lag0" not in bases:
                record["by_depth"].append(cell)
                continue
            belief_basis = bases["action_lag0"]

            cell["e2a_decodability_horizon"] = e2a_decodability_horizon(
                act, episodes.tokens, belief_basis, mean, train_idx, test_idx
            )
            cell["e2b_position_restricted"] = e2b_position_restricted(
                model, tokens_t, depth, belief_basis, mean, rng, process.steps_per_segment
            )
            readout = probe_basis(fit_probe(a_tr, f_tr).weight, groups["action_lag0"])
            cell["e2c_patching"] = e2c_patching(
                model, tokens_t, depth, belief_basis, readout, act, predictive, rng
            )

            # R4: the contrast that survives the subspaces overlapping.
            if "metric" in bases:
                ang = principal_angles(belief_basis, bases["metric"])
                b_only = complement_basis(belief_basis, bases["metric"])
                m_only = complement_basis(bases["metric"], belief_basis)
                r = min(b_only.shape[1], m_only.shape[1])
                base_loss = intervened_loss(model, tokens_t, depth, None)
                draws = [
                    intervened_loss(model, tokens_t, depth,
                                    mean_ablate(random_basis(rng, act.shape[-1], r), mean)) - base_loss
                    for _ in range(N_CONTROL)
                ] if r > 0 else [0.0]
                # R4 shows the complement is the load-bearing object, so it is
                # also the one to patch: the full erasure basis is dominated by
                # redundant directions the loss does not depend on, and patching
                # 60 of 128 directions transplants the donor's computation
                # wholesale whatever those directions mean.
                if r > 0:
                    cell["e2c_patching_complement"] = e2c_patching(
                        model, tokens_t, depth, b_only[:, :r], readout, act, predictive, rng
                    )
                cell["r4_complements"] = {
                    "first_principal_angle_deg": float(ang.min()),
                    "n_angles_below_10deg": int((ang < 10).sum()),
                    "matched_rank": int(r),
                    "belief_minus_metric_delta": (
                        intervened_loss(model, tokens_t, depth, mean_ablate(b_only[:, :r], mean)) - base_loss
                        if r > 0 else float("nan")
                    ),
                    "metric_minus_belief_delta": (
                        intervened_loss(model, tokens_t, depth, mean_ablate(m_only[:, :r], mean)) - base_loss
                        if r > 0 else float("nan")
                    ),
                    "random_delta_mean": float(np.mean(draws)),
                    "random_delta_sd": float(np.std(draws)),
                }

            record["by_depth"].append(cell)
            p = cell["e2c_patching"]["readout"]
            h = cell["e2a_decodability_horizon"]
            b2 = cell["e2b_position_restricted"]
            print(f"  {_depth_name(depth):<14} rank={belief_basis.shape[1]:>3}  "
                  f"E2c readout rank {p['rank']}: cos={p['belief']['cosine']:+.3f} "
                  f"slope={p['belief']['slope']:+.3f} (random {p['random_mean_cosine']:+.3f})", flush=True)
            print(f"  {'':<14} E2c cosine by patch rank: "
                  + "  ".join(f"r{q['rank']}:{q['belief']['cosine']:+.2f}/{q['random_mean_cosine']:+.2f}"
                              for q in cell["e2c_patching"]["by_rank"]), flush=True)
            if "e2c_patching_complement" in cell:
                q = cell["e2c_patching_complement"]["by_rank"][-1]
                print(f"  {'':<14} E2c COMPLEMENT (belief minus metric) rank {q['rank']}: "
                      f"cos={q['belief']['transplant_cosine']:+.3f} "
                      f"(random {q['random_mean_transplant_cosine']:+.3f})", flush=True)
            print(f"  {'':<14} E2c transplant cosine by rank: "
                  + "  ".join(f"r{q['rank']}:{q['belief']['transplant_cosine']:+.2f}/"
                              f"{q['random_mean_transplant_cosine']:+.2f}"
                              for q in cell["e2c_patching"]["by_rank"]), flush=True)
            print(f"  {'':<14} E2a t+k decodability drop: "
                  + "  ".join(f"k{r['k']}:{r['r2_intact']-r['r2_after_erasure']:+.3f}" for r in h), flush=True)
            if b2["runnable"]:
                print(f"  {'':<14} E2b excess by offset (block=4): "
                      + "  ".join(f"+{r['offset']}:{r['excess']:+.4f}"
                                  for r in b2["by_offset"] if r["block_len"] == 4), flush=True)

        results[name] = {**record, "wall_seconds": time.perf_counter() - t0}
        (OUTPUT_DIR / f"phase5_05_causal{TAG}.json").write_text(json.dumps(results, indent=2, default=float))

    print(f"\nwrote {OUTPUT_DIR / 'phase5_05_causal.json'}", flush=True)


if __name__ == "__main__":
    main()
