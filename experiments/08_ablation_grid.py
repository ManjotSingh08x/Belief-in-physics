"""Stage 5.1 -- ablation at every depth, every checkpoint, with honest controls.

Phase 4 ran the causal test at one depth per checkpoint, chosen by the argmax of
`action_lag0` R^2. That argmax is an order statistic over five nearly-equal
numbers (pendulum won by 0.0016), it moves during training, and every sign flip
in the phase-4 ablation coincides with it moving. Depth explains the phase-4
pattern at least as well as the belief does, so the pattern cannot be read until
depth is held fixed and swept.

The final residual stream is a special case worth stating: there the only path
to the loss is `logits = W_U LN(x)` with `|V|` between 8 and 24, so at most
`|V| - 1` of 128 directions can move the next-token loss at all. Erasing 35
directions chosen to be uncorrelated with the logits is then guaranteed to cost
nothing, and "load-bearing" collapses into "linearly present in the logits" --
exactly the distinction this project exists to draw. Any ablation reported at
`resid_post_{n_layers-1}` is uninformative by construction.

Four repairs, all in one pass:

**R1** every depth, every checkpoint.
**R2** three nulls, not one. Isotropic random subspaces are the wrong control
because a fitted basis sits in high-variance directions; a shuffled-target INLP
basis is variance-matched and information-free, and the top-r principal
directions are the most destructive variance-matched deletion available.
**R3** a rank ladder from prefixes of the erasure basis, so the belief at rank 35
and the metric at rank 48 can be compared at equal rank.
**R7** `N_CONTROL` raised to 20 with percentiles reported, because a z-score
against 5 draws is not a significance level.

Run:  uv run python experiments/08_ablation_grid.py
Env:  OUTPUT_DIR, SYSTEMS, N_EVAL, N_CONTROL, MAX_ERASURE_RANK, CHECKPOINTS.
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
    erasure_basis,
    intervened_loss,
    mean_ablate,
    pca_basis,
    principal_angles,
    random_basis,
    resample_ablate,
    variance_fraction,
)
from models.analysis import _sequence_split, residual_streams_batched
from models.ablation import _grouped_r2
from models.probe import fit_probe
from models.train import checkpoint_paths, pick_device
from models.transformer import ModelConfig, TinyTransformer
from physics.branch import feature_groups, forward_features
from physics.branch_configs import BRANCH_CONFIGS, make_branch_process

N_EVAL = int(os.environ.get("N_EVAL", 512))
N_CONTROL = int(os.environ.get("N_CONTROL", 20))
MAX_ERASURE_RANK = int(os.environ.get("MAX_ERASURE_RANK", 64))
TRAIN_FRAC = 0.7
EVAL_SEED = 20_260_828
GROUPS = ("action_lag0", "z0", "metric")
RANK_LADDER = (2, 4, 8, 16, 24, 32, 48, 64)

OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "experiments/outputs-03"))
SYSTEMS = os.environ.get("SYSTEMS", ",".join(BRANCH_CONFIGS)).split(",")
ONLY = os.environ.get("CHECKPOINTS")
TAG = os.environ.get("TAG", "")


def _checkpoints(name: str):
    found = checkpoint_paths(OUTPUT_DIR / "checkpoints", name, TAG)
    if ONLY:
        keep = {int(x) for x in ONLY.split(",")}
        found = [f for f in found if f[0] in keep]
    return found


def _depth_name(depth: int) -> str:
    return "embedding" if depth == 0 else f"resid_post_{depth - 1}"


def _summary(draws: list[float]) -> dict:
    a = np.array(draws)
    return {
        "mean": float(a.mean()), "sd": float(a.std()),
        "p05": float(np.quantile(a, 0.05)), "p95": float(np.quantile(a, 0.95)),
        "max": float(a.max()), "n": int(a.size),
    }


def _pca_variance_curve(flat: np.ndarray) -> np.ndarray:
    """Cumulative share of residual variance in the top-r principal directions."""
    centred = flat - flat.mean(axis=0)
    s = np.linalg.svd(centred, full_matrices=False, compute_uv=False)
    power = s**2
    return np.cumsum(power) / power.sum()


def _controls(model, tokens_t, depth, flat, mean, rank, base, rng, cache, widths):
    """The nulls at one rank. Cached: they do not depend on which group is erased.

    Three of them, because the isotropic one is not enough. A basis fitted to a
    real feature sits in high-variance directions, and deleting variance costs
    loss on its own, so an isotropic random subspace of the same rank is a
    strictly easier deletion. `pca_topr` is the opposite extreme -- the most
    destructive rank-r deletion there is -- and `shuffled_inlp` is the same
    fitting procedure run against a target carrying no information, which
    isolates whatever the procedure itself selects for.
    """
    key = (depth, rank)
    if key in cache:
        return cache[key]
    d_model = flat.shape[1]

    random_draws = [
        intervened_loss(model, tokens_t, depth, mean_ablate(random_basis(rng, d_model, rank), mean)) - base
        for _ in range(N_CONTROL)
    ]
    pcs = pca_basis(flat, rank)
    pca_delta = intervened_loss(model, tokens_t, depth, mean_ablate(pcs, mean)) - base

    shuffled = []
    for width in widths:
        fake = rng.normal(size=(flat.shape[0], width))
        basis, _ = erasure_basis(flat, fake, flat[:1], fake[:1], floor=-np.inf, max_rank=rank)
        if basis.shape[1] == 0:
            continue
        shuffled.append(
            intervened_loss(model, tokens_t, depth, mean_ablate(basis, mean)) - base
        )

    out = {
        "random": _summary(random_draws),
        "pca_delta": pca_delta,
        "pca_variance_fraction": variance_fraction(pcs, flat),
        "shuffled_inlp": _summary(shuffled) if shuffled else None,
        "random_variance_fraction": variance_fraction(random_basis(rng, d_model, rank), flat),
    }
    cache[key] = out
    return out


def _variance_matched_pca(model, tokens_t, depth, flat, mean, target_fraction, curve, base, cache):
    """Damage from deleting the *fewest* principal directions that carry as much
    variance as the erasure basis does.

    This is the control the isotropic one cannot be. If erasing 60 fitted
    directions carrying 13% of the variance costs less than erasing the 2
    principal directions carrying the same 13%, then the fitted subspace is
    specifically avoiding the directions the loss depends on, and calling it
    "not load-bearing" is describing the selection, not the model.
    """
    rank_v = int(np.searchsorted(curve, target_fraction) + 1)
    rank_v = max(1, min(rank_v, flat.shape[1]))
    key = ("vpca", depth, rank_v)
    if key not in cache:
        cache[key] = (
            intervened_loss(model, tokens_t, depth, mean_ablate(pca_basis(flat, rank_v), mean)) - base
        )
    return {"rank": rank_v, "delta_loss": cache[key], "variance_fraction": float(curve[rank_v - 1])}


def analyse_checkpoint(model, tokens_t, episodes, features, groups, rng, train_idx, test_idx):
    streams = residual_streams_batched(model, episodes.tokens, tokens_t.device.type)
    base = intervened_loss(model, tokens_t, 0, None)
    n_features = features.shape[-1]
    f_train = features[train_idx].reshape(-1, n_features).astype(np.float64)
    f_test = features[test_idx].reshape(-1, n_features).astype(np.float64)

    out = {"base_loss": base, "by_depth": []}
    for depth, activations in enumerate(streams):
        d_model = activations.shape[-1]
        flat = activations.reshape(-1, d_model)
        mean = flat.mean(axis=0)
        a_train = activations[train_idx].reshape(-1, d_model).astype(np.float64)
        a_test = activations[test_idx].reshape(-1, d_model).astype(np.float64)
        cache: dict = {}
        widths = tuple({groups[g].stop - groups[g].start for g in GROUPS})

        record = {"depth": depth, "name": _depth_name(depth), "groups": {}, "controls": {}}
        curve = _pca_variance_curve(flat)
        bases = {}
        for group in GROUPS:
            cols = groups[group]
            intact = _grouped_r2(fit_probe(a_train, f_train[:, cols]), a_test, f_test[:, cols])
            basis, history = erasure_basis(
                a_train, f_train[:, cols], a_test, f_test[:, cols],
                floor=max(0.02, 0.1 * intact),  # same criterion phase 4 used, for comparability
                max_rank=MAX_ERASURE_RANK,
            )
            rank = basis.shape[1]
            if rank == 0:
                continue
            bases[group] = basis

            ladder = []
            ranks = sorted({r for r in RANK_LADDER if r <= rank} | {rank})
            for r in ranks:
                ctrl = _controls(model, tokens_t, depth, flat, mean, r, base, rng, cache, widths)
                damage = intervened_loss(model, tokens_t, depth, mean_ablate(basis[:, :r], mean)) - base
                vf = variance_fraction(basis[:, :r], flat)
                vpca = _variance_matched_pca(model, tokens_t, depth, flat, mean, vf, curve, base, cache)
                draws = ctrl["random"]
                ladder.append(
                    {
                        "rank": r,
                        "delta_loss": damage,
                        "variance_fraction": vf,
                        "excess_over_random": damage - draws["mean"],
                        "excess_in_control_sds": (damage - draws["mean"]) / draws["sd"] if draws["sd"] > 0 else float("nan"),
                        "exceeds_random_p95": bool(damage > draws["p95"]),
                        "excess_over_pca_topr": damage - ctrl["pca_delta"],
                        "excess_over_variance_matched_pca": damage - vpca["delta_loss"],
                        "variance_matched_pca_rank": vpca["rank"],
                        "excess_over_shuffled_inlp": (
                            damage - ctrl["shuffled_inlp"]["mean"] if ctrl["shuffled_inlp"] else float("nan")
                        ),
                    }
                )
                record["controls"][str(r)] = {
                    "random_delta": draws["mean"], "random_sd": draws["sd"],
                    "random_p95_delta": draws["p95"],
                    "pca_delta": ctrl["pca_delta"],
                    "pca_variance_fraction": ctrl["pca_variance_fraction"],
                    "random_variance_fraction": ctrl["random_variance_fraction"],
                    "shuffled_inlp_delta": (
                        ctrl["shuffled_inlp"]["mean"] if ctrl["shuffled_inlp"] else None
                    ),
                }

            reference = activations[rng.permutation(activations.shape[0])]
            resample = intervened_loss(model, tokens_t, depth, resample_ablate(basis, reference)) - base

            record["groups"][group] = {
                "rank": rank,
                "hit_rank_cap": rank >= MAX_ERASURE_RANK,
                "r2_before": history[0],
                "r2_after": history[-1],
                "variance_fraction": variance_fraction(basis, flat),
                "ladder": ladder,
                "delta_loss_resample": resample,
                "delta_loss": ladder[-1]["delta_loss"],
                "excess": ladder[-1]["excess_over_random"],
                "excess_in_control_sds": ladder[-1]["excess_in_control_sds"],
                "excess_over_variance_matched_pca": ladder[-1]["excess_over_variance_matched_pca"],
            }

        # R4: the leading principal angles say whether the bases are separate
        # objects at all. If they are not, no single-group ablation is about one
        # group, and the complement contrast in experiment 11 is required.
        angles = {}
        for i, a in enumerate(GROUPS):
            for b in GROUPS[i + 1 :]:
                if a in bases and b in bases:
                    ang = principal_angles(bases[a], bases[b])
                    angles[f"{a}|{b}"] = {
                        "mean": float(ang.mean()), "first": float(ang.min()),
                        "n_below_10deg": int((ang < 10).sum()),
                    }
        record["principal_angles_deg"] = angles
        out["by_depth"].append(record)
    return out


def main() -> None:
    device = pick_device()
    training = json.loads((OUTPUT_DIR / "phase2_branch_training.json").read_text())
    print(f"device={device} systems={SYSTEMS} n_eval={N_EVAL} n_control={N_CONTROL} "
          f"max_rank={MAX_ERASURE_RANK}", flush=True)

    results = {}
    for name in SYSTEMS:
        checkpoints = _checkpoints(name)
        if not checkpoints:
            print(f"!! {name}: no checkpoints", flush=True)
            continue
        t0 = time.perf_counter()
        process = make_branch_process(name)
        episodes = process.sample_batch(np.random.default_rng(EVAL_SEED), N_EVAL)
        features = forward_features(process, episodes.tokens)
        groups = feature_groups(process)
        tokens_t = torch.as_tensor(episodes.tokens, dtype=torch.long, device=device)
        rng = np.random.default_rng(0)
        train_idx, test_idx = _sequence_split(N_EVAL, TRAIN_FRAC, rng)
        config = ModelConfig(**training[name]["model"])
        print(f"\n=== {name} === {len(checkpoints)} checkpoints, vocab={config.vocab_size}", flush=True)

        curve = []
        for tokens_seen, path in checkpoints:
            model = TinyTransformer(config)
            model.load_state_dict(torch.load(path, map_location=device))
            model = model.to(device).eval()
            entry = analyse_checkpoint(
                model, tokens_t, episodes, features, groups, rng, train_idx, test_idx
            )
            entry["tokens"] = tokens_seen
            curve.append(entry)
            line = "  ".join(
                f"{d['name'][-2:] if d['depth'] else 'em'}:"
                + ",".join(
                    f"{d['groups'][g]['excess_in_control_sds']:+.0f}" if g in d["groups"] else "--"
                    for g in GROUPS
                )
                for d in entry["by_depth"]
            )
            print(f"  {tokens_seen:>11,}  excess sd (belief,z0,metric) by depth: {line}", flush=True)

        results[name] = {
            "vocab_size": config.vocab_size,
            "n_layers": config.n_layers,
            "d_model": config.d_model,
            "curve": curve,
            "wall_seconds": time.perf_counter() - t0,
        }
        (OUTPUT_DIR / f"phase5_01_ablation_grid{TAG}.json").write_text(
            json.dumps(results, indent=2, default=float)
        )
        print(f"  [{name} done in {results[name]['wall_seconds']:.0f}s]", flush=True)

    print(f"\nwrote {OUTPUT_DIR / 'phase5_01_ablation_grid.json'}", flush=True)


if __name__ == "__main__":
    main()
