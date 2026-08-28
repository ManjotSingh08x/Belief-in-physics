"""Phase 4 -- how the belief geometry appears over training, over depth, and what it costs to remove.

Three questions, one pass over the checkpoints written by 03, because the
expensive part (the exact beliefs) is computed once per system and reused.

1. **Emergence.** R^2 per feature group at every checkpoint. Does the metric
   arrive before the belief, or the other way round?
2. **Depth.** The same R^2 at every residual-stream depth, so we can see where
   each quantity is built and whether that layer moves during training.
3. **Cost.** Erase a quantity from the stream, then measure the increase in
   next-token loss against a random subspace of the same rank. Erase, not
   single-shot ablate: removing a probe's row space leaves the feature readable
   by a refit, and the loss after such an intervention describes deleting
   something that is still there. `erasure_basis` iterates until the target is
   undecodable and reports the rank that took.

Also recorded: principal angles between the erased subspaces, and cross-probing
after erasure. Both exist to keep (3) honest -- if the subspaces overlap then
erasing the belief also erases the metric, and the damage numbers cannot be read
as independent without knowing how much they share.

Run:  uv run python experiments/06_analyse_checkpoints.py
Env:  OUTPUT_DIR (default experiments/outputs-03), SYSTEMS, N_EVAL, N_CONTROL,
      MAX_ERASURE_RANK.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root

import json
import os
import time

import numpy as np
import torch

from models.ablation import (
    ablated_loss,
    ablated_stream,
    erasure_basis,
    principal_angles,
    random_basis,
)
from models.analysis import _sequence_split, residual_streams_batched
from models.probe import fit_probe, grouped_r2, shuffled_control
from models.train import pick_device
from models.transformer import ModelConfig, TinyTransformer
from physics.branch import feature_groups, forward_features
from physics.branch_configs import BRANCH_CONFIGS, make_branch_process

N_EVAL = int(os.environ.get("N_EVAL", 512))
N_CONTROL = int(os.environ.get("N_CONTROL", 5))  # random subspaces per rank
MAX_ERASURE_RANK = int(os.environ.get("MAX_ERASURE_RANK", 48))  # half of d_model
TRAIN_FRAC = 0.7
EVAL_SEED = 20_260_828
ABLATION_GROUPS = ("action_lag0", "z0", "metric")

OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "experiments/outputs-03"))
SYSTEMS = os.environ.get("SYSTEMS", ",".join(BRANCH_CONFIGS)).split(",")


def _checkpoints(name: str) -> list[tuple[int, Path]]:
    paths = (OUTPUT_DIR / "checkpoints").glob(f"{name}_*.pt")
    found = []
    for path in paths:
        suffix = path.stem[len(name) + 1 :]
        if suffix.isdigit():
            found.append((int(suffix), path))
    return sorted(found)


def _depth_name(depth: int) -> str:
    return "embedding" if depth == 0 else f"resid_post_{depth - 1}"


def _probe_all_depths(streams, features, groups, train_idx, test_idx, rng):
    """One probe per depth: held-out R^2 per group, plus the shuffled control."""
    records = []
    for depth, activations in enumerate(streams):
        d = activations.shape[-1]
        a_train = activations[train_idx].reshape(-1, d).astype(np.float64)
        a_test = activations[test_idx].reshape(-1, d).astype(np.float64)
        f_train = features[train_idx].reshape(-1, features.shape[-1]).astype(np.float64)
        f_test = features[test_idx].reshape(-1, features.shape[-1]).astype(np.float64)

        probe = fit_probe(a_train, f_train)
        records.append(
            {
                "depth": depth,
                "name": _depth_name(depth),
                "r2_by_group": grouped_r2(probe, a_test, f_test, groups),
                "r2_shuffled_control": shuffled_control(rng, a_train, f_train),
                "probe_weight": probe.weight,  # kept in memory only, for ablation bases
            }
        )
    return records


def _ablate(model, tokens_t, streams, record, groups, rng, features, train_idx, test_idx):
    """Damage per group at one depth, after actually erasing the feature.

    A single-shot ablation of the probe's row space does not remove the target --
    a refit recovers it from redundant directions -- so the loss measured after
    one is the loss of deleting something still present. `erasure_basis` iterates
    until the target is undecodable, and the rank it needs is reported alongside,
    because a control at rank 3 says nothing about an erasure that took rank 20.
    """
    depth = record["depth"]
    base_loss = ablated_loss(model, tokens_t, 0, None)
    activations = streams[depth]
    d_model = activations.shape[-1]
    mean = activations.reshape(-1, d_model).mean(axis=0)

    a_train = activations[train_idx].reshape(-1, d_model).astype(np.float64)
    a_test = activations[test_idx].reshape(-1, d_model).astype(np.float64)
    f_train = features[train_idx].reshape(-1, features.shape[-1]).astype(np.float64)
    f_test = features[test_idx].reshape(-1, features.shape[-1]).astype(np.float64)

    control_cache: dict[int, tuple[float, float]] = {}
    out = {"depth": depth, "name": record["name"], "base_loss": base_loss, "groups": {}}

    for group in ABLATION_GROUPS:
        columns = groups[group]
        intact = record["r2_by_group"][group]
        if not np.isfinite(intact):
            continue
        basis, history = erasure_basis(
            a_train, f_train[:, columns], a_test, f_test[:, columns],
            floor=max(0.02, 0.1 * intact), max_rank=MAX_ERASURE_RANK,
        )
        rank = basis.shape[1]
        if rank == 0:
            continue
        damage = ablated_loss(model, tokens_t, depth, basis, mean) - base_loss

        if rank not in control_cache:
            draws = [
                ablated_loss(model, tokens_t, depth, random_basis(rng, d_model, rank), mean) - base_loss
                for _ in range(N_CONTROL)
            ]
            control_cache[rank] = (float(np.mean(draws)), float(np.std(draws)))
        control, control_sd = control_cache[rank]

        out["groups"][group] = {
            "rank": rank,
            "hit_rank_cap": rank >= MAX_ERASURE_RANK,
            "r2_before": history[0],
            "r2_after": history[-1],
            "delta_loss": damage,
            "delta_loss_random_control": control,
            "delta_loss_random_control_sd": control_sd,
            "excess": damage - control,
            "excess_in_control_sds": (damage - control) / control_sd if control_sd > 0 else float("nan"),
            "basis": basis,  # popped by the caller before serialising
        }
    return out


def _geometry(model, tokens_t, streams, record, features, groups, train_idx, test_idx, erasure):
    """Subspace overlap, and what survives a group's removal, at one depth.

    `principal_angles` near 0 means one group's directions sit inside another's;
    near 90 means they are separate. `cross_r2` is the same question causally:
    fit every group again after removing one, and see which ones went with it.
    """
    depth = record["depth"]
    bases = {g: np.asarray(erasure[g]) for g in ABLATION_GROUPS if g in erasure}
    mean = streams[depth].reshape(-1, streams[depth].shape[-1]).mean(axis=0)

    angles = {}
    for i, a in enumerate(ABLATION_GROUPS):
        for b in ABLATION_GROUPS[i + 1 :]:
            if a in bases and b in bases:
                angles[f"{a}|{b}"] = principal_angles(bases[a], bases[b]).tolist()

    f_train = features[train_idx].reshape(-1, features.shape[-1]).astype(np.float64)
    f_test = features[test_idx].reshape(-1, features.shape[-1]).astype(np.float64)

    cross = {}
    for group, basis in bases.items():
        if basis.shape[1] == 0:
            continue
        stream = ablated_stream(model, tokens_t, depth, basis, mean)
        d = stream.shape[-1]
        probe = fit_probe(stream[train_idx].reshape(-1, d).astype(np.float64), f_train)
        cross[group] = grouped_r2(
            probe, stream[test_idx].reshape(-1, d).astype(np.float64), f_test, groups
        )
    return {"depth": depth, "name": record["name"], "principal_angles_deg": angles, "cross_r2": cross}


def main() -> None:
    device = pick_device()
    training = json.loads((OUTPUT_DIR / "phase2_branch_training.json").read_text())
    print(f"device={device}  systems={SYSTEMS}  n_eval={N_EVAL}", flush=True)

    results = {}
    for name in SYSTEMS:
        t0 = time.perf_counter()
        checkpoints = _checkpoints(name)
        if not checkpoints:
            print(f"!! {name}: no checkpoints under {OUTPUT_DIR / 'checkpoints'}", flush=True)
            continue

        process = make_branch_process(name)
        episodes = process.sample_batch(np.random.default_rng(EVAL_SEED), N_EVAL)
        groups = feature_groups(process)
        print(
            f"\n=== {name} === {len(checkpoints)} checkpoints, "
            f"exact beliefs over {process.n_branches(process.M):,} branches",
            flush=True,
        )
        features = forward_features(process, episodes.tokens)
        tokens_t = torch.as_tensor(episodes.tokens, dtype=torch.long, device=device)

        rng = np.random.default_rng(0)
        train_idx, test_idx = _sequence_split(N_EVAL, TRAIN_FRAC, rng)
        config = ModelConfig(**training[name]["model"])

        curve = []
        for tokens_seen, path in checkpoints:
            model = TinyTransformer(config)
            model.load_state_dict(torch.load(path, map_location=device))
            model = model.to(device).eval()

            streams = residual_streams_batched(model, episodes.tokens, device)
            records = _probe_all_depths(streams, features, groups, train_idx, test_idx, rng)

            # Depth chosen on the headline group, then reused for the causal work
            # so ablation and geometry describe the same place in the network.
            best = max(records, key=lambda r: r["r2_by_group"]["action_lag0"])
            ablation = _ablate(
                model, tokens_t, streams, best, groups, rng, features, train_idx, test_idx
            )
            bases = {g: v.pop("basis") for g, v in ablation["groups"].items()}
            entry = {
                "tokens": tokens_seen,
                "loss": ablated_loss(model, tokens_t, 0, None),
                "best_depth": best["name"],
                "by_depth": [
                    {k: v for k, v in r.items() if k != "probe_weight"} for r in records
                ],
                "ablation": [ablation],
                "geometry": _geometry(
                    model, tokens_t, streams, best, features, groups, train_idx, test_idx, bases
                ),
            }
            curve.append(entry)
            r2 = best["r2_by_group"]
            print(
                f"  {tokens_seen:>11,} tokens  loss={entry['loss']:.4f}  @{best['name']}  "
                f"simplex={r2['action_lag0']:.3f}  z0={r2['z0']:.3f}  metric={r2['metric']:.3f}",
                flush=True,
            )

        results[name] = {
            "n_eval": N_EVAL,
            "n_actions": process.n_actions,
            "n_z0": process.n_z0,
            "feature_groups": {k: [v.start, v.stop] for k, v in groups.items()},
            "curve": curve,
            "wall_seconds": time.perf_counter() - t0,
        }

    (OUTPUT_DIR / "phase4_checkpoint_analysis.json").write_text(
        json.dumps(results, indent=2, default=str)
    )
    print(f"\nwrote {OUTPUT_DIR / 'phase4_checkpoint_analysis.json'}", flush=True)


if __name__ == "__main__":
    main()
