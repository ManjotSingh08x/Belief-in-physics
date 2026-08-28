"""Phase 4 -- how the belief geometry appears over training, over depth, and what it costs to remove.

Three questions, one pass over the checkpoints written by 03, because the
expensive part (the exact beliefs) is computed once per system and reused.

1. **Emergence.** R^2 per feature group at every checkpoint. Does the metric
   arrive before the belief, or the other way round?
2. **Depth.** The same R^2 at every residual-stream depth, so we can see where
   each quantity is built and whether that layer moves during training.
3. **Cost.** Mean-ablate the subspace a probe reads and measure the increase in
   next-token loss, against a random subspace of the same rank. Decodable is
   not the same as used, and only the excess over the control is interpretable.

Also recorded: principal angles between the group subspaces, and cross-probing
after ablation. Both exist to keep (3) honest -- the three subspaces overlap, so
ablating the belief partly ablates the metric, and the raw damage numbers cannot
be read as independent without knowing how much they share.

Run:  uv run python experiments/06_analyse_checkpoints.py
Env:  OUTPUT_DIR (default experiments/outputs-03), SYSTEMS, N_EVAL, N_CONTROL.
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
    principal_angles,
    probe_basis,
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


def _ablate(model, tokens_t, streams, records, groups, rng):
    """Damage per group per depth, minus a matched-rank random-subspace control."""
    base_loss = ablated_loss(model, tokens_t, 0, None)
    control_cache: dict[tuple[int, int], float] = {}
    out = []

    for record in records:
        depth = record["depth"]
        mean = streams[depth].reshape(-1, streams[depth].shape[-1]).mean(axis=0)
        d_model = mean.shape[0]
        entry = {"depth": depth, "name": record["name"], "base_loss": base_loss, "groups": {}}

        for group in ABLATION_GROUPS:
            basis = probe_basis(record["probe_weight"], groups[group])
            rank = basis.shape[1]
            if rank == 0:
                continue
            damage = ablated_loss(model, tokens_t, depth, basis, mean) - base_loss

            key = (depth, rank)
            if key not in control_cache:
                control_cache[key] = float(
                    np.mean(
                        [
                            ablated_loss(model, tokens_t, depth, random_basis(rng, d_model, rank), mean)
                            for _ in range(N_CONTROL)
                        ]
                    )
                    - base_loss
                )
            entry["groups"][group] = {
                "rank": rank,
                "delta_loss": damage,
                "delta_loss_random_control": control_cache[key],
                "excess": damage - control_cache[key],
            }
        out.append(entry)
    return out


def _geometry(model, tokens_t, streams, record, features, groups, train_idx, test_idx):
    """Subspace overlap, and what survives a group's removal, at one depth.

    `principal_angles` near 0 means one group's directions sit inside another's;
    near 90 means they are separate. `cross_r2` is the same question causally:
    fit every group again after removing one, and see which ones went with it.
    """
    depth = record["depth"]
    weight = record["probe_weight"]
    bases = {g: probe_basis(weight, groups[g]) for g in ABLATION_GROUPS}
    mean = streams[depth].reshape(-1, streams[depth].shape[-1]).mean(axis=0)

    angles = {}
    for i, a in enumerate(ABLATION_GROUPS):
        for b in ABLATION_GROUPS[i + 1 :]:
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
            entry = {
                "tokens": tokens_seen,
                "loss": ablated_loss(model, tokens_t, 0, None),
                "best_depth": best["name"],
                "by_depth": [
                    {k: v for k, v in r.items() if k != "probe_weight"} for r in records
                ],
                "ablation": _ablate(model, tokens_t, streams, records, groups, rng),
                "geometry": _geometry(
                    model, tokens_t, streams, best, features, groups, train_idx, test_idx
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
