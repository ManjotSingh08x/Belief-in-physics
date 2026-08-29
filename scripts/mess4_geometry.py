"""Belief geometry of the no-no-op Mess-4 variants, against their parents.

No transformer anywhere in here. The belief simplex is a property of the
*process*, so this runs on CPU and answers "what geometry does this action set
produce" before any GPU time is spent on learning it.

Three things are measured per process, all from the exact forward algorithm:

  geometry     the action_lag0 marginal as a point cloud -- a 2-simplex for the
               3-action parents (the classic Mess3 triangle) and a 3-simplex for
               the Mess-4 variants
  identifiability  oracle recovery of the true last action from the exact
               posterior, and normalised belief entropy, both as a function of
               how many observations have arrived since the kick
  metric image how that cloud maps into E[m], which is collinear for the graded
               pendulum ladder and two-dimensional for the predator-prey cross

Run:  uv run python scripts/mess4_geometry.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from physics.branch import feature_groups, forward_features  # noqa: E402
from physics.branch_configs import make_branch_process  # noqa: E402

PAIRS = [("pendulum", "pendulum_mess4"), ("predator_prey", "predator_prey_mess4")]
N_SEQ = 256
OUT = Path("experiments/outputs-mess4")


def simplex_embedding(a: int) -> np.ndarray:
    """Vertices of a regular (a-1)-simplex, so barycentric coords plot directly."""
    if a == 3:
        ang = np.array([90.0, 210.0, 330.0]) * np.pi / 180.0
        return np.stack([np.cos(ang), np.sin(ang)], axis=1)
    if a == 4:
        v = np.array([[1.0, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]])
        return v / np.sqrt(3.0)
    raise ValueError(f"no embedding for {a} actions")


def collect(name: str, seed: int = 0) -> dict:
    proc = make_branch_process(name)
    rng = np.random.default_rng(seed)
    ep = proc.sample_batch(rng, N_SEQ)
    feats = forward_features(proc, ep.tokens)
    g = feature_groups(proc)

    a, spp = proc.n_actions, proc.steps_per_segment
    simplex = feats[:, :, g["action_lag0"]].reshape(-1, a)
    metric = feats[:, :, g["metric"]].reshape(-1, feats.shape[-1] - g["metric"].start)

    seg = proc.segment_of_position
    since = np.tile(np.arange(spp) + 1, proc.M)  # observations since the kick
    truth = ep.actions[:, seg].reshape(-1)  # true last action at each position
    since_flat = np.tile(since, (N_SEQ, 1)).reshape(-1)

    ent = -(simplex * np.log(np.clip(simplex, 1e-12, None))).sum(1) / np.log(a)
    hit = simplex.argmax(1) == truth

    by_since = [
        {
            "obs_since_kick": int(s),
            "entropy_normalised": float(ent[since_flat == s].mean()),
            "oracle_recovery": float(hit[since_flat == s].mean()),
        }
        for s in range(1, spp + 1)
    ]

    # How much of the simplex cloud is visible to the metric? Regress each
    # principal direction of the cloud onto E[m]; a collinear action set puts
    # essentially all of it on one direction.
    centred = simplex - simplex.mean(0)
    _, sv, vt = np.linalg.svd(centred, full_matrices=False)
    pcs = centred @ vt[: a - 1].T
    m0 = metric[:, 0] - metric[:, 0].mean()
    corr = [float(abs(np.corrcoef(pcs[:, i], m0)[0, 1])) for i in range(a - 1)]

    return {
        "name": name,
        "n_actions": a,
        "M": proc.M,
        "seq_len": proc.seq_len,
        "n_branches": int(proc.n_branches(proc.M)),
        "chance": 1.0 / a,
        "by_obs_since_kick": by_since,
        "entropy_mean": float(ent.mean()),
        "oracle_recovery_mean": float(hit.mean()),
        "recovery_over_chance": float(hit.mean() * a),
        "simplex_pc_variance_fraction": (sv[: a - 1] ** 2 / (sv**2).sum()).tolist(),
        "abs_corr_pc_with_metric": corr,
        "_simplex": simplex,
        "_metric": metric,
        "_since": since_flat,
        "_truth": truth,
        "_pcs": pcs,
    }


def _scatter_simplex(ax, d, colour, cmap, label):
    verts = simplex_embedding(d["n_actions"])
    xy = d["_simplex"] @ verts
    if verts.shape[1] == 2:
        for i in range(len(verts)):  # outline
            j = (i + 1) % len(verts)
            ax.plot(*zip(verts[i], verts[j]), color="0.75", lw=0.8, zorder=1)
        sc = ax.scatter(xy[:, 0], xy[:, 1], c=colour, s=1.2, cmap=cmap, alpha=0.35, lw=0)
        ax.set_aspect("equal")
        ax.axis("off")
    else:
        for i in range(4):
            for j in range(i + 1, 4):
                ax.plot(*zip(verts[i], verts[j]), color="0.8", lw=0.7, zorder=1)
        sc = ax.scatter(xy[:, 0], xy[:, 1], xy[:, 2], c=colour, s=1.0, cmap=cmap, alpha=0.3, lw=0)
        ax.set_axis_off()
        ax.view_init(elev=18, azim=35)
    ax.set_title(label, fontsize=9)
    return sc


def figure(parent: dict, child: dict, path: Path) -> None:
    fig = plt.figure(figsize=(13.5, 7.2))
    fig.suptitle(
        f"{parent['name']} -> {child['name']}: belief geometry of the action_lag0 marginal\n"
        f"{parent['n_actions']} actions (with no-op) vs {child['n_actions']} actions (none), "
        f"variance-matched",
        fontsize=11,
    )
    for col, d in enumerate((parent, child)):
        proj = "3d" if d["n_actions"] == 4 else None
        ax = fig.add_subplot(2, 3, 1 + 3 * col, projection=proj)
        sc = _scatter_simplex(ax, d, d["_since"], "viridis", f"{d['name']}\ncoloured by obs since kick")
        fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)

        ax = fig.add_subplot(2, 3, 2 + 3 * col, projection=proj)
        _scatter_simplex(ax, d, d["_truth"], "tab10", "coloured by TRUE last action")

        ax = fig.add_subplot(2, 3, 3 + 3 * col)
        m = d["_metric"]
        if m.shape[1] >= 2:
            ax.scatter(m[:, 0], m[:, 1], c=d["_truth"], s=1.0, cmap="tab10", alpha=0.3, lw=0)
            ax.set_xlabel("E[m] dim 0", fontsize=8)
            ax.set_ylabel("E[m] dim 1", fontsize=8)
        else:
            ax.scatter(m[:, 0], d["_pcs"][:, 0], c=d["_truth"], s=1.0, cmap="tab10", alpha=0.3, lw=0)
            ax.set_xlabel("E[m]", fontsize=8)
            ax.set_ylabel("simplex PC1", fontsize=8)
        ax.set_title("metric image, |corr(PC, E[m])| = "
                     + ", ".join(f"{c:.2f}" for c in d["abs_corr_pc_with_metric"]), fontsize=8)
        ax.tick_params(labelsize=7)

    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def curves(records: list[dict], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for d in records:
        x = [r["obs_since_kick"] for r in d["by_obs_since_kick"]]
        ls = "--" if "mess4" not in d["name"] else "-"
        axes[0].plot(x, [r["entropy_normalised"] for r in d["by_obs_since_kick"]], ls, label=d["name"])
        axes[1].plot(x, [r["oracle_recovery"] for r in d["by_obs_since_kick"]], ls, label=d["name"])
        axes[1].axhline(d["chance"], color="0.8", lw=0.6, zorder=0)
    axes[0].axhspan(0.50, 0.76, color="0.9", zorder=0)
    axes[0].set_ylabel("normalised belief entropy")
    axes[1].set_ylabel("oracle recovery of last action")
    for ax in axes:
        ax.set_xlabel("observations since kick")
        ax.legend(fontsize=7)
        ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    records, summary = [], []
    for parent_name, child_name in PAIRS:
        parent, child = collect(parent_name), collect(child_name)
        figure(parent, child, OUT / f"mess4_geometry_{parent_name}.png")
        records += [parent, child]
        for d in (parent, child):
            print(
                f"{d['name']:22s} A={d['n_actions']} branches={d['n_branches']:>6d} "
                f"entropy={d['entropy_mean']:.3f} recovery={d['oracle_recovery_mean']:.3f} "
                f"({d['recovery_over_chance']:.2f}x chance) "
                f"PC var={['%.2f' % v for v in d['simplex_pc_variance_fraction']]} "
                f"|corr with E[m]|={['%.2f' % c for c in d['abs_corr_pc_with_metric']]}"
            )
            summary.append({k: v for k, v in d.items() if not k.startswith("_")})
    curves(records, OUT / "mess4_identifiability.png")
    (OUT / "mess4_geometry.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"wrote {OUT}/")


if __name__ == "__main__":
    main()
