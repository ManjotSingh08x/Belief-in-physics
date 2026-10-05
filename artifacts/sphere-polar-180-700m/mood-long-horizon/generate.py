"""Extend existing mood forks for 80 seconds under explicit common-future controls."""
import argparse
import csv
import hashlib
import json
import os
import sys
from dataclasses import replace
from pathlib import Path
os.environ.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MPLBACKEND="Agg")
import numpy as np
import matplotlib.pyplot as plt

HORIZON = 2000
COLORS = ["#3690ff", "#ff9233", "#26af65", "#ba59d4"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def persistent_start(values, threshold):
    """First point below threshold through the observed remainder (at least 100 steps)."""
    violations = np.flatnonzero(values > threshold)
    start = int(violations[-1]+1) if len(violations) else 0
    return start if start <= len(values)-101 else None


def distances(states):
    theta, psi = states[..., 0], states[..., 1]
    xyz = np.stack((np.sin(theta)*np.cos(psi), np.sin(theta)*np.sin(psi), np.cos(theta)), -1)
    cart = np.linalg.norm(xyz[:, :, :, None]-xyz[:, :, None, :], axis=-1).max(axis=(2, 3))
    return np.ptp(theta, axis=2), cart


def self_check():
    assert persistent_start(np.r_[0, 1, np.zeros(101)], .1) == 2
    assert persistent_start(np.ones(120), .1) is None
    z = np.zeros((1, 1, 4, 2, 4)); z[..., 0] = .6
    z[:, :, 1, :, 1] = 1
    theta, xyz = distances(z)
    assert np.all(theta == 0) and np.all(xyz > .1)


def main():
    parser = argparse.ArgumentParser()
    for name in ("source", "mood-input", "token-input", "snapshot", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    self_check()
    snap = json.loads(args.snapshot.read_text())
    for name, value in snap["members"].items():
        if name.endswith((".py", ".ipynb")):
            assert sha(args.source/name) == value
    mood_hashes = json.loads((args.mood_input/"REMOTE_SHA256.json").read_text())
    assert sha(args.mood_input/"mood_arrays.npz") == mood_hashes["mood_arrays.npz"]
    token_hashes = json.loads((args.token_input/"SHA256.json").read_text())
    recipe_file = args.token_input/"long_context_k10.json"
    assert sha(recipe_file) == token_hashes[recipe_file.name]
    recipe = json.loads(recipe_file.read_text())
    sys.path.insert(0, str(args.source))
    from physics.messk import MessDriven, MessKProcess
    from physics.systems.sphere_polar import SphereBall, THETA_MIN, THETA_MAX
    cfg = recipe["original_experiment"]["physics"]
    chain_cfg = recipe["original_experiment"]["hmm"]
    proc = MessDriven(chain=MessKProcess(**chain_cfg),
                      system=SphereBall(**{k:cfg[k] for k in ("g", "length", "gamma", "rate_max")}),
                      delta_v=cfg["delta_v"], n_steps=20, dt=cfg["dt"],
                      integration_dt=cfg["integration_dt"], m=125, obs_bins=cfg["obs_bins"])
    with np.load(args.mood_input/"mood_arrays.npz", allow_pickle=False) as data:
        initial, original_letters, observed = data["initial_states"], data["letters"], data["tokens"]
        anchors, weights = data["anchors"], data["trained_projected"][:, data["anchors"]]
        short = data["action_branch_theta"].copy()
    _, letters = proc.chain.sample(np.random.default_rng(20261005), 4, proc.m)
    assert np.array_equal(letters[:, :25], original_letters)
    start_states = np.empty((4, 4, 4)); z = initial.copy()
    for t in range(500):
        if t % 20 == 0:
            z = proc.system.kick(z, proc.actions[letters[:, t//20]])
        z = proc.flow(z)
        assert np.array_equal(proc.observe(z), observed[:, t])
        if t in anchors:
            start_states[:, np.flatnonzero(anchors == t)[0]] = z
    args.output.mkdir(parents=True, exist_ok=True)
    arrays = dict(start_states=start_states, anchors=anchors, mood_weights=weights,
                  common_letters=letters, original_letters=original_letters, observed_tokens=observed)
    summary = dict(experiment_id="sphere-mood-forks-80s-20261005", source_revision=snap["source_revision"],
                   model_name=recipe["model_name"], configuration_sha256=recipe["configuration_sha256"],
                   script_sha256=sha(Path(__file__)), input_mood_arrays_sha256=mood_hashes["mood_arrays.npz"],
                   runtime_source_files_verified=26, horizon_observations=HORIZON, seconds=HORIZON*proc.dt,
                   dt=proc.dt, integration_dt=proc.integration_dt, reference_integration_dt=.0025,
                   chain_seed=20261005, training_seed=0, protocols={})
    bin_width = (proc.obs_ranges[0][1]-proc.obs_ranges[0][0])/180
    offsets = np.arange(HORIZON+1)
    for scenario in ("shared_future_kicks", "no_later_kicks"):
        def evolve(flow):
            q = np.repeat(start_states[:, :, None], 4, axis=2)
            out = np.empty((4, 4, 4, HORIZON+1, 4)); out[..., 0, :] = q
            for step in range(HORIZON):
                if step == 0:
                    q = flow.system.kick(q, flow.actions[None, None])
                elif step % 20 == 0 and scenario == "shared_future_kicks":
                    ticks = (anchors+1)//20+step//20
                    common = flow.actions[letters[:, ticks]]
                    q = flow.system.kick(q, common[:, :, None])
                q = flow.flow(q); out[..., step+1, :] = q
            return out
        states = evolve(proc)
        reference = evolve(replace(proc, integration_dt=.0025))
        assert np.isfinite(states).all() and np.isfinite(reference).all()
        assert np.allclose(states[..., 1:11, 0], short, atol=1e-12)
        assert np.all(states[..., 0] > THETA_MIN) and np.all(states[..., 0] < THETA_MAX)
        assert np.all(np.abs(states[..., 2:]) < proc.system.rate_max)
        tokens, ref_tokens = proc.observe(states), proc.observe(reference)
        spread, cart = distances(states)
        same_bin = np.ptp(tokens, axis=2) == 0
        arrays[scenario+"_states"] = states
        arrays[scenario+"_theta_spread"] = spread
        arrays[scenario+"_xyz_diameter"] = cart
        arrays[scenario+"_tokens"] = tokens
        checks = dict(reference_max_theta_error=float(np.max(np.abs(states[..., 0]-reference[..., 0]))),
                      reference_token_agreement=float(np.mean(tokens == ref_tokens)),
                      observable_bin_clipping_fraction=float(((states[..., 0] < .05) | (states[..., 0] > 1.25)).mean()),
                      polar_or_rate_contacts=0, forks=[])
        for path in range(4):
            for ai in range(4):
                checks["forks"].append(dict(path=path+1, after_token=int(anchors[ai]+1),
                    below_one_bin_through_remainder=persistent_start(spread[path, ai], bin_width),
                    same_token_through_remainder=persistent_start((~same_bin[path, ai]).astype(int), 0),
                    terminal_theta_spread=float(spread[path, ai, -1]),
                    terminal_xyz_diameter=float(cart[path, ai, -1])))
        summary["protocols"][scenario] = checks
        for path in range(4):
            fig, axes = plt.subplots(2, 2, figsize=(14, 9), layout="constrained")
            for ai, ax in enumerate(axes.flat):
                for mood, color in enumerate(COLORS):
                    ax.plot(offsets, states[path, ai, mood, :, 0], color=color, linewidth=1,
                            label=f"Mood {mood} · initial weight {weights[path, ai, mood]:.1%}")
                meeting = checks["forks"][path*4+ai]["below_one_bin_through_remainder"]
                if meeting is not None:
                    ax.axvline(meeting, color="black", linestyle="--", linewidth=1,
                               label=f"spread < one bin from {meeting} ({meeting*.04:.1f}s)")
                ax.set(title=f"Fork after token {anchors[ai]+1}", xlabel="future observations", ylabel="polar angle (rad)")
                ax.legend(fontsize=7)
            fig.suptitle(f"Path {path+1} · 2000 future observations / 80 seconds · {scenario.replace('_',' ')}\n"
                         "Initial mood fork followed by an explicit common-future control; physics replay, not a new transformer horizon")
            fig.savefig(args.output/f"{scenario}_path_{path+1}.png", dpi=150); plt.close(fig)
        print(scenario, checks["reference_max_theta_error"], "complete", flush=True)
    # Close-up and separation for the same preselected fork after token 320, path 1.
    fig, axes = plt.subplots(2, 3, figsize=(18, 9), layout="constrained")
    for row, scenario in enumerate(summary["protocols"]):
        states = arrays[scenario+"_states"][0, 2]
        spread = arrays[scenario+"_theta_spread"][0, 2]
        cart = arrays[scenario+"_xyz_diameter"][0, 2]
        for mood, color in enumerate(COLORS):
            for ax, limit in ((axes[row, 0], 150), (axes[row, 1], HORIZON)):
                ax.plot(offsets[:limit+1]*.04, states[mood, :limit+1, 0], color=color, linewidth=1.5, label=f"Mood {mood}")
        axes[row, 0].set(title=scenario.replace('_',' ')+" · first 6 seconds", xlabel="future seconds", ylabel="polar angle (rad)")
        axes[row, 1].set(title="Full 80-second horizon", xlabel="future seconds", ylabel="polar angle (rad)")
        axes[row, 2].semilogy(offsets*.04, np.maximum(spread, 1e-12), label="polar-angle spread (rad)")
        axes[row, 2].semilogy(offsets*.04, np.maximum(cart, 1e-12), label="3D position diameter (sphere radii)")
        axes[row, 2].axhline(bin_width, color="black", linestyle="--", label="one observation-bin width")
        axes[row, 2].set(title="Angle agreement versus physical separation", xlabel="future seconds", ylabel="separation (log scale)")
        for ax in axes[row]:
            ax.legend(fontsize=8)
    fig.suptitle("Same fork after token 320 · randomized path 1 · four mood-conditioned dominant-action branches\n"
                 "Common later inputs isolate forgetting of the first kick; matching polar angles does not imply matching azimuths")
    fig.savefig(args.output/"long_horizon_convergence_path1_token320.png", dpi=150); plt.close(fig)
    np.savez_compressed(args.output/"long_horizon_arrays.npz", **arrays)
    with (args.output/"convergence.csv").open("w", newline="") as f:
        rows = [dict(protocol=protocol, **fork) for protocol, checks in summary["protocols"].items() for fork in checks["forks"]]
        writer = csv.DictWriter(f, fieldnames=rows[0].keys(), lineterminator="\n"); writer.writeheader(); writer.writerows(rows)
    (args.output/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    (args.output/"REMOTE_SHA256.json").write_text(json.dumps({p.name:sha(p) for p in sorted(args.output.iterdir()) if p.is_file() and p.name!="REMOTE_SHA256.json"}, indent=2)+"\n")
    print("complete", flush=True)


if __name__ == "__main__":
    main()
