"""Frozen-model mood readouts and mood-conditioned physics fans on saved 500-token paths."""
import argparse
import hashlib
import importlib.util
import json
import os
import resource
import sys
from dataclasses import replace
from pathlib import Path

os.environ.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="2", MPLBACKEND="Agg", BELIEF_SPHERE_HORIZON="16")
resource.setrlimit(resource.RLIMIT_AS, (12 * 1024**3, 12 * 1024**3))
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

LABEL = "long_context_k10"
ANCHORS = np.array([19, 159, 319, 479])
COLORS = ["#3690ff", "#ff9233", "#26af65", "#ba59d4"]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def import_file(path):
    spec = importlib.util.spec_from_file_location("existing_diagnostics", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def long_features(model, tokens, context):
    """Native prefix then causal rolling context, retaining all four block outputs."""
    batch, length = tokens.shape
    out = np.empty((batch, length, 512), np.float32)
    prefix = min(context, length)
    out[:, :prefix] = np.concatenate(model.residual_streams(torch.as_tensor(tokens[:, :prefix]))[1:], -1)
    for start in range(prefix, length, 8):
        positions = np.arange(start, min(start+8, length))
        windows = np.stack([tokens[:, t-context+1:t+1] for t in positions]).reshape(-1, context)
        last = np.concatenate([s[:, -1] for s in model.residual_streams(torch.as_tensor(windows))[1:]], -1)
        out[:, positions] = last.reshape(len(positions), batch, 512).transpose(1, 0, 2)
    assert np.isfinite(out).all()
    return out


def self_check(project):
    class Mock:
        def residual_streams(self, tokens):
            a = tokens.numpy()[..., None].astype(np.float32)
            return [np.zeros((*a.shape[:2], 128))] + [np.repeat(a, 128, -1)]*4
    tokens = np.arange(14).reshape(2, 7).astype(np.int64)
    x = long_features(Mock(), tokens, 3)
    assert np.array_equal(x[..., 0], tokens)
    q = project(np.array([[-.2, .2, .3, .7]]))
    assert np.all(q >= 0) and np.allclose(q.sum(-1), 1)
    emission = np.array([[.95, .05], [.05, .95]])
    belief = np.array([.3, .7])
    assert np.allclose(belief @ emission, [.32, .68])
    print("Verified rolling alignment, simplex projection and mood-to-action mixing.")


def plot_moods(arrays, out, tick_only=False):
    positions = np.arange(19, 500, 20) if tick_only else np.arange(500)
    fig, axes = plt.subplots(4, 3, figsize=(18, 11), layout="constrained")
    for path in range(4):
        for col, (key, title) in enumerate((("oracle", "Oracle · hidden-action conditioned"),
                                          ("trained_projected", "Trained · simplex-projected probe"),
                                          ("random_init_projected", "Matched random · simplex-projected probe"))):
            ax = axes[path, col]
            image = ax.imshow(arrays[key][path, positions].T, cmap="magma", vmin=0, vmax=1,
                              aspect="auto", interpolation="nearest", extent=(0, 500, 3.5, -.5))
            ax.axvline(320, color="cyan", linestyle="--", linewidth=1)
            ax.set(title=f"{title} · path {path+1}", ylabel="predictive next-tick mood",
                   xlabel="observed token position", yticks=range(4), yticklabels=[f"Mood {i}" for i in range(4)])
    fig.colorbar(image, ax=axes, label="Projected readout / oracle probability", shrink=.8)
    fig.suptitle("Sphere k=10 · 500-token randomized-start paths · four predictive moods\n" +
                 ("Tick-end readouts: 25 ticks, one estimate per 20 tokens" if tick_only else
                  "Token-resolution readouts: intermediate phases extrapolate a tick-end probe") +
                 " · dashed line: sliding context begins")
    fig.savefig(out / ("mood_heatmaps_tick_ends.png" if tick_only else "mood_heatmaps_500_tokens.png"), dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    for name in ("source", "models", "input", "diagnostics", "snapshot", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    assert digest(args.diagnostics) == "86f2b816c3e7d50a1859f6a76512658cd5bc7176c4074885af7069fb9a51c44e"
    helper = import_file(args.diagnostics)
    self_check(helper.project_simplex)
    if args.self_check:
        return
    args.output.mkdir(parents=True, exist_ok=True)
    snapshot = json.loads(args.snapshot.read_text())
    for name, value in snapshot["members"].items():
        if name.endswith((".py", ".ipynb")):
            assert digest(args.source / name) == value, name
    manifest = json.loads((args.input / "SHA256.json").read_text())
    for name in (LABEL+".json", LABEL+"_probabilities.npz"):
        assert digest(args.input / name) == manifest[name]
    recipe = json.loads((args.input / (LABEL+".json")).read_text())
    sys.path.insert(0, str(args.source))
    from experiments.sphere_vast_train import configuration_sha256, run_name, scope
    from experiments.sphere_staging_probe import sequence_splits, token_features
    ns = scope("half", args.models)
    exp, training = next(pair for pair in ns["ACTIVE_RUNS"] if run_name(*pair) == recipe["model_name"])
    proc = exp.build_process()
    meta_path = args.models / (recipe["model_name"]+".json")
    meta = json.loads(meta_path.read_text())
    assert meta["source_revision"] == snapshot["source_revision"] == recipe["training_source_revision"]
    assert meta["configuration_sha256"] == configuration_sha256(exp, training, proc)
    assert training.k == 10 and proc.n_steps == 20 and meta["model"]["n_ctx"] == 320
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    with np.load(args.input / (LABEL+"_probabilities.npz"), allow_pickle=False) as saved:
        tokens = saved["random_tokens"].copy()
        letters = saved["random_letters"].copy()
        initial = saved["random_initial_states"].copy()
    longer = replace(proc, m=25)
    replay = longer.sample_batch(np.random.default_rng(20261005), 4, initial_state=initial)
    assert np.array_equal(tokens, replay["tokens"]) and np.array_equal(letters, replay["letters"])
    # Reconstruct the original fit only; none of the long random-start targets enter it.
    original = ns["sample_physics_batch"](proc, np.random.default_rng(20260929), meta["probe_trajectories"], exp_cfg=exp)
    tick_ends = np.arange(19, 320, 20)
    parts = sequence_splits(original["tokens"])
    splits = tuple((part[:, None]*16+np.arange(16)).reshape(-1) for part in parts)
    for a in range(3):
        for b in range(a+1, 3):
            assert {r.tobytes() for r in original["tokens"][parts[a]]}.isdisjoint(
                {r.tobytes() for r in original["tokens"][parts[b]]})
    y = original["beliefs"][:, tick_ends].reshape(-1, 4)
    arrays = dict(tokens=tokens, letters=letters, initial_states=initial, oracle=replay["beliefs"],
                  predictive_moods=replay["moods"], emission=proc.chain.E, transition=proc.chain.T,
                  anchors=ANCHORS)
    scores = {}
    for tag, suffix, key in (("trained", ".pth", "checkpoint_sha256"),
                             ("random_init", "_random_init.pt", "random_checkpoint_sha256")):
        weight = args.models / (recipe["model_name"]+suffix)
        assert digest(weight) == meta[key]
        model = training.build_model(proc).eval()
        model.load_state_dict(torch.load(weight, map_location="cpu", weights_only=True))
        chunks = []
        for start in range(0, len(original["tokens"]), 16):
            streams = model.residual_streams(torch.as_tensor(original["tokens"][start:start+16]))
            chunks.append(token_features(streams, tick_ends)[helper.READOUT])
            if start % 512 == 0:
                print(tag, "original features", start, "/", len(original["tokens"]), flush=True)
        x = np.concatenate(chunks).reshape(-1, 512).astype(np.float32)
        expected = next(r for r in meta["probes"] if r["model_type"] == tag and r["mode"] == helper.READOUT and r["target"] == "belief")
        probe, alpha, _ = helper.fit_probe(x, y, splits, float(expected["alpha"]))
        reproduction = float(r2_score(y[splits[2]], probe.predict(x[splits[2]])))
        assert abs(reproduction-float(expected["test_r2"])) < 1e-4
        scaler, ridge = probe.steps[0][1], probe.steps[1][1]
        for name, value in (("mean", scaler.mean_), ("scale", scaler.scale_),
                            ("coef", ridge.coef_), ("intercept", ridge.intercept_)):
            arrays[f"{tag}_probe_{name}"] = value
        xf = long_features(model, tokens, 320)
        raw = probe.predict(xf.reshape(-1, 512)).reshape(4, 500, 4)
        projected = helper.project_simplex(raw)
        assert np.isfinite(raw).all() and np.allclose(projected.sum(-1), 1, atol=1e-6)
        arrays[tag+"_features"] = xf
        arrays[tag+"_raw"], arrays[tag+"_projected"] = raw, projected
        scores[tag] = dict(weight_sha256=meta[key], ridge_alpha=alpha, original_test_r2=reproduction,
                           expected_original_test_r2=float(expected["test_r2"]),
                           tick_end_500_raw_r2=float(r2_score(replay["beliefs"][:, 19::20].reshape(-1, 4), raw[:, 19::20].reshape(-1, 4))),
                           tick_end_500_projected_mse=float(np.mean((projected[:, 19::20]-replay["beliefs"][:, 19::20])**2)),
                           raw_rows_outside_0_1_fraction=float(((raw < 0) | (raw > 1)).any(-1).mean()))
        print(tag, "frozen readout complete", scores[tag], flush=True)
        del chunks, streams, x, xf, model
    # Replay the full physical state, then fork the next kick from the same state.
    z = initial.copy()
    states = np.empty((4, 500, 4))
    for t in range(500):
        if t % 20 == 0:
            z = proc.system.kick(z, proc.actions[letters[:, t//20]])
        z = proc.flow(z); states[:, t] = z
        assert np.array_equal(proc.observe(z), tokens[:, t])
    branches = np.empty((4, len(ANCHORS), 4, 10))
    branch_tokens = np.empty_like(branches, dtype=np.int64)
    for ai, t in enumerate(ANCHORS):
        q = proc.system.kick(np.repeat(states[:, t, None], 4, axis=1), proc.actions[None])
        for h in range(10):
            q = proc.flow(q)
            branches[:, ai, :, h] = proc.system.observable(q)
            branch_tokens[:, ai, :, h] = proc.observe(q)
        for path in range(4):
            assert np.array_equal(branch_tokens[path, ai, letters[path, (t+1)//20]], tokens[path, t+1:t+11])
    arrays["action_branch_theta"] = branches
    arrays["action_branch_tokens"] = branch_tokens
    arrays["mood_conditional_mean_theta"] = np.einsum("ml,palh->pamh", proc.chain.E, branches)
    dominant = proc.chain.E.argmax(-1)
    arrays["dominant_action_by_mood"] = dominant
    mood_token_mass = np.zeros((4, len(ANCHORS), 4, 10, 181))
    for path in range(4):
        for ai in range(len(ANCHORS)):
            for mood in range(4):
                for action in range(4):
                    mood_token_mass[path, ai, mood, np.arange(10), branch_tokens[path, ai, action]] += proc.chain.E[mood, action]
    arrays["mood_conditional_token_mass"] = mood_token_mass
    assert np.allclose(mood_token_mass.sum(-1), 1)
    fine = replace(proc, integration_dt=.0025)
    fine_theta = np.empty_like(branches)
    fine_tokens = np.empty_like(branch_tokens)
    for ai, t in enumerate(ANCHORS):
        q = fine.system.kick(np.repeat(states[:, t, None], 4, axis=1), fine.actions[None])
        for h in range(10):
            q = fine.flow(q)
            fine_theta[:, ai, :, h] = fine.system.observable(q)
            fine_tokens[:, ai, :, h] = fine.observe(q)
    assert np.isfinite(branches).all() and np.array_equal(fine_tokens, branch_tokens)
    branch_reference_error = float(np.max(np.abs(fine_theta-branches)))
    plot_moods(arrays, args.output); plot_moods(arrays, args.output, True)
    for path in range(4):
        fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
        for ai, ax in enumerate(axes.flat):
            t = ANCHORS[ai]
            b = arrays["trained_projected"][path, t]
            for mood, color in enumerate(COLORS):
                ax.plot(np.arange(1, 11), branches[path, ai, dominant[mood]], color=color, linewidth=2,
                        label=f"Mood {mood}: probe weight {b[mood]:.1%}; dominant kick 95%")
                ax.plot(np.arange(1, 11), arrays["mood_conditional_mean_theta"][path, ai, mood],
                        color=color, linestyle=":", linewidth=1)
            ax.plot(np.arange(1, 11), replay["observable"][path, t+1:t+11, 0], color="black", linestyle="--", label="actual continuation")
            ax.set(title=f"Same state after token {t+1}; fork the next kick", xlabel="future observation offset", ylabel="polar angle (rad)")
            ax.legend(fontsize=7)
        fig.suptitle(f"Path {path+1}: four mood-conditioned dominant-action continuations\n"
                     "Physics replay from known simulator state; mood weights from frozen transformer probe · dotted: emission-weighted mean")
        fig.savefig(args.output / f"mood_trajectory_fans_path_{path+1}.png", dpi=150); plt.close(fig)
    # One representative anchor is fixed in advance, not selected for visible separation.
    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True, layout="constrained")
    for mood, ax in enumerate(axes):
        im = ax.imshow(mood_token_mass[0, 2, mood].T, origin="lower", aspect="auto", cmap="magma", vmin=0, vmax=1,
                       extent=(.5, 10.5, -.5, 180.5))
        ax.plot(np.arange(1, 11), tokens[0, 320:330], color="cyan", label="actual future")
        ax.set(title=f"Next-tick Mood {mood} · decoded weight {arrays['trained_projected'][0, 319, mood]:.1%}", ylabel="future token ID")
    axes[-1].set_xlabel("future offset after token 320")
    fig.colorbar(im, ax=axes, label="Conditional probability from known state + HMM emissions")
    fig.suptitle("Four mood-conditioned future distributions · randomized path 1\nPhysics/HMM replay, not the transformer's native token head")
    fig.savefig(args.output / "four_mood_future_heatmaps_path1_token320.png", dpi=150); plt.close(fig)
    np.savez_compressed(args.output / "mood_arrays.npz", **arrays)
    report = dict(experiment_id="sphere-mood-500-k10-20261005", hypothesis_id="H-R2-05-polar-ladder-capacity",
                  model_name=recipe["model_name"], source_revision=meta["source_revision"],
                  configuration_sha256=meta["configuration_sha256"], metadata_sha256=digest(meta_path),
                  source_input_sha256=manifest[LABEL+"_probabilities.npz"], diagnostics_helper_sha256=digest(args.diagnostics),
                  script_sha256=digest(Path(__file__)), runtime_source_files_verified=26,
                  training_seed=0, initial_seed=20261006, action_seed=20261005, original_probe_eval_seed=20260929,
                  original_probe_trajectories=meta["probe_trajectories"], split_counts=[len(part) for part in parts],
                  readout=helper.READOUT, readout_fit_on_long_paths=False, context=320, trajectory_tokens=500,
                  tick_length=20, ticks=25, k=10, anchors_observed_token_counts=(ANCHORS+1).tolist(),
                  python=sys.version, numpy=np.__version__, torch=torch.__version__, scores=scores,
                  acceptance="diagnostic only; one training seed, four OOD starts, projected linear readouts",
                  mood_semantics="predictive next-tick moods; oracle conditions on hidden action letters",
                  forecast_semantics="known simulator state + HMM emission mixture; transformer provides probe mood weights only")
    report["forecast_reference_dt"] = .0025
    report["forecast_reference_token_agreement"] = 1.0
    report["forecast_reference_max_theta_error"] = branch_reference_error
    (args.output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    (args.output / "REMOTE_SHA256.json").write_text(json.dumps({p.name:digest(p) for p in sorted(args.output.iterdir()) if p.is_file() and p.name != "REMOTE_SHA256.json"}, indent=2)+"\n")
    print("complete", flush=True)


if __name__ == "__main__":
    main()
