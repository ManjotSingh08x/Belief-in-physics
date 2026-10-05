"""Compare actual k10 token probabilities with belief-weighted physical action ridges."""
import argparse
import hashlib
import json
import os
import sys
from dataclasses import replace
from pathlib import Path
os.environ.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MPLBACKEND="Agg")
import numpy as np
import matplotlib.pyplot as plt


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mix(bins, action_probabilities):
    out = np.zeros((*bins.shape[:-1], 181), np.float64)
    for action in range(4):
        np.put_along_axis(out, bins[..., action, None],
                         np.take_along_axis(out, bins[..., action, None], -1)+action_probabilities[..., action, None], -1)
    assert np.allclose(out.sum(-1), 1, atol=2e-6)
    return out


def main():
    parser = argparse.ArgumentParser()
    for name in ("source", "token-input", "mood-input", "snapshot", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    assert np.allclose(mix(np.array([[5, 5, 7, 8]]), np.array([[.1, .2, .3, .4]]))[0, [5, 7, 8]], [.3, .3, .4])
    snap = json.loads(args.snapshot.read_text())
    for name, value in snap["members"].items():
        if name.endswith((".py", ".ipynb")):
            assert sha(args.source/name) == value
    mh = json.loads((args.mood_input/"REMOTE_SHA256.json").read_text())
    th = json.loads((args.token_input/"SHA256.json").read_text())
    assert sha(args.mood_input/"mood_arrays.npz") == mh["mood_arrays.npz"]
    assert sha(args.token_input/"long_context_k10_probabilities.npz") == th["long_context_k10_probabilities.npz"]
    assert sha(args.token_input/"long_context_k10.json") == th["long_context_k10.json"]
    recipe = json.loads((args.token_input/"long_context_k10.json").read_text())
    with np.load(args.token_input/"long_context_k10_probabilities.npz", allow_pickle=False) as z:
        native, tokens = z["trained_random_probabilities"].copy(), z["random_tokens"].copy()
    with np.load(args.mood_input/"mood_arrays.npz", allow_pickle=False) as z:
        initial, letters = z["initial_states"].copy(), z["letters"].copy()
        oracle, decoded = z["oracle"].copy(), z["trained_projected"].copy()
        emission = z["emission"].copy()
    sys.path.insert(0, str(args.source))
    from physics.messk import MessDriven, MessKProcess
    from physics.systems.sphere_polar import SphereBall
    cfg = recipe["original_experiment"]["physics"]
    proc = MessDriven(chain=MessKProcess(**recipe["original_experiment"]["hmm"]),
        system=SphereBall(**{k:cfg[k] for k in ("g", "length", "gamma", "rate_max")}),
        delta_v=cfg["delta_v"], n_steps=20, dt=.04, integration_dt=.005, m=25, obs_bins=181)
    states = np.empty((4, 500, 4)); z = initial.copy()
    for t in range(500):
        if t % 20 == 0:
            z = proc.system.kick(z, proc.actions[letters[:, t//20]])
        z = proc.flow(z); states[:, t] = z
        assert np.array_equal(proc.observe(z), tokens[:, t])
    source, target = np.arange(490), np.arange(490)+10
    crossing = source//20 != target//20
    def forks(flow):
        q = np.repeat(states[:, :490, None], 4, axis=2)
        for h in range(1, 11):
            kick = (source+h) % 20 == 0
            q[:, kick] = flow.system.kick(q[:, kick], flow.actions[None, None])
            q = flow.flow(q)
        return flow.observe(q), q[..., 0]
    bins, theta = forks(proc)
    fine_bins, fine_theta = forks(replace(proc, integration_dt=.0025))
    assert np.array_equal(bins, fine_bins)
    assert np.all((bins == tokens[:, 10:, None]).any(-1))
    assert np.all(bins[:, ~crossing] == tokens[:, 10:, None][:, ~crossing])
    oracle_action, decoded_action = oracle[:, :490] @ emission, decoded[:, :490] @ emission
    oracle_mass, decoded_mass = mix(bins, oracle_action), mix(bins, decoded_action)
    allowed = np.zeros_like(native, dtype=bool)
    np.put_along_axis(allowed, bins, True, -1)
    distinct = np.sum(np.sort(bins, -1)[..., 1:] != np.sort(bins, -1)[..., :-1], -1)+1
    scores = dict(crossing_positions_per_path=int(crossing.sum()),
        crossing_four_distinct_bins_fraction=float((distinct[:, crossing] == 4).mean()),
        native_probability_on_physical_branches_crossing_mean=float((native*allowed).sum(-1)[:, crossing].mean()),
        oracle_crossing_action_probability_min=float(oracle_action[:, crossing].min()),
        oracle_crossing_action_probability_max=float(oracle_action[:, crossing].max()))
    args.output.mkdir(parents=True, exist_ok=True)
    arrays = dict(source_positions=source, target_positions=target, crosses_kick=crossing,
        physical_states=states, tokens=tokens, letters=letters, emission=emission,
        candidate_bins=bins, candidate_theta=theta, native_probabilities=native,
        oracle_action_probabilities=oracle_action, decoded_action_probabilities=decoded_action,
        oracle_token_probabilities=oracle_mass, decoded_token_probabilities=decoded_mass)
    for path in range(4):
        for zoom in (False, True):
            selection = (target >= 20) & (target < 40) if zoom else np.ones(490, dtype=bool)
            positions = target[selection]
            fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True, layout="constrained")
            panels = ((oracle_mass, "Known state + oracle mood belief → four action ridges"),
                      (native, "Actual trained transformer token head · original softmax"),
                      (decoded_mass, "Known state + decoded mood weights → physical mixture (hybrid diagnostic)"))
            for ax, (p, title) in zip(axes, panels):
                im = ax.imshow(p[path, selection].T, origin="lower", aspect="auto", cmap="magma", vmin=0, vmax=1,
                               interpolation="nearest", extent=(positions[0]-.5, positions[-1]+.5, -.5, 180.5))
                ax.plot(positions, tokens[path, positions], color="cyan", linewidth=1, label="actual target token")
                for action, color in enumerate(("#44ff77", "#ffd43b", "#ff709f", "#73a5ff")):
                    ridge = np.where(crossing[selection], bins[path, selection, action], np.nan)
                    ax.plot(positions, ridge, color=color, linestyle=":", linewidth=1,
                            label=f"action {action} physical ridge" if zoom else None)
                for tick in range(20, 500, 20):
                    if positions[0] <= tick <= positions[-1]:
                        ax.axvline(tick-.5, color="white", linestyle="--", linewidth=.7)
                if zoom:
                    ax.set_ylim(bins[path, selection].min()-6, bins[path, selection].max()+6)
                    ax.legend(ncol=5, fontsize=7, loc="upper right")
                ax.set(title=title, ylabel="target observation token ID")
            axes[-1].set_xlabel("target position j (zero-based); prediction uses history only through source j−10")
            fig.colorbar(im, ax=axes, label="Probability; dotted ridges show candidate locations, not model mass")
            fig.suptitle(f"Sphere n=20, k=10 · random path {path+1} · " +
                ("target20–29: next kick unknown; target30–39: kick already in observed source history" if zoom else
                 "500-token path: branching while lookahead crosses an unseen kick; collapse after kick is observed"))
            fig.savefig(args.output/f"path_{path+1}_{'kick20_zoom' if zoom else '500_tokens'}.png", dpi=150)
            plt.close(fig)
    np.savez_compressed(args.output/"ridge_arrays.npz", **arrays)
    report = dict(experiment_id="sphere-k10-four-ridge-check-20261005", source_revision=snap["source_revision"],
        model_name=recipe["model_name"], configuration_sha256=recipe["configuration_sha256"],
        training_seed=0, n=20, k=10, index_convention="zero-based target j, observed source j-10",
        native_context=310, decoded_probe_context=320, score_contexts_differ=True,
        mood_input_sha256=mh["mood_arrays.npz"], token_input_sha256=th["long_context_k10_probabilities.npz"],
        script_sha256=sha(Path(__file__)), reference_token_agreement=1.0,
        reference_max_theta_error=float(np.max(np.abs(theta-fine_theta))), scores=scores,
        interpretation="Oracle knows exact physical state and hidden action history; hybrid uses known state plus projected linear mood readout; native probabilities untouched")
    (args.output/"summary.json").write_text(json.dumps(report, indent=2)+"\n")
    (args.output/"REMOTE_SHA256.json").write_text(json.dumps({p.name:sha(p) for p in sorted(args.output.iterdir()) if p.is_file() and p.name!="REMOTE_SHA256.json"}, indent=2)+"\n")
    print(json.dumps(scores), flush=True)


if __name__ == "__main__":
    main()
