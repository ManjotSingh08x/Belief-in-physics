"""Actual k-ahead token distributions on 500-token Sphere trajectories; CPU only."""
import argparse
import hashlib
import json
import os
import resource
import sys
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

os.environ.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="2", MPLBACKEND="Agg", BELIEF_SPHERE_HORIZON="16")
resource.setrlimit(resource.RLIMIT_AS, (12 * 1024**3, 12 * 1024**3))
import numpy as np
import torch
import matplotlib.pyplot as plt

REVISION = "dc80726a8c987e8c3a9171299dac00d03a074311"
CASES = (("long_context_k10", "half", "sphere_polar_c11_n_up_gamma_constant_d0p14_a0p95_m16_d128l4h2mlp512_700m_seed0_k10"),
         ("long_context_k21", "n_plus_one", "sphere_polar_c11_n_up_gamma_constant_d0p14_a0p95_m16_d128l4h2mlp512_700m_seed0_k21"),
         ("belief_winner_k2", "half", "sphere_polar_c08_n_down_gamma_up_d0p14_a0p95_m16_d128l4h2mlp512_700m_seed0_k2"))
CHAIN_SEED, INITIAL_SEED = 20261005, 20261006


@torch.no_grad()
def predict(model, tokens, k, context):
    """All targets t+k are scored; future tokens never enter the input window."""
    batch, length = tokens.shape
    count = length - k
    assert 1 <= k < length and 1 <= context <= model.config.n_ctx
    p = np.empty((batch, count, model.config.vocab_size), dtype=np.float32)
    nll = np.empty((batch, count), dtype=np.float32)
    prefix = min(context, count)
    logs = torch.log_softmax(model(torch.as_tensor(tokens[:, :prefix], dtype=torch.long)), -1)
    p[:, :prefix] = logs.exp().numpy()
    target = torch.as_tensor(tokens[:, k:k+prefix], dtype=torch.long)
    nll[:, :prefix] = -logs.gather(-1, target[..., None]).squeeze(-1).numpy()
    for start in range(prefix, count, 8):
        positions = np.arange(start, min(start + 8, count))
        windows = np.stack([tokens[:, t-context+1:t+1] for t in positions])
        logs = torch.log_softmax(model(torch.as_tensor(windows.reshape(-1, context), dtype=torch.long))[:, -1], -1)
        logs = logs.reshape(len(positions), batch, -1).transpose(0, 1)
        p[:, positions] = logs.exp().numpy()
        target = torch.as_tensor(tokens[:, positions + k], dtype=torch.long)
        nll[:, positions] = -logs.gather(-1, target[..., None]).squeeze(-1).numpy()
    assert np.isfinite(p).all() and np.isfinite(nll).all()
    assert (p >= 0).all() and np.allclose(p.sum(-1), 1, atol=2e-6)
    return p, nll


def panel(ax, probabilities, tokens, k, context, title):
    count = len(probabilities)
    image = ax.imshow(probabilities.T, origin="lower", aspect="auto", cmap="magma", vmin=0, vmax=1,
                      extent=(-.5, count-.5, -.5, 180.5))
    ax.plot(np.arange(count), tokens[k:], color="cyan", linewidth=1.1, label=f"actual token at t+{k}")
    if count > context:
        ax.axvline(context-.5, color="white", linestyle="--", linewidth=1, label="sliding context starts")
    ax.set(xlabel="source position t", ylabel="token id", title=title, ylim=(-.5, 180.5))
    ax.set_yticks([0, 45, 90, 135, 180]); ax.legend(loc="upper right", fontsize=8)
    return image


def run_case(label, mode, name, args):
    from experiments.sphere_vast_train import configuration_sha256, run_name, scope
    from physics.systems.sphere_polar import THETA_MIN, THETA_MAX
    ns = scope(mode, args.models)
    experiment, training = next(pair for pair in ns["ACTIVE_RUNS"] if run_name(*pair) == name)
    original = experiment.build_process()
    meta = json.loads((args.models / (name + ".json")).read_text())
    assert meta["source_revision"] == REVISION and meta["tokens_seen"] == 699924480
    assert meta["configuration_sha256"] == configuration_sha256(experiment, training, original)
    assert args.tokens % experiment.n == 0
    process = replace(original, m=args.tokens // experiment.n)
    reference = replace(process, system=replace(process.system, integration_dt=.0025))
    # Archived polar dynamics omit state_names. Supply only the metadata needed
    # by the existing start sampler; no evolution equation or saved source is changed.
    start_view = SimpleNamespace(initial_state=process.system.initial_state, rate_max=process.system.rate_max,
                                 state_names=("theta", "psi", "dtheta", "dpsi"))
    batches, states, checks = {}, {}, {}
    for initial_mode in ("fixed", "random"):
        initial = ns["InitialStateConfig"](mode=initial_mode, jitter_std=args.jitter).sample(
            start_view, np.random.default_rng(INITIAL_SEED), args.paths)
        batch = process.sample_batch(np.random.default_rng(CHAIN_SEED), args.paths, initial_state=initial)
        finer = reference.sample_batch(np.random.default_rng(CHAIN_SEED), args.paths, initial_state=initial)
        assert np.array_equal(batch["letters"], finer["letters"])
        theta = batch["observable"][..., 0]
        rates = np.stack((batch["metric"][..., 0] / process.system.length,
                          batch["metric"][..., 1] / (process.system.length * np.sin(theta))), axis=-1)
        assert np.isfinite(theta).all() and np.isfinite(rates).all()
        lo, hi = process.obs_ranges[0]
        checks[initial_mode] = {
            "observation_range": [float(theta.min()), float(theta.max())],
            "observation_bin_clipping_fraction": float(((theta < lo) | (theta > hi)).mean()),
            "sampled_polar_boundary_contacts": int(((theta <= THETA_MIN+1e-10) | (theta >= THETA_MAX-1e-10)).sum()),
            "sampled_rate_limit_contacts": int((np.abs(rates) >= process.system.rate_max-1e-8).sum()),
            "finer_integration_dt": .0025,
            "reference_token_agreement": float((batch["tokens"] == finer["tokens"]).mean()),
            "reference_max_polar_error_rad": float(np.max(np.abs(theta-finer["observable"][..., 0]))),
        }
        batches[initial_mode], states[initial_mode] = batch, initial
    assert np.array_equal(batches["fixed"]["letters"], batches["random"]["letters"])
    assert not np.array_equal(states["fixed"], states["random"])
    assert len(np.unique(states["random"], axis=0)) == args.paths
    context = meta["model"]["n_ctx"] - training.k
    result = {"label": label, "model_name": name, "training_source_revision": REVISION,
              "configuration_sha256": meta["configuration_sha256"], "original_experiment": meta["experiment"],
              "architecture": meta["model"], "trajectory_tokens": args.tokens, "hmm_ticks": process.m,
              "k_lookahead": training.k, "inference_context": context,
              "fixed_and_random_starts_share_identical_HMM_letters": True,
              "initial_states": {key: value.tolist() for key, value in states.items()},
              "physics_checks": checks, "scores": {}}
    arrays = {f"{key}_tokens": value["tokens"] for key, value in batches.items()}
    arrays.update({f"{key}_initial_states": value for key, value in states.items()})
    arrays.update({f"{key}_observable": value["observable"] for key, value in batches.items()})
    arrays.update({f"{key}_letters": value["letters"] for key, value in batches.items()})
    for tag, suffix, hash_key in (("trained", ".pth", "checkpoint_sha256"),
                                  ("random_init", "_random_init.pt", "random_checkpoint_sha256")):
        weight = args.models / (name + suffix)
        digest = hashlib.sha256(weight.read_bytes()).hexdigest()
        assert digest == meta[hash_key]
        model = training.build_model(original).eval()
        model.load_state_dict(torch.load(weight, weights_only=True, map_location="cpu"))
        scores = {"weight_sha256": digest}
        for initial_mode, batch in batches.items():
            p, nll = predict(model, batch["tokens"], training.k, context)
            arrays[f"{tag}_{initial_mode}_probabilities"] = p
            arrays[f"{tag}_{initial_mode}_nll"] = nll
            scores[initial_mode] = {"mean_nll_nats": float(nll.mean()),
                                    "mean_nll_first_window": float(nll[:, :context].mean()),
                                    "mean_nll_sliding_window": float(nll[:, context:].mean()),
                                    "path_nll_nats": nll.mean(-1).tolist(),
                                    "top1_accuracy": float((p.argmax(-1) == batch["tokens"][:, training.k:]).mean())}
            if tag == "trained" and initial_mode == "random":
                for i in range(args.paths):
                    fig, ax = plt.subplots(figsize=(16, 6), constrained_layout=True)
                    heading = f"Sphere n={experiment.n} γ={experiment.physics.gamma} δv={experiment.physics.delta_v} α={experiment.hmm.alpha} | k={training.k} d128 | random start {i+1} | mean NLL={nll[i].mean():.3f}"
                    image = panel(ax, p[i], batch["tokens"][i], training.k, context, heading)
                    fig.suptitle(f"{args.tokens}-token trajectory · context capped at {context} trained positions · training used fixed starts")
                    fig.colorbar(image, ax=ax, label="Model token probability")
                    fig.savefig(args.output / f"{label}_random_start_{i+1}.png", dpi=130); plt.close(fig)
        result["scores"][tag] = scores
        del model
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), constrained_layout=True)
    for ax, (tag, initial_mode) in zip(axes, (("trained", "fixed"), ("trained", "random"), ("random_init", "random"))):
        p, nll = arrays[f"{tag}_{initial_mode}_probabilities"][0], arrays[f"{tag}_{initial_mode}_nll"][0]
        image = panel(ax, p, batches[initial_mode]["tokens"][0], training.k, context,
                      f"{tag} · {initial_mode} start · mean NLL={nll.mean():.3f}")
    fig.suptitle(f"Sphere {label} · matched action path 1 · {args.tokens} tokens · k={training.k} · context {context}/{meta['model']['n_ctx']}")
    fig.colorbar(image, ax=axes, label="Model token probability")
    fig.savefig(args.output / f"{label}_controls.png", dpi=130); plt.close(fig)
    np.savez_compressed(args.output / f"{label}_probabilities.npz", **arrays)
    (args.output / f"{label}.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print("completed", label, flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--models", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tokens", type=int, default=500)
    parser.add_argument("--paths", type=int, default=4)
    parser.add_argument("--jitter", type=float, default=.1)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    assert 400 <= args.tokens <= 1000 and args.tokens % 20 == 0 and 2 <= args.paths <= 8
    assert 0 < args.jitter <= .2
    sys.path.insert(0, str(args.source)); torch.set_num_threads(2); torch.set_num_interop_threads(1)
    if args.self_check:
        class Echo(torch.nn.Module):
            config = SimpleNamespace(n_ctx=16, vocab_size=7)
            def forward(self, x): return torch.nn.functional.one_hot(x, 7).float() * 5
        tokens = np.arange(70).reshape(2, 35) % 7
        p, nll = predict(Echo(), tokens, 3, 13)
        assert p.shape == (2, 32, 7) and np.array_equal(p.argmax(-1), tokens[:, :-3])
        assert np.allclose(nll, np.log(np.exp(5)+6))
        from experiments.sphere_vast_train import scope
        ns = scope("half", Path("/tmp"))
        system = ns["ACTIVE_RUNS"][0][0].build_process().system
        view = SimpleNamespace(initial_state=system.initial_state, rate_max=system.rate_max,
                               state_names=("theta", "psi", "dtheta", "dpsi"))
        random = ns["InitialStateConfig"](mode="random", jitter_std=.1)
        a = random.sample(view, np.random.default_rng(INITIAL_SEED), 4)
        assert np.array_equal(a, random.sample(view, np.random.default_rng(INITIAL_SEED), 4))
        assert a.shape == (4, 4) and np.isfinite(a).all() and not np.array_equal(a, system.initial_state(4))
        print("Causal window/target alignment, normalization, NLL and reproducible random starts passed."); return
    assert args.models and args.output
    args.output.mkdir(parents=True, exist_ok=True)
    assert not (args.output / "summary.json").exists(), "completed output already exists"
    results = [run_case(label, mode, name, args) for label, mode, name in CASES]
    report = {"experiment_id": "sphere-token-500-random-start-eval-20261005", "training_source_revision": REVISION,
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "chain_seed": CHAIN_SEED,
              "initial_seed": INITIAL_SEED, "initial_distribution": "Independent Gaussian jitter around (0.6,0,0,2), using existing bounded InitialStateConfig sampler",
              "initial_coordinate_order": ["theta", "psi", "dtheta", "dpsi"], "jitter_std": args.jitter,
              "trajectory_tokens": args.tokens, "paths_per_initial_mode": args.paths,
              "inference_protocol": "Causal prefix then sliding context of n_ctx-k tokens; positions reset in each window. Last k untrained source positions are avoided.",
              "interpretation": "Frozen models trained on 16-tick fixed-start sequences. Long/random-start evaluation is outside their original training protocol, not newly randomized or 500-context training.",
              "probability_semantics": "Native softmax over 181 observation tokens at t+k; no fitted belief probe or probability projection",
              "uniform_baseline_nll": float(np.log(181)), "cpu_host": os.uname().nodename,
              "environment": {"torch": torch.__version__, "numpy": np.__version__}, "results": results}
    (args.output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    manifest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(args.output.iterdir()) if p.is_file() and p.name != "SHA256.json"}
    (args.output / "SHA256.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__": main()
