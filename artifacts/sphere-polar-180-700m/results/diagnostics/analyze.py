"""Bounded CPU diagnostics of completed Sphere models; never trains or edits bundles."""
import argparse
import csv
import hashlib
import json
import os
import resource
import sys
from pathlib import Path

os.environ.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="2", MPLBACKEND="Agg")
resource.setrlimit(resource.RLIMIT_AS, (12 * 1024**3, 12 * 1024**3))
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REVISION = "dc80726a8c987e8c3a9171299dac00d03a074311"
READOUT = "all_block_layers_single_token"
FRESH_SEED = 20261002


def project_simplex(p):
    """Euclidean projection; raw probe R2 is always reported separately."""
    u = np.sort(p, axis=-1)[..., ::-1]
    c = np.cumsum(u, axis=-1) - 1
    k = np.sum(u - c / np.arange(1, p.shape[-1] + 1) > 0, axis=-1)
    theta = np.take_along_axis(c, (k - 1)[..., None], axis=-1) / k[..., None]
    return np.maximum(p - theta, 0)


def bootstrap_gap(y, trained, random, repeats=500):
    # Resample whole trajectories, keeping all ticks and paired predictions together.
    rng = np.random.default_rng(20261003)
    gaps = []
    for _ in range(repeats):
        index = rng.integers(len(y), size=len(y))
        truth = y[index].reshape(-1, y.shape[-1])
        gaps.append(r2_score(truth, trained[index].reshape(truth.shape)) -
                    r2_score(truth, random[index].reshape(truth.shape)))
    return np.quantile(gaps, [.025, .975]).tolist()


def fit_probe(x, y, splits, alpha=None):
    fit, validation, _ = splits
    best = None
    for penalty in ([alpha] if alpha is not None else (1., 10., 100., 1000., 10000.)):
        model = make_pipeline(StandardScaler(), Ridge(alpha=penalty, solver="lsqr"))
        model.fit(x[fit], y[fit])
        score = float(r2_score(y[validation], model.predict(x[validation])))
        if best is None or score > best[0]:
            best = score, model, penalty
    return best[1], best[2], best[0]


def probabilities(p, y, moods):
    projected = project_simplex(p)
    onehot = np.eye(4)[moods]
    confidence = projected.max(-1)
    correct = projected.argmax(-1) == moods
    bins = []
    for lo in np.arange(0, 1, .1):
        mask = (confidence >= lo) & (confidence < lo + .1 + (1e-8 if lo > .89 else 0))
        if mask.any():
            bins.append({"confidence": float(confidence[mask].mean()),
                         "accuracy": float(correct[mask].mean()), "count": int(mask.sum())})
    return {"raw_entries_outside_0_1_fraction": float(((p < 0) | (p > 1)).mean()),
            "raw_rows_outside_simplex_fraction": float(((p < 0).any(-1) | (p > 1).any(-1)).mean()),
            "raw_sum_max_abs_error": float(np.max(np.abs(p.sum(-1) - 1))),
            "raw_min": float(p.min()), "raw_max": float(p.max()),
            "projected_belief_r2": float(r2_score(y, projected)),
            "projected_belief_mse": float(np.mean((y - projected)**2)),
            "projected_mood_brier_sum": float(np.mean(np.sum((onehot - projected)**2, axis=-1))),
            "projected_mood_log_loss": float(-np.log(np.maximum(projected[np.arange(len(p)), moods], 1e-12)).mean()),
            "projected_mood_accuracy": float(correct.mean()), "reliability_bins": bins}


def ranking(rows, output):
    pairs = {}
    for row in rows:
        if row["mode"] == READOUT:
            pairs.setdefault((row["model_name"], row["target"]), {})[row["model_type"]] = row
    assert len(pairs) == 360 and all(set(p) == {"trained", "random_init"} for p in pairs.values())
    ranked = []
    for (name, target), pair in pairs.items():
        row = pair["trained"]
        ranked.append({key: row[key] for key in
                       ("model_name", "lookahead_mode", "hmm_alpha", "n", "gamma", "delta_v", "k_lookahead")}
                      | {"target": target, "trained_r2": float(row["test_r2"]),
                         "random_r2": float(pair["random_init"]["test_r2"]),
                         "gap": float(row["test_r2"]) - float(pair["random_init"]["test_r2"])})
    ranked.sort(key=lambda row: (row["target"], -row["gap"]))
    with (output / "gap_ranking.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, ranked[0].keys()); writer.writeheader(); writer.writerows(ranked)
    fig, axes = plt.subplots(2, 3, figsize=(15, 11), constrained_layout=True)
    alphas = [.5, .7, .85, .95]
    for ti, target in enumerate(("belief", "physics")):
        subset = [r for r in ranked if r["target"] == target]
        for mi, mode in enumerate(("fixed", "half", "n_plus_one")):
            data = [r for r in subset if r["lookahead_mode"] == mode]
            conditions = sorted({(int(r["model_name"].split("_c")[1][:2]), r["n"], r["gamma"], r["delta_v"]) for r in data})
            values = np.array([[next(r["gap"] for r in data if int(r["model_name"].split("_c")[1][:2]) == c[0]
                                   and float(r["hmm_alpha"]) == a) for a in alphas] for c in conditions])
            ax = axes[ti, mi]
            im = ax.imshow(values, aspect="auto", cmap="viridis", vmin=0, vmax=max(r["gap"] for r in subset))
            ax.set(title=f"{target.capitalize()} gain · {mode}", xlabel="HMM emission accuracy α")
            ax.set_xticks(range(4), alphas)
            ax.set_yticks(range(15), [f"c{c[0]:02}  n={c[1]} γ={c[2]} δv={c[3]}" for c in conditions], fontsize=8)
            for i in range(15):
                for j in range(4): ax.text(j, i, f"{values[i,j]:.3f}", ha="center", va="center", fontsize=8,
                                          color="white" if values[i,j] < .55 * max(r["gap"] for r in subset) else "black")
            fig.colorbar(im, ax=ax, label="Trained − matched random test R²")
    fig.suptitle("180 complete Sphere models · one training seed · all four block outputs")
    fig.savefig(output / "grid_gaps.png", dpi=160); plt.close(fig)
    belief = [r for r in ranked if r["target"] == "belief"]
    physics = [r for r in ranked if r["target"] == "physics"]
    return [("widest_belief", belief[0]), ("widest_physics", physics[0]), ("smallest_belief", belief[-1])]


def diagnose(label, row, args, rows):
    from experiments.sphere_staging_probe import sequence_splits, token_features
    from experiments.sphere_vast_train import configuration_sha256, run_name, scope
    name = row["model_name"]
    ns = scope(row["lookahead_mode"], args.models)
    experiment, training = next(pair for pair in ns["ACTIVE_RUNS"] if run_name(*pair) == name)
    process = experiment.build_process()
    meta = json.loads((args.models / (name + ".json")).read_text())
    assert meta["source_revision"] == REVISION and meta["tokens_seen"] == 699924480
    assert meta["configuration_sha256"] == configuration_sha256(experiment, training, process)
    original = ns["sample_physics_batch"](process, np.random.default_rng(20260929), 2048, exp_cfg=experiment)
    fresh = ns["sample_physics_batch"](process, np.random.default_rng(FRESH_SEED), 2048, exp_cfg=experiment)
    original_histories = {r.tobytes() for r in original["tokens"]}
    keep = np.array([i for i, r in enumerate(fresh["tokens"]) if r.tobytes() not in original_histories])
    fresh = {key: value[keep] for key, value in fresh.items()}
    assert len(keep) >= 1800
    last = np.arange(experiment.n - 1, process.seq_len, experiment.n)
    sequence_parts = sequence_splits(original["tokens"])
    splits = tuple((part[:, None] * 16 + np.arange(16)).reshape(-1) for part in sequence_parts)
    for left in range(3):
        for right in range(left + 1, 3):
            assert {r.tobytes() for r in original["tokens"][sequence_parts[left]]}.isdisjoint(
                {r.tobytes() for r in original["tokens"][sequence_parts[right]]})
    prefix_overlap = []
    for pos in last:
        fit_prefixes = {r[:pos+1].tobytes() for r in original["tokens"][sequence_parts[0]]}
        prefix_overlap.append(float(np.mean([r[:pos+1].tobytes() in fit_prefixes
                                             for r in original["tokens"][sequence_parts[2]]])))
    targets = lambda batch: {"belief": batch["beliefs"][:, last], "physics": batch["metric"][:, last]}
    original_y, fresh_y = targets(original), targets(fresh)
    for batch in (original, fresh):
        assert np.isfinite(batch["beliefs"]).all() and np.isfinite(batch["metric"]).all()
        assert np.allclose(batch["beliefs"].sum(-1), 1) and batch["beliefs"].min() >= 0
        assert np.allclose(batch["beliefs"][:, last], process.chain.beliefs(batch["letters"]))
    result = {"selection": label, "model_name": name, "setting": row,
              "source_revision": REVISION, "configuration_sha256": meta["configuration_sha256"],
              "original_split_counts": [len(p) for p in sequence_parts], "fresh_test_trajectories": len(keep),
              "fresh_full_histories_removed": 2048 - len(keep),
              "test_prefix_seen_in_fit_fraction_by_tick": prefix_overlap,
              "true_belief_simplex_and_letter_alignment_passed": True, "models": {}}
    predictions = {}
    for tag, suffix, key in (("trained", ".pth", "checkpoint_sha256"),
                             ("random_init", "_random_init.pt", "random_checkpoint_sha256")):
        weight = args.models / (name + suffix)
        assert hashlib.sha256(weight.read_bytes()).hexdigest() == meta[key]
        model = training.build_model(process).eval()
        model.load_state_dict(torch.load(weight, weights_only=True, map_location="cpu"))
        def features(tokens):
            # Keep tick rows immediately instead of retaining all 5 full residual streams.
            return np.concatenate([token_features(model.residual_streams(torch.as_tensor(chunk, dtype=torch.long)), last)[READOUT]
                                   for chunk in np.array_split(tokens, np.arange(16, len(tokens), 16))]).astype(np.float32)
        cut = process.seq_len // 2
        tokens = torch.as_tensor(fresh["tokens"][:4], dtype=torch.long)
        changed = tokens.clone(); changed[:, cut:] = (changed[:, cut:] + 17) % process.n_obs
        with torch.no_grad():
            before, after = model.residual_streams(tokens), model.residual_streams(changed)
            causal_error = max(float(np.max(np.abs(a[:, :cut] - b[:, :cut]))) for a, b in zip(before, after))
            assert causal_error <= 1e-5
            losses = [float(model.loss(torch.as_tensor(chunk, dtype=torch.long)))
                      for chunk in np.array_split(fresh["tokens"][:256], 16)]
        x, xf = features(original["tokens"]), features(fresh["tokens"])
        scores = {"weight_sha256": meta[key], "causal_future_perturbation_max_error": causal_error,
                  "fresh_k_ahead_cross_entropy_nats": float(np.mean(losses))}
        predictions[tag] = {}
        for target in ("belief", "physics"):
            expected = next(r for r in rows if r["model_name"] == name and r["mode"] == READOUT
                            and r["target"] == target and r["model_type"] == tag)
            y = original_y[target].reshape(-1, original_y[target].shape[-1])
            probe, penalty, validation = fit_probe(x.reshape(-1, 512), y, splits, float(expected["alpha"]))
            reproduction = float(r2_score(y[splits[2]], probe.predict(x.reshape(-1, 512)[splits[2]])))
            error = abs(reproduction - float(expected["test_r2"]))
            assert error < 1e-4, (name, tag, target, error)
            pred = probe.predict(xf.reshape(-1, 512)).reshape(fresh_y[target].shape)
            predictions[tag][target] = pred
            shuffled = y.copy()
            rng = np.random.default_rng(20261004)
            for part in splits: shuffled[part] = y[rng.permutation(part)]
            null, null_penalty, _ = fit_probe(x.reshape(-1, 512), shuffled, splits)
            scores[target] = {"ridge_alpha": penalty, "validation_r2": validation,
                              "original_test_r2": reproduction, "reproduction_abs_error": error,
                              "fresh_test_r2": float(r2_score(fresh_y[target].reshape(-1, y.shape[1]), pred.reshape(-1, y.shape[1]))),
                              "shuffled_fit_and_validation_labels_fresh_r2": float(r2_score(fresh_y[target].reshape(-1, y.shape[1]), null.predict(xf.reshape(-1, 512)))),
                              "shuffled_ridge_alpha": null_penalty}
            if target == "belief":
                scores[target]["probability_checks"] = probabilities(pred.reshape(-1, 4), fresh_y[target].reshape(-1, 4), fresh["hmm_state_path"][:, 1:].reshape(-1))
        result["models"][tag] = scores
        del model, x, xf
    # Baseline has the current observation bin and tick number, but no history.
    baseline = lambda tokens: np.concatenate((np.eye(181, dtype=np.float32)[tokens[:, last]],
                                               np.broadcast_to(np.eye(16, dtype=np.float32), (len(tokens), 16, 16))), axis=-1).reshape(-1, 197)
    x, xf = baseline(original["tokens"]), baseline(fresh["tokens"])
    result["fresh_gap_bootstrap_95_ci"] = {}
    result["current_token_and_tick_baseline"] = {}
    for target in ("belief", "physics"):
        y = original_y[target].reshape(-1, original_y[target].shape[-1])
        probe, penalty, _ = fit_probe(x, y, splits)
        result["current_token_and_tick_baseline"][target] = {
            "fresh_test_r2": float(r2_score(fresh_y[target].reshape(-1, y.shape[-1]), probe.predict(xf))), "ridge_alpha": penalty}
        result["fresh_gap_bootstrap_95_ci"][target] = bootstrap_gap(fresh_y[target], predictions["trained"][target], predictions["random_init"][target])
    np.savez_compressed(args.output / (label + "_heldout_predictions.npz"),
                        tokens=fresh["tokens"], true_belief=fresh_y["belief"], true_physics=fresh_y["physics"],
                        mood=fresh["hmm_state_path"][:, 1:], **{f"{tag}_{target}": value for tag, pred in predictions.items() for target, value in pred.items()})
    fig, axes = plt.subplots(4, 3, figsize=(13, 10), constrained_layout=True)
    for i in range(4):
        values = (fresh_y["belief"][i], project_simplex(predictions["trained"]["belief"][i]), project_simplex(predictions["random_init"]["belief"][i]))
        for j, value in enumerate(values):
            ax = axes[i, j]; im = ax.imshow(value.T, vmin=0, vmax=1, cmap="magma", aspect="auto")
            ax.set(title=f"{'Oracle action-conditioned belief' if j == 0 else ('Trained' if j == 1 else 'Matched random') + ' · simplex projected'} · path {i+1}", xlabel="HMM tick")
            ax.set_xticks([0, 3, 7, 11, 15], [1, 4, 8, 12, 16]); ax.set_yticks(range(4), ["Mood 0", "Mood 1", "Mood 2", "Mood 3"])
    fig.colorbar(im, ax=axes, label="Probability")
    fig.suptitle(f"{label.replace('_',' ')} · first four eligible fresh histories (no visual selection)\nα={row['hmm_alpha']}, n={row['n']}, γ={row['gamma']}, δv={row['delta_v']}, k={row['k_lookahead']}")
    fig.savefig(args.output / (label + "_probability_heatmaps.png"), dpi=150); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for tag, color in (("trained", "#1976d2"), ("random_init", "#e67e22")):
        bins = result["models"][tag]["belief"]["probability_checks"]["reliability_bins"]
        axes[0].plot([b["confidence"] for b in bins], [b["accuracy"] for b in bins], "o-", label=tag, color=color)
        for target, ax in (("belief", axes[1]), ("physics", axes[2])):
            per_tick = [r2_score(fresh_y[target][:, tick], predictions[tag][target][:, tick]) for tick in range(16)]
            ax.plot(range(1,17), per_tick, label=tag, color=color)
    axes[0].plot([0,1], [0,1], "k--", alpha=.4); axes[0].set(xlabel="Projected confidence", ylabel="Latent mood accuracy", title="Reliability (binned; not certification)")
    for target, ax in (("belief", axes[1]), ("physics", axes[2])): ax.set(xlabel="HMM tick", ylabel="Raw R²", title=f"{target.capitalize()} recovery by time")
    for ax in axes: ax.legend()
    fig.savefig(args.output / (label + "_validity.png"), dpi=150); plt.close(fig)
    (args.output / (label + ".json")).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print("completed", label, name, flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--models", type=Path)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args(); sys.path.insert(0, str(args.source)); torch.set_num_threads(2)
    if args.self_check:
        p = project_simplex(np.array([[-.2, .2, .3, .7], [.25, .25, .25, .25]]))
        assert (p >= 0).all() and np.allclose(p.sum(-1), 1)
        assert np.allclose(p[1], .25)
        rng = np.random.default_rng(1); y = rng.normal(size=(30, 16, 4))
        assert bootstrap_gap(y, y, y, 10) == [0., 0.]
        from experiments.sphere_staging_probe import self_check
        self_check(); print("diagnostic self-check passed"); return
    assert args.models and args.csv and args.output
    args.output.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(args.csv.open()))
    assert len(rows) == 2160
    selected = ranking(rows, args.output)
    results = [diagnose(label, row, args, rows) for label, row in selected]
    report = {"hypothesis_id": "H-R2-05-polar-ladder-capacity", "source_revision": REVISION,
              "diagnostic_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "original_eval_seed": 20260929, "fresh_eval_seed": FRESH_SEED,
              "readout": READOUT, "training_seed": 0, "bootstrap_repeats": 500,
              "cpu_host": os.uname().nodename, "torch_version": torch.__version__,
              "numpy_version": np.__version__, "results": results,
              "limitations": ["Three cases selected after inspecting 180 settings; fresh data checks these cases only.",
                              "Bootstrap intervals quantify evaluation sampling, not training-seed variance.",
                              "Oracle belief conditions on hidden action letters, not observation tokens; it is not an exact token-conditioned posterior.",
                              "Linear probes may leave the probability simplex; probability plots and proper scores use explicitly labeled projection.",
                              "Full histories are disjoint; early causal prefixes may recur across splits. Causality is checked separately."]}
    (args.output / "diagnostic_summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    manifest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(args.output.iterdir()) if p.is_file() and p.name != "SHA256.json"}
    (args.output / "SHA256.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__": main()
