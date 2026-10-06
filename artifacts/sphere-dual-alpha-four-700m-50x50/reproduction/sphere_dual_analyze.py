"""CPU-only held-out probes and native 500-token heatmaps for the four joint models."""
import argparse
import csv
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import resource
import socket
import zipfile

os.environ.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='2', MPLBACKEND='Agg')
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from experiments.sphere_dual_four import runs
from experiments.sphere_staging_probe import ridge_score, sequence_splits, token_features
from experiments.sphere_vast_train import configuration_sha256, run_name

EVALUATION_SEED = 20261009


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def grouped_prefix_splits(tokens, last):
    """Split trajectories; drop later-split rows sharing an earlier causal prefix."""
    sequence = sequence_splits(tokens)
    seen, parts, dropped = set(), [], []
    for trajectories in sequence:
        rows, current, removed = [], set(), 0
        for i in trajectories:
            for tick, t in enumerate(last):
                prefix = hashlib.sha256(np.asarray(tokens[i, :t + 1], dtype='<i4').tobytes()).digest()
                if prefix in seen:
                    removed += 1
                else:
                    rows.append(i * len(last) + tick)
                    current.add(prefix)
        assert seen.isdisjoint(current)
        seen.update(current)
        parts.append(np.array(rows, dtype=np.int64))
        dropped.append(removed)
    if any(len(p) < 32 for p in parts):
        raise ValueError('insufficient distinct held-out causal histories')
    return sequence, tuple(parts), dropped


def simplex(p):
    # Same Euclidean projection as the archived diagnostic; raw R2 stays separate.
    u = np.sort(p, axis=-1)[..., ::-1]
    c = np.cumsum(u, axis=-1) - 1
    k = np.sum(u - c / np.arange(1, p.shape[-1] + 1) > 0, axis=-1)
    return np.maximum(p - np.take_along_axis(c, (k - 1)[..., None], -1) / k[..., None], 0)


def native_metrics(model, tokens, k, indices):
    totals = np.zeros(5, dtype=np.float64)
    calibration = np.zeros((10, 3), dtype=np.float64)
    normalization_error = 0.
    with torch.no_grad():
        for start in range(0, len(indices), 8):
            t = torch.as_tensor(tokens[indices[start:start + 8]], dtype=torch.long)
            lp = model(t)[:, :-k].log_softmax(-1)
            p = lp.exp()
            truth = t[:, k:]
            confidence, predicted = p.max(-1)
            correct = predicted == truth
            target_lp = lp.gather(-1, truth[..., None]).squeeze(-1)
            target_p = p.gather(-1, truth[..., None]).squeeze(-1)
            count = truth.numel()
            totals += [float(-target_lp.double().sum()), float((p.square().sum(-1) - 2 * target_p + 1).double().sum()),
                       float(-(p * lp).double().sum()), float(correct.sum()), count]
            normalization_error = max(normalization_error, float((p.sum(-1) - 1).abs().max()))
            bins = (confidence * 10).long().clamp(max=9)
            for i in range(10):
                mask = bins == i
                calibration[i] += [int(mask.sum()), float(confidence[mask].double().sum()), int(correct[mask].sum())]
    if not np.isfinite(totals).all() or normalization_error > 1e-5:
        raise ValueError('invalid native probability distribution')
    count = totals[-1]
    ece = sum(abs(c - a) for n, c, a in calibration if n) / count
    return dict(zip(('nll', 'brier_sum', 'entropy', 'accuracy'), (totals[:-1] / count).tolist())) | {
        'positions': int(count), 'ece_10_bins': float(ece), 'max_probability_sum_error': normalization_error,
        'reliability': [{'count': int(n), 'confidence': c / n, 'accuracy': a / n} for n, c, a in calibration if n]}


def tick_streams(model, tokens, last):
    # Keep ticks as we go: avoids holding every token at every layer in RAM.
    chunks = [[] for _ in range(5)]
    for start in range(0, len(tokens), 8):
        streams = model.residual_streams(torch.as_tensor(tokens[start:start + 8], dtype=torch.long))
        for depth, stream in enumerate(streams):
            chunks[depth].append(stream[:, last].copy())
    return [np.concatenate(part) for part in chunks]


def action_counterfactuals(proc, batch, paths):
    """Four physical endpoints, not four neural mood-conditioned rollouts."""
    z = batch['initial_state'][paths].copy()
    history = []
    for t in range(500):
        if t % proc.n_steps == 0:
            z = proc.system.kick(z, proc.actions[batch['letters'][paths, t // proc.n_steps]])
        z = proc.flow(z)
        history.append(z.copy())
    history = np.stack(history, axis=1)
    np.testing.assert_array_equal(proc.observe(history), batch['tokens'][paths])
    branch = np.repeat(history[:, :490, None], 4, axis=2)
    for ahead in range(1, 11):
        kicks = (np.arange(490) + ahead) % proc.n_steps == 0
        branch[:, kicks] = proc.system.kick(branch[:, kicks], proc.actions)
        branch = proc.flow(branch)
    tokens = proc.observe(branch)
    assert np.any(tokens == batch['tokens'][paths, 10:, None], axis=-1).all()
    return tokens


def heatmaps(model, tokens, paths, k, directory, tag, counterfactual):
    with torch.no_grad():
        probability = model(torch.as_tensor(tokens[paths], dtype=torch.long))[:, :-k].softmax(-1).numpy()
    np.savez_compressed(directory / f'{tag}_native_probabilities.npz', joint=probability,
                        source_tokens=tokens[paths], future_tokens=tokens[paths, k:], trajectory_indices=paths, k=k,
                        physical_action_counterfactual_tokens=counterfactual)
    theta = probability.reshape(len(paths), -1, 50, 50).sum(-1)
    psi = probability.reshape(len(paths), -1, 50, 50).sum(-2)
    future = tokens[paths, k:]
    for label, data, actual, vmax in (('joint', probability, future, 1.),
                                     ('theta', theta, future // 50, 1.), ('psi', psi, future % 50, 1.)):
        fig, axes = plt.subplots(len(paths), 1, figsize=(16, 3 * len(paths)), constrained_layout=True)
        for path, ax in enumerate(np.atleast_1d(axes)):
            im = ax.imshow(data[path].T, origin='lower', aspect='auto', cmap='magma', vmin=0, vmax=vmax,
                           extent=(-.5, data.shape[1] - .5, -.5, data.shape[2] - .5))
            # Periodic seam jumps are breaks, not trajectories through all azimuth bins.
            line = actual[path].astype(float)
            if label == 'psi':
                line[np.r_[False, np.abs(np.diff(line)) > 25]] = np.nan
            if label != 'joint':
                fork_bins = counterfactual[path] // 50 if label == 'theta' else counterfactual[path] % 50
                for action in range(4):
                    curve = fork_bins[:, action].astype(float)
                    if label == 'psi':
                        curve[np.r_[False, np.abs(np.diff(curve)) > 25]] = np.nan
                    ax.plot(curve, linewidth=.65, alpha=.6, linestyle='--',
                            label=f'physical action {action} at next impulse')
                ax.plot(line, color='cyan', linewidth=.8, label='actual token at t+10')
                ax.legend(loc='upper right')
            ax.set(title=f'{tag}: {label} | held-out randomized path {int(paths[path])} | native k=10 head',
                   xlabel='source token position t (0–489 of 500)', ylabel=f'{label} bin')
        fig.colorbar(im, ax=np.atleast_1d(axes).tolist(), label='probability', fraction=.015)
        fig.savefig(directory / f'{tag}_{label}_heatmaps.png', dpi=160)
        plt.close(fig)
    mass = np.zeros(counterfactual.shape[:2])
    for action in range(4):
        unique = np.all(counterfactual[..., action, None] != counterfactual[..., :action], axis=-1)
        mass += np.take_along_axis(probability, counterfactual[..., action, None], axis=-1).squeeze(-1) * unique
    return {'mean_probability_mass_on_four_physical_endpoint_bins': float(mass.mean()),
            'minimum_unique_physical_endpoints': int(min(len(np.unique(row)) for row in counterfactual.reshape(-1, 4)))}


def analyze(args):
    torch.set_num_threads(2)
    _, selected = runs(args.screen, args.results)
    exp, tr = selected[args.index]
    proc = exp.build_process()
    name = run_name(exp, tr)
    checkpoint = args.results / (name + '.pth')
    meta = json.loads(checkpoint.with_suffix('.json').read_text())
    random = args.results / (name + '_random_init.pt')
    if (meta['configuration_sha256'] != configuration_sha256(exp, tr, proc)
        or meta['source_revision'] != args.training_revision or meta['tokens_seen'] != 699904000
        or meta['experiment'] != json.loads(json.dumps(exp.metadata()))
        or meta['training'] != json.loads(json.dumps(asdict(tr)))
        or digest(checkpoint) != meta['checkpoint_sha256'] or digest(random) != meta['random_checkpoint_sha256']
        or (args.results / (name + '.ready')).read_text().strip() != meta['checkpoint_sha256']):
        raise ValueError('not a verified full joint-angle checkpoint and matched random control')
    directory = args.output / name
    if (directory / 'analysis.json').exists():
        raise ValueError('analysis already exists; verify it instead of duplicating')
    directory.mkdir(parents=True, exist_ok=True)
    batch = exp.sample_batch(proc, np.random.default_rng(EVALUATION_SEED), args.trajectories)
    tokens = batch['tokens']
    # The final k query positions receive no direct prediction loss in a shifted objective.
    last = np.arange(exp.n - 1, 500 - tr.k, exp.n)
    sequence, splits, dropped = grouped_prefix_splits(tokens, last)
    targets = {key: batch[value][:, last].reshape(-1, batch[value].shape[-1])
               for key, value in (('belief', 'beliefs'), ('physics', 'metric'))}
    paths = sequence[2][:4]
    counterfactual = action_counterfactuals(proc, batch, paths)
    np.savez_compressed(directory / 'heldout_paths.npz', **{key: value[paths] for key, value in batch.items()
                        if isinstance(value, np.ndarray)}, trajectory_indices=paths)
    rows, native, belief_predictions, belief_validity = [], {}, {}, {}
    for tag, file in (('trained', checkpoint), ('random_init', random)):
        model = tr.build_model(proc).eval()
        model.load_state_dict(torch.load(file, map_location='cpu', weights_only=True))
        with torch.no_grad():
            test = torch.as_tensor(tokens[:1], dtype=torch.long)
            full, prefix = model(test), model(test[:, :101])
            error = float((full[:, :101] - prefix).abs().max())
            if error > 2e-5:
                raise ValueError('future tokens affect a causal prefix')
        native[tag] = native_metrics(model, tokens, 10, sequence[2]) | {'causal_prefix_max_error': error}
        native[tag].update(heatmaps(model, tokens, paths, 10, directory, tag, counterfactual))
        streams = tick_streams(model, tokens, last)
        features = token_features(streams, np.arange(len(last)))
        del streams
        for mode, feature in features.items():
            x = feature.reshape(-1, feature.shape[-1])
            for target, y in targets.items():
                score = ridge_score(x, y, splits)
                rows.append({'model_name': name, 'model_type': tag, 'mode': mode, 'target': target,
                             'control': 'matched', **score})
                if mode == 'all_block_layers_single_token':
                    shuffled = y.copy()
                    fit_sequence = sequence[0]
                    source = np.random.default_rng(88).permutation(fit_sequence)
                    view = shuffled.reshape(len(tokens), len(last), -1)
                    view[fit_sequence] = view[source].copy()
                    rows.append({'model_name': name, 'model_type': tag, 'mode': mode, 'target': target,
                                 'control': 'shuffled_fit_trajectories', **ridge_score(x, shuffled, splits)})
                    if target == 'belief':
                        readout = make_pipeline(StandardScaler(), Ridge(alpha=score['alpha'], solver='lsqr'))
                        readout.fit(x[splits[0]], y[splits[0]])
                        prediction = readout.predict(x).reshape(len(tokens), len(last), 4)
                        belief_predictions[tag] = prediction[paths]
                        raw = prediction.reshape(-1, 4)[splits[2]]
                        projected = simplex(raw)
                        belief_validity[tag] = {
                            'raw_entries_outside_0_1_fraction': float(((raw < 0) | (raw > 1)).mean()),
                            'raw_sum_max_error': float(np.max(np.abs(raw.sum(-1) - 1))),
                            'projected_belief_mse': float(np.mean((projected - y[splits[2]]) ** 2)),
                            'projected_sum_max_error': float(np.max(np.abs(projected.sum(-1) - 1)))}
        del model, features, feature, x
    oracle = batch['beliefs'][paths][:, last]
    np.savez_compressed(directory / 'belief_readouts.npz', oracle=oracle, **belief_predictions,
                        ticks=last, trajectory_indices=paths)
    fig, axes = plt.subplots(4, 3, figsize=(17, 10), constrained_layout=True)
    for col, (tag, belief) in enumerate([('oracle action-conditioned belief', oracle),
                                       *[(tag + ' linear probe; simplex projected', simplex(p))
                                         for tag, p in belief_predictions.items()]]):
        for path in range(4):
            im = axes[path, col].imshow(belief[path].T, aspect='auto', origin='lower', vmin=0, vmax=1, cmap='magma',
                                       extent=(0, int(last[-1]) + 1, -.5, 3.5))
            axes[path, col].set(title=f'{tag} | path {int(paths[path])}', yticks=range(4),
                               yticklabels=[f'Mood {i}' for i in range(4)], xlim=(0, 500),
                               xlabel='token position (supervised query ticks only)')
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=.015, label='probability')
    fig.savefig(directory / 'belief_heatmaps.png', dpi=160)
    plt.close(fig)
    csv_path = directory / 'probes.csv'
    with csv_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    reloaded = list(csv.DictReader(csv_path.open()))
    assert len(rows) == len(reloaded) == 16
    for original, exported in zip(rows, reloaded):
        for key in ('alpha', 'validation_r2', 'test_r2', 'feature_dim', 'train_rows_per_feature'):
            assert float(exported[key]) == original[key] and np.isfinite(original[key])
    evidence = {'model_name': name, 'training_source_revision': args.training_revision,
                'analysis_sha256': digest(Path(__file__)), 'configuration_sha256': meta['configuration_sha256'],
                'checkpoint_sha256': digest(checkpoint), 'random_checkpoint_sha256': digest(random),
                'experiment': exp.metadata(), 'probes': rows, 'native_head_metrics': native,
                'belief_readout_validity': belief_validity,
                'protocol': {'evaluation_seed': EVALUATION_SEED, 'split_seed': 29, 'trajectories': len(tokens),
                             'trajectory_split_counts': [len(p) for p in sequence], 'retained_tick_rows': [len(p) for p in splits],
                             'dropped_duplicate_causal_prefix_rows': dropped, 'cross_split_causal_prefix_overlap': 0,
                             'probe_targets': 'action-conditioned four-mood oracle belief; two tangent velocities',
                             'native_target': 'joint token t+10 given true token prefix through t; teacher forced',
                             'four_dashed_curves': 'physical endpoints for four actions at next impulse, conditioned on actual current state; not neural mood-conditioned rollouts',
                             'random_control': 'saved pre-training initialization; same test trajectories',
                             'heatmap_paths': paths.tolist(), 'model_context_capacity': 500,
                             'longest_directly_supervised_prefix': 490, 'head_evaluated_positions': 490,
                             'probe_query_positions': last.tolist()},
                'environment': {'host': socket.gethostname(), 'torch': torch.__version__, 'numpy': np.__version__},
                'acceptance_state': 'full training, held-out diagnostics and exports complete',
                'scientific_acceptance': 'one training seed; selected by physics preflight, not a model-performance sweep',
                'artifact_sha256': {p.name: digest(p) for p in directory.iterdir() if p.is_file()}}
    (directory / 'analysis.json').write_text(json.dumps(evidence, indent=2) + '\n')
    archive_path = directory.parent / (directory.name + '.zip')
    with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(directory.iterdir()):
            archive.write(p, p.name)
    with zipfile.ZipFile(archive_path) as archive:
        members = {name: hashlib.sha256(archive.read(name)).hexdigest() for name in archive.namelist()}
    assert members == {p.name: digest(p) for p in directory.iterdir()}
    (directory / 'EXPORT_VERIFIED.json').write_text(json.dumps({'zip_sha256': digest(archive_path),
             'member_sha256': members, 'csv_metadata_numerical_agreement': True}, indent=2) + '\n')
    print('analysis_complete', name, flush=True)


def self_check():
    from models.transformer import ModelConfig, TinyTransformer
    tokens = np.arange(800).reshape(400, 2)
    tokens[300:350] = tokens[:50]
    _, parts, _ = grouped_prefix_splits(tokens, [0, 1])
    keys = [{tokens[row // 2, :row % 2 + 1].tobytes() for row in part} for part in parts]
    assert keys[0].isdisjoint(keys[1]) and keys[0].isdisjoint(keys[2]) and keys[1].isdisjoint(keys[2])
    p = simplex(np.array([[2., -1., .2, .3], [.25, .25, .25, .25]]))
    assert np.all(p >= 0) and np.allclose(p.sum(-1), 1)
    model = TinyTransformer(ModelConfig(vocab_size=9, n_ctx=12, n_layers=1, d_model=8, n_heads=2, d_mlp=16)).eval()
    test_tokens = np.random.default_rng(9).integers(0, 9, (8, 12))
    with torch.no_grad():
        full = model(torch.as_tensor(test_tokens))
        prefix = model(torch.as_tensor(test_tokens[:, :5]))
        assert torch.allclose(full[:, :5], prefix, atol=2e-6)
    metric = native_metrics(model, test_tokens, 2, np.arange(8))
    assert metric['positions'] == 80 and metric['nll'] > 0 and 0 <= metric['accuracy'] <= 1
    from experiments.sphere_dual_screen import process, starts
    for n in (10, 20):
        proc = process(dict(n=n, gamma=.35, delta_v=.2, alpha=.95))
        initial = starts(proc.system, 74, 4)
        batch = proc.sample_batch(np.random.default_rng(75), 4, initial_state=initial)
        batch['initial_state'] = initial
        assert action_counterfactuals(proc, batch, np.arange(4)).shape == (4, 490, 4)
    print('prefix grouping, probability normalization and causal alignment passed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-check', action='store_true')
    for name in ('screen', 'results', 'output'):
        parser.add_argument('--' + name, type=Path)
    parser.add_argument('--training-revision')
    parser.add_argument('--index', type=int, choices=range(4))
    parser.add_argument('--trajectories', type=int, default=2048)
    args = parser.parse_args()
    resource.setrlimit(resource.RLIMIT_AS, (12 * 1024**3, 12 * 1024**3))
    if args.self_check:
        self_check()
    else:
        if args.trajectories < 256 or any(getattr(args, key) is None for key in
                                        ('screen', 'results', 'output', 'training_revision', 'index')):
            parser.error('four-model manifest, revision and at least256 trajectories required')
        analyze(args)
