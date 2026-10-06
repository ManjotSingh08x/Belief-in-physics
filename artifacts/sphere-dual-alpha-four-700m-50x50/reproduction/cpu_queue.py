"""Wait for verified GPU release, then analyze the four models sequentially on CPU."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile


def verify(directory):
    evidence = json.loads((directory / 'analysis.json').read_text())
    export = json.loads((directory / 'EXPORT_VERIFIED.json').read_text())
    archive = directory.parent / (directory.name + '.zip')
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == export['zip_sha256']
    with zipfile.ZipFile(archive) as z:
        hashes = {name: hashlib.sha256(z.read(name)).hexdigest() for name in z.namelist()}
    assert hashes == export['member_sha256']
    for name, expected in hashes.items():
        assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == expected
    with (directory / 'probes.csv').open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len(evidence['probes']) == 16
    for original, exported in zip(evidence['probes'], rows):
        for key in ('alpha', 'validation_r2', 'test_r2', 'feature_dim', 'train_rows_per_feature'):
            assert float(exported[key]) == original[key]
    return evidence


def main(args):
    control = args.control
    receipt = json.loads((control / 'training_receipt.json').read_text())
    plan = json.loads((control / 'plan.json').read_text())
    assert receipt['models'] == plan['total'] == len(plan['jobs']) == 4
    deadline = time.time() + 18*3600
    while not (control/'NATIVE_TRANSFER_VERIFIED.json').exists():
        if time.time()>deadline:
            raise RuntimeError('native full-model transfer not verified in bounded wait')
        time.sleep(30)
    state=json.loads((control/'NATIVE_TRANSFER_VERIFIED.json').read_text())
    if state.get('verified_full_models') != 4 or not state.get('gpu_workers_finished'):
        raise RuntimeError('native GPU workers/full-model transfer unverified')
    if len(Path('/proc/swaps').read_text().splitlines()) != 1:
        raise RuntimeError('CPU host swap must be disabled')
    output = control / 'analysis'
    env = dict(os.environ, PYTHONPATH=str(control / 'source'), OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='2')
    evidence = []
    for job in plan['jobs']:
        directory = output / job['name']
        if not (directory / 'EXPORT_VERIFIED.json').exists():
            log = (control / f'analysis_{job["index"]}.log').open('a')
            command = [sys.executable, str(control / 'sphere_dual_analyze.py'), '--screen', str(control / 'screen.json'),
                       '--results', str(control / 'results'), '--output', str(output), '--index', str(job['index']),
                       '--training-revision', receipt['source_revision'], '--trajectories', '2048']
            subprocess.run(command, env=env, cwd=control / 'source', stdout=log, stderr=log, check=True, timeout=10800)
            log.close()
        evidence.append(verify(directory))
        (control / 'analysis_status.json').write_text(json.dumps({'phase': 'cpu_analysis',
            'validated_models': len(evidence), 'updated_at': time.time()}, indent=2) + '\n')
    ranking = []
    for model in evidence:
        for target in ('belief', 'physics'):
            pair = {r['model_type']: r for r in model['probes'] if r['mode'] == 'all_block_layers_single_token'
                    and r['target'] == target and r['control'] == 'matched'}
            assert set(pair) == {'trained', 'random_init'}
            exp = model['experiment']
            ranking.append({'model_name': model['model_name'], 'n': exp['n'], 'gamma': exp['physics']['gamma'],
                'delta_v': exp['physics']['delta_v'], 'alpha': exp['hmm']['alpha'], 'target': target,
                'trained_r2': pair['trained']['test_r2'], 'random_r2': pair['random_init']['test_r2'],
                'gap': pair['trained']['test_r2'] - pair['random_init']['test_r2']})
    ranking.sort(key=lambda r: (r['target'], -r['gap']))
    with (output / 'gap_ranking.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, ranking[0].keys()); writer.writeheader(); writer.writerows(ranking)
    summary = {'models': 4, 'requested_tokens_each': 700000000, 'actual_tokens_each': 699904000,
               'training_source_revision': receipt['source_revision'], 'training_host': receipt['host'],
               'gpu_completion': state, 'ranking': ranking,
               'native_head_metrics': {m['model_name']: m['native_head_metrics'] for m in evidence},
               'scientific_acceptance': 'one training seed measured; physical preflight selected settings, not proven optimum',
               'analysis_script_sha256': hashlib.sha256((control / 'sphere_dual_analyze.py').read_bytes()).hexdigest()}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    (control / 'analysis_status.json').write_text(json.dumps({'phase': 'complete', 'validated_models': 4,
        'exports_verified': 4, 'updated_at': time.time()}, indent=2) + '\n')
    print('four CPU analyses and exports verified', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--control', type=Path, required=True)
    args = parser.parse_args()
    try:
        main(args)
    except BaseException as error:
        (args.control / 'analysis_status.json').write_text(json.dumps({'phase': 'failed',
            'error': type(error).__name__, 'message': str(error), 'updated_at': time.time()}, indent=2) + '\n')
        raise
