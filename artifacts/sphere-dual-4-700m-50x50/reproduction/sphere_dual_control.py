"""Guard one exact $5 rental. Credential comes from stdin and remains in CPU memory."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
import threading
import time
import urllib.request

REMOTE_SOURCE = '/workspace/sphere-dual-source'
REMOTE_OUTPUT = '/workspace/sphere-dual-results'


def budget_deadline(receipt):
    prior = receipt.get('prior_rental_reserve', 0.)
    available = 5 - receipt['transfer_reserve'] - prior
    if not 0 <= prior < 5 or available <= 0 or not 0 < receipt['guard_hourly_rate'] <= .20:
        raise ValueError('invalid hard budget receipt')
    return receipt['created_at'] + available / receipt['guard_hourly_rate'] * 3600


def main(args, credential):
    receipt = json.loads((args.control / 'rental_receipt.json').read_text())
    instance_id = receipt['instance_id']
    assert receipt['max_total_dollars'] == 5 and receipt['guard_hourly_rate'] <= .20
    deadline = budget_deadline(receipt)
    output = args.control / 'results'
    output.mkdir(exist_ok=True)
    plan = json.loads((args.control / 'plan.json').read_text())
    assert plan['total'] == len(plan['jobs']) == 4
    state = {'instance_id': instance_id, 'deadline': deadline, 'phase': 'setup',
             'max_total_dollars': 5, 'models_expected': 4, 'controller_pid': __import__('os').getpid()}
    lock = threading.RLock()
    release_lock = threading.Lock()
    released = threading.Event()
    connection = []
    def update(**fields):
        with lock:
            state.update(fields, updated_at=time.time())
            p = args.control / 'control_status.json'
            temp = p.with_suffix('.tmp')
            temp.write_text(json.dumps(state, indent=2) + '\n')
            temp.replace(p)
    def api(path, method='GET'):
        req = urllib.request.Request('https://console.vast.ai' + path, method=method,
                                     headers={'Authorization': 'Bearer ' + credential})
        with urllib.request.urlopen(req, timeout=20) as r:
            return json.load(r)
    def destroy(reason):
        with release_lock:
            if released.is_set():
                return
            update(phase='releasing', reason=reason)
            for _ in range(20):
                try:
                    live = api('/api/v1/instances')['instances']
                    if not any(i['id'] == instance_id for i in live):
                        released.set()
                        update(phase='gpu_released', destroyed_verified=True, destroyed_at=time.time())
                        return
                    api('/api/v0/instances/' + str(instance_id) + '/', 'DELETE')
                except Exception as e:
                    update(teardown_error=type(e).__name__)
                time.sleep(10)
            update(phase='teardown_failed')
    def watchdog():
        while not released.wait(15):
            if time.time() >= deadline - 300:
                destroy('absolute_budget_deadline')
                return
    threading.Thread(target=watchdog, daemon=True).start()
    update()
    options = ['-i', '/home/chayan/.ssh/sphere_vast_ed25519', '-o', 'IdentitiesOnly=yes',
               '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=accept-new', '-o', 'ConnectTimeout=10']
    def remote(command, timeout=30):
        return subprocess.run(connection + [command], check=True, text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout).stdout
    def sync():
        host, port = receipt['ssh_host'], receipt['ssh_port']
        subprocess.run(['rsync', '-a', '--partial', '-e', shlex.join(['ssh', *options, '-p', str(port)]),
                        'root@' + host + ':' + REMOTE_OUTPUT + '/', str(output) + '/'],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120)
    try:
        boot_deadline = min(time.time() + 900, deadline - 600)
        ssh_ready = False
        for attempt in range(60):
            if time.time() >= boot_deadline:
                break
            live = api('/api/v1/instances')['instances']
            item = next(i for i in live if i['id'] == instance_id)
            update(provider_status=item.get('actual_status'), setup_attempt=attempt + 1)
            if item.get('label') != receipt['label'] or float(item['dph_total']) > receipt['guard_hourly_rate']:
                raise RuntimeError('exact rental or hourly rate mismatch')
            host, port = item.get('ssh_host'), item.get('ssh_port')
            if host and port and re.fullmatch(r'[A-Za-z0-9.-]+', host):
                receipt.update(ssh_host=host, ssh_port=int(port))
                connection[:] = ['ssh', '-n', *options, '-p', str(port), 'root@' + host]
                try:
                    remote('true', 15)
                    ssh_ready = True
                    break
                except Exception:
                    pass
            time.sleep(10)
        if not ssh_ready:
            raise RuntimeError('GPU SSH unavailable')
        (args.control / 'rental_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
        remote('mkdir -p ' + REMOTE_SOURCE + ' ' + REMOTE_OUTPUT)
        scp = ['scp', *options, '-P', str(receipt['ssh_port'])]
        for name in ('source.tar.gz', 'screen.json', 'plan.json'):
            subprocess.run(scp + [str(args.control / name), 'root@' + receipt['ssh_host'] + ':/' + 'workspace/' + name],
                           stdin=subprocess.DEVNULL, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120)
        digest = hashlib.sha256((args.control / 'source.tar.gz').read_bytes()).hexdigest()
        check = 'import pathlib,hashlib; assert hashlib.sha256(pathlib.Path("/workspace/source.tar.gz").read_bytes()).hexdigest()==' + repr(digest)
        remote(shlex.join(['python', '-c', check]))
        remote('tar -xzf /workspace/source.tar.gz -C ' + REMOTE_SOURCE)
        update(phase='installing_dependencies', source_archive_sha256=digest)
        remote('command -v rsync || (apt-get update -qq && apt-get install -y -qq rsync)', 300)
        remote('python -m pip install --quiet numpy matplotlib pandas scikit-learn ipywidgets tqdm pytest', 600)
        remote('cd ' + REMOTE_SOURCE + ' && PYTHONPATH=. python tests/check_sphere_dual_four.py /workspace/screen.json /workspace/gpu_preflight.json', 300)
        command = ['python', '-m', 'experiments.sphere_dual_launch', '--screen', '/workspace/screen.json',
                   '--plan', '/workspace/plan.json', '--output', REMOTE_OUTPUT, '--deadline', str(deadline),
                   '--revision', receipt['source_revision']]
        spawn = ('import subprocess,pathlib; p=pathlib.Path(' + repr(REMOTE_OUTPUT) + '); '
                 'assert not (p/"LAUNCH_PID").exists(); log=(p/"launch.log").open("a"); '
                 'c=subprocess.Popen(' + repr(command) + ',cwd=' + repr(REMOTE_SOURCE) + ',stdout=log,stderr=log,start_new_session=True); '
                 '(p/"LAUNCH_PID").write_text(str(c.pid)); print(c.pid)')
        launcher_pid = int(remote(shlex.join(['python', '-c', spawn])).strip())
        update(phase='benchmark_then_training', launcher_pid=launcher_pid)
        while not released.is_set():
            sync()
            ready = []
            for job in plan['jobs']:
                name = job['name']
                marker = output / (name + '.ready')
                if not marker.exists():
                    continue
                meta = json.loads((output / (name + '.json')).read_text())
                weights = output / (name + '.pth')
                random = output / (name + '_random_init.pt')
                if (meta['configuration_sha256'] != job['configuration_sha256'] or meta['source_revision'] != receipt['source_revision']
                    or meta['tokens_seen'] != job['actual_tokens'] or meta['requested_tokens'] != 700000000
                    or hashlib.sha256(weights.read_bytes()).hexdigest() != meta['checkpoint_sha256']
                    or marker.read_text().strip() != meta['checkpoint_sha256']
                    or hashlib.sha256(random.read_bytes()).hexdigest() != meta['random_checkpoint_sha256']):
                    raise RuntimeError('transferred checkpoint verification failed')
                ready.append(name)
            update(verified_full_models=len(ready), ready_names=ready,
                   campaign_guard_spend=(time.time() - receipt['created_at']) / 3600 * receipt['guard_hourly_rate'] + receipt['transfer_reserve'] + receipt.get('prior_rental_reserve', 0.))
            if (output / 'GPU_FAILURE.json').exists():
                raise RuntimeError('GPU launcher failed; artifacts synchronized')
            if len(ready) == 4:
                sync()
                destroy('four_models_transferred_and_verified')
                break
            if time.time() >= deadline - 600:
                remote('kill -TERM ' + str(launcher_pid))
                time.sleep(20)
                sync()
                destroy('hard_budget_stop')
                break
            time.sleep(30)
    except BaseException as error:
        update(failure=type(error).__name__, failure_message=str(error))
        if connection:
            try:
                sync()
            except Exception:
                pass
        destroy('failure')
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--control', type=Path, required=True)
    args = p.parse_args()
    key = sys.stdin.readline().strip()
    if not key:
        raise ValueError('missing memory-only provider credential')
    main(args, key)
