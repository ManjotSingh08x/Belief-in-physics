"""One approved four-model study, using qualified one-hour Research Studio slices."""
import hashlib,json,subprocess,time
from dataclasses import asdict
from datetime import UTC,datetime,timedelta
from pathlib import Path
from closed_run_store import RunStore
from research_harness.run_types import Recipe,Resources

p=Path(__file__).resolve().parent;project=p/'source'
manifest=json.loads((p/'manifest.json').read_text());plan=json.loads((p/'plan.json').read_text())
revision=subprocess.check_output(['git','-C',str(project),'rev-parse','HEAD'],text=True).strip()
assert revision==manifest['source_revision']
store=RunStore(p/'run-control')
existing=store.list_requests(limit=100)
if any(store.attempt_count(r.id) and store.get_attempt(r.id).state in ('running','unknown','dispatch_intent') for r in existing):
    raise RuntimeError('existing attempt needs reconciliation; never duplicate')
first=next((i for i in range(16) if not any(r.recipe_name==f'sphere-alpha-slice-{i:02d}' for r in existing)),None)
if first is None:raise RuntimeError('slice bound reached')
python='/opt/watchdog/users/chayan-bit/research-world-model/exp028/venv/bin/python'
for i in range(first,16):
    if all((p/'results'/(j['name']+'.ready')).exists() for j in plan['jobs']):break
    while True:
        row=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(',')
        if int(row[0])>=12288 and int(row[1])<20:break
        (p/'controller_status.json').write_text(json.dumps(dict(phase='waiting_for_gpu_headroom',free_mib=int(row[0]),utilization_percent=int(row[1]),updated_at=time.time()))+'\n')
        if (p/'production_started.json').exists() and time.time()>json.loads((p/'production_started.json').read_text())['started_at']+plan['max_runtime_seconds']:raise RuntimeError('campaign runtime limit reached while waiting')
        time.sleep(30)
    recipe=Recipe(f'sphere-alpha-slice-{i:02d}',revision,'gpu',project,p/'runner-artifacts',
                  (python,'experiments/sphere_dual_slice.py','--control',str(p)),{},Resources(12*1024**3,3600,400),False,'kratos')
    store.register_recipe(recipe)
    request=store.submit(recipe.name,revision,{},revision,'chayan',datetime.now(UTC)+timedelta(hours=2),
        lineage={'hypothesis_id':'H-SPHERE-JOINT-ALPHA','experiment_id':'sphere-dual-alpha-four',
        'dataset_version':'Mess4-joint50-bounded-uniform-500','config_digest':manifest['files']['plan.json'],
        'seed':0,'environment':'Kratos-qualified-torch2.11.0-cu128-py3.14.4; four FP32 workers'})
    store.approve(request.id,request.digest,'chayan')
    attempt=store.dispatch(request.id,'chayan')
    state=dict(phase='running_slice',slice=i,controller_pid=__import__('os').getpid(),request_id=request.id,
               digest=request.digest,attempt=asdict(attempt),updated_at=time.time(),user_authorization='Kratos only using 4 workers')
    (p/'controller_status.json').write_text(json.dumps(state,default=str,indent=2)+'\n');print(json.dumps(state,default=str),flush=True)
    while True:
        attempt=store.reconcile(request.id)
        if attempt.state in ('ready_for_review','failed','unknown'):
            state.update(phase=attempt.state,attempt=asdict(attempt),updated_at=time.time())
            (p/'controller_status.json').write_text(json.dumps(state,default=str,indent=2)+'\n')
            store.backup(p/'run-control-backup.sqlite3')
            if attempt.state!='ready_for_review':raise RuntimeError('slice failed/unknown; preserve and inspect, never duplicate')
            break
        time.sleep(10)
    if (p/'results/SLICE_FAILURE.json').exists():raise RuntimeError('slice scientific/process failure; inspect')
else:raise RuntimeError('slice bound exhausted')
state.update(phase='four_gpu_models_complete',updated_at=time.time())
(p/'controller_status.json').write_text(json.dumps(state,default=str,indent=2)+'\n');print(json.dumps(state,default=str),flush=True)
