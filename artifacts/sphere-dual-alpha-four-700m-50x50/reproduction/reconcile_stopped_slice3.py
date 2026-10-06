import pathlib,json,hashlib,subprocess,time,sys
p=pathlib.Path('/opt/watchdog/users/chayan-bit/sphere-dual-alpha-four-20261006');sys.path.insert(0,str(p))
from closed_run_store import RunStore
s=json.loads((p/'controller_status.json').read_text()); assert s['slice']==3
for pid in [s['controller_pid']]+json.loads((p/'results/TRAIN_PIDS.json').read_text()): assert not pathlib.Path('/proc/'+str(pid)).exists(),pid
manifest=json.loads((p/'manifest.json').read_text())
for n,h in manifest['files'].items(): assert hashlib.file_digest((p/n).open('rb'),'sha256').hexdigest()==h,n
assert subprocess.check_output(['git','-C',str(p/'source'),'rev-parse','HEAD'],text=True).strip()==manifest['source_revision']
proof=dict(checked_at=time.time(),all_owned_workers_stopped=True,files={f.name:hashlib.file_digest(f.open('rb'),'sha256').hexdigest() for f in (p/'results').iterdir() if f.is_file()})
store=RunStore(p/'run-control');store.backup(p/'run-control-before-slice3-recovery.sqlite3')
a=store.reconcile(s['request_id']);assert a.state in ('unknown','failed'),a.state
if a.state=='unknown': a=store.resolve_unknown(s['request_id'],'chayan',confirm_stopped=True)
assert a.state=='failed';proof['failed_attempt_id']=a.id;proof['reconciled_state']=a.state
store.backup(p/'run-control-after-slice3-recovery.sqlite3')
(p/'SLICE3_PRESERVATION.json').write_text(json.dumps(proof,indent=2)+chr(10));print(json.dumps(proof))
