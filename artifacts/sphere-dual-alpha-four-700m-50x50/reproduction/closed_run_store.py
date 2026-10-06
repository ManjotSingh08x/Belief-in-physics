"""Private study controller: close every SQLite connection after its transaction."""
from contextlib import contextmanager
from research_harness.run_control import RunStore as BaseRunStore

class RunStore(BaseRunStore):
    @contextmanager
    def _connect(self):
        connection = super()._connect()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

if __name__ == '__main__':
    import gc, json, os, resource, sqlite3, tempfile
    old = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (128, old[1]))
    with tempfile.TemporaryDirectory() as root:
        gc.disable()
        baseline = BaseRunStore(__import__('pathlib').Path(root)/'baseline')
        reproduced = False
        try:
            for i in range(2000): baseline.list_requests(limit=1)
        except sqlite3.OperationalError as error:
            reproduced = True
            baseline_error = str(error)
        gc.collect()
        assert reproduced, 'baseline did not reproduce; reconsider diagnosis'
        fixed = RunStore(__import__('pathlib').Path(root)/'fixed')
        before = len(os.listdir('/proc/self/fd'))
        for _ in range(2000): fixed.list_requests(limit=1)
        after = len(os.listdir('/proc/self/fd'))
        assert after == before, (before, after)
        gc.enable()
        resource.setrlimit(resource.RLIMIT_NOFILE, old)
        print(json.dumps(dict(baseline_error=baseline_error, baseline_failed_after_reads=i,
                             fixed_reads=2000, descriptors_before=before, descriptors_after=after)))
