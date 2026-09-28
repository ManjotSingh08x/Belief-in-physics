#!/usr/bin/env python3
"""Probe the maximum number of concurrent _worker.py processes at batch_size=1024.

Spawns 1 … MAX_PROBE workers simultaneously against a minimal smoke job queue,
waits for all to finish (or die), measures peak RSS and VRAM, increments until
any worker exits non-zero or the system runs out of GPU memory.

Usage:
    python scripts/probe_max_workers.py [--max-probe N] [--device DEVICE]

Output: table of (n_workers, status, peak_vram_mb, peak_rss_mb, wall_s)
        and a final "safe max workers = K" line.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Repo root resolution
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SWEEPS_DIR = ROOT / "experiments" / "pendulum-sweeps"
WORKER_SCRIPT = SWEEPS_DIR / "_worker.py"

# Minimal single smoke job (n_steps=5 → seq_len=20, tiny model d=32)
SMOKE_JOB = {
    "job_id": "probe_dt0.2_gamma0.3_dv0.7_n5_d32_seed0",
    "config_name": "probe_dt0.2_gamma0.3_dv0.7_n5",
    "delta_v": 0.7,
    "gamma": 0.3,
    "dt": 0.2,
    "n_steps": 5,
    "m": 4,
    "seq_len": 20,
    "width": 32,
    "seed": 0,
    "horizons": [1, 2, 5],
}

# Worker args that stay constant across all probe rounds
FIXED_WORKER_ARGS = [
    "--total-tokens", "20480",   # ~2 gradient steps at batch=1024
    "--batch-size", "1024",      # the invariant under test
    "--n-probe-traj", "16",
    "--cv-folds", "2",
    "--ridge-alphas", "1.0,10.0",
    "--n-layers", "2",
    "--n-heads", "1",
    "--log-every", "5",
]


# ---------------------------------------------------------------------------
# VRAM / RSS helpers
# ---------------------------------------------------------------------------

def _total_vram_mb(device: str) -> float:
    """Return total VRAM (MiB) on `device`, 0 if CPU."""
    if not device.startswith("cuda"):
        return 0.0
    try:
        import torch
        idx = int(device.split(":")[-1]) if ":" in device else 0
        return torch.cuda.get_device_properties(idx).total_memory / 1024**2
    except Exception:
        return 0.0


def _nvidia_smi_used_mb() -> float:
    """Read max used VRAM (MiB) across GPUs via nvidia-smi."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
        )
        return max(float(x) for x in out.strip().splitlines())
    except Exception:
        return 0.0


def _vram_mb(device: str) -> float:
    """Return current VRAM allocated (MiB) on `device`, 0 if CPU."""
    if not device.startswith("cuda"):
        return 0.0
    try:
        import torch
        idx = int(device.split(":")[-1]) if ":" in device else 0
        return torch.cuda.memory_allocated(idx) / 1024**2
    except Exception:
        return 0.0


def _peak_vram_mb(device: str) -> float:
    """Return peak VRAM allocated (MiB) since last reset, 0 if CPU."""
    if not device.startswith("cuda"):
        return 0.0
    try:
        import torch
        idx = int(device.split(":")[-1]) if ":" in device else 0
        return torch.cuda.max_memory_allocated(idx) / 1024**2
    except Exception:
        return 0.0


def _reset_peak_vram(device: str) -> None:
    if not device.startswith("cuda"):
        return
    try:
        import torch
        idx = int(device.split(":")[-1]) if ":" in device else 0
        torch.cuda.reset_peak_memory_stats(idx)
    except Exception:
        pass


def _rss_mb(pid: int) -> float:
    """Read /proc/<pid>/status VmRSS in MiB (Linux only)."""
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    except Exception:
        pass
    return 0.0


# ---------------------------------------------------------------------------
# Single probe round
# ---------------------------------------------------------------------------

def probe_round(
    n_workers: int,
    tmp_dir: Path,
    device: str,
) -> dict:
    """Spawn n_workers processes, wait for all, return result dict."""
    round_dir = tmp_dir / f"w{n_workers}"
    round_dir.mkdir(parents=True, exist_ok=True)

    # Each worker needs its own job_id so they don't skip each other's work.
    # Give every worker a distinct job (same physics, different seed) so they
    # all actually train instead of seeing an already-completed job.
    jobs = []
    for i in range(n_workers):
        j = dict(SMOKE_JOB)
        j["job_id"] = f"probe_w{n_workers}_worker{i}"
        j["seed"] = i
        jobs.append(j)

    queue_path = round_dir / "_job_queue.json"
    queue_path.write_text(json.dumps(jobs, indent=2))

    _reset_peak_vram(device)
    t0 = time.time()
    procs: list[subprocess.Popen] = []

    for i in range(n_workers):
        device_arg = f"cuda:{i % max(1, _gpu_count())}" if device.startswith("cuda") else "cpu"
        cmd = [
            sys.executable, str(WORKER_SCRIPT),
            "--queue-file", str(queue_path),
            "--output-dir", str(round_dir),
            "--worker-id", str(i),
            "--device", device_arg,
            *FIXED_WORKER_ARGS,
        ]
        p = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        procs.append(p)

    # Poll until done; sample peak RSS across all workers and peak VRAM via nvidia-smi
    peak_rss = 0.0
    peak_vram = 0.0
    while any(p.poll() is None for p in procs):
        for p in procs:
            if p.pid:
                peak_rss = max(peak_rss, _rss_mb(p.pid))
        if device.startswith("cuda"):
            peak_vram = max(peak_vram, _nvidia_smi_used_mb())
        time.sleep(0.2)

    wall_s = time.time() - t0

    # Collect exit codes and tail of stdout
    results = []
    for i, p in enumerate(procs):
        stdout, _ = p.communicate()
        results.append({"worker": i, "rc": p.returncode, "tail": stdout[-400:]})

    ok = all(r["rc"] == 0 for r in results)
    failed = [r for r in results if r["rc"] != 0]

    return {
        "n_workers": n_workers,
        "ok": ok,
        "failed": failed,
        "peak_vram_mb": round(peak_vram, 1),
        "peak_rss_mb": round(peak_rss, 1),
        "wall_s": round(wall_s, 1),
    }


def _gpu_count() -> int:
    try:
        import torch
        return max(1, torch.cuda.device_count())
    except Exception:
        return 1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-probe", type=int, default=16,
                        help="Maximum number of workers to probe (default: 16)")
    parser.add_argument("--device", type=str, default=None,
                        help="Base device: 'cpu', 'cuda', 'cuda:0'. Auto-detected if omitted.")
    parser.add_argument("--out", type=Path, default=None,
                        help="Write JSON results to this file.")
    args = parser.parse_args()

    # Auto-detect device
    device = args.device
    if device is None:
        try:
            import torch
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

    assert WORKER_SCRIPT.exists(), f"Worker script not found: {WORKER_SCRIPT}"

    import tempfile
    tmp_root = tempfile.mkdtemp(prefix="probe_max_workers_")
    tmp_dir = Path(tmp_root)

    total_vram = _total_vram_mb(device)
    max_vram_allowed = 0.8 * total_vram if total_vram > 0 else 0.0

    print(f"\n{'='*60}")
    print(f"  Worker saturation probe  |  batch_size=1024  |  device={device}")
    if total_vram > 0:
        print(f"  Total VRAM: {total_vram:.1f} MiB  |  80% limit: {max_vram_allowed:.1f} MiB")
    print(f"{'='*60}")
    print(f"{'workers':>8}  {'status':>8}  {'vram_MB':>9}  {'rss_MB':>8}  {'wall_s':>7}")
    print(f"{'-'*60}")

    rows = []
    safe_max = 0

    for n in range(1, args.max_probe + 1):
        result = probe_round(n, tmp_dir, device)
        status = "OK" if result["ok"] else "FAIL"
        print(
            f"{n:>8}  {status:>8}  "
            f"{result['peak_vram_mb']:>9.1f}  "
            f"{result['peak_rss_mb']:>8.1f}  "
            f"{result['wall_s']:>7.1f}",
            flush=True,
        )
        rows.append(result)

        if not result["ok"]:
            print(f"\n  First failure at {n} workers.")
            for f in result["failed"]:
                print(f"  Worker {f['worker']} (rc={f['rc']}) tail:\n{f['tail']}")
            break

        if max_vram_allowed > 0 and result["peak_vram_mb"] > max_vram_allowed:
            print(f"\n  Stopped: {n} workers reached peak VRAM {result['peak_vram_mb']:.1f} MiB, exceeding 80% GPU margin ({max_vram_allowed:.1f} MiB).")
            break

        safe_max = n

    print(f"\n{'='*60}")
    print(f"  Safe max workers at batch_size=1024 on {device}: {safe_max}")
    print(f"{'='*60}\n")

    if args.out:
        args.out.write_text(json.dumps({"device": device, "batch_size": 1024, "safe_max": safe_max, "rounds": rows}, indent=2))
        print(f"Results written to {args.out}")


if __name__ == "__main__":
    main()


# ── self-check ──────────────────────────────────────────────────────────────
def _demo():
    """Quick sanity: confirm SMOKE_JOB schema matches worker expectations."""
    required = {"job_id", "config_name", "delta_v", "gamma", "dt", "n_steps", "m", "seq_len", "width", "seed"}
    missing = required - SMOKE_JOB.keys()
    assert not missing, f"SMOKE_JOB missing keys: {missing}"
    assert SMOKE_JOB["m"] * SMOKE_JOB["n_steps"] == SMOKE_JOB["seq_len"]
    print("SMOKE_JOB schema OK")

if __name__ == "__main__":
    pass  # _demo() not called automatically — run with: python -c "from probe_max_workers import _demo; _demo()"
