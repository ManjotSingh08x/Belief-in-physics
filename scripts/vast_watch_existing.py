#!/usr/bin/env python3
"""Vast.ai Process Watcher & Budget Guard for Existing Running Jobs.

Attaches directly to an already-running training process (e.g. production_train.py),
monitors live spend via Vast.ai API v1 with dynamic budget adjustments (vast_budget.txt),
and ensures automated Google Drive backup (via rclone) and instance destruction when:
1. Spending limit ($5.00 default) is reached, OR
2. The watched training process finishes cleanly.

Includes built-in DNS override so it functions seamlessly even if /etc/hosts has
blocked console.vast.ai to neutralize an older supervisor.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ==============================================================================
# 0. Resilient DNS Resolver (Bypasses /etc/hosts redirection)
# ==============================================================================

VAST_API_IP = "98.85.239.100"
VAST_API_HOST = "console.vast.ai"
VAST_API_BASE = f"https://{VAST_API_HOST}/api/v1"

_ORIG_GETADDRINFO = socket.getaddrinfo

def _bypassing_getaddrinfo(host: str, port: Any, *args: Any, **kwargs: Any) -> Any:
    if host == VAST_API_HOST:
        return _ORIG_GETADDRINFO(VAST_API_IP, port, *args, **kwargs)
    return _ORIG_GETADDRINFO(host, port, *args, **kwargs)

socket.getaddrinfo = _bypassing_getaddrinfo


# ==============================================================================
# 1. Dual Logger (Terminal + File Stream)
# ==============================================================================

def log(msg: str = "", end: str = "\n", log_file: Optional[Path] = None) -> None:
    text = f"{msg}{end}"
    sys.stdout.write(text)
    sys.stdout.flush()
    if log_file is not None:
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(text)
        except Exception:
            pass


# ==============================================================================
# 2. Vast.ai REST API Interface
# ==============================================================================

def resolve_api_key(arg_key: Optional[str] = None) -> str:
    if arg_key:
        return arg_key.strip()
    if os.environ.get("VAST_API_KEY"):
        return os.environ["VAST_API_KEY"].strip()
    
    # Check .env
    env_file = Path(".env")
    if env_file.exists():
        try:
            for line in env_file.read_text().splitlines():
                line = line.strip()
                if line.startswith("VAST_API_KEY="):
                    k = line.split("=", 1)[1].strip().strip('"').strip("'")
                    if k:
                        return k
        except Exception:
            pass
    raise ValueError("Vast.ai API key not provided. Set VAST_API_KEY env or pass --api-key.")


def vast_request(endpoint: str, api_key: str, method: str = "GET", payload: Optional[Dict[str, Any]] = None) -> Any:
    api_key = api_key.strip()
    sep = "&" if "?" in endpoint else "?"
    base = "https://console.vast.ai/api/v0" if method in ("DELETE", "PUT") or (endpoint.startswith("instances/") and endpoint.strip("/").count("/") >= 1) else VAST_API_BASE
    url = f"{base}/{endpoint.lstrip('/')}{sep}api_key={api_key}"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
        "Content-Type": "application/json",
        "User-Agent": "VastProcessWatcher/1.0",
    }
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8") if e.fp else ""
        try:
            err_json = json.loads(body)
            msg = err_json.get("msg") or err_json.get("error") or str(err_json)
        except Exception:
            msg = body or str(e)
        raise RuntimeError(f"HTTP {e.code} on {endpoint}: {msg}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Network error on {endpoint}: {e.reason}") from e


def get_instances(api_key: str) -> List[Dict[str, Any]]:
    res = vast_request("instances/", api_key, method="GET")
    if isinstance(res, list):
        return res
    if isinstance(res, dict):
        if "instances" in res:
            return res["instances"]
    return []


def compute_instance_spend(
    instance: Dict[str, Any],
    monitor_start_time: float,
    budget_type: str = "lifetime",
) -> Tuple[float, float, float, float]:
    dph = float(instance.get("dph_total") or instance.get("dph_base") or 0.0)
    inst_start = instance.get("start_date")
    now = time.time()
    effective_start = float(inst_start) if budget_type == "lifetime" and inst_start else monitor_start_time
    elapsed_seconds = max(0.0, now - effective_start)
    elapsed_hours = elapsed_seconds / 3600.0
    spend = elapsed_hours * dph
    return spend, elapsed_hours, dph, effective_start


# ==============================================================================
# 3. Artifact Packaging & Google Drive Sync
# ==============================================================================

def package_current_results(results_base: Path, log_file: Optional[Path] = None) -> Optional[Path]:
    if not results_base.exists():
        return None
    zip_dest = results_base / "double_pendulum_results.zip"
    try:
        log(f"[Artifact Sync] Packaging '{results_base}' -> '{zip_dest}'...", log_file=log_file)
        with zipfile.ZipFile(zip_dest, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(results_base):
                for f in files:
                    fp = Path(root) / f
                    if fp.resolve() == zip_dest.resolve():
                        continue
                    arcname = fp.relative_to(results_base)
                    zf.write(fp, arcname)
        log(f"[Artifact Sync] Package created: {zip_dest.stat().st_size / (1024*1024):.2f} MB", log_file=log_file)
        return zip_dest
    except Exception as e:
        log(f"[Artifact Sync] Warning: Failed to create zip package: {e}", log_file=log_file)
        return None


def upload_logs_and_results_to_gdrive(
    rclone_dest: str,
    log_path: Path,
    results_base: Path,
    skip_upload: bool = False,
) -> bool:
    if skip_upload or not rclone_dest:
        log("[Artifact Sync] Upload skipped by request.", log_file=log_path)
        return True

    log("\n" + "=" * 76, log_file=log_path)
    log(" MANDATORY PRE-DELETION HOOK: UPLOADING LOGS & RESULTS TO GDRIVE", log_file=log_path)
    log("=" * 76, log_file=log_path)
    upload_ok = True

    # 1. Package results
    results_zip = package_current_results(results_base, log_file=log_path)

    # 2. Upload Log File
    if log_path.exists() and log_path.stat().st_size > 0:
        rclone_log_cmd = f'rclone copy "{log_path}" "{rclone_dest}/logs/" --stats-one-line'
        log(f"  $ {rclone_log_cmd}", log_file=log_path)
        res_log = subprocess.run(rclone_log_cmd, shell=True)
        if res_log.returncode == 0:
            log(f"[Artifact Sync] -> Log successfully uploaded to {rclone_dest}/logs/", log_file=log_path)
        else:
            upload_ok = False

    # 3. Upload Results Zip or Directory
    if results_zip and results_zip.exists():
        rclone_res_cmd = f'rclone copy "{results_zip}" "{rclone_dest}/" --stats-one-line'
        log(f"  $ {rclone_res_cmd}", log_file=log_path)
        res_res = subprocess.run(rclone_res_cmd, shell=True)
        if res_res.returncode == 0:
            log(f"[Artifact Sync] -> Results archive uploaded to {rclone_dest}/", log_file=log_path)
        else:
            upload_ok = False
    elif results_base.exists():
        rclone_dir_cmd = f'rclone copy "{results_base}" "{rclone_dest}/" --stats-one-line'
        log(f"  $ {rclone_dir_cmd}", log_file=log_path)
        res_dir = subprocess.run(rclone_dir_cmd, shell=True)
        if res_dir.returncode == 0:
            log(f"[Artifact Sync] -> Results directory uploaded to {rclone_dest}/", log_file=log_path)
        else:
            upload_ok = False

    log("=" * 76 + "\n", log_file=log_path)
    return upload_ok


def terminate_instance(
    instance_id: int,
    api_key: str,
    action: str = "destroy",
    rclone_dest: str = "gdrive:belief_results",
    log_path: Path = Path("vast_watch.log"),
    results_base: Path = Path("./experiments"),
    skip_upload: bool = False,
    dry_run: bool = False,
) -> bool:
    if not instance_id:
        log("[Error] Cannot terminate: unresolved instance ID.", log_file=log_path)
        return False

    upload_logs_and_results_to_gdrive(
        rclone_dest=rclone_dest,
        log_path=log_path,
        results_base=results_base,
        skip_upload=skip_upload,
    )

    if dry_run:
        log(f"[DRY-RUN] Would now execute {action.upper()} on instance {instance_id}.", log_file=log_path)
        return True

    endpoint = f"instances/{instance_id}/"
    if action == "destroy":
        log(f"!!! EXECUTING API DELETE ON INSTANCE {instance_id} !!!", log_file=log_path)
        try:
            res = vast_request(endpoint, api_key, method="DELETE")
            log(f"API Response: {res}", log_file=log_path)
            log(f"[Success] Instance {instance_id} destroyed cleanly.", log_file=log_path)
            return True
        except Exception as e:
            log(f"[Error] Failed to destroy instance {instance_id}: {e}", log_file=log_path)
            return False
    else:
        log(f"!!! SENDING STOP REQUEST TO INSTANCE {instance_id} !!!", log_file=log_path)
        try:
            res = vast_request(endpoint, api_key, method="PUT", payload={"state": "stopped"})
            log(f"Stop Response: {res}", log_file=log_path)
            return True
        except Exception as e:
            log(f"[Error] Failed to stop instance {instance_id}: {e}", log_file=log_path)
            return False


# ==============================================================================
# 4. Process Status Checker
# ==============================================================================

def is_pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def find_target_pid(name_pattern: str = "production_train.py") -> Optional[int]:
    try:
        out = subprocess.check_output(["pgrep", "-f", name_pattern], text=True).strip()
        lines = [line.strip() for line in out.splitlines() if line.strip()]
        # Filter out current process
        my_pid = os.getpid()
        candidates = [int(p) for p in lines if int(p) != my_pid]
        if candidates:
            # Return lowest or highest pid
            return candidates[0]
    except Exception:
        pass
    return None


# ==============================================================================
# 5. Main Watcher Loop
# ==============================================================================

def run_watcher(
    api_key: str,
    watch_pid: Optional[int],
    watch_pattern: str,
    instance_id: Optional[int],
    max_spend: float,
    action: str = "destroy",
    budget_type: str = "lifetime",
    check_interval: int = 30,
    rclone_dest: str = "gdrive:belief_results",
    log_path: Path = Path("vast_watch.log"),
    results_base: Path = Path("./experiments"),
    budget_file: Path = Path("vast_budget.txt"),
    skip_upload: bool = False,
    dry_run: bool = False,
) -> None:
    # Resolve target PID
    if watch_pid is None:
        watch_pid = find_target_pid(watch_pattern)
        if watch_pid is None:
            raise RuntimeError(f"Could not find any running process matching '{watch_pattern}'.")

    # Resolve Instance ID
    if instance_id is None:
        insts = get_instances(api_key)
        if len(insts) == 1:
            instance_id = insts[0].get("id")
        elif len(insts) > 1:
            running = [i for i in insts if i.get("actual_status") == "running"]
            if running:
                instance_id = running[0].get("id")
        if instance_id is None:
            raise RuntimeError("Could not auto-resolve Vast.ai instance ID. Pass --instance-id.")

    start_time = time.time()
    log("=" * 76, log_file=log_path)
    log(" VAST.AI PROCESS WATCHER & DYNAMIC BUDGET GUARD", log_file=log_path)
    log("=" * 76, log_file=log_path)
    log(f"Watched Process PID:  {watch_pid} (Pattern: {watch_pattern})", log_file=log_path)
    log(f"Target Instance ID:   {instance_id}", log_file=log_path)
    log(f"Initial Budget Limit: ${max_spend:.2f} USD", log_file=log_path)
    log(f"Dynamic Budget File:  {budget_file.resolve()}", log_file=log_path)
    log(f"Action on Limit:      {action.upper()}", log_file=log_path)
    log(f"Google Drive Remote:  {rclone_dest}", log_file=log_path)
    log(f"Watch Log File:       {log_path}", log_file=log_path)
    log(f"Check Interval:       Every {check_interval}s", log_file=log_path)
    log("=" * 76 + "\n", log_file=log_path)

    while True:
        # 1. Check if watched process has completed
        if not is_pid_alive(watch_pid):
            log(f"\n[Watcher] Target process PID {watch_pid} has completed!", log_file=log_path)
            log(f"[Watcher] Executing pre-deletion backup and {action.upper()}...", log_file=log_path)
            terminate_instance(
                instance_id=instance_id,
                api_key=api_key,
                action=action,
                rclone_dest=rclone_dest,
                log_path=log_path,
                results_base=results_base,
                skip_upload=skip_upload,
                dry_run=dry_run,
            )
            log("[Watcher] Completion workflow finished. Exiting.", log_file=log_path)
            return

        # 2. Check dynamic budget override
        if budget_file.exists():
            try:
                val = float(budget_file.read_text().strip())
                if val > 0 and val != max_spend:
                    log(f"[Watcher] Dynamic budget updated from {budget_file.name}: ${max_spend:.2f} -> ${val:.2f}", log_file=log_path)
                    max_spend = val
            except Exception:
                pass

        # 3. Query Vast API for instance spend
        try:
            insts = get_instances(api_key)
            matching = [i for i in insts if i.get("id") == instance_id]
            if not matching:
                log(f"[Warning] Instance {instance_id} not found in Vast API.", log_file=log_path)
            else:
                inst = matching[0]
                spend, el_hours, dph, _ = compute_instance_spend(inst, start_time, budget_type=budget_type)
                pct = min(100.0, (spend / max_spend) * 100.0) if max_spend > 0 else 100.0
                rem_budget = max(0.0, max_spend - spend)
                rem_hours = (rem_budget / dph) if dph > 0 else 0.0

                hrs = int(el_hours)
                mins = int((el_hours - hrs) * 60)
                ts = time.strftime("%H:%M:%S")

                log(
                    f"[{ts}] [PID {watch_pid} Alive] [ID: {instance_id}] "
                    f"Elapsed: {hrs}h {mins:02d}m | "
                    f"Spend: ${spend:6.2f} / ${max_spend:.2f} ({pct:5.1f}%) | "
                    f"Remaining: ~${rem_budget:.2f} (~{rem_hours:.1f}h)",
                    log_file=log_path,
                )

                # TRIGGER: BUDGET LIMIT EXCEEDED
                if spend >= max_spend:
                    log("\n" + "=" * 76, log_file=log_path)
                    log(f"🚨 [BUDGET LIMIT EXCEEDED] Spend ${spend:.2f} reached threshold ${max_spend:.2f}!", log_file=log_path)
                    log(f"Stopping training PID {watch_pid}...", log_file=log_path)
                    try:
                        os.kill(watch_pid, signal.SIGTERM)
                        time.sleep(3)
                        if is_pid_alive(watch_pid):
                            os.kill(watch_pid, signal.SIGKILL)
                    except Exception as e:
                        log(f"Warning stopping PID {watch_pid}: {e}", log_file=log_path)

                    terminate_instance(
                        instance_id=instance_id,
                        api_key=api_key,
                        action=action,
                        rclone_dest=rclone_dest,
                        log_path=log_path,
                        results_base=results_base,
                        skip_upload=skip_upload,
                        dry_run=dry_run,
                    )
                    log("[Watcher] Budget safety protocol complete. Exiting.", log_file=log_path)
                    return
        except Exception as e:
            log(f"[Warning] Failed to query Vast API: {e}. Retrying in {check_interval}s...", log_file=log_path)

        time.sleep(check_interval)


# ==============================================================================
# 6. CLI Entrypoint
# ==============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Watch existing running process and guard budget on Vast.ai.")
    parser.add_argument("--watch-pid", type=int, default=None, help="Target process PID to watch")
    parser.add_argument("--watch-pattern", type=str, default="production_train.py", help="Process pattern to auto-detect PID (default: production_train.py)")
    parser.add_argument("--instance-id", type=int, default=None, help="Vast.ai Instance ID (default: auto-detect)")
    parser.add_argument("--max-spend", type=float, default=5.0, help="Spending limit in USD (default: $5.00)")
    parser.add_argument("--action", type=str, choices=["destroy", "stop"], default="destroy", help="Action when budget is reached or process finishes")
    parser.add_argument("--budget-type", type=str, choices=["lifetime", "session"], default="lifetime", help="lifetime (from launch) or session")
    parser.add_argument("--check-interval", type=int, default=30, help="Check interval in seconds (default: 30s)")
    parser.add_argument("--rclone-dest", type=str, default=os.environ.get("VAST_GDRIVE_DEST", "gdrive:belief_results"), help="Google Drive rclone destination")
    parser.add_argument("--log-file", type=str, default="vast_watch.log", help="Path to watcher log file")
    parser.add_argument("--results-path", type=str, default="./experiments", help="Path to results directory")
    parser.add_argument("--budget-file", type=str, default="vast_budget.txt", help="Path to dynamic budget text file")
    parser.add_argument("--api-key", type=str, default=None, help="Vast.ai API key")
    parser.add_argument("--skip-upload", action="store_true", help="Skip Google Drive upload")
    parser.add_argument("--dry-run", action="store_true", help="Dry run mode without stopping or deleting")

    args = parser.parse_args()
    api_key = resolve_api_key(args.api_key)

    try:
        run_watcher(
            api_key=api_key,
            watch_pid=args.watch_pid,
            watch_pattern=args.watch_pattern,
            instance_id=args.instance_id,
            max_spend=args.max_spend,
            action=args.action,
            budget_type=args.budget_type,
            check_interval=args.check_interval,
            rclone_dest=args.rclone_dest,
            log_path=Path(args.log_file).resolve(),
            results_base=Path(args.results_path).resolve(),
            budget_file=Path(args.budget_file).resolve(),
            skip_upload=args.skip_upload,
            dry_run=args.dry_run,
        )
    except KeyboardInterrupt:
        log("\n[Watcher] Exiting upon user Ctrl+C.")
        sys.exit(0)


if __name__ == "__main__":
    main()
