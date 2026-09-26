#!/usr/bin/env python3
"""Vast.ai Instance Cost, Budget Safety Guard & Automated Results Exporter.

Monitors an active Vast.ai instance and terminates/destroys it if spending
exceeds a predefined budget (default: $5.00), preventing unexpected charges.

MANDATORY SAFETY RULE:
In any case where the instance is to be deleted/destroyed (whether due to budget
limit reached, training command completion, error, or user interrupt), the service
FIRST uploads its own execution logs (vast_monitor.log) and all current results
(double_pendulum_results.zip or dynamically packaged partial results) to Google Drive
through rclone, and ONLY THEN issues the API call to destroy the instance.

Zero external Python dependencies: uses standard library (urllib, subprocess, zipfile).
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
import dotenv
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

VAST_API_BASE = "https://console.vast.ai/api/v1"
dotenv.load_dotenv()
# ==============================================================================
# 1. Dual Tee Logger (Terminal + File Stream)
# ==============================================================================

class TeeLogger:
    """Tees stdout and stderr simultaneously to terminal and a log file."""
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.terminal_stdout = sys.stdout
        self.terminal_stderr = sys.stderr
        self.log_file = open(log_path, "a", encoding="utf-8")

    def write_out(self, msg: str) -> None:
        self.terminal_stdout.write(msg)
        self.terminal_stdout.flush()
        self.log_file.write(msg)
        self.log_file.flush()

    def write_err(self, msg: str) -> None:
        self.terminal_stderr.write(msg)
        self.terminal_stderr.flush()
        self.log_file.write(msg)
        self.log_file.flush()

    def flush(self) -> None:
        self.terminal_stdout.flush()
        self.terminal_stderr.flush()
        self.log_file.flush()

    def close(self) -> None:
        self.log_file.close()


GLOBAL_LOGGER: Optional[TeeLogger] = None


def log(msg: str = "", end: str = "\n") -> None:
    full_msg = f"{msg}{end}"
    if GLOBAL_LOGGER is not None:
        GLOBAL_LOGGER.write_out(full_msg)
    else:
        print(msg, end=end, flush=True)


# ==============================================================================
# 2. Vast.ai REST API Helper & Credentials Resolution
# ==============================================================================

def resolve_api_key(arg_key: Optional[str] = None) -> str:
    """Resolves Vast.ai API key from CLI argument, environment, or config file."""
    if arg_key and arg_key.strip():
        return arg_key.strip()

    # Check environment variables
    for env_var in ("VAST_API_KEY", "VAST_KEY", "VASTAI_API_KEY"):
        val = os.environ.get(env_var, "").strip()
        if val:
            return val

    # Check ~/.vast_api_key or ~/.config/vastai/vast_api_key
    home = Path.home()
    candidates = [
        home / ".vast_api_key",
        home / ".config" / "vastai" / "vast_api_key",
    ]
    for p in candidates:
        if p.exists():
            try:
                key = p.read_text().strip()
                if key:
                    return key
            except Exception:
                pass

    raise ValueError(
        "Vast.ai API key not found. Please provide it via:\n"
        "  1. CLI argument: --api-key <YOUR_KEY>\n"
        "  2. Environment:  export VAST_API_KEY=<YOUR_KEY>\n"
        "  3. Config file:  echo <YOUR_KEY> > ~/.vast_api_key"
    )


def resolve_instance_id(cli_id: Optional[int] = None) -> Optional[int]:
    """Resolves target Vast.ai instance ID from CLI, environment, or container metadata.

    Particularly critical in Organization / Team environments where multiple instances
    may be active concurrently under the same account.
    """
    if cli_id is not None:
        return cli_id

    # 1. Direct environment variables (set by user or Vast startup)
    for env_var in ("VAST_INSTANCE_ID", "CONTAINER_ID", "VAST_CONTAINER_ID", "INSTANCE_ID"):
        val = os.environ.get(env_var, "").strip()
        if val and val.isdigit():
            return int(val)

    # 2. VAST_CONTAINERLABEL (often 'C.1482910')
    label = os.environ.get("VAST_CONTAINERLABEL", "").strip()
    if label:
        digits = "".join(ch for ch in label if ch.isdigit())
        if digits:
            return int(digits)

    # 3. /etc/environment (injected in container sessions)
    etc_env = Path("/etc/environment")
    if etc_env.exists():
        try:
            for line in etc_env.read_text().splitlines():
                if "=" in line:
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip('"').strip("'")
                    if k in ("VAST_INSTANCE_ID", "CONTAINER_ID", "VAST_CONTAINER_ID", "INSTANCE_ID") and v.isdigit():
                        return int(v)
                    if k == "VAST_CONTAINERLABEL":
                        digits = "".join(ch for ch in v if ch.isdigit())
                        if digits:
                            return int(digits)
        except Exception:
            pass

    # 4. /proc/1/environ (container PID 1 environment)
    proc_env = Path("/proc/1/environ")
    if proc_env.exists():
        try:
            raw = proc_env.read_bytes().split(b"\x00")
            for entry in raw:
                try:
                    line = entry.decode("utf-8", errors="ignore")
                    if "=" in line:
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip('"').strip("'")
                        if k in ("VAST_INSTANCE_ID", "CONTAINER_ID", "VAST_CONTAINER_ID", "INSTANCE_ID") and v.isdigit():
                            return int(v)
                        if k == "VAST_CONTAINERLABEL":
                            digits = "".join(ch for ch in v if ch.isdigit())
                            if digits:
                                return int(digits)
                except Exception:
                    pass
        except Exception:
            pass

    return None


def vast_request(
    endpoint: str,
    api_key: str,
    method: str = "GET",
    payload: Optional[Dict[str, Any]] = None,
) -> Any:
    """Executes authenticated REST request to Vast.ai API using urllib."""
    api_key = api_key.strip()
    sep = "&" if "?" in endpoint else "?"
    url = f"{VAST_API_BASE}/{endpoint.lstrip('/')}{sep}api_key={api_key}"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
        "User-Agent": "VastCostMonitor/1.0",
    }
    data = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(payload).encode("utf-8")

    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = resp.read().decode("utf-8")
            if not body:
                return {}
            return json.loads(body)
    except urllib.error.HTTPError as e:
        err_msg = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Vast.ai API HTTP {e.code} Error on {method} {url}: {err_msg}")
    except Exception as e:
        raise RuntimeError(f"Vast.ai API Network Error on {method} {url}: {e}")


def get_instances(api_key: str) -> List[Dict[str, Any]]:
    """Retrieves all active instances for the account."""
    res = vast_request("instances/", api_key, method="GET")
    if isinstance(res, list):
        return res
    if isinstance(res, dict):
        if "instances" in res:
            return res["instances"]
        if res.get("success") is False:
            err_msg = res.get("msg") or res.get("error") or res
            log(f"[Warning] Vast.ai API rejected listing instances: {err_msg}")
    return []


def send_webhook(webhook_url: Optional[str], message: str) -> None:
    """Optional Discord/Slack webhook notification."""
    if not webhook_url:
        return
    try:
        req = urllib.request.Request(
            webhook_url,
            data=json.dumps({"content": message, "text": message}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=5)
    except Exception as e:
        log(f"[Notice] Failed to send webhook alert: {e}")


# ==============================================================================
# 3. Packaging & Google Drive Upload via rclone
# ==============================================================================

def package_current_results(results_base: Path) -> Optional[Path]:
    """Finds or dynamically bundles all available results into a distribution zip."""
    if not results_base.exists():
        return None

    # 1. Check if complete master archive already exists
    master_zip = results_base / "double_pendulum_results.zip"
    if master_zip.exists() and master_zip.stat().st_size > 0:
        return master_zip

    # 2. Check in parent or output_base
    parent_master = results_base.parent / "double_pendulum_results.zip"
    if parent_master.exists() and parent_master.stat().st_size > 0:
        return parent_master

    # 3. Dynamically bundle whatever results currently exist on disk
    bundle_zip = results_base / f"current_results_snapshot_{int(time.time())}.zip"
    log(f"[Packaging] Master zip not found; bundling current partial results into {bundle_zip.name}...")

    added_arcnames = set()
    files_found = 0
    with zipfile.ZipFile(bundle_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        # Include state manifest and master CSV if present
        for meta_file in ["pipeline_state.json", "master_probe_results.csv"]:
            p = results_base / meta_file
            if p.exists() and p.is_file() and p.name not in added_arcnames:
                zf.write(p, arcname=p.name)
                added_arcnames.add(p.name)
                files_found += 1
            p_par = results_base.parent / meta_file
            if p_par.exists() and p_par.is_file() and p_par.name not in added_arcnames:
                zf.write(p_par, arcname=p_par.name)
                added_arcnames.add(p_par.name)
                files_found += 1

        # Include everything in results/ directory
        res_sub = results_base / "results"
        target_dir = res_sub if res_sub.exists() else results_base
        for f in target_dir.rglob("*"):
            if f.is_file() and f != bundle_zip and not f.name.endswith(".tmp"):
                arcname = str(f.relative_to(results_base))
                if arcname not in added_arcnames:
                    zf.write(f, arcname=arcname)
                    added_arcnames.add(arcname)
                    files_found += 1

    if files_found > 0:
        log(f"[Packaging] Successfully bundled {files_found} files ({bundle_zip.stat().st_size / 1024:.1f} KB) -> {bundle_zip}")
        return bundle_zip
    else:
        log("[Packaging] No result files found to bundle.")
        return None


def upload_logs_and_results_to_gdrive(
    rclone_dest: str,
    log_path: Path,
    results_base: Path,
) -> bool:
    """MANDATORY PRE-DELETION HOOK:

    Uploads service execution logs (vast_monitor.log) and current training results
    to Google Drive via rclone before allowing the instance to be deleted.
    """
    log("\n" + "=" * 76)
    log(" MANDATORY PRE-DELETION HOOK: UPLOADING LOGS & CURRENT RESULTS TO GDRIVE")
    log("=" * 76)
    log(f"Rclone Destination: {rclone_dest}")
    log(f"Service Log File:   {log_path}")
    log(f"Results Base Path:  {results_base}")

    # Ensure log buffers are flushed to disk before copying
    if GLOBAL_LOGGER is not None:
        GLOBAL_LOGGER.flush()

    upload_ok = True

    # 1. Package available results
    results_zip = package_current_results(results_base)

    # 2. Upload Service Log to Google Drive
    if log_path.exists() and log_path.stat().st_size > 0:
        log(f"\n[Artifact Sync] Uploading service execution log ({log_path.stat().st_size:,} bytes)...")
        rclone_log_cmd = f'rclone copy "{log_path}" "{rclone_dest}/logs/" --stats-one-line'
        log(f"  $ {rclone_log_cmd}")
        res_log = subprocess.run(rclone_log_cmd, shell=True)
        if res_log.returncode == 0:
            log(f"[Artifact Sync] -> Service log successfully uploaded to {rclone_dest}/logs/{log_path.name}")
        else:
            log(f"[Artifact Sync] Warning: rclone copy for log file exited with code {res_log.returncode}")
            upload_ok = False
    else:
        log("[Artifact Sync] Notice: Log file is empty or missing; skipping log upload.")

    # 3. Upload Results to Google Drive
    if results_zip and results_zip.exists():
        log(f"\n[Artifact Sync] Uploading results archive ({results_zip.stat().st_size / (1024*1024):.2f} MB)...")
        rclone_res_cmd = f'rclone copy "{results_zip}" "{rclone_dest}/" --stats-one-line'
        log(f"  $ {rclone_res_cmd}")
        res_results = subprocess.run(rclone_res_cmd, shell=True)
        if res_results.returncode == 0:
            log(f"[Artifact Sync] -> Results successfully uploaded to {rclone_dest}/{results_zip.name}")
        else:
            log(f"[Artifact Sync] Warning: rclone copy for results exited with code {res_results.returncode}")
            upload_ok = False
    elif results_base.exists():
        log(f"\n[Artifact Sync] Uploading entire results directory '{results_base}'...")
        rclone_dir_cmd = f'rclone copy "{results_base}" "{rclone_dest}/" --stats-one-line'
        log(f"  $ {rclone_dir_cmd}")
        res_dir = subprocess.run(rclone_dir_cmd, shell=True)
        if res_dir.returncode == 0:
            log(f"[Artifact Sync] -> Directory successfully uploaded to {rclone_dest}/")
        else:
            log(f"[Artifact Sync] Warning: rclone copy for directory exited with code {res_dir.returncode}")
            upload_ok = False
    else:
        log(f"[Artifact Sync] Notice: No results found at {results_base} to upload.")

    log("=" * 76)
    if upload_ok:
        log("[Artifact Sync] Pre-deletion Google Drive upload PASSED successfully!")
    else:
        log("[Artifact Sync] Pre-deletion upload encountered warnings/errors (see above).")
    log("=" * 76 + "\n")
    return upload_ok


def sync_to_local_machine(
    instance: Dict[str, Any],
    local_dir: str,
    remote_path: str,
    ssh_key_path: Optional[str] = None,
) -> bool:
    """Pulls remote files to local machine using rsync / scp if monitoring from laptop."""
    local_target = Path(local_dir).expanduser().resolve()
    local_target.mkdir(parents=True, exist_ok=True)
    ssh_host = instance.get("ssh_host")
    ssh_port = instance.get("ssh_port")
    if not ssh_host or not ssh_port:
        log(f"[Local Sync] Instance {instance.get('id')} missing ssh_host/ssh_port; cannot pull.")
        return False

    log(f"\n[Local Sync] Pulling '{remote_path}' from root@{ssh_host}:{ssh_port} -> {local_target}...")
    ssh_opts = f"-p {ssh_port} -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
    if ssh_key_path:
        ssh_opts += f" -i {ssh_key_path}"

    rsync_cmd = f"rsync -avz -e 'ssh {ssh_opts}' root@{ssh_host}:{remote_path} '{local_target}/'"
    log(f"  $ {rsync_cmd}")
    res = subprocess.run(rsync_cmd, shell=True)
    if res.returncode == 0:
        log(f"[Local Sync] Artifacts successfully transferred to {local_target}!")
        return True

    log(f"[Local Sync] rsync exited with code {res.returncode}. Attempting fallback via scp...")
    scp_cmd = f"scp -P {ssh_port} -o StrictHostKeyChecking=no root@{ssh_host}:{remote_path} '{local_target}/'"
    log(f"  $ {scp_cmd}")
    res_scp = subprocess.run(scp_cmd, shell=True)
    if res_scp.returncode == 0:
        log(f"[Local Sync] Artifacts successfully transferred to {local_target} via scp!")
        return True

    log("[Local Sync] Error: Both rsync and scp failed to transfer artifacts.")
    return False


# ==============================================================================
# 4. Instance Termination Actions (with Mandatory Pre-Deletion Hook)
# ==============================================================================

def terminate_instance(
    instance: Dict[str, Any],
    api_key: str,
    action: str = "destroy",
    rclone_dest: Optional[str] = "gdrive:belief_results",
    log_path: Optional[Path] = None,
    results_base: Optional[Path] = None,
    sync_local_dir: Optional[str] = None,
    remote_path: str = "./experiments/double_pendulum_results.zip",
    ssh_key_path: Optional[str] = None,
    skip_upload: bool = False,
    dry_run: bool = False,
) -> bool:
    """MANDATORY INVARIANT:

    If the instance has to be deleted (action == 'destroy'):
    First upload service logs and current results to Google Drive via rclone
    (and optionally pull to local machine), and ONLY THEN delete the instance.
    """
    instance_id = instance.get("id")

    if action == "destroy":
        log("\n" + "#" * 76)
        log(f"# INITIATING DELETION SEQUENCE FOR INSTANCE {instance_id}")
        log("#" * 76)

        # MANDATORY STEP 1: Upload to Google Drive via rclone
        if not skip_upload and rclone_dest:
            upload_logs_and_results_to_gdrive(
                rclone_dest=rclone_dest,
                log_path=log_path or Path("vast_monitor.log"),
                results_base=results_base or Path("./experiments"),
            )

        # OPTIONAL STEP 2: Pull to local machine if running on local laptop
        if sync_local_dir:
            sync_to_local_machine(
                instance=instance,
                local_dir=sync_local_dir,
                remote_path=remote_path,
                ssh_key_path=ssh_key_path,
            )

        # MANDATORY STEP 3: ONLY NOW execute the API DELETE call
        if dry_run:
            log(f"[DRY-RUN] Pre-deletion upload completed. Would now execute DELETE on instance {instance_id}.")
            return True

        if not instance_id or instance_id == 0:
            log(f"[Error] Cannot destroy instance: instance ID is unresolved ({instance_id}). Please specify --instance-id <ID> explicitly.")
            return False

        endpoint = f"instances/{instance_id}/"
        log(f"\n!!! EXECUTING API DELETE ON INSTANCE {instance_id} !!!")
        try:
            res = vast_request(endpoint, api_key, method="DELETE")
            log(f"Vast.ai API Response: {res}")
            if isinstance(res, dict) and res.get("success") is False:
                err_msg = res.get("msg") or res.get("error") or res
                log(f"[Error] Vast.ai API rejected instance destruction: {err_msg}")
                return False
            log(f"[Success] Instance {instance_id} destroyed cleanly after all data was secured.")
            return True
        except Exception as e:
            log(f"[Error] Failed to destroy instance {instance_id}: {e}")
            return False

    elif action == "stop":
        if dry_run:
            log(f"[DRY-RUN] Would execute STOP on instance {instance_id}.")
            return True

        if not instance_id or instance_id == 0:
            log(f"[Error] Cannot stop instance: instance ID is unresolved ({instance_id}). Please specify --instance-id <ID> explicitly.")
            return False

        endpoint = f"instances/{instance_id}/"
        log(f"!!! SENDING PUT REQUEST TO STOP INSTANCE {instance_id} !!!")
        try:
            res = vast_request(endpoint, api_key, method="PUT", payload={"state": "stopped"})
            log(f"Stop request response: {res}")
            if isinstance(res, dict) and res.get("success") is False:
                err_msg = res.get("msg") or res.get("error") or res
                log(f"[Error] Vast.ai API rejected instance stop: {err_msg}")
                return False
            return True
        except Exception as e:
            log(f"Error stopping instance {instance_id}: {e}")
            return False
    else:
        raise ValueError(f"Unknown action: {action}. Must be 'destroy' or 'stop'.")


# ==============================================================================
# 5. Cost & Spend Calculation Logic
# ==============================================================================

def compute_instance_spend(
    instance: Dict[str, Any],
    monitor_start_time: float,
    budget_type: str = "lifetime",
) -> Tuple[float, float, float, float]:
    """Computes spend ($), elapsed hours, rate ($/hr), and start timestamp."""
    dph = float(instance.get("dph_total") or instance.get("dph_base") or 0.0)
    inst_start = instance.get("start_date")

    now = time.time()
    if budget_type == "session" or not inst_start:
        effective_start = monitor_start_time
    else:
        effective_start = float(inst_start)

    elapsed_seconds = max(0.0, now - effective_start)
    elapsed_hours = elapsed_seconds / 3600.0
    spend = elapsed_hours * dph

    return spend, elapsed_hours, dph, effective_start


# ==============================================================================
# 6. Monitoring Loop & Process Supervision
# ==============================================================================

def run_monitor(
    api_key: str,
    target_instance_id: Optional[int],
    max_spend: float,
    action: str = "destroy",
    budget_type: str = "lifetime",
    check_interval: int = 30,
    cmd: Optional[str] = None,
    destroy_on_cmd_finish: bool = False,
    rclone_dest: str = "gdrive:belief_results",
    log_path: Path = Path("vast_monitor.log"),
    results_base: Path = Path("./experiments"),
    sync_local_dir: Optional[str] = None,
    remote_path: str = "./experiments/double_pendulum_results.zip",
    ssh_key_path: Optional[str] = None,
    skip_upload: bool = False,
    webhook_url: Optional[str] = None,
    dry_run: bool = False,
) -> None:
    """Main monitoring loop with auto-termination and pre-termination artifact sync."""
    target_instance_id = resolve_instance_id(target_instance_id)
    monitor_start_time = time.time()
    log("=" * 76)
    log(" VAST.AI BUDGET SAFETY & AUTOMATED GDRIVE RESULTS EXPORTER")
    log("=" * 76)
    log(f"Spending Limit:       ${max_spend:.2f} USD")
    log(f"Action on Limit:      {action.upper()} instance")
    log(f"Budget Metric:        {budget_type.upper()} ({'From instance launch' if budget_type == 'lifetime' else 'From monitor start'})")
    log(f"Target Instance:      {target_instance_id if target_instance_id else 'Auto-detect from Vast API'}")
    log(f"Check Interval:       Every {check_interval} seconds")
    log(f"Google Drive Remote:  {rclone_dest}")
    log(f"Service Log File:     {log_path}")
    log(f"Results Directory:    {results_base}")
    log(f"Supervised Command:   {cmd if cmd else 'None (Stand-alone polling daemon)'}")
    if sync_local_dir:
        log(f"Sync to Local Laptop: {sync_local_dir} (pulling {remote_path})")
    log(f"Dry-Run Mode:         {dry_run}")
    log("=" * 76)

    # Launch supervised command if requested
    proc: Optional[subprocess.Popen] = None
    if cmd:
        log(f"\n[Supervisor] Launching child process under active monitoring:\n  $ {cmd}\n")
        proc = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        def pipe_stdout():
            if proc and proc.stdout:
                for line in iter(proc.stdout.readline, ""):
                    log(line, end="")
                proc.stdout.close()

        t_pipe = threading.Thread(target=pipe_stdout, daemon=True)
        t_pipe.start()

    try:
        while True:
            # Check if supervised child command completed
            if proc is not None:
                ret = proc.poll()
                if ret is not None:
                    log(f"\n[Supervisor] Supervised command completed with exit code: {ret}")

                    # Fetch instance details for termination and sync
                    instances = get_instances(api_key)
                    matching_inst = None
                    if target_instance_id is not None:
                        for inst in instances:
                            if inst.get("id") == target_instance_id:
                                matching_inst = inst
                                break
                    if matching_inst is None:
                        if instances and len(instances) == 1:
                            matching_inst = instances[0]
                        else:
                            matching_inst = {"id": target_instance_id or 0}

                    if destroy_on_cmd_finish:
                        log(f"[Supervisor] --destroy-on-cmd-finish is active. Initiating pre-deletion upload and {action.upper()}...")
                        terminate_instance(
                            instance=matching_inst,
                            api_key=api_key,
                            action=action,
                            rclone_dest=rclone_dest,
                            log_path=log_path,
                            results_base=results_base,
                            sync_local_dir=sync_local_dir,
                            remote_path=remote_path,
                            ssh_key_path=ssh_key_path,
                            skip_upload=skip_upload,
                            dry_run=dry_run,
                        )
                    else:
                        # Upload without destroying
                        log("\n[Supervisor] Uploading logs and results to Google Drive upon task completion...")
                        upload_logs_and_results_to_gdrive(
                            rclone_dest=rclone_dest,
                            log_path=log_path,
                            results_base=results_base,
                            sync_local_dir=sync_local_dir,
                            remote_path=remote_path,
                            ssh_key_path=ssh_key_path,
                            skip_upload=skip_upload,
                        )
                        if sync_local_dir:
                            sync_to_local_machine(
                                instance=matching_inst,
                                local_dir=sync_local_dir,
                                remote_path=remote_path,
                                ssh_key_path=ssh_key_path,
                            )
                    break

            # Fetch active instances
            try:
                instances = get_instances(api_key)
            except Exception as e:
                log(f"[Warning] Failed to query Vast API ({e}). Retrying in {check_interval}s...")
                time.sleep(check_interval)
                continue

            if not instances:
                log(f"[VastMonitor] No active instances found. Checking again in {check_interval}s...")
                time.sleep(check_interval)
                continue

            # Identify target instance(s)
            if target_instance_id is not None:
                active_targets = [i for i in instances if i.get("id") == target_instance_id]
                if not active_targets:
                    log(f"[VastMonitor] Target instance ID {target_instance_id} is no longer running or not found.")
                    break
            else:
                if len(instances) == 1:
                    target_instance_id = instances[0].get("id")
                    active_targets = instances
                    log(f"[VastMonitor] Auto-locked onto sole active instance ID: {target_instance_id}")
                else:
                    log(f"[VastMonitor] NOTICE: Organization account has {len(instances)} active instances.")
                    active_targets = [i for i in instances if i.get("actual_status") in ("running", "loading", None)]
                    if not active_targets:
                        active_targets = instances
                    active_targets = instances

            # Check spend on each target instance
            for inst in active_targets:
                inst_id = inst.get("id")
                gpu_name = inst.get("gpu_name", "GPU")

                spend, el_hours, dph, _ = compute_instance_spend(inst, monitor_start_time, budget_type=budget_type)
                pct = min(100.0, (spend / max_spend) * 100.0) if max_spend > 0 else 100.0
                rem_budget = max(0.0, max_spend - spend)
                rem_hours = (rem_budget / dph) if dph > 0 else 0.0

                hrs = int(el_hours)
                mins = int((el_hours - hrs) * 60)

                timestamp_str = time.strftime("%H:%M:%S")
                log(
                    f"[{timestamp_str}] [ID: {inst_id}] {gpu_name} (${dph:.2f}/hr) | "
                    f"Elapsed: {hrs}h {mins:02d}m | "
                    f"Spend: ${spend:6.2f} / ${max_spend:.2f} ({pct:5.1f}%) | "
                    f"Remaining: ~${rem_budget:.2f} (~{rem_hours:.1f}h)"
                )

                # TRIGGER: BUDGET LIMIT EXCEEDED
                if spend >= max_spend:
                    alert_msg = (
                        f"🚨 [Vast.ai Budget Alert] Instance {inst_id} ({gpu_name}) has spent "
                        f"${spend:.2f}, reaching the ${max_spend:.2f} budget limit! "
                        f"Executing pre-deletion data backup and {action.upper()} action..."
                    )
                    log("\n" + "=" * 76)
                    log(alert_msg)
                    log("=" * 76)
                    send_webhook(webhook_url, alert_msg)

                    # Terminate supervised child process first
                    if proc is not None and proc.poll() is None:
                        log(f"[Supervisor] Stopping child process PID {proc.pid}...")
                        try:
                            proc.terminate()
                            proc.wait(timeout=5)
                        except Exception:
                            proc.kill()

                    # Execute mandatory pre-deletion upload, then destroy
                    terminate_instance(
                        instance=inst,
                        api_key=api_key,
                        action=action,
                        rclone_dest=rclone_dest,
                        log_path=log_path,
                        results_base=results_base,
                        sync_local_dir=sync_local_dir,
                        remote_path=remote_path,
                        ssh_key_path=ssh_key_path,
                        skip_upload=skip_upload,
                        dry_run=dry_run,
                    )
                    log("[VastMonitor] Budget safety protocol complete. Exiting.")
                    return

            time.sleep(check_interval)

    except KeyboardInterrupt:
        log("\n[VastMonitor] Interrupted by user (Ctrl+C).")
        if proc is not None and proc.poll() is None:
            log("[Supervisor] Terminating child process...")
            proc.terminate()
        sys.exit(0)


# ==============================================================================
# 7. CLI Entrypoint
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Vast.ai Budget Monitor & Auto-Termination Guard with Pre-Deletion GDrive Upload")
    parser.add_argument("--api-key", type=str, default=None, help="Vast.ai API key (or set VAST_API_KEY env)")
    parser.add_argument("--instance-id", type=int, default=None, help="Specific instance ID to monitor (default: auto-detect)")
    parser.add_argument("--max-spend", type=float, default=5.0, help="Maximum spend threshold in USD (default: $5.00)")
    parser.add_argument("--action", type=str, choices=["destroy", "stop"], default="destroy", help="Action when budget is reached (default: destroy)")
    parser.add_argument("--budget-type", type=str, choices=["lifetime", "session"], default="lifetime", help="'lifetime' (from launch) or 'session' (from monitor start)")
    parser.add_argument("--check-interval", type=int, default=30, help="Seconds between API polling checks (default: 30s)")
    parser.add_argument("--cmd", type=str, default=None, help="Optional training shell command to supervise")
    parser.add_argument("--destroy-on-cmd-finish", action="store_true", help="Automatically destroy instance when --cmd finishes")
    parser.add_argument("--rclone-dest", type=str, default=os.environ.get("VAST_GDRIVE_DEST", "gdrive:belief_results"), help="Google Drive rclone destination (default: 'gdrive:belief_results')")
    parser.add_argument("--log-file", type=str, default="vast_monitor.log", help="Path to tee service output logs (default: 'vast_monitor.log')")
    parser.add_argument("--results-path", type=str, default="./experiments", help="Path to results directory or zip (default: './experiments')")
    parser.add_argument("--sync-local-dir", type=str, default=None, help="Local directory to pull remote results into via rsync/scp")
    parser.add_argument("--remote-path", type=str, default="./experiments/double_pendulum_results.zip", help="Path on remote instance to sync (default: ./experiments/double_pendulum_results.zip)")
    parser.add_argument("--ssh-key-path", type=str, default=None, help="Path to SSH private key for rsync/scp")
    parser.add_argument("--skip-upload", action="store_true", help="Bypass mandatory pre-deletion upload (testing only)")
    parser.add_argument("--webhook-url", type=str, default=None, help="Optional Discord/Slack webhook URL for alerts")
    parser.add_argument("--dry-run", action="store_true", help="Log spend metrics without actually stopping or destroying instances")

    args = parser.parse_args()

    # Initialize Global Tee Logger
    global GLOBAL_LOGGER
    log_file_path = Path(args.log_file).resolve()
    GLOBAL_LOGGER = TeeLogger(log_file_path)

    api_key = resolve_api_key(args.api_key)

    try:
        run_monitor(
            api_key=api_key,
            target_instance_id=args.instance_id,
            max_spend=args.max_spend,
            action=args.action,
            budget_type=args.budget_type,
            check_interval=args.check_interval,
            cmd=args.cmd,
            destroy_on_cmd_finish=args.destroy_on_cmd_finish,
            rclone_dest=args.rclone_dest,
            log_path=log_file_path,
            results_base=Path(args.results_path).resolve(),
            sync_local_dir=args.sync_local_dir,
            remote_path=args.remote_path,
            ssh_key_path=args.ssh_key_path,
            skip_upload=args.skip_upload,
            webhook_url=args.webhook_url,
            dry_run=args.dry_run,
        )
    finally:
        if GLOBAL_LOGGER:
            GLOBAL_LOGGER.close()


if __name__ == "__main__":
    main()
