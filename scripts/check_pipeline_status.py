#!/usr/bin/env python3
"""Pipeline Status Checker for Double Pendulum Production Runs.

Inspects pipeline_state.json and filesystem checkpoints to report which
models are TRAINING (GPU), PROBING (CPU), QUEUED, and COMPLETED.

Usage:
    python scripts/check_pipeline_status.py
    python scripts/check_pipeline_status.py --dir /workspace/Belief-in-physics/experiments
    python scripts/check_pipeline_status.py --watch 5
    python scripts/check_pipeline_status.py --all
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional


def find_default_output_dir(cli_dir: Optional[str] = None) -> Path:
    """Finds the most likely experiments output directory."""
    if cli_dir:
        return Path(cli_dir).expanduser().resolve()

    candidates = [
        Path("./experiments"),
        Path("/workspace/Belief-in-physics/experiments"),
        Path("/kaggle/working"),
    ]
    for c in candidates:
        if (c / "pipeline_state.json").exists() or (c / "results").exists():
            return c.resolve()

    return Path("./experiments").resolve()


def scan_pipeline(output_dir: Path) -> Dict[str, Any]:
    """Reads pipeline_state.json or falls back to filesystem inspection."""
    state_file = output_dir / "pipeline_state.json"
    results_dir = output_dir / "results" if (output_dir / "results").exists() else output_dir

    data: Dict[str, Any] = {
        "source": "none",
        "output_dir": str(output_dir),
        "total": 0,
        "completed": [],
        "probing": [],
        "training": [],
        "queued": [],
        "counts": Counter(),
        "fs_checkpoints": 0,
        "fs_probes": 0,
        "fs_zips": 0,
        "last_updated": None,
    }

    # 1. Inspect Filesystem directly
    if results_dir.exists():
        data["fs_checkpoints"] = len(list(results_dir.glob("*/*_final.pt")))
        data["fs_probes"] = len(list(results_dir.glob("*/*_probes.csv")))
        data["fs_zips"] = len(list(results_dir.glob("*.zip")))

    # 2. Inspect pipeline_state.json if available
    if state_file.exists():
        try:
            raw = json.loads(state_file.read_text())
            data["source"] = "pipeline_state.json"
            data["total"] = raw.get("total_models", 0)
            data["last_updated"] = raw.get("timestamp")

            job_statuses: Dict[str, str] = raw.get("job_statuses", {})
            for jid, status in job_statuses.items():
                st_upper = status.upper()
                if st_upper == "TRAINING":
                    data["training"].append(jid)
                elif st_upper == "PROBING":
                    data["probing"].append(jid)
                elif st_upper == "COMPLETED":
                    data["completed"].append(jid)
                elif st_upper == "QUEUED":
                    data["queued"].append(jid)

            if not data["completed"]:
                data["completed"] = raw.get("completed_models", [])

            # Check filesystem to reconcile actual disk ground truth
            if results_dir.exists():
                probed_files = list(results_dir.glob("*/*_probes.csv"))
                ckpt_files = list(results_dir.glob("*/*_final.pt"))

                probed_ids = {p.parent.name for p in probed_files}
                ckpt_ids = {p.parent.name for p in ckpt_files}

                # 1. Models with probe CSV are COMPLETED
                for pid in probed_ids:
                    if pid not in data["completed"]:
                        data["completed"].append(pid)
                    if pid in data["probing"]:
                        data["probing"].remove(pid)
                    if pid in data["training"]:
                        data["training"].remove(pid)

                # 2. Models with checkpoint .pt but no probe CSV are in the PROBING pipeline
                for cid in ckpt_ids:
                    if cid in data["completed"]:
                        continue
                    if cid not in data["probing"]:
                        data["probing"].append(cid)
                    if cid in data["training"]:
                        data["training"].remove(cid)

            if not data["total"]:
                data["total"] = len(job_statuses) or (len(data["completed"]) + len(data["training"]) + len(data["probing"]) + len(data["queued"]))

            data["counts"]["COMPLETED"] = len(data["completed"])
            data["counts"]["PROBING"] = len(data["probing"])
            data["counts"]["TRAINING"] = len(data["training"])
            data["counts"]["QUEUED"] = len(data["queued"])
            return data
        except Exception as e:
            data["source"] = f"error reading json ({e})"

    # 3. Fallback: Reconstruct state from filesystem if pipeline_state.json is missing
    if results_dir.exists():
        data["source"] = "filesystem_scan"
        probed_files = list(results_dir.glob("*/*_probes.csv"))
        ckpt_files = list(results_dir.glob("*/*_final.pt"))

        probed_ids = {p.parent.name for p in probed_files}
        ckpt_ids = {p.parent.name for p in ckpt_files}

        data["completed"] = sorted(probed_ids)
        # Checkpoints that haven't been probed yet are likely currently probing or awaiting probe
        data["probing"] = sorted(ckpt_ids - probed_ids)
        data["counts"]["COMPLETED"] = len(data["completed"])
        data["counts"]["PROBING"] = len(data["probing"])
        data["total"] = max(len(data["completed"]) + len(data["probing"]), data["fs_zips"])

    return data


def format_status_report(data: Dict[str, Any], show_all: bool = False) -> str:
    """Formats the scanned status data into a readable terminal dashboard."""
    lines: List[str] = []
    total = data["total"]
    num_completed = len(data["completed"])
    pct = (num_completed / max(1, total)) * 100.0 if total > 0 else 0.0

    lines.append("=" * 72)
    lines.append("  DOUBLE PENDULUM PRODUCTION PIPELINE MONITOR")
    lines.append("=" * 72)
    lines.append(f"Output Directory: {data['output_dir']}")
    lines.append(f"Data Source:      {data['source']}")
    if data["last_updated"]:
        updated_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(data["last_updated"]))
        elapsed_sec = int(time.time() - data["last_updated"])
        lines.append(f"State Last Saved: {updated_str} ({elapsed_sec}s ago)")

    lines.append("-" * 72)
    lines.append(f"OVERALL PROGRESS: {num_completed}/{total} Models Completed ({pct:5.1f}%)")
    lines.append("-" * 72)

    # Category counts
    cnt = data["counts"]
    num_train = len(data["training"])
    num_probe = len(data["probing"])
    
    # RTX 5090 typically runs 12 train workers and 2 probe workers
    active_train_est = min(12, num_train)
    queued_train_est = max(0, num_train - active_train_est)
    active_probe_est = min(2, num_probe)
    queued_probe_est = max(0, num_probe - active_probe_est)

    lines.append(f"  ✓ COMPLETED:          {len(data['completed']):4d}")
    lines.append(f"  ⚙ PROBING PIPELINE:   {num_probe:4d}  ({active_probe_est} active on CPU, {queued_probe_est} waiting in queue)")
    lines.append(f"  ⚡ TRAINING PIPELINE:  {num_train:4d}  ({active_train_est} active on GPU, {queued_train_est} waiting in queue)")
    if data["queued"]:
        lines.append(f"  ⏳ UNTOUCHED QUEUED:   {len(data['queued']):4d}")
    if data["fs_checkpoints"] or data["fs_probes"] or data["fs_zips"]:
        lines.append(f"  📁 DISK RECONCILIATION: {data['fs_checkpoints']} trained (.pt) | {data['fs_probes']} probed (.csv) | {data['fs_zips']} packaged (.zip)")

    # Active Training (GPU)
    lines.append("-" * 72)
    lines.append(f"ACTIVE GPU TRAINING WORKERS (Showing up to 12):")
    if data["training"]:
        for jid in sorted(data["training"])[:12]:
            lines.append(f"  [TRAIN-GPU] {jid}")
        if len(data["training"]) > 12:
            lines.append(f"  ... and {len(data['training']) - 12} more jobs queued in worker pool")
    else:
        lines.append("  (No active GPU training workers reported)")

    # Active Probing (CPU)
    lines.append("-" * 72)
    lines.append(f"PROBING QUEUE ({num_probe} models finished training):")
    if data["probing"]:
        for i, jid in enumerate(sorted(data["probing"])):
            tag = "[PROBE-ACTIVE]" if i < 2 else "[PROBE-QUEUED]"
            lines.append(f"  {tag} {jid}")
    else:
        lines.append("  (No models currently in probing pipeline)")

    # Completed Models
    lines.append("-" * 72)
    if data["completed"]:
        if show_all:
            lines.append(f"ALL COMPLETED MODELS ({len(data['completed'])}):")
            for jid in sorted(data["completed"]):
                lines.append(f"  [DONE]  {jid}")
        else:
            recent_count = min(8, len(data["completed"]))
            lines.append(f"RECENTLY COMPLETED MODELS (showing {recent_count} of {len(data['completed'])}, use --all to view all):")
            for jid in data["completed"][-recent_count:]:
                lines.append(f"  [DONE]  {jid}")
    else:
        lines.append("COMPLETED MODELS: None yet")

    lines.append("=" * 72)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Find which models are training, probing, and completed.")
    parser.add_argument("--dir", "-d", type=str, default=None, help="Root experiments directory (default: auto-detect)")
    parser.add_argument("--all", "-a", action="store_true", help="Display full list of all completed models")
    parser.add_argument("--watch", "-w", type=int, default=None, metavar="SECONDS", help="Live watch mode: refresh every N seconds")
    parser.add_argument("--json", "-j", action="store_true", help="Output raw JSON instead of formatted text")
    args = parser.parse_args()

    output_dir = find_default_output_dir(args.dir)

    if args.watch:
        try:
            while True:
                os.system("clear" if os.name == "posix" else "cls")
                data = scan_pipeline(output_dir)
                if args.json:
                    print(json.dumps(data, indent=2, default=str))
                else:
                    print(format_status_report(data, show_all=args.all))
                    print(f"\n[Watch mode: refreshing every {args.watch}s | Press Ctrl+C to exit]")
                time.sleep(args.watch)
        except KeyboardInterrupt:
            print("\nExiting monitor.")
    else:
        data = scan_pipeline(output_dir)
        if args.json:
            print(json.dumps(data, indent=2, default=str))
        else:
            print(format_status_report(data, show_all=args.all))


if __name__ == "__main__":
    main()
