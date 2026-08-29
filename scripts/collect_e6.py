"""E6: assemble the kick-magnitude sweep from the Kaggle kernel log.

The sweep runs one training and one analysis per magnitude inside a single
kernel session, and only JSON comes back. An untagged output filename in the
first version of `10_myopic.py` meant the four runs overwrote each other, so the
per-magnitude numbers are recovered from the kernel's stdout, which recorded all
of them. The bug is fixed; this exists so the completed 6-GPU-hour sweep does not
have to be repeated to get its own numbers back.

Run:  uv run python scripts/collect_e6.py <kernel.log> [more.log ...]
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get("OUTPUT_DIR", ROOT / "experiments/outputs-03"))

PATTERNS = {
    "scale": re.compile(r"^myopic kick_scale=([\d.]+)"),
    "loss": re.compile(r"^(\w+): eval loss=([\d.]+)"),
    "e1a": re.compile(r"E1a\s+belief from p: linear=([-\d.]+) log-linear=([-\d.]+)"),
    "stream": re.compile(r"E1a\s+stream @(\w+): full=([-\d.]+)\s+top-\d+-PCs=([-\d.]+)\s+random-\d+-proj=([-\d.]+)"),
    "e1b": re.compile(r"E1b\s+([\d,]+) pairs.*paired R\^2=([-\d.]+) \(shuffled ([-+\d.]+)\).*mean \|db\|=([\d.]+)"),
    "e1c": re.compile(r"E1c\s+R\^2\(belief \| p\^\(1\.\.k\)\):\s+(.*)$"),
}


def _text(path: Path) -> str:
    raw = path.read_text(encoding="utf-8", errors="replace")
    try:  # Kaggle serves the log as a JSON array of events
        return "\n".join(str(e.get("data", "")) for e in json.loads(raw))
    except Exception:
        return raw


def parse(paths) -> list[dict]:
    rows, current, pending_loss = [], None, None
    for path in paths:
        for line in _text(Path(path)).splitlines():
            line = line.strip()
            if m := PATTERNS["loss"].match(line):
                pending_loss = float(m.group(2))
            if m := PATTERNS["scale"].match(line):
                current = {"kick_scale": float(m.group(1)), "eval_loss": pending_loss}
                rows.append(current)
                continue
            if current is None:
                continue
            if m := PATTERNS["e1a"].search(line):
                current["belief_from_p_linear"] = float(m.group(1))
                current["belief_from_log_p_linear"] = float(m.group(2))
            elif m := PATTERNS["stream"].search(line):
                current["depth"] = m.group(1)
                current["belief_from_stream"] = float(m.group(2))
                current["belief_from_top_pcs"] = float(m.group(3))
                current["belief_from_random_proj"] = float(m.group(4))
            elif m := PATTERNS["e1b"].search(line):
                current["n_pairs"] = int(m.group(1).replace(",", ""))
                current["paired_r2"] = float(m.group(2))
                current["paired_r2_shuffled"] = float(m.group(3))
                current["mean_belief_gap"] = float(m.group(4))
            elif m := PATTERNS["e1c"].search(line):
                current["horizon"] = [
                    float(v) for v in re.findall(r"k\d+:([-\d.]+)", m.group(1))
                ]
    return rows


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    rows = parse(sys.argv[1:])

    # The baseline magnitude was run locally, so splice it in from that JSON.
    local = OUT / "phase5_03_myopic.json"
    if local.exists():
        d = json.loads(local.read_text()).get("pendulum")
        if d:
            rows.append(
                {
                    "kick_scale": 1.0,
                    "eval_loss": None,
                    "belief_from_p_linear": d["e1a"]["belief_from_p_linear"],
                    "belief_from_log_p_linear": d["e1a"]["belief_from_log_p_linear"],
                    "depth": d["e1a"]["dimension_controls"]["depth"],
                    "belief_from_stream": d["e1a"]["dimension_controls"]["belief_from_full_stream"],
                    "belief_from_top_pcs": d["e1a"]["dimension_controls"]["belief_from_top_pcs_of_stream"],
                    "belief_from_random_proj": d["e1a"]["dimension_controls"]["belief_from_random_projection_of_stream"],
                    "n_pairs": d["e1b"].get("n_pairs_used"),
                    "paired_r2": d["e1b"].get("paired_r2"),
                    "paired_r2_shuffled": d["e1b"].get("paired_r2_shuffled_control"),
                    "mean_belief_gap": d["e1b"].get("mean_belief_gap_of_used_pairs"),
                    "horizon": [c["belief_from_p1_to_pk"] for c in d["e1c_horizon"]],
                }
            )
    rows.sort(key=lambda r: r["kick_scale"])
    base_kick = 1.5  # pendulum's phase-1 magnitude
    for r in rows:
        r["kick"] = base_kick * r["kick_scale"]

    path = OUT / "phase5_06_kick_sweep.json"
    path.write_text(json.dumps({"pendulum": rows}, indent=2))
    print(f"{'scale':>6}{'kick':>7}{'loss':>9}{'coupling':>10}{'stream':>9}"
          f"{'paired':>9}{'shuffled':>10}{'|db|':>8}")
    for r in rows:
        loss = "-" if r["eval_loss"] is None else f"{r['eval_loss']:.4f}"
        print(f"{r['kick_scale']:>6.2f}{r['kick']:>7.2f}{loss:>9}"
              f"{r['belief_from_p_linear']:>10.3f}{r['belief_from_stream']:>9.3f}"
              f"{r['paired_r2']:>9.3f}{r['paired_r2_shuffled']:>10.3f}{r['mean_belief_gap']:>8.3f}")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
