"""Merge phase-5 JSONs that were produced by separate partial runs.

Two situations produce them. A long run can be interrupted and restarted for the
systems it did not reach, and a follow-up can deliberately rerun one depth with
an extra measurement added. Both write the same filename, so the pieces are
merged here rather than by hand.

Merging is per system and, where both sides carry `by_depth`, per depth: a later
file wins on any key it defines, and depths only the earlier file has are kept.

Run:  uv run python scripts/merge_results.py <out.json> <older.json> <newer.json> [...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _merge_depths(old: list, new: list) -> list:
    by_name = {d.get("name", d.get("depth")): dict(d) for d in old}
    for d in new:
        key = d.get("name", d.get("depth"))
        by_name[key] = {**by_name.get(key, {}), **d}
    return sorted(by_name.values(), key=lambda d: d.get("depth", 0))


def merge(parts: list[dict]) -> dict:
    out: dict = {}
    for part in parts:
        for system, value in part.items():
            if system not in out:
                out[system] = value
                continue
            merged = {**out[system], **value}
            if "by_depth" in out[system] and "by_depth" in value:
                merged["by_depth"] = _merge_depths(out[system]["by_depth"], value["by_depth"])
            out[system] = merged
    return out


def main() -> None:
    if len(sys.argv) < 4:
        raise SystemExit(__doc__)
    target, sources = Path(sys.argv[1]), [Path(p) for p in sys.argv[2:]]
    parts = [json.loads(p.read_text()) for p in sources]
    result = merge(parts)
    target.write_text(json.dumps(result, indent=2, default=float))
    for p, part in zip(sources, parts):
        print(f"  {p.name}: {sorted(part)}")
    print(f"-> {target}: {sorted(result)}")
    for system, value in result.items():
        if isinstance(value, dict) and "by_depth" in value:
            print(f"   {system}: depths {[d.get('name') for d in value['by_depth']]}")


def _demo() -> None:
    a = {"x": {"by_depth": [{"depth": 0, "name": "e", "p": 1}, {"depth": 1, "name": "L0", "p": 2}]}}
    b = {"x": {"by_depth": [{"depth": 1, "name": "L0", "q": 9}]}, "y": {"k": 1}}
    m = merge([a, b])
    assert sorted(m) == ["x", "y"]
    depths = {d["name"]: d for d in m["x"]["by_depth"]}
    assert depths["e"]["p"] == 1, "a depth only the older file has must survive"
    assert depths["L0"] == {"depth": 1, "name": "L0", "p": 2, "q": 9}, "keys must union, newer winning"
    print("merge ok")


if __name__ == "__main__":
    _demo() if "--self-check" in sys.argv else main()
