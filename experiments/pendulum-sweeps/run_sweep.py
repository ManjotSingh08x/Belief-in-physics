import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")

def run_notebook(nb_path: Path):
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    ctx = {
        "__file__": str(nb_path),
        "display": lambda *args, **kwargs: None,
    }

    code_cells = [c for c in nb["cells"] if c.get("cell_type") == "code"]
    print(f"Loaded {len(code_cells)} code cells from {nb_path.name}")

    for idx, cell in enumerate(code_cells, start=1):
        source = "".join(cell.get("source", []))
        clean_lines = [
            line for line in source.splitlines()
            if not line.strip().startswith("%") and not line.strip().startswith("!")
        ]
        code = "\n".join(clean_lines).strip()
        if not code:
            continue

        first_line = clean_lines[0].strip() if clean_lines else ""
        print(f"\n[Cell {idx}/{len(code_cells)}] {first_line[:60]}...")
        exec(code, ctx)

    print("\nExecution complete.")

if __name__ == "__main__":
    notebook_file = Path(__file__).resolve().parent / "three_phase_pipeline.ipynb"
    run_notebook(notebook_file)
