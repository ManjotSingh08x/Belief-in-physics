"""Package one non-overlapping Sphere sweep slice as a Kaggle notebook."""

import argparse
import base64
import io
import json
import subprocess
import zipfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--owner", required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--stop", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0 <= args.start < args.stop <= 60:
        parser.error("the Sphere seed-0 alpha sweep has exactly 60 runs")

    root = Path(__file__).resolve().parents[1]
    source = root / "notebooks/three_phase_pipeline.ipynb"
    notebook = json.loads(source.read_text())
    archive_bytes = io.BytesIO()
    with zipfile.ZipFile(archive_bytes, "w", zipfile.ZIP_DEFLATED) as archive:
        for package in ("models", "physics"):
            for path in sorted((root / package).rglob("*.py")):
                archive.write(path, path.relative_to(root))
    payload = base64.b64encode(archive_bytes.getvalue()).decode("ascii")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    settings = {
        "BELIEF_REPO_ROOT": "/kaggle/working/Belief-in-physics",
        "BELIEF_SOURCE_REVISION": revision,
        "BELIEF_RUN_FULL_SWEEP": "1",
        "BELIEF_SYSTEM_FILTER": "sphere",
        "BELIEF_SEED_FILTER": "0",
        "BELIEF_RUN_START": str(args.start),
        "BELIEF_RUN_STOP": str(args.stop),
        "BELIEF_NOTEBOOK_DEVICE": "cuda",
    }
    setup = "import base64, io, os, zipfile\nfrom pathlib import Path\n"
    setup += "root = Path('/kaggle/working/Belief-in-physics')\nroot.mkdir(parents=True, exist_ok=True)\n"
    setup += f"with zipfile.ZipFile(io.BytesIO(base64.b64decode({payload!r}))) as archive:\n    archive.extractall(root)\n"
    setup += "\n".join(
        f"os.environ[{key!r}] = {value!r}" for key, value in settings.items()
    ) + "\n"
    notebook["cells"].insert(1, {
        "cell_type": "code", "execution_count": None, "metadata": {},
        "outputs": [], "source": setup.splitlines(keepends=True),
    })
    slug = f"sphere-alpha-sweep-{args.start:02d}-{args.stop:02d}"
    metadata = {
        "id": f"{args.owner}/{slug}", "title": slug.replace("-", " "),
        "code_file": "sphere_alpha_sweep.ipynb", "language": "python",
        "kernel_type": "notebook", "is_private": "true",
        "enable_gpu": "true", "enable_tpu": "false", "enable_internet": "true",
        "dataset_sources": [], "competition_sources": [],
        "kernel_sources": [], "model_sources": [],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "sphere_alpha_sweep.ipynb").write_text(json.dumps(notebook, indent=1))
    (args.output / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2))
    print(metadata["id"])


if __name__ == "__main__":
    main()
