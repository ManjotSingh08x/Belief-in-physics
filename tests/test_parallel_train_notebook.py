import json
from pathlib import Path
import subprocess
import sys
import tempfile

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
SWEEPS_DIR = ROOT / "experiments" / "pendulum-sweeps"
if str(SWEEPS_DIR) not in sys.path:
    sys.path.insert(0, str(SWEEPS_DIR))

NOTEBOOK_PATH = SWEEPS_DIR / "parallel_train.ipynb"
RANGE_CSV = SWEEPS_DIR / "pendulum_range.csv"


def test_notebook_json_validity():
    assert NOTEBOOK_PATH.exists()
    content = json.loads(NOTEBOOK_PATH.read_text())
    assert content["nbformat"] == 4
    cells = content["cells"]
    code_cells = [c for c in cells if c["cell_type"] == "code"]
    assert len(code_cells) == 7


def test_load_configs_and_m_computation():
    assert RANGE_CSV.exists()
    df = pd.read_csv(RANGE_CSV)
    assert len(df) == 25

    def compute_m(n_steps):
        return 500 // n_steps

    configs = []
    for _, row in df.iterrows():
        n = int(row["n_steps"])
        m = compute_m(n)
        configs.append({
            "delta_v": float(row["delta_v"]),
            "gamma": float(row["damping_value"]),
            "dt": float(row["dt"]),
            "n_steps": n,
            "m": m,
            "seq_len": m * n,
        })

    assert len(configs) == 25
    # Verify n_steps=15 special handling (m=33, seq_len=495)
    n15_cfgs = [c for c in configs if c["n_steps"] == 15]
    assert len(n15_cfgs) > 0
    for c in n15_cfgs:
        assert c["m"] == 33
        assert c["seq_len"] == 495


def test_job_queue_building_and_filtering():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        df = pd.read_csv(RANGE_CSV)

        def config_name(row):
            return f"dt{row['dt']:g}_gamma{row['damping_value']:g}_dv{row['delta_v']:g}_n{int(row['n_steps'])}"

        # SMOKE mode: 1 width (32), 1 seed (0) -> 25 jobs
        jobs_smoke = []
        for _, row in df.iterrows():
            cfg_name = config_name(row)
            jobs_smoke.append({
                "job_id": f"{cfg_name}_d32_seed0",
                "config_name": cfg_name,
            })
        assert len(jobs_smoke) == 25

        # Production mode: 3 widths x 3 seeds -> 225 jobs
        jobs_prod = []
        for _, row in df.iterrows():
            cfg_name = config_name(row)
            for w in [32, 64, 128]:
                for s in [0, 1, 2]:
                    jobs_prod.append({
                        "job_id": f"{cfg_name}_d{w}_seed{s}",
                    })
        assert len(jobs_prod) == 225

        # Test filtering
        state_file = tmp_path / "pipeline_state.json"
        state_file.write_text(json.dumps({"completed": [jobs_smoke[0]["job_id"], jobs_smoke[1]["job_id"]]}))
        completed = set(json.loads(state_file.read_text()).get("completed", []))
        pending = [j for j in jobs_smoke if j["job_id"] not in completed]
        assert len(pending) == 23


def test_probe_aggregation():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Create two fake job folders with probes.csv
        c1_dir = tmp_path / "cfg1"
        c1_dir.mkdir(parents=True)
        df1 = pd.DataFrame([{"job": 1, "test_r2": 0.8}, {"job": 1, "test_r2": 0.7}])
        df1.to_csv(c1_dir / "d32_seed0_probes.csv", index=False)

        c2_dir = tmp_path / "cfg2"
        c2_dir.mkdir(parents=True)
        df2 = pd.DataFrame([{"job": 2, "test_r2": 0.9}])
        df2.to_csv(c2_dir / "d32_seed0_probes.csv", index=False)

        # Aggregate
        all_csvs = sorted(tmp_path.rglob("*_probes.csv"))
        master_df = pd.concat([pd.read_csv(f) for f in all_csvs], ignore_index=True)
        master_csv = tmp_path / "probe_results_all.csv"
        master_df.to_csv(master_csv, index=False)

        assert master_csv.exists()
        loaded = pd.read_csv(master_csv)
        assert len(loaded) == 3
        assert list(loaded["test_r2"]) == [0.8, 0.7, 0.9]


def test_multi_worker_concurrency():
    """Verify that multiple concurrent worker subprocesses do not collide."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        queue_path = tmp_path / "_job_queue.json"
        worker_script = SWEEPS_DIR / "_worker.py"

        # Create 2 tiny jobs
        jobs = []
        for i in range(2):
            cfg_name = f"test_concurrency_n10_cfg{i}"
            jobs.append({
                "job_id": f"{cfg_name}_d32_seed0",
                "config_name": cfg_name,
                "delta_v": 0.7,
                "gamma": 0.3,
                "dt": 0.2,
                "n_steps": 10,
                "m": 4,
                "seq_len": 40,
                "width": 32,
                "seed": 0,
                "horizons": [1, 5, 10],
            })
        queue_path.write_text(json.dumps(jobs))

        # Launch 2 worker subprocesses concurrently
        procs = []
        for w_id in range(2):
            cmd = [
                sys.executable,
                str(worker_script),
                "--queue-file", str(queue_path),
                "--output-dir", str(tmp_path),
                "--worker-id", str(w_id),
                "--device", "cpu",
                "--total-tokens", "500",
                "--batch-size", "4",
                "--n-probe-traj", "8",
                "--cv-folds", "2",
                "--ridge-alphas", "1.0,10.0",
                "--n-layers", "2",
                "--log-every", "10",
            ]
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            procs.append(p)

        for p in procs:
            stdout, stderr = p.communicate()
            assert p.returncode == 0, f"Worker failed:\nSTDOUT:{stdout}\nSTDERR:{stderr}"

        # Verify state file has both completed jobs without duplicates
        state_file = tmp_path / "pipeline_state.json"
        assert state_file.exists()
        state = json.loads(state_file.read_text())
        assert len(state["completed"]) == 2
        assert len(set(state["completed"])) == 2


def test_notebook_cell_execution():
    """Execute cells 1-7 from parallel_train.ipynb in a temporary directory."""
    nb = json.loads(NOTEBOOK_PATH.read_text())
    code_cells = [c["source"] for c in nb["cells"] if c["cell_type"] == "code"]
    assert len(code_cells) == 7

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Environment namespace
        ns = {"display": lambda x: None}

        # Cell 1: Constants & setup
        c1 = "".join(code_cells[0])
        exec(c1, ns)
        # Override output dir and hyperparameters for ultra-fast unit test execution
        ns["OUTPUT_DIR"] = tmp_path
        ns["TOTAL_TOKENS"] = 500
        ns["BATCH_SIZE"] = 4
        ns["N_PROBE_TRAJECTORIES"] = 8
        ns["CV_FOLDS"] = 2
        ns["RIDGE_ALPHAS"] = (1.0, 10.0)
        ns["N_LAYERS"] = 2
        ns["LOG_EVERY"] = 10
        ns["DEVICE"] = "cpu"
        ns["NUM_WORKERS"] = 1

        # Cell 2: Load configs
        c2 = "".join(code_cells[1])
        exec(c2, ns)
        assert len(ns["configs"]) == 25

        # Cell 3: Build jobs
        c3 = "".join(code_cells[2])
        exec(c3, ns)
        assert len(ns["jobs"]) == 25

        # Limit to 1 job for fast test execution
        ns["jobs"] = ns["jobs"][:1]

        # Cell 4: Filter completed
        c4 = "".join(code_cells[3])
        exec(c4, ns)
        assert len(ns["pending"]) == 1

        # Cell 5: Spawn workers
        c5 = "".join(code_cells[4])
        exec(c5, ns)

        # Cell 6: Monitor
        c6 = "".join(code_cells[5])
        exec(c6, ns)

        # Cell 7: Aggregate
        c7 = "".join(code_cells[6])
        exec(c7, ns)

        # Verify final outputs
        master_csv = tmp_path / "probe_results_all.csv"
        assert master_csv.exists()
        df = pd.read_csv(master_csv)
        assert len(df) == 72
        assert "test_r2" in df.columns
        assert "cv_r2" in df.columns
        assert "target" in df.columns
        assert "mode" in df.columns

        # Verify pipeline state
        state_file = tmp_path / "pipeline_state.json"
        assert state_file.exists()
        state = json.loads(state_file.read_text())
        assert len(state["completed"]) == 1

