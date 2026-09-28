import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import r2_score
import torch

import sys

SWEEPS_DIR = Path(__file__).resolve().parent.parent / "experiments" / "pendulum-sweeps"
if str(SWEEPS_DIR) not in sys.path:
    sys.path.insert(0, str(SWEEPS_DIR))

from _worker import (
    LookaheadTransformer,
    SampledPhysicsSystem,
    build_process,
    claim_next_job,
    cross_validated_ridge,
    _ridge_r2_gpu,
    extract_probe_features_and_targets,
    extract_raw_targets,
    is_pid_alive,
    make_sampler,
    mark_completed,
    release_job,
    PROBE_MODES,
)
from models.transformer import ModelConfig
from physics.systems.pendulum import Pendulum


def test_sampled_physics_system():
    pendulum = Pendulum(gamma=0.3)
    sampled = SampledPhysicsSystem(base=pendulum, integration_dt=0.01)
    z0 = np.array([0.5, -0.2])
    z_sampled = sampled.flow(z0, 0.2)
    z_direct = pendulum.flow(z0, 0.2, substeps=20)
    assert np.allclose(z_sampled, z_direct)


def test_lookahead_transformer():
    cfg = ModelConfig(
        vocab_size=181,
        n_ctx=50,
        n_layers=2,
        n_heads=1,
        d_model=32,
        d_mlp=128,
        seed=0,
    )
    model = LookaheadTransformer(cfg, k=5)
    tokens = torch.randint(0, 181, (4, 50))
    loss = model.loss(tokens)
    assert loss.ndim == 0
    assert torch.isfinite(loss)

    # Check loss matches manual cross entropy
    logits = model(tokens)
    expected_loss = torch.nn.functional.cross_entropy(
        logits[:, :-5].reshape(-1, logits.shape[-1]),
        tokens[:, 5:].reshape(-1),
    )
    assert torch.allclose(loss, expected_loss)


def test_build_process_and_sampler():
    proc = build_process(delta_v=0.7, gamma=0.3, dt=0.2, n_steps=10, m=50)
    assert proc.seq_len == 500
    assert proc.n_obs == 181

    sampler = make_sampler(proc)
    rng = np.random.default_rng(42)
    batch = sampler(rng, 4)
    assert batch.shape == (4, 500)
    assert batch.min() >= 0
    assert batch.max() < 181


def test_job_claiming_and_completion():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        queue_path = tmp_path / "_job_queue.json"
        state_path = tmp_path / "pipeline_state.json"

        jobs = [
            {"job_id": "job_1", "config_name": "cfg1"},
            {"job_id": "job_2", "config_name": "cfg2"},
            {"job_id": "job_3", "config_name": "cfg3"},
        ]
        queue_path.write_text(json.dumps(jobs))

        j1 = claim_next_job(queue_path, state_path)
        assert j1["job_id"] == "job_1"

        mark_completed(state_path, "job_1")
        # Idempotent mark
        mark_completed(state_path, "job_1")

        j2 = claim_next_job(queue_path, state_path)
        assert j2["job_id"] == "job_2"

        mark_completed(state_path, "job_2")
        j3 = claim_next_job(queue_path, state_path)
        assert j3["job_id"] == "job_3"

        mark_completed(state_path, "job_3")
        j4 = claim_next_job(queue_path, state_path)
        assert j4 is None


def _claim_helper(args):
    q_path, s_path, w_id = args
    return claim_next_job(q_path, s_path, worker_id=w_id)


def test_concurrent_claims_no_race_condition():
    from concurrent.futures import ProcessPoolExecutor

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        queue_path = tmp_path / "_job_queue.json"
        state_path = tmp_path / "pipeline_state.json"

        jobs = [
            {"job_id": f"job_{i}", "config_name": f"cfg{i}"}
            for i in range(8)
        ]
        queue_path.write_text(json.dumps(jobs))

        worker_args = [(queue_path, state_path, i) for i in range(8)]
        with ProcessPoolExecutor(max_workers=8) as executor:
            claimed_jobs = list(executor.map(_claim_helper, worker_args))

        claimed_ids = [j["job_id"] for j in claimed_jobs if j is not None]
        assert len(claimed_ids) == 8
        assert len(set(claimed_ids)) == 8, "Multiple workers claimed the same job ID"

        # Verify state file recorded in_progress correctly
        state = json.loads(state_path.read_text())
        assert len(state["in_progress"]) == 8
        for i in range(8):
            assert f"job_{i}" in state["in_progress"]


def test_release_job_and_recovery():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        queue_path = tmp_path / "_job_queue.json"
        state_path = tmp_path / "pipeline_state.json"

        jobs = [{"job_id": "job_oom_test", "config_name": "cfg_oom"}]
        queue_path.write_text(json.dumps(jobs))

        j1 = claim_next_job(queue_path, state_path, worker_id="worker_1")
        assert j1["job_id"] == "job_oom_test"

        state = json.loads(state_path.read_text())
        assert "job_oom_test" in state["in_progress"]

        # Worker catches OOM and releases job
        release_job(state_path, "job_oom_test")
        state_after = json.loads(state_path.read_text())
        assert "job_oom_test" not in state_after["in_progress"]
        assert "job_oom_test" not in state_after["completed"]

        # Next worker claims the released job
        j2 = claim_next_job(queue_path, state_path, worker_id="worker_2")
        assert j2 is not None
        assert j2["job_id"] == "job_oom_test"


def test_dead_pid_reclaiming():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        queue_path = tmp_path / "_job_queue.json"
        state_path = tmp_path / "pipeline_state.json"

        jobs = [{"job_id": "job_dead_worker", "config_name": "cfg_dead"}]
        queue_path.write_text(json.dumps(jobs))

        # Simulate dead worker with a non-existent PID
        dead_pid = 999999999
        assert not is_pid_alive(dead_pid)
        state_path.write_text(json.dumps({
            "completed": [],
            "in_progress": {
                "job_dead_worker": {
                    "worker_id": "dead_worker",
                    "pid": dead_pid,
                    "claimed_at": 1000.0,
                }
            }
        }))

        # Live worker claims next job
        j = claim_next_job(queue_path, state_path, worker_id="alive_worker")
        assert j is not None
        assert j["job_id"] == "job_dead_worker"

        state = json.loads(state_path.read_text())
        assert state["in_progress"]["job_dead_worker"]["worker_id"] == "alive_worker"


def test_target_extraction():
    proc = build_process(delta_v=0.7, gamma=0.3, dt=0.2, n_steps=10, m=50)
    rng = np.random.default_rng(123)
    batch = proc.sample_batch(rng, 8)
    targets = extract_raw_targets(batch, proc)

    assert targets["physics_token"].shape == (8, 500, 1)
    assert targets["belief_token"].shape == (8, 500, 3)
    assert targets["physics_cycle"].shape == (8, 50, 1)
    assert targets["belief_cycle"].shape == (8, 50, 3)
    assert np.all(np.isfinite(targets["physics_token"]))
    assert np.all(np.isfinite(targets["belief_token"]))


def test_probe_features_and_targets_all_6_modes():
    N, seq_len, d_model = 8, 500, 32
    n_steps, m = 10, 50
    L = 5
    streams = [np.random.randn(N, seq_len, d_model).astype(np.float32) for _ in range(L)]
    raw_targets = {
        "physics_token": np.random.randn(N, seq_len, 1).astype(np.float32),
        "belief_token": np.random.randn(N, seq_len, 3).astype(np.float32),
        "physics_cycle": np.random.randn(N, m, 1).astype(np.float32),
        "belief_cycle": np.random.randn(N, m, 3).astype(np.float32),
    }

    # 1. single_layer_single_token
    X, yp, yb, g = extract_probe_features_and_targets(streams, raw_targets, "single_layer_single_token", n_steps, m)
    assert X.shape == (N * seq_len * L, d_model)
    assert yp.shape == (N * seq_len * L, 1)
    assert yb.shape == (N * seq_len * L, 3)
    assert g.shape == (N * seq_len * L,)
    assert set(g) == set(range(N))

    # 2. all_layers_single_token
    X, yp, yb, g = extract_probe_features_and_targets(streams, raw_targets, "all_layers_single_token", n_steps, m)
    assert X.shape == (N * seq_len, L * d_model)
    assert yp.shape == (N * seq_len, 1)
    assert yb.shape == (N * seq_len, 3)
    assert g.shape == (N * seq_len,)

    # 3. single_layer_cycle
    X, yp, yb, g = extract_probe_features_and_targets(streams, raw_targets, "single_layer_cycle", n_steps, m)
    assert X.shape == (N * m * L, n_steps * d_model)
    assert yp.shape == (N * m * L, 1)
    assert yb.shape == (N * m * L, 3)
    assert g.shape == (N * m * L,)

    # 4. all_layers_cycle
    X, yp, yb, g = extract_probe_features_and_targets(streams, raw_targets, "all_layers_cycle", n_steps, m)
    assert X.shape == (N * m, n_steps * L * d_model)
    assert yp.shape == (N * m, 1)
    assert yb.shape == (N * m, 3)
    assert g.shape == (N * m,)

    # 5. single_layer_last_token
    X, yp, yb, g = extract_probe_features_and_targets(streams, raw_targets, "single_layer_last_token", n_steps, m)
    assert X.shape == (N * m, d_model)
    assert yp.shape == (N * m, 1)
    assert yb.shape == (N * m, 3)
    assert g.shape == (N * m,)

    # 6. all_layers_last_token
    X, yp, yb, g = extract_probe_features_and_targets(streams, raw_targets, "all_layers_last_token", n_steps, m)
    assert X.shape == (N * m, L * d_model)
    assert yp.shape == (N * m, 1)
    assert yb.shape == (N * m, 3)
    assert g.shape == (N * m,)


def test_cross_validated_ridge():
    rng = np.random.default_rng(42)
    N_groups = 20
    rows_per_group = 10
    total_samples = N_groups * rows_per_group
    groups = np.repeat(np.arange(N_groups), rows_per_group)

    # 1. Linear relationship
    X = rng.standard_normal((total_samples, 10))
    w = rng.standard_normal((10, 1))
    y = X @ w + 0.01 * rng.standard_normal((total_samples, 1))

    dev_mask = np.isin(groups, np.arange(15))
    test_mask = ~dev_mask

    res = cross_validated_ridge(X, y, groups, dev_mask, test_mask)
    assert res["test_r2"] > 0.9
    assert res["cv_r2"] > 0.9

    # 2. Uncorrelated random data
    y_random = rng.standard_normal((total_samples, 1))
    res_rand = cross_validated_ridge(X, y_random, groups, dev_mask, test_mask)
    assert res_rand["test_r2"] < 0.3


def test_ridge_r2_gpu():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((100, 5)).astype(np.float32)
    w = np.array([[1.], [-2.], [0.5], [0.], [3.]], dtype=np.float32)
    y = (X @ w + 0.01 * rng.standard_normal((100, 1))).astype(np.float32)

    # CPU solve
    r2_cpu = _ridge_r2_gpu(X[:80], y[:80], X[80:], y[80:], alpha=1.0, device="cpu")
    assert r2_cpu > 0.95

    # GPU solve if available
    if torch.cuda.is_available():
        r2_gpu = _ridge_r2_gpu(X[:80], y[:80], X[80:], y[80:], alpha=1.0, device="cuda:0")
        assert r2_gpu > 0.95
        assert abs(r2_cpu - r2_gpu) < 1e-4


def test_cross_validated_ridge_gpu():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    rng = np.random.default_rng(42)
    N_groups = 10
    rows_per_group = 8
    total_samples = N_groups * rows_per_group
    groups = np.repeat(np.arange(N_groups), rows_per_group)

    X = rng.standard_normal((total_samples, 6))
    w = rng.standard_normal((6, 3))
    y = X @ w + 0.01 * rng.standard_normal((total_samples, 3))

    dev_mask = np.isin(groups, np.arange(8))
    test_mask = ~dev_mask

    res = cross_validated_ridge(X, y, groups, dev_mask, test_mask, device="cuda:0")
    assert res["test_r2"] > 0.9
    assert res["cv_r2"] > 0.9


def test_cpu_gpu_ridge_equivalence():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    rng = np.random.default_rng(123)
    N_groups = 15
    rows_per_group = 12
    total_samples = N_groups * rows_per_group
    groups = np.repeat(np.arange(N_groups), rows_per_group)

    X = rng.standard_normal((total_samples, 8)).astype(np.float32)
    w = rng.standard_normal((8, 3)).astype(np.float32)
    y = (X @ w + 0.05 * rng.standard_normal((total_samples, 3))).astype(np.float32)

    dev_mask = np.isin(groups, np.arange(10))
    test_mask = ~dev_mask

    res_cpu = cross_validated_ridge(
        X, y, groups, dev_mask, test_mask,
        ridge_alphas=(0.1, 1.0, 10.0, 50.0), cv_folds=3, device="cpu",
    )
    res_gpu = cross_validated_ridge(
        X, y, groups, dev_mask, test_mask,
        ridge_alphas=(0.1, 1.0, 10.0, 50.0), cv_folds=3, device="cuda:0",
    )

    assert res_cpu["alpha"] == res_gpu["alpha"]
    assert res_cpu["feature_dim"] == res_gpu["feature_dim"]
    np.testing.assert_allclose(res_cpu["cv_r2"], res_gpu["cv_r2"], rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(res_cpu["test_r2"], res_gpu["test_r2"], rtol=1e-4, atol=1e-5)



def test_generate_summary_plot_3d_tetrahedron():
    from _worker import generate_summary_plot
    from physics.messk import simplex_embedding

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        proc = build_process(delta_v=0.7, gamma=0.3, dt=0.2, n_steps=5, m=4)

        probe_rows = []
        for m_name in PROBE_MODES:
            for m_type in ["trained", "random_init"]:
                for t_name in ["belief", "physics"]:
                    probe_rows.append({
                        "mode": m_name,
                        "model_type": m_type,
                        "target": t_name,
                        "test_r2": 0.8 if m_type == "trained" else 0.1,
                    })
        probe_df = pd.DataFrame(probe_rows)
        job = {
            "config_name": "test_cfg",
            "delta_v": 0.7, "gamma": 0.3, "dt": 0.2, "n_steps": 5, "m": 4,
        }

        N_vis, m_cycles = 8, 4
        # Synthetic 3D belief on simplex
        tet = simplex_embedding(4)
        hmm_states = np.random.randint(0, 4, size=(N_vis, m_cycles))
        belief_gt = tet[hmm_states] + 0.02 * np.random.randn(N_vis, m_cycles, 3)
        belief_pred = tet[hmm_states] + 0.05 * np.random.randn(N_vis, m_cycles, 3)

        generate_summary_plot(
            tmp_path, job, width=32, seed=0, horizons=[(1, "k1")],
            loss_histories={}, probe_df=probe_df, proc=proc, device="cpu",
            n_layers=2,
            belief_gt=belief_gt,
            belief_pred=belief_pred,
            hmm_states=hmm_states,
        )
        png_path = tmp_path / "d32_seed0_summary.png"
        assert png_path.exists()
        assert png_path.stat().st_size > 10_000


def test_training_and_plotting():
    from _worker import generate_summary_plot
    from models.train import TrainConfig, train
    import pandas as pd

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        proc = build_process(delta_v=0.7, gamma=0.3, dt=0.2, n_steps=5, m=4)
        seq_len = proc.seq_len  # 20
        sampler = make_sampler(proc)

        cfg = ModelConfig(vocab_size=181, n_ctx=seq_len, n_layers=2, n_heads=1, d_model=32, d_mlp=128, seed=0)
        model = LookaheadTransformer(cfg, k=1)
        train_cfg = TrainConfig(total_tokens=500, batch_size=4, learning_rate=1e-3, seed=0, log_every=10, checkpoint_at=())

        report = train(model, sampler, seq_len, train_cfg, device="cpu")
        assert report["tokens_seen"] == report["steps"] * 4 * seq_len
        assert "history" in report
        assert report["final"] is not None

        pt_path = tmp_path / "d32_k1_seed0_final.pt"
        torch.save(model.state_dict(), pt_path)

        # Probe df for plotting test
        probe_rows = []
        for m_name in PROBE_MODES:
            for m_type in ["trained", "random_init"]:
                for t_name in ["belief", "physics"]:
                    probe_rows.append({
                        "mode": m_name,
                        "model_type": m_type,
                        "target": t_name,
                        "test_r2": 0.5 if m_type == "trained" else 0.1,
                    })
        probe_df = pd.DataFrame(probe_rows)
        job = {
            "config_name": "test_cfg",
            "delta_v": 0.7, "gamma": 0.3, "dt": 0.2, "n_steps": 5, "m": 4,
        }
        loss_histories = {"k1": report["history"]}
        generate_summary_plot(
            tmp_path, job, width=32, seed=0, horizons=[(1, "k1")],
            loss_histories=loss_histories, probe_df=probe_df, proc=proc, device="cpu",
            n_layers=2,
        )
        png_path = tmp_path / "d32_seed0_summary.png"
        assert png_path.exists()
        assert png_path.stat().st_size > 10_000


def test_worker_cli_full_cycle():
    import subprocess
    import sys

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        queue_path = tmp_path / "_job_queue.json"
        cfg_name = "test_dt0.2_gamma0.3_dv0.7_n10"
        job_id = f"{cfg_name}_d32_seed0"

        job = {
            "job_id": job_id,
            "config_name": cfg_name,
            "delta_v": 0.7,
            "gamma": 0.3,
            "dt": 0.2,
            "n_steps": 10,
            "m": 5,
            "seq_len": 50,
            "width": 32,
            "seed": 0,
            "horizons": [1, 5, 10],
        }
        queue_path.write_text(json.dumps([job]))

        worker_script = SWEEPS_DIR / "_worker.py"
        cmd = [
            sys.executable,
            str(worker_script),
            "--queue-file", str(queue_path),
            "--output-dir", str(tmp_path),
            "--worker-id", "0",
            "--device", "cpu",
            "--total-tokens", "1000",
            "--batch-size", "4",
            "--n-probe-traj", "16",
            "--cv-folds", "2",
            "--ridge-alphas", "1.0,10.0",
            "--n-layers", "2",
            "--log-every", "10",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        assert res.returncode == 0, f"Worker failed:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"

        cfg_dir = tmp_path / cfg_name
        assert cfg_dir.exists()

        # Check 3 horizons trained: .pt and .json
        for h in ["k1", "kn2", "kn"]:
            pt_file = cfg_dir / f"d32_{h}_seed0_final.pt"
            json_file = cfg_dir / f"d32_{h}_seed0_final.json"
            assert pt_file.exists(), f"Missing {pt_file.name}"
            assert json_file.exists(), f"Missing {json_file.name}"
            meta = json.loads(json_file.read_text())
            assert meta["k_suffix"] == h
            assert meta["vocab_size"] == 181
            assert meta["d_model"] == 32
            assert meta["seed"] == 0

        # Check probe CSV has 72 rows
        probe_csv = cfg_dir / "d32_seed0_probes.csv"
        assert probe_csv.exists()
        probe_df = pd.read_csv(probe_csv)
        assert len(probe_df) == 72
        assert set(probe_df["horizon"]) == {"k1", "kn2", "kn"}
        assert set(probe_df["model_type"]) == {"trained", "random_init"}
        assert set(probe_df["mode"]) == set(PROBE_MODES)
        assert set(probe_df["target"]) == {"physics", "belief"}

        # Check summary plot exists
        summary_png = cfg_dir / "d32_seed0_summary.png"
        assert summary_png.exists()
        assert summary_png.stat().st_size > 10_000

        # Check pipeline_state.json contains completed job
        state_file = tmp_path / "pipeline_state.json"
        assert state_file.exists()
        state = json.loads(state_file.read_text())
        assert job_id in state["completed"]

        # Run worker second time: queue empty, exits cleanly
        res2 = subprocess.run(cmd, capture_output=True, text=True)
        assert res2.returncode == 0
        assert "Queue empty and no active jobs. Exiting." in res2.stdout


