# Double Pendulum Production Training & Multi-Platform Queue Architecture

This document specifies the architecture, execution pipeline, and re-entrant queue framework for `kaggle_deployment/double_pendulum_kaggle_train.ipynb` and `scripts/production_train.py`, a high-throughput, multi-stage training and probing pipeline supporting three execution modes: **`rtx5090`** (Vast.ai / local workstation with 5 concurrent workers on 1x RTX 5090), **`kaggle`** (dual-GPU T4x2 with 2 workers), and **`local_cpu`** (single-worker CPU development/smoke).

---

## 1. System Mission & Multi-Stage Architecture

The goal of this pipeline is to execute a comprehensive, empirical investigation of the chaotic, Lagrangian **Double Pendulum** (`double_pendulum_mess4`). 

The pipeline runs across distinct phases with parameter distillation gates, followed by a **270-model production queue** and a **re-entrant continuation framework**:

```mermaid
flowchart TD
    subgraph Phase1["Phase 1: Physics Screening & 10-Tuple Distillation (CPU-Only, Single Seed)"]
        S1["Sweep 1A: Δv vs n Sweep (Candidate Impulse Horizons, Single Seed)"]
        D1["Distillation 1A: Select Top-Ranked Candidate (Δv, n) Pairs"]
        OptG["Energy-Balance Step: Compute Optimal γ* & Generate 9 Gammas
        γ = k · γ* and k · γ* ± dγ, where k ∈ {0.03, 0.07, 0.10}, dγ = 0.05"]
        S2["Sweep 1B: (Chosen Δv, n) vs 9 Candidate Gammas (Single Seed)"]
        D2["Distillation 1B: Select Top 10 Best Physics Tuples (Δv, γ, n)
        Export Artifact: distilled_physics_tuples.csv & .json"]
        S1 --> D1 --> OptG --> S2 --> D2
    end

    subgraph Phase2["Phase 2: 270-Model Production Queue (Multi-Platform Concurrency)"]
        Q["Model Job Queue:
        10 Physics Tuples × 3 Widths (32, 64, 128) × 3 Lookaheads (_k1, _k_mid, _k_full) × 3 Seeds (0, 1, 2)
        = 270 Total Models | Batch Size: 1024 | Token Budget: 400M tokens/model"]
        
        subgraph ModeDispatch["Hardware Concurrency Dispatch by Mode"]
            direction TB
            M_5090["Mode 'rtx5090' (Vast.ai / Local Workstation):
            5 Concurrent Worker Processes on cuda:0 (32GB VRAM, Ryzen 7 16-thread)
            Total Run Time: ~2.2–2.5 Hours"]
            M_Kaggle["Mode 'kaggle' (Kaggle Dual Tesla T4):
            2 Workers across cuda:0 and cuda:1
            Total Run Time: ~61 Hours (~5 sessions)"]
            M_CPU["Mode 'local_cpu':
            1 Worker on CPU (Micro-batch 16)"]
        end

        Probe["Synchronous In-Worker Probing & Artifact Bundling:
        4-Way Ridge Probes (Trained vs Random Init)
        Standardized Lookahead Probability Map
        Instant Packaging of results/{model_id}.zip"]
        
        Q --> ModeDispatch
        ModeDispatch --> Probe
    end

    subgraph Phase3["Phase 3: Emergence Synthesis & Packaging"]
        Out["Consolidated Master Archive:
        master_probe_results.csv (2,160 probe rows)
        5-Panel Synthesis Dashboard (.png)
        double_pendulum_results.zip"]
        Probe --> Out
    end

    subgraph Phase4["Phase 4: Re-entrant Continuation (Dormant by Default)"]
        ReQ["Re-entrant Continuation Interface:
        requeue_model(...) / requeue_top_models(...)
        REQUEUE_JOBS = []"]
        Out -.-> ReQ
        ReQ -.-> Q
    end

    D2 --> Q
```

---

## 2. Environment Setup, Multi-Platform Modes & Authenticated Repository Resolution

At the very top of `kaggle_deployment/double_pendulum_kaggle_train.ipynb` (and in `scripts/production_train.py`), **Cell 1** initializes the execution environment, establishes the hardware platform mode (`rtx5090`, `kaggle`, or `local_cpu`), handles private GitHub authentication, and injects module paths:

```python
# ==============================================================================
# Cell 1: Multi-Platform Setup, Hardware Modes & Authenticated Repository Init
# ==============================================================================
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

# ------------------------------------------------------------------------------
# Hardware Platform Mode Detection ('rtx5090', 'kaggle', 'local_cpu')
# ------------------------------------------------------------------------------
IS_KAGGLE = Path("/kaggle").exists()

def detect_platform_mode() -> str:
    env_mode = os.environ.get("BELIEF_PLATFORM", "").lower().strip()
    if env_mode in ("rtx5090", "kaggle", "local_cpu"):
        return env_mode
    if IS_KAGGLE:
        return "kaggle"
    if torch.cuda.is_available():
        dev_name = torch.cuda.get_device_name(0).lower()
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if "5090" in dev_name or vram_gb >= 28.0:
            return "rtx5090"
        if torch.cuda.device_count() >= 2:
            return "kaggle"
        return "rtx5090"
    return "local_cpu"

PLATFORM_MODE = detect_platform_mode()

# Configure workers, device mappings, and output paths per platform mode
if PLATFORM_MODE == "rtx5090":
    # 1x RTX 5090 (32GB VRAM) on Vast.ai / Local Workstation with 16-thread Ryzen CPU:
    # 5 concurrent worker processes multiplexed onto cuda:0 (12.5GB VRAM total)
    NUM_WORKERS = int(os.environ.get("BELIEF_NUM_WORKERS", "5"))
    WORKER_DEVICES = ["cuda:0"] * NUM_WORKERS
    OUTPUT_BASE = Path(os.environ.get("BELIEF_OUTPUT_DIR", "./experiments"))
    RESULTS_DIR = OUTPUT_BASE / "results"
elif PLATFORM_MODE == "kaggle":
    # Dual Tesla T4 (16GB VRAM each) on Kaggle:
    # 2 workers distributed across cuda:0 and cuda:1
    DEVICE_COUNT = torch.cuda.device_count()
    NUM_WORKERS = max(1, min(2, DEVICE_COUNT))
    WORKER_DEVICES = [f"cuda:{i}" for i in range(NUM_WORKERS)] if DEVICE_COUNT > 0 else ["cpu"]
    OUTPUT_BASE = Path("/kaggle/working")
    RESULTS_DIR = OUTPUT_BASE / "results"
else:  # local_cpu
    # Local CPU debug or smoke validation:
    NUM_WORKERS = int(os.environ.get("BELIEF_NUM_WORKERS", "1"))
    WORKER_DEVICES = ["cpu"] * NUM_WORKERS
    OUTPUT_BASE = Path("./experiments")
    RESULTS_DIR = OUTPUT_BASE / "results"

OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print(f"Platform Mode:    {PLATFORM_MODE.upper()}")
print(f"Compute Devices:  {WORKER_DEVICES}")
print(f"Worker Pool Size: {NUM_WORKERS} concurrent worker processes")
print(f"Output Base Path: {OUTPUT_BASE}")

# ------------------------------------------------------------------------------
# Repository Resolution & Authenticated Private Git Clone
# ------------------------------------------------------------------------------
REPO_OWNER = "ManjotSingh08x"
REPO_NAME = "Belief-in-physics"
REPO_BRANCH = os.environ.get("BELIEF_REPO_BRANCH", "double_pendulum_system")

# If using Kaggle Secrets (Add-ons -> Secrets -> GITHUB_TOKEN or GH_TOKEN), it is read automatically.
# Otherwise, paste your token here or set the GITHUB_TOKEN environment variable:
GITHUB_TOKEN = ""

def get_auth_token() -> Optional[str]:
    if GITHUB_TOKEN.strip():
        return GITHUB_TOKEN.strip()
    try:
        from kaggle_secrets import UserSecretsClient
        user_secrets = UserSecretsClient()
        for k in ["GITHUB_TOKEN", "GH_TOKEN", "github_token", "gh_token"]:
            try:
                tok = user_secrets.get_secret(k)
                if tok and tok.strip():
                    return tok.strip()
            except Exception:
                pass
    except Exception:
        pass
    return os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")

def resolve_repository_root() -> Path:
    # 1. Local checkout detection
    candidate = Path.cwd()
    for p in [candidate, candidate.parent, candidate.parent.parent]:
        if (p / "physics" / "systems" / "double_pendulum.py").exists():
            return p.resolve()

    # 2. Check attached Kaggle Datasets (/kaggle/input)
    input_root = Path("/kaggle/input")
    if input_root.exists():
        for marker in input_root.rglob("physics/systems/double_pendulum.py"):
            source_dir = marker.parents[2]
            target_repo = Path("/kaggle/working") / REPO_NAME
            if not target_repo.exists():
                shutil.copytree(source_dir, target_repo, dirs_exist_ok=True)
            return target_repo.resolve()

    # 3. Clone from private GitHub repository via HTTPS with Token Auth
    target_repo = (Path("/kaggle/working") if IS_KAGGLE else Path.cwd()) / REPO_NAME
    if (target_repo / "physics" / "systems" / "double_pendulum.py").exists():
        return target_repo.resolve()

    token = get_auth_token()
    if token:
        clone_url = f"https://x-access-token:{token}@github.com/{REPO_OWNER}/{REPO_NAME}.git"
        print(f"Cloning private repo '{REPO_OWNER}/{REPO_NAME}' (branch: {REPO_BRANCH}) with GitHub Token...")
    else:
        clone_url = f"https://github.com/{REPO_OWNER}/{REPO_NAME}.git"
        print(f"Attempting clone of '{REPO_OWNER}/{REPO_NAME}' (branch: {REPO_BRANCH})...")

    subprocess.run(["git", "clone", "--depth", "1", "--branch", REPO_BRANCH, clone_url, str(target_repo)], check=True)
    return target_repo.resolve()

ROOT = resolve_repository_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
print(f"Repository Root successfully linked: {ROOT}")

# ------------------------------------------------------------------------------
# Core Physics & Pipeline Imports
# ------------------------------------------------------------------------------
from physics.systems.double_pendulum import DoublePendulum
from physics.messk import MessKProcess, simplex_embedding
from physics.messk_configs import make_process, SYSTEMS, DOUBLE_PENDULUM_BINS
from physics.visualise import compute_lyapunov, continuous_trajectory_unwrap
from physics.controls import compute_optimal_gamma, grid_screen, stability, trace

# ------------------------------------------------------------------------------
# Global Execution Mode & Fast Verification (SMOKE Mode)
# ------------------------------------------------------------------------------
# Set SMOKE = True for a rapid 3-minute dry-run validating end-to-end multi-GPU
# workers, probing, probability maps, state manifest, and packaging.
# Set SMOKE = False for full 270-model production training (400M tokens/model).
SMOKE = os.environ.get("BELIEF_SMOKE", "False").lower() in ("true", "1", "yes")
print(f"Pipeline Mode: {'SMOKE (Rapid Dry-Run)' if SMOKE else 'PRODUCTION (Full 270-Model Matrix)'}")
```

### Invariant: Dual-Mode Execution (Interactive Override + Headless Automation)
The notebook operates seamlessly in two modalities:
1. **Headless 'Run All' / Kaggle Background Commit**: If executed without interactive intervention, distillation gates automatically select the top-ranked candidates (the "green ones" passing stability and Bayes-gap thresholds) and proceed through the full 270-model queue.
2. **Interactive Exploration**: When executed cell-by-cell, interactive override lists (e.g. `CHOSEN_PAIR_IDS`, `CHOSEN_TUPLE_IDS`) allow manually choosing specific physical configurations before downstream training begins.

---

## 3. Phase 1: Physics Screening & 10-Tuple Distillation (CPU-Only)

### Invariant: Constant Sequence Length ($m \times n = 500$)
- In accordance with the [three_phase_pipeline.ipynb](file:///home/vedansh/projects/ML/Belief-in-physics/notebooks/three_phase_pipeline.ipynb) specification, all generated trajectories enforce a strictly fixed sequence length of **$m \times n = 500$ tokens**.
- When varying integration impulse step size $n$, the impulse cycle count $m$ dynamically scales as:
  $$m = \frac{500}{n}$$
  - For $n = 5 \implies m = 100$ cycles ($100 \times 5 = 500$ tokens)
  - For $n = 10 \implies m = 50$ cycles ($50 \times 10 = 500$ tokens)
  - For $n = 20 \implies m = 25$ cycles ($25 \times 20 = 500$ tokens)
  - For $n = 25 \implies m = 20$ cycles ($20 \times 25 = 500$ tokens)
- This guarantees identical context window length ($n_{\text{ctx}} = 500$) across all models while systematically varying the continuous physics integration duration between discrete kicks.

### Sweep 1A: 20-Pair $(\Delta v \times n)$ Grid Screening
- **Exposed Parameter Lists**:
  ```python
  DELTA_V_CANDIDATES = [0.6, 0.9, 1.2, 1.5, 2.0]
  N_CANDIDATES       = [5, 10, 20, 25]  # Divisors ensuring exact integer m = 500 // n
  ```
  - Full factorial: $5 \times 4 = 20$ candidate pairs.
  - Constraint: Evaluated with baseline viscous damping $\gamma_1 = \gamma_2 = 0.65$ (authoritative contractive baseline for $dt=0.2$ from `docs/HYPERPARAM.md`).
  - Sequence length invariant: `m = 500 // n`, yielding exactly 500 tokens per trajectory.
  - All other physics parameters remain fixed at defaults (`th1_0 = 0.9, th2_0 = -0.4, dt = 0.2, omega_max = 10.0`, 2D mixed-radix `(50, 50)` bins with vocabulary 2500, `joint1_action_gain = 2.0`).
- **Screening Checks (Single Seed, `seed=42`)**:
  - Lyapunov exponent $\lambda$ (contractive margin $\lambda < -0.30/\text{s}$), observation clipping ($< 1\%$), used bins ($> 10\%$), driven-vs-free gap, and Bayes-optimal gap ($\ge 0.15$ nats). (Single seed evaluation avoids redundant overhead prior to final Phase 3 production).
- **Interactive Output**:
  - HTML heatmap matrix colored by GO / NO-GO status across $(\Delta v, n)$.
  - Numbered summary DataFrame (`pair_id`: `0` to `19`).

### Distillation Checkpoint 1A: Select Candidate Pairs for Optimal Damping Calculation
- Interactive list cell:
  ```python
  # Enter chosen pair IDs from Sweep 1A for optimal gamma calculation & gamma sweep
  CHOSEN_PAIR_IDS = [2, 5, 6, 8, 9, 11, 13, 14, 17, 18]
  ```
- *Headless Fallback*: If left unedited, automatically selects the top 10 ranked $(\Delta v, n)$ candidate pairs from Sweep 1A based on Bayes gap and stability margin.

### Intermediate Step: Steady-State Energy Balance ($\gamma^*$ Calculation & 9-Point Damping Generation)
- Sourced directly from `notebooks/explorer.ipynb` Section 6 (`compute_optimal_gamma`):
  Computes the exact viscous damping parameter $\gamma^*$ where average energy dissipated balances kick injection:
  $$\langle \Delta E_{\text{injected}} \rangle = \langle \Delta E_{\text{dissipated}} \rangle$$
  via ODE rollout simulation:
  ```python
  res = compute_optimal_gamma(
      system_name="double_pendulum_mess4",
      m=500 // n,
      n_steps=n,
      dt=0.2,
      delta_v=delta_v,
      stay=0.7,
      alpha=0.7,
      damping_field="gamma1",  # sets gamma1 = gamma2
  )
  gamma_opt = res["gamma_opt"]
  ```
- **9-Point Damping Generation per Tuple**:
  For each selected $(\Delta v, n)$ pair, take its calculated optimal $\gamma^*$ and generate **9 candidate damping values**: 3 unperturbed scalings ($k \cdot \gamma^*$) plus 6 perturbed values ($k \cdot \gamma^* \pm d\gamma$) with $d\gamma = 0.05$ across scaling factors $k \in \{0.03, 0.07, 0.10\}$:
  $$\gamma \in \{ k \cdot \gamma^*, \quad k \cdot \gamma^* \pm d\gamma \quad \text{for } k \in \{0.03, 0.07, 0.10\} \}$$
  Explicitly generating 9 candidate damping values per chosen $(\Delta v, n)$ tuple:
  1. $\gamma_1 = 0.03 \cdot \gamma^*$ (unperturbed)
  2. $\gamma_2 = 0.03 \cdot \gamma^* - 0.05$
  3. $\gamma_3 = 0.03 \cdot \gamma^* + 0.05$
  4. $\gamma_4 = 0.07 \cdot \gamma^*$ (unperturbed)
  5. $\gamma_5 = 0.07 \cdot \gamma^* - 0.05$
  6. $\gamma_6 = 0.07 \cdot \gamma^* + 0.05$
  7. $\gamma_7 = 0.10 \cdot \gamma^*$ (unperturbed)
  8. $\gamma_8 = 0.10 \cdot \gamma^* - 0.05$
  9. $\gamma_9 = 0.10 \cdot \gamma^* + 0.05$
  *(Enforcing a safety clamp $\gamma \ge 0.05$ to guarantee strictly dissipative physics, with $\gamma_1 = \gamma_2 = \gamma$)*.

### Sweep 1B: Damping Sweep on Chosen Tuples $(\text{Chosen } (\Delta v, n) \times 9\text{ Candidate } \gamma)$
- Evaluates the Cartesian product of chosen $(\Delta v, n)$ pairs and their respective 9 candidate $\gamma$ values (e.g. 10 pairs $\times$ 9 gammas = 90 physics candidate runs, evaluated on single seed `seed=42`).
- Evaluates contractive stability ($\lambda < -0.30/\text{s}$, 0% clipping, used bins $> 10\%$, Bayes gap $\ge 0.15$ nats, driven-vs-free trajectory separation).
- Produces ranked evaluation matrix and diagnostic breakdown.

### Distillation Checkpoint 2: Select Top 10 Physics Tuples (`ExperimentConfig`)
- Following the naming conventions of [three_phase_pipeline.ipynb](file:///home/vedansh/projects/ML/Belief-in-physics/notebooks/three_phase_pipeline.ipynb), data generation configurations are encapsulated as `ExperimentConfig` objects.
- Exactly **10 tuples** showing optimal contractive dynamics and high Bayes gap are selected for the downstream 270-model pipeline:
  ```python
  # Selected 10 best ExperimentConfig tuples: (delta_v, gamma, n, m with m*n=500)
  SELECTED_EXPERIMENT_CONFIGS = [
      ExperimentConfig(
          name="dp_dv1.2_g0.56_n10",
          physics=DoublePendulumConfig(delta_v=1.2, gamma1=0.56, gamma2=0.56, dt=0.2, obs_bins=(50, 50)),
          hmm=HMMConfig(n_states=4, alpha=0.7, stay=0.7),
          m=50, n=10, seed=20260925,
      ),
      ExperimentConfig(
          name="dp_dv1.2_g0.61_n10",
          physics=DoublePendulumConfig(delta_v=1.2, gamma1=0.61, gamma2=0.61, dt=0.2, obs_bins=(50, 50)),
          hmm=HMMConfig(n_states=4, alpha=0.7, stay=0.7),
          m=50, n=10, seed=20260925,
      ),
      ExperimentConfig(
          name="dp_dv1.2_g0.66_n10",
          physics=DoublePendulumConfig(delta_v=1.2, gamma1=0.66, gamma2=0.66, dt=0.2, obs_bins=(50, 50)),
          hmm=HMMConfig(n_states=4, alpha=0.7, stay=0.7),
          m=50, n=10, seed=20260925,
      ),
      ExperimentConfig(
          name="dp_dv0.9_g0.50_n10",
          physics=DoublePendulumConfig(delta_v=0.9, gamma1=0.50, gamma2=0.50, dt=0.2, obs_bins=(50, 50)),
          hmm=HMMConfig(n_states=4, alpha=0.7, stay=0.7),
          m=50, n=10, seed=20260925,
      ),
      ExperimentConfig(
          name="dp_dv0.9_g0.55_n10",
          physics=DoublePendulumConfig(delta_v=0.9, gamma1=0.55, gamma2=0.55, dt=0.2, obs_bins=(50, 50)),
          hmm=HMMConfig(n_states=4, alpha=0.7, stay=0.7),
          m=50, n=10, seed=20260925,
      ),
      ExperimentConfig(
          name="dp_dv1.5_g0.70_n10",
          physics=DoublePendulumConfig(delta_v=1.5, gamma1=0.70, gamma2=0.70, dt=0.2, obs_bins=(50, 50)),
          hmm=HMMConfig(n_states=4, alpha=0.7, stay=0.7),
          m=50, n=10, seed=20260925,
      ),
      ExperimentConfig(
          name="dp_dv1.2_g0.55_n20",
          physics=DoublePendulumConfig(delta_v=1.2, gamma1=0.55, gamma2=0.55, dt=0.2, obs_bins=(50, 50)),
          hmm=HMMConfig(n_states=4, alpha=0.7, stay=0.7),
          m=25, n=20, seed=20260925,
      ),
      ExperimentConfig(
          name="dp_dv1.2_g0.60_n20",
          physics=DoublePendulumConfig(delta_v=1.2, gamma1=0.60, gamma2=0.60, dt=0.2, obs_bins=(50, 50)),
          hmm=HMMConfig(n_states=4, alpha=0.7, stay=0.7),
          m=25, n=20, seed=20260925,
      ),
      ExperimentConfig(
          name="dp_dv1.5_g0.75_n20",
          physics=DoublePendulumConfig(delta_v=1.5, gamma1=0.75, gamma2=0.75, dt=0.2, obs_bins=(50, 50)),
          hmm=HMMConfig(n_states=4, alpha=0.7, stay=0.7),
          m=25, n=20, seed=20260925,
      ),
      ExperimentConfig(
          name="dp_dv2.0_g0.85_n10",
          physics=DoublePendulumConfig(delta_v=2.0, gamma1=0.85, gamma2=0.85, dt=0.2, obs_bins=(50, 50)),
          hmm=HMMConfig(n_states=4, alpha=0.7, stay=0.7),
          m=50, n=10, seed=20260925,
      ),
  ]
  ```
- *Headless Fallback & Automated Top-Ranking*: If left unedited, automatically selects the top 10 ranked "GO" (Green) tuples from Sweep 1B based on pass rate, stability margin, and Bayes gap, wrapping them in `ExperimentConfig` objects with $m = 500 // n$.
- **Export Level 1 Distillation Artifact**:
  The selected physical tuples and their stability/Bayes benchmarks are exported as an authoritative reference artifact:
  - `/kaggle/working/results/distilled_physics_tuples.csv`
  - `/kaggle/working/results/distilled_physics_tuples.json`
  - Captures: `tuple_id`, `delta_v`, `gamma1`, `gamma2`, `n_steps`, `m`, `pass_rate`, `lyapunov_mean`, `bayes_gap_mean`, `clipped_mean`, `used_bins_mean`, `gap_free_mean`, and recipe timestamp. Bundled into all downstream zip packages.

---

## 4. Phase 2: The 270-Model Production Queue (Deep Training, 3 Seeds)

### 4.1 The 270-Model Matrix
Rather than pre-selecting a single model width, the pipeline directly integrates all 3 candidate residual stream widths alongside impulse horizons and multi-seed replications. The complete production queue trains:

$$\text{Total Models} = 10\text{ physics tuples} \times 3\text{ widths } (32, 64, 128) \times 3\text{ lookaheads } (k \in \{1, \text{int}(n/2), n\}) \times 3\text{ seeds } (0, 1, 2) = \mathbf{270\text{ models}}$$

- **Residual Stream Widths ($d_{\text{model}}$)**:
  - `WIDTH_CANDIDATES = [32, 64, 128]`
  - Scaled MLP dimension: $d_{\text{mlp}} = 4 \times d_{\text{model}}$ (widths 32/128, 64/256, 128/512).
  - Fixed transformer depth & heads: $n_{\text{layers}} = 4, n_{\text{heads}} = 2$.
  - Fixed emission fidelity: $\alpha = 0.70$ (nominal Mess-4 standard from `docs/HYPERPARAM.md`).
- **Lookahead Horizons ($k$) & Explicit Suffixes**:
  - $k = 1$: Immediate next-token prediction $\to$ suffix `_k1` (e.g. `dmodel_128_dmlp_512_k1`).
  - $k = \text{int}(n/2)$: Mid-cycle lookahead (halfway through impulse cycle) $\to$ suffix `_k_mid` (e.g. `dmodel_128_dmlp_512_k_mid`).
  - $k = n$: Full-cycle lookahead (end of impulse cycle) $\to$ suffix `_k_full` (e.g. `dmodel_128_dmlp_512_k_full`).
- **Data Generation Seeds**:
  - `SEEDS = [0, 1, 2]` (3 independent random initializations and physics rollout seeds per configuration).
- **Model Naming Convention (per `three_phase_pipeline.ipynb`)**:
  - `exp_cfg.name`: encapsulates data generation parameters, e.g. `dp_dv1.2_g2.1_n10`.
  - `training_cfg.name`: encapsulates architecture, lookahead suffix, and seed, e.g. `dmodel_128_dmlp_512_k_mid`.
  - Saved model checkpoint: `{exp_cfg.name}_{training_cfg.name}_seed{seed}_final.pt`
    *(Examples: `dp_dv1.2_g2.1_n10_dmodel_128_dmlp_512_k1_seed0_final.pt`, `dp_dv1.2_g2.1_n10_dmodel_128_dmlp_512_k_mid_seed0_final.pt`, `dp_dv1.2_g2.1_n10_dmodel_128_dmlp_512_k_full_seed0_final.pt`)*
  - Metadata sidecar: `{exp_cfg.name}_{training_cfg.name}_seed{seed}_final.json`
  - Lookahead probability map: `{exp_cfg.name}_{training_cfg.name}_seed{seed}_lookahead_prob_map.png`
- **Sequence Length Invariant**:
  - Constant $m \times n = 500$ tokens per trajectory across all 270 models ($n \in \{5, 10, 20, 25\}$ with $m = 500 // n$).
- **Batch Size & Token Budget**:
  - **Batch Size**: `BATCH_SIZE = 1024` sequences per step ($512,000$ tokens per gradient step for maximum GPU saturation).
  - **Deep Training Budget**: `DEEP_TRAIN_TOKENS = 400_000_000` (400M tokens per model, corresponding to 781 gradient steps). Empirically locked via 15-model pilot saturation profiling (`experiments/results_saturation/`), capturing 98.16% of total loss reduction with <0.1 nat residual gap compared to 1B tokens.
  - (When `SMOKE = True`, scales to 2 models at 5M tokens for a rapid 3-minute validation).
- **Batch Size Retention Invariant (Optimization Dynamics)**:
  - `BATCH_SIZE = 1024` is **strictly locked**. Do **NOT** increase to 2048 or 4096 even when running on an RTX 5090 with 32 GB VRAM.
  - *Hard Rationale*:
    1. **CPU Simulation Bottleneck**: On modern GPUs, the transformer forward/backward pass takes $<0.3\text{ ms}$, while RK4 CPU physics takes $\approx 450\text{ ms}$ per step. Because the CPU must simulate the exact same total 400M tokens, doubling batch size halves steps but doubles step duration, yielding zero speedup.
    2. **Optimization Steps**: At $B=1024$, the model undergoes 781 gradient updates. Doubling to $B=2048$ cuts updates to 390; quadrupling to $B=4096$ cuts updates to 195. With fewer than 400 steps, Adam optimizer momentum buffers do not stabilize, causing severe underfitting and higher loss.
    3. **Concurrency Scaling Rule**: Concurrency is scaled through **worker processes (5 workers in `rtx5090` mode)**, NOT through batch size.
- **Disk Budget Invariant (Storage Headroom Preservation)**:
  - Save **ONLY the final model weights** (`_final.pt`), JSON sidecar (`_final.json`), probe tables, and lookahead probability map upon completing training. Intermediate checkpoints are skipped to preserve disk headroom.

### 4.2 Parallelism Architecture & Platform Modes ('rtx5090', 'kaggle', 'local_cpu')

The queue execution engine dynamically adapts its concurrency, device mapping, and process model according to the active `PLATFORM_MODE`:

1. **Mode 1: `rtx5090` (Vast.ai / Workstation — 5 Concurrent Worker Processes)**:
   - **Hardware Profile**: 1x NVIDIA GeForce RTX 5090 (32 GB GDDR7, 108.1 TFLOPS) paired with an 8-core / 16-thread AMD Ryzen 7 host CPU.
   - **Worker Pool**: Spawns **`NUM_WORKERS = 5` concurrent worker processes** via `torch.multiprocessing` / `ProcessPoolExecutor`.
   - **Device Mapping**: All 5 workers share `cuda:0`. Total VRAM consumption: $5 \times 2.5\text{ GB} = \mathbf{12.5\text{ GB}}$ (leaving 19.5 GB free VRAM out of 32 GB).
   - **Zero GIL Contention**: Each worker runs in an independent Python process mapped to a dedicated physical CPU core, executing RK4 physics simulations at full Zen 4 single-core IPC.
   - **Expected Performance**: Combined throughput of **~3.2M to 3.8M tokens/second**, completing the entire 270-model production run in **~2.2 to 2.5 hours total** (under $1.50 rental cost on Vast.ai).

2. **Mode 2: `kaggle` (Dual Tesla T4 — 2 Thread/Process Workers)**:
   - **Hardware Profile**: 2x NVIDIA Tesla T4 (16 GB GDDR6 each) on Kaggle with 2–4 shared vCPUs.
   - **Worker Pool**: Spawns `NUM_WORKERS = 2` workers: Worker 0 assigned to `cuda:0`, Worker 1 assigned to `cuda:1`.
   - **Expected Performance**: Combined throughput of ~490k tokens/second, taking **~61 hours total**. Automatically chunks across ~5 sequential 12-hour Kaggle sessions using `pipeline_state.json`.

3. **Mode 3: `local_cpu` (Single Worker Sequential / Smoke)**:
   - **Hardware Profile**: Local CPU execution without CUDA.
   - **Worker Pool**: `NUM_WORKERS = 1` worker on `cpu` with micro-batch size 16 for rapid dry-runs and pipeline testing.

4. **Dual-Path Auto-Resume & Session Recovery**:
   - Maintains a centralized manifest file at `{OUTPUT_BASE}/pipeline_state.json`.
   - **Startup Check**:
     1. Inspects `pipeline_state.json` to find already completed jobs.
     2. If starting on Kaggle where `/kaggle/working` is fresh, automatically searches `/kaggle/input` for any attached previous run output dataset, copies existing model directories to `{RESULTS_DIR}/`, and reconstructs `pipeline_state.json`.
     3. Completed models are bypassed; execution picks up seamlessly from the first pending job.

5. **Synchronous In-Worker Probing & Instant Model Packaging**:
   - Probing and probability map generation run **synchronously within the worker process** immediately upon reaching 400M tokens (~10-15s per model).
   - The worker saves `{model_id}_final.pt`, `{model_id}_final.json`, `{model_id}_probes.csv`, `{model_id}_probes.json`, and `{model_id}_lookahead_prob_map.png` into `{RESULTS_DIR}/{model_id}/`.
   - The worker immediately compresses this folder into `{RESULTS_DIR}/{model_id}.zip`.
   - Guaranteed atomic completeness: every completed model is 100% packaged and downloadable even if a session terminates prematurely.

### 4.3 Real-Time Observability & Live Health Verification Protocol

To ensure full transparency during multi-hour execution runs on Kaggle, Vast.ai, or local workstations, workers and the central dispatcher maintain active streaming telemetry and incremental persistence:

1. **Live Step-Level Loss Streaming**:
   - Every 50 gradient steps (out of 781 steps per 400M tokens), workers log:
     ```text
     [{model_id}] Step {step}/{total_steps} ({pct}%) | Tokens: {tokens_seen}M/{total_tokens}M | Loss: {loss:.4f} | LR: {lr:.2e}
     ```
   - **Empirical Health Criteria**:
     - **Initial Loss**: Step 1 starts around $\ln(2500) \approx 7.82\text{ nats}$ (uniform prior over $50 \times 50$ 2D token grid).
     - **Rapid Descent**: Loss descends sharply within the first 100–200 steps to $\approx 2.0 - 2.5\text{ nats}$.
     - **Optimizer Stability**: Monotonic cosine learning rate annealing, gradient norm $\le 1.0$, and non-NaN cross-entropy loss confirm stable numerical dynamics.

2. **In-Worker Probe Summary Printing**:
   - Immediately following the 781 training steps, in-worker synchronous probing finishes in $\approx 10-15\text{ s}$ and prints a diagnostic summary:
     ```text
     [{model_id}] Probe Results -> Max Belief R²: {belief_r2:+.3f} | Max Physics R²: {phys_r2:+.3f} | Gain over Random: {gain:+.3f}
     [{model_id}] Finished in {runtime:.1f}s. Packaged -> {model_id}.zip
     ```
   - **Empirical Health Criteria**:
     - $\text{Max Belief } R^2 > 0.50$ confirms linear readability of the hidden Mess-4 belief states.
     - $\text{Gain over Random} > +0.30$ confirms representational emergence from physics-driven training rather than random projection artifacts.

3. **Immediate Per-Model Artifact Availability**:
   - The moment a model finishes, `{RESULTS_DIR}/{model_id}/` immediately contains:
     - `{model_id}_lookahead_prob_map.png`: Visual heatmap of predicted probabilities vs actual future tokens $t+k$ (verifying smooth continuous curvature rather than horizontal plateaus).
     - `{model_id}_probes.csv` & `.json`: Exhaustive 4-mode $\times$ 2-domain probe results.
     - `{model_id}_final.pt` & `_final.json`: Final weights and configuration sidecar.
     - `{RESULTS_DIR}/{model_id}.zip`: Standalone zip archive ready for instant download without waiting for the full queue to terminate.

4. **Incremental Master CSV & Central Manifest Updates**:
   - Both `{OUTPUT_BASE}/master_probe_results.csv` and `{OUTPUT_BASE}/pipeline_state.json` are incrementally appended and flushed to disk as each model completes.
   - Allows users to query completed models and partial results midway through the 270-model run without waiting for all jobs to complete.

---

## 5. Phase 3: 4-Way Probing Scheme, Emergence Synthesis & Artifact Packaging

### 5.1 Standard 4-Way Linear Ridge Probing Protocol
Directly adopting the evaluation architecture of [three_phase_pipeline.ipynb](file:///home/vedansh/projects/ML/Belief-in-physics/notebooks/three_phase_pipeline.ipynb), representation emergence is analyzed across **4 distinct feature views** and tested against matched **randomly initialized baselines**:

1. **Four Feature Representation Views (`probe_modes`)**:
   - `all_layers_single_token`: Concatenated residual streams across all transformer blocks at impulse cycle boundaries ($t = c \times n - 1$). Feature dim: $n_{\text{layers}} \times d_{\text{model}}$.
   - `single_layer_cycle`: Residual stream of selected block across all $n$ tokens within the impulse cycle. Feature dim: $n \times d_{\text{model}}$.
   - `single_layer_last_token`: Residual stream of selected block at the last token before an impulse perturbation ($t = c \times n - 1$). **Cycle 0 ($t = n - 1$) is strictly excluded**, evaluating from Cycle 1 onwards ($t = 2n-1, 3n-1, \dots, mn-1$) to guarantee HMM belief synchronization past prior initialization. Feature dim: $d_{\text{model}}$.
   - `single_layer_all_token`: All individual tokens across the entire sequence from the last layer probed as distinct independent data points ($N \times (m \times n)$ points). Feature dim: $d_{\text{model}}$.

2. **Two Target Domains**:
   - `physics`: True continuous angular velocities $(v_1, v_2) = (\omega_1, \omega_2)$ at cycle ends.
   - `belief`: Exact hidden Markov posterior belief vector $P(\text{mood} \mid \text{tokens}_{1:t}) \in \Delta^3$ embedded in 3D simplex coordinates via `simplex_embedding(4)`.

3. **Evaluation Protocol & On-The-Fly Test Data**:
   - Evaluation trajectories ($N_{\text{eval}} = 512$) are generated **fresh on the fly** for each probe evaluation from `proc`.
   - Trajectories split into train/dev ($80\%$) and held-out test ($20\%$) using `GroupKFold(n_splits=5)`.
   - Feature matrices standardized via `StandardScaler()`.
   - Regularization tuned via `GridSearchCV` over `ridge_alphas = (0.1, 1.0, 10.0, 50.0, 100.0)`.
   - Evaluated on untouched test trajectories reporting `test_r2`.

4. **Matched Randomly Initialized Control (`random_init`)**:
   - For **every single trained model**, an identical architecture initialized with random weights (`random_model`, matching model seed) is probed under the exact same 4 modes and targets.
   - Computes:
     - `trained`: Test $R^2$ of trained model
     - `random_init`: Test $R^2$ of untrained control
     - `trained_minus_random`: Net representational gain strictly attributable to training

### 5.2 Authoritative Artifact Inventory & Directory Structure
Outputs are partitioned into structured per-model subdirectories under `{RESULTS_DIR}/` (auto-resolving to `/kaggle/working/results/` in `kaggle` mode, or `./experiments/results/` in `rtx5090` / `local_cpu` mode):

```text
OUTPUT_BASE/ (e.g. /kaggle/working/ or ./experiments/)
├── pipeline_state.json                         # Central resume manifest & job queue status
├── master_probe_results.csv                    # Consolidated 2,160-row probe evaluation table
├── double_pendulum_results.zip                 # Consolidated single zip containing all models & summaries
└── results/
    ├── distilled_physics_tuples.csv            # Authoritative Level 1 chosen physics parameters
    ├── distilled_physics_tuples.json           # Structured JSON of Level 1 screening benchmarks
    ├── scaling_curves.png                      # Synthesis Dashboard: Width scaling (32, 64, 128) vs R²
    ├── lookahead_comparison.png                # Synthesis Dashboard: Horizon (_k1, _k_mid, _k_full)
    ├── probe_views_breakdown.png               # Synthesis Dashboard: 4-way probe comparison
    ├── physics_vs_belief_scatter.png           # Synthesis Dashboard: Linear readability decoupling
    ├── trained_vs_random_gain.png              # Synthesis Dashboard: Net emergence gain
    ├── {model_id}/                             # Dedicated per-model directory
    │   ├── {model_id}_final.pt                 # Final model weights at 400M tokens
    │   ├── {model_id}_final.json               # Configs, tokens seen, and final loss sidecar
    │   ├── {model_id}_probes.csv               # 4-mode probe scores (trained vs random_init)
    │   ├── {model_id}_probes.json              # Structured probe hyperparameters & CV scores
    │   └── {model_id}_lookahead_prob_map.png   # Standardized lookahead probability heatmap
    ├── {model_id}.zip                          # Per-model zip archive (ready for instant download)
    ...
```

| Artifact | File Location Pattern | Content |
|---|---|---|
| **Level 1 Physics Table** | `results/distilled_physics_tuples.csv` | Distilled top-10 physics parameters and empirical stability/Bayes metrics |
| **Level 1 Physics Sidecar** | `results/distilled_physics_tuples.json` | Structured JSON containing parameter definitions and screening metrics |
| **Final Model Weights** | `results/{model_id}/{model_id}_final.pt` | PyTorch `state_dict` of the trained model at 400M tokens |
| **Model Metadata Sidecar** | `results/{model_id}/{model_id}_final.json` | Complete `ExperimentConfig`, `TrainingConfig`, model shape, tokens seen, final train/eval loss |
| **Lookahead Probability Map** | `results/{model_id}/{model_id}_lookahead_prob_map.png` | Standardized lookahead probability heatmap evaluated on fixed held-out sequence |
| **Per-Model Probe Table** | `results/{model_id}/{model_id}_probes.csv` | Full probe evaluations across 4 modes $\times$ 2 targets for both `trained` and `random_init` |
| **Per-Model Probe Sidecar** | `results/{model_id}/{model_id}_probes.json` | Structured JSON containing hyperparameters and CV scores |
| **Per-Model Archive** | `results/{model_id}.zip` | Individual compressed zip containing the single model's complete artifacts |
| **State Manifest** | `{OUTPUT_BASE}/pipeline_state.json` | Central pipeline execution state, completed model registry, and auto-resume manifest |
| **Master Probe CSV** | `{OUTPUT_BASE}/master_probe_results.csv` | Consolidated results across all 270 models (2,160 evaluated probe rows) |
| **Master Distribution Package** | `{OUTPUT_BASE}/double_pendulum_results.zip` | Consolidated archive of all model folders, sidecars, maps, CSVs, and state manifest |

### 5.3 Lookahead Probability Map & 5-Panel Synthesis Dashboard
1. **Lookahead Probability Map (Standardized Test Sequence)**:
   - Evaluated on a **standardized fixed held-out test trajectory** (fixed evaluation seed) so all 270 models can be compared side-by-side on the exact same physical trajectory:
     ```python
     fig, ax = plt.subplots(figsize=(13, 5))
     ax.imshow(probs[:-k].T, origin="lower", aspect="auto", cmap="magma")
     ax.plot(np.arange(len(actual)), actual, color="cyan", lw=1, label="actual next token")
     ax.set(xlabel="source position t", ylabel="token id",
            title=f"{chosen_key} k={k} lookahead probability heatmap; mean NLL={nll.mean():.3f}")
     ax.legend()
     plt.savefig(prob_map_path, dpi=150, bbox_inches="tight")
     ```
   - Saved into `results/{model_id}/` and bundled in `{model_id}.zip` and `double_pendulum_results.zip`.
2. **Comprehensive 5-Panel Synthesis Dashboard**:
   At the conclusion of the queue, the pipeline generates 5 high-resolution summary figures in `results/`:
   - `scaling_curves.png`: Model width scaling ($d_{\text{model}} \in \{32, 64, 128\}$) vs belief $R^2$ and physics $R^2$.
   - `lookahead_comparison.png`: Forecasting horizon comparison ($k=1$ vs $k=\text{int}(n/2)$ vs $k=n$).
   - `probe_views_breakdown.png`: Linear readability comparison across the 4 representation views.
   - `physics_vs_belief_scatter.png`: Representation decoupling scatter plots ($R^2_{\text{belief}}$ vs. $R^2_{\text{physics}}$) across $\Delta v \times \gamma$.
   - `trained_vs_random_gain.png`: Net emergence gain (`trained_minus_random`) isolating non-trivial feature learning.

### 5.4 Two-Tier Packaging Strategy
1. **Per-Model Subfolders & Instant Zips (`results/{model_id}/` & `{model_id}.zip`)**:
   - Generated immediately upon job completion for fault-tolerant downloading during multi-hour runs.
2. **Consolidated Master Package (`{OUTPUT_BASE}/double_pendulum_results.zip`)**:
   - Single-click archive packaging the entire `results/` hierarchy, manifest, and master CSVs.

---

## 6. Phase 4: Re-entrant Model Queue Framework (Continuation & Further Training)

A dedicated re-entrant framework allows any completed model to be reloaded and sent back into the training pipeline to train for additional tokens without restarting from scratch.

### 6.1 Architecture of a `ModelJob`
Every model in the pipeline is tracked by a stateful metadata descriptor and persisted in `pipeline_state.json`:
```python
@dataclass
class ModelJob:
    job_id: str                   # e.g. "dp_dv1.2_g2.1_n10_dmodel_128_dmlp_512_k_mid_seed0"
    exp_config: ExperimentConfig  # complete data generation configuration (m*n=500)
    train_config: TrainingConfig  # model architecture & lookahead k
    k: int                        # lookahead horizon (1, int(n/2), or n)
    k_suffix: str                 # "_k1", "_k_mid", or "_k_full"
    seed: int                     # data generation seed (0, 1, or 2)
    tokens_seen: int              # cumulative tokens trained so far
    target_tokens: int            # target tokens for current run
    checkpoint_path: Path         # path to final .pt weights
    sidecar_path: Path            # path to final metadata .json
    probe_csv_path: Path          # path to probe results .csv
    prob_map_path: Path           # path to lookahead probability map .png
    status: str                   # "QUEUED", "TRAINING", "PROBING", "COMPLETED"
    probe_history: list[dict]     # chronological history of probe scores across 4 modes
```

### 6.2 Re-queuing Mechanism (Dormant by Default)
By default, `REQUEUE_JOBS = []` so a standard 'Run All' run terminates cleanly after Phase 3. Any subset of models can be re-enqueued for deeper training interactively via:
```python
# Re-queue specific models for further training (e.g. train an extra 500M tokens)
requeue_model("dp_dv1.2_g2.1_n10_dmodel_128_dmlp_512_k_mid_seed0", additional_tokens=500_000_000)

# Or re-queue all top-performing models that meet an R² threshold:
requeue_top_models(metric="belief_r2", top_n=5, additional_tokens=1_000_000_000)
```
- **How Continuation Works**:
  1. The worker loads final weights from `{exp_cfg.name}_{training_cfg.name}_seed{seed}_final.pt` and restores optimizer state.
  2. The streaming data generator resumes using the model's exact `ExperimentConfig` and seed.
  3. The model trains for `additional_tokens` (e.g. from 400M to 800M tokens).
  4. The updated final weights overwrite `_final.pt`, and `_final.json` updates with new `tokens_seen` and loss.
  5. The probing worker re-evaluates the extended model across all 4 probe modes and refreshes the Lookahead Probability Map.
  6. `pipeline_state.json`, `master_probe_results.csv`, and emergence curves update in real time.

---

## 7. Operational Decisions & Implementation Details

The design decisions for the pipeline are resolved as follows:

- **Decision 1 (Checkpoint Storage Policy)**: **APPROVED**. Save **only the final model weights** (`_final.pt`), lookahead probability maps, probe CSVs/JSONs, and metadata sidecars. No intermediate checkpoint `.pt` files will be saved to disk, preserving disk headroom.
- **Decision 2 (Session Auto-Resume & State Resilience)**: **APPROVED**. Dual-path check inspecting `{OUTPUT_BASE}/pipeline_state.json` first, and if on Kaggle checking `/kaggle/input` for attached previous run outputs, copying completed models to `{RESULTS_DIR}/` before starting.
- **Decision 3 (Sequence Length Invariant)**: **APPROVED**. Fixed $m \times n = 500$ tokens across all tuples ($n \in \{5, 10, 20, 25\}$, with $m = 500 // n$).
- **Decision 4 (Probing Scheme & Baselines)**: **APPROVED**. Evaluates across 4 representation views against matched `random_init` controls. For `single_layer_last_token`, **Cycle 0 is excluded** ($t = 2n-1, \dots$), evaluating from Cycle 1 onwards. Test data is generated on the fly.
- **Decision 5 (Multi-Platform Hardware Modes & Worker Concurrency)**: **CONFIRMED**. Adaptive hardware dispatch across 3 modes:
  - `rtx5090`: Spawns **5 concurrent worker processes** on `cuda:0` utilizing 5 dedicated physical cores of the Ryzen 7 CPU, taking **~2.2 to 2.5 hours total**.
  - `kaggle`: Spawns **2 workers** across `cuda:0` and `cuda:1` taking **~61 hours total** across ~5 sessions.
  - `local_cpu`: Spawns **1 worker** on CPU for dry-runs and smoke tests.
- **Decision 6 (Two-Tier Packaging Architecture)**: **CONFIRMED**. Every model outputs to its own folder (`results/{model_id}/`) and is immediately zipped to `{model_id}.zip` upon job completion. A single master archive `double_pendulum_results.zip` consolidates all results for 1-click download.
- **Decision 7 (Direct Width Integration & Level 1 Artifact)**: **CONFIRMED**. Phase 2 separate width distillation sweep is removed; all 3 widths ($d_{\text{model}} \in \{32, 64, 128\}$) are trained in the main production queue (270 models total). The Level 1 chosen physics parameters are exported as `distilled_physics_tuples.csv` and `.json`.
- **Decision 8 (5-Panel Synthesis Dashboard)**: **CONFIRMED**. Automated generation of 5 synthesis figures (`scaling_curves.png`, `lookahead_comparison.png`, `probe_views_breakdown.png`, `physics_vs_belief_scatter.png`, `trained_vs_random_gain.png`) alongside `master_probe_results.csv`.
- **Decision 9 (Locked Batch Size Invariant)**: **LOCKED**. `BATCH_SIZE = 1024` sequences per step ($512,000$ tokens/step, 781 gradient steps) is strictly preserved. It is not increased to 2048 or 4096 because CPU physics generation bounds total time, and reducing gradient steps would degrade Adam optimizer convergence. Concurrency is scaled through workers (5 workers), not batch size.
- **Decision 10 (Deployment Location & Authenticated Private Git Access)**: **CONFIRMED**. All deployment files live in `kaggle_deployment/double_pendulum_kaggle_train.ipynb` (plus `scripts/production_train.py`). Cell 1 implements authenticated private repository cloning supporting Kaggle Secrets (`GITHUB_TOKEN` / `GH_TOKEN`), inline tokens, and attached datasets.
- **Decision 11 (Real-Time Observability & Incremental Manifest Flush)**: **CONFIRMED**. Real-time step loss logging every 50 gradient steps (`loss`, `lr`, `tokens_seen`), in-worker headline probe metrics printed immediately upon completion, and incremental persistence of both `pipeline_state.json` and `master_probe_results.csv` as each model completes, enabling live monitoring and partial queue evaluation without waiting for the full run to terminate.

---

## 8. Execution Readiness & Invariant Verification

All design decisions and physical invariants are resolved and locked:
1. **Multi-Platform Modes & Imports**: Cell 1 resolves repository root (with authenticated private cloning), auto-detects platform mode (`rtx5090`, `kaggle`, or `local_cpu`), establishes `SMOKE` flag, and configures worker concurrency (`NUM_WORKERS = 5` for RTX 5090, `2` for Kaggle T4x2).
2. **Sequence Length**: Fixed $m \times n = 500$ tokens across all impulse step sizes ($n \in \{5, 10, 20, 25\}$ with $m = 500 // n \in \{100, 50, 25, 20\}$).
3. **Phase 1 Physics**: (Δv, $n$) sweep first $\to$ steady-state energy-balance $\gamma^*$ calculation $\to$ 9-point damping generation (3 unperturbed $k \cdot \gamma^*$ + 6 perturbed $k \cdot \gamma^* \pm 0.05$ across $k \in \{0.03, 0.07, 0.10\}$) $\to$ damping sweep on chosen tuples (single seed `seed=42`) $\to$ Distillation 1B exporting `distilled_physics_tuples.csv` and `.json`.
4. **Phase 2 Deep Training Matrix**: 270 models (10 physics tuples $\times$ 3 widths [32, 64, 128] $\times$ 3 lookaheads `_k1`, `_k_mid`, `_k_full` $\times$ 3 seeds `[0, 1, 2]`) trained to 400M tokens/model (`DEEP_TRAIN_TOKENS = 400_000_000`, empirically calibrated via loss saturation profiling) at `BATCH_SIZE = 1024` sequences per step ($512,000$ tokens per gradient step, 781 steps).
5. **State Resilience & Live Observability**: Full auto-resume supported via `{OUTPUT_BASE}/pipeline_state.json` and `/kaggle/input` detection. Real-time streaming step loss logs (every 50 steps), live probe summaries, and incremental persistence of `pipeline_state.json` and `master_probe_results.csv` after every model completes.
6. **4-Way Linear Ridge Probes & Probability Map**: Grouped 5-Fold cross-validation across all 4 representation views (Cycle 0 excluded for last-token probe) against matched `random_init` controls, saving `{exp_cfg.name}_{training_cfg.name}_seed{seed}_final.pt`, `.json`, `_probes.csv`, and standardized `_lookahead_prob_map.png` packaged into both per-model `{model_id}.zip` archives and the master `{OUTPUT_BASE}/double_pendulum_results.zip`.
7. **Synthesis Analytics**: 5-panel dashboard summarizing width scaling, lookahead horizons, probe views, physics vs belief decoupling, and trained vs random emergence.
