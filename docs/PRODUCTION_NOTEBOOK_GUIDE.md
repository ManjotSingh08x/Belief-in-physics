# Double Pendulum Production Training Notebook Guide

This document is the authoritative user and operational guide for running the production training pipeline using **[`kaggle_deployment/double_pendulum_kaggle_train.ipynb`](file:///home/vedansh/projects/ML/Belief-in-physics/kaggle_deployment/double_pendulum_kaggle_train.ipynb)** (and its command-line orchestrator **[`scripts/production_train.py`](file:///home/vedansh/projects/ML/Belief-in-physics/scripts/production_train.py)**).

The pipeline executes the complete, 3-phase experimental investigation of the chaotic, Lagrangian **Double Pendulum** (`double_pendulum_mess4`) across a full **270-model production queue**, testing how hidden Markovian belief states survive through chaotic physical dynamics into transformer residual streams.

---

## 1. Quick Start & Execution Modes

The notebook automatically detects your compute environment and configures worker concurrency, device allocation, and directory paths:

| Platform Mode | Target Environment | Worker Concurrency | Device Allocation | Token Budget | Expected Duration |
|---|---|---|---|---|---|
| **`rtx5090`** | Vast.ai / Local Workstation | **5 Workers** (`ProcessPoolExecutor`) | `cuda:0` (multiplexed, ~12.5GB VRAM) | 400M / model | **~2.2 – 2.5 hours total** |
| **`kaggle`** | Kaggle Dual Tesla T4 | **2 Workers** (`ProcessPoolExecutor`) | `cuda:0` and `cuda:1` | 400M / model | **~61 hours total** (~5 sessions) |
| **`local_cpu`** | Local CPU / Smoke Test | **1 Worker** | `cpu` (micro-batch 16) | 5M / model | Rapid dry-run / verification |

---

## 2. Cell-by-Cell Notebook Walkthrough

### Cell 1: Overview & Architecture (Markdown)
Summarizes the 4-phase system architecture, parameter matrices, and execution modalities.

### Cell 2: Multi-Platform Setup, Hardware Modes & Authenticated Repository Init (Code)
- **Platform Detection**: Automatically detects `rtx5090`, `kaggle`, or `local_cpu`. Can be overridden via environment variable `BELIEF_PLATFORM`.
- **Private Repository Access**:
  - Automatically reads Kaggle Secrets for `GITHUB_TOKEN` or `GH_TOKEN`.
  - Supports manual token entry via `GITHUB_TOKEN = "ghp_..."`.
  - Checks local checkout, `/kaggle/input` attached datasets, and authenticated Git clone.
- **Physics Invariants Initialized**:
  - Enforces 2D mixed-radix observation grid `DOUBLE_PENDULUM_BINS = (50, 50)` ($2,500$ vocabulary).
  - Integration timestep $dt = 0.2$.
  - Fixed sequence length $m \times n = 500$ tokens per trajectory.
- **Execution Mode Flag (`SMOKE`)**:
  - Set `SMOKE = True` (or environment variable `BELIEF_SMOKE=1`) for a rapid 3-minute validation run across 2 models at 5M tokens.
  - Set `SMOKE = False` for the full 270-model production run (400M tokens/model).

### Cell 3 & 4: Phase 1 — Physics Screening & 10-Tuple Distillation (Code & Markdown)
Isolates 10 optimal contractive physical parameter configurations:
1. **Sweep 1A**: Evaluates 20 candidate pairs $(\Delta v \times n)$ with $\Delta v \in \{0.6, 0.9, 1.2, 1.5, 2.0\}$ and $n \in \{5, 10, 20, 25\}$ ($m = 500 // n$) under contractive baseline damping $\gamma_1 = \gamma_2 = 0.65$.
2. **Distillation Checkpoint 1A**:
   - **Interactive Override**: `CHOSEN_PAIR_IDS = [2, 5, 6, 8, ...]` allows manual selection.
   - **Headless Fallback**: If `None`, automatically ranks and selects top candidate pairs based on composite stability and Bayes gap score:
     $$\text{Score} = \text{Bayes Gap} - 0.5 \times \max(0, \lambda) - 10.0 \times \text{Clipped}$$
3. **Energy-Balance Step**:
   Computes the steady-state damping $\gamma^*$ balancing impulse injection via `compute_optimal_gamma(...)`. Generates 9 candidate gammas per pair:
   $$\gamma \in \{k \cdot \gamma^*, \quad k \cdot \gamma^* \pm 0.05 \quad \text{for } k \in \{0.03, 0.07, 0.10\}\}$$
4. **Sweep 1B**: Screens the Cartesian product of chosen pairs $\times$ 9 candidate gammas ($90$ physics runs).
5. **Distillation Checkpoint 2**:
   - Selects the top 10 `ExperimentConfig` physics tuples.
   - Exports authoritative Level 1 artifacts:
     - `results/distilled_physics_tuples.csv`
     - `results/distilled_physics_tuples.json`

### Cell 5 & 6: Phase 2 — The 270-Model Production Queue (Code & Markdown)
Constructs and dispatches the main production queue:
$$\text{Total Models} = 10\text{ physics tuples} \times 3\text{ widths } (32, 64, 128) \times 3\text{ lookaheads } (k \in \{1, \lfloor n/2 \rfloor, n\}) \times 3\text{ seeds } (0, 1, 2) = \mathbf{270\text{ models}}$$

- **Locked Batch Size Invariant**: `BATCH_SIZE = 1024` sequences per step ($512,000$ tokens/step, 781 gradient steps) with micro-batching (256 on GPU, 16 on CPU) and gradient accumulation.
- **Real-Time Step Loss Logging**:
  Every 50 steps, the worker prints live telemetry:
  ```text
  [dp_dv1.2_g0.61_n10_dmodel_128_dmlp_512_k_mid_seed0] Step 100/781 (12.8%) | Tokens:  51.2M / 400.0M | Loss: 2.1450 | LR: 9.85e-04
  ```
- **In-Worker Synchronous Probing**:
  Directly upon reaching 400M tokens, the worker evaluates 4 representation views $\times$ 2 targets against matched `random_init` controls:
  1. `all_layers_single_token`: Residual streams concatenated across all layers at cycle boundaries.
  2. `single_layer_cycle`: Last layer residual streams across all $n$ tokens within each cycle.
  3. `single_layer_last_token`: Last layer residual stream at cycle boundaries. **Cycle 0 ($t=n-1$) is strictly excluded**, evaluating from Cycle 1 onwards to ensure belief state synchronization.
  4. `single_layer_all_token`: All individual sequence tokens from the last layer probed independently.
- **Probe Headline Verdict**:
  Immediately prints headline probe scores:
  ```text
  [model_id] Probe Results -> Max Belief R²: +0.812 | Max Physics R²: +0.895 | Gain over Random: +0.584
  [model_id] Finished in 142.1s. Packaged -> model_id.zip
  ```
- **Lookahead Probability Map**:
  Generates and saves `{model_id}_lookahead_prob_map.png` evaluated on a fixed held-out sequence (seed $2026$).
- **Instant Atomic Packaging**:
  Compresses `{model_id}/` into `{model_id}.zip` immediately upon completion.
- **Incremental Master Manifest Flush**:
  Updates `{OUTPUT_BASE}/master_probe_results.csv` and `{OUTPUT_BASE}/pipeline_state.json` after every completed model.
- **Dual-Path Auto-Resume**:
  Bypasses any model whose `.zip` and `_probes.csv` already exist on disk.

### Cell 7 & 8: Phase 3 — Emergence Synthesis & 5-Panel Dashboard (Code & Markdown)
Consolidates all completed model evaluations and generates 5 synthesis figures in `results/`:
1. **`scaling_curves.png`**: Model width ($d_{\text{model}} \in \{32, 64, 128\}$) vs. Test $R^2$ for Trained vs. Random controls across Belief and Physics.
2. **`lookahead_comparison.png`**: Forecasting horizon comparison ($k=1$ vs. $k=\lfloor n/2 \rfloor$ vs. $k=n$).
3. **`probe_views_breakdown.png`**: Linear readability comparison across the 4 representation views.
4. **`physics_vs_belief_scatter.png`**: Representation decoupling scatter plot ($R^2_{\text{belief}}$ vs. $R^2_{\text{physics}}$) featuring:
   - Distinct colors per model
   - Distinct marker shapes per probe view (`o` for all layers, `s` for cycle tokens, `^` for last token, `D` for all tokens)
   - Zero reference crosshairs ($x=0, y=0$)
5. **`trained_vs_random_gain.png`**: Net emergence gain (`trained_minus_random`) isolating non-trivial learned structure.
- **Master Distribution Package**: Consolidates all results into `{OUTPUT_BASE}/double_pendulum_results.zip`.
- **Inline Visualization**: Displays all 5 dashboard figures and sample lookahead probability maps inline in the notebook.

### Cell 9 & 10: Phase 4 — Re-entrant Model Queue Framework (Code & Markdown)
Provides an interface for continuing training on top-performing models without restarting from scratch:
```python
# Example: Continue training top 5 belief models for an additional 500M tokens
from scripts.production_train import requeue_top_models

requeue_top_models(master_probe_df, metric="trained_r2", target="belief", top_n=5, additional_tokens=500_000_000)
```
- By default, `REQUEUE_JOBS = []` is dormant, allowing standard 'Run All' runs to terminate cleanly after Phase 3.

### Cell 11: Artifact Inventory & Download Guide (Markdown)
Details file paths, manifest contents, and instructions for continuing across multi-session Kaggle environments.

---

## 3. How to Check Intermediate Results During Execution

You do not need to wait for all 270 models to finish. You can verify health in real time:

### A. Live Step Loss Criteria
- **Initial Step**: Loss starts around $\ln(2500) \approx 7.82\text{ nats}$ (uniform prior over $50 \times 50$ 2D token grid).
- **Early Convergence**: Loss drops steeply within the first 100–200 steps to $\approx 2.0 - 2.5\text{ nats}$.
- If loss is steadily descending and not NaN, the Adam optimizer and continuous physics data generator are functioning properly.

### B. In-Worker Probe Criteria
- **Belief $R^2 > 0.50$**: Indicates hidden Markovian belief states are linearly accessible.
- **Gain over Random $> +0.30$**: Confirms linear readability emerged from training dynamics rather than random projection artifacts.

### C. Lookahead Probability Maps
- Open any generated `{model_id}_lookahead_prob_map.png`.
- The cyan actual trajectory should trace cleanly along a high-probability ridge, showing distinct steps without horizontal plateauing.

### D. Live Inspection from another Notebook Cell
Run this snippet in any scratch cell while training is active:
```python
import pandas as pd
from IPython.display import Image, display

# 1. View summary of completed models:
master_csv = OUTPUT_BASE / "master_probe_results.csv"
if master_csv.exists():
    df = pd.read_csv(master_csv)
    display(df.groupby(["model_id", "target"])[["trained_r2", "trained_minus_random"]].max())

# 2. View latest probability map:
prob_maps = sorted(RESULTS_DIR.glob("*/*_lookahead_prob_map.png"))
if prob_maps:
    display(Image(filename=str(prob_maps[-1])))
```

---

## 4. Multi-Session Kaggle Continuation Guide

Because Kaggle GPU sessions are capped at 12 hours and the dual-T4 pipeline requires ~61 hours total, the run naturally spans ~5 sessions:

1. **Session 1**: Run the notebook. As models finish, individual `{model_id}.zip` files and `{OUTPUT_BASE}/pipeline_state.json` are written to `/kaggle/working/`.
2. **At Session Expiry**:
   - Go to Kaggle Notebook Output $\to$ Click **"New Dataset"** from Output $\to$ Name it e.g. `double-pendulum-session1`.
3. **Session 2 Onward**:
   - In the Kaggle notebook sidebar, click **"Add Input"** $\to$ Add your previous session's dataset.
   - Run Cell 2: The notebook automatically detects previous outputs in `/kaggle/input`, restores `pipeline_state.json`, and resumes from the first pending model.
4. **Final Session**:
   - The last session finishes the remaining models, runs Phase 3 Emergence Synthesis, and packages `{OUTPUT_BASE}/double_pendulum_results.zip`.

---

## 5. Running from CLI / Vast.ai Workstation

For headless execution on a dedicated RTX 5090 workstation:
```bash
# Full 270-model production run with 5 concurrent workers:
uv run python scripts/production_train.py --platform rtx5090 --tokens 400000000 --out-dir ./experiments/production

# Fast 3-minute smoke validation test:
uv run python scripts/production_train.py --smoke --platform local_cpu --out-dir /tmp/prod_smoke
```
