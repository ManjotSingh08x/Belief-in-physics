# Belief-in-Physics: Sweep Execution Plan

## The Problem

We need to train transformers across multiple physics systems and HMM configurations to answer the core research question: **does the transformer's internal representation scale with belief-tracking difficulty?**

Naive combinatorial sweep = 800+ training runs. This plan uses CPU-only pre-flight vetting to cut that to **~28 runs** with zero wasted GPU time.

---

## Decisions Locked (from /grill-me)

| Decision | Choice |
|---|---|
| Physics grid | System-specific (Δv, γ) ranges per system |
| Observation channels | Fixed per system (not swept) |
| HMM sweep axis | Alpha (emission fidelity) only, stay fixed at 0.7 |
| Alpha values | {0.5, 0.85, 0.95} + baseline 0.7 |
| Sequence length m | Scaled per alpha via empirical sync-length measurement |
| n-dilution ablation | Yes, best 2 systems, n ∈ {5, 20} with energy compensation |
| Multi-seed | 3 seeds on best config per system (all 4 systems) |
| Pre-flight checks | Lyapunov + clipping + used bins + gap mean + energy drift + Bayes gap + sync curve |
| Pre-flight code location | `experiments/00_preflight.py` |
| Stability extensions | Add Bayes gap + sync curve into `V.stability()` |

---

## Pipeline Overview

```
Stage 0: Lock dt per system           (0 runs, 1 min CPU)
    │
Stage 0a: Screen Δv × γ grid          (0 runs, ~5 min CPU per system)
    │
Stage 0b: Recalibrate n               (0 runs, ~1 min CPU per system)
    │
Stage 1: Baseline training             (4 runs)
    │
Stage 2: Alpha sweep                   (12 runs)
    │
Stage 3: n-dilution ablation           (4 runs)
    │
Stage 4: Multi-seed validation          (8 runs)

Total: 28 transformer training runs
```

---

## Stage 0: Physics Pre-Flight (0 Training Runs)

**Goal:** Find one clean (Δv, γ) operating point per system. Pure CPU, no transformer.

### Step 0.0: Lock dt per system

dt is an integrator accuracy knob, not a scientific axis. Pick smallest value where RK4 energy conservation holds (compare dt vs dt/2 integration on free trajectory, require < 0.1% energy drift over 1000 steps).

Current committed values (already validated by existing tests):

| System | dt | Why |
|---|---|---|
| Pendulum | 0.02 | Simple ODE, stable at large steps |
| Predator-Prey | 0.05 | Log-space populations, moderate stiffness |
| Sphere | 0.04 | Angular coordinates, moderate dynamics |
| Double Pendulum | 0.01 | Chaotic, needs small dt for RK4 stability |

**Action:** Confirm these pass energy-conservation check. Fixed forever after.

### Step 0a: Screen Δv × γ Grid (3 × 3 per system)

Each system gets its own grid, scaled to natural units:

| System | Δv candidates | γ candidates |
|---|---|---|
| Pendulum | {0.3, 0.55, 0.8} | {1.5, 2.5, 4.0} |
| Predator-Prey | {0.15, 0.35, 0.6} | {1.0, 2.0, 3.5} |
| Sphere | {0.08, 0.15, 0.25} | {1.5, 3.0, 5.0} |
| Double Pendulum | {0.5, 1.2, 2.0} | {1.5, 2.5, 4.0} |

> [!IMPORTANT]
> These grid values are initial guesses based on existing committed defaults and natural unit scales. Need your review — you know the physics better. Adjust before running.

**Per candidate config, run these 7 checks (all CPU, ~2 sec each):**

#### Check 1: Lyapunov Divergence (already in `V.stability()`)
- Twin-trajectory test: perturb z0 by ε = 1e-8, replay same letters
- **Pass:** λ < 0.05 /s
- **Fail:** Physical chaos, trajectory forgets initial conditions

#### Check 2: Observation Clipping (already in `V.stability()`)
- Fraction of observation values hitting bin boundary
- **Pass:** clipped < 1.0%
- **Fail:** Energy too high, observable saturates at extremes

#### Check 3: Used Bins (already in `V.stability()`)
- Fraction of vocabulary actually emitted
- **Pass:** used_bins > 10% of n_obs
- **Fail:** Observable stuck in narrow range, most tokens never seen

#### Check 4: Driven-vs-Free Gap (already in `V.stability()`)
- Mean distance between kicked and unkicked trajectories
- **Pass:** gap_free_mean > 1e-3
- **Fail:** Kicks leave no visible trace in observations

#### Check 5: Energy Drift (already in `V.stability()`)
- Free-trajectory energy change over full rollout
- **Pass:** |drift| < 1%
- **Fail:** Integrator leaking or gaining energy (dt too large)

#### Check 6: Bayes-Optimal Gap (NEW — add to `V.stability()`)
- Compare cross-entropy of tokens given exact letter sequence vs. marginal prior (no letter knowledge)
- Computed by: replay deterministic trajectory for each of K^m letter sequences (or Monte Carlo sample ~1000 letter sequences), measure average log-likelihood gap
- **Pass:** gap > 0.15 nats
- **Fail:** Tokens carry almost no information about which letter was active — kicks too weak relative to natural dynamics

#### Check 7: Forward-Algorithm Sync Curve (NEW — add to `V.stability()`)
- Total-variation distance between belief-from-uniform-prior and belief-from-true-prior, plotted vs number of letters observed
- Uses existing `MessKProcess.beliefs()` machinery
- **Output:** sync_length = number of letters until TV < 0.01 (same as `memory_length()` but returned as curve, not scalar)
- **Purpose:** Provides empirical m requirement for Stage 2 alpha sweep. Not a pass/fail gate here — informational.

**Output of Stage 0a:** A 3×3 table per system with pass/fail verdicts. Select **top 4 candidates** per system, ranked by Bayes gap among all survivors (all 6 go/no-go checks passed). If fewer than 4 pass, take all survivors and expand the grid.

### Step 0b: Recalibrate n (0 runs)

For each of the 4 surviving candidates per system, test n ∈ {5, 10, 15, 20} with dt and (Δv, γ) held fixed. Rerun same 7 checks.

- If default n = 10 passes: keep it.
- If n = 10 fails (e.g. clipping): lower to n = 5, recheck.
- If n = 10 shows very low Bayes gap: raise to n = 15 or 20.

**Output:** 4 locked (Δv, γ, dt, n) candidates per system, each confirmed to pass all checks. These 16 total configs (4 systems × 4 candidates) go to Stage 1.

---

## Stage 1: Baseline Training + Physics Config Selection (16 Runs)

**Goal:** Train all 4 candidate (Δv, γ) configs per system at standard HMM parameters. Use actual probe R² to pick the single best physics config per system.

| Runs | System | Candidates | Alpha | Stay | m | Seed |
|---|---|---|---|---|---|---|
| 1-4 | Pendulum | 4 (Δv, γ) combos | 0.7 | 0.7 | 16 | 0 |
| 5-8 | Predator-Prey | 4 (Δv, γ) combos | 0.7 | 0.7 | 16 | 0 |
| 9-12 | Sphere | 4 (Δv, γ) combos | 0.7 | 0.7 | 16 | 0 |
| 13-16 | Double Pendulum | 4 (Δv, γ) combos | 0.7 | 0.7 | 16 | 0 |

Each run:
1. Train transformer on streaming physics batches
2. Save model weights + JSON sidecar to `OUTPUT_DIR`
3. Run Phase 3 ridge probing (physics R² and belief R²)
4. Save probe results CSV + JSON alongside model

**Narrowing step:** After all 16 runs complete, pick **1 winner per system** = the (Δv, γ) candidate with highest belief probe R² (trained minus random-init). That winner's physics config is locked for Stages 2-4.

**Success gate:** At least 1 candidate per system shows trained probe R² > random-init probe R² for belief target.

---

## Stage 2: Alpha Sweep (12 Runs)

**Goal:** Core scientific contribution. Test whether transformer representations scale with emission ambiguity.

### Pre-step: Compute empirical sync-length per alpha

Before training, run `MessKProcess(n_states=4, alpha=alpha, stay=0.7).memory_length()` plus the full TV-distance curve from Stage 0a Check 7 for each alpha:

| Alpha | Expected Sync Length (letters) | m (= 2 × sync length, rounded up) |
|---|---|---|
| 0.50 | Measure empirically | 2 × measured |
| 0.70 | ~7 (known) | 16 (baseline) |
| 0.85 | Measure empirically | 2 × measured |
| 0.95 | Measure empirically | 2 × measured |

> [!IMPORTANT]
> At alpha = 0.5, emissions are very noisy. Sync length will be longer than at alpha = 0.7. At alpha = 0.95, emissions are nearly deterministic, sync length will be much shorter. m must scale accordingly — training a model at alpha = 0.5 with m = 16 may not give the belief enough context to synchronize.

### Training Matrix

| Run | System | Alpha | m (empirical) | Seed |
|---|---|---|---|---|
| 5-7 | Pendulum | 0.5, 0.85, 0.95 | per-alpha | 0 |
| 8-10 | Predator-Prey | 0.5, 0.85, 0.95 | per-alpha | 0 |
| 11-13 | Sphere | 0.5, 0.85, 0.95 | per-alpha | 0 |
| 14-16 | Double Pendulum | 0.5, 0.85, 0.95 | per-alpha | 0 |

Each run: same pipeline as Stage 1 (train + probe + save).

**Output:** Headline figure: Belief Probe R² vs Alpha, one curve per system.

---

## Stage 3: n-Dilution Ablation (4 Runs)

**Goal:** Test whether information dilution (more tokens per kick cycle) degrades belief tracking, holding physics energy constant.

### Energy Compensation Rule

When changing n from baseline n₀ = 10 to n_new, adjust γ:

```
γ_new = γ_baseline × (n₀ / n_new)
```

This holds `exp(-γ × n × dt)` constant, so dissipated energy per tick is identical.

Example for Pendulum (γ_baseline = 2.5):
- n = 5: γ = 2.5 × (10/5) = 5.0
- n = 10: γ = 2.5 (baseline)
- n = 20: γ = 2.5 × (10/20) = 1.25

> [!WARNING]
> Before training, re-run Stage 0a checks (especially clipping and Bayes gap) on each energy-compensated config. Compensation keeps average energy constant but changes transient dynamics — a config can still fail checks.

### Training Matrix (Best 2 Systems)

Systems chosen after Stage 2 results (pick one stable + one chaotic/marginal).

| Run | System | n | γ (compensated) | Seed |
|---|---|---|---|---|
| 17 | System A | 5 | γ_baseline × 2 | 0 |
| 18 | System A | 20 | γ_baseline × 0.5 | 0 |
| 19 | System B | 5 | γ_baseline × 2 | 0 |
| 20 | System B | 20 | γ_baseline × 0.5 | 0 |

Baseline n = 10 already trained in Stage 1. Three data points per system: n ∈ {5, 10, 20}.

**Output:** Belief Probe R² vs n (information dilution), physics held constant.

---

## Stage 4: Multi-Seed Validation (8 Runs)

**Goal:** Error bars for publication.

Best config per system (chosen from Stage 2 results), 2 additional seeds:

| Run | System | Config | Seed |
|---|---|---|---|
| 21-22 | Pendulum | Best alpha | 1, 2 |
| 23-24 | Predator-Prey | Best alpha | 1, 2 |
| 25-26 | Sphere | Best alpha | 1, 2 |
| 27-28 | Double Pendulum | Best alpha | 1, 2 |

**Output:** Mean ± standard error for probe R², trained-minus-random delta, loss curves.

---

## What Gets Saved (File Structure)

All outputs under `experiments/outputs-three-phase/`:

```
experiments/outputs-three-phase/
├── preflight/
│   ├── preflight_results.json          # Full Stage 0 grid results
│   ├── sync_curves.json                # TV-distance curves per (stay, alpha)
│   └── winning_configs.json            # Locked (Δv, γ, dt, n) per system
│
├── {exp_name}_{train_name}_seed{N}_final.pt       # Model weights
├── {exp_name}_{train_name}_seed{N}_final.json     # Training metadata sidecar
├── {exp_name}_{train_name}_seed{N}_probes.csv     # Per-run probe results
├── {exp_name}_{train_name}_seed{N}_probes.json    # Per-run probe metadata
│
└── probe_results_all.csv              # Master table across all runs
```

---

## Code Changes Required

### 1. Extend `physics/visualise.py::stability()`
Add two new fields to the returned dict:
- `bayes_gap`: float — cross-entropy gap (with letter knowledge vs without)
- `sync_curve`: list of (letter_count, tv_distance) pairs

### 2. New script: `experiments/00_preflight.py`
- Takes system-specific (Δv, γ) grid as input
- Calls `V.trace()` + `V.stability()` per candidate
- Outputs structured JSON with pass/fail verdicts
- Prints human-readable summary table

### 3. Notebook `three_phase_pipeline.ipynb` (already partially done)
- `EXPERIMENT_CONFIGS` is a list (done)
- `TRAINING_CONFIGS` is a list (already was)
- Phase 2 loops over all experiment × training combos (done)
- Phase 3 probes and saves per-run (done)

---

## Run Budget Summary

| Stage | GPU Runs | CPU-Only Work |
|---|:---:|---|
| Stage 0 (dt lock) | 0 | Energy conservation check, 4 systems |
| Stage 0a (Δv × γ screen) | 0 | 36 candidate configs, 7 checks each |
| Stage 0b (n recalibration) | 0 | ~64 configs (4 n values × 16 candidates) |
| Stage 1 (Baseline + selection) | **16** | Narrow to 1 winner per system |
| Stage 2 (Alpha sweep) | **12** | Sync-length measurement per alpha |
| Stage 3 (n-dilution) | **4** | Energy-compensated check per config |
| Stage 4 (Multi-seed) | **8** | — |
| **TOTAL** | **40** | All pre-flight < 30 min total on CPU |

---

## Executive Summary

**The research question:** Does a transformer's internal representation of hidden Markov state scale with belief-tracking difficulty?

**The experimental strategy:** We test this across 4 physical systems (pendulum, predator-prey, sphere, double pendulum) by varying emission fidelity alpha — the knob that controls how ambiguous each observation token is about the hidden state. Higher alpha = easy (nearly deterministic emissions), lower alpha = hard (noisy emissions, transformer must integrate longer context to infer belief).

**How we avoid wasting compute:** Before training any transformer, we run cheap CPU-only physics checks on every candidate configuration. 7 automated checks verify the physics is stable (not chaotic), the discretization is clean (tokens use a reasonable fraction of the vocabulary without clipping), and the kicks actually leave a detectable trace in the observations (Bayes gap). From a 3×3 grid per system (36 total candidates), only the top 4 per system that pass all checks proceed to training.

**What we actually train:** 40 transformers total. 16 baseline runs (4 candidate physics configs × 4 systems) to GPU-select the best operating point per system. 12 alpha-sweep models (3 new alpha values × 4 systems) to answer the core scientific question. 4 information-dilution ablations (testing whether more tokens between kicks degrades tracking). 8 multi-seed repeats for publication-grade error bars.

**What we measure:** Ridge regression probes on frozen transformer residual streams, predicting either the exact HMM belief or the physical metric. Trained models are compared against architecture-matched random-initialization baselines. The headline result is a plot of Belief Probe R² vs Alpha across all 4 physical systems.
