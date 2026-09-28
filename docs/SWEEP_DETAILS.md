# Belief-in-Physics Sweep Plan

The experiment is split into two notebooks:

1. `notebooks/explorer.ipynb` screens physics configurations without training a transformer.
2. `notebooks/three_phase_pipeline.ipynb` builds the training manifest, trains, sanity-probes, and packages each run.

## Parameter meanings

- `dt`: physics sampling gap between emitted tokens. The integrator uses `integration_dt=0.01`, so `dt=0.05` performs five internal physics steps per token.
- `n`: impulse spacing, in sampled tokens.
- `m`: impulse cycles per trajectory. Sequence length is exactly `m * n`.
- `delta_v`: impulse magnitude.
- damping: `gamma` for pendulum/sphere, `gamma1=gamma2` for double pendulum, and effective damping `1/kappa` for predator-prey.
- `alpha`: HMM emission fidelity.

`dt` is fixed by system and is not a sweep axis:

| system | dt | internal steps/token |
|---|---:|---:|
| pendulum | 0.02 | 2 |
| predator-prey | 0.05 | 5 |
| sphere | 0.20 | 20 |
| double pendulum | 0.01 | 1 |

Mechanical angular velocities are unbounded during screening. Predator-prey keeps its finite log-population bound because exponentiating unbounded log states is numerically unsafe.

## Explorer cascade

For each system:

1. Screen a 4 x 4 `delta_v` by damping grid across 10 seeds.
2. Rank stable cells and keep six base candidates.
3. Expand each base into five conditions:
   - base `n`, base damping;
   - `n` up, damping down, constant `n * damping`;
   - `n` down, damping up, constant `n * damping`;
   - `n` up, damping constant;
   - `n` down, damping constant.
4. This yields 30 candidates per system.
5. Keep 15 stable finalists, balanced at three per condition where possible.

For predator-prey, constant-product comparisons use effective damping `1/kappa`; therefore decreasing damping means increasing `kappa`.

### Initial grids

| system | delta_v | raw damping parameter |
|---|---|---|
| pendulum | 0.30, 0.55, 0.80, 1.10 | gamma: 0.40, 0.80, 1.20, 2.00 |
| predator-prey | 0.15, 0.35, 0.60, 0.85 | kappa: 3, 5, 8, 12 |
| sphere | 0.08, 0.15, 0.25, 0.40 | gamma: 0.15, 0.35, 0.70, 1.20 |
| double pendulum | 0.50, 1.00, 1.50, 2.00 | gamma1=gamma2: 0.50, 1.50, 2.50, 4.00 |

### Stability evidence

Each candidate records:

- finite trajectory check;
- Lyapunov exponent;
- observation clipping;
- state safety-bound hits;
- used token bins;
- driven-versus-free causal gap;
- positional token entropy (`bayes_gap`);
- HMM synchronization curve and length.

A grid cell is `GO` when at least 80% of its 10 seeds pass and mean `bayes_gap >= 0.15` nats. Any candidate relying on a state safety bound for more than 0.1% of samples fails the sweep.

The committed screen produced 15 finalists for each of the four systems. The exact 60 tuples are embedded in `SCREENED_CANDIDATES` in the training notebook and can be regenerated in the explorer. The Sphere finalists were originally selected at `dt=0.04`; all 15 were rechecked with the same 10-seed stability screen at `dt=0.20` and remained GO (14 passed 10/10 seeds, one passed 9/10). Their ranking at the new sampling gap has not been recomputed.

## Training manifest

The base manifest is:

```
15 physics candidates/system
* 4 systems
* 4 alpha values (0.50, 0.70, 0.85, 0.95)
* 3 seeds (0, 1, 2)
= 720 runs
```

The default residual-stream dimension is 64 so this count remains 720. Setting `BELIEF_SWEEP_RESIDUAL_DIMS=1` sweeps `[32, 64, 128]`, producing 2160 runs. Batch size is 1024 and the full-token budget is 2 billion per run; smoke mode uses a small bounded run.

The notebook executes only the first manifest row by default. Set `BELIEF_RUN_FULL_SWEEP=1` on the intended compute job to activate the complete manifest.

`BELIEF_K_MODE` selects the forecast horizon per physics candidate without changing the slice order: `fixed` (default, `TrainingConfig.k=1`), `half` (`k=max(1,n//2)`, so `n=5` uses `k=2`), or `n_plus_one` (`k=n+1`). Each mode is a separate 720-run manifest, or 60 Sphere seed-0 d64 runs after the Sphere/seed filters. Non-default runs include the resolved `k` in their artifact names, and the Kaggle packager gives each mode a distinct notebook slug. No lookahead sweep is activated by default.

## Per-run artifacts

Every completed run writes exactly these primary files:

- `.pth`: final transformer state dictionary;
- `.json`: complete experiment, physics, HMM, model, training, loss, and probe metadata;
- `.csv`: trained and random-baseline ridge probe results.

Phase 3 packages those three files into one same-named `.zip` for Lightning storage and verifies the archive member list. A consolidated `probe_results_all.csv` is also written for cross-run analysis.

## Probe protocol

The frozen trained transformer and an architecture-matched random initialization are both probed for physics and belief targets. Ridge strength is selected with grouped cross-validation, with whole trajectories held out. Four feature views are evaluated:

1. single layer, single token;
2. all layers concatenated, single token;
3. single layer, all `n` tokens in an impulse cycle concatenated;
4. all layers and all `n` cycle tokens concatenated.

The notebook is a sanity analysis. Publication analysis should load the zipped saved runs in a separate notebook.
