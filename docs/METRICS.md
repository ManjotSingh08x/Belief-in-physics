# Experiment and metrics

What each quantity is and how it is computed. No results here.

## Pipeline

| script | does |
|---|---|
| `03_train_branch.py` | trains one transformer per system, 500M tokens, snapshots at 9 log-spaced token counts including 0 |
| `04_probe_branch.py` | probes the final model, trained vs random-init |
| `06_analyse_checkpoints.py` | probes every checkpoint at every depth; erases each quantity and measures the loss cost |
| `make_analysis_figures.py` | the four figures in `figures/analysis/` |

Model: 4 layers, 1 head, `d_model=128`, `d_mlp=512`, `lr=1e-3`, cosine schedule over the full 500M.
A checkpoint is a snapshot part-way along one run, not a model trained to that budget and annealed.

## Probe targets

`b` is the exact posterior over branches `(z0, w_1..w_m)`. Every target is a marginal of `b`, so
every one is exactly linear in it. One ridge probe per depth predicts all columns at once;
R² is scored per column and averaged within a block, so a large-scale block cannot mask a
failure elsewhere.

| block | width | definition |
|---|---|---|
| `action_lag0` | A | `P(most recent perturbation)` -- the simplex point |
| `action_lag1/2` | A | same, one and two segments back |
| `z0` | 4 | `P(initial condition)` |
| `metric` | 1-2 | `E_b[m] = b @ metric_table` |

`metric` per system: pendulum `omega`; double_pendulum `(omega1, omega2)`; sphere tangential
velocity `(u, cos(phi) w)`; predator_prey `(dx/dt, dx/dt / dy/dt)` clipped to `+/-20`.

## Metrics

**R²** -- held-out, sequence-level 70/30 split (whole sequences, never positions, so no
position leaks between train and test).

**Shuffled control** -- refit after permuting the context-to-target correspondence. Must
collapse to ~0, else the probe is doing the modelling rather than reading it off.

**Untrained baseline** -- the same architecture at the same seed, probed before any training.
A quantity counts as learned only if the trained R² exceeds it; a random projection of the
residual stream already predicts some targets well.

**Best depth** -- the depth maximising `action_lag0` R². Ablation and geometry are computed
there, so the causal numbers describe one place in the network.

**Erasure rank** -- iterative nullspace projection (Ravfogel et al. 2020): fit a probe, delete
its row space, refit, repeat until R² falls below `max(0.02, 0.1 x intact)`. One pass is not
enough -- a feature is stored redundantly, so deleting the readout directions leaves copies a
refit recovers. The rank required counts how many directions the feature is spread over.
Capped at 48 of 128; `hit_rank_cap` marks an erasure that stopped early and is therefore
incomplete.

**Δ loss** -- next-token cross entropy after mean-ablating the erasure basis at the best depth,
minus the unablated loss. Mean rather than zero so the stream stays on-distribution.

**Excess Δ loss** -- Δ loss minus the mean Δ loss of `N_CONTROL=5` random subspaces of the same
rank, reported with the control's standard deviation. Deleting any `r` directions costs
something, so only the excess is interpretable. Positive means load-bearing. On an untrained
model ablation *lowers* loss (near-uniform logits, deleting variance removes noise), so a
negative value early is expected.

**Principal angles** -- between two erasure bases, in degrees. Compared against the mean angle
between random subspaces of the same ranks, because two rank-40 subspaces in 128 dimensions
already overlap substantially; that null, not 90 degrees, is the independence line.

**Cross-R²** -- refit every block after erasing one. Measures how much of block `h` lived in
block `g`'s subspace.
