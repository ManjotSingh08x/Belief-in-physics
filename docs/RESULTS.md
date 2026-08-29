# Mess-4 results

All four seed-0 models completed 500M-token training and all CPU analyses completed on held-out sequence splits.
The exact target was the same four-state predictive-belief tetrahedron for every physical system.

## Headline

The transformers learned the physical dynamics, but the results do not support a dedicated belief-state representation beyond recent observations.
Pendulum and predator-prey developed belief decodability above random initialization, while sphere did not clear the preregistered `+0.05` baseline margin and double pendulum learned no belief signal.
For every system, some raw observation-token window matched or exceeded the validation-selected residual-stream belief probe.
The predicted geometry remained a compressed central cloud rather than reconstructing the exact fractal tetrahedron.

## Training

| System | Final eval loss | Uniform-token loss | T4 time |
|---|---:|---:|---:|
| Pendulum | `0.385` | `5.198` | `23.3 min` |
| Predator-prey | `0.314` | `5.198` | `24.7 min` |
| Sphere | `0.449` | `5.198` | `24.2 min` |
| Double pendulum | `0.378` | `5.198` | `27.8 min` |

All runs used exactly `499,998,720` generated tokens, seed `0`, and eight checkpoints.

## Final probes

Layer selection used validation sequences and the values below are from the untouched test split.

| System | Belief R2 | Random-init R2 | Gain | Physical-metric R2 | Best belief depth | Baseline criterion |
|---|---:|---:|---:|---:|---|---|
| Pendulum | `0.145` | `0.065` | `+0.081` | `0.246` | after block 2 | pass |
| Predator-prey | `0.176` | `0.049` | `+0.127` | `0.608` | after block 3 | pass |
| Sphere | `0.094` | `0.048` | `+0.046` | `0.699` | after block 1 | fail |
| Double pendulum | `0.009` | `0.015` | `-0.006` | `0.127` | after block 2 | fail |

Shuffled-target R2 was approximately `0.003` for every system.
True-mood R2 remained approximately zero, so the probes did not recover the hidden state itself.

## Myopic control

This table uses the strongest raw-token window for each system.
A negative excess means raw recent observations predicted the exact belief better than the transformer residual stream.

| System | Best window | Raw-window belief R2 | Residual belief R2 | Residual excess |
|---|---:|---:|---:|---:|
| Pendulum | `16` tokens | `0.279` | `0.125` | `-0.155` |
| Predator-prey | `8` tokens | `0.202` | `0.164` | `-0.038` |
| Sphere | `32` tokens | `0.174` | `0.076` | `-0.098` |
| Double pendulum | `16` tokens | `0.026` | `0.017` | `-0.009` |

Therefore none of the four systems provides evidence that the residual stream contains more belief information than an appropriately controlled recent-token baseline.

## Emergence

A crossing is reported only when learned gain over initialization exceeds `0.05`.
The token count is where the target reached `75%` of its own final learned gain.

| System | Belief gain | Belief crossing | Metric gain | Metric crossing |
|---|---:|---:|---:|---:|
| Pendulum | `+0.086` | `4.32M` | `+0.227` | `8.24M` |
| Predator-prey | `+0.117` | `64.46M` | `+0.417` | `11.57M` |
| Sphere | `+0.039` | none | `+0.301` | `11.03M` |
| Double pendulum | `-0.001` | none | `+0.106` | `9.09M` |

No system developed a meaningful true-mood gain.

## Geometry

The exact tetrahedral target is [`figures/messk/mess4_belief_geometry.png`](../figures/messk/mess4_belief_geometry.png).
The clearer square projection is [`figures/messk/mess4_belief_geometry_planar.png`](../figures/messk/mess4_belief_geometry_planar.png).
The square view is deliberately lossy and is not used for any score.

Checkpoint-by-layer tetrahedral predictions are:

- [Pendulum](../figures/transformer-belief/pendulum_mess4_checkpoint_layers.png)
- [Predator-prey](../figures/transformer-belief/predator_prey_mess4_checkpoint_layers.png)
- [Sphere](../figures/transformer-belief/sphere_mess4_checkpoint_layers.png)
- [Double pendulum](../figures/transformer-belief/double_pendulum_mess4_checkpoint_layers.png)

Planar square projections are:

- [Pendulum](../figures/transformer-belief/pendulum_mess4_checkpoint_layers_planar.png)
- [Predator-prey](../figures/transformer-belief/predator_prey_mess4_checkpoint_layers_planar.png)
- [Sphere](../figures/transformer-belief/sphere_mess4_checkpoint_layers_planar.png)
- [Double pendulum](../figures/transformer-belief/double_pendulum_mess4_checkpoint_layers_planar.png)

Pendulum and predator-prey clouds expand and separate modestly with training but remain far from the exact fractal support.
Sphere shows only weak belief improvement despite strong physical-metric recovery.
Double-pendulum geometry remains at initialization-level quality throughout training.

## Interpretation boundary

These are single-seed results and therefore do not estimate run-to-run variance.
They establish a controlled negative result for the current seed and architecture: next-token training learns useful physical state variables, but apparent belief decodability is explainable by recent observation history.

Raw machine-readable results are in [`experiments/results-messk/`](../experiments/results-messk/).
