# Mess-4 results

All four seed-0 models completed 500M-token training and all CPU analyses completed on held-out sequence splits.
The exact target was the same four-state predictive-belief tetrahedron for every physical system.

## Headline

The transformers learned the physical dynamics well.
Whether they carry belief information beyond recent observations depends on how the residual stream is read, and the two readouts disagree in sign.
Reading one layer at one token position, every system's raw-token control matched or beat the probe.
Reading all four blocks across a whole HMM tick, every system's probe beats its raw-token control, by `+0.03` to `+0.12`.
The honest summary is a small positive excess that is readout-sensitive at a single seed, not the clean negative result the first pass suggested.
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

Therefore, **at this readout**, none of the four systems provides evidence that the residual stream contains more belief information than an appropriately controlled recent-token baseline.
The readout matters, and the next section shows it changes the sign of this conclusion.

## What is actually reachable

`07_ceiling.py` computes the exact upper bound rather than estimating it.
Each system is deterministic given its letters - fixed initial state, no process noise - so the posterior over letter sequences is obtained by replaying every candidate prefix and keeping the ones whose tokens match the observation exactly.
Summing the surviving forward vectors and normalising gives `E[belief | tokens]` with no approximation.

| System | Ceiling R2 | Widest beam | Oracle loss | Achieved eval loss | Loss gap |
|---|---:|---:|---:|---:|---:|
| Pendulum | `0.9887` | `1545` | `0.120` | `0.385` | `3.2x` |
| Predator-prey | `0.9997` | `3` | `0.109` | `0.314` | `2.9x` |
| Sphere | `1.0000` | `1` | `0.070` | `0.449` | `6.4x` |
| Double pendulum | `0.9895` | `3` | `0.034` | `0.378` | `11.2x` |

`256` sequences per system, no sequence hit the `20000` beam cap, so these are exact.

The observation channel destroys almost nothing: the letter sequence is very nearly recoverable from the tokens.
A belief R2 of about `0.99` is therefore attainable in principle, and the whole gap between the measured `0.25`-`0.67` and that ceiling is model capacity or readout, not lost information.

`oracle_loss` is the next-token cross entropy of a predictor that knows the exact physical state and the exact belief.
Every run sits `3x` to `11x` above it, with no train-eval gap at any point and loss still falling at `500M` tokens, which is the signature of underfitting rather than of a data limit.

## Readout dependence

`06_reference_probe.py` re-reads the same four checkpoints on the same held-out sequences with the feature construction used by the external Mess-3 study: one probe row per HMM tick, built from all four block outputs at all ten physics positions in that tick, so `4 * 128 * 10 = 5120` features instead of one layer at one position.
The raw-token control is scored on the identical rows and split, with its window and ridge penalty chosen on validation.
`6144` evaluation sequences give `11.5` training rows per feature.

| System | Trained | Random init | Gain | Raw tokens (window) | Residual excess |
|---|---:|---:|---:|---:|---:|
| Pendulum | `0.572` | `0.366` | `+0.207` | `0.454` (`W=20`) | `+0.118` |
| Predator-prey | `0.669` | `0.346` | `+0.323` | `0.639` (`W=40`) | `+0.029` |
| Sphere | `0.641` | `0.420` | `+0.221` | `0.572` (`W=40`) | `+0.068` |
| Double pendulum | `0.252` | `0.185` | `+0.067` | `0.174` (`W=20`) | `+0.077` |

Under this readout the residual excess is positive for all four systems rather than negative for all four, and every system clears the `+0.05` gain-over-initialization margin.
The excess is small, and predator-prey is within noise of its raw-token control at a single seed.

Two facts constrain how these numbers should be read.
An **untrained** network of the same shape reaches `0.366` to `0.420` on this construction, so the baseline is not zero.
Fitting and scoring on the same rows, as the external script does, adds a further `0.06` to `0.17`: our in-sample scores are `0.662`, `0.742`, `0.724` and `0.409`.

The single-position readout in `02_probe.py` and the whole-tick readout here are both legitimate measurements of different things, and the earlier tables are not withdrawn.
What is withdrawn is the claim that the myopic control wins regardless of readout.

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
Next-token training clearly learns useful physical state variables.
The belief question is not settled here: the residual stream carries a small amount of belief information beyond a matched recent-token window under the whole-tick readout, and none under the single-position readout, at one seed.
Deciding between those requires more seeds and a readout chosen before the numbers are seen, neither of which this run has.

Raw machine-readable results are in [`experiments/results-messk/`](../experiments/results-messk/).
