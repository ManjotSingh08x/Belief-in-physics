# Mess-4 results

All four seed-0 models completed 500M-token training and all CPU analyses completed on held-out sequence splits.
The exact target was the same four-state predictive-belief tetrahedron for every physical system.

## Headline

At `8` layers and `d_model=256`, trained with plain Adam on `1B` tokens, the spherical pendulum recovers the predictive belief to `R2 = 0.952` on untouched test sequences, against an exact ceiling of `1.0000` and a raw-token control of `0.572`.
Predator-prey reaches `0.823`, the pendulum `0.698`, the double pendulum `0.311`.
All four now exceed their own raw-observation-token control, by `+0.14` to `+0.38`, so the earlier readout-sensitive near-tie is gone.

The first round of models, at `4` layers and `d_model=128`, was underfit rather than uninformative.
Its negative result is recorded below unchanged, and it was wrong about the conclusion but right about the method: the raw-token control is what makes any of these numbers interpretable.
The predicted geometry for the smaller models remained a compressed central cloud rather than the exact fractal tetrahedron.

## Scaled-up runs

`8` layers, `d_model=256`, `4` heads, `d_mlp=1024`, so `6,291,456` non-embedding parameters against the first round's `786,432`.
Plain Adam with `weight_decay=0`, `1B` tokens, seed `0`, tag `_d256l8`.
Every system's `500M`-token checkpoint already beats the first round's *final* loss, so the gain is capacity rather than the doubled token budget.

| System | Wall | Eval loss @500M | Eval loss @1B | First round final | Oracle floor | Above floor |
|---|---:|---:|---:|---:|---:|---:|
| Pendulum | `4h15m` | `0.269` | `0.218` | `0.385` | `0.120` | `1.8x` |
| Sphere | `4h43m` | `0.209` | `0.164` | `0.449` | `0.070` | `2.3x` |
| Double pendulum | `4h54m` | `0.344` | `0.316` | `0.378` | `0.034` | `9.3x` |
| Predator-prey | `5h13m` | `0.222` | `0.187` | `0.314` | `0.109` | `1.7x` |

Belief probe, whole-tick readout over the top four blocks, penalty tuned on validation, test scored once.
Validation and test agree to within `0.006` on every row.

| System | First round | Scaled up | Random init | Raw tokens | Excess over raw | Ceiling | Of ceiling |
|---|---:|---:|---:|---:|---:|---:|---:|
| Sphere | `0.641` | `0.952` | `0.443` | `0.572` | `+0.380` | `1.0000` | `95.2%` |
| Predator-prey | `0.669` | `0.823` | `0.438` | `0.639` | `+0.184` | `0.9997` | `82.4%` |
| Pendulum | `0.572` | `0.698` | `0.433` | `0.454` | `+0.243` | `0.9887` | `70.6%` |
| Double pendulum | `0.252` | `0.311` | `0.217` | `0.174` | `+0.136` | `0.9895` | `31.4%` |

The raw-token control is unchanged from the first round, as it must be, since it does not depend on the model.
That it reproduces exactly is a check that the protocol itself is stable across the two rounds.

### What separates the four

The ceiling is about `0.99` for all four, so the spread is not explained by information lost in the observation channel.
Two different causes appear instead.

The double pendulum is simply not converged.
It sits `9.3x` above its own oracle loss floor while the other three sit at `1.7x` to `2.3x`, and it is chaotic, so tracking its state well enough to infer the letters is a much harder problem.
Its `31.4%` is a statement about training budget, not about whether the belief is representable.

Among the three converged systems, the ordering follows how sharply the tokens identify the letter sequence, which `07_ceiling.py` reports as the width of its consistency beam.
Sphere's beam collapses to `1` surviving letter sequence and it reaches `95.2%`; predator-prey's is `3` and it reaches `82.4%`; the pendulum's widens to `1545` and it reaches `70.6%`.
A wide beam means the letters remain genuinely ambiguous from the observations for longer, so the model faces a harder inference problem even where the belief is formally recoverable.

### Readout depth is not the binding constraint

Holding the pendulum model fixed and varying only how many blocks the probe reads:

| Blocks read | Features | Belief R2 |
|---:|---:|---:|
| `1` | `2560` | `0.553` |
| `2` | `5120` | `0.621` |
| `4` | `10240` | `0.698` |

About `+0.07` per doubling.
Extrapolating to all `8` blocks gives roughly `0.77`, so the pendulum's remaining gap is mostly the model rather than the probe.

## First round: training

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

An **untrained** network of the same shape reaches `0.366` to `0.420` on this construction, so the baseline for this readout is not zero.

Every probe in this repository is scored on rows it was not fitted on, and any selection of depth, window or penalty happens on validation.
The in-sample figures below are recorded once, as the explanation of the external script's protocol, and are used for nothing else.
Fitting and scoring on the same rows adds `0.06` to `0.17` here: `0.662`, `0.742`, `0.724` and `0.409` against the held-out `0.572`, `0.669`, `0.641` and `0.252`.
`06_reference_probe.py` no longer computes an in-sample score at all.

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
Every number here is seed `0`.

What the scaled-up round supports: at sufficient capacity, next-token training on binned physical observations does build a linearly decodable predictive belief over the hidden chain, clearly beyond what a matched recent-observation window explains.
The sphere case is strong, at `95%` of an exactly computed ceiling with a `+0.38` margin over its control.

What it does not support: any claim that this happens uniformly, or at the capacity the first round used.
The double pendulum is still far from its loss floor and its `0.311` says nothing yet about whether the belief is there.
The pendulum's `0.698` is a real gap that the depth ablation shows is not a probe artefact.

The open items are more seeds, a longer double-pendulum run, and geometry figures for the scaled-up checkpoints, none of which exist yet.

Raw machine-readable results are in [`experiments/results-messk/`](../experiments/results-messk/).
