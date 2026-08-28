# Results -- belief simplex probes

Kaggle T4, 500M tokens per system. `03_train_branch.py` -> `04_probe_branch.py` ->
`06_analyse_checkpoints.py`. Raw JSON in `experiments/results-branch/`; figures in
`figures/simplex_*.png` and `figures/analysis/`. Metric definitions in `METRICS.md`.

## Design

Hidden state is a **branch** `(z0, w_1..w_m)`: an initial condition from a set
of 4, plus the word of perturbations applied so far. Between perturbations the
true ODE is integrated exactly, so given a branch the trajectory is
deterministic. The only random events are `z0`, the perturbations, and the
emission.

Perturbations are **not** in the vocabulary. The model sees binned observations
only and must infer which kicks occurred. That inference is what makes the
belief a spread point in a simplex instead of a delta.

Child index is `parent * A + a`, so the flat branch index _is_ the action word
and any lag's marginal is a reshape-and-sum. The forward algorithm is exact by
enumeration; `_brute_force_posterior` agrees to `atol=1e-9`.

|                 | vocab | seq_len | M   | steps/seg | A   | branches | actions                    |
| --------------- | ----- | ------- | --- | --------- | --- | -------- | -------------------------- |
| pendulum        | 16    | 64      | 8   | 8         | 3   | 26,244   | `-dv, noop, +dv`           |
| predator_prey   | 8     | 48      | 6   | 8         | 5   | 62,500   | `+/-prey, +/-pred, noop`   |
| sphere          | 24    | 56      | 7   | 8         | 4   | 65,536   | `kick0/120/240, noop`      |
| double_pendulum | 12    | 56      | 7   | 8         | 4   | 65,536   | `kick1, kick2, both, noop` |

Model: 4 layers, 1 head, `d_model=128`, `d_mlp=512`, 500M tokens, `lr=1e-3`.

## Probe targets

One ridge probe per depth maps the residual stream to the whole target vector;
R^2 is held out and reported per column block. All blocks are marginals of the
same belief `b`, hence **exactly** linear in it.

| block           | width | definition                                            |
| --------------- | ----- | ----------------------------------------------------- |
| `action_lag0`   | A     | `P(most recent perturbation)` -- the headline simplex |
| `action_lag1/2` | A     | same, one and two segments back                       |
| `z0`            | 4     | `P(initial condition)`                                |
| `metric`        | 1-2   | `E_b[m] = b @ metric_table`                           |

`metric` per system: pendulum `omega`; double_pendulum `(omega1, omega2)`;
sphere tangential velocity `(u, cos(phi) w)`; predator_prey `(dx/dt,
dx/dt / dy/dt)` clipped to `+/-20` (the ratio is singular where `dy/dt = 0`).

Controls: random-init model at the same seed, sequence-level train/test split
(0.7), shuffled-target fit. `baseline_intact` requires trained to exceed
random-init by `> 0.05`.

## Numbers

At 500M tokens, probe at the best depth. `untr.` is the same architecture untrained.

| system          | loss (uniform) | H     | simplex R^2 | untr. | shuffled | metric R^2 | untr. | z0 R^2 | untr. |
| --------------- | -------------- | ----- | ----------- | ----- | -------- | ---------- | ----- | ------ | ----- |
| pendulum        | 1.797 (2.773)  | 0.673 | **0.590** @L3 | 0.156 | 0.005 | **0.891** | 0.110 | 0.119 | 0.088 |
| predator_prey   | 1.280 (2.079)  | 0.753 | **0.681** @L2 | 0.179 | 0.009 | **0.676** | 0.427 | 0.403 | 0.422 |
| sphere          | 1.712 (3.178)  | 0.765 | 0.074 @L2     | 0.034 | 0.006 | **0.587** | 0.346 | 0.617 | 0.624 |
| double_pendulum | 1.803 (2.485)  | 0.654 | **0.137** @L1 | 0.048 | 0.008 | **0.328** | 0.136 | 0.116 | 0.077 |

Erasure at the best depth, final checkpoint. `excess` is over a random subspace of the same
rank, in units of that control's standard deviation.

| system | quantity | rank | R^2 after | excess (sd) |
|---|---|---|---|---|
| pendulum | belief | 35 | 0.052 | -0.040 (-4) |
| pendulum | metric | 48 (capped) | 0.119 | **+0.137 (+15)** |
| predator_prey | all three | 48 (capped) | 0.27-0.30 | inconclusive |
| sphere | belief | 17 | 0.014 | -0.005 (-2) |
| double_pendulum | belief | 22 | 0.016 | **+0.050 (+22)** |
| double_pendulum | metric | 12 | 0.023 | **+0.014 (+25)** |

## Reading

**Learned vs. artifact.** Against the untrained baseline: the simplex is learned in pendulum
(+0.43), predator_prey (+0.50) and weakly in double_pendulum (+0.09). It is **not** learned in
sphere (+0.04). `metric` is learned in all four. **`z0` is learned in none of them** --
predator_prey and sphere score *lower* than untrained, and the other two gain under 0.04. A
random projection of the residual stream already predicts the initial condition, so no `z0`
number in this project supports a claim about the model.

**The representation sharpens after the loss converges.** Pendulum loss is 1.8163 at 12M and
1.7975 at 500M, a 1% move, while the simplex goes 0.356 -> 0.590. predator_prey loss is flat
from 5M while its simplex goes 0.272 -> 0.681. Neither simplex curve has plateaued at 500M.

**Metric arrives before belief.** Pendulum at 12M: metric at 86% of its final value, simplex at
60%. Metric saturates by ~30M; the simplex is still climbing at 500M.

**Erasure separates decodable from used.** double_pendulum uses both belief and metric causally
(+22 and +25 sd). Pendulum uses the metric strongly (+15 sd) but erasing its belief costs
*less* than a random subspace of equal rank (-4 sd), despite R^2 = 0.590. predator_prey is
inconclusive: all three erasures hit the rank cap with R^2 still at 0.27-0.30, so "not used"
and "not erased" cannot be told apart.

**The three subspaces overlap well above chance.** Principal angles run 20-29 degrees below the
random-subspace null for every pair in every system. The rank-2-4 probe readout directions are
near-orthogonal, but the full subspaces carrying each quantity are not.

**Blocks are not comparable.** `metric` is 1-2 smooth columns the model needs anyway for
next-token prediction; the simplex is 3-5 and the stronger claim.
