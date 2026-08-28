# Results -- belief simplex probes

Kaggle T4, `experiments/03_train_branch.py` then `04_probe_branch.py`.
Raw JSON in `experiments/results-branch/`. Figures in `figures/simplex_*.png`.

## Design

Hidden state is a **branch** `(z0, w_1..w_m)`: an initial condition from a set
of 4, plus the word of perturbations applied so far. Between perturbations the
true ODE is integrated exactly, so given a branch the trajectory is
deterministic. The only random events are `z0`, the perturbations, and the
emission.

Perturbations are **not** in the vocabulary. The model sees binned observations
only and must infer which kicks occurred. That inference is what makes the
belief a spread point in a simplex instead of a delta.

Child index is `parent * A + a`, so the flat branch index *is* the action word
and any lag's marginal is a reshape-and-sum. The forward algorithm is exact by
enumeration; `_brute_force_posterior` agrees to `atol=1e-9`.

| | vocab | seq_len | M | steps/seg | A | branches | actions |
|---|---|---|---|---|---|---|---|
| pendulum | 16 | 64 | 8 | 8 | 3 | 26,244 | `-dv, noop, +dv` |
| predator_prey | 8 | 48 | 6 | 8 | 5 | 62,500 | `+/-prey, +/-pred, noop` |
| sphere | 24 | 56 | 7 | 8 | 4 | 65,536 | `kick0/120/240, noop` |
| double_pendulum | 12 | 56 | 7 | 8 | 4 | 65,536 | `kick1, kick2, both, noop` |

Model: 4 layers, 1 head, `d_model=128`, `d_mlp=512`, 100M tokens, `lr=1e-3`.

## Probe targets

One ridge probe per depth maps the residual stream to the whole target vector;
R^2 is held out and reported per column block. All blocks are marginals of the
same belief `b`, hence **exactly** linear in it.

| block | width | definition |
|---|---|---|
| `action_lag0` | A | `P(most recent perturbation)` -- the headline simplex |
| `action_lag1/2` | A | same, one and two segments back |
| `z0` | 4 | `P(initial condition)` |
| `metric` | 1-2 | `E_b[m] = b @ metric_table` |

`metric` per system: pendulum `omega`; double_pendulum `(omega1, omega2)`;
sphere tangential velocity `(u, cos(phi) w)`; predator_prey `(dx/dt,
dx/dt / dy/dt)` clipped to `+/-20` (the ratio is singular where `dy/dt = 0`).

Controls: random-init model at the same seed, sequence-level train/test split
(0.7), shuffled-target fit. `baseline_intact` requires trained to exceed
random-init by `> 0.05`.

## Numbers

| system | eval loss (uniform) | H | oracle (chance) | **simplex R^2** | rand-init | shuffled | metric R^2 | z0 R^2 |
|---|---|---|---|---|---|---|---|---|
| pendulum | 1.797 (2.773) | 0.673 | 0.715 (0.333) | **0.490** @L3 | 0.156 | 0.005 | 0.874 | 0.102 |
| predator_prey | 1.281 (2.079) | 0.753 | 0.557 (0.200) | **0.509** @L2 | 0.179 | 0.008 | 0.659 | 0.347 |
| sphere | 1.726 (3.178) | 0.765 | 0.727 (0.250) | **0.077** @L3 | 0.034 | 0.006 | 0.604 | 0.606 |
| double_pendulum | 1.823 (2.485) | 0.654 | 0.521 (0.250) | **0.130** @L1 | 0.048 | 0.007 | 0.306 | 0.091 |

`H` = normalised belief entropy. `oracle` = fraction of sequences where the
exact belief's argmax is the true last action, i.e. the ceiling any probe could
reach through the belief. `@Lk` = `resid_post_k`.

## Reading

**Holds.** pendulum and predator_prey recover the simplex at ~3x random-init
with the shuffled control near zero. `H` in `[0.65, 0.77]` everywhere, so no
result is a collapsed belief in disguise. Every model beats uniform token loss
by a wide margin.

**Fails.** sphere: `0.077` vs `0.034` is under the `0.05` margin, and both are
near zero -- the perturbation posterior is essentially absent. Its `z0` R^2 is
the highest of the four, so the model tracks *which trajectory* it is on but not
*which kick* landed. The three tangent kicks are plausibly near-degenerate under
the equal-area observable.

**Weak.** double_pendulum clears the margin at `0.130` but is low, and its
metric R^2 `0.306` is the worst -- expected for a chaotic system.

**Blocks are not comparable.** `metric` is 1-2 smooth columns the model needs
anyway for next-token prediction; the simplex is 3-5 columns and the stronger
claim. Pendulum's `0.874` vs `0.490` is the expected ordering, not evidence the
metric is "more encoded".

**Low `z0` R^2 may be low target variance**, not absence -- pendulum's `0.102`
is not by itself evidence the initial condition is unrepresented.
