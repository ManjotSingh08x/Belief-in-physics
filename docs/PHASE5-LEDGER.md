# Does the transformer build the HMM's belief state? The full ledger

Every phase-5 result, signed for or against the hypothesis, for all four systems,
followed by an answer to a single question: **which of the negatives would more
parameters fix?**

Regenerate the tables with `uv run python scripts/phase5_ledger.py`.

Model under test throughout: 4 layers, 1 head, `d_model = 128`, `d_mlp = 512`,
**806k parameters**, 500M tokens.

## The one table that decides the scaling question

Excess next-token loss over the **exact Bayes floor** computed from the HMM. This
is not a proxy for capacity, it is the thing itself: a model at the floor has
solved the task it is trained on, and no amount of extra capacity can improve its
predictions.

| system | Bayes floor | L(12M) | L(500M) | excess @500M | fraction of its 12M excess still unclosed |
|---|---|---|---|---|---|
| pendulum | 1.7966 | 1.8163 | 1.7975 | **0.0009** | 4.5% |
| predator_prey | 1.2779 | 1.2843 | 1.2773 | **-0.0005** | at the floor |
| sphere | 1.6625 | 1.7540 | 1.7071 | **0.0446** | 48.7% |
| double_pendulum | 1.7677 | 1.8896 | 1.8027 | **0.0351** | 28.8% |

predator_prey scores marginally below the plug-in floor estimate, which is
estimator noise on a finite sample, not a violation -- read it as "at the floor".

**The systems split two and two, and every other result in this project splits
the same way.**

## Positive results, by system

**P1. The belief is decodable, and it is learned.** Linear ridge probe at the best
depth; `untr` is the identical architecture untrained at the same seed.

| system | depth | linear | untrained | MLP | MLP untrained |
|---|---|---|---|---|---|
| pendulum | resid_post_3 | **0.590** | 0.153 | **0.873** | 0.315 |
| predator_prey | resid_post_2 | **0.681** | 0.176 | **0.857** | 0.323 |
| sphere | resid_post_2 | 0.074 | 0.033 | 0.126 | 0.043 |
| double_pendulum | resid_post_1 | 0.137 | 0.048 | 0.290 | 0.124 |

Positive in pendulum and predator_prey. Weak in double_pendulum, absent in sphere.

**P2. It is not the next-token distribution in disguise.** This is the load-bearing
positive of the whole project, because a many-to-one map from beliefs to
predictive distributions is what makes "belief state" a stronger claim than
"the thing the model is trained to output".

| system | R^2(b given p) | R^2(b given p at k=1..8) | matched-pair R^2 |
|---|---|---|---|
| pendulum | 0.023 | 0.168 | **0.713** |
| predator_prey | 0.128 | 0.179 | **0.738** |
| sphere | 0.015 | 0.055 | not runnable |
| double_pendulum | 0.023 | 0.057 | 0.017 |

The matched-pair number is the strong form: pairs with near-identical predictive
distributions (`TV < 0.014`) but different beliefs, regressed on the *difference*
of residual streams, which differences out every myopic component
non-parametrically. Positive in two systems.

**P3. The stream tracks the posterior, not the hidden truth.** On held-out
positions where the posterior's mode disagrees with the action that actually
fired:

| system | disagree | probe mode = belief | probe mode = truth | chance |
|---|---|---|---|---|
| pendulum | 36% | **0.691** | 0.200 | 0.333 |
| predator_prey | 53% | **0.616** | 0.147 | 0.200 |
| sphere | 49% | 0.388 | 0.216 | 0.250 |
| double_pendulum | 44% | 0.330 | 0.246 | 0.250 |

Agreement with the truth is *below chance* precisely where the belief points
elsewhere, which a partial decoding of the hidden action cannot produce.

**P4. It is not a lexical bag.** A causal one-hot bag of the last 8 tokens plus
phase recovers only 0.260 / 0.215 / 0.052 / 0.028 of the belief, against the
model's 0.590 / 0.681 / 0.074 / 0.137. (It does get 93% of the *metric*, which is
why the metric is the weaker claim.)

**P5. Coupling to `p(next)` orders emergence.** 24 of 24 cells negative across four
systems, two target families and two normalisations.

**P6. Methodological gates pass.** The impossible probe (`y - b`) scores -0.004 to
-0.011 with CIs below zero in all four systems, and a position-only baseline
scores -0.0002 to -0.0005. There is no leakage and no positional artifact.

**P7. The one causal positive.** Erasing the belief-specific complement
(`belief` minus `metric`, matched rank) beats a matched random subspace by more
than 2 sd in 10 of 14 cells, largest at `resid_post_0` where C12 says the belief is
built.

## Negative results, by system

**N1. The belief is not decodable in sphere, and barely in double_pendulum.**
0.074 and 0.137 linear, against oracle information that is plainly present
(below).

**N2. The representation is not linear.** Linear captures only 47-79% of what an
MLP reads from the same activations:

| system | linear | MLP | linear as fraction of MLP |
|---|---|---|---|
| pendulum | 0.590 | 0.873 | 0.68 |
| predator_prey | 0.681 | 0.857 | 0.79 |
| sphere | 0.074 | 0.126 | 0.59 |
| double_pendulum | 0.137 | 0.290 | 0.47 |

**N3. Ablation: nothing clears a variance-matched null.** Best belief cell per
system, excluding the degenerate embedding, in nats:

| system | vs isotropic random | vs variance-matched PCA |
|---|---|---|
| pendulum | -0.0004 | -0.0738 |
| predator_prey | -0.0595 | -0.1714 |
| sphere | +0.0322 | -0.0148 |
| double_pendulum | +0.0801 | **+0.0595** |

**N4. Multi-step rollout (E2d) is negative where the belief is decodable.** Best
excess KL over a matched-rank control, in control sd:

| system | best | reading |
|---|---|---|
| pendulum | -0.3 | negative at every depth and every k |
| predator_prey | -1.6 | negative at every depth and every k, worsening with k |
| sphere | **+9.5** | positive |
| double_pendulum | **+7.0** | positive |

**N5. The erasure never removed the feature.** After INLP drops the linear probe
from 0.590 to 0.052, an MLP still reads the belief at **0.852** (98% of intact).
Same in predator_prey: 0.681 -> 0.193 linear, 0.857 -> 0.836 MLP.

**N6. Patching the complement is split.** Two systems positive, two negative, and
the largest sd figure (+22 sd, predator_prey) is a 0.098 absolute gap inflated by
a control sd of 0.004.

**N7. E2a and E2b are null** at every non-degenerate depth in all four systems.

**N8. `z0` is not learned anywhere.** sphere scores 0.617 trained against 0.624
untrained; predator_prey 0.403 against 0.422.

## The negatives split into four classes, and only two are about parameters

**E. The information is available in every system**, so nothing below is a limit
of the process:

| system | chance | oracle recovery at end of segment | model linear | model MLP |
|---|---|---|---|---|
| pendulum | 0.33 | 0.72 | 0.590 | 0.873 |
| predator_prey | 0.20 | 0.54 | 0.681 | 0.857 |
| sphere | 0.25 | **0.71** | **0.074** | 0.126 |
| double_pendulum | 0.25 | **0.74** | **0.137** | 0.290 |

Sphere's posterior identifies the hidden perturbation at 0.71 against 0.25
chance, better than the pendulum's 0.72-versus-0.33 in absolute terms. The
information is there and the model is not taking it.

### Class 1 -- underfitting. More parameters should fix these. (N1, and N4's positives)

sphere and double_pendulum are the two systems with **48.7% and 28.8% of their
loss excess unclosed**, and they are exactly the two where the belief is not
decoded. They are also the two where the causal tests come out *positive*, which
on this reading is the signature of a model still using a small, fragile,
not-yet-redundant representation.

Prediction, falsifiable: at larger `d_model` these two move toward the floor,
belief R^2 rises toward the pendulum's, and **their causal positives shrink**.

### Class 2 -- linearity, not capacity. Parameters may help, untested. (N2)

The pendulum's information is in the stream at 0.873 and read linearly at 0.590.
That gap is a fact about representational *format*, not about how much the model
knows. Width plausibly linearises features. This has not been tested here, and
unlike class 1 there is no headroom argument that predicts it.

### Class 3 -- not about parameters at all. Scale should NOT fix these, and may worsen them. (N3, N4's negatives, N6, N7)

pendulum and predator_prey sit at **0.0009 and -0.0005 nats from the exact Bayes
floor**. These models have solved the task. There is no loss left for extra
capacity to buy, therefore no gradient pressure that would make a bigger model
start *using* a belief subspace it currently does not use. A model can only be
pushed toward an algorithm by a loss it is failing to achieve, and these are not
failing.

Worse, width works against the measurement. Every one of these is a
**rank-limited linear intervention**: erase `r` directions and see what breaks. As
`d_model` grows, a fixed feature has more room to be represented redundantly, so
the fraction of it captured by any rank-`r` subspace falls, and the erasure
removes proportionally less. The intervention gets weaker as the model gets
bigger. Expect these nulls to become *more* null.

### Class 4 -- an instrument limit. Scale strictly worsens it. (N5)

R6 is not a fact about the model, it is a fact about INLP: it guarantees only
*linear* non-decodability. Every ablation in this project therefore measures
deletion of linear readability, not deletion of a feature. This is worse at every
larger width for the same redundancy reason. No parameter count fixes it; a
different instrument does.

## What the ledger says, in one line each

- **Claim 1 (the belief is more than `p`) is the survivor**, and it is only
  demonstrated in the two systems that reached the Bayes floor.
- **Every clean causal test is negative in exactly those two systems**, and
  positive only in the two that have not converged.
- The most economical reading of the whole ledger: the belief-state geometry is a
  **by-product of an optimal predictor** rather than a mechanism it consults, and
  the causal positives in sphere and double_pendulum are what a *partially
  trained* model looks like, not what a belief-using model looks like.
- That reading makes a sharp prediction, which is the natural next experiment:
  **train sphere and double_pendulum wider until they reach their floors. If the
  causal positives disappear as the loss excess closes, the hypothesis is
  finished.** If instead belief R^2 rises *and* the causal effect survives at the
  floor, it is the strongest result the project could get.
