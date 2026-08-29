# Phase 5 -- results

Methods and falsifiers are in `PHASE5-METHODS.md`. Numbers below come from
`phase5_00_validity.json` through `phase5_07_emergence.json` and are reproducible
with `uv run python scripts/phase5_summary.py`.

Still one seed per system unless a row says otherwise. The replication seeds were
running when this was written.

## Summary: what survived, what did not

| claim, as phase 4 stated it | verdict |
|---|---|
| 1. The belief is not recoverable from the optimal next-token distribution | **survives, and strengthened.** Tested three more ways, including one that could have killed it |
| 2. The representation keeps sharpening after the loss converges | **refuted.** The loss had not converged; excess over the Bayes floor fell 95% across the same window |
| 3. Erasure separates decodable from used | **withdrawn, and the method with it.** The depth was chosen by a coin flip, the sign flips when depth is held fixed, an MLP still reads the target at 98% of intact after the erasure, and only 2 of 19 belief cells clear a variance-matched null |
| 4. The metric emerges before the belief because it couples more to p | **direction supported, mechanism not shown.** Higher coupling emerges earlier in 24 of 24 cells across four systems and two target families, but `alpha` cannot move coupling without also moving how much of the target is a linear image of `p` |
| z0 is not learned in any system | **survives** |
| The three subspaces overlap far above chance | **survives, and matters more than stated** |

Two results that were not claims before and are now the strongest things here:
positions with the same predictive distribution but different beliefs are
separated in the residual stream (E1b), and the stream tracks the posterior
rather than the hidden truth where the two disagree (V2).

## 5.0 The gates

**V1, the impossible probe: PASS on all four systems.** A probe fitted to `y - b`
scores -0.004 to -0.011 with bootstrap CIs entirely below zero, while the same
probe at the same depth finds the real target. There is no leakage, and the
pipeline is validated end to end for the first time.

**Position alone explains nothing.** A probe from one-hot phase-in-segment to the
belief scores -0.0002 to -0.0005; from absolute position, -0.002 to -0.006. The
belief R^2 is not the model knowing where it is in the sequence.

**V4, the Bayes floor. Claim 2 is refuted.**

| system | floor | L(12M) | L(500M) | excess 12M | excess 500M | reduction |
|---|---|---|---|---|---|---|
| pendulum | 1.7966 | 1.8163 | 1.7975 | 0.0197 | 0.0009 | **95%** |
| predator_prey | 1.2779 | 1.2843 | 1.2773 | 0.0064 | -0.0005 | at the floor |
| sphere | 1.6625 | 1.7540 | 1.7071 | 0.0915 | 0.0446 | 51% |
| double_pendulum | 1.7677 | 1.8896 | 1.8027 | 0.1219 | 0.0351 | 71% |

The pendulum's "1% move" in raw loss is a 95% reduction in the only part of the
loss that can move. The belief R^2 rising from 0.356 to 0.590 over the same
window is not decoupled from the loss; the two improve together. predator_prey
reaches its floor (the small negative is finite-sample noise; its realised floor
is 1.2763).

Two systems are still far from optimal at 500M: sphere at +0.045 and
double_pendulum at +0.035. Those are the two whose belief R^2 is lowest.

**C12, where each quantity is built.** The residual stream is additive so R^2 is
monotone in depth; only the increment locates the computation. Pendulum,
`action_lag0`, per-layer gain:

```
embedding  +0.089   L0 +0.193   L1 +0.215   L2 +0.092   L3 +0.002
```

Phase 4 ran its causal test at `resid_post_3`, where the increment is **+0.002**,
chosen over `resid_post_2` by a margin of **0.0016** R^2.

**C19.** double_pendulum's metric is 0.328 at the belief's depth and 0.415 at its
own, a 0.088 understatement that feeds directly into the claim-4 ordering. The
other three systems lose 0.000 to 0.011.

**C21. The sphere story in phase 4 section 5 was wrong, and so was my rewording
of it.** The phase-3 field is measured at the final position only. As a function
of observations since the kick:

| system | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | chance |
|---|---|---|---|---|---|---|---|---|---|
| pendulum | 0.43 | 0.53 | 0.62 | 0.67 | 0.70 | 0.70 | 0.71 | 0.72 | 0.33 |
| predator_prey | 0.34 | 0.39 | 0.41 | 0.45 | 0.47 | 0.50 | 0.52 | 0.54 | 0.20 |
| sphere | 0.30 | 0.36 | 0.41 | 0.47 | 0.54 | 0.60 | 0.65 | 0.71 | 0.25 |
| double_pendulum | 0.32 | 0.40 | 0.48 | 0.55 | 0.61 | 0.66 | 0.70 | 0.74 | 0.25 |

Sphere's information does arrive late, but it arrives: by the end of a segment
its oracle recovery is 0.71, essentially the pendulum's 0.72. The information is
present and **the model fails to extract it**. Sphere is a model failure, not an
information deficit, which is consistent with it having the largest excess loss
of the four.

## 5.1 The full ablation grid: F1 confirmed, and a null that nothing clears

Every depth, every checkpoint, three nulls, a rank ladder, and a resample arm.
This is the experiment phase 4 should have run, and it retracts phase 4's
erasure section outright.

**F1: the phase-4 sign does not survive holding depth fixed.** Phase 4 re-chose
the readout depth at every checkpoint by an argmax over near-equal R^2 values,
so its "excess Δloss over training" curve mixes two moving parts. Holding depth
fixed, the pendulum's belief excess (nats over a matched random subspace) reads:

| depth | 0M | 2M | 5M | 12M | 30M | 60M | 120M | 250M | 500M |
|---|---|---|---|---|---|---|---|---|---|
| embedding | +0.016 | +0.214 | +0.327 | +0.399 | +0.397 | +0.393 | +0.379 | +0.378 | +0.382 |
| resid_post_0 | +0.064 | +0.093 | +0.100 | -0.115 | -0.106 | -0.090 | -0.085 | -0.086 | -0.074 |
| resid_post_1 | +0.055 | +0.070 | -0.098 | -0.093 | -0.062 | -0.050 | -0.009 | +0.007 | -0.000 |
| resid_post_2 | +0.052 | -0.033 | -0.058 | -0.117 | -0.129 | -0.110 | -0.095 | -0.081 | -0.072 |
| resid_post_3 | +0.046 | -0.050 | -0.058 | -0.107 | -0.107 | -0.114 | -0.091 | -0.058 | -0.046 |

Each row is a different story, and three of the five change sign inside the
window. predator_prey is the same shape. Any claim read off a curve that lets
the depth move is unreadable.

**F2 confirmed exactly.** At `resid_post_3` the erasure has almost nothing to
act on: erasing the pendulum's rank-35 belief basis costs **+0.0009 nats**
against +0.0426 for a matched random subspace. At the final residual stream only
`vocab_size - 1` directions can move the loss at all -- 15 of 128 for the
pendulum, 7 for predator_prey. Phase 4 read its headline at a depth where the
measurement is bounded near zero by construction.

**The variance-matched null erases the rest.** Three nulls were run: an isotropic
random subspace, the top-r principal components, and the fewest principal
components carrying the same variance fraction as the target basis. The third is
the honest one -- an erasure that removes 23% of the stream's variance should be
compared against another that removes 23%, not against a random subspace that
removes 3%.

Belief erasure, final checkpoint, nats above each null:

| system | depth | rank | var frac | Δloss | vs random (sd) | vs var-matched PCA |
|---|---|---|---|---|---|---|
| pendulum | embedding | 20 | 0.135 | 0.3962 | +0.382 (97) | **+0.337** |
| pendulum | resid_post_0 | 60 | 0.106 | 0.1518 | -0.074 (-3) | -0.101 |
| pendulum | resid_post_1 | 60 | 0.230 | 0.1930 | -0.000 (-0) | -0.074 |
| pendulum | resid_post_2 | 46 | 0.113 | 0.0083 | -0.072 (-4) | -0.351 |
| pendulum | resid_post_3 | 35 | 0.034 | 0.0009 | -0.046 (-3) | -0.527 |
| predator_prey | resid_post_0 | 66 | 0.152 | 0.0471 | -0.060 (-2) | -0.171 |
| predator_prey | resid_post_3 | 66 | 0.056 | 0.0008 | -0.172 (-5) | -0.849 |
| sphere | resid_post_1 | 20 | 0.078 | 0.0530 | +0.032 (+7) | -0.015 |
| double_pendulum | resid_post_1 | 22 | 0.150 | 0.0835 | +0.051 (+14) | -0.062 |
| double_pendulum | resid_post_2 | 16 | 0.123 | 0.0634 | +0.050 (+26) | -0.011 |
| double_pendulum | resid_post_3 | 27 | 0.026 | 0.1349 | +0.080 (+2) | **+0.060** |

Of 19 belief cells across all four systems, **two** clear the variance-matched
null: the pendulum embedding and double_pendulum's final layer. The metric grid
is the same -- 18 cells, one clearance (the predator_prey embedding). Every
positive result phase 4 reported, including double_pendulum's +22 sd and the
pendulum metric's +15 sd, is a comparison against a null that removes less
variance than the treatment.

**And the embedding clearances are degenerate.** At the embedding the belief
probe reads token identity, so erasing its subspace deletes the input rather
than a computed feature. That is why the embedding row is large, flat across
training in predator_prey (+0.486 at 2M, +0.532 at 500M), and present in the
untrained model. It is not evidence about belief use.

The one survivor worth naming is **double_pendulum at `resid_post_3`**: +0.1349
nats, +0.060 above the variance-matched null, at a depth where at most 11 of 128
directions can move the loss. It is a real effect in the only system where the
belief probe is weak (R^2 = 0.137), which is the opposite of what claim 3
predicted.

**The resample arm rules out a mean-shift artifact.** Replacing the subspace with
a phase-matched activation from another sequence, rather than the mean, gives
larger damage everywhere (pendulum `resid_post_1`: 0.314 vs 0.193), so the mean
ablation is not merely knocking the stream off its manifold. It does not change
any sign.

## 5.2 What the R^2 numbers mean

**V2. The stream tracks the posterior, not the truth.** On held-out positions
where the posterior's mode disagrees with the action that actually fired, a probe
fitted to the belief lands on the belief's mode far more often than on the truth:

| system | disagreements | probe mode = belief mode | probe mode = truth | chance |
|---|---|---|---|---|
| pendulum | 36% | **0.689** | 0.202 | 0.333 |
| predator_prey | 53% | **0.616** | 0.147 | 0.200 |

Agreement with the truth is *below* chance on exactly the positions where the
belief points elsewhere, which is what a representation of the posterior should
do and what a partial decoding of the hidden action could not. This is the
single cleanest piece of evidence that the simplex framing is not decoration,
and it did not exist before phase 5.

**V3. The dumb baseline is well short on the belief and close on the metric.**

| system | features | d_model | bag: belief | model: belief | bag: metric | model: metric |
|---|---|---|---|---|---|---|
| pendulum | 136 | 128 | 0.260 | 0.590 | **0.829** | 0.891 |
| predator_prey | 72 | 128 | 0.215 | 0.681 | 0.477 | 0.676 |

A bag of the last 8 observation tokens plus the phase in segment, with roughly as
many features as `d_model`, recovers 44% of the pendulum's belief R^2 but **93%
of its metric R^2**. The metric result is therefore a much smaller claim than it
reads as: `E_b[omega]` is close to a function of the last few tokens. The belief
result is not.

**V5. The linear probe substantially understates the belief.** MLP 0.873 against
linear 0.590 for the pendulum, 0.857 against 0.681 for predator_prey. The
untrained baseline also rises (0.316 and 0.318), so the learned gain under an MLP
is 0.557 and 0.539, comparable to the linear gain. The ordering survives; the
levels do not.

**R6. "Erased" is the wrong word, everywhere in this project.**

| system | depth | rank | linear before -> after | MLP before -> after |
|---|---|---|---|---|
| pendulum | resid_post_3 | 35 | 0.590 -> 0.052 | **0.873 -> 0.852** |
| predator_prey | resid_post_2 | 64 | 0.681 -> 0.193 | **0.857 -> 0.836** |

Iterative nullspace projection removes linear decodability and essentially
nothing else. After an intervention that drops the linear probe by 91%, an MLP
still recovers the belief at 98% of its intact value. **Every ablation result in
phases 4 and 5 measures the deletion of a linearly readable component, not the
deletion of a feature.** A null in such an experiment is therefore weak evidence
about use: the model can still reach the belief through the nonlinear path that
survived. This is the most important caveat in phase 5 and it applies to the
causal section below.

**C17.** Scoring the free coordinates of the simplex rather than the dependent
columns moves the pendulum from 0.590 to 0.627 and predator_prey from 0.681 to
0.690. Real but small; the phase-4 numbers were not distorted by this.

**C16.** Belief R^2 by entropy quartile, pendulum: 0.20 / 0.48 / 0.15 / -0.14
from low to high entropy. Part of this is mechanical, since a narrow stratum has
less target variance to explain, but the highest-entropy bin (just after a kick,
when the posterior is nearly uniform) carries no signal at all. The pooled number
is not driven by position, though: a position-only probe scores -0.0002.

**V6.** Bootstrap CIs over held-out sequences are narrow: pendulum belief 0.590
with [0.557, 0.613], predator_prey 0.681 with [0.663, 0.696]. The n = 154 concern
is real for the design but does not by itself make these numbers fragile.

## 5.3 / 5.4 Claim 1 survives four attacks

**E1a. The gap is orthogonality, not linearity.**

| system | linear | log-linear | ceiling r=0.02 | r=0.04 | singleton frac | stream | top-`n_obs` PCs | random `n_obs` proj |
|---|---|---|---|---|---|---|---|---|
| pendulum | 0.023 | 0.032 | 0.116 | 0.061 | 0.38 | **0.612** | 0.303 | 0.224 |
| predator_prey | 0.128 | 0.132 | 0.268 | 0.195 | 0.20 | **0.696** | 0.233 | 0.240 |
| sphere | 0.015 | 0.004 | 0.963 | 0.732 | **0.91** | 0.082 | 0.040 | 0.022 |
| double_pendulum | 0.023 | 0.022 | 0.266 | 0.148 | 0.64 | 0.151 | 0.083 | 0.042 |

The conditional-variance ceiling is the best possible predictor of the belief
from `p`, of any functional form. For the pendulum it sits between 0.061 and
0.116 against a stream value of 0.612, so nearly all of the gap is real
orthogonality. Using `log p` instead of `p`, the matched comparison for a
residual stream that is linearly related to logits, moves nothing.

**Sphere's ceiling is not usable and is not quoted anywhere:** 91% of its
clusters are singletons at r=0.02, which forces the estimate toward 1 by
construction. Its emission alphabet is large enough that near-duplicate
predictive distributions barely occur.

C4 is addressed: dimension-matched controls give 0.303 and 0.224 for the
pendulum, still an order of magnitude above `p`'s 16 features at 0.023.

**E1c. The deflationary reading is ruled out.** `R^2(belief | p^(1..K))`:

| system | k=1 | k=2 | k=4 | k=8 | pairwise joint at k=2 |
|---|---|---|---|---|---|
| pendulum | 0.023 | 0.106 | 0.119 | **0.168** | 0.130 |
| predator_prey | 0.128 | 0.154 | 0.159 | **0.179** | 0.173 |
| sphere | 0.015 | 0.024 | 0.035 | 0.055 | 0.104 |
| double_pendulum | 0.023 | 0.030 | 0.041 | 0.057 | 0.088 |

Eight steps of the exact future predictive law explain at most 18% of the belief.
Since marginals are not the joint, the pairwise joint `p(x_{t+1}, x_{t+2})` is
reported too and adds almost nothing. Next-token training at t+1..t+K is
therefore **not** a sufficient explanation for the belief being present at t.

The contrast with the metric is the sharpest single result in this section:
`R^2(metric | p^(1..2))` is **0.968** for the pendulum. The metric essentially
*is* the two-step predictive distribution. The belief is not.

**E1b. The categorical form of the claim.** Positions matched to within
TV < 0.02 in predictive distribution but differing by more than 0.2 in belief;
the residual-stream difference regressed on the belief difference; split so both
endpoints of a pair fall on the same side.

| system | pairs used | mean TV | mean `|db|` | paired R^2 | shuffled | pooled R^2 |
|---|---|---|---|---|---|---|
| pendulum | 84,407 | 0.0136 | 0.671 | **0.713** | +0.001 | 0.612 |
| predator_prey | 85,247 | 0.0134 | 0.489 | **0.738** | -0.002 | 0.696 |
| sphere | - | - | - | not runnable | - | 0.082 |
| double_pendulum | 47,398 | 0.0041 | 0.539 | **0.017** | -0.002 | 0.151 |

For the two systems with a strong pooled result, holding the predictive
distribution fixed *raises* the recovered R^2. That is the Shai et al. claim in
its categorical form: belief states that produce the same next-token
distribution are still separated in the residual stream.

double_pendulum is a clean null here: its pooled 0.151 is essentially all
variation in `p`, and nothing survives the matching. Sphere has no power -- only
3.6% of its TV-matched pairs have a belief gap above the threshold, which is a
fact about the process rather than a result.

An earlier version of this analysis split pairs by one endpoint only. Because
each pair is symmetrised into two rows that are exact negations, that could put a
row and its negation on opposite sides. The effect is bounded by in-sample
optimism and was immaterial for the pendulum (0.712 to 0.713), but
double_pendulum moved from 0.117 to 0.017 and sphere from an underpowered 0.628
to not runnable. `tests/test_phase5.py` carries the regression.

## 5.5 / 5.6 The causal picture

Read everything here against R6: the intervention removes linear readability,
not the feature.

**F1 and F2 confirmed.** At `resid_post_3`, erasing the pendulum's rank-35 belief
basis costs **Δloss = +0.0009 nats**, against +0.0426 for a random subspace of
the same rank. That is the depth phase 4 chose, by a margin of 0.0016 R^2, and
the depth whose own contribution to the belief is +0.002. Nothing could have been
detected there.

**E2b, position-restricted ablation: negative.** Corrupting the belief subspace
in a block of 4 positions and reading the loss at the following positions gives
excesses of order 1e-3 or negative at every non-embedding depth in every system.
There is no evidence that the belief is stored at t for later use rather than
recomputed. The embedding is degenerate and is reported but not counted: the
belief probe there reads token identity, so patching it substitutes the input,
and it scores +0.11 to +0.33.

**E2c, patching: mostly negative.** Transplant-alignment cosine, belief basis
against a matched-rank random subspace, at the depth after the first block:

```
pendulum       resid_post_2   r2:-0.03/+0.43   r8:-0.02/+0.65   r46:+0.20/+0.88
predator_prey  resid_post_2   r4:+0.42/+0.70   r8:+0.43/+0.86   r64:+0.69/+0.98
sphere         resid_post_2   r3:+0.06/+0.30   r8:+0.09/+0.48   r17:+0.18/+0.61
```

Moving the belief subspace moves the model *less* toward the donor's predictive
distribution than moving an arbitrary subspace of the same size does. The
exception is double_pendulum, which beats its control at `resid_post_1` and
`resid_post_2` (0.66 against 0.47, 0.67 against 0.53) -- notable because that is
the one system where E1b found nothing.

**E2a, decodability horizon.** The drop in R^2 onto the token at t+k after
erasing the belief is 0.001 to 0.013 and flat or falling in k, at every depth
except the embedding. The intact head is itself weak (R^2 0.04 to 0.14 onto a
one-hot future token), so this is a low-power measurement and is reported as a
property of the representation, not as evidence about use.

**R4, complement erasure: the strongest causal positive in phase 5.**

First, the premise. The first principal angle between the belief and metric
erasure bases is **0.0 degrees** in 12 of 14 cells, with up to 47 angles below 10
degrees. The two bases share their leading directions outright, so no
un-complemented ablation of either is about one quantity.

Erasing only the part of each basis orthogonal to the other, at matched rank:

| system | depth | rank | belief - metric | metric - belief | random |
|---|---|---|---|---|---|
| pendulum | resid_post_0 | 20 | **+0.1192** | **+0.1855** | +0.0164 ± 0.0039 |
| pendulum | resid_post_1 | 14 | +0.0240 | +0.0197 | +0.0075 ± 0.0030 |
| pendulum | resid_post_2 | 23 | +0.0075 | +0.0557 | +0.0147 ± 0.0046 |
| predator_prey | resid_post_0 | 39 | **+0.2534** | +0.0977 | +0.0285 ± 0.0091 |
| predator_prey | resid_post_2 | 37 | +0.0453 | **+0.1380** | +0.0449 ± 0.0143 |
| sphere | resid_post_1 | 20 | +0.0511 | +0.0217 | +0.0195 ± 0.0038 |
| double_pendulum | resid_post_0 | 29 | +0.1027 | +0.0950 | +0.0571 ± 0.0067 |
| double_pendulum | resid_post_2 | 16 | +0.0314 | +0.0527 | +0.0129 ± 0.0022 |

Across the 14 (system, depth) cells, the belief-specific complement exceeds the
matched random control by more than 2 sd in **10**, sits between 0 and 2 sd in 2,
and falls below it in 2. The metric-specific complement exceeds by more than 2 sd
in 12. The four belief cells that do not clear 2 sd are:

```
pendulum       resid_post_2   +0.0075 vs random +0.0147 +- 0.0046   z = -1.6
predator_prey  resid_post_1   +0.0587 vs random +0.0386 +- 0.0152   z = +1.3
predator_prey  resid_post_2   +0.0453 vs random +0.0449 +- 0.0143   z =  0.0
sphere         resid_post_2   +0.0098 vs random +0.0146 +- 0.0021   z = -2.3
```

Three of the four are at `resid_post_2`, and the effect is largest at
`resid_post_0`, which C12 identifies as where the belief is built. The pattern is
consistent across the whole of phase 5: the belief-specific directions matter
where the belief is being constructed and stop mattering downstream.

The full erasure basis is not load-bearing because it is dominated by redundant
low-variance directions the loss does not depend on; the unique 20 of its 60
directions are.

**R4 by patching, not erasure: split, and weaker than it looks.** The same
belief-minus-metric complement, at `resid_post_0`, driven the other way -- patch
the complement from a donor sequence and ask whether the logits move where the
HMM says they should. Alignment cosine at the full complement rank:

| system | rank | belief complement | matched random | excess |
|---|---|---|---|---|
| pendulum | 19 | +0.427 | +0.612 ± 0.053 | **-3.5 sd** |
| predator_prey | 39 | +0.937 | +0.839 ± 0.004 | +22.3 sd (**+0.098 absolute**) |
| sphere | 24 | +0.330 | +0.415 ± 0.031 | **-2.7 sd** |
| double_pendulum | 29 | +0.548 | +0.426 ± 0.037 | +3.3 sd (+0.122 absolute) |

Two positive, two negative, and the sd units are misleading in the one large
cell: predator_prey's random control has a standard deviation of 0.004, so a
0.098 absolute gap becomes "+22 sd". Read the absolute column.

This matters for how much weight the erasure result above can carry. Erasure and
patching ask different questions -- *is this subspace load-bearing* versus *does
moving it move the model the way the generative model says it should* -- and the
belief complement passes the first in 10 of 14 cells while passing the second in
2 of 4. A representation the model genuinely used as a belief should do both.
The honest summary is that the complement contrast is a real effect on the loss
and an inconsistent one on the logit direction.

This is the contrast phase 4 should have run. It does not rescue claim 3 as
stated -- the phase-4 numbers remain unusable -- but it replaces the null with a
positive result at the right depth and the right rank.

## 5.6b E6, the kick-magnitude sweep: claim 1 is not an artifact of the phase-1 threshold

Pendulum retrained at four multiples of the magnitude phase 1 chose, on Kaggle T4.

| kick scale | kick | eval loss | coupling `R^2(b|p)` | stream, pooled | stream, matched pairs | shuffled |
|---|---|---|---|---|---|---|
| 0.50 | 0.75 | 1.7462 | **-0.002** | 0.398 | 0.542 | -0.007 |
| 0.75 | 1.12 | 1.7689 | 0.006 | 0.582 | 0.699 | +0.003 |
| 1.00 | 1.50 | 1.7975 | 0.023 | 0.612 | 0.713 | -0.003 |
| 1.50 | 2.25 | 1.8353 | 0.033 | 0.548 | 0.654 | -0.003 |
| 2.50 | 3.75 | 1.9127 | 0.051 | 0.538 | 0.455 | -0.005 |

Coupling rises monotonically with magnitude, by a factor of about 25 over the
range, and stays under 0.06 throughout. The residual-stream number stays between
0.40 and 0.61 at every magnitude and is not monotone. At half the chosen kick the
coupling is **exactly zero** and the stream still recovers the belief at 0.398
pooled and 0.542 matched.

So the gap claim 1 reports is a property of the representation, not a restatement
of the phase-1 separability threshold. C6 is closed.

**Limit of this sweep:** coupling never exceeds 0.051 within the usable magnitude
range, so the high-coupling regime is untested. Reaching it would need a kick
large enough to saturate or wrap the observable, at which point the kick would be
trivially detectable for the wrong reason, which is what the phase-1 threshold
was protecting against.

## 5.6c E2d, multi-step rollout: the belief buys nothing at longer horizons

The strongest form of the causal question. Erase the belief subspace, roll the
model forward k tokens, and compare the resulting predictive law against the
exact HMM law for the same prefix. Marginals are Rao-Blackwellised -- 32 sampled
continuations, averaging the *distribution* at step k rather than the samples --
so the estimator noise is far below the effect being looked for. The reference is
a random subspace of the same rank, with the same rollout seed on both arms.

Excess KL over the matched-rank random control, at each depth's own erasure rank.
Negative means erasing the belief hurt the k-step law **less** than erasing a
random subspace of the same size.

| system | depth | rank | intact KL (k=1) | k=1 | k=2 | k=4 | k=6 |
|---|---|---|---|---|---|---|---|
| pendulum | resid_post_0 | 62 | 0.0051 | -0.093 (-1.2sd) | -0.142 (-1.8) | -0.118 (-1.7) | -0.050 (-0.6) |
| pendulum | resid_post_2 | 49 | 0.0051 | -0.120 (-7.2sd) | -0.139 (-8.6) | -0.176 (-6.8) | -0.149 (-5.1) |
| predator_prey | resid_post_1 | 64 | 0.0019 | -0.044 (-2.7sd) | -0.056 (-2.9) | -0.064 (-2.5) | -0.051 (-1.7) |
| predator_prey | resid_post_2 | 65 | 0.0019 | -0.082 (-2.8sd) | -0.116 (-2.9) | -0.159 (-3.0) | -0.172 (-3.0) |
| sphere | resid_post_0 | 29 | 0.0743 | **+0.072 (+9.5sd)** | +0.083 (+8.8) | +0.110 (+8.4) | +0.124 (+6.6) |
| sphere | resid_post_2 | 17 | 0.0743 | -0.007 (-2.7sd) | -0.011 (-2.8) | -0.015 (-2.4) | -0.020 (-2.4) |
| double_pendulum | resid_post_0 | 38 | 0.0880 | -0.013 (-0.4sd) | -0.016 (-0.8) | -0.022 (-1.3) | -0.021 (-0.9) |
| double_pendulum | resid_post_2 | 17 | 0.0880 | **+0.055 (+5.7sd)** | +0.063 (+6.3) | +0.061 (+5.9) | +0.044 (+7.0) |

**Where the belief is well decoded, erasing it is cheaper than erasing noise, and
the gap widens with the horizon.** Pendulum and predator_prey are negative at
every depth and every k, most sharply at `resid_post_2` where the probe is
strongest. This is the same pattern as the loss-based ablation in 5.1, now with
the failure mode that most plausibly hid an effect -- a myopic single-token
readout -- removed. It does not.

**The two positives are in the two systems where the belief is barely there.**
Sphere at `resid_post_0` (+9.5 sd) and double_pendulum at `resid_post_2`
(+5.7 sd) are the same two systems whose belief probes score 0.074 and 0.137, and
double_pendulum is the only system that cleared the variance-matched null in 5.1.
Two readings are available and this experiment does not separate them. Either a
small, genuinely used belief representation exists in exactly the systems where a
linear probe reads it worst, or -- more likely -- the erasure in those systems is
removing something else the model needs, and the belief basis is merely the
vehicle. Note that their **intact** KL is 15 to 45 times the pendulum's (0.074
and 0.088 against 0.0051), so the models whose rollouts are already far from the
exact law are the ones with room for an erasure to look damaging.

The horizon result is the part worth keeping. If the belief were a forward model
the network consults to predict several tokens ahead, the cost of removing it
should grow with k. In the two systems where it is decodable, the cost is
negative and grows *more* negative with k.

## 5.7 E5: coupling does order emergence

*This section reverses what the pilot said.* The pilot ran one system (pendulum),
one target family, and scored emergence at an **absolute** learned-gain threshold.
All three of those were wrong, and fixing them flips the sign.

**Two families, because one is not a control.** The first family built each target
by interpolating a random functional of the belief toward `p(next)`. Every member
is an arbitrary functional the model has no reason to build, so a null there says
little. The second family interpolates from the *actual* belief marginal instead:
at `alpha = 0` the target IS `action_lag0`, and `alpha` walks it toward `p`.

**Absolute thresholds are confounded.** Higher-`alpha` targets do not merely emerge
sooner, they end higher: in predator_prey the `alpha = 1.0` target reaches a final
learned gain of 0.291 and `alpha = 0.0` reaches 0.101. A target with three times
the asymptote crosses a fixed gain of 0.05 sooner almost mechanically. The
confound-free emergence time is the crossing at a **fraction of each target's own
final gain**.

Spearman(coupling, log emergence time); negative means higher coupling emerges
earlier. `n` is the targets clearing `learned_gain > 0.10`.

| system | frac | random family | belief family | pooled |
|---|---|---|---|---|
| pendulum | 0.50 | -0.49 (8) | -0.68 (7) | -0.56 (15) |
| pendulum | 0.75 | -0.71 (8) | -0.96 (7) | -0.83 (15) |
| predator_prey | 0.50 | -0.97 (14) | -1.00 (7) | -0.98 (21) |
| predator_prey | 0.75 | -0.95 (14) | -1.00 (7) | -0.94 (21) |
| sphere | 0.50 | -0.21 (9) | -1.00 (4) | -0.35 (13) |
| sphere | 0.75 | -0.53 (9) | -0.80 (4) | -0.56 (13) |
| double_pendulum | 0.50 | -0.88 (9) | -1.00 (5) | -0.87 (14) |
| double_pendulum | 0.75 | -0.97 (9) | -1.00 (5) | -0.95 (14) |

**Every cell is negative**: 24 of 24 across four systems, two families and two
normalisations. The belief family, the one anchored at a quantity the model
actually represents, is at or near -1.00 in three of four systems. Claim 4's
direction is supported.

Two limits keep this from being decisive. The sphere's random family is weak
(-0.21 at frac 0.50), and the sphere is the system where nothing else is learned
either. And the design is correlational within a family: `alpha` moves coupling,
but it also moves how much of the target is a linear image of `p`, and those are
not separable by construction -- a target more nearly equal to `p` is both more
coupled and more nearly free. So the honest statement is that **coupling to
`p(next)` predicts emergence order, and this experiment cannot show it causes it.**

The real-target couplings that motivated claim 4 are consistent with it but do not
sit on the fitted line. In pendulum the metric couples at 0.153 and crosses at
0.4M while the belief couples at 0.015 and crosses at 3.7M -- the right order. In
sphere the belief couples at 0.019 and is never learned. But the pendulum's own
synthetic targets at coupling 0.025 cross between "never" and 2.4M, a spread wider
than the gap being explained, so coupling alone does not fix the timing.

V3 gives a second variable that is not the same thing: a causal bag of the last 8
tokens recovers **93%** of the pendulum's metric R^2 but only **44%** of its
belief R^2. How directly the recent observation window reveals a quantity is a
distinct axis from how much that quantity overlaps the next-token distribution,
and this phase did not separate them.
