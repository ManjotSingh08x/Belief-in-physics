# Phase 5 -- methods

What each experiment computes and what would falsify it. No results here; those
are in `PHASE5-RESULTS.md`. Phase 1-4 definitions are in `METRICS.md` and are
not repeated.

Phase 5 exists because a review of phases 1-4 found three defects that make the
earlier headlines unreadable rather than merely uncertain, and about twenty
confounds that change what the numbers mean. Every script below is either a
repair, a validity check that can fail, or an experiment with its falsifier
stated up front.

## Scripts

| script | stage | contents |
|---|---|---|
| `07_validity.py` | 5.0 | V1 impossible probe, V4 Bayes floor, C12 delta-R^2 per layer, C19, C21 |
| `08_ablation_grid.py` | 5.1 | R1 every depth x checkpoint, R2 three nulls, R3 rank ladder, R5 resample, R7 N=20 |
| `09_probe_semantics.py` | 5.2 | V2 belief-or-label, V3 token-window baseline, V5 MLP ceiling, R6 nonlinear-after-erasure, C16, C17, V6 |
| `10_myopic.py` | 5.3-5.4 | E1a ceiling, E1b matched pairs, E1c horizon curve |
| `11_causal.py` | 5.5-5.6 | E2a decodability horizon, E2b position-restricted, E2c patching, R4 complements |
| `13_rollout.py` | 5.6 | E2d rollout divergence against the exact k-step law |
| `12_emergence.py` | 5.7 | E5 matched-target coupling vs emergence |
| `03_train_branch.py` | 5.7 | seeds and the kick sweep, via `SEED`, `KICK_SCALE`, `TAG` |
| `scripts/make_phase5_figures.py` | - | one figure per question |

New library code: `physics/myopic.py` (exact k-step predictive laws, Bayes
floor, conditional-variance ceiling), `models/bootstrap.py` (sequence-level
resampling), `models/probe_extra.py` (MLP probe, token-window features, Helmert
simplex coordinates, entropy stratification), and the intervention primitives
appended to `models/ablation.py`.

## The three defects being repaired

**F1, the ablation depth was not held fixed.** Phase 4 ran the causal test at one
depth per checkpoint, chosen by the argmax of `action_lag0` R^2 over five nearly
equal numbers. That argmax moves during training, and every sign flip in the
phase-4 ablation coincides with it moving. `08` sweeps depth instead.

**F2, the final residual stream has almost no power.** There the only path to the
loss is `logits = W_U LN(x)`, so at most `vocab_size - 1` of 128 directions can
move the next-token loss at all. Erasing directions selected to be uncorrelated
with the logits is then guaranteed to cost nothing. `11` excludes that depth.

**F3, no Bayes floor.** Cross entropy has a large irreducible offset, so a 1%
move in the raw loss is not a statement about convergence. `07` computes
`H_floor` exactly from the HMM and every loss is reported as `L - H_floor`.

## Definitions

**Bayes floor** `H_floor = E[H(p_t)]`, the expected entropy of the exact
predictive distribution, computed from the branch HMM with no model involved.
Reported alongside the realised form `E[-log p_t(x_{t+1})]`; the two agree in
expectation and the gap is finite-sample noise.

**`p^(k)`** the law of the token at `t + k` given the prefix through `t`, exact.
The belief is pushed forward without conditioning, splitting `n_actions` ways at
each segment boundary. Positions where `t + k` runs past the sequence are NaN,
never zero.

**Coupling** `R^2(target | p^(1))`, a linear fit from the optimal next-token
distribution to a target, on the same held-out split the probes use. This is the
axis claim 4 is about, and it is measured rather than assumed.

**Conditional-variance ceiling** `1 - E[Var(b | p)] / Var(b)`, the best possible
predictor of the belief from `p` of any functional form. Estimated by greedy
leader clustering in total variation: points inside a cluster have near-identical
`p`, so within-cluster variance of the belief is variance no function of `p`
could explain. Biased upward at small radius (singletons have zero within-
variance) and downward at large radius (a fat cluster mixes different `p`), so
the radius is swept and the singleton fraction reported. A ceiling estimated on a
system whose singleton fraction is near 1 is not usable.

**Impossible probe (V1)** a probe fitted to `y - b`, the one-hot true action minus
the exact posterior over it. The model sees the same tokens the filter sees, so
it cannot know the filter's own error and the held-out R^2 must be 0. Any
positive value invalidates every probe number in the project.

**Token-window baseline (V3)** ridge from a one-hot bag of the last
`steps_per_segment` observations plus the position in segment. For the pendulum
that is 136 features against `d_model = 128`. Causal by construction, so it
competes on the same information the model has.

**MLP probe (V5)** two hidden layers, standardised inputs and targets, early
stopping on a validation slice of the training rows. Separates "not represented"
from "not linearly represented".

**Helmert simplex coordinates (C17)** an orthonormal basis of the tangent space of
the `(A-1)`-simplex. A block of `A` marginals has `A - 1` degrees of freedom, so
scoring `A` dependent columns is not the same as scoring the free ones. The map
is an isometry, so distances are preserved.

**Sequence bootstrap (V6)** 1000 resamples of the held-out *sequences*, never of
positions. Positions inside one sequence share a belief history, so the
effective n is ~154, not ~10k.

**Three ablation nulls (R2)** isotropic random subspaces of matched rank;
top-`r` principal directions, the most destructive rank-`r` deletion available;
and a variance-matched PCA control, the fewest principal directions carrying as
much variance as the erasure basis does. The variance fraction each basis spans
is reported so the reader can see which comparison is fair. A fitted basis sits
in high-variance directions, so the isotropic null is the easiest of the three.

**Resample ablation (R5)** replaces the subspace component with another
sequence's at the *same position*, rather than with the pooled mean. Mean
ablation over pooled positions also deletes the phase-conditional mean, which is
a real signal in a segmented process.

**Complement erasure (R4)** `belief - metric` and `metric - belief` at matched
rank. When the first principal angle between two bases is a few degrees they are
not separate objects and no single-group ablation is about one group.

## Experiments and their falsifiers

**E1a, the ceiling.** If the ceiling is near the linear number, claim 1 is an
information statement. If it is far above, most of the reported gap is linearity
rather than orthogonality and claim 1 must be restated as "not *linearly*
available from `p`".

**E1b, matched pairs.** Nearest-neighbour match positions in `p`-space with
`TV < eps` but `||db|| > delta`, then regress the residual-stream difference on
the belief difference, split by sequence, with the pair sign symmetrised and a
shuffled control. Holding `p` fixed within a pair differences out every myopic
component non-parametrically. *Falsifier:* paired R^2 near zero while pooled R^2
is high. *Power check reported first:* the joint distribution of `(TV, ||db||)`;
if the pairs that satisfy the TV constraint all have `||db|| ~ 0`, the design has
no power and that is a fact about the process, not a null.

**E1c, the myopic horizon.** `R^2(b | p^(1..K))` against K, plus the pairwise
joint `p(x_{t+1}, x_{t+2})` since marginals are not the joint. *This is the
deflationary reading:* if K = 3 reaches 0.9, next-token training at t+1..t+3 --
which attends to position t through the KV cache -- is a complete explanation for
the belief being present at t.

**E2a, decodability horizon.** R^2 of a linear head onto the token at t+k, before
and after erasing the belief. Reported as a property of the representation and
never as evidence of use: the belief is by construction the sufficient statistic
for t+k, so damage growing with k is entailed by the generative model alone.

**E2b, position-restricted ablation.** Corrupt the belief subspace at position t
only (and at a block of 4, since one position is a small perturbation), then read
the per-position loss at t+1..t+k, which are themselves untouched but attend to
t. Target positions sit at a fixed phase in the segment. *Falsifier:* damage
confined to position t, which would mean the belief is recomputed everywhere
rather than stored. That is an available and real negative.

**E2c, subspace patching.** Overwrite the belief component at position t' with
the one from a donor at the same position index, so the positional embedding and
the phase in segment are identical and only the history differs. Alignment is
measured against two references: the HMM counterfactual
`log(b_donor E) - log(b_recipient E)`, and the model's own full-transplant logit
difference, which does not require the model to match the optimum. Rank is swept
rather than fixed, because patching 60 of 128 directions transplants the donor's
computation wholesale and a *random* subspace then scores high for an
uninteresting reason. *Falsifier:* alignment at or below the matched-rank random
subspace. Ablation asks whether removing it hurts; patching asks whether changing
it moves the model where the interpretation says it should.

**E2d, rollout divergence.** Cut every sequence at the last position of a
segment, roll the model forward k steps autoregressively under an intervention,
and compare its k-step marginal to the exact `p^(k)` by KL. The model's marginal
is estimated Rao-Blackwellised: sample `k - 1` tokens, then average the
*distribution* at step k over replicas rather than histogramming sampled tokens.
Every arm uses the same rollout seed, so the arms differ by the intervention and
not by which continuations were drawn. Sampling error compounds with k in every
arm and any rank-r deletion costs something, so only `excess(k) = KL(ablated, k)
- mean KL(random, k)` is interpretable, with the control spread recomputed at
each k rather than carried over from k = 1.

**E5, coupling versus emergence.** Within one model, a family of targets of
matched width `n_actions` and matched scale,
`f_alpha = alpha * (emission columns) + (1 - alpha) * base`. The `alpha = 1` end
is the next-token distribution restricted to those bins, so its coupling is high
by construction; the `alpha = 0` end has coupling near 0. The test is a Spearman
correlation across ~20 targets inside one model, which can fail, rather than four
ordinal comparisons across four runs, which cannot.

Two things that a first pass got wrong and that the result depends on:

*Two families, because one of them is not a control.* With `base` a random
functional of the belief, every member is an arbitrary quantity the model has no
reason to build, and a null says little. The second family sets `base` to the
**actual belief marginal**, so `alpha = 0` IS `action_lag0` and the family passes
through the place the real quantities live. Both families are run and reported
separately; agreement between them is the evidence.

*Emergence is read at a fraction of each target's own final learned gain, not at
an absolute threshold.* Learned gain is `R^2(t) - R^2(untrained)` at the same
seed. Higher-`alpha` targets do not only emerge sooner, they end higher -- a
target whose asymptotic gain is 0.29 crosses an absolute 0.05 before one whose
asymptote is 0.10, for reasons that have nothing to do with timing. Normalising
by the target's own asymptote removes that confound. This does not reintroduce
the "half of final" problem the belief has, because the synthetic targets do
plateau within the window and any that do not are excluded by
`learned_gain > 0.10`. Both scorings are reported; only the fractional one is
read.

**E6, the kick-magnitude sweep.** Phase 1 chose each `kick` as the *smallest*
value clearing its separability threshold, so claim 1's headline coupling was set
by that choice. Retrain the pendulum at four multiples and plot `R^2(b | p)` and
`R^2(b | resid)` against magnitude. If the residual-stream number stays high
while the coupling rises, claim 1 is a property of the representation; if they
move together, it is a restatement of the phase-1 threshold.

**Seeds.** Three additional seeds per system, so the emergence ordering and every
level is reported as a range rather than as an n = 1 observation.

## Running it

CPU, in order; each stage writes one JSON into `OUTPUT_DIR`:

```bash
uv run python experiments/07_validity.py
uv run python experiments/08_ablation_grid.py
uv run python experiments/09_probe_semantics.py
uv run python experiments/10_myopic.py
DEVICE=cpu uv run python experiments/11_causal.py
uv run python experiments/13_rollout.py
uv run python experiments/12_emergence.py
uv run python scripts/make_phase5_figures.py
```

The GPU work is retraining only, and runs on Kaggle T4 through
`kernel-seeds` and `kernel-kicks`, which train and then run the analysis in the
same session so only JSON comes back. Env knobs: `OUTPUT_DIR`, `SYSTEMS`,
`N_EVAL`, `N_CONTROL`, `MAX_ERASURE_RANK`, `DEVICE`, `SEED`, `KICK_SCALE`, `TAG`.
