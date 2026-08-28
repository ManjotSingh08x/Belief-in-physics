# Simulator interface + 4 physical systems (Phase 1)

## Context

The `belief-geometry` project tests whether a next-token predictor's residual
stream linearly encodes the *belief state* over a hidden physical system, and
whether conserved quantities fall out of that geometry. It currently has
exactly one hard-coded system (planar Kepler), one observation channel, and an
approximate belief from a bootstrap particle filter.

This work builds the generalised data-generation layer that the next round of
experiments needs, in a separate repo (`Finding-belief-in-physics`, bucket
`dsg`):

- a **factory** that turns a config dict into a simulator,
- **four physical systems** (pendulum, predator-prey, ball on a sphere, double
  pendulum) rather than one,
- **perturbation tokens**: some tokens are *actions* applied to the system, not
  observations, making this a controlled HMM / POMDP rather than a passive HMM,
- an **explicit discrete HMM** so the ground-truth belief is *exact* (forward
  algorithm) instead of particle-filter-approximate.

The intended outcome of Phase 1 is: four working simulators with exact beliefs,
visual evidence that each one behaves physically, and a quantitative check that
the perturbations are actually detectable. Training and probing are Phase 2 and
3 and are only sketched here.

**Scope of this plan: Phase 1 only.** No transformer training, no probes.

## Decisions already taken

| Decision | Choice |
|---|---|
| `simulate_batch(num_simulations, N, M)` | `N` = sim steps between perturbations; `M` = perturbations per sequence |
| Belief representation | Explicit discrete HMM, exact forward algorithm |
| Emission binning | Coarse, equal-measure bins (~500-1000 max), not the 361x181 / 361x361 originally specified |
| Repo | Already created; bucket `dsg`. No repo/remote work in this plan. |

## The one design decision everything rests on

**The discrete HMM is the generator, not an approximation of it.**

The tempting version is: integrate the ODE, then bin the result. That makes the
"exact" forward-algorithm belief a belief in the *wrong* model — the discrete
surrogate — so it is not ground truth for anything.

Instead, build the transition matrix from the true dynamics by Ulam's method
(Monte-Carlo transfer-operator discretisation), then **sample the data from
that matrix**. The belief the forward algorithm returns is then exactly the
posterior of the process that actually generated the tokens, while the kernel
itself is physically faithful because it came from integrating the real ODE.

This mirrors the principle already stated in `belief-geometry`'s
`observation.py`: define the emission *as* the categorical law that both the
generator and the filter use, so any mismatch in the results is a finding
rather than a discretisation artefact.

## Sequence structure

Total simulated steps per sequence = `M * N`. Observations are emitted every
`K` steps. `N % K == 0` is required and validated.

```
[ P_0 | o_1 o_2 ... o_{N/K} | P_1 | o_1 ... o_{N/K} | P_2 | ... ]
   \____________ segment 0 ___________/ \____ segment 1 ____/
```

- `M` perturbation tokens, `M * (N // K)` observation tokens.
- Sequence length in tokens = `M * (1 + N // K)`.
- Vocabulary = `n_obs_bins + n_actions`; action tokens occupy a disjoint block
  appended after the observation bins (same trick as `AngleChannel.blank_token`
  in the existing repo).

An action token is **information-free about the latent state** — the model is
told which action was applied, so observing it triggers no Bayes update, only a
deterministic push `b <- b P_a`. This preserves the pure-pushforward steps that
the downstream conservation analysis reads, exactly as `BLANK` does today.

Belief recursion:

| event | update |
|---|---|
| perturbation token `a` | `b <- b P_a` (no Bayes; information-free) |
| K steps of free flow | `b <- b T` |
| observation token `o` | `b <- normalize(b * E[:, o])` |

## Files

Package `src/beliefphysics/` (name changeable). Kept pure numpy/scipy so it has
no torch dependency, matching the layering in `belief-geometry`.

| File | Contents |
|---|---|
| `dynamics.py` | `PhysicalSystem` Protocol: `state_dim`, `flow(z, dt, substeps)`, `apply_action(z, a)`, `n_actions`, `energy(z)`, `observable(z)`, `metrics(z)` |
| `grid.py` | `LatentGrid`: per-dim bin edges, `index(z)`, `centers`, `sample_in_bin(rng, idx, n)`. Handles periodic dims (angle wrap) and equal-measure binning |
| `hmm.py` | `DiscreteHMM` frozen dataclass holding `T`, `{P_a}`, `E`, `metric_table`; builders; `sample_batch()`; `forward()` (exact belief) |
| `channels.py` | Emission binning per observable, incl. equal-area sphere cells |
| `factory.py` | `make_simulator(config: dict) -> Simulator`, name->builder registry, dict -> frozen dataclass conversion, validation |
| `separability.py` | Closed-form TV diagnostics for perturbation detectability |
| `viz.py` | The demonstration figures |
| `systems/pendulum.py`<br>`systems/predator_prey.py`<br>`systems/sphere.py`<br>`systems/double_pendulum.py` | One `PhysicalSystem` implementation each |
| `cache.py` | Transition matrices are expensive; cache to `cache/<config-hash>.npz` |

Config dicts are converted to nested frozen dataclasses **at the factory
boundary** and never passed around as raw dicts internally — `belief-geometry`
uses frozen dataclasses everywhere and has no dict-config precedent.

## Public interface

```python
config = {
    "system": {"name": "pendulum", "g": 9.81, "L": 1.0, "gamma": 0.15, "dt": 0.01},
    "hmm": {"latent_bins": (128, 96), "emission_bins": 64, "samples_per_bin": 200},
    "perturbation": {"kind": "velocity_kick", "magnitudes": [-0.8, 0.8], "include_noop": True},
    "K": 10,
    "seed": 0,
}

sim = make_simulator(config)
ep = sim.simulate_batch(num_simulations=1024, N=50, M=8)
```

`simulate_batch` returns a frozen `Episodes` dataclass mirroring
`belief-geometry`'s `data.py`:

| field | shape | role |
|---|---|---|
| `tokens` | `(n, L)` int | **all the model ever sees** |
| `beliefs` | `(n, L, n_latent)` float32 or `None` | exact posterior, analysis only |
| `is_prediction` | `(n, L)` bool | True on pure-pushforward steps |
| `is_action` | `(n, L)` bool | True on perturbation tokens |
| `latent` | `(n, L)` int | true latent bin index |
| `metrics` | `(n, L, d_metric)` float | ground-truth metric at the true state |

`beliefs` is `(n, L, n_latent)` and for the double pendulum `n_latent` is
~330k, so it is gated behind `with_beliefs=False` by default and only computed
for a small held-out evaluation set — the same cheap-path/expensive-path split
`data.py::generate` already uses.

## Transition-matrix construction

For each latent bin `i`: draw `samples_per_bin` points uniformly inside the
bin, integrate the ODE forward `K` steps, histogram the destination bins.
Row `i` of `T` is that histogram, normalised. Vectorised across all bins,
chunked to bound memory. Probabilities below a floor are pruned and the row
renormalised, to keep `T` sparse.

Only **one** `T` is needed (free flow, shared by every segment). The
perturbations are instantaneous, so each `P_a` is a cheap deterministic remap:
push each bin centre through `apply_action` and record the destination. This is
the difference between building 1 expensive matrix and `n_actions` of them.

## Grid sizing

Latent grids are deliberately **finer than the emission bins** so the posterior
stays genuinely spread — if the observation pinned the latent state the belief
would collapse to a delta and "linear functional of the belief" degenerates
into "nonlinear function of the state", the one case this whole line of work
cannot use.

| system | latent state | latent grid | emission | actions |
|---|---|---|---|---|
| pendulum | `(theta, omega)` | 128 x 96 = 12k | theta, 64 bins | `+dv`, `-dv`, noop |
| predator-prey | `(log x, log y)` | 96 x 96 = 9k | ratio `x/(x+y)`, 64 bins | `+/-` prey, `+/-` pred, noop |
| ball on sphere | `(lat, lon, v_t1, v_t2)` | 24 x 24 x 12 x 12 = 83k | ~512 equal-area cells | 3 tangent dirs, noop |
| double pendulum | `(th1, th2, w1, w2)` | 24^4 = 332k | theta2, 128 bins | kick b1, b2, both, noop |

**Known risk — the double pendulum.** It is chaotic, so destinations spread and
`nnz` per row approaches `samples_per_bin`, which could reach ~66M nonzeros
(~500 MB in float32). Mitigations, in order: lower `K`, prune harder, drop the
grid to `20^4 = 160k`. Build it last and measure before tuning. If it stays
intractable at usable resolution, it degrades to the particle-filter path
rather than blocking the other three.

## Physics per system

| system | dynamics | damping | metric (probe target) |
|---|---|---|---|
| pendulum | `th'' = -(g/L) sin th - gamma th'` | viscous `-gamma*omega` | `omega` |
| predator-prey | `x' = ax - bxy - a x^2/kappa`, `y' = dxy - cy` | logistic self-limitation on prey (spirals to fixed point) | `dx/dt`, and `dx / dy` |
| ball on sphere | geodesic flow on S^2 with tangential friction | sliding friction `-mu * |v| * vhat` | tangential velocity vector |
| double pendulum | standard Lagrangian, RK4 | viscous on both joints | `(omega1, omega2)` |

Integrators: velocity-Verlet where the system is separable and undamped (so
energy error stays bounded rather than drifting, per `systems.py`'s reasoning),
RK4 otherwise. Damping breaks symplecticity anyway, so RK4 is honest for the
damped systems.

Predator-prey and the sphere run on transformed coordinates (`log` populations,
equal-area sphere cells) so bins carry roughly equal prior mass instead of the
grid wasting most of its resolution on regions the system never visits.

### Metric selection

Each system exposes a `metrics` dict, defaulting to the metric specified in the
request. Because the HMM is discrete, the belief-space metric is *exactly*
linear: `E_b[m] = b @ metric_table`, where `metric_table` is
`(n_latent, d_metric)` evaluated at bin centres. That linearity is the reason
a linear probe is the right instrument, and it is exact here rather than
approximate.

A `rank_metrics()` diagnostic scores candidates on (a) variance across the
belief trajectory, (b) predictability from observation history, (c)
conditioning of the linear map. The specified metric stays the default and is
only replaced when a candidate wins on evidence, with the numbers recorded.

## Perturbation separability

The requirement that a perturbation's effect be distinguishable from an
unperturbed trajectory becomes a computation, done **in closed form from the
HMM matrices** — no Monte-Carlo:

```
p_a(t) = b0 @ P_a @ T^t @ E        # predictive obs distribution after action a
p_0(t) = b0 @ T^t @ E              # same, no-op
```

Thresholds, averaged over a set of initial beliefs `b0`:

- `TV(p_a(1), p_0(1)) >= 0.15` — visible at the first post-kick observation
- `mean_t TV(p_a(t), p_0(t)) >= 0.10` over the segment — not washed out by damping
- `TV(p_a, p_b) >= 0.10` for distinct actions — the actions are distinguishable
  from *each other*, not merely from the no-op

`choose_magnitude()` sweeps perturbation magnitude and picks the **smallest**
value clearing all thresholds — smallest, so the kick does not saturate or wrap
the observable and thereby become trivially detectable for the wrong reason.
The chosen magnitudes are then frozen into the per-system default configs, with
the sweep curve saved as evidence.

## Visualisation

`viz.py` produces one multi-panel figure per system, written to `figures/`:

1. **Phase portrait / trajectory** with perturbation events marked.
2. **Token raster** (time x bin), observation and action tokens coloured
   differently.
3. **Belief heatmap** — exact posterior over latent bins vs time, true state
   overlaid. This is the panel that demonstrates the HMM is correct: the true
   state must stay inside the high-probability region, and the posterior must
   visibly sharpen at observations and spread during free flow.
4. **Separability**: TV vs magnitude sweep, plus the pairwise action TV matrix.
5. **Energy/damping sanity**: energy decaying monotonically under damping,
   conserved when damping is off.

`matplotlib` is a new dependency and goes in a `viz` extra, not the base
install — base stays numpy/scipy only.

## Verification

Run in order; each is cheap and pure-numpy except the last.

**Unit tests** (`tests/test_<module>.py`, one per module, mirroring the
existing repo's convention of private `_helper()`s and fixed integer seeds
rather than shared fixtures):

- `T` and `E` rows sum to 1 to `atol=1e-12`.
- Fine-grid limit: applying `T` to a delta at bin `i` concentrates near
  `flow(centre_i)` — the Ulam matrix really is the transfer operator.
- Forward algorithm matches brute-force enumeration over all latent paths on a
  toy 5-state / 3-observation HMM. This is the exactness proof.
- Belief normalises to 1 at every step of a long sequence.
- Undamped pendulum conserves energy over 10^4 steps to a fixed tolerance;
  damped energy is monotone decreasing.
- `N % K != 0` raises a clear error.
- `simulate_batch(n, N, M).tokens.shape == (n, M * (1 + N // K))`, and
  `is_action.sum(axis=1) == M`.
- Belief mean tracks the true state strictly better than the stationary prior
  (mean squared error comparison) — the filter is actually doing work.
- Each system's chosen perturbation magnitude clears all three separability
  thresholds.

**End-to-end**:

```bash
uv run --with pytest pytest -q
python scripts/build_hmms.py          # builds + caches all four; prints nnz, build time, memory
python scripts/validate_simulators.py # separability report + metric ranking, writes results/phase1.json
python scripts/make_figures.py        # writes figures/<system>.png
```

Phase 1 is done when all four figures show a posterior that tracks the true
state, `results/phase1.json` records every system clearing its separability
thresholds, and the double-pendulum build cost is measured rather than
estimated.

## Deferred

- **Phase 2**: transformer training (4 layers, 1 head, `d_model=128`) on the
  generated tokens, plus the random-init probe control — fit probes on an
  untrained model first; if the correlation survives there, the baseline is
  broken and the trained-model result means nothing. `belief-geometry`'s
  `model.py` (`TrainConfig`, `build`, `train`, `residual_stream`) and
  `probe.py` (`fit_probe`, `spectrum_on_activations`, `shuffled_control`) are
  already generic over `(activations, is_prediction, targets)` and should be
  reused unchanged rather than rewritten.
- **Phase 3**: belief-state extraction probes and the metric comparison across
  all four systems.
