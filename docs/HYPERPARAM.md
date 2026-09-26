# Every adjustable parameter

One place for everything that can be turned, what it does, and what it is set to now.
`env` marks a parameter that can be changed without editing code.
Derived values are listed where they are constrained by others, so that changing an input does not silently violate an assumption elsewhere.

Update this file in the same commit as any change to a value here.

## 1. Hidden process, the Mess-4 chain

`physics/messk_configs.py:CHAIN`, `physics/messk.py:MessKProcess`.

| Parameter | Value | What it does |
|---|---|---|
| `n_states` | `4` | Number of moods and letters, K |
| `alpha` | `0.7` | Probability a mood emits its own letter |
| `stay` | `0.7` | Probability a mood stays put on the next tick |
| `x` *(derived)* | `0.1` | `(1-stay)/(K-1)`, probability of each specific other mood |

The belief memory is about seven letters at these values, which is what `03_myopic.py` sizes its token windows against.

## 2. Observation channel

`physics/messk_configs.py:DRIVER`, `physics/messk.py:MessDriven`.

| Parameter | Value | What it does |
|---|---|---|
| `m` | `16` | Chain ticks per sequence |
| `n_steps` | `10` | Physics integration steps per tick |
| `obs_bins` | `181` | Bins per observation channel. An int, or a tuple to bin several channels into one token |
| `n_obs` *(derived)* | `181` | `prod(obs_bins)`, the vocabulary size |
| `seq_len` *(derived)* | `160` | `m * n_steps`, tokens per sequence |
| `delta_v` | per system | Scale of the impulse a letter applies |
| `dt` | per system | Integration timestep |

There is no observation noise. The only stochasticity in a sequence is the letter draw, which is why `07_ceiling.py` can compute the exact posterior by replaying candidate prefixes.

Every system exposes a second observation channel, listed in its own table below. `obs_bins = 181` bins only the first, which is what every committed run does. `obs_bins = (181, 181)` bins both and combines them in mixed radix, so one token names a cell of the product grid and the vocabulary is `32761`. The cost is occupancy: two correlated channels trace a curve through the grid rather than filling it, so the double pendulum at `(181, 181)` occupies about 18% of its cells, and the uniform-prediction loss rises from `ln(181) = 5.20` to `ln(32761) = 10.40` nats, which makes losses incomparable across bin settings.

Measured clipping at the committed defaults, over 256 sequences: pendulum `2.90%`, and zero for the other three. The pendulum's `theta` leaves its `+-pi/2` range, so those samples land on an edge bin and lose their value.

## 3. Physical systems

Every system defines four distinct, non-zero actions whose sum is zero, so no system receives a directional drift from a uniform letter distribution.

### Pendulum

`delta_v = 0.5477225575051661` (sqrt of 0.3), `dt = 0.02`. Channels: `theta`, `omega`. Release: `theta0 = 0.0`, `omega0 = 1.0`.

| Parameter | Value | What it does |
|---|---|---|
| `g` | `9.8` | Gravity |
| `length` | `1.0` | Pendulum length |
| `gamma` | `1.2` | Viscous damping on omega, tuned so only 2.45% of tokens hit the edge |
| `omega_max` | `8.0` | Angular-velocity clamp |
| `obs_range` | `(-pi/2, pi/2)` | Binned range of theta, one bin per degree |
| `actions` | `scale * [-3, -1, +1, +3]` | Variance-matched ladder `{-1.64, -0.55, +0.55, +1.64}` |
| `initial_state` | `theta=0, omega=1` | Fixed release, identical for every sequence |
| `metric_names` | `("omega",)` | Physical probe target |

### Predator-prey
Channels: `prey_share`, `log_prey`. Release: `x0 = 3.0`, `y0 = 2.0`.

`delta_v = 0.35`, `dt = 0.05`.

| Parameter | Value | What it does |
|---|---|---|
| `a` | `1.0` | Prey growth rate |
| `b` | `0.6` | Predation rate |
| `c` | `0.8` | Predator death rate |
| `d` | `0.4` | Predator growth per prey eaten |
| `kappa` | `5.0` | Prey carrying capacity, the term that turns closed orbits into an inward spiral |
| `log_bound` | `3.5` | Clamp on the log-populations |
| `obs_range` | `(0.0, 1.0)` | Binned prey share `x/(x+y)` |
| `actions` | `scale * [prey down, prey up, predator down, predator up]` | Fixed impulse in log space, so a fixed multiplicative factor on the population |
| `initial_state` | `log 3, log 2` | Fixed populations |
| `metric_names` | `("dx_dt", "dy_dt")` | Physical probe targets |

### Spherical pendulum
Channels: `theta`, `psi`. Release: `theta0 = 0.6`, `psi_dot0 = 2.0`.

`delta_v = 0.12`, `dt = 0.04`.

| Parameter | Value | What it does |
|---|---|---|
| `g` | `9.8` | Gravity |
| `length` | `1.0` | Rod length |
| `gamma` | `0.35` | Damping on theta only; psi is undamped so angular momentum about the vertical is conserved |
| `rate_max` | `6.0` | Angular-rate clamp |
| `THETA_MIN` | `0.05` | Guard on the `cot(theta)` singularity at the bottom |
| `THETA_MAX` | `pi - 0.15` | Guard at the top |
| `obs_range` | `(0.05, 1.25)` | The theta band actually visited, not the full coordinate range |
| `actions` | `scale * [north, south, west, east]` | Tangent-plane impulses |
| `initial_state` | `theta=0.6, psi=0, theta_dot=0, psi_dot=2.0` | A conical swing, off the bottom and already rotating |
| `metric_names` | `("v_meridional", "v_azimuthal")` | Physical probe targets |

`delta_v` was reduced from `0.35` to `0.12` because the larger value clipped 2.18% of steps at the theta floor and 5.1% at the rate ceiling.
At `0.12` neither clip fires and all four action pairs still separate.

### Double pendulum
Channels: `theta1`, `theta2`. Release: `th1_0 = 0.9`, `th2_0 = -0.4`, `w1_0 = w2_0 = 0.0`.

`delta_v = 1.2`, `dt = 0.2`, `obs_bins = (50, 50)` (mixed-radix tokens `theta1 * 50 + theta2`, vocabulary `2500`). Damping scales dynamically via $k \cdot \gamma^*$ ($k \in \{0.3, 0.5, 0.7\}$); at the contractive operating point ($\gamma \approx 0.65$): $\lambda = -0.21/\text{s}$, clipping $0.0\%$, Bayes gap $= 0.231\text{ nats}$, Action Signal Gap $= 1.21$, $100\%$ stability pass rate across seeds.

| Parameter | Value | What it does |
|---|---|---|
| `g` | `9.8` | Gravity |
| `l1`, `l2` | `1.0`, `1.0` | Arm lengths |
| `m1`, `m2` | `1.0`, `1.0` | Bob masses |
| `gamma1`, `gamma2` | `2.5`, `2.5` | Viscous joint friction; screened optimal range [2.5, 3.0] with 3.0 yielding 100% pass rate, Bayes gap 1.96 nats, and lambda=-0.81/s |
| `omega_max` | `10.0` | Rate clamp |
| `joint1_action_gain` | `2.0` | Amplifies joint-1 kicks so all four actions give distinct observation sequences |
| `obs_ranges` | `((-pi, pi), (-pi, pi))` | Binned range of (theta1, theta2) |
| `initial_state` | `(0.9, -0.4, 0, 0)` | Asymmetric release, so kicks to either joint are observable |
| `metric_names` | `("omega1", "omega2")` | Physical probe targets |

## 4. Model architecture

`models/transformer.py:ModelConfig`, set from `experiments/01_train.py`.

| Parameter | Code default | Current run | What it does |
|---|---|---|---|
| `n_layers` `env NUM_LAYERS` | `4` | `8` | Transformer blocks |
| `d_model` `env EMBED_DIM` | `128` | `256` | Residual stream width |
| `n_heads` `env NUM_HEADS` | `1` | `4` | Attention heads |
| `d_mlp` `env D_MLP` | `512` | `1024` | MLP hidden width |
| `seed` `env SEED` | `0` | `0` | Initialisation seed |
| `vocab_size` *(derived)* | `181` | `181` | Equals `n_obs` |
| `n_ctx` *(derived)* | `160` | `160` | Equals `seq_len` |
| non-embedding parameters *(derived)* | `786,432` | `6,291,456` | `n_layers * (4 d_model^2 + 2 d_model d_mlp)` |

The model is pre-LayerNorm with default PyTorch initialisation and no depth-scaled residual init.

## 5. Training

`models/train.py:TrainConfig`, `experiments/01_train.py`.

| Parameter | Value | What it does |
|---|---|---|
| optimiser | `Adam` | Plain Adam, not AdamW |
| `weight_decay` `env` | `0.0` | Zero, because Adam with a non-zero decay is coupled L2, which is neither Adam nor AdamW |
| `learning_rate` `env` | `1e-3` | Peak learning rate |
| schedule | cosine to zero | Defined over the full `total_tokens`, so a mid-run checkpoint is not an annealed model |
| `warmup_frac` | `0.02` | Linear warmup fraction |
| `grad_clip` | `1.0` | Global gradient-norm clip |
| `batch_size` `env` | `128` | Sequences per step, so 20,480 tokens per step |
| `total_tokens` `env TOTAL_TOKENS` | `1_000_000_000` | Was `500_000_000` for the 4x128 runs |
| `log_every` | `100` | Steps between eval-loss records |
| eval batch | `256` | Fixed held-out sequences used for eval loss |
| `CHECKPOINT_FRACTIONS` | `0.004, 0.01, 0.024, 0.06, 0.24, 0.5, 1.0` | Log-spaced, plus `0` for the random-init control |
| `TAG` `env` | `_d256l8` | Filename suffix separating runs that share one output directory |

The generator is unlimited and every step samples a fresh batch, so there is no finite corpus to memorise and no train-eval gap is expected.

## 6. Analysis

Shared across experiments: `OUTPUT_DIR` `env`, `CONFIGS` `env`, `ANALYSIS_COMMIT` `env`, `EVAL_SEED = 20260829`, `SPLIT_SEED = 0`.
No probe is ever scored on the rows it was fitted on, and any selection of depth, window or penalty happens on validation.

| Experiment | Parameters |
|---|---|
| `02_probe.py` | `N_EVAL=512` `env`, `TRAIN_FRAC=0.6`, `VALIDATION_FRAC=0.2`, `HEADLINE="belief"` |
| `03_myopic.py` | `N_EVAL=384` `env`, `WINDOWS=1,2,4,8,16,32,64,80` `env`, `SPARSE_ALPHAS=0.1,1,10,100,1e3,1e4` |
| `04_emergence.py` | `N_EVAL=384` `env`, `FRACTIONS=0.5,0.75` `env` (fraction of final gain that defines a crossing), `MIN_GAIN=0.05` |
| `05_geometry.py` | `N_EVAL=256` `env`, `MAX_POINTS=2500` `env`, `PROJECTION=tetrahedron\|square` `env`, `FIGURE_DIR` `env`, six plot colours |
| `06_reference_probe.py` | `N_EVAL=6144` `env`, `N_DEPTHS=4` `env` (blocks the probe reads), `RIDGE_ALPHAS=1,10,100,1e3,1e4` `env`, `WINDOWS=10,20,40,80` `env`, `TOKEN_ALPHAS=1,10,100,1e3`, `TRAIN_FRAC=0.6`, `VALIDATION_FRAC=0.2`, `MOMENT_CHUNK=4096`, `TAG` `env`, `REPORT_NAME` `env` |
| `07_ceiling.py` | `N_SEQ=256` `env`, `MAX_BEAM=20000` `env` (cap on the consistency-filter beam) |

`N_EVAL` in `06` must keep rows-per-feature comfortably above one.
At `N_EVAL=6144` with `N_DEPTHS=4` and `d_model=256` the probe has 10,240 features and 5.76 rows per feature.

## 7. Infrastructure

| Parameter | Value | What it does |
|---|---|---|
| Kaggle `machine_shape` | `NvidiaTeslaT4` | Required; the default P100 is `sm_60`, which Kaggle's preinstalled torch cannot target |
| GPU sessions | 2 per account, 2 accounts | Four systems train concurrently |
| `OMP_NUM_THREADS` | `16` | Staging CPU threads for the analysis passes |

The Kaggle client must be run with IPv6 disabled.
This network resolves `www.kaggle.com` to a NAT64 address with no working gateway, and urllib3 blocks on it for the full TCP connect timeout on every call.

## 8. Interactive notebooks

`notebooks/simulator.ipynb` (physics and channel only, no torch) and `notebooks/transformer.ipynb` (train, snapshot, probe, compare).
Both build their process through `make_process`, so every parameter in this document is reachable from a free-text box with no slider bound.

`notebooks/explorer.ipynb` is the visual counterpart: sliders over every parameter, ten graph types, and a stability verdict.
It imports no torch either.
Its controls come from `physics/controls.py` and its panels from `physics/visualise.py`, so the UI is version-controlled rather than pasted into cells.
Each slider carries its own editable range boxes, so a value far outside the default neighbourhood is reachable without leaving the slider.

None of the three notebooks changes a default; they are views over the same code the experiments run.

### Stability thresholds

These decide the `stable` verdict in `physics/visualise.py::stability`.
They are reporting thresholds, not physics, and nothing in the training pipeline reads them.

| Threshold | Value | Why |
|---|---|---|
| `lyapunov` | `> 0.05 /s` fails | fitted from a twin started `1e-8` away, over the stretch before the twins saturate at 10% of the state scale; above zero the trajectory forgets its own initial condition |
| `clipped` | `> 1%` fails | the same 1% budget the per-system `obs_range` entries in section 3 were chosen against |
| `gap_free_mean` | `< 1e-3` fails | driven and free runs indistinguishable, so the letters left no trace to learn from |
| `used_bins` | `< 10%` of `n_obs` fails | most of the vocabulary never emitted |

Measured at the committed defaults, one sequence of `m=40` ticks at seed 3:

| System | lyapunov | clipped | bins used | driven-vs-free gap | verdict |
|---|---:|---:|---:|---:|---|
| pendulum | -0.579 | 0.00% | 105/181 | 1.250 | stable |
| predator-prey | -0.197 | 0.00% | 131/181 | 0.897 | stable |
| sphere | +0.081 | 0.00% | 47/181 | 0.703 | marginally chaotic |
| double pendulum | +1.579 | 0.00% | 166/181 | 7.379 | chaotic, as expected |

The double pendulum's positive exponent is the system, not a misconfiguration.
The sphere's is small and its bin use is the lowest of the four, both consequences of `obs_range = (0.05, 1.25)` being far wider than the band theta visits - see the sphere entry in section 3.
