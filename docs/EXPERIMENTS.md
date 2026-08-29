# Mess-4 belief experiments

## Question

Does an ordinary next-token transformer recover the exact predictive belief over a hidden four-state Markov chain when it sees only the observations of a physical system driven by that chain?

## Generator

Each sequence uses the causal graph `mood -> letter -> physical action -> observation tokens`.
The transformer sees only observation tokens.
The exact analysis target is

\[
b_t = P(M_{t+1}\mid L_1,\ldots,L_t).
\]

The four-state transition and emission matrices both have diagonal probability `0.7` and off-diagonal probability `0.1`.
The forward update is `b <- normalize(b @ (T * E[:, letter, None]))`.
The belief forgets its prior to total variation below `0.01` after approximately seven letters.
Its exact reachable geometry is a three-dimensional fractal inside a regular tetrahedron, shared by all four systems.

Each sequence has `16` chain ticks, `10` physics steps per tick, `160` observation tokens, and a vocabulary of `181` bins.
Every letter has a distinct non-zero action, and every four-action set sums to zero.

## Physical systems

| System | State | Four letter actions | Observable | Probe metrics | `dt` |
|---|---|---|---|---|---:|
| Pendulum | `(theta, omega)` | `delta omega = {-1.643, -0.548, +0.548, +1.643}` | `theta` in 181 bins over `[-pi/2, pi/2]` | `omega` | `0.02` |
| Predator-prey | `(log x, log y)` | prey down/up, predator down/up, magnitude `0.35` | prey share `x/(x+y)` | `(dx/dt, dy/dt)` | `0.05` |
| Spherical pendulum | `(theta, psi, theta_dot, psi_dot)` | north/south, west/east tangent impulses, magnitude `0.12` | polar angle `theta` | meridional and azimuthal velocity | `0.04` |
| Double pendulum | `(theta1, theta2, omega1, omega2)` | joint 1 `-/+2.4`, joint 2 `-/+1.2` | wrapped `theta2` | `(omega1, omega2)` | `0.01` |

The pendulum uses damping `gamma=1.2` to prevent the strong balanced kicks from accumulating at the angle-bin boundary.
The spherical action scale was selected so a seeded 1,024-sequence preflight produced no coordinate or token-boundary clipping while all action pairs remained distinguishable.
The double-pendulum joint-1 actions are larger because `theta2` observes joint 1 only indirectly.
Tests require every action to differ from an unperturbed trajectory and from every other action, both at the initial state and on typical visited states.

## Transformer and training

| Choice | Value |
|---|---:|
| Architecture | decoder-only causal transformer |
| Blocks | `4` |
| Heads | `1` |
| Residual width | `128` |
| MLP width | `512` |
| Batch size | `128` |
| Optimizer | AdamW |
| Learning rate | `1e-3` with warmup and cosine decay |
| Training budget | `500M` fresh tokens per system |
| Seed | `0` only |
| Objective | next-observation-token cross entropy |

Checkpoints are saved at `0`, `2M`, `5M`, `12M`, `30M`, `120M`, `250M`, and `500M` tokens.
No auxiliary belief loss, physical supervision, action token, or reseeded run is used.

## Experiments

### `01_train.py`

Trains one model per system and saves the random initialization, final model, loss history, and all eight checkpoints.
Each GPU job writes a distinct training report, and the four reports are merged into `messk_01_training.json` before CPU analysis.

### `02_probe.py`

Fits linear probes at the embedding and after each of the four blocks.
Targets are the three independent tetrahedral belief coordinates, the four-dimensional true-mood one-hot, and each system's physical metrics.
Sequences are split `60%/20%/20%` into train, validation, and test sets.
Validation selects the layer, and the reported number is measured once on the untouched test set.
The same probes are run on the trained model and an architecture-matched random initialization, with a shuffled-target control.

### `03_myopic.py`

Compares the validation-selected residual-stream belief probe against sparse ridge regression from the last `1,2,4,8,16,32,64,80` raw observation tokens plus position within the physics tick.
The 80-token window spans eight ticks and therefore covers the full seven-letter belief memory.
Ridge strength is selected on validation sequences before test scoring.
The headline is `excess = residual R2 - raw-window R2`; an excess near zero means the transformer adds no representation beyond recent observations.

### `04_emergence.py`

Runs probes at every checkpoint and selects depth using validation sequences at each checkpoint.
For each target, it reports test R2 and the log-interpolated token count where the target reaches `50%` and `75%` of its own final learned gain over initialization.
Targets with gain below `0.05` receive no emergence crossing.

### `05_geometry.py`

Plots the transformer's predicted belief geometry for all eight checkpoints and all five depths.
The pale cloud is the exact reachable set, and prediction colours denote the dominant exact belief state.
Predictions are never clipped or projected back into the tetrahedron, so geometric failures remain visible.
Each panel reports test R2 and the fraction of predicted points outside the simplex.

## Compute and outputs

Training runs on four Kaggle T4 GPUs across two authenticated accounts, with one physical system per kernel.
Pendulum and sphere run under `chayanaggarwal45`, while predator-prey and double pendulum run under `chayanagiuwdhwekj`.
All four seed-0 models therefore train concurrently.
All probing and plotting ran on `staging-entity` CPU and completed in `218` seconds.

| Output | Contents |
|---|---|
| `messk_01_training.json` | merged training metadata and loss curves |
| `messk_02_probes.json` | trained, random-init, shuffled, layer-wise probe scores |
| `messk_03_myopic.json` | raw-window control and residual excess |
| `messk_04_emergence.json` | checkpoint curves and emergence crossings |
| `messk_05_geometry.json` | checkpoint-by-layer geometry scores |
| `figures/messk/mess4_belief_geometry.png` | exact tetrahedral belief geometry |
| `figures/transformer-belief/*.png` | predicted geometry across checkpoints and layers |

## Run

```bash
uv run pytest -q
uv run python scripts/messk_geometry.py
uv run python experiments/01_train.py
uv run python experiments/02_probe.py
uv run python experiments/03_myopic.py
uv run python experiments/04_emergence.py
uv run python experiments/05_geometry.py
```

Training is the only GPU step.
All claims are based on held-out sequences, explicit controls, and the single preregistered seed `0`.
