# Universal System Analyzer CLI Guide (`analyze_system.py`)

A command-line diagnostics and parameter exploration engine designed for researchers and AI agents. It gives **100% full control over every parameter** across all four physical systems (**Pendulum**, **Predator-Prey**, **Double Pendulum**, and **Sphere**) without writing ad-hoc Python scripts.

The script sits alongside `explorer.ipynb` at:
`notebooks/analyze_system.py`

---

## 1. Parameter Discovery (`--list-params`)

You never need to guess which parameters exist for a system. Run `--list-params` to view every tunable field and its current default:

```bash
# List all parameters for Predator-Prey
.venv/bin/python notebooks/analyze_system.py -s predator_prey_mess4 --list-params

# List all parameters for Double Pendulum
.venv/bin/python notebooks/analyze_system.py -s double_pendulum_mess4 --list-params

# List all parameters for Pendulum
.venv/bin/python notebooks/analyze_system.py -s pendulum_mess4 --list-params

# List all parameters for Sphere
.venv/bin/python notebooks/analyze_system.py -s sphere_mess4 --list-params
```

---

## 2. Complete Parameter Reference Across All Systems

You can override **ANY** parameter below by passing it directly as a CLI flag (`--<name> <value>`) or via `-p <name>=<value>`.

### A. Universal Drivers & Integration (All Systems)
| Parameter | Default | Description | Example Override |
| :--- | :--- | :--- | :--- |
| `--m` | 16 (or 40) | Sequence length in discrete ticks | `--m 40` |
| `--n-steps` / `-n` | 10 | Continuous RK4 integration steps per tick | `-n 15` |
| `--dt` | varies | Timestep in seconds | `--dt 0.04` |
| `--delta-v` / `--dv` | varies | Perturbation impulse scale | `--dv 0.50` |
| `--obs-bins` | 181 | Observation discretization vocabulary | `--obs-bins 256` |
| `--stay` | 0.7 | Markov chain self-transition probability | `--stay 0.85` |
| `--alpha` | 0.7 | Markov chain emission accuracy | `--alpha 0.9` |
| `--seed` | 0 | Simulation random seed | `--seed 42` |

### B. Predator-Prey Parameters (`predator_prey_mess4`)
| Parameter | Default | Description | Example Override |
| :--- | :--- | :--- | :--- |
| `--a` | 1.0 | Prey birth rate | `--a 1.2` |
| `--b` | 0.6 | Predation attack rate | `--b 0.5` |
| `--c` | 0.8 | Predator natural death / mortality rate | `--c 0.9` |
| `--d` | 0.4 | Predator conversion efficiency | `--d 0.35` |
| `--kappa` | 5.0 | Carrying capacity (ecological damping) | `--kappa 3.5` |
| `--log-bound` | 3.5 | Safety clamp on log populations | `--log-bound 4.0` |
| `--x0` | 3.0 | Initial prey population count | `--x0 4.5` |
| `--y0` | 2.0 | Initial predator population count | `--y0 1.5` |
| `--obs-range` | (0.0, 1.0) | Sensor range (prey share bounds) | `--obs-range "(0.0, 1.0)"` |

### C. Pendulum Parameters (`pendulum_mess4`)
| Parameter | Default | Description | Example Override |
| :--- | :--- | :--- | :--- |
| `--gamma` | 1.2 | Viscous friction damping | `--gamma 0.50` |
| `--length` | 1.0 | Pendulum rod length (meters) | `--length 1.5` |
| `--g` | 9.8 | Gravitational acceleration | `--g 9.81` |
| `--omega-max` | 8.0 | Velocity clamp limit | `--omega-max 25.0` |
| `--theta0` | 0.0 | Initial release angle (rad) | `--theta0 0.5` |
| `--omega0` | 1.0 | Initial angular velocity (rad/s) | `--omega0 0.0` |
| `--obs-range` | (-1.571, 1.571) | Sensor range (radians) | `--obs-range "(-3.14, 3.14)"` |

### D. Double Pendulum Parameters (`double_pendulum_mess4`)
| Parameter | Default | Description | Example Override |
| :--- | :--- | :--- | :--- |
| `--l1`, `--l2` | 1.0, 1.0 | Lengths of arm 1 and arm 2 | `--l1 1.2 --l2 0.8` |
| `--m1`, `--m2` | 1.0, 1.0 | Masses of joint 1 and joint 2 | `--m1 2.0 --m2 0.5` |
| `--gamma1`, `--gamma2` | 0.5, 0.5 | Viscous friction on joint 1 and 2 | `--gamma1 1.2 --gamma2 1.2` |
| `--joint1-action-gain` | 2.0 | Gain multiplier on joint 1 kicks | `--joint1-action-gain 3.0` |
| `--th1-0`, `--th2-0` | 0.9, -0.4 | Initial release angles (rad) | `--th1-0 0.5 --th2-0 0.0` |
| `--w1-0`, `--w2-0` | 0.0, 0.0 | Initial angular velocities | `--w1-0 1.0` |
| `--omega-max` | 10.0 | Joint angular velocity limit | `--omega-max 30.0` |
| `--obs-range` | (-3.142, 3.142) | Sensor angle observation range | `--obs-range "(-3.14, 3.14)"` |

### E. Sphere Ball Parameters (`sphere_mess4`)
| Parameter | Default | Description | Example Override |
| :--- | :--- | :--- | :--- |
| `--gamma` | 0.35 | Isotropic sliding friction (meridional and azimuthal) | `--gamma 0.70` |
| `--length` | 1.0 | Sphere radius | `--length 2.0` |
| `--rate-max` | 6.0 | Angular rate cap | `--rate-max 12.0` |
| `--theta0` | 0.6 | Initial polar angle off bottom | `--theta0 0.8` |
| `--psi-dot0` | 2.0 | Initial azimuthal rotation rate | `--psi-dot0 3.5` |
| `--obs-range` | (0.05, 1.25) | Polar angle observation range | `--obs-range "(0.02, 1.50)"` |

---

## 3. The Four Modes in Action

### Mode 1: `check` (Full Diagnostics with Multiple Custom Parameters)
Pass any combination of parameters to evaluate stability, clipping, and state coordinate bounds:

```bash
# Example: Predator-prey with custom birth/death rates and carrying capacity
.venv/bin/python notebooks/analyze_system.py \
  -s predator_prey_mess4 \
  --mode check \
  --a 1.5 \
  --b 0.4 \
  --kappa 3.5 \
  --x0 4.0 \
  --dv 0.30 \
  --m 40

# Example: Double pendulum with custom joint masses, lengths, and heavy damping
.venv/bin/python notebooks/analyze_system.py \
  -s double_pendulum_mess4 \
  --mode check \
  --l1 1.2 \
  --m1 2.0 \
  --gamma1 1.5 \
  --gamma2 1.5 \
  --dv 0.8 \
  --m 40
```

---

### Mode 2: `sweep` (Scan ANY Physical Parameter with Fixed Overrides)
Scan **any parameter** across any range while holding all other custom parameters fixed at non-default values:

```bash
# Example 1: Sweep carrying capacity (kappa) while holding a=1.5, b=0.4, dv=0.4, and m=40 fixed
.venv/bin/python notebooks/analyze_system.py \
  -s predator_prey_mess4 \
  --mode sweep \
  --sweep-param kappa \
  --sweep-min 2.0 \
  --sweep-max 8.0 \
  --sweep-steps 4 \
  --a 1.5 \
  --b 0.4 \
  --dv 0.4 \
  --m 40

# Example 2: Sweep joint damping (gamma1) on Double Pendulum while holding l1=1.5, m1=2.0 fixed
.venv/bin/python notebooks/analyze_system.py \
  -s double_pendulum_mess4 \
  --mode sweep \
  --sweep-param gamma1 \
  --sweep-min 0.2 \
  --sweep-max 2.0 \
  --sweep-steps 10 \
  --l1 1.5 \
  --m1 2.0

# Example 3: Sweep integration timestep (dt) on Pendulum while holding dv=0.8, omega_max=25.0 fixed
.venv/bin/python notebooks/analyze_system.py \
  -s pendulum_mess4 \
  --mode sweep \
  --sweep-param dt \
  --sweep-min 0.01 \
  --sweep-max 0.08 \
  --sweep-steps 8 \
  --dv 0.8 \
  --omega-max 25.0
```

The output explicitly displays all held-fixed baseline overrides before the sweep table:
```text
===========================================================================
PARAMETER SWEEP: PREDATOR_PREY_MESS4
===========================================================================
Swept Parameter: 'kappa' from 2.0000 to 8.0000 (4 points)
Fixed Baseline Overrides (held constant throughout sweep):
  a                  = 1.5        (default was 1.0)
  b                  = 0.4        (default was 0.6)
  delta_v            = 0.4        (default was 0.35)
  m                  = 40         (default was 16)
---------------------------------------------------------------------------
       kappa | Stable |   Lyapunov |  Clipped |  Used Bins |  Obs Min |  Obs Max
---------------------------------------------------------------------------
      2.0000 |    YES |    -0.1410 |    0.00% |   78/181  |    0.411 |    0.891
      4.0000 |    YES |    -0.1911 |    0.00% |  123/181  |    0.059 |    0.848
      6.0000 |    YES |    -0.0683 |    0.00% |  133/181  |    0.040 |    0.870
      8.0000 |    YES |    -0.0767 |    0.00% |  133/181  |    0.038 |    0.870
---------------------------------------------------------------------------
```

#### In the Jupyter Notebook UI (`explorer.ipynb`):
The `sweep_ui()` widget also includes a **`fixed overrides`** text input box. You can directly type e.g.:
`a=1.5, kappa=3.5, delta_v=0.4`
and click **run sweep**. The scan will run while holding those exact parameters constant!

---

### Mode 3: `seeds` (Reliability Across Random Environments)
Run multiple seeds with any custom parameter setup to ensure zero clipping under rare kick sequences:

```bash
# Example: Check 15 seeds on heavily driven predator-prey
.venv/bin/python notebooks/analyze_system.py \
  -s predator_prey_mess4 \
  --mode seeds \
  --n-seeds 15 \
  --kappa 4.0 \
  --dv 0.45 \
  --m 40
```

---

### Mode 4: `balance` (Steady-State Energy Equilibrium)
Computes the exact optimal damping where energy bled equals energy injected:

```bash
# Example: Calculate optimal gamma for pendulum with custom kick and timestep
.venv/bin/python notebooks/analyze_system.py \
  -s pendulum_mess4 \
  --mode balance \
  --dv 0.60 \
  --dt 0.02 \
  --n-steps 10
```

---

## 4. Metric Interpretation Quick-Guide

When answering research queries on whether a physical parameter setup is suitable for training transformers:

1. **Lyapunov Exponent (λ)**:
   * **λ <= +0.05 /s**: STABLE / Contractive. Trajectories remain predictable.
   * **λ > +0.05 /s**: CHAOTIC. Nearby states explode exponentially. Unusable for clean belief recovery.
2. **Clipped Fraction**:
   * **Target: 0.0% (Must be < 1.0%)**.
   * If `clipped > 1%`: The trajectory exceeds `obs_range`. Tokens saturate against boundary bins, destroying gradient signals.
3. **Used Vocabulary**:
   * **Target: > 40% (e.g. > 70 of 181 bins)**.
   * If vocabulary usage is < 20%, the system is over-damped or kicks are too small; the observable stays trapped in a tiny cluster of bins.
4. **Gap Free Mean**:
   * **Must be > 1e-3**.
   * Measures how much the Markov kicks displace the physical trajectory compared to an un-driven run. If this is zero, kicks leave no trace for the model to detect.

---

## 5. Calibrated 5-Regime Portfolios

To systematically probe whether transformers learn continuous physics simulations and internal belief states across distinct physical dynamics, we provide a calibrated 5-regime portfolio.

All diagnostic plots can be batch-generated into `notebooks/outputs/` using:
```bash
.venv/bin/python notebooks/analyze_system.py -s predator_prey_mess4 --generate-portfolio
```

### Predator-Prey 5-Regime Reference Table

| Regime Name | Physical Character | Δv (dv) | κ (kappa) | dt (s) | n | m (Physics Explorer) | m (Transformer Training, m*n = 250) | Diagnostic Highlights (m=150) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Regime 1: Gentle / Fine-Grained** | Mild perturbation, tight deterministic orbit | 0.18 | 5.0 | 0.05 | 10 | 150 | 25 | λ = -0.187/s, 0% clip, 90/181 bins (50%), tight 1D return curve |
| **Regime 2: Balanced Baseline** | Canonical ecology, broad phase space coverage | 0.35 | 5.0 | 0.05 | 10 | 150 | 25 | λ = -0.155/s, 0% clip, 146/181 bins (81%), rich multi-ring phase diagram |
| **Regime 3: Heavily Damped** | Rapid relaxation, short memory of past kicks | 0.35 | 2.5 | 0.05 | 10 | 150 | 25 | λ = -0.251/s, 0% clip, 114/181 bins (63%), inward spiral collapse |
| **Regime 4: Wide-Orbit** | Weak damping, high entropy, long memory | 0.30 | 7.0 | 0.05 | 10 | 150 | 25 | λ = -0.108/s, 0% clip, 152/181 bins (84%), maximum token diversity |
| **Regime 5: Sparse Episodic Shocks**| Long continuous arcs, sharp discrete kicks | 0.40 | 5.0 | 0.04 | 25 | 150 | 10 | λ = -4.455/s, 0% clip, 121/181 bins (67%), 25 RK4 steps between kicks |

### Sequence Budget Rationale

1. **Physics Exploration (`m = 150`, 150 ticks)**:
   * Total physical duration: 75.0 seconds (~8.3 natural orbital cycles at T_0 ≈ 9.06s).
   * Generates rich, dense trajectories for multi-cycle return maps, Fourier spectra, and Lyapunov convergence.

2. **Transformer Training Budget (`m * n = 250 tokens`)**:
   * For n = 10 (Regimes 1-4): `m = 25 ticks` -> `25 * 10 = 250 tokens`. Total physical time: `25 * 10 * 0.05s = 12.5s` (~1.4 cycles).
   * For n = 25 (Regime 5): `m = 10 ticks` -> `10 * 25 = 250 tokens`. Total physical time: `10 * 25 * 0.04s = 10.0s` (~1.1 cycles).
   * **Transformer Benefit**: Covering ~1.1 to 1.4 full ecological cycles within the 250-token context window ensures that self-attention mechanisms can correlate phase relationships across a complete boom-bust cycle.
### Sphere Pendulum 5-Regime Reference Table

Batch-generate via:
```bash
.venv/bin/python notebooks/analyze_system.py -s sphere_mess4 --generate-portfolio
```

| Regime Name | Physical Character | Δv (dv) | γ (gamma) | dt (s) | n | m | Diagnostic Highlights (m=150) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Regime 1: Conical Precession** | Mild tangent kicks, steady precession swirl | 0.08 | 0.25 | 0.04 | 10 | 150 | λ = +0.001/s, 0% clip, 1203 bins, smooth rosette patterns |
| **Regime 2: Balanced Baseline** | Canonical spherical pendulum, isotropic damping | 0.15 | 0.35 | 0.04 | 10 | 150 | λ = -0.005/s, 0% clip, 1115 bins, balanced polar/azimuth wander |
| **Regime 3: Strongly Damped** | High friction, decays rapidly toward lower bowl | 0.15 | 0.70 | 0.04 | 10 | 150 | λ = +0.007/s, 0% clip, 928 bins, localized precession |
| **Regime 4: Wide Meridian** | Energetic perturbation, large theta swings | 0.25 | 0.40 | 0.04 | 10 | 150 | λ = -0.007/s, 0% clip, 1155 bins, broad polar coverage |
| **Regime 5: Long-Period Drift** | Fine timestep, slow continuous geodesic drift | 0.12 | 0.20 | 0.02 | 20 | 150 | λ = +0.009/s, 0% clip, 1975 bins, high token diversity |
