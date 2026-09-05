# System Analyzer CLI Guide (`analyze_system.py`)

A fast command-line diagnostics tool designed for researchers and AI agents to analyze, debug, and sweep physical parameters across all four Mess-4 systems (**Pendulum**, **Predator-Prey**, **Double Pendulum**, and **Sphere**) without writing ad-hoc Python code.

The script sits alongside `explorer.ipynb` at:
`notebooks/analyze_system.py`

---

## 1. Quick Start

Run the analyzer from the repository root using the virtual environment:

```bash
# Basic health check on default Predator-Prey
.venv/bin/python notebooks/analyze_system.py --system predator_prey_mess4 --mode check

# Basic health check on Pendulum with custom delta_v
.venv/bin/python notebooks/analyze_system.py --system pendulum_mess4 --dv 0.33 --mode check

# Display all flags and options
.venv/bin/python notebooks/analyze_system.py --help
```

---

## 2. Four Analysis Modes

### Mode 1: `check` (Single Detailed Diagnostic)
Runs an end-to-end simulation trace and prints a comprehensive stability report:
* **Stability Verdict**: PASS or FAIL with explicit failure reasons.
* **Lyapunov Exponent (λ)**: Divergence rate per second (target: λ <= +0.05 /s).
* **Clipped Fraction**: Percentage of observations hitting the outer bin boundary (budget: < 1.0%).
* **Vocabulary Usage**: Number and percentage of active observation bins (out of 181).
* **State Extrema**: Min and max values visited by all coordinates during the rollout.

```bash
# Example: Evaluate a custom predator-prey setup
.venv/bin/python notebooks/analyze_system.py \
  --system predator_prey_mess4 \
  --mode check \
  --kappa 4.0 \
  --dv 0.25 \
  --m 40
```

---

### Mode 2: `seeds` (Multi-Seed Reliability Test)
A parameter setting might look stable on `seed=0`, but experience clipping or chaotic divergence on a different sequence of kicks. The `seeds` mode runs across N random seeds to compute pass rate and worst-case clipping.

```bash
# Example: Test 20 random seeds on a heavy pendulum setup
.venv/bin/python notebooks/analyze_system.py \
  --system pendulum_mess4 \
  --mode seeds \
  --n-seeds 20 \
  --dv 0.55 \
  --gamma 1.2 \
  --m 40
```

---

### Mode 3: `sweep` (1D Parameter Scan)
Scans any single parameter across a range and prints a clean ASCII table of stability metrics.

```bash
# Example 1: Scan prey carrying capacity (kappa) from 2.0 to 12.0
.venv/bin/python notebooks/analyze_system.py \
  --system predator_prey_mess4 \
  --mode sweep \
  --sweep-param kappa \
  --sweep-min 2.0 \
  --sweep-max 12.0 \
  --sweep-steps 6

# Example 2: Scan pendulum kick strength (delta_v) from 0.1 to 1.0
.venv/bin/python notebooks/analyze_system.py \
  --system pendulum_mess4 \
  --mode sweep \
  --sweep-param delta_v \
  --sweep-min 0.1 \
  --sweep-max 1.0 \
  --sweep-steps 10
```

---

### Mode 4: `balance` (Steady-State Energy Equilibrium)
Runs Monte Carlo trajectory integration to compute the exact optimal damping (gamma*) where the average energy bled by friction matches the average energy injected by kicks.

```bash
# Example: Calculate optimal damping for a specific kick scale
.venv/bin/python notebooks/analyze_system.py \
  --system pendulum_mess4 \
  --mode balance \
  --dv 0.55 \
  --dt 0.02 \
  --n-steps 10
```

---

## 3. Command-Line Flag Reference

| Flag | Short | Description | Example |
| :--- | :--- | :--- | :--- |
| `--system` | `-s` | Target system (`pendulum_mess4`, `predator_prey_mess4`, etc.) | `-s predator_prey_mess4` |
| `--mode` | | `check`, `seeds`, `sweep`, or `balance` | `--mode sweep` |
| `--m` | | Sequence length in discrete ticks | `--m 40` |
| `--n-steps` | `-n` | Continuous ODE integration steps per tick | `-n 10` |
| `--dt` | | Continuous integration timestep in seconds | `--dt 0.02` |
| `--delta-v` | `--dv` | Perturbation impulse scale | `--dv 0.35` |
| `--seed` | | Base random seed | `--seed 42` |
| `--gamma` | | Viscous damping (Pendulum / Double Pendulum) | `--gamma 1.2` |
| `--kappa` | | Carrying capacity damping (Predator-Prey) | `--kappa 5.0` |
| `--omega-max` | | Velocity ceiling | `--omega-max 25.0` |
| `--obs-range` | | Sensor range tuple | `--obs-range "(-3.14, 3.14)"` |
| `--param-extra` | `-p` | Extra system parameters as `name=val` | `-p a=1.2 -p b=0.5` |
| `--stay` | | Markov chain self-transition probability | `--stay 0.8` |
| `--alpha` | | Markov chain emission accuracy | `--alpha 0.7` |
| `--sweep-param` | | Parameter to scan in sweep mode | `--sweep-param kappa` |
| `--sweep-min` | | Minimum value for sweep | `--sweep-min 1.0` |
| `--sweep-max` | | Maximum value for sweep | `--sweep-max 10.0` |
| `--sweep-steps` | | Number of grid points for sweep | `--sweep-steps 10` |

---

## 4. How Future Agents Should Interpret Metrics

When answering user queries about whether a configuration is usable for transformer representation training:

1. **Lyapunov Exponent (λ)**:
   * **λ <= 0.05 /s**: Contractive / predictable. Required for stable token sequences.
   * **λ > 0.05 /s**: Chaotic. Small perturbations diverge exponentially. Reject or increase damping.
2. **Clipped Fraction**:
   * **Must be < 1.0%** (ideally 0.0%).
   * If `clipped > 1%`: The trajectory is hitting the observation wall (`obs_range`), squishing tokens into bin 0 or 180 and destroying gradient signal.
3. **Used Vocabulary**:
   * **Target: > 40% (e.g. > 70 of 181 bins)**.
   * If vocabulary usage drops below 15-20%, the system is over-damped or kicks are too small, meaning tokens stay trapped in 2 or 3 center bins.
4. **Gap Free Mean**:
   * **Must be > 1e-3**.
   * Measures the distance between driven and free trajectories. If this is zero, kicks leave no visible trace on the physics, giving the transformer nothing to learn.
