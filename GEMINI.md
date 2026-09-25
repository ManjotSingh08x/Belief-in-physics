# Belief-in-Physics: Agent Context & Engineering Rules (Double Pendulum Branch)

This file provides persistent context and defines the operational interface between developer preferences and repository invariants for Google Antigravity and Gemini agents operating on the `double_pendulum_system` branch.

---

## 1. Branch Focus & System Architecture

- **Active Branch**: `double_pendulum_system` (dedicated strictly to the Double Pendulum system).
- **Domain Mission**: Investigating how hidden Markovian belief states ($P(\text{next mood} \mid \text{letters so far})$ from a 4-mood Mess-4 chain) survive transmission through chaotic, non-linear Lagrangian double-pendulum dynamics and become linearly readable from transformer residual streams.
- **Physical Setup**:
  - **State Vector**: $z = (\theta_1, \theta_2, \omega_1, \omega_2)$ where $\theta_1, \theta_2$ are joint angles and $\omega_1, \omega_2$ are angular velocities.
  - **Release**: Asymmetric initial state at rest: $\theta_{1,0} = 0.9$, $\theta_{2,0} = -0.4$, $\omega_{1,0} = \omega_{2,0} = 0.0$.
  - **Dynamics**: Standard Lagrangian equations of motion integrated via RK4 with viscous joint friction $-\gamma_i \omega_i$ and rate clamping (`omega_max = 10.0`).
  - **Damping**: $\gamma_1 = 2.5, \gamma_2 = 2.5$ (screened contractive operating point with $\lambda < -0.30/\text{s}$ and 0% clipping across seeds).
  - **Actions / Kicks**: 4 balanced, non-zero cardinal impulses ($\Delta v = 1.2$):
    - Letter 0: Joint 1 negative kick ($-\text{gain} \times \Delta v$)
    - Letter 1: Joint 1 positive kick ($+\text{gain} \times \Delta v$)
    - Letter 2: Joint 2 negative kick ($-\Delta v$)
    - Letter 3: Joint 2 positive kick ($+\Delta v$)
    - `joint1_action_gain = 2.0` (amplifies base joint kick so both joints produce distinct trajectories).
- **Observation Space & 2D Discretization**:
  - Observables are wrapped angles $(\theta_1, \theta_2) \in [-\pi, \pi]^2$.
  - Discretized into a 2D mixed-radix token grid: `DOUBLE_PENDULUM_BINS = (30, 30)` ($30 \times 30 = 900$ tokens total vocabulary).
  - Token formula: $\text{token} = \text{index}(\theta_1) \times 30 + \text{index}(\theta_2)$.
  - Physical probe targets (`metric_names`): `("omega1", "omega2")`.

---

## 2. Essential Commands & Toolchains

Always run commands from the repository root:

- **Environment Synchronization**:
  ```bash
  uv sync
  ```
- **Double Pendulum Unit & Physics Tests**:
  ```bash
  uv run pytest tests/test_double_pendulum_physics.py
  uv run pytest tests/test_explorer_notebook.py
  uv run pytest tests/test_rk4_substeps.py
  ```
- **Run Full Test Suite**:
  ```bash
  uv run pytest
  ```
- **CLI Sanity Check for Double Pendulum**:
  ```bash
  uv run python -c "from physics.messk_configs import make_process; p = make_process('double_pendulum_mess4'); print(f'vocab={p.n_obs}, actions=\n{p.actions}')"
  ```
- **Interactive Exploration & UI**:
  ```bash
  uv run jupyter lab  # Open notebooks/explorer.ipynb or notebooks/three_phase_pipeline.ipynb
  ```
- **Dependency Management**:
  ```bash
  uv add <package> && uv remove <package>
  ```
  *(Always stage `pyproject.toml` and `uv.lock` together).*

---

## 3. Directory Layout & Architectural Boundaries

- `physics/systems/double_pendulum.py`: Canonical `DoublePendulum` class (ODEs, RK4 integration, angle wrapping, energy calculations).
- `physics/messk_configs.py`: System registration (`SYSTEMS["double_pendulum"]`), default grid bins (`DOUBLE_PENDULUM_BINS = (30, 30)`), and configuration helpers (`set_double_pendulum_bins`).
- `physics/controls.py`: Interactive explorer widgets, grid screening (`grid_screen`), and step recalibration (`n_recalibrate`) for `double_pendulum_mess4`.
- `physics/visualise.py`: Trajectory visualization, 2D token grid density plots, continuous trajectory unwrap, and Lyapunov computation.
- `notebooks/explorer.ipynb`: Interactive physics UI focused on double pendulum stability, parameter sweeps, and token distributions.
- `notebooks/three_phase_pipeline.ipynb`: Three-phase training, baseline selection, and probing pipeline.
- `docs/HYPERPARAM.md`: Authoritative inventory of double pendulum physical parameters and stability benchmarks.
- `docs/SWEEP_DETAILS.md`: Preflight protocol (Stage 0: $\Delta v \times \gamma$ grid screening, Lyapunov, Bayes gap, sync length).
- `tests/test_double_pendulum_physics.py`: Automated regression tests for energy, damping stability, wrapping, and tokenization.

### Boundary Rules
- **NEVER** define ODE integration or discretization logic inside notebooks or experiment scripts. All physics belongs in `physics/systems/double_pendulum.py` or `physics/messk_configs.py`.
- **Outputs**: Transient outputs go to `experiments/outputs-*` (gitignored). Persistent findings go to `experiments/results-*`.

---

## 4. Hard Invariants & Technical Rules

- **Hyperparameter Inventory Invariant (`docs/HYPERPARAM.md`)**:
  - `docs/HYPERPARAM.md` is the single source of truth for all parameters.
  - **MANDATORY**: Any change to `gamma1`, `gamma2`, `delta_v`, `dt`, `obs_bins`, `joint1_action_gain`, or `initial_state` MUST update `docs/HYPERPARAM.md` in the exact same commit.
  - Always record empirical stability baselines (Lyapunov exponent $\lambda$, clipping rate, Bayes gap) next to modified parameter values.
- **2D Token Ordering & Mixed-Radix Invariant**:
  - Channel 0 is ALWAYS $\theta_1$; Channel 1 is ALWAYS $\theta_2$.
  - Token discretization MUST follow $\text{token} = \text{index}(\theta_1) \times B_2 + \text{index}(\theta_2)$. Never reverse the channels.
  - Both angles MUST be wrapped to $[-\pi, \pi]$ before discretization.
- **Stability Baseline Invariant**:
  - Double pendulum dynamics must remain in the contractive regime ($\lambda < -0.30/\text{s}$, clipping $= 0\%$, multi-seed stability pass rate $\ge 8/10$ across 10 seeds).
- **Git Branch Discipline**:
  - Work strictly on feature branches or `double_pendulum_system`. Never commit directly to `main`.
  - Always commit `pyproject.toml` and `uv.lock` together.
- **Knowledge Graph (`graphify`)**:
  - Query graph before major refactoring: `graphify query "<question>"`.
  - Update graph after code changes: `graphify update .`.

---

## 5. User Preferences & Collaboration Interface

- **Task Scope**: All current work is dedicated strictly to the Double Pendulum system (`double_pendulum_mess4`). Disregard unrelated physical systems unless comparative calibration is explicitly requested.
- **Communication Tone**: Concise, technical, GitHub-flavored markdown. No redundant recitation of unchanged code.
- **Verification Rule**: Always run `uv run pytest tests/test_double_pendulum_physics.py` before marking any code edit complete.
- **Code Modification Discipline**:
  - Keep edits surgical and minimal.
  - Preserve all physics comments, Lagrangian derivations, and docstrings.
  - Use clickable links for all referenced file paths and code symbols.
- **Arbitration Protocol**:
  - If a user prompt requests a parameter tweak that violates contractive stability ($\lambda \ge 0$) or forgets to update `docs/HYPERPARAM.md`, immediately flag the conflict and apply the compliant fix.
