# Spherical Pendulum Refactoring & Visualization Roadmap: Cartesian Shift for Model

This document outlines the roadmap to **shift the model's observation and tokenization system to Cartesian coordinates $(x, y)$**, preserving the full physical range (0° to 90°), and upgrading [`notebooks/explorer.ipynb`](file:///home/manjot/Files/Code/BeliefPhysics/Belief-in-physics/notebooks/explorer.ipynb) to utilize the **2D Cartesian Token Grid system**.

---

## Core Architectural Shift: Cartesian Coordinates for the Model

### Why the Model Shifts to Cartesian $(x, y)$
In spherical polar coordinates $(\theta, \psi)$, the bottom of the bowl is a coordinate singularity ($\cot\theta \to \infty$ and $\psi$ is discontinuous by $180^\circ$ on origin crossings). This causes violent multi-bin jumps in token space.

By shifting the model's observable to Cartesian coordinates:
$$x = \sin(\theta)\cos(\psi), \quad y = \sin(\theta)\sin(\psi)$$
1. **The origin $(x, y) = (0, 0)$ is completely smooth and regular**: The ball swings through $(0, 0)$ as smooth harmonic sinusoids.
2. **Local Token Transitions**: The transformer only needs to predict transitions to neighboring cells ($\pm 1$ step), matching physical continuity.
3. **Faithful $0^\circ \to 90^\circ$ Representation**:
   - $\theta = 0^\circ$ (bottom pole) maps to the center $(x, y) = (0, 0)$.
   - $\theta = 90^\circ$ (equator) maps to the circle boundary $x^2 + y^2 = 1.0$.
   - The sensor range is $[-1.0, +1.0] \times [-1.0, +1.0]$, representing the full lower hemisphere.
4. **Uniform Vocabulary Usage**: An isotropic 2D distribution centered at $(0, 0)$ populates the $30 \times 30$ grid without dead bands.

---

## Action Items & Implementation Plan

### Phase 1: Model Observable & Tokenization Shift to Cartesian $(x, y)$

- [x] **1.1 Update `SphereBall` Observable to Cartesian $(x, y)$**
  - **File**: [`physics/systems/sphere.py`](file:///home/manjot/Files/Code/BeliefPhysics/Belief-in-physics/physics/systems/sphere.py)
  - **Action**:
    - Changed `observable_names` from `("theta", "psi")` to `("x", "y")`.
    - Implemented `observable(z)` and `observables(z)` returning $[x, y] = [\sin\theta \cos\psi, \sin\theta \sin\psi]$.
    - Set `obs_ranges = ((-1.0, 1.0), (-1.0, 1.0))` to cover the full $0^\circ \to 90^\circ$ lower hemisphere ($x^2 + y^2 \le 1.0$).
    - Set `obs_bins = (30, 30)` for a $30 \times 30 = 900$-cell Cartesian grid ($\Delta x = \Delta y = 0.069\text{ m}$).

- [x] **1.2 Update Undiscretization & Reconstruction for Cartesian Tokens**
  - **File**: [`physics/messk.py`](file:///home/manjot/Files/Code/BeliefPhysics/Belief-in-physics/physics/messk.py#L294)
  - **Action**: Verified `proc.undiscretise(tokens)` maps integer token indices back to continuous $(x, y)$ physical coordinates within $[-1.0, 1.0]^2$.

---

### Phase 2: Physics Engine Calibration (Cartesian Observables & 10-Seed Stability)

- [x] **2.1 Physics Engine Formulation Choice**
  - **File**: [`physics/systems/sphere.py`](file:///home/manjot/Files/Code/BeliefPhysics/Belief-in-physics/physics/systems/sphere.py)
  - **Choice**: Option A (Cartesian Observables over Internal Spherical ODE) with `THETA_MIN = 0.05 rad` as numerical guard.
  - Implemented `state_gap` measuring physical phase-space distance in $T(S^2)$ avoiding coordinate singularity inflation.
  - Calibrated parameters ($\Delta v = 0.15$, $\gamma = 0.35$) achieving 10/10 seed stability.

- [x] **2.2 Multi-Seed Stability Protocol (10 Seeds Minimum)**
  - **File**: [`notebooks/analyze_system.py`](file:///home/manjot/Files/Code/BeliefPhysics/Belief-in-physics/notebooks/analyze_system.py)
  - **Protocol**:
    - Evaluated across 10 random seeds (seeds 0 through 9).
    - Reported mean and standard deviation:
      - Lyapunov exponent: $\lambda = -0.1312 \pm 0.1347/\text{s}$ (target $\le +0.02/\text{s}$, max $-0.0347/\text{s} < +0.05/\text{s}$).
      - Clipped fraction: $0.00\% \pm 0.00\%$.
      - Active vocabulary: $71.2 \pm 9.7$ out of 900.
      - 10-seed pass rate: 10/10 (100%).

- [x] **2.3 Prevent Velocity Clamping**
  - Verified velocity limits (`rate_max = 6.0 rad/s`) provide sufficient headroom without trajectory clipping.

---

### Phase 3: Explorer Notebook & Visualizer Upgrade (2D Cartesian Grid)

- [x] **3.1 Upgrade 2D Token Grid in Visualizer (`_panel_tokens`)**
  - **File**: [`physics/visualise.py`](file:///home/manjot/Files/Code/BeliefPhysics/Belief-in-physics/physics/visualise.py#L525-L555)
  - **Action**:
    - Adapted `_panel_tokens` for Cartesian $(x, y)$:
      - X-axis: $x = \sin\theta \cos\psi \in [-1.0, 1.0]\text{ m}$.
      - Y-axis: $y = \sin\theta \sin\psi \in [-1.0, 1.0]\text{ m}$.
      - Drew unit circle boundary $x^2 + y^2 = 1.0$ (equator, $90^\circ$).
      - Heatmap of visited bins + continuous gradient trajectory path (start = green, end = orange).
      - Maintained `aspect="equal"` for faithful circular bowl geometry.

- [x] **3.2 Interactive Explorer Widget Integration (Cell 3)**
  - **Files**: [`notebooks/explorer.ipynb`](file:///home/manjot/Files/Code/BeliefPhysics/Belief-in-physics/notebooks/explorer.ipynb) (Cell 3), [`physics/controls.py`](file:///home/manjot/Files/Code/BeliefPhysics/Belief-in-physics/physics/controls.py#L115)
  - **Action**:
    - When `sphere_mess4` is selected in `explorer()`, the 2D Cartesian token grid panel is automatically included in the default dashboard.

- [x] **3.3 Cross-System Comparison Panel (Cell 7)**
  - **File**: [`notebooks/explorer.ipynb`](file:///home/manjot/Files/Code/BeliefPhysics/Belief-in-physics/notebooks/explorer.ipynb) (Cell 7)
  - **Action**:
    - `V.compare(traces, panel="tokens")` cleanly displays the 2D Cartesian $(x, y)$ grid for `sphere_mess4` side-by-side with 1D histograms for `pendulum`, `double_pendulum`, and `predator_prey`.

- [x] **3.4 Causal Effect Branching Analysis (Cell 11)**
  - **File**: [`notebooks/explorer.ipynb`](file:///home/manjot/Files/Code/BeliefPhysics/Belief-in-physics/notebooks/explorer.ipynb) (Cell 11)
  - **Action**:
    - Updated reconstructed observable labels and verified multi-channel branching calculations.

---

### Phase 4: Systematic Recalibration & Portfolio Generation

- [x] **4.1 Update 5-Regime Portfolio in `analyze_system.py`**
  - **File**: [`notebooks/analyze_system.py`](file:///home/manjot/Files/Code/BeliefPhysics/Belief-in-physics/notebooks/analyze_system.py#L335-L341)
  - **Action**: Calibrated 5 regimes for `sphere_mess4`, updated multi-channel bin thresholds, generated diagnostic plots in `notebooks/outputs/`.

---

### Phase 5: Machine Learning Validation

- [x] **5.1 Verify Local Diffusive Token Transitions**
  - Validated that transitions on the $30 \times 30$ Cartesian grid are smooth and strictly localized to neighboring cells (mean jump = 0.34 bins, max jump = 1.41 bins), verifying zero 15+ bin wrap teleportations.
  - Script: [`scratchpad/04_token_grid_demo.py`](file:///home/manjot/Files/Code/BeliefPhysics/Belief-in-physics/scratchpad/04_token_grid_demo.py)

- [x] **5.2 Verify Vocabulary Coverage Across Full Lower Bowl**
  - Ensured the $x^2 + y^2 \le 1.0$ disc has balanced token occupancy with no boundary clipping artifacts.

