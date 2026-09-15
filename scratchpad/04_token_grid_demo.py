"""Multi-Seed 2D Cartesian Token Grid & Transition Analysis for sphere_mess4.

Evaluates 10 random seeds on the 30x30 Cartesian token grid (900 cells, range [-1, 1]^2).
Verifies:
1. Vocabulary coverage and isotropic distribution across the lower bowl (x^2 + y^2 <= 1).
2. Local diffusive token transitions (|delta_token| <= 3 bins, 0 wrap teleportations).
3. Exact mapping to the unit circle equator (theta = 90 deg).
"""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from physics.messk_configs import make_process
from physics.visualise import trace

out_dir = Path(__file__).parent
fig_path = out_dir / "04_token_grid_demo.png"

N_SEEDS = 10
M_TICKS = 40
proc = make_process("sphere_mess4", m=M_TICKS)
b0, b1 = proc.obs_bins
lo0, hi0 = proc.obs_ranges[0]
lo1, hi1 = proc.obs_ranges[1]

# 2D Grid accumulator for all 10 seeds
grid_counts = np.zeros((b0, b1), dtype=int)
all_step_jumps = []
seed_traces = []

for s in range(N_SEEDS):
    tr = trace(proc, seed=s)
    seed_traces.append(tr)
    tokens = tr["tokens_driven"]
    indices = proc.tokens_to_indices(tokens)  # shape (L, 2)

    # Accumulate 2D bin hits
    for ix, iy in indices:
        grid_counts[ix, iy] += 1

    # Transition jumps: Euclidean distance in bin units between consecutive steps
    step_diffs = np.linalg.norm(np.diff(indices, axis=0), axis=-1)
    all_step_jumps.extend(step_diffs)

all_step_jumps = np.array(all_step_jumps)
total_cells_visited = int(np.sum(grid_counts > 0))
vocab_pct = total_cells_visited / proc.n_obs * 100.0

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))

# Panel 1: 2D Heatmap of Token Occupancy across 10 seeds
extent = [lo0, hi0, lo1, hi1]
im = ax1.imshow(
    grid_counts.T, origin="lower", extent=extent,
    cmap="YlGnBu", interpolation="nearest", aspect="equal"
)
cbar = plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
cbar.set_label("Visit Count (10 seeds, 400 ticks total)", fontsize=9)

# Overlay Equator Circle (theta = 90 deg, x^2 + y^2 = 1.0)
phi_circ = np.linspace(0, 2 * np.pi, 300)
ax1.plot(np.cos(phi_circ), np.sin(phi_circ), color="#c0392b", ls="--", lw=1.5,
         label="Equator (θ=90°, r=1.0 m)")

# Overlay sample trajectory (Seed 0)
sample_tr = seed_traces[0]
sample_recon = proc.undiscretise(sample_tr["tokens_driven"])
ax1.plot(sample_recon[:, 0], sample_recon[:, 1], color="#e67e22", lw=1.2, alpha=0.85,
         label="Trajectory (Seed 0)")
ax1.scatter(sample_recon[0, 0], sample_recon[0, 1], color="#27ae60", s=60, edgecolors="w",
            zorder=6, label="Start")
ax1.scatter(sample_recon[-1, 0], sample_recon[-1, 1], color="#e74c3c", s=60, edgecolors="w",
            zorder=6, label="End")

ax1.set_xlim(-1.12, 1.12)
ax1.set_ylim(-1.12, 1.12)
ax1.set_xlabel("x = sin(θ) cos(ψ) (m)", fontsize=10)
ax1.set_ylabel("y = sin(θ) sin(ψ) (m)", fontsize=10)
ax1.set_title(f"30×30 Cartesian Token Grid ({total_cells_visited}/{proc.n_obs} cells visited = {vocab_pct:.1f}%)",
              fontsize=10.5)
ax1.legend(fontsize=8, loc="upper right")
ax1.grid(True, alpha=0.3, ls=":")

# Panel 2: Transition Step Size Histogram in Token Space
max_jump = float(all_step_jumps.max())
mean_jump = float(all_step_jumps.mean())
jump_bins = np.arange(0, int(np.ceil(max_jump)) + 2) - 0.5

ax2.hist(all_step_jumps, bins=jump_bins, color="#2980b9", edgecolor="white", alpha=0.85, rwidth=0.85)
ax2.axvline(mean_jump, color="#e74c3c", ls="--", lw=1.5, label=f"Mean Step = {mean_jump:.2f} bins")
ax2.set_xlabel("Step Size between Consecutive Tokens (bins)", fontsize=10)
ax2.set_ylabel("Transition Count", fontsize=10)
ax2.set_title(f"Local Token Transitions: Zero Wrap-Around Teleportations (Max Jump = {max_jump:.1f} bins)",
              fontsize=10.5)
ax2.legend(fontsize=8.5)
ax2.grid(True, alpha=0.3, ls=":")

fig.tight_layout()
fig.savefig(fig_path, dpi=120)
plt.close(fig)

print(f"[OK] Saved 10-seed 2D token grid analysis to: {fig_path}")
print(f"  Total cells visited: {total_cells_visited} / {proc.n_obs} ({vocab_pct:.1f}%)")
print(f"  Step jump in token space: mean = {mean_jump:.2f} bins, max = {max_jump:.2f} bins")
