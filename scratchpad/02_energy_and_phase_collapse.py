"""Visualizing the 16:1 Energy Deficit and Token Grid Wastage in SphereBall.

This script executes the exact SphereBall simulation from physics/systems/sphere.py
under default hyperparameters and demonstrates the physical collapse, velocity clamping,
and token vocabulary desert.
"""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from physics.messk_configs import make_process
from physics import visualise as V

out_dir = Path(__file__).parent
fig_path = out_dir / "02_energy_and_phase_collapse.png"

# Run default process with m=40 ticks
proc = make_process("sphere_mess4", m=40)
tr = V.trace(proc, seed=0)

t_state = tr["t_state"]
t_tick = np.arange(proc.m) * (proc.n_steps * proc.dt)
energy = tr["energy_driven"]
th = tr["states_driven"][:, 0]
psi = tr["states_driven"][:, 1]
dth = tr["states_driven"][:, 2]
dpsi = tr["states_driven"][:, 3]
tokens = tr["tokens_driven"]

# Calculate energy flow
E_ground = -proc.system.g * proc.system.length  # -9.8 J
P_inj = 0.5 * (proc.delta_v ** 2) / (proc.n_steps * proc.dt)  # Watts
# Dissipation rate over time: P_diss = gamma * v^2 = gamma * (L^2 * dth^2 + L^2 * sin^2(th) * dpsi^2)
v_sq = (proc.system.length ** 2) * (dth**2 + (np.sin(th)**2) * (dpsi**2))
P_diss = proc.system.gamma * v_sq

# Token grid counts
indices = proc.tokens_to_indices(tokens)
grid_counts = np.zeros((proc.obs_bins[0], proc.obs_bins[1]), dtype=int)
for r_th, c_psi in indices:
    grid_counts[r_th, c_psi] += 1

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.patch.set_facecolor("#fafafa")

# Panel 1: Energy Collapse
axes[0, 0].plot(t_state, energy, color="#2980b9", lw=1.8, label="Driven Total Energy E(t)")
axes[0, 0].axhline(E_ground, color="#c0392b", ls="--", lw=1.4, label=f"Ground State (-mgL = {E_ground:.1f} J)")
axes[0, 0].set_title("Energy Plunge to Ground State (Orbital Death in <10s)", fontsize=11, fontweight="bold")
axes[0, 0].set_xlabel("Time (s)")
axes[0, 0].set_ylabel("Total Mechanical Energy (J)")
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].legend(loc="upper right")

# Panel 2: Power Dissipation vs Injection
axes[0, 1].plot(t_state, P_diss, color="#e74c3c", lw=1.3, alpha=0.85, label="Instantaneous Dissipation P_diss (W)")
axes[0, 1].axhline(P_inj, color="#27ae60", ls="--", lw=1.8, label=f"Mean Kick Power P_inj ({P_inj:.3f} W)")
axes[0, 1].set_title("The 16:1 Power Deficit: Dissipation >> Injection", fontsize=11, fontweight="bold")
axes[0, 1].set_xlabel("Time (s)")
axes[0, 1].set_ylabel("Power (Watts)")
axes[0, 1].set_ylim(0, 0.6)
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].legend(loc="upper right")

# Panel 3: Polar Angle and Velocity Clamping
axes[1, 0].plot(t_state, th, color="#34495e", lw=1.4, label=r"$\theta(t)$ (rad)")
axes[1, 0].axhline(0.05, color="#c0392b", ls=":", lw=1.5, label="THETA_MIN (0.05 rad clamp)")
twin = axes[1, 0].twinx()
twin.plot(t_state, dpsi, color="#8e44ad", lw=1.2, ls="--", alpha=0.75, label=r"$\dot{\psi}(t)$ (rad/s)")
twin.axhline(6.0, color="#d35400", ls="--", lw=1.2, label=r"$\pm$rate_max clamp ($\pm$6 rad/s)")
twin.axhline(-6.0, color="#d35400", ls="--", lw=1.2)
axes[1, 0].set_title(r"Freezing at $\theta=0.05$ & Azimuthal Rate Squeezed to Clamps", fontsize=11, fontweight="bold")
axes[1, 0].set_xlabel("Time (s)")
axes[1, 0].set_ylabel(r"$\theta$ (rad)", color="#34495e")
twin.set_ylabel(r"$\dot{\psi}$ (rad/s)", color="#8e44ad")
axes[1, 0].grid(True, alpha=0.3)
h1, l1 = axes[1, 0].get_legend_handles_labels()
h2, l2 = twin.get_legend_handles_labels()
axes[1, 0].legend(handles=h1 + h2, loc="upper right", fontsize=8)

# Panel 4: 2D Token Grid Heatmap
im = axes[1, 1].imshow(grid_counts, cmap="Blues", origin="lower", aspect="auto",
                       extent=[-np.pi, np.pi, proc.obs_ranges[0][0], proc.obs_ranges[0][1]])
axes[1, 1].set_title(f"2D Token Grid: {np.count_nonzero(grid_counts)}/900 Bins Used (>80% Wastage)",
                     fontsize=11, fontweight="bold")
axes[1, 1].set_xlabel(r"Azimuth $\psi$ (rad)")
axes[1, 1].set_ylabel(r"Polar Angle $\theta$ (rad)")
cbar = plt.colorbar(im, ax=axes[1, 1])
cbar.set_label("Visit Count")
axes[1, 1].axhline(0.20, color="#e74c3c", ls="--", lw=1.2, label=r"Max steady $\theta \approx 0.15$")
axes[1, 1].legend(loc="upper right", fontsize=8)

plt.tight_layout()
plt.savefig(fig_path, dpi=120)
plt.close(fig)
print(f"Saved: {fig_path}")
