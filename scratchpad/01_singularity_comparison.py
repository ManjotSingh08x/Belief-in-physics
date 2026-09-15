"""Visualizing the Polar Singularity vs Cartesian Smoothness in Spherical Pendulums.

This script demonstrates why spherical polar coordinates (theta, psi) break down
numerically and machine-learning-wise when a trajectory passes near the bottom pole,
while Cartesian coordinates (x, y) remain perfectly smooth and continuous.
"""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

out_dir = Path(__file__).parent
fig_path = out_dir / "01_singularity_comparison.png"

# Simulate a near-meridional swing passing close to the pole (b = impact parameter)
t = np.linspace(0, 4.0, 1000)
omega = np.sqrt(9.8 / 1.0)  # ~3.13 rad/s natural frequency
b = 0.03  # impact parameter off the pole in y
A = 0.40  # amplitude in x

# Smooth Cartesian trajectory
x = A * np.sin(omega * t)
y = b * np.cos(omega * t * 0.5)  # slow precession
vx = A * omega * np.cos(omega * t)
vy = -b * (omega * 0.5) * np.sin(omega * t * 0.5)

# Convert to spherical polar coordinates
r = np.sqrt(x**2 + y**2)
theta = np.arcsin(np.clip(r, 0, 0.999))  # polar angle
psi = np.arctan2(y, x)                   # azimuth [-pi, pi]

# Velocities in spherical polar coordinates
dtheta = (x * vx + y * vy) / (np.maximum(r, 1e-6) * np.cos(theta))
dpsi = (x * vy - y * vx) / np.maximum(r**2, 1e-6)

# Discretization into 30 bins each
x_bins = np.clip(np.floor((x + 0.5) / 1.0 * 30), 0, 29).astype(int)
y_bins = np.clip(np.floor((y + 0.5) / 1.0 * 30), 0, 29).astype(int)

th_bins = np.clip(np.floor((theta - 0.05) / 0.5 * 30), 0, 29).astype(int)
psi_bins = np.clip(np.floor((psi + np.pi) / (2 * np.pi) * 30), 0, 29).astype(int)

fig, axes = plt.subplots(4, 2, figsize=(13, 11), sharex="col")
fig.patch.set_facecolor("#fafafa")

# Colors
C_X, C_Y = "#1f77b4", "#2ca02c"
C_TH, C_PSI = "#d62728", "#9467bd"

# Row 1: Coordinates
axes[0, 0].plot(t, x, label="x (m)", color=C_X, lw=1.6)
axes[0, 0].plot(t, y, label="y (m)", color=C_Y, lw=1.6, ls="--")
axes[0, 0].set_title("Cartesian: Smooth Continuous Signals", fontsize=11, fontweight="bold")
axes[0, 0].set_ylabel("Position (m)")
axes[0, 0].legend(loc="upper right")
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(t, theta, label=r"$\theta$ (rad)", color=C_TH, lw=1.6)
axes[0, 1].plot(t, psi, label=r"$\psi$ (rad)", color=C_PSI, lw=1.6, ls="--")
axes[0, 1].set_title(r"Spherical Polar: V-Cusps & $\pi$-Jumps at Pole", fontsize=11, fontweight="bold")
axes[0, 1].set_ylabel("Angle (rad)")
axes[0, 1].legend(loc="upper right")
axes[0, 1].grid(True, alpha=0.3)

# Row 2: Velocities / Rates
axes[1, 0].plot(t, vx, label="vx (m/s)", color=C_X, lw=1.4)
axes[1, 0].plot(t, vy, label="vy (m/s)", color=C_Y, lw=1.4, ls="--")
axes[1, 0].set_title("Cartesian: Bounded, Smooth Velocities", fontsize=11, fontweight="bold")
axes[1, 0].set_ylabel("Velocity (m/s)")
axes[1, 0].legend(loc="upper right")
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].plot(t, dtheta, label=r"$\dot{\theta}$ (rad/s)", color=C_TH, lw=1.4)
axes[1, 1].plot(t, dpsi, label=r"$\dot{\psi}$ (rad/s)", color=C_PSI, lw=1.4, ls="--")
axes[1, 1].set_title(r"Spherical Polar: $\dot{\psi} \to \infty$ Singularity Explosion", fontsize=11, fontweight="bold")
axes[1, 1].set_ylabel("Angular Rate (rad/s)")
axes[1, 1].set_ylim(-30, 30)
axes[1, 1].legend(loc="upper right")
axes[1, 1].grid(True, alpha=0.3)

# Row 3: Discretized Token Step Trajectory
axes[2, 0].step(t, x_bins, label="x token bin", color=C_X, where="post", lw=1.3)
axes[2, 0].step(t, y_bins, label="y token bin", color=C_Y, where="post", lw=1.3, ls="--")
axes[2, 0].set_title("Cartesian Token Sequence: Smooth Local Step Transitions", fontsize=11, fontweight="bold")
axes[2, 0].set_ylabel("Token Bin [0..29]")
axes[2, 0].legend(loc="upper right")
axes[2, 0].grid(True, alpha=0.3)

axes[2, 1].step(t, th_bins, label=r"$\theta$ token bin", color=C_TH, where="post", lw=1.3)
axes[2, 1].step(t, psi_bins, label=r"$\psi$ token bin", color=C_PSI, where="post", lw=1.3, ls="--")
axes[2, 1].set_title(r"Spherical Token Sequence: Violent 15-Bin Discontinuous Teleports", fontsize=11, fontweight="bold")
axes[2, 1].set_ylabel("Token Bin [0..29]")
axes[2, 1].legend(loc="upper right")
axes[2, 1].grid(True, alpha=0.3)

# Row 4: 2D Spatial Path in Grid Space
axes[3, 0].plot(x, y, color="#2c3e50", lw=1.5)
axes[3, 0].scatter([x[0]], [y[0]], color="#27ae60", s=40, label="start", zorder=5)
axes[3, 0].scatter([0], [0], color="#c0392b", marker="x", s=50, label="pole (0,0)", zorder=5)
axes[3, 0].set_title("Cartesian Phase Plane: Passes Smoothly through Center", fontsize=11, fontweight="bold")
axes[3, 0].set_xlabel("x (m)")
axes[3, 0].set_ylabel("y (m)")
axes[3, 0].axis("equal")
axes[3, 0].legend(loc="upper right")
axes[3, 0].grid(True, alpha=0.3)

axes[3, 1].plot(theta, psi, color="#2c3e50", lw=1.5)
axes[3, 1].scatter([theta[0]], [psi[0]], color="#27ae60", s=40, label="start", zorder=5)
axes[3, 1].axvline(0.05, color="#c0392b", ls=":", label=r"THETA_MIN (0.05 clamp)")
axes[3, 1].set_title(r"Spherical Plane: Trapped along Artificial $\theta$ Clamp", fontsize=11, fontweight="bold")
axes[3, 1].set_xlabel(r"$\theta$ (rad)")
axes[3, 1].set_ylabel(r"$\psi$ (rad)")
axes[3, 1].legend(loc="upper right")
axes[3, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(fig_path, dpi=120)
plt.close(fig)
print(f"Saved: {fig_path}")
