"""Demonstrating the Superiority of Cartesian Representation for ML in Spherical Pendulum.

This script uses the production SphereBall and sphere_mess4 process with live
Cartesian observables (x, y) = (sin theta cos psi, sin theta sin psi).
It shows how Cartesian coordinates eliminate origin singularities and wrap-around jumps.
"""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from physics.messk_configs import make_process
from physics.visualise import trace

out_dir = Path(__file__).parent
fig_path = out_dir / "03_cartesian_projection_demo.png"

proc = make_process("sphere_mess4", m=40)
tr = trace(proc, seed=1)

obs = tr["obs_driven"]  # shape (N, 2): (x, y)
tokens = tr["tokens_driven"]  # shape (N,): combined tokens
recon = proc.undiscretise(tokens)  # shape (N, 2): reconstructed (x, y)
t = tr["t"]

# Also compute spherical state coordinates for comparison
states = tr["states_driven"]  # shape (N+1, 4): (theta, psi, dtheta, dpsi)
theta = states[:-1, 0]
psi = states[:-1, 1]

fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

# Panel 1: Spherical coordinates with wrap discontinuity
psi_plot = psi.copy()
jumps = np.abs(np.diff(psi_plot, prepend=psi_plot[0])) > np.pi
axes[0].plot(t, theta, label="θ (polar angle)", color="#1f77b4", lw=1.3)
axes[0].plot(t, psi_plot, label="ψ (azimuth, ±π wrap)", color="#d62728", lw=1.1, ls="--")
axes[0].set_ylabel("spherical (rad)")
axes[0].set_title("Spherical State: (θ, ψ) suffers coordinate singularity at θ=0 and wrap at ±π")
axes[0].legend(fontsize=8, loc="upper right")
axes[0].grid(True, alpha=0.3)

# Panel 2: Continuous Cartesian Observables (x, y)
axes[1].plot(t, obs[:, 0], label="x = sin(θ) cos(ψ)", color="#2ca02c", lw=1.4)
axes[1].plot(t, obs[:, 1], label="y = sin(θ) sin(ψ)", color="#ff7f0e", lw=1.4)
axes[1].axhline(1.0, color="#c0392b", ls=":", lw=0.8, alpha=0.6)
axes[1].axhline(-1.0, color="#c0392b", ls=":", lw=0.8, alpha=0.6)
axes[1].set_ylabel("cartesian (m)")
axes[1].set_title("Cartesian Observables: Smooth sinusoids through (0, 0) without any coordinate jumps")
axes[1].legend(fontsize=8, loc="upper right")
axes[1].grid(True, alpha=0.3)

# Panel 3: Discrete Reconstructed Tokens
axes[2].step(t, recon[:, 0], where="post", label="recon x token", color="#2ca02c", lw=1.1, ls=":")
axes[2].step(t, recon[:, 1], where="post", label="recon y token", color="#ff7f0e", lw=1.1, ls=":")
axes[2].set_ylabel("token recon (m)")
axes[2].set_xlabel("time (s)")
axes[2].set_title(f"Reconstructed from 30×30 Token Grid (Vocabulary = {proc.n_obs})")
axes[2].legend(fontsize=8, loc="upper right")
axes[2].grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(fig_path, dpi=120)
plt.close(fig)
print(f"[OK] Saved production Cartesian demo plot to: {fig_path}")
