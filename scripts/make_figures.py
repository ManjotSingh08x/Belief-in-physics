"""Write one demonstration figure per system to figures/."""
from __future__ import annotations

from pathlib import Path

from physics.configs import DEFAULT_CONFIGS
from physics.factory import make_simulator
from physics.viz import make_figure

OUT_DIR = Path(__file__).resolve().parent.parent / "figures"

# Which latent grid dim the belief-marginal panel projects onto -- matched to
# what each system's observable actually reads (theta2, not theta1, for the
# double pendulum; a position proxy for the two systems with no single-dim
# observable).
MARGINAL_DIM = {"pendulum": 0, "predator_prey": 0, "sphere": 0, "double_pendulum": 1}


def main() -> None:
    for name, config in DEFAULT_CONFIGS.items():
        sim = make_simulator(config)
        path = make_figure(name, sim, OUT_DIR, N=sim.K * 6, M=6, marginal_dim=MARGINAL_DIM[name])
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
