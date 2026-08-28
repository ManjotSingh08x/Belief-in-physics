"""Contract every physical system implements. State `z` has shape (..., state_dim);
all methods are vectorised over arbitrary leading batch dims.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class PhysicalSystem(Protocol):
    state_dim: int
    n_actions: int
    dims_periodic: tuple[bool, ...]

    @property
    def domain(self) -> tuple[tuple[float, float], ...]: ...

    def flow(self, z: np.ndarray, dt: float, substeps: int) -> np.ndarray:
        """Integrate z forward by dt*substeps, taking `substeps` internal steps."""
        ...

    def apply_action(self, z: np.ndarray, action: int) -> np.ndarray:
        """Instantaneous, deterministic perturbation. action in [0, n_actions)."""
        ...

    def energy(self, z: np.ndarray) -> np.ndarray: ...

    def observable(self, z: np.ndarray) -> np.ndarray:
        """Quantity the emission channel bins into observation tokens."""
        ...

    def metrics(self, z: np.ndarray) -> np.ndarray:
        """(..., d_metric) ground-truth analysis targets (probe labels)."""
        ...
