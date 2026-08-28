"""Disk cache for built DiscreteHMMs, keyed by a hash of the config dict
(transition-matrix construction is the expensive part of make_simulator).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from .grid import LatentGrid
from .hmm import DiscreteHMM

CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"


def config_hash(config: dict) -> str:
    blob = json.dumps(config, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def load(config: dict) -> DiscreteHMM | None:
    path = CACHE_DIR / f"{config_hash(config)}.npz"
    if not path.exists():
        return None
    z = np.load(path, allow_pickle=False)
    T_shape = tuple(z["T_shape"])
    T = sp.csr_matrix((z["T_data"], z["T_indices"], z["T_indptr"]), shape=T_shape)
    n_actions = int(z["n_actions"])
    P_actions = tuple(
        sp.csr_matrix((z[f"P{i}_data"], z[f"P{i}_indices"], z[f"P{i}_indptr"]), shape=T_shape)
        for i in range(n_actions)
    )
    grid = LatentGrid(
        bin_counts=tuple(int(x) for x in z["grid_bin_counts"]),
        domain=tuple(map(tuple, z["grid_domain"].tolist())),
        periodic=tuple(bool(x) for x in z["grid_periodic"]),
    )
    return DiscreteHMM(
        T=T,
        P_actions=P_actions,
        E=z["E"],
        metric_table=z["metric_table"],
        grid=grid,
        action_names=tuple(str(x) for x in z["action_names"]),
    )


def save(config: dict, hmm: DiscreteHMM) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{config_hash(config)}.npz"
    data = {
        "T_data": hmm.T.data,
        "T_indices": hmm.T.indices,
        "T_indptr": hmm.T.indptr,
        "T_shape": np.array(hmm.T.shape),
        "E": hmm.E,
        "metric_table": hmm.metric_table,
        "n_actions": hmm.n_actions,
        "action_names": np.array(hmm.action_names),
        "grid_bin_counts": np.array(hmm.grid.bin_counts),
        "grid_domain": np.array(hmm.grid.domain),
        "grid_periodic": np.array(hmm.grid.periodic),
    }
    for i, P in enumerate(hmm.P_actions):
        data[f"P{i}_data"] = P.data
        data[f"P{i}_indices"] = P.indices
        data[f"P{i}_indptr"] = P.indptr
    np.savez_compressed(path, **data)
    return path


def _demo() -> None:
    import tempfile

    from .systems.pendulum import Pendulum
    from .builder import build_transition_matrix, build_action_map

    global CACHE_DIR
    with tempfile.TemporaryDirectory() as tmp:
        CACHE_DIR = Path(tmp)  # noqa: F824 (module-level override for the demo)
        system = Pendulum(gamma=0.0)
        grid = LatentGrid(bin_counts=(8, 8), domain=system.domain, periodic=system.dims_periodic)
        rng = np.random.default_rng(0)
        T = build_transition_matrix(system, grid, K=3, dt=0.02, samples_per_bin=32, rng=rng)
        P = (build_action_map(system, grid, 0), build_action_map(system, grid, 1))
        E = np.full((grid.n_bins, 4), 0.25)
        hmm = DiscreteHMM(T=T, P_actions=P, E=E, metric_table=np.zeros((grid.n_bins, 1)), grid=grid, action_names=("a", "b"))
        cfg = {"system": {"name": "pendulum"}, "seed": 0}
        save(cfg, hmm)
        loaded = load(cfg)
        assert loaded is not None
        assert (loaded.T != hmm.T).nnz == 0
        assert np.array_equal(loaded.E, hmm.E)
        assert loaded.grid.bin_counts == hmm.grid.bin_counts
        assert load({"system": {"name": "other"}}) is None
    print("cache ok")


if __name__ == "__main__":
    _demo()
