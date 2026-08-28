"""Build (or load from cache) all four DiscreteHMMs and report nnz / build
time / memory, per the Phase 1 verification plan.
"""
from __future__ import annotations

import time

from physics.configs import DEFAULT_CONFIGS
from physics.factory import make_simulator


def main() -> None:
    for name, config in DEFAULT_CONFIGS.items():
        t0 = time.perf_counter()
        sim = make_simulator(config)
        dt = time.perf_counter() - t0
        hmm = sim.hmm
        T_bytes = hmm.T.data.nbytes + hmm.T.indices.nbytes + hmm.T.indptr.nbytes
        P_bytes = sum(P.data.nbytes + P.indices.nbytes + P.indptr.nbytes for P in hmm.P_actions)
        E_bytes = hmm.E.nbytes
        print(
            f"{name:16s} n_latent={hmm.n_latent:7d} n_obs={hmm.n_obs:5d} n_actions={hmm.n_actions} "
            f"T.nnz={hmm.T.nnz:9d} build_s={dt:7.2f} "
            f"mem_MB={(T_bytes + P_bytes + E_bytes) / 1e6:6.2f}"
        )


if __name__ == "__main__":
    main()
