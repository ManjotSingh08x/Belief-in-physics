#!/usr/bin/env python3
"""Thin CLI shim over physics.messk_configs, physics.visualise, and physics.controls."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from physics.messk_configs import MESSK_CONFIGS, make_process
from physics.visualise import stability, trace
from physics.controls import compute_optimal_gamma, tunable_fields


def main() -> None:
    parser = argparse.ArgumentParser(description="System stability & parameter diagnostics.")
    parser.add_argument("-s", "--system", choices=sorted(MESSK_CONFIGS), default="pendulum_mess4")
    parser.add_argument("--mode", choices=["check", "seeds", "balance", "params"], default="check")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("-m", type=int, default=40)
    args = parser.parse_args()

    if args.mode == "params":
        proc = make_process(args.system)
        print(f"Tunable fields for {args.system}: {dict(tunable_fields(proc.system))}")
        return

    if args.mode == "balance":
        res = compute_optimal_gamma(args.system, m=args.m, seed=args.seed)
        print(f"[{args.system}] Optimal gamma: {res['gamma_opt']:.4f} (ratio: {res['balance_ratio']:.1%})")
        return

    if args.mode == "seeds":
        passes, lyaps = [], []
        for s in range(args.n_seeds):
            stab = stability(trace(make_process(args.system, m=args.m), seed=s))
            passes.append(stab["stable"])
            lyaps.append(stab["lyapunov"])
        print(f"[{args.system}] {sum(passes)}/{args.n_seeds} passed, mean lyapunov: {np.mean(lyaps):+.3f}")
        return

    stab = stability(trace(make_process(args.system, m=args.m), seed=args.seed))
    print(f"[{args.system}] Stable: {stab['stable']}, Lyapunov: {stab['lyapunov']:+.3f}, Clipped: {stab['clipped']:.1%}")
    if stab["reasons"]:
        print(f"  Reasons: {'; '.join(stab['reasons'])}")


if __name__ == "__main__":
    main()
