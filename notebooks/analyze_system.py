#!/usr/bin/env python3
"""System Analyzer CLI: Command-line tool for physical and statistical diagnostics.

Runs stability evaluations, parameter sweeps, multi-seed variance checks,
and energy balance calculations for Mess-4 systems (Pendulum, Predator-Prey,
Double Pendulum, and Sphere) without having to write ad-hoc Python code.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from physics.messk_configs import MESSK_CONFIGS, make_process
from physics.visualise import stability, trace
from physics.controls import compute_optimal_gamma, tunable_fields


def build_process(args) -> tuple[any, dict]:
    """Instantiate a MessDriven process using CLI flags."""
    system_kwargs = {}
    if args.gamma is not None:
        system_kwargs["gamma"] = float(args.gamma)
    if args.kappa is not None:
        system_kwargs["kappa"] = float(args.kappa)
    if args.omega_max is not None:
        system_kwargs["omega_max"] = float(args.omega_max)
    if args.obs_range is not None:
        try:
            system_kwargs["obs_range"] = ast.literal_eval(args.obs_range)
        except Exception:
            parts = [float(x) for x in args.obs_range.replace("(", "").replace(")", "").split(",")]
            system_kwargs["obs_range"] = (parts[0], parts[1])
    if args.param_extra:
        for item in args.param_extra:
            k, v = item.split("=")
            system_kwargs[k.strip()] = float(v.strip())

    proc_kwargs = {}
    if args.dt is not None:
        proc_kwargs["dt"] = float(args.dt)
    if args.n_steps is not None:
        proc_kwargs["n_steps"] = int(args.n_steps)
    if args.delta_v is not None:
        proc_kwargs["delta_v"] = float(args.delta_v)
    if args.m is not None:
        proc_kwargs["m"] = int(args.m)

    chain_kwargs = {}
    if args.stay is not None:
        chain_kwargs["stay"] = float(args.stay)
    if args.alpha is not None:
        chain_kwargs["alpha"] = float(args.alpha)
    if chain_kwargs:
        proc_kwargs["chain"] = chain_kwargs

    if system_kwargs:
        proc_kwargs["system"] = system_kwargs

    proc = make_process(args.system, **proc_kwargs)
    return proc, proc_kwargs


def cmd_check(args) -> None:
    """Run a single in-depth diagnostic trace and stability check."""
    proc, _ = build_process(args)
    tr = trace(proc, seed=args.seed)
    st = stability(tr)

    state_coords = tr["states_driven"]
    min_vals = state_coords.min(axis=0)
    max_vals = state_coords.max(axis=0)
    coord_names = getattr(proc.system, "state_names", [f"z{i}" for i in range(state_coords.shape[-1])])

    t_tick = proc.n_steps * proc.dt
    status_str = "PASS (STABLE)" if st["stable"] else "FAIL (UNSTABLE)"

    print("=" * 65)
    print(f"SYSTEM REPORT: {args.system.upper()} | STATUS: {status_str}")
    print("=" * 65)
    print(f"Parameters:")
    print(f"  m (ticks):          {proc.m}")
    print(f"  n_steps (per tick): {proc.n_steps}")
    print(f"  dt:                 {proc.dt:.4f} s")
    print(f"  T_tick (interval):  {t_tick:.4f} s")
    print(f"  delta_v (kick):     {proc.delta_v:.4f}")
    if hasattr(proc.system, "gamma"):
        print(f"  gamma (damping):    {proc.system.gamma:.4f}")
    if hasattr(proc.system, "kappa"):
        print(f"  kappa (damping):    {proc.system.kappa:.4f}")
    if hasattr(proc.system, "omega_max"):
        print(f"  omega_max (limit):  {proc.system.omega_max:.4f}")

    print("\nStability Metrics:")
    print(f"  Stable:             {st['stable']}")
    if st["reasons"]:
        print(f"  Failure Reasons:    {', '.join(st['reasons'])}")
    print(f"  Lyapunov Exponent:  {st['lyapunov']:+.4f} / s (target <= +0.05)")
    print(f"  Clipped Fraction:   {st['clipped']:.2%} (budget < 1.00%)")
    print(f"  Used Vocabulary:    {st['used_bins']} / {st['n_obs']} bins ({st['used_bins']/st['n_obs']*100:.1f}%)")
    print(f"  Observed Sensor:    [{st['observed_range'][0]:.3f}, {st['observed_range'][1]:.3f}]")
    print(f"  Sensor Limit:       [{st['obs_range'][0]:.3f}, {st['obs_range'][1]:.3f}]")
    print(f"  Gap Free Mean:      {st['gap_free_mean']:.4f}")

    print("\nPhysical State Extrema across Rollout:")
    for name, v_min, v_max in zip(coord_names, min_vals, max_vals):
        print(f"  {name:15s}: min = {v_min:+.4f}, max = {v_max:+.4f}")
    print("=" * 65)


def cmd_seeds(args) -> None:
    """Run stability over multiple seeds to detect rare clipping/divergence."""
    proc, _ = build_process(args)
    n_seeds = args.n_seeds
    clipped_arr, lyap_arr, bins_arr, stable_count = [], [], [], 0

    print(f"Evaluating {args.system} across {n_seeds} random seeds (0 to {n_seeds-1})...")
    for s in range(n_seeds):
        tr = trace(proc, seed=s)
        st = stability(tr)
        clipped_arr.append(st["clipped"])
        lyap_arr.append(st["lyapunov"])
        bins_arr.append(st["used_bins"])
        if st["stable"]:
            stable_count += 1

    print("\n" + "=" * 65)
    print(f"MULTI-SEED STABILITY SUMMARY ({n_seeds} seeds)")
    print("=" * 65)
    print(f"Pass Rate:            {stable_count} / {n_seeds} ({stable_count / n_seeds * 100:.1f}%)")
    print(f"Lyapunov Exponent:    mean = {np.mean(lyap_arr):+.4f} / s, min = {np.min(lyap_arr):+.4f}, max = {np.max(lyap_arr):+.4f}")
    print(f"Clipped Fraction:     mean = {np.mean(clipped_arr):.2%}, max = {np.max(clipped_arr):.2%}")
    print(f"Used Bins:            mean = {np.mean(bins_arr):.1f} / {proc.n_obs} ({np.mean(bins_arr)/proc.n_obs*100:.1f}%)")
    print("=" * 65)


def cmd_sweep(args) -> None:
    """Scan one parameter over a range and display a tabular summary."""
    param_name = args.sweep_param
    val_lo = float(args.sweep_min)
    val_hi = float(args.sweep_max)
    steps = int(args.sweep_steps)
    values = np.linspace(val_lo, val_hi, steps)

    print(f"Sweeping parameter '{param_name}' from {val_lo:.3f} to {val_hi:.3f} ({steps} points)...")
    print("-" * 75)
    print(f"{param_name:>10s} | {'Stable':>6s} | {'Lyapunov':>10s} | {'Clipped':>8s} | {'Used Bins':>10s} | {'Obs Min':>8s} | {'Obs Max':>8s}")
    print("-" * 75)

    for v in values:
        # Clone args and inject parameter
        sweep_args = argparse.Namespace(**vars(args))
        if param_name in ("gamma", "kappa", "omega_max"):
            setattr(sweep_args, param_name, v)
        elif param_name in ("dt", "delta_v", "m", "n_steps", "stay", "alpha"):
            setattr(sweep_args, param_name, v)
        else:
            extra = list(sweep_args.param_extra or [])
            extra.append(f"{param_name}={v}")
            sweep_args.param_extra = extra

        try:
            proc, _ = build_process(sweep_args)
            tr = trace(proc, seed=args.seed)
            st = stability(tr)
            st_flag = "YES" if st["stable"] else "NO"
            obs_lo, obs_hi = st["observed_range"]
            print(f"{v:10.4f} | {st_flag:>6s} | {st['lyapunov']:+10.4f} | {st['clipped']:8.2%} | {st['used_bins']:4d}/{st['n_obs']:<4d} | {obs_lo:8.3f} | {obs_hi:8.3f}")
        except Exception as exc:
            print(f"{v:10.4f} | {'ERROR':>6s} | {str(exc)[:40]}")
    print("-" * 75)


def cmd_balance(args) -> None:
    """Compute optimal viscous damping balancing kick injection with friction."""
    sys_params = {}
    if args.gamma is not None:
        sys_params["gamma"] = float(args.gamma)
    if args.kappa is not None:
        sys_params["kappa"] = float(args.kappa)
    if args.omega_max is not None:
        sys_params["omega_max"] = float(args.omega_max)

    print(f"Solving steady-state energy equilibrium for {args.system}...")
    res = compute_optimal_gamma(
        system_name=args.system,
        m=int(args.m or 40),
        n_steps=int(args.n_steps or 10),
        dt=float(args.dt or 0.02),
        delta_v=float(args.delta_v or 0.55),
        stay=float(args.stay or 0.7),
        alpha=float(args.alpha or 0.7),
        system_params=sys_params,
        seed=args.seed,
    )
    print("=" * 65)
    print("STEADY-STATE ENERGY EQUILIBRIUM REPORT")
    print("=" * 65)
    print(f"Optimal Damping (gamma*):        {res['gamma_opt']:.4f}")
    print(f"Mean Energy Injected per Tick:   {res['mean_injected']:.4f}")
    print(f"Mean Energy Dissipated per Tick: {res['mean_dissipated']:.4f}")
    print(f"Energy Balance Ratio (bled/inj): {res['balance_ratio']:.1%} (target: 100%)")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(
        description="Mess-4 System Diagnostic & Parameter Evaluation Tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--system", "-s", default="predator_prey_mess4", choices=sorted(MESSK_CONFIGS),
                        help="Target Mess-4 system")
    parser.add_argument("--mode", choices=["check", "seeds", "sweep", "balance"], default="check",
                        help="Analysis mode: check (single trace), seeds (multi-seed), sweep (1D parameter scan), balance (energy balance)")
    
    # Process / Driver parameters
    parser.add_argument("--m", type=int, default=None, help="Sequence length in ticks (e.g. 40)")
    parser.add_argument("--n-steps", "-n", type=int, default=None, help="Integration steps per tick (e.g. 10)")
    parser.add_argument("--dt", type=float, default=None, help="Integration timestep in seconds (e.g. 0.02 or 0.05)")
    parser.add_argument("--delta-v", "--dv", type=float, default=None, help="Impulse kick scale (e.g. 0.35, 0.55)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")

    # Specific system parameters
    parser.add_argument("--gamma", type=float, default=None, help="Viscous damping on pendulum / double pendulum")
    parser.add_argument("--kappa", type=float, default=None, help="Prey carrying capacity damping on predator-prey")
    parser.add_argument("--omega-max", type=float, default=None, help="Velocity limit (e.g. 8.0, 25.0)")
    parser.add_argument("--obs-range", type=str, default=None, help="Sensor range tuple, e.g. '(-3.1416, 3.1416)'")
    parser.add_argument("--param-extra", "-p", action="append", help="Extra system fields as name=val, e.g. -p a=1.2")

    # Chain parameters
    parser.add_argument("--stay", type=float, default=None, help="Markov chain stay probability (default 0.7)")
    parser.add_argument("--alpha", type=float, default=None, help="Markov chain emission accuracy (default 0.7)")

    # Mode-specific options
    parser.add_argument("--n-seeds", type=int, default=10, help="Number of seeds for --mode seeds")
    parser.add_argument("--sweep-param", type=str, default="kappa", help="Parameter name to sweep in --mode sweep")
    parser.add_argument("--sweep-min", type=float, default=1.0, help="Min sweep value")
    parser.add_argument("--sweep-max", type=float, default=10.0, help="Max sweep value")
    parser.add_argument("--sweep-steps", type=int, default=10, help="Number of steps in sweep")

    args = parser.parse_args()

    if args.mode == "check":
        cmd_check(args)
    elif args.mode == "seeds":
        cmd_seeds(args)
    elif args.mode == "sweep":
        cmd_sweep(args)
    elif args.mode == "balance":
        cmd_balance(args)


if __name__ == "__main__":
    main()
