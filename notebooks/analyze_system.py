#!/usr/bin/env python3
"""System Analyzer CLI: Command-line tool for physical and statistical diagnostics.

Runs stability evaluations, arbitrary parameter sweeps, multi-seed variance checks,
and energy balance calculations for all Mess-4 systems (Pendulum, Predator-Prey,
Double Pendulum, and Sphere) with full control over all physical, driver, and chain parameters.
"""

from __future__ import annotations

import argparse
import ast
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


def get_system_param_info(system_name: str) -> dict:
    """Return categorized parameter fields and defaults for a system."""
    base_proc = make_process(system_name)
    sys_fields = dict(tunable_fields(base_proc.system))
    sys_fields["obs_range"] = getattr(base_proc.system, "obs_range", None)
    
    driver_fields = {
        "m": base_proc.m,
        "n_steps": base_proc.n_steps,
        "dt": base_proc.dt,
        "delta_v": base_proc.delta_v,
        "obs_bins": getattr(base_proc, "obs_bins", 181),
    }
    chain_fields = {
        "stay": getattr(base_proc.chain, "stay", 0.7),
        "alpha": getattr(base_proc.chain, "alpha", 0.7),
        "n_states": getattr(base_proc.chain, "n_states", 4),
    }
    return {
        "system": sys_fields,
        "driver": driver_fields,
        "chain": chain_fields,
    }


SCRIPT_CONTROL_KEYS = {
    "system", "mode", "seed", "n_seeds", "sweep_param", "sweep_min",
    "sweep_max", "sweep_steps", "list_params", "generate_portfolio",
    "save_plot", "param",
}


def parse_dynamic_args(unparsed_args: list[str]) -> dict:
    """Parse unknown CLI flags (--key value or --key) into typed dictionary."""
    d = {}
    i = 0
    while i < len(unparsed_args):
        arg = unparsed_args[i]
        if arg.startswith("--"):
            key = arg[2:].replace("-", "_")
            if i + 1 < len(unparsed_args) and not unparsed_args[i + 1].startswith("--"):
                val_str = unparsed_args[i + 1]
                i += 2
                try:
                    d[key] = ast.literal_eval(val_str)
                except Exception:
                    try:
                        d[key] = float(val_str)
                    except Exception:
                        d[key] = val_str
            else:
                d[key] = True
                i += 1
        else:
            i += 1
    return d


def create_process_from_dict(system_name: str, all_params: dict) -> tuple:
    """Instantiate a MessDriven process with arbitrary param overrides."""
    info = get_system_param_info(system_name)
    sys_field_names = set(info["system"].keys())
    chain_field_names = set(info["chain"].keys())
    driver_field_names = set(info["driver"].keys())

    if "dv" in all_params and "delta_v" not in all_params:
        all_params = dict(all_params)
        all_params["delta_v"] = all_params.pop("dv")

    sys_kwargs = {}
    chain_kwargs = {}
    driver_kwargs = {}

    for k, v in all_params.items():
        if k in SCRIPT_CONTROL_KEYS:
            continue
        if v is None:
            continue

        # Handle tuple string parsing for obs_range or obs_bins
        if k in ("obs_range", "obs_bins") and isinstance(v, str):
            try:
                v = ast.literal_eval(v)
            except Exception:
                parts = [float(x) for x in v.replace("(", "").replace(")", "").split(",")]
                v = tuple(parts)

        if k in sys_field_names:
            sys_kwargs[k] = v
        elif k in chain_field_names:
            chain_kwargs[k] = v
        elif k in driver_field_names:
            driver_kwargs[k] = v
        else:
            # Fallback: attempt system override
            sys_kwargs[k] = v

    proc_kwargs = dict(driver_kwargs)
    if sys_kwargs:
        proc_kwargs["system"] = sys_kwargs
    if chain_kwargs:
        proc_kwargs["chain"] = chain_kwargs

    proc = make_process(system_name, **proc_kwargs)
    return proc, proc_kwargs


def cmd_list_params(system_name: str) -> None:
    """Print all available tunable parameters for the specified system."""
    info = get_system_param_info(system_name)
    print("=" * 68)
    print(f"ALL TUNABLE PARAMETERS FOR: {system_name.upper()}")
    print("=" * 68)
    print("\n1. PHYSICAL SYSTEM PARAMETERS (use --<name> <val> or -p <name>=<val>):")
    for name, val in sorted(info["system"].items()):
        print(f"  --{name:<18s} (default: {val})")

    print("\n2. DRIVER & SIMULATION PARAMETERS:")
    for name, val in sorted(info["driver"].items()):
        print(f"  --{name:<18s} (default: {val})")

    print("\n3. MARKOV CHAIN PARAMETERS:")
    for name, val in sorted(info["chain"].items()):
        print(f"  --{name:<18s} (default: {val})")
    print("=" * 68)


def cmd_check(system_name: str, params: dict, seed: int = 0) -> None:
    """Run single in-depth diagnostic trace and stability check."""
    proc, _ = create_process_from_dict(system_name, params)
    tr = trace(proc, seed=seed)
    st = stability(tr)

    state_coords = tr["states_driven"]
    min_vals = state_coords.min(axis=0)
    max_vals = state_coords.max(axis=0)
    coord_names = getattr(proc.system, "state_names", [f"z{i}" for i in range(state_coords.shape[-1])])

    t_tick = proc.n_steps * proc.dt
    status_str = "PASS (STABLE)" if st["stable"] else "FAIL (UNSTABLE)"

    print("=" * 68)
    print(f"SYSTEM REPORT: {system_name.upper()} | STATUS: {status_str}")
    print("=" * 68)
    print("Simulation Drivers:")
    print(f"  m (ticks):          {proc.m}")
    print(f"  n_steps (per tick): {proc.n_steps}")
    print(f"  dt:                 {proc.dt:.4f} s")
    print(f"  T_tick (interval):  {t_tick:.4f} s")
    print(f"  delta_v (kick):     {proc.delta_v:.4f}")
    
    print("\nPhysical Parameters (Active):")
    for k, v in tunable_fields(proc.system).items():
        print(f"  {k:<18s}: {v}")
    if hasattr(proc.system, "obs_range"):
        print(f"  {'obs_range':<18s}: {proc.system.obs_range}")

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
    print("=" * 68)

    save_plot = params.get("save_plot")
    if save_plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from physics.visualise import plot
        out_path = Path(save_plot)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig = plot(tr, ncols=2, title=f"{system_name.upper()} Diagnostic Plot")
        fig.savefig(out_path, dpi=110)
        plt.close(fig)
        print(f"\n[OK] Diagnostic plot saved to: {out_path}")


def cmd_seeds(system_name: str, params: dict, n_seeds: int = 10) -> None:
    """Run stability over multiple seeds to detect rare clipping/divergence."""
    proc, _ = create_process_from_dict(system_name, params)
    clipped_arr, lyap_arr, bins_arr, stable_count = [], [], [], 0

    print(f"Evaluating {system_name} across {n_seeds} random seeds (0 to {n_seeds-1})...")
    for s in range(n_seeds):
        tr = trace(proc, seed=s)
        st = stability(tr)
        clipped_arr.append(st["clipped"])
        lyap_arr.append(st["lyapunov"])
        bins_arr.append(st["used_bins"])
        if st["stable"]:
            stable_count += 1

    print("\n" + "=" * 68)
    print(f"MULTI-SEED STABILITY SUMMARY ({n_seeds} seeds)")
    print("=" * 68)
    print(f"Pass Rate:            {stable_count} / {n_seeds} ({stable_count / n_seeds * 100:.1f}%)")
    print(f"Lyapunov Exponent:    mean = {np.mean(lyap_arr):+.4f} / s, min = {np.min(lyap_arr):+.4f}, max = {np.max(lyap_arr):+.4f}")
    print(f"Clipped Fraction:     mean = {np.mean(clipped_arr):.2%}, max = {np.max(clipped_arr):.2%}")
    print(f"Used Bins:            mean = {np.mean(bins_arr):.1f} / {proc.n_obs} ({np.mean(bins_arr)/proc.n_obs*100:.1f}%)")
    print("=" * 68)


def cmd_sweep(system_name: str, params: dict, sweep_param: str, val_lo: float, val_hi: float, steps: int, seed: int = 0) -> None:
    """Scan ANY parameter over a range while holding all other custom parameters fixed."""
    sweep_param = sweep_param.replace("-", "_")
    values = np.linspace(val_lo, val_hi, steps)

    info = get_system_param_info(system_name)
    all_defaults = {}
    all_defaults.update(info["system"])
    all_defaults.update(info["driver"])
    all_defaults.update(info["chain"])

    # Identify non-default parameters held fixed during the sweep
    fixed_overrides = {}
    for k, v in params.items():
        if k in SCRIPT_CONTROL_KEYS or k == sweep_param:
            continue
        if v is None:
            continue
        if k in all_defaults:
            if all_defaults[k] != v:
                fixed_overrides[k] = v
        else:
            fixed_overrides[k] = v

    print("=" * 75)
    print(f"PARAMETER SWEEP: {system_name.upper()}")
    print("=" * 75)
    print(f"Swept Parameter: '{sweep_param}' from {val_lo:.4f} to {val_hi:.4f} ({steps} points)")
    if fixed_overrides:
        print("Fixed Baseline Overrides (held constant throughout sweep):")
        for k, v in sorted(fixed_overrides.items()):
            def_v = all_defaults.get(k, "none")
            print(f"  {k:<18s} = {v:<10} (default was {def_v})")
    else:
        print("Fixed Baseline Overrides: None (all other parameters held at system defaults)")
    print("-" * 75)
    print(f"{sweep_param:>12s} | {'Stable':>6s} | {'Lyapunov':>10s} | {'Clipped':>8s} | {'Used Bins':>10s} | {'Obs Min':>8s} | {'Obs Max':>8s}")
    print("-" * 75)

    for v in values:
        curr_params = dict(params)
        curr_params[sweep_param] = v
        try:
            proc, _ = create_process_from_dict(system_name, curr_params)
            tr = trace(proc, seed=seed)
            st = stability(tr)
            st_flag = "YES" if st["stable"] else "NO"
            obs_lo, obs_hi = st["observed_range"]
            print(f"{v:12.4f} | {st_flag:>6s} | {st['lyapunov']:+10.4f} | {st['clipped']:8.2%} | {st['used_bins']:4d}/{st['n_obs']:<4d} | {obs_lo:8.3f} | {obs_hi:8.3f}")
        except Exception as exc:
            print(f"{v:12.4f} | {'ERROR':>6s} | {str(exc)[:40]}")
    print("-" * 75)


def cmd_balance(system_name: str, params: dict, seed: int = 0) -> None:
    """Compute optimal viscous damping balancing kick injection with friction."""
    info = get_system_param_info(system_name)
    sys_params = {k: params[k] for k in info["system"] if k in params}

    print(f"Solving steady-state energy equilibrium for {system_name}...")
    res = compute_optimal_gamma(
        system_name=system_name,
        m=int(params.get("m", 40)),
        n_steps=int(params.get("n_steps", 10)),
        dt=float(params.get("dt", 0.02)),
        delta_v=float(params.get("delta_v", 0.55)),
        stay=float(params.get("stay", 0.7)),
        alpha=float(params.get("alpha", 0.7)),
        system_params=sys_params,
        seed=seed,
    )
    print("=" * 68)
    print("STEADY-STATE ENERGY EQUILIBRIUM REPORT")
    print("=" * 68)
    print(f"Optimal Damping (gamma*):        {res['gamma_opt']:.4f}")
    print(f"Mean Energy Injected per Tick:   {res['mean_injected']:.4f}")
    print(f"Mean Energy Dissipated per Tick: {res['mean_dissipated']:.4f}")
    print(f"Energy Balance Ratio (bled/inj): {res['balance_ratio']:.1%} (target: 100%)")
    print("=" * 68)


PORTFOLIOS = {
    "predator_prey_mess4": [
        ("regime1_gentle", "Regime 1: Gentle / Fine-Grained Ecology", {"delta_v": 0.18, "kappa": 5.0, "dt": 0.05, "n_steps": 10, "m": 150}),
        ("regime2_baseline", "Regime 2: Balanced Ecological Baseline", {"delta_v": 0.35, "kappa": 5.0, "dt": 0.05, "n_steps": 10, "m": 150}),
        ("regime3_heavily_damped", "Regime 3: Heavily Damped / Short Memory", {"delta_v": 0.35, "kappa": 2.5, "dt": 0.05, "n_steps": 10, "m": 150}),
        ("regime4_wide_orbit", "Regime 4: Wide-Orbit / High-Entropy Swings", {"delta_v": 0.30, "kappa": 7.0, "dt": 0.05, "n_steps": 10, "m": 150}),
        ("regime5_sparse_shocks", "Regime 5: Sparse Episodic Shocks / Fine Sampling", {"delta_v": 0.40, "kappa": 5.0, "dt": 0.04, "n_steps": 25, "m": 150}),
    ],
    "pendulum_mess4": [
        ("regime1_linear", "Regime 1: Near-Linear Harmonic", {"delta_v": 0.20, "gamma": 0.234, "dt": 0.02, "n_steps": 10, "m": 150}),
        ("regime2_balanced", "Regime 2: Balanced Non-Linear", {"delta_v": 0.33, "gamma": 0.234, "dt": 0.02, "n_steps": 10, "m": 150}),
        ("regime3_default", "Regime 3: Default Non-Linear Baseline", {"delta_v": 0.55, "gamma": 1.10, "dt": 0.02, "n_steps": 10, "m": 150}),
        ("regime4_sparse_heavy", "Regime 4: Sparse Heavy Hits", {"delta_v": 0.80, "gamma": 1.00, "dt": 0.04, "n_steps": 15, "m": 150, "omega_max": 25.0}),
        ("regime5_amnesiac", "Regime 5: Heavily Damped Amnesiac", {"delta_v": 0.75, "gamma": 2.50, "dt": 0.02, "n_steps": 10, "m": 150}),
    ],
    "sphere_mess4": [
        ("regime1_conical", "Regime 1: Conical Precession / Low Shocks", {"delta_v": 0.08, "gamma": 0.25, "dt": 0.04, "n_steps": 10, "m": 150}),
        ("regime2_balanced", "Regime 2: Balanced Isotropic Baseline", {"delta_v": 0.15, "gamma": 0.35, "dt": 0.04, "n_steps": 10, "m": 150}),
        ("regime3_damped", "Regime 3: Strongly Damped Swirl", {"delta_v": 0.15, "gamma": 0.70, "dt": 0.04, "n_steps": 10, "m": 150}),
        ("regime4_wide_meridian", "Regime 4: Wide Meridional Excursions", {"delta_v": 0.25, "gamma": 0.40, "dt": 0.04, "n_steps": 10, "m": 150}),
        ("regime5_fine_drift", "Regime 5: Long-Period Geodesic Drift", {"delta_v": 0.12, "gamma": 0.20, "dt": 0.02, "n_steps": 20, "m": 150}),
    ],
}


def cmd_generate_portfolio(system_name: str, out_dir_str: str = "notebooks/outputs", seed: int = 0) -> None:
    """Generate diagnostic reports and 10-panel visual plots for all portfolio regimes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from physics.visualise import plot

    if system_name not in PORTFOLIOS:
        print(f"No preset portfolio defined for {system_name}. Available: {list(PORTFOLIOS.keys())}")
        return

    out_dir = Path(out_dir_str)
    out_dir.mkdir(parents=True, exist_ok=True)
    regimes = PORTFOLIOS[system_name]

    print("=" * 75)
    print(f"GENERATING FULL VISUAL PORTFOLIO FOR {system_name.upper()} ({len(regimes)} REGIMES)")
    print(f"Target directory: {out_dir.resolve()}")
    print("=" * 75)

    for slug, title, p_overrides in regimes:
        proc, _ = create_process_from_dict(system_name, p_overrides)
        tr = trace(proc, seed=seed)
        st = stability(tr)
        st_flag = "PASS (STABLE)" if st["stable"] else "FAIL (UNSTABLE)"

        img_path = out_dir / f"{system_name}_{slug}.png"
        fig = plot(tr, ncols=2, title=f"{system_name.upper()} - {title}")
        fig.savefig(img_path, dpi=110)
        plt.close(fig)

        print(f"\n[Generated] {title}")
        print(f"  File:       {img_path}")
        print(f"  Status:     {st_flag}")
        print(f"  Lyapunov:   {st['lyapunov']:+.4f} / s")
        print(f"  Clipped:    {st['clipped']:.2%}")
        print(f"  Used Bins:  {st['used_bins']} / {st['n_obs']} ({st['used_bins']/st['n_obs']*100:.1f}%)")
        print(f"  Obs Range:  [{st['observed_range'][0]:.3f}, {st['observed_range'][1]:.3f}]")

    print("\n" + "=" * 75)
    print(f"[COMPLETE] All {len(regimes)} diagnostic plots generated in {out_dir}/")
    print("=" * 75)


def main():
    parser = argparse.ArgumentParser(
        description="Mess-4 Universal Physical Diagnostics & Parameter Evaluation Tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument("--system", "-s", default="predator_prey_mess4", choices=sorted(MESSK_CONFIGS),
                        help="Target Mess-4 system")
    parser.add_argument("--mode", choices=["check", "seeds", "sweep", "balance"], default="check",
                        help="Analysis mode: check (single trace), seeds (multi-seed), sweep (parameter scan), balance (energy balance)")
    parser.add_argument("--list-params", action="store_true", help="List all tunable parameters for the specified system and exit")
    parser.add_argument("--generate-portfolio", action="store_true",
                        help="Batch-generate diagnostic plots for all 5 portfolio regimes to notebooks/outputs/")
    parser.add_argument("--save-plot", type=str, default=None,
                        help="Path to save the 10-panel diagnostic plot image, e.g. notebooks/outputs/my_plot.png")

    # Common driver & chain parameters
    parser.add_argument("--m", type=int, default=None, help="Sequence length in ticks (e.g. 40)")
    parser.add_argument("--n-steps", "-n", type=int, default=None, help="Integration steps per tick (e.g. 10)")
    parser.add_argument("--dt", type=float, default=None, help="Integration timestep in seconds (e.g. 0.02 or 0.05)")
    parser.add_argument("--delta-v", "--dv", type=float, default=None, help="Impulse kick scale (e.g. 0.35, 0.55)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--stay", type=float, default=None, help="Markov chain stay probability")
    parser.add_argument("--alpha", type=float, default=None, help="Markov chain emission accuracy")
    parser.add_argument("--obs-bins", type=str, default=None, help="Bins count or tuple, e.g. 181 or '(64, 32)'")

    # Common system parameters (for quick autocomplete)
    parser.add_argument("--gamma", type=float, default=None, help="Damping (Pendulum / Double Pendulum / Sphere)")
    parser.add_argument("--gamma1", type=float, default=None, help="Joint 1 damping (Double Pendulum)")
    parser.add_argument("--gamma2", type=float, default=None, help="Joint 2 damping (Double Pendulum)")
    parser.add_argument("--kappa", type=float, default=None, help="Carrying capacity damping (Predator-Prey)")
    parser.add_argument("--omega-max", type=float, default=None, help="Velocity limit (Pendulum / Double Pendulum)")
    parser.add_argument("--obs-range", type=str, default=None, help="Observation bounds tuple, e.g. '(-3.14, 3.14)'")

    # Generic key-value parameter input
    parser.add_argument("--param", "-p", action="append", help="Set any parameter as name=val, e.g. -p a=1.2 -p l1=1.5")

    # Mode-specific options
    parser.add_argument("--n-seeds", type=int, default=10, help="Number of seeds for --mode seeds")
    parser.add_argument("--sweep-param", type=str, default="kappa", help="ANY parameter name to sweep in --mode sweep")
    parser.add_argument("--sweep-min", type=float, default=1.0, help="Min sweep value")
    parser.add_argument("--sweep-max", type=float, default=10.0, help="Max sweep value")
    parser.add_argument("--sweep-steps", type=int, default=10, help="Number of steps in sweep")

    # Parse known and dynamic unparsed args
    known_args, unparsed = parser.parse_known_args()

    if known_args.list_params:
        cmd_list_params(known_args.system)
        return

    # Combine known args into param dict
    params = {}
    for k, v in vars(known_args).items():
        if v is not None:
            params[k] = v

    # Parse -p name=val pairs
    if known_args.param:
        for item in known_args.param:
            if "=" in item:
                k, v = item.split("=", 1)
                try:
                    params[k.strip()] = ast.literal_eval(v.strip())
                except Exception:
                    try:
                        params[k.strip()] = float(v.strip())
                    except Exception:
                        params[k.strip()] = v.strip()

    # Parse any arbitrary CLI flags like --a 1.2 or --th1_0 0.5
    dynamic_kwargs = parse_dynamic_args(unparsed)
    params.update(dynamic_kwargs)
    if known_args.generate_portfolio:
        cmd_generate_portfolio(known_args.system, seed=known_args.seed)
        return

    # Route to selected mode
    mode = known_args.mode
    if mode == "check":
        cmd_check(known_args.system, params, seed=known_args.seed)
    elif mode == "seeds":
        cmd_seeds(known_args.system, params, n_seeds=known_args.n_seeds)
    elif mode == "sweep":
        cmd_sweep(known_args.system, params, known_args.sweep_param, known_args.sweep_min, known_args.sweep_max, known_args.sweep_steps, seed=known_args.seed)
    elif mode == "balance":
        cmd_balance(known_args.system, params, seed=known_args.seed)


if __name__ == "__main__":
    main()
