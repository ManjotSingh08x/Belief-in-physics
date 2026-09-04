"""Everything you can look at without a transformer: traces, stability, panels.

The question this answers is "is this parameter setting a sane physical system",
which has to be settled before any token from it is worth training on. So every
panel here compares the *driven* trajectory against the *free* one from the same
release condition: the difference between them is the entire causal footprint of
the chain, and if it is invisible the dataset carries no signal, while if it
explodes the dataset is measuring the clamps rather than the physics.

Pure numpy and matplotlib on purpose - no torch - so the simulator side stays
loadable on a machine with no model installed.
"""

from __future__ import annotations

from dataclasses import fields

import numpy as np

#: Phase-plane coordinates per system, by class name. Not derivable from the
#: dataclass: which two state coordinates make a *readable* portrait is a fact
#: about the physics, not about the array layout. Anything unlisted falls back
#: to the first two coordinates, which is right for any (position, velocity) pair.
PHASE_SPACE: dict[str, dict] = {
    "Pendulum": {"x": "theta (rad)", "y": "omega (rad/s)"},
    "DoublePendulum": {"x": "theta1 (rad)", "y": "theta2 (rad)"},
    "PredatorPrey": {
        "x": "prey x", "y": "predator y",
        "project": lambda s: (np.exp(s[..., 0]), np.exp(s[..., 1])),
    },
    "SphereBall": {"sphere": True, "x": "theta (rad)", "y": "psi (rad)"},
}

PANELS = (
    "observable", "energy", "metric", "phase", "states",
    "divergence", "spectrum", "tokens", "belief", "return_map",
)

INK, MUTED, GRID = "#0b0b0b", "#666666", "#e1e0d9"
C_DRIVEN, C_FREE = "#2a78d6", "#888888"
C_A, C_B = "#eb6834", "#1baf7a"
C_ENERGY = "#4a3aa7"
LETTER_COLOURS = ("#2a78d6", "#eb6834", "#1baf7a", "#c0392b")


def tunable_fields(system) -> dict[str, float]:
    """Every scalar knob on a system dataclass, with its current value.

    Derived from the dataclass rather than listed by hand, so a field added to a
    system appears in the notebook without anyone remembering to add it. The
    exclusions are the non-scalars: ranges and identifier lists.
    """
    return {
        f.name: getattr(system, f.name)
        for f in fields(system)
        if f.name not in {"obs_range", "metric_names", "state_names", "observable_names"}
        and isinstance(getattr(system, f.name), (int, float))
        and not isinstance(getattr(system, f.name), bool)
    }


def state_names(system) -> tuple[str, ...]:
    names = getattr(system, "state_names", None)
    if names:
        return tuple(names)
    return tuple(f"z{i}" for i in range(system.initial_state(1).shape[-1]))


def trace(proc, letters=None, seed: int = 0, twin_eps: float = 1e-8) -> dict:
    """One driven trajectory, its undriven twin, and a nearby-start twin.

    Three rollouts rather than one, because each answers a different question.
    `free` is the same release condition with no kicks, so driven-minus-free is
    the chain's effect. `twin` starts `twin_eps` away and is otherwise identical
    to `driven`, so driven-minus-twin measures the system's own sensitivity to
    initial conditions - the number that says whether this parameter setting is
    stable or chaotic, independent of how hard the chain is kicking it.
    """
    rng = np.random.default_rng(seed)
    if letters is None:
        _, letters = proc.chain.sample(rng, 1, proc.m)
        letters = letters[0]
    letters = np.asarray(letters, dtype=np.int64).reshape(-1)
    if len(letters) != proc.m:
        raise ValueError(f"expected {proc.m} letters, got {len(letters)}")

    sysm, actions, dt = proc.system, proc.actions, proc.dt
    z_d = sysm.initial_state(1)
    z_f = sysm.initial_state(1)
    z_t = sysm.initial_state(1) + twin_eps

    states = {"driven": [z_d[0].copy()], "free": [z_f[0].copy()], "twin": [z_t[0].copy()]}
    energy = {k: [float(sysm.energy(z)[0])] for k, z in
              (("driven", z_d), ("free", z_f), ("twin", z_t))}
    obs = {"driven": [], "free": []}
    metric = {"driven": [], "free": []}

    for t in range(proc.m):
        action = actions[letters[t : t + 1]]
        z_d = sysm.kick(z_d, action)
        z_t = sysm.kick(z_t, action)
        for _ in range(proc.n_steps):
            z_d, z_f, z_t = sysm.flow(z_d, dt), sysm.flow(z_f, dt), sysm.flow(z_t, dt)
            for key, z in (("driven", z_d), ("free", z_f), ("twin", z_t)):
                states[key].append(z[0].copy())
                energy[key].append(float(sysm.energy(z)[0]))
            for key, z in (("driven", z_d), ("free", z_f)):
                obs[key].append(proc.channels(z)[0])
                metric[key].append(sysm.metric(z)[0])

    out = {
        "proc": proc,
        "letters": letters,
        "dt": dt,
        "t": np.arange(proc.seq_len) * dt,
        "t_state": np.arange(proc.seq_len + 1) * dt,
        "tick_times": np.arange(proc.m) * proc.n_steps * dt,
        "beliefs": proc.chain.beliefs(letters[None])[0],
    }
    for key in states:
        out[f"states_{key}"] = np.asarray(states[key])
        out[f"energy_{key}"] = np.asarray(energy[key])
    for key in obs:
        out[f"obs_{key}"] = np.asarray(obs[key])
        out[f"metric_{key}"] = np.asarray(metric[key])
        out[f"tokens_{key}"] = proc.discretise(np.asarray(obs[key]))
    out["gap_free"] = np.linalg.norm(out["states_driven"] - out["states_free"], axis=-1)
    out["gap_twin"] = np.linalg.norm(out["states_driven"] - out["states_twin"], axis=-1)
    return out


def lyapunov(tr: dict, eps: float = 1e-8) -> float:
    """Finite-time exponent from the nearby-start twin, per unit time.

    Fitted over the stretch before the twins saturate, since once they are a
    system-size apart the separation stops growing and the slope collapses
    towards zero, which would read as "stable" for the most chaotic settings.
    """
    gap = np.maximum(tr["gap_twin"], 1e-300)
    ceiling = 0.1 * max(float(np.abs(tr["states_driven"]).max()), 1e-9)
    grown = np.flatnonzero(gap > ceiling)
    stop = int(grown[0]) if len(grown) else len(gap)
    stop = max(stop, 8)
    if stop <= 2:
        return 0.0
    t, y = tr["t_state"][:stop], np.log(gap[:stop] / max(eps, 1e-300))
    return float(np.polyfit(t, y, 1)[0])


def stability(tr: dict) -> dict:
    """The numbers that decide whether a parameter setting is usable.

    A setting fails for one of three separable reasons, so they are reported
    separately rather than as one score: the observable leaves its range and the
    tokens saturate (`clipped`), the trajectory diverges from its own twin
    (`lyapunov`), or the driven and free runs are indistinguishable so the
    letters left no trace to learn from (`gap_free_mean`).
    """
    proc = tr["proc"]
    report = proc.bin_report(tr["obs_driven"])
    exponent = lyapunov(tr)
    e = tr["energy_free"]
    drift = float((e[-1] - e[0]) / max(abs(e[0]), 1e-12))
    reasons = []
    if report["clipped"] > 0.01:
        reasons.append(f"clips {report['clipped']:.1%} of samples")
    if exponent > 0.05:
        reasons.append(f"diverges from its own twin (lambda={exponent:+.2f}/s)")
    if tr["gap_free"].mean() < 1e-3:
        reasons.append("kicks leave no visible trace")
    if report["used_bins"] < 0.1 * report["n_obs"]:
        reasons.append(f"uses {report['used_bins']}/{report['n_obs']} bins")
    return {
        "lyapunov": exponent,
        "clipped": report["clipped"],
        "per_channel_clipped": report["per_channel_clipped"],
        "used_bins": report["used_bins"],
        "n_obs": report["n_obs"],
        "energy_drift_free": drift,
        "gap_free_mean": float(tr["gap_free"].mean()),
        "gap_free_max": float(tr["gap_free"].max()),
        "observed_range": report["observed_range"],
        "obs_range": report["obs_range"],
        "stable": not reasons,
        "reasons": reasons,
    }


def _mark_kicks(ax, tr) -> None:
    for tt, letter in zip(tr["tick_times"], tr["letters"]):
        ax.axvline(tt, color=LETTER_COLOURS[letter % len(LETTER_COLOURS)],
                   alpha=0.25, linestyle=":", lw=1.0)


def _panel_observable(ax, tr) -> None:
    proc = tr["proc"]
    for c, name in enumerate(proc.channel_names):
        ax.plot(tr["t"], tr["obs_driven"][:, c], color=(C_DRIVEN, C_A)[c % 2], lw=1.4,
                label=f"driven {name}")
        ax.plot(tr["t"], tr["obs_free"][:, c], color=C_FREE, lw=1.1, ls="--",
                label=f"free {name}")
    for (lo, hi) in proc.obs_ranges:
        ax.axhline(lo, color="#c0392b", lw=0.7, alpha=0.5)
        ax.axhline(hi, color="#c0392b", lw=0.7, alpha=0.5)
    twin = ax.twinx()
    twin.step(tr["t"], tr["tokens_driven"], color=MUTED, alpha=0.3, where="post")
    twin.set_ylabel("token", color=MUTED)
    _mark_kicks(ax, tr)
    ax.set_ylabel("observable")
    ax.set_title("observable and its token (red = range edge)")
    ax.legend(fontsize=7, loc="upper right")


def _panel_energy(ax, tr) -> None:
    ax.plot(tr["t_state"], tr["energy_driven"], color=C_ENERGY, lw=1.4, label="driven")
    ax.plot(tr["t_state"], tr["energy_free"], color=C_FREE, lw=1.2, ls="--", label="free")
    _mark_kicks(ax, tr)
    ax.set_ylabel("energy")
    ax.set_title("energy: kicks inject, damping dissipates")
    ax.legend(fontsize=7)


def _panel_metric(ax, tr) -> None:
    names = tr["proc"].system.metric_names
    for c, name in enumerate(names):
        colour = (C_A, C_B)[c % 2]
        ax.plot(tr["t"], tr["metric_driven"][:, c], color=colour, lw=1.3, label=f"driven {name}")
        ax.plot(tr["t"], tr["metric_free"][:, c], color=colour, lw=1.0, ls="--",
                alpha=0.6, label=f"free {name}")
    ax.set_ylabel("metric")
    ax.set_title("probed metric (the physical quantity the readout targets)")
    ax.legend(fontsize=7)


def _phase_xy(tr, key):
    spec = PHASE_SPACE.get(type(tr["proc"].system).__name__, {})
    s = tr[f"states_{key}"]
    if "project" in spec:
        return spec["project"](s)
    return s[..., 0], s[..., 1]


def _panel_phase(ax, tr) -> None:
    spec = PHASE_SPACE.get(type(tr["proc"].system).__name__, {})
    if spec.get("sphere"):
        return _panel_sphere(ax, tr)
    for key, colour, ls in (("free", C_FREE, "--"), ("driven", C_DRIVEN, "-")):
        x, y = _phase_xy(tr, key)
        ax.plot(x, y, color=colour, lw=1.1, ls=ls, alpha=0.85, label=key)
    x, y = _phase_xy(tr, "driven")
    ax.scatter(x[0], y[0], color="#1baf7a", s=45, zorder=5, label="start")
    ax.scatter(x[-1], y[-1], color="#c0392b", s=45, zorder=5, label="end")
    ax.set_xlabel(spec.get("x", "z0"))
    ax.set_ylabel(spec.get("y", "z1"))
    ax.set_title("phase portrait")
    ax.legend(fontsize=7)


def _panel_sphere(ax, tr) -> None:
    """The sphere's portrait is the sphere itself; a flat plot hides the winding."""
    def xyz(key):
        s = tr[f"states_{key}"]
        th, psi = s[:, 0], s[:, 1]
        return np.sin(th) * np.cos(psi), np.sin(th) * np.sin(psi), -np.cos(th)

    # The wireframe spans only the band theta actually visits, padded. Drawing the
    # whole hemisphere is honest but forces a box the trajectory rattles around in,
    # and a shallow cone then reads as a flat line.
    th_all = tr["states_driven"][:, 0]
    pad = 0.25 * max(float(np.ptp(th_all)), 0.05)
    v = np.linspace(max(th_all.min() - pad, 1e-3), min(th_all.max() + pad, np.pi / 2), 12)
    u = np.linspace(0, 2 * np.pi, 32)
    ax.plot_wireframe(np.outer(np.cos(u), np.sin(v)), np.outer(np.sin(u), np.sin(v)),
                      -np.outer(np.ones_like(u), np.cos(v)), color=GRID, alpha=0.4, lw=0.5)
    span = float(np.sin(v).max())
    ax.set_xlim(-span, span)
    ax.set_ylim(-span, span)
    ax.set_box_aspect((1.0, 1.0, 0.55))
    for key, colour, ls in (("free", C_FREE, "--"), ("driven", C_DRIVEN, "-")):
        x, y, z = xyz(key)
        ax.plot(x, y, z, color=colour, lw=1.3, ls=ls, label=key)
    x, y, z = xyz("driven")
    ax.scatter([x[0]], [y[0]], [z[0]], color="#1baf7a", s=35, label="start")
    ax.scatter([x[-1]], [y[-1]], [z[-1]], color="#c0392b", s=35, label="end")
    ax.set_title("trajectory on the sphere")
    ax.legend(fontsize=7)


def _panel_states(ax, tr) -> None:
    names = state_names(tr["proc"].system)
    for c, name in enumerate(names):
        ax.plot(tr["t_state"], tr["states_driven"][:, c], lw=1.2, label=name)
    _mark_kicks(ax, tr)
    ax.set_ylabel("state")
    ax.set_title("every state coordinate (driven)")
    ax.legend(fontsize=7, ncol=2)


def _panel_divergence(ax, tr) -> None:
    ax.semilogy(tr["t_state"], np.maximum(tr["gap_free"], 1e-16), color=C_DRIVEN, lw=1.3,
                label="driven vs free (chain's footprint)")
    ax.semilogy(tr["t_state"], np.maximum(tr["gap_twin"], 1e-16), color="#c0392b", lw=1.3,
                label="driven vs 1e-8 twin (sensitivity)")
    ax.set_ylabel("|state gap|")
    ax.set_title(f"divergence, lambda = {lyapunov(tr):+.3f} / s")
    ax.legend(fontsize=7)


def _panel_spectrum(ax, tr) -> None:
    for key, colour in (("driven", C_DRIVEN), ("free", C_FREE)):
        x = tr[f"obs_{key}"][:, 0]
        x = x - x.mean()
        freq = np.fft.rfftfreq(len(x), d=tr["dt"])
        ax.semilogy(freq[1:], np.maximum(np.abs(np.fft.rfft(x))[1:], 1e-12),
                    color=colour, lw=1.1, label=key)
    ax.set_xlabel("frequency (Hz)")
    ax.set_title("spectrum of channel 0 (a clean peak = periodic, broadband = chaotic)")
    ax.legend(fontsize=7)


def _panel_tokens(ax, tr) -> None:
    proc = tr["proc"]
    counts = np.bincount(tr["tokens_driven"], minlength=proc.n_obs)
    ax.bar(np.arange(proc.n_obs), counts, width=1.0, color=C_DRIVEN)
    used = int((counts > 0).sum())
    ax.set_xlabel("token")
    ax.set_ylabel("count")
    ax.set_title(f"vocabulary use: {used}/{proc.n_obs} bins in this sequence")


def _panel_belief(ax, tr) -> None:
    for k in range(tr["beliefs"].shape[1]):
        ax.step(np.arange(tr["proc"].m), tr["beliefs"][:, k], where="post", lw=1.2,
                color=LETTER_COLOURS[k % len(LETTER_COLOURS)], label=f"P(mood {k})")
    ax.scatter(np.arange(tr["proc"].m), np.full(tr["proc"].m, -0.05),
               c=[LETTER_COLOURS[l % len(LETTER_COLOURS)] for l in tr["letters"]],
               marker="s", s=22)
    ax.set_ylim(-0.12, 1.02)
    ax.set_xlabel("tick (squares = letters)")
    ax.set_title("exact predictive belief - identical across all four systems")
    ax.legend(fontsize=7, ncol=2)


def _panel_return_map(ax, tr) -> None:
    x = tr["obs_driven"][:, 0]
    ax.scatter(x[:-1], x[1:], s=4, alpha=0.5, color=C_DRIVEN, label="driven")
    f = tr["obs_free"][:, 0]
    ax.scatter(f[:-1], f[1:], s=4, alpha=0.5, color=C_FREE, label="free")
    ax.set_xlabel("obs[n]")
    ax.set_ylabel("obs[n+1]")
    ax.set_title("return map (a thin curve = deterministic, a cloud = folded)")
    ax.legend(fontsize=7)


_PANEL_FUNCS = {
    "observable": _panel_observable, "energy": _panel_energy, "metric": _panel_metric,
    "phase": _panel_phase, "states": _panel_states, "divergence": _panel_divergence,
    "spectrum": _panel_spectrum, "tokens": _panel_tokens, "belief": _panel_belief,
    "return_map": _panel_return_map,
}


def plot(tr: dict, panels=PANELS, ncols: int = 2, height: float = 3.4, title: str = ""):
    """Draw the chosen panels for one trace, in a grid."""
    import matplotlib.pyplot as plt

    panels = [p for p in panels if p in _PANEL_FUNCS]
    if not panels:
        raise ValueError(f"no known panels in {panels}; have {sorted(_PANEL_FUNCS)}")
    nrows = int(np.ceil(len(panels) / ncols))
    fig = plt.figure(figsize=(7.0 * ncols, height * nrows))
    is_sphere = PHASE_SPACE.get(type(tr["proc"].system).__name__, {}).get("sphere")
    for i, name in enumerate(panels, start=1):
        kw = {"projection": "3d"} if (name == "phase" and is_sphere) else {}
        ax = fig.add_subplot(nrows, ncols, i, **kw)
        _PANEL_FUNCS[name](ax, tr)
        if not kw:
            ax.grid(True, color=GRID, ls="--", alpha=0.6)
    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97 if title else 1))
    return fig


def compare(traces: dict[str, dict], panel: str = "observable", height: float = 3.2):
    """The same panel for several systems, stacked, so they are read side by side."""
    import matplotlib.pyplot as plt

    is_sphere = {k: PHASE_SPACE.get(type(t["proc"].system).__name__, {}).get("sphere")
                 for k, t in traces.items()}
    fig = plt.figure(figsize=(9.0, height * len(traces)))
    for i, (name, tr) in enumerate(traces.items(), start=1):
        kw = {"projection": "3d"} if (panel == "phase" and is_sphere[name]) else {}
        ax = fig.add_subplot(len(traces), 1, i, **kw)
        _PANEL_FUNCS[panel](ax, tr)
        ax.set_title(f"{name}: {ax.get_title()}", fontsize=10)
        if not kw:
            ax.grid(True, color=GRID, ls="--", alpha=0.6)
    fig.tight_layout()
    return fig


def sweep(make_proc, field: str, values, seed: int = 0) -> list[dict]:
    """`stability` at each value of one parameter - the stable-region scan.

    `make_proc(**{field: value})` is passed in rather than a config name, so the
    same sweep works over a system field, a driver field or the chain.
    """
    rows = []
    for value in values:
        tr = trace(make_proc(**{field: value}), seed=seed)
        rows.append({field: value, **stability(tr)})
    return rows


def _demo() -> None:
    from .messk_configs import MESSK_CONFIGS, make_process

    for name in sorted(MESSK_CONFIGS):
        proc = make_process(name, m=8)
        tr = trace(proc, seed=1)
        assert tr["states_driven"].shape[0] == proc.seq_len + 1
        assert tr["obs_driven"].shape == (proc.seq_len, len(proc.obs_bins))
        assert np.array_equal(tr["tokens_driven"], proc.rollout(tr["letters"])["tokens"][0]), \
            f"{name}: trace must reproduce rollout exactly"
        assert tr["gap_free"][0] == 0, f"{name}: both runs share the release condition"
        assert tr["gap_free"][1:].max() > 0, f"{name}: the kicks must move the state"
        s = stability(tr)
        assert 0.0 <= s["clipped"] <= 1.0 and s["used_bins"] >= 1
        assert isinstance(s["stable"], bool)

    # A chaotic setting must read as more sensitive than a heavily damped one.
    from .systems.double_pendulum import DoublePendulum
    calm = trace(make_process("double_pendulum_mess4", m=8,
                              system={"gamma1": 8.0, "gamma2": 8.0}), seed=1)
    wild = trace(make_process("double_pendulum_mess4", m=8,
                              system={"gamma1": 0.0, "gamma2": 0.0}), seed=1)
    assert lyapunov(wild) > lyapunov(calm), (lyapunov(wild), lyapunov(calm))
    assert set(tunable_fields(DoublePendulum())) >= {"g", "l1", "m1", "gamma1", "w1_0"}
    assert len(state_names(DoublePendulum())) == 4
    print(f"visualise ok (chaotic lambda {lyapunov(wild):+.2f} vs damped {lyapunov(calm):+.2f})")


if __name__ == "__main__":
    _demo()
