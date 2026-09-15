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
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection

#: Blue-to-yellow time progression colormap for physical trajectories
TIME_GRADIENT = LinearSegmentedColormap.from_list(
    "time_gradient",
    ["#1a5fb4", "#00b4d8", "#2ec4b6", "#ffb703", "#fb8500"]
)

#: Phase-plane coordinates per system, by class name. Not derivable from the
#: dataclass: which two state coordinates make a *readable* diagram is a fact
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
    "causal_effect",
)

INK, MUTED, GRID = "#0b0b0b", "#666666", "#e1e0d9"
C_DRIVEN, C_FREE = "#2a78d6", "#888888"
C_A, C_B = "#eb6834", "#1baf7a"
C_ENERGY = "#4a3aa7"
LETTER_COLOURS = (
    "#8b0000",  # 0: - strong dv (dark deep red)
    "#ff7043",  # 1: - weak dv (light red / warm orange)
    "#48c774",  # 2: + weak dv (light green)
    "#005a20",  # 3: + strong dv (dark green)
)

TICKER_CONFIG = {
    0: {"name": "- strong dv", "marker": "v", "color": "#8b0000", "size": 48, "label": "-strong dv"},
    1: {"name": "- weak dv",   "marker": "v", "color": "#ff7043", "size": 26, "label": "-weak dv"},
    2: {"name": "+ weak dv",   "marker": "^", "color": "#48c774", "size": 26, "label": "+weak dv"},
    3: {"name": "+ strong dv", "marker": "^", "color": "#005a20", "size": 48, "label": "+strong dv"},
}


def ticker_for_letter(letter: int) -> dict:
    """Ticker styling: direction (triangle up/down), strength (color & size)."""
    return TICKER_CONFIG[int(letter) % len(TICKER_CONFIG)]


def ticker_legend_elements():
    """Legend proxy items showing the 4 perturbation ticker types."""
    from matplotlib.lines import Line2D
    return [
        Line2D([0], [0], marker="v", color="w", markerfacecolor="#8b0000",
               markersize=7, label="-strong dv"),
        Line2D([0], [0], marker="v", color="w", markerfacecolor="#ff7043",
               markersize=5, label="-weak dv"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#48c774",
               markersize=5, label="+weak dv"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#005a20",
               markersize=7, label="+strong dv"),
    ]


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
    min_bins = min(0.1 * report["n_obs"], max(10, 0.1 * len(tr["obs_driven"])))
    if report["used_bins"] < min_bins:
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


def _mark_kicks(ax, tr, y_data=None) -> None:
    proc = tr["proc"]
    for k, (tt, letter) in enumerate(zip(tr["tick_times"], tr["letters"])):
        cfg = ticker_for_letter(letter)
        ax.axvline(tt, color=cfg["color"], alpha=0.25, linestyle=":", lw=1.0)
        if y_data is not None:
            idx = min(k * proc.n_steps, len(y_data) - 1)
            ax.scatter([tt], [y_data[idx]], marker=cfg["marker"], color=cfg["color"],
                       s=cfg["size"], zorder=5, edgecolors="#ffffff", linewidths=0.5)


def _panel_observable(ax, tr) -> None:
    proc = tr["proc"]
    for c, name in enumerate(proc.channel_names):
        # Mask wrap jumps if coordinate is an azimuthal/wrapping angle
        y_d = tr["obs_driven"][:, c].copy()
        y_f = tr["obs_free"][:, c].copy()
        if "psi" in name.lower() or "th" in name.lower():
            j_d = np.abs(np.diff(y_d, prepend=y_d[0])) > np.pi
            j_f = np.abs(np.diff(y_f, prepend=y_f[0])) > np.pi
            y_d[j_d] = np.nan
            y_f[j_f] = np.nan

        ax.plot(tr["t"], y_d, color=(C_DRIVEN, C_A)[c % 2], lw=1.4,
                label=f"driven {name}")
        ax.plot(tr["t"], y_f, color=C_FREE, lw=1.1, ls="--",
                label=f"free {name}")
    for (lo, hi) in proc.obs_ranges:
        ax.axhline(lo, color="#c0392b", lw=0.7, alpha=0.5)
        ax.axhline(hi, color="#c0392b", lw=0.7, alpha=0.5)

    recon_obs = proc.undiscretise(tr["tokens_driven"])
    twin = None
    if len(proc.obs_bins) == 1:
        twin = ax.twinx()
        twin.step(tr["t"], tr["tokens_driven"], color=MUTED, alpha=0.3, where="post")
        twin.set_ylabel("token", color=MUTED)
        twin.grid(False)
        (lo, hi), bins = proc.obs_ranges[0], proc.obs_bins[0]
        def _sync_twin(a):
            y_l, y_h = a.get_ylim()
            twin.set_ylim(
                (y_l - lo) / (hi - lo) * (bins - 1),
                (y_h - lo) / (hi - lo) * (bins - 1),
            )
        ax.callbacks.connect("ylim_changed", _sync_twin)
    else:
        # Multi-channel system: reconvert tokens and mask wrap jumps
        step_colors = ("#e67e22", "#9b59b6")
        for c, name in enumerate(proc.channel_names):
            col = step_colors[c % len(step_colors)]
            y_tok = recon_obs[:, c].copy()
            if "psi" in name.lower() or "th" in name.lower():
                j_tok = np.abs(np.diff(y_tok, prepend=y_tok[0])) > np.pi
                y_tok[j_tok] = np.nan
            ax.step(tr["t"], y_tok, color=col, lw=1.1, ls=":", where="post", alpha=0.75,
                    label=f"token {name} (recon)")

    _mark_kicks(ax, tr, y_data=tr["obs_driven"][:, 0])
    ax.set_ylabel("observable (rad)" if "theta" in proc.channel_names[0] else "observable")
    title_suffix = f" & tokens reconverted to ({', '.join(proc.channel_names)})" if len(proc.obs_bins) > 1 else " and its token"
    ax.set_title(f"observable{title_suffix} (red = range edge)")
    y_lo, y_hi = ax.get_ylim()
    ax.set_ylim(y_lo, y_hi + 0.22 * (y_hi - y_lo))
    if twin is not None:
        _sync_twin(ax)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles + ticker_legend_elements(), fontsize=7, loc="upper right", ncol=3)


def _panel_energy(ax, tr) -> None:
    ax.plot(tr["t_state"], tr["energy_driven"], color=C_ENERGY, lw=1.4, label="driven")
    ax.plot(tr["t_state"], tr["energy_free"], color=C_FREE, lw=1.2, ls="--", label="free")
    _mark_kicks(ax, tr, y_data=tr["energy_driven"])
    ax.set_ylabel("energy")
    ax.set_title("energy: kicks inject, damping dissipates")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles + ticker_legend_elements(), fontsize=7, loc="upper right", ncol=3)


def _panel_metric(ax, tr) -> None:
    names = tr["proc"].system.metric_names
    for c, name in enumerate(names):
        colour = (C_A, C_B)[c % 2]
        ax.plot(tr["t"], tr["metric_driven"][:, c], color=colour, lw=1.3, label=f"driven {name}")
        ax.plot(tr["t"], tr["metric_free"][:, c], color=colour, lw=1.0, ls="--",
                alpha=0.6, label=f"free {name}")
    _mark_kicks(ax, tr, y_data=tr["metric_driven"][:, 0])
    ax.set_ylabel("metric")
    ax.set_title("probed metric (the physical quantity the readout targets)")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles + ticker_legend_elements(), fontsize=7, loc="upper right", ncol=3)


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

    # Free run in dashed subtle line
    xf, yf = _phase_xy(tr, "free")
    ax.plot(xf, yf, color=C_FREE, lw=1.0, ls="--", alpha=0.6, label="free")

    # Driven trajectory in blue-to-yellow time gradient
    xd, yd = _phase_xy(tr, "driven")
    points = np.array([xd, yd]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap=TIME_GRADIENT, norm=plt.Normalize(0, 1), lw=1.8, alpha=0.9)
    lc.set_array(np.linspace(0, 1, len(segments)))
    ax.add_collection(lc)

    ax.scatter(xd[0], yd[0], color="#1a5fb4", s=50, edgecolors="#ffffff", lw=1.2, zorder=6, label="start")
    ax.scatter(xd[-1], yd[-1], color="#fb8500", s=50, edgecolors="#ffffff", lw=1.2, zorder=6, label="end")

    ax.set_xlabel(spec.get("x", "z0"))
    ax.set_ylabel(spec.get("y", "z1"))
    ax.set_title("phase diagram (time: blue \u2192 yellow)")
    ax.legend(fontsize=7, loc="upper right")


def _panel_sphere(ax, tr) -> None:
    """The sphere's phase diagram is the sphere itself; drawn as a clean, enlarged spherical bowl."""
    def xyz(key):
        s = tr[f"states_{key}"]
        th, psi = s[:, 0], s[:, 1]
        return np.sin(th) * np.cos(psi), np.sin(th) * np.sin(psi), -np.cos(th)

    th_all = tr["states_driven"][:, 0]
    th_max = min(max(float(th_all.max()) * 1.15, 0.75), np.pi * 0.48)

    # 1. Draw the spherical bowl from bottom pole (theta=0, z=-1) up to th_max
    v = np.linspace(0.0, th_max, 16)
    u = np.linspace(0, 2 * np.pi, 32)
    X_bowl = np.outer(np.cos(u), np.sin(v))
    Y_bowl = np.outer(np.sin(u), np.sin(v))
    Z_bowl = -np.outer(np.ones_like(u), np.cos(v))

    # Subtle wireframe grid of the bowl
    ax.plot_wireframe(X_bowl, Y_bowl, Z_bowl, color=GRID, alpha=0.35, lw=0.5)

    # Rim circle at top of visited bowl
    u_rim = np.linspace(0, 2 * np.pi, 64)
    ax.plot(np.sin(th_max) * np.cos(u_rim), np.sin(th_max) * np.sin(u_rim),
        -np.cos(th_max) * np.ones_like(u_rim), color="#95a5a6", ls=":", lw=0.8, alpha=0.6)

    # Mark bottom pole (0, 0, -1) as a reference anchor
    ax.scatter([0], [0], [-1.0], color="#7f8c8d", marker="+", s=40, alpha=0.6)

    # 2. Free trajectory (subtle dashed grey)
    xf, yf, zf = xyz("free")
    ax.plot(xf, yf, zf, color="#95a5a6", lw=1.0, ls="--", alpha=0.55)

    # 3. Driven trajectory in blue-to-yellow gradient
    xd, yd, zd = xyz("driven")
    points = np.array([xd, yd, zd]).T.reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    lc = Line3DCollection(segments, cmap=TIME_GRADIENT, norm=plt.Normalize(0, 1))
    lc.set_array(np.linspace(0, 1, len(segments)))
    lc.set_linewidth(2.2)
    ax.add_collection3d(lc)

    # Only mark start and end points
    ax.scatter([xd[0]], [yd[0]], [zd[0]], color="#1a5fb4", s=55, edgecolors="#ffffff", lw=1.2, zorder=8)
    ax.text(xd[0], yd[0], zd[0] + 0.03, "start", color="#1a5fb4", fontsize=8.5, fontweight="bold", zorder=9)

    ax.scatter([xd[-1]], [yd[-1]], [zd[-1]], color="#fb8500", s=55, edgecolors="#ffffff", lw=1.2, zorder=8)
    ax.text(xd[-1], yd[-1], zd[-1] + 0.03, "end", color="#fb8500", fontsize=8.5, fontweight="bold", zorder=9)

    # 4. Framing, zoom, and clean transparent panes
    span_xy = float(np.sin(th_max))
    z_min = -1.0
    z_max = -float(np.cos(th_max))
    span_z = z_max - z_min

    ax.set_xlim(-span_xy * 1.02, span_xy * 1.02)
    ax.set_ylim(-span_xy * 1.02, span_xy * 1.02)
    ax.set_zlim(z_min - 0.02, z_max + 0.02)

    # Zoom enlarged to fill frame (zoom=1.75 eliminates empty margins)
    ax.set_box_aspect((1.0, 1.0, max(span_z / span_xy, 0.45)), zoom=1.75)
    ax.view_init(elev=24, azim=45)

    # Clean transparent panes
    for p in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        p.fill = False
        p.set_edgecolor("none")

    ax.tick_params(labelsize=6.5, pad=-4)
    ax.set_xlabel("X (east)", fontsize=7.5, labelpad=-7)
    ax.set_ylabel("Y (north)", fontsize=7.5, labelpad=-7)
    ax.set_zlabel("Z (up)", fontsize=7.5, labelpad=-7)
    ax.set_title("trajectory on the sphere (time: blue \u2192 yellow)", fontsize=9.5)



def _panel_states(ax, tr) -> None:
    proc = tr["proc"]
    names = state_names(proc.system)
    t = tr["t_state"]
    states = tr["states_driven"]

    # Detect if state has rate coordinates with very different scales from angles (e.g. Sphere, Double Pendulum)
    has_rates = len(names) >= 3 and any("dot" in n.lower() or n.lower() in ("dtheta", "dpsi", "w1", "w2", "omega") for n in names)

    if has_rates:
        angle_indices = [i for i, n in enumerate(names) if not ("dot" in n.lower() or n.lower() in ("dtheta", "dpsi", "w1", "w2", "omega"))]
        rate_indices = [i for i, n in enumerate(names) if i not in angle_indices]

        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
        for idx in angle_indices:
            name = names[idx]
            vals = states[:, idx].copy()
            if "psi" in name.lower() or "th" in name.lower():
                jumps = np.abs(np.diff(vals, prepend=vals[0])) > np.pi
                vals[jumps] = np.nan
            ax.plot(t, vals, lw=1.3, color=colors[idx % len(colors)], label=f"{name} (rad)")
        ax.set_ylabel("angle (rad)")

        twin = ax.twinx()
        twin.grid(False)
        for idx in rate_indices:
            name = names[idx]
            vals = states[:, idx]
            twin.plot(t, vals, lw=1.1, ls="--", color=colors[idx % len(colors)], alpha=0.75, label=f"{name} (rad/s)")
        twin.set_ylabel("rate (rad/s)", color="#555555")

        _mark_kicks(ax, tr, y_data=states[:, angle_indices[0]])
        ax.set_title("state coordinates: angles (left) & rates (twin right)")

        # Provide clean headroom so the multi-row legend doesn't overlap curves
        y_a_lo, y_a_hi = ax.get_ylim()
        ax.set_ylim(y_a_lo, y_a_hi + 0.28 * (y_a_hi - y_a_lo))
        y_r_lo, y_r_hi = twin.get_ylim()
        twin.set_ylim(y_r_lo, y_r_hi + 0.28 * (y_r_hi - y_r_lo))

        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = twin.get_legend_handles_labels()
        ax.legend(handles=h1 + h2 + ticker_legend_elements(), fontsize=6.5, ncol=min(4, len(names)), loc="upper right")
    else:
        for c, name in enumerate(names):
            vals = states[:, c].copy()
            if "psi" in name.lower() or "th" in name.lower():
                jumps = np.abs(np.diff(vals, prepend=vals[0])) > np.pi
                vals[jumps] = np.nan
            ax.plot(t, vals, lw=1.2, label=name)
        _mark_kicks(ax, tr, y_data=states[:, 0])
        ax.set_ylabel("state")
        ax.set_title("every state coordinate (driven)")
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles + ticker_legend_elements(), fontsize=7, ncol=3, loc="upper right")


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
    if len(proc.obs_bins) == 1:
        counts = np.bincount(tr["tokens_driven"], minlength=proc.n_obs)
        ax.bar(np.arange(proc.n_obs), counts, width=1.0, color=C_DRIVEN)
        used = int((counts > 0).sum())
        ax.set_xlabel("token")
        ax.set_ylabel("count")
        ax.set_title(f"vocabulary use: {used}/{proc.n_obs} bins in this sequence")
    else:
        # Multi-channel system: decode combined tokens back to per-channel physical coordinates.
        recon = proc.undiscretise(tr["tokens_driven"])
        name0 = proc.channel_names[0] if len(proc.channel_names) > 0 else "ch0"
        name1 = proc.channel_names[1] if len(proc.channel_names) > 1 else "ch1"
        (lo0, hi0), (lo1, hi1) = proc.obs_ranges[:2]
        b0, b1 = proc.obs_bins[:2]

        H, xedges, yedges = np.histogram2d(
            recon[:, 0], recon[:, 1],
            bins=[b0, b1],
            range=[(lo0, hi0), (lo1, hi1)],
        )
        ax.imshow(
            H.T, origin="lower", extent=[lo0, hi0, lo1, hi1],
            aspect="auto", cmap="Blues", interpolation="nearest", alpha=0.65
        )

        # Build continuous path segments without wrap-around cuts in channel 1 (e.g. psi)
        th = recon[:, 0]
        psi = recon[:, 1]
        is_angle = "psi" in name1.lower() or "th" in name1.lower()
        jumps = np.abs(np.diff(psi, prepend=psi[0])) > np.pi if is_angle else np.zeros_like(psi, dtype=bool)

        segments = []
        t_vals = []
        N = len(th)
        for i in range(N - 1):
            if not jumps[i + 1]:
                segments.append([[th[i], psi[i]], [th[i + 1], psi[i + 1]]])
                t_vals.append(i / N)

        if segments:
            lc = LineCollection(segments, cmap=TIME_GRADIENT, norm=plt.Normalize(0, 1), lw=1.6, alpha=0.9)
            lc.set_array(np.array(t_vals))
            ax.add_collection(lc)

        ax.scatter(th[0], psi[0], color="#1a5fb4", s=45, edgecolors="#ffffff", lw=1.0, zorder=6, label="start")
        ax.scatter(th[-1], psi[-1], color="#fb8500", s=45, edgecolors="#ffffff", lw=1.0, zorder=6, label="end")
        ax.set_xlabel(f"{name0} (rad)" if "th" in name0.lower() else name0)
        ax.set_ylabel(f"{name1} (rad)" if "psi" in name1.lower() or "th" in name1.lower() else name1)
        used = int((np.bincount(tr["tokens_driven"], minlength=proc.n_obs) > 0).sum())
        ax.set_title(f"token grid ({b0}\u00d7{b1}): {used}/{proc.n_obs} cells (blue \u2192 yellow)")
        ax.legend(fontsize=7, loc="upper right")


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


def _panel_causal_effect(ax, tr, tick: int = 4) -> None:
    """One letter's causal effect: counterfactual rollout branching at `tick`."""
    proc = tr["proc"]
    letters = tr["letters"]
    k = proc.chain.n_states

    tick = min(tick, max(0, proc.m - 1))
    variants = np.tile(letters, (k, 1))
    variants[:, tick] = np.arange(k)
    out = proc.rollout(variants)

    t = np.arange(proc.seq_len) * proc.dt
    t_branch = tick * proc.n_steps * proc.dt

    for l in range(k):
        is_drawn = (l == letters[tick])
        style = dict(lw=2.0, alpha=1.0, zorder=4) if is_drawn else dict(lw=1.1, alpha=0.65, zorder=3)
        cfg = ticker_for_letter(l)
        label = f"{cfg['name']}" + (" (drawn)" if is_drawn else "")
        ax.plot(t, out["observable"][l, :, 0], color=cfg["color"], label=label, **style)

    for (lo, hi) in proc.obs_ranges[:1]:
        ax.axhline(lo, color="#c0392b", lw=0.7, alpha=0.4, ls=":")
        ax.axhline(hi, color="#c0392b", lw=0.7, alpha=0.4, ls=":")

    branch = tick * proc.n_steps
    tok = out["tokens"][:, branch:]

    if len(proc.obs_bins) == 1:
        twin = ax.twinx()
        twin.grid(False)
        for l in range(k):
            is_drawn = (l == letters[tick])
            style = dict(lw=1.5, alpha=0.35, zorder=4) if is_drawn else dict(lw=0.8, alpha=0.2, zorder=3)
            cfg = ticker_for_letter(l)
            twin.step(t, out["tokens"][l], color=cfg["color"], where="post", **style)
        twin.set_ylabel("token", color=MUTED)
        (lo, hi), bins = proc.obs_ranges[0], proc.obs_bins[0]
        y_lo, y_hi = ax.get_ylim()
        twin.set_ylim(
            (y_lo - lo) / (hi - lo) * (bins - 1),
            (y_hi - lo) / (hi - lo) * (bins - 1),
        )
        sep = np.abs(tok[:, None, :] - tok[None, :, :]).max(-1).mean()
        sep_label = f"sep = {sep:.1f} bins"
    else:
        # Multi-channel: reconvert tokens to physical angles
        for l in range(k):
            is_drawn = (l == letters[tick])
            style = dict(lw=1.5, alpha=0.4, zorder=4) if is_drawn else dict(lw=0.8, alpha=0.2, zorder=3)
            cfg = ticker_for_letter(l)
            recon_l = proc.undiscretise(out["tokens"][l])
            ax.step(t, recon_l[:, 0], color=cfg["color"], ls=":", where="post", **style)
        indices = proc.tokens_to_indices(tok)
        diff = indices[:, None, :, :] - indices[None, :, :, :]
        sep = np.linalg.norm(diff, axis=-1).max(-1).mean()
        sep_label = f"grid sep = {sep:.1f} bins"

    ax.axvline(t_branch, color=INK, ls="--", lw=1.0, alpha=0.7)
    ax.set_ylabel(f"{proc.channel_names[0]} (rad)" if "theta" in proc.channel_names[0] else "observable")
    ax.set_title(f"causal effect at tick {tick} ({sep_label})")
    ax.legend(fontsize=7, loc="upper right", ncol=min(4, k))


def causal_effect(tr_or_proc, tick: int = 4, seed: int = 0):
    """Standalone 2-row figure: continuous observable (top) and tokens (bottom)."""
    import matplotlib.pyplot as plt

    if isinstance(tr_or_proc, dict) and "proc" in tr_or_proc:
        proc = tr_or_proc["proc"]
        letters = tr_or_proc["letters"]
    else:
        proc = tr_or_proc
        tr = trace(proc, seed=seed)
        letters = tr["letters"]

    tick = min(tick, max(0, proc.m - 1))
    k = proc.chain.n_states
    variants = np.tile(letters, (k, 1))
    variants[:, tick] = np.arange(k)
    out = proc.rollout(variants)

    fig, axes = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True)
    t = np.arange(proc.seq_len) * proc.dt
    t_branch = tick * proc.n_steps * proc.dt

    for l in range(k):
        is_drawn = (l == letters[tick])
        style = dict(lw=2.0, alpha=1.0) if is_drawn else dict(lw=1.1, alpha=0.7)
        cfg = ticker_for_letter(l)
        label = f"{cfg['name']}" + (" (drawn)" if is_drawn else "")
        axes[0].plot(t, out["observable"][l, :, 0], color=cfg["color"], label=label, **style)
        if len(proc.obs_bins) > 1:
            recon_l = proc.undiscretise(out["tokens"][l])
            axes[1].step(t, recon_l[:, 0], color=cfg["color"], where="post",
                         label=f"{label} ({proc.channel_names[0]})", **style)
            if recon_l.shape[-1] > 1:
                axes[1].step(t, recon_l[:, 1], color=cfg["color"], where="post", ls="--", alpha=0.45,
                             label=f"{label} ({proc.channel_names[1]})" if is_drawn else None)
        else:
            axes[1].step(t, out["tokens"][l], color=cfg["color"], where="post", label=label, **style)

    axes[0].axvline(t_branch, color=INK, ls="--", lw=1.0, alpha=0.7)
    axes[1].axvline(t_branch, color=INK, ls="--", lw=1.0, alpha=0.7)
    axes[0].set_ylabel(f"continuous {proc.channel_names[0]}")
    axes[1].set_ylabel(f"token recon ({', '.join(proc.channel_names)})" if len(proc.obs_bins) > 1 else "token")
    axes[1].set_xlabel("time (s)")

    branch = tick * proc.n_steps
    tok = out["tokens"][:, branch:]
    if len(proc.obs_bins) > 1:
        indices = proc.tokens_to_indices(tok)
        diff = indices[:, None, :, :] - indices[None, :, :, :]
        sep = np.linalg.norm(diff, axis=-1).max(-1).mean()
        sep_str = f"grid sep = {sep:.1f} bins"
    else:
        sep = np.abs(tok[:, None, :] - tok[None, :, :]).max(-1).mean()
        sep_str = f"mean sep = {sep:.1f} bins"

    sys_name = type(proc.system).__name__
    axes[0].set_title(f"{sys_name}: alternative letters at tick {tick} ({sep_str})")
    axes[0].legend(fontsize=7, ncol=min(4, k), loc="upper right")
    if len(proc.obs_bins) > 1:
        axes[1].legend(fontsize=6.5, ncol=min(4, k), loc="upper right")
    fig.tight_layout()
    return fig


_PANEL_FUNCS = {
    "observable": _panel_observable, "energy": _panel_energy, "metric": _panel_metric,
    "phase": _panel_phase, "states": _panel_states, "divergence": _panel_divergence,
    "spectrum": _panel_spectrum, "tokens": _panel_tokens, "belief": _panel_belief,
    "return_map": _panel_return_map, "causal_effect": _panel_causal_effect,
    "causal": _panel_causal_effect,
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
