"""ipywidgets controls for the simulator notebooks.

Kept out of the notebook so the UI is diffable and importable rather than a
cell of pasted code, and so two notebooks cannot drift into two different
versions of the same panel.

Every numeric control is a slider *with its own editable bounds*. A plain slider
cannot answer "what happens at gamma = 50" when it was built around a default of
1.2, and a plain text box gives no feel for the shape of the response. The pair
does both: drag to sweep, retype the bound to leave the neighbourhood entirely.
"""

from __future__ import annotations

import ipywidgets as W
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display

from .messk_configs import MESSK_CONFIGS, SYSTEMS, make_process
from .visualise import PANELS, auto_dt, plot, stability, trace, tunable_fields

LABEL_W = "150px"


def _bounds(value: float) -> tuple[float, float]:
    """A first guess at a useful slider range, given only the default."""
    span = max(abs(value), 1.0) * 2.0
    lo = 0.0 if value >= 0 and abs(value) > 1e-9 else -span
    return float(lo), float(value + span)


def param_row(name: str, value: float, integer: bool = False) -> tuple[W.Widget, W.Widget]:
    """One parameter: a slider, plus the two boxes that move the slider's ends."""
    lo, hi = _bounds(value)
    if integer:
        lo, hi = int(max(1, lo)), int(max(hi, value + 1))
        slider = W.IntSlider(value=int(value), min=lo, max=hi, description=name,
                             continuous_update=False, style={"description_width": LABEL_W},
                             layout=W.Layout(width="440px"))
        box = W.IntText
    else:
        slider = W.FloatSlider(value=value, min=lo, max=hi, step=(hi - lo) / 200 or 0.01,
                               description=name, continuous_update=False, readout_format=".4g",
                               style={"description_width": LABEL_W},
                               layout=W.Layout(width="440px"))
        box = W.FloatText
    lo_box = box(value=lo, layout=W.Layout(width="90px"))
    hi_box = box(value=hi, layout=W.Layout(width="90px"))

    def retune(_=None) -> None:
        low, high = lo_box.value, hi_box.value
        if high <= low:
            return
        # Set bounds in order that prevents slider.min > slider.max during update
        if low > slider.max:
            slider.max = high
            slider.min = low
        else:
            slider.min = low
            slider.max = high
        if not integer:
            slider.step = (high - low) / 200 or 0.01

    lo_box.observe(retune, "value")
    hi_box.observe(retune, "value")
    return slider, W.HBox([slider, W.Label("range"), lo_box, hi_box])


class ParamPanel:
    """The sliders for one system, rebuilt whenever the system changes."""

    def __init__(self) -> None:
        self.box = W.VBox([])
        self.sliders: dict[str, W.Widget] = {}

    def rebuild(self, system, on_change) -> None:
        self.sliders = {}
        rows = []
        for name, value in tunable_fields(system).items():
            slider, row = param_row(name, float(value))
            slider.observe(on_change, "value")
            self.sliders[name] = slider
            rows.append(row)
        self.box.children = tuple(rows)

    @property
    def values(self) -> dict[str, float]:
        return {name: s.value for name, s in self.sliders.items()}


def parse_bins(text: str) -> tuple[int, ...]:
    """"181" or "181x181" - the second form bins two channels into one token."""
    parts = [p for p in str(text).replace(",", "x").split("x") if p.strip()]
    bins = tuple(int(p) for p in parts)
    if not bins or any(b < 2 for b in bins):
        raise ValueError(f"need at least 2 bins per channel, got {text!r}")
    return bins


def classify_regime(lam: float) -> tuple[str, str, str]:
    """Classify dynamical stability regime based on Lyapunov exponent."""
    if lam < -0.02:
        return "CONTRACTIVE", "#137333", "#e6f4ea"
    elif lam > 0.05:
        return "CHAOTIC", "#c5221f", "#fce8e6"
    return "MARGINAL", "#b06000", "#fef7e0"


def _verdict(report: dict, multi_reports: list[tuple[int, dict]] | None = None, active_seed: int = 0) -> str:
    """Format a multi-seed aggregate stability audit table with transformer guidance."""
    if not multi_reports:
        multi_reports = [(active_seed, report)]

    lams = [r["lyapunov"] for _, r in multi_reports]
    passes = [r["stable"] for _, r in multi_reports]
    clips = [r["clipped"] for _, r in multi_reports]
    bins = [r["used_bins"] for _, r in multi_reports]
    gaps = [r["gap_free_mean"] for _, r in multi_reports]
    n_seeds = len(multi_reports)
    pass_count = sum(passes)

    mean_lam = float(np.mean(lams))
    std_lam = float(np.std(lams))
    min_lam = float(np.min(lams))
    max_lam = float(np.max(lams))
    mean_clip = float(np.mean(clips))
    max_clip = float(np.max(clips))
    mean_bins = float(np.mean(bins))
    mean_gap = float(np.mean(gaps))
    n_obs = report.get("n_obs", 181)

    regime_name, reg_col, reg_bg = classify_regime(mean_lam)
    if regime_name == "CONTRACTIVE" and any(classify_regime(l)[0] == "CHAOTIC" for l in lams):
        regime_name = "MOSTLY CONTRACTIVE (MILD CHAOS IN SOME SEEDS)"

    if pass_count == n_seeds:
        status_text = f"STABLE ({pass_count}/{n_seeds} seeds pass)"
        status_col, status_bg = "#137333", "#e6f4ea"
    elif pass_count >= n_seeds * 0.7:
        status_text = f"MARGINAL ({pass_count}/{n_seeds} seeds pass)"
        status_col, status_bg = "#b06000", "#fef7e0"
    else:
        unstable_seeds = [s for s, r in multi_reports if not r["stable"]]
        status_text = f"UNSTABLE ({pass_count}/{n_seeds} seeds pass; fail on seeds {unstable_seeds})"
        status_col, status_bg = "#c5221f", "#fce8e6"

    if mean_lam < -0.02 and max_lam <= 0.05 and mean_clip < 0.005:
        guidance = (
            "Optimal contractive dynamics. Perturbations contract exponentially, preserving causal chain memory. "
            "Clean token distributions ideal for autoregressive transformer training with low cross-entropy."
        )
        g_bg, g_border, g_col = "#f0fdf4", "#22c55e", "#15803d"
    elif mean_lam <= 0.05 and max_lam <= 0.10:
        guidance = (
            "Marginal / weakly damped dynamics. Trajectories maintain long-lived oscillations with slow dissipation. "
            "Attention heads must track ongoing phase drift without relying on strong contractive damping."
        )
        g_bg, g_border, g_col = "#fffbeb", "#f59e0b", "#b45309"
    else:
        guidance = (
            "Chaotic regime detected (&lambda; > 0). Exponential sensitivity quickly destroys causal letter predictability; "
            "autoregressive error will compound. Increase viscous damping (&gamma;) or reduce kick scale (&delta;v)."
        )
        g_bg, g_border, g_col = "#fef2f2", "#ef4444", "#b91c1c"

    channels = report.get("channels", ("observable",))
    obs_ranges = report.get("obs_ranges", (report.get("obs_range", (-1.0, 1.0)),))
    observed = report.get("per_channel_observed_range", ((report.get("observed_range", (0.0, 0.0))),))
    channels_info = []
    for i, ch in enumerate(channels):
        lo, hi = obs_ranges[i] if i < len(obs_ranges) else (-1.0, 1.0)
        olo, ohi = observed[i] if i < len(observed) else (0.0, 0.0)
        channels_info.append(f"<b>{ch}</b>: domain [{lo:.2f}, {hi:.2f}] (active visits [{olo:.2f}, {ohi:.2f}])")
    ch_str = " &nbsp;|&nbsp; ".join(channels_info)

    seed_rows = []
    for s, r in multi_reports:
        s_lam = r["lyapunov"]
        s_reg, s_col, s_bg = classify_regime(s_lam)
        reasons_str = "; ".join(r["reasons"])
        s_verdict = (
            "<span style='color:#16a34a;font-weight:700'>&#10004; PASS</span>"
            if r["stable"] else
            f"<span style='color:#dc2626;font-weight:700'>&#10008; FAIL</span> <span style='font-size:9.5px;color:#64748b'>({reasons_str})</span>"
        )
        is_act = (s == active_seed)
        row_style = "background:#eff6ff;font-weight:600;" if is_act else ""
        act_tag = " <span style='color:#2563eb;font-size:9.5px;'>(active)</span>" if is_act else ""
        s_clip = r["clipped"]
        s_bins = r["used_bins"]
        seed_rows.append(
            f"<tr style='border-bottom:1px solid #f1f5f9;{row_style}'>"
            f"<td style='padding:3px 6px;text-align:left;'>Seed {s}{act_tag}</td>"
            f"<td style='padding:3px 6px;'><span style='background:{s_bg};color:{s_col};padding:1px 5px;border-radius:3px;font-size:9.5px;font-weight:700;'>{s_reg}</span></td>"
            f"<td style='padding:3px 6px;'>{s_lam:+.3f}</td>"
            f"<td style='padding:3px 6px;'>{s_clip:.2%}</td>"
            f"<td style='padding:3px 6px;'>{s_bins}/{n_obs}</td>"
            f"<td style='padding:3px 6px;text-align:left;'>{s_verdict}</td>"
            f"</tr>"
        )
    seed_table_html = "".join(seed_rows)

    pct_cov = mean_bins / max(n_obs, 1)
    act_pct = report["used_bins"] / max(n_obs, 1)
    act_lam = report["lyapunov"]
    act_clip = report["clipped"]
    act_gap_m = report["gap_free_mean"]
    act_gap_x = report["gap_free_max"]

    return (
        f"<div style='font-family:-apple-system,BlinkMacSystemFont,\"Segoe UI\",Roboto,sans-serif;font-size:11.5px;line-height:1.45;color:#1e293b;background:#ffffff;border:1px solid #cbd5e1;border-radius:8px;padding:12px 14px;box-shadow:0 1px 3px rgba(0,0,0,0.05);max-width:620px;'>"
        f"<div style='display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid #e2e8f0;padding-bottom:8px;margin-bottom:8px;'>"
        f"<div><span style='font-size:13px;font-weight:700;color:#0f172a;'>Multi-Seed Stability Audit</span>"
        f"<span style='margin-left:8px;padding:2px 8px;border-radius:12px;font-size:10.5px;font-weight:700;background:{status_bg};color:{status_col};'>{status_text}</span></div>"
        f"<div style='font-size:11px;'><b>Dynamics:</b> <span style='padding:2px 7px;border-radius:4px;font-weight:700;background:{reg_bg};color:{reg_col};'>{regime_name}</span></div>"
        f"</div>"
        f"<div style='background:{g_bg};border-left:3px solid {g_border};padding:6px 10px;border-radius:0 4px 4px 0;margin-bottom:9px;font-size:11px;color:{g_col};'>"
        f"<b>Transformer Guidance:</b> {guidance}"
        f"</div>"
        f"<table style='width:100%;border-collapse:collapse;margin-bottom:9px;font-size:11px;'>"
        f"<thead><tr style='background:#f8fafc;border-bottom:1px solid #e2e8f0;text-align:left;color:#64748b;'>"
        f"<th style='padding:4px 6px;'>Metric</th>"
        f"<th style='padding:4px 6px;'>Aggregate ({n_seeds} seeds)</th>"
        f"<th style='padding:4px 6px;'>Active (Seed {active_seed})</th>"
        f"<th style='padding:4px 6px;'>Transformer Target</th>"
        f"</tr></thead>"
        f"<tbody>"
        f"<tr style='border-bottom:1px solid #f1f5f9;'>"
        f"<td style='padding:4px 6px;font-weight:600;'>Lyapunov (&lambda;)</td>"
        f"<td style='padding:4px 6px;'><b>{mean_lam:+.3f} &plusmn; {std_lam:.3f}</b> /s <span style='color:#64748b;font-size:10px;'>[{min_lam:+.2f}, {max_lam:+.2f}]</span></td>"
        f"<td style='padding:4px 6px;font-weight:600;'>{act_lam:+.3f} /s</td>"
        f"<td style='padding:4px 6px;color:#16a34a;'>&le; 0.00 /s (contractive)</td>"
        f"</tr>"
        f"<tr style='border-bottom:1px solid #f1f5f9;'>"
        f"<td style='padding:4px 6px;font-weight:600;'>Boundary Clipping</td>"
        f"<td style='padding:4px 6px;'>{mean_clip:.2%} <span style='color:#64748b;font-size:10px;'>(max {max_clip:.2%})</span></td>"
        f"<td style='padding:4px 6px;'>{act_clip:.2%}</td>"
        f"<td style='padding:4px 6px;color:#16a34a;'>&lt; 0.5% (no edge artifacts)</td>"
        f"</tr>"
        f"<tr style='border-bottom:1px solid #f1f5f9;'>"
        f"<td style='padding:4px 6px;font-weight:600;'>Vocab Coverage</td>"
        f"<td style='padding:4px 6px;'>{mean_bins:.1f} / {n_obs} ({pct_cov:.1%})</td>"
        f"<td style='padding:4px 6px;'>{report['used_bins']} / {n_obs} ({act_pct:.1%})</td>"
        f"<td style='padding:4px 6px;color:#2563eb;'>Balanced token use</td>"
        f"</tr>"
        f"<tr style='border-bottom:1px solid #f1f5f9;'>"
        f"<td style='padding:4px 6px;font-weight:600;'>Action Signal Gap</td>"
        f"<td style='padding:4px 6px;'>mean {mean_gap:.3g}</td>"
        f"<td style='padding:4px 6px;'>mean {act_gap_m:.3g} (max {act_gap_x:.3g})</td>"
        f"<td style='padding:4px 6px;color:#16a34a;'>&gt; 1e-3 (distinguishable)</td>"
        f"</tr>"
        f"</tbody></table>"
        f"<div style='font-size:10.5px;color:#475569;margin-bottom:4px;padding:2px 0;'>{ch_str}</div>"
        f"<div style='font-weight:700;color:#334155;margin:8px 0 4px;font-size:10.5px;text-transform:uppercase;letter-spacing:0.5px;'>Per-Seed Breakdown ({n_seeds} Realizations):</div>"
        f"<table style='width:100%;border-collapse:collapse;font-size:10.5px;text-align:center;border:1px solid #e2e8f0;'>"
        f"<thead><tr style='background:#f8fafc;color:#475569;border-bottom:1px solid #cbd5e1;'>"
        f"<th style='padding:3px 6px;text-align:left;'>Seed</th>"
        f"<th style='padding:3px 6px;'>Regime</th>"
        f"<th style='padding:3px 6px;'>&lambda; (/s)</th>"
        f"<th style='padding:3px 6px;'>Clipped</th>"
        f"<th style='padding:3px 6px;'>Bins Used</th>"
        f"<th style='padding:3px 6px;text-align:left;'>Verdict</th>"
        f"</tr></thead>"
        f"<tbody>{seed_table_html}</tbody>"
        f"</table>"
        f"</div>"
    )


def explorer(default: str = "pendulum_mess4", panels=("observable", "energy", "metric", "phase")):
    """The whole single-system dashboard: every parameter, every graph.

    Returns the widget rather than displaying it, so a notebook cell can place
    it and a test can build it without a kernel front end.
    """
    import io

    system_dd = W.Dropdown(options=sorted(MESSK_CONFIGS), value=default, description="system",
                           style={"description_width": LABEL_W})
    panel_sel = W.SelectMultiple(options=PANELS, value=tuple(panels), description="graphs",
                                 rows=len(PANELS), style={"description_width": LABEL_W},
                                 layout=W.Layout(width="440px"))
    seed = W.IntText(value=0, description="seed", style={"description_width": LABEL_W})
    spec_default = make_process(default)
    bins_default_str = "x".join(str(b) for b in spec_default.obs_bins) if len(spec_default.obs_bins) > 1 else str(spec_default.obs_bins[0])
    bins = W.Text(value=bins_default_str, description="bins per channel",
                  style={"description_width": LABEL_W},
                  layout=W.Layout(width="440px"))
    live = W.Checkbox(value=True, description="redraw on change")
    draw_btn = W.Button(description="redraw", button_style="primary")

    m_s, m_row = param_row("m (chain ticks)", 40, integer=True)
    n_s, n_row = param_row("n_steps (per tick)", 10, integer=True)
    dt_s, dt_row = param_row("dt", 0.02)
    dv_s, dv_row = param_row("delta_v (kick scale)", 0.55)
    alpha_s, alpha_row = param_row("alpha (chain)", 0.7)
    stay_s, stay_row = param_row("stay (chain)", 0.7)
    for s in (alpha_s, stay_s):
        s.min, s.max, s.step = 0.0, 1.0, 0.005

    params = ParamPanel()
    img_out = W.Image(format="png", layout=W.Layout(max_width="100%"))
    out = W.Output()
    report_html = W.HTML()

    def build():
        proc = make_process(
            system_dd.value,
            system=params.values,
            chain={"alpha": alpha_s.value, "stay": stay_s.value},
            m=int(m_s.value), n_steps=int(n_s.value), dt=float(dt_s.value),
            delta_v=float(dv_s.value), obs_bins=parse_bins(bins.value),
        )
        return proc, trace(proc, seed=int(seed.value))

    def redraw(_=None) -> None:
        if not live.value and _ is not None:
            return
        with out:
            out.clear_output(wait=True)
            try:
                proc, tr = build()
            except Exception as exc:  # a bad parameter is a normal event here
                report_html.value = f"<pre style='color:#c0392b'>{type(exc).__name__}: {exc}</pre>"
                return

            active_seed = int(seed.value)
            active_rep = stability(tr)
            # Evaluate 6 realizations across seeds for multi-seed stability audit
            other_seeds = [s for s in range(6) if s != active_seed][:5]
            multi_reports = [(active_seed, active_rep)]
            for s in other_seeds:
                multi_reports.append((s, stability(trace(proc, seed=s))))

            report_html.value = _verdict(active_rep, multi_reports=multi_reports, active_seed=active_seed)
            fig = plot(tr, panels=panel_sel.value, title=system_dd.value)
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
            plt.close(fig)
            img_out.value = buf.getvalue()

    def reset_system(_=None, build_only: bool = False) -> None:
        spec = make_process(system_dd.value)
        params.rebuild(spec.system, redraw)
        dt_s.value, dv_s.value = spec.dt, spec.delta_v
        bins.value = "x".join(str(b) for b in spec.obs_bins) if len(spec.obs_bins) > 1 else str(spec.obs_bins[0])
        if system_dd.value == "sphere_mess4" and "tokens" not in panel_sel.value:
            cur = list(panel_sel.value)
            if "metric" in cur:
                cur[cur.index("metric")] = "tokens"
            else:
                cur.append("tokens")
            panel_sel.value = tuple(cur)
        if not build_only:
            redraw()

    for w in (panel_sel, seed, bins, m_s, n_s, dt_s, dv_s, alpha_s, stay_s):
        w.observe(redraw, "value")
    system_dd.observe(reset_system, "value")
    draw_btn.on_click(lambda _: redraw())

    reset_system(build_only=True)
    redraw()

    controls = W.VBox([
        W.HBox([system_dd, live, draw_btn]),
        W.HTML("<b>driver</b> - how the chain meets the physics"),
        m_row, n_row, dt_row, dv_row, bins,
        W.HTML("<b>chain</b>"), alpha_row, stay_row,
        W.HTML("<b>system</b> - every dataclass field, read off the system itself"),
        params.box,
        W.HTML("<b>view</b>"), panel_sel, seed,
    ])
    return W.VBox([W.HBox([controls, report_html]), img_out, out])


def sweep_ui(default: str = "double_pendulum_mess4"):
    """Scan one parameter and plot where the trajectory stays stable."""
    import ast

    system_dd = W.Dropdown(options=sorted(MESSK_CONFIGS), value=default, description="system",
                           style={"description_width": LABEL_W})
    field_dd = W.Dropdown(options=[], description="parameter",
                          style={"description_width": LABEL_W})
    base_text = W.Text(value="", description="fixed overrides",
                       placeholder="e.g. a=1.5, kappa=3.5, delta_v=0.4",
                       style={"description_width": LABEL_W}, layout=W.Layout(width="500px"))
    lo = W.FloatText(value=0.0, description="from", style={"description_width": "60px"},
                     layout=W.Layout(width="180px"))
    hi = W.FloatText(value=4.0, description="to", style={"description_width": "60px"},
                     layout=W.Layout(width="180px"))
    steps = W.IntSlider(value=17, min=3, max=61, description="points",
                        style={"description_width": LABEL_W})
    run = W.Button(description="run sweep", button_style="primary")
    out = W.Output()

    def reset(_=None) -> None:
        proc = make_process(system_dd.value)
        options = ["delta_v", "dt", "n_steps", "stay", "alpha"] + sorted(tunable_fields(proc.system))
        field_dd.options = options
        field_dd.value = "delta_v"

    def parse_overrides(text: str, proc) -> tuple[dict, dict, dict]:
        sys_fields = set(tunable_fields(proc.system).keys()) | {"obs_range"}
        chain_fields = {"stay", "alpha", "n_states"}
        driver_fields = {"m", "n_steps", "dt", "delta_v", "obs_bins"}
        s_kw, c_kw, d_kw = {}, {}, {}
        for item in text.split(","):
            item = item.strip()
            if not item or "=" not in item:
                continue
            k, v = item.split("=", 1)
            k, v = k.strip().replace("-", "_"), v.strip()
            try:
                val = ast.literal_eval(v)
            except Exception:
                val = v
            if k in sys_fields:
                s_kw[k] = val
            elif k in chain_fields:
                c_kw[k] = val
            elif k in driver_fields:
                d_kw[k] = int(val) if k in ("m", "n_steps") else val
            else:
                s_kw[k] = val
        return s_kw, c_kw, d_kw

    def go(_=None) -> None:
        with out:
            out.clear_output(wait=True)
            values = np.linspace(lo.value, hi.value, int(steps.value))
            name, rows = field_dd.value, []
            base_proc = make_process(system_dd.value)
            base_sys, base_chain, base_driver = parse_overrides(base_text.value, base_proc)

            for v in values:
                sys_kw = dict(base_sys)
                chain_kw = dict(base_chain)
                driver_kw = dict(base_driver)

                if name in tunable_fields(base_proc.system):
                    sys_kw[name] = float(v)
                elif name in ("stay", "alpha", "n_states"):
                    chain_kw[name] = float(v)
                elif name == "n_steps":
                    driver_kw["n_steps"] = max(1, int(round(v)))
                elif name == "m":
                    driver_kw["m"] = max(1, int(round(v)))
                else:
                    driver_kw[name] = float(v)

                call_kw = dict(driver_kw)
                if sys_kw:
                    call_kw["system"] = sys_kw
                if chain_kw:
                    call_kw["chain"] = chain_kw

                m_val = call_kw.pop("m", 24)
                try:
                    tr = trace(make_process(system_dd.value, m=m_val, **call_kw), seed=0)
                    rows.append({"value": v, **stability(tr)})
                except Exception as exc:
                    rows.append({"value": v, "lyapunov": np.nan, "clipped": np.nan,
                                 "used_bins": 0, "n_obs": 1, "gap_free_mean": np.nan,
                                 "stable": False, "reasons": [str(exc)]})

            fig, axes = plt.subplots(1, 3, figsize=(15, 3.6))
            v = [r["value"] for r in rows]
            axes[0].plot(v, [r["lyapunov"] for r in rows], "o-", color="#c0392b")
            axes[0].axhline(0, color="#0b0b0b", lw=0.8)
            axes[0].set_title("lyapunov (>0 = chaotic)")
            axes[1].plot(v, [100 * r["clipped"] for r in rows], "o-", color="#eb6834")
            axes[1].axhline(1.0, color="#0b0b0b", lw=0.8, ls="--")
            axes[1].set_title("clipped % (dashed = 1% budget)")
            axes[2].plot(v, [r["used_bins"] / max(r["n_obs"], 1) * 100 for r in rows],
                         "o-", color="#2a78d6")
            axes[2].set_title("vocabulary used %")
            for ax in axes:
                ax.set_xlabel(name)
                ax.grid(True, ls="--", alpha=0.5)
            good = [f"{r['value']:.4g}" for r in rows if r["stable"]]
            override_str = f" (fixed: {base_text.value.strip()})" if base_text.value.strip() else ""
            fig.suptitle(f"{system_dd.value}{override_str}: stable at {name} in "
                         f"{{{', '.join(good) if good else 'nothing in this range'}}}",
                         fontsize=11)
            fig.tight_layout()
            plt.show()
            plt.close(fig)

    system_dd.observe(reset, "value")
    run.on_click(go)
    reset()
    return W.VBox([W.HBox([system_dd, field_dd]), base_text, W.HBox([lo, hi, steps, run]), out])


def compute_optimal_gamma(
    system_name: str = "pendulum_mess4",
    m: int = 40,
    n_steps: int = 10,
    dt: float = 0.02,
    delta_v: float = 0.55,
    stay: float = 0.7,
    alpha: float = 0.7,
    system_params: dict | None = None,
    n_trajs: int = 32,
    seed: int = 42,
    max_iter: int = 6,
    damping_field: str = "gamma",
) -> dict:
    """Find the optimal viscous damping parameter balancing kick injection with dissipation."""
    sys_kw = dict(system_params or {})
    if damping_field not in sys_kw:
        proc_check = make_process(system_name, system=sys_kw)
        tf = tunable_fields(proc_check.system)
        if damping_field not in tf:
            for cand in ("gamma", "gamma1", "kappa"):
                if cand in tf:
                    damping_field = cand
                    break

    gamma = float(sys_kw.get(damping_field, 1.0))
    warmup = max(5, m // 4)
    active_ticks = m - warmup
    t_inj, t_diss = 0.0, 0.0
    cum_inj, cum_diss = [], []

    for iteration in range(max_iter):
        rng = np.random.default_rng(seed + iteration)
        sys_kw[damping_field] = gamma
        if damping_field == "gamma1":
            sys_kw["gamma2"] = gamma
        proc = make_process(
            system_name,
            system=sys_kw,
            chain={"stay": stay, "alpha": alpha},
            m=m, n_steps=n_steps, dt=dt, delta_v=delta_v,
        )
        sysm = proc.system
        letters = proc.chain.sample(rng, n_trajs, m)[1]
        actions = proc.actions
        z = sysm.initial_state(n_trajs)

        t_inj, t_diss = 0.0, 0.0
        cum_inj, cum_diss = [], []
        total_omega_sq = 0.0

        for t in range(m):
            action = actions[letters[:, t]]
            e_before = sysm.energy(z)
            z_kicked = sysm.kick(z, action)
            e_after = sysm.energy(z_kicked)
            dE = float((e_after - e_before).mean())

            z_flow = z_kicked
            w_sq = 0.0
            for _ in range(n_steps):
                z_next = proc.flow(z_flow)
                if damping_field == "kappa":
                    x = np.exp(z_flow[..., 0])
                    x_star = getattr(sysm, "c", 0.8) / getattr(sysm, "d", 0.4)
                    a = getattr(sysm, "a", 1.0)
                    w_sq += float((a * (x - x_star) ** 2).mean()) * dt
                else:
                    v_mid = sysm.metric(0.5 * (z_flow + z_next))
                    w_sq += float((v_mid ** 2).sum(axis=-1).mean()) * dt
                z_flow = z_next
            z = z_flow

            dE_diss = float((e_after - sysm.energy(z_flow)).mean())

            if t >= warmup:
                t_inj += dE
                total_omega_sq += w_sq
                t_diss += dE_diss
                cum_inj.append(t_inj)
                cum_diss.append(t_diss)

        if damping_field == "kappa":
            gamma_est = float(total_omega_sq / max(t_inj, 1e-9))
        else:
            gamma_est = float(t_inj / max(total_omega_sq, 1e-9))
        if abs(gamma - gamma_est) < 0.005 or iteration == max_iter - 1:
            gamma = gamma_est
            break
        gamma = 0.5 * (gamma + gamma_est)

    mean_inj = t_inj / active_ticks
    mean_diss = t_diss / active_ticks
    return {
        "gamma_opt": gamma,
        "mean_injected": mean_inj,
        "mean_dissipated": mean_diss,
        "balance_ratio": mean_diss / max(mean_inj, 1e-9),
        "cum_injected": np.array(cum_inj),
        "cum_dissipated": np.array(cum_diss),
    }


def optimal_gamma_ui(default_system: str = "pendulum_mess4") -> W.Widget:
    """Interactive widget to compute and verify the optimal viscous damping gamma."""
    spec = make_process(default_system)
    system_dd = W.Dropdown(options=sorted(MESSK_CONFIGS), value=default_system, description="system",
                           style={"description_width": LABEL_W})
    m_s, m_row = param_row("m (ticks)", 40, integer=True)
    n_s, n_row = param_row("n_steps (per tick)", spec.n_steps, integer=True)
    dt_s, dt_row = param_row("dt", spec.dt)
    dv_s, dv_row = param_row("delta_v (kick scale)", spec.delta_v)
    stay_s, stay_row = param_row("stay (chain)", 0.7)
    alpha_s, alpha_row = param_row("alpha (chain)", 0.7)
    for s in (stay_s, alpha_s):
        s.min, s.max, s.step = 0.0, 1.0, 0.01

    calc_btn = W.Button(description="Compute Optimal Gamma", button_style="primary",
                        layout=W.Layout(width="240px"))
    report_html = W.HTML()
    img_out = W.Image(format="png", layout=W.Layout(max_width="100%"))
    out = W.Output()

    def on_system_change(_=None):
        new_spec = make_process(system_dd.value)
        dt_s.value = new_spec.dt
        dv_s.value = new_spec.delta_v
        n_s.value = new_spec.n_steps

    system_dd.observe(on_system_change, "value")

    def calculate(_=None):
        with out:
            out.clear_output(wait=True)
            report_html.value = "<i>Simulating ODE rollouts & finding optimal gamma...</i>"
            res = compute_optimal_gamma(
                system_name=system_dd.value,
                m=int(m_s.value),
                n_steps=int(n_s.value),
                dt=float(dt_s.value),
                delta_v=float(dv_s.value),
                stay=float(stay_s.value),
                alpha=float(alpha_s.value),
            )
            g_opt = res["gamma_opt"]
            inj = res["mean_injected"]
            diss = res["mean_dissipated"]
            ratio = res["balance_ratio"]

            report_html.value = (
                f"<div style='font-family: monospace; font-size: 13px; line-height: 1.6; padding: 10px; "
                f"background: #f8f9fa; border-radius: 4px; border: 1px solid #dee2e6; margin-bottom: 8px;'>"
                f"<b>Optimal Viscous Damping (&gamma;*):</b> <span style='color: #2b78d6; font-size: 16px; font-weight: bold;'>{g_opt:.4f}</span><br>"
                f"&bull; Energy added per tick: <b>{inj:.4f}</b> J<br>"
                f"&bull; Energy bled per tick: <b>{diss:.4f}</b> J<br>"
                f"&bull; Energy balance ratio (bled / added): <b>{ratio:.1%}</b> (target = 100%)"
                f"</div>"
            )

            fig, ax = plt.subplots(figsize=(8, 3.2))
            ticks = np.arange(len(res["cum_injected"]))
            ax.plot(ticks, res["cum_injected"], label="Cumulative energy injected (kicks)", color="#c0392b", lw=1.8)
            ax.plot(ticks, res["cum_dissipated"], label="Cumulative energy bled (viscosity)", color="#1baf7a", lw=1.8, ls="--")
            ax.set_xlabel("tick (post-warmup)")
            ax.set_ylabel("cumulative energy")
            ax.set_title(f"Steady-State Energy Equilibrium (gamma* = {g_opt:.4f})")
            ax.grid(True, ls="--", alpha=0.5)
            ax.legend(fontsize=8, loc="upper left")
            plt.tight_layout()
            import io
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
            plt.close(fig)
            img_out.value = buf.getvalue()

    calc_btn.on_click(calculate)
    calculate()

    return W.VBox([
        W.HTML("<h4>Steady-State Energy Balance: Optimal Viscous Damping (&gamma;*)</h4>"
               "<p style='color: #666; font-size: 12px; margin-top: -6px;'>"
               "Finds the exact damping &gamma; where average energy bled by friction equals average energy injected by kicks."
               "</p>"),
        system_dd,
        m_row, n_row, dt_row, dv_row, stay_row, alpha_row,
        W.HBox([calc_btn]),
        report_html,
        img_out,
        out,
    ])


def _detect_damping_field(system_name: str, damping_field: str | None = None) -> str:
    if damping_field is not None:
        return damping_field
    proc = make_process(system_name)
    tf = tunable_fields(proc.system)
    for cand in ("gamma", "gamma1", "kappa"):
        if cand in tf:
            return cand
    return "gamma"


def _fixed_dt(system_name: str) -> float:
    """Committed physics sampling gap; dt is not a sweep axis."""
    return float(SYSTEMS[MESSK_CONFIGS[system_name]]["dt"])


def _damping_strength(field: str, value: float) -> float:
    """Positive damping coefficient used by n*damping comparisons."""
    return 1.0 / value if field == "kappa" else value


def _damping_value(field: str, strength: float) -> float:
    return 1.0 / strength if field == "kappa" else strength


def _sweep_system_kwargs(
    system_name: str,
    field: str,
    damping_value: float,
    system_overrides: dict | None = None,
    unbounded_rates: bool = True,
) -> dict:
    """System kwargs for screening, with velocity clamps disabled by default."""
    system_key = MESSK_CONFIGS[system_name]
    kwargs = dict(system_overrides or {})
    if unbounded_rates:
        if system_key in {"pendulum", "double_pendulum"}:
            kwargs.setdefault("omega_max", np.inf)
        elif system_key == "sphere":
            kwargs.setdefault("rate_max", np.inf)
    kwargs[field] = float(damping_value)
    if field == "gamma1":
        kwargs["gamma2"] = float(damping_value)
    return kwargs


def _screen_process(proc, n_seeds: int) -> dict:
    reports = [stability(trace(proc, seed=seed)) for seed in range(n_seeds)]
    values = lambda key: [report[key] for report in reports]
    bound_failures = sum(report["state_bound_fraction"] > 0.001 for report in reports)
    pass_rate = sum(
        report["stable"] and report["state_bound_fraction"] <= 0.001
        for report in reports
    ) / n_seeds
    reasons = {reason for report in reports for reason in report["reasons"]}
    if bound_failures:
        reasons.add(f"state safety bound exceeded on {bound_failures}/{n_seeds} seeds")
    return {
        "pass_rate": pass_rate,
        "lyapunov_mean": float(np.mean(values("lyapunov"))),
        "lyapunov_std": float(np.std(values("lyapunov"))),
        "clipped_mean": float(np.mean(values("clipped"))),
        "state_bound_mean": float(np.mean(values("state_bound_fraction"))),
        "used_bins_mean": float(np.mean(values("used_bins"))),
        "gap_free_mean": float(np.mean(values("gap_free_mean"))),
        "bayes_gap_mean": float(np.mean(values("bayes_gap"))),
        "verdict": "GO" if pass_rate >= 0.8 and float(np.mean(values("bayes_gap"))) >= 0.15 else "NO-GO",
        "reasons": sorted(reasons),
    }


def _candidate_rank(cell: dict) -> tuple:
    return (
        cell["pass_rate"],
        -cell["state_bound_mean"],
        -cell["clipped_mean"],
        cell["bayes_gap_mean"],
        cell["used_bins_mean"],
    )


def grid_screen(
    system_name: str,
    delta_v_values: list[float],
    damping_values: list[float],
    damping_field: str | None = None,
    dt: float | None = None,
    n_steps: int = 10,
    m: int = 24,
    n_seeds: int = 10,
    obs_bins=181,
    system_overrides: dict | None = None,
    unbounded_rates: bool = True,
    verbose: bool = False,
) -> dict:
    """Screen a 2D grid of (delta_v, damping) configs across multiple random seeds.

    For each cell, runs trace() + stability() across n_seeds. Reports per-cell:
      - pass_rate: fraction of seeds passing all checks
      - mean and std of lyapunov exponent
      - mean clipping percentage
      - mean used bins
      - mean bayes gap
      - GO / NO-GO verdict
    """
    field = _detect_damping_field(system_name, damping_field)
    if dt is None:
        dt = _fixed_dt(system_name)

    grid_results = []
    for dv in delta_v_values:
        for d_val in damping_values:
            sys_kw = _sweep_system_kwargs(
                system_name, field, d_val, system_overrides, unbounded_rates
            )
            proc = make_process(
                system_name,
                system=sys_kw,
                delta_v=float(dv),
                dt=float(dt),
                m=int(m),
                n_steps=int(n_steps),
                obs_bins=obs_bins,
            )
            cell = {
                "delta_v": float(dv),
                field: float(d_val),
                "damping_field": field,
                "damping_value": float(d_val),
                "damping_strength": _damping_strength(field, float(d_val)),
                **_screen_process(proc, n_seeds),
            }
            grid_results.append(cell)
            if verbose:
                print(f"dv={dv:.4f} {field}={d_val:.4f} -> {cell['verdict']} "
                      f"(pass={cell['pass_rate']:.0%}, bg={cell['bayes_gap_mean']:.3f}, "
                      f"lyap={cell['lyapunov_mean']:+.2f})")

    passing = [c for c in grid_results if c["verdict"] == "GO"]
    res = GridScreenResult({
        "system": system_name,
        "dt": float(dt),
        "damping_field": field,
        "n_steps": int(n_steps),
        "m": int(m),
        "unbounded_rates": unbounded_rates,
        "grid": grid_results,
        "passing": passing,
    })
    return res


def render_grid_html(result: dict) -> str:
    """Format grid screening output as an HTML heatmap matrix."""
    system = result.get("system", "")
    field = result.get("damping_field", "damping")
    dt = result.get("dt", 0.0)
    grid = result.get("grid", [])
    if not grid:
        return "<p>Empty grid results.</p>"

    dvs = sorted(list({c["delta_v"] for c in grid}))
    damps = sorted(list({c["damping_value"] for c in grid}), reverse=True)
    lookup = {(c["delta_v"], c["damping_value"]): c for c in grid}

    html = [
        "<div style='font-family: -apple-system, BlinkMacSystemFont, \"Segoe UI\", Roboto, monospace; margin: 12px 0;'>",
        f"<h4 style='margin: 0 0 8px 0; font-size: 14px;'>Grid Screening: {system} (dt={dt:.4f}, damping={field})</h4>",
        "<table style='border-collapse: collapse; text-align: center; font-size: 12px; border: 1px solid #334155;'>",
        "<tr>",
        f"<th style='border: 1px solid #334155; padding: 7px 10px; background: #0f172a; color: #94a3b8; font-weight: 600;'>{field} \\ Δv</th>",
    ]
    for dv in dvs:
        html.append(f"<th style='border: 1px solid #334155; padding: 7px 10px; background: #1e293b; color: #f8fafc; font-weight: 600;'>{dv:.4g}</th>")
    html.append("</tr>")

    for damp in damps:
        html.append("<tr>")
        html.append(f"<td style='border: 1px solid #334155; padding: 7px 10px; font-weight: 600; background: #1e293b; color: #f8fafc;'>{damp:.4g}</td>")
        for dv in dvs:
            cell = lookup.get((dv, damp))
            if cell is None:
                html.append("<td style='border: 1px solid #334155; padding: 6px; background: #1e293b; color: #64748b;'>-</td>")
                continue
            pr = cell["pass_rate"]
            bg = cell["bayes_gap_mean"]
            ly = cell["lyapunov_mean"]
            bound = cell["state_bound_mean"]
            if pr >= 0.8 and bg >= 0.15:
                bg_col = "#d4edda"
                text_col = "#0f5132"
                border_col = "#badbcc"
            elif pr >= 0.5:
                bg_col = "#fff3cd"
                text_col = "#664d03"
                border_col = "#ffecb5"
            else:
                bg_col = "#f8d7da"
                text_col = "#842029"
                border_col = "#f5c2c7"

            cell_content = f"<b style='font-size: 13px;'>{pr:.0%}</b><br><span style='font-size: 10px; opacity: 0.9;'>bg={bg:.2f}<br>λ={ly:+.2f}<br>bound={bound:.1%}</span>"
            html.append(f"<td style='border: 1px solid {border_col}; padding: 6px 8px; background: {bg_col}; color: {text_col}; min-width: 80px;'>{cell_content}</td>")
        html.append("</tr>")

    html.append("</table>")
    html.append("<p style='font-size: 11px; opacity: 0.8; margin-top: 6px;'>Passing criteria: pass_rate ≥ 80% and bayes_gap ≥ 0.15 nats</p>")
    html.append("</div>")
    return "\n".join(html)


class GridScreenResult(dict):
    """Dictionary holding grid_screen results with rich HTML display in notebooks."""

    def _repr_html_(self) -> str:
        return render_grid_html(self)


def display_grid(result: dict) -> None:
    """Render and display grid screen result in a notebook."""
    from IPython.display import HTML, display
    display(HTML(render_grid_html(result)))



def n_recalibrate(
    system_name: str,
    delta_v: float,
    damping_value: float,
    damping_field: str | None = None,
    dt: float | None = None,
    n_values: list[int] = (5, 10, 15, 20),
    m: int = 24,
    n_seeds: int = 10,
    obs_bins=181,
    system_overrides: dict | None = None,
    unbounded_rates: bool = True,
) -> list[dict]:
    """Test multiple impulse spacings for a locked (delta_v, damping, dt) config.

    Returns list of dicts, one per n value, with stability metrics
    aggregated over n_seeds seeds.
    """
    field = _detect_damping_field(system_name, damping_field)
    if dt is None:
        dt = _fixed_dt(system_name)

    results = []
    for n in n_values:
        sys_kw = _sweep_system_kwargs(
            system_name, field, damping_value, system_overrides, unbounded_rates
        )
        proc = make_process(
            system_name,
            system=sys_kw,
            delta_v=float(delta_v),
            dt=float(dt),
            m=int(m),
            n_steps=int(n),
            obs_bins=obs_bins,
        )
        results.append({
            "n": int(n),
            "delta_v": float(delta_v),
            "damping_field": field,
            "damping_value": float(damping_value),
            "damping_strength": _damping_strength(field, float(damping_value)),
            "dt": float(dt),
            **_screen_process(proc, n_seeds),
        })
    return results


def cascade_screen(
    system_name: str,
    delta_v_values: list[float],
    damping_values: list[float],
    *,
    dt: float | None = None,
    base_n: int = 10,
    n_down: int = 5,
    n_up: int = 20,
    m: int = 24,
    n_seeds: int = 10,
    obs_bins=181,
    keep_stage1: int = 6,
    keep_final: int = 15,
    system_overrides: dict | None = None,
    unbounded_rates: bool = True,
) -> dict:
    """Two-stage screen: delta_v x damping, then the five requested n/damping conditions."""
    if not (0 < n_down < base_n < n_up):
        raise ValueError("require 0 < n_down < base_n < n_up")
    if keep_stage1 < 1 or keep_final < 1:
        raise ValueError("candidate counts must be positive")
    dt = _fixed_dt(system_name) if dt is None else float(dt)
    stage1 = grid_screen(
        system_name,
        delta_v_values,
        damping_values,
        dt=dt,
        n_steps=base_n,
        m=m,
        n_seeds=n_seeds,
        obs_bins=obs_bins,
        system_overrides=system_overrides,
        unbounded_rates=unbounded_rates,
    )
    pool = stage1["passing"] or stage1["grid"]
    bases = sorted(pool, key=_candidate_rank, reverse=True)[:keep_stage1]
    field = stage1["damping_field"]
    expanded = []
    for base_rank, base in enumerate(bases, 1):
        strength = base["damping_strength"]
        conditions = (
            ("base", base_n, strength),
            ("n_up_gamma_down", n_up, strength * base_n / n_up),
            ("n_down_gamma_up", n_down, strength * base_n / n_down),
            ("n_up_gamma_constant", n_up, strength),
            ("n_down_gamma_constant", n_down, strength),
        )
        for condition, n, condition_strength in conditions:
            raw_damping = _damping_value(field, condition_strength)
            sys_kw = _sweep_system_kwargs(
                system_name, field, raw_damping, system_overrides, unbounded_rates
            )
            proc = make_process(
                system_name,
                system=sys_kw,
                delta_v=base["delta_v"],
                dt=dt,
                m=m,
                n_steps=n,
                obs_bins=obs_bins,
            )
            expanded.append({
                "system": system_name,
                "base_rank": base_rank,
                "condition": condition,
                "dt": dt,
                "n": n,
                "delta_v": base["delta_v"],
                "damping_field": field,
                "damping_value": raw_damping,
                "damping_strength": condition_strength,
                "n_times_damping": n * condition_strength,
                **_screen_process(proc, n_seeds),
            })

    stable = [candidate for candidate in expanded if candidate["verdict"] == "GO"]
    selected = []
    quota = max(1, keep_final // 5)
    for condition in ("base", "n_up_gamma_down", "n_down_gamma_up", "n_up_gamma_constant", "n_down_gamma_constant"):
        candidates = sorted(
            (candidate for candidate in stable if candidate["condition"] == condition),
            key=_candidate_rank,
            reverse=True,
        )
        selected.extend(candidates[:quota])
    if len(selected) < keep_final:
        selected_ids = {id(candidate) for candidate in selected}
        remainder = sorted(
            (candidate for candidate in stable if id(candidate) not in selected_ids),
            key=_candidate_rank,
            reverse=True,
        )
        selected.extend(remainder[:keep_final - len(selected)])
    return {
        "system": system_name,
        "dt": dt,
        "base_n": base_n,
        "damping_field": field,
        "stage1": stage1,
        "bases": bases,
        "expanded": expanded,
        "selected": selected[:keep_final],
    }


def grid_screen_ui(system_name: str = "pendulum_mess4") -> W.Widget:
    """Interactive widget to configure and run grid_screen."""
    system_dd = W.Dropdown(
        options=sorted(MESSK_CONFIGS),
        value=system_name if system_name in MESSK_CONFIGS else sorted(MESSK_CONFIGS)[0],
        description="System:",
        style={"description_width": LABEL_W},
    )
    field_lbl = W.HTML(value="")
    dv_input = W.Text(value="0.3, 0.55, 0.8", description="Δv values:", style={"description_width": LABEL_W})
    damping_input = W.Text(value="0.8, 1.2, 1.6", description="Damping values:", style={"description_width": LABEL_W})
    dt_box = W.FloatText(value=0.02, description="dt:", style={"description_width": LABEL_W}, layout=W.Layout(width="220px"))
    auto_dt_btn = W.Button(description="Auto dt", button_style="info", layout=W.Layout(width="90px"))
    n_box = W.IntText(value=10, description="n_steps:", style={"description_width": LABEL_W}, layout=W.Layout(width="220px"))
    m_box = W.IntText(value=24, description="m (ticks):", style={"description_width": LABEL_W}, layout=W.Layout(width="220px"))
    seeds_box = W.IntText(value=10, description="n_seeds:", style={"description_width": LABEL_W}, layout=W.Layout(width="220px"))
    run_btn = W.Button(description="Run Grid Screen", button_style="primary", layout=W.Layout(width="200px"))
    out = W.Output()

    def update_field(*_):
        f = _detect_damping_field(system_dd.value)
        field_lbl.value = f"<span style='color:#555;'>Damping field: <b>{f}</b></span>"
        damping_input.description = f"{f} values:"
    system_dd.observe(update_field, "value")
    update_field()

    def on_auto_dt(_):
        res = auto_dt(system_dd.value)
        dt_box.value = res["dt"]
    auto_dt_btn.on_click(on_auto_dt)

    def on_run(_):
        out.clear_output()
        with out:
            try:
                dvs = [float(x.strip()) for x in dv_input.value.split(",") if x.strip()]
                damps = [float(x.strip()) for x in damping_input.value.split(",") if x.strip()]
                print(f"Running grid screen on {system_dd.value} ({len(dvs)}×{len(damps)} cells × {seeds_box.value} seeds)...")
                res = grid_screen(
                    system_name=system_dd.value,
                    delta_v_values=dvs,
                    damping_values=damps,
                    dt=dt_box.value,
                    n_steps=n_box.value,
                    m=m_box.value,
                    n_seeds=seeds_box.value,
                )
                import pandas as pd
                df = pd.DataFrame(res["grid"])
                display(df)
                print(f"Done! {len(res['passing'])}/{len(res['grid'])} cells passed GO criteria.")
            except Exception as e:
                print(f"Error: {e}")

    run_btn.on_click(on_run)

    return W.VBox([
        W.HTML("<h4>2D Parameter Grid Screening (10 Seeds)</h4>"),
        system_dd,
        field_lbl,
        dv_input,
        damping_input,
        W.HBox([dt_box, auto_dt_btn]),
        W.HBox([n_box, m_box, seeds_box]),
        run_btn,
        out,
    ])


def _demo() -> None:
    import matplotlib

    matplotlib.use("Agg")  # the demo runs headless; the notebook picks its own
    assert parse_bins("181") == (181,) and parse_bins("181x181") == (181, 181)
    assert parse_bins("64, 32") == (64, 32)
    for bad in ("", "1", "0x5"):
        try:
            parse_bins(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{bad!r} should be rejected")

    slider, row = param_row("gamma", 1.2)
    assert slider.min <= 1.2 <= slider.max
    row.children[3].value = 500.0  # retype the top of the range
    assert slider.max == 500.0, slider.max
    row.children[2].value = -500.0
    assert slider.min == -500.0 and slider.max == 500.0

    tr = trace(make_process("pendulum_mess4", m=6), seed=0)
    assert "STABLE" in _verdict(stability(tr))
    assert explorer() is not None and sweep_ui() is not None
    res = grid_screen("pendulum_mess4", [0.55], [1.2], n_steps=6, m=6, n_seeds=2)
    assert len(res["grid"]) == 1
    recal = n_recalibrate("pendulum_mess4", 0.55, 1.2, dt=0.02, n_values=[5, 10], m=6, n_seeds=2)
    assert len(recal) == 2
    assert grid_screen_ui() is not None
    print("controls ok")


if __name__ == "__main__":
    _demo()
