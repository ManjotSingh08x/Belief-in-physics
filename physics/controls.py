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
import numpy as np
from IPython.display import display

from .messk_configs import MESSK_CONFIGS, make_process
from .visualise import PANELS, plot, stability, trace, tunable_fields

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
        # Widen before narrowing, or an intermediate state raises min > max.
        slider.min, slider.max = min(low, slider.min), max(high, slider.max)
        slider.min, slider.max = low, high
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


def _verdict(report: dict) -> str:
    colour = "#1baf7a" if report["stable"] else "#c0392b"
    headline = "STABLE" if report["stable"] else "UNSTABLE: " + "; ".join(report["reasons"])
    lo, hi = report["obs_range"]
    olo, ohi = report["observed_range"]
    return (
        f"<div style='font-family:monospace;font-size:12px'>"
        f"<b style='color:{colour}'>{headline}</b><br>"
        f"lyapunov {report['lyapunov']:+.3f} /s &nbsp; "
        f"clipped {report['clipped']:.2%} (per channel "
        f"{', '.join(f'{c:.2%}' for c in report['per_channel_clipped'])})<br>"
        f"bins used {report['used_bins']}/{report['n_obs']} &nbsp; "
        f"channel 0 range [{lo:.3g}, {hi:.3g}] but visits [{olo:.3g}, {ohi:.3g}]<br>"
        f"driven-vs-free gap mean {report['gap_free_mean']:.3g} max {report['gap_free_max']:.3g}"
        f" &nbsp; free-run energy drift {report['energy_drift_free']:+.2%}"
        f"</div>"
    )


def explorer(default: str = "pendulum_mess4", panels=("observable", "energy", "metric", "phase")):
    """The whole single-system dashboard: every parameter, every graph.

    Returns the widget rather than displaying it, so a notebook cell can place
    it and a test can build it without a kernel front end.
    """
    import matplotlib.pyplot as plt

    system_dd = W.Dropdown(options=sorted(MESSK_CONFIGS), value=default, description="system",
                           style={"description_width": LABEL_W})
    panel_sel = W.SelectMultiple(options=PANELS, value=tuple(panels), description="graphs",
                                 rows=len(PANELS), style={"description_width": LABEL_W},
                                 layout=W.Layout(width="440px"))
    seed = W.IntText(value=0, description="seed", style={"description_width": LABEL_W})
    bins = W.Text(value="181", description="bins per channel",
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
        return trace(proc, seed=int(seed.value))

    def redraw(_=None) -> None:
        if not live.value and _ is not None:
            return
        with out:
            out.clear_output(wait=True)
            try:
                tr = build()
            except Exception as exc:  # a bad parameter is a normal event here
                report_html.value = f"<pre style='color:#c0392b'>{type(exc).__name__}: {exc}</pre>"
                return
            report_html.value = _verdict(stability(tr))
            fig = plot(tr, panels=panel_sel.value, title=system_dd.value)
            plt.show()
            plt.close(fig)

    def reset_system(_=None, build_only: bool = False) -> None:
        spec = make_process(system_dd.value)
        params.rebuild(spec.system, redraw)
        dt_s.value, dv_s.value = spec.dt, spec.delta_v
        if not build_only:
            redraw()

    for w in (panel_sel, seed, bins, m_s, n_s, dt_s, dv_s, alpha_s, stay_s):
        w.observe(redraw, "value")
    system_dd.observe(reset_system, "value")
    draw_btn.on_click(lambda _: redraw())

    reset_system(build_only=True)
    controls = W.VBox([
        W.HBox([system_dd, live, draw_btn]),
        W.HTML("<b>driver</b> - how the chain meets the physics"),
        m_row, n_row, dt_row, dv_row, bins,
        W.HTML("<b>chain</b>"), alpha_row, stay_row,
        W.HTML("<b>system</b> - every dataclass field, read off the system itself"),
        params.box,
        W.HTML("<b>view</b>"), panel_sel, seed,
    ])
    return W.VBox([W.HBox([controls, report_html]), out])


def sweep_ui(default: str = "double_pendulum_mess4"):
    """Scan one parameter and plot where the trajectory stays stable."""
    import matplotlib.pyplot as plt

    system_dd = W.Dropdown(options=sorted(MESSK_CONFIGS), value=default, description="system",
                           style={"description_width": LABEL_W})
    field_dd = W.Dropdown(options=[], description="parameter",
                          style={"description_width": LABEL_W})
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
        options = ["delta_v", "dt", "n_steps"] + sorted(tunable_fields(proc.system))
        field_dd.options = options
        field_dd.value = "delta_v"

    def go(_=None) -> None:
        with out:
            out.clear_output(wait=True)
            values = np.linspace(lo.value, hi.value, int(steps.value))
            name, rows = field_dd.value, []
            for v in values:
                kwargs = ({"system": {name: float(v)}} if name in tunable_fields(
                    make_process(system_dd.value).system) else {name: float(v)})
                if name == "n_steps":
                    kwargs = {"n_steps": max(1, int(round(v)))}
                try:
                    tr = trace(make_process(system_dd.value, m=24, **kwargs), seed=0)
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
            fig.suptitle(f"{system_dd.value}: stable at {name} in "
                         f"{{{', '.join(good) if good else 'nothing in this range'}}}",
                         fontsize=11)
            fig.tight_layout()
            plt.show()
            plt.close(fig)

    system_dd.observe(reset, "value")
    run.on_click(go)
    reset()
    return W.VBox([W.HBox([system_dd, field_dd]), W.HBox([lo, hi, steps, run]), out])


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
    print("controls ok")


if __name__ == "__main__":
    _demo()
