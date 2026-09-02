"""Figures for `notebooks/lab.ipynb`. Matplotlib only, no widget imports.

Split from `lab.py` so the pipelines can run headless (on staging, in a script)
without a display stack, and so a plotting change never touches the numbers.
"""

from __future__ import annotations

import numpy as np

LETTER_COLOURS = ("tab:blue", "tab:orange", "tab:green", "tab:red")


def plot_generation(proc, batch, index: int = 0, tick: int | None = None, counterfactual=None):
    """The data-generation pipeline, with no transformer anywhere in it."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    step = np.arange(proc.seq_len)
    kicks = np.arange(proc.m) * proc.n_steps
    letters = batch["letters"][index]
    lo, hi = proc.obs_ranges[0]

    ax = axes[0, 0]
    observed = batch["observable"][index]
    for channel, name in enumerate(proc.channel_names):
        ax.plot(step, observed[:, channel], lw=1.3, zorder=3, label=name)
    ax.axhline(lo, color="tab:red", lw=0.8, ls=":")
    ax.axhline(hi, color="tab:red", lw=0.8, ls=":", label="channel 0 range")
    for start, letter in zip(kicks, letters):
        ax.axvspan(start, start + proc.n_steps, color=LETTER_COLOURS[letter], alpha=0.16, lw=0)
    ax.set_title("trajectory, shaded by the letter kicking each tick")
    ax.set_xlabel("physics step")
    ax.legend(fontsize=7, loc="lower right")

    ax = axes[0, 1]
    if counterfactual is None:
        ax.text(0.5, 0.5, "pick a tick to branch", ha="center", va="center")
        ax.set_axis_off()
    else:
        branch = counterfactual["tick"] * proc.n_steps
        for letter in range(proc.chain.n_states):
            actual = letter == counterfactual["actual"]
            ax.plot(step, counterfactual["observable"][letter][:, 0],
                    color=LETTER_COLOURS[letter], lw=2.0 if actual else 1.1,
                    alpha=1.0 if actual else 0.75, ls="-" if actual else "--",
                    label=f"letter {letter}" + (" (actual)" if actual else ""))
        ax.axvline(branch, color="0.3", lw=1.0)
        ax.set_title(f"same history, different letter at tick {counterfactual['tick']}: "
                     f"mean separation {counterfactual['separation']:.1f} tokens")
        ax.set_xlabel("physics step")
        ax.legend(fontsize=7)

    ax = axes[1, 0]
    ax.plot(step, batch["tokens"][index], color="tab:purple", lw=1.0, drawstyle="steps-post")
    for start in kicks:
        ax.axvline(start, color="0.85", lw=0.7, zorder=0)
    ax.set_ylim(-1, proc.n_obs)
    ax.set_title(f"what the model sees: {'x'.join(map(str, proc.obs_bins))} = {proc.n_obs} tokens")
    ax.set_xlabel("physics step")
    ax.set_ylabel("token")

    ax = axes[1, 1]
    report = proc.bin_report(batch["observable"])
    ax.bar(np.arange(proc.n_obs), report["counts"], width=1.0, color="tab:purple")
    ax.set_title(f"bin occupancy: {report['used_bins']}/{report['n_obs']} used, "
                 f"{report['clipped']:.2%} clipped")
    if len(proc.obs_bins) > 1:
        ax.set_yscale("log")
    ax.set_xlabel("token")
    ax.set_ylabel("count")

    ax = axes[2, 0]
    truth = batch["beliefs"][index][proc.n_steps - 1 :: proc.n_steps]
    for state in range(proc.chain.n_states):
        ax.plot(np.arange(proc.m), truth[:, state], color=LETTER_COLOURS[state], lw=1.5,
                label=f"mood {state}")
    ax.set_ylim(0, 1)
    ax.set_title("exact belief P(next mood | letters)")
    ax.set_xlabel("tick")
    ax.legend(fontsize=7, ncol=4)

    ax = axes[2, 1]
    for j, metric_name in enumerate(proc.system.metric_names):
        ax.plot(step, batch["metric"][index, :, j], lw=1.2, label=metric_name)
    for start in kicks:
        ax.axvline(start, color="0.85", lw=0.7, zorder=0)
    ax.set_title("probed metric, every step (kicks are the discontinuities)")
    ax.set_xlabel("physics step")
    ax.legend(fontsize=7)

    fig.tight_layout()
    return fig


def plot_all_systems(survey: dict):
    """One row per system: trajectory, counterfactual, occupancy, metric.

    The belief is identical across the four rows by construction, so it is not
    plotted four times. What the rows are for is the *channel*: how visibly one
    letter changes the observable, and whether the bins land where the data is.
    """
    import matplotlib.pyplot as plt

    names = sorted(survey)
    fig, axes = plt.subplots(len(names), 4, figsize=(19, 3.1 * len(names)))
    for row, name in enumerate(names):
        entry = survey[name]
        proc, batch, cf = entry["proc"], entry["batch"], entry["counterfactual"]
        step = np.arange(proc.seq_len)
        lo, hi = proc.obs_ranges[0]

        ax = axes[row, 0]
        for channel in range(len(proc.obs_bins)):
            ax.plot(step, batch["observable"][0][:, channel], lw=1.1)
        for start, letter in zip(np.arange(proc.m) * proc.n_steps, batch["letters"][0]):
            ax.axvspan(start, start + proc.n_steps, color=LETTER_COLOURS[letter], alpha=0.15, lw=0)
        ax.axhline(lo, color="tab:red", lw=0.7, ls=":")
        ax.axhline(hi, color="tab:red", lw=0.7, ls=":")
        ax.set_ylabel(name.replace("_mess4", ""), fontsize=9)
        if row == 0:
            ax.set_title("trajectory, shaded by letter")

        ax = axes[row, 1]
        for letter in range(proc.chain.n_states):
            actual = letter == cf["actual"]
            ax.plot(step, cf["observable"][letter][:, 0], color=LETTER_COLOURS[letter],
                    lw=1.8 if actual else 1.0, ls="-" if actual else "--",
                    alpha=1.0 if actual else 0.75)
        ax.axvline(cf["tick"] * proc.n_steps, color="0.3", lw=0.9)
        ax.set_title("one letter changed" if row == 0 else "")
        ax.text(0.99, 0.03, f"separation {entry['separation']:.1f} tokens", fontsize=7,
                ha="right", transform=ax.transAxes)

        ax = axes[row, 2]
        report = entry["bins"]
        ax.bar(np.arange(proc.n_obs), report["counts"], width=1.0, color="tab:purple")
        if len(proc.obs_bins) > 1:
            ax.set_yscale("log")
        ax.set_title("bin occupancy" if row == 0 else "")
        ax.text(0.5, 0.92, f"{report['used_bins']}/{report['n_obs']} used, "
                           f"{report['clipped']:.2%} clipped", fontsize=7,
                ha="center", transform=ax.transAxes)

        ax = axes[row, 3]
        for j, metric_name in enumerate(proc.system.metric_names):
            ax.plot(step, batch["metric"][0, :, j], lw=1.0, label=metric_name)
        ax.set_title("probed metric" if row == 0 else "")
        ax.legend(fontsize=6, loc="upper right")

    for ax in axes[-1]:
        ax.set_xlabel("physics step")
    fig.tight_layout()
    return fig


def plot_training(report: dict):
    """Loss curve with the uniform-prediction line and the snapshot points."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 4))
    history = report["history"]
    tokens = [h["step"] * report["seq_len"] * report.get("batch_size", 128) for h in history]
    ax.plot(tokens, [h["train_loss"] for h in history], lw=0.9, alpha=0.6, label="train")
    ax.plot(tokens, [h["eval_loss"] for h in history], lw=1.6, label="eval")
    ax.axhline(report["uniform_loss"], color="tab:red", ls=":",
               label=f"uniform ({report['uniform_loss']:.2f} nats)")
    for at in report.get("checkpoint_paths", {}):
        ax.axvline(int(at), color="0.8", lw=0.8, zorder=0)
    ax.set_xlabel("tokens seen")
    ax.set_ylabel("cross entropy (nats)")
    ax.set_title("training, with snapshot points marked")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_probe_table(rows: list[dict]):
    """Belief and metric R2 against training progress, plus the loss."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    labels = [str(r["label"]) for r in rows]
    x = np.arange(len(rows))

    ax = axes[0]
    ax.bar(x - 0.2, [r["belief_r2"] for r in rows], 0.4, label="belief", color="tab:blue")
    ax.bar(x + 0.2, [r["metric_r2"] for r in rows], 0.4, label="metric", color="tab:orange")
    ax.axhline(0, color="0.3", lw=0.8)
    ax.set_xticks(x, labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("held-out R2")
    ax.set_title("what the residual stream carries, per checkpoint")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(x, [r["eval_loss"] for r in rows], "o-", color="0.2")
    ax.set_xticks(x, labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("eval loss (nats)")
    ax.set_title("next-token loss, same checkpoints")

    fig.tight_layout()
    return fig


def _demo() -> None:
    import matplotlib
    matplotlib.use("Agg")

    from physics.messk_configs import make_process

    from .lab import counterfactual

    proc = make_process("pendulum_mess4", m=6, n_steps=5)
    batch = proc.sample_batch(np.random.default_rng(0), 4)
    cf = counterfactual(proc, batch["letters"][0], tick=2)
    assert len(plot_generation(proc, batch, 0, 2, cf).axes) == 6
    assert len(plot_generation(proc, batch, 0).axes) == 6, "must render without a counterfactual"

    from .lab import survey_systems

    survey = survey_systems(n=8, m=4, n_steps=5)
    assert len(plot_all_systems(survey).axes) == 16

    report = {"history": [{"step": 0, "train_loss": 5.2, "eval_loss": 5.1},
                          {"step": 10, "train_loss": 1.0, "eval_loss": 1.1}],
              "seq_len": proc.seq_len, "uniform_loss": float(np.log(proc.n_obs)),
              "checkpoint_paths": {0: "a", 1000: "b"}}
    assert len(plot_training(report).axes) == 1
    rows = [{"label": 0, "belief_r2": 0.2, "metric_r2": 0.9, "eval_loss": 5.1},
            {"label": 1000, "belief_r2": 0.8, "metric_r2": 0.99, "eval_loss": 0.4}]
    assert len(plot_probe_table(rows).axes) == 2
    print("labplots ok")


if __name__ == "__main__":
    _demo()
