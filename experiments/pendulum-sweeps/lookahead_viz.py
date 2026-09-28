"""Lookahead prediction and counterfactual branching visualization module.

Provides:
- compute_counterfactual_branches: Replays physical ODE to evaluate counterfactual branches for each impulse letter.
- compute_horizon_branches: Evaluates horizon-aware branching mask and Bayes-optimal predictive distribution.
- prepare_run_data: Precomputes lookahead predictions, probabilities, and Bayes-optimal alignment data for a trained model.
- plot_stacked_horizons: Arranges horizons (k=1, k=n//2, k=n) in 3 full-width rows stacked vertically.
- plot_lookahead_branching: Detailed 2-panel inspection with time cursor tau and probability distribution slice.
- launch_interactive_dashboard: Interactive ipywidgets dashboard for exploration in Jupyter.
- plot_enhanced_summary: Comprehensive summary plot with 3 stacked horizon rows, loss curves, and probe metrics.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from models.explore import next_token_probs as token_probabilities
from models.transformer import ModelConfig
from physics.messk import MessDriven

BRANCH_COLORS = ["#ff5555", "#ffb86c", "#50fa7b", "#8be9fd"]  # L0 (red), L1 (orange), L2 (green), L3 (blue)
BRANCH_NAMES = ["L0 (-1.64)", "L1 (-0.55)", "L2 (+0.55)", "L3 (+1.64)"]


def compute_counterfactual_branches(proc: MessDriven, letters: np.ndarray) -> np.ndarray:
    """Replays physical ODE with letters L0, L1, L2, L3 at each tick to get exact physical branches.

    Uses 1-tick forward simulation from cached pre-kick states for 25x acceleration.

    Args:
        proc: MessDriven physics process.
        letters: 1D or 2D array containing the letter sequence of length m.

    Returns:
        branch_tokens: Array of shape (4, seq_len) with discrete tokens for each counterfactual impulse.
    """
    letters_1d = np.asarray(letters, dtype=np.int64).ravel()
    m_ticks = proc.m
    n_steps = proc.n_steps
    seq_len = proc.seq_len
    sys_ = proc.system
    actions = proc.actions

    z = sys_.initial_state(1)
    z_before_kicks = np.empty((m_ticks, 2))
    for j in range(m_ticks):
        z_before_kicks[j] = z[0]
        z = sys_.kick(z, actions[letters_1d[j : j + 1]])
        for s in range(n_steps):
            z = sys_.flow(z, proc.dt)

    branch_tokens = np.zeros((4, seq_len), dtype=np.int64)
    for j in range(m_ticks):
        z_init_4 = np.repeat(z_before_kicks[j : j + 1], 4, axis=0)
        z_kicked = sys_.kick(z_init_4, actions[np.arange(4)])
        step_obs = np.empty((4, n_steps, len(proc.obs_bins)))
        for s in range(n_steps):
            z_kicked = sys_.flow(z_kicked, proc.dt)
            step_obs[:, s] = proc.channels(z_kicked)
        branch_tokens[:, j * n_steps : (j + 1) * n_steps] = proc.discretise(step_obs)

    return branch_tokens


def compute_horizon_branches(
    proc: MessDriven,
    tokens: np.ndarray,
    letters: np.ndarray,
    k: int,
    all_branches: np.ndarray | None = None,
    beliefs: np.ndarray | None = None,
) -> dict[str, Any]:
    """Computes horizon-aware counterfactual branches and Bayes-optimal predictive distribution.

    For lookahead prediction at target tau from source t = tau - k:
    - If t < J * n (impulse J unobserved at source):
      Impulse L_J at tick J occurs in (t, tau]. The prediction is stochastic with 4 physical branches.
      is_branching[tau] = True.
      Bayes optimal observer uses predictive belief b_t to compute P(L_J = l) = (b_t @ E)[l].
    - If t >= J * n (impulse J already observed in context):
      No impulses occur in (t, tau]. Physics is completely deterministic given context.
      is_branching[tau] = False.
      Counterfactual branches collapse; non-actual branches are masked with NaN.
      Bayes optimal observer predicts P(actual_token) = 1.0.

    Args:
        proc: MessDriven physics process.
        tokens: Actual discrete token trajectory (seq_len,).
        letters: Actual impulse letter sequence (m,).
        k: Lookahead horizon step count.
        all_branches: Optional precomputed array of shape (4, seq_len).
        beliefs: Optional predictive beliefs array of shape (seq_len, 4).

    Returns:
        Dict with:
        - branch_tokens: Array (4, seq_len) with non-actual branches set to np.nan during deterministic steps.
        - all_branches: Array (4, seq_len) with all 4 counterfactual tokens at every step.
        - is_branching: Boolean array (seq_len,) indicating stochastic steps.
        - bayes_probs: Array (seq_len, n_obs) of Bayes-optimal predictive probabilities.
        - bayes_nll: Array (seq_len,) of negative log likelihoods under Bayes-optimal observer.
        - mean_bayes_nll: Float mean Bayes NLL over the forecast range [k..seq_len-1].
        - pct_branching: Float percentage of forecast steps that are stochastic branches.
    """
    letters_1d = np.asarray(letters, dtype=np.int64).ravel()
    seq_len = proc.seq_len
    n_steps = proc.n_steps
    E = proc.chain.E

    if all_branches is None:
        all_branches = compute_counterfactual_branches(proc, letters_1d)

    if beliefs is None:
        tick_beliefs = proc.chain.beliefs(np.atleast_2d(letters_1d))
        beliefs = np.repeat(tick_beliefs, n_steps, axis=1)[0]

    is_branching = np.zeros(seq_len, dtype=bool)
    branch_tokens = np.full((4, seq_len), np.nan, dtype=np.float64)
    bayes_probs = np.zeros((seq_len, proc.n_obs), dtype=np.float64)
    bayes_nll = np.zeros(seq_len, dtype=np.float64)

    for tau in range(k, seq_len):
        t_src = tau - k
        J = tau // n_steps
        t_kick = J * n_steps
        actual_l = letters_1d[J]
        actual_tok = tokens[tau]

        if t_src < t_kick:
            # Stochastic regime: impulse at t_kick was unobserved at source t_src
            is_branching[tau] = True
            branch_tokens[:, tau] = all_branches[:, tau]
            b_src = beliefs[t_src]
            p_letters = b_src @ E
            for l in range(4):
                tok_l = all_branches[l, tau]
                bayes_probs[tau, tok_l] += p_letters[l]
            bayes_nll[tau] = -np.log(max(bayes_probs[tau, actual_tok], 1e-12))
        else:
            # Deterministic regime: impulse at t_kick was already observed at source t_src
            is_branching[tau] = False
            branch_tokens[actual_l, tau] = actual_tok
            bayes_probs[tau, actual_tok] = 1.0
            bayes_nll[tau] = 0.0

    mean_bayes_nll = float(bayes_nll[k:].mean()) if k < seq_len else 0.0
    pct_branching = float(is_branching[k:].mean() * 100.0) if k < seq_len else 0.0

    return {
        "branch_tokens": branch_tokens,
        "all_branches": all_branches,
        "is_branching": is_branching,
        "bayes_probs": bayes_probs,
        "bayes_nll": bayes_nll,
        "mean_bayes_nll": mean_bayes_nll,
        "pct_branching": pct_branching,
    }


def prepare_run_data(
    run: dict[str, Any],
    batch: dict[str, np.ndarray],
    device: str = "cpu",
    traj_idx: int = 0,
) -> dict[str, Any]:
    """Precomputes model predictions, probabilities, and physical alignments for a run.

    Args:
        run: Dict containing 'model', 'k' or 'training_config' with k, 'proc', 'run_name', 'config_name'.
        batch: Dict from proc.sample_batch containing 'tokens', 'letters', and optionally 'beliefs'.
        device: Device to run inference on.
        traj_idx: Index of trajectory within batch to analyze.

    Returns:
        Dictionary with precomputed forecast probabilities, targets, branches, Bayes NLL, and suboptimality gap.
    """
    model = run["model"]
    cfg = run.get("training_config")
    k = run.get("k", getattr(cfg, "k", 1))
    k_suffix = run.get("k_suffix", f"k{k}")
    proc = run["proc"]

    tokens = batch["tokens"][traj_idx]
    letters = batch["letters"][traj_idx]
    beliefs_batch = batch.get("beliefs")
    beliefs_traj = beliefs_batch[traj_idx] if beliefs_batch is not None else None
    seq_len = proc.seq_len

    all_branches = compute_counterfactual_branches(proc, letters)
    h_data = compute_horizon_branches(
        proc,
        tokens,
        letters,
        k,
        all_branches=all_branches,
        beliefs=beliefs_traj,
    )

    probs = token_probabilities(model, tokens, device)
    probs_k = probs[:-k] if k < len(tokens) else np.zeros((0, proc.n_obs))
    actual_k = tokens[k:] if k < len(tokens) else np.zeros(0, dtype=np.int64)

    if len(actual_k) > 0:
        model_nll = -np.log(np.maximum(probs_k[np.arange(len(actual_k)), actual_k], 1e-12))
    else:
        model_nll = np.array([0.0])

    mean_model_nll = float(model_nll.mean()) if len(model_nll) > 0 else 0.0
    mean_bayes_nll = h_data["mean_bayes_nll"]
    suboptimality_gap = mean_model_nll - mean_bayes_nll

    return {
        "run_name": run.get("run_name", f"Model (k={k})"),
        "config_name": run.get("config_name", getattr(cfg, "name", f"k={k}")),
        "k": k,
        "k_suffix": k_suffix,
        "proc": proc,
        "tokens": tokens,
        "letters": letters,
        "beliefs": beliefs_traj,
        "seq_len": seq_len,
        "n_steps": proc.n_steps,
        "kicks": np.arange(proc.m) * proc.n_steps,
        "branch_tokens": h_data["branch_tokens"],
        "all_branches": all_branches,
        "is_branching": h_data["is_branching"],
        "probs_k": probs_k,
        "actual_k": actual_k,
        "nll": model_nll,
        "model_nll": model_nll,
        "mean_model_nll": mean_model_nll,
        "bayes_probs": h_data["bayes_probs"],
        "bayes_nll": h_data["bayes_nll"],
        "mean_bayes_nll": mean_bayes_nll,
        "suboptimality_gap": suboptimality_gap,
        "pct_branching": h_data["pct_branching"],
    }


def plot_stacked_horizons(
    horizon_data_list: Sequence[dict[str, Any]],
    save_path: Path | str | None = None,
    title: str | None = None,
    figsize: tuple[float, float] = (16.0, 11.0),
    max_display_tokens: int | None = 100,
) -> plt.Figure:
    """Plots prediction horizons arranged in full-width rows stacked one below the other.

    Each row stretches across the breadth of the figure, displaying:
    - Dark background with Magma PowerNorm probability heatmap.
    - Translucent shading for stochastic branching windows.
    - 4 counterfactual impulse branches (active during stochastic steps, masked during deterministic steps).
    - Scatter dots for single-step branches (ensuring k=1 branches are clearly visible).
    - Actual token trajectory in electric cyan.
    - White dashed tick boundaries and yellow letter labels.
    - Title showing Model NLL, Bayes Optimal NLL, Suboptimality Gap, and Branching Percentage.

    Args:
        horizon_data_list: List of dicts returned by prepare_run_data (e.g. for k1, kn2, kn).
        save_path: Optional path to save the generated figure.
        title: Optional suptitle for the entire figure.
        figsize: Figure dimensions (width, height).
        max_display_tokens: Maximum tokens to show on the x-axis for high-detail view (defaults to 100).

    Returns:
        plt.Figure object.
    """
    n_rows = len(horizon_data_list)
    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=1,
        figsize=figsize,
        sharex=True,
        dpi=120,
    )
    if n_rows == 1:
        axes = [axes]

    for idx, (ax, data) in enumerate(zip(axes, horizon_data_list)):
        k = data["k"]
        k_suffix = data.get("k_suffix", f"k{k}")
        seq_len = data["seq_len"]
        proc = data["proc"]
        n_steps = proc.n_steps
        probs_k = data["probs_k"]
        tokens = data["tokens"]
        letters = data["letters"]
        branch_tokens = data["branch_tokens"]
        is_branching = data.get("is_branching", np.ones(seq_len, dtype=bool))
        n_obs = proc.n_obs

        display_len = min(seq_len, max_display_tokens) if max_display_tokens is not None else seq_len
        target_times = np.arange(k, display_len)
        br_mask = is_branching[k:display_len]
        probs_k_disp = probs_k[: max(0, display_len - k)]

        ax.set_facecolor("#050508")

        if len(probs_k_disp) > 0:
            vmax = max(0.01, float(np.max(probs_k_disp)) * 0.85)
            norm = mcolors.PowerNorm(gamma=0.45, vmin=0, vmax=vmax)
            extent = [k - 0.5, display_len - 0.5, 0, n_obs - 1]
            ax.imshow(
                probs_k_disp.T,
                origin="lower",
                aspect="auto",
                cmap="magma",
                extent=extent,
                norm=norm,
                interpolation="nearest",
            )

            # Translucent tint for stochastic branching windows
            for j in range(proc.m):
                t_kick = j * n_steps
                if t_kick >= display_len:
                    break
                br_start = max(k, t_kick)
                br_end = min(display_len, t_kick + k)
                if br_end > br_start:
                    ax.axvspan(br_start - 0.5, br_end - 0.5, color="#ff79c6", alpha=0.06, lw=0)

            # Plot counterfactual branches (NaNs naturally omitted by matplotlib)
            for l in range(4):
                y_branch = branch_tokens[l, k:display_len]
                ax.plot(
                    target_times,
                    y_branch,
                    color=BRANCH_COLORS[l],
                    lw=1.5,
                    alpha=0.8,
                    label=f"Branch {BRANCH_NAMES[l]}",
                )
                # Scatter markers for isolated / single-step branching
                if np.any(br_mask):
                    ax.scatter(
                        target_times[br_mask],
                        y_branch[br_mask],
                        color=BRANCH_COLORS[l],
                        s=16,
                        alpha=0.9,
                        zorder=4,
                    )

            # Actual trajectory in electric cyan
            ax.plot(
                target_times,
                tokens[k:display_len],
                color="#00ffff",
                lw=2.2,
                alpha=0.95,
                label="Actual Trajectory",
            )

        for tick_idx, tick_pos in enumerate(data["kicks"]):
            if tick_pos >= display_len:
                break
            ax.axvline(tick_pos, color="white", lw=0.7, ls="--", alpha=0.35)
            if tick_pos >= k:
                ax.text(
                    tick_pos + 0.3,
                    n_obs - 6,
                    f"L{letters[tick_idx]}",
                    color="yellow",
                    fontsize=8.5,
                    alpha=0.85,
                    va="top",
                    ha="left",
                )

        mean_model_nll = float(data.get("mean_model_nll", data["nll"].mean() if len(data["nll"]) > 0 else 0.0))
        mean_bayes_nll = float(data.get("mean_bayes_nll", 0.0))
        gap = float(data.get("suboptimality_gap", mean_model_nll - mean_bayes_nll))
        pct_br = float(data.get("pct_branching", is_branching[k:].mean() * 100 if len(is_branching) > k else 0.0))

        regime_desc = (
            "1-step kick rollouts (remainder deterministic)"
            if k == 1
            else (
                f"First {k} steps guess, remainder deterministic"
                if pct_br < 95
                else "Full cycle stochastic branching"
            )
        )

        ax.set_xlim(0, display_len)
        ax.set_ylim(0, n_obs - 1)
        ax.set_ylabel("Token Bin", fontsize=9.5)
        ax.set_title(
            f"Horizon {k_suffix} (k={k}) | Model NLL: {mean_model_nll:.3f} | Bayes Optimal NLL: {mean_bayes_nll:.3f} | "
            f"Gap (Excess NLL): {gap:+.3f} nats | Branching: {pct_br:.1f}% [{regime_desc}]",
            fontsize=10.5,
            fontweight="bold",
            pad=6,
        )

        if idx == 0:
            ax.legend(
                loc="upper right",
                fontsize=8,
                ncol=5,
                framealpha=0.75,
                facecolor="#181824",
                edgecolor="#44475a",
                labelcolor="white",
            )

    axes[-1].set_xlabel("Target Time Step τ (Lookahead Target)", fontsize=10.5)

    if title:
        fig.suptitle(title, fontsize=12.5, fontweight="bold", y=0.995)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=120)
    return fig


def plot_lookahead_branching(
    data: dict[str, Any],
    tau: int | None = None,
    save_path: Path | str | None = None,
    max_display_tokens: int | None = 100,
) -> plt.Figure:
    """Detailed 2-panel inspection with time cursor tau and probability distribution slice.

    Panel 0 (top): Stretched breadth view of trajectory and counterfactual branches, displaying whether
    the current cursor tau is in the stochastic or deterministic regime.
    Panel 1 (bottom): Direct comparison between Model predictive distribution and Bayes-optimal predictive
    distribution at cursor tau, highlighting physical counterfactual branches and Bayes gap.

    Args:
        data: Dict returned by prepare_run_data.
        tau: Target time step to inspect (defaults to k + 15 within display range).
        save_path: Optional path to save figure.
        max_display_tokens: Maximum tokens to show on the top panel x-axis (defaults to 100).

    Returns:
        plt.Figure object.
    """
    k = data["k"]
    seq_len = data["seq_len"]
    proc = data["proc"]
    n_steps = data["n_steps"]
    probs_k = data["probs_k"]
    tokens = data["tokens"]
    letters = data["letters"]
    branch_tokens = data["branch_tokens"]
    all_branches = data.get("all_branches", branch_tokens)
    is_branching = data.get("is_branching", np.ones(seq_len, dtype=bool))
    bayes_probs = data.get("bayes_probs")
    E = proc.chain.E

    display_len = min(seq_len, max_display_tokens) if max_display_tokens is not None else seq_len

    if tau is None or tau < k:
        tau = min(k + 15, display_len - 1)
    tau = min(max(k, tau), seq_len - 1)

    t_source = tau - k
    slice_probs = probs_k[t_source]
    J = tau // n_steps
    actual_letter = letters[J]
    actual_tok = tokens[tau]
    is_br_tau = is_branching[tau]

    bayes_slice = (
        bayes_probs[tau]
        if bayes_probs is not None
        else np.zeros(proc.n_obs)
    )

    probs_k_disp = probs_k[: max(0, display_len - k)]
    target_times = np.arange(k, display_len)

    norm = mcolors.PowerNorm(gamma=0.45, vmin=0, vmax=max(0.01, float(np.max(probs_k_disp)) * 0.85))
    extent = [k - 0.5, display_len - 0.5, 0, proc.n_obs - 1]

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(16, 9), height_ratios=[1.8, 1.2], dpi=110)
    ax0.set_facecolor("#050508")
    ax0.imshow(
        probs_k_disp.T,
        origin="lower",
        aspect="auto",
        cmap="magma",
        extent=extent,
        norm=norm,
        interpolation="nearest",
    )

    # Shading for stochastic intervals within display range
    for j in range(proc.m):
        t_kick = j * n_steps
        if t_kick >= display_len:
            break
        br_start = max(k, t_kick)
        br_end = min(display_len, t_kick + k)
        if br_end > br_start:
            ax0.axvspan(br_start - 0.5, br_end - 0.5, color="#ff79c6", alpha=0.06, lw=0)

    for l in range(4):
        ax0.plot(
            target_times,
            branch_tokens[l, k:display_len],
            color=BRANCH_COLORS[l],
            lw=1.5,
            alpha=0.8,
            label=f"Branch {BRANCH_NAMES[l]}",
        )
    ax0.plot(target_times, tokens[k:display_len], color="#00ffff", lw=2.2, alpha=0.95, label="Actual Trajectory")

    for tick_idx, tick_pos in enumerate(data["kicks"]):
        if tick_pos >= display_len:
            break
        ax0.axvline(tick_pos, color="white", lw=0.7, ls="--", alpha=0.35)
        if tick_pos >= k:
            ax0.text(
                tick_pos + 0.3,
                proc.n_obs - 6,
                f"L{letters[tick_idx]}",
                color="yellow",
                fontsize=8.5,
                alpha=0.85,
                va="top",
            )

    if tau < display_len:
        ax0.axvline(tau, color="yellow", lw=1.4, ls="-", alpha=0.9)
        ax0.scatter([tau], [actual_tok], color="yellow", s=45, zorder=5)

    ax0.set_xlim(0, display_len)
    ax0.set_ylim(0, proc.n_obs - 1)
    ax0.set_ylabel("Token Bin", fontsize=9.5)
    regime_str = (
        "STOCHASTIC (4 Counterfactual Branches | Impulse Unobserved)"
        if is_br_tau
        else "DETERMINISTIC (Impulse in Context | Physics Deterministic)"
    )
    ax0.set_title(
        f"{data['run_name']} | Horizon k={k} ({data['config_name']}) | Cursor at Target τ={tau} (Source t={t_source}) | Regime: {regime_str} | First {display_len} Tokens Detailed View",
        fontsize=11,
        fontweight="bold",
    )
    ax0.legend(
        loc="upper right",
        fontsize=8,
        ncol=5,
        framealpha=0.75,
        facecolor="#181824",
        edgecolor="#44475a",
        labelcolor="white",
    )

    # Panel 1: Probability slice comparison: Model vs Bayes Optimal
    bins_x = np.arange(proc.n_obs)
    ax1.fill_between(bins_x, 0, slice_probs, color="#bd93f9", alpha=0.25)
    ax1.plot(bins_x, slice_probs, color="#bd93f9", lw=2.0, label="Model P(token | context)")
    ax1.plot(bins_x, bayes_slice, color="#f1fa8c", lw=1.8, ls="--", label="Bayes Optimal P(token | context)")

    for l in range(4):
        tok_l = all_branches[l, tau]
        p_model_l = slice_probs[tok_l]
        if is_br_tau and data.get("beliefs") is not None:
            p_bayes_l = float((data["beliefs"][t_source] @ E)[l])
        else:
            p_bayes_l = 1.0 if l == actual_letter else 0.0

        is_actual = (l == actual_letter)
        if is_br_tau or is_actual:
            lbl = f"Branch {BRANCH_NAMES[l]}: bin {tok_l} (Model: {p_model_l:.2f}, Bayes: {p_bayes_l:.2f})"
            if is_actual:
                lbl += " [ACTUAL]"
            ax1.vlines(
                tok_l,
                0,
                max(p_model_l, p_bayes_l),
                color=BRANCH_COLORS[l],
                lw=2.2 if is_actual else 1.3,
                label=lbl,
            )
            ax1.scatter([tok_l], [p_model_l], color=BRANCH_COLORS[l], s=55 if is_actual else 35, zorder=5)

    curr_branches = all_branches[:, tau] if is_br_tau else np.array([actual_tok])
    x_min = max(0, int(np.min(curr_branches)) - 12)
    x_max = min(proc.n_obs - 1, int(np.max(curr_branches)) + 12)
    ax1.set_xlim(x_min, x_max)
    y_max = max(0.1, float(np.max(slice_probs[x_min : x_max + 1])) * 1.3, float(np.max(bayes_slice)) * 1.1)
    ax1.set_ylim(0, y_max)
    ax1.set_xlabel("Token Bin (Continuous Angle Discretization)", fontsize=9.5)
    ax1.set_ylabel("Probability", fontsize=9.5)

    model_nll_tau = -np.log(max(slice_probs[actual_tok], 1e-12))
    bayes_nll_tau = -np.log(max(bayes_slice[actual_tok], 1e-12))
    gap_tau = model_nll_tau - bayes_nll_tau
    ax1.set_title(
        f"Predictive Distribution at τ={tau}: Model NLL={model_nll_tau:.3f} | Bayes Optimal NLL={bayes_nll_tau:.3f} | Suboptimality Gap: {gap_tau:+.3f} nats",
        fontsize=10.5,
        fontweight="bold",
    )
    ax1.grid(True, linestyle=":", alpha=0.35)
    ax1.legend(loc="upper right", fontsize=8, ncol=2)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=110)
    return fig


def plot_enhanced_summary(
    config_dir: Path,
    job: dict[str, Any],
    width: int,
    seed: int,
    horizons: list[tuple[int, str]],
    loss_histories: dict[str, list[dict]],
    probe_df: pd.DataFrame,
    proc: MessDriven,
    device: str = "cpu",
    n_layers: int = 4,
    n_heads: int = 1,
    save_path: Path | str | None = None,
    max_display_tokens: int | None = 100,
) -> plt.Figure:
    """Generates comprehensive summary plot with the 3 horizon heatmaps arranged in full-width rows.

    Layout:
    - Rows 0, 1, 2: Horizons k1, kn2, kn stacked vertically, stretched across the breadth (first 100 tokens detailed view).
    - Row 3: Loss curves across horizons (full width).
    - Row 4: Probe R² bars (Belief on col 0, Physics on col 1, Metadata summary on col 2).
    """
    from _worker import LookaheadTransformer, PROBE_MODES

    fig = plt.figure(figsize=(18, 19), constrained_layout=True)
    gs = fig.add_gridspec(5, 3, height_ratios=[2.2, 2.2, 2.2, 1.8, 2.0])

    eval_rng = np.random.default_rng(seed + 9999)
    eval_batch = proc.sample_batch(eval_rng, 1)
    tokens_single = eval_batch["tokens"][0]
    letters_single = eval_batch["letters"][0]
    beliefs_single = eval_batch.get("beliefs", [None])[0]
    all_branches = compute_counterfactual_branches(proc, letters_single)

    display_len = min(proc.seq_len, max_display_tokens) if max_display_tokens is not None else proc.seq_len

    # Rows 0, 1, 2: 3 Stacked Horizon Heatmaps
    axes_horizons = []
    for row_idx, (k, k_suffix) in enumerate(horizons):
        ax = fig.add_subplot(gs[row_idx, :])
        axes_horizons.append(ax)
        ax.set_facecolor("#050508")

        h_data = compute_horizon_branches(
            proc,
            tokens_single,
            letters_single,
            k,
            all_branches=all_branches,
            beliefs=beliefs_single,
        )
        b_tokens = h_data["branch_tokens"]
        is_br = h_data["is_branching"]

        model_id = f"d{width}_{k_suffix}_seed{seed}"
        pt_path = config_dir / f"{model_id}_final.pt"
        json_path = config_dir / f"{model_id}_final.json"

        if pt_path.exists():
            layer_count = n_layers
            if json_path.exists():
                try:
                    meta = json.loads(json_path.read_text())
                    layer_count = meta.get("n_layers", n_layers)
                except Exception:
                    pass

            cfg = ModelConfig(
                vocab_size=proc.n_obs,
                n_ctx=proc.seq_len,
                n_layers=layer_count,
                n_heads=n_heads,
                d_model=width,
                d_mlp=4 * width,
                seed=seed,
            )
            model = LookaheadTransformer(cfg, k=k)
            model.load_state_dict(torch.load(pt_path, map_location=device))
            model.to(device).eval()

            probs = token_probabilities(model, tokens_single, device=device)
            if k < len(tokens_single):
                probs_k = probs[:-k]
                actual_k = tokens_single[k:]
                probs_k_disp = probs_k[: max(0, display_len - k)]
                target_times = np.arange(k, display_len)
                br_mask = is_br[k:display_len]

                norm = mcolors.PowerNorm(gamma=0.45, vmin=0, vmax=max(0.01, float(np.max(probs_k_disp)) * 0.85))
                extent = [k - 0.5, display_len - 0.5, 0, proc.n_obs - 1]

                ax.imshow(
                    probs_k_disp.T,
                    origin="lower",
                    aspect="auto",
                    cmap="magma",
                    extent=extent,
                    norm=norm,
                    interpolation="nearest",
                )

                # Shading for stochastic branching windows
                for j in range(proc.m):
                    t_kick = j * proc.n_steps
                    if t_kick >= display_len:
                        break
                    br_start = max(k, t_kick)
                    br_end = min(display_len, t_kick + k)
                    if br_end > br_start:
                        ax.axvspan(br_start - 0.5, br_end - 0.5, color="#ff79c6", alpha=0.06, lw=0)

                for l in range(4):
                    y_branch = b_tokens[l, k:display_len]
                    ax.plot(
                        target_times,
                        y_branch,
                        color=BRANCH_COLORS[l],
                        lw=1.5,
                        alpha=0.8,
                        label=f"Branch {BRANCH_NAMES[l]}",
                    )
                    if np.any(br_mask):
                        ax.scatter(
                            target_times[br_mask],
                            y_branch[br_mask],
                            color=BRANCH_COLORS[l],
                            s=16,
                            alpha=0.9,
                            zorder=4,
                        )

                ax.plot(
                    target_times,
                    actual_k[: max(0, display_len - k)],
                    color="#00ffff",
                    lw=2.2,
                    alpha=0.95,
                    label="Actual Trajectory",
                )

                model_nll = -np.log(np.maximum(probs_k[np.arange(len(actual_k)), actual_k], 1e-12)).mean()
                bayes_nll = h_data["mean_bayes_nll"]
                gap = model_nll - bayes_nll
                pct_br = h_data["pct_branching"]
                regime_desc = (
                    "1-step kick rollouts (remainder deterministic)"
                    if k == 1
                    else (
                        f"First {k} steps guess, remainder deterministic"
                        if pct_br < 95
                        else "Full cycle stochastic branching"
                    )
                )

                ax.set_title(
                    f"Horizon {k_suffix} (k={k}) | Model NLL: {model_nll:.3f} | Bayes Optimal NLL: {bayes_nll:.3f} | "
                    f"Gap: {gap:+.3f} nats | Branching: {pct_br:.1f}% [{regime_desc}] | First {display_len} Tokens Detailed View",
                    fontsize=10.5,
                    fontweight="bold",
                    pad=6,
                )
            else:
                ax.text(0.5, 0.5, f"k={k} >= seq_len", color="white", ha="center", va="center")
        else:
            ax.text(0.5, 0.5, f"Missing model checkpoint: {pt_path.name}", color="white", ha="center", va="center")

        for tick_idx, tick_pos in enumerate(np.arange(proc.m) * proc.n_steps):
            if tick_pos >= display_len:
                break
            ax.axvline(tick_pos, color="white", lw=0.7, ls="--", alpha=0.35)
            if tick_pos >= k:
                ax.text(
                    tick_pos + 0.3,
                    proc.n_obs - 6,
                    f"L{letters_single[tick_idx]}",
                    color="yellow",
                    fontsize=8.5,
                    alpha=0.85,
                    va="top",
                )

        ax.set_xlim(0, display_len)
        ax.set_ylim(0, proc.n_obs - 1)
        ax.set_ylabel("Token Bin", fontsize=9.5)
        if row_idx == 0:
            ax.legend(
                loc="upper right",
                fontsize=8,
                ncol=5,
                framealpha=0.75,
                facecolor="#181824",
                edgecolor="#44475a",
                labelcolor="white",
            )

    axes_horizons[-1].set_xlabel("Target Time Step τ (Lookahead Target)", fontsize=10.5)

    # Row 3: Loss curves across horizons (colspan=3)
    ax_loss = fig.add_subplot(gs[3, :])
    colors = {"k1": "#2ecc71", "kn2": "#3498db", "kn": "#9b59b6"}
    for k, k_suffix in horizons:
        hist = loss_histories.get(k_suffix, [])
        if hist:
            steps = [h["step"] for h in hist]
            eval_losses = [h["eval_loss"] for h in hist]
            ax_loss.plot(
                steps,
                eval_losses,
                label=f"Eval Loss {k_suffix} (k={k})",
                color=colors.get(k_suffix, "black"),
                lw=1.8,
            )
    ax_loss.axhline(np.log(proc.n_obs), ls=":", color="gray", lw=1.5, label=f"Random ln({proc.n_obs})")
    ax_loss.set_title(f"Loss Curves — {job['config_name']} (d={width}, seed={seed})", fontweight="bold")
    ax_loss.set_xlabel("Training Steps")
    ax_loss.set_ylabel("Cross Entropy Loss")
    ax_loss.legend(loc="upper right")
    ax_loss.grid(True, alpha=0.3)

    # Row 4: Probe R² bars (Belief on col 0, Physics on col 1, Info on col 2)
    for col_idx, target_name in enumerate(["belief", "physics"]):
        ax_probe = fig.add_subplot(gs[4, col_idx])
        sub_df = probe_df[probe_df["target"] == target_name]
        modes = PROBE_MODES
        x = np.arange(len(modes))
        bar_w = 0.35

        trained_r2 = []
        random_r2 = []
        for m_name in modes:
            m_trained = sub_df[(sub_df["mode"] == m_name) & (sub_df["model_type"] == "trained")]["test_r2"]
            m_random = sub_df[(sub_df["mode"] == m_name) & (sub_df["model_type"] == "random_init")]["test_r2"]
            trained_r2.append(float(m_trained.max()) if len(m_trained) > 0 else 0.0)
            random_r2.append(float(m_random.max()) if len(m_random) > 0 else 0.0)

        ax_probe.bar(x - bar_w / 2, trained_r2, bar_w, label="Trained (best k)", color="#2980b9")
        ax_probe.bar(x + bar_w / 2, random_r2, bar_w, label="Random init", color="#e74c3c")
        ax_probe.set_xticks(x)
        ax_probe.set_xticklabels([m.replace("_", "\n") for m in modes], rotation=45, ha="right", fontsize=8)
        ax_probe.set_ylabel("Test R²")
        ax_probe.set_ylim(-0.1, 1.05)
        ax_probe.set_title(f"{target_name.capitalize()} Probe R² by Mode", fontweight="bold")
        ax_probe.legend(loc="upper left")
        ax_probe.grid(True, axis="y", alpha=0.3)

    # Row 4, Col 2: Info panel
    ax_info = fig.add_subplot(gs[4, 2])
    ax_info.axis("off")
    info_text = (
        f"Config: {job['config_name']}\n"
        f"delta_v: {job['delta_v']}\n"
        f"gamma: {job['gamma']}\n"
        f"dt: {job['dt']}\n"
        f"n_steps (n): {job['n_steps']}\n"
        f"m (cycles): {job['m']}\n"
        f"seq_len: {job['m'] * job['n_steps']}\n"
        f"d_model: {width}\n"
        f"seed: {seed}\n"
        f"Total probe rows: {len(probe_df)}"
    )
    ax_info.text(
        0.1,
        0.5,
        info_text,
        fontsize=11,
        family="monospace",
        va="center",
        bbox=dict(boxstyle="round,pad=0.8", facecolor="#f8f9fa", edgecolor="#ced4da"),
    )

    out_path = Path(save_path) if save_path else (config_dir / f"d{width}_seed{seed}_summary.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    return fig


def launch_interactive_dashboard(
    trained_runs: dict[str, dict[str, Any]],
    batch: dict[str, np.ndarray],
    device: str = "cpu",
):
    """Launches interactive UI with Configuration Dropdown + synchronized Slider."""
    import ipywidgets as widgets
    from IPython.display import display

    unique_runs = {}
    for chosen in trained_runs.values():
        name = chosen["run_name"]
        if name not in unique_runs:
            unique_runs[name] = chosen

    print("Precomputing lookahead branch data for available runs...")
    cached_data = {
        name: prepare_run_data(run, batch, device)
        for name, run in unique_runs.items()
    }

    run_names = sorted(list(unique_runs.keys()), key=lambda name: cached_data[name]["k"])
    default_run = run_names[-1]

    run_dropdown = widgets.Dropdown(
        options=run_names,
        value=default_run,
        description="Model Config:",
        layout=widgets.Layout(width="400px"),
    )

    init_data = cached_data[default_run]
    tau_slider = widgets.IntSlider(
        value=max(init_data["k"], 25),
        min=init_data["k"],
        max=init_data["seq_len"] - 1,
        step=1,
        description="Target Time τ:",
        continuous_update=True,
        layout=widgets.Layout(width="650px"),
    )

    def on_run_change(change):
        selected_data = cached_data[change["new"]]
        new_k = selected_data["k"]
        tau_slider.min = new_k
        if tau_slider.value < new_k:
            tau_slider.value = new_k

    run_dropdown.observe(on_run_change, names="value")

    def dashboard(config_name, tau):
        selected_data = cached_data[config_name]
        plot_lookahead_branching(selected_data, tau)

    out = widgets.interactive_output(dashboard, {"config_name": run_dropdown, "tau": tau_slider})
    display(widgets.VBox([run_dropdown, tau_slider, out]))
