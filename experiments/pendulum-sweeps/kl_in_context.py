"""In-Context KL-Divergence and k-HMM Crossover Analysis.

Implements the experimental protocol adapted from Poncin et al. (2026),
"Large Language Models Develop Belief State Geometry In-Context" (arXiv:2609.17376v1):
1. Per-position KL divergence at usable tokens against 0-HMM and 1-HMM baselines (linear & log scale).
2. Decomposition of model KL into in-subspace belief error vs off-branch physics mass leakage.
3. Average converged KL vs k-HMM Markov order k and identification of crossover k*.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import torch

from lookahead_viz import (
    BRANCH_COLORS,
    BRANCH_NAMES,
    compute_counterfactual_branches,
    compute_horizon_branches,
)
from models.transformer import ModelConfig, TinyTransformer
from physics.messk import MessDriven, MessKProcess
from physics.systems.pendulum import Pendulum


class SampledPhysicsWrapper:
    """Wraps pendulum flow to execute fixed-step substeps for numerical precision."""

    def __init__(self, base: Pendulum, integration_dt: float = 0.01):
        self.base = base
        self.integration_dt = integration_dt

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base, name)

    def flow(self, z: np.ndarray, dt: float, substeps: int = 1) -> np.ndarray:
        internal_steps = round(dt / self.integration_dt)
        return self.base.flow(z, dt, substeps=internal_steps * substeps)


def load_run(
    config_name: str,
    output_dir: Path | str,
    width: int = 128,
    seed: int = 0,
    horizons: Sequence[str] = ("k1", "kn2", "kn"),
    m_eval: int | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    """Loads trained horizon models and constructs matching MessDriven process."""
    c_dir = Path(output_dir) / config_name
    meta_files = list(c_dir.glob("*.json"))
    if not meta_files:
        raise FileNotFoundError(f"No sidecar metadata JSON found in {c_dir}")

    meta = json.loads(meta_files[0].read_text())
    dt = float(meta["dt"])
    gamma = float(meta["gamma"])
    delta_v = float(meta["delta_v"])
    n_steps = int(meta["n_steps"])
    m_train = int(meta["m"])
    m_run = m_eval if m_eval is not None else m_train

    pendulum = Pendulum(gamma=gamma)
    sampled = SampledPhysicsWrapper(base=pendulum, integration_dt=0.01)
    chain = MessKProcess(n_states=4, alpha=0.7, stay=0.7)
    proc = MessDriven(
        chain=chain,
        system=sampled,
        delta_v=delta_v,
        m=m_run,
        n_steps=n_steps,
        dt=dt,
        obs_bins=int(meta.get("vocab_size", 181)),
        random_init=bool(meta.get("random_init", True)),
        init_theta_std=float(meta.get("init_theta_std", 0.3)),
    )

    models = {}
    for h_suffix in horizons:
        pt_path = c_dir / f"d{width}_{h_suffix}_seed{seed}_final.pt"
        json_path = c_dir / f"d{width}_{h_suffix}_seed{seed}_final.json"
        if not pt_path.exists() or not json_path.exists():
            raise FileNotFoundError(f"Checkpoint {pt_path} or sidecar missing")

        h_meta = json.loads(json_path.read_text())
        k_val = int(h_meta["k"])
        n_ctx = int(h_meta["seq_len"])
        if proc.seq_len > n_ctx:
            raise ValueError(
                f"Requested evaluation seq_len ({proc.seq_len}) exceeds model n_ctx ({n_ctx})"
            )

        model_cfg = ModelConfig(
            vocab_size=int(h_meta["vocab_size"]),
            n_ctx=n_ctx,
            n_layers=int(h_meta["n_layers"]),
            n_heads=int(h_meta.get("n_heads", 1)),
            d_model=width,
            d_mlp=int(h_meta.get("d_mlp", 4 * width)),
            seed=seed,
        )
        model = TinyTransformer(model_cfg)
        state_dict = torch.load(pt_path, map_location=device)
        model.load_state_dict(state_dict)
        model.k = k_val
        model.k_suffix = h_suffix
        model.to(device).eval()
        models[h_suffix] = model

    return {
        "proc": proc,
        "models": models,
        "config_name": config_name,
        "width": width,
        "seed": seed,
        "m_eval": m_run,
        "n_steps": n_steps,
    }


def sample_eval_batch(
    proc: MessDriven,
    n_seq: int = 1000,
    seed: int = 20260830,
) -> dict[str, np.ndarray]:
    """Generates evaluation trajectories with randomized initial conditions."""
    rng = np.random.default_rng(seed)
    return proc.sample_batch(rng, n_seq)


def batched_branch_tokens(
    proc: MessDriven,
    batch: dict[str, np.ndarray],
    chunk_size: int = 256,
) -> np.ndarray:
    """Precomputes discrete tokens for all 4 impulse letters at all ticks and phases.

    Returns:
        branch_tokens: Array of shape (N, m, 4, n_steps) with discrete token IDs.
    """
    letters = batch["letters"]  # (N, m)
    initial_state = batch.get("initial_state")  # (N, 2)
    n_seq = len(letters)
    m = proc.m
    n_steps = proc.n_steps
    actions = proc.actions  # (4, 1)

    out = np.empty((n_seq, m, 4, n_steps), dtype=np.int64)

    for start in range(0, n_seq, chunk_size):
        end = min(start + chunk_size, n_seq)
        b_size = end - start
        b_letters = letters[start:end]

        if initial_state is not None:
            z = initial_state[start:end].copy()
        else:
            z = proc.system.initial_state(b_size)

        z_pre = np.empty((b_size, m, 2), dtype=np.float64)
        for t in range(m):
            z_pre[:, t] = z
            z = proc.system.kick(z, actions[b_letters[:, t]])
            for s in range(n_steps):
                z = proc.system.flow(z, proc.dt)

        for t in range(m):
            z_cand = np.repeat(z_pre[:, t], 4, axis=0)  # (b_size * 4, 2)
            acts = np.tile(actions, (b_size, 1))  # (b_size * 4, 1)
            z_kicked = proc.system.kick(z_cand, acts)

            step_obs = np.empty((b_size * 4, n_steps, len(proc.obs_bins)), dtype=np.float64)
            for s in range(n_steps):
                z_kicked = proc.system.flow(z_kicked, proc.dt)
                step_obs[:, s] = proc.channels(z_kicked)

            tokens_t = proc.discretise(step_obs).reshape(b_size, 4, n_steps)
            out[start:end, t] = tokens_t

    return out


def true_letter_probs(proc: MessDriven, batch: dict[str, np.ndarray]) -> np.ndarray:
    """Computes exact ground-truth predictive letter probability at each tick.

    Returns:
        p_true: Array of shape (N, m, 4) where p_true[i, t, l] = P(letter_{t} = l | letters_{0..t-1}).
    """
    beliefs = batch["beliefs"][:, :: proc.n_steps, :]  # (N, m, 4)
    E = proc.chain.E  # (4, 4)
    return np.einsum("nti,ij->ntj", beliefs, E)


def khmm_letter_probs(
    proc: MessDriven,
    letters: np.ndarray,
    k: int,
) -> np.ndarray:
    """Computes analytic k-HMM letter probabilities using Markov order-k approximation.

    Args:
        proc: MessDriven process containing chain transition operators.
        letters: Array (N, m) of emitted integer letters.
        k: Markov order approximation (k=0 is stationary marginal; k>=1 uses last k letters).

    Returns:
        q_k: Array of shape (N, m, 4) with next-letter probabilities under k-HMM.
    """
    n_seq, m = letters.shape
    n_states = proc.chain.n_states
    E = proc.chain.E
    joint = proc.chain.joint

    out = np.empty((n_seq, m, n_states), dtype=np.float64)
    stationary = np.full(n_states, 1.0 / n_states)

    if k == 0:
        stat_prob = stationary @ E
        out[:] = stat_prob[None, None, :]
        return out

    for i in range(n_seq):
        seq_letters = letters[i]
        b = stationary.copy()
        for t in range(m):
            # Advance belief with emitted letter at tick t
            b = b @ joint[seq_letters[t]]
            tot = b.sum()
            b = b / tot if tot > 0 else stationary

            if t + 1 <= k:
                # When k covers all observed letters 0..t, equals exact belief
                b_pred = b
            else:
                # Recalculate belief from last k tokens only: letters[t - k + 1 : t + 1]
                w_letters = seq_letters[t - k + 1 : t + 1]
                b_curr = stationary.copy()
                for l in w_letters:
                    b_curr = b_curr @ joint[l]
                    tot_w = b_curr.sum()
                    b_curr = b_curr / tot_w if tot_w > 0 else stationary
                b_pred = b_curr

            out[i, t] = b_pred @ E

    return out



def compute_collision_merged_kl(
    p_letters: np.ndarray,
    q_target: np.ndarray,
    branch_tokens: np.ndarray,
    is_model: bool = False,
    eps: float = 1e-12,
) -> tuple[float, float, float]:
    """Computes branch-collision-merged KL divergence and physics leakage decomposition.

    Args:
        p_letters: 1D array of 4 ground-truth letter probabilities.
        q_target: Either 1D array of 4 letter probabilities (k-HMM), or full vocabulary
                  softmax distribution (model).
        branch_tokens: 1D array of 4 discrete token IDs produced by the 4 impulse letters.
        is_model: If True, treats q_target as full vocab distribution and calculates leakage Z.
        eps: Small floor to prevent numerical zero in log.

    Returns:
        (total_kl, belief_kl, leakage)
    """
    # Identify unique branch tokens and group letters that collide
    unique_toks, inverse_idx = np.unique(branch_tokens, return_inverse=True)
    n_groups = len(unique_toks)

    p_merged = np.zeros(n_groups, dtype=np.float64)
    for l_idx, g_idx in enumerate(inverse_idx):
        p_merged[g_idx] += p_letters[l_idx]

    if is_model:
        q_merged = np.array([q_target[tok] for tok in unique_toks], dtype=np.float64)
        z = float(np.sum(q_merged))
        z_safe = max(z, eps)
        q_norm = q_merged / z_safe

        # In-subspace belief KL
        belief_kl = float(
            np.sum(p_merged * np.log(np.maximum(p_merged, eps) / np.maximum(q_norm, eps)))
        )
        leakage = float(-np.log(z_safe))
        total_kl = belief_kl + leakage
    else:
        q_merged = np.zeros(n_groups, dtype=np.float64)
        for l_idx, g_idx in enumerate(inverse_idx):
            q_merged[g_idx] += q_target[l_idx]

        total_kl = float(
            np.sum(p_merged * np.log(np.maximum(p_merged, eps) / np.maximum(q_merged, eps)))
        )
        belief_kl = total_kl
        leakage = 0.0

    return total_kl, belief_kl, leakage


def run_kl_experiment(
    run_data: dict[str, Any],
    n_seq: int = 1000,
    ks: Sequence[int] = (0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 16),
    burn_in_ticks: int = 0,
    seed: int = 20260830,
    device: str = "cpu",
    batch_size: int = 64,
) -> dict[str, Any]:
    """Runs the complete in-context KL divergence evaluation across usable tokens."""
    proc = run_data["proc"]
    models = run_data["models"]
    m = proc.m
    n_steps = proc.n_steps

    print(f"Sampling {n_seq} evaluation trajectories (m={m}, seq_len={proc.seq_len})...")
    batch = sample_eval_batch(proc, n_seq=n_seq, seed=seed)
    tokens = batch["tokens"]  # (N, seq_len)
    letters = batch["letters"]  # (N, m)

    print("Precomputing batched physical counterfactual branches...")
    branches = batched_branch_tokens(proc, batch)  # (N, m, 4, n_steps)

    print("Precomputing ground-truth letter probabilities...")
    p_true = true_letter_probs(proc, batch)  # (N, m, 4)

    print(f"Precomputing k-HMM letter probabilities for orders {ks}...")
    q_khmm = {k_order: khmm_letter_probs(proc, letters, k_order) for k_order in ks}

    results = {
        "config_name": run_data["config_name"],
        "width": run_data["width"],
        "seed": run_data["seed"],
        "m_eval": m,
        "n_steps": n_steps,
        "ks": list(ks),
        "horizons": {},
    }

    for h_suffix, model in models.items():
        k_val = model.k
        usable = proc.usable_positions(k_val)
        # Filter usable positions past burn-in
        if burn_in_ticks > 0:
            usable = usable[usable >= burn_in_ticks * n_steps]

        n_usable = len(usable)
        print(f"\nEvaluating Horizon {h_suffix} (k={k_val}) across {n_usable} usable positions...")

        # Model softmax predictions over all usable positions
        model_kl_all = np.empty((n_seq, n_usable), dtype=np.float64)
        model_belief_kl_all = np.empty((n_seq, n_usable), dtype=np.float64)
        model_leakage_all = np.empty((n_seq, n_usable), dtype=np.float64)
        khmm_kl_all = {k_order: np.empty((n_seq, n_usable), dtype=np.float64) for k_order in ks}

        # Chunked model forward passes
        for b_start in range(0, n_seq, batch_size):
            b_end = min(b_start + batch_size, n_seq)
            tok_tensor = torch.as_tensor(tokens[b_start:b_end], dtype=torch.long, device=device)
            with torch.no_grad():
                logits = model(tok_tensor)  # (B, seq_len, vocab)
                probs = torch.softmax(logits, dim=-1).cpu().numpy()

            for seq_idx in range(b_start, b_end):
                sub_idx = seq_idx - b_start
                for pos_idx, p in enumerate(usable):
                    t_source = p // n_steps
                    tau = p + k_val
                    t_target = tau // n_steps
                    s_target = tau % n_steps

                    c_branches = branches[seq_idx, t_target, :, s_target]
                    p_curr = p_true[seq_idx, t_source]
                    q_mod = probs[sub_idx, p]

                    b0, b1, b2, b3 = c_branches
                    if b0 != b1 and b0 != b2 and b0 != b3 and b1 != b2 and b1 != b3 and b2 != b3:
                        # Fast path: all 4 branches map to distinct tokens (no collision)
                        q_mod_b = q_mod[c_branches]
                        z = float(np.sum(q_mod_b))
                        z_safe = max(z, 1e-12)
                        lkg = -float(np.log(z_safe))
                        bel_kl = float(
                            np.sum(p_curr * np.log(np.maximum(p_curr, 1e-12) / np.maximum(q_mod_b / z_safe, 1e-12)))
                        )
                        tot_kl = bel_kl + lkg

                        model_kl_all[seq_idx, pos_idx] = tot_kl
                        model_belief_kl_all[seq_idx, pos_idx] = bel_kl
                        model_leakage_all[seq_idx, pos_idx] = lkg

                        for k_order in ks:
                            q_k_letter = q_khmm[k_order][seq_idx, t_source]
                            k_kl = float(
                                np.sum(p_curr * np.log(np.maximum(p_curr, 1e-12) / np.maximum(q_k_letter, 1e-12)))
                            )
                            khmm_kl_all[k_order][seq_idx, pos_idx] = k_kl
                    else:
                        tot_kl, bel_kl, lkg = compute_collision_merged_kl(
                            p_curr, q_mod, c_branches, is_model=True
                        )
                        model_kl_all[seq_idx, pos_idx] = tot_kl
                        model_belief_kl_all[seq_idx, pos_idx] = bel_kl
                        model_leakage_all[seq_idx, pos_idx] = lkg

                        for k_order in ks:
                            q_k_letter = q_khmm[k_order][seq_idx, t_source]
                            k_kl, _, _ = compute_collision_merged_kl(
                                p_curr, q_k_letter, c_branches, is_model=False
                            )
                            khmm_kl_all[k_order][seq_idx, pos_idx] = k_kl


        # Compute summary statistics across sequences
        results["horizons"][h_suffix] = {
            "k_val": k_val,
            "usable_positions": usable,
            "model_kl_mean": np.mean(model_kl_all, axis=0),
            "model_kl_sem": stats.sem(model_kl_all, axis=0),
            "model_belief_kl_mean": np.mean(model_belief_kl_all, axis=0),
            "model_leakage_mean": np.mean(model_leakage_all, axis=0),
            "khmm_kl_mean": {k_order: np.mean(khmm_kl_all[k_order], axis=0) for k_order in ks},
            "khmm_kl_sem": {k_order: stats.sem(khmm_kl_all[k_order], axis=0) for k_order in ks},
            # Converged KL (mean over usable positions in the second half of sequence)
            "converged_model_kl": float(np.mean(model_kl_all[:, n_usable // 2 :])),
            "converged_model_belief_kl": float(np.mean(model_belief_kl_all[:, n_usable // 2 :])),
            "converged_model_leakage": float(np.mean(model_leakage_all[:, n_usable // 2 :])),
            "converged_khmm_kl": {
                k_order: float(np.mean(khmm_kl_all[k_order][:, n_usable // 2 :])) for k_order in ks
            },
        }

    return results


def save_kl_results(results: dict[str, Any], path: Path | str) -> None:
    """Saves evaluation results dictionary to disk."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(results, p)


def load_kl_results(path: Path | str) -> dict[str, Any]:
    """Loads evaluation results dictionary from disk."""
    return torch.load(Path(path), weights_only=False)


def find_crossover_k(ks: Sequence[int], khmm_kls: Sequence[float], model_kl: float) -> float:
    """Finds exact linear interpolation crossover where k-HMM KL matches model KL."""
    for i in range(len(ks) - 1):
        y0, y1 = khmm_kls[i], khmm_kls[i + 1]
        if (y0 >= model_kl >= y1) or (y0 <= model_kl <= y1):
            if abs(y1 - y0) < 1e-12:
                return float(ks[i])
            fraction = (y0 - model_kl) / (y0 - y1)
            return float(ks[i] + fraction * (ks[i + 1] - ks[i]))
    return float(ks[-1]) if khmm_kls[-1] <= model_kl else float(ks[0])


def plot_kl_vs_position(
    results: dict[str, Any],
    yscale: str = "both",
    decompose: bool = True,
    min_pos: int = 0,
    save_path: Path | str | None = None,
) -> plt.Figure:
    """Generates Plot 1: In-context KL divergence vs token position."""
    horizons = results["horizons"]
    n_rows = len(horizons)
    n_cols = 2 if yscale == "both" or decompose else 1

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7.5 * n_cols, 3.8 * n_rows), squeeze=False)

    for row_idx, (h_suffix, h_data) in enumerate(horizons.items()):
        raw_pos = h_data["usable_positions"]
        k_val = h_data["k_val"]
        mask = raw_pos >= min_pos
        pos = raw_pos[mask]

        # Left panel: Primary KL plot (Linear scale or Log scale)
        ax = axes[row_idx, 0]
        m_mean = h_data["model_kl_mean"][mask]
        m_sem = h_data["model_kl_sem"][mask]

        ax.plot(pos, m_mean, color="#2b5c8f", lw=2.0, label=f"Model ({h_suffix})")
        ax.fill_between(pos, m_mean - 1.96 * m_sem, m_mean + 1.96 * m_sem, color="#2b5c8f", alpha=0.2)

        # 0-HMM baseline
        if 0 in h_data["khmm_kl_mean"]:
            ax.plot(pos, h_data["khmm_kl_mean"][0][mask], color="#e74c3c", lw=1.6, ls="--", label="0-HMM (marginal)")
        # 1-HMM baseline
        if 1 in h_data["khmm_kl_mean"]:
            ax.plot(pos, h_data["khmm_kl_mean"][1][mask], color="#e67e22", lw=1.6, ls=":", label="1-HMM")

        ax.set_ylabel(r"$D_{\mathrm{KL}}(P_{\mathrm{true}} \parallel Q)$ [nats]")
        start_note = f" (from pos {min_pos})" if min_pos > 0 else ""
        ax.set_title(f"Horizon {h_suffix} (k={k_val}) — Prediction KL vs Position{start_note}")
        ax.grid(True, alpha=0.3, ls=":")
        ax.legend(loc="upper right", framealpha=0.9, fontsize=9)
        if yscale == "log":
            ax.set_yscale("log")

        # Right panel: Log scale or Decomposition panel
        if n_cols > 1:
            ax_right = axes[row_idx, 1]
            if decompose:
                bel_mean = h_data["model_belief_kl_mean"][mask]
                leak_mean = h_data["model_leakage_mean"][mask]
                ax_right.plot(pos, m_mean, color="#2b5c8f", lw=2.0, label="Total Model KL")
                ax_right.plot(pos, bel_mean, color="#27ae60", lw=1.8, ls="-.", label="In-Subspace Belief KL")
                ax_right.plot(pos, leak_mean, color="#8e44ad", lw=1.8, ls=":", label="Physics Leakage (-ln Z)")
                ax_right.set_ylabel("KL Component [nats]")
                ax_right.set_title(f"Horizon {h_suffix} — Belief vs Physics Leakage Decomposition{start_note}")
                ax_right.grid(True, alpha=0.3, ls=":")
                ax_right.legend(loc="upper right", framealpha=0.9, fontsize=9)
                if yscale == "log":
                    ax_right.set_yscale("log")
            else:
                ax_right.plot(pos, m_mean, color="#2b5c8f", lw=2.0, label=f"Model ({h_suffix})")
                if 0 in h_data["khmm_kl_mean"]:
                    ax_right.plot(pos, h_data["khmm_kl_mean"][0][mask], color="#e74c3c", lw=1.6, ls="--", label="0-HMM")
                if 1 in h_data["khmm_kl_mean"]:
                    ax_right.plot(pos, h_data["khmm_kl_mean"][1][mask], color="#e67e22", lw=1.6, ls=":", label="1-HMM")
                ax_right.set_yscale("log")
                ax_right.set_ylabel(r"$D_{\mathrm{KL}}$ (log scale)")
                ax_right.set_title(f"Horizon {h_suffix} — Log Scale{start_note}")
                ax_right.grid(True, alpha=0.3, ls=":")
                ax_right.legend(loc="upper right", framealpha=0.9, fontsize=9)

        axes[row_idx, 0].set_xlabel("Token Position (Usable)")
        if n_cols > 1:
            axes[row_idx, 1].set_xlabel("Token Position (Usable)")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_kl_vs_k(
    results: dict[str, Any],
    burn_in_tokens: int = 100,
    save_path: Path | str | None = None,
) -> plt.Figure:
    """Generates Plot 2: Converged KL vs k-HMM order and model crossover order k*."""
    horizons = results["horizons"]
    ks = results["ks"]
    n_h = len(horizons)

    fig, axes = plt.subplots(1, n_h, figsize=(5.0 * n_h, 4.2), squeeze=False)

    for idx, (h_suffix, h_data) in enumerate(horizons.items()):
        ax = axes[0, idx]
        k_val = h_data["k_val"]
        raw_pos = h_data["usable_positions"]
        mask = raw_pos >= burn_in_tokens

        if np.any(mask):
            tot_kl = float(np.mean(h_data["model_kl_mean"][mask]))
            bel_kl = float(np.mean(h_data["model_belief_kl_mean"][mask]))
            khmm_vals = [float(np.mean(h_data["khmm_kl_mean"][k_order][mask])) for k_order in ks]
        else:
            tot_kl = h_data["converged_model_kl"]
            bel_kl = h_data.get("converged_model_belief_kl", tot_kl)
            khmm_vals = [h_data["converged_khmm_kl"][k_order] for k_order in ks]

        k_star_tot = find_crossover_k(ks, khmm_vals, tot_kl)
        k_star_bel = find_crossover_k(ks, khmm_vals, bel_kl)

        # Plot k-HMM curve
        ax.plot(ks, khmm_vals, marker="o", color="#e67e22", lw=2.0, label="k-HMM Predictors")

        # Plot model total horizontal reference
        ax.axhline(tot_kl, color="#2b5c8f", lw=1.8, ls="--", label=f"Model Total = {tot_kl:.4f}")

        # Plot in-subspace belief horizontal reference
        ax.axhline(bel_kl, color="#27ae60", lw=1.8, ls="-.", label=f"Model Belief = {bel_kl:.4f}")

        # Highlight crossover for belief
        ax.axvline(k_star_bel, color="#27ae60", lw=1.2, ls=":", label=f"Belief k* = {k_star_bel:.2f}")
        ax.scatter([k_star_bel], [bel_kl], color="#27ae60", s=50, zorder=5)

        ax.set_xlabel("Markov Order k")
        ax.set_ylabel(r"Converged $D_{\mathrm{KL}}$ [nats]")
        ax.set_title(
            f"Horizon {h_suffix} (k={k_val}, pos≥{burn_in_tokens})\nBelief $k^* = {k_star_bel:.2f}$ | Total $k^* = {k_star_tot:.2f}$"
        )
        ax.grid(True, alpha=0.3, ls=":")
        ax.legend(loc="upper right", framealpha=0.9, fontsize=8.0)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_lookahead_probabilities(
    run_data: dict[str, Any],
    window_tokens: int = 100,
    traj_idx: int = 0,
    seed: int = 42,
    device: str = "cpu",
    save_path: Path | str | None = None,
) -> plt.Figure:
    """Plots full model lookahead predictive distributions across 181 token bins.

    Matches _worker.py and lookahead_viz.py style:
    - 3 stacked rows for horizons (k=1, k=n//2, k=n).
    - Heatmap of full model predictive distribution over all 181 token bins (magma colormap).
    - Overlaid 4 counterfactual branches (red, orange, green, blue).
    - Overlaid actual trajectory (cyan).
    - Kick boundaries with letter labels L0..L3.
    - Stochastic branching window shading (#ff79c6).
    - Shows the LAST window_tokens (default 100) of the sequence.
    """
    proc = run_data["proc"]
    models = run_data["models"]
    seq_len = proc.seq_len
    n_obs = proc.n_obs
    n_steps = proc.n_steps

    rng = np.random.default_rng(seed)
    batch = proc.sample_batch(rng, max(1, traj_idx + 1))
    tokens = batch["tokens"][traj_idx]
    letters = batch["letters"][traj_idx]
    beliefs = batch.get("beliefs")
    beliefs_traj = beliefs[traj_idx] if beliefs is not None else None
    init_state = batch.get("initial_state")[traj_idx] if batch.get("initial_state") is not None else None

    all_branches = compute_counterfactual_branches(proc, letters, initial_state=init_state)

    start = max(0, seq_len - window_tokens)
    end = seq_len
    target_times = np.arange(start, end)
    n_win = end - start

    fig, axes = plt.subplots(len(models), 1, figsize=(18, 4.0 * len(models)), sharex=True, dpi=120)
    if len(models) == 1:
        axes = [axes]

    for idx, (k_suffix, model) in enumerate(models.items()):
        ax = axes[idx]
        k = model.k
        ax.set_facecolor("#050508")

        h_data = compute_horizon_branches(
            proc,
            tokens,
            letters,
            k,
            all_branches=all_branches,
            beliefs=beliefs_traj,
            initial_state=init_state,
        )
        b_tokens = h_data["branch_tokens"]
        is_br = h_data["is_branching"]

        with torch.no_grad():
            tok_t = torch.as_tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
            logits = model(tok_t)[0]
            probs = torch.softmax(logits, dim=-1).cpu().numpy()

        # Target tau is predicted from source p = tau - k
        # For tau in [start, end), source p in [start - k, end - k)
        probs_window = probs[start - k : end - k]
        actual_window = tokens[start:end]

        model_nll = -np.log(np.maximum(probs_window[np.arange(n_win), actual_window], 1e-12)).mean()
        bayes_nll = h_data["bayes_nll"][start:end].mean() if len(h_data["bayes_nll"]) >= end else 0.0
        gap = model_nll - bayes_nll
        pct_br = float(is_br[start:end].mean() * 100)

        regime_desc = (
            "1-step kick rollouts (remainder deterministic)"
            if k == 1
            else (
                f"First {k} steps guess, remainder deterministic"
                if pct_br < 95
                else "Full cycle stochastic branching"
            )
        )

        vmax = max(0.01, float(np.max(probs_window)) * 0.85)
        norm = mcolors.PowerNorm(gamma=0.45, vmin=0, vmax=vmax)
        extent = [start - 0.5, end - 0.5, 0, n_obs - 1]

        ax.imshow(
            probs_window.T,
            origin="lower",
            aspect="auto",
            cmap="magma",
            extent=extent,
            norm=norm,
            interpolation="nearest",
        )

        # Stochastic branching tint
        for j in range(proc.m):
            t_kick = j * n_steps
            br_s = max(start, t_kick)
            br_e = min(end, t_kick + k)
            if br_e > br_s:
                ax.axvspan(br_s - 0.5, br_e - 0.5, color="#ff79c6", alpha=0.06, lw=0)

        # Counterfactual branches
        br_mask = is_br[start:end]
        for l in range(4):
            y_branch = b_tokens[l, start:end]
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

        # Actual trajectory
        ax.plot(
            target_times,
            actual_window,
            color="#00ffff",
            lw=2.2,
            alpha=0.95,
            label="Actual Trajectory",
        )

        # Kick vertical lines and labels
        for j in range(proc.m):
            t_kick = j * n_steps
            if start <= t_kick < end:
                ax.axvline(t_kick, color="white", lw=0.7, ls="--", alpha=0.35)
                ax.text(
                    t_kick + 0.3,
                    n_obs - 6,
                    f"L{letters[j]}",
                    color="yellow",
                    fontsize=8.5,
                    alpha=0.85,
                    va="top",
                    ha="left",
                )

        ax.set_xlim(start - 0.5, end - 0.5)
        ax.set_ylim(0, n_obs - 1)
        ax.set_ylabel("Token Bin", fontsize=9.5)
        ax.set_title(
            f"Horizon {k_suffix} (k={k}) | Model NLL: {model_nll:.3f} | Bayes Optimal NLL: {bayes_nll:.3f} | "
            f"Gap: {gap:+.3f} nats | Branching: {pct_br:.1f}% [{regime_desc}] | Last {window_tokens} Tokens Detailed View",
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
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=120)
    return fig
