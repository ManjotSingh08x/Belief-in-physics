from __future__ import annotations

import json
from pathlib import Path
import sys
from dataclasses import dataclass

EXPERIMENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = EXPERIMENT_DIR.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from IPython.display import HTML, clear_output, display
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from models.analysis import residual_streams_batched, tick_features
from models.transformer import ModelConfig, TinyTransformer
from physics.messk import MessDriven, MessKProcess, simplex_embedding
from physics.systems.pendulum import Pendulum

OUTPUT_DIR = EXPERIMENT_DIR / "outputs"
ANALYSIS_CSV = OUTPUT_DIR / "analysis.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_CACHED_DF: pd.DataFrame | None = None


class LookaheadTransformer(TinyTransformer):
    def __init__(self, config: ModelConfig, k: int):
        super().__init__(config)
        self.k = k

    def loss(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.k < 1 or self.k >= tokens.shape[1]:
            raise ValueError(f"lookahead k must satisfy 1 <= k < {tokens.shape[1]}")
        logits = self(tokens)
        return F.cross_entropy(
            logits[:, :-self.k].reshape(-1, logits.shape[-1]),
            tokens[:, self.k:].reshape(-1),
        )


@dataclass(frozen=True)
class SampledPhysicsSystem:
    base: object
    integration_dt: float

    def __getattr__(self, name: str):
        return getattr(self.base, name)

    def flow(self, z: np.ndarray, dt: float, substeps: int = 1) -> np.ndarray:
        internal_steps = round(dt / self.integration_dt)
        if internal_steps < 1 or not np.isclose(dt, internal_steps * self.integration_dt):
            raise ValueError("physics sampling gap dt must be an integer multiple of integration_dt")
        return self.base.flow(z, dt, substeps=internal_steps * substeps)


def build_process_from_meta(exp_meta: dict, hmm_meta: dict | None = None) -> MessDriven:
    phys_meta = exp_meta.get("physics", {})
    pendulum_fields = {"g", "length", "gamma", "omega_max", "theta0", "omega0"}
    sim_kwargs = {k: v for k, v in phys_meta.items() if k in pendulum_fields}
    sim = Pendulum(**sim_kwargs)
    integration_dt = float(phys_meta.get("integration_dt", 0.01))
    sampled = SampledPhysicsSystem(base=sim, integration_dt=integration_dt)

    hmm_d = hmm_meta or exp_meta.get("hmm", {})
    chain = MessKProcess(
        n_states=int(hmm_d.get("n_states", 4)),
        alpha=float(hmm_d.get("alpha", 0.7)),
        stay=float(hmm_d.get("stay", 0.7)),
    )
    return MessDriven(
        chain=chain,
        system=sampled,
        delta_v=float(phys_meta.get("delta_v", 0.55)),
        m=int(exp_meta.get("m", 16)),
        n_steps=int(exp_meta.get("n", 10)),
        dt=float(phys_meta.get("dt", 0.2)),
        obs_bins=phys_meta.get("obs_bins", 181),
    )


def sample_analysis_batch(
    proc: MessDriven,
    rng: np.random.Generator,
    n_trajectories: int,
    init_cfg: dict | None = None,
) -> dict:
    z0 = proc.system.initial_state(n_trajectories).astype(np.float64, copy=True)
    if init_cfg and init_cfg.get("mode") == "random":
        jitter = float(init_cfg.get("jitter_std", 0.1))
        z0 += rng.normal(0.0, jitter, size=z0.shape)
        if hasattr(proc.system, "omega_max"):
            z0[:, 1] = np.clip(z0[:, 1], -proc.system.omega_max, proc.system.omega_max)
    raw = proc.sample_batch(rng, n_trajectories, initial_state=z0)
    return {
        "tokens": raw["tokens"],
        "observable": raw["observable"],
        "metric": raw["metric"],
        "letters": raw["letters"],
        "beliefs": raw["beliefs"],
        "states": raw["states"],
    }


def discover_checkpoints() -> dict[str, dict]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoints = {}
    for json_file in sorted(OUTPUT_DIR.glob("*_final.json")):
        base_name = json_file.stem.replace("_final", "")
        pt_file = json_file.with_suffix(".pt")
        if pt_file.exists():
            with open(json_file) as f:
                meta = json.load(f)
            checkpoints[base_name] = {
                "name": base_name,
                "json_path": json_file,
                "pt_path": pt_file,
                "meta": meta,
            }
    return checkpoints


def load_models_for_run(ckpt_info: dict, device: str = DEVICE):
    meta = ckpt_info["meta"]
    cfg_d = meta["model"]
    train_d = meta.get("training", {})
    k = train_d.get("k", 1)
    model_cfg = ModelConfig(
        vocab_size=cfg_d["vocab_size"],
        n_ctx=cfg_d["n_ctx"],
        n_layers=cfg_d["n_layers"],
        n_heads=cfg_d["n_heads"],
        d_model=cfg_d["d_model"],
        d_mlp=cfg_d["d_mlp"],
        seed=cfg_d.get("seed", 0),
    )
    if k > 1:
        trained_model = LookaheadTransformer(model_cfg, k)
        random_model = LookaheadTransformer(model_cfg, k)
    else:
        trained_model = TinyTransformer(model_cfg)
        random_model = TinyTransformer(model_cfg)

    trained_model.load_state_dict(torch.load(ckpt_info["pt_path"], map_location=device))
    trained_model.to(device).eval()
    random_model.to(device).eval()
    return trained_model, random_model, model_cfg, meta


def extract_probe_targets(batch: dict, proc: MessDriven, n_steps: int = 10) -> dict[str, np.ndarray]:
    last = np.arange(n_steps - 1, proc.seq_len, n_steps)
    belief = batch["beliefs"][:, last] @ simplex_embedding(proc.chain.n_states)
    metric = batch["metric"][:, last]
    return {
        "physics": metric.reshape(-1, metric.shape[-1]),
        "belief": belief.reshape(-1, belief.shape[-1]),
    }


def extract_probe_features(streams: list[np.ndarray], mode: str, n_steps: int = 10) -> np.ndarray:
    if mode == "single_layer_single_token":
        return tick_features([streams[-1]], n_steps, whole_tick=False).astype(np.float32)
    elif mode == "all_layers_single_token":
        return tick_features(streams, n_steps, whole_tick=False).astype(np.float32)
    elif mode == "single_layer_cycle":
        return tick_features([streams[-1]], n_steps, whole_tick=True).astype(np.float32)
    elif mode == "all_layers_cycle":
        return tick_features(streams, n_steps, whole_tick=True).astype(np.float32)
    elif mode.startswith("layer_"):
        parts = mode.split("_")
        depth = int(parts[1])
        whole = len(parts) > 2 and parts[2] == "cycle"
        return tick_features([streams[depth]], n_steps, whole_tick=whole).astype(np.float32)
    raise ValueError(f"Unknown feature mode: {mode}")


def cross_validated_ridge(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    dev_mask: np.ndarray,
    test_mask: np.ndarray,
    ridge_alphas: tuple = (0.1, 1.0, 10.0, 50.0),
    cv_folds: int = 5,
) -> dict:
    estimator = make_pipeline(StandardScaler(), Ridge(solver="auto"))
    search = GridSearchCV(
        estimator,
        {"ridge__alpha": list(ridge_alphas)},
        cv=GroupKFold(n_splits=cv_folds),
        scoring="r2",
        n_jobs=1,
    )
    search.fit(X[dev_mask], y[dev_mask], groups=groups[dev_mask])
    prediction = search.best_estimator_.predict(X[test_mask])
    return {
        "alpha": float(search.best_params_["ridge__alpha"]),
        "cv_r2": float(search.best_score_),
        "test_r2": float(r2_score(y[test_mask], prediction, multioutput="uniform_average")),
        "feature_dim": int(X.shape[1]),
        "dev_rows_per_feature": float(dev_mask.sum() / X.shape[1]),
    }


def load_analysis_csv() -> pd.DataFrame | None:
    global _CACHED_DF
    for candidate in [ANALYSIS_CSV, OUTPUT_DIR / "probe_results_all.csv"]:
        if candidate.exists():
            try:
                df = pd.read_csv(candidate)
                if len(df) > 0:
                    _CACHED_DF = df
                    return _CACHED_DF
            except Exception:
                pass
    return None


def save_analysis_csv(df: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(ANALYSIS_CSV, index=False)


def run_precomputation_job(
    target_models: list[str],
    macro_modes: bool,
    layerwise_token: bool,
    layerwise_cycle: bool,
    targets: list[str],
    n_traj: int = 512,
    cv_folds: int = 5,
    alphas: tuple = (0.1, 1.0, 10.0, 50.0),
    progress_callback=None,
    log_callback=None,
) -> pd.DataFrame:
    global _CACHED_DF
    checkpoints = discover_checkpoints()
    new_rows = []

    for model_name in target_models:
        if model_name not in checkpoints:
            continue
        ckpt_info = checkpoints[model_name]
        if log_callback:
            log_callback(f"\nProcessing model: {model_name}")

        trained_m, random_m, m_cfg, meta = load_models_for_run(ckpt_info, device=DEVICE)
        exp_meta = meta.get("experiment", {})
        proc = build_process_from_meta(exp_meta, meta.get("hmm"))
        n_steps = exp_meta.get("n", 10)

        eval_rng = np.random.default_rng(exp_meta.get("seed", 20260917) + 3)
        init_cfg = exp_meta.get("initial_state")
        batch = sample_analysis_batch(proc, eval_rng, n_traj, init_cfg=init_cfg)
        target_dict = extract_probe_targets(batch, proc, n_steps=n_steps)

        groups = np.repeat(np.arange(n_traj), proc.m)
        order = np.random.default_rng(exp_meta.get("seed", 20260917) + 4).permutation(n_traj)
        n_test = max(1, int(n_traj * 0.2))
        test_seqs = set(order[:n_test].tolist())
        test_mask = np.array([g in test_seqs for g in groups])
        dev_mask = ~test_mask

        modes = []
        if macro_modes:
            modes.extend(["single_layer_single_token", "all_layers_single_token", "single_layer_cycle", "all_layers_cycle"])
        if layerwise_token:
            modes.extend([f"layer_{l}_single_token" for l in range(m_cfg.n_layers + 1)])
        if layerwise_cycle:
            modes.extend([f"layer_{l}_cycle" for l in range(m_cfg.n_layers + 1)])

        for model_tag, model_obj in (("trained", trained_m), ("random_init", random_m)):
            streams = residual_streams_batched(model_obj.to(DEVICE).eval(), batch["tokens"], DEVICE)
            for mode in modes:
                X = extract_probe_features(streams, mode, n_steps=n_steps)
                for t_name in targets:
                    score = cross_validated_ridge(
                        X,
                        target_dict[t_name],
                        groups,
                        dev_mask,
                        test_mask,
                        ridge_alphas=alphas,
                        cv_folds=cv_folds,
                    )
                    row = {
                        "experiment": exp_meta.get("name", "standard"),
                        "system": meta.get("process", "pendulum_mess4"),
                        "training": meta.get("training", {}).get("name", model_name),
                        "seed": meta.get("training", {}).get("seed", 0),
                        "k_lookahead": meta.get("training", {}).get("k", 1),
                        "d_model": m_cfg.d_model,
                        "n_layers": m_cfg.n_layers,
                        "model_type": model_tag,
                        "mode": mode,
                        "target": t_name,
                        **score,
                    }
                    new_rows.append(row)
                    if progress_callback:
                        progress_callback(1)
                    if log_callback:
                        log_callback(f"  [{model_tag:11s}] {mode:25s} | {t_name:7s} -> Test R²={score['test_r2']:.4f} (CV={score['cv_r2']:.4f})")

    new_df = pd.DataFrame(new_rows)
    existing_df = load_analysis_csv()
    if existing_df is not None and len(existing_df) > 0:
        combined = pd.concat([existing_df, new_df]).drop_duplicates(
            subset=["experiment", "training", "model_type", "mode", "target"], keep="last"
        ).reset_index(drop=True)
    else:
        combined = new_df

    _CACHED_DF = combined
    save_analysis_csv(combined)
    return combined


def run_precomputation(
    target_models: list[str] | str = "all",
    macro_modes: bool = True,
    layerwise_token: bool = True,
    layerwise_cycle: bool = False,
    targets: list[str] | tuple = ("belief", "physics"),
    n_traj: int = 512,
    cv_folds: int = 5,
    alphas: tuple = (0.1, 1.0, 10.0, 50.0),
    verbose: bool = True,
) -> pd.DataFrame:
    checkpoints = discover_checkpoints()
    if target_models == "all" or target_models == ["all"]:
        models = list(checkpoints.keys())
    elif isinstance(target_models, str):
        models = [target_models]
    else:
        models = list(target_models)

    return run_precomputation_job(
        target_models=models,
        macro_modes=macro_modes,
        layerwise_token=layerwise_token,
        layerwise_cycle=layerwise_cycle,
        targets=list(targets),
        n_traj=n_traj,
        cv_folds=cv_folds,
        alphas=alphas,
        log_callback=print if verbose else None,
    )


def plot_macro_overview(
    run_name: str = "next-token",
    df: pd.DataFrame | None = None,
    targets: list[str] | tuple = ("belief", "physics"),
    save_path: Path | str | None = None,
) -> plt.Figure | None:
    current_df = df if df is not None else load_analysis_csv()
    if current_df is None or len(current_df) == 0:
        print("No analysis data available.")
        return None

    macro_modes = ["single_layer_single_token", "all_layers_single_token", "single_layer_cycle", "all_layers_cycle"]
    sub = current_df[(current_df["training"] == run_name) & (current_df["mode"].isin(macro_modes))]
    if len(sub) == 0:
        print(f"No macro mode data found for '{run_name}'.")
        return None

    fig, axes = plt.subplots(len(targets), 2, figsize=(13, 3.8 * len(targets)), squeeze=False)
    for r_idx, tgt in enumerate(targets):
        t_sub = sub[sub["target"] == tgt]
        piv = t_sub.pivot(index="mode", columns="model_type", values="test_r2").reindex(macro_modes)
        ax1, ax2 = axes[r_idx, 0], axes[r_idx, 1]

        piv.plot.bar(ax=ax1, color=["#e74c3c", "#2980b9"], alpha=0.85, edgecolor="k", width=0.6)
        ax1.set_title(f"{run_name} | {tgt.capitalize()} Held-Out R²", fontweight="bold")
        ax1.set_ylabel("Test R²")
        ax1.set_ylim(-0.05, 1.05)
        ax1.axhline(0, color="k", lw=0.8, ls="--")
        ax1.grid(axis="y", alpha=0.3)
        ax1.tick_params(axis="x", rotation=18)
        ax1.legend(["Random Init", "Trained"])

        if "trained" in piv.columns and "random_init" in piv.columns:
            delta = piv["trained"] - piv["random_init"]
            colors = ["#27ae60" if d >= 0 else "#c0392b" for d in delta]
            delta.plot.bar(ax=ax2, color=colors, edgecolor="k", width=0.6)
            ax2.set_title(f"{run_name} | Net Gain (Trained − Random)", fontweight="bold")
            ax2.set_ylabel("Δ Test R²")
            ax2.axhline(0, color="k", lw=0.8, ls="--")
            ax2.grid(axis="y", alpha=0.3)
            ax2.tick_params(axis="x", rotation=18)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=100)
    plt.show()
    return fig


def plot_layerwise_emergence(
    run_name: str = "next-token",
    df: pd.DataFrame | None = None,
    targets: list[str] | tuple = ("belief", "physics"),
    save_path: Path | str | None = None,
) -> plt.Figure | None:
    current_df = df if df is not None else load_analysis_csv()
    if current_df is None or len(current_df) == 0:
        print("No analysis data available.")
        return None

    sub = current_df[(current_df["training"] == run_name) & (current_df["mode"].str.startswith("layer_")) & (current_df["mode"].str.endswith("_single_token"))].copy()
    if len(sub) == 0:
        print(f"No layerwise data for '{run_name}'. Run precomputation with layerwise mode enabled.")
        return None

    sub["depth"] = sub["mode"].apply(lambda m: int(m.split("_")[1]))
    sub["depth_name"] = sub["depth"].apply(lambda d: "emb" if d == 0 else f"L{d}")
    sub = sub.sort_values("depth")

    fig, axes = plt.subplots(1, len(targets), figsize=(6.8 * len(targets), 4.2), squeeze=False)
    for c_idx, tgt in enumerate(targets):
        ax = axes[0, c_idx]
        t_sub = sub[sub["target"] == tgt]
        for m_type, style, col in [("trained", "-o", "#2980b9"), ("random_init", "--s", "#e74c3c")]:
            m_sub = t_sub[t_sub["model_type"] == m_type].sort_values("depth")
            if len(m_sub) > 0:
                ax.plot(m_sub["depth_name"], m_sub["test_r2"], style, color=col, lw=2, ms=6, label=m_type)
        ax.set_title(f"{tgt.capitalize()} Layerwise Trajectory ({run_name})", fontweight="bold")
        ax.set_xlabel("Residual Stream Depth")
        ax.set_ylabel("Test R²")
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(0, color="k", lw=0.8, ls="--")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(loc="lower right" if tgt == "belief" else "upper left")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=100)
    plt.show()
    return fig


def plot_sweep_comparison(
    df: pd.DataFrame | None = None,
    targets: list[str] | tuple = ("belief", "physics"),
    save_path: Path | str | None = None,
) -> plt.Figure | None:
    current_df = df if df is not None else load_analysis_csv()
    if current_df is None or len(current_df) == 0:
        print("No analysis data available.")
        return None

    macro_sub = current_df[current_df["mode"].isin(["single_layer_single_token", "all_layers_cycle"])].copy()
    if len(macro_sub) == 0:
        print("No sweep comparison data available.")
        return None

    fig, axes = plt.subplots(len(targets), 1, figsize=(11, 4.2 * len(targets)), squeeze=False)
    for r_idx, tgt in enumerate(targets):
        ax = axes[r_idx, 0]
        piv = macro_sub[macro_sub["target"] == tgt].pivot_table(index="training", columns=["model_type", "mode"], values="test_r2")
        piv.plot.bar(ax=ax, width=0.7)
        ax.set_title(f"{tgt.capitalize()} Comparison Across Sweep Horizons", fontweight="bold")
        ax.set_ylabel("Test R²")
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(0, color="k", lw=0.8, ls="--")
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis="x", rotation=0)
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=100)
    plt.show()
    return fig


def get_metrics_table(run_name: str | None = None, df: pd.DataFrame | None = None) -> pd.DataFrame:
    current_df = df if df is not None else load_analysis_csv()
    if current_df is None or len(current_df) == 0:
        return pd.DataFrame()

    cols = ["training", "model_type", "mode", "target", "test_r2", "cv_r2", "alpha", "feature_dim"]
    sub = current_df[[c for c in cols if c in current_df.columns]]
    if run_name is not None:
        sub = sub[sub["training"] == run_name]
    return sub.sort_values(["target", "mode", "model_type"])


def build_precomputation_widget() -> widgets.VBox:
    discovered = discover_checkpoints()
    names = list(discovered.keys())
    model_options = [("All Models", "all")] + [(n, n) for n in names]

    model_sel = widgets.SelectMultiple(
        options=model_options,
        value=("all",),
        description="Models:",
        rows=max(3, min(len(model_options), 6)),
        layout=widgets.Layout(width="420px"),
    )

    chk_macro = widgets.Checkbox(value=True, description="Phase 3 Modes (Macro)", indent=False)
    chk_layerwise_token = widgets.Checkbox(value=True, description="Layerwise Single-Token (emb, L1..L4)", indent=False)
    chk_layerwise_cycle = widgets.Checkbox(value=False, description="Layerwise Cycle", indent=False)

    chk_belief = widgets.Checkbox(value=True, description="Belief State (Simplex)", indent=False)
    chk_physics = widgets.Checkbox(value=True, description="Physical Observables", indent=False)

    slider_n_traj = widgets.IntSlider(value=512, min=128, max=2048, step=128, description="Trajectories:", layout=widgets.Layout(width="380px"))
    slider_cv_folds = widgets.IntSlider(value=5, min=2, max=10, step=1, description="CV Folds:", layout=widgets.Layout(width="380px"))
    text_alphas = widgets.Text(value="0.1, 1.0, 10.0, 50.0", description="Alphas:", layout=widgets.Layout(width="380px"))

    btn_run = widgets.Button(description="Run Precomputation", button_style="primary", icon="play", layout=widgets.Layout(width="200px", height="36px"))
    btn_load = widgets.Button(description="Load analysis.csv", button_style="info", icon="folder-open", layout=widgets.Layout(width="180px", height="36px"))

    prog = widgets.IntProgress(value=0, min=0, max=100, description="Progress:", bar_style="info", layout=widgets.Layout(width="600px"))
    status_lbl = widgets.Label(value="Ready.")
    console = widgets.Output(layout=widgets.Layout(height="200px", overflow="auto", border="1px solid #ddd", padding="6px"))

    def on_load(b):
        with console:
            clear_output()
            df = load_analysis_csv()
            if df is not None:
                status_lbl.value = f"Loaded {len(df)} records from {ANALYSIS_CSV.name}."
                display(df.head(10))
            else:
                status_lbl.value = "No existing cache found."
                print("No analysis.csv or probe_results_all.csv found in outputs/.")

    btn_load.on_click(on_load)

    def on_run(b):
        btn_run.disabled = True
        btn_load.disabled = True
        status_lbl.value = "Running cross-validated probing..."

        with console:
            clear_output()
            sel = list(model_sel.value)
            models = list(discovered.keys()) if "all" in sel else sel
            if not models:
                print("No models selected.")
                btn_run.disabled = False
                btn_load.disabled = False
                return

            targets = []
            if chk_physics.value:
                targets.append("physics")
            if chk_belief.value:
                targets.append("belief")
            if not targets:
                print("Select at least one target.")
                btn_run.disabled = False
                btn_load.disabled = False
                return

            try:
                alphas = tuple(float(x.strip()) for x in text_alphas.value.split(",") if x.strip())
            except ValueError:
                alphas = (0.1, 1.0, 10.0, 50.0)

            total_jobs = 0
            for m in models:
                meta = discovered[m]["meta"]
                n_layers = meta["model"]["n_layers"]
                m_count = (4 if chk_macro.value else 0) + ((n_layers + 1) if chk_layerwise_token.value else 0) + ((n_layers + 1) if chk_layerwise_cycle.value else 0)
                total_jobs += m_count * len(targets) * 2

            prog.max = max(1, total_jobs)
            prog.value = 0

            df = run_precomputation_job(
                target_models=models,
                macro_modes=chk_macro.value,
                layerwise_token=chk_layerwise_token.value,
                layerwise_cycle=chk_layerwise_cycle.value,
                targets=targets,
                n_traj=slider_n_traj.value,
                cv_folds=slider_cv_folds.value,
                alphas=alphas,
                progress_callback=lambda step: setattr(prog, "value", prog.value + step),
                log_callback=print,
            )
            status_lbl.value = f"Finished! Total {len(df)} records saved to analysis.csv."
            print(f"\nCompleted. Cached to {ANALYSIS_CSV.resolve()}")

        btn_run.disabled = False
        btn_load.disabled = False

    btn_run.on_click(on_run)

    return widgets.VBox([
        widgets.HTML("<h3>Precomputation Controller</h3>"),
        widgets.HBox([
            model_sel,
            widgets.VBox([
                widgets.HTML("<b>Feature Modes:</b>"),
                chk_macro, chk_layerwise_token, chk_layerwise_cycle,
                widgets.HTML("<b>Targets:</b>"),
                chk_belief, chk_physics,
            ]),
        ]),
        widgets.HBox([slider_n_traj, slider_cv_folds]),
        text_alphas,
        widgets.HBox([btn_run, btn_load]),
        prog,
        status_lbl,
        widgets.HTML("<b>Log Console:</b>"),
        console,
    ])


def build_viewer_widget() -> widgets.VBox:
    df = load_analysis_csv()
    if df is None or len(df) == 0:
        return widgets.VBox([
            widgets.HTML("<h3>Results Viewer</h3>"),
            widgets.HTML("<p>No precomputed results found. Use the controller above to compute or load <code>analysis.csv</code>.</p>"),
        ])

    runs = sorted(df["training"].unique().tolist())
    run_dropdown = widgets.Dropdown(options=runs, value=runs[0], description="Run:", layout=widgets.Layout(width="360px"))
    target_toggle = widgets.ToggleButtons(options=["belief", "physics", "both"], value="belief", description="Target:", button_style="info")
    btn_refresh = widgets.Button(description="Refresh Data", icon="refresh", layout=widgets.Layout(width="140px"))

    tab = widgets.Tab()
    out_macro = widgets.Output()
    out_layerwise = widgets.Output()
    out_sweep = widgets.Output()
    out_table = widgets.Output()

    tab.children = [out_macro, out_layerwise, out_sweep, out_table]
    tab.set_title(0, "Macro Overview")
    tab.set_title(1, "Layerwise Emergence")
    tab.set_title(2, "Sweep Comparison")
    tab.set_title(3, "Metrics Table")

    def render():
        current_df = _CACHED_DF if _CACHED_DF is not None else load_analysis_csv()
        if current_df is None:
            return
        run_name = run_dropdown.value
        tgt_choice = target_toggle.value
        targets = ["belief", "physics"] if tgt_choice == "both" else [tgt_choice]

        with out_macro:
            clear_output(wait=True)
            plot_macro_overview(run_name, df=current_df, targets=targets)

        with out_layerwise:
            clear_output(wait=True)
            plot_layerwise_emergence(run_name, df=current_df, targets=targets)

        with out_sweep:
            clear_output(wait=True)
            plot_sweep_comparison(df=current_df, targets=targets)

        with out_table:
            clear_output(wait=True)
            tbl = get_metrics_table(run_name, df=current_df)
            display(HTML(f"<b>{run_name}</b> ({len(tbl)} rows):"))
            display(tbl)

    def on_update(change=None):
        render()

    def on_refresh_clicked(b):
        load_analysis_csv()
        runs_updated = sorted(_CACHED_DF["training"].unique().tolist()) if _CACHED_DF is not None else []
        if runs_updated:
            run_dropdown.options = runs_updated
        render()

    run_dropdown.observe(on_update, names="value")
    target_toggle.observe(on_update, names="value")
    btn_refresh.on_click(on_refresh_clicked)

    render()

    return widgets.VBox([
        widgets.HTML("<h3>Interactive Results Dashboard</h3>"),
        widgets.HBox([run_dropdown, target_toggle, btn_refresh]),
        tab,
    ])
