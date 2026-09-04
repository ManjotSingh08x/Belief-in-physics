"""The pipelines behind `notebooks/lab.ipynb`: generate, train, probe.

Kept out of the notebook so the three stages are testable and under version
control, and so the notebook is a set of sliders over code rather than code
pasted into cells.

The three stages are deliberately separable. Generation needs no model, so its
panels answer "is the data any good" before a transformer is involved at all -
which is the question that actually bites, because a channel that clips half its
samples or uses a tenth of its bins will produce a confident, meaningless probe
score downstream.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import torch

from physics.visualise import tunable_fields  # noqa: F401  (re-exported for the notebooks)

from .explore import fit_readout, load_model
from .train import TrainConfig, pick_device, train
from .transformer import ModelConfig, TinyTransformer


def counterfactual(proc, letters: np.ndarray, tick: int) -> dict:
    """Replay one sequence with every possible letter at `tick`.

    This is the causal effect of a single observed token's cause: the
    trajectories agree exactly up to `tick`, and differ afterwards only because
    of that one impulse. Anything the model could infer about the letter has to
    come from the gap between these curves.
    """
    letters = np.asarray(letters, dtype=np.int64).reshape(-1)
    if not 0 <= tick < proc.m:
        raise ValueError(f"tick must be in [0, {proc.m}), got {tick}")
    variants = np.tile(letters, (proc.chain.n_states, 1))
    variants[:, tick] = np.arange(proc.chain.n_states)
    out = proc.rollout(variants)
    out["tick"] = tick
    out["actual"] = int(letters[tick])
    # How far apart the alternatives stay, in tokens, after the branch point.
    branch = tick * proc.n_steps
    tokens = out["tokens"][:, branch:]
    out["separation"] = float(np.abs(tokens[:, None, :] - tokens[None, :, :]).max(-1).mean())
    return out


def survey_systems(rng_seed: int = 20_260_829, n: int = 256, tick: int = 4, **overrides) -> dict:
    """Generate from all four systems under one set of shared settings.

    The chain, the tick structure and the bin count are common to every system by
    construction, so the belief is literally the same object in all four. What
    differs is only the physical channel it passes through, which is what makes a
    side-by-side comparison meaningful rather than four unrelated plots.
    """
    from physics.messk_configs import MESSK_CONFIGS, make_process

    out = {}
    for name in sorted(MESSK_CONFIGS):
        proc = make_process(name, **overrides)
        batch = proc.sample_batch(np.random.default_rng(rng_seed), n)
        report = proc.bin_report(batch["observable"])
        cf = counterfactual(proc, batch["letters"][0], min(tick, proc.m - 1))
        belief = batch["beliefs"].reshape(-1, proc.chain.n_states)
        out[name] = {
            "proc": proc,
            "batch": batch,
            "counterfactual": cf,
            "bins": report,
            "separation": cf["separation"],
            "belief_entropy": float(
                -(belief * np.log(np.clip(belief, 1e-12, None))).sum(1).mean()
                / np.log(proc.chain.n_states)
            ),
        }
    return out


@dataclass(frozen=True)
class Arch:
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 1
    d_mlp: int | None = None

    def config(self, proc) -> ModelConfig:
        return ModelConfig(
            vocab_size=proc.n_obs,
            n_ctx=proc.seq_len,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            d_model=self.d_model,
            d_mlp=self.d_mlp or 4 * self.d_model,
        )


def run_training(
    proc,
    arch: Arch,
    out_dir: str | Path,
    name: str = "lab",
    total_tokens: int = 2_000_000,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    fractions: tuple[float, ...] = (0.0, 0.1, 0.25, 0.5, 1.0),
    seed: int = 0,
    device: str | None = None,
    progress=None,
) -> dict:
    """Train one model, snapshotting at `fractions` of the token budget.

    Snapshots are the point: a single final checkpoint cannot show *when* the
    belief appears, and 0.0 gives the random-initialisation control for free
    from the same run rather than from a separately constructed model.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = device or pick_device()
    model = TinyTransformer(arch.config(proc))
    saved: dict[int, str] = {}

    def snapshot(tokens_seen: int, m) -> None:
        path = out / f"{name}_{tokens_seen}.pt"
        torch.save({k: v.cpu() for k, v in m.state_dict().items()}, path)
        saved[tokens_seen] = str(path)
        if progress is not None:
            progress(tokens_seen, total_tokens)

    config = TrainConfig(
        total_tokens=total_tokens,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed,
        checkpoint_at=tuple(sorted({int(f * total_tokens) for f in fractions})),
    )
    report = train(
        model,
        lambda rng, n: proc.sample_batch(rng, n)["tokens"],
        proc.seq_len,
        config,
        device=device,
        on_checkpoint=snapshot,
    )
    report["checkpoint_paths"] = saved
    report["arch"] = asdict(arch)
    report["uniform_loss"] = float(np.log(proc.n_obs))
    (out / f"{name}_report.json").write_text(json.dumps(report, indent=1, default=str))
    return report


def probe_checkpoints(
    proc,
    paths: dict[int, str] | dict[str, str],
    device: str | None = None,
    n_fit: int = 3_072,
    n_depths: int = 4,
    whole_tick: bool = True,
    alpha: float = 100.0,
) -> list[dict]:
    """Held-out belief and metric R2 for each checkpoint, in one table.

    Every checkpoint is scored by the same protocol on the same sequences, so
    the column is comparable down its length. It is not comparable to
    `experiments/06_reference_probe.py`, which tunes the penalty on a
    validation split rather than fixing it.
    """
    device = device or pick_device()
    rows = []
    for label, path in sorted(paths.items(), key=lambda kv: kv[0]):
        model = load_model(path, device=device)
        readout = fit_readout(
            model, proc, device=device, n_fit=n_fit,
            n_depths=min(n_depths, model.config.n_layers + 1),
            whole_tick=whole_tick, alpha=alpha,
        )
        with torch.no_grad():
            evaluation = proc.sample_batch(np.random.default_rng(4_242), 256)["tokens"]
            loss = float(model.loss(torch.as_tensor(evaluation, dtype=torch.long, device=device)))
        rows.append({
            "label": label,
            "belief_r2": readout.r2["belief"],
            "metric_r2": readout.r2["metric"],
            "eval_loss": loss,
            "rows_per_feature": readout.rows_per_feature,
        })
    return rows


def _demo() -> None:
    import tempfile

    from physics.messk_configs import make_process

    proc = make_process("pendulum_mess4", m=4, n_steps=5)
    letters = np.array([0, 3, 1, 2])

    cf = counterfactual(proc, letters, tick=1)
    assert cf["tokens"].shape == (4, proc.seq_len), cf["tokens"].shape
    before = slice(0, 1 * proc.n_steps)
    assert (cf["tokens"][:, before] == cf["tokens"][0, before]).all(), "must agree before the branch"
    assert cf["separation"] > 0, "distinct letters must separate the trajectory"
    assert np.array_equal(cf["tokens"][cf["actual"]], proc.rollout(letters)["tokens"][0])

    with tempfile.TemporaryDirectory() as tmp:
        report = run_training(
            proc, Arch(d_model=32, n_layers=2, d_mlp=64), tmp,
            total_tokens=40_000, batch_size=8, fractions=(0.0, 1.0), device="cpu",
        )
        assert len(report["checkpoint_paths"]) == 2, report["checkpoint_paths"]
        table = probe_checkpoints(
            proc, report["checkpoint_paths"], device="cpu",
            n_fit=512, n_depths=2, whole_tick=False, alpha=10.0,
        )
        assert len(table) == 2 and all(r["rows_per_feature"] > 2 for r in table), table
    from physics.systems.pendulum import Pendulum

    knobs = tunable_fields(Pendulum())
    assert {"gamma", "theta0", "omega0", "g"} <= set(knobs), knobs
    assert "obs_range" not in knobs and "metric_names" not in knobs, knobs

    survey = survey_systems(n=16, m=4, n_steps=5)
    assert len(survey) == 4, sorted(survey)
    assert all(v["separation"] > 0 for v in survey.values()), "every system must react to its letters"
    assert all(0.0 < v["belief_entropy"] <= 1.0 for v in survey.values())

    print(f"lab ok (separation {cf['separation']:.1f} tokens, {len(table)} checkpoints probed)")


if __name__ == "__main__":
    _demo()
