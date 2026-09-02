"""Interactive inspection of one sequence: ground truth against what the model says.

The notebook that drives this is `notebooks/explore.ipynb`. Everything with any
logic in it lives here rather than in notebook cells, so it is testable and so a
saved notebook is a thin view over code that is under version control.

Three things are shown for a single sequence, and they answer different questions:

- the **next-token distribution**, which is the model's actual output and needs no
  probe at all, so it is the only panel with nothing fitted between the model and
  the picture;
- the **belief**, read out of the residual stream by a ridge probe fitted on other
  sequences, which is the object the whole experiment is about;
- the **physical metric**, read out by the same fit, which is the reference point:
  a window of recent tokens already carries most of it, so the belief panel is
  only interesting to the extent it beats what the metric panel gets for free.

The probe here is fitted on a small batch for responsiveness, so its numbers are
noisier than `experiments/06_reference_probe.py`. Use that script for a number to
quote and this module to see what the number is made of.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge

from .analysis import residual_streams_batched, tick_features
from .transformer import ModelConfig, TinyTransformer

#: `n_heads` is the one shape a checkpoint does not pin down: the qkv projection
#: has the same size however the heads divide it. Both trained architectures are
#: listed, and `load_model` takes an override for anything else.
HEADS_BY_WIDTH = {128: 1, 256: 4}


def find_checkpoints(output_dir: str | Path) -> dict[str, Path]:
    """Every `.pt` under `output_dir`, labelled by path, newest name order."""
    root = Path(output_dir)
    return {str(p.relative_to(root)): p for p in sorted(root.rglob("*.pt"))}


def config_from_state_dict(state: dict, n_heads: int | None = None) -> ModelConfig:
    """Recover the architecture from the weights, so any checkpoint just loads."""
    vocab_size, d_model = state["tok_emb.weight"].shape
    layers = {int(k.split(".")[1]) for k in state if k.startswith("blocks.")}
    width = int(d_model)
    return ModelConfig(
        vocab_size=int(vocab_size),
        n_ctx=int(state["pos_emb.weight"].shape[0]),
        n_layers=max(layers) + 1,
        n_heads=n_heads or HEADS_BY_WIDTH.get(width, 1),
        d_model=width,
        d_mlp=int(state["blocks.0.mlp.0.weight"].shape[0]),
    )


def load_model(path: str | Path, device: str = "cpu", n_heads: int | None = None):
    state = torch.load(path, map_location=device)
    model = TinyTransformer(config_from_state_dict(state, n_heads))
    model.load_state_dict(state)
    return model.to(device).eval()


def _targets(batch: dict, proc) -> tuple[np.ndarray, dict[str, slice]]:
    """Belief probabilities and physical metric at each tick's final step."""
    last = np.arange(proc.n_steps - 1, proc.seq_len, proc.n_steps)
    belief = batch["beliefs"][:, last, :]
    metric = batch["metric"][:, last, :]
    stacked = np.concatenate([belief, metric], axis=-1)
    k = belief.shape[-1]
    return (
        stacked.reshape(-1, stacked.shape[-1]).astype(np.float64),
        {"belief": slice(0, k), "metric": slice(k, stacked.shape[-1])},
    )


@dataclass(frozen=True)
class Readout:
    """A fitted linear map from residual stream to belief and metric."""

    ridge: Ridge
    groups: dict[str, slice]
    depths: tuple[int, ...]
    whole_tick: bool
    r2: dict[str, float]
    rows_per_feature: float

    def predict(self, model, tokens: np.ndarray, proc, device: str) -> dict[str, np.ndarray]:
        streams = residual_streams_batched(model, tokens, device, depths=self.depths)
        rows = self.ridge.predict(tick_features(streams, proc.n_steps, self.whole_tick))
        out = {name: rows[:, columns] for name, columns in self.groups.items()}
        out["belief"] = np.clip(out["belief"], 0.0, None)
        out["belief"] /= np.maximum(out["belief"].sum(1, keepdims=True), 1e-12)
        return {k: v.reshape(len(tokens), proc.m, -1) for k, v in out.items()}


def fit_readout(
    model,
    proc,
    device: str = "cpu",
    n_fit: int = 3_072,
    n_depths: int = 4,
    whole_tick: bool = True,
    alpha: float = 100.0,
    seed: int = 12_345,
) -> Readout:
    """Fit belief and metric probes on `n_fit` fresh sequences.

    A different seed from anything the notebook then displays, so the sequence on
    screen is genuinely held out from the map drawing it.

    The reported R2 is on a fifth of these sequences that the fit never saw. An
    in-sample number here would be worse than useless: the whole-tick readout on
    the wide model has 10,240 features, which will fit its own training rows
    almost perfectly at any `n_fit` the notebook can afford.
    """
    batch = proc.sample_batch(np.random.default_rng(seed), n_fit)
    depths = tuple(range(model.config.n_layers + 1 - n_depths, model.config.n_layers + 1))
    streams = residual_streams_batched(model, batch["tokens"], device, depths=depths)
    features = tick_features(streams, proc.n_steps, whole_tick)
    targets, groups = _targets(batch, proc)
    if features.shape[0] < 2 * features.shape[1]:
        raise ValueError(
            f"{features.shape[0]} rows for {features.shape[1]} features; raise n_fit "
            f"above {2 * features.shape[1] // proc.m} or turn whole_tick off"
        )
    held = np.zeros(len(features), dtype=bool)
    held[np.random.default_rng(seed + 1).permutation(len(features))[: len(features) // 5]] = True
    ridge = Ridge(alpha=alpha).fit(features[~held], targets[~held])
    predicted = ridge.predict(features[held])
    truth = targets[held]
    scores = {
        name: float(
            1.0
            - ((truth[:, c] - predicted[:, c]) ** 2).sum()
            / ((truth[:, c] - truth[:, c].mean(0)) ** 2).sum()
        )
        for name, c in groups.items()
    }
    return Readout(
        ridge, groups, depths, whole_tick, scores, float((~held).sum() / features.shape[1])
    )


@torch.no_grad()
def next_token_probs(model, tokens: np.ndarray, device: str = "cpu") -> np.ndarray:
    """(L, vocab) distribution over the token following each position."""
    logits = model(torch.as_tensor(tokens[None], dtype=torch.long, device=device))
    return torch.softmax(logits[0], dim=-1).cpu().numpy()


def plot_sequence(proc, batch, index, model, readout, device: str = "cpu"):
    """Four panels for one sequence: physics, belief, metric, next-token."""
    import matplotlib.pyplot as plt

    tokens = batch["tokens"][index]
    predicted = readout.predict(model, batch["tokens"][index : index + 1], proc, device)
    ticks = np.arange(proc.m)
    step = np.arange(proc.seq_len)
    kicks = ticks * proc.n_steps

    fig, axes = plt.subplots(4, 1, figsize=(13, 12), height_ratios=[2, 2, 2, 3])

    ax = axes[0]
    observed = np.atleast_2d(batch["observable"][index].T).T  # (L, channels)
    for channel, name in enumerate(proc.channel_names):
        ax.plot(step, observed[:, channel], lw=1.2, label=name)
    if len(proc.obs_bins) == 1:
        lo, hi = proc.obs_ranges[0]
        ax.plot(step, lo + tokens / (proc.n_obs - 1) * (hi - lo),
                color="tab:orange", lw=0.8, ls="--", label=f"token ({proc.n_obs} bins)")
    for tick, letter in zip(kicks, batch["letters"][index]):
        ax.axvline(tick, color="0.85", lw=0.8, zorder=0)
        ax.text(tick, ax.get_ylim()[1], str(letter), fontsize=7, color="tab:red", va="top")
    ax.set_ylabel("observable")
    ax.set_title(f"physics and its quantisation into {'x'.join(map(str, proc.obs_bins))} bins; "
                 "red digits are the HMM letter kicking that tick")
    ax.legend(loc="lower right", fontsize=8)

    ax = axes[1]
    truth = batch["beliefs"][index][proc.n_steps - 1 :: proc.n_steps]
    for state in range(proc.chain.n_states):
        line, = ax.plot(ticks, truth[:, state], lw=1.6, label=f"mood {state}")
        ax.plot(ticks, predicted["belief"][0, :, state], lw=1.2, ls="--", color=line.get_color())
    ax.set_ylabel("P(next mood)")
    ax.set_ylim(0, 1)
    ax.set_title(f"belief: exact (solid) against linear readout (dashed), held-out R2 {readout.r2['belief']:+.3f}")
    ax.legend(ncol=4, fontsize=8)

    ax = axes[2]
    metric_truth = batch["metric"][index][proc.n_steps - 1 :: proc.n_steps]
    for j, name in enumerate(proc.system.metric_names):
        line, = ax.plot(ticks, metric_truth[:, j], lw=1.6, label=name)
        ax.plot(ticks, predicted["metric"][0, :, j], lw=1.2, ls="--", color=line.get_color())
    ax.set_ylabel("metric")
    ax.set_xlabel("tick")
    ax.set_title(f"physical metric, same fit, held-out R2 {readout.r2['metric']:+.3f}")
    ax.legend(ncol=4, fontsize=8)

    ax = axes[3]
    probs = next_token_probs(model, tokens, device)
    ax.imshow(probs[:-1].T, aspect="auto", origin="lower", cmap="magma",
              extent=(0, proc.seq_len - 1, 0, proc.n_obs - 1))
    ax.plot(step[:-1], tokens[1:], color="cyan", lw=0.9, label="true next token")
    for tick in kicks:
        ax.axvline(tick, color="white", lw=0.4, alpha=0.4)
    nll = -np.log(np.maximum(probs[np.arange(proc.seq_len - 1), tokens[1:]], 1e-12))
    ax.set_ylabel("token")
    ax.set_xlabel("position")
    ax.set_title(f"model next-token distribution, no probe involved; mean NLL {nll.mean():.3f} nats")
    ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    return fig


def _demo() -> None:
    from physics.messk_configs import make_process

    proc = make_process("pendulum_mess4", m=4)
    config = ModelConfig(vocab_size=proc.n_obs, n_ctx=proc.seq_len, n_layers=2, d_model=32, d_mlp=64)
    model = TinyTransformer(config).eval()

    recovered = config_from_state_dict(model.state_dict(), n_heads=1)
    assert recovered.n_layers == 2 and recovered.d_model == 32, recovered
    assert recovered.vocab_size == proc.n_obs and recovered.d_mlp == 64, recovered

    readout = fit_readout(model, proc, n_fit=256, n_depths=2, whole_tick=False, alpha=10.0)
    assert readout.rows_per_feature > 2, readout.rows_per_feature
    batch = proc.sample_batch(np.random.default_rng(7), 4)
    predicted = readout.predict(model, batch["tokens"], proc, "cpu")
    assert predicted["belief"].shape == (4, proc.m, proc.chain.n_states), predicted["belief"].shape
    assert np.allclose(predicted["belief"].sum(-1), 1.0), "belief rows must be a distribution"
    assert predicted["metric"].shape[-1] == len(proc.system.metric_names)

    probs = next_token_probs(model, batch["tokens"][0])
    assert probs.shape == (proc.seq_len, proc.n_obs), probs.shape
    assert np.allclose(probs.sum(-1), 1.0)
    fig = plot_sequence(proc, batch, 0, model, readout)
    assert len(fig.axes) == 4, len(fig.axes)

    print(f"explore ok (held-out R2 {readout.r2})")


if __name__ == "__main__":
    _demo()
