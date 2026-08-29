"""Stage 5.6 -- E2d: does ablating the belief distort the model's k-step law?

The next-token loss is nearly blind to the belief, so a t+1 ablation has almost
no power by construction. The belief is the sufficient statistic for the whole
future, so the place to look for its effect is the *joint* law of the next
several tokens, which the branch HMM gives exactly.

Procedure. Cut every evaluation sequence at a fixed prefix (the last position of
a segment, where the posterior is most informed), roll the model forward k steps
autoregressively under an intervention, and compare its k-step marginal to the
exact `p^(k)` by KL.

The model's k-step marginal is estimated Rao-Blackwellised: sample k - 1 tokens,
then read the *distribution* at step k and average over replicas, rather than
histogramming sampled tokens. Same estimator, far less variance.

Two things make the raw KL uninterpretable and only the excess meaningful.
Sampling error compounds with k in every arm, and any rank-r deletion costs
something. So the quantity reported is

    excess(k) = KL(ablated, k) - mean_over_controls KL(random subspace, k)

with the control spread recomputed at each k, never carried over from k = 1,
because its variance grows with k too.

Run:  uv run python experiments/13_rollout.py
Env:  OUTPUT_DIR, SYSTEMS, N_EVAL, N_ROLLOUT, K_MAX, N_CONTROL, DEPTHS, TAG.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import os
import time

import numpy as np
import torch

from models.ablation import _grouped_r2, erasure_basis, mean_ablate, random_basis
from models.analysis import _sequence_split, residual_streams_batched
from models.probe import fit_probe
from models.train import pick_device
from models.transformer import ModelConfig, TinyTransformer
from physics.branch import feature_groups, forward_features
from physics.branch_configs import BRANCH_CONFIGS, make_branch_process, scaled_kick
from physics.myopic import predictive_stack

N_EVAL = int(os.environ.get("N_EVAL", 256))
N_ROLLOUT = int(os.environ.get("N_ROLLOUT", 32))
K_MAX = int(os.environ.get("K_MAX", 6))
N_CONTROL = int(os.environ.get("N_CONTROL", 8))
TRAIN_FRAC = 0.7
EVAL_SEED = 20_260_828
KICK_SCALE = float(os.environ.get("KICK_SCALE", 1.0))
TAG = os.environ.get("TAG", "")
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "experiments/outputs-03"))
SYSTEMS = os.environ.get("SYSTEMS", ",".join(BRANCH_CONFIGS)).split(",")
DEPTHS = os.environ.get("DEPTHS")


def _depth_name(d: int) -> str:
    return "embedding" if d == 0 else f"resid_post_{d - 1}"


@torch.no_grad()
def rollout_marginals(model, prefix, t0, k_max, n_rollout, depth, edit, generator):
    """`(k_max, n_seq, vocab)`: the model's law for the token at `t0 + k`.

    Each replica samples its own continuation, and the distribution -- not the
    sampled token -- is what gets averaged, so the estimator is exact in the
    limit of many replicas and already low-variance at 32.
    """
    n, L = prefix.shape
    work = prefix.repeat_interleave(n_rollout, dim=0).clone()
    out = torch.zeros(k_max, n, model.config.vocab_size, device=prefix.device)

    for k in range(1, k_max + 1):
        pos = t0 + k - 1  # the position whose output predicts token t0 + k
        if pos + 1 >= L:
            break
        x = model._embed(work)
        for i, block in enumerate(model.blocks):
            if edit is not None and i == depth:
                x = edit(x)
            x = block(x)
        if edit is not None and depth == len(model.blocks):
            x = edit(x)
        probs = torch.softmax(model.unembed(model.ln_f(x))[:, pos], dim=-1)
        out[k - 1] = probs.view(n, n_rollout, -1).mean(1)
        work[:, pos + 1] = torch.multinomial(probs, 1, generator=generator).squeeze(1)
    return out.cpu().numpy()


def _kl(exact, model_p):
    """KL(exact || model), averaged over sequences. Exact is the reference, so a
    token the model cannot produce is what should be punished."""
    m = np.clip(model_p, 1e-9, None)
    e = np.clip(exact, 1e-12, None)
    return float((e * (np.log(e) - np.log(m))).sum(-1).mean())


def main() -> None:
    device = pick_device()
    training = json.loads((OUTPUT_DIR / f"phase2_branch_training{TAG}.json").read_text())
    print(f"device={device} systems={SYSTEMS} n_eval={N_EVAL} n_rollout={N_ROLLOUT} "
          f"k_max={K_MAX}", flush=True)
    results = {}

    for name in SYSTEMS:
        t0_wall = time.perf_counter()
        print(f"\n=== {name} ===", flush=True)
        process = make_branch_process(name, **scaled_kick(name, KICK_SCALE))
        episodes = process.sample_batch(np.random.default_rng(EVAL_SEED), N_EVAL)
        features = forward_features(process, episodes.tokens).astype(np.float64)
        groups = feature_groups(process)
        rng = np.random.default_rng(0)
        train_idx, test_idx = _sequence_split(N_EVAL, TRAIN_FRAC, rng)
        config = ModelConfig(**training[name]["model"])

        model = TinyTransformer(config)
        model.load_state_dict(torch.load(OUTPUT_DIR / f"{name}{TAG}_trained.pt", map_location=device))
        model = model.to(device).eval()

        # Cut where the posterior is most informed and a full horizon still fits.
        sps = process.steps_per_segment
        cut = ((process.seq_len - K_MAX - 1) // sps) * sps - 1
        if cut < sps:
            print(f"!! {name}: sequence too short for k_max={K_MAX}", flush=True)
            continue
        exact = predictive_stack(process, episodes.tokens, horizon=K_MAX)[:, cut].astype(np.float64)
        tokens_t = torch.as_tensor(episodes.tokens, dtype=torch.long, device=device)
        streams = residual_streams_batched(model, episodes.tokens, device)

        depths = [int(x) for x in DEPTHS.split(",")] if DEPTHS else list(range(1, len(streams) - 1))
        record = {"cut_position": int(cut), "n_rollout": N_ROLLOUT, "by_depth": []}

        for depth in depths:
            act = streams[depth].astype(np.float64)
            d_model = act.shape[-1]
            mean = act.reshape(-1, d_model).mean(0)
            a_tr = act[train_idx].reshape(-1, d_model)
            a_te = act[test_idx].reshape(-1, d_model)
            f_tr = features[train_idx].reshape(-1, features.shape[-1])
            f_te = features[test_idx].reshape(-1, features.shape[-1])
            cols = groups["action_lag0"]
            intact_r2 = _grouped_r2(fit_probe(a_tr, f_tr[:, cols]), a_te, f_te[:, cols])
            basis, hist = erasure_basis(
                a_tr, f_tr[:, cols], a_te, f_te[:, cols],
                floor=max(0.02, 0.1 * intact_r2), max_rank=64,
            )
            if basis.shape[1] == 0:
                continue
            rank = basis.shape[1]

            gen = torch.Generator(device=device)
            def run(edit, seed):
                gen.manual_seed(seed)
                return rollout_marginals(
                    model, tokens_t, cut, K_MAX, N_ROLLOUT, depth, edit, gen
                )

            # The same rollout seed everywhere, so the arms differ by the
            # intervention and not by which continuations were sampled.
            base = run(None, 12345)
            ablated = run(mean_ablate(basis, mean), 12345)
            controls = [
                run(mean_ablate(random_basis(rng, d_model, rank), mean), 12345)
                for _ in range(N_CONTROL)
            ]

            rows = []
            for k in range(K_MAX):
                if not np.isfinite(exact[:, k]).all():
                    break
                kl_base = _kl(exact[:, k], base[k])
                kl_abl = _kl(exact[:, k], ablated[k])
                draws = np.array([_kl(exact[:, k], c[k]) for c in controls])
                rows.append(
                    {
                        "k": k + 1,
                        "kl_intact": kl_base,
                        "kl_ablated": kl_abl,
                        "kl_control_mean": float(draws.mean()),
                        "kl_control_sd": float(draws.std()),
                        # Excess over the matched-rank control, in that control's
                        # own spread at this k -- never carried over from k = 1.
                        "excess": kl_abl - float(draws.mean()),
                        "excess_in_control_sds": (
                            (kl_abl - float(draws.mean())) / draws.std() if draws.std() > 0 else float("nan")
                        ),
                        "damage_over_intact": kl_abl - kl_base,
                    }
                )
            record["by_depth"].append(
                {"depth": depth, "name": _depth_name(depth), "rank": rank,
                 "r2_before": hist[0], "r2_after": hist[-1], "by_k": rows}
            )
            print(f"  {_depth_name(depth):<14} rank={rank:>3}  excess KL by k: "
                  + "  ".join(f"k{r['k']}:{r['excess']:+.4f}({r['excess_in_control_sds']:+.1f}sd)"
                              for r in rows), flush=True)
            print(f"  {'':<14} intact KL by k:  "
                  + "  ".join(f"k{r['k']}:{r['kl_intact']:.4f}" for r in rows), flush=True)

        results[name] = {**record, "wall_seconds": time.perf_counter() - t0_wall}
        (OUTPUT_DIR / f"phase5_06_rollout{TAG}.json").write_text(
            json.dumps(results, indent=2, default=float)
        )

    print(f"\nwrote {OUTPUT_DIR / f'phase5_06_rollout{TAG}.json'}", flush=True)


if __name__ == "__main__":
    main()
