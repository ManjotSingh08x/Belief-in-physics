"""The exact upper bound on belief recovery from observation tokens alone.

Every system here is deterministic given its letter sequence: the initial state
is fixed and the flow has no process noise, so the token stream is an exact
function of the letters. That makes the posterior over letter sequences
computable without approximation - replay every candidate prefix, keep the ones
whose tokens match the observation exactly, and weight the survivors by the
chain's forward vector.

Two numbers come out.

`ceiling_r2` is the R2 of the Bayes-optimal predictor `E[belief | tokens]`
against the true belief. No model of any size, linear probe or not, can exceed
it. If it sits near 0.65 then asking a probe for 0.90 is asking for information
the observation channel already destroyed; if it sits near 1.0 the gap to our
measured probes is model capacity or readout.

`oracle_loss` is the next-token cross entropy of a predictor that knows the
exact physical state and the exact belief. It is a lower bound on achievable
training loss, so the gap to a run's `eval_loss` bounds the headroom left.

Run:  uv run python experiments/07_ceiling.py
Env:  OUTPUT_DIR, CONFIGS, N_SEQ, MAX_BEAM.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from physics.messk_configs import MESSK_CONFIGS, make_process

N_SEQ = int(os.environ.get("N_SEQ", 256))
MAX_BEAM = int(os.environ.get("MAX_BEAM", 20_000))
EVAL_SEED = 20_260_830

OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "experiments/outputs-messk"))
CONFIGS = os.environ.get("CONFIGS", ",".join(MESSK_CONFIGS)).split(",")
ANALYSIS_COMMIT = os.environ.get("ANALYSIS_COMMIT", "unknown")


def replay(proc, states: np.ndarray, letters: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One tick of physics for a batch of candidates: new states and their tokens."""
    z = proc.system.kick(states, proc.actions[letters])
    tokens = np.empty((len(letters), proc.n_steps), dtype=np.int64)
    for step in range(proc.n_steps):
        z = proc.system.flow(z, proc.dt)
        tokens[:, step] = proc.observe(z)
    return z, tokens


def posterior_beliefs(proc, observed: np.ndarray, max_beam: int) -> tuple[np.ndarray, int, int]:
    """`E[belief | tokens]` after each tick, by exact consistency filtering.

    The beam holds one unnormalised forward vector per surviving letter prefix.
    Summing those vectors and normalising is exactly the posterior-weighted
    average of the per-prefix beliefs, because a prefix's posterior weight is
    its forward vector's own mass.
    """
    k = proc.chain.n_states
    joint = proc.chain.joint
    z = proc.system.initial_state(1)
    alpha = np.full((1, k), 1.0 / k)
    out = np.empty((proc.m, k))
    capped, widest = 0, 1

    for tick in range(proc.m):
        beam = len(z)
        candidates = np.repeat(z, k, axis=0)
        letters = np.tile(np.arange(k), beam)
        forward = np.einsum("pi,lij->plj", alpha, joint).reshape(beam * k, k)

        candidates, tokens = replay(proc, candidates, letters)
        target = observed[tick * proc.n_steps : (tick + 1) * proc.n_steps]
        keep = np.all(tokens == target, axis=1)
        if not keep.any():
            raise RuntimeError(f"no candidate reproduces tick {tick}; replay is not bit-exact")
        z, alpha = candidates[keep], forward[keep]
        widest = max(widest, len(z))
        if len(z) > max_beam:
            best = np.argsort(-alpha.sum(1))[:max_beam]
            z, alpha = z[best], alpha[best]
            capped += 1
        out[tick] = alpha.sum(0) / alpha.sum()
    return out, capped, widest


def oracle_entropy(proc, batch: dict, index: int) -> float:
    """Summed next-token entropy for a predictor that knows the state and belief.

    Only the first token of each tick is uncertain: inside a tick the flow is
    deterministic, so an oracle predicting it has zero entropy.
    """
    k = proc.chain.n_states
    emission = proc.chain.E
    letters = batch["letters"][index]
    z = proc.system.initial_state(1)
    total = 0.0
    for tick in range(proc.m):
        if tick > 0:
            belief = batch["beliefs"][index, (tick - 1) * proc.n_steps]
            letter_probs = belief @ emission
            branch = proc.system.kick(np.repeat(z, k, axis=0), proc.actions[np.arange(k)])
            first = proc.discretise(proc.system.observable(proc.system.flow(branch, proc.dt)))
            mass = np.bincount(first, weights=letter_probs, minlength=proc.n_obs)
            live = mass[mass > 0]
            total += float(-(live * np.log(live)).sum())
        z, _ = replay(proc, z, letters[tick : tick + 1])
    return total


def _r2(truth: np.ndarray, prediction: np.ndarray) -> float:
    residual = ((truth - prediction) ** 2).sum(axis=0)
    total = ((truth - truth.mean(axis=0)) ** 2).sum(axis=0)
    return float(np.mean(1.0 - residual / total))


def _demo() -> None:
    proc = make_process("pendulum_mess4")
    batch = proc.sample_batch(np.random.default_rng(1), 2)
    truth = batch["beliefs"][0, :: proc.n_steps]
    posterior, capped, _ = posterior_beliefs(proc, batch["tokens"][0], MAX_BEAM)
    assert capped == 0 and posterior.shape == truth.shape
    assert np.all(posterior >= -1e-12) and np.allclose(posterior.sum(axis=1), 1.0)


def main() -> None:
    _demo()
    results = {}
    print(f"n_seq={N_SEQ}  max_beam={MAX_BEAM}  configs={CONFIGS}", flush=True)

    for name in CONFIGS:
        started = time.perf_counter()
        proc = make_process(name)
        batch = proc.sample_batch(np.random.default_rng(EVAL_SEED), N_SEQ)
        truth = batch["beliefs"][:, :: proc.n_steps, :]

        posterior = np.empty_like(truth)
        capped_sequences, widest, entropy = 0, 1, 0.0
        for index in range(N_SEQ):
            posterior[index], capped, beam = posterior_beliefs(
                proc, batch["tokens"][index], MAX_BEAM
            )
            capped_sequences += capped > 0
            widest = max(widest, beam)
            entropy += oracle_entropy(proc, batch, index)

        flat_truth = truth.reshape(-1, truth.shape[-1])
        results[name] = {
            "ceiling_r2": _r2(flat_truth, posterior.reshape(-1, truth.shape[-1])),
            "oracle_loss": entropy / (N_SEQ * (proc.seq_len - 1)),
            "n_seq": N_SEQ,
            "max_beam": MAX_BEAM,
            "widest_beam": int(widest),
            "capped_sequences": int(capped_sequences),
            "evaluation_seed": EVAL_SEED,
            "analysis_commit": ANALYSIS_COMMIT,
            "wall_seconds": time.perf_counter() - started,
        }
        entry = results[name]
        print(
            f"  {name:<22} ceiling R2 {entry['ceiling_r2']:+.4f}   "
            f"oracle loss {entry['oracle_loss']:.4f}   "
            f"widest beam {entry['widest_beam']}   capped {entry['capped_sequences']}/{N_SEQ}",
            flush=True,
        )

    path = OUTPUT_DIR / "messk_07_ceiling.json"
    path.write_text(json.dumps(results, indent=2, default=float))
    print(f"\nwrote {path}", flush=True)


if __name__ == "__main__":
    main()
