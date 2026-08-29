"""Named Mess-K configurations.

`mess3` is the canonical case: three moods, alpha 0.7, x 0.15, impulses
(l+1)*0.3, theta rounded to whole degrees and clamped to +-90. It is kept as the
reference, because a result on Mess-4 is only readable against a Mess-3 number
produced by the same pipeline.

`mess4` is the same in every respect except that the chain has four moods. The
stay probability is held at 0.7 rather than the per-neighbour `x`, so the chain's
persistence -- and therefore how long the belief takes to forget -- is matched
between the two rather than drifting with K.
"""

from __future__ import annotations

from .messk import MessKProcess, MessPendulum

MESSK_CONFIGS: dict[str, dict] = {
    "mess3": {"n_states": 3, "alpha": 0.7, "stay": 0.7},
    "mess4": {"n_states": 4, "alpha": 0.7, "stay": 0.7},
}

PENDULUM = {"delta_v": 0.3, "m": 16, "n_steps": 10, "dt": 0.02, "gamma": 0.5, "g": 9.8}


def make_process(name: str, **overrides) -> MessPendulum:
    if name not in MESSK_CONFIGS:
        raise ValueError(f"unknown config {name!r}; have {sorted(MESSK_CONFIGS)}")
    chain = MessKProcess(**{**MESSK_CONFIGS[name], **overrides.pop("chain", {})})
    return MessPendulum(chain=chain, **{**PENDULUM, **overrides})


def _demo() -> None:
    p3, p4 = make_process("mess3"), make_process("mess4")
    assert abs(p3.chain.x - 0.15) < 1e-12, "mess3 must reproduce their x=0.15"
    assert p3.n_obs == p4.n_obs == 181
    assert p3.seq_len == p4.seq_len == 160
    assert len(p4.kicks) == 4 and abs(p4.kicks[0] - 0.3) < 1e-12
    print(f"configs ok (mess3 kicks {p3.kicks}, mess4 kicks {p4.kicks})")


if __name__ == "__main__":
    _demo()
