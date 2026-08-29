"""Named Mess-K configurations: a chain crossed with a physical system.

`mess3` is the canonical chain: three moods, alpha 0.7, x 0.15. It is kept as
the reference, because a result on Mess-4 is only readable against a Mess-3
number produced by the same pipeline. `mess4` is the same in every respect
except that the chain has four moods. The stay probability is held at 0.7 rather
than the per-neighbour `x`, so the chain's persistence -- and therefore how long
the belief takes to forget -- is matched between the two rather than drifting
with K.

The four systems span the range of how hard the observation makes the inference:
a pendulum whose angle tracks the impulse almost directly, a predator-prey
oscillator whose response is delayed by a whole cycle, a ball on a sphere whose
observable is one coordinate of a two-dimensional motion, and a chaotic double
pendulum where the trace of any single impulse is scrambled within a few ticks.
The chain, the belief, the binning and the model are identical across all four,
so a difference in the results is a difference in the physics.

Every letter maps to a distinct non-zero impulse `(l+1) * delta_v`. No
configuration has an action that leaves the state untouched.
"""

from __future__ import annotations

from .messk import MessDriven, MessKProcess
from .systems.double_pendulum import DoublePendulum
from .systems.pendulum import Pendulum
from .systems.predator_prey import PredatorPrey
from .systems.sphere import SphereBall

CHAINS: dict[str, dict] = {
    "mess3": {"n_states": 3, "alpha": 0.7, "stay": 0.7},
    "mess4": {"n_states": 4, "alpha": 0.7, "stay": 0.7},
}

#: system factory, plus the driver settings that system needs. `delta_v` and
#: `dt` are per-system because the systems run on different timescales and
#: different state scales; `m` and `n_steps` are held fixed so every run has the
#: same sequence length and the same number of chain ticks to infer from.
SYSTEMS: dict[str, dict] = {
    "pendulum": {"factory": Pendulum, "delta_v": 0.3, "dt": 0.02},
    "predator_prey": {"factory": PredatorPrey, "delta_v": 0.25, "dt": 0.05},
    "sphere": {"factory": SphereBall, "delta_v": 0.35, "dt": 0.04},
    "double_pendulum": {"factory": DoublePendulum, "delta_v": 0.4, "dt": 0.01},
}

DRIVER = {"m": 16, "n_steps": 10, "n_obs": 181}

#: Every (system, chain) pair, named "<system>_<chain>".
MESSK_CONFIGS: dict[str, tuple[str, str]] = {
    f"{s}_{c}": (s, c) for s in SYSTEMS for c in CHAINS
}


def make_process(name: str, **overrides) -> MessDriven:
    if name not in MESSK_CONFIGS:
        raise ValueError(f"unknown config {name!r}; have {sorted(MESSK_CONFIGS)}")
    system_name, chain_name = MESSK_CONFIGS[name]

    spec = dict(SYSTEMS[system_name])
    factory = spec.pop("factory")
    system = factory(**overrides.pop("system", {}))
    chain = MessKProcess(**{**CHAINS[chain_name], **overrides.pop("chain", {})})
    return MessDriven(chain=chain, system=system, **{**DRIVER, **spec, **overrides})


def _demo() -> None:
    p3 = make_process("pendulum_mess3")
    assert abs(p3.chain.x - 0.15) < 1e-12, "mess3 must reproduce their x=0.15"

    assert len(MESSK_CONFIGS) == 8, sorted(MESSK_CONFIGS)
    for name in MESSK_CONFIGS:
        proc = make_process(name)
        assert proc.n_obs == 181 and proc.seq_len == 160, name
        assert len(proc.kicks) == proc.chain.n_states, name
        assert all(k > 0 for k in proc.kicks), f"{name} has a no-op action"
        assert len(set(proc.kicks)) == len(proc.kicks), f"{name} repeats an impulse"

    p4 = make_process("double_pendulum_mess4")
    assert abs(p4.kicks[0] - 0.4) < 1e-12 and len(p4.kicks) == 4
    print(f"configs ok ({len(MESSK_CONFIGS)} runs, pendulum_mess3 kicks {p3.kicks})")


if __name__ == "__main__":
    _demo()
