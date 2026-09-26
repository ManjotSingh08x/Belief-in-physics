"""The four Mess-4 physical systems.

Every system uses the same four-state chain: stay probability 0.7, and emission
accuracy 0.7. The exact predictive belief is therefore the same tetrahedral
object in all four experiments. What differs is the physical meaning of its four
letters:

- pendulum: strong negative, weak negative, weak positive, strong positive kick;
- predator-prey: prey down/up, predator down/up;
- spherical pendulum: north/south, west/east tangent kick;
- double pendulum: joint 1 negative/positive, joint 2 negative/positive.

Every action is distinct and non-zero. Each action set sums to zero, so no system
receives a directional drift merely because its HMM letters are uniform.
"""

from __future__ import annotations

import numpy as np

from .messk import MessDriven, MessKProcess
from .systems.double_pendulum import DoublePendulum
from .systems.pendulum import Pendulum
from .systems.predator_prey import PredatorPrey
from .systems.sphere import SphereBall

CHAIN = {"n_states": 4, "alpha": 0.7, "stay": 0.7}

#: `delta_v` is the weak pendulum kick and the cardinal-action magnitude for the
#: other systems. The pendulum value sqrt(0.3) gives the variance-matched ladder
#: {-1.64,-0.55,+0.55,+1.64}, whose mean squared impulse is 1.5.
#: Bin resolution for double pendulum: set to an int (e.g. 50 -> 50x50 = 2500) or tuple (e.g. (50, 50)).
#: Changing this one variable changes the double pendulum binning across the entire project.
DOUBLE_PENDULUM_BINS: int | tuple[int, int] = (50, 50)

SYSTEMS: dict[str, dict] = {
    "pendulum": {"factory": Pendulum, "delta_v": 0.5477225575051661, "dt": 0.02},
    "predator_prey": {"factory": PredatorPrey, "delta_v": 0.35, "dt": 0.05},
    "sphere": {"factory": SphereBall, "delta_v": 0.15, "dt": 0.04, "obs_bins": (30, 30)},
    "double_pendulum": {"factory": DoublePendulum, "delta_v": 1.2, "dt": 0.2, "obs_bins": DOUBLE_PENDULUM_BINS},
}

DRIVER = {"m": 16, "n_steps": 10, "obs_bins": 181}
MESSK_CONFIGS: dict[str, str] = {f"{system}_mess4": system for system in SYSTEMS}


def set_double_pendulum_bins(bins: int | tuple[int, int]) -> None:
    """Set double pendulum observation bins in a single line.

    Can be called directly in scripts or notebooks:
        set_double_pendulum_bins(25)        # 25x25 = 625 tokens
        set_double_pendulum_bins((20, 30))  # 20x30 = 600 tokens
    """
    global DOUBLE_PENDULUM_BINS
    norm_bins = (bins, bins) if isinstance(bins, int) else tuple(bins)
    DOUBLE_PENDULUM_BINS = norm_bins
    SYSTEMS["double_pendulum"]["obs_bins"] = norm_bins


def make_process(name: str, **overrides) -> MessDriven:
    if name not in MESSK_CONFIGS:
        raise ValueError(f"unknown config {name!r}; have {sorted(MESSK_CONFIGS)}")
    system_name = MESSK_CONFIGS[name]
    spec = dict(SYSTEMS[system_name])
    factory = spec.pop("factory")
    system = factory(**overrides.pop("system", {}))
    chain = MessKProcess(**{**CHAIN, **overrides.pop("chain", {})})
    cfg = {**DRIVER, **spec, **overrides}
    # For 2-channel systems (double_pendulum, sphere), allow passing an int B for symmetric (B, B) bins
    obs_bins = cfg.get("obs_bins")
    if isinstance(obs_bins, int) and system_name in ("double_pendulum", "sphere"):
        cfg["obs_bins"] = (obs_bins, obs_bins)
    return MessDriven(chain=chain, system=system, **cfg)


def _demo() -> None:
    assert len(MESSK_CONFIGS) == 4, sorted(MESSK_CONFIGS)
    for name in MESSK_CONFIGS:
        proc = make_process(name)
        actions = proc.actions
        assert proc.chain.n_states == actions.shape[0] == 4, name
        cfg_bins = SYSTEMS[MESSK_CONFIGS[name]].get("obs_bins", DRIVER["obs_bins"])
        if isinstance(cfg_bins, int) and MESSK_CONFIGS[name] in ("double_pendulum", "sphere"):
            cfg_bins = (cfg_bins, cfg_bins)
        expected_vocab = int(np.prod(cfg_bins)) if isinstance(cfg_bins, (tuple, list)) else int(cfg_bins)
        assert proc.n_obs == expected_vocab and proc.seq_len == 160, name
        assert (actions != 0).any(axis=1).all(), f"{name} contains a no-op"
        assert len({tuple(row) for row in actions}) == 4, f"{name} repeats an action"
        assert abs(actions.sum(axis=0)).max() < 1e-12, f"{name} is directionally biased"

    pendulum = make_process("pendulum_mess4").actions[:, 0]
    assert abs(pendulum[0] + 1.6431676725) < 1e-9
    assert abs(pendulum[1] + 0.5477225575) < 1e-9
    print(f"configs ok ({len(MESSK_CONFIGS)} Mess-4 systems, pendulum {pendulum.tolist()})")


if __name__ == "__main__":
    _demo()
