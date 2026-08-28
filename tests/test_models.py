"""Run each model module's `_demo()` self-check under pytest.

The demos already assert the properties that matter (causality, exact linearity
of the feature map, chunking invariance, probe recovery and collapse). Wiring
them in here means a regression fails the suite rather than only failing when
someone runs the module by hand.
"""

import pytest

from models import analysis, features, probe, transformer


@pytest.mark.parametrize("module", [transformer, features, probe, analysis], ids=lambda m: m.__name__)
def test_module_self_check(module):
    module._demo()
