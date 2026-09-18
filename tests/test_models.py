"""Run each model module's `_demo()` self-check under pytest.

The demos already assert the properties that matter (causality, exact linearity
of the feature map, chunking invariance, probe recovery and collapse). Wiring
them in here means a regression fails the suite rather than only failing when
someone runs the module by hand.
"""

import numpy as np
import pytest

from models import analysis, features, probe, transformer
from models.probe_extra import sparse_token_window_features


@pytest.mark.parametrize("module", [transformer, features, probe, analysis], ids=lambda m: m.__name__)
def test_module_self_check(module):
    module._demo()


def test_best_layer_is_selected_on_validation_not_test():
    records = [
        {
            "name": "validation_winner",
            "r2_by_group_validation": {"belief": 0.8},
            "r2_by_group": {"belief": 0.1},
        },
        {
            "name": "test_winner",
            "r2_by_group_validation": {"belief": 0.2},
            "r2_by_group": {"belief": 0.9},
        },
    ]
    assert analysis.best_layer(records, "belief")["name"] == "validation_winner"


def test_sparse_token_window_is_ordered_causal_and_has_tick_phase():
    tokens = np.array([[0, 1, 2, 3], [3, 2, 1, 0]])
    features = sparse_token_window_features(tokens, vocab_size=4, window=3, steps_per_tick=2)
    assert features.shape == (8, 14)
    dense = features.toarray().reshape(2, 4, 14)
    assert dense[0, 3, 3] == 1.0
    assert dense[0, 3, 4 + 2] == 1.0
    assert dense[0, 3, 8 + 1] == 1.0
    assert dense[0, 3, 12 + 1] == 1.0
    assert dense[0, 0, 4:12].sum() == 0.0
