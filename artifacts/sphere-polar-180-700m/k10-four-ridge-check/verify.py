"""Check branch bin mixtures, untouched native probabilities, alignment and hashes."""
import hashlib
import json
from pathlib import Path
import numpy as np
root = Path(__file__).resolve().parent
r = json.loads((root/"summary.json").read_text())
assert hashlib.sha256((root/"generate.py").read_bytes()).hexdigest() == r["script_sha256"]
with np.load(root/"ridge_arrays.npz", allow_pickle=False) as z:
    d = {k:z[k] for k in z.files}
with np.load(root.parent/"token-heatmaps-500-random-start"/"long_context_k10_probabilities.npz", allow_pickle=False) as z:
    assert np.array_equal(d["native_probabilities"], z["trained_random_probabilities"])
    assert np.array_equal(d["tokens"], z["random_tokens"])
with np.load(root.parent/"mood-heatmaps-500-k10"/"mood_arrays.npz", allow_pickle=False) as z:
    assert np.allclose(d["oracle_action_probabilities"], z["oracle"][:, :490] @ z["emission"])
    assert np.allclose(d["decoded_action_probabilities"], z["trained_projected"][:, :490] @ z["emission"])
assert np.array_equal(d["target_positions"], d["source_positions"]+10)
crossing = d["source_positions"]//20 != d["target_positions"]//20
assert np.array_equal(crossing, d["crosses_kick"])
assert np.array_equal(d["candidate_bins"], np.clip(np.rint((d["candidate_theta"]-.05)/1.2*180), 0, 180).astype(int))
for prefix in ("oracle", "decoded"):
    mass = np.zeros((4, 490, 181))
    for p in range(4):
        for t in range(490):
            for a in range(4):
                mass[p, t, d["candidate_bins"][p, t, a]] += d[prefix+"_action_probabilities"][p, t, a]
    assert np.allclose(mass, d[prefix+"_token_probabilities"])
    assert np.allclose(mass.sum(-1), 1, atol=2e-6)
allowed = np.zeros((4, 490, 181), bool)
np.put_along_axis(allowed, d["candidate_bins"], True, -1)
assert np.all(np.take_along_axis(allowed, d["tokens"][:, 10:, None], -1))
assert np.all(d["candidate_bins"][:, ~crossing] == d["tokens"][:, 10:, None][:, ~crossing])
value = float((d["native_probabilities"]*allowed).sum(-1)[:, crossing].mean())
assert value == r["scores"]["native_probability_on_physical_branches_crossing_mean"]
for name, value in json.loads((root/"SHA256.json").read_text()).items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest() == value
print("Verified source/target alignment, physical bins, all mixtures, untouched native predictions, branch scores and hashes.")
