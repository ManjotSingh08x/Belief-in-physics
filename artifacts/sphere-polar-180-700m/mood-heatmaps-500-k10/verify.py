"""Independent numerical checks of mood readouts, probabilities, forecasts and exports."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

root = Path(__file__).resolve().parent
report = json.loads((root / "summary.json").read_text())
assert hashlib.sha256((root / "generate.py").read_bytes()).hexdigest() == report["script_sha256"]
input_path = root.parent / "token-heatmaps-500-random-start" / "long_context_k10_probabilities.npz"
assert hashlib.sha256(input_path.read_bytes()).hexdigest() == report["source_input_sha256"]
with np.load(root / "mood_arrays.npz", allow_pickle=False) as z:
    data = {key:z[key] for key in z.files}
with np.load(input_path, allow_pickle=False) as z:
    assert np.array_equal(data["tokens"], z["random_tokens"])
    assert np.array_equal(data["letters"], z["random_letters"])
    assert np.array_equal(data["initial_states"], z["random_initial_states"])
assert all(np.isfinite(value).all() for value in data.values())
belief = np.full((4, 4), .25)
oracle = []
for tick in range(25):
    belief = (belief * data["emission"][:, data["letters"][:, tick]].T) @ data["transition"]
    belief /= belief.sum(-1, keepdims=True)
    oracle.append(belief.copy())
assert np.allclose(data["oracle"], np.repeat(np.stack(oracle, axis=1), 20, axis=1))
for tag in ("trained", "random_init"):
    features = data[tag+"_features"].copy()
    features -= data[tag+"_probe_mean"]
    features /= data[tag+"_probe_scale"]
    raw = features @ data[tag+"_probe_coef"].T + data[tag+"_probe_intercept"]
    assert np.allclose(raw, data[tag+"_raw"], atol=2e-6, rtol=2e-6)
    # Independent simplex projection by solving sum(max(raw-theta,0))=1.
    lo = raw.min(-1, keepdims=True)-1; hi = raw.max(-1, keepdims=True)
    for _ in range(50):
        mid = (lo+hi)/2
        over = np.maximum(raw-mid, 0).sum(-1, keepdims=True) > 1
        lo, hi = np.where(over, mid, lo), np.where(over, hi, mid)
    projected = np.maximum(raw-(lo+hi)/2, 0)
    assert np.allclose(projected, data[tag+"_projected"], atol=3e-6)
    assert np.all(data[tag+"_projected"] >= 0) and np.allclose(data[tag+"_projected"].sum(-1), 1, atol=1e-6)
    y = data["oracle"][:, 19::20].reshape(-1, 4)
    p = raw[:, 19::20].reshape(-1, 4)
    r2 = np.mean(1-((y-p)**2).sum(0)/((y-y.mean(0))**2).sum(0))
    assert np.isclose(r2, report["scores"][tag]["tick_end_500_raw_r2"], atol=2e-6)
    assert abs(report["scores"][tag]["original_test_r2"]-report["scores"][tag]["expected_original_test_r2"]) < 1e-4
mean = np.einsum("ml,palh->pamh", data["emission"], data["action_branch_theta"])
assert np.allclose(mean, data["mood_conditional_mean_theta"])
mass = np.zeros_like(data["mood_conditional_token_mass"])
for path in range(4):
    for ai, t in enumerate(data["anchors"]):
        actual_action = data["letters"][path, (t+1)//20]
        assert np.array_equal(data["action_branch_tokens"][path, ai, actual_action], data["tokens"][path, t+1:t+11])
        for mood in range(4):
            for action in range(4):
                for step in range(10):
                    mass[path, ai, mood, step, data["action_branch_tokens"][path, ai, action, step]] += data["emission"][mood, action]
assert np.allclose(mass, data["mood_conditional_token_mass"]) and np.allclose(mass.sum(-1), 1)
rows = list(csv.DictReader((root / "mood_probabilities.csv").open()))
assert len(rows) == 2000
for index, row in enumerate(rows):
    path, t = divmod(index, 500)
    assert int(row["path"]) == path+1 and int(row["observed_token_count"]) == t+1
    for key in ("oracle", "trained_raw", "trained_projected", "random_init_raw", "random_init_projected"):
        for mood in range(4):
            assert float(row[f"{key}_mood_{mood}"]) == float(data[key][path, t, mood])
for name, value in json.loads((root / "SHA256.json").read_text()).items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == value
print("Verified oracle filtering, frozen probe weights, projections, mood/action mixtures, actual branch replay, 2000 CSV rows and hashes.")
