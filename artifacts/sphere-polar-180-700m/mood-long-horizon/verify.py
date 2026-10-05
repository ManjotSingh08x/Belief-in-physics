"""Check long-horizon trajectories, separation metrics and CSV/metadata agreement."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

root = Path(__file__).resolve().parent
report = json.loads((root/"summary.json").read_text())
assert hashlib.sha256((root/"generate.py").read_bytes()).hexdigest() == report["script_sha256"]
source = root.parent/"mood-heatmaps-500-k10"/"mood_arrays.npz"
assert hashlib.sha256(source.read_bytes()).hexdigest() == report["input_mood_arrays_sha256"]
with np.load(source, allow_pickle=False) as z:
    short = z["action_branch_theta"]
    weights = z["trained_projected"][:, z["anchors"]]
with np.load(root/"long_horizon_arrays.npz", allow_pickle=False) as z:
    data = {key:z[key] for key in z.files}
assert np.array_equal(weights, data["mood_weights"])
assert np.array_equal(data["common_letters"][:, :25], data["original_letters"])
assert all(np.isfinite(value).all() for value in data.values())


def start(values, threshold):
    passing = values <= threshold
    # Independently check every suffix, ignoring agreements shorter than 101 samples.
    for i in range(len(values)-100):
        if passing[i:].all():
            return i
    return None


rows = list(csv.DictReader((root/"convergence.csv").open()))
assert len(rows) == 32
for protocol, checks in report["protocols"].items():
    states = data[protocol+"_states"]
    assert states.shape == (4, 4, 4, 2001, 4)
    assert np.array_equal(states[..., 0, :], np.repeat(data["start_states"][:, :, None], 4, 2))
    assert np.allclose(states[..., 1:11, 0], short, atol=1e-12)
    theta_spread = states[..., 0].max(2)-states[..., 0].min(2)
    theta, psi = states[..., 0], states[..., 1]
    xyz = np.stack([np.sin(theta)*np.cos(psi), np.sin(theta)*np.sin(psi), np.cos(theta)], -1)
    diameter = np.maximum.reduce([np.linalg.norm(xyz[:, :, i]-xyz[:, :, j], axis=-1)
                                  for i in range(4) for j in range(i+1, 4)])
    assert np.allclose(theta_spread, data[protocol+"_theta_spread"])
    assert np.allclose(diameter, data[protocol+"_xyz_diameter"])
    tokens = data[protocol+"_tokens"]
    assert np.array_equal(tokens, np.clip(np.rint((theta-.05)/1.2*180), 0, 180).astype(np.int64))
    assert checks["reference_max_theta_error"] < (1.2/180)/100
    for fork in checks["forks"]:
        path, ai = fork["path"]-1, np.flatnonzero(data["anchors"]+1 == fork["after_token"])[0]
        assert start(theta_spread[path, ai], 1.2/180) == fork["below_one_bin_through_remainder"]
        same = np.all(tokens[path, ai] == tokens[path, ai, :1], axis=0)
        assert start((~same).astype(int), 0) == fork["same_token_through_remainder"]
        assert float(theta_spread[path, ai, -1]) == fork["terminal_theta_spread"]
        assert np.isclose(diameter[path, ai, -1], fork["terminal_xyz_diameter"], atol=1e-12)
        row = next(r for r in rows if r["protocol"] == protocol and int(r["path"]) == fork["path"] and int(r["after_token"]) == fork["after_token"])
        for key, value in fork.items():
            assert row[key] == ("" if value is None else str(value))
for name, value in json.loads((root/"SHA256.json").read_text()).items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest() == value
print("Verified 32 forks, short-horizon agreement, polar/3D separation, sustained agreement, CSV numbers and file hashes.")
