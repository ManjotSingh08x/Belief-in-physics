"""Independently verify every exported token rank/probability against saved softmax."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

root = Path(__file__).resolve().parent
source = root.parent / "token-heatmaps-500-random-start"
report = json.loads((root / "summary.json").read_text())
assert hashlib.sha256((source / "long_context_k10_probabilities.npz").read_bytes()).hexdigest() == report["source_npz_sha256"]
assert hashlib.sha256((source / "long_context_k10.json").read_bytes()).hexdigest() == report["source_recipe_sha256"]
with np.load(source / "long_context_k10_probabilities.npz", allow_pickle=False) as data:
    p = data["trained_random_probabilities"]
    target = data["random_tokens"][:, 10:]
    nll = data["trained_random_nll"]
    expected = np.empty((4, 490, 4), dtype=int)
    for path in range(4):
        for t in range(490):
            expected[path, t] = sorted(range(181), key=lambda j: (-float(p[path, t, j]), j))[:4]
    rows = list(csv.DictReader((root / "predictions.csv").open()))
    assert len(rows) == 7840
    seen = set()
    for row in rows:
        path, t, rank = int(row["path"])-1, int(row["source_t"]), int(row["rank"])-1
        assert (path, t, rank) not in seen
        seen.add((path, t, rank))
        token = int(row["predicted_token"])
        assert token == expected[path, t, rank]
        assert int(row["target_t"]) == t+10 and int(row["actual_token"]) == target[path, t]
        assert float(row["probability"]) == float(p[path, t, token])
    assert seen == {(path, t, rank) for path in range(4) for t in range(490) for rank in range(4)}
    values = np.take_along_axis(p, expected, -1)
    for path, score in enumerate(report["scores"]):
        assert score["path"] == path+1
        assert score["mean_top4_mass"] == float(values[path].sum(-1).mean())
        assert score["top4_accuracy"] == float((expected[path] == target[path, :, None]).any(-1).mean())
        assert score["mean_nll_nats"] == float(nll[path].mean())
for name, digest in json.loads((root / "SHA256.json").read_text()).items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
print("Verified all 7,840 exported predictions, target alignment, ranks, original probabilities, scores and hashes.")
