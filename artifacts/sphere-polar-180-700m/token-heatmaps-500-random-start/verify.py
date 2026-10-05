"""Verify heatmap arrays, numerical scores, target alignment and published hashes."""
import hashlib
import json
import os
import zipfile
from pathlib import Path
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import numpy as np

root = Path(__file__).resolve().parent
report = json.loads((root / "summary.json").read_text())
assert report["trajectory_tokens"] == 500 and len(report["results"]) == 3
assert hashlib.sha256((root / "plot_tokens.py").read_bytes()).hexdigest() == report["script_sha256"]
for case in report["results"]:
    archive = root / (case["label"] + "_probabilities.npz")
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
    with np.load(archive, allow_pickle=False) as data:
        assert np.array_equal(data["fixed_letters"], data["random_letters"])
        assert not np.array_equal(data["fixed_initial_states"], data["random_initial_states"])
        assert case["inference_context"] == case["architecture"]["n_ctx"] - case["k_lookahead"]
        for tag in ("trained", "random_init"):
            for mode in ("fixed", "random"):
                tokens = data[mode + "_tokens"]
                p = data[f"{tag}_{mode}_probabilities"]
                nll = data[f"{tag}_{mode}_nll"]
                target = tokens[:, case["k_lookahead"]:]
                assert tokens.shape == (4, 500) and p.shape == (*target.shape, 181)
                assert np.isfinite(p).all() and np.isfinite(nll).all()
                assert p.min() >= 0 and p.max() <= 1 and np.allclose(p.sum(-1), 1, atol=2e-6)
                actual_p = np.take_along_axis(p, target[..., None], axis=-1)[..., 0]
                assert np.allclose(actual_p, np.exp(-nll), rtol=1e-5, atol=1e-7)
                score = case["scores"][tag][mode]
                assert np.isclose(nll.mean(), score["mean_nll_nats"], atol=1e-6)
                assert np.allclose(nll.mean(-1), score["path_nll_nats"], atol=1e-6)
                assert np.isclose((p.argmax(-1) == target).mean(), score["top1_accuracy"], atol=1e-12)
    assert len(list(root.glob(case["label"] + "_random_start_*.png"))) == 4
for name, digest in json.loads((root / "SHA256.json").read_text()).items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
print("Verified 12 random-start heatmaps, 3 control panels, probability sums, k-ahead targets, NLL and file hashes.")
