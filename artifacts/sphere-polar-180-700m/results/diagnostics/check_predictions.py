"""Inspect saved predictions without fitting or running the models again."""
import hashlib
import json
import os
import resource
from pathlib import Path
os.environ["OPENBLAS_NUM_THREADS"] = "1"
resource.setrlimit(resource.RLIMIT_AS, (12 * 1024**3, 12 * 1024**3))
import numpy as np
from sklearn.metrics import r2_score

root = Path(__file__).resolve().parent
summary = json.loads((root / "diagnostic_summary.json").read_text())
results = []
for case in summary["results"]:
    label = case["selection"]
    with np.load(root / (label + "_heldout_predictions.npz"), allow_pickle=False) as data:
        assert all(np.isfinite(data[key]).all() for key in data.files)
        assert np.allclose(data["true_belief"].sum(-1), 1)
        # stay=.7 and off-diagonal transition=.1 imply predictive mood bounds [.1,.7].
        assert data["true_belief"].min() >= .1 - 1e-12 and data["true_belief"].max() <= .7 + 1e-12
        row = {"selection": label, "all_saved_arrays_finite": True, "oracle_predictive_bounds_passed": True,
               "models": {}}
        for tag in ("trained", "random_init"):
            scores = {}
            for target in ("belief", "physics"):
                truth, prediction = data["true_" + target], data[tag + "_" + target]
                assert truth.shape == prediction.shape
                scores[target + "_r2_by_tick"] = [float(r2_score(truth[:, t], prediction[:, t])) for t in range(16)]
            raw = data[tag + "_belief"]
            scores["raw_entries_outside_predictive_bounds_fraction"] = float(((raw < .1) | (raw > .7)).mean())
            scores["raw_rows_outside_predictive_bounds_fraction"] = float(((raw < .1) | (raw > .7)).any(-1).mean())
            row["models"][tag] = scores
        results.append(row)
report = {"source_revision": summary["source_revision"], "fresh_eval_seed": summary["fresh_eval_seed"],
          "check_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          "predictive_mood_probability_bounds": [.1, .7], "results": results,
          "interpretation": "Terminal token features are not directly supervised by the k-ahead objective; time-resolved scores qualify pooled R2. Probability simplex projection alone does not enforce predictive mood bounds."}
(root / "saved_prediction_checks.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
print("Verified finite prediction arrays, predictive belief bounds and all 16 time-resolved scores.")
