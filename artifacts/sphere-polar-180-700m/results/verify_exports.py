"""Check the consolidated campaign exports with Python's standard library."""
import csv
import hashlib
import json
import math
import statistics
import zipfile
from pathlib import Path

p = Path(__file__).resolve().parent
rows = list(csv.DictReader((p / "probe_results.csv").open()))
report = json.loads((p / "campaign_summary.json").read_text())
assert len(rows) == 2160 and len({r["model_name"] for r in rows}) == 180
assert all(r["source_revision"] == report["source_revision"] for r in rows)
assert all(math.isfinite(float(r[k])) for r in rows for k in ("test_r2", "validation_r2"))
for item in report["summary"]:
    selected = [r for r in rows if r["mode"] == item["readout"] and r["target"] == item["target"] and r["model_type"] == item["model_type"] and (item["lookahead_mode"] == "all" or r["lookahead_mode"] == item["lookahead_mode"])]
    values = [float(r["test_r2"]) for r in selected]
    assert len(values) == item["count"]
    assert math.isclose(statistics.mean(values), item["mean_r2"], abs_tol=1e-12)
    assert min(values) == item["min_r2"] and max(values) == item["max_r2"]
for item in report["paired_gains"]:
    selected = [r for r in rows if r["mode"] == item["readout"] and r["target"] == item["target"] and (item["lookahead_mode"] == "all" or r["lookahead_mode"] == item["lookahead_mode"])]
    scores = {(r["model_name"], r["model_type"]): float(r["test_r2"]) for r in selected}
    names = {r["model_name"] for r in selected}
    gains = [scores[n, "trained"] - scores[n, "random_init"] for n in names]
    assert len(gains) == item["count"] and sum(g > 0 for g in gains) == item["improved_count"]
    assert math.isclose(statistics.mean(gains), item["mean_gain"], abs_tol=1e-12)
metadata_hashes = json.loads((p / "metadata/SHA256.json").read_text())
assert len(metadata_hashes) == 180
for name, digest in metadata_hashes.items():
    file = p / "metadata" / name
    assert hashlib.sha256(file.read_bytes()).hexdigest() == digest
    meta = json.loads(file.read_text())
    assert meta["source_revision"] == report["source_revision"] and meta["tokens_seen"] == 699924480
    assert meta["training"]["seed"] == 0 and meta["model"]["d_model"] == 128
    assert meta["model"]["n_layers"] == 4 and meta["model"]["n_heads"] == 2 and meta["model"]["d_mlp"] == 512
    assert all(math.isfinite(meta["final_loss"][key]) for key in ("train_loss", "eval_loss"))
    selected = [r for r in rows if r["model_name"] == file.stem]
    assert len(selected) == len(meta["probes"]) == 12
    for probe in meta["probes"]:
        row = next(r for r in selected if all(str(probe[k]) == r[k] for k in ("mode", "target", "model_type")))
        assert row["configuration_sha256"] == meta["configuration_sha256"]
        assert all(math.isclose(float(row[k]), float(probe[k]), abs_tol=1e-12)
                   for k in ("alpha", "validation_r2", "test_r2", "feature_dim", "train_rows_per_feature"))
diagnostics = p / "diagnostics"
if (diagnostics / "SHA256.json").exists():
    for name, digest in json.loads((diagnostics / "SHA256.json").read_text()).items():
        assert hashlib.sha256((diagnostics / name).read_bytes()).hexdigest() == digest
    ranking = list(csv.DictReader((diagnostics / "gap_ranking.csv").open()))
    assert len(ranking) == 360
    for r in ranking:
        assert math.isclose(float(r["gap"]), float(r["trained_r2"]) - float(r["random_r2"]), abs_tol=1e-12)
        for tag, key in (("trained", "trained_r2"), ("random_init", "random_r2")):
            original = next(v for v in rows if v["model_name"] == r["model_name"] and v["target"] == r["target"]
                            and v["model_type"] == tag and v["mode"] == "all_block_layers_single_token")
            assert float(original["test_r2"]) == float(r[key])
    diagnostic = json.loads((diagnostics / "diagnostic_summary.json").read_text())
    assert hashlib.sha256((diagnostics / "analyze.py").read_bytes()).hexdigest() == diagnostic["diagnostic_script_sha256"]
    assert len(diagnostic["results"]) == 3 and diagnostic["fresh_eval_seed"] == 20261002
    for case in diagnostic["results"]:
        assert case["fresh_test_trajectories"] >= 1800 and case["true_belief_simplex_and_letter_alignment_passed"]
        for model in case["models"].values():
            assert model["causal_future_perturbation_max_error"] <= 1e-5
            assert all(model[target]["reproduction_abs_error"] < 1e-4 for target in ("belief", "physics"))
    saved = json.loads((diagnostics / "saved_prediction_checks.json").read_text())
    assert hashlib.sha256((diagnostics / "check_predictions.py").read_bytes()).hexdigest() == saved["check_script_sha256"]
    assert len(saved["results"]) == 3 and all(v["all_saved_arrays_finite"] and v["oracle_predictive_bounds_passed"] for v in saved["results"])
    for file in diagnostics.glob("*.npz"):
        with zipfile.ZipFile(file) as archive:
            assert archive.testzip() is None
source = json.loads((p / "SOURCE_SNAPSHOT.json").read_text())
assert source["source_revision"] == report["source_revision"]
assert hashlib.sha256((p / "source_snapshot.zip").read_bytes()).hexdigest() == source["archive_sha256"]
with zipfile.ZipFile(p / "source_snapshot.zip") as archive:
    assert set(archive.namelist()) == set(source["members"]) and archive.testzip() is None
    for name, digest in source["members"].items():
        assert hashlib.sha256(archive.read(name)).hexdigest() == digest
if (p / "PUBLISH_SHA256.json").exists():
    for name, digest in json.loads((p / "PUBLISH_SHA256.json").read_text()).items():
        assert hashlib.sha256((p / name).read_bytes()).hexdigest() == digest
print("Verified 180 models, 2160 rows, metadata hashes, CSV agreement, summaries and available diagnostics.")
