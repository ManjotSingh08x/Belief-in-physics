"""Render the four highest-probability k=10 observation tokens from saved arrays."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

os.environ.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MPLBACKEND="Agg")
import numpy as np


def top4(p):
    assert p.shape[-1] == 181 and np.isfinite(p).all()
    assert p.min() >= 0 and np.allclose(p.sum(-1), 1, atol=2e-6)
    # Stable ordering breaks probability ties by ascending token ID.
    ids = np.argsort(-p, axis=-1, kind="stable")[..., :4]
    values = np.take_along_axis(p, ids, axis=-1)
    assert np.all(values[..., :-1] >= values[..., 1:])
    assert np.all(values.sum(-1) <= 1 + 2e-6)
    return ids, values


def self_check():
    p = np.zeros((2, 181))
    p[0, [13, 7, 99, 4, 2]] = [.4, .3, .15, .1, .05]
    p[1, :5] = .2
    ids, values = top4(p)
    assert ids.tolist() == [[13, 7, 99, 4], [0, 1, 2, 3]]
    assert np.allclose(values[0], [.4, .3, .15, .1])
    assert np.allclose(values.sum(-1), [.95, .8])  # No renormalization.


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    self_check()
    if args.self_check:
        print("Verified ranked token selection, ties, and original probability mass.")
        return
    assert args.input is not None
    source = args.input / "long_context_k10_probabilities.npz"
    expected = json.loads((args.input / "SHA256.json").read_text())
    for name in (source.name, "long_context_k10.json"):
        assert hashlib.sha256((args.input / name).read_bytes()).hexdigest() == expected[name]
    recipe = json.loads((args.input / "long_context_k10.json").read_text())
    assert recipe["k_lookahead"] == 10 and recipe["inference_context"] == 310
    with np.load(source, allow_pickle=False) as data:
        p = data["trained_random_probabilities"].copy()
        target = data["random_tokens"][:, 10:].copy()
        nll = data["trained_random_nll"].copy()
    assert p.shape == (4, 490, 181) and target.shape == (4, 490)
    assert np.allclose(np.take_along_axis(p, target[..., None], -1)[..., 0],
                       np.exp(-nll), rtol=1e-5, atol=1e-7)
    ids, values = top4(p)
    args.output.mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    colors = ["#44ff77", "#ffd43b", "#ff709f", "#73a5ff"]
    scores = []
    with (args.output / "predictions.csv").open("w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["path", "source_t", "target_t", "actual_token", "rank", "predicted_token", "probability"])
        for path in range(4):
            for t in range(490):
                for rank in range(4):
                    writer.writerow([path+1, t, t+10, int(target[path, t]), rank+1,
                                     int(ids[path, t, rank]), float(values[path, t, rank])])
            sparse = np.zeros_like(p[path])
            np.put_along_axis(sparse, ids[path], values[path], axis=-1)
            assert np.allclose(sparse.sum(-1), values[path].sum(-1))
            hit = (ids[path] == target[path, :, None]).any(-1)
            score = dict(path=path+1, mean_top4_mass=float(values[path].sum(-1).mean()),
                         top4_accuracy=float(hit.mean()), mean_nll_nats=float(nll[path].mean()))
            scores.append(score)
            fig, axes = plt.subplots(2, 1, figsize=(17, 8), sharex=True,
                                     gridspec_kw={"height_ratios": [3, 1]}, layout="constrained")
            image = axes[0].imshow(sparse.T, origin="lower", aspect="auto", cmap="magma", vmin=0, vmax=1,
                                   extent=(-.5, 489.5, -.5, 180.5))
            for rank, color in enumerate(colors):
                axes[0].scatter(np.arange(490), ids[path, :, rank], s=9, facecolors="none",
                                edgecolors=color, linewidths=.5)
                axes[1].plot(values[path, :, rank], color=color, linewidth=1, label=f"rank {rank+1}")
            axes[0].plot(target[path], color="cyan", linewidth=1, label="actual token at t+10")
            handles = [Line2D([], [], color=c, marker="o", markerfacecolor="none", linestyle="none",
                              label=f"rank {r+1}") for r, c in enumerate(colors)]
            handles.append(Line2D([], [], color="cyan", label="actual token at t+10"))
            axes[0].legend(handles=handles, loc="upper right", ncol=5, fontsize=8)
            axes[0].set(ylabel="predicted observation token ID", yticks=[0, 45, 90, 135, 180],
                        title=f"Sphere k=10 | randomized start {path+1} | four most probable tokens at each t\n"
                              f"Top-4 mass {score['mean_top4_mass']:.1%}; actual token in top 4 {score['top4_accuracy']:.1%}")
            fig.colorbar(image, ax=axes[0], label="original softmax probability (top 4 only)", pad=.01)
            axes[1].plot(values[path].sum(-1), color="black", linestyle=":", linewidth=1, label="top-4 total mass")
            axes[1].set(ylabel="probability", xlabel="source position t (prediction target is t+10)", ylim=(0, 1.02))
            axes[1].legend(loc="upper right", ncol=5, fontsize=8)
            for ax in axes:
                ax.axvline(309.5, color="gray", linestyle="--", linewidth=1)
            fig.savefig(args.output / f"k10_top4_random_start_{path+1}.png", dpi=150)
            plt.close(fig)
    report = dict(source_npz_sha256=expected[source.name], source_recipe_sha256=expected["long_context_k10.json"],
                  training_source_revision=recipe["training_source_revision"], model_name=recipe["model_name"],
                  configuration_sha256=recipe["configuration_sha256"], k=10, inference_context=310,
                  trajectory_tokens=500, selection="descending probability; ascending token ID on ties",
                  probabilities_renormalized=False, semantics="positionwise ranks, not four coherent physical branches",
                  initial_states=recipe["initial_states"]["random"], scores=scores)
    (args.output / "summary.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(scores))


if __name__ == "__main__":
    main()
