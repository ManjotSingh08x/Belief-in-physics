"""Combine each 500-token mood heatmap with its four conditional forecast fans."""
import argparse
import csv
import os
from pathlib import Path
os.environ.update(OPENBLAS_NUM_THREADS="1", MPLBACKEND="Agg")
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=Path, required=True, help="Original token heatmap packet")
args = parser.parse_args()
root = Path(__file__).resolve().parent
with np.load(root / "mood_arrays.npz", allow_pickle=False) as z:
    data = {key:z[key] for key in z.files}
with np.load(args.input / "long_context_k10_probabilities.npz", allow_pickle=False) as z:
    theta = z["random_observable"][..., 0]
colors = ["#3690ff", "#ff9233", "#26af65", "#ba59d4"]
keys = ["oracle", "trained_raw", "trained_projected", "random_init_raw", "random_init_projected"]
with (root / "mood_probabilities.csv").open("w", newline="") as f:
    writer = csv.writer(f, lineterminator="\n")
    writer.writerow(["path", "observed_token_count"]+[f"{key}_mood_{m}" for key in keys for m in range(4)])
    for path in range(4):
        for t in range(500):
            writer.writerow([path+1, t+1]+[float(data[key][path, t, m]) for key in keys for m in range(4)])
for path in range(4):
    fig = plt.figure(figsize=(15, 12), layout="constrained")
    grid = fig.add_gridspec(3, 2, height_ratios=[.6, 1, 1])
    ax = fig.add_subplot(grid[0, :])
    im = ax.imshow(data["trained_projected"][path].T, cmap="magma", vmin=0, vmax=1,
                   aspect="auto", interpolation="nearest", extent=(.5, 500.5, 3.5, -.5))
    ax.set(yticks=range(4), yticklabels=[f"Mood {m}" for m in range(4)], xlabel="observed token count",
           title="Frozen transformer probe · predictive next-tick mood · simplex projected")
    ax.axvline(320.5, color="cyan", linestyle="--", linewidth=1)
    for t in data["anchors"]:
        ax.axvline(t+1, color="white", linestyle=":", linewidth=.8)
    fig.colorbar(im, ax=ax, label="Projected mood weight", pad=.01)
    for ai, t in enumerate(data["anchors"]):
        ax = fig.add_subplot(grid[1+ai//2, ai % 2])
        for mood, color in enumerate(colors):
            action = data["dominant_action_by_mood"][mood]
            b = data["trained_projected"][path, t, mood]
            future = np.r_[theta[path, t], data["action_branch_theta"][path, ai, action]]
            mean = np.r_[theta[path, t], data["mood_conditional_mean_theta"][path, ai, mood]]
            ax.plot(np.arange(11), future, color=color, linewidth=2,
                    label=f"Mood {mood}: weight {b:.1%}; dominant kick 95%")
            ax.plot(np.arange(11), mean, color=color, linestyle=":", linewidth=1)
        ax.plot(np.arange(11), theta[path, t:t+11], color="black", linestyle="--", label="actual continuation")
        ax.set(title=f"Fork after token {t+1} from the same state", xlabel="future observation offset", ylabel="polar angle (rad)")
        ax.legend(fontsize=8)
    fig.suptitle(f"Sphere k=10 · randomized path {path+1} · 500 observed tokens\n"
                 "Four mood-conditioned continuations: solid = dominant action; dotted = emission-weighted mean\n"
                 "Transformer probe supplies mood weights; physics replay uses known simulator state")
    fig.savefig(root / f"path_{path+1}_moods_and_trajectories.png", dpi=150)
    plt.close(fig)
print("Rendered four combined mood/trajectory panels and exported 2000 mood rows.")
