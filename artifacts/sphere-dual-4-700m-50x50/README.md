# Sphere joint-angle experiment: four models, 700M tokens each

All four models completed **699,904,000 actual input tokens** (700M requested). Four CPU analyses and exports passed local ZIP/member SHA256 and CSV/metadata numerical-agreement checks. Vast instance **54348163 is destroyed**, independently confirmed absent with zero volumes. Guarded elapsed rental cost including the reserved prior rental and transfer allowances was **$1.82**, within the $5 cap; this is a conservative budget calculation, not a final provider invoice.

## Joint 3D probability views

First run `python3 reproduction/restore_large_files.py` from this packet directory (or any directory; the script finds its packet root). It verifies and reconstructs the original larger files from ≤8 MiB parts, preserving existing files with different contents. Then open [the 3D index](joint3d/index.html), choose a setting and held-out path, then rotate the surface or move the token-position slider. Each standalone HTML compares trained and matched random weights: **x = theta bin, y = circular azimuth bin, z = joint probability**. All 2,500 probabilities are displayed at each query. The shared peak-height/colour zoom is labelled and optional; numerical probabilities are unchanged. The HTML includes its plotting library and works offline. GitHub displays HTML source; download/open the file in a browser.

Cyan is the actual future joint token. Coloured markers are physical four-action endpoints recomputed from the actual current state. They are **not four neural mood-conditioned rollouts**. Azimuth bins 49 and 0 are neighbours. Native predictions are teacher forced. The 500-token sequences provide directly supervised source queries **0–489**, targeting tokens **10–499**.

## Largest trained-minus-random gaps

Comparisons below use the validation-selected ridge probe on all four transformer blocks. The belief target is the action-conditioned four-mood oracle belief; the physics target is two tangent velocities.

| gamma | delta_v | alpha | Belief R² trained | random | gap | Physics R² trained | random | gap |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.7 | 0.3 | 0.95 | 0.472215 | 0.122496 | 0.349719 | 0.926326 | 0.195472 | 0.730854 |
| 0.35 | 0.2 | 0.85 | 0.291352 | 0.040848 | 0.250504 | 0.930163 | 0.210649 | 0.719514 |
| 0.35 | 0.2 | 0.95 | 0.324086 | 0.054489 | 0.269596 | 0.934272 | 0.222767 | 0.711506 |
| 0.7 | 0.2 | 0.95 | 0.423086 | 0.101326 | 0.321760 | 0.952241 | 0.295743 | 0.656497 |

The widest gaps in this four-setting experiment were at **n=20, gamma=.7, delta_v=.3, alpha=.95**: belief **+0.349719 R²** and physics **+0.730854 R²**. This does not establish an optimum across other settings or seeds.

## Held-out native prediction quality

Each setting uses 2,048 fresh trajectories, with grouped trajectory splits and exclusion of duplicate causal prefixes across splits. Native metrics are measured on 200,900 held-out positions per model. NLL is in nats; lower NLL and Brier are better.

| gamma | delta_v | alpha | NLL trained | random | Accuracy trained | random | Brier trained | ECE trained |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.7 | 0.3 | 0.95 | 2.011777 | 7.991271 | 35.449% | 0.039% | 0.773739 | 0.058184 |
| 0.35 | 0.2 | 0.85 | 2.008832 | 7.994362 | 32.880% | 0.037% | 0.789522 | 0.045349 |
| 0.35 | 0.2 | 0.95 | 1.909311 | 7.994251 | 36.272% | 0.042% | 0.765699 | 0.051398 |
| 0.7 | 0.2 | 0.95 | 1.660135 | 7.993669 | 42.302% | 0.041% | 0.707955 | 0.037534 |

The lowest measured native NLL is **1.660135** at gamma=.7, delta_v=.2, alpha=.95, with **42.302%** joint-token accuracy. The setting with the largest probe gap is a different setting.

## Validity checks and limitations

- Saved matched random weights were captured before optimization and independently hash checked. Training source, recipes, model weights and exports retain separate digests. No old polar-only models or failed partial attempts are counted as completed joint-angle models.
- Exactly one training seed (seed0) per setting. Evaluation trajectories and physical stress seeds do not replace training-seed replication; no uncertainty intervals or optimal-performance claim are made.
- No cross-split duplicate causal-prefix overlap was retained. Full-context versus prefix-only logits differed by at most 6.68e-6 for trained models; native probability-sum errors were at most 4.42e-6.
- Shuffled-fit whole-trajectory controls produced negative belief R² in every setting. Physics controls retained small positive trained R² (0.041–0.114), much lower than matched probes (0.926–0.952). Whole-trajectory shuffling retains tick alignment and can retain shared temporal structure; it is not a universal leakage proof.
- Linear belief readouts are not native probability heads. Trained readouts had 1.12–3.15% of raw entries outside [0,1], and raw sum errors up to .0525. Four-mood heatmaps explicitly use simplex projection; raw R² is reported separately. They do not prove calibrated HMM filtering.
- Trained native ECE is .0375–.0582. The nearly uniform random model has a smaller ECE because its confidence and accuracy are both tiny; that does not imply useful prediction quality. Calibration remains imperfect.
- Physics selection screened 90 coarse configurations; the selected four passed fixed finer-step, extra-seed and 4,096-trajectory stress checks (4,464 trajectories / 14 seeds per setting). Starts are uniform coordinate offsets [-.1,.1] for theta/dtheta/dpsi plus independent uniform azimuth [-pi,pi). This is measured physical safety under that audited distribution, not a global stability theorem.
- Rejected unbounded Gaussian starts caused rare clipping/momentum drift in the larger audit. Rejected candidates and that failure are preserved under negative_evidence. No safety threshold was relaxed. Neutral azimuth remains relevant, so theta contraction does not establish full spatial convergence.
- Physical counterfactual endpoints may share bins, and the neural distribution need not form four large peaks. Neither peak creation nor smoothing was applied. These figures do not establish long-horizon neural rollout convergence.

## Contents and reproduction

- `models/`: four full trained `.pth` weights, four saved random controls, final metadata and ready markers.
- `exports/`: the four original verified ZIPs, including probability arrays, held-out paths, probe CSVs, plots and analysis metadata. `analysis/` also exposes plots, CSVs and metadata for convenient reading.
- `joint3d/`: 16 standalone 3D views (four held-out paths per setting), input/output manifests and a preview.
- `reproduction/`: immutable training source archive, exact CPU analysis script, 3D exporter, post-run controller release fix and CPU environment versions.
- `evidence/` and `negative_evidence/`: source/recipe/hash/billing checks and rejected-screen/audit evidence. Eight 250M/500M partial snapshot files remain independently verified locally and on staging; their hashes and metadata are included here, but they are not the four full models.
- `LARGE_FILES.json` and `large-parts/`: lossless storage of four ZIPs and 16 standalone HTML files under the repository’s 10 MB per-file limit. `PACKET_MANIFEST.json` lists every stored file plus the reconstructed originals; no probability precision was changed.

Training revision: `98d5b417fc3a162cd61f45a371e0209783261ab9`. Training archive SHA256: `28c7b4183b4d66996a01cb514f99796e1cb0378d464802154219e473a2a1f427`. CPU analysis revision: `1411bf9`, script SHA256: `b5da1dd21570954a191f998123e5bfded59e7077f6b1b5d4f16d67080db76d06`. 3D exporter revision: `450bb61`; source and saved-probability digests are in each 3D manifest.

From this packet directory, extract one ZIP and regenerate its 3D pages using NumPy and the included Plotly bundle:

```sh
python3 reproduction/restore_large_files.py
mkdir extracted
unzip exports/sphere_dual50_c0_n20_g0.7_dv0.3_a0.95_d128l4h2mlp512_700m_seed0_k10.zip -d extracted/c0
cp analysis/sphere_dual50_c0_n20_g0.7_dv0.3_a0.95_d128l4h2mlp512_700m_seed0_k10/EXPORT_VERIFIED.json extracted/c0/EXPORT_VERIFIED.json
python reproduction/sphere_dual_joint3d.py --self-check
python reproduction/sphere_dual_joint3d.py --analysis extracted/c0 --output regenerated --plotly-js joint3d/plotly-4.1.1.min.js
```

For CPU probe reproduction, unpack `reproduction/source.tar.gz` into a fresh directory, set `PYTHONPATH` to it, and run `reproduction/sphere_dual_analyze.py --screen evidence/screen.json --results models --output new-analysis --index 0 --training-revision 98d5b417fc3a162cd61f45a371e0209783261ab9 --trajectories 2048` using the versions in `reproduction/CPU_ENVIRONMENT.json`. Repeat indices 1–3. This reads the completed weights; no new training or rental is required.

The original release endpoint returned HTTP308. The exact instance was successfully destroyed via the canonical trailing-slash endpoint and independently checked absent with zero volumes; `reproduction/sphere_dual_control.py` contains that one-line repair. The frozen training archive was not rewritten.
