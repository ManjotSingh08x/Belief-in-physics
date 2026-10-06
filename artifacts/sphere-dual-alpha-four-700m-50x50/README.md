# Sphere joint-angle alpha ablation

Four models completed 699,904,000 input tokens each on Kratos. All four CPU exports, original ZIP/member hashes and CSV/metadata values were verified independently. All 16 interactive 3D pages were verified locally: their probabilities exactly match the saved float32 predictions.

## Results

These are test R² from validation-selected ridge probes on held-out trajectories, using concatenated activations from all four blocks at a single supervised tick. The physics target is the two tangent velocities. Native metrics evaluate the joint-token head; NLL is in nats.

| Alpha | Belief trained R² | Belief random R² | Belief gap | Velocity trained R² | Velocity random R² | Velocity gap | Native NLL | Native accuracy | Native ECE |
|---|---|---|---|---|---|---|---|---|---|
| 0.50 | 0.306217 | 0.063504 | 0.242713 | 0.927406 | 0.180246 | 0.747160 | 2.254252 | 0.277432 | 0.048355 |
| 0.65 | 0.341546 | 0.078854 | 0.262692 | 0.925521 | 0.179425 | 0.746096 | 2.222514 | 0.282768 | 0.044978 |
| 0.75 | 0.389490 | 0.089650 | 0.299841 | 0.929248 | 0.175142 | 0.754106 | 2.169254 | 0.297287 | 0.042812 |
| 0.85 | 0.431029 | 0.102651 | 0.328378 | 0.925469 | 0.182276 | 0.743193 | 2.121581 | 0.315231 | 0.046949 |

Alpha 0.85 has the largest measured belief gap in this set (+0.328378); alpha 0.75 has the largest velocity gap (+0.754106). One training seed was used per model, so training variance and significance are unknown. All trained shuffled-fit belief scores are negative. Shuffled-fit velocity scores remain small but positive (0.039–0.051); every control is retained in `probes.csv`. The belief gaps here are below the previously published alpha 0.95 setting with the same gamma and kick; this is a descriptive comparison, not a multi-seed significance claim.

## Heatmaps and branching

Each model includes native joint-token, theta and circular-azimuth heatmaps for held-out randomized paths 1340, 436, 1659 and 910, four-mood probe heatmaps, and matched random controls. [Open the joint 3D index](joint3d/index.html) after restoring the standalone HTML files. Each view has theta bin, azimuth bin and joint probability axes, with a source-position slider.

Inspection of all 16 paths in both angle marginals shows a dominant ridge with localized widening and faint secondary bands. Consistently clear four-way high-probability splits are not established by these displays. Inspect crossing windows 10–19, 30–39, … before impulses at 20, 40, … . Repeated azimuth diagonals are circular wrapping. This qualitative finding uses existing predictions only; no additional neural inference, smoothing or artificial peaks were used.

The native head predicts the joint token at t+10 from observed tokens through t. Four-mood panels are linear-probe/simplex readouts. Dashed curves and 3D action markers are physical counterfactual endpoints recomputed from the actual current state; they are not neural mood-conditioned rollouts. Marginals can merge distinct joint outcomes and physical endpoints can share bins. Full spatial convergence is unproven.

## Recipe and evaluation

- Alpha is the HMM emission probability, not an angle: 0.50, 0.65, 0.75 and 0.85. Stay 0.7; n=20; gamma=0.7; delta_v=0.3.
- Joint token = theta_bin × 50 + azimuth_bin; 50×50 bins, 2,500 vocabulary. Azimuth bins 49 and 0 are circular neighbours.
- d128, four layers, two heads, MLP512; 500-token context; k=10; seed0; FP32. Global batch1,024, microbatch32; 1,367 Adam updates. Requested700,000,000 / actual699,904,000 input tokens per model.
- Starts use uniform coordinate offsets [-0.1,0.1] for theta/dtheta/dpsi and independent uniform azimuth [-pi,pi).
- Training revision: `92bb2afa0fd61e86a0402cbffb2a50907ba0822f`. Archive SHA256: `1177588a13e4efec9afa3e23b8fda3332acb6aa7c87c0edb7d6f026c8875bbf0`.
- Kratos RTX5070Ti, Python3.14.4, torch2.11.0+cu128. Exact sliced resume preserved model, Adam, scheduler, stream and CPU/CUDA RNG. Final model weights, saved random weights and metadata are under `models/`.
- Existing safety gates used 4,464 trajectories / 14 physics seeds per setting, fine dt=0.0025 agreement and stress checks, with zero pole/rate contacts and bin-changing actions. Physics seeds do not measure neural-training variance.
- Analysis used 2,048 fresh trajectories, evaluation seed20261009 and split seed29, with 1,228/410/410 trajectory groups. Duplicate causal-prefix rows were excluded; cross-split overlap is zero. Ridge regularization was selected on validation and scores reported on test. Saved random and shuffled-fit controls are included.
- The native head is evaluated at source positions0–489 only, corresponding to directly supervised prefixes up to490 tokens. Probes use ticks19,39,…,479. This teacher-forced diagnostic does not measure autonomous rollout performance.

## Reproduction and integrity

[Reproduction instructions](reproduction/README.md) restore already measured outputs without training or inference. Large files are stored as lossless parts, each at most8MiB. Run from this packet root:

```sh
python3 reproduction/restore_large_files.py --root .
```

This reconstructs byte-identical original ZIPs, standalone HTML and the frozen training source archive. Extract each original ZIP to its named `analysis/MODEL_NAME` directory to recover the saved NPZ probabilities. `LARGE_FILES.json` carries part/original hashes; `PACKET_MANIFEST.json` carries the published file hashes. Each model's `EXPORT_VERIFIED.json` carries its original ZIP/member hashes. `analysis.json` and `probes.csv` retain all 16 probe rows and agree numerically.

The exact analysis script is independently pinned to SHA256 `b5da1dd21570954a191f998123e5bfded59e7077f6b1b5d4f16d67080db76d06`. The 3D script reads saved NPZ only. Completion, source/config/checkpoint identity, three-host hash agreement and cleanup proofs are under `evidence/`. Process completion and scientific acceptance are separate.

## Negative attempts and cleanup

The paid RTX5090 benchmark failed the additional $2 gate with zero optimizer updates. The RTX4060Ti attempt failed transport; neither paid attempt counts as a completed model. Both exact rentals were independently verified absent with zero associated persistent volumes. There is no active paid rental for this experiment. The cumulative $2 cap remained unchanged, and Kratos training finished within the original 12-hour deadline. Its measured 201,898 input-token/s benchmark is not a claim of global hardware optimality.

Slice0 lost its login session after SQLite descriptor exhaustion; a private connection-closure fix preserved the shared harness. Slice3 later lost SSH transport; its exact optimizer capsules were verified across Kratos, Mac and staging before resumption. Failure receipts, capsule checks and request evidence are preserved. Partial/failure binaries remain at their original preservation locations, with hashes and references in `evidence/`; only the final four model/random pairs are duplicated here. All owned training workers and the coordinator exited, and bounded keepers/caffeinate were stopped. Unrelated Kratos jobs were left running.
