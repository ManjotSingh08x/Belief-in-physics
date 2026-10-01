# Belief State Existence Tests: Experiment Design

## Context

Probe R² alone cannot answer "does belief exist in the residual stream?" High R² might be dimensionality artifact. Low R² might miss nonlinear encoding. Need multiple converging lines of evidence.

Goal: identify which physical configs show **cleanest** belief representation, to orient future intervention work.

---

## 1. Current Data: What It Already Tells Us

### Belief-physics anti-correlation (key finding)

From `all_layers_single_token` (the only clean mode):

```
corr(physics_R², belief_R²) = -0.75
```

Configs that learn physics best learn belief worst. Physics-heavy configs (low gamma=0.3, low dv=0.7) score physics R²=0.81 but belief R²=0.35. Belief-heavy configs (gamma=1.0, dv=1.2) score belief R²=0.66 but physics R²=0.47.

This anti-correlation is itself evidence: if probes were fitting noise, physics and belief R² would co-vary positively (both benefiting from richer representations). Anti-correlation means the model is allocating representational capacity between two genuinely different quantities.

### Physics-belief entanglement in the data

Measured conditional dependence (physics → belief linear R²):

| Config | Physics→Belief R² | Residual variance |
|---|---|---|
| gamma=0.3, dv=0.7 | 0.40 | **0.60** (most independent) |
| gamma=0.5, dv=0.7 | 0.50 | 0.50 |
| gamma=0.8, dv=1.0 | 0.60 | 0.40 |
| gamma=1.0, dv=1.2 | 0.63 | 0.37 |
| gamma=1.0, dv=1.2, n=20 | 0.70 | 0.30 (most entangled) |

Lower gamma = more residual variance = belief carries more information **beyond** physics. But the anti-correlation above means models with gamma=0.3 have *lower* belief probe R². Paradox? No: low damping means pendulum remembers kicks longer, so physics state itself encodes more of the HMM history. The model doesn't need a separate belief representation because the physics already carries it.

> [!IMPORTANT]
> **The sweet spot for existence tests: configs where belief is decorrelated from physics AND the model still achieves reasonable belief probe R².** From current data: **gamma=0.8, dv=1.2, n=10** — belief gap=+0.50, physics→belief R²=0.60, residual=0.40.

### Best configs ranked by belief gap (trustworthy mode)

| Config | Belief gap | Physics gap | Belief residual |
|---|---|---|---|
| **gamma=1.0, dv=1.2, n=10** | **+0.50** | +0.55 | 0.37 |
| **gamma=0.8, dv=1.2, n=10** | **+0.50** | +0.64 | 0.40 |
| gamma=0.8, dv=1.0, n=10 | +0.45 | +0.61 | 0.40 |
| gamma=1.0, dv=1.2, n=15 | +0.44 | +0.39 | 0.37 |
| gamma=0.8, dv=1.2, n=15 | +0.44 | +0.44 | 0.40 |

---

## 2. Existence Tests

### Test A: Residual Orthogonality — "Belief Beyond Physics"

**Question**: After removing all physics information from the residual stream, does belief remain?

**Method**: 
1. Fit ridge probe: `residual_stream → (theta, omega)` = physics subspace
2. Project residual stream onto the **null space** of physics probe weights
3. Probe the residual for belief
4. If belief R² > 0 in the null space: belief representation is **orthogonal** to physics

```python
# Pseudocode
W_phys = ridge_fit(X, y_physics)  # (d_model, 2)
# Project out physics subspace
Q, _ = np.linalg.qr(W_phys)  # orthonormal basis of physics subspace
X_null = X - X @ Q @ Q.T      # remove physics directions
# Now probe X_null for belief
score_belief_in_null = ridge_fit(X_null, y_belief)
```

**Why this is decisive**: random networks can linearly decode physics from token embeddings (tokens = angle). But random networks cannot have learned to **separate** belief from physics into orthogonal subspaces. Only training does that.

**Controls**: same procedure on random model (should find ~0 belief in null space).

### Test B: Layer-wise Emergence Curves

**Question**: At which layer does belief become decodable? Does it emerge differently from physics?

**Method**: Probe each layer independently (already have `single_layer_single_token` data). Plot R² per layer.

**Expected signature of true belief**:
- Physics R² high at layer 0 (embedding carries angle directly)
- Belief R² **near zero at layer 0**, rises at layers 2-3 (requires multi-step temporal computation)
- If belief R² is flat across layers: suspicious (might be token-identity artifact)

**From current data**: `single_layer_single_token` belief R² is 0.05-0.15 across configs — too noisy to see layer structure. Need **per-layer** breakdown (not pooled across layers as current mode does by stacking layers into the sample dimension).

**Fix**: modify probe to report R² separately per layer, not pooled. The existing [probe_layers](file:///home/manjot/Files/Code/BeliefPhysics/Belief-in-physics/models/analysis.py#L82-L130) function already does this — use it.

### Test C: PCA Alignment — "Does Training Concentrate Belief?"

**Question**: In the trained model, does belief information concentrate into a few principal components? Does random model show this?

**Method**:
1. PCA on residual stream at last layer
2. Project belief targets onto each PC
3. Plot cumulative R² vs number of PCs

**Expected signature**:
- Trained: belief R² saturates quickly (top 5-10 PCs capture most belief info)
- Random: belief R² grows linearly with PCs (no concentration = random projection)

If trained model concentrates belief into few PCs, the model has formed a low-dimensional belief subspace — the very subspace you'd target for intervention.

**Bonus**: the PCs that carry belief are the **candidate intervention directions**. This test doubles as subspace discovery.

### Test D: Temporal Selectivity Within Cycle

**Question**: Does belief decodability change within a cycle (across the n_steps positions)?

**Method**: Probe at position t=1, t=2, ..., t=n_steps within each cycle separately.

**Expected signature**:
- Belief R² should **increase** across positions within a cycle. Why: at t=1 (right after kick), the model has seen one new observation and can start updating belief. By t=n_steps, it has seen the full trajectory arc and has maximum evidence about which kick occurred.
- Physics R² should be roughly constant (every position has an angle).

This distinguishes "model computes belief" from "model memorizes tokens." Memorization would show flat selectivity. Active computation shows temporal gradient.

### Test E: Cross-Configuration Transfer

**Question**: Does the belief probe generalize across physics configurations?

**Method**:
1. Train probe on config A (e.g., gamma=0.8, dv=1.2)
2. Test on config B (e.g., gamma=1.0, dv=1.0)
3. Both use same chain (alpha=0.7, stay=0.7), so belief dynamics are identical
4. Only physics differs

**If probe transfers**: model represents belief in a **physics-invariant** way. Strong evidence of abstract belief representation.

**If probe fails**: belief representation is entangled with physics-specific features. Still useful — tells you the model encodes belief differently per regime.

### Test F: Matched-Physics Divergence

**Question**: When two trajectories have the same physical state (theta, omega) but different beliefs, do residual streams differ?

**Method**:
1. Sample many trajectories
2. Find "matched pairs": timesteps where theta and omega are within epsilon but beliefs differ maximally
3. Compute L2 distance of residual streams between matched pairs
4. Compare with L2 distance between unmatched pairs
5. If matched-pair distance > 0: model encodes something beyond physics at that position

**Why this matters**: directly tests the existence claim. No probe needed. Pure representation geometry.

**This is the natural precursor to activation patching**: if you find positions where physics matches but residual streams differ, those are exactly the positions where patching the belief subspace should change predictions.

### Test G: Belief Geometry — Simplex Structure

**Question**: Does the model's belief representation have the geometric structure of a simplex?

**Method**:
1. Extract residual streams at cycle-end positions
2. Color-code by ground-truth HMM state (argmax of belief)
3. Apply UMAP/t-SNE/PCA to residual streams
4. Check: do 4 clusters emerge? Do they form a tetrahedral structure (matching the 4-state simplex)?

**Expected signature**:
- Trained: clusters corresponding to HMM states, with transition trajectories between them
- Random: uniform cloud, no structure

**Quantify**: Adjusted Mutual Information between k-means(4) cluster assignments and ground-truth HMM states.

### Test H: Information-Theoretic — Mutual Information

**Question**: How much information does the residual stream carry about belief, compared with the input ceiling?

**Method**:
1. Compute MI(residual_stream, belief) using KSG estimator (or binned estimator)
2. Compute MI(token_history, belief) — this is the theoretical ceiling
3. Compute MI(random_residual_stream, belief) — baseline

**If MI(trained) >> MI(random) and MI(trained) approaches MI(tokens)**: model efficiently compresses token history into belief-relevant representation.

Simpler alternative: use the **token window baseline** ([token_window_features](file:///home/manjot/Files/Code/BeliefPhysics/Belief-in-physics/physics/messk.py#L456-L471) already exists). Probe one-hot encoding of last W tokens for belief. Compare with residual stream probe. If residual stream beats token window, the model is computing something beyond memorization.

---

## 3. Configuration Selection Criteria

For each config, compute a **belief separability score**:

```
score = belief_gap × (1 - physics_to_belief_R²) × temporal_gradient
```

Where:
- `belief_gap` = trained belief R² - random belief R² (from clean mode)
- `1 - physics_to_belief_R²` = how much belief is independent of physics
- `temporal_gradient` = slope of within-cycle belief R² (Test D)

### Recommendations for next sweep

| Priority | Config | Rationale |
|---|---|---|
| 🥇 | gamma=1.0, dv=1.2, n=10 | Highest belief gap (+0.50), high decorrelation |
| 🥈 | gamma=0.8, dv=1.2, n=10 | Same gap, slightly more residual variance |
| 🥉 | gamma=0.8, dv=1.0, n=10 | Good gap (+0.45), lower kick strength tests robustness |
| Control | gamma=0.3, dv=0.7, n=10 | Low gap (+0.30), high entanglement — negative control |

**n_steps=10 dominates.** n=10 with m=50 gives 500 tokens: enough context, reasonable feature dims. n=20 inflates feature dimensions in cycle modes and the within/total variance ratio drops to 0.08 (physics explains almost everything).

---

## 4. Experiment Priority Order

Phase 1 (quick, use existing models):
1. **Test B**: Per-layer R² curves (just re-run probes per layer)
2. **Test A**: Null-space probing (one extra regression step)
3. **Test G**: UMAP/clustering of residual streams

Phase 2 (moderate effort):
4. **Test D**: Temporal selectivity (probe at each within-cycle position)
5. **Test C**: PCA alignment curves (trained vs random)
6. **Test F**: Matched-physics divergence analysis

Phase 3 (new experiments):
7. **Test E**: Cross-config transfer (requires probing across saved models)
8. **Test H**: MI estimation (requires careful estimator choice)

---

## 5. Key Insight: Memory Length = 7 Ticks

`MessKProcess(alpha=0.7, stay=0.7).memory_length() = 7` ticks.

Belief depends on last ~7 kicks. With n_steps=10, that's 70 tokens of history. Model context = 500 tokens. So the model has **more than enough** context to compute belief. The question is whether it *does*.

This also means: with n_steps=10, the model must attend back ~7 cycles (70 tokens). If attention patterns show systematic lookback to cycle boundaries, that's mechanical evidence of belief computation.

**Suggested additional test**: attention pattern analysis at cycle-end tokens. Do attention heads attend to previous cycle-end positions (where kicks land)? If yes, the model is selectively gathering kick evidence — the raw material for belief updates.

---

## Summary

| Test | What it proves | Effort | Priority |
|---|---|---|---|
| A: Null-space probing | Belief orthogonal to physics | Low | 🥇 |
| B: Layer emergence | Belief computed, not inherited | Low | 🥇 |
| C: PCA alignment | Belief concentrated in subspace | Low | 🥈 |
| D: Temporal selectivity | Active computation within cycle | Medium | 🥈 |
| E: Cross-config transfer | Abstract belief representation | Medium | 🥉 |
| F: Matched-physics | Belief exists beyond physics state | Medium | 🥈 |
| G: Geometric structure | Simplex geometry preserved | Low | 🥇 |
| H: Mutual information | Quantitative information content | High | 🥉 |

write an implementation plan to create a python script and an accompanying jupyter notebook which does the following:
- the python scripts contains the functions requried for implementing the tests and the plotting systems whle the jupyter notebook contains functionality to load configurations and model weights, display the results, rerun the plots with different settings and configurations. all of the plots follow the similar structure:
-- for each model configuration, it is allocatted 1-2 rows of subplots and they ar evertically stacked. for each plot we want per model, they are arranged horizontally with max 3-4 subplots per row and if there are multiple plots maybe 2 rows per configuration. for example if we have 3 models to test and 4 plots per model we will have a 3 height 4 width (in units of subplots) image. The tests should be done on atleast 3 different seeds and error bars should be included. 

- the python script should ipmlement test A: null-space probing, by getting the physics subspace and then projecting the residual stream on the null space and then probing the remaining for belief. this should be done on all layers concatenated single token setting and on all layers concatenated and all tokens in a cycle concatenated setting. along with this, the ridge regression should be used with alpha = 1. also at all points use gpu based regression. 
test B: probe all layers seperately on three variations per model: all tokens, tokens in a cycle concatenated and last token before pertubation setting. show the curves for each model for both random config and trained config side by side. each plot shold have 3 curves corresponding to the probing variaiton with x axis layer + concatenation of all layers and y axis r^2. 
test C: PCA alignment. perform this test on 3 features configruations: all_layers cycle_tokens concatenated, all_layers single token, single layer all tokens concatenated. each plot is for these different feature configuration for trained vs random. 
test D: Temporal selectivity. this test should be done on a single feature  configuration: all layers concatenated and tokens not concatenated ofcourse. the plots are 2, one for belief and one for physics and x axis is token position and y axis is r^2 of ridge probe.
- test G: simple Umap. colorcode according to the ground truth belief state distribution. like if the ground turth belief is (0.1, 0.4, 0.2, 0.3) then the color of a point should be say green*0.1 + red*0.4, blue*0.2 + another bright color*0.3 (choose a good color theme). then plot the umap of this with the color coding. do this for 3 feature config: all_layers cycle_tokens concatenated, all_layers single token, single layer all tokens concatenated. and with 2 different seeds and error bars for the different seeds. use 20 epochs for training the umap.
- test F:Matched-physics divergence analysis. do this for last token before pertubation case with all layers concatenated. you may need to search and brute through many simulations. take care to do it with n = 10 and with m = 50. test behaviour on longer end (such as at cycle 30 to 45). keep the remaining tests as future todos.

