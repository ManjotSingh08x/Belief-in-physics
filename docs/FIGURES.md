# Figure key

Every symbol, axis and colour in `figures/`.

## `simplex_<system>.png` -- four panels, produced by `scripts/make_simplex_figures.py`

One point per token position, `256 sequences x seq_len` positions. The plotted
vector is `action_lag0`: the exact posterior over which perturbation opened the
current segment, a point in the `(A-1)`-simplex where `A = n_actions`.

Colour is the **ground-truth action**, not the argmax, so the panels show
whether the geometry separates the hidden cause rather than merely spreading.
Colormap `viridis`, discretised to `A` levels.

**Title line 2** -- `A` actions, `n_z0` initial conditions, `n_z0 * A^M`
branches, `seq_len` tokens.

### Panel 1 (top-left) -- simplex projection

`simplex_to_2d`: vertex `k` at angle `2*pi*k/A` on the unit circle, point
plotted at `sum_k p_k * vertex_k`. Uniform belief lands at the origin; a
collapsed belief lands on a labelled vertex. Grey polygon is the vertex hull.
Vertex labels are `process.action_names`.

- `A = 3` (pendulum): bijective, the triangle is faithful.
- `A > 3`: marked **LOSSY**. Distinct beliefs can coincide. Read nothing
  quantitative off this panel -- use panel 2.

### Panel 2 (top-right) -- PCA

SVD of the same points, mean-centred, first two right singular vectors.
Axes `PC1`/`PC2` are in belief units but carry no fixed orientation.
Title states the fraction of variance the 2D view retains; `100%` for `A = 3`
confirms panel 1 lost nothing there.

### Panel 3 (bottom-left) -- belief uncertainty

Histogram over all positions of normalised Shannon entropy
`H(p) / log A`, in `[0, 1]`. Crimson line is the mean.

`0` = collapsed, so the target is a discrete label and a high probe R^2 would
mean nothing. `1` = uniform, so there is nothing to probe. The configs are
tuned to sit between; the test suite asserts `0.3 < H < 0.9`.

### Panel 4 (bottom-right) -- one sequence through time

Sequence 0 only. `x` = token position, `y` = `P(last perturbation = k)`, one
line per action in the panel-1 vertex colours. Dashed grey verticals mark
segment boundaries, where a new hidden perturbation fires and the posterior
resets to uniform (`1/A`) before re-sharpening on the next `steps_per_segment`
observations.

## `<system>.png` -- Phase 1 simulator sanity checks

Produced by `physics/viz.py` on the **earlier Ulam path**, not the branch
process. Retained as evidence the physics integrates correctly; the belief
panel there is over Ulam latent bins and is not the object probed in
`RESULTS.md`.

| panel | axes | meaning |
|---|---|---|
| metric trajectory | position vs metric value | red marks a perturbation |
| token raster | position vs token id | emitted observation stream |
| posterior marginal | position vs latent bin | Ulam posterior, true state overlaid; must stay inside the bright band |
| TV curve | steps after a kick vs total variation | perturbation separability from no-op; decays as damping washes the kick out |
