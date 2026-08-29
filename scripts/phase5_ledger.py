"""Every phase-5 result, signed for or against "the transformer builds the HMM belief".

Five blocks, in the order the argument runs:

  A  loss headroom -- how far each model is from the exact Bayes floor, which
     decides whether "add parameters" is even a coherent lever for that system
  B  decodability -- is the belief in the stream at all
  C  is it the belief rather than a myopic or lexical proxy
  D  causal -- is it used
  E  is the information available in the process at all, model aside

Run:  uv run python scripts/phase5_ledger.py
"""
import json
from pathlib import Path
D = Path("experiments/outputs-03")
L = lambda n: json.load(open(D / n))
val = L("phase5_00_validity.json")["systems"]
grid, sem, myo = L("phase5_01_ablation_grid.json"), L("phase5_02_probe_semantics.json"), L("phase5_03_myopic.json")
cau, roll, emg = L("phase5_05_causal.json"), L("phase5_06_rollout.json"), L("phase5_07_emergence.json")
SYS = list(val)
best = lambda s: next(d for d in sem[s]["trained"]["by_depth"] if d["name"] == sem[s]["trained"]["best_depth"])
bestu = lambda s: next(d for d in sem[s]["untrained"]["by_depth"] if d["name"] == sem[s]["trained"]["best_depth"])

print("A. LOSS HEADROOM  (is the model capacity-limited at the task it is trained on?)")
print(f"{'system':<17}{'Bayes floor':>12}{'L(12M)':>9}{'L(500M)':>9}{'excess@500M':>12}{'excess left':>12}")
for s in SYS:
    bf, lc = val[s]["bayes_floor"], val[s]["loss_curve"]
    f = bf["plugin"]
    e12 = next(c["excess_loss"] for c in lc if c["tokens"] >= 12_000_000)
    e500 = lc[-1]["excess_loss"]
    print(f"{s:<17}{f:>12.4f}{f+e12:>9.4f}{f+e500:>9.4f}{e500:>12.4f}{100*e500/e12:>11.1f}%")

print("\nB. DECODABILITY  (positive: the belief is present)")
print(f"{'system':<17}{'depth':>13}{'lin':>7}{'untr':>7}{'MLP':>7}{'MLPuntr':>9}{'metric':>8}{'z0':>7}")
for s in SYS:
    b, u = best(s), bestu(s)
    print(f"{s:<17}{b['name']:>13}{b['r2_by_group']['action_lag0']:>7.3f}{u['r2_by_group']['action_lag0']:>7.3f}"
          f"{b['v5_mlp']['action_lag0']:>7.3f}{u['v5_mlp']['action_lag0']:>9.3f}"
          f"{b['r2_by_group']['metric']:>8.3f}{b['r2_by_group']['z0']:>7.3f}")

print("\nC. IS IT THE BELIEF, NOT SOMETHING CHEAPER?  (the hypothesis's real content)")
print(f"{'system':<17}{'R2(b|p)':>9}{'R2(b|p1..8)':>12}{'paired':>8}{'pooled':>8}{'V2 bel':>8}{'V2 tru':>8}{'chance':>8}{'bag':>7}")
for s in SYS:
    m, v3 = myo[s], sem[s]["v3_token_window"]
    v2 = max(sem[s]["trained"]["v2_belief_or_label"], key=lambda r: r["probe_agrees_with_belief"])
    e = m["e1b"]
    pr = e.get("paired_r2")
    po = e.get("pooled_r2")
    cell = lambda x: "n/a" if x is None else f"{x:.3f}"
    print(f"{s:<17}{m['e1a']['belief_from_p_linear']:>9.3f}{m['e1c_horizon'][-1]['belief_from_p1_to_pk']:>12.3f}"
          f"{cell(pr):>8}{cell(po):>8}"
          f"{v2['probe_agrees_with_belief']:>8.3f}{v2['probe_agrees_with_truth']:>8.3f}{v2['chance']:>8.3f}"
          f"{v3['r2_by_group']['action_lag0']:>7.3f}")

print("\nD. CAUSAL  (negative: is the belief USED?)")
print(f"{'system':<17}{'abl vs random':>15}{'abl vs varPCA':>15}{'E2d best sd':>13}{'R4 erase':>10}{'R4 patch':>10}")
for s in SYS:
    cells = [d["groups"]["action_lag0"]["ladder"][-1] for d in grid[s]["curve"][-1]["by_depth"]
             if d["name"] != "embedding" and "action_lag0" in d.get("groups", {})]
    ex_r = max(c["excess_over_random"] for c in cells)
    ex_v = max(c.get("excess_over_variance_matched_pca", float("-inf")) for c in cells)
    e2d = max(max(k["excess_in_control_sds"] for k in dep["by_k"]) for dep in roll[s]["by_depth"])
    r4 = [d["r4_complements"] for d in cau[s]["by_depth"] if "r4_complements" in d]
    npass = sum(1 for r in r4 if r["belief_minus_metric_delta"] > r["random_delta_mean"] + 2 * r["random_delta_sd"])
    pc = [d["e2c_patching_complement"] for d in cau[s]["by_depth"] if "e2c_patching_complement" in d]
    pr = pc[0]["by_rank"][-1] if pc else None
    ps = "n/a" if pr is None else f"{(pr['belief']['cosine']-pr['random_mean_cosine'])/max(pr['random_sd_cosine'],1e-9):+.1f}sd"
    print(f"{s:<17}{ex_r:>15.4f}{ex_v:>15.4f}{e2d:>13.1f}{f'{npass}/{len(r4)}':>10}{ps:>10}")

print("\nE. IS THE INFORMATION EVEN AVAILABLE?  (oracle on the HMM, no model)")
for s in SYS:
    c = val[s]["c21_recovery_by_phase"]
    print(f"  {s:<17} chance={c['chance']:.2f}  oracle@end-of-segment={c['oracle_recovery'][-1]:.2f}  "
          f"model linear={best(s)['r2_by_group']['action_lag0']:.3f}  MLP={best(s)['v5_mlp']['action_lag0']:.3f}")
