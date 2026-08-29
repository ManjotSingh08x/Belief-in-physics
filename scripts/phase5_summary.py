"""Print the phase-5 tables. Reads whatever JSONs exist; skips the rest.

Run:  uv run python scripts/phase5_summary.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get("OUTPUT_DIR", ROOT / "experiments/outputs-03"))


def load(name):
    p = OUT / name
    return json.loads(p.read_text()) if p.exists() else None


def head(t):
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}")


def validity(d):
    head("5.0  V1 impossible probe (gate) and V4 Bayes floor")
    print(f"{'system':<17}{'V1 worst R2':>13}{'V1 CI hi':>10}{'gate':>7}"
          f"{'floor':>9}{'L(12M)':>9}{'L(500M)':>9}{'excess 12M':>12}{'excess 500M':>13}")
    for n, v in d["systems"].items():
        w = max(v["v1_impossible_probe"], key=lambda r: r["r2_impossible"])
        c = v["loss_curve"]
        e12 = next(x for x in c if x["tokens"] == 12_000_000)
        print(f"{n:<17}{w['r2_impossible']:>13.4f}{w['ci'][1]:>10.4f}"
              f"{'PASS' if v['v1_pass'] else 'FAIL':>7}"
              f"{v['bayes_floor']['plugin']:>9.4f}{e12['loss']:>9.4f}{c[-1]['loss']:>9.4f}"
              f"{e12['excess_loss']:>12.4f}{c[-1]['excess_loss']:>13.4f}")

    head("5.0  C21 oracle recovery vs observations since the kick (no model)")
    for n, v in d["systems"].items():
        c = v["c21_recovery_by_phase"]
        print(f"  {n:<17} chance={c['chance']:.2f}  "
              + " ".join(f"{s}:{r:.2f}" for s, r in
                         zip(c["observations_since_kick"], c["oracle_recovery"])))

    head("5.0  C12 where each quantity is COMPUTED (delta R2 per layer, final ckpt)")
    for n, v in d["reread_from_phase4"].items():
        print(f"  {n}  (phase-4 chose {v['chosen_depth']}, argmax margin {v['chosen_margin']:.4f})")
        for g, p in v["per_group"].items():
            print(f"    {g:<12} " + " ".join(f"{x:+.3f}" for x in p["increments"])
                  + f"   own argmax @{p['argmax_depth']}  at_chosen={p['at_chosen_depth']:.3f}")


def ablation(d):
    head("5.1  R1 belief excess Δloss (sd units) at EVERY depth, final checkpoint")
    print(f"{'system':<17}{'depth':<15}{'rank':>5}{'varfrac':>9}{'Δloss':>9}"
          f"{'exRandom':>10}{'sd':>7}{'>p95':>6}{'exVMPCA':>10}{'resample':>10}")
    for n, v in d.items():
        final = v["curve"][-1]
        for dep in final["by_depth"]:
            g = dep["groups"].get("action_lag0")
            if not g:
                continue
            l = g["ladder"][-1]
            print(f"{n:<17}{dep['name']:<15}{g['rank']:>5}{g['variance_fraction']:>9.3f}"
                  f"{g['delta_loss']:>9.4f}{l['excess_over_random']:>10.4f}"
                  f"{l['excess_in_control_sds']:>7.1f}{str(l['exceeds_random_p95']):>6}"
                  f"{l['excess_over_variance_matched_pca']:>10.4f}{g['delta_loss_resample']:>10.4f}")

    head("5.1  the same for the metric")
    for n, v in d.items():
        for dep in v["curve"][-1]["by_depth"]:
            g = dep["groups"].get("metric")
            if not g:
                continue
            l = g["ladder"][-1]
            print(f"{n:<17}{dep['name']:<15}{g['rank']:>5}{g['variance_fraction']:>9.3f}"
                  f"{g['delta_loss']:>9.4f}{l['excess_over_random']:>10.4f}"
                  f"{l['excess_in_control_sds']:>7.1f}{str(l['exceeds_random_p95']):>6}"
                  f"{l['excess_over_variance_matched_pca']:>10.4f}{g['delta_loss_resample']:>10.4f}")

    head("5.1  F1: does the phase-4 sign survive holding depth fixed?")
    for n, v in d.items():
        print(f"  {n}")
        for dep_i in range(len(v["curve"][0]["by_depth"])):
            row = []
            for c in v["curve"]:
                dep = c["by_depth"][dep_i]
                g = dep["groups"].get("action_lag0")
                row.append("  --  " if not g else f"{g['excess']:+.3f}")
            print(f"    {v['curve'][0]['by_depth'][dep_i]['name']:<14} " + " ".join(row))
        print(f"    {'tokens':<14} " + " ".join(f"{c['tokens']/1e6:>6.0f}M" for c in v["curve"]))

    head("5.1  R4 first principal angle between belief and metric bases (final)")
    for n, v in d.items():
        for dep in v["curve"][-1]["by_depth"]:
            a = dep.get("principal_angles_deg", {}).get("action_lag0|metric")
            if a:
                print(f"  {n:<17}{dep['name']:<15} first={a['first']:>6.2f} deg  "
                      f"mean={a['mean']:>6.2f}  n_below_10deg={a['n_below_10deg']}")


def semantics(d):
    head("5.2  V3 token-window baseline vs the transformer")
    print(f"{'system':<17}{'features':>9}{'d_model':>8}{'bag belief':>12}{'model belief':>14}"
          f"{'bag metric':>12}{'model metric':>14}")
    for n, v in d.items():
        b = v["v3_token_window"]
        best = next(x for x in v["trained"]["by_depth"] if x["name"] == v["trained"]["best_depth"])
        print(f"{n:<17}{b['n_features']:>9}{b['d_model']:>8}"
              f"{b['r2_by_group']['action_lag0']:>12.3f}{best['r2_by_group']['action_lag0']:>14.3f}"
              f"{b['r2_by_group']['metric']:>12.3f}{best['r2_by_group']['metric']:>14.3f}")

    head("5.2  V5 nonlinear ceiling, C17 simplex coords, V6 CI (trained, best depth)")
    print(f"{'system':<17}{'linear':>9}{'CI':>18}{'simplexcoord':>14}{'MLP':>9}"
          f"{'untr linear':>13}{'untr MLP':>10}")
    for n, v in d.items():
        b = next(x for x in v["trained"]["by_depth"] if x["name"] == v["trained"]["best_depth"])
        u = next(x for x in v["untrained"]["by_depth"] if x["name"] == v["untrained"]["best_depth"])
        ci = b["v6_ci_belief"]
        interval = "[{:.3f}, {:.3f}]".format(ci["lo"], ci["hi"])
        print(f"{n:<17}{b['r2_by_group']['action_lag0']:>9.3f}"
              f"{interval:>18}"
              f"{b['c17_r2_simplex_coords']['action_lag0']:>14.3f}"
              f"{b['v5_mlp']['action_lag0']:>9.3f}"
              f"{u['r2_by_group']['action_lag0']:>13.3f}{u['v5_mlp']['action_lag0']:>10.3f}")

    head("5.2  V2 belief or label, on positions where the posterior mode is wrong")
    for n, v in d.items():
        rows = v["trained"]["v2_belief_or_label"]
        if not rows:
            continue
        r = max(rows, key=lambda x: x["r2_belief_on_disagreements"])
        print(f"  {n:<17}@{r['name']:<14} {r['disagree_fraction']:.0%} disagree  "
              f"probe->belief {r['probe_agrees_with_belief']:.3f}  "
              f"probe->truth {r['probe_agrees_with_truth']:.3f}  chance {r['chance']:.3f}")

    head("5.2  C16 belief R2 by belief-entropy quartile (low entropy = near-collapsed)")
    for n, v in d.items():
        b = next(x for x in v["trained"]["by_depth"] if x["name"] == v["trained"]["best_depth"])
        print(f"  {n:<17}" + "  ".join(
            f"[{s['entropy_range'][0]:.2f}-{s['entropy_range'][1]:.2f}]:{s['r2']:.3f}"
            for s in b["c16_stratified_belief"]))

    head("5.2  R6 is the target erased, or only linearly erased?")
    for n, v in d.items():
        r = v.get("r6_nonlinear_after_erasure")
        if r:
            print(f"  {n:<17}@{r['depth']:<14} rank {r['rank']:>3}  linear "
                  f"{r['linear_before']:.3f}->{r['linear_after']:.3f}   "
                  f"MLP {r['mlp_before']:.3f}->{r['mlp_after']:.3f}")


def myopic(d):
    head("5.3  E1a belief from the optimal next-token distribution, and the ceiling")
    print(f"{'system':<17}{'linear':>8}{'log-lin':>9}{'ceiling r=.02':>14}{'r=.04':>8}"
          f"{'singleton@.02':>15}{'stream':>8}{'topPC':>7}{'randproj':>9}")
    for n, v in d.items():
        e = v["e1a"]
        sw = {s["radius"]: s for s in e["radius_sweep"]}
        dc = e["dimension_controls"]
        print(f"{n:<17}{e['belief_from_p_linear']:>8.3f}{e['belief_from_log_p_linear']:>9.3f}"
              f"{sw[0.02]['ceiling']:>14.3f}{sw[0.04]['ceiling']:>8.3f}"
              f"{sw[0.02]['singleton_fraction']:>15.2f}"
              f"{dc['belief_from_full_stream']:>8.3f}"
              f"{dc['belief_from_top_pcs_of_stream']:>7.3f}"
              f"{dc['belief_from_random_projection_of_stream']:>9.3f}")

    head("5.3  E1c horizon: R2(belief | p^(1..K)) and R2(metric | p^(1..K))")
    for n, v in d.items():
        h = v["e1c_horizon"]
        print(f"  {n:<17} belief " + " ".join(f"k{c['k']}:{c['belief_from_p1_to_pk']:.3f}" for c in h))
        print(f"  {'':<17} metric " + " ".join(f"k{c['k']}:{c['metric_from_p1_to_pk']:.3f}" for c in h))
        j = h[1].get("belief_from_joint_pair")
        if j is not None:
            print(f"  {'':<17} pairwise joint p(x_t+1, x_t+2) -> belief: {j:.3f}")

    head("5.4  E1b matched pairs: same next-token distribution, different belief")
    print(f"{'system':<17}{'pairs':>10}{'mean TV':>10}{'mean |db|':>11}"
          f"{'paired R2':>11}{'shuffled':>10}{'pooled R2':>11}")
    for n, v in d.items():
        e = v["e1b"]
        if not e.get("runnable"):
            print(f"{n:<17}{e['n_candidate_pairs']:>10}  NOT RUNNABLE")
            continue
        print(f"{n:<17}{e['n_pairs_used']:>10,}{e['mean_tv_of_used_pairs']:>10.4f}"
              f"{e['mean_belief_gap_of_used_pairs']:>11.3f}{e['paired_r2']:>11.3f}"
              f"{e['paired_r2_shuffled_control']:>10.3f}"
              f"{v['e1a']['dimension_controls']['belief_from_full_stream']:>11.3f}")


def causal(d):
    head("5.5  E2c patching: does moving the belief subspace move the model where it should?")
    print("     transplant cosine, belief / matched-rank random. The embedding is degenerate:")
    print("     the belief probe there reads token identity, so patching it substitutes the input.")
    for n, v in d.items():
        print(f"  {n}  (final depth {v['final_depth_excluded']} excluded: only "
              f"{v['vocab_size'] - 1} of 128 directions can move the loss)")
        for dep in v["by_depth"]:
            p = dep.get("e2c_patching")
            if not p:
                continue
            print(f"    {dep['name']:<14} " + "  ".join(
                f"r{q['rank']}:{q['belief']['transplant_cosine']:+.2f}/"
                f"{q['random_mean_transplant_cosine']:+.2f}" for q in p["by_rank"]))

    head("5.5  E2b position-restricted: corrupt the belief at t, read the loss downstream")
    for n, v in d.items():
        print(f"  {n}")
        for dep in v["by_depth"]:
            b = dep.get("e2b_position_restricted")
            if not b or not b.get("runnable"):
                continue
            rows = [r for r in b["by_offset"] if r["block_len"] == 4]
            print(f"    {dep['name']:<14} " + "  ".join(
                f"+{r['offset']}:{r['excess']:+.4f}" for r in rows))

    head("5.6  E2a decodability of the token at t+k, drop after erasing the belief")
    for n, v in d.items():
        for dep in v["by_depth"]:
            h = dep.get("e2a_decodability_horizon")
            if not h:
                continue
            print(f"  {n:<17}{dep['name']:<14} intact k1={h[0]['r2_intact']:.3f}   drop "
                  + " ".join(f"k{r['k']}:{r['r2_intact'] - r['r2_after_erasure']:+.3f}" for r in h))

    head("5.5  R4 complement erasure at matched rank")
    for n, v in d.items():
        for dep in v["by_depth"]:
            r = dep.get("r4_complements")
            if not r:
                continue
            print(f"  {n:<17}{dep['name']:<14} 1st angle={r['first_principal_angle_deg']:>5.1f}deg "
                  f"({r['n_angles_below_10deg']} below 10)  rank={r['matched_rank']:>3}  "
                  f"belief-metric={r['belief_minus_metric_delta']:+.4f}  "
                  f"metric-belief={r['metric_minus_belief_delta']:+.4f}  "
                  f"random={r['random_delta_mean']:+.4f}+-{r['random_delta_sd']:.4f}")


MIN_GAIN = 0.10


def _gain_crossing(axis, curve, thresh):
    """Tokens at which the learned gain first reaches `thresh`, log-interpolated."""
    g = np.asarray(curve) - curve[0]
    hit = np.flatnonzero(g >= thresh)
    if not hit.size:
        return None
    i = int(hit[0])
    if i == 0:
        return float(axis[0])
    x0, x1, y0, y1 = np.log(max(axis[i - 1], 1)), np.log(axis[i]), g[i - 1], g[i]
    return float(np.exp(x0 + (thresh - y0) / (y1 - y0) * (x1 - x0)))


def _fractional_rho(entry, frac, family=None):
    """Spearman(coupling, log emergence time) at a FRACTION of each target's own gain.

    An absolute gain threshold conflates "learned sooner" with "learned more":
    a target ending at gain 0.29 crosses 0.05 before one ending at 0.10 almost
    mechanically. Normalising by the target's own asymptote removes that.
    """
    axis, pts = entry["tokens_axis"], []
    for t in entry["targets"]:
        if t["alpha"] is None or t["learned_gain"] <= MIN_GAIN:
            continue
        if family is not None and t["family"] != family:
            continue
        c = _gain_crossing(axis, t["curve"], frac * t["learned_gain"])
        if c:
            pts.append((t["coupling_to_p"], np.log(c)))
    if len(pts) < 4:
        return None, len(pts)
    rho = spearmanr([p[0] for p in pts], [p[1] for p in pts]).correlation
    return float(rho), len(pts)


def emergence(d):
    head("5.7  E5 coupling vs emergence, within one model, matched-width targets")
    print("  rho at a fraction of each target's OWN final learned gain (confound-free);")
    print("  negative = higher coupling to p(next) emerges earlier.")
    print(f"  {'system':<17}{'frac':<6}{'random':>14}{'belief':>14}{'both':>14}")
    for n, v in d.items():
        for frac in (0.5, 0.75):
            cells = []
            for fam in ("random", "belief", None):
                rho, k = _fractional_rho(v, frac, fam)
                cells.append("     --" if rho is None else f"{rho:+.2f} (n={k})")
            print(f"  {n:<17}{frac:<6}" + "".join(f"{c:>14}" for c in cells))
    print()
    for n, v in d.items():
        print(f"  {n}   Spearman(coupling, log tokens to ABSOLUTE gain threshold): "
              + "  ".join(f"{t}:{s['spearman_coupling_vs_log_tokens']}" for t, s in v["spearman"].items()))
        for r in sorted(v["targets"], key=lambda r: -r["coupling_to_p"]):
            c = r["crossings"]
            print(f"    {r['target']:<18} alpha={str(r['alpha']):>5}  coupling={r['coupling_to_p']:+.3f}"
                  f"  final={r['final_r2']:+.3f}  untr={r['untrained_r2']:+.3f}  "
                  + "  ".join(f"@{t}:{'never' if c[t] is None else f'{c[t]/1e6:.1f}M'}" for t in c))


def main():
    for filename, fn in (
        ("phase5_00_validity.json", validity),
        ("phase5_01_ablation_grid.json", ablation),
        ("phase5_02_probe_semantics.json", semantics),
        ("phase5_03_myopic.json", myopic),
        ("phase5_05_causal.json", causal),
        ("phase5_07_emergence.json", emergence),
    ):
        d = load(filename)
        if d is None:
            print(f"\n[skip {filename}]")
            continue
        try:
            fn(d)
        except Exception as exc:
            print(f"\n!! {filename}: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
