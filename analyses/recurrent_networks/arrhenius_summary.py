"""Number trace of the escape ladder and the Arrhenius slopes of Fig. 2e (right).

Reads the escape-ladder output (escape_ladder.py) and prints every quantity in the form
name = value [CI] (n, test, P): the self-test calibrations, the networks, per-gain means, the ordering of the
gains (declared four gain levels and all five gains), the exclusion bounds, the Arrhenius slopes, the
over-prediction of the change in escape time, censoring, the finite-difference minimiser on the first network,
the internal consistency of the minimum-action paths and the decision.  Computes nothing that escape_ladder.py did
not already write, except the two figure entries below.

arrhenius_slopes.json: the two Arrhenius regressions of log mean first-passage time (primary transition rule,
separatrix_saddle_lefteig) on the minimum action (fw_action_mean) and on the fitted-density barrier
(barrier_generative_nats_DENSITY), each with its slope, 95% seed-bootstrap interval, number of networks,
one-sample t-test P values against slopes 1 and 0 and mean within-network R^2, rounded as printed in the trace;
and the number of networks whose escape times resolve at both ends of the gain range out of the networks with a
complete ladder.

Input   rnn_escape_simulation.json in the network cache paths.cache("rnn") (--ladder; the stored output is the
        Source Data file analysis_inputs/stored_results/rnn_escape_simulation.json).
Outputs figure_reproduction/inputs/arrhenius_slopes.json (the values of Fig. 2e, right) and
        analyses/recurrent_networks/out/escape_ladder_trace.txt.
Reported in  Fig. 2e right (minimum action 0.832 [0.796, 0.880], density barrier 0.170 [0.104, 0.234]; five of
        the seven networks with complete ladders) and the escape simulation of the Methods.
Runtime  seconds.

    python analyses/recurrent_networks/arrhenius_summary.py
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from analyses.common import paths  # noqa: E402

ap = argparse.ArgumentParser(description="number trace of the escape ladder and the Arrhenius slopes")
ap.add_argument("--ladder", default=os.path.join(paths.cache("rnn"), "rnn_escape_simulation.json"),
                help="escape-ladder output (default: rnn_escape_simulation.json in the network cache)")
args = ap.parse_args()
RES = Path(args.ladder)

R = json.loads(RES.read_text(encoding="utf-8"))
G = R["gains"]
grp = R["group"]
S = R["selftest"]
lines = []


def P(s=""):
    print(s, flush=True)
    lines.append(s)


P("=" * 92)
P("ESCAPE LADDER NUMBER TRACE -- " + RES.name)
P(f"  wall {R['wall_time_min']:.1f} min   torch {R['versions']['torch']}  "
  f"numpy {R['versions']['numpy']}  scipy {R['versions']['scipy']}  "
  f"device {R['versions']['device']}")
P(f"  alpha {R['conventions']['alpha']}  sigma {R['conventions']['sigma']}  "
  f"D = sigma^2/(2 alpha) = {R['conventions']['D']}")
P("=" * 92)

P("\n-- ESTIMATOR CALIBRATION (self-tests) --")
a = S["double_well_1d"]
P(f"gmam_selftest_1d_double_well_exact = 5.0 ; correct_minimiser = "
  f"{a['correct_minimiser']:.4f} ; attractor_lib_gmam = {a['attractor_lib_gmam']:.4f} "
  f"; straight_line = {a['straight_line']:.4f}  (cannot discriminate)")
c = S["curved_valley_2d"]
P(f"gmam_selftest_curved_valley_exact = 5.0 ; correct_minimiser = "
  f"{c['correct_minimiser']:.4f} (rel. err {c['correct_minimiser_rel_error']:.2e}) ; "
  f"attractor_lib_gmam = {c['attractor_lib_gmam']:.3f} (rel. err "
  f"{c['attractor_lib_rel_error']:.2f}) ; straight_line = {c['straight_line']:.3f}")
P(f"attractor_lib_gmam_descent_fraction = {c['attractor_lib_descent_fraction']:.3f} ; "
  f"path |y|max = {c['attractor_lib_path_max_abs_y']:.4f} vs true 1.0")
k = S["arrhenius_calibration_1d"]
P(f"arrhenius_calibration_fixed_prefactor_slope = {k['slope']:.4f} "
  f"[{k['ci_lo']:.4f},{k['ci_hi']:.4f}] (5 rungs, OLS, R2 {k['r2']:.4f}; "
  f"predicted 1.0)")
k2 = S["arrhenius_calibration_1d_moving_prefactor"]
P(f"arrhenius_calibration_moving_prefactor_slope = {k2['slope']:.4f} "
  f"[{k2['ci_lo']:.4f},{k2['ci_hi']:.4f}] (analytic {k2['predicted_slope']:.4f}, "
  f"R2 {k2['r2']:.4f})")

P("\n-- ANCHORS --")
A = R["anchors"]
P(f"n_anchor_candidates_examined = {A['n_examined']} ; n_usable = {A['n_usable']} "
  f"; seeds = {A['selected_seeds']}")
lc = R["ladder_coverage"]
P(f"cells = {lc['n_cells']} ; usable = {lc['n_cells_ok']} ; "
  f"seeds_with_full_ladder = {lc['n_seeds_with_full_ladder']}")
fe = [c_["escape"].get("fused_step_max_abs_error_vs_reference_step")
      for c_ in R["cells"] if "escape" in c_]
P(f"fused_step_max_abs_error_vs_reference_step = {max(x for x in fe if x is not None)}")

P("\n-- PER-GAIN MEANS OVER SEEDS (n = %d) --" % lc["n_seeds_with_full_ladder"])
hdr = ["measure"] + [f"g={g:.2f}" for g in G]
P("  " + f"{hdr[0]:<42s}" + "".join(f"{h:>13s}" for h in hdr[1:]))
show = ["saddle_unstable_eigenvalue", "well_slowest_relaxation_rate_mean",
        "well_slowest_relaxation_rate_lo", "well_slowest_relaxation_rate_hi",
        "fw_action_lo_to_saddle", "fw_action_hi_to_saddle", "fw_action_mean",
        "fw_action_lo_to_hi", "fw_action_hi_to_lo",
        "critical_displacement_mean", "recovery_1e_fixed_mean",
        "recovery_1e_halfcrit_mean", "fp_separation",
        "barrier_generative_nats_DENSITY", "mean_dwell_midpoint_DENSITY_RUN"]
rules = [k_ for k_ in grp["per_seed_series"] if k_.startswith("mfpt[")]
show += rules
for nm in show:
    v = grp["ordering"].get(nm)
    if not v:
        continue
    P("  " + f"{nm:<42s}" + "".join(f"{x:>13.4f}" for x in v["per_gain_mean"]))
P("  (sd over seeds)")
for nm in show:
    v = grp["ordering"].get(nm)
    if not v:
        continue
    P("  " + f"{nm:<42s}" + "".join(f"{x:>13.4f}" for x in v["per_gain_sd"]))

for tag, block in (("declared, the four gain levels 0.76/0.84/0.90/1.00",
                    grp["ordering_four_gain_levels"]),
                   ("secondary, all five gains including the 0.86 anchor",
                    grp["ordering"])):
    P(f"\n-- ORDERING vs GAIN [{tag}] (mean within-seed Spearman rho; within-seed "
      "gain-label permutation null, 10,000 draws) --")
    for nm in show:
        v = block.get(nm)
        if not v:
            continue
        P(f"rho[{nm}] = {v['mean_spearman']:+.4f} "
          f"[{v['ci_lo']:+.4f},{v['ci_hi']:+.4f}] (n = {v['n_seeds']} seeds, "
          f"within-seed label permutation, P = {v['p_perm']:.4f}); strictly "
          f"monotone in {v['n_seeds_strictly_monotone']}/{v['n_seeds']} seeds "
          f"(per-seed chance {v['per_seed_null_prob_perfect_monotone']:.4f})")

P("\n-- EXCLUSION BOUNDS, gain 0.76 -> 1.00 (seed-level bootstrap) --")
for nm in show:
    b = grp["exclusion_bounds_B0_to_B3"].get(nm)
    if not b:
        continue
    fold = ""
    if "fold_change" in b:
        fold = (f" ; fold = {b['fold_change']['mean']:.3f} "
                f"[{b['fold_change']['ci_lo']:.3f},{b['fold_change']['ci_hi']:.3f}]")
    P(f"d[{nm}] = {b['mean_change']:+.4f} [{b['ci_lo']:+.4f},{b['ci_hi']:+.4f}] "
      f"(n = {b['n_seeds']} seeds, bootstrap){fold}")

P("\n-- THE DECIDING TEST: Arrhenius slopes (H0: slope = 1) --")
for key, v in grp["arrhenius"].items():
    if "prefactor_corrected" in key:
        continue
    P(f"slope[{key}] = {v['mean_slope']:+.4f} "
      f"[{v['ci_lo_boot']:+.4f},{v['ci_hi_boot']:+.4f}] "
      f"(n = {v['n_seeds']} seeds, one-sample t vs 1, P = {v['p_vs_h0']:.3e}; "
      f"vs 0, P = {v['p_vs_zero']:.3e}; mean within-seed R2 = {v['mean_r2']:.4f})")
P("  [secondary, outside the declared analysis: prefactor-corrected]")
for key, v in grp["arrhenius"].items():
    if "prefactor_corrected" not in key:
        continue
    if not key.startswith(("fw_action_mean", "barrier_generative")):
        continue
    P(f"slope[{key}] = {v['mean_slope']:+.4f} "
      f"[{v['ci_lo_boot']:+.4f},{v['ci_hi_boot']:+.4f}] (n = {v['n_seeds']}, "
      f"P vs 1 = {v['p_vs_h0']:.3e})")

P("\n-- OVER-PREDICTION of the change in escape time, gain 0.76 -> 1.00 --")
for rn, v in grp["over_prediction"].items():
    P(f"[{rn}] measured = {v['measured_fold']['median']:.2f}x "
      f"[{v['measured_fold']['lo']:.2f},{v['measured_fold']['hi']:.2f}] ; "
      f"FW predicts {v['fw_predicted_fold']['median']:.2f}x "
      f"(ratio {v['fw_over_prediction']['median']:.2f}x "
      f"[{v['fw_over_prediction']['lo']:.2f},{v['fw_over_prediction']['hi']:.2f}]) ; "
      f"density predicts {v['density_predicted_fold']['median']:.4g}x "
      f"(ratio {v['density_over_prediction']['median']:.4g}x "
      f"[{v['density_over_prediction']['lo']:.4g},"
      f"{v['density_over_prediction']['hi']:.4g}])")

P("\n-- CENSORING and horizons --")
cens = {}
for c_ in R["cells"]:
    if "escape" not in c_:
        continue
    for d in ("lo_to_hi", "hi_to_lo"):
        for rn, st in c_["escape"][d]["rules"].items():
            cens.setdefault(rn, []).append(st["censored_frac"])
for rn, v in cens.items():
    P(f"censored_fraction[{rn}] = max {max(v):.4f}, mean {np.mean(v):.4f} "
      f"(n = {len(v)} cell-directions)")

P("\n-- THE FINITE-DIFFERENCE MINIMISER (gmam) ON THE CELLS OF THE FIRST NETWORK --")
for c_ in R["cells"]:
    if c_.get("anchor_seed") != 20260922 or "fw_action" not in c_:
        continue
    f_ = c_["fw_action"]
    if f_.get("attractor_lib_gmam_A_to_B") is None:
        continue
    P(f"  gain {c_['gain']:.2f}: correct S(lo->hi) = {f_['A_to_B']:.4f} ; "
      f"attractor_lib.gmam = {f_['attractor_lib_gmam_A_to_B']:.4f} ; "
      f"straight line = {f_['straight_line_A_to_B']:.4f}  "
      f"(overstates by {f_['attractor_lib_gmam_A_to_B']/f_['A_to_B']:.1f}x)")

P("\n-- INTERNAL CONSISTENCY of the minimum-action paths --")
r1, r2 = [], []
for c_ in R["cells"]:
    if "fw_action" not in c_ or not c_.get("saddle_found"):
        continue
    f_ = c_["fw_action"]
    r1.append(abs(f_["A_to_B"] - f_["A_to_saddle"]))
    r1.append(abs(f_["B_to_A"] - f_["B_to_saddle"]))
    r2.append(max(f_["saddle_to_B"], f_["saddle_to_A"]))
P(f"max |S(A->B) - S(A->saddle)| over cells = {max(r1):.5f} "
  f"(the downhill leg must cost nothing)")
P(f"max S(saddle->well) over cells = {max(r2):.5f} (must be ~0)")

P("\n-- DECISION --")
d = grp["decision"]
P(f"independent measures entered: {d['independent_measures']}")
P(f"ordering failures: {d['failures'] or 'none'}")
P(f"strictly monotone per measure: {d['per_measure_seeds_strictly_monotone']}")
P(f"wells_order_the_gains = {d['jacobian_split']['wells_order_the_gains']}")
P(f"FW slope = {d['fw_slope']:+.4f} {d['fw_slope_ci']} ; includes 1: "
  f"{d['fw_slope_includes_one']}")
P(f"density slope = {d['density_slope']:+.4f} {d['density_slope_ci']} ; includes 1: "
  f"{d['density_slope_includes_one']}")

TRACE = paths.out("analyses", "recurrent_networks", "out", "escape_ladder_trace.txt")
with open(TRACE, "w", encoding="utf-8", newline="\n") as fh:
    fh.write("\n".join(lines))
P(f"\nwritten: {TRACE}")


# --------------------------------------------------------------------- Fig. 2e (right)
def slope_entry(v):
    """One regression as printed in the trace above (slope and interval to 4 decimals, P to 4 significant
    digits, R^2 to 4 decimals)."""
    return dict(slope=float(f"{v['mean_slope']:+.4f}"),
                ci95=[float(f"{v['ci_lo_boot']:+.4f}"), float(f"{v['ci_hi_boot']:+.4f}")],
                n_networks=int(v["n_seeds"]),
                p_vs_slope_1=float(f"{v['p_vs_h0']:.3e}"),
                p_vs_slope_0=float(f"{v['p_vs_zero']:.3e}"),
                mean_within_network_r2=float(f"{v['mean_r2']:.4f}"))


rule = R["settings"]["primary_rule"]
resolved = grp["over_prediction"][rule]
arr = dict(
    description=("Arrhenius regressions of log mean first-passage time on two barrier estimates across the gain "
                 "levels of trained bistable recurrent networks (128 chains per level). Slope and 95% interval "
                 "over networks whose escape times are resolved at both ends of the gain range."),
    response="log mean first-passage time to the separatrix saddle",
    minimum_action=slope_entry(grp["arrhenius"][f"fw_action_mean -> log_mfpt[{rule}]"]),
    density_barrier=slope_entry(grp["arrhenius"][f"barrier_generative_nats_DENSITY -> log_mfpt[{rule}]"]),
    n_networks_resolved=int(resolved["n_seeds_resolved"]),
    n_networks=int(resolved["n_seeds_total"]),
)
SLOPES = paths.out("figure_reproduction", "inputs", "arrhenius_slopes.json")
with open(SLOPES, "w", encoding="utf-8", newline="\n") as fh:
    fh.write(json.dumps(arr, ensure_ascii=False, indent=1) + "\n")
P(f"written: {SLOPES}")
