# -*- coding: utf-8 -*-
"""Reported digit-span (ds003838) task-location values, copied from the output of ds003838_position_slopes.py.

Reads the group summary ('summary/position') of analyses/task_location/out/digit_span_position_analysis.json
(or the file given with --input) and writes
  figure_reproduction/inputs/digit_span_position_slopes.json
      memory, listen-only and memory-minus-listen slopes on the primary input u3 with their 95% bootstrap
      intervals: 0.06399, -0.00022 and 0.06421 [0.04565, 0.08285] (Fig. 4a; Extended Data Fig. 4a;
      Supplementary Table 7a)
  analyses/task_location/out/digit_span_supplementary_values.json
      the memory-minus-listen differences on the alternative inputs u1 (0.01684) and u2 (0.02267) with their
      intervals (Extended Data Fig. 4a and Supplementary Table 7a, source-only rows), the one-slot-shift control
      0.03540 [0.02210, 0.04902] and the slot-map permutation P 0.000999 with 1,000 draws (Supplementary Note 6),
      each with the key it is copied from.
Nothing is recomputed.  Runtime: under 1 s.

    python analyses/task_location/ds003838_summary.py [--input <digit_span_position_analysis.json>]
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyses.common import paths  # noqa: E402
from ds003838_common import SEED, log  # noqa: E402

DATASET = "ds003838 EEG digit span, n = 64 participants"
FIGURE_KEYS = ["slope_memory_u3", "slope_memory_u3_ci95", "slope_control_u3", "slope_control_u3_ci95",
               "did_u3_mean", "did_u3_ci95"]


def dump(obj, target):
    Path(target).write_text(json.dumps(obj, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    log(f"-> {target}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=None,
                    help="output of ds003838_position_slopes.py (default: its location under NDIF_OUT or the "
                         "repository)")
    a = ap.parse_args()
    src = Path(a.input) if a.input else Path(paths.out("analyses", "task_location", "out",
                                                         "digit_span_position_analysis.json"))
    pos = json.loads(src.read_text(encoding="utf-8"))
    S = pos["summary"]

    # the values of each input definition, as they are named in the reported tables
    N = {}
    for un in ("u3", "u1", "u2"):
        p = S["position"][un]
        N[f"did_{un}_mean"] = p["did"]["mean"]
        N[f"did_{un}_ci95"] = p["did"]["ci95"]
        N[f"did_{un}_perm_p"] = p["permutation"]["p_one_sided"]
        N[f"slope_memory_{un}"] = p["slope_memory"]["mean"]
        N[f"slope_memory_{un}_ci95"] = p["slope_memory"]["ci95"]
        N[f"slope_control_{un}"] = p["slope_control"]["mean"]
        N[f"slope_control_{un}_ci95"] = p["slope_control"]["ci95"]
        N[f"did_{un}_shift_null_mean"] = p["shift_null"]["mean"] if p["shift_null"] else None

    dump(dict(dataset=DATASET, seed=SEED, numbers={k: N[k] for k in FIGURE_KEYS}),
         paths.out("figure_reproduction", "inputs", "digit_span_position_slopes.json"))

    u3 = S["position"]["u3"]
    sup = dict(
        dataset=DATASET, seed=SEED, n_subjects=S["n_subjects"],
        numbers=dict(
            did_u1_mean=N["did_u1_mean"], did_u1_ci95=N["did_u1_ci95"],
            did_u2_mean=N["did_u2_mean"], did_u2_ci95=N["did_u2_ci95"],
            did_u3_one_slot_shift_mean=N["did_u3_shift_null_mean"],
            did_u3_one_slot_shift_ci95=u3["shift_null"]["ci95"] if u3["shift_null"] else None,
            did_u3_slot_map_permutation_p=N["did_u3_perm_p"],
            did_u3_slot_map_permutation_draws=u3["permutation"]["n_draws"]),
        source=dict(
            did_u1_mean="summary/position/u1/did/mean", did_u1_ci95="summary/position/u1/did/ci95",
            did_u2_mean="summary/position/u2/did/mean", did_u2_ci95="summary/position/u2/did/ci95",
            did_u3_one_slot_shift_mean="summary/position/u3/shift_null/mean",
            did_u3_one_slot_shift_ci95="summary/position/u3/shift_null/ci95",
            did_u3_slot_map_permutation_p="summary/position/u3/permutation/p_one_sided",
            did_u3_slot_map_permutation_draws="summary/position/u3/permutation/n_draws"),
        note=("Differences in differences (memory minus listen-only within-set-size slopes) on the alternative "
              "inputs u1 = serial position and u2 = min(k, 9); the one-slot circular shift of u3 is a control; the "
              "slot-map permutation P is one-sided, (1 + #null >= observed) / (1 + draws), and the permutation null "
              "range is not an effect-size interval."))
    dump(sup, paths.out("analyses", "task_location", "out", "digit_span_supplementary_values.json"))
    for k, v in sup["numbers"].items():
        log(f"{k} = {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
