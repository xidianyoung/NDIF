# -*- coding: utf-8 -*-
r"""Table of the four observation-aware sleep rate corrections (Fig. 5e, Supplementary Table 11).

Eight rows: the deeper/shallower rate ratios S1/Wake and S2/S1 (stage codes N1/Wake, N2/N1) under
four corrections, each taken from the output of the analysis that computes it:

  0  two-lag multivariate IV (2 s)      stage_contrasts.py, by_window/2/contrasts/lam_mean_iv|<pair>
  1  surrogate subtraction + bootstrap  stage_contrasts.py, by_window/2/contrasts/lam_mean_ncbc|<pair>
     (2 s)                              rows 0 and 1: the stored fold is shallower/deeper, so the
                                        table gives its reciprocal, with the reciprocal limits
                                        exchanged (1/fold, 1/fold_hi, 1/fold_lo)
  2  random-direction two-lag (1 s)     random_direction_rates.py, contrasts/<pair>/random/dlog_lam
  3  six-lag grid fit (1 s)             ../rate_identifiability/sleep_lag_profile.py,
                                        <shallow>-><deep>|lam_calibrated|1
                                        rows 2 and 3: fold and limits copied

Columns: contrast, method_index, method, window_s, fold, ci95_lo, ci95_hi, n_participants,
bootstrap_draws (4,000; 10,000 for the six-lag fit), participant_summary.  Floats are written
with '%.17g'.  Nothing is fitted or resampled here.

Inputs   figure_reproduction/inputs/sleep_stage_contrasts.json,
         analyses/sleep_stage_contrasts/out/sleep_random_direction_two_lag.json,
         analyses/rate_identifiability/out/sleep_lag_profile.json (below NDIF_OUT when it is set;
         each can be given on the command line, e.g. the Source Data copies
         analysis_inputs/sleep/sleep_random_direction_two_lag.json and sleep_multilag_analysis.json)
Output   figure_reproduction/inputs/sleep_rate_corrections.csv
Runtime  under a second.

    python analyses/sleep_stage_contrasts/rate_corrections.py
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from analyses.common import paths  # noqa: E402

PAIRS = (("Wake", "N1"), ("N1", "N2"))
METHODS = (
    ("Multivariate two-lag IV (2 s)", 2,
     "median participant log contrast after mean night log rates"),
    ("Surrogate + bootstrap (2 s)", 2,
     "median participant log contrast after mean night log rates"),
    ("Random-direction two-lag (1 s)", 1,
     "median participant contrast from median directional log differences"),
    ("Six-lag grid fit (1 s)", 1,
     "mean participant log contrast after mean night cell rates"),
)
COLUMNS = ("contrast", "method_index", "method", "window_s", "fold", "ci95_lo", "ci95_hi",
           "n_participants", "bootstrap_draws", "participant_summary")


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rows(stage_contrasts, random_direction, six_lag):
    con = stage_contrasts["by_window"]["2"]["contrasts"]
    rd = random_direction["contrasts"]
    six = six_lag["contrasts"] if "contrasts" in six_lag else six_lag["rungs"]
    out = []
    for a, b in PAIRS:
        pair = f"{a}>{b}"
        for mi, (label, window, summary) in enumerate(METHODS):
            if mi < 2:
                item = con[f"{('lam_mean_iv', 'lam_mean_ncbc')[mi]}|{pair}"]
                fold, lo, hi = 1 / item["fold"], 1 / item["fold_hi"], 1 / item["fold_lo"]
            elif mi == 2:
                item = rd[pair]["random"]["dlog_lam"]
                fold, lo, hi = item["fold"], item["fold_lo"], item["fold_hi"]
            else:
                item = six[f"{a}->{b}|lam_calibrated|1"]
                fold, lo, hi = item["fold"], item["fold_lo"], item["fold_hi"]
            out.append((f"{b}/{a}", mi, label, window, fold, lo, hi, item["n"],
                        10000 if mi == 3 else 4000, summary))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--contrasts",
                    default=paths.out("figure_reproduction", "inputs", "sleep_stage_contrasts.json"))
    ap.add_argument("--random-direction",
                    default=paths.out("analyses", "sleep_stage_contrasts", "out",
                                      "sleep_random_direction_two_lag.json"))
    ap.add_argument("--six-lag",
                    default=paths.out("analyses", "rate_identifiability", "out",
                                      "sleep_lag_profile.json"))
    a = ap.parse_args()

    table = rows(load(a.contrasts), load(a.random_direction), load(a.six_lag))
    p = paths.out("figure_reproduction", "inputs", "sleep_rate_corrections.csv")
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(COLUMNS)
        for r in table:
            w.writerow(["%.17g" % v if isinstance(v, float) else v for v in r])
    for r in table:
        print(f"  {r[0]:8s} {r[2]:32s} fold {r[4]:.3f} [{r[5]:.3f}, {r[6]:.3f}]  n = {r[7]}")
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
