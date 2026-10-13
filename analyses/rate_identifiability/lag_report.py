# -*- coding: utf-8 -*-
r"""The phase-surrogate share and the calibrated white share of the sleep state by stage (1.0-s windows).

For every stage with a calibrated fit at 1.0-s windows, two rows are tabulated from the per-stage
summaries of sleep_lag_profile.py (subject means with percentile-bootstrap 95% intervals):

  "<stage> 1.0 s nu_fit"    the white share nu of the calibrated single-timescale fit
  "<stage> 1.0 s nu_inst"   the share of the observed variance reproduced by phase-randomised copies
                            of the same bouts (the instrument-noise covariance of the band-power
                            estimator, carried into the state space)

Columns: name, value (mean over subjects), lo, hi (95% interval), n (subjects), p and note (empty);
values rounded to five decimals.

Input:  analyses/rate_identifiability/out/sleep_lag_profile.json (sleep_lag_profile.py).
Output: figure_reproduction/inputs/phase_surrogate_variance_by_stage.csv (12 rows).
Supports Fig. 3b: phase-surrogate shares 0.36-0.57 (bars) and calibrated white shares rising from
0.45 at wake to 0.83 at stage 4 (circles).
Runtime: seconds.

    python analyses/rate_identifiability/lag_report.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyses.common import paths                                                 # noqa: E402
from lag_lib import STAGES                                                        # noqa: E402

KEYS = ("nu_fit", "nu_inst")


def main():
    src = paths.out("analyses", "rate_identifiability", "out", "sleep_lag_profile.json")
    p2 = json.loads(Path(src).read_text(encoding="utf-8"))
    rows = []

    def add(name, value, ci=None, n=None, p=None, note=""):
        rows.append(dict(name=name,
                         value=value,
                         lo=(ci[0] if ci else ""), hi=(ci[1] if ci else ""),
                         n=(n if n is not None else ""),
                         p=(p if p is not None else ""), note=note))

    for g in STAGES:
        v = p2["per_stage"].get(f"{g}|1", {})
        if "a_fit" not in v:
            continue
        for key in KEYS:
            if key in v:
                add(f"{g} 1.0 s {key}", round(v[key]["mean"], 5),
                    ci=(round(v[key]["lo"], 5), round(v[key]["hi"], 5)),
                    n=v[key]["n"])

    out = paths.out("figure_reproduction", "inputs", "phase_surrogate_variance_by_stage.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["name", "value", "lo", "hi", "n", "p", "note"],
                           lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
