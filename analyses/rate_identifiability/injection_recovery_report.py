# -*- coding: utf-8 -*-
"""Collect the injection-recovery calibrations of the four cohorts into the table that Fig. 3d reads.

Nothing is recomputed: every row is read from a calibration file written by one of the cohort arms.

  COG-BCI   analyses/workload_certificates/features/cog_state.py --cboot 50
            -> analyses/workload_certificates/features/out/rescaling_cog_calibration.json
  HCP fMRI  injection_recovery_fmri.py      -> out/injection_recovery_fmri_calibration.json
  ds003838  injection_recovery_ds003838.py  -> out/injection_recovery_ds003838_calibration.json
  HCP MEG   injection_recovery_meg.py       -> out/injection_recovery_meg_calibration.json

One row per injection rung (cohorts in the order COG-BCI, HCP fMRI, ds003838, HCP MEG):
  INJ-S             stride k against stride k+1 of the recorded series (dist / dyn series in COG-BCI), truth
                    ln((k+1)/k), and the same decimation on a VAR(1) surrogate of the same recordings;
  INJ-T             exact time rescaling of the recorded innovation stream, truth ln c (in HCP fMRI also at the
                    decimation's own block length);
  INJ-D             pure drive (truth for the rate exactly zero; the correlation-invariance ratio at a known
                    shift of the stationary correlation);
  INJ-R             pure reshaping of the relaxation spectrum (recovered reshaping RMS at a known truth).
Columns: cohort, quantity, mean, lo, hi (bootstrap over participants), n, n_positive, p, note.

Output   figure_reproduction/inputs/injection_recovery.csv
Reported Fig. 3d (innovation rescaling 0.68-1.11 of the injected change in COG-BCI, HCP fMRI and HCP MEG,
         -0.007-0.58 in ds003838; temporal decimation -0.157-0.663)
Runtime  seconds
Usage    python analyses/rate_identifiability/injection_recovery_report.py
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from analyses.common import paths                                          # noqa: E402

HERE = Path(__file__).resolve().parent
COG_CALIB = REPO / "analyses" / "workload_certificates" / "features" / "out" / "rescaling_cog_calibration.json"
KEYS = ["cohort", "quantity", "mean", "lo", "hi", "n", "n_positive", "p", "note"]


def J(p):
    p = Path(p) if p else None
    return json.loads(p.read_text(encoding="utf-8")) if p is not None and p.exists() else {}


def row(rows, pred, cohort, quantity, c, note=""):
    if isinstance(c, dict) and "mean" in c:
        rows.append(dict(prediction=pred, cohort=cohort, quantity=quantity,
                         mean=c["mean"], lo=c["lo"], hi=c["hi"], n=c.get("n"),
                         n_positive=c.get("n_pos"), p=c.get("p"), note=note))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cog", default=str(COG_CALIB))
    ap.add_argument("--fmri", default=paths.out("analyses", "rate_identifiability", "out",
                                                "injection_recovery_fmri_calibration.json"))
    ap.add_argument("--ds003838", default=paths.out("analyses", "rate_identifiability", "out",
                                                    "injection_recovery_ds003838_calibration.json"))
    ap.add_argument("--meg", default=paths.out("analyses", "rate_identifiability", "out",
                                               "injection_recovery_meg_calibration.json"))
    a = ap.parse_args()
    CC, CF, CD, CM = J(a.cog), J(a.fmri), J(a.ds003838), J(a.meg)
    rows = []

    # ---------------- calibration
    for name, C in (("COG-BCI", CC), ("HCP fMRI", CF), ("ds003838", CD), ("HCP MEG", CM)):
        for blk, tag in (("decimation_dist", "INJ-S dist"), ("decimation_dyn", "INJ-S dyn"),
                         ("decimation", "INJ-S"),
                         ("decimation_var1_control", "INJ-S on a VAR(1) surrogate"),
                         ("rescale", "INJ-T"), ("rescale_n19", "INJ-T at the decimation's n"),
                         ("drive", "INJ-D"), ("reshape", "INJ-R")):
            for k, v in (C.get(blk) or {}).items():
                if not isinstance(v, dict):
                    continue
                t = v.get("truth_dlog_lam")
                if t is not None and "recovered_dlog_lam" in v:
                    rows.append(dict(
                        prediction="CALIB", cohort=name,
                        quantity=f"{tag} {k}: dlog lambda recovered (truth {t:+.4f})",
                        mean=v["recovered_dlog_lam"]["mean"], lo=v["recovered_dlog_lam"]["lo"],
                        hi=v["recovered_dlog_lam"]["hi"], n=v.get("n"),
                        note=(f"fraction {v.get('fraction_recovered', float('nan')):.3f}; "
                              f"common factor fraction "
                              f"{v.get('fraction_recovered_dbar', float('nan')):.3f}; "
                              f"dlog D {v.get('dlog_D', {}).get('mean', float('nan')):+.4f}; "
                              f"landing slope {v.get('slope_landing', float('nan')):+.2f}")))
                if "recovered_reshape_rms" in v:
                    rows.append(dict(
                        prediction="CALIB", cohort=name,
                        quantity=f"{tag} {k}: reshaping recovered "
                                 f"(truth {v['truth_reshape_rms']['mean']:.4f})",
                        mean=v["recovered_reshape_rms"]["mean"],
                        lo=v["recovered_reshape_rms"]["lo"], hi=v["recovered_reshape_rms"]["hi"],
                        n=v.get("n"),
                        note=f"common-factor leak {v['recovered_dbar']['mean']:+.4f}"))
                if "true_corr_shift" in v and "p1_d_ratio" in v:
                    rows.append(dict(
                        prediction="CALIB", cohort=name,
                        quantity=f"{tag} {k}: P1 ratio at a true shift of "
                                 f"{v['true_corr_shift']['mean']:.4f}",
                        mean=v["p1_d_ratio"]["mean"], lo=v["p1_d_ratio"]["lo"],
                        hi=v["p1_d_ratio"]["hi"], n=v.get("n"),
                        note=f"rate leak {v['recovered_dlog_lam']['mean']:+.4f}"))

    target = paths.out("figure_reproduction", "inputs", "injection_recovery.csv")
    with open(target, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, KEYS, extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        for r in rows:
            if r["quantity"].startswith("INJ-"):
                w.writerow(r)
    print(f"injection_recovery.csv: {sum(r['quantity'].startswith('INJ-') for r in rows)} rows -> {target}")


if __name__ == "__main__":
    main()
