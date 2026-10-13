"""Separation at which a basin divides in each record's own geometry (Fig. 1f).

The threshold of exactly 2 is the result of Behboodian (1970) for an equal-weight, equal-covariance pair of
Gaussians.  The recordings have neither equal regime covariances nor, in two of the four cohorts, equal
weights.  Per record this script computes the population division threshold: the generating separation
delta_div at which the two-component mixture with the record's own Sigma_0, Sigma_1 and weight w1, with the
displacement along the record's own direction (normalised in the mean covariance), acquires a second mode.
No sampling is involved: it is the exact ridgeline mode count (landscape_lib.ridgeline_modes_2, 1,201
points) at the population parameters, bisected in delta on [0, 12] to a tolerance of 0.001.

Also per record:
    delta_div_real_w05   the same with w1 = 0.5;
    delta_div_equalcov   Sigma_0 = Sigma_1 = (Sigma_0 + Sigma_1) / 2 and w1 = 0.5, which must return 2 and is
                         the check on the bisection.

Input    paths.cache("separation_floors")/generators/<cohort>.pkl (extract_generators.py).
Outputs  analyses/separation_floors/out/division_threshold.json, division_threshold.csv.
Reported in  Fig. 1f, separation at which a basin divides (participant medians 2.34 HCP fMRI, 2.58 HCP MEG,
         2.76 ds003838, 2.33 eegmmidb): /cohorts/<cohort>/delta_div_real_subject_median.

Runtime about 2 min on 12 workers.

    python analyses/separation_floors/division_threshold.py --workers 12
"""

from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import argparse                                                       # noqa: E402
import csv                                                            # noqa: E402
import json                                                           # noqa: E402
import pickle                                                         # noqa: E402
import sys                                                            # noqa: E402
import time                                                           # noqa: E402
from concurrent.futures import ProcessPoolExecutor                    # noqa: E402
from pathlib import Path                                              # noqa: E402

import numpy as np                                                    # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from analyses.common import paths                                     # noqa: E402
from analyses.landscape_readouts import landscape_lib as AL           # noqa: E402

COHORTS = ["hcp_fmri", "hcp_meg", "ds003838", "eegmmidb"]


def bisect_divide(mu, u, S1, S2, w1, hi=12.0, tol=1e-3):
    """Smallest delta at which the population mixture has two modes; inf if none below `hi`."""
    def bimodal(g):
        return AL.ridgeline_modes_2(mu, S1, mu + g * u, S2, w1=w1, n=1201).n_modes >= 2
    if bimodal(0.0):
        return 0.0
    if not bimodal(hi):
        return float("inf")
    lo = 0.0
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if bimodal(mid):
            hi = mid
        else:
            lo = mid
    return float(hi)


def one(rec):
    r0, r1 = rec["regimes"]
    S1, S2 = r0["Sigma"], r1["Sigma"]
    Sbar = 0.5 * (S1 + S2)
    e = r1["mu"] - r0["mu"]
    s = float(np.sqrt(max(e @ np.linalg.solve(Sbar, e), 1e-300)))
    u = e / s
    w1 = rec["w1_real"]
    return dict(cohort=rec["cohort"], record=rec["record"], subject=rec["subject"],
                sep_real=float(rec["sep_real"]), w1=float(w1),
                cond0=float(np.linalg.cond(S1)), cond1=float(np.linalg.cond(S2)),
                delta_div_real=bisect_divide(r0["mu"], u, S1, S2, w1),
                delta_div_real_w05=bisect_divide(r0["mu"], u, S1, S2, 0.5),
                delta_div_equalcov=bisect_divide(r0["mu"], u, Sbar, Sbar, 0.5))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--cohort", nargs="+", default=COHORTS)
    args = ap.parse_args()
    gen = os.path.join(paths.cache("separation_floors"), "generators")
    res = dict(created=time.strftime("%Y-%m-%d %H:%M:%S"), cohorts={})
    allrows = []
    for c in args.cohort:
        with open(os.path.join(gen, f"{c}.pkl"), "rb") as fh:
            recs = pickle.load(fh)
        rows = []
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for r in ex.map(one, recs, chunksize=2):
                rows.append(r)
        allrows += rows
        subs = sorted(set(r["subject"] for r in rows))
        per_sub = np.array([np.median([r["delta_div_real"] for r in rows if r["subject"] == s])
                            for s in subs])
        v = np.array([r["delta_div_real"] for r in rows])
        vw = np.array([r["delta_div_real_w05"] for r in rows])
        eq = np.array([r["delta_div_equalcov"] for r in rows])

        def q(x):
            x = np.asarray(x, float)
            x = x[np.isfinite(x)]
            return [float(np.percentile(x, p)) for p in (5, 25, 50, 75, 95)] if len(x) else None
        res["cohorts"][c] = dict(
            n_records=len(rows), n_subjects=len(subs),
            delta_div_real_subject_median=float(np.median(per_sub)),
            delta_div_real_subject_q=q(per_sub),
            delta_div_real_record_q=q(v),
            delta_div_real_w05_record_q=q(vw),
            delta_div_equalcov_record_q=q(eq),
            equalcov_check_max_abs_dev_from_2=float(np.max(np.abs(eq[np.isfinite(eq)] - 2.0))),
            n_infinite=int(np.sum(~np.isfinite(v))))
        x = res["cohorts"][c]
        print(f"{c:10s} n={len(rows)} subj={len(subs)}  population division threshold in this "
              f"cohort's own geometry: subject median {x['delta_div_real_subject_median']:.3f} "
              f"(record 5-95% {x['delta_div_real_record_q'][0]:.3f}-{x['delta_div_real_record_q'][4]:.3f}); "
              f"at w1=0.5 {x['delta_div_real_w05_record_q'][2]:.3f}; equal-covariance check "
              f"{x['delta_div_equalcov_record_q'][2]:.4f} (max |dev from 2| "
              f"{x['equalcov_check_max_abs_dev_from_2']:.2e})", flush=True)

    target = paths.out("analyses", "separation_floors", "out", "division_threshold.json")
    with open(target, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(res, indent=1))
    with open(paths.out("analyses", "separation_floors", "out", "division_threshold.csv"), "w",
              newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(allrows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(allrows)
    print("wrote division_threshold.json and division_threshold.csv")


if __name__ == "__main__":
    main()
