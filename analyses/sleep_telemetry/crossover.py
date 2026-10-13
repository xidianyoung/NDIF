# -*- coding: utf-8 -*-
r"""Telemetry temazepam/placebo crossover of the directional-spread contrasts (Fig. 4c,
Supplementary Table 9) and the alternative-processing spread contrasts of the cassette nights
(Supplementary Table 8, second part).

Cells (spread_lib.cellify) are built from the per-direction rows of bout_spread_rows.py: the cell
spread is the median spread of the night-stage's kept directions; the cell rate is the median
over directions of the one-step rate -ln r(1)/dt (`onestep`) or of the same rate after
correcting r(1) for the per-bout centring of 30-window blocks (`onestep_bc`, the inverse of a
simulated bias curve, generator seeded 20260921 + 1).  A cell enters a contrast if at least half
of its directions have a finite rate.

Contrasts (spread_lib.contrast_stat): per adjacent pair, night log ratios shallower/deeper are
averaged within participant; fold = exp(median), 95% percentile interval from participant
resamples (generator 20260921 + 7, fresh per contrast), sign and Wilcoxon tests; and a
within-night stage-label permutation null size-matched to the observed participant count
(spread_lib.perm_null, generator 20260921 + 11).

Alternative processing (cassette)  `onestep` spread contrasts over 129 of the 153 cassette
    nights, --nboot 4000 resamples and --nperm 5000 permutations:
    Wake/S1 1.186 [1.055, 1.245] (n 52), S1/S2 1.000 (67), S2/S3 1.180 (40), S3/S4 1.111 (14);
    label-permutation P 0.0004, 0.9942, 0.0004, 0.0500.  The 24 nights in NOT_IN_ALTERNATIVE_SET
    are not part of this contrast set.
Telemetry (44 nights; ST7161/ST7162 have no two usable stages: 42 nights, 21 participants, each
    with one placebo and one temazepam night, from ST-subjects.xls)  per arm, the `onestep`
    spread and `onestep_bc` rate contrasts (2,000 resamples, generator 20260921 + 41; 2,000
    permutations, 20260921 + 43): placebo S2/S3 1.178 [1.120, 1.314] (n 16), temazepam 1.196
    [1.071, 1.284] (15), placebo S3/S4 1.282 (7), temazepam 1.185 (8); per stage, the paired
    temazepam/placebo contrast (S2 spread 1.002, Wilcoxon P 0.5854); and the cassette contrasts
    beside them (`cassette_comparison`).  Stage codes N1-N4 stand for the R&K stages S1-S4.

Inputs   paths.cache("sleep_bout_rows") (NDIF_SLEEP_BOUT_ROWS_CACHE): the two row files of
         bout_spread_rows.py (or the Source Data copies, analysis_inputs/telemetry/stored_outputs);
         ST-subjects.xls in NDIF_SLEEPEDF_RAW.
Outputs  figure_reproduction/inputs/sleep_telemetry_crossover.json
         analyses/sleep_telemetry/out/alternative_processing.csv   (below NDIF_OUT when set)
Runtime  about 1 min on one core.

    python analyses/sleep_telemetry/crossover.py --nperm 5000 --nboot 4000
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse                                                       # noqa: E402
import csv                                                            # noqa: E402
import json                                                           # noqa: E402
import math                                                           # noqa: E402
import sys                                                            # noqa: E402
import time                                                           # noqa: E402
import warnings                                                       # noqa: E402
from pathlib import Path                                              # noqa: E402

import numpy as np                                                    # noqa: E402
from scipy import stats as sst                                        # noqa: E402

np.seterr(all="ignore")
warnings.filterwarnings("ignore", category=RuntimeWarning)

HERE = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))
from analyses.common import paths  # noqa: E402
import spread_lib as L  # noqa: E402

SEED = L.SEED
NOT_IN_ALTERNATIVE_SET = (
    "SC4121", "SC4122", "SC4131", "SC4141", "SC4142", "SC4151", "SC4152", "SC4162",
    "SC4241", "SC4251", "SC4252", "SC4261", "SC4262", "SC4271", "SC4272", "SC4281",
    "SC4282", "SC4292", "SC4341", "SC4342", "SC4351", "SC4431", "SC4432", "SC4441")
DESIGN = ("Sleep-EDF Telemetry is a temazepam/placebo crossover: each participant contributes one night "
          "under each condition.  The arms are analysed separately; within an arm each participant "
          "contributes exactly one night.")
OUT_JSON = ("figure_reproduction", "inputs", "sleep_telemetry_crossover.json")
OUT_CSV = ("analyses", "sleep_telemetry", "out", "alternative_processing.csv")
CSV_HEADER = ["arm", "quantity", "contrast", "n_subjects", "k_a_greater", "fold_subject",
              "ci95_lo", "ci95_hi", "sign_p_subject", "wilcoxon_p_subject",
              "n_nights", "k_nights", "fold_night", "sign_p_night",
              "perm_null_fold_mean", "perm_p_two_sided",
              "perm_null_n_subj_mean_before_subsample"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--nperm", type=int, default=5000)
    ap.add_argument("--nboot", type=int, default=L.N_BOOT)
    a = ap.parse_args()
    t0 = time.time()

    cas = L.load_rows("cassette")
    tel = L.load_rows("telemetry")
    alt_nights = [dict(n, directions=[d for d in n.get("directions", [])])
                  for n in cas["nights"] if n["stem"] not in NOT_IN_ALTERNATIVE_SET]

    # ---- bias curve of the per-bout-centred one-step estimator -------------
    bc = L.bias_curve(np.random.default_rng(SEED + 1))
    inv = L.make_bias_inverse(bc)

    # ---- cassette cells and contrasts ----------------------------------------
    arms = {arm: L.cellify(alt_nights, arm, inv) for arm in ("onestep", "onestep_bc")}
    wcheck = L.check_cell_width(arms["onestep"], alt_nights)
    assert wcheck["max_abs_diff"] == 0.0, wcheck
    print(f"{len(alt_nights)} cassette nights; cell-width check: {wcheck}", flush=True)

    contrasts = {}
    for arm, key in (("onestep", "width"), ("onestep_bc", "lam")):
        for (aa, bb) in L.CONTRASTS:
            rows = L.contrast_rows(arms[arm], aa, bb, key, require_resolved=True)
            st = L.contrast_stat(rows, np.random.default_rng(SEED + 7), n_boot=a.nboot)
            if st.get("testable") and key == "width":
                st["perm_null"] = L.perm_null(arms[arm], aa, bb, key, True, st,
                                              np.random.default_rng(SEED + 11), a.nperm)
            contrasts[f"{arm}|{key}|{aa}>{bb}"] = st
            s = st.get("subject", {})
            print(f"  {arm:10s} {key:5s} {aa}>{bb}: n={s.get('n_subjects')} "
                  f"fold={s.get('fold')} perm_p={st.get('perm_null', {}).get('p_perm_two_sided')}",
                  flush=True)

    with open(paths.out(*OUT_CSV), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(CSV_HEADER)
        for (aa, bb) in L.CONTRASTS:
            v = contrasts[f"onestep|width|{aa}>{bb}"]
            if not v.get("testable"):
                w.writerow(["onestep", "width", f"{aa}>{bb}"] + [""] * 14)
                continue
            s, n, pn = v["subject"], v["night"], v.get("perm_null", {})
            w.writerow(["onestep", "width", f"{aa}>{bb}", s["n_subjects"], s["k_a_greater"],
                        round(s["fold"], 5), round(s["fold_ci95"][0], 5), round(s["fold_ci95"][1], 5),
                        s["sign_p"], s["wilcoxon_p"], n["n_nights"], n["k_a_greater"],
                        round(n["fold"], 5), n["sign_p"],
                        pn.get("null_fold_mean"), pn.get("p_perm_two_sided"),
                        (pn.get("null_n_subjects_before_subsample") or {}).get("mean")])

    # ---- telemetry crossover ---------------------------------------------------
    dmap = L.telemetry_drug_map()
    tel_nights = [n for n in tel["nights"] if "rows" in n]
    labels = {n["stem"]: L.telemetry_label(n["stem"], dmap) for n in tel_nights}
    tarms = {}
    for drug in ("placebo", "temazepam"):
        sub = [n for n in tel_nights if labels[n["stem"]] == drug]
        tarms[drug] = dict(nights=sub, cells={est: L.cellify(sub, est, inv)
                                              for est in ("onestep", "onestep_bc")})
    telout = dict(
        design=DESIGN,
        n_recordings=len(tel_nights),
        n_subjects=len({s[:5] for s in labels}),
        assignment={k: v for k, v in sorted(labels.items())},
        dropped=tel["errors"],
        arms={}, drug_contrast={}, cassette_comparison={})
    for drug in ("placebo", "temazepam"):
        A = dict(n_nights=len(tarms[drug]["nights"]),
                 n_subjects=len({n["stem"][:5] for n in tarms[drug]["nights"]}),
                 stage_n={}, contrasts={})
        for s in L.STAGES:
            A["stage_n"][s] = sum(1 for m in tarms[drug]["cells"]["onestep"].values() if s in m)
        for arm, key in [("onestep", "width"), ("onestep_bc", "lam")]:
            for (aa, bb) in L.CONTRASTS:
                rows = L.contrast_rows(tarms[drug]["cells"][arm], aa, bb, key, require_resolved=True)
                st = L.contrast_stat(rows, np.random.default_rng(SEED + 41), n_boot=2000)
                if st.get("testable"):
                    st["perm_null"] = L.perm_null(tarms[drug]["cells"][arm], aa, bb, key, True, st,
                                                  np.random.default_rng(SEED + 43), 2000)
                A["contrasts"][f"{arm}|{key}|{aa}>{bb}"] = st
        telout["arms"][drug] = A
    # within-participant drug contrast per stage
    for arm, key in [("onestep", "width"), ("onestep_bc", "lam")]:
        for s in L.STAGES:
            P = {st[:5]: c[s][key] for st, c in tarms["placebo"]["cells"][arm].items() if s in c
                 and c[s]["resolved"] and c[s].get(key)}
            T = {st[:5]: c[s][key] for st, c in tarms["temazepam"]["cells"][arm].items() if s in c
                 and c[s]["resolved"] and c[s].get(key)}
            both = sorted(set(P) & set(T))
            if len(both) < 3:
                telout["drug_contrast"][f"{arm}|{key}|{s}"] = dict(n_subjects=len(both),
                                                                   testable=False)
                continue
            lr = np.array([math.log(T[x] / P[x]) for x in both], float)
            k = int((lr > 0).sum())
            telout["drug_contrast"][f"{arm}|{key}|{s}"] = dict(
                testable=True, n_subjects=len(both), k_temazepam_greater=k,
                fold_temazepam_over_placebo=float(np.exp(np.median(lr))),
                sign_p=float(sst.binomtest(k, len(lr), 0.5).pvalue),
                wilcoxon_p=float(sst.wilcoxon(lr).pvalue) if len(lr) >= 6 else None)
    # the cassette contrasts beside them
    for arm, key in [("onestep", "width"), ("onestep_bc", "lam")]:
        for (aa, bb) in L.CONTRASTS:
            telout["cassette_comparison"][f"{arm}|{key}|{aa}>{bb}"] = \
                contrasts[f"{arm}|{key}|{aa}>{bb}"]["subject"]
    p = paths.out(*OUT_JSON)
    Path(p).write_text(json.dumps(telout, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {p}\n      {paths.out(*OUT_CSV)}  ({(time.time() - t0) / 60:.2f} min)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
