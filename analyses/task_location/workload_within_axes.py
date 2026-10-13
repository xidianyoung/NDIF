# -*- coding: utf-8 -*-
"""Workload EEG (COG-BCI, 28 participants): within-participant, cross-validated axes.

The group analysis (workload_axes.py) reads every cell along a leave-one-participant-out GROUP axis.  That
estimator addresses a shared cortical direction, but it cannot separate "the manipulation does not move the state"
from "the manipulation moves each participant's state along a direction of their own".  This script estimates each
participant's axis on one half of their OWN data and reads the effect on the other half, with the two halves
swapped, and with the load arm split the same way so that the two domains are matched:

  sleepiness : axis from the eyes-open resting epochs  -> read on the eyes-closed epochs, and back
  load       : axis from the n-back family             -> read on the MATB family,     and back

No window that contributes an axis contributes its readout.  Spaces S248 and S8 and the group summaries (10,000
bootstrap resamples, 10,000 sign-flip draws, seed 20260921) as in workload_axes.py.

Inputs:  as workload_axes.py (NDIF_COGBCI_CACHE, NDIF_COGBCI_RAW).
Output:  analyses/task_location/out/workload_within_person.json (NDIF_OUT relocates it).
Reported in: Extended Data Fig. 4b and Supplementary Table 7b: own cross-task axis 0.273 [0.137, 0.402]
         (key S248/LP_cv) and its 8-PC variant 0.245 [0.114, 0.371] (key S8/LP_cv).
Runtime: about 15 s on one desktop.

    python analyses/task_location/workload_within_axes.py --boot 10000 --perm 10000
"""
from __future__ import annotations
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")       # one BLAS thread: the setting of the stored outputs
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse, json, sys, time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO / "analyses" / "workload_certificates" / "features"))
from analyses.common import paths                                   # noqa: E402
import cog_features as C                                            # noqa: E402
from cog_features import log, cos_unit                              # noqa: E402
from workload_axes import Data, summarise, _ps_from, resid          # noqa: E402


def sleep_cells_eye(d, w, eye, key="Z"):
    got = d.sleep_m(eye)
    if got is None or w is None:
        return None
    ks, mw, mb = got
    if mw.std() == 0:
        return None
    groups = [resid(d.rest[k][key], d.rest[k]["seg"]) for k in ks]
    mu = [d.rest[k][key].mean(axis=0) for k in ks]
    return _ps_from(groups, mu, w, mw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--perm", type=int, default=10000)
    a = ap.parse_args()
    t0 = time.time()
    subs = C.load_subjects()
    kss, rsme = C.load_kss(), C.load_rsme()
    pca = C.fit_group_pca(subs)
    ids = sorted(subs)
    D = {i: Data(subs[i], pca, kss, rsme) for i in ids}
    res = {"design": "within-participant cross-validated axes"}

    for key, spacename in (("Z", "S248"), ("Z8", "S8")):
        cells = {"SP_cv": [], "SS_cv": [], "LP_cv": [], "LS_cv": [],
                 "SP_insample": [], "LP_insample": []}
        for i in ids:
            d = D[i]
            # ---- sleepiness: EO axis -> EC readout, and back
            pos, spr, pos_in = [], [], []
            for est, app in (("EO", "EC"), ("EC", "EO")):
                w = d._axis_eye(est, key)
                r = sleep_cells_eye(d, w, app, key)
                if r:
                    pos.append(r["beta_pos"]); spr.append(r["beta_spr"])
                r_in = sleep_cells_eye(d, w, est, key)
                if r_in:
                    pos_in.append(r_in["beta_pos"])
            cells["SP_cv"].append(float(np.mean(pos)) if pos else np.nan)
            cells["SS_cv"].append(float(np.mean(spr)) if spr else np.nan)
            cells["SP_insample"].append(float(np.mean(pos_in)) if pos_in else np.nan)
            # ---- load: n-back axis -> MATB readout, and back
            pos, spr, pos_in = [], [], []
            for est, app in (("nback", "matb"), ("matb", "nback")):
                w = d.w_load(key, est)
                if w is None:
                    continue
                r = d.load_cells(w, key, family=app)
                if r:
                    pos.append(float(np.mean([x["beta_pos"] for x in r])))
                    spr.append(float(np.mean([x["beta_spr"] for x in r])))
                r_in = d.load_cells(w, key, family=est)
                if r_in:
                    pos_in.append(float(np.mean([x["beta_pos"] for x in r_in])))
            cells["LP_cv"].append(float(np.mean(pos)) if pos else np.nan)
            cells["LS_cv"].append(float(np.mean(spr)) if spr else np.nan)
            cells["LP_insample"].append(float(np.mean(pos_in)) if pos_in else np.nan)

        summ = {k: summarise(k, v, a.boot, a.perm) for k, v in cells.items()}
        I = np.array(cells["LP_cv"]) - np.array(cells["LS_cv"]) \
            - (np.array(cells["SP_cv"]) - np.array(cells["SS_cv"]))
        summ["I_signed_cv"] = summarise("I_signed_cv", I, a.boot, a.perm)
        # transfer ratios against the matched in-sample ceiling
        summ["transfer_ratio_sleep"] = (summ["SP_cv"]["mean"] / summ["SP_insample"]["mean"]
                                        if summ["SP_insample"]["mean"] else None)
        summ["transfer_ratio_load"] = (summ["LP_cv"]["mean"] / summ["LP_insample"]["mean"]
                                       if summ["LP_insample"]["mean"] else None)
        # matched within-participant axis reliabilities
        rel_s = [abs(cos_unit(D[i]._axis_eye("EO", key), D[i]._axis_eye("EC", key))) for i in ids]
        rel_l = [abs(cos_unit(D[i].w_load(key, "nback"), D[i].w_load(key, "matb"))) for i in ids]
        summ["axis_reliability_sleep_EO_EC"] = summarise(
            "rel", [x for x in rel_s if np.isfinite(x)], a.boot, a.perm)
        summ["axis_reliability_load_nback_matb"] = summarise(
            "rel", [x for x in rel_l if np.isfinite(x)], a.boot, a.perm)
        res[spacename] = summ
        log(f"{spacename}: SP_cv={summ['SP_cv']['mean']:+.4f} (in-sample {summ['SP_insample']['mean']:+.4f}) "
            f"LP_cv={summ['LP_cv']['mean']:+.4f} (in-sample {summ['LP_insample']['mean']:+.4f}) "
            f"SS_cv={summ['SS_cv']['mean']:+.4f} LS_cv={summ['LS_cv']['mean']:+.4f}")

    res["elapsed_s"] = time.time() - t0
    Path(paths.out("analyses", "task_location", "out", "workload_within_person.json")).write_text(
        json.dumps(res, indent=1, default=float), encoding="utf-8", newline="\n")
    log(f"DONE {time.time()-t0:.0f}s -> workload_within_person.json")


if __name__ == "__main__":
    main()
