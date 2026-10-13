# -*- coding: utf-8 -*-
r"""The window-scaling exponent of two rate estimators on the sleep state.

A recording analysed at 2.0-s and at 1.0-s windows is the SAME process.  A rate estimator that
identifies a restoring rate must return the same lambda either way; the ratio lambda(1 s)/lambda(2 s)
is 1 for an exact estimator.  Writing lambda ~ dt^(-theta), theta = 0 is an identified rate and
theta = 1 is a quantity that is the sampling rate and nothing else.

The ratio is computed on the SAME nights and stages for both estimators, from the per-recording rows
of sleep_lag_profile.py (uninjected cassette recordings):

  lam_calibrated  -ln(a)/dt with a from the calibrated single-timescale fit (lag_lib)
  lam_onestep     -ln r(1)/dt, the one-step estimator

for the window pairs 1 s / 2 s, 0.5 s / 1 s and 0.5 s / 2 s.  Subject is the unit: the mean over a
subject's night x stage cells of log lambda(fine) - log lambda(coarse), then a percentile-bootstrap
95% interval over subjects (lag_lib.boot_mean, 10,000 draws, one random stream with seed 20260922,
drawn in the order window pairs, then stages).  theta = log(ratio)/log(coarse/fine) with its interval
mapped through the same transformation.

Input:  analyses/rate_identifiability/out/sleep_lag_profile_rows.csv (sleep_lag_profile.py).
Output: figure_reproduction/inputs/window_scaling_exponent.json.
Supports Fig. 3c and Supplementary Table 1: theta of the one-step estimator 1.001 [0.978, 1.024]
and of the calibrated estimator 0.493 [0.424, 0.566] over the widest span (0.5 s to 2.0 s); pooled
ratios lambda(1 s)/lambda(2 s) per stage.
Runtime: seconds.

    python analyses/rate_identifiability/window_scaling.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyses.common import paths                                                 # noqa: E402
from lag_lib import STAGES, boot_mean                                             # noqa: E402


def _pair_block(piv, rng):
    out = {}
    for g in list(STAGES) + ["ALL"]:
        cell = {}
        for key in ("lam_calibrated", "lam_onestep"):
            d = piv[key]
            d = d if g == "ALL" else d[d.index.get_level_values("stage") == g]
            d = d[(d.w1 > 0) & (d.w2 > 0)]
            if len(d) < 5:
                continue
            sub = d.reset_index()
            sub["subject"] = sub.stem.str[:5]
            per = sub.groupby("subject").apply(
                lambda x: float(np.mean(np.log(x.w1) - np.log(x.w2))),
                include_groups=False)
            st = boot_mean(per.values, rng=rng)
            st["ratio"] = float(np.exp(st["mean"]))
            st["ratio_lo"] = float(np.exp(st["lo"]))
            st["ratio_hi"] = float(np.exp(st["hi"]))
            st["n_cells"] = int(len(d))
            cell[key] = st
        if cell:
            out[g] = cell
    return out


def main():
    df = pd.read_csv(paths.out("analyses", "rate_identifiability", "out", "sleep_lag_profile_rows.csv"))
    df = df[(df.inj_kind.isna()) | (df.inj_kind.astype(str) == "")]
    df = df[df.stem.astype(str).str.startswith("SC")]
    rng = np.random.default_rng(20260922)
    res = dict(note=("lambda(1 s) / lambda(2 s) on the same night and stage; an exact estimator "
                     "returns 1.000.  The calibrated estimator fits a = argmin over the expected "
                     "r_hat; the one-step estimator is -ln r(1)/dt."),
               per_stage={}, pooled={})
    pairs = [(1.0, 2.0), (0.5, 1.0), (0.5, 2.0)]
    have = set(df.win_s.unique())
    pairs = [p for p in pairs if p[0] in have and p[1] in have]
    res["pairs"] = [list(p) for p in pairs]
    res["by_pair"] = {}
    for fine, coarse in pairs:
        piv = {}
        for key in ("lam_calibrated", "lam_onestep", "nu_fit", "a_fit", "parity"):
            w1 = df[df.win_s == fine].set_index(["stem", "stage"])[key]
            w2 = df[df.win_s == coarse].set_index(["stem", "stage"])[key]
            piv[key] = pd.DataFrame({"w1": w1, "w2": w2}).dropna()
        res["by_pair"]["%g_over_%g" % (fine, coarse)] = _pair_block(piv, rng)
    piv = {}
    for key in ("lam_calibrated", "lam_onestep", "nu_fit", "a_fit", "parity"):
        w1 = df[df.win_s == 1.0].set_index(["stem", "stage"])[key]
        w2 = df[df.win_s == 2.0].set_index(["stem", "stage"])[key]
        piv[key] = pd.DataFrame({"w1": w1, "w2": w2}).dropna()

    for g in list(STAGES) + ["ALL"]:
        cell = {}
        for key in ("lam_calibrated", "lam_onestep"):
            d = piv[key]
            d = d if g == "ALL" else d[d.index.get_level_values("stage") == g]
            d = d[(d.w1 > 0) & (d.w2 > 0)]
            if len(d) < 5:
                continue
            # subject is the unit
            sub = d.reset_index()
            sub["subject"] = sub.stem.str[:5]
            per = sub.groupby("subject").apply(
                lambda x: float(np.mean(np.log(x.w1) - np.log(x.w2))),
                include_groups=False)
            st = boot_mean(per.values, rng=rng)
            st["ratio"] = float(np.exp(st["mean"]))
            st["ratio_lo"] = float(np.exp(st["lo"]))
            st["ratio_hi"] = float(np.exp(st["hi"]))
            st["n_cells"] = int(len(d))
            cell[key] = st
        for key in ("nu_fit", "a_fit", "parity"):
            d = piv[key]
            d = d if g == "ALL" else d[d.index.get_level_values("stage") == g]
            if len(d) < 5:
                continue
            sub = d.reset_index()
            sub["subject"] = sub.stem.str[:5]
            cell[f"{key}_w1"] = boot_mean(sub.groupby("subject").w1.mean().values, rng=rng)
            cell[f"{key}_w2"] = boot_mean(sub.groupby("subject").w2.mean().values, rng=rng)
        res["per_stage"][g] = cell

    # ---- the scaling exponent: lambda ~ dt^(-theta).
    # theta = 0 is an identified restoring rate; theta = 1 is a quantity that is
    # the sampling rate and nothing else.
    res["theta"] = {}
    for key in ("lam_calibrated", "lam_onestep"):
        rows_ = []
        for (fine, coarse) in pairs:
            b = res["by_pair"]["%g_over_%g" % (fine, coarse)].get("ALL", {}).get(key)
            if not b:
                continue
            th = np.log(b["ratio"]) / np.log(coarse / fine)
            th_lo = np.log(b["ratio_lo"]) / np.log(coarse / fine)
            th_hi = np.log(b["ratio_hi"]) / np.log(coarse / fine)
            rows_.append(dict(fine=fine, coarse=coarse, ratio=b["ratio"],
                              theta=float(th), theta_lo=float(th_lo),
                              theta_hi=float(th_hi)))
        res["theta"][key] = dict(
            per_pair=rows_,
            widest=(rows_[-1] if rows_ else None),
            note=("lambda ~ dt^(-theta); theta = 0 is an identified rate, "
                  "theta = 1 is the sampling rate itself"))

    a = res["per_stage"].get("ALL", {})
    if "lam_calibrated" in a and "lam_onestep" in a:
        res["pooled"] = dict(
            calibrated_ratio=a["lam_calibrated"]["ratio"],
            calibrated_ci=[a["lam_calibrated"]["ratio_lo"], a["lam_calibrated"]["ratio_hi"]],
            onestep_ratio=a["lam_onestep"]["ratio"],
            onestep_ci=[a["lam_onestep"]["ratio_lo"], a["lam_onestep"]["ratio_hi"]],
            n_subjects=a["lam_calibrated"]["n"], n_cells=a["lam_calibrated"]["n_cells"])
    out = paths.out("figure_reproduction", "inputs", "window_scaling_exponent.json")
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(res, indent=1, default=float, ensure_ascii=False) + "\n")
    print(json.dumps(res["pooled"], indent=1))
    for g in STAGES:
        c = res["per_stage"].get(g, {})
        if "lam_calibrated" in c:
            print(f"{g:5s} calibrated {c['lam_calibrated']['ratio']:.3f} "
                  f"[{c['lam_calibrated']['ratio_lo']:.3f}, {c['lam_calibrated']['ratio_hi']:.3f}]"
                  f"   one-step {c['lam_onestep']['ratio']:.3f} "
                  f"[{c['lam_onestep']['ratio_lo']:.3f}, {c['lam_onestep']['ratio_hi']:.3f}]"
                  f"   n={c['lam_calibrated']['n']}")
    print("->", out)


if __name__ == "__main__":
    main()
