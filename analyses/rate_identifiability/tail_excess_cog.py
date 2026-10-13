# -*- coding: utf-8 -*-
r"""Tail excess of the state autocorrelation in COG-BCI: one relaxation timescale plus an additive white component
against several timescales (tail_excess_lib).

Cohort.  The COG-BCI participants after the duplicate exclusion of cog_features.py x 3 sessions x two task
families (n-back 0/1/2, MATB easy/med/diff), the 62-channel 248-feature log band-power state, robust z on each
participant's own Flanker reference and a group PCA to d = 8 fitted on the LOAD-BLIND Flanker task.

WINDOWS.  Every autocorrelation here is on the `dyn` series: 1-s windows at a 1-s step, i.e. NON-OVERLAPPING.  A
block is a maximal run of CONSECUTIVE window indices inside one record; every artefact drop starts a new block, so
lag k means exactly k windows and exactly k seconds.  The 50 %-overlap `dist` series is pushed through the
identical pipeline and reported as the counterfactual.

The two task families are analysed separately: the n-back family carries a large PERIOD-2 component (the task's
own stimulus train at 2 s against a 1-s window), MATB carries none; the test is repeated with the parity term
fitted as a third basis component.

CALIBRATION, end to end on the recordings, injected into the RAW 248-d log band powers before the robust z, the
PCA and any axis: a known additive white component, a known second timescale (a_slow = 0.95), and temporal
decimation by stride s (r_s(k) = r(sk)).

This module also provides the state construction (build, lopo, D_PCA, MAXLAG, TAIL) used by
tail_excess_contrast.py and tail_excess_calibration.py.

Inputs   NDIF_COGBCI_CACHE (sub-XX.npz)
Outputs  analyses/rate_identifiability/out/tail_excess_cog.json, tail_excess_cog_rows.csv,
         tail_excess_cog_profile.csv, tail_excess_cog_calibration.csv
Reported per-participant single-cohort verdicts (observed points of the identifiability map; Extended Data Fig. 7
         uses the contrast and calibration scripts)
Runtime  about 26 min on one desktop core
Usage    python analyses/rate_identifiability/tail_excess_cog.py [--nrep 200] [--perm 2000]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import injection_lib as L                                                  # noqa: E402
import cog_features as C                                                   # noqa: E402
from cog_features import SESSIONS, FAMILIES, log                           # noqa: E402
import tail_excess_lib as A                                                # noqa: E402

DT = 1.0
D_PCA = 8
SEED = 20260922
MAXLAG = 10
TAIL = (3, 10)
MIN_BLOCK = MAXLAG + 8
N_CTRL = 20


# --------------------------------------------------------------------------- data
def blocks_for(sub, pca, ses, rec, series="dyn", inject=None, rng=None, stride=1):
    """Contiguity-aware blocks of the d = 8 state for one record.

    `inject` acts on the RAW feature matrix, before the robust z and before the PCA, which is where
    an end-to-end injection must enter.
    """
    k = f"{ses}|{rec}|{series}"
    if f"{k}|F" not in sub.d:
        return []
    F = sub.d[f"{k}|F"].astype(float)
    keep = sub.d[f"{k}|keep"]
    seg = sub.d[f"{k}|seg"]
    if inject is not None:
        F = inject(F, seg, rng)
    Z = pca.transform(sub.z(F))
    out = []
    for g in np.unique(seg):
        idx = np.flatnonzero((seg == g) & keep)
        if len(idx) == 0:
            continue
        for b in A.split_contiguous(Z[idx], idx, MIN_BLOCK * stride):
            if stride > 1:
                out += [b[o::stride] for o in range(stride) if len(b[o::stride]) >= MIN_BLOCK]
            else:
                out.append(b)
    return out


def build(subs, pca, series="dyn", inject=None, seed=SEED, stride=1, with_axis=True):
    """{sid: {(family, level): [blocks]}} and, when asked, the per-participant load axis."""
    rng = np.random.default_rng(seed)
    data, axes = {}, {}
    for sid, s in subs.items():
        per, mus = {}, {}
        for fam, recs in FAMILIES.items():
            for lev, rec in enumerate(recs):
                B, M = [], []
                for ses in SESSIONS:
                    B += blocks_for(s, pca, ses, rec, series, inject, rng, stride)
                    if with_axis:
                        Fd = s.raw(ses, rec, "dist")[0]
                        if Fd is not None and len(Fd):
                            M.append(pca.transform(s.z(Fd)).mean(axis=0))
                if B:
                    per[(fam, lev)] = B
                if M:
                    mus[(fam, lev)] = np.mean(M, axis=0)
        if per:
            data[sid] = per
        v = [(mus[(f, 2)] - mus[(f, 0)]) / (np.linalg.norm(mus[(f, 2)] - mus[(f, 0)]) + 1e-12)
             for f in FAMILIES if (f, 2) in mus and (f, 0) in mus]
        if v:
            axes[sid] = np.mean(v, axis=0) / np.linalg.norm(np.mean(v, axis=0))
    return data, axes


def lopo(axes, sid):
    v = np.mean([a for s, a in axes.items() if s != sid], axis=0)
    return v / (np.linalg.norm(v) + 1e-12)


def sweep(data, axes, fam, lev, nrep, seed=SEED, parity=False, tail=TAIL, ceiling=True):
    """Run the per-unit test over every participant that has this family and level."""
    units, nulls, profiles, ceil = {}, [], [], []
    for sid in sorted(data):
        B = data[sid].get((fam, lev))
        if not B or sid not in axes:
            continue
        w = lopo(axes, sid)
        u = A.analyse_unit(B, w, np.random.default_rng(seed + sid), nrep, MAXLAG, tail, parity)
        if u is None:
            continue
        units[sid] = u
        nulls.append(u["null_E"])
        profiles.append((np.array(u["ks"]), np.array(u["rho"])))
        if ceiling:
            ceil.append(A.half_ceiling(B, w, np.random.default_rng(seed + 1000 + sid),
                                       MAXLAG, tail, parity))
    return units, nulls, profiles, ceil


def summarise(units, nulls, profiles, ceil, perm, seed=SEED):
    gE = A.group_stat([u["E_tail"] for u in units.values()], "tail excess", nulls, seed)
    gR = A.group_stat([u["E_tail_raw"] for u in units.values()], "tail excess (raw)",
                      [u["null_E_raw"] for u in units.values()], seed)
    gS = A.group_stat([u["slope"] for u in units.values()], "ratio slope",
                      [u["null_slope"] for u in units.values()], seed)
    lp = A.lagperm_null(profiles, perm, seed)
    gS["p_lagperm"] = A.perm_p(lp, gS["mean"], "greater")
    gS["lagperm_hi"] = float(np.percentile(lp, 97.5)) if len(lp) else np.nan
    return dict(n=len(units), tail_excess=gE, tail_excess_raw=gR, ratio_slope=gS,
                ceiling_halfdiff=A.boot_ci([c for c in ceil if np.isfinite(c)]),
                parity=A.boot_ci([u["parity_amp"] for u in units.values()]),
                exp_index=A.boot_ci([u["exp_index"] for u in units.values()]),
                a_hat=A.boot_ci([u["a"] for u in units.values()]),
                nu_hat=A.boot_ci([u["nu"] for u in units.values()]),
                mean_r=np.mean([u["r"] for u in units.values()], axis=0).tolist(),
                mean_pred=np.mean([u["pred"] for u in units.values()], axis=0).tolist())


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nrep", type=int, default=200)
    ap.add_argument("--perm", type=int, default=2000)
    ap.add_argument("--skip-calib", action="store_true")
    a = ap.parse_args()
    t0 = time.time()

    subs = C.load_subjects()
    pca = C.fit_group_pca(subs, D_PCA, seed=SEED)
    log(f"COG-BCI: {len(subs)} participants; PCA d={D_PCA} on the load-blind Flanker task")
    data, axes = build(subs, pca)
    log(f"  {len(data)} participants with usable non-overlapping dyn blocks, "
        f"{len(axes)} with a load axis")

    RES = dict(cohort="COG-BCI", n_participants=len(data), dt=DT, d=D_PCA, maxlag=MAXLAG,
               tail=list(TAIL), window_s=1.0, step_s=1.0, overlap="none",
               r1_floor=A.R1_FLOOR, generated=time.strftime("%Y-%m-%d %H:%M:%S"))
    rows, prof = [], []

    # ============================================== 1. the deciding test, per family and level
    for fam in FAMILIES:
        for lev in (0, 2):
            units, nulls, profiles, ceil = sweep(data, axes, fam, lev, a.nrep)
            if not units:
                continue
            key = f"{fam}_L{lev}"
            RES[key] = summarise(units, nulls, profiles, ceil, a.perm)
            RES[key]["n_admissible"] = len(units)
            RES[key]["n_attempted"] = sum(1 for s in data if (fam, lev) in data[s])
            for sid, u in units.items():
                rows.append(dict(cohort="COG-BCI", family=fam, level=lev, direction="load_axis",
                                 subject=sid, n_blocks=u["n_blocks"], n_win=u["n_win"],
                                 r1=u["r"][1], r2=u["r"][2], r3=u["r"][3], r5=u["r"][5],
                                 exp_index=u["exp_index"], parity=u["parity_amp"], a=u["a"],
                                 nu=u["nu"], E_tail=u["E_tail"], E_tail_raw=u["E_tail_raw"],
                                 E_tail_corr=u["E_tail_corr"], slope=u["slope"],
                                 slope2=u["slope2"]))
            mr = np.array(RES[key]["mean_r"]); mp = np.array(RES[key]["mean_pred"])
            for k in range(1, MAXLAG):
                prof.append(dict(cohort="COG-BCI", family=fam, level=lev, k=k, lag_s=k * DT,
                                 r_obs=mr[k], r_single_timescale=mp[k],
                                 rho_obs=mr[k + 1] / mr[k] if mr[k] > 1e-3 else np.nan,
                                 rho_single_timescale=mp[k + 1] / mp[k] if mp[k] > 1e-3 else np.nan,
                                 rho2_obs=(mr[k + 2] / mr[k] if k + 2 <= MAXLAG and mr[k] > 1e-3
                                           else np.nan)))
            g, c = RES[key]["tail_excess"], RES[key]["ceiling_halfdiff"]
            log(f"  {key}: n={len(units)}/{RES[key]['n_attempted']} "
                f"exp_index={RES[key]['exp_index']['mean']:.3f} "
                f"parity={RES[key]['parity']['mean']:+.3f} a={RES[key]['a_hat']['mean']:.3f} "
                f"nu={RES[key]['nu_hat']['mean']:.3f}")
            log(f"        E_tail={g['mean']:+.4f} [{g['lo']:+.4f},{g['hi']:+.4f}] vs single timescale "
                f"{g.get('null_mean', float('nan')):+.4f} "
                f"[{g.get('null_lo', float('nan')):+.4f},{g.get('null_hi', float('nan')):+.4f}] "
                f"P={g.get('p_single_timescale_model', float('nan')):.4f}; ceiling (split-half) "
                f"{c['mean']:+.4f} [{c['lo']:+.4f},{c['hi']:+.4f}]")

    # ============================================== 2. parity-corrected repeat
    for fam in FAMILIES:
        for lev in (0, 2):
            units, nulls, profiles, ceil = sweep(data, axes, fam, lev, a.nrep, parity=True,
                                                 tail=(4, TAIL[1]), ceiling=False)
            if units:
                RES[f"{fam}_L{lev}_parityfit"] = summarise(units, nulls, profiles, ceil, a.perm)
                g = RES[f"{fam}_L{lev}_parityfit"]["tail_excess"]
                log(f"  {fam} L{lev} parity fitted (moments r1,r2,r3; tail 4-{TAIL[1]}): "
                    f"E_tail={g['mean']:+.4f} [{g['lo']:+.4f},{g['hi']:+.4f}] "
                    f"P(single timescale)={g.get('p_single_timescale_model', float('nan')):.4f}")

    # ============================================== 3. control directions
    for fam in FAMILIES:
        Ea, Ec, nadm, natt = [], [], 0, 0
        for sid in sorted(data):
            B = data[sid].get((fam, 0))
            if not B or sid not in axes:
                continue
            r2 = np.random.default_rng(SEED + 500 + sid)
            w = lopo(axes, sid)
            u = A.analyse_unit(B, w, None, 0, MAXLAG, TAIL)
            if u is None:
                continue
            cv = []
            for v in A.orth_directions(w, N_CTRL, r2):
                natt += 1
                uu = A.analyse_unit(B, v, None, 0, MAXLAG, TAIL)
                if uu:
                    nadm += 1
                    cv.append(uu["E_tail"])
            if cv:
                Ea.append(u["E_tail"]); Ec.append(float(np.mean(cv)))
        if Ea:
            RES[f"{fam}_direction"] = dict(
                task_aligned=A.boot_ci(Ea), control_mean=A.boot_ci(Ec),
                paired_difference=A.boot_ci(np.array(Ea) - np.array(Ec)),
                n_control_directions=N_CTRL, control_admissible=nadm, control_attempted=natt)
            d = RES[f"{fam}_direction"]["paired_difference"]
            log(f"  {fam} L0 task-aligned {np.mean(Ea):+.4f} vs {nadm}/{natt} admissible "
                f"orthogonal controls {np.mean(Ec):+.4f}; paired {d['mean']:+.4f} "
                f"[{d['lo']:+.4f},{d['hi']:+.4f}]")

    # ============================================== 4. the overlapping-window counterfactual
    ov = {}
    d_ov, _ = build(subs, pca, series="dist", with_axis=False)
    for fam in FAMILIES:
        E, X, P = [], [], []
        for sid in sorted(d_ov):
            B = d_ov[sid].get((fam, 0))
            if not B or sid not in axes:
                continue
            u = A.analyse_unit(B, lopo(axes, sid), None, 0, MAXLAG, TAIL)
            if u:
                E.append(u["E_tail"]); X.append(u["exp_index"]); P.append(u["a"])
        if E:
            ov[fam] = dict(E_tail=A.boot_ci(E), exp_index=A.boot_ci(X), a_hat=A.boot_ci(P))
            log(f"  OVERLAP counterfactual {fam} L0 (0.5-s step, 50 % overlap): "
                f"E_tail={ov[fam]['E_tail']['mean']:+.4f} "
                f"(non-overlapping {RES[f'{fam}_L0']['tail_excess']['mean']:+.4f}), "
                f"exp_index={ov[fam]['exp_index']['mean']:.3f}")
    RES["overlap_counterfactual"] = ov

    # ============================================== 5. end-to-end calibration
    calib = []
    if not a.skip_calib:
        for nu_inj in (0.10, 0.25, 0.50):
            dW, _ = build(subs, pca, inject=lambda F, s, r, n=nu_inj: A.inject_white(F, n, r),
                          seed=SEED + 11, with_axis=False)
            E = [u["E_tail"] for sid in sorted(dW) if (("matb", 0) in dW[sid] and sid in axes)
                 for u in [A.analyse_unit(dW[sid][("matb", 0)], lopo(axes, sid), None, 0,
                                          MAXLAG, TAIL)] if u]
            NU = [u["nu"] for sid in sorted(dW) if (("matb", 0) in dW[sid] and sid in axes)
                  for u in [A.analyse_unit(dW[sid][("matb", 0)], lopo(axes, sid), None, 0,
                                           MAXLAG, TAIL)] if u]
            if E:
                g = A.boot_ci(E)
                calib.append(dict(cohort="COG-BCI", injection="white", size=nu_inj, a_slow="",
                                  E_tail=g["mean"], lo=g["lo"], hi=g["hi"], n=g["n"],
                                  nu_hat=float(np.mean(NU))))
                log(f"  CALIB white nu={nu_inj}: E_tail={g['mean']:+.4f} "
                    f"[{g['lo']:+.4f},{g['hi']:+.4f}] nu_hat={np.mean(NU):.3f}")
        for wt in (0.02, 0.05, 0.10, 0.20):
            dS, _ = build(subs, pca,
                          inject=lambda F, s, r, w_=wt: A.inject_slow(F, s, w_, 0.95, r),
                          seed=SEED + 12, with_axis=False)
            E = [u["E_tail"] for sid in sorted(dS) if (("matb", 0) in dS[sid] and sid in axes)
                 for u in [A.analyse_unit(dS[sid][("matb", 0)], lopo(axes, sid), None, 0,
                                          MAXLAG, TAIL)] if u]
            if E:
                g = A.boot_ci(E)
                p = A.signflip_p(np.array(E) - RES["matb_L0"]["tail_excess"]["mean"])
                calib.append(dict(cohort="COG-BCI", injection="slow", size=wt, a_slow=0.95,
                                  E_tail=g["mean"], lo=g["lo"], hi=g["hi"], n=g["n"],
                                  nu_hat=""))
                log(f"  CALIB second timescale weight={wt} a_slow=0.95: E_tail={g['mean']:+.4f} "
                    f"[{g['lo']:+.4f},{g['hi']:+.4f}]")
    RES["calibration"] = calib

    # ============================================== 6. decimation ground truth on the lag axis
    dec = {}
    for stride in (2, 3):
        dD, _ = build(subs, pca, stride=stride, with_axis=False)
        E = [u["E_tail"] for sid in sorted(dD) if (("matb", 0) in dD[sid] and sid in axes)
             for u in [A.analyse_unit(dD[sid][("matb", 0)], lopo(axes, sid), None, 0,
                                      MAXLAG, TAIL)] if u]
        AH = [u["a"] for sid in sorted(dD) if (("matb", 0) in dD[sid] and sid in axes)
              for u in [A.analyse_unit(dD[sid][("matb", 0)], lopo(axes, sid), None, 0,
                                       MAXLAG, TAIL)] if u]
        if E:
            dec[str(stride)] = dict(E_tail=A.boot_ci(E), a_hat=A.boot_ci(AH))
            log(f"  DECIMATION stride {stride} (exact time rescaling, truth a -> a^{stride}): "
                f"E_tail={dec[str(stride)]['E_tail']['mean']:+.4f}, "
                f"a_hat={dec[str(stride)]['a_hat']['mean']:.3f} "
                f"(undecimated {RES['matb_L0']['a_hat']['mean']:.3f})")
    RES["decimation"] = dec

    L.out("tail_excess_cog.json").write_text(json.dumps(RES, indent=1, default=float), encoding="utf-8")
    _csv(L.out("tail_excess_cog_rows.csv"), rows)
    _csv(L.out("tail_excess_cog_profile.csv"), prof)
    _csv(L.out("tail_excess_cog_calibration.csv"), calib)
    log(f"done in {time.time() - t0:.0f}s -> tail_excess_cog.json")


def _csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, list(rows[0].keys()))
        wr.writeheader()
        for r in rows:
            wr.writerow(r)


if __name__ == "__main__":
    main()
