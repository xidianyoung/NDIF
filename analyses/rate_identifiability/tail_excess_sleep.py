# -*- coding: utf-8 -*-
r"""Tail excess of the state autocorrelation in the Sleep-EDF recordings, per sleep stage, and the non-REM depth
ladder on the uncorrected and the white-share-corrected one-step rate (observed points of the identifiability map,
sleep).

Under the single-timescale model the uncorrelated share of the autocorrelation is exactly nu and the true one-step
factor is a = r(k+1)/r(k); the corrected rate is -ln(a)/dt, the uncorrected one -ln(r(1))/dt.

Cohort and state space (lag_lib, the sleep window cache): 153 Sleep-EDF Cassette nights (prefix SC; the 44
Telemetry nights with --prefix ST), bouts of 60 consecutive 1-s windows, robust z + PCA (d = 4) fitted on each
recording's own pooled bouts.  1-s windows at a 1-s step, NON-OVERLAPPING; lag k is exactly k seconds and nothing
spans a bout boundary.  The unit is the participant, not the night: two Cassette nights per subject are averaged
within subject before any group test.  Per-recording draws use generators seeded with SEED + recording index.

Inputs   the sleep window cache (paths.cache("sleep_states"); NDIF_SLEEP_STATES_CACHE), see lag_lib.py
Outputs  analyses/rate_identifiability/out/tail_excess_sleep.json, tail_excess_sleep_rows.csv (read by
         identifiability_map.py: stage N2, columns lam_corrected, lam_uncorrected, nu), tail_excess_sleep_ladder.csv,
         tail_excess_sleep_profile.csv (file names carry _ST for the Telemetry prefix)
Reported Fig. 3a / Extended Data Fig. 1 (observed point of the sleep recordings)
Runtime  about 7 min on one desktop core
Usage    python analyses/rate_identifiability/tail_excess_sleep.py [--nrep 200] [--prefix SC]
"""
from __future__ import annotations

import argparse
import collections
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
import lag_lib as R                                                        # noqa: E402
import tail_excess_lib as A                                                # noqa: E402

DT = 1.0
WIN = 1.0
SEED = 20260922
MAXLAG = 12
TAIL = (3, 12)
STAGES = ("Wake", "N1", "N2", "N3", "N4")
RUNGS = (("Wake", "N1"), ("N1", "N2"), ("N2", "N3"), ("N3", "N4"))
N_CTRL = 20


def log(m):
    print(time.strftime("[%H:%M:%S] ") + str(m), flush=True)


def subject_of(stem):
    """Sleep-EDF Cassette: SC4ssN -> subject ss, night N.  Telemetry: ST7ssN likewise."""
    return stem[:5]


def prepare(stem, win=WIN, inject=None, rng=None):
    """{stage: (blocks, ...)} in the d = 4 state, plus the depth axis."""
    feat, meta = R.load_cell(stem, win)
    if feat is None:
        return None
    if inject is not None:
        feat = {g: np.stack([inject(b, np.zeros(len(b), int), rng) for b in v])
                for g, v in feat.items()}
    Z, evr = R.state_space(feat)
    blocks = {g: [np.asarray(b, float) for b in v] for g, v in Z.items()}
    deep = [g for g in ("N3", "N4") if g in Z]
    light = [g for g in ("Wake", "N1") if g in Z]
    ax = None
    if deep and light:
        d = np.mean([Z[g].reshape(-1, Z[g].shape[-1]).mean(0) for g in deep], axis=0)
        l = np.mean([Z[g].reshape(-1, Z[g].shape[-1]).mean(0) for g in light], axis=0)
        u = d - l
        if np.linalg.norm(u) > 1e-9:
            ax = u / np.linalg.norm(u)
    if ax is None:                      # fall back to the state's own leading direction
        allz = np.concatenate([Z[g].reshape(-1, Z[g].shape[-1]) for g in Z], 0)
        w, V = np.linalg.eigh(np.cov(allz.T))
        ax = V[:, -1] / np.linalg.norm(V[:, -1])
    return blocks, ax, evr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nrep", type=int, default=200)
    ap.add_argument("--stems", type=int, default=0)
    ap.add_argument("--prefix", default="SC")
    ap.add_argument("--skip-calib", action="store_true")
    a = ap.parse_args()
    t0 = time.time()

    stems = R.cache_stems(a.prefix, WIN)
    if a.stems:
        stems = stems[: a.stems]
    log(f"sleep ({a.prefix}): {len(stems)} recordings, "
        f"{len(set(map(subject_of, stems)))} participants, 1-s non-overlapping windows")

    RES = dict(cohort=f"Sleep-EDF {a.prefix}", n_records=len(stems),
               n_participants=len(set(map(subject_of, stems))), dt=DT, d=R.D_STATE,
               maxlag=MAXLAG, tail=list(TAIL), window_s=WIN, step_s=WIN, overlap="none",
               r1_floor=A.R1_FLOOR, generated=time.strftime("%Y-%m-%d %H:%M:%S"))
    rows, prof = [], []
    per = collections.defaultdict(dict)          # (subject, stage) -> unit
    ctrl = collections.defaultdict(dict)

    for si, stem in enumerate(stems):
        try:
            q = prepare(stem)
        except Exception as e:                                            # noqa: BLE001
            log(f"  {stem}: {e}")
            continue
        if q is None:
            continue
        blocks, ax, _ = q
        sub = subject_of(stem)
        rg = np.random.default_rng(SEED + si)
        for g in STAGES:
            B = blocks.get(g)
            if not B or len(B) < R.MIN_BOUTS:
                continue
            u = A.analyse_unit(B, ax, rg, a.nrep, MAXLAG, TAIL)
            if u is None:
                continue
            per[g].setdefault(sub, []).append(u)
            cv = [uu for v in A.orth_directions(ax, N_CTRL, rg)
                  for uu in [A.analyse_unit(B, v, None, 0, MAXLAG, TAIL)] if uu]
            if cv:
                ctrl[g].setdefault(sub, []).append(float(np.mean([c["E_tail"] for c in cv])))
            rows.append(dict(cohort=a.prefix, record=stem, subject=sub, stage=g,
                             n_bouts=u["n_blocks"], r1=u["r"][1], r2=u["r"][2], r3=u["r"][3],
                             exp_index=u["exp_index"], a=u["a"], nu=u["nu"],
                             E_tail=u["E_tail"], E_tail_corr=u["E_tail_corr"],
                             slope=u["slope"], parity=u["parity_amp"],
                             lam_uncorrected=-np.log(max(u["r"][1], 1e-6)) / DT,
                             lam_corrected=-np.log(max(u["a"], 1e-6)) / DT))
        if (si + 1) % 25 == 0:
            log(f"  {si+1}/{len(stems)} recordings")

    # ---- collapse to one value per participant per stage
    def sub_mean(g, key):
        return {s: float(np.mean([u[key] for u in v])) for s, v in per[g].items()}

    for g in STAGES:
        if g not in per or not per[g]:
            continue
        E = sub_mean(g, "E_tail")
        nulls = []
        for v in per[g].values():                    # average the nights within a participant
            ns = [u["null_E"] for u in v if u["null_E"]]
            if not ns:
                continue
            m = min(len(x) for x in ns)
            nulls.append(np.mean([x[:m] for x in ns], axis=0).tolist())
        gE = A.group_stat(list(E.values()), f"tail excess {g}", nulls)
        prf = [(np.array(v[0]["ks"]), np.nanmean(np.array([u["rho"] for u in v]), axis=0))
               for v in per[g].values()]
        gS = A.group_stat(list(sub_mean(g, "slope").values()), f"ratio slope {g}")
        lp = A.lagperm_null(prf, 1000, SEED)
        gS["p_lagperm"] = A.perm_p(lp, gS["mean"], "greater")
        mr = np.mean([np.mean([u["r"] for u in v], axis=0) for v in per[g].values()], axis=0)
        mp = np.mean([np.mean([u["pred"] for u in v], axis=0) for v in per[g].values()], axis=0)
        RES[g] = dict(n=len(E), tail_excess=gE, ratio_slope=gS,
                      exp_index=A.boot_ci(list(sub_mean(g, "exp_index").values())),
                      nu_hat=A.boot_ci(list(sub_mean(g, "nu").values())),
                      a_hat=A.boot_ci(list(sub_mean(g, "a").values())),
                      lam_uncorrected=A.boot_ci(
                          [-np.log(max(np.mean([u["r"][1] for u in v]), 1e-6)) / DT
                           for v in per[g].values()]),
                      lam_corrected=A.boot_ci([-np.log(max(np.mean([u["a"] for u in v]), 1e-6)) / DT
                                               for v in per[g].values()]),
                      control_E_tail=A.boot_ci([float(np.mean(v)) for v in ctrl[g].values()])
                      if g in ctrl and ctrl[g] else None,
                      mean_r=mr.tolist(), mean_pred=mp.tolist())
        for k in range(1, MAXLAG):
            prof.append(dict(cohort=a.prefix, stage=g, k=k, lag_s=k * DT, r_obs=mr[k],
                             r_single_timescale=mp[k], rho_obs=mr[k + 1] / mr[k] if mr[k] > 1e-3 else np.nan,
                             rho_single_timescale=mp[k + 1] / mp[k] if mp[k] > 1e-3 else np.nan))
        log(f"  {g}: n={len(E)} exp_index={RES[g]['exp_index']['mean']:.3f} "
            f"nu={RES[g]['nu_hat']['mean']:.3f} a={RES[g]['a_hat']['mean']:.3f} "
            f"E_tail={gE['mean']:+.4f} [{gE['lo']:+.4f},{gE['hi']:+.4f}] vs single timescale "
            f"{gE.get('null_mean', float('nan')):+.4f} P={gE.get('p_single_timescale_model', float('nan')):.4f}")

    # ---- the depth ladder on the uncorrected and the corrected rate
    ladder = []
    for lo, hi in RUNGS:
        if lo not in per or hi not in per:
            continue
        common = sorted(set(per[lo]) & set(per[hi]))
        if len(common) < 5:
            continue
        def val(g, s, key):
            return float(np.mean([u[key] for u in per[g][s]]))

        def r1(g, s):
            return float(np.mean([u["r"][1] for u in per[g][s]]))

        def lam(x):
            """lambda = -ln(factor)/dt; the factor is r(1) uncorrected and a = r(k+1)/r(k)
            corrected, the two differing by exactly the white share nu under the single-timescale model."""
            return max(-np.log(max(min(x, 0.999999), 1e-6)) / DT, 1e-9)

        dl_un = np.array([np.log(lam(r1(hi, s)) / lam(r1(lo, s))) for s in common])
        dl_co = np.array([np.log(lam(val(hi, s, "a")) / lam(val(lo, s, "a"))) for s in common])
        dnu = np.array([val(hi, s, "nu") - val(lo, s, "nu") for s in common])
        gu, gc, gn = A.boot_ci(dl_un), A.boot_ci(dl_co), A.boot_ci(dnu)
        ladder.append(dict(rung=f"{lo}->{hi}", n=len(common),
                           dlog_lam_uncorrected=gu["mean"], un_lo=gu["lo"], un_hi=gu["hi"],
                           p_un=A.signflip_p(dl_un), fold_un=float(np.exp(gu["mean"])),
                           dlog_lam_corrected=gc["mean"], co_lo=gc["lo"], co_hi=gc["hi"],
                           p_co=A.signflip_p(dl_co), fold_co=float(np.exp(gc["mean"])),
                           dnu=gn["mean"], dnu_lo=gn["lo"], dnu_hi=gn["hi"],
                           p_dnu=A.signflip_p(dnu)))
        log(f"  LADDER {lo}->{hi} (n={len(common)}): uncorrected rate x{np.exp(gu['mean']):.3f} "
            f"[{np.exp(gu['lo']):.3f},{np.exp(gu['hi']):.3f}] P={A.signflip_p(dl_un):.4g} | "
            f"CORRECTED x{np.exp(gc['mean']):.3f} [{np.exp(gc['lo']):.3f},{np.exp(gc['hi']):.3f}] "
            f"P={A.signflip_p(dl_co):.4g} | dnu {gn['mean']:+.3f} "
            f"[{gn['lo']:+.3f},{gn['hi']:+.3f}] P={A.signflip_p(dnu):.4g}")
    RES["ladder"] = ladder

    # ---- what nu is physically: does it depend on the direction?
    dirdep = {}
    for g in STAGES:
        if g in ctrl and ctrl[g] and g in per:
            common = sorted(set(per[g]) & set(ctrl[g]))
            if len(common) >= 5:
                d = np.array([float(np.mean([u["E_tail"] for u in per[g][s]]))
                              - float(np.mean(ctrl[g][s])) for s in common])
                dirdep[g] = dict(paired_difference=A.boot_ci(d), p=A.signflip_p(d), n=len(common))
    RES["direction_dependence"] = dirdep

    # ---- calibration: a known white component, and a known rate change
    calib = []
    if not a.skip_calib:
        sub_stems = stems[:40]
        for nu_inj in (0.10, 0.25):
            NU, E = [], []
            for si, stem in enumerate(sub_stems):
                rg = np.random.default_rng(SEED + 41 + si)
                try:
                    q = prepare(stem, inject=lambda F, s, r, n=nu_inj: A.inject_white(F, n, r),
                                rng=rg)
                except Exception:                                         # noqa: BLE001
                    continue
                if q is None:
                    continue
                B = q[0].get("N2")
                if not B:
                    continue
                u = A.analyse_unit(B, q[1], None, 0, MAXLAG, TAIL)
                if u:
                    NU.append(u["nu"]); E.append(u["E_tail"])
            if NU:
                calib.append(dict(injection="white", size=nu_inj, stage="N2",
                                  nu_hat=float(np.mean(NU)), E_tail=float(np.mean(E)),
                                  n=len(NU)))
                log(f"  CALIB white nu={nu_inj} into N2: nu_hat={np.mean(NU):.3f}, "
                    f"E_tail={np.mean(E):+.4f}")
        for wt in (0.05, 0.10):
            E = []
            for si, stem in enumerate(sub_stems):
                rg = np.random.default_rng(SEED + 51 + si)
                try:
                    q = prepare(stem, inject=lambda F, s, r, w_=wt: A.inject_slow(F, s, w_, 0.95, r),
                                rng=rg)
                except Exception:                                         # noqa: BLE001
                    continue
                if q is None:
                    continue
                B = q[0].get("N2")
                if not B:
                    continue
                u = A.analyse_unit(B, q[1], None, 0, MAXLAG, TAIL)
                if u:
                    E.append(u["E_tail"])
            if E:
                calib.append(dict(injection="slow", size=wt, stage="N2", nu_hat="",
                                  E_tail=float(np.mean(E)), n=len(E)))
                log(f"  CALIB second timescale weight={wt} into N2: E_tail={np.mean(E):+.4f}")
    RES["calibration"] = calib

    tag = "" if a.prefix == "SC" else f"_{a.prefix}"
    L.out(f"tail_excess_sleep{tag}.json").write_text(json.dumps(RES, indent=1, default=float),
                                                    encoding="utf-8")
    _csv(L.out(f"tail_excess_sleep{tag}_rows.csv"), rows)
    _csv(L.out(f"tail_excess_sleep{tag}_ladder.csv"), ladder)
    _csv(L.out(f"tail_excess_sleep{tag}_profile.csv"), prof)
    log(f"done in {time.time() - t0:.0f}s -> tail_excess_sleep{tag}.json")


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
