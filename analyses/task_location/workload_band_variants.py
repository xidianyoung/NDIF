# -*- coding: utf-8 -*-
"""Workload EEG (COG-BCI, 28 participants): the within-participant cross-validated cells recomputed with the
self-contained implementation of workload_axes_lib.py, each cross-validation half kept separately, and the same
cells on band-restricted features.

  S248_kss             cells on the 248-feature state with the within-session KSS label: n-back -> MATB and
                       MATB -> n-back transfer (LP_nb2mb, LP_mb2nb), in-sample references (LP_in_mb, LP_in_nb),
                       the eyes-open/eyes-closed halves of the sleepiness cells, the paired contrasts
  S8_kss               the same on the group PCA (d = 8)
  S248_clock           the same with a KSS-free label (-1 at the start, +1 at the end of a session), the ratio of
                       its sleepiness cell to the KSS one, the agreement between the two label axes and the KSS
                       change within sessions
  halfsplit_ceilings   matched within-family half-split ceilings of the group axes (500 splits)
  S248_noGamma         the S248_kss cells on 4-30 Hz features (theta, alpha, beta)
  S248_thetaAlpha      the S248_kss cells on 4-12 Hz features (theta, alpha)
Group summaries: participant means with percentile bootstrap intervals (10,000 resamples) and two-sided sign-flip P
values (10,000 draws), seed 777001.

Inputs:  the COG-BCI feature cache (NDIF_COGBCI_CACHE) and KSS.txt of the release (NDIF_COGBCI_RAW).
Outputs: analyses/task_location/out/workload_band_variants.json and workload_band_variant_rows.csv (one row per
         participant: the S248_kss cells); NDIF_OUT relocates them.
Reported in: Extended Data Fig. 4b,d and Supplementary Table 7b,d: within-person n-back -> MATB 0.401, MATB ->
         n-back 0.145, in-sample references 0.871 (MATB) and 0.669 (n-back) (keys S248_kss/LP_nb2mb, LP_mb2nb,
         LP_in_mb, LP_in_nb); own cross-task axis on 4-12 Hz features 0.096 (S248_thetaAlpha/LP_cv) and on 4-30 Hz
         features 0.143 (S248_noGamma/LP_cv).
Runtime: about 30 s on one desktop.

    python analyses/task_location/workload_band_variants.py
"""
from __future__ import annotations
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")       # one BLAS thread: the setting of the stored outputs
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import json, sys, time
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyses.common import paths                                # noqa: E402
import workload_axes_lib as L                                    # noqa: E402
from workload_axes_lib import summ, cos_u, gmean_axis, EYES, SEED  # noqa: E402


def cv_cells(S, key="Z", label="kss"):
    """Per-participant CV cells, and each CV half kept separately."""
    ids = sorted(S)
    R = {k: [] for k in ("SP_cv", "SS_cv", "LP_cv", "LS_cv", "SP_in", "LP_in",
                         "SP_EOtoEC", "SP_ECtoEO", "SS_EOtoEC", "SS_ECtoEO",
                         "LP_nb2mb", "LP_mb2nb", "LS_nb2mb", "LS_mb2nb",
                         "LP_in_nb", "LP_in_mb", "SP_in_EO", "SP_in_EC")}
    for i in ids:
        d = S[i]
        pos, spr, pin = [], [], []
        for est, app, tag in (("EO", "EC", "EOtoEC"), ("EC", "EO", "ECtoEO")):
            w = d.axis_rest(est, key, label)
            r = d.read_rest(w, app, key, label)
            ri = d.read_rest(w, est, key, label)
            if r:
                pos.append(r[0]); spr.append(r[1])
                R[f"SP_{tag}"].append(r[0]); R[f"SS_{tag}"].append(r[1])
            else:
                R[f"SP_{tag}"].append(np.nan); R[f"SS_{tag}"].append(np.nan)
            if ri:
                pin.append(ri[0]); R[f"SP_in_{est}"].append(ri[0])
            else:
                R[f"SP_in_{est}"].append(np.nan)
        R["SP_cv"].append(float(np.mean(pos)) if pos else np.nan)
        R["SS_cv"].append(float(np.mean(spr)) if spr else np.nan)
        R["SP_in"].append(float(np.mean(pin)) if pin else np.nan)

        pos, spr, pin = [], [], []
        for est, app, tag in (("nback", "matb", "nb2mb"), ("matb", "nback", "mb2nb")):
            w = d.axis_load(key, est)
            r = d.read_load(w, key, family=app)
            ri = d.read_load(w, key, family=est)
            if r:
                pos.append(r[0]); spr.append(r[1])
                R[f"LP_{tag}"].append(r[0]); R[f"LS_{tag}"].append(r[1])
            else:
                R[f"LP_{tag}"].append(np.nan); R[f"LS_{tag}"].append(np.nan)
            if ri:
                pin.append(ri[0]); R["LP_in_nb" if est=="nback" else "LP_in_mb"].append(ri[0])
            else:
                R["LP_in_nb" if est=="nback" else "LP_in_mb"].append(np.nan)
        R["LP_cv"].append(float(np.mean(pos)) if pos else np.nan)
        R["LS_cv"].append(float(np.mean(spr)) if spr else np.nan)
        R["LP_in"].append(float(np.mean(pin)) if pin else np.nan)
    return ids, {k: np.array(v, float) for k, v in R.items()}


def axis_reliability(S, key, label="kss"):
    ids = sorted(S)
    rs = [abs(cos_u(S[i].axis_rest("EO", key, label), S[i].axis_rest("EC", key, label))) for i in ids]
    rl = [abs(cos_u(S[i].axis_load(key, "nback"), S[i].axis_load(key, "matb"))) for i in ids]
    return np.array(rs, float), np.array(rl, float)


def block(S, key, label, tag, res, boot=10000):
    ids, R = cv_cells(S, key, label)
    I = R["LP_cv"] - R["LS_cv"] - (R["SP_cv"] - R["SS_cv"])
    out = {k: summ(v, boot) for k, v in R.items()}
    out["I_signed_cv"] = summ(I, boot)
    out["LP_minus_SP_cv"] = summ(R["LP_cv"] - R["SP_cv"], boot)
    out["LS_minus_SS_cv"] = summ(R["LS_cv"] - R["SS_cv"], boot)
    rs, rl = axis_reliability(S, key, label)
    out["axis_rel_sleep"] = summ(rs, boot)
    out["axis_rel_load"] = summ(rl, boot)
    out["transfer_ratio_sleep"] = out["SP_cv"]["mean"] / out["SP_in"]["mean"]
    out["transfer_ratio_load"] = out["LP_cv"]["mean"] / out["LP_in"]["mean"]
    res[tag] = out
    return ids, R, out


def main():
    t0 = time.time()
    res = {"seed": SEED}

    print("== building the 248-feature cohort ==")
    S, C, pca = L.build()
    res["pca_evr_total"] = float(pca.explained_variance_ratio_.sum())
    res["n_participants"] = len(S)

    ids, R, out = block(S, "Z", "kss", "S248_kss", res)
    print(f"  S248 KSS : LP_cv {out['LP_cv']['mean']:+.4f} [{out['LP_cv']['lo']:+.4f},{out['LP_cv']['hi']:+.4f}]"
          f"  SP_cv {out['SP_cv']['mean']:+.4f} [{out['SP_cv']['lo']:+.4f},{out['SP_cv']['hi']:+.4f}]"
          f"  LS_cv {out['LS_cv']['mean']:+.4f}  SS_cv {out['SS_cv']['mean']:+.4f}"
          f"  I {out['I_signed_cv']['mean']:+.4f} [{out['I_signed_cv']['lo']:+.4f},{out['I_signed_cv']['hi']:+.4f}] p={out['I_signed_cv']['p']:.3f}")
    print(f"  halves: LP nb->mb {out['LP_nb2mb']['mean']:+.4f} [{out['LP_nb2mb']['lo']:+.4f},{out['LP_nb2mb']['hi']:+.4f}] p={out['LP_nb2mb']['p']:.4f}"
          f" | mb->nb {out['LP_mb2nb']['mean']:+.4f} [{out['LP_mb2nb']['lo']:+.4f},{out['LP_mb2nb']['hi']:+.4f}] p={out['LP_mb2nb']['p']:.4f}")
    print(f"             SP EO->EC {out['SP_EOtoEC']['mean']:+.4f} [{out['SP_EOtoEC']['lo']:+.4f},{out['SP_EOtoEC']['hi']:+.4f}] p={out['SP_EOtoEC']['p']:.4f}"
          f" | EC->EO {out['SP_ECtoEO']['mean']:+.4f} [{out['SP_ECtoEO']['lo']:+.4f},{out['SP_ECtoEO']['hi']:+.4f}] p={out['SP_ECtoEO']['p']:.4f}")
    print(f"  paired LP_cv-SP_cv {out['LP_minus_SP_cv']['mean']:+.4f} "
          f"[{out['LP_minus_SP_cv']['lo']:+.4f},{out['LP_minus_SP_cv']['hi']:+.4f}] p={out['LP_minus_SP_cv']['p']:.3f}")

    block(S, "Z8", "kss", "S8_kss", res)
    print(f"  S8   KSS : LP_cv {res['S8_kss']['LP_cv']['mean']:+.4f}  SP_cv {res['S8_kss']['SP_cv']['mean']:+.4f}"
          f"  I {res['S8_kss']['I_signed_cv']['mean']:+.4f}")

    # ---------------- KSS-free clock label
    print("== KSS-free clock label ==")
    ids2, rows2, out2 = block(S, "Z", "clock", "S248_clock", res)
    print(f"  S248 CLOCK: LP_cv {out2['LP_cv']['mean']:+.4f}  SP_cv {out2['SP_cv']['mean']:+.4f} "
          f"[{out2['SP_cv']['lo']:+.4f},{out2['SP_cv']['hi']:+.4f}] p={out2['SP_cv']['p']:.4f}"
          f"  SS_cv {out2['SS_cv']['mean']:+.4f}  I {out2['I_signed_cv']['mean']:+.4f}")
    res["clock_reproduction_ratio_SP"] = out2["SP_cv"]["mean"] / out["SP_cv"]["mean"]

    # cos(sleep axis, clock axis) per participant and per eye, vs the sleep axis's own reliability
    a_cos, a_rel = [], []
    for i in ids:
        for eye in EYES:
            wk = S[i].axis_rest(eye, "Z", "kss")
            wc = S[i].axis_rest(eye, "Z", "clock")
            c = abs(cos_u(wk, wc))
            if np.isfinite(c):
                a_cos.append(c)
        r = abs(cos_u(S[i].axis_rest("EO", "Z", "kss"), S[i].axis_rest("EC", "Z", "kss")))
        if np.isfinite(r):
            a_rel.append(r)
    res["cos_kssaxis_clockaxis"] = summ(a_cos)
    res["sleepaxis_own_reliability"] = summ(a_rel)
    print(f"  |cos|(KSS axis, clock axis) = {res['cos_kssaxis_clockaxis']['mean']:.4f} "
          f"[{res['cos_kssaxis_clockaxis']['lo']:.4f},{res['cos_kssaxis_clockaxis']['hi']:.4f}]"
          f"   vs the sleep axis's OWN EO/EC reliability {res['sleepaxis_own_reliability']['mean']:.4f}")

    # how much KSS information is actually in the label
    kss = L.load_kss()
    dk, npos, nneg, nzero = [], 0, 0, 0
    for i in ids:
        for s in (1, 2, 3):
            k = kss.get((i, s))
            if k and "beginning" in k and "end" in k:
                v = k["end"] - k["beginning"]
                dk.append(v)
                npos += v > 0; nneg += v < 0; nzero += v == 0
    dk = np.array(dk, float)
    res["dKSS"] = dict(n=len(dk), mean=float(dk.mean()), sd=float(dk.std(ddof=1)),
                          n_positive=int(npos), n_negative=int(nneg), n_zero=int(nzero),
                          frac_nonpositive=float((nneg + nzero) / len(dk)))
    print(f"  dKSS(end-beg): n={len(dk)} mean {dk.mean():+.3f} sd {dk.std(ddof=1):.3f} "
          f"| +{npos} / -{nneg} / 0:{nzero}  -> {100*(nneg+nzero)/len(dk):.0f}% of sessions are not 'sleepier at the end'")

    # correlation between the KSS-within label and the clock label, per participant/eye
    cc = []
    for i in ids:
        for eye in EYES:
            k1 = S[i].m_kss_within(eye); k2 = S[i].m_clock(eye)
            if k1[0] is None or k2[0] is None:
                continue
            a, b = k1[1], k2[1]
            if np.std(a) == 0 or np.std(b) == 0:
                continue
            cc.append(float(np.corrcoef(a, b)[0, 1]))
    res["corr_label_kss_vs_clock"] = summ(cc)
    print(f"  corr(KSS-within label, clock label) = {res['corr_label_kss_vs_clock']['mean']:.3f} "
          f"[{res['corr_label_kss_vs_clock']['lo']:.3f},{res['corr_label_kss_vs_clock']['hi']:.3f}]")

    # ---------------- matched within-family half-split ceilings
    print("== matched within-family half-split ceilings ==")
    wl = {i: S[i].axis_load("Z") for i in ids}
    ws = {i: S[i].axis_sleep_both_eyes("Z", "kss") for i in ids}
    wc = {i: S[i].axis_sleep_both_eyes("Z", "clock") for i in ids}
    GL, GS, GC = gmean_axis(wl), gmean_axis(ws), gmean_axis(wc)
    rng = np.random.default_rng(SEED)
    ok_l = [i for i in ids if wl[i] is not None]
    ok_s = [i for i in ids if ws[i] is not None]
    hl, hs, hc = [], [], []
    for _ in range(500):
        p = rng.permutation(ok_l); hl.append(abs(cos_u(gmean_axis(wl, p[:len(p)//2]), gmean_axis(wl, p[len(p)//2:]))))
        q = rng.permutation(ok_s); hs.append(abs(cos_u(gmean_axis(ws, q[:len(q)//2]), gmean_axis(ws, q[len(q)//2:]))))
        hc.append(abs(cos_u(gmean_axis(wc, q[:len(q)//2]), gmean_axis(wc, q[len(q)//2:]))))
    def pc(v):
        v = np.array([x for x in v if np.isfinite(x)])
        return dict(mean=float(v.mean()), lo=float(np.percentile(v, 2.5)), hi=float(np.percentile(v, 97.5)))
    res["halfsplit_ceilings"] = dict(halfsplit_load=pc(hl), halfsplit_sleep=pc(hs), halfsplit_clock=pc(hc),
                     group_signed_cos_load_sleep=cos_u(GL, GS),
                     group_signed_cos_load_clock=cos_u(GL, GC),
                     group_signed_cos_sleep_clock=cos_u(GS, GC))
    print(f"  half-split load {res['halfsplit_ceilings']['halfsplit_load']['mean']:.3f}  sleep {res['halfsplit_ceilings']['halfsplit_sleep']['mean']:.3f}"
          f"  clock {res['halfsplit_ceilings']['halfsplit_clock']['mean']:.3f}")
    print(f"  group signed cos(load,sleep) {res['halfsplit_ceilings']['group_signed_cos_load_sleep']:+.3f}"
          f"  cos(load,clock) {res['halfsplit_ceilings']['group_signed_cos_load_clock']:+.3f}"
          f"  cos(sleep,clock) {res['halfsplit_ceilings']['group_signed_cos_sleep_clock']:+.3f}")

    # ---------------- band-restricted features
    for tag, keep in (("noGamma", ("theta", "alpha", "beta")), ("thetaAlpha", ("theta", "alpha"))):
        print(f"== band-restricted features: {tag} ==")
        Sx, Cx, px = L.build(band_keep=keep)
        _, _, o = block(Sx, "Z", "kss", f"S248_{tag}", res)
        print(f"  {tag}: LP_cv {o['LP_cv']['mean']:+.4f} [{o['LP_cv']['lo']:+.4f},{o['LP_cv']['hi']:+.4f}] p={o['LP_cv']['p']:.4f}"
              f"  SP_cv {o['SP_cv']['mean']:+.4f} [{o['SP_cv']['lo']:+.4f},{o['SP_cv']['hi']:+.4f}] p={o['SP_cv']['p']:.4f}"
              f"  I {o['I_signed_cv']['mean']:+.4f} [{o['I_signed_cv']['lo']:+.4f},{o['I_signed_cv']['hi']:+.4f}]")
        wlx = {i: Sx[i].axis_load("Z") for i in sorted(Sx)}
        wsx = {i: Sx[i].axis_sleep_both_eyes("Z", "kss") for i in sorted(Sx)}
        res[f"S248_{tag}"]["group_signed_cos"] = cos_u(gmean_axis(wlx), gmean_axis(wsx))
        del Sx, Cx

    res["elapsed_s"] = time.time() - t0
    Path(paths.out("analyses", "task_location", "out", "workload_band_variants.json")).write_text(
        json.dumps(res, indent=1, default=float), encoding="utf-8", newline="\n")

    import csv
    with open(paths.out("analyses", "task_location", "out", "workload_band_variant_rows.csv"), "w",
              newline="") as fh:
        w = csv.writer(fh)
        keys = sorted(R)
        w.writerow(["sid"] + keys)
        for n, i in enumerate(ids):
            w.writerow([i] + [R[k][n] for k in keys])
    print(f"DONE {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
