# -*- coding: utf-8 -*-
r"""Instrument-only white share in COG-BCI: is the white share nu of the analysed state sensor and spectral-estimation
noise, or a white dynamical component shared across the state?

Two mechanisms produce the same nu in a single series:
  * INSTRUMENT.  Sensor noise and the sampling noise of a short-time spectral estimate are INDEPENDENT between
    frequency bands and between channels, so each band carries its own white component;
  * SHARED.  A white component of the population state appears in every band that reads the state, so the white
    parts of two bands are CORRELATED at lag zero.
Two band-restricted series p and q, built from the SAME raw features through the SAME linear functional restricted
to one band each, with a shared correlated component of one-step factor a plus white parts, have
    cross(k) = s a^k  for k >= 1 ,    cross(0) = s + (shared white variance) ,
so the cross-band white share  nu_x = 1 - cross(1) / (a cross(0))  is ZERO when the white components are
band-specific and equals the within-band nu when they are entirely shared.  The same is done for two disjoint halves
of the montage.  The 248 raw features are exactly 62 channels x 4 bands in C order, so a band-restricted functional
is a literal subset of the feature vector.  Non-overlapping 1-s windows (the `dyn` series), MATB low load, blocks
split at every gap; the participants after the duplicate exclusion of cog_features.py.

Inputs   NDIF_COGBCI_CACHE
Outputs  analyses/rate_identifiability/out/band_white_share.json, band_white_share_rows.csv (read by
         identifiability_map.py: column nu_full, the within-series white share per participant)
Reported Fig. 3a / Extended Data Fig. 1 (instrument-only white share beside the observed points)
Runtime  about 1 min on one desktop core
Usage    python analyses/rate_identifiability/instrument_white_share.py
"""
from __future__ import annotations

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
import tail_excess_lib as A                                                # noqa: E402

SEED = 20260922
MAXLAG = 8
TAIL = (3, 8)
NCH, NB = 62, 4
BANDS = ("theta", "alpha", "beta", "gamma")


def log(m):
    print(time.strftime("[%H:%M:%S] ") + str(m), flush=True)


def cross_acf(Pa, Pb, maxlag):
    """Cross-correlation of two series over blocks, each block centred on its own mean.

    cross(k) = sum_t pa_t pb_{t+k}, normalised by sqrt(sum pa^2 sum pb^2) on the common base
    sample, so cross(0) and cross(k) are on the same footing.
    """
    num = np.zeros(maxlag + 1)
    da = db = 0.0
    for a, b in zip(Pa, Pb):
        if len(a) < maxlag + 2:
            continue
        x = a - a.mean()
        y = b - b.mean()
        m = len(x) - maxlag
        da += float(np.sum(x[:m] ** 2))
        db += float(np.sum(y[:m] ** 2))
        for k in range(maxlag + 1):
            num[k] += float(np.sum(x[:m] * y[k:m + k]))
    if da <= 0 or db <= 0:
        return None
    return num / np.sqrt(da * db)


def main():
    import cog_features as C
    import tail_excess_cog as G
    t0 = time.time()
    subs = C.load_subjects()
    pca = C.fit_group_pca(subs, G.D_PCA, seed=SEED)
    data, axes = G.build(subs, pca)
    log(f"COG-BCI: {len(data)} participants; nu attribution on the MATB low-load records")

    RES = dict(generated=time.strftime("%Y-%m-%d %H:%M:%S"), cohort="COG-BCI", maxlag=MAXLAG,
               note=("nu_x is the cross-band (or cross-montage) white share; it is 0 when the white "
                     "component is specific to each band or channel subset -- the instrument -- and "
                     "equals the within-series nu when it is entirely shared, which only a neural "
                     "white component can be."))
    rows = []
    per_band = {b: [] for b in BANDS}
    within, cross_band, cross_ch = [], [], []

    for sid in sorted(data):
        if ("matb", 0) not in data[sid] or sid not in axes:
            continue
        w = G.lopo(axes, sid)
        sub = subs[sid]
        c = (pca.components_.T @ w) / sub.iqr                  # p = F c + const, exactly
        # rebuild the raw per-record blocks with the same contiguity rule
        P_full, P_band, P_half = [], {b: [] for b in BANDS}, {0: [], 1: []}
        for ses in C.SESSIONS:
            k = f"{ses}|MATBeasy|dyn"
            if f"{k}|F" not in sub.d:
                continue
            F = sub.d[f"{k}|F"].astype(float)
            keep, seg = sub.d[f"{k}|keep"], sub.d[f"{k}|seg"]
            for g in np.unique(seg):
                idx = np.flatnonzero((seg == g) & keep)
                if len(idx) == 0:
                    continue
                for blk in A.split_contiguous(F[idx], idx, MAXLAG + 8):
                    P_full.append(blk @ c)
                    for bi, bn in enumerate(BANDS):
                        cb = np.zeros_like(c)
                        cb.reshape(NCH, NB)[:, bi] = c.reshape(NCH, NB)[:, bi]
                        P_band[bn].append(blk @ cb)
                    for hi in (0, 1):
                        ch = np.zeros_like(c)
                        sel = slice(0, NCH // 2) if hi == 0 else slice(NCH // 2, NCH)
                        ch.reshape(NCH, NB)[sel, :] = c.reshape(NCH, NB)[sel, :]
                        P_half[hi].append(blk @ ch)
        if len(P_full) < 2:
            continue
        uf = A.analyse_unit([p[:, None] for p in P_full], np.array([1.0]), None, 0, MAXLAG, TAIL)
        if uf is None:
            continue
        nu_w, a_w = uf["nu"], uf["a"]
        within.append(nu_w)
        row = dict(subject=sid, nu_full=nu_w, a_full=a_w, n_blocks=len(P_full))
        # ---- per-band nu
        for bn in BANDS:
            ub = A.analyse_unit([p[:, None] for p in P_band[bn]], np.array([1.0]), None, 0,
                                MAXLAG, TAIL)
            if ub:
                per_band[bn].append(ub["nu"])
                row["nu_" + bn] = ub["nu"]
                row["a_" + bn] = ub["a"]
        # ---- cross-band white share, averaged over the 6 band pairs
        xs = []
        for i in range(NB):
            for j in range(i + 1, NB):
                cx = cross_acf(P_band[BANDS[i]], P_band[BANDS[j]], MAXLAG)
                if cx is None or abs(cx[0]) < 0.02:
                    continue
                ai = A.analyse_unit([p[:, None] for p in P_band[BANDS[i]]], np.array([1.0]),
                                    None, 0, MAXLAG, TAIL)
                aj = A.analyse_unit([p[:, None] for p in P_band[BANDS[j]]], np.array([1.0]),
                                    None, 0, MAXLAG, TAIL)
                if ai is None or aj is None:
                    continue
                ash = float(np.sqrt(max(ai["a"], 1e-6) * max(aj["a"], 1e-6)))
                xs.append(1.0 - cx[1] / (ash * cx[0]))
        if xs:
            cross_band.append(float(np.mean(xs)))
            row["nu_cross_band"] = cross_band[-1]
        # ---- cross-montage white share
        cx = cross_acf(P_half[0], P_half[1], MAXLAG)
        if cx is not None and abs(cx[0]) > 0.02:
            a0 = A.analyse_unit([p[:, None] for p in P_half[0]], np.array([1.0]), None, 0,
                                MAXLAG, TAIL)
            a1 = A.analyse_unit([p[:, None] for p in P_half[1]], np.array([1.0]), None, 0,
                                MAXLAG, TAIL)
            if a0 and a1:
                ash = float(np.sqrt(max(a0["a"], 1e-6) * max(a1["a"], 1e-6)))
                cross_ch.append(1.0 - cx[1] / (ash * cx[0]))
                row["nu_cross_montage"] = cross_ch[-1]
        rows.append(row)

    RES["nu_within"] = A.boot_ci(within)
    RES["nu_cross_band"] = A.boot_ci(cross_band)
    RES["nu_cross_montage"] = A.boot_ci(cross_ch)
    RES["nu_per_band"] = {b: A.boot_ci(v) for b, v in per_band.items() if v}
    n = min(len(within), len(cross_band))
    if n >= 5:
        d = np.array(within[:n]) - np.array(cross_band[:n])
        RES["within_minus_cross_band"] = dict(**A.boot_ci(d), p=A.signflip_p(d))
    n2 = min(len(within), len(cross_ch))
    if n2 >= 5:
        d2 = np.array(within[:n2]) - np.array(cross_ch[:n2])
        RES["within_minus_cross_montage"] = dict(**A.boot_ci(d2), p=A.signflip_p(d2))
    # spread across bands against a matched ceiling (the participant-to-participant spread)
    if RES["nu_per_band"]:
        m = np.array([RES["nu_per_band"][b]["mean"] for b in RES["nu_per_band"]])
        RES["across_band_sd"] = float(np.std(m, ddof=1))
        RES["across_participant_sd"] = float(RES["nu_within"]["sd"])

    log(f"  nu within the full axis      {RES['nu_within']['mean']:.3f} "
        f"[{RES['nu_within']['lo']:.3f},{RES['nu_within']['hi']:.3f}] n={RES['nu_within']['n']}")
    for b in BANDS:
        if b in RES["nu_per_band"]:
            q = RES["nu_per_band"][b]
            log(f"  nu in the {b:6s} band only   {q['mean']:.3f} [{q['lo']:.3f},{q['hi']:.3f}]")
    log(f"  nu SHARED across bands       {RES['nu_cross_band']['mean']:.3f} "
        f"[{RES['nu_cross_band']['lo']:.3f},{RES['nu_cross_band']['hi']:.3f}]")
    log(f"  nu SHARED across the montage {RES['nu_cross_montage']['mean']:.3f} "
        f"[{RES['nu_cross_montage']['lo']:.3f},{RES['nu_cross_montage']['hi']:.3f}]")
    if "within_minus_cross_band" in RES:
        q = RES["within_minus_cross_band"]
        log(f"  within minus cross-band      {q['mean']:+.3f} [{q['lo']:+.3f},{q['hi']:+.3f}] "
            f"P={q['p']:.4g}   (a positive difference means the white component is BAND-SPECIFIC, "
            f"i.e. the instrument)")
    if "within_minus_cross_montage" in RES:
        q = RES["within_minus_cross_montage"]
        log(f"  within minus cross-montage   {q['mean']:+.3f} [{q['lo']:+.3f},{q['hi']:+.3f}] "
            f"P={q['p']:.4g}")

    L.out("band_white_share.json").write_text(json.dumps(RES, indent=1, default=float), encoding="utf-8")
    if rows:
        keys = sorted({k for r in rows for k in r})
        with open(L.out("band_white_share_rows.csv"), "w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, keys)
            wr.writeheader()
            for r in rows:
                wr.writerow(r)
    log(f"done in {time.time() - t0:.0f}s -> band_white_share.json")


if __name__ == "__main__":
    main()
