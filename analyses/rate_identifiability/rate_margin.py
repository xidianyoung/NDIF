# -*- coding: utf-8 -*-
"""Equivalence margin for the relaxation rate, and the end-to-end injections that calibrate it, in the COG-BCI
workload cohort (28 participants after the duplicate exclusion of cog_features.py).

(1) THE MARGIN: the design's own 80 %-power resolution, measured by simulation on the data's own geometry.  Each
    participant's LOWEST-LOAD cell supplies its own fitted (M, Q) and its own block lengths.  A gain arm raises
    every eigenvalue's MODULUS to the power g with its argument untouched (every mode's -ln|mu|/dt is multiplied
    by exactly g, Q untouched: lambda -> g lambda, D unchanged); a drive arm sets Q -> c Q (D -> c D, lambda
    unchanged).  Both arms go through the identical estimator, including the bias correction.  The margin is the
    fold change resolved at 80 % power against the matched no-change band.  No load contrast enters it.

(2) THE END-TO-END INJECTIONS INTO THE RECORDED DATA.  Temporal decimation of the window series by an integer
    stride s, n matched between the two arms by truncation: for any VAR(1) the estimand -(1/d) sum ln|eig M| / dt
    at stride s is exactly s times its value at stride 1, so the truth is ln(s).  And the static covariance
    reshaping control: the state is multiplied by a known invertible linear map T = I + (factor - 1) u u'
    (factor 2, random unit u per participant), which cannot change any eigenvalue of M (similarity), so the truth
    for the rate is exactly zero and the truth for D is the exactly computable change in tr(Q).  The recovered
    values are the estimator's false-positive size for a purely representational change.

State construction: the 248-feature log band-power state, participant-specific Flanker-referenced robust z and a
group PCA to d = 8 fitted on the load-blind Flanker task, non-overlapping 1-s windows, blocks matched across the
three load levels (build() below); estimators cog_fit_lib (analyses/workload_certificates/features).

Inputs   NDIF_COGBCI_CACHE (sub-XX.npz, analyses/workload_certificates/features/cog_cache.py)
Output   analyses/rate_identifiability/out/rate_margin_cog.json
Reported Supplementary Table 2, static covariance reshaping control: log-trace error -0.000646 [-0.002245, 0.001014]
         (cog.end_to_end_reshaping.dlog_D_minus_truth), rate shift 0.000919 [-0.012462, 0.015727]
         (cog.end_to_end_reshaping.recovered_dlog_lam)
Runtime  about 10.5 min on one desktop core
Usage    python analyses/rate_identifiability/rate_margin.py [--cohort cog] [--draws 260] [--boot 300]
"""
from __future__ import annotations

import argparse
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
import cog_fit_lib as R                                                    # noqa: E402

SEED = 20260922
DT = 1.0
GAIN_GRID = (1.02, 1.05, 1.10, 1.20, 1.40)
DRIVE_GRID = (1.02, 1.05, 1.10, 1.20)
LEVELS = [0, 1, 2]
D_PCA = 8

# --------------------------------------------------------------------------- state construction
def build(subs, cols=None, d=D_PCA):
    """Group PCA (load-blind: fitted on the Flanker reference task) then per-cell blocks."""
    from sklearn.decomposition import PCA
    rng = np.random.default_rng(SEED)
    X = []
    for s in subs.values():
        Z = s.z(s.ref_raw("dist"))
        if cols is not None:
            Z = Z[:, cols]
        if len(Z) > 1200:
            Z = Z[rng.choice(len(Z), 1200, replace=False)]
        X.append(Z)
    p = PCA(n_components=d, random_state=SEED).fit(np.concatenate(X, 0))
    log(f"  PCA d={d} on {sum(len(x) for x in X)} Flanker windows, "
        f"{100*p.explained_variance_ratio_.sum():.1f}% of variance")

    data = {}
    for sid, s in subs.items():
        cells = {}
        for fam, recs in FAMILIES.items():
            for si, ses in enumerate(SESSIONS, start=1):
                ok, per = True, {}
                for lev, rec in enumerate(recs):
                    Zd, sd = s.zrec(ses, rec, "dist")
                    Zy, sy = s.zrec(ses, rec, "dyn")
                    if Zd is None or Zy is None or len(Zd) < 60 or len(Zy) < 60:
                        ok = False
                        break
                    if cols is not None:
                        Zd, Zy = Zd[:, cols], Zy[:, cols]
                    Zd8 = p.transform(Zd)
                    Zy8 = p.transform(Zy)
                    blocks = [Zy8[sy == g] for g in np.unique(sy) if (sy == g).sum() >= 20]
                    per[lev] = dict(mu=Zd8.mean(axis=0), blocks=blocks, n_dist=len(Zd8))
                if ok and all(len(per[l]["blocks"]) for l in LEVELS):
                    mb = R.match_blocks([per[l]["blocks"] for l in LEVELS])
                    if mb is None:
                        continue
                    for l in LEVELS:
                        per[l]["blocks"] = mb[l]
                    cells[(fam, si)] = per
        if cells:
            data[sid] = cells
    return p, data



def mat_power_real(M, g):
    """A pure GAIN change: every eigenvalue's MODULUS is raised to the power g and its argument is
    left alone, so every mode's decay rate -ln|mu|/dt is multiplied by exactly g and no rotation is
    touched.  Built from the eigendecomposition rather than from logm(M), so it is defined for every
    real stable M, including the 80% of cells that carry a negative real eigenvalue and for which
    the principal logarithm is not real.  Conjugate pairs stay conjugate, so the result is real."""
    ev, V = np.linalg.eig(M)
    mod = np.abs(ev)
    ev_g = (mod ** g) * np.exp(1j * np.angle(ev))
    Mg = V @ np.diag(ev_g) @ np.linalg.inv(V)
    return np.real(Mg)


def chol_like(Q):
    w, V = np.linalg.eigh(0.5 * (Q + Q.T))
    return V @ np.diag(np.sqrt(np.maximum(w, 1e-18))) @ V.T


def draw_pair(geo, kind, g, rng, boot, burn=100):
    """One simulated participant: arm A from (M, Q), arm B from the manipulated pair."""
    M, Q, lens = geo
    if kind == "gain":
        Mb, Qb = mat_power_real(M, g), Q
    elif kind == "drive":
        Mb, Qb = M, g * Q
    else:
        Mb, Qb = M, Q
    La, Lb = chol_like(Q), chol_like(Qb)
    a = R._sim_blocks_batch(M, La, lens, 1, rng, burn=burn)
    b = R._sim_blocks_batch(Mb, Lb, lens, 1, rng, burn=burn)
    fa = R.fit_cell([x[0] for x in a], DT, n_boot=boot, rng=rng, burn=burn)
    fb = R.fit_cell([x[0] for x in b], DT, n_boot=boot, rng=rng, burn=burn)
    if fa is None or fb is None or not (fa["stable"] and fb["stable"]):
        return None
    return (float(np.log(fb["lam_mean"] / fa["lam_mean"])),
            float(np.log(fb["D_tr"] / fa["D_tr"])))


def ladder(geos, kind, grid, n_sub, n_cells, draws, boot, rng, band=None, burn=100):
    """Per-CELL draws, then the design's own aggregation: a participant is the mean of n_cells
    cells and the group is the mean of n_sub participants, exactly as the analysis aggregates."""
    out = {}
    for g in grid:
        dl, dd = [], []
        for i in range(draws):
            r = draw_pair(geos[i % len(geos)], kind, g, rng, boot, burn)
            if r is not None:
                dl.append(r[0]); dd.append(r[1])
        dl, dd = np.array(dl), np.array(dd)
        rec = dict(g=g, n_draws=len(dl),
                   cell_dlog_lam_mean=float(np.mean(dl)), cell_dlog_lam_sd=float(np.std(dl, ddof=1)),
                   cell_dlog_D_mean=float(np.mean(dd)), cell_dlog_D_sd=float(np.std(dd, ddof=1)),
                   dlog_lam_mean=float(np.mean(dl)), dlog_D_mean=float(np.mean(dd)),
                   truth_dlog_lam=float(np.log(g)) if kind == "gain" else 0.0,
                   truth_dlog_D=float(np.log(g)) if kind == "drive" else 0.0)
        rec["attenuation_lam"] = (rec["dlog_lam_mean"] / rec["truth_dlog_lam"]
                                  if rec["truth_dlog_lam"] else float("nan"))
        rec["attenuation_D"] = (rec["dlog_D_mean"] / rec["truth_dlog_D"]
                                if rec["truth_dlog_D"] else float("nan"))
        gm_l = np.mean(rng.choice(dl, (6000, n_sub * n_cells)), axis=1)
        gm_d = np.mean(rng.choice(dd, (6000, n_sub * n_cells)), axis=1)
        rec["group_sd_lam"] = float(np.std(gm_l, ddof=1))
        rec["group_sd_D"] = float(np.std(gm_d, ddof=1))
        if band is not None:
            rec["power_lam"] = float(np.mean((gm_l < band["lam"][0]) | (gm_l > band["lam"][1])))
            rec["power_D"] = float(np.mean((gm_d < band["D"][0]) | (gm_d > band["D"][1])))
        out[f"{g:g}"] = rec
        log(f"    {kind} g={g:g}: dlog lam {rec['dlog_lam_mean']:+.4f} (truth "
            f"{rec['truth_dlog_lam']:+.4f}), dlog D {rec['dlog_D_mean']:+.4f} (truth "
            f"{rec['truth_dlog_D']:+.4f})"
            + (f", power_lam {rec['power_lam']:.2f}, power_D {rec['power_D']:.2f}"
               if band is not None else ""))
    return out


def eighty(tab, key, truthkey):
    """Smallest fold change reaching 80% power, by linear interpolation in log g."""
    gs = sorted(float(k) for k in tab)
    p = [tab[f"{g:g}"][key] for g in gs]
    for i in range(len(gs)):
        if p[i] >= 0.8:
            if i == 0:
                return gs[0]
            x0, x1, y0, y1 = np.log(gs[i - 1]), np.log(gs[i]), p[i - 1], p[i]
            return float(np.exp(x0 + (0.8 - y0) * (x1 - x0) / (y1 - y0)))
    return float("nan")


# --------------------------------------------------------------------------- cohort geometries
def geos_cog(boot):
    subs = C.load_subjects()
    _, data = build(subs)
    geos, raws = [], []
    rng = np.random.default_rng(SEED)
    for sid in sorted(data):
        cells = data[sid]
        k = max(cells, key=lambda q: sum(len(b) for b in cells[q][0]["blocks"]))
        blocks = cells[k][0]["blocks"]
        f = R.fit_cell(blocks, DT, n_boot=boot, rng=rng)
        if f is None or not f["stable"]:
            continue
        geos.append((f["M"], f["Q"], [len(b) for b in blocks]))
        raws.append((sid, blocks))
    return geos, raws


# --------------------------------------------------------------------------- end-to-end injection
def decimation_injection(raws, strides, boot, rng, burn=100):
    """Known truth ln(s) injected into the REAL windows by temporal decimation, n matched."""
    out = {}
    for s in strides:
        dl, dd = [], []
        for _, blocks in raws:
            A, B = [], []
            for b in blocks:
                n = len(b) // s
                if n < 12:
                    continue
                A.append(np.asarray(b[:n], float))          # stride 1, truncated to n
                B.append(np.asarray(b[: n * s : s], float))  # stride s, same n
            if not A:
                continue
            fa = R.fit_cell(A, DT, n_boot=boot, rng=rng, burn=burn)
            fb = R.fit_cell(B, DT, n_boot=boot, rng=rng, burn=burn)
            if fa is None or fb is None or not (fa["stable"] and fb["stable"]):
                continue
            dl.append(np.log(fb["lam_mean"] / fa["lam_mean"]))
            dd.append(np.log(fb["D_tr"] / fa["D_tr"]))
        c = R.boot_ci(dl, seed=SEED)
        out[f"stride_{s}"] = dict(truth_dlog_lam=float(np.log(s)),
                                  recovered=c, fraction=c["mean"] / float(np.log(s)),
                                  dlog_D=R.boot_ci(dd, seed=SEED))
        log(f"    decimation s={s}: truth dlog lam {np.log(s):+.4f}, recovered "
            f"{c['mean']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}] -> fraction "
            f"{c['mean']/np.log(s):.3f}  (dlog D {out[f'stride_{s}']['dlog_D']['mean']:+.4f})")
    return out


def reshape_injection(raws, boot, rng, factor=2.0, burn=100):
    """A static invertible linear reshaping of the RAW state: truth for lambda is exactly zero
    (similarity), truth for D is the exactly computable change in tr(Q)."""
    dl, dd = [], []
    for _, blocks in raws:
        d = blocks[0].shape[1]
        u = rng.standard_normal(d); u /= np.linalg.norm(u)
        T = np.eye(d) + (factor - 1.0) * np.outer(u, u)
        fa = R.fit_cell([np.asarray(b, float) for b in blocks], DT, n_boot=boot, rng=rng, burn=burn)
        fb = R.fit_cell([np.asarray(b, float) @ T.T for b in blocks], DT, n_boot=boot, rng=rng,
                        burn=burn)
        if fa is None or fb is None or not (fa["stable"] and fb["stable"]):
            continue
        dl.append(np.log(fb["lam_mean"] / fa["lam_mean"]))
        truth = np.log(np.trace(T @ fa["Q"] @ T.T) / np.trace(fa["Q"]))
        dd.append(np.log(fb["D_tr"] / fa["D_tr"]) - truth)
    return dict(factor=factor, truth_dlog_lam=0.0, recovered_dlog_lam=R.boot_ci(dl, seed=SEED),
                dlog_D_minus_truth=R.boot_ci(dd, seed=SEED))


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="cog", choices=["cog"])
    ap.add_argument("--draws", type=int, default=260)
    ap.add_argument("--boot", type=int, default=300)
    ap.add_argument("--cells", type=int, default=0)
    ap.add_argument("--burn", type=int, default=100)
    a = ap.parse_args()
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    res = dict(seed=SEED, dt_s=DT,
               draws=a.draws, boot_in_ladder=a.boot,
               note=("The margin uses ONLY the lowest-load cell of each participant, so no load "
                     "contrast enters it."))
    cohorts = [a.cohort]
    for coh in cohorts:
        log(f"cohort {coh}")
        geos, raws = {"cog": geos_cog}[coh](a.boot)
        n_sub = len(geos)
        log(f"  {n_sub} participants with a usable lowest-load geometry")
        lam0 = [float(-np.mean(np.log(np.abs(np.linalg.eigvals(M)))) / DT) for M, _, _ in geos]
        D0 = [float(np.trace(Q) / (2 * Q.shape[0] * DT)) for _, Q, _ in geos]
        n_cells = a.cells or {"cog": 6}[coh]
        null = ladder(geos, "null", (1.0,), n_sub, n_cells, a.draws, a.boot, rng, None, a.burn)["1"]
        dlm = null["group_sd_lam"]
        dlD = null["group_sd_D"]
        band = dict(lam=[null["dlog_lam_mean"] - 1.96 * dlm, null["dlog_lam_mean"] + 1.96 * dlm],
                    D=[null["dlog_D_mean"] - 1.96 * dlD, null["dlog_D_mean"] + 1.96 * dlD])
        log(f"  no-change band (group mean, 95%): dlog lam {band['lam'][0]:+.4f}..{band['lam'][1]:+.4f}"
            f"  dlog D {band['D'][0]:+.4f}..{band['D'][1]:+.4f}")
        gl = ladder(geos, "gain", GAIN_GRID, n_sub, n_cells, a.draws, a.boot, rng, band, a.burn)
        dr = ladder(geos, "drive", DRIVE_GRID, n_sub, n_cells, a.draws, a.boot, rng, band, a.burn)
        g80 = eighty(gl, "power_lam", "truth_dlog_lam")
        c80 = eighty(dr, "power_D", "truth_dlog_D")
        inj = decimation_injection(raws, (2, 3), a.boot, rng, a.burn)
        rsh = reshape_injection(raws, a.boot, rng, burn=a.burn)
        res[coh] = dict(
            n_participants=n_sub, n_cells_per_participant=n_cells, lambda_level0=R.boot_ci(lam0, seed=SEED),
            D_level0=R.boot_ci(D0, seed=SEED), no_change=null, no_change_band=band,
            gain_ladder=gl, drive_ladder=dr,
            EQUIVALENCE_MARGIN_lambda_fold=g80,
            EQUIVALENCE_MARGIN_dlog_lambda=float(np.log(g80)) if np.isfinite(g80) else float("nan"),
            resolution_D_fold=c80,
            resolution_dlog_D=float(np.log(c80)) if np.isfinite(c80) else float("nan"),
            end_to_end_decimation=inj, end_to_end_reshaping=rsh)
        log(f"  MARGIN lambda: 80% power at fold {g80:.4f} -> |dlog lambda| = "
            f"{np.log(g80):.4f}   ({time.time()-t0:.0f}s)")
        log(f"  resolution  D: 80% power at fold {c80:.4f} -> |dlog D| = {np.log(c80):.4f}")
        L.out(f"rate_margin_{a.cohort}.json").write_text(json.dumps(res, indent=1, default=float))
    log(f"done in {time.time()-t0:.0f}s -> rate_margin_{a.cohort}.json")


if __name__ == "__main__":
    main()
