"""Two-regime generator parameters of every recording of the four awake cohorts (input of Fig. 1f).

Per record and per regime k in {0, 1} (the two task conditions), in the cohort's state space and with the
regime fit of the equilibrium count (landscape_lib.fit_linear_regime):

    mu_k      (d,)      regime mean of all windows of that regime
    Sigma_k   (d, d)    covariance of those windows (anisotropic, different between regimes)
    M_k       (d, d)    VAR(1) one-step map
    Q_k       (d, d)    innovation covariance implied by (M_k, Sigma_k), Sigma_k - M_k Sigma_k M_k',
                        projected onto the positive semidefinite cone
    seg_k     list      the segment lengths
    B_k       (d, d)    between-segment dispersion of the segment means in excess of what a stationary
                        VAR(1) with (M_k, Sigma_k) on those same segments predicts (non-stationarity)

and per record the Mahalanobis separation of the two regime means under the mean covariance (sep_real,
the observed separation of Fig. 1f) and the weight w1 of regime 0.

Cohorts
  hcp_fmri   7 fronto-parietal parcels, dt = 0.72 s, 0-back vs 2-back, one segment per task block
             (analyses/cohort_states/hcp_fmri_states.run_states: robust z per run, volume labels with a 5-s
             haemodynamic shift from the run's EV files, block identity per load).
  hcp_meg    d = 8, dt = 1.0 s, 0-back vs 2-back; each trial contributes one lag-1 pair (X, Y) of the
             per-trial state cache, so a regime has one segment of length 2 per trial.
  ds003838   d = 8, dt = 1.0 s, listen-only (control) vs memorise; reference state space fitted on the
             control windows of the 0.5-s-step series; dynamics on the 1-s-step series, runs of at least
             three windows that pass the amplitude screen.
  eegmmidb   d = 8, dt = 1.0 s, rest (T0) vs task (T1/T2) blocks of the 12 task runs
             (analyses/cohort_states/eegmmidb_states.subject_blocks); reference state space fitted on the
             rest windows of the 0.5-s-step series, one segment per block of the 1-s-step series.
  n_iso      the sample count per regime of the isotropic reference generator of the benchmark
             (floor_benchmark.py), per cohort.

Inputs   NDIF_HCP_FMRI_PARCELS (parcel series and EV files of every run, layout of
         analyses/cohort_states/hcp_fmri_states.py); the HCP MEG per-trial state cache (paths.cache("meg");
         NDIF_MEG_CACHE); the ds003838 feature cache (paths.cache("ds003838"); NDIF_DS003838_CACHE); the
         eegmmidb EDF files (NDIF_EEGMMIDB_RAW).
Outputs  paths.cache("separation_floors")/generators/<cohort>.pkl (list of per-record dicts) and
         <cohort>_summary.json (NDIF_SEPARATION_FLOORS_CACHE).
Supports Fig. 1f (observed separations; generators of the floor benchmark): 99 participants (198 runs)
         HCP fMRI, 83 HCP MEG, 64 ds003838, 106 eegmmidb.

Runtime a few minutes per cohort (eegmmidb reads 106 x 12 EDF files).

    python analyses/separation_floors/extract_generators.py --cohort hcp_fmri hcp_meg ds003838 eegmmidb --workers 8
"""

from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import argparse                                                       # noqa: E402
import glob                                                           # noqa: E402
import json                                                           # noqa: E402
import pickle                                                         # noqa: E402
import sys                                                            # noqa: E402
import time                                                           # noqa: E402
from concurrent.futures import ProcessPoolExecutor                    # noqa: E402
from pathlib import Path                                              # noqa: E402

import numpy as np                                                    # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from analyses.cohort_states import ds003838_states as DS              # noqa: E402
from analyses.cohort_states import eegmmidb_states as EE              # noqa: E402
from analyses.cohort_states import hcp_fmri_states as HF              # noqa: E402
from analyses.common import paths                                     # noqa: E402
from analyses.landscape_readouts import landscape_lib as LL           # noqa: E402

# sample count per regime of the isotropic reference generator, per cohort
N_ISO = dict(hcp_fmri=152, hcp_meg=296, ds003838=1458, eegmmidb=1260)
DT = dict(hcp_fmri=0.72, hcp_meg=1.0, ds003838=1.0, eegmmidb=1.0)
DIM = dict(hcp_fmri=7, hcp_meg=8, ds003838=8, eegmmidb=8)


def generator_dir():
    d = os.path.join(paths.cache("separation_floors"), "generators")
    os.makedirs(d, exist_ok=True)
    return d


def log(m):
    print(time.strftime("[%H:%M:%S] ") + str(m), flush=True)


# --------------------------------------------------------------------------- regime parameters
def _psd(S, floor=0.0):
    S = 0.5 * (np.asarray(S, float) + np.asarray(S, float).T)
    w, V = np.linalg.eigh(S)
    n_clip = int(np.sum(w < floor))
    w = np.maximum(w, floor)
    return V @ np.diag(w) @ V.T, n_clip


def _seg_mean_var(M, Sigma, L):
    """Var(mean of a length-L block of a stationary VAR(1) with one-step map M, stationary covariance Sigma)."""
    d = len(Sigma)
    tot = L * np.asarray(Sigma, float)
    Mk = np.eye(d)
    for k in range(1, L):
        Mk = Mk @ M
        C = Mk @ Sigma                      # Cov(x_{t+k}, x_t) = M^k Sigma
        tot = tot + (L - k) * (C + C.T)
    return tot / (L * L)


def regime_params(segs, label, dt):
    """Fit one regime (landscape_lib.fit_linear_regime) and add the generator quantities."""
    segs = [np.asarray(s, float) for s in segs if len(s) >= 2]
    if not segs:
        return None
    r = LL.fit_linear_regime(segs, label, dt=dt)
    d = len(r.mu)
    Q, n_clip = _psd(r.Sigma - r.M @ r.Sigma @ r.M.T)
    # measured between-segment mean dispersion, and what a stationary VAR(1) predicts for it
    Ls = np.array([len(s) for s in segs], float)
    w = Ls / Ls.sum()
    ms = np.array([s.mean(0) for s in segs])
    mbar = (w[:, None] * ms).sum(0)
    dm = ms - mbar
    B_obs = (w[:, None, None] * np.einsum("si,sj->sij", dm, dm)).sum(0)
    V = [_seg_mean_var(r.M, r.Sigma, int(L)) for L in Ls]
    sum_w2V = sum((wi * wi) * Vi for wi, Vi in zip(w, V))
    B_pred = sum(wi * (1.0 - 2.0 * wi) * Vi for wi, Vi in zip(w, V)) + sum_w2V
    B_exc, n_clip_B = _psd(B_obs - B_pred)
    return dict(label=label, mu=r.mu, Sigma=r.Sigma, M=r.M, Q=Q, Q_clipped=n_clip,
                n_windows=int(sum(Ls)), n_increments=int(r.n), n_segments=len(segs),
                seg_lens=[int(L) for L in Ls], logm_ok=bool(r.logm_ok),
                B_obs=B_obs, B_pred=B_pred, B_excess=B_exc, B_clipped=n_clip_B,
                rho=float(max(abs(np.linalg.eigvals(r.M)))), d=d)


def pack_record(cohort, rid, subject, segs0, segs1, extra=None):
    dt = DT[cohort]
    a = regime_params(segs0, "r0", dt)
    b = regime_params(segs1, "r1", dt)
    if a is None or b is None:
        return None
    Sbar = 0.5 * (a["Sigma"] + b["Sigma"])
    dmu = b["mu"] - a["mu"]
    sep = float(np.sqrt(dmu @ np.linalg.solve(Sbar, dmu)))
    rec = dict(cohort=cohort, record=rid, subject=subject, d=a["d"], dt=dt,
               n_iso=N_ISO[cohort], regimes=[a, b], sep_real=sep,
               w1_real=float(a["n_windows"] / (a["n_windows"] + b["n_windows"])))
    if extra:
        rec.update(extra)
    return rec


# --------------------------------------------------------------------------- cohort: HCP fMRI
def _fmri_one(subj):
    out = []
    for run in HF.RUNS:
        if not (HF.parcel_file(subj, run).exists() and HF.ev_dir(subj, run).is_dir()):
            continue
        Z, lab, bid0, bid1 = HF.run_states(subj, run)
        segs = []
        for ci, bid in ((0, bid0), (1, bid1)):
            s = [Z[(lab == ci) & (bid == i)] for i in range(bid.max() + 1)]
            segs.append([x for x in s if len(x) >= 2])
        rec = pack_record("hcp_fmri", f"{subj}_{run}", subj, segs[0], segs[1])
        if rec is not None:
            out.append(rec)
    return out


def cohort_hcp_fmri(workers):
    subs = sorted({s for s, _ in HF.available_runs()})
    log(f"hcp_fmri: {len(subs)} subjects")
    recs = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, out in enumerate(ex.map(_fmri_one, subs), 1):
            recs += out
            if i % 20 == 0:
                log(f"  {i}/{len(subs)} subjects, {len(recs)} runs")
    return recs


# --------------------------------------------------------------------------- cohort: HCP MEG
def cohort_hcp_meg(workers):
    files = sorted(glob.glob(os.path.join(paths.cache("meg"), "*.npz")))
    log(f"hcp_meg: {len(files)} cached participants")
    recs = []
    for f in files:
        z = np.load(f)
        segs = []
        for X, Y in ((z["X0"], z["Y0"]), (z["X1"], z["Y1"])):
            segs.append([np.vstack([X[i], Y[i]]) for i in range(len(X))])
        stem = os.path.basename(f)[:-4]
        rec = pack_record("hcp_meg", stem, stem, segs[0], segs[1])
        if rec is not None:
            recs.append(rec)
    return recs


# --------------------------------------------------------------------------- cohort: ds003838
def _ds_one(sub):
    S = DS.load_cache(sub)
    blocks = []
    for j in range(S["n"]):
        blocks.append(dict(cond=S["cond"][j], block=S["block"][j], role=S["role"][j],
                           F_dist=np.asarray(S["F_dist"][j], float),
                           F_dyn=np.asarray(S["F_dyn"][j], float),
                           keep=np.asarray(S["keep"][j], bool),
                           flag_y=np.asarray(S["flag_y"][j], bool)))
    mem = [b for b in blocks if b["cond"] == 1]
    ctl = [b for b in blocks if b["cond"] == 0]
    folds = sorted(set(b["block"] for b in blocks))
    if len(mem) < 20 or len(ctl) < 10 or len(folds) < 5:
        return None
    F_ref = np.vstack([b["F_dist"][b["keep"]] for b in ctl])       # control windows, 0.5-s step
    sp = LL.RefSpace(LL.D).fit(F_ref)
    for b in blocks:
        Zy = sp.transform(b["F_dyn"])
        b["Zy_runs"] = [Zy[r] for r in LL.split_runs(~b["flag_y"]) if len(r) >= 3]
    segs0 = [r for b in ctl for r in b["Zy_runs"]]
    segs1 = [r for b in mem for r in b["Zy_runs"]]
    return pack_record("ds003838", sub, sub, segs0, segs1)


def cohort_ds003838(workers):
    subs = sorted(os.path.basename(p)[:-4]
                  for p in glob.glob(os.path.join(paths.cache("ds003838"), "sub-*.npz")))
    log(f"ds003838: {len(subs)} cached participants")
    recs = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, r in enumerate(ex.map(_ds_one, subs), 1):
            if r is not None:
                recs.append(r)
            if i % 16 == 0:
                log(f"  {i}/{len(subs)}")
    return recs


# --------------------------------------------------------------------------- cohort: eegmmidb
def _eeg_one(name):
    sd = EE.DATA / name
    present = [r for r in EE.TASK_RUNS if (sd / f"{name}R{r:02d}.edf").exists()]
    if len(present) < 6:
        return None
    try:
        blocks = EE.subject_blocks(sd, present)
    except Exception as e:                       # unreadable recordings are skipped
        print(f"  {name}: skipped ({e!r})", flush=True)
        return None
    sel = [b for b in blocks if b["run"] in set(EE.TASK_RUNS)]
    lab = np.array([0 if b["desc"] == "T0" else 1 for b in sel])
    if len(sel) < 20 or lab.sum() < 10 or (lab == 0).sum() < 10:
        return None
    F_ref = np.vstack([b["F_dist"] for b, l in zip(sel, lab) if l == 0])
    sp = LL.RefSpace(LL.D).fit(F_ref)
    segs = ([sp.transform(b["F_dyn"]) for b, l in zip(sel, lab) if l == 0],
            [sp.transform(b["F_dyn"]) for b, l in zip(sel, lab) if l == 1])
    return pack_record("eegmmidb", name, name, segs[0], segs[1])


def cohort_eegmmidb(workers):
    subs = sorted(p.name for p in EE.DATA.iterdir() if p.is_dir() and p.name.startswith("S"))
    log(f"eegmmidb: {len(subs)} subject directories")
    recs = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, r in enumerate(ex.map(_eeg_one, subs), 1):
            if r is not None:
                recs.append(r)
            if i % 10 == 0:
                log(f"  {i}/{len(subs)}, kept {len(recs)}")
    return recs


COHORTS = dict(hcp_fmri=cohort_hcp_fmri, hcp_meg=cohort_hcp_meg,
               ds003838=cohort_ds003838, eegmmidb=cohort_eegmmidb)


def summarise(cohort, recs):
    def q(v):
        v = np.asarray(v, float)
        return [float(np.percentile(v, p)) for p in (5, 25, 50, 75, 95)]
    segs = [L for r in recs for k in (0, 1) for L in r["regimes"][k]["seg_lens"]]
    return dict(
        cohort=cohort, n_records=len(recs),
        n_subjects=len(set(r["subject"] for r in recs)),
        d=recs[0]["d"], dt=recs[0]["dt"], n_iso_per_regime=recs[0]["n_iso"],
        n_windows_per_regime_median=[float(np.median([r["regimes"][k]["n_windows"] for r in recs]))
                                     for k in (0, 1)],
        n_increments_per_regime_median=[float(np.median([r["regimes"][k]["n_increments"] for r in recs]))
                                        for k in (0, 1)],
        n_segments_per_regime_median=[float(np.median([r["regimes"][k]["n_segments"] for r in recs]))
                                      for k in (0, 1)],
        segment_length_q5_25_50_75_95=q(segs),
        spectral_radius_q=q([r["regimes"][k]["rho"] for r in recs for k in (0, 1)]),
        sep_real_q=q([r["sep_real"] for r in recs]),
        sep_real_median=float(np.median([r["sep_real"] for r in recs])),
        w1_real_median=float(np.median([r["w1_real"] for r in recs])),
        anisotropy_condition_number_q=q([float(np.linalg.cond(r["regimes"][k]["Sigma"]))
                                         for r in recs for k in (0, 1)]),
        nonstationarity_excess_trace_ratio_q=q(
            [float(np.trace(r["regimes"][k]["B_excess"]) /
                   max(np.trace(r["regimes"][k]["B_pred"]), 1e-30)) for r in recs for k in (0, 1)]),
        logm_ok_fraction=float(np.mean([r["regimes"][k]["logm_ok"] for r in recs for k in (0, 1)])),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cohort", nargs="+", default=list(COHORTS))
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    out = generator_dir()
    for c in args.cohort:
        t0 = time.time()
        recs = COHORTS[c](args.workers)
        if not recs:
            log(f"{c}: nothing extracted")
            continue
        with open(os.path.join(out, f"{c}.pkl"), "wb") as f:
            pickle.dump(recs, f, protocol=4)
        s = summarise(c, recs)
        s["elapsed_min"] = round((time.time() - t0) / 60.0, 2)
        with open(os.path.join(out, f"{c}_summary.json"), "w", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(s, indent=1))
        log(f"{c}: {len(recs)} records, {s['n_subjects']} subjects, "
            f"median segments/regime {s['n_segments_per_regime_median']}, "
            f"median windows/regime {s['n_windows_per_regime_median']}, "
            f"median separation {s['sep_real_median']:.4f}, {s['elapsed_min']:.1f} min")


if __name__ == "__main__":
    main()
