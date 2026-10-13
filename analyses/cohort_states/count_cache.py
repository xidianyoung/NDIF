# -*- coding: utf-8 -*-
"""Count cache: the pooled state series of every recording of the equilibrium-count analysis (646 recordings).

For each recording the cache holds the state series in its cohort's state space, its segment structure and the
condition label of every sample, so that the equilibrium count can be recomputed with any assignment of samples
to regimes (analyses/landscape_readouts) while records, state space and counter stay fixed.

Cohorts (record = participant, run or night)
  ds003838  64 participants  memorize / listen-only; states from the feature cache of ds003838_states.py: RefSpace
                             (d = 8) fitted on the unflagged dist windows of the listen-only sequences; dyn series
                             split at flagged windows, runs of >= 3 windows are segments (dt = 1 s)
  eegmmidb  104 participants rest / task; eegmmidb_states.py; every block with >= 2 dyn windows is a segment
  meg       83 participants  0-back / 2-back; the per-trial cache of hcp_meg_states.py; each trial's two dyn
                             states are one segment (dt = 1 s)
  fmri      198 runs         0-back / 2-back, 99 participants x 2 runs; hcp_fmri_states.py; each block with >= 2
                             volumes is a segment (dt = 0.72 s)
  sleep     197 nights       NREM / REM / Wake, 153 cassette + 44 telemetry; sleep_states.py; the states of each
                             stage group cut into consecutive 2-sample segments (dt = 1 s)
449 awake recordings (ds003838, eegmmidb, MEG, fMRI) and 197 sleep nights.

Membership.  ds003838, MEG and fMRI: every recording in the cohort's state input.  eegmmidb and sleep: the records
of reference_pipeline_counts.csv.  That table lists, per record, the equilibrium count of the per-participant
reference landscape pipeline (one VAR(1) regime per condition on the condition labels, equilibria of the
mixture-weighted drift field; for sleep also the ridgeline mode count of the three stage Gaussians); it is
copied into the cache metadata (pub_n_stable, pub_n_eq, pub_modes_K) and fixes two cohorts: eegmmidb has 104
of 109 participants (S088, S092 and S100 are 128-Hz recordings whose annotated blocks cannot be read at 160 Hz;
the per-participant pipeline stopped for S035 and S063, at the singular covariance of their artifact proxies) and
sleep has all 197 nights.  For the fMRI runs the reference count is recomputed from the cached labels
(hcp_fmri_states.reference_count) and compared with the table (label_check_ok; label_reconstruction_check).

Inputs: the state inputs of the five cohorts (see the cohort modules; environment variables NDIF_DS003838_CACHE,
NDIF_EEGMMIDB_RAW, NDIF_MEG_CACHE, NDIF_HCP_FMRI_PARCELS, NDIF_SLEEPEDF_RAW).
Output (paths.cache("count"), environment variable NDIF_COUNT_CACHE, default analyses/_work/count), per cohort
directory {ds003838, eegmmidb, meg, fmri, sleep}:
  <record>.npz   Z (N, d) float32 pooled dyn state series, segments concatenated in order; seg (N,) int32 segment
                 index; lab (N,) int16 condition label (index into regimes); ds003838, eegmmidb and sleep also
                 Zdist (M, d) float32 dist series, labdist (M,) int16 and (ds003838, eegmmidb) segdist (M,) int32
  _meta.json     dt, d, K, regimes, unit and one entry per record (subject, n, n_seg and the reference counts)
Supports: the 646-recording composition (Methods, Human recordings; Supplementary Fig. 2a) and the inputs of the
drift-replacement, separation-threshold and readout analyses (Fig. 2b,c,f).
Runtime: about 25 min for all cohorts with 6 workers on one desktop (sleep 9 min, eegmmidb 2 min, fMRI 3 min),
after the ds003838 and MEG caches exist.  Threads: two per process, as in the stored run.

    python analyses/cohort_states/count_cache.py --cohort all --workers 6
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import argparse
import csv
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))
from analyses.common import paths                                  # noqa: E402

CACHE = Path(paths.cache("count"))
REFERENCE = HERE / "reference_pipeline_counts.csv"
COHORTS = ["fmri", "meg", "ds003838", "eegmmidb", "sleep"]


def log(m):
    print(time.strftime("[%H:%M:%S] ") + str(m), flush=True)


def reference_counts(cohort):
    """record -> dict(subject, n_stable, n_equilibria, ridgeline_modes_K, labels) of reference_pipeline_counts.csv."""
    def num(v):
        return int(v) if v != "" else None
    out = {}
    with open(REFERENCE, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["cohort"] != cohort:
                continue
            out[r["record"]] = dict(subject=r["subject"], n_stable=num(r["n_stable"]),
                                    n_equilibria=num(r["n_equilibria"]),
                                    ridgeline_modes_K=num(r["ridgeline_modes_K"]),
                                    labels=(r["labels"].split("|") if r["labels"] else None))
    return out


def _save(cohort, rec, Z, seg, lab, Zdist=None, labdist=None, segdist=None):
    d = CACHE / cohort
    d.mkdir(parents=True, exist_ok=True)
    kw = dict(Z=np.asarray(Z, np.float32), seg=np.asarray(seg, np.int32),
              lab=np.asarray(lab, np.int16))
    if Zdist is not None:
        kw["Zdist"] = np.asarray(Zdist, np.float32)
        kw["labdist"] = np.asarray(labdist, np.int16)
        if segdist is not None:
            kw["segdist"] = np.asarray(segdist, np.int32)
    np.savez_compressed(d / f"{rec}.npz", **kw)


def _keep(name, records):
    return records is None or name in records


# =============================================================================== fMRI (HCP WM)
def build_fmri(records=None):
    import hcp_fmri_states as WM
    pub_map = reference_counts("fmri")
    meta = dict(cohort="fmri", dt=WM.TR, d=None, K=2, regimes=list(WM.CONDS),
                unit="run (99 subjects x 2 runs)", records={})
    agree = tot = 0
    for subj, run in WM.available_runs():
        rec = f"{subj}__{run}"
        if not _keep(rec, records):
            continue
        F, lab, bid0, bid1 = WM.run_states(subj, run)
        Zs, segs, labs, s = [], [], [], 0
        for ci, bid in ((0, bid0), (1, bid1)):
            for i in range(bid.max() + 1):
                m = (lab == ci) & (bid == i)
                if m.sum() >= 2:
                    Zs.append(F[m]); segs.append(np.full(int(m.sum()), s)); labs.append(np.full(int(m.sum()), ci))
                    s += 1
        if not Zs:
            continue
        Z = np.vstack(Zs); seg = np.concatenate(segs); lb = np.concatenate(labs)
        _save("fmri", rec, Z, seg, lb)
        # the reference count recomputed from the cached labels, against reference_pipeline_counts.csv
        chk = WM.reference_count(F, lab, bid0, bid1)
        p = pub_map.get(rec, {})
        ok = (chk.get("n_stable") == p.get("n_stable") and chk.get("n_equilibria") == p.get("n_equilibria"))
        agree += int(ok); tot += 1
        meta["records"][rec] = dict(subject=subj, run=run, n=int(len(Z)), n_seg=int(seg.max() + 1),
                                    pub_n_stable=p.get("n_stable"), pub_n_eq=p.get("n_equilibria"),
                                    recomputed_n_stable=chk.get("n_stable"),
                                    recomputed_n_eq=chk.get("n_equilibria"), label_check_ok=bool(ok))
        meta["d"] = int(Z.shape[1])
        if tot % 25 == 0:
            log(f"  fmri {tot} runs, label check {agree}/{tot}")
    meta["label_reconstruction_check"] = dict(n_runs=tot, n_reproducing_published_count=agree)
    (CACHE / "fmri" / "_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    log(f"fmri: {tot} runs cached; recomputed reference count equals the table in {agree}/{tot}")


# =============================================================================== HCP MEG
def build_meg(records=None):
    import hcp_meg_states as MG
    src = Path(paths.cache("meg"))
    pub = reference_counts("meg")
    meta = dict(cohort="meg", dt=float(MG.STEP_DYN), d=int(MG.D), K=2, regimes=["0bk", "2bk"],
                unit="subject (83)", records={},
                note=("segments are the 2-window trial segments (the two dyn states of each trial) of the "
                      "per-trial HCP MEG cache written by hcp_meg_states.py"))
    for f in sorted(src.glob("*.npz")):
        subj = f.stem
        if not _keep(subj, records):
            continue
        z = np.load(f)
        Zs, segs, labs, s = [], [], [], 0
        for ld in (0, 1):
            X, Y = z[f"X{ld}"], z[f"Y{ld}"]
            for i in range(len(X)):
                Zs.append(np.vstack([X[i], Y[i]])); segs.append(np.full(2, s)); labs.append(np.full(2, ld))
                s += 1
        Z = np.vstack(Zs); seg = np.concatenate(segs); lb = np.concatenate(labs)
        _save("meg", subj, Z, seg, lb)
        p = pub.get(subj, {})
        meta["records"][subj] = dict(subject=subj, n=int(len(Z)), n_seg=int(seg.max() + 1),
                                     pub_n_stable=p.get("n_stable"), pub_n_eq=p.get("n_equilibria"))
    (CACHE / "meg" / "_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    log(f"meg: {len(meta['records'])} subjects cached")


# =============================================================================== ds003838
def build_ds(records=None):
    import ds003838_states as DS
    from states_lib import RefSpace, split_runs

    pub = reference_counts("ds003838")
    subs = sorted(p.stem for p in DS.CACHE.glob("sub-*.npz"))
    meta = dict(cohort="ds003838", dt=float(DS.STEP_DYN), d=int(DS.D), K=2,
                regimes=["control", "memory"], unit="subject (64)", records={})
    for sub in subs:
        if not _keep(sub, records):
            continue
        S = DS.load_cache(sub)
        F_ref = np.vstack([S["F_dist"][i][S["keep"][i]] for i in range(S["n"]) if S["cond"][i] == 0])
        sp = RefSpace(DS.D).fit(F_ref)
        Zs, segs, labs, s = [], [], [], 0
        Zd_all, lab_d, seg_d, sd = [], [], [], 0
        for i in range(S["n"]):
            Zy = sp.transform(S["F_dyn"][i])
            ci = int(S["cond"][i])
            for r in split_runs(~S["flag_y"][i]):
                if len(r) >= 3:
                    Zs.append(Zy[r]); segs.append(np.full(len(r), s)); labs.append(np.full(len(r), ci)); s += 1
            # the dist series is segmented at flagged windows and sequence boundaries, so increments
            # never straddle a rejected window or a sequence boundary
            Zdi = sp.transform(S["F_dist"][i])
            for r in split_runs(S["keep"][i]):
                Zd_all.append(Zdi[r]); lab_d.append(np.full(len(r), ci))
                seg_d.append(np.full(len(r), sd)); sd += 1
        Z = np.vstack(Zs); seg = np.concatenate(segs); lb = np.concatenate(labs)
        _save("ds003838", sub, Z, seg, lb, np.vstack(Zd_all), np.concatenate(lab_d),
              np.concatenate(seg_d))
        p = pub.get(sub, {})
        meta["records"][sub] = dict(subject=sub, n=int(len(Z)), n_seg=int(seg.max() + 1),
                                    pub_n_stable=p.get("n_stable"), pub_n_eq=p.get("n_equilibria"))
        log(f"  ds003838 {sub}: {len(Z)} dyn samples, {seg.max()+1} segments")
    (CACHE / "ds003838" / "_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    log(f"ds003838: {len(meta['records'])} subjects cached")


# =============================================================================== eegmmidb
def _ee_one(sub):
    import eegmmidb_states as EE
    st = EE.subject_states(sub)
    if st is None:
        return sub, "too few blocks", None
    sel, lab, sp = st
    Zs, segs, labs, s = [], [], [], 0
    Zd_all, lab_d, seg_d, sd = [], [], [], 0
    for b, l in zip(sel, lab):
        Zy = sp.transform(b["F_dyn"])
        if len(Zy) >= 2:
            Zs.append(Zy); segs.append(np.full(len(Zy), s)); labs.append(np.full(len(Zy), int(l))); s += 1
        Zd = sp.transform(b["F_dist"])
        Zd_all.append(Zd); lab_d.append(np.full(len(Zd), int(l)))
        seg_d.append(np.full(len(Zd), sd)); sd += 1
    Z = np.vstack(Zs); seg = np.concatenate(segs); lb = np.concatenate(labs)
    _save("eegmmidb", sub, Z, seg, lb, np.vstack(Zd_all), np.concatenate(lab_d),
          np.concatenate(seg_d))
    return sub, "ok", dict(n=int(len(Z)), n_seg=int(seg.max() + 1))


def _ee_safe(sub):
    try:
        return _ee_one(sub)
    except Exception as e:  # noqa: BLE001
        return sub, f"ERROR {e!r}", None


def build_eegmmidb(workers=6, records=None):
    import eegmmidb_states as EE
    from concurrent.futures import ProcessPoolExecutor
    pub = reference_counts("eegmmidb")
    subs = sorted(s for s in pub if _keep(s, records))
    todo = [s for s in subs if not (CACHE / "eegmmidb" / f"{s}.npz").exists()]
    log(f"eegmmidb: {len(subs)} subjects with a reference count, {len(todo)} to build")
    meta = dict(cohort="eegmmidb", dt=float(EE.STEP_DYN), d=int(EE.D), K=2,
                regimes=["rest", "task"], unit="subject (104, contrast rest_vs_task)", records={})
    if todo:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for i, r in enumerate(ex.map(_ee_safe, todo), 1):
                if i % 10 == 0 or r[1] != "ok":
                    log(f"  [{i}/{len(todo)}] {r[0]} {r[1]}")
    for s in subs:
        f = CACHE / "eegmmidb" / f"{s}.npz"
        if not f.exists():
            continue
        z = np.load(f)
        p = pub.get(s, {})
        meta["records"][s] = dict(subject=s, n=int(len(z["Z"])), n_seg=int(z["seg"].max() + 1),
                                  pub_n_stable=p.get("n_stable"), pub_n_eq=p.get("n_equilibria"))
    (CACHE / "eegmmidb" / "_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    log(f"eegmmidb: {len(meta['records'])} subjects cached")


# =============================================================================== sleep
def _sleep_one(item):
    import sleep_states as SL
    subj, psg, hyp = item
    eeg, sfreq, chans, epochs = SL.load_eeg(Path(psg), Path(hyp))
    features = SL.stage_features(eeg, sfreq, epochs)
    if len(features) < 2:
        return subj, f"only {len(features)} conditions", None
    states = SL.state_space(features)
    labels = sorted(states)
    chunk = max(2, int(SL.WIN_S / SL.STEP_S))
    Zs, segs, labs, s = [], [], [], 0
    for li, lname in enumerate(labels):
        Z = states[lname]
        for i in range(0, len(Z) - chunk + 1, chunk):
            Zs.append(Z[i:i + chunk]); segs.append(np.full(chunk, s)); labs.append(np.full(chunk, li)); s += 1
    if not Zs:
        return subj, "no segments", None
    Zc = np.vstack(Zs); seg = np.concatenate(segs); lb = np.concatenate(labs)
    Zd = np.vstack([states[l] for l in labels])
    ld = np.concatenate([np.full(len(states[l]), li) for li, l in enumerate(labels)])
    _save("sleep", subj, Zc, seg, lb, Zd, ld)
    return subj, "ok", dict(n=int(len(Zc)), n_seg=int(seg.max() + 1), labels=labels)


def _sleep_safe(item):
    try:
        return _sleep_one(item)
    except Exception as e:  # noqa: BLE001
        return item[0], f"ERROR {e!r}", None


def build_sleep(workers=6, records=None):
    import sleep_states as SL
    from concurrent.futures import ProcessPoolExecutor
    nights = SL.discover_nights()
    ref = reference_counts("sleep")
    pub = {s: dict(n_stable=v["n_stable"], n_eq=v["n_equilibria"], labels=v["labels"],
                   modes_K=v["ridgeline_modes_K"]) for s, v in ref.items()}
    items = [(s, str(v[0]), str(v[1])) for s, v in nights.items() if s in pub and _keep(s, records)]
    todo = [it for it in items if not (CACHE / "sleep" / f"{it[0]}.npz").exists()]
    log(f"sleep: {len(items)} nights with a reference count, {len(todo)} to build")
    meta = dict(cohort="sleep", dt=float(SL.STEP_S), d=int(SL.D_STATE), K=3,
                regimes=["NREM", "REM", "Wake"], unit="night (197; subject is SC4ss / ST7ss)",
                records={})
    if todo:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for i, r in enumerate(ex.map(_sleep_safe, todo), 1):
                if i % 10 == 0 or r[1] != "ok":
                    log(f"  [{i}/{len(todo)}] {r[0]} {r[1]}")
    for s, _, _ in items:
        f = CACHE / "sleep" / f"{s}.npz"
        if not f.exists():
            continue
        z = np.load(f)
        p = pub[s]
        meta["records"][s] = dict(night=s, subject=s[:5], which_night=s[5:],
                                  collection=("cassette" if s.startswith("SC") else "telemetry"),
                                  n=int(len(z["Z"])), n_seg=int(z["seg"].max() + 1),
                                  pub_n_stable=p["n_stable"], pub_n_eq=p["n_eq"],
                                  pub_modes_K=p["modes_K"], labels=p["labels"])
    (CACHE / "sleep" / "_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    log(f"sleep: {len(meta['records'])} nights cached")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cohort", default="all", help="one of fmri, meg, ds003838, eegmmidb, sleep, or all")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--records", nargs="*", default=None,
                    help="build only these records (participant, <subject>__<run> or night); the cohort metadata "
                         "then lists only them (default: every record)")
    a = ap.parse_args()
    records = set(a.records) if a.records else None
    t0 = time.time()
    todo = COHORTS if a.cohort == "all" else [a.cohort]
    for c in todo:
        log(f"=== {c} ===")
        try:
            if c == "fmri":
                build_fmri(records)
            elif c == "meg":
                build_meg(records)
            elif c == "ds003838":
                build_ds(records)
            elif c == "eegmmidb":
                build_eegmmidb(a.workers, records)
            elif c == "sleep":
                build_sleep(a.workers, records)
        except Exception:
            traceback.print_exc()
    log(f"done in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
