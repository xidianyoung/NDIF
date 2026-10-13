"""Assess the two load cohorts: COG-BCI (28 participants after the duplicate exclusion) and HCP MEG
(83 participants), under the lag-1 least-squares rate and the primary configuration.

Unit of assessment: one recording per participant and task family, pooled over sessions and load
levels, with the load level as the label (so G5 can be evaluated where there are two levels).

  COG-BCI  features/cog_state.build(): the 248-feature log band-power state reduced to
           d = 8 by a PCA fitted on the load-blind Flanker reference task, non-overlapping
           1-s windows, matched blocks per level.
  HCP MEG  <MEG cache>/<subject>.npz: the 8-dimensional per-trial state
           on the 2.5-s trial clock, contiguous trial runs as segments (as in
           analyses/primary_configuration/benchmark_04_meg_clock.py), levels 0-back and 2-back
           as the label.  The rate channel of G3 is evaluated from the generator fit also where the
           fit places the white share at its boundary (G3 boundary_rule=False; MEG_GATE_KW).

    python analyses/workload_certificates/benchmark_06_load_cohorts.py --workers 4

Writes the COG-BCI outputs next to this script (corpus_cog/<sid>_<family>.json, corpus_cog.log,
corpus_load_summary_cog.json) and the HCP MEG outputs to analyses/primary_configuration/out (corpus_meg/<subject>.json,
corpus_meg.log, corpus_load_summary_meg.json).
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import collections
import glob
import json
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(os.path.dirname(HERE))                       # the repository root
sys.path.insert(0, PKG)
sys.path.insert(0, os.path.join(PKG, "analyses", "primary_configuration"))
sys.path.insert(0, os.path.join(HERE, "features"))               # the COG-BCI state construction after the duplicate exclusion

from benchmark_05_corpus_full import certificate_record, jsonable   # noqa: E402

# HCP MEG per-trial state cache (one .npz per participant) and the COG-BCI feature cache are derived from the source
# data under their access terms and are not redistributed; set NDIF_MEG_CACHE and NDIF_COGBCI_CACHE.
MEG_CACHE = os.environ.get("NDIF_MEG_CACHE", os.path.join(PKG, "analyses", "primary_configuration", "meg_feature_cache"))
OUT_COG = HERE
OUT_MEG = os.path.join(PKG, "analyses", "primary_configuration", "out")
GATES = ["G0", "G1", "G2", "G3", "G4", "G5", "G6", "G7"]
DT_TRIAL = 2.5
# Gate settings of the HCP MEG assessments: G3 evaluates the rate channel from the generator fit even where the fit
# places the white share at its boundary (ndif.gates.NU_BOUNDARY).  The COG-BCI assessments use the defaults.
MEG_GATE_KW = dict(G3=dict(boundary_rule=False))


# ----------------------------------------------------------------------------- COG-BCI
def cog_recordings():
    """(sid, family) -> (segments, labels) from the COG-BCI state construction."""
    import cog_features as C
    import cog_state as CS
    subs = C.load_subjects()
    _, data = CS.build(subs)
    out = {}
    for sid, cells in data.items():
        by_fam = collections.defaultdict(list)
        for (fam, si), per in cells.items():
            for lev in sorted(per):
                for blk in per[lev]["blocks"]:
                    by_fam[fam].append((np.asarray(blk, float), lev))
        for fam, items in by_fam.items():
            out[(sid, fam)] = items
    return out


def meg_recordings():
    out = {}
    for f in sorted(glob.glob(os.path.join(MEG_CACHE, "*.npz"))):
        z = np.load(f, allow_pickle=True)
        items = []
        for lev in (0, 1):
            X, tr = np.asarray(z[f"X{lev}"], float), np.asarray(z[f"trial{lev}"], int)
            order = np.argsort(tr)
            X, tr = X[order], tr[order]
            cur = [X[0]]
            for i in range(1, len(tr)):
                if tr[i] == tr[i - 1] + 1:
                    cur.append(X[i])
                else:
                    if len(cur) >= 3:
                        items.append((np.asarray(cur), lev))
                    cur = [X[i]]
            if len(cur) >= 3:
                items.append((np.asarray(cur), lev))
        out[os.path.basename(f).replace(".npz", "")] = items
    return out


def certify_items(name, items, dt, modality, out_path):
    from ndif import Design, certify
    t0 = time.time()
    try:
        segs = [s for s, _ in items]
        X = np.concatenate(segs, 0)
        e = np.cumsum([0] + [len(s) for s in segs])
        bounds = [(int(e[i]), int(e[i + 1])) for i in range(len(segs))]
        lab = np.concatenate([np.full(len(s), lev, float) for s, lev in items])
        design = Design(dt=dt, window_s=dt, step_s=dt, segments=bounds, labels=lab,
                        name=name, modality=modality)
        gate_kw = MEG_GATE_KW if modality == "MEG" else {}
        cert = certify(X, design, which=GATES, configuration="primary", rng=1, estimator="ols", **gate_kw)
        rec = dict(kind="cell", recording=name, estimator="ols", n_segments=len(segs),
                   n_windows=int(X.shape[0]), dim=int(X.shape[1]), dt=dt,
                   levels_present=sorted({int(l) for _, l in items}),
                   gates_consulted=GATES, replicates="package defaults",
                   **certificate_record(cert))
        rec["wall_s"] = time.time() - t0
        tmp = out_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(rec, fh, indent=1)
        os.replace(tmp, out_path)
        return out_path, None
    except Exception:
        return out_path, traceback.format_exc()


def summarize(in_dir, label, log_path):
    rows = []
    for f in sorted(glob.glob(os.path.join(in_dir, "*.json"))):
        with open(f, encoding="utf-8") as fh:
            rows.append(json.load(fh))
    lines = ["=" * 78, f"NDIF certificates, {label}, lag-1 least-squares rate, package defaults",
             f"{len(rows)} recordings", "=" * 78]
    summ = dict(n=len(rows))
    for lvl in ("L1", "L2", "L3"):
        st = collections.Counter(r["levels"][lvl]["status"] for r in rows)
        bind = collections.Counter(r["levels"][lvl]["binding_gate"] for r in rows
                                   if r["levels"][lvl]["status"] == "refused")
        summ[lvl] = dict(status=dict(st), binding_refused=dict(bind))
        lines.append(f"  {lvl}: " + ", ".join(f"{k} {v}" for k, v in st.items())
                     + ("  | binds: " + ", ".join(f"{k} x{v}" for k, v in bind.most_common()) if bind else ""))

    def g(r, name):
        for x in r["gates"]:
            if x["gate"] == name:
                return x
        return None

    def med(xs):
        xs = [x for x in xs if x is not None and np.isfinite(x)]
        return float(np.median(xs)) if xs else None

    summ["median_g1_nu"] = med([g(r, "G1")["value"] for r in rows if g(r, "G1")])
    summ["median_g3_recovery"] = med([g(r, "G3")["value"] for r in rows if g(r, "G3")])
    summ["median_detection_margin_fold"] = med([(g(r, "G3")["detail"] or {}).get("detection_margin_fold")
                                                for r in rows if g(r, "G3")])
    summ["g1_indeterminate"] = sum(1 for r in rows if g(r, "G1") and g(r, "G1")["passed"] is None)
    summ["rows"] = [dict(recording=r["recording"], L1=r["levels"]["L1"]["status"],
                         L2=r["levels"]["L2"]["status"], L3=r["levels"]["L3"]["status"],
                         L3_binding=r["levels"]["L3"]["binding_gate"],
                         g1_nu=(g(r, "G1")["value"] if g(r, "G1") else None),
                         g3_recovery=(g(r, "G3")["value"] if g(r, "G3") else None),
                         detection_margin_fold=((g(r, "G3")["detail"] or {}).get("detection_margin_fold")
                                                if g(r, "G3") else None)) for r in rows]
    lines.append(f"  median G1 white share {summ['median_g1_nu']}, G1 indeterminate in "
                 f"{summ['g1_indeterminate']} of {len(rows)}; median G3 recovery {summ['median_g3_recovery']}; "
                 f"median detection margin x{summ['median_detection_margin_fold']}")
    with open(log_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--cohorts", default="cog,meg")
    a = ap.parse_args()
    tasks = []
    if "cog" in a.cohorts:
        d = os.path.join(OUT_COG, "corpus_cog"); os.makedirs(d, exist_ok=True)
        for (sid, fam), items in cog_recordings().items():
            p = os.path.join(d, f"sub-{sid:02d}_{fam}.json")
            if not os.path.exists(p):
                tasks.append((f"COG-BCI sub-{sid:02d}/{fam}", items, 1.0, "EEG", p))
    if "meg" in a.cohorts:
        d = os.path.join(OUT_MEG, "corpus_meg"); os.makedirs(d, exist_ok=True)
        for subj, items in meg_recordings().items():
            p = os.path.join(d, f"{subj}.json")
            if not os.path.exists(p):
                tasks.append((f"HCP MEG {subj}", items, DT_TRIAL, "MEG", p))
    print(f"{len(tasks)} recordings to certify ({a.workers} workers)", flush=True)
    t0 = time.time(); fail = 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(certify_items, *t) for t in tasks]
        for k, f in enumerate(as_completed(futs), 1):
            out, err = f.result()
            if err:
                fail += 1
                print(f"  ERROR {os.path.basename(out)}: {err.strip().splitlines()[-1][:120]}", flush=True)
            if k % 10 == 0 or k == len(tasks):
                print(f"  {k}/{len(tasks)} failed={fail} {(time.time()-t0)/60:.1f} min", flush=True)
    if "cog" in a.cohorts:
        s = summarize(os.path.join(OUT_COG, "corpus_cog"), "COG-BCI (28 participants x 2 task families; sub-27 and sub-28 session 1 excluded as duplicates)",
                      os.path.join(OUT_COG, "corpus_cog.log"))
        with open(os.path.join(OUT_COG, "corpus_load_summary_cog.json"), "w", encoding="utf-8") as fh:
            json.dump(jsonable({"COG-BCI": s}), fh, indent=1)
    if "meg" in a.cohorts:
        s = summarize(os.path.join(OUT_MEG, "corpus_meg"), "HCP MEG working memory (83 participants, trial clock)",
                      os.path.join(OUT_MEG, "corpus_meg.log"))
        with open(os.path.join(OUT_MEG, "corpus_load_summary_meg.json"), "w", encoding="utf-8") as fh:
            json.dump(jsonable({"HCP MEG": s}), fh, indent=1)
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
