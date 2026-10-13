"""Per-record table of the landscape readouts of the 646 recordings (Fig. 2f; separations of Fig. 2c).

Two parts.

sleep_ladder  The Gaussian-mixture (BIC) readout of the 197 scored sleep nights: landscape_lib.ladder_basins
              on the overlapping-window state series of each night (Zdist of the count cache), fitted on a
              seeded subsample of --cap windows drawn without replacement and stratified by the stage
              grouping (labdist), so that the subsample keeps the night's stage proportions (a whole night
              costs about 150 s per fit).  Seed 20260921; the subsample size is stored with every value.
              Output analyses/landscape_readouts/out/sleep_ladder_readouts.json (resumable).
table         One row per record: the reference count of the mixture drift field
              (reference_pipeline_field, from analyses/cohort_states/reference_pipeline_counts.csv), the recount with the condition labels
              (sup_recount) and its regime separation (sup_separation), every label-free arm of
              count_recount.py, and the Gaussian-mixture (ladder_basins) and pairwise maximum-entropy
              (ising_basins) readouts.  A record's subject is the HCP participant for the two fMRI runs and
              the Sleep-EDF subject (first five characters) for the nights; otherwise the record.

Readouts per cohort: ds003838 from ds003838_pooled_readouts.py; sleep Gaussian mixture from the
sleep_ladder part; the readouts of the other cohorts (eegmmidb, HCP MEG, HCP fMRI, and the sleep pairwise
maximum-entropy readout) are read from --reference-readouts when given
({cohort: {record: {"ladder_basins": .., "ising_basins": ..}}}) and are left empty otherwise; Fig. 2f
uses the ds003838 rows only.

Inputs  analyses/cohort_states/reference_pipeline_counts.csv (--reference-counts),
        paths.cache("count") (sleep nights for the sleep_ladder part; NDIF_COUNT_CACHE),
        paths.cache("count_rows") (count_recount.py; NDIF_COUNT_ROWS_CACHE),
        analyses/landscape_readouts/out/ds003838_pooled_readouts.json, out/sleep_ladder_readouts.json.
Output  figure_reproduction/inputs/landscape_readout_counts.csv (the values of Fig. 2f; also read by
        drift_replacement.py for the separation threshold).
Reported in  Fig. 2f (five readouts on the 64 ds003838 recordings); Fig. 2c threshold (via
        drift_replacement.py).

Runtime: sleep_ladder about 15 min on 4 workers; table a few seconds.

    python analyses/landscape_readouts/readout_table.py --part all --workers 4 --cap 10000
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse                                                       # noqa: E402
import csv                                                            # noqa: E402
import glob                                                           # noqa: E402
import json                                                           # noqa: E402
import sys                                                            # noqa: E402
import time                                                           # noqa: E402
from pathlib import Path                                              # noqa: E402

import numpy as np                                                    # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from analyses.common import paths                                     # noqa: E402
from analyses.landscape_readouts import landscape_lib as LL           # noqa: E402

COHORTS = ["ds003838", "eegmmidb", "meg", "fmri", "sleep"]
ARMS = ["u1_gmm_Kfixed_segment", "u1w_gmm_Kfixed_sample", "u2_gmm_BIC", "u3_hmm_Kfixed"]
SEED = 20260921


# --------------------------------------------------------------------------- sleep Gaussian-mixture readout
def sleep_ladder_one(args):
    cache, rec, cap = args
    z = np.load(os.path.join(cache, "sleep", f"{rec}.npz"))
    X = z["Zdist"].astype(float)
    lab = z["labdist"].astype(int)
    rng = np.random.default_rng(SEED)
    if len(X) > cap:
        idx = []
        for k in np.unique(lab):
            w = np.where(lab == k)[0]
            take = max(1, int(round(cap * len(w) / len(X))))
            idx.append(rng.choice(w, size=min(take, len(w)), replace=False))
        idx = np.sort(np.concatenate(idx))
        X = X[idx]
    t0 = time.time()
    try:
        m, k = LL.ladder_basins(X)
        out = dict(ladder_basins=(None if (m is None or (isinstance(m, float) and np.isnan(m))) else int(m)),
                   ladder_K_BIC=int(k))
    except Exception as e:  # noqa: BLE001
        out = dict(ladder_error=repr(e))
    out.update(n_used=int(len(X)), n_total=int(len(z["Zdist"])), seconds=round(time.time() - t0, 1))
    return rec, out


def _sleep_safe(a):
    try:
        return sleep_ladder_one(a)
    except Exception as e:  # noqa: BLE001
        return a[1], dict(ladder_error=repr(e))


def run_sleep_ladder(workers, cap):
    cache = paths.cache("count")
    target = paths.out("analyses", "landscape_readouts", "out", "sleep_ladder_readouts.json")
    recs = sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(cache, "sleep", "*.npz")))
    done = {}
    if os.path.exists(target):
        with open(target, encoding="utf-8") as fh:
            done = json.load(fh)
    todo = [(cache, r, cap) for r in recs if r not in done]
    print(f"{len(recs)} nights, {len(todo)} to do", flush=True)

    def save():
        with open(target, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(done, indent=1))

    from concurrent.futures import ProcessPoolExecutor
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, (r, o) in enumerate(ex.map(_sleep_safe, todo), 1):
            done[r] = o
            if i % 10 == 0 or i == len(todo):
                save()
                print(f"[{i}/{len(todo)}] {r} {o.get('ladder_basins')} ({(time.time()-t0)/60:.1f} min)",
                      flush=True)
    save()


# --------------------------------------------------------------------------- per-record table
def subject_of(cohort, rec):
    if cohort == "fmri":
        return rec.split("__")[0]
    if cohort == "sleep":
        return rec[:5]
    return rec


def n_stable(arm):
    """The count an arm reports, or None where the arm could not define one."""
    if not isinstance(arm, dict) or arm.get("error"):
        return None
    if arm.get("K_eff") == 0 or arm.get("undefined_no_regime"):
        return None
    return arm.get("n_stable")


def _load_json(p):
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def load_reference(reference_counts, reference_readouts):
    """Per record: reference count of the mixture drift field and the two pooled readouts."""
    ref = {c: {} for c in COHORTS}
    other = _load_json(reference_readouts) if reference_readouts else {}
    with open(reference_counts, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            c, rec = row["cohort"], row["record"]
            if c not in ref:
                continue
            v = (other.get(c, {}) or {}).get(rec, {}) or {}
            ref[c][rec] = dict(field=(int(row["n_stable"]) if row["n_stable"] != "" else None),
                               ladder=v.get("ladder_basins"), ising=v.get("ising_basins"))
    ds = paths.out("analyses", "landscape_readouts", "out", "ds003838_pooled_readouts.json")
    if not os.path.exists(ds):
        raise SystemExit(f"missing {ds}: run ds003838_pooled_readouts.py first")
    for sub, v in _load_json(ds).items():
        if sub in ref["ds003838"]:
            ref["ds003838"][sub].update(ladder=v.get("ladder_basins"), ising=v.get("ising_basins"))
    sl = paths.out("analyses", "landscape_readouts", "out", "sleep_ladder_readouts.json")
    if os.path.exists(sl):
        for rec, v in _load_json(sl).items():
            if rec in ref["sleep"]:
                ref["sleep"][rec]["ladder"] = (v or {}).get("ladder_basins")
    return ref


def run_table(reference_counts, reference_readouts):
    rows_dir = paths.cache("count_rows")
    ref = load_reference(reference_counts, reference_readouts)
    per_record = []
    for c in COHORTS:
        files = sorted(glob.glob(os.path.join(rows_dir, c, "*.json")))
        if not files:
            continue
        rows = [_load_json(p) for p in files]
        rows = [r for r in rows if not r.get("error")]
        for r in rows:
            rec = r["record"]
            d = dict(cohort=c, record=rec, subject=subject_of(c, rec), n=r["n"], d=r["d"],
                     n_segments=r["n_seg"], K_reference_pipeline=r["K_reference_pipeline"],
                     reference_pipeline_field=ref[c].get(rec, {}).get("field"),
                     sup_recount=n_stable(r["sup"]),
                     sup_separation=r["sup"].get("separation_mahalanobis"),
                     ladder_basins=ref[c].get(rec, {}).get("ladder"),
                     ising_basins=ref[c].get(rec, {}).get("ising"),
                     perm_frac_same=r.get("perm_labels", {}).get("frac_equal_to_sup"),
                     km_resolved=(r.get("km_labelfree") or {}).get("resolved"),
                     km_wells=(r.get("km_labelfree") or {}).get("n_wells"),
                     km_valid_bins=(r.get("km_labelfree") or {}).get("n_valid_bins"),
                     km_labelaxis_resolved=(r.get("km_label_axis") or {}).get("resolved"),
                     km_labelaxis_wells=(r.get("km_label_axis") or {}).get("n_wells"),
                     mixture_fit_subsampled=(r.get("fit_subsample") or {}).get("used"),
                     mixture_fit_n=(r.get("fit_subsample") or {}).get("n_fit"))
            for arm in ARMS:
                a = r.get(arm, {}) or {}
                d[arm] = n_stable(a)
                d[arm + "__Keff"] = a.get("K_eff")
                d[arm + "__sep"] = a.get("separation_mahalanobis")
                d[arm + "__frac_pairs_kept"] = a.get("frac_pairs_kept")
            d["K_BIC"] = (r.get("u2_gmm_BIC", {}) or {}).get("K_BIC")
            per_record.append(d)

    first = ["cohort", "record", "subject", "n", "d", "n_segments", "K_reference_pipeline",
             "reference_pipeline_field", "sup_recount", "sup_separation"]
    keys = sorted({k for r in per_record for k in r})
    order = first + [k for k in keys if k not in first]
    target = paths.out("figure_reproduction", "inputs", "landscape_readout_counts.csv")
    with open(target, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=order, lineterminator="\n")
        w.writeheader()
        for r in per_record:
            w.writerow(r)
    print(f"wrote {len(per_record)} records to {os.path.basename(target)}")
    if not reference_readouts:
        print("no --reference-readouts: the Gaussian-mixture and maximum-entropy columns of eegmmidb, "
              "HCP MEG, HCP fMRI and the sleep maximum-entropy column are left empty")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--part", choices=["all", "sleep_ladder", "table"], default="all")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--cap", type=int, default=10000, help="windows per night for the sleep readout")
    ap.add_argument("--reference-counts",
                    default=paths.repo("analyses", "cohort_states", "reference_pipeline_counts.csv"),
                    help="reference count of the mixture drift field per record (column n_stable)")
    ap.add_argument("--reference-readouts", default=None,
                    help="JSON with the readouts of the cohorts other than ds003838 (see the docstring)")
    a = ap.parse_args()
    if a.part in ("all", "sleep_ladder"):
        run_sleep_ladder(a.workers, a.cap)
    if a.part in ("all", "table"):
        run_table(a.reference_counts, a.reference_readouts)


if __name__ == "__main__":
    main()
