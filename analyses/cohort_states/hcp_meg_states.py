# -*- coding: utf-8 -*-
"""HCP MEG working memory: per-participant cache of the within-trial state pairs and the load axis.

State construction (one participant, runs 6 and 7 of Wrkmem, tmegpreproc TIM files: trials aligned to stimulus onset)
  sensors  the magnetometers common to both runs (the release drops a few channels per run), in femtotesla
  trials   every task trial (fixation trials dropped; trials with non-finite samples dropped), analysed span
           [0, 2.5) s after stimulus onset; at least 40 trials are required
  features log10 band power of 1-s windows (theta 4-8, alpha 8-12, beta 12-30, gamma 30-45 Hz) per sensor,
           taken within trials only: dist windows at [0, 1), [0.5, 1.5), [1, 2), [1.5, 2.5) s and dyn windows at
           [0, 1) and [1, 2) s (dt = 1 s)
  space    robust z (median / IQR) + PCA to d = 8, both fitted on the dist windows of the 0-back trials
           (states_lib.RefSpace, random_state 20260728)
  pairs    per load (0 = 0-back, 1 = 2-back) and trial, the two dyn states of the trial in time order

Input: NDIF_HCP_MEG_RAW (<subject>/MEG/Wrkmem/tmegpreproc/<subject>_MEG_{6,7}-Wrkmem_tmegpreproc_TIM.mat).
Output (paths.cache("meg"), environment variable NDIF_MEG_CACHE, default
analyses/primary_configuration/meg_feature_cache), one <subject>.npz per participant:
  X0, Y0, X1, Y1   (n_trials, 8) the first and second dyn state of every trial, per load
  trial0, trial1   trial indices of the pairs (order of the task trials over both runs)
  v                (8,) unit load axis, mean dist state of 0-back minus 2-back
  v_a, v_b         the same axis from the odd and from the even trials only
This is the per-trial state cache read by analyses/primary_configuration/benchmark_04_meg_clock.py and
analyses/workload_certificates/benchmark_06_load_cohorts.py, and by count_cache.py (MEG arm: 83 participants).
Runtime: about 7 min for the 83 participants with 6 workers on one desktop (about 0.7 GB per run in memory).
Threads: the numerical libraries' default thread count, as for the stored cache.

    python analyses/cohort_states/hcp_meg_states.py --workers 6
"""
from __future__ import annotations

import argparse
import gc
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import scipy.io as sio

HERE = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))
from analyses.common import paths                                  # noqa: E402
from states_lib import short_time_band_power, log_features, RefSpace  # noqa: E402

DATA = Path(paths.path("NDIF_HCP_MEG_RAW"))
OUT = Path(paths.cache("meg"))
RUNS = (6, 7)
D = 8
WIN, STEP_DIST, STEP_DYN = 1.0, 0.5, 1.0
BANDS = {"theta": (4, 8), "alpha": (8, 12), "beta": (12, 30), "gamma": (30, 45)}
TRIAL_SPAN = 2.5                # analysed span of each trial: [0, 2.5) s after onset
SCALE = 1e15                    # tesla -> femtotesla


def run_file(subj, run):
    return DATA / subj / "MEG" / "Wrkmem" / "tmegpreproc" / f"{subj}_MEG_{run}-Wrkmem_tmegpreproc_TIM.mat"


def scan_labels(subj, run):
    """Sensor labels of one run."""
    d = sio.loadmat(str(run_file(subj, run)), squeeze_me=True, struct_as_record=False,
                    variable_names=["data"])["data"]
    labels = [str(x) for x in np.atleast_1d(d.label)]
    del d
    gc.collect()
    return labels


def load_run(subj, run, keep_labels):
    """Per-trial feature dicts of one run on the common sensors (keep_labels, in that order)."""
    d = sio.loadmat(str(run_file(subj, run)), squeeze_me=True, struct_as_record=False,
                    variable_names=["data"])["data"]
    fs = float(d.fsample)
    labels = [str(x) for x in np.atleast_1d(d.label)]
    pick = np.array([labels.index(l) for l in keep_labels])
    T = np.asarray(d.trialinfo, float)
    tms = d.time
    per_trial_time = getattr(tms, "dtype", None) == object
    trials = []
    for k in range(len(d.trial)):
        if not np.isfinite(T[k, 1]):                 # fixation trial (no block number)
            continue
        tm = np.asarray(tms[k] if per_trial_time else tms, float)
        i0 = int(np.searchsorted(tm, 0.0))
        i1 = i0 + int(round(TRIAL_SPAN * fs))
        x = np.asarray(d.trial[k], float)
        if i1 > x.shape[1]:
            i1 = x.shape[1]
        x = x[pick, i0:i1] * SCALE
        if not np.isfinite(x).all():
            continue
        Fd = log_features(short_time_band_power(x, fs, WIN, STEP_DIST, BANDS))
        Fy = log_features(short_time_band_power(x, fs, WIN, STEP_DYN, BANDS))
        trials.append(dict(run=run, load=0 if T[k, 4] == 1 else 1, F_dist=Fd, F_dyn=Fy))
    del d
    gc.collect()
    return trials


def window_table(trials, field, offsets):
    """Windows of all trials stacked, with the trial index, load and within-trial time of each window."""
    F = np.vstack([t[field] for t in trials])
    meta = dict(
        trial=np.concatenate([np.full(len(t[field]), i) for i, t in enumerate(trials)]),
        load=np.concatenate([np.full(len(t[field]), t["load"]) for t in trials]),
        t_in_trial=np.concatenate([np.asarray(offsets[:len(t[field])], float) for t in trials]))
    return F, meta


def one(subj):
    t0 = time.time()
    present = [r for r in RUNS if run_file(subj, r).exists()]
    label_sets = {r: scan_labels(subj, r) for r in present}
    common = [l for l in label_sets[present[0]] if all(l in set(label_sets[r]) for r in present)]
    trials = []
    for r in present:
        trials += load_run(subj, r, common)
    if len(trials) < 40:
        return (subj, "too few trials")
    Fd, md = window_table(trials, "F_dist", [0.5 * WIN + STEP_DIST * k for k in range(8)])  # centres 0.5..2.0 s
    Fy, my = window_table(trials, "F_dyn", [0.5 * WIN + STEP_DYN * k for k in range(8)])    # centres 0.5, 1.5 s
    sp = RefSpace(D).fit(Fd[md["load"] == 0])
    Zd, Zy = sp.transform(Fd), sp.transform(Fy)

    def axis(mask):
        a = Zd[(md["load"] == 0) & mask].mean(0)
        b = Zd[(md["load"] == 1) & mask].mean(0)
        u = a - b
        return u / (np.linalg.norm(u) + 1e-12)

    v = axis(np.ones(len(Zd), bool))
    odd = (md["trial"] % 2) == 1
    v_a, v_b = axis(odd), axis(~odd)

    out = {}
    for ld in (0, 1):
        Xs, Ys, tt = [], [], []
        for t in np.unique(my["trial"][my["load"] == ld]):
            m = my["trial"] == t
            idx = np.where(m)[0][np.argsort(my["t_in_trial"][m])]
            if len(idx) >= 2:
                Xs.append(Zy[idx[0]]); Ys.append(Zy[idx[1]]); tt.append(int(t))
        out[f"X{ld}"] = np.array(Xs); out[f"Y{ld}"] = np.array(Ys)
        out[f"trial{ld}"] = np.array(tt, int)
    out["v"] = v; out["v_a"] = v_a; out["v_b"] = v_b
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"{subj}.npz", **out)
    return (subj, round(time.time() - t0, 1))


def _safe(s):
    try:
        return one(s)
    except Exception as e:  # noqa: BLE001
        return (s, f"ERROR {e!r}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--subjects", nargs="*", default=None, help="participants to build (default: all)")
    a = ap.parse_args()
    subs = sorted(p.name for p in DATA.iterdir() if p.is_dir() and any(run_file(p.name, r).exists() for r in RUNS))
    if a.subjects:
        subs = [s for s in subs if s in set(a.subjects)]
    subs = [s for s in subs if not (OUT / f"{s}.npz").exists()]
    print(f"{len(subs)} to do", flush=True)
    t0 = time.time()
    if a.workers > 1 and subs:
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            for i, r in enumerate(ex.map(_safe, subs), 1):
                print(f"[{i}/{len(subs)}] {r}  ({(time.time()-t0)/60:.1f} min)", flush=True)
    else:
        for i, s in enumerate(subs, 1):
            print(f"[{i}/{len(subs)}] {_safe(s)}", flush=True)
    print("done", (time.time() - t0) / 60, "min")


if __name__ == "__main__":
    main()
