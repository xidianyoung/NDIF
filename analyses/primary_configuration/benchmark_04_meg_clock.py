"""HCP MEG identifiability coordinate on the trial clock, without a lag-1 floor.

  CLOCK.  The MEG state carries one window per trial, so its lag-1 autocorrelation is
          a per-trial quantity and its sampling interval is the 2.5 s trial clock, not a
          1-s window.

  FLOOR.  Every participant in the cached state series is used.  The effect of
          restricting the cohort to units whose lag-1 autocorrelation clears 0.08 is
          reported separately, because such a selection acts on the very statistic that
          fixes the coordinate.

    python analyses/primary_configuration/benchmark_04_meg_clock.py
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))   # the repository root

from ndif.estimators import white_share                              # noqa: E402

# HCP MEG per-trial state cache (one .npz per participant), derived from the HCP source data under their access
# terms; not redistributed.  Set NDIF_MEG_CACHE or pass --cache.
CACHE = os.environ.get("NDIF_MEG_CACHE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "meg_feature_cache"))
DT_TRIAL = 2.5          # nominal trial SOA: the clock this cohort's lag-1 lives on
R1_FLOOR = 0.08         # lag-1 floor, applied only to measure the effect of floor selection


def per_subject(path, level):
    """Rebuild the per-trial state sequence on the load axis, as contiguous runs."""
    z = np.load(path, allow_pickle=True)
    X, Y, tr, v = z[f"X{level}"], z[f"Y{level}"], z[f"trial{level}"], z["v"]
    v = np.asarray(v, float)
    v = v / np.linalg.norm(v)
    x = np.asarray(X, float) @ v                       # state at trial t
    t = np.asarray(tr, int)
    order = np.argsort(t)
    x, t = x[order], t[order]
    runs, cur = [], [x[0]]
    for i in range(1, len(t)):
        if t[i] == t[i - 1] + 1:
            cur.append(x[i])
        else:
            if len(cur) >= 3:
                runs.append(np.asarray(cur))
            cur = [x[i]]
    if len(cur) >= 3:
        runs.append(np.asarray(cur))
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, default=0, help="0 = 0-back, 1 = 2-back")
    ap.add_argument("--cache", default=CACHE, help="directory of the per-trial MEG state cache (default: NDIF_MEG_CACHE)")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  "out", "meg_clock.json"))
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.cache, "*.npz")))
    if not files:
        print(f"no MEG cache at {a.cache}")
        return 2
    print(f"{len(files)} subjects in the cache; lag-1 is one trial = {DT_TRIAL} s\n")

    rows = []
    for f in files:
        sid = os.path.basename(f).replace(".npz", "")
        runs = per_subject(f, a.level)
        if not runs or sum(len(r) for r in runs) < 12:
            rows.append(dict(subject=sid, usable=False))
            continue
        w = white_share([r[:, None] for r in runs])
        r1, r2 = w["r1"], w["r2"]
        rows.append(dict(subject=sid, usable=True, n_runs=len(runs),
                         n_points=int(sum(len(r) for r in runs)),
                         r1=r1, r2=r2, a=w["a"], nu=w["nu"], ok=bool(w["ok"]),
                         clears_floor=bool(np.isfinite(r1) and r1 >= R1_FLOOR)))

    good = [r for r in rows if r.get("usable") and r.get("ok")]
    allr = [r for r in rows if r.get("usable") and np.isfinite(r.get("r1", np.nan))]
    kept = [r for r in good if r["clears_floor"]]

    def med(rs, k):
        v = [r[k] for r in rs if r.get(k) is not None and np.isfinite(r[k])]
        return float(np.median(v)) if v else float("nan")

    out = {}
    for label, rs in (("all units, no floor", good), ("floor-selected units", kept)):
        a_med, nu_med = med(rs, "a"), med(rs, "nu")
        lam_per_trial = -np.log(a_med) if 0 < a_med < 1 else float("nan")
        out[label] = dict(
            n=len(rs), a_median=a_med, nu_median=nu_med,
            lam_dt_trial_clock=lam_per_trial,                    # dt = one trial
            lam_dt_if_read_as_1s=lam_per_trial * DT_TRIAL,       # the value on a 1-s window clock
            lam_per_second=lam_per_trial / DT_TRIAL)
        print(f"{label}:")
        print(f"  n = {len(rs)} of {len(allr)} usable")
        print(f"  median a  = {a_med:.4f}   median nu = {nu_med:.4f}")
        print(f"  lambda*dt on the TRIAL clock      = {lam_per_trial:.4f}")
        print(f"  lambda*dt if read as a 1 s window = {lam_per_trial * DT_TRIAL:.4f}")
        print(f"  lambda per second                 = {lam_per_trial / DT_TRIAL:.4f}\n")

    sel = dict(n_usable=len(allr), n_clearing_floor=sum(r["clears_floor"] for r in allr),
               min_r1=float(min(r["r1"] for r in allr)),
               floor=R1_FLOOR)
    sel["fraction_kept"] = sel["n_clearing_floor"] / max(sel["n_usable"], 1)
    print(f"floor effect: {sel['n_clearing_floor']} of {sel['n_usable']} units clear "
          f"r1 >= {R1_FLOOR} ({sel['fraction_kept']:.0%}); smallest r1 in the cache is "
          f"{sel['min_r1']:.4f}")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(dict(dt_trial_s=DT_TRIAL, level=a.level, summary=out,
                       floor_effect=sel, rows=rows), fh, indent=1)
    print(f"\nwritten: {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
