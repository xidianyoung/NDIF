"""NDIF on real human recordings, by sleep stage.

Certificates for Sleep-EDF polysomnography (Kemp et al. 2000, via PhysioNet),
read from the supplied feature cache (analyses/sleep_feature_cache): 1-s
windows of eight unstandardized log band-power features (4 bands x 2
channels), one segment per scored 60-s bout.  The gates are run through the
public ``certify`` entry point.  Sleep stages differ in how separated and how
slow the state is, so the certificate is expected to differ between stages of
the same night.

    python examples/example_04_corpus.py --n 40            # default cache: analyses/sleep_feature_cache

The cache directory holds ``*_win1.npz`` files with ``X`` of shape
(n_bouts, n_windows, n_features) and a ``stage`` label per bout.
"""

import argparse
import collections
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # the repository root

from ndif import Design, certify

DEFAULT_CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "analyses", "sleep_feature_cache")

STAGES = ["Wake", "N1", "N2", "N3", "N4", "REM"]


def recordings(cache, limit):
    """One (name, stage, segments) triple per (recording, stage) with enough epochs."""
    files = sorted(glob.glob(os.path.join(cache, "*_win1.npz")))[:limit]
    for f in files:
        z = np.load(f, allow_pickle=True)
        X, stage = z["X"], z["stage"]
        win_s = float(z["win_s"])
        name = os.path.basename(f).replace("_win1.npz", "")
        for st in STAGES:
            idx = np.flatnonzero(stage == st)
            if idx.size < 6:                      # too few epochs to certify
                continue
            yield name, st, [np.asarray(X[i], float) for i in idx], win_s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=DEFAULT_CACHE)
    ap.add_argument("--n", type=int, default=25, help="recordings to read")
    ap.add_argument("--stages", default=",".join(STAGES))
    ap.add_argument("--estimators", default="ols,kalman",
                    help="certify once per estimator; the point of the run")
    a = ap.parse_args()
    want = set(a.stages.split(","))

    if not os.path.isdir(a.cache):
        print(f"cache not found: {a.cache}\nPass --cache <dir>.")
        return 1

    ests = [e.strip() for e in a.estimators.split(",") if e.strip()]
    tally = collections.defaultdict(collections.Counter)
    binding = collections.defaultdict(collections.Counter)
    recov = collections.defaultdict(list)
    margin = collections.defaultdict(list)
    nseen = collections.Counter()

    for name, st, segs, win_s in recordings(a.cache, a.n):
        if st not in want:
            continue
        X = np.concatenate(segs, axis=0)
        edges = np.cumsum([0] + [len(s) for s in segs])
        bounds = [(edges[i], edges[i + 1]) for i in range(len(segs))]
        design = Design(dt=win_s, window_s=win_s, step_s=win_s, segments=bounds,
                        name=f"{name}/{st}", modality="polysomnography")
        for est in ests:
            cert = certify(X, design, configuration="benchmark",
                           rng=1, estimator=est,
                           G3=dict(n_rep=8), G7=dict(n_rep=40))
            key = (st, est)
            for lvl in ("L1", "L2", "L3"):
                tally[key][f"{lvl}:{cert.levels[lvl].status}"] += 1
            binding[key][str(cert.levels["L3"].binding_gate)] += 1
            g3 = cert.gate("G3")
            if g3 is not None and np.isfinite(g3.value):
                recov[key].append(float(g3.value))
            if g3 is not None:
                m = (g3.detail or {}).get("detection_margin_fold")
                if m is not None and np.isfinite(m):
                    margin[key].append(float(m))
            nseen[key] += 1

    print("=" * 78)
    print("NDIF certificates on Sleep-EDF, by scored stage")
    print("=" * 78)
    print(f"{'stage':>6} {'estimator':>10} {'n':>4} {'L2 adm':>8} {'L3 adm':>8} "
          f"{'median G3':>10} {'MDE fold':>9}  binding gates for L3")
    print("-" * 78)
    for st in STAGES:
        for est in ests:
            key = (st, est)
            if not nseen[key]:
                continue
            n = nseen[key]
            l2 = tally[key]["L2:admitted"] / n
            l3 = tally[key]["L3:admitted"] / n
            med = np.median(recov[key]) if recov[key] else float("nan")
            gates = ", ".join(f"{k} x{v}" for k, v in binding[key].most_common(3))
            mm = np.median(margin[key]) if margin[key] else float("nan")
            print(f"{st:>6} {est:>10} {n:>4} {l2:>7.0%} {l3:>7.0%} "
                  f"{med:>10.3f} {mm:>9.2f}  {gates}")
    print("-" * 78)
    print("Read the last two columns together.  Where the certificate admits a")
    print("level-3 claim it is because a known injected truth came back through")
    print("this recording's own chain, and where it refuses, the gate that binds")
    print("is named.  The verdict is a property of the recording, so it differs")
    print("between stages of the same night.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
