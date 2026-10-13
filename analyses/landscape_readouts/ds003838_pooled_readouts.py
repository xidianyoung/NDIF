"""Gaussian-mixture (BIC) and pairwise maximum-entropy basin counts of the 64 ds003838 recordings.

For each participant the pooled state cloud of all sequences (memorise and listen, encoding period) is
built in the reference state space of the participant: robust z (median and IQR) and a d = 8 PCA, both
fitted on the windows of the listen-only (control) sequences that pass the amplitude screen
(landscape_lib.RefSpace).  On the kept 1-s windows (0.5-s step) of all sequences:

  ladder_basins  landscape_lib.ladder_basins on the robust-z features (PCA to 8 dimensions, Gaussian
                 mixture with K = 1..6 chosen by BIC, ridgeline mode count of the selected mixture);
  ising_basins   landscape_lib.ising_basins on the 8 principal-component scores (median binarisation,
                 exact pairwise maximum-entropy fit, local minima of the basin graph).

Input   the ds003838 feature cache (paths.cache("ds003838"); NDIF_DS003838_CACHE; written by
        analyses/cohort_states/ds003838_states.py, read with its load_cache): per-sequence log band power
        F_dist on 1-s windows with 0.5-s step, amplitude flags flag_d, sequence condition.
Output  analyses/landscape_readouts/out/ds003838_pooled_readouts.json
        {participant: {ladder_basins, ladder_K_BIC, ising_basins, n_windows}}, read by readout_table.py.
Reported in  Fig. 2f, "Gaussian mixture, BIC" (1 x 41, 2 x 17, 3 x 4, 5 x 2) and "pairwise maximum
        entropy" (2 x 7, 4 x 25, 6 x 20, 8 x 10, 10 x 2) on the 64 recordings.

Runtime about 6 min on one core (about 5 s per participant).

    python analyses/landscape_readouts/ds003838_pooled_readouts.py
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse                                                       # noqa: E402
import glob                                                           # noqa: E402
import json                                                           # noqa: E402
import sys                                                            # noqa: E402
import time                                                           # noqa: E402
from pathlib import Path                                              # noqa: E402

import numpy as np                                                    # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from analyses.common import paths                                     # noqa: E402
from analyses.cohort_states import ds003838_states as DS              # noqa: E402
from analyses.landscape_readouts import landscape_lib as LL           # noqa: E402


def pooled_readouts(sub):
    """The two pooled readouts of one participant."""
    S = DS.load_cache(sub)
    keep = S["keep"]
    F_ref = np.vstack([S["F_dist"][i][keep[i]] for i in range(S["n"]) if S["cond"][i] == 0])
    sp = LL.RefSpace(LL.D).fit(F_ref)
    Fz_all = np.vstack([sp.z(S["F_dist"][i])[keep[i]] for i in range(S["n"])])
    Zd_all = np.vstack([sp.transform(S["F_dist"][i])[keep[i]] for i in range(S["n"])])
    pooled = dict(n_windows=int(len(Fz_all)))
    try:
        modes, k = LL.ladder_basins(Fz_all)
        pooled["ladder_basins"] = None if np.isnan(modes) else int(modes)
        pooled["ladder_K_BIC"] = int(k)
    except Exception as e:  # noqa: BLE001
        pooled["ladder_error"] = repr(e)
    try:
        pooled["ising_basins"] = int(LL.ising_basins(Zd_all.T, n_vars_max=15))
    except Exception as e:  # noqa: BLE001
        pooled["ising_error"] = repr(e)
    return pooled


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--subjects", nargs="*", default=None,
                    help="participants to compute (default: every participant in the cache)")
    a = ap.parse_args()
    cache = paths.cache("ds003838")
    subs = a.subjects or sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(cache, "sub-*.npz")))
    out = {}
    t0 = time.time()
    for i, sub in enumerate(subs, 1):
        out[sub] = pooled_readouts(sub)
        print(f"[{i}/{len(subs)}] {sub} GMM-BIC {out[sub].get('ladder_basins')} "
              f"(K {out[sub].get('ladder_K_BIC')}), MEM {out[sub].get('ising_basins')} "
              f"({time.time() - t0:.0f} s)", flush=True)
    target = paths.out("analyses", "landscape_readouts", "out", "ds003838_pooled_readouts.json")
    with open(target, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(out, fh, indent=1)
        fh.write("\n")
    print(f"wrote {os.path.basename(target)}")


if __name__ == "__main__":
    main()
