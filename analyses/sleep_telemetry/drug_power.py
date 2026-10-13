# -*- coding: utf-8 -*-
r"""Bounds of the paired temazepam/placebo contrasts in the Sleep-EDF telemetry crossover: fold,
interval, tests and the resampling power of the participant sign test against known folds
(Supplementary Table 9; Fig. 4c drug contrast; Results).

For every stage and quantity (`onestep` directional spread, `onestep_bc` rate; cells as in
crossover.py) with at least 6 participants who have the stage resolved in both nights:
  lr        participant log ratio temazepam/placebo
  fold      exp(median lr), 95% percentile interval from 4,000 participant resamples
            (generator seeded 20260921 + 57), sign test, two-sided Wilcoxon test
  power     for each fold f in 1.02-2.00 (spread_lib.FOLD_GRID) and each direction, lr is centred
            on its median, shifted by +-ln f, participants are resampled 4,000 times and the
            two-sided sign test is run at alpha 0.05; the power is the share of resamples with
            P < 0.05 in the direction of the shift (generator seeded 20260921 + 55, fresh per fold)
  mde80     the smallest fold in the grid with power >= 0.8, upward and downward
Reported (S2 spread, n 21): fold 1.002 [0.929, 1.123], Wilcoxon P 0.5854; 80%-power folds 1.125
upward and 1.20 downward; power at 1.20 0.9295 upward and 0.8033 downward.

Inputs   paths.cache("sleep_bout_rows")/telemetry_crossover_rows.json (bout_spread_rows.py, or the
         Source Data copy) and ST-subjects.xls in NDIF_SLEEPEDF_RAW.
Output   figure_reproduction/inputs/sleep_drug_contrast_bounds.json (below NDIF_OUT when set)
Runtime  under a minute on one core.

    python analyses/sleep_telemetry/drug_power.py
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import json                                                           # noqa: E402
import math                                                           # noqa: E402
import sys                                                            # noqa: E402
import time                                                           # noqa: E402
from pathlib import Path                                              # noqa: E402

import numpy as np                                                    # noqa: E402
from scipy import stats as sst                                        # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))
from analyses.common import paths  # noqa: E402
import spread_lib as L  # noqa: E402

SEED = L.SEED
OUT_JSON = ("figure_reproduction", "inputs", "sleep_drug_contrast_bounds.json")


def main() -> int:
    t0 = time.time()
    tel = L.load_rows("telemetry")
    bc = L.bias_curve(np.random.default_rng(SEED + 1))
    inv = L.make_bias_inverse(bc)

    dmap = L.telemetry_drug_map()
    tn = [x for x in tel["nights"] if "rows" in x]
    lab = {x["stem"]: L.telemetry_label(x["stem"], dmap) for x in tn}
    tc = {}
    for drug in ("placebo", "temazepam"):
        sub = [x for x in tn if lab[x["stem"]] == drug]
        tc[drug] = {a: L.cellify(sub, a, inv) for a in ("onestep", "onestep_bc")}
    drug = {}
    for arm, key in [("onestep", "width"), ("onestep_bc", "lam")]:
        for s in L.STAGES:
            P = {st[:5]: c[s][key] for st, c in tc["placebo"][arm].items()
                 if s in c and c[s]["resolved"] and c[s].get(key)}
            T = {st[:5]: c[s][key] for st, c in tc["temazepam"][arm].items()
                 if s in c and c[s]["resolved"] and c[s].get(key)}
            both = sorted(set(P) & set(T))
            if len(both) < 6:
                drug[f"{arm}|{key}|{s}"] = dict(n_subjects=len(both), testable=False)
                continue
            lr = np.array([math.log(T[x] / P[x]) for x in both], float)
            k = int((lr > 0).sum())
            grid = {}
            for f in L.FOLD_GRID:
                for sg, nm in ((+1, "up"), (-1, "down")):
                    grid[f"{nm}_x{f}"] = round(L.power_by_resampling(
                        lr, sg * math.log(f), np.random.default_rng(SEED + 55)), 4)
            mde = {}
            for nm in ("up", "down"):
                mde[nm] = next((f for f in L.FOLD_GRID if grid[f"{nm}_x{f}"] >= 0.8), None)
            bsr = np.random.default_rng(SEED + 57)
            bo = np.array([np.median(lr[bsr.integers(0, len(lr), len(lr))]) for _ in range(4000)])
            drug[f"{arm}|{key}|{s}"] = dict(
                testable=True, n_subjects=len(both), k_temazepam_greater=k,
                fold=float(np.exp(np.median(lr))),
                ci95=[float(np.exp(np.percentile(bo, 2.5))),
                      float(np.exp(np.percentile(bo, 97.5)))],
                sign_p=float(sst.binomtest(k, len(lr), 0.5).pvalue),
                wilcoxon_p=float(sst.wilcoxon(lr).pvalue),
                mde80_up=mde["up"], mde80_down=mde["down"], power_by_fold=grid)

    p = paths.out(*OUT_JSON)
    Path(p).write_text(json.dumps(dict(seed=SEED, drug_contrast_bounds=drug), indent=1) + "\n",
                       encoding="utf-8", newline="\n")
    for k, v in drug.items():
        if v.get("testable"):
            print(f"  {k:26s} n={v['n_subjects']:2d} fold={v['fold']:.4f} "
                  f"[{v['ci95'][0]:.3f},{v['ci95'][1]:.3f}] P={v['sign_p']:.3g} "
                  f"W={v['wilcoxon_p']:.3g} MDE80 up=x{v['mde80_up']} down=x{v['mde80_down']}")
        else:
            print(f"  {k:26s} n={v['n_subjects']} not testable")
    print(f"wrote {p} ({time.time() - t0:.1f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
