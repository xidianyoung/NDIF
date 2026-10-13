# -*- coding: utf-8 -*-
"""HCP working-memory fMRI: the 7-parcel state series of each run and its 0-back / 2-back block labels.

State construction (one run, WM_LR or WM_RL; 405 volumes, TR = 0.72 s)
  series   the 7 parcel time series of the run (detrended parcel means of the MNI-space volumes; the parcellation
           of the volumes is not part of this repository) in robust z: (x - median) / max(IQR, 1e-9) per parcel
           over the whole run (states_lib.median_iqr_z)
  labels   from the task EV files: a volume belongs to a block of load 0-back (0) or 2-back (1) if its midpoint
           falls in [onset + 5 s, onset + 5 s + duration) of that block (hemodynamic shift 5 s); all other
           volumes (fixation, cues) are 2 and are not used
  segments the volumes of each block of each load, in time order (dt = TR)

Input (NDIF_HCP_FMRI_PARCELS, one directory)
  <subject>__<run>__atlas_fpn7.nii.npy   (405, 7) parcel series of one run
  <subject>__<run>__EVs/                 the run's task EV files as released by the HCP
                                         (MNINonLinear/Results/tfMRI_<run>/EVs/): {0bk,2bk}_{body,faces,places,tools}.txt,
                                         rows (onset s, duration s, amplitude)
Used by count_cache.py (fMRI arm: 198 runs of 99 participants).  No computation is run on import.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))
from analyses.common import paths                                          # noqa: E402
from states_lib import median_iqr_z, fit_linear_regime, find_equilibria   # noqa: E402

PARCELS = Path(paths.path("NDIF_HCP_FMRI_PARCELS"))
ATLAS = "atlas_fpn7.nii"
TR = 0.72
N_VOLS = 405
HRF_SHIFT_S = 5.0
RUNS = ("WM_LR", "WM_RL")
CONDS = ("0bk", "2bk")
CATS = ("body", "faces", "places", "tools")
SEED = 20260819                    # seed of the equilibrium search of reference_count


def parcel_file(subj, run):
    return PARCELS / f"{subj}__{run}__{ATLAS}.npy"


def ev_dir(subj, run):
    return PARCELS / f"{subj}__{run}__EVs"


def available_runs():
    """(subject, run) pairs with a parcel series and EV files, subjects in sorted order, runs LR then RL."""
    subs = sorted({p.name.split("__")[0] for p in PARCELS.glob(f"*__*__{ATLAS}.npy")})
    return [(s, r) for s in subs for r in RUNS if parcel_file(s, r).exists() and ev_dir(s, r).is_dir()]


def read_blocks(subj, run):
    """Per load: list of (onset, duration) task blocks from the EV files, in onset order."""
    out = {c: [] for c in CONDS}
    for c in CONDS:
        for cat in CATS:
            rows = np.loadtxt(ev_dir(subj, run) / f"{c}_{cat}.txt", ndmin=2)
            for onset, dur, _amp in rows:
                out[c].append((float(onset), float(dur)))
        out[c].sort()
    return out


def tr_labels(blocks):
    """Label of each of the 405 volumes: 0 = 0-back, 1 = 2-back, 2 = fixation / other."""
    mid = (np.arange(N_VOLS) + 0.5) * TR
    lab = np.full(N_VOLS, 2, np.int8)
    for ci, c in enumerate(CONDS):
        for onset, dur in blocks[c]:
            lab[(mid >= onset + HRF_SHIFT_S)
                & (mid < onset + HRF_SHIFT_S + dur)] = ci
    return lab


def block_ids(blocks, cond_idx, lab):
    """For the volumes of one load, the index of its (onset-ordered) block; -1 elsewhere."""
    c = CONDS[cond_idx]
    mid = (np.arange(N_VOLS) + 0.5) * TR
    bid = np.full(N_VOLS, -1, np.int32)
    for i, (onset, dur) in enumerate(blocks[c]):
        bid[(mid >= onset + HRF_SHIFT_S)
            & (mid < onset + HRF_SHIFT_S + dur)] = i
    bid[lab != cond_idx] = -1
    return bid


def run_states(subj, run):
    """(F, lab, bid0, bid1): robust-z parcel series of the run, volume labels and block indices per load."""
    F = median_iqr_z(np.load(parcel_file(subj, run)).astype(float))
    blocks = read_blocks(subj, run)
    lab = tr_labels(blocks)
    return F, lab, block_ids(blocks, 0, lab), block_ids(blocks, 1, lab)


def reference_count(Z, lab, bid0, bid1):
    """Equilibria of the drift field from one VAR(1) regime per load (increments within blocks, dt = TR):
    n_equilibria, n_stable, n_unstable."""
    regimes = []
    for ci, bid in ((0, bid0), (1, bid1)):
        segs = [Z[(lab == ci) & (bid == i)] for i in range(bid.max() + 1)]
        segs = [s for s in segs if len(s) >= 2]
        if not segs:
            return dict(error="no segments")
        regimes.append(fit_linear_regime(segs, CONDS[ci], dt=TR))
    eqs = find_equilibria(regimes, softness=1.0, n_starts=200, seed=SEED)
    return dict(n_equilibria=len(eqs),
                n_stable=int(sum(e["stable"] for e in eqs)),
                n_unstable=int(sum(not e["stable"] for e in eqs)))
