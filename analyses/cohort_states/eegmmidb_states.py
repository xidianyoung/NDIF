# -*- coding: utf-8 -*-
"""PhysioNet EEG Motor Movement/Imagery (eegmmidb): annotated blocks and their state space.

State construction (one participant)
  blocks   every annotated T0 (rest), T1 or T2 (task) interval of at least 1 s in the twelve task runs
           R03-R14, read from the EDF files (64 EEG channels, 160 Hz, microvolts)
  features log10 band power of 1-s windows (theta 4-8, alpha 8-12, beta 12-30, gamma 30-45 Hz) per channel, in a
           dist series (windows every 0.5 s) and a dyn series (non-overlapping, every 1.0 s; dt = 1 s)
  space    robust z (median / IQR) + PCA to d = 8, both fitted on the dist windows of the rest blocks
           (states_lib.RefSpace, random_state 20260728)
  contrast rest (T0) vs task (T1 or T2); a participant needs at least 20 blocks and 10 of each class

Input: NDIF_EEGMMIDB_RAW (one directory per participant, S001/S001R03.edf ...).
Used by count_cache.py (eegmmidb arm).  No computation is run on import; the per-participant states are built
by count_cache.py (about 2 min for the 104 participants with 6 workers on one desktop).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))
from analyses.common import paths                                  # noqa: E402
from states_lib import short_time_band_power, log_features, RefSpace  # noqa: E402

DATA = Path(paths.path("NDIF_EEGMMIDB_RAW"))
SFREQ = 160.0
WIN, STEP_DIST, STEP_DYN = 1.0, 0.5, 1.0
D = 8
BANDS = {"theta": (4, 8), "alpha": (8, 12), "beta": (12, 30), "gamma": (30, 45)}
TASK_RUNS = list(range(3, 15))


def load_run(path):
    """EEG of one run (microvolts) and its T0/T1/T2 blocks as (desc, first sample, end sample)."""
    import mne
    raw = mne.io.read_raw_edf(str(path), preload=True, verbose="ERROR")
    raw.pick("eeg")
    x = raw.get_data() * 1e6  # volts -> microvolts
    blocks = []
    for onset, dur, desc in zip(raw.annotations.onset, raw.annotations.duration, raw.annotations.description):
        if desc not in ("T0", "T1", "T2"):
            continue
        i0, i1 = int(round(onset * SFREQ)), int(round((onset + dur) * SFREQ))
        if i1 - i0 >= int(WIN * SFREQ):
            blocks.append((desc, i0, min(i1, x.shape[1])))
    return x, blocks


def subject_blocks(sdir, runs):
    """List of dicts, one per annotated block, with its dist and dyn log band-power features."""
    out = []
    for r in runs:
        f = sdir / f"{sdir.name}R{r:02d}.edf"
        if not f.exists():
            continue
        x, blocks = load_run(f)
        for bi, (desc, i0, i1) in enumerate(blocks):
            seg = x[:, i0:i1]
            Fd = log_features(short_time_band_power(seg, SFREQ, WIN, STEP_DIST, BANDS))
            Fy = log_features(short_time_band_power(seg, SFREQ, WIN, STEP_DYN, BANDS))
            if len(Fd) and len(Fy):
                out.append(dict(run=r, block=bi, desc=desc, F_dist=Fd, F_dyn=Fy))
    return out


def subject_states(sub):
    """The rest-vs-task states of one participant, or None with fewer than 20 blocks or 10 of a class.

    Returns (blocks, labels, space): the selected blocks of the task runs in file order, their labels
    (0 = rest, 1 = task) and the fitted RefSpace.
    """
    sdir = DATA / sub
    blocks = subject_blocks(sdir, TASK_RUNS)
    sel = [b for b in blocks if b["run"] in set(TASK_RUNS)]
    lab = np.array([0 if b["desc"] == "T0" else 1 for b in sel])
    if len(sel) < 20 or lab.sum() < 10 or (lab == 0).sum() < 10:
        return None
    F_ref = np.vstack([b["F_dist"] for b, l in zip(sel, lab) if l == 0])
    sp = RefSpace(D).fit(F_ref)
    return sel, lab, sp
