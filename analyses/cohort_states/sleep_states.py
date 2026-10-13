# -*- coding: utf-8 -*-
"""Sleep-EDF Expanded: per-night state series of the scored stage groups (Wake, NREM, REM).

State construction (one night; cassette SC* and telemetry ST* recordings)
  signal   EEG Fpz-Cz and Pz-Oz (the first two EEG channels if these are absent), volts, 100 Hz
  stages   expert hypnogram: W -> Wake, 1-4 -> NREM, R -> REM (movement and unscored epochs dropped)
  features log10 band power (delta 0.5-4, theta 4-8, alpha 8-12, beta 12-30 Hz) of 2-s Hann windows moved in 1-s
           steps within each scored hypnogram interval; windows with log10 total power <= -20 (flat line) dropped
  space    robust z (median / IQR) + PCA to d = 4, both fitted on all retained windows of the night pooled over
           the stage groups (states_lib.RefSpace, random_state 20260728)

Input: NDIF_SLEEPEDF_RAW, one directory with the *-PSG.edf recordings and their *-Hypnogram.edf files.
Used by count_cache.py (sleep arm: 197 nights = 153 cassette + 44 telemetry).  No computation is run on import.
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
from states_lib import short_time_band_power, RefSpace             # noqa: E402

DATA_DIR = Path(paths.path("NDIF_SLEEPEDF_RAW"))
BANDS = {"delta": (0.5, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 12.0), "beta": (12.0, 30.0)}
D_STATE = 4
WIN_S = 2.0
STEP_S = 1.0
STAGE_MAP = {"W": "Wake", "1": "NREM", "2": "NREM", "3": "NREM", "4": "NREM", "R": "REM"}
EEG_CHANNELS = ["EEG Fpz-Cz", "EEG Pz-Oz"]
FLAT_LOG10_TOTAL = -20.0


def discover_nights() -> dict:
    """Night ID (first six characters of the PSG file name, e.g. SC4001) -> (psg_path, hypnogram_path)."""
    nights = {}
    for psg in sorted(DATA_DIR.glob("*-PSG.edf")):
        night = psg.name[:6]
        hyps = sorted(DATA_DIR.glob(f"{night}*-Hypnogram.edf"))
        if not hyps:
            print(f"[discover] {night}: PSG present but no hypnogram -> skip", flush=True)
            continue
        nights[night] = (psg, hyps[0])
    return nights


def load_eeg(psg_path: Path, hyp_path: Path):
    """(eeg (2, n), sfreq, channels, epochs [(onset_s, dur_s, stage), ...] in onset order)."""
    import mne
    raw = mne.io.read_raw_edf(str(psg_path), preload=True, verbose="error")
    sfreq = raw.info["sfreq"]
    chans = [c for c in EEG_CHANNELS if c in raw.ch_names]
    if len(chans) < 2:
        chans = [c for c in raw.ch_names if "EEG" in c.upper()][:2]
    if len(chans) < 2:
        raise RuntimeError(f"could not find 2 EEG channels in {raw.ch_names}")
    picks = [raw.ch_names.index(c) for c in chans]
    eeg = raw.get_data(picks=picks).astype(np.float64)

    ann = mne.read_annotations(str(hyp_path))
    epochs = []
    for onset, dur, desc in zip(ann.onset, ann.duration, ann.description):
        if "Sleep stage" not in desc:
            continue
        s = desc.split()[-1].strip()
        if s in STAGE_MAP:
            epochs.append((float(onset), float(dur), s))
    epochs.sort(key=lambda x: x[0])
    return eeg, sfreq, chans, epochs


def stage_features(eeg, sfreq, epochs):
    """Log band-power windows per stage group {Wake, NREM, REM} (groups without windows omitted)."""
    groups = {"Wake": [], "NREM": [], "REM": []}
    n_s = int(WIN_S * sfreq)
    n_flat = 0
    for onset, dur, s in epochs:
        group = STAGE_MAP[s]
        i0 = int(round(onset * sfreq))
        i1 = int(round((onset + dur) * sfreq))
        i1 = min(i1, eeg.shape[1])
        seg = eeg[:, i0:i1]
        if seg.shape[1] < n_s:
            continue
        P = short_time_band_power(seg, sfreq, WIN_S, STEP_S, BANDS)
        if len(P) == 0:
            continue
        total_power = P.sum(axis=(1, 2))
        keep = np.log10(np.maximum(total_power, 1e-30)) > FLAT_LOG10_TOTAL
        n_flat += int((~keep).sum())
        P = P[keep]
        if len(P) == 0:
            continue
        F = np.log10(np.maximum(P, 1e-30)).reshape(P.shape[0], -1)
        groups[group].append(F)
    if n_flat:
        print(f"    [flat-line] dropped {n_flat} windows", flush=True)
    return {g: np.concatenate(v, axis=0) for g, v in groups.items() if v}


def state_space(features):
    """{stage group: (n_windows, 4) states}: RefSpace(4) fitted on the windows of all groups pooled
    (in the order Wake, NREM, REM)."""
    sp = RefSpace(D_STATE).fit(np.vstack(list(features.values())))
    return {k: sp.transform(v) for k, v in features.items()}
