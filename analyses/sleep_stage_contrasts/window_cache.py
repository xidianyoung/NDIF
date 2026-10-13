# -*- coding: utf-8 -*-
r"""Sleep-EDF window cache: log10 band power per scored bout at window lengths of 2.0, 1.0 and 0.5 s.

For every Sleep-EDF Expanded recording (153 cassette nights SC*, 44 telemetry nights ST*), the two
EEG channels Fpz-Cz and Pz-Oz are cut into non-overlapping windows (step = window) inside each
scored bout, the power in four bands (delta 0.5-4, theta 4-8, alpha 8-12, beta 12-30 Hz; Hann
window, one-sided density, scipy.signal.spectrogram) is summed per band and log10-transformed,
giving 8 features per window.  The robust z and the PCA of the state space are fitted downstream,
per recording (sleep_common.state_space).

Bout rule: fine hypnogram stages W/1/2/3/4/R (codes Wake, N1, N2, N3, N4, REM); the first and last
Wake bout of a recording are dropped (lights-off segments); 0 < duration <= 3600 s; duration
>= 90 s to be used; the first 20 s of every bout are dropped as the stage-onset transient and the
next 60 s are used, so every bout contributes the same span (30 windows at 2 s, 60 at 1 s, 120 at
0.5 s).  A bout is rejected if any window in its first 80 s is flat (log10 total power <= -20),
so that no lag pair spans a gap; the number of bouts this rejects is recorded in cache_meta.json
(`n_bouts_flatreject`, and `n_bouts_flatreject_kept_by_splicing` for the bouts that a rule
dropping the flat windows and joining the rest would have kept).

Inputs   Sleep-EDF Expanded EDF files and hypnograms in NDIF_SLEEPEDF_RAW
         (<stem>*-PSG.edf, <stem>*-Hypnogram.edf).
Outputs  paths.cache("sleep_states") (NDIF_SLEEP_STATES_CACHE, default analyses/_work/sleep_states):
           <stem>_win2.npz, <stem>_win1.npz, <stem>_win0.5.npz   X (n_bout, n_win, 8) float32,
                stage, onset, dur, win_s, sfreq, n_skip, n_use, bins (FFT bins per band, JSON)
           cache_meta.json   bout counts per recording and window
         The 153 cassette files <stem>_win1.npz are the files of analyses/sleep_feature_cache
         (read by the primary- and G8-configuration sleep drivers).
Used by  stage_contrasts.py (Fig. 4c, Fig. 5b,e, Supplementary Tables 8 and 11),
         random_direction_rates.py, noise_cache.py and ../rate_identifiability.
Runtime  about 1 min with 4 workers (197 recordings); existing files are skipped.

    python analyses/sleep_stage_contrasts/window_cache.py --library both --workers 4
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse                                                       # noqa: E402
import json                                                           # noqa: E402
import sys                                                            # noqa: E402
import time                                                           # noqa: E402
import traceback                                                      # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed     # noqa: E402
from pathlib import Path                                              # noqa: E402

import numpy as np                                                    # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from analyses.common import paths  # noqa: E402

SLEEP = Path(paths.path("NDIF_SLEEPEDF_RAW"))
CACHE = Path(paths.cache("sleep_states"))
CACHE.mkdir(parents=True, exist_ok=True)

BANDS = {"delta": (0.5, 4.0), "theta": (4.0, 8.0),
         "alpha": (8.0, 12.0), "beta": (12.0, 30.0)}
EEG_CHANNELS = ["EEG Fpz-Cz", "EEG Pz-Oz"]
FINE_MAP = {"W": "Wake", "1": "N1", "2": "N2", "3": "N3", "4": "N4", "R": "REM"}
STAGES = ("Wake", "N1", "N2", "N3", "N4", "REM")
MIN_BOUT_S = 90.0
MAX_BOUT_S = 3600.0
SKIP_S = 20.0           # stage-onset transient dropped from every bout
USE_S = 60.0            # span used from every bout
FLAT_LOG10_TOTAL = -20.0
WINDOWS = (2.0, 1.0, 0.5)


def pair_recordings(lib: str):
    prefix = "SC" if lib == "cassette" else "ST"
    out = []
    for p in sorted(SLEEP.glob(f"{prefix}*-PSG.edf")):
        stem = p.name[:6]
        hyps = sorted(SLEEP.glob(f"{stem}*-Hypnogram.edf"))
        if hyps:
            out.append((stem, p, hyps[0]))
    return out


def load_eeg(psg_path, hyp_path):
    """Two EEG channels and the scored bouts (2 of 7 channels read, lazily, so four workers
    fit in memory)."""
    import mne
    raw = mne.io.read_raw_edf(str(psg_path), preload=False, verbose="error")
    sfreq = float(raw.info["sfreq"])
    chans = [c for c in EEG_CHANNELS if c in raw.ch_names]
    if len(chans) < 2:
        chans = [c for c in raw.ch_names if "EEG" in c.upper()][:2]
    if len(chans) < 2:
        raise RuntimeError(f"fewer than two EEG channels: {raw.ch_names}")
    picks = [raw.ch_names.index(c) for c in chans]
    eeg = np.asarray(raw.get_data(picks=picks), dtype=np.float64)
    ann = mne.read_annotations(str(hyp_path))
    epochs = []
    for onset, dur, desc in zip(ann.onset, ann.duration, ann.description):
        if "Sleep stage" not in desc:
            continue
        s = desc.split()[-1].strip()
        if s in FINE_MAP:
            epochs.append((float(onset), float(dur), s))
    epochs.sort(key=lambda x: x[0])
    return eeg, sfreq, chans, epochs


def band_power(x, sfreq, win_s):
    """Non-overlapping short-time band power, (n_win, n_chan, n_band)."""
    from scipy import signal as sps
    nperseg = int(round(win_s * sfreq))
    if x.shape[1] < nperseg:
        return np.zeros((0, x.shape[0], len(BANDS)))
    f, t, S = sps.spectrogram(x, fs=sfreq, window="hann", nperseg=nperseg,
                              noverlap=0, detrend="constant",
                              scaling="density", mode="psd", axis=-1)
    df = f[1] - f[0]
    out = np.empty((S.shape[-1], x.shape[0], len(BANDS)))
    for bi, (lo, hi) in enumerate(BANDS.values()):
        m = (f >= lo) & (f < hi)
        out[:, :, bi] = (S[:, m, :].sum(axis=1) * df).T
    return out


def n_bins_per_band(sfreq, win_s):
    """How many FFT bins each band gets at this window."""
    nperseg = int(round(win_s * sfreq))
    f = np.fft.rfftfreq(nperseg, 1.0 / sfreq)
    return {b: int(((f >= lo) & (f < hi)).sum()) for b, (lo, hi) in BANDS.items()}


def usable_bouts(epochs):
    idx_wake = [i for i, (o, d, s) in enumerate(epochs) if FINE_MAP.get(s) == "Wake"]
    drop = {idx_wake[0], idx_wake[-1]} if idx_wake else set()
    fit = []
    for i, (onset, dur, s) in enumerate(epochs):
        g = FINE_MAP.get(s)
        if g is None or i in drop or not (0 < dur <= MAX_BOUT_S):
            continue
        if dur >= MIN_BOUT_S:
            fit.append((onset, dur, g))
    return fit


def run_one(args):
    stem, psg, hyp = args
    t0 = time.time()
    outs = {w: CACHE / f"{stem}_win{w:g}.npz" for w in WINDOWS}
    if all(p.exists() for p in outs.values()):
        return dict(stem=stem, cached=True)

    eeg, sfreq, chans, epochs = load_eeg(psg, hyp)
    if chans[:2] != list(EEG_CHANNELS):
        return dict(stem=stem, error=f"channel mismatch: {chans}")
    bouts = usable_bouts(epochs)

    meta = dict(stem=stem, subject=stem[:5], sfreq=sfreq,
                n_bouts_scored=len(bouts), per_window={})
    for win_s in WINDOWS:
        n_skip = int(round(SKIP_S / win_s))
        n_use = int(round(USE_S / win_s))
        blocks, stages, onsets, durs = [], [], [], []
        n_flatreject = n_flat_splice_would_keep = n_short = 0
        for onset, dur, g in bouts:
            i0 = int(round(onset * sfreq))
            i1 = min(int(round((onset + dur) * sfreq)), eeg.shape[1])
            seg = eeg[:, i0:i1]
            P = band_power(seg, sfreq, win_s)
            if len(P) < n_skip + n_use:
                n_short += 1
                continue
            ok = np.log10(np.maximum(P.sum(axis=(1, 2)), 1e-30)) > FLAT_LOG10_TOTAL
            if not ok[: n_skip + n_use].all():
                n_flatreject += 1
                if int(ok.sum()) >= n_skip + n_use:
                    n_flat_splice_would_keep += 1
                continue
            F = np.log10(np.maximum(P, 1e-30)).reshape(P.shape[0], -1)
            blk = F[n_skip:n_skip + n_use]
            if not np.isfinite(blk).all():
                continue
            blocks.append(blk.astype(np.float32))
            stages.append(g); onsets.append(onset); durs.append(dur)
        if blocks:
            np.savez_compressed(
                outs[win_s],
                X=np.stack(blocks),                       # (n_bout, n_use, 8)
                stage=np.array(stages), onset=np.array(onsets, float),
                dur=np.array(durs, float), win_s=win_s, sfreq=sfreq,
                n_skip=n_skip, n_use=n_use,
                bins=json.dumps(n_bins_per_band(sfreq, win_s)))
        else:
            np.savez_compressed(outs[win_s], X=np.zeros((0, 0, 8), np.float32),
                                stage=np.array([]), onset=np.array([]),
                                dur=np.array([]), win_s=win_s, sfreq=sfreq,
                                n_skip=n_skip, n_use=n_use,
                                bins=json.dumps(n_bins_per_band(sfreq, win_s)))
        meta["per_window"][f"{win_s:g}"] = dict(
            n_bouts_used=len(blocks), n_bouts_short=n_short,
            n_bouts_flatreject=n_flatreject,
            n_bouts_flatreject_kept_by_splicing=n_flat_splice_would_keep,
            bins=n_bins_per_band(sfreq, win_s))
    meta["seconds"] = round(time.time() - t0, 1)
    return meta


def _worker(a):
    try:
        return run_one(a)
    except Exception:  # noqa: BLE001
        return dict(stem=a[0], error=traceback.format_exc(limit=3))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--library", choices=["cassette", "telemetry", "both"], default="both")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None,
                    help="only the first N recordings (a check run)")
    a = ap.parse_args()

    libs = ["cassette", "telemetry"] if a.library == "both" else [a.library]
    recs = []
    for lib in libs:
        recs += pair_recordings(lib)
    if a.limit:
        recs = recs[: a.limit]
    print(f"{len(recs)} recordings; windows {WINDOWS}; {a.workers} workers", flush=True)

    metas, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(_worker, r): r[0] for r in recs}
        for k, f in enumerate(as_completed(futs), 1):
            m = f.result()
            metas.append(m)
            if "error" in m:
                print(f"  [{k}/{len(recs)}] {m['stem']} ERROR "
                      f"{m['error'].splitlines()[-1][:80]}", flush=True)
            elif k % 20 == 0 or k == len(recs):
                print(f"  {k}/{len(recs)}  {(time.time() - t0) / 60:.1f} min", flush=True)
    metas.sort(key=lambda m: m["stem"])
    (CACHE / "cache_meta.json").write_text(json.dumps(metas, indent=1), encoding="utf-8",
                                          newline="\n")
    fresh = [m for m in metas if "per_window" in m]
    if fresh:
        for w in WINDOWS:
            k = f"{w:g}"
            used = sum(m["per_window"][k]["n_bouts_used"] for m in fresh)
            rej = sum(m["per_window"][k]["n_bouts_flatreject"] for m in fresh)
            dev = sum(m["per_window"][k]["n_bouts_flatreject_kept_by_splicing"]
                      for m in fresh)
            print(f"  win {k}s: {used} bouts used, {rej} flat-rejected "
                  f"({dev} of them kept by a splicing rule)")
        print("  bins per band:", fresh[0]["per_window"]["2"]["bins"],
              fresh[0]["per_window"]["1"]["bins"], fresh[0]["per_window"]["0.5"]["bins"])
    print(f"done in {(time.time() - t0) / 60:.1f} min -> {CACHE}")


if __name__ == "__main__":
    main()
