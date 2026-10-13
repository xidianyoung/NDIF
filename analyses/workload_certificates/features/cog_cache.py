# -*- coding: utf-8 -*-
"""COG-BCI feature cache: the zipped participant archives of the COG-BCI release -> short-time log10 band power.

Features: theta 4-8, alpha 8-12, beta 12-30, gamma 30-45 Hz on 62 channels (248 features), 1.0-s Hann windows, a
50 %-overlap `dist` series for every moment and a non-overlapping `dyn` series for every propagator fit.

Preprocessing (the archives are raw: common reference, highpass 0 Hz, multi-millivolt DC offsets), applied
identically to every record of every condition: 62 common EEG channels (the 63 labels shared by all 29
participants, minus ECG1; Cz dropped for sub-10..29 so one montage carries everyone) -> 1.0 Hz FIR high-pass ->
average reference -> microvolts.  Window rejection: per-channel peak-to-peak > 500 uV, or a flat channel
(< 0.01 uV); the rejection mask is stored per window (`keep`), together with the segment index (`seg`, records are
split at n-back block markers and at boundary annotations).

All 29 released participants are cached; the duplicate exclusion of the public release (sub-27 duplicates sub-17;
sub-28 session 1 duplicates sub-25 session 1) is applied when the cache is read (cog_features.py).

Input    NDIF_COGBCI_RAW: sub-XX.zip (COG-BCI release, Zenodo https://doi.org/10.5281/zenodo.6874128)
Output   NDIF_COGBCI_CACHE (paths.cache("cogbci"), default analyses/workload_certificates/features/cog_feature_cache):
         one sub-XX.npz per participant with keys "<session>|<record>|<dist|dyn>|F" (float32, windows x 248),
         "...|keep" (bool), "...|seg" (int16) and "meta" (JSON: channels, bands, window settings, per-record status);
         existing files are skipped
Scratch  each archive is extracted to NDIF_WORK/cogbci_extract/<sub> and removed afterwards
Read by  cog_features.py (and through it cog_state.py, benchmark_06_load_cohorts.py and the scripts in
         analyses/rate_identifiability and analyses/task_location that use the COG-BCI cohort)
Runtime  about 2 min per participant (one desktop core)
Usage    python analyses/workload_certificates/features/cog_cache.py [--subs 1,2,...,29]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
from scipy import signal as sps

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from analyses.common import paths                                          # noqa: E402

CACHE = Path(paths.cache("cogbci"))
ZIPDIR = Path(paths.path("NDIF_COGBCI_RAW"))
SFREQ = 500.0
WIN, STEP_DIST, STEP_DYN = 1.0, 0.5, 1.0
BANDS = {"theta": (4.0, 8.0), "alpha": (8.0, 12.0), "beta": (12.0, 30.0), "gamma": (30.0, 45.0)}
HP = 1.0
PTP_MAX_UV = 500.0
PTP_FLAT_UV = 0.01

# the 63 labels common to all 29 participants, minus ECG1 -> 62 EEG channels, sub-01 header order
CHANNELS = ['Fp1', 'Fz', 'F3', 'F7', 'FT9', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'Pz', 'P3',
            'P7', 'O1', 'Oz', 'O2', 'P4', 'P8', 'TP10', 'CP6', 'CP2', 'FCz', 'C4', 'T8', 'FT10',
            'FC6', 'FC2', 'F4', 'F8', 'Fp2', 'AF7', 'AF3', 'AFz', 'F1', 'F5', 'FT7', 'FC3', 'C1',
            'C5', 'TP7', 'CP3', 'P1', 'P5', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'P6', 'P2', 'CPz',
            'CP4', 'TP8', 'C6', 'C2', 'FC4', 'FT8', 'F6', 'AF8', 'AF4', 'F2']
RECORDS = ["RS_Beg_EO", "RS_Beg_EC", "RS_End_EO", "RS_End_Ec",
           "zeroBACK", "oneBACK", "twoBACK",
           "MATBeasy", "MATBmed", "MATBdiff", "Flanker", "PVT"]
NBACK_BLOCK = {"zeroBACK": ("6011", "6012"), "oneBACK": ("6111", "6112"), "twoBACK": ("6211", "6212")}
SESSIONS = ["ses-S1", "ses-S2", "ses-S3"]


def log(m):
    print(time.strftime("[%H:%M:%S] ") + m, flush=True)



# ------------------------------------------------------------------ features
def short_time_band_power(x, sfreq, win_s, step_s, bands):
    """Band power per window: Hann spectrogram (density), summed over each band's bins times df."""
    nperseg = int(round(win_s * sfreq)); nstep = int(round(step_s * sfreq))
    if x.shape[1] < nperseg:
        return np.zeros((0, x.shape[0], len(bands)))
    f, t, S = sps.spectrogram(x, fs=sfreq, window="hann", nperseg=nperseg, noverlap=nperseg - nstep,
                              detrend="constant", scaling="density", mode="psd", axis=-1)
    df = f[1] - f[0]
    out = np.empty((S.shape[-1], x.shape[0], len(bands)))
    for bi, (lo, hi) in enumerate(bands.values()):
        m = (f >= lo) & (f < hi)
        out[:, :, bi] = (S[:, m, :].sum(axis=1) * df).T
    return out


def log_features(P):
    return np.log10(np.maximum(P, 1e-30)).reshape(P.shape[0], -1)


def window_ptp(x, sfreq, win_s, step_s):
    """Per-window min/max per channel -> (n_win,) max and min peak-to-peak across channels."""
    nperseg = int(round(win_s * sfreq)); nstep = int(round(step_s * sfreq))
    n = x.shape[1]
    if n < nperseg:
        return np.zeros(0), np.zeros(0)
    nw = 1 + (n - nperseg) // nstep
    mx = np.empty(nw); mn = np.empty(nw)
    for i in range(nw):
        seg = x[:, i * nstep: i * nstep + nperseg]
        p = seg.max(axis=1) - seg.min(axis=1)
        mx[i] = p.max(); mn[i] = p.min()
    return mx, mn


# ------------------------------------------------------------------ io
def read_record(path):
    import mne
    mne.set_log_level("ERROR")
    raw = mne.io.read_raw_eeglab(str(path), preload=True)
    missing = [c for c in CHANNELS if c not in raw.ch_names]
    if missing:
        raise RuntimeError(f"missing channels {missing} in {path}")
    raw.pick(CHANNELS)
    raw.filter(l_freq=HP, h_freq=None, fir_design="firwin", picks="all", verbose="ERROR")
    x = raw.get_data() * 1e6                      # -> microvolts
    x = x - x.mean(axis=0, keepdims=True)         # average reference over the 62 kept channels
    ann = raw.annotations
    return x, ann


def segments_for(record, x, ann):
    """(start, stop) sample pairs, boundary annotations splitting them."""
    n = x.shape[1]
    if record in NBACK_BLOCK:
        s_code, e_code = NBACK_BLOCK[record]
        onsets = [(float(o), str(d)) for o, d in zip(ann.onset, ann.description)]
        segs, open_at = [], None
        for o, d in onsets:
            if d == s_code:
                open_at = o
            elif d == e_code and open_at is not None:
                segs.append((int(round(open_at * SFREQ)), int(round(o * SFREQ))))
                open_at = None
        if not segs:                                # fall back to the whole record, logged upstream
            segs = [(0, n)]
    else:
        segs = [(0, n)]
    # split at boundary annotations
    bnds = sorted(int(round(float(o) * SFREQ)) for o, d in zip(ann.onset, ann.description)
                  if str(d) == "boundary")
    out = []
    for a, b in segs:
        a = max(0, a); b = min(n, b)
        cuts = [a] + [c for c in bnds if a < c < b] + [b]
        for i in range(len(cuts) - 1):
            if cuts[i + 1] - cuts[i] >= int(round(WIN * SFREQ)):
                out.append((cuts[i], cuts[i + 1]))
    return out


def process_record(path):
    x, ann = read_record(path)
    segs = segments_for(path.stem, x, ann)
    res = {}
    for series, step in (("dist", STEP_DIST), ("dyn", STEP_DYN)):
        Fs, keeps, segid = [], [], []
        for si, (a, b) in enumerate(segs):
            seg = x[:, a:b]
            F = log_features(short_time_band_power(seg, SFREQ, WIN, step, BANDS))
            mx, mn = window_ptp(seg, SFREQ, WIN, step)
            k = min(len(F), len(mx))
            if k == 0:
                continue
            Fs.append(F[:k].astype(np.float32))
            keeps.append((mx[:k] <= PTP_MAX_UV) & (mn[:k] >= PTP_FLAT_UV))
            segid.append(np.full(k, si, dtype=np.int16))
        if Fs:
            res[series] = (np.concatenate(Fs, 0), np.concatenate(keeps, 0), np.concatenate(segid, 0))
        else:
            res[series] = (np.zeros((0, len(CHANNELS) * 4), np.float32),
                           np.zeros(0, bool), np.zeros(0, np.int16))
    return res, len(segs), x.shape[1] / SFREQ


def extract_subject(sub, workdir):
    zp = ZIPDIR / f"{sub}.zip"
    z = zipfile.ZipFile(zp)
    dest = workdir / sub
    if dest.exists():
        shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(parents=True, exist_ok=True)
    members = [n for n in z.namelist() if "/eeg/" in n and (n.endswith(".set") or n.endswith(".fdt"))]
    mapping = {}
    for n in members:
        parts = n.split("/")
        ses = [p for p in parts if p.startswith("ses-")]
        if not ses:
            continue
        ses = ses[0]
        fname = parts[-1]
        tgt = dest / ses / fname
        tgt.parent.mkdir(parents=True, exist_ok=True)
        with z.open(n) as src, open(tgt, "wb") as fh:
            shutil.copyfileobj(src, fh, 1 << 20)
        mapping.setdefault(ses, []).append(fname)
    z.close()
    return dest, mapping


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subs", default=",".join(str(i) for i in range(1, 30)))
    ap.add_argument("--workdir", default=paths.work("cogbci_extract"))
    a = ap.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    workdir = Path(a.workdir); workdir.mkdir(parents=True, exist_ok=True)
    subs = [f"sub-{int(s):02d}" for s in a.subs.split(",")]
    for sub in subs:
        t0 = time.time()
        npz_path = CACHE / f"{sub}.npz"
        if npz_path.exists():
            log(f"{sub} cached, skip"); continue
        dest, mapping = extract_subject(sub, workdir)
        log(f"{sub} extracted {sum(len(v) for v in mapping.values())} files in {time.time()-t0:.0f}s")
        store, meta = {}, {"sub": sub, "channels": CHANNELS, "bands": BANDS, "win_s": WIN,
                           "step_dist": STEP_DIST, "step_dyn": STEP_DYN, "hp_hz": HP,
                           "ptp_max_uv": PTP_MAX_UV, "records": {}}
        for ses in SESSIONS:
            for rec in RECORDS:
                p = dest / ses / f"{rec}.set"
                if not p.exists():
                    meta["records"][f"{ses}|{rec}"] = {"status": "missing"}
                    log(f"  !! {sub} {ses} {rec} missing")
                    continue
                try:
                    res, nseg, dur = process_record(p)
                except Exception as e:
                    meta["records"][f"{ses}|{rec}"] = {"status": f"error:{type(e).__name__}:{e}"}
                    log(f"  !! {sub} {ses} {rec} ERROR {type(e).__name__} {e}")
                    continue
                for series, (F, k, sid) in res.items():
                    store[f"{ses}|{rec}|{series}|F"] = F
                    store[f"{ses}|{rec}|{series}|keep"] = k
                    store[f"{ses}|{rec}|{series}|seg"] = sid
                meta["records"][f"{ses}|{rec}"] = {
                    "status": "ok", "dur_s": float(dur), "n_seg": int(nseg),
                    "n_dist": int(len(res["dist"][0])), "n_dyn": int(len(res["dyn"][0])),
                    "keep_frac_dist": float(res["dist"][1].mean()) if len(res["dist"][1]) else 0.0,
                    "keep_frac_dyn": float(res["dyn"][1].mean()) if len(res["dyn"][1]) else 0.0}
            log(f"  {sub} {ses} done ({time.time()-t0:.0f}s)")
        store["meta"] = np.array(json.dumps(meta))
        np.savez_compressed(npz_path, **store)
        shutil.rmtree(dest, ignore_errors=True)
        log(f"{sub} DONE {time.time()-t0:.0f}s -> {npz_path.name} "
            f"({npz_path.stat().st_size/1e6:.0f} MB)")


if __name__ == "__main__":
    main()
