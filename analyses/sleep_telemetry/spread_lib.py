# -*- coding: utf-8 -*-
r"""Shared code of the 2-s bout-spread analyses: Sleep-EDF loading, the per-bout band-power blocks,
the spread and lag-1 estimators, night-stage cells, participant-level contrasts with a size-matched
within-night label permutation null, the resampling power of a paired contrast and the telemetry
drug assignment.

Imported by bout_spread_rows.py, crossover.py and drug_power.py.  The state space (robust z + PCA,
d = 4) is ../sleep_stage_contrasts/sleep_common.build_state_space.

THE 2-S BOUT CHAIN
------------------
Two EEG channels (Fpz-Cz, Pz-Oz), 2-s Hann windows at zero overlap, power in delta 0.5-4, theta
4-8, alpha 8-12, beta 12-30 Hz, log10 (8 features per window).  Bouts: fine hypnogram stages
(Wake, N1-N4, REM; the R&K stages Wake, S1-S4, REM), the first and last Wake bout dropped, 90 to
3600 s long; flat windows (log10 total power <= -20) are dropped, then the first 10 windows of
the bout are skipped and exactly the next 30 used, so every bout contributes 60 s; a stage needs
at least 5 such bouts.  Along every direction from a stage's mean to another stage's mean, the
spread is the standard deviation of the bout-centred projections; a direction is kept if its
pooled lag-1 coefficient is <= 0.6 and its odd/even split-half spread ratio is within 0.7-1.4.

Rows directory: analyses.common.paths.cache("sleep_bout_rows") (NDIF_SLEEP_BOUT_ROWS_CACHE,
default analyses/_work/sleep_bout_rows), files sleep_alternative_processing_rows.json (cassette)
and telemetry_crossover_rows.json (telemetry), the names under which the Source Data hold them
(analysis_inputs/telemetry/stored_outputs/).
"""
from __future__ import annotations

import collections
import json
import math
import struct
import sys
from pathlib import Path

import numpy as np
from scipy import stats as sst

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "analyses" / "sleep_stage_contrasts"))
from analyses.common import paths  # noqa: E402
from sleep_common import build_state_space  # noqa: E402,F401  (re-exported for the scripts)

SLEEP = Path(paths.path("NDIF_SLEEPEDF_RAW"))
ROWS = Path(paths.cache("sleep_bout_rows"))
ROWS_FILE = {"cassette": "sleep_alternative_processing_rows.json",
             "telemetry": "telemetry_crossover_rows.json"}

# ---- band power and bouts -------------------------------------------------
BANDS = {"delta": (0.5, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 12.0), "beta": (12.0, 30.0)}
EEG_CHANNELS = ["EEG Fpz-Cz", "EEG Pz-Oz"]
D_STATE = 4
WIN_S = 2.0
STEP_S = WIN_S                 # zero overlap: no MA correlation in a lagged quantity
MIN_BOUT_S = 90.0
MAX_BOUT_S = 3600.0
SKIP_WINDOWS = 10              # 20 s of stage-onset transient
USE_WINDOWS = 30               # 60 s, identical for every bout
MIN_BOUTS = 5
RHO_MAX = 0.6                  # above this the projection is drifting, not relaxing
SPLIT_LO, SPLIT_HI = 0.7, 1.4  # odd/even split-half spread ratio must land inside
FLAT_LOG10_TOTAL = -20.0
FINE_MAP = {"W": "Wake", "1": "N1", "2": "N2", "3": "N3", "4": "N4", "R": "REM"}
DEPTH = {"N1": 1, "N2": 2, "N3": 3, "N4": 4}
AROUSAL = {"Wake": 0, "N1": 1, "N2": 2, "N3": 3, "N4": 4, "REM": 1.5}
BOUT_STAGES = tuple(sorted(set(FINE_MAP.values())))     # processing order of the stages

# ---- rate estimators and contrasts ------------------------------------------
DT = float(WIN_S)
ACF_MAXLAG = 5
ACF_FLOOR = 0.05
SEED = 20260921
NW = 30                        # windows per bout
STAGES = ["Wake", "N1", "N2", "N3", "N4", "REM"]
CONTRASTS = [("Wake", "N1"), ("N1", "N2"), ("N2", "N3"), ("N3", "N4")]
N_BOOT = 4000

# ---- resampling power --------------------------------------------------------
N_RESAMPLE = 4000
FOLD_GRID = [1.02, 1.03, 1.05, 1.075, 1.10, 1.125, 1.15, 1.20, 1.25, 1.35, 1.50, 1.75, 2.00]


# =========================================================================== EEG
def pair_recordings(lib: str) -> list[tuple[str, Path, Path]]:
    """(stem, psg, hypnogram) pairs. PSG and hypnogram suffixes differ."""
    prefix = "SC" if lib == "cassette" else "ST"
    psgs = sorted(SLEEP.glob(f"{prefix}*-PSG.edf"))
    out = []
    for p in psgs:
        stem = p.name[:6]
        hyps = sorted(SLEEP.glob(f"{stem}*-Hypnogram.edf"))
        if hyps:
            out.append((stem, p, hyps[0]))
    return out


def load_real_eeg(psg_path: Path, hyp_path: Path):
    """(eeg (2,n), sfreq, chans, epochs [(onset_s, dur_s, stage_str), ...]).

    Scored epochs are kept if their stage code is one of W, 1, 2, 3, 4, R."""
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
        if s in FINE_MAP:
            epochs.append((float(onset), float(dur), s))
    epochs.sort(key=lambda x: x[0])
    return eeg, sfreq, chans, epochs


def short_time_band_power(x, sfreq, win_s, step_s, bands=None):
    """Short-time band power, (n_win, n_chan, n_band)."""
    from scipy import signal as sps
    bands = bands or BANDS
    nperseg = int(round(win_s * sfreq))
    nstep = int(round(step_s * sfreq))
    noverlap = nperseg - nstep
    x = np.asarray(x, dtype=float)  # ensure float64 for scipy
    if x.shape[1] < nperseg:
        return np.zeros((0, x.shape[0], len(bands)), dtype=float)

    f, t, S = sps.spectrogram(x, fs=sfreq, window="hann", nperseg=nperseg,
                              noverlap=noverlap, detrend="constant",
                              scaling="density", mode="psd", axis=-1)
    df = f[1] - f[0]
    out = np.empty((S.shape[-1], x.shape[0], len(bands)), dtype=float)
    for bi, (lo, hi) in enumerate(bands.values()):
        m = (f >= lo) & (f < hi)
        out[:, :, bi] = (S[:, m, :].sum(axis=1) * df).T
    return out


def usable_bouts_fine(epochs):
    """Bout hygiene with the fine stage map: first and last Wake bout dropped, 0 < dur <= 3600 s,
    dur >= 90 s to be used."""
    idx_wake = [i for i, (o, d, s) in enumerate(epochs) if FINE_MAP.get(s) == "Wake"]
    drop = {idx_wake[0], idx_wake[-1]} if idx_wake else set()
    fit = {g: [] for g in BOUT_STAGES}
    for i, (onset, dur, s) in enumerate(epochs):
        g = FINE_MAP.get(s)
        if g is None or i in drop or not (0 < dur <= MAX_BOUT_S):
            continue
        if dur >= MIN_BOUT_S:
            fit[g].append((onset, dur))
    return fit


def bout_features(eeg, sfreq, onset, dur):
    """Exactly USE_WINDOWS windows from one bout, or None.

    Band power of the windows strictly inside the annotation interval, flat-line
    windows rejected, then truncated so every bout contributes an identical number
    of samples (and the untruncated remainder after the skipped windows).
    """
    i0 = int(round(onset * sfreq))
    i1 = min(int(round((onset + dur) * sfreq)), eeg.shape[1])
    seg = eeg[:, i0:i1]
    if seg.shape[1] < int(WIN_S * sfreq):
        return None
    P = short_time_band_power(seg, sfreq, WIN_S, STEP_S, BANDS)
    if len(P) == 0:
        return None
    keep = np.log10(np.maximum(P.sum(axis=(1, 2)), 1e-30)) > FLAT_LOG10_TOTAL
    P = P[keep]
    if len(P) < SKIP_WINDOWS + USE_WINDOWS:
        return None
    F = np.log10(np.maximum(P, 1e-30)).reshape(P.shape[0], -1)
    trunc = F[SKIP_WINDOWS:SKIP_WINDOWS + USE_WINDOWS]
    full = F[SKIP_WINDOWS:]
    if not (np.isfinite(trunc).all() and np.isfinite(full).all()):
        return None
    return trunc, full


def spread(blocks) -> float:
    """Stationary spread, each block centred on its own mean (equal effort per bout)."""
    n = sum(len(b) for b in blocks)
    if n == 0:
        return float("nan")
    ss = sum(float(((b - b.mean()) ** 2).sum()) for b in blocks)
    return float(np.sqrt(ss / n))


def ar1(blocks) -> float:
    """Lag-1 coefficient pooled across blocks; no pair spans a bout boundary."""
    num = den = 0.0
    for b in blocks:
        if len(b) < 3:
            continue
        c = b - b.mean()
        num += float(c[:-1] @ c[1:])
        den += float(c[:-1] @ c[:-1])
    return float(num / den) if den > 0 else float("nan")


# ===================================================================== estimators
def _lam_from_r(r: np.ndarray) -> tuple:
    """(lambda, n_lags) from the ACF shape over lags 1..5, fit through the origin on the lags
    before the first r(k) <= ACF_FLOOR."""
    n = 0
    for k in range(ACF_MAXLAG):
        if not np.isfinite(r[k]) or r[k] <= ACF_FLOOR:
            break
        n = k + 1
    if n == 0:
        return float("nan"), 0
    t = (np.arange(1, n + 1) * DT)
    y = np.log(r[:n])
    return float(-(t @ y) / (t @ t)), n


def _D_from(lam: float, m2: float) -> float:
    if not np.isfinite(lam) or lam <= 0:
        return float("nan")
    return float(lam * m2 / (2.0 * (1.0 - np.exp(-lam * DT))))


def lam_onestep(d: dict) -> float:
    """lambda from the decay of the one-step map, -ln(r(1))/dt."""
    r = d["acf"][0]
    if not np.isfinite(r) or r <= 0.0 or r >= 1.0:
        return float("nan")
    return float(-math.log(r) / DT)


def lam_shape(d: dict) -> tuple:
    """5-lag ACF shape fit, floor 0.05."""
    lam, nlag = _lam_from_r(np.asarray(d["acf"], float))
    return lam, nlag


def D_from(lam: float, m2: float) -> float:
    return _D_from(lam, m2)


def bias_curve(rng, n_blocks: int = 20000, grid=None) -> dict:
    """E[r_hat(1)] as a function of the true rho, for 30-window per-bout-centred
    blocks pooled as the cells pool them; also the spread attenuation."""
    if grid is None:
        grid = np.concatenate([np.linspace(0.005, 0.30, 60), np.linspace(0.31, 0.95, 65)])
    rh, wf = [], []
    for rho in grid:
        x = np.empty((n_blocks, NW))
        x[:, 0] = rng.standard_normal(n_blocks)
        s = math.sqrt(max(1.0 - rho * rho, 1e-12))
        e = rng.standard_normal((n_blocks, NW - 1)) * s
        for t in range(1, NW):
            x[:, t] = rho * x[:, t - 1] + e[:, t - 1]
        c = x - x.mean(axis=1, keepdims=True)
        num = (c[:, :-1] * c[:, 1:]).sum()
        den = (c[:, :-1] * c[:, :-1]).sum()
        rh.append(num / den)
        wf.append(math.sqrt((c * c).sum() / c.size))      # true sd is 1.0
    return dict(rho_true=[float(x) for x in grid],
                r1_expected=[float(x) for x in rh],
                width_factor=[float(x) for x in wf])


def make_bias_inverse(bc: dict):
    """rho_hat -> rho_true, by monotone interpolation of the measured curve."""
    rt = np.asarray(bc["rho_true"], float)
    rh = np.asarray(bc["r1_expected"], float)
    o = np.argsort(rh)
    rh, rt = rh[o], rt[o]

    def inv(r_hat: float) -> float:
        if not np.isfinite(r_hat) or r_hat <= 0 or r_hat >= 1:
            return float("nan")
        if r_hat < rh[0] or r_hat > rh[-1]:
            return float("nan")          # outside the calibrated range
        return float(np.interp(r_hat, rh, rt))

    def vinv(arr):
        arr = np.asarray(arr, float)
        out = np.interp(arr, rh, rt, left=np.nan, right=np.nan)
        return np.where(np.isfinite(arr) & (arr > 0) & (arr < 1), out, np.nan)
    inv.vec = vinv
    return inv


# ========================================================================= cells
def cellify(nights_rows, arm: str, inv=None) -> dict:
    """{stem: {stage: cell}}.  arm in {'onestep','onestep_bc','shape'}."""
    by = collections.defaultdict(list)
    for n in nights_rows:
        for d in n.get("directions", []):
            by[(d["stem"], d["stage"])].append(d)
    out = collections.defaultdict(dict)
    for (stem, stage), ds in sorted(by.items()):
        lam, cen = [], 0
        for d in ds:
            if arm == "shape":
                l, nl = lam_shape(d)
                ok = (nl >= 2) and np.isfinite(l) and l > 0
            else:
                l = lam_onestep(d)
                if arm == "onestep_bc" and np.isfinite(l):
                    rt = inv(d["acf"][0])
                    l = float(-math.log(rt) / DT) if (np.isfinite(rt) and 0 < rt < 1) else float("nan")
                ok = np.isfinite(l) and l > 0
            if ok:
                lam.append((l, D_from(l, d["m2"]), d["width"]))
            else:
                cen += 1
        nk = len(ds)
        nr = len(lam)
        out[stem][stage] = dict(
            stem=stem, subject=stem[:5], stage=stage,
            width=float(np.median([d["width"] for d in ds])),
            n_dir=nk, n_res=nr, n_censored=cen,
            resolved=bool(nr > 0 and nr * 2 >= nk),
            lam=float(np.median([x[0] for x in lam])) if lam else None,
            D=float(np.median([x[1] for x in lam])) if lam else None,
            id_ratio=float(np.median([x[1] / (x[0] * x[2] ** 2) for x in lam])) if lam else None)
    return dict(out)


def check_cell_width(cells, nights_rows):
    """The cell width recomputed here must equal the cell width stored in the rows."""
    dep = {(r["stem"], r["stage"]): r["width"] for n in nights_rows for r in n.get("rows", [])}
    d = [abs(dep[k] - cells[k[0]][k[1]]["width"]) for k in dep if k[0] in cells and k[1] in cells[k[0]]]
    return dict(n=len(d), max_abs_diff=float(max(d)) if d else None)


# ================================================================= contrasts
def contrast_rows(cells, a, b, key, require_resolved=True):
    rows = []
    for stem, m in cells.items():
        if a not in m or b not in m:
            continue
        ca, cb = m[a], m[b]
        if require_resolved and not (ca["resolved"] and cb["resolved"]):
            continue
        va, vb = ca.get(key), cb.get(key)
        if va is None or vb is None or not (np.isfinite(va) and np.isfinite(vb)) or va <= 0 or vb <= 0:
            continue
        rows.append((stem, ca["subject"], float(math.log(va / vb))))
    return rows


def _subject_means(rows):
    by = collections.defaultdict(list)
    for stem, sub, lr in rows:
        by[sub].append(lr)
    subs = sorted(by)
    return subs, np.array([float(np.mean(by[s])) for s in subs])


def contrast_stat(rows, rng, n_boot=N_BOOT):
    """Participant-level paired contrast (participant = the unit; night level beside it)."""
    if len(rows) < 3:
        return dict(testable=False, n_nights=len(rows))
    lr_n = np.array([r[2] for r in rows], float)
    subs, sv = _subject_means(rows)
    ns, ks = len(sv), int((sv > 0).sum())
    boot = []
    for _ in range(n_boot):
        p = rng.integers(0, ns, ns)
        boot.append(float(np.median(sv[p])))
    boot = np.array(boot)
    return dict(
        testable=True,
        subject=dict(
            n_subjects=ns, k_a_greater=ks,
            fold=float(np.exp(np.median(sv))),
            mean_fold=float(np.exp(np.mean(sv))),
            fold_ci95=[float(np.exp(np.percentile(boot, 2.5))),
                       float(np.exp(np.percentile(boot, 97.5)))],
            sign_p=float(sst.binomtest(ks, ns, 0.5).pvalue),
            wilcoxon_p=float(sst.wilcoxon(sv).pvalue) if ns >= 6 else None,
            median_log_ratio=float(np.median(sv))),
        night=dict(
            n_nights=len(lr_n), k_a_greater=int((lr_n > 0).sum()),
            fold=float(np.exp(np.median(lr_n))),
            sign_p=float(sst.binomtest(int((lr_n > 0).sum()), len(lr_n), 0.5).pvalue),
            median_log_ratio=float(np.median(lr_n))))


def perm_null(cells, a, b, key, require_resolved, obs, rng, n_perm):
    """Within-night permutation of stage labels among that night's own cells.

    SIZE MATCHING: the resolvability rule is a property of the cell, so a
    permutation can land the contrast on more resolvable cells than the observed
    one.  Every permutation whose participant count exceeds the observed one is
    subsampled, uniformly over participants, down to the observed count.
    """
    stems = sorted(cells)
    packs = []
    for s in stems:
        m = cells[s]
        st = list(m)
        packs.append((s, m[st[0]]["subject"], st,
                      np.array([(m[x].get(key) if (m[x].get(key) is not None
                                                   and np.isfinite(m[x].get(key))
                                                   and m[x].get(key) > 0) else np.nan)
                                for x in st], float),
                      np.array([bool(m[x]["resolved"]) for x in st])))
    obs_ns = obs["subject"]["n_subjects"]
    obs_med = obs["subject"]["median_log_ratio"]
    meds, fracs, ns_raw = [], [], []
    for _ in range(n_perm):
        by = collections.defaultdict(list)
        for stem, sub, st, val, res in packs:
            if a not in st or b not in st:
                continue
            p = rng.permutation(len(st))
            ia, ib = st.index(a), st.index(b)
            va, vb = val[p[ia]], val[p[ib]]
            if require_resolved and not (res[p[ia]] and res[p[ib]]):
                continue
            if not (np.isfinite(va) and np.isfinite(vb)):
                continue
            by[sub].append(math.log(va / vb))
        if len(by) < 3:
            continue
        subs = sorted(by)
        ns_raw.append(len(subs))
        sv = np.array([np.mean(by[s]) for s in subs], float)
        if len(sv) > obs_ns:
            sv = sv[rng.choice(len(sv), obs_ns, replace=False)]
        meds.append(float(np.median(sv)))
        fracs.append(float((sv > 0).mean()))
    meds, fracs, ns_raw = np.array(meds), np.array(fracs), np.array(ns_raw, float)
    n = len(meds)
    if n == 0:
        return dict(n_perm=0)
    p_ge = float(((meds >= obs_med).sum() + 1) / (n + 1))
    p_le = float(((meds <= obs_med).sum() + 1) / (n + 1))
    return dict(n_perm=int(n),
                observed_n_subjects=int(obs_ns),
                null_n_subjects_before_subsample=dict(
                    mean=float(ns_raw.mean()), p5=float(np.percentile(ns_raw, 5)),
                    p95=float(np.percentile(ns_raw, 95)),
                    frac_exceeding_observed=float((ns_raw > obs_ns).mean())),
                null_median_log_ratio_mean=float(meds.mean()),
                null_median_log_ratio_p95=[float(np.percentile(meds, 2.5)),
                                           float(np.percentile(meds, 97.5))],
                null_fold_mean=float(np.exp(meds.mean())),
                null_frac_mean=float(fracs.mean()),
                p_perm_ge=p_ge, p_perm_le=p_le,
                p_perm_two_sided=float(min(1.0, 2.0 * min(p_ge, p_le))))


_BINOM = {}


def _binom_p(k: int, n: int) -> float:
    key = (k, n)
    if key not in _BINOM:
        _BINOM[key] = float(sst.binomtest(k, n, 0.5).pvalue)
    return _BINOM[key]


def power_by_resampling(sv: np.ndarray, shift: float, rng, n=N_RESAMPLE):
    """sv = observed participant-level log ratios.  Centre, add a known shift,
    resample participants, run the two-sided sign test at alpha 0.05; the share of
    resamples with P < 0.05 in the direction of the shift."""
    c = sv - np.median(sv)
    ns = len(c)
    hit = 0
    for _ in range(n):
        d = c[rng.integers(0, ns, ns)] + shift
        k = int((d > 0).sum())
        p = _binom_p(k, ns)
        if p < 0.05 and ((k > ns / 2) == (shift > 0)):
            hit += 1
    return hit / n


# ========================================================== rows and drug labels
def load_rows(library: str) -> dict:
    """The per-night rows of bout_spread_rows.py for one library."""
    return json.loads((ROWS / ROWS_FILE[library]).read_text(encoding="utf-8"))


def telemetry_drug_map() -> dict:
    """(participant, night) -> 'placebo' | 'temazepam' from ST-subjects.xls (Sleep-EDF telemetry):
    column 0 participant, column 3 the placebo night, column 5 the temazepam night, from the
    third row on."""
    sheets = read_xls(SLEEP / "ST-subjects.xls")
    rows = next(iter(sheets.values()))
    m = {}
    for r in rows[2:]:
        sub = int(r[0])
        m[(sub, int(r[3]))] = "placebo"
        m[(sub, int(r[5]))] = "temazepam"
    return m


def telemetry_label(stem: str, dmap: dict):
    # ST7 <participant:02d> <night:1d>
    return dmap.get((int(stem[3:5]), int(stem[5])))


# ===================================================================== xls reader
# A minimal reader for ST-subjects.xls (a BIFF8 workbook in an OLE2 container), so that no Excel
# reader package is needed.  It reads the OLE2 header, FAT and miniFAT chains and directory to
# extract the "Workbook" stream, then the BIFF8 records BOF, SST (+ CONTINUE), BOUNDSHEET, LABELSST,
# LABEL, NUMBER, RK, MULRK and FORMULA (cached value).  It raises on anything it does not
# understand; it is not a general xls reader.  Its cells give the same assignment as
# pandas.read_excel(header=None) on this file.
OLE_MAGIC = bytes.fromhex("d0cf11e0a1b11ae1")
ENDOFCHAIN, FREESECT = 0xFFFFFFFE, 0xFFFFFFFF


def _ole_streams(data: bytes) -> dict[str, bytes]:
    if data[:8] != OLE_MAGIC:
        raise ValueError("not an OLE2 file")
    ssz = 1 << struct.unpack_from("<H", data, 30)[0]      # sector size
    mssz = 1 << struct.unpack_from("<H", data, 32)[0]     # mini sector size
    n_fat = struct.unpack_from("<I", data, 44)[0]
    dir_start = struct.unpack_from("<I", data, 48)[0]
    mini_cutoff = struct.unpack_from("<I", data, 56)[0]
    mini_start = struct.unpack_from("<I", data, 60)[0]
    difat_start = struct.unpack_from("<I", data, 68)[0]
    n_difat = struct.unpack_from("<I", data, 72)[0]

    def sector(i: int) -> bytes:
        off = 512 + i * ssz
        return data[off:off + ssz]

    # DIFAT: 109 entries in the header, then a chain of DIFAT sectors
    fat_sectors = [struct.unpack_from("<I", data, 76 + 4 * i)[0] for i in range(109)]
    fat_sectors = [s for s in fat_sectors if s not in (FREESECT, ENDOFCHAIN)]
    nxt, seen = difat_start, 0
    while nxt not in (ENDOFCHAIN, FREESECT) and seen < n_difat + 1:
        sec = sector(nxt)
        per = ssz // 4 - 1
        for i in range(per):
            v = struct.unpack_from("<I", sec, 4 * i)[0]
            if v not in (FREESECT, ENDOFCHAIN):
                fat_sectors.append(v)
        nxt = struct.unpack_from("<I", sec, 4 * per)[0]
        seen += 1
    if len(fat_sectors) < n_fat:
        raise ValueError(f"FAT truncated: {len(fat_sectors)} of {n_fat}")

    fat: list[int] = []
    for s in fat_sectors[:n_fat]:
        sec = sector(s)
        fat += list(struct.unpack_from(f"<{ssz // 4}I", sec, 0))

    def chain(start: int) -> list[int]:
        out, cur, guard = [], start, 0
        while cur not in (ENDOFCHAIN, FREESECT) and guard < len(fat) + 8:
            out.append(cur)
            cur = fat[cur] if cur < len(fat) else ENDOFCHAIN
            guard += 1
        return out

    def read_chain(start: int, size: int | None = None) -> bytes:
        b = b"".join(sector(i) for i in chain(start))
        return b[:size] if size is not None else b

    # directory
    dir_data = read_chain(dir_start)
    entries = []
    for off in range(0, len(dir_data), 128):
        e = dir_data[off:off + 128]
        if len(e) < 128:
            break
        nlen = struct.unpack_from("<H", e, 64)[0]
        if nlen < 2:
            continue
        name = e[:nlen - 2].decode("utf-16-le", "replace")
        etype = e[66]
        start = struct.unpack_from("<I", e, 116)[0]
        size = struct.unpack_from("<Q", e, 120)[0]
        entries.append((name, etype, start, size))
    if not entries:
        raise ValueError("no OLE directory entries")

    root = next((e for e in entries if e[1] == 5), None)
    mini_stream = read_chain(root[2]) if root and root[2] != ENDOFCHAIN else b""
    minifat: list[int] = []
    if mini_start not in (ENDOFCHAIN, FREESECT):
        mb = read_chain(mini_start)
        minifat = list(struct.unpack_from(f"<{len(mb) // 4}I", mb, 0))

    def read_mini(start: int, size: int) -> bytes:
        out, cur, guard = [], start, 0
        while cur not in (ENDOFCHAIN, FREESECT) and guard < len(minifat) + 8:
            off = cur * mssz
            out.append(mini_stream[off:off + mssz])
            cur = minifat[cur] if cur < len(minifat) else ENDOFCHAIN
            guard += 1
        return b"".join(out)[:size]

    streams = {}
    for name, etype, start, size in entries:
        if etype != 2:                     # 2 = stream
            continue
        streams[name] = (read_mini(start, size) if size < mini_cutoff
                         else read_chain(start, size))
    return streams


def _records(buf: bytes):
    i = 0
    while i + 4 <= len(buf):
        rid, ln = struct.unpack_from("<HH", buf, i)
        yield rid, buf[i + 4:i + 4 + ln]
        i += 4 + ln


def _unicode_string(b: bytes, off: int) -> tuple[str, int]:
    """BIFF8 XLUnicodeString; returns (text, bytes consumed)."""
    n = struct.unpack_from("<H", b, off)[0]
    flags = b[off + 2]
    p = off + 3
    if flags & 0x08:                       # rich text runs
        rt = struct.unpack_from("<H", b, p)[0]; p += 2
    else:
        rt = 0
    if flags & 0x04:                       # far-east extension
        sz = struct.unpack_from("<I", b, p)[0]; p += 4
    else:
        sz = 0
    if flags & 0x01:
        s = b[p:p + 2 * n].decode("utf-16-le", "replace"); p += 2 * n
    else:
        s = b[p:p + n].decode("latin1"); p += n
    p += 4 * rt + sz
    return s, p - off


def _sst(payload: bytes, conts: list[bytes]) -> list[str]:
    """Shared string table, following CONTINUE records with their flag byte."""
    blob = payload
    parts = [payload] + conts
    out: list[str] = []
    total = struct.unpack_from("<I", blob, 4)[0]
    buf = b"".join(parts)
    off = 8
    for _ in range(total):
        if off + 3 > len(buf):
            break
        try:
            s, used = _unicode_string(buf, off)
        except Exception:  # noqa: BLE001
            break
        out.append(s)
        off += used
    return out


def _rk(v: int) -> float:
    if v & 0x02:
        num = float(v >> 2)
    else:
        num = struct.unpack("<d", struct.pack("<q", (v >> 2) << 34))[0]
    return num / 100.0 if v & 0x01 else num


def read_xls(path: str | Path) -> dict[str, list[list]]:
    """{sheet_name: [[cell, ...], ...]} of a BIFF8 .xls file."""
    data = Path(path).read_bytes()
    streams = _ole_streams(data)
    wb = streams.get("Workbook") or streams.get("Book")
    if wb is None:
        raise ValueError(f"no Workbook stream; found {sorted(streams)}")

    recs = list(_records(wb))
    # shared strings
    sst: list[str] = []
    for i, (rid, pl) in enumerate(recs):
        if rid == 0x00FC:                  # SST
            conts = []
            for rid2, pl2 in recs[i + 1:]:
                if rid2 == 0x003C:         # CONTINUE
                    conts.append(pl2[1:] if pl2 and pl2[0] in (0, 1) else pl2)
                else:
                    break
            sst = _sst(pl, conts)
            break

    # sheet names, in BOUNDSHEET order
    names = []
    for rid, pl in recs:
        if rid == 0x0085:                  # BOUNDSHEET
            n = pl[6]
            flags = pl[7]
            nm = (pl[8:8 + 2 * n].decode("utf-16-le", "replace") if flags & 1
                  else pl[8:8 + n].decode("latin1"))
            names.append(nm)

    sheets: dict[str, list[list]] = {}
    cur: list[list] | None = None
    sheet_i = -1
    for rid, pl in recs:
        if rid == 0x0809:                  # BOF
            if len(pl) >= 4 and struct.unpack_from("<H", pl, 2)[0] == 0x0010:
                sheet_i += 1
                cur = []
                sheets[names[sheet_i] if sheet_i < len(names) else f"Sheet{sheet_i}"] = cur
            continue
        if cur is None:
            continue

        def put(r, c, v):
            while len(cur) <= r:
                cur.append([])
            row = cur[r]
            while len(row) <= c:
                row.append(None)
            row[c] = v

        if rid == 0x00FD and len(pl) >= 10:            # LABELSST
            r, c, idx = struct.unpack_from("<HH", pl, 0) + (struct.unpack_from("<I", pl, 6)[0],)
            put(r, c, sst[idx] if idx < len(sst) else None)
        elif rid == 0x0204 and len(pl) >= 8:           # LABEL
            r, c = struct.unpack_from("<HH", pl, 0)
            s, _ = _unicode_string(pl, 6)
            put(r, c, s)
        elif rid == 0x0203 and len(pl) >= 14:          # NUMBER
            r, c = struct.unpack_from("<HH", pl, 0)
            put(r, c, struct.unpack_from("<d", pl, 6)[0])
        elif rid == 0x027E and len(pl) >= 10:          # RK
            r, c = struct.unpack_from("<HH", pl, 0)
            put(r, c, _rk(struct.unpack_from("<i", pl, 6)[0]))
        elif rid == 0x00BD and len(pl) >= 6:           # MULRK
            r, c0 = struct.unpack_from("<HH", pl, 0)
            n = (len(pl) - 6) // 6
            for k in range(n):
                put(r, c0 + k, _rk(struct.unpack_from("<i", pl, 4 + 6 * k + 2)[0]))
        elif rid == 0x0006 and len(pl) >= 14:          # FORMULA, cached value
            r, c = struct.unpack_from("<HH", pl, 0)
            raw = pl[6:14]
            if raw[6:8] == b"\xff\xff":
                put(r, c, None)
            else:
                put(r, c, struct.unpack("<d", raw)[0])
    return sheets
