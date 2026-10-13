# -*- coding: utf-8 -*-
"""Digit-span EEG (OpenNeuro ds003838): per-participant feature cache of the encoding-period windows.

Computes, for every usable sequence (memorize or listen-only) of every participant, the log10 band power of
1-s windows on the 64 EEG channels (theta 4-8, alpha 8-12, beta 12-30, gamma 30-45 Hz) in two series:
  dist  windows every 0.5 s (moments of the state distribution)
  dyn   non-overlapping windows every 1.0 s (every drift fit; dt = 1 s)
with the amplitude QC flag of each window, the nuisance and residual-artifact columns, the peripheral columns
(heart rate, PPG amplitude, pupil diameter where present) and two item-slot labels per window:
  slot_true(_dyn)  the item interval [o_k, o_k+1) of the events file that contains the window centre (the last
                   interval is [o_n, o_n + median inter-item interval))
  slot_pub(_dyn)   the slot from the nominal 2.0-s inter-item interval, floor((centre - t0) / 2.0) + 1
and, per memorize sequence, the recall outcome of every item ('correctly recalled' / 'forgotten').

Inputs
  NDIF_DS003838_DERIV  preprocessed recordings, one directory per participant (sub-XXX/) with
                       sub-XXX_task-memory_preproc_250Hz.npy (channels x samples, microvolts, 250 Hz),
                       sub-XXX_sequences.json, sub-XXX_qc.json, sub-XXX_nuisance_1s.npz,
                       sub-XXX_periph_1s.npz and sub-XXX_pupil_1s.npz (where recorded), _done.json
  NDIF_DS003838_RAW    the BIDS dataset (sub-XXX/eeg/sub-XXX_task-memory_events.tsv: true item onsets and
                       recall outcomes)
Output (paths.cache("ds003838"), environment variable NDIF_DS003838_CACHE, default analyses/_work/ds003838)
  sub-XXX.npz          per sequence j: s{j:03d}_F_dist, _F_dyn (float32), _flag_d, _flag_y (bool), _t_start,
                       _t_dyn (s), _slot_true, _slot_pub, _slot_true_dyn, _slot_pub_dyn (int16), _nuis, _resid,
                       _periph (float32, NaN -> 0), _item_ok (int16), _onsets (s)
  sub-XXX_meta.json    channels, settings, QC summary and one entry per sequence (cond 0 = listen-only,
                       1 = memorize; block, role, n_digits, t0, correct, mem_index, window counts)
  _cache_report.json   one line per participant (cached or the exclusion reason)
Participants are excluded when their QC record marks them excluded, when fewer than 8 ICA components are
labelled brain or more than 30 were removed, or when the events file and the sequence list disagree.  The
derivative itself was produced by standard preprocessing (Methods, Human recordings); that step is not part of
this repository.

Supports: the 64 digit-span recordings of the count cache (count_cache.py; Fig. 2b,c,f; Supplementary Fig. 2a;
Methods, Human recordings) and the digit-span analyses that read the same cache (analyses/task_location,
analyses/rate_identifiability).
Runtime: 0.1-0.4 min per participant, about 23 min for the 64 participants on one process (about 4 min with 6
workers) on one desktop.
Threads: two per worker, as in the stored run.

    python analyses/cohort_states/ds003838_states.py --workers 6
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))
from analyses.common import paths                                      # noqa: E402
from states_lib import short_time_band_power, log_features, artifact_proxies  # noqa: E402

DERIV = Path(paths.path("NDIF_DS003838_DERIV"))
DATA = Path(paths.path("NDIF_DS003838_RAW"))
CACHE = Path(paths.cache("ds003838"))

SF = 250.0                           # sampling rate of the derivative (Hz)
SOA = 2.0                            # nominal inter-item interval of the slot_pub labels (s)
WIN, STEP_DIST, STEP_DYN = 1.0, 0.5, 1.0
D = 8                                # latent dimension of the state space built from this cache
BANDS = {"theta": (4, 8), "alpha": (8, 12), "beta": (12, 30), "gamma": (30, 45)}
PTP_UV, PTP_NCH = 500.0, 3           # a window is flagged if more than 3 channels exceed 500 uV peak to peak
PTP_ANY_UV = 2000.0                  # ... or any channel exceeds 2000 uV
NUIS_KEYS = ["ic_eye", "ic_muscle", "ic_heart", "ic_linechan", "emg55_95"]      # removed-artifact space
EV_RE = re.compile(r"^(control|memory)\s+(\d+)/(\d+)")


def log(msg):
    print(time.strftime("[%H:%M:%S] ") + msg, flush=True)


# --------------------------------------------------------------------------- derivative
def load_subject(sub):
    d = DERIV / sub
    X = np.load(d / f"{sub}_task-memory_preproc_250Hz.npy", mmap_mode="r")
    seqs = json.loads((d / f"{sub}_sequences.json").read_text(encoding="utf-8"))
    qc = json.loads((d / f"{sub}_qc.json").read_text(encoding="utf-8"))
    nz = dict(np.load(d / f"{sub}_nuisance_1s.npz"))
    periph = dict(np.load(d / f"{sub}_periph_1s.npz")) if (d / f"{sub}_periph_1s.npz").exists() else None
    pupil = dict(np.load(d / f"{sub}_pupil_1s.npz")) if (d / f"{sub}_pupil_1s.npz").exists() else None
    return X, seqs, qc, nz, periph, pupil


def grid_index(t_start):
    """Nearest global 1-s window (grid step 0.5 s) for a window starting at t_start."""
    return int(round(t_start / 0.5))


def seq_features(X, names, s, nz, periph, pupil):
    """All per-window quantities of one sequence's encoding period."""
    i0, i1 = int(s["t0"] * SF), int(s["t_end"] * SF)
    x = np.asarray(X[:, i0:i1], dtype=np.float64)
    if x.shape[1] < int(WIN * SF):
        return None
    Pd = short_time_band_power(x, SF, WIN, STEP_DIST, BANDS)          # (n_w, 64, 4)
    Py = short_time_band_power(x, SF, WIN, STEP_DYN, BANDS)
    n_w, n_y = Pd.shape[0], Py.shape[0]
    scalp = np.array([n != "FCz" for n in names])
    nper, nstep = int(WIN * SF), int(STEP_DIST * SF)
    view = np.lib.stride_tricks.sliding_window_view(x[scalp], nper, axis=1)[:, ::nstep, :][:, :n_w, :]
    ptp = view.max(axis=2) - view.min(axis=2)
    flag_d = ((ptp > PTP_UV).sum(axis=0) > PTP_NCH) | (ptp.max(axis=0) > PTP_ANY_UV)
    view_y = np.lib.stride_tricks.sliding_window_view(x[scalp], nper, axis=1)[:, ::nper, :][:, :n_y, :]
    ptp_y = view_y.max(axis=2) - view_y.min(axis=2)
    flag_y = ((ptp_y > PTP_UV).sum(axis=0) > PTP_NCH) | (ptp_y.max(axis=0) > PTP_ANY_UV)
    art = artifact_proxies(x[scalp], SF, WIN, STEP_DIST)               # emg30_45, log ptp, spike fraction
    spike = art[:n_w, 2]
    starts = s["t0"] + STEP_DIST * np.arange(n_w)
    g = np.array([grid_index(t) for t in starts])
    g = np.clip(g, 0, len(nz["grid_t"]) - 1)
    N = np.column_stack([nz[k][g] for k in NUIS_KEYS] + [spike])
    emg_clean = nz["emg55_95_clean"][g] if "emg55_95_clean" in nz else np.full(n_w, np.nan)
    Rz = np.column_stack([art[:n_w, 0], art[:n_w, 1], spike, emg_clean])
    P = None
    if periph is not None:
        gp = np.clip(g, 0, len(periph["grid_t"]) - 1)
        cols = [periph["hr"][gp], periph["ppg_amp"][gp]]
        if pupil is not None:
            gq = np.clip(g, 0, len(pupil["grid_t"]) - 1)
            cols.append(pupil["diameter"][gq])
        P = np.column_stack(cols)
    digit = np.clip(np.floor((starts + WIN / 2 - s["t0"]) / SOA).astype(int) + 1, 1, s["n"])
    return dict(F_dist=log_features(Pd), F_dyn=log_features(Py),
                flag_d=flag_d, flag_y=flag_y, nuis=N, resid=Rz, periph=P, digit=digit, t_start=starts,
                n_w=n_w, n_y=n_y)


def build_blocks(X, names, seqs, nz, periph, pupil):
    """Usable sequences with at least three unflagged dist windows, with their features."""
    out = []
    mem_order = 0
    for k, s in enumerate(seqs):
        if not s.get("usable", True):
            continue
        f = seq_features(X, names, s, nz, periph, pupil)
        if f is None or (~f["flag_d"]).sum() < 3:
            continue
        b = dict(seq=k, cond=0 if s["cond"] == "control" else 1, block=int(s["block"]), role=s["role"],
                 n_digits=int(s["n"]), t0=float(s["t0"]), correct=bool(s["correct"]),
                 mem_index=(mem_order if s["cond"] == "memory" else None))
        if s["cond"] == "memory":
            mem_order += 1
        b.update(f)
        out.append(b)
    return out


# --------------------------------------------------------------------------- true item onsets
def read_item_onsets(events_tsv: Path):
    """Sequences in file order, each with the true per-item onsets and per-item recall outcome.

    A sequence is opened by item 1 and continued by later items of the same condition and length, the rule by
    which the derivative's sequence list was built, so the list is index-aligned with sub-XXX_sequences.json.
    """
    seqs, cur = [], None
    with open(events_tsv, encoding="utf-8") as f:
        h = f.readline().rstrip("\n").split("\t")
        i_on, i_tt, i_val = h.index("onset"), h.index("trial_type"), h.index("value")
        for line in f:
            c = line.rstrip("\n").split("\t")
            if c[i_val] == "boundary":
                continue
            m = EV_RE.match(c[i_tt])
            if not m:
                continue
            cond, k, n = m.group(1), int(m.group(2)), int(m.group(3))
            t = float(c[i_on])
            item_ok = 1 if ("correctly recalled" in c[i_tt]) else (0 if "forgotten" in c[i_tt] else -1)
            if k == 1:
                cur = dict(cond=cond, n=n, onsets=[t], items=[k], ok=[item_ok])
                seqs.append(cur)
            elif cur is not None and cur["cond"] == cond and cur["n"] == n:
                cur["onsets"].append(t); cur["items"].append(k); cur["ok"].append(item_ok)
    return seqs


def slot_edges(onsets, n):
    """Interval edges of the n item slots: [o_1, o_2, ..., o_n, o_n + median inter-item interval]."""
    o = np.asarray(onsets, dtype=float)
    if len(o) >= 2:
        soa = float(np.median(np.diff(o)))
    else:
        soa = 2.0323
    return np.concatenate([o, [o[-1] + soa]]), soa


def slots_of(centres, edges, n):
    """Item slot (1..n) containing each window centre, clipped to [1, n]."""
    k = np.searchsorted(edges, np.asarray(centres, float), side="right")
    return np.clip(k, 1, n).astype(np.int16)


# --------------------------------------------------------------------------- one participant
def build_one(sub, force=False):
    out_npz = CACHE / f"{sub}.npz"
    out_meta = CACHE / f"{sub}_meta.json"
    if out_npz.exists() and out_meta.exists() and not force:
        return dict(subject=sub, cached=True)
    t0 = time.time()
    X, seqs, qc, nz, periph, pupil = load_subject(sub)
    n_brain = sum(1 for l in qc["ica"]["labels"] if l == "brain")
    n_excl = len(qc["ica"]["exclude_primary"])
    if qc.get("excluded") or n_brain < 8 or n_excl > 30:
        return dict(subject=sub, excluded=qc.get("excluded") or f"ICA {n_brain} brain / {n_excl} excluded")
    names = qc["channels"]

    ev = read_item_onsets(DATA / sub / "eeg" / f"{sub}_task-memory_events.tsv")
    if len(ev) != len(seqs):
        return dict(subject=sub, excluded=f"events/sequences length mismatch {len(ev)} vs {len(seqs)}")
    t0_err = float(np.max([abs(ev[i]["onsets"][0] - seqs[i]["t0"]) for i in range(len(seqs))]))
    if t0_err > 1e-3:
        return dict(subject=sub, excluded=f"sequence t0 mismatch {t0_err:.4f} s")

    blocks = build_blocks(X, names, seqs, nz, periph, pupil)
    if not blocks:
        return dict(subject=sub, excluded="no usable sequences")

    rec = {}
    meta = dict(subject=sub, has_pupil=bool(pupil is not None), has_periph=bool(periph is not None),
                n_channels=len(names), channels=names, sequences=[],
                soa_published=SOA, win_s=WIN, step_dist_s=STEP_DIST, step_dyn_s=STEP_DYN,
                bands=BANDS, qc=dict(bads=qc["bads"], ica_excluded=n_excl,
                                     usable_sequence_fraction=qc["usable_sequence_fraction"]))
    soas = []
    for j, b in enumerate(blocks):
        k = b["seq"]
        e = ev[k]
        edges, soa = slot_edges(e["onsets"], b["n_digits"])
        soas.append(soa)
        c_dist = b["t_start"] + WIN / 2.0
        t_dyn = b["t0"] + STEP_DYN * np.arange(b["n_y"])
        c_dyn = t_dyn + WIN / 2.0
        slot_true = slots_of(c_dist, edges, b["n_digits"])
        slot_true_dyn = slots_of(c_dyn, edges, b["n_digits"])
        slot_pub = np.asarray(b["digit"], dtype=np.int16)
        slot_pub_dyn = np.clip(np.floor((c_dyn - b["t0"]) / SOA).astype(int) + 1, 1, b["n_digits"]).astype(np.int16)
        pre = f"s{j:03d}_"
        rec[pre + "F_dist"] = b["F_dist"].astype(np.float32)
        rec[pre + "F_dyn"] = b["F_dyn"].astype(np.float32)
        rec[pre + "flag_d"] = b["flag_d"].astype(bool)
        rec[pre + "flag_y"] = b["flag_y"].astype(bool)
        rec[pre + "t_start"] = b["t_start"].astype(np.float64)
        rec[pre + "t_dyn"] = t_dyn.astype(np.float64)
        rec[pre + "slot_true"] = slot_true
        rec[pre + "slot_pub"] = slot_pub
        rec[pre + "slot_true_dyn"] = slot_true_dyn
        rec[pre + "slot_pub_dyn"] = slot_pub_dyn
        rec[pre + "nuis"] = np.nan_to_num(b["nuis"].astype(np.float32), nan=0.0)
        rec[pre + "resid"] = np.nan_to_num(b["resid"].astype(np.float32), nan=0.0)
        if b["periph"] is not None:
            rec[pre + "periph"] = np.nan_to_num(b["periph"].astype(np.float32), nan=0.0)
        item_ok = np.asarray(e["ok"], dtype=np.int16)
        rec[pre + "item_ok"] = item_ok
        rec[pre + "onsets"] = np.asarray(e["onsets"], dtype=np.float64)
        meta["sequences"].append(dict(
            idx=j, seq=int(k), cond=int(b["cond"]), block=int(b["block"]), role=b["role"],
            n_digits=int(b["n_digits"]), t0=float(b["t0"]), correct=bool(b["correct"]),
            mem_index=(None if b["mem_index"] is None else int(b["mem_index"])),
            n_w=int(b["n_w"]), n_y=int(b["n_y"]), soa_true=float(soa),
            n_mislabelled=int((slot_true != slot_pub).sum()),
            frac_mislabelled=float(np.mean(slot_true != slot_pub)),
            has_periph=bool(b["periph"] is not None)))
    meta["soa_true_median"] = float(np.median(soas))
    meta["soa_true_mean"] = float(np.mean(soas))
    meta["frac_windows_mislabelled"] = float(
        np.mean(np.concatenate([rec[f"s{j:03d}_slot_true"] != rec[f"s{j:03d}_slot_pub"] for j in range(len(blocks))])))
    meta["n_sequences"] = len(blocks)
    meta["minutes"] = round((time.time() - t0) / 60, 2)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE / f"{sub}.npz.tmp.npz", **rec)
    os.replace(CACHE / f"{sub}.npz.tmp.npz", out_npz)
    out_meta.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return dict(subject=sub, n_sequences=len(blocks), soa=meta["soa_true_median"],
                frac_mislabelled=meta["frac_windows_mislabelled"], minutes=meta["minutes"])


def load_cache(sub):
    """Per-sequence arrays of the feature cache as lists indexed by sequence (keep = not flag_d)."""
    z = np.load(CACHE / f"{sub}.npz")
    meta = json.loads((CACHE / f"{sub}_meta.json").read_text(encoding="utf-8"))
    n = len(meta["sequences"])
    g = lambda k: [z[f"s{j:03d}_{k}"] for j in range(n)]  # noqa: E731
    S = dict(meta=meta, sub=sub, n=n,
             F_dist=g("F_dist"), F_dyn=g("F_dyn"),
             flag_d=g("flag_d"), flag_y=g("flag_y"),
             t_start=g("t_start"), t_dyn=g("t_dyn"),
             slot_true=g("slot_true"), slot_pub=g("slot_pub"),
             slot_true_dyn=g("slot_true_dyn"), slot_pub_dyn=g("slot_pub_dyn"),
             nuis=g("nuis"), resid=g("resid"), item_ok=g("item_ok"), onsets=g("onsets"))
    S["periph"] = [z[f"s{j:03d}_periph"] if f"s{j:03d}_periph" in z else None for j in range(n)]
    S["keep"] = [~f for f in S["flag_d"]]
    S["keep_y"] = [~f for f in S["flag_y"]]
    for k in ("cond", "block", "n_digits", "t0", "correct", "mem_index"):
        S[k] = [m[k] for m in meta["sequences"]]
    S["role"] = [m["role"] for m in meta["sequences"]]
    return S


def _worker(args):
    sub, force = args
    try:
        return build_one(sub, force=force)
    except Exception as e:  # noqa: BLE001
        import traceback
        return dict(subject=sub, error=repr(e), traceback=traceback.format_exc()[-2000:])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--subjects", nargs="*", default=None, help="participants to build (default: all)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--force", action="store_true", help="rebuild participants already in the cache")
    a = ap.parse_args()
    subs = sorted(p.name for p in DERIV.iterdir()
                  if p.is_dir() and p.name.startswith("sub-") and (p / "_done.json").exists())
    if a.subjects:
        subs = [s for s in subs if s in set(a.subjects)]
    log(f"{len(subs)} participants queued, {a.workers} workers")
    CACHE.mkdir(parents=True, exist_ok=True)
    res = []
    if a.workers <= 1:
        for s in subs:
            r = _worker((s, a.force)); res.append(r); log(str(r)[:200])
    else:
        import multiprocessing as mp
        with mp.Pool(a.workers) as pool:
            for r in pool.imap_unordered(_worker, [(s, a.force) for s in subs]):
                res.append(r); log(str(r)[:200])
    (CACHE / "_cache_report.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    ok = [r for r in res if "error" not in r and "excluded" not in r]
    log(f"done: {len(ok)}/{len(subs)} cached")
    return 0


if __name__ == "__main__":
    sys.exit(main())
