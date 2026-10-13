# -*- coding: utf-8 -*-
r"""Tail excess of the state autocorrelation in HCP fMRI, with the haemodynamic control that decides whether a
several-timescale verdict here is cortical or vascular.

Cohort.  99 participants x 2 working-memory runs, the 7-parcel fronto-parietal state at TR = 0.72 s, with the
archived block labels, read from the HCP fMRI state cache (``<count cache>/fmri``).  A block is one task block of
38 consecutive TRs; blocks never span a task block, so lag k is exactly k TRs = 0.72 k s.  TRs are samples, not
sliding windows; the overlapping counterfactual is supplied by a 2-TR boxcar at a 1-TR step.

THE HAEMODYNAMIC CONTROL.  The canonical HRF is a low-pass filter, so at TR = 0.72 s an HRF-filtered process has a
lag-1 autocorrelation of about 0.97 and an autocorrelation that is NOT a single exponential.  The arm therefore
carries a matched HRF reference: white neural activity convolved with the SPM canonical HRF, sampled at the TR,
plus white observation noise tuned to reproduce the observed r(1), pushed through the identical estimator.

CALIBRATION.  The injections enter the RAW parcel time series (NDIF_HCP_FMRI_PARCELS,
<subject>__<run>__atlas_fpn7.nii.npy), i.e. BEFORE the robust z and before any projection; the cached TR subset is
re-derived from the archived block onsets (NDIF_HCP_FMRI_LABELS, <subject>.raw.json).

This module also provides the state loader (load_subjects, load_axis, lopo, MAXLAG, TAIL0, TAIL1, PARCELS) used by
tail_excess_contrast.py and tail_excess_calibration.py.

Inputs   NDIF_COUNT_CACHE/fmri; for the calibration NDIF_HCP_FMRI_PARCELS and NDIF_HCP_FMRI_LABELS
         (default data/hcp_wm_labels)
Outputs  analyses/rate_identifiability/out/tail_excess_fmri.json, tail_excess_fmri_rows.csv,
         tail_excess_fmri_profile.csv, tail_excess_fmri_calibration.csv
Reported per-participant single-cohort verdicts (observed points of the identifiability map)
Runtime  about 22 min on one desktop core
Usage    python analyses/rate_identifiability/tail_excess_fmri.py [--nrep 300] [--perm 2000]
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyses.common import paths                                          # noqa: E402
import injection_lib as L                                                  # noqa: E402
import tail_excess_lib as A                                                # noqa: E402

CACHE = L.FMRI_CACHE
PARCELS = Path(paths.path("NDIF_HCP_FMRI_PARCELS"))
LABELS = Path(os.path.abspath(os.environ["NDIF_HCP_FMRI_LABELS"]) if os.environ.get("NDIF_HCP_FMRI_LABELS")
              else paths.repo("data", "hcp_wm_labels"))
DT = 0.72
SEED = 20260922
MAXLAG = 10
TAIL0, TAIL1 = 3, 10
MIN_BLOCK = MAXLAG + 8
N_CTRL = 20


def log(m):
    print(time.strftime("[%H:%M:%S] ") + str(m), flush=True)



# ----------------------------------------------------------------------------- HCP WM block labels and robust z
TR = 0.72
N_VOLS = 405
HRF_SHIFT_S = 5.0
CONDS = ("0bk", "2bk")


def tr_labels(blocks: dict) -> np.ndarray:
    """Label each of the 405 TRs: 0 = 0bk, 1 = 2bk, 2 = fixation/other.

    A TR belongs to a block if its midpoint falls inside the HRF-shifted
    window [onset + 5 s, onset + 5 s + duration).
    """
    mid = (np.arange(N_VOLS) + 0.5) * TR
    lab = np.full(N_VOLS, 2, np.int8)
    for ci, c in enumerate(CONDS):
        for onset, dur in blocks[c]:
            lab[(mid >= onset + HRF_SHIFT_S)
                & (mid < onset + HRF_SHIFT_S + dur)] = ci
    return lab


def block_ids(blocks: dict, cond_idx: int, lab: np.ndarray) -> np.ndarray:
    """For TRs of one condition, which of its (onset-ordered) blocks each is in."""
    c = CONDS[cond_idx]
    mid = (np.arange(N_VOLS) + 0.5) * TR
    bid = np.full(N_VOLS, -1, np.int32)
    for i, (onset, dur) in enumerate(blocks[c]):
        bid[(mid >= onset + HRF_SHIFT_S)
            & (mid < onset + HRF_SHIFT_S + dur)] = i
    bid[lab != cond_idx] = -1
    return bid


def robust_z(F: np.ndarray) -> np.ndarray:
    med = np.median(F, axis=0)
    iqr = np.subtract(*np.percentile(F, [75, 25], axis=0))
    return (F - med) / np.maximum(iqr, 1e-9)

# --------------------------------------------------------------------------- data
def load_subjects(limit=0, inject=None, seed=SEED):
    """{subj: {0: [blocks], 1: [blocks]}} from the archived labels.

    With `inject` the RAW parcel series is reloaded, the injection applied, and the robust z
    re-run, so the calibration is end to end from before standardisation.
    """
    rng = np.random.default_rng(seed)
    by = collections.defaultdict(lambda: {0: [], 1: []})
    for f in sorted(CACHE.glob("*.npz")):
        subj, run = f.stem.split("__")[0], f.stem.split("__")[1]
        z = np.load(f)
        Z, seg, lab = z["Z"].astype(float), z["seg"], z["lab"]
        if inject is not None:
            p = PARCELS / f"{subj}__{run}__atlas_fpn7.nii.npy"
            if not p.exists():
                continue
            raw = np.load(p).astype(float)
            raw = inject(raw, np.zeros(len(raw), int), rng)
            Fz = robust_z(raw)
            # the cached Z is a row-subset of robust_z(raw) in block order; re-extract by matching
            # the block structure, which is what seg encodes
            pos, Zs = 0, []
            order = _tr_order(subj, run, len(Z))
            if order is None or len(order) != len(Z):
                continue
            Z = Fz[order]
        for s in np.unique(seg):
            m = seg == s
            if m.sum() >= MIN_BLOCK:
                by[subj][int(lab[m][0])].append(Z[m])
    subs = {k: v for k, v in by.items() if len(v[0]) >= 2 and len(v[1]) >= 2}
    ks = sorted(subs)
    if limit:
        ks = ks[:limit]
    return {k: subs[k] for k in ks}


_ORDER = {}


def _tr_order(subj, run, n):
    """The TR indices the cache kept, recomputed from the archived labels exactly as the cache did."""
    key = (subj, run)
    if key in _ORDER:
        return _ORDER[key]
    arch = LABELS / f"{subj}.raw.json"
    if not arch.exists():
        _ORDER[key] = None
        return None
    raw = json.loads(arch.read_text(encoding="utf-8"))
    rv = raw.get("runs", {}).get(run)
    if not rv or not rv.get("seq"):
        _ORDER[key] = None
        return None
    blocks = {c: [] for c in CONDS}
    for onset, dur, load in rv["seq"]:
        blocks[CONDS[int(load)]].append((float(onset), float(dur)))
    for c in CONDS:
        blocks[c].sort()
    lab = tr_labels(blocks)
    bid0, bid1 = block_ids(blocks, 0, lab), block_ids(blocks, 1, lab)
    idx = []
    for ci, bid in ((0, bid0), (1, bid1)):
        for i in range(bid.max() + 1):
            m = (lab == ci) & (bid == i)
            if m.sum() >= 2:
                idx.append(np.flatnonzero(m))
    out = np.concatenate(idx) if idx else None
    _ORDER[key] = out
    return out


def load_axis(subs):
    ax = {}
    for sid, v in subs.items():
        m0, m1 = np.concatenate(v[0]).mean(0), np.concatenate(v[1]).mean(0)
        u = m1 - m0
        ax[sid] = u / (np.linalg.norm(u) + 1e-12)
    return ax


def lopo(ax, sid):
    v = np.mean([a for s, a in ax.items() if s != sid], axis=0)
    return v / (np.linalg.norm(v) + 1e-12)


# --------------------------------------------------------------------------- the HRF reference
def spm_hrf(tr, length=32.0):
    t = np.arange(0, length, tr)
    from scipy.special import gammaln
    def g(t, a, b):
        return np.exp(a * np.log(b) + (a - 1) * np.log(np.maximum(t, 1e-6)) - b * t - gammaln(a))
    h = g(t, 6.0, 1.0) - g(t, 16.0, 1.0) / 6.0
    return h / np.abs(h).sum()


def hrf_reference(r1_target, lens, maxlag, nrep, rng, tr=DT):
    """White neural activity -> SPM canonical HRF -> TR sampling -> white observation noise tuned so
    the simulated r(1) equals `r1_target`.  Everything this returns is vascular by construction."""
    h = spm_hrf(tr)
    burn = 200
    # neural-only ACF at this TR, measured once
    def draw(nu):
        num = np.zeros(maxlag + 1); den = np.zeros(nrep); nm = np.zeros((nrep, maxlag + 1))
        for n in lens:
            if n < maxlag + 2:
                continue
            e = rng.standard_normal((nrep, n + burn + len(h)))
            y = np.apply_along_axis(lambda v: np.convolve(v, h, mode="same"), 1, e)
            y = y[:, burn:burn + n]
            y = y / (y.std(axis=1, keepdims=True) + 1e-12)
            y = np.sqrt(1 - nu) * y + np.sqrt(nu) * rng.standard_normal((nrep, n))
            y = y - y.mean(axis=1, keepdims=True)
            m = n - maxlag
            den += np.sum(y[:, :m] ** 2, axis=1)
            for k in range(maxlag + 1):
                nm[:, k] += np.sum(y[:, :m] * y[:, k:m + k], axis=1)
        ok = den > 0
        return (nm[ok] / den[ok][:, None]).mean(axis=0)
    lo, hi = 0.0, 0.999
    for _ in range(18):
        nu = 0.5 * (lo + hi)
        r = draw(nu)
        if r[1] > r1_target:
            lo = nu
        else:
            hi = nu
    nu = 0.5 * (lo + hi)
    return draw(nu), float(nu)


# --------------------------------------------------------------------------- per unit
def analyse_unit(B, w, rng, nrep, maxlag=MAXLAG, tail=(TAIL0, TAIL1)):
    return A.analyse_unit(B, w, rng, nrep, maxlag, tail)


def group(vals, label, nullmat=None):
    return A.group_stat(vals, label, nullmat, SEED)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nrep", type=int, default=300)
    ap.add_argument("--perm", type=int, default=2000)
    ap.add_argument("--subs", type=int, default=0)
    ap.add_argument("--skip-calib", action="store_true")
    a = ap.parse_args()
    t0 = time.time()

    subs = load_subjects(a.subs)
    ax = load_axis(subs)
    log(f"HCP fMRI: {len(subs)} participants, TR = {DT}s")
    RES = dict(cohort="HCP fMRI", n_participants=len(subs), dt=DT, d=7, maxlag=MAXLAG,
               tail=[TAIL0, TAIL1], overlap="none (TRs are samples)",
               generated=time.strftime("%Y-%m-%d %H:%M:%S"))
    rows, prof = [], []

    for lev, name in ((0, "0bk"), (1, "2bk")):
        units, nulls, profiles, ceil = {}, [], [], []
        for sid in sorted(subs):
            B = subs[sid][lev]
            if len(B) < 2:
                continue
            u = analyse_unit(B, lopo(ax, sid), np.random.default_rng(SEED + L.seed_offset(sid) % 10000),
                             a.nrep)
            if u is None:
                continue
            units[sid] = u
            nulls.append(u["null_E"])
            profiles.append((np.array(u["ks"]), np.array(u["rho"])))
            ceil.append(A.half_ceiling(B, lopo(ax, sid),
                                       np.random.default_rng(SEED + 1000 + L.seed_offset(sid) % 10000),
                                       MAXLAG, (TAIL0, TAIL1)))
            rows.append(dict(cohort="fMRI", level=name, direction="load_axis", subject=sid,
                             n_blocks=u["n_blocks"], n_win=u["n_win"], r1=u["r"][1], r2=u["r"][2],
                             r3=u["r"][3], exp_index=u["exp_index"], parity=u["parity_amp"],
                             a=u["a"], nu=u["nu"], E_tail=u["E_tail"],
                             E_tail_corr=u["E_tail_corr"], slope=u["slope"], slope2=u["slope2"]))
        if not units:
            continue
        gE = group([u["E_tail"] for u in units.values()], f"tail excess {name}", nulls)
        gS = group([u["slope"] for u in units.values()], f"ratio slope {name}",
                   [u["null_slope"] for u in units.values()])
        lp = A.lagperm_null(profiles, a.perm, SEED)
        gS["p_lagperm"] = A.perm_p(lp, gS["mean"], "greater")
        gS["lagperm_hi"] = float(np.percentile(lp, 97.5)) if len(lp) else np.nan
        RES[name] = dict(n=len(units), tail_excess=gE, ratio_slope=gS,
                         ceiling_halfdiff=A.boot_ci([c for c in ceil if np.isfinite(c)]),
                         tail_excess_raw=group([u["E_tail_raw"] for u in units.values()],
                                               "tail excess (raw)",
                                               [u["null_E_raw"] for u in units.values()]),
                         n_admissible=len(units), n_attempted=len(subs),
                         exp_index=A.boot_ci([u["exp_index"] for u in units.values()]),
                         parity=A.boot_ci([u["parity_amp"] for u in units.values()]),
                         a_hat=A.boot_ci([u["a"] for u in units.values()]),
                         nu_hat=A.boot_ci([u["nu"] for u in units.values()]),
                         mean_r=np.mean([u["r"] for u in units.values()], axis=0).tolist(),
                         mean_pred=np.mean([u["pred"] for u in units.values()], axis=0).tolist())
        mr, mp = np.array(RES[name]["mean_r"]), np.array(RES[name]["mean_pred"])
        for k in range(1, MAXLAG):
            prof.append(dict(cohort="fMRI", level=name, k=k, lag_s=k * DT, r_obs=mr[k],
                             r_single_timescale=mp[k], rho_obs=mr[k + 1] / mr[k] if mr[k] > 1e-3 else np.nan,
                             rho_single_timescale=mp[k + 1] / mp[k] if mp[k] > 1e-3 else np.nan))
        log(f"  {name}: n={len(units)} exp_index={RES[name]['exp_index']['mean']:.3f} "
            f"a={RES[name]['a_hat']['mean']:.3f} nu={RES[name]['nu_hat']['mean']:.3f} "
            f"E_tail={gE['mean']:+.4f} [{gE['lo']:+.4f},{gE['hi']:+.4f}] vs single timescale "
            f"{gE.get('null_mean', float('nan')):+.4f} P={gE.get('p_single_timescale_model', float('nan')):.4f} "
            f"ceiling (split-half) {RES[name]['ceiling_halfdiff']['mean']:+.4f} "
            f"[{RES[name]['ceiling_halfdiff']['lo']:+.4f},"
            f"{RES[name]['ceiling_halfdiff']['hi']:+.4f}]")

    # ---- the haemodynamic reference
    rng = np.random.default_rng(SEED + 3)
    sid0 = sorted(subs)[0]
    lens = [len(b) for b in subs[sid0][0]]
    r1t = float(RES["0bk"]["mean_r"][1])
    rh, nu_h = hrf_reference(r1t, lens, MAXLAG, 300, rng)
    fh = A.fit_single_timescale(rh, lens, MAXLAG)
    RES["hrf_reference"] = dict(
        r1_target=r1t, nu_needed=nu_h, r=rh.tolist(), pred=fh["pred"].tolist(),
        a=fh["a"], nu=fh["nu"], E_tail=A.tail_excess(rh, fh["pred"], TAIL0, TAIL1),
        exp_index=A.exp_index(rh),
        note=("white neural activity -> SPM canonical HRF -> TR sampling -> white observation "
              "noise tuned to the observed r(1).  Any tail excess here is vascular by construction."))
    log(f"  HRF reference: nu needed {nu_h:.3f} to reach r(1)={r1t:.3f}; "
        f"E_tail={RES['hrf_reference']['E_tail']:+.4f}; exp_index={RES['hrf_reference']['exp_index']:.3f}")

    # ---- control directions
    Ea, Ec = [], []
    for sid in sorted(subs):
        B = subs[sid][0]
        r2 = np.random.default_rng(SEED + 500 + L.seed_offset(sid) % 10000)
        w = lopo(ax, sid)
        u = analyse_unit(B, w, None, 0)
        if u is None:
            continue
        Ea.append(u["E_tail"])
        cv = []
        for v in A.orth_directions(w, N_CTRL, r2):
            RES.setdefault("_ctrl_att", 0)
            RES["_ctrl_att"] += 1
            uu = analyse_unit(B, v, None, 0)
            if uu:
                cv.append(uu["E_tail"])
        if cv:
            Ec.append(float(np.mean(cv)))
            RES["_ctrl_adm"] = RES.get("_ctrl_adm", 0) + len(cv)
    n = min(len(Ea), len(Ec))
    RES["direction"] = dict(task_aligned=A.boot_ci(Ea), control_mean=A.boot_ci(Ec),
                            paired_difference=A.boot_ci(np.array(Ea[:n]) - np.array(Ec[:n])),
                            n_control_directions=N_CTRL,
                            control_admissible=RES.pop("_ctrl_adm", 0),
                            control_attempted=RES.pop("_ctrl_att", 0))
    d = RES["direction"]["paired_difference"]
    log(f"  direction: task-aligned {np.mean(Ea):+.4f} vs {N_CTRL} controls {np.mean(Ec):+.4f}; "
        f"paired {d['mean']:+.4f} [{d['lo']:+.4f},{d['hi']:+.4f}]")

    # ---- the sliding-window counterfactual (what an overlapping choice would have produced)
    Eo = []
    for sid in sorted(subs):
        BB = []
        for b in subs[sid][0]:
            sm = np.stack([np.convolve(b[:, i], np.ones(2) / 2, mode="valid")
                           for i in range(b.shape[1])], 1)
            BB.append(sm)
        u = analyse_unit(BB, lopo(ax, sid), None, 0)
        if u:
            Eo.append(u["E_tail"])
    RES["overlap_counterfactual"] = dict(E_tail=A.boot_ci(Eo), window_trs=2, step_trs=1)
    log(f"  OVERLAP counterfactual (2-TR boxcar at 1-TR step): "
        f"E_tail={RES['overlap_counterfactual']['E_tail']['mean']:+.4f} "
        f"against {RES['0bk']['tail_excess']['mean']:+.4f} non-overlapping")

    # ---- end-to-end calibration, injected before robust_z
    calib = []
    if not a.skip_calib:
        for nu_inj in (0.10, 0.25, 0.50):
            def inj(F, seg, rr, nu=nu_inj):
                return A.inject_white(F, nu, rr)
            sW = load_subjects(a.subs, inj, SEED + 21)
            E = [u["E_tail"] for sid in sorted(sW)
                 for u in [analyse_unit(sW[sid][0], lopo(ax, sid), None, 0)] if u]
            if E:
                g = A.boot_ci(E)
                calib.append(dict(cohort="fMRI", injection="white", size=nu_inj, a_slow="",
                                  E_tail=g["mean"], lo=g["lo"], hi=g["hi"], n=g["n"]))
                log(f"  CALIB white nu={nu_inj}: E_tail={g['mean']:+.4f} [{g['lo']:+.4f},{g['hi']:+.4f}]")
        for wt in (0.02, 0.05, 0.10, 0.20):
            def inj(F, seg, rr, w_=wt):
                return A.inject_slow(F, seg, w_, 0.95, rr)
            sS = load_subjects(a.subs, inj, SEED + 22)
            E = [u["E_tail"] for sid in sorted(sS)
                 for u in [analyse_unit(sS[sid][0], lopo(ax, sid), None, 0)] if u]
            if E:
                g = A.boot_ci(E)
                calib.append(dict(cohort="fMRI", injection="slow", size=wt, a_slow=0.95,
                                  E_tail=g["mean"], lo=g["lo"], hi=g["hi"], n=g["n"]))
                log(f"  CALIB second timescale weight={wt}: E_tail={g['mean']:+.4f} "
                    f"[{g['lo']:+.4f},{g['hi']:+.4f}]")
    RES["calibration"] = calib

    L.out("tail_excess_fmri.json").write_text(json.dumps(RES, indent=1, default=float), encoding="utf-8")
    _csv(L.out("tail_excess_fmri_rows.csv"), rows)
    _csv(L.out("tail_excess_fmri_profile.csv"), prof)
    _csv(L.out("tail_excess_fmri_calibration.csv"), calib)
    log(f"done in {time.time() - t0:.0f}s -> tail_excess_fmri.json")


def _csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, list(rows[0].keys()))
        wr.writeheader()
        for r in rows:
            wr.writerow(r)


if __name__ == "__main__":
    main()
