# -*- coding: utf-8 -*-
"""Shared helpers of the injection-recovery, rate-margin and tail-excess scripts of this directory.

* The block-wise propagator fit (``cog_fit_lib``), the time-rescaling estimators and injections
  (``cog_rescaling_lib``) and the COG-BCI state construction (``cog_features``, ``cog_state``) are the modules of
  ``analyses/workload_certificates/features``; this module puts that directory on the import path, so every
  cohort is analysed with the identical estimator code.
* Locations of the cohort caches read here:
    HCP fMRI working-memory state, one ``<subject>__<run>.npz`` per run (``Z``, ``seg``, ``lab``):
        ``<count cache>/fmri`` (``NDIF_COUNT_CACHE``);
    ds003838 digit-span per-sequence features, ``sub-XXX.npz`` + ``sub-XXX_meta.json``:
        ``NDIF_DS003838_CACHE``;
    HCP MEG per-trial state (``X0``, ``Y0``, ``X1``, ``Y1``, ``v``, ``trial0``, ``trial1``):
        ``NDIF_MEG_CACHE`` (written by ``analyses/cohort_states/hcp_meg_states.py``).
* The ds003838 design: the per-sequence cache loader, the robust z + PCA reference space fitted on the
  listen-only arm (blind to memory load), and the load-bin block lists of the item-count contrasts
  (bin A = slots 1-4, B = 5-8, C = 9-13; C1 = B vs A in set sizes 9 and 13, C2 = C vs A in set size 13).

Outputs of the scripts that are not stored package files go to ``analyses/rate_identifiability/out``
(or below ``NDIF_OUT`` when it is set).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats as sst
from sklearn.decomposition import PCA

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from analyses.common import paths                                          # noqa: E402

FEATURES = REPO / "analyses" / "workload_certificates" / "features"
if str(FEATURES) not in sys.path:
    sys.path.insert(0, str(FEATURES))

import cog_fit_lib as R                                                     # noqa: E402

FMRI_CACHE = Path(paths.cache("count")) / "fmri"
DS_CACHE = Path(paths.cache("ds003838"))
MEG_CACHE = Path(paths.cache("meg"))

D_PCA = 8
BINS = {"A": (1, 4), "B": (5, 8), "C": (9, 13)}
CONTRASTS = {"C1": ("A", "B", (9, 13)), "C2": ("A", "C", (13,))}


def out(name):
    """Output file of this directory (``analyses/rate_identifiability/out/<name>``)."""
    return Path(paths.out("analyses", "rate_identifiability", "out", name))


def log(m):
    print(time.strftime("[%H:%M:%S] ") + str(m), flush=True)


def seed_offset(key):
    """Per-unit seed offset: Python's built-in hash of the participant key."""
    return hash(key)


# ----------------------------------------------------------------------------- reference space
class RefSpace:
    """Robust z (median / raw IQR, IQR <= 0 -> 1) + PCA fitted on the reference condition."""

    def __init__(self, d, seed=20260728):
        self.d, self.seed = d, seed

    def fit(self, F_ref):
        self.med = np.median(F_ref, axis=0)
        iqr = sst.iqr(F_ref, axis=0)
        iqr[~np.isfinite(iqr) | (iqr <= 0)] = 1.0
        self.iqr = iqr
        self.pca = PCA(n_components=self.d, random_state=self.seed).fit(self.z(F_ref))
        return self

    def z(self, F):
        return (F - self.med) / self.iqr

    def transform(self, F):
        return self.pca.transform(self.z(F))


# ----------------------------------------------------------------------------- ds003838 cache
def load_cache(sub):
    """Per-sequence arrays of the ds003838 feature cache, as lists indexed by sequence."""
    z = np.load(DS_CACHE / f"{sub}.npz")
    meta = json.loads((DS_CACHE / f"{sub}_meta.json").read_text(encoding="utf-8"))
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


def subjects():
    return sorted(p.stem for p in DS_CACHE.glob("sub-*.npz"))


# ----------------------------------------------------------------------------- ds003838 design
def prep(sub, drop_first=0):
    """State space (fitted on the listen-only arm, blind to load bin) and per-bin blocks."""
    S = load_cache(sub)
    Fc = [S["F_dyn"][i][S["keep_y"][i]] for i in range(S["n"]) if S["cond"][i] == 0]
    Fc = [f for f in Fc if len(f)]
    if not Fc:
        return None
    sp = RefSpace(D_PCA).fit(np.vstack(Fc))
    out = {}
    for arm in (1, 0):
        for L in (5, 9, 13):
            for bn, (lo, hi) in BINS.items():
                blocks, dmu, half = [], [], []
                for i in range(S["n"]):
                    if S["cond"][i] != arm or S["n_digits"][i] != L:
                        continue
                    k = S["slot_true_dyn"][i]
                    m = S["keep_y"][i] & (k >= lo + drop_first) & (k <= hi + drop_first)
                    if m.sum() < 5:
                        continue
                    idx = np.flatnonzero(m)
                    if np.any(np.diff(idx) != 1):        # keep only the contiguous run
                        b = np.split(idx, np.flatnonzero(np.diff(idx) != 1) + 1)
                        idx = max(b, key=len)
                    if len(idx) < 5:
                        continue
                    Z = sp.transform(S["F_dyn"][i][idx])
                    blocks.append(Z)
                    half.append(0 if S["block"][i] < 5 else 1)
                    kd = S["slot_true"][i]
                    md = S["keep"][i] & (kd >= lo + drop_first) & (kd <= hi + drop_first)
                    if md.sum() >= 4:
                        dmu.append(sp.transform(S["F_dist"][i][md]).mean(axis=0))
                if blocks:
                    out[(arm, L, bn)] = dict(blocks=blocks, half=np.array(half),
                                             mu=np.mean(dmu, axis=0) if dmu else None,
                                             mu_half=[np.mean([dmu[j] for j in range(len(dmu))
                                                               if half[j] == h], axis=0)
                                                      if any(h == hh for hh in half) else None
                                                      for h in (0, 1)])
    return sp, out, S


def contrast_cells(out, arm, lo_bin, hi_bin, Ls, sel=None):
    """Matched block lists for the two bins of one contrast, pooled across the allowed set sizes."""
    A, B, mua, mub = [], [], [], []
    for L in Ls:
        ka, kb = (arm, L, lo_bin), (arm, L, hi_bin)
        if ka not in out or kb not in out:
            continue
        ba, bb = out[ka]["blocks"], out[kb]["blocks"]
        if sel is not None:
            ba = [b for b, h in zip(ba, out[ka]["half"]) if h == sel]
            bb = [b for b, h in zip(bb, out[kb]["half"]) if h == sel]
        if not ba or not bb:
            continue
        mb = R.match_blocks([ba, bb])
        if mb is None:
            continue
        A += mb[0]; B += mb[1]
        if out[ka]["mu"] is not None and out[kb]["mu"] is not None:
            mua.append(out[ka]["mu"]); mub.append(out[kb]["mu"])
    if not A:
        return None
    dmu = (np.mean(mub, axis=0) - np.mean(mua, axis=0)) if mua else None
    return A, B, dmu


def match_arms(cells_mem, cells_ctl):
    """Equalise the block structure BETWEEN arms so the difference in differences is bias-free."""
    if cells_mem is None or cells_ctl is None:
        return cells_mem, cells_ctl
    Am, Bm, dm = cells_mem
    Ac, Bc, dc = cells_ctl
    m = R.match_blocks([Am, Bm, Ac, Bc])
    if m is None:
        return None, None
    return (m[0], m[1], dm), (m[2], m[3], dc)
