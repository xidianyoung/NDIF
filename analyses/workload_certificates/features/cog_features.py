# -*- coding: utf-8 -*-
"""COG-BCI: shared loading, state space, axes and readouts.  Imported by cog_state.py and the drivers.
"""
from __future__ import annotations
import json, sys, time
import os
from pathlib import Path

import numpy as np
from scipy import stats as sst
from sklearn.decomposition import PCA

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parent.parent
OUT = HERE / "out"
CACHE = Path(os.environ.get("NDIF_COGBCI_CACHE", str(HERE / "cog_feature_cache")))
RAW = Path(os.environ.get("NDIF_COGBCI_RAW", "COG-BCI"))          # the COG-BCI release (not redistributed)

SEED = 20260921
# COG-BCI duplicate exclusion (public release): sub-27 duplicates sub-17; sub-28 session 1 duplicates sub-25 session 1
EXCLUDE_SUBS = {27}                      # EEG = byte copy of sub-17 (public release)
EXCLUDE_SESSIONS = {(28, "ses-S1")}      # EEG = byte copy of sub-25 ses-S1 (public release)
SESSIONS = ["ses-S1", "ses-S2", "ses-S3"]
NBACK = ["zeroBACK", "oneBACK", "twoBACK"]
MATB = ["MATBeasy", "MATBmed", "MATBdiff"]
FAMILIES = {"nback": NBACK, "matb": MATB}
REST = ["RS_Beg_EO", "RS_Beg_EC", "RS_End_EO", "RS_End_Ec"]
REST_EO = ["RS_Beg_EO", "RS_End_EO"]
REST_EC = ["RS_Beg_EC", "RS_End_Ec"]
REFREC = "Flanker"
D_PCA = 8
NCH, NB = 62, 4
BAND_GEO = np.array([np.sqrt(4 * 8), np.sqrt(8 * 12), np.sqrt(12 * 30), np.sqrt(30 * 45)])
# RSME condition codes in RSME.txt
RSME_CODE = {"MATBeasy": 1, "MATBmed": 2, "MATBdiff": 3,
             "zeroBACK": 4, "oneBACK": 5, "twoBACK": 6, "PVT": 7, "Flanker": 8}
# notebook.Order task numbering
NB_TASK = {1: "PVT", 2: "Flanker", 3: "twoBACK", 4: "oneBACK", 5: "zeroBACK",
           6: "MATBeasy", 7: "MATBmed", 8: "MATBdiff"}


def log(m):
    print(time.strftime("[%H:%M:%S] ") + m, flush=True)


# ------------------------------------------------------------------ behaviour
def load_kss():
    """{(sub, sess): {'beginning': v, 'end': v, 'after_PVT': v}}  (condition 3 is a derived value,
    takes negative numbers, and is not used)."""
    out = {}
    with open(RAW / "KSS.txt") as fh:
        next(fh)
        for line in fh:
            p = line.strip().split(",")
            if len(p) < 5:
                continue
            sub, sess, score, cond = int(p[0]), int(p[1]), float(p[2]), int(p[3])
            if cond in (0, 1, 2):
                out.setdefault((sub, sess), {})[{0: "beginning", 1: "end", 2: "after_PVT"}[cond]] = score
    return out


def load_rsme():
    out = {}
    with open(RAW / "RSME.txt") as fh:
        next(fh)
        for line in fh:
            p = line.strip().split(",")
            if len(p) < 5:
                continue
            out[(int(p[0]), int(p[1]), int(p[2]))] = float(p[3])
    return out


def load_order():
    """{(sub, sess): [task names in temporal order]} from notebook.mat."""
    import scipy.io as sio
    m = sio.loadmat(RAW / "notebook.mat", squeeze_me=False, struct_as_record=False)
    nb = m["notebook"][0, 0]
    out = {}
    for si, fld in enumerate(nb._fieldnames):
        s = getattr(nb, fld)[0, 0]
        for sj, sfld in enumerate(s._fieldnames):
            ses = getattr(s, sfld)
            if np.size(ses) == 0:
                continue
            ses = ses[0, 0]
            try:
                order = np.atleast_1d(ses.Order).ravel().astype(int)
            except Exception:
                continue
            if len(order) == 8:
                out[(si + 1, sj + 1)] = [NB_TASK[int(t)] for t in order]
    return out


# ------------------------------------------------------------------ cache
class Sub:
    """One participant's cached features, with the z/PCA reference fitted on Flanker."""

    def __init__(self, sid):
        self.sid = sid
        self.name = f"sub-{sid:02d}"
        self.d = np.load(CACHE / f"{self.name}.npz", allow_pickle=True)
        self.meta = json.loads(str(self.d["meta"]))
        self.inject = None   # end-to-end calibration hook: applied to the RAW log band powers,
                             # before the robust z, before the PCA, before any axis estimation

    def raw(self, ses, rec, series="dist", keep=True):
        k = f"{ses}|{rec}|{series}"
        if (self.sid, ses) in EXCLUDE_SESSIONS:
            return None, None
        if f"{k}|F" not in self.d:
            return None, None
        F = self.d[f"{k}|F"]
        m = self.d[f"{k}|keep"] if keep else np.ones(len(F), bool)
        seg = self.d[f"{k}|seg"]
        return F[m].astype(np.float64), seg[m]

    def ref_raw(self, series="dist"):
        Fs = []
        for ses in SESSIONS:
            F, _ = self.raw(ses, REFREC, series)
            if F is not None and len(F):
                Fs.append(F)
        return np.concatenate(Fs, 0) if Fs else None

    def fit_ref(self):
        F = self.ref_raw("dist")
        self.med = np.median(F, axis=0)
        iqr = sst.iqr(F, axis=0)
        iqr[~np.isfinite(iqr) | (iqr <= 0)] = 1.0
        self.iqr = iqr
        return self

    def z(self, F):
        return (F - self.med) / self.iqr

    def zrec(self, ses, rec, series="dist"):
        F, seg = self.raw(ses, rec, series)
        if F is None or len(F) == 0:
            return None, None
        if self.inject is not None:
            F = self.inject(self, ses, rec, series, F, seg)
        return self.z(F), seg


def load_subjects(sids=None):
    sids = sids or [i for i in range(1, 30) if (CACHE / f"sub-{i:02d}.npz").exists()]
    sids = [i for i in sids if i not in EXCLUDE_SUBS]
    subs = {}
    for i in sids:
        subs[i] = Sub(i).fit_ref()
    return subs


def fit_group_pca(subs, d=D_PCA, n_per_sub=1200, seed=SEED):
    """Group PCA on the pooled z-scored Flanker windows, equal n per participant."""
    rng = np.random.default_rng(seed)
    X = []
    for i, s in subs.items():
        Z = s.z(s.ref_raw("dist"))
        if len(Z) > n_per_sub:
            Z = Z[rng.choice(len(Z), n_per_sub, replace=False)]
        X.append(Z)
    X = np.concatenate(X, 0)
    p = PCA(n_components=d, random_state=seed).fit(X)
    return p


# ------------------------------------------------------------------ manipulation-blind directions
def gain_tilt_raw():
    """Uniform gain and spectral tilt as unit vectors in the RAW 248-d log-power space."""
    g = np.ones(NCH * NB)
    t = np.tile(np.log10(BAND_GEO) - np.log10(BAND_GEO).mean(), NCH)
    # feature order is (channel, band) C-order: reshape(NCH, NB)
    g = g / np.linalg.norm(g)
    t = t - (t @ g) * g
    t = t / np.linalg.norm(t)
    return g, t


def gain_tilt_z(sub):
    """Same two directions expressed in that participant's z-space (dF -> dF / iqr)."""
    g, t = gain_tilt_raw()
    G = np.column_stack([g / sub.iqr, t / sub.iqr])
    Q, _ = np.linalg.qr(G)
    return Q                                   # 248 x 2 orthonormal basis of span{gain, tilt}


def share_in(Q, w):
    """Fraction of the unit vector w's squared length inside the column span of Q."""
    w = w / np.linalg.norm(w)
    return float(np.sum((Q.T @ w) ** 2))


# ------------------------------------------------------------------ readouts
def _std_m(m):
    m = np.asarray(m, float)
    s = m.std(ddof=1)
    return (m - m.mean()) / (s if s > 0 else 1.0)


def position_spread(groups, w, mvals):
    """groups: list of (n_i, d) arrays, one per condition; mvals: manipulation value per group.

    beta_pos = slope of mean(p) on standardised m, divided by the pooled within-condition SD of p
    beta_spr = slope of log SD(p) on standardised m
    """
    w = np.asarray(w, float); w = w / np.linalg.norm(w)
    ps = [Z @ w for Z in groups]
    ms = _std_m(mvals)
    mu = np.array([p.mean() for p in ps])
    sd = np.array([p.std(ddof=1) for p in ps])
    n = np.array([len(p) for p in ps])
    pooled = np.sqrt(np.sum((n - 1) * sd ** 2) / max(1, np.sum(n - 1)))
    den = np.sum((ms - ms.mean()) ** 2)
    b_pos = float(np.sum((ms - ms.mean()) * (mu - mu.mean())) / den) / (pooled if pooled > 0 else np.nan)
    ls = np.log(np.maximum(sd, 1e-12))
    b_spr = float(np.sum((ms - ms.mean()) * (ls - ls.mean())) / den)
    return dict(beta_pos=b_pos, beta_spr=b_spr, pooled_sd=float(pooled),
                mu=mu.tolist(), sd=sd.tolist(), n=n.tolist())


def ar1_lambda(p, seg, dt=1.0):
    """AR(1) restoring rate from contiguous dyn windows only; lambda = -ln(a)/dt."""
    num = den = 0.0
    for s in np.unique(seg):
        x = p[seg == s]
        if len(x) < 10:
            continue
        x = x - x.mean()
        num += float(np.sum(x[:-1] * x[1:])); den += float(np.sum(x[:-1] ** 2))
    if den <= 0:
        return np.nan, np.nan
    a = num / den
    if not (0 < a < 1):
        return np.nan, float(a)
    return float(-np.log(a) / dt), float(a)


# ------------------------------------------------------------------ inference
def boot_mean(x, n=10000, seed=SEED):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 2:
        return float("nan"), float("nan"), float("nan"), len(x)
    rng = np.random.default_rng(seed)
    b = np.array([np.mean(rng.choice(x, len(x), replace=True)) for _ in range(n)])
    return float(np.mean(x)), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)), int(len(x))


def signflip_p(x, n=10000, seed=SEED):
    """Size-matched sign-flip permutation over participants (two-sided)."""
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 2:
        return float("nan")
    rng = np.random.default_rng(seed + 1)
    obs = abs(np.mean(x))
    s = rng.choice([-1.0, 1.0], size=(n, len(x)))
    null = np.abs((s * x).mean(axis=1))
    return float((1 + np.sum(null >= obs)) / (n + 1))


def cos_unit(a, b):
    if a is None or b is None:
        return float("nan")
    a = np.asarray(a, float); b = np.asarray(b, float)
    if a.ndim != 1 or b.ndim != 1 or a.shape != b.shape:
        return float("nan")
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    return float(a @ b / (na * nb))
