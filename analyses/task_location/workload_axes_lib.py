# -*- coding: utf-8 -*-
"""Workload EEG (COG-BCI): a second, self-contained implementation of the within-participant cross-validated
readouts, used by workload_band_variants.py.

It reads only the feature cache and the KSS.txt table of the release and imports nothing from cog_features.py or
workload_axes.py: every readout (cache access, Flanker reference, band masks, segment-mean removal, axes, position
and spread slopes, bootstrap and sign-flip summaries with seed 777001) is derived again from its definition, so that
an agreement with workload_within_axes.py does not rest on shared code.  Duplicate exclusion as in the rest of the
cohort: sub-27; sub-28 session 1.  No computation is run on import.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
from scipy import stats as sst
from sklearn.decomposition import PCA

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from analyses.common import paths  # noqa: E402

CACHE = Path(paths.cache("cogbci"))
RAW = Path(paths.path("NDIF_COGBCI_RAW"))

SESSIONS = ["ses-S1", "ses-S2", "ses-S3"]
NBACK = ["zeroBACK", "oneBACK", "twoBACK"]
MATB = ["MATBeasy", "MATBmed", "MATBdiff"]
FAM = {"nback": NBACK, "matb": MATB}
EYES = {"EO": ["RS_Beg_EO", "RS_End_EO"], "EC": ["RS_Beg_EC", "RS_End_Ec"]}
NCH, NB = 62, 4
BANDNAMES = ["theta", "alpha", "beta", "gamma"]
SEED = 777001
# COG-BCI duplicate exclusion (public release): sub-27 duplicates sub-17; sub-28 session 1 duplicates sub-25 session 1
EXCLUDE_SUBS = {27}
EXCLUDE_SESSIONS = {(28, "ses-S1")}


# ---------------------------------------------------------------- behaviour
def load_kss():
    out = {}
    with open(RAW / "KSS.txt") as fh:
        next(fh)
        for line in fh:
            p = line.strip().split(",")
            if len(p) < 5:
                continue
            sub, ses, sc, cond = int(p[0]), int(p[1]), float(p[2]), int(p[3])
            if cond in (0, 1):
                out.setdefault((sub, ses), {})[{0: "beginning", 1: "end"}[cond]] = sc
    return out


# ---------------------------------------------------------------- cache access
class Cache:
    def __init__(self, sid, band_mask=None):
        self.sid = sid
        self.name = f"sub-{sid:02d}"
        self.d = np.load(CACHE / f"{self.name}.npz", allow_pickle=True)
        self.meta = json.loads(str(self.d["meta"]))
        self.band_mask = band_mask          # boolean over the 248 raw features, or None
        self.inject = None

    def raw(self, ses, rec):
        k = f"{ses}|{rec}|dist"
        if (self.sid, ses) in EXCLUDE_SESSIONS:
            return None, None
        if f"{k}|F" not in self.d:
            return None, None
        F = self.d[f"{k}|F"].astype(np.float64)
        keep = self.d[f"{k}|keep"]
        seg = self.d[f"{k}|seg"][keep]
        F = F[keep]
        if self.inject is not None:
            F = self.inject(self.sid, ses, rec, F, seg)
        if self.band_mask is not None:
            F = F[:, self.band_mask]
        return F, seg

    def fit_ref(self):
        Fs = []
        for ses in SESSIONS:
            F, _ = self.raw(ses, "Flanker")
            if F is not None and len(F):
                Fs.append(F)
        F = np.concatenate(Fs, 0)
        self.med = np.median(F, axis=0)
        iqr = sst.iqr(F, axis=0)
        iqr[~np.isfinite(iqr) | (iqr <= 0)] = 1.0
        self.iqr = iqr
        self._ref = F
        return self

    def z(self, F):
        return (F - self.med) / self.iqr

    def zrec(self, ses, rec):
        F, seg = self.raw(ses, rec)
        if F is None or len(F) == 0:
            return None, None
        return self.z(F), seg


def band_mask(drop=("gamma",), keep=None):
    """Boolean mask over the 248 raw (channel, band) features, C-order reshape(62, 4)."""
    m = np.ones((NCH, NB), bool)
    for bi, bn in enumerate(BANDNAMES):
        if keep is not None:
            m[:, bi] = bn in keep
        elif bn in drop:
            m[:, bi] = False
    return m.ravel()


# ---------------------------------------------------------------- one participant
def resid(Z, seg):
    Z = np.asarray(Z, float).copy()
    for s in np.unique(seg):
        k = seg == s
        Z[k] -= Z[k].mean(axis=0, keepdims=True)
    return Z


class Sub:
    """Load cells and rest epochs, in S248 (key 'Z') and optionally S8 (key 'Z8')."""

    def __init__(self, c: Cache, kss, pca=None):
        self.sid = c.sid
        self.c = c
        self.load = {}
        self.rest = {}
        to8 = (lambda Z: (Z - pca.mean_) @ pca.components_.T) if pca is not None else None
        for fam, recs in FAM.items():
            for si, ses in enumerate(SESSIONS, start=1):
                cell, ok = {}, True
                for lev, rec in enumerate(recs):
                    Z, seg = c.zrec(ses, rec)
                    m = c.meta["records"].get(f"{ses}|{rec}", {})
                    if Z is None or len(Z) < 60 or int(m.get("n_dyn", 0)) < 30:
                        ok = False; break
                    e = dict(Z=Z, seg=seg)
                    if to8 is not None:
                        e["Z8"] = to8(Z)
                    cell[lev] = e
                if ok:
                    self.load[(fam, si)] = cell
        for eye, recs in EYES.items():
            for si, ses in enumerate(SESSIONS, start=1):
                k = kss.get((c.sid, si))
                if k is None or "beginning" not in k or "end" not in k:
                    continue
                for tp, rec in zip(("beg", "end"), recs):
                    Z, seg = c.zrec(ses, rec)
                    if Z is None or len(Z) < 40:
                        continue
                    e = dict(Z=Z, seg=seg, kss=k["beginning"] if tp == "beg" else k["end"])
                    if to8 is not None:
                        e["Z8"] = to8(Z)
                    self.rest[(eye, si, tp)] = e

    # ---- labels
    def rest_keys(self, eye):
        ks = [(eye, s, t) for s in (1, 2, 3) for t in ("beg", "end") if (eye, s, t) in self.rest]
        return ks if len(ks) >= 4 else None

    def m_kss_within(self, eye):
        ks = self.rest_keys(eye)
        if ks is None:
            return None, None
        v = np.array([self.rest[k]["kss"] for k in ks], float)
        ses = np.array([k[1] for k in ks])
        mw = v.copy()
        for s in np.unique(ses):
            mw[ses == s] -= v[ses == s].mean()
        return ks, mw

    def m_clock(self, eye):
        """KSS-FREE label: -1 at the beginning of a session, +1 at the end, session-centred.
        Carries the temporal contrast and none of the arousal report."""
        ks = self.rest_keys(eye)
        if ks is None:
            return None, None
        v = np.array([1.0 if k[2] == "end" else -1.0 for k in ks], float)
        ses = np.array([k[1] for k in ks])
        for s in np.unique(ses):
            v[ses == s] -= v[ses == s].mean()
        return ks, v

    # ---- axes
    def axis_rest(self, eye, key="Z", label="kss"):
        got = self.m_kss_within(eye) if label == "kss" else self.m_clock(eye)
        ks, m = got
        if ks is None or np.std(m) == 0:
            return None
        M = np.stack([self.rest[k][key].mean(axis=0) for k in ks])
        mc = m - m.mean()
        v = (mc @ (M - M.mean(axis=0))) / np.sum(mc ** 2)
        n = np.linalg.norm(v)
        return v / n if n > 0 else None

    def axis_load(self, key="Z", family=None):
        ds = []
        for (fam, ses), cell in self.load.items():
            if family and fam != family:
                continue
            ds.append(cell[2][key].mean(axis=0) - cell[0][key].mean(axis=0))
        if not ds:
            return None
        v = np.mean(ds, axis=0)
        n = np.linalg.norm(v)
        return v / n if n > 0 else None

    def axis_sleep_both_eyes(self, key="Z", label="kss"):
        vs = [self.axis_rest(e, key, label) for e in EYES]
        vs = [v for v in vs if v is not None]
        if not vs:
            return None
        v = np.mean(vs, axis=0)
        n = np.linalg.norm(v)
        return v / n if n > 0 else None

    # ---- readouts
    @staticmethod
    def _beta(groups, mu, w, m):
        w = np.asarray(w, float); w = w / np.linalg.norm(w)
        ms = np.asarray(m, float)
        s = ms.std(ddof=1)
        ms = (ms - ms.mean()) / (s if s > 0 else 1.0)
        pm = np.array([float(v @ w) for v in mu])
        sd = np.array([float((np.asarray(g, float) @ w).std(ddof=1)) for g in groups])
        n = np.array([len(g) for g in groups])
        pooled = float(np.sqrt(np.sum((n - 1) * sd ** 2) / max(1, np.sum(n - 1))))
        den = float(np.sum((ms - ms.mean()) ** 2))
        bp = float(np.sum((ms - ms.mean()) * (pm - pm.mean())) / den) / (pooled if pooled > 0 else np.nan)
        ls = np.log(np.maximum(sd, 1e-12))
        bs = float(np.sum((ms - ms.mean()) * (ls - ls.mean())) / den)
        return bp, bs

    def read_rest(self, w, eye, key="Z", label="kss"):
        if w is None:
            return None
        got = self.m_kss_within(eye) if label == "kss" else self.m_clock(eye)
        ks, m = got
        if ks is None or np.std(m) == 0:
            return None
        groups = [resid(self.rest[k][key], self.rest[k]["seg"]) for k in ks]
        mu = [self.rest[k][key].mean(axis=0) for k in ks]
        return self._beta(groups, mu, w, m)

    def read_load(self, w, key="Z", family=None):
        if w is None:
            return None
        bp, bs = [], []
        for (fam, ses), cell in self.load.items():
            if family and fam != family:
                continue
            groups = [resid(cell[l][key], cell[l]["seg"]) for l in (0, 1, 2)]
            mu = [cell[l][key].mean(axis=0) for l in (0, 1, 2)]
            a, b = self._beta(groups, mu, w, [0, 1, 2])
            bp.append(a); bs.append(b)
        if not bp:
            return None
        return float(np.mean(bp)), float(np.mean(bs))


# ---------------------------------------------------------------- cohort
def build(band_keep=None, with_pca=True, sids=None, inject=None):
    sids = sids or sorted(int(p.stem.split("-")[1]) for p in CACHE.glob("sub-*.npz"))
    sids = [i for i in sids if i not in EXCLUDE_SUBS]
    bm = None if band_keep is None else band_mask(keep=band_keep)
    kss = load_kss()
    caches = {}
    for i in sids:
        c = Cache(i, bm)
        if inject is not None:
            c.inject = inject
        caches[i] = c.fit_ref()
    pca = None
    if with_pca:
        rng = np.random.default_rng(SEED)
        X = []
        for i, c in caches.items():
            Z = c.z(c._ref)
            if len(Z) > 1200:
                Z = Z[rng.choice(len(Z), 1200, replace=False)]
            X.append(Z)
        pca = PCA(n_components=8, random_state=SEED).fit(np.concatenate(X, 0))
    S = {i: Sub(caches[i], kss, pca) for i in sids}
    return S, caches, pca


# ---------------------------------------------------------------- inference
def boot_ci(x, n=10000, seed=SEED):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 2:
        return dict(mean=float("nan"), lo=float("nan"), hi=float("nan"), n=int(len(x)))
    rng = np.random.default_rng(seed)
    b = x[rng.integers(0, len(x), size=(n, len(x)))].mean(axis=1)
    return dict(mean=float(x.mean()), lo=float(np.percentile(b, 2.5)),
                hi=float(np.percentile(b, 97.5)), n=int(len(x)))


def signflip(x, n=10000, seed=SEED + 5):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 2:
        return float("nan")
    rng = np.random.default_rng(seed)
    obs = abs(x.mean())
    s = rng.choice([-1.0, 1.0], size=(n, len(x)))
    return float((1 + np.sum(np.abs((s * x).mean(axis=1)) >= obs)) / (n + 1))


def summ(x, n=10000):
    d = boot_ci(x, n)
    d["p"] = signflip(x, n)
    return d


def cos_u(a, b):
    if a is None or b is None:
        return float("nan")
    a = np.asarray(a, float); b = np.asarray(b, float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    return float(a @ b / (na * nb))


def gmean_axis(axes, keys=None):
    vs = [axes[j] for j in (keys if keys is not None else axes) if axes.get(j) is not None]
    if not vs:
        return None
    v = np.mean(vs, axis=0)
    n = np.linalg.norm(v)
    return v / n if n > 0 else None
