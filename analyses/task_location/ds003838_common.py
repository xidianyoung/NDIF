# -*- coding: utf-8 -*-
"""Digit-span EEG (ds003838): feature-cache access, the three input definitions u1/u2/u3, and the state-space and
estimator helpers used by ds003838_position_slopes.py.

Contents
  load_cache, recall_curve, u_maps   per-sequence arrays of the feature cache; the inputs u1, u2, u3
  RefSpace, D                        robust z (median / IQR) + PCA to d = 8, fitted on a reference set of windows
  lda_axis, split_runs               OAS-regularised discriminant axis; runs of True in a mask
  detrend_within_condition           degree-3 polynomial session detrend of the features, per condition
  OUFit, fit_ou, bias_correct        VAR(1) propagator of state segments and its bootstrap bias correction

The feature cache (``paths.cache("ds003838")``, environment variable NDIF_DS003838_CACHE) holds one
``sub-XXX.npz`` and one ``sub-XXX_meta.json`` per participant; the format is described in README.md.
No computation is run on import.
"""
from __future__ import annotations
import json, sys, time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import stats as sst
from sklearn.decomposition import PCA

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from analyses.common import paths  # noqa: E402

CACHE = Path(paths.cache("ds003838"))
SEED = 20260921
D = 8                              # latent dimension of the state space
RANDOM_STATE = 20260728            # random_state of the reference PCA
DEG = 3                            # degree of the within-condition session detrend


def log(msg):
    print(time.strftime("[%H:%M:%S] ") + str(msg), flush=True)


def pkg_versions():
    import numpy, scipy, sklearn
    v = dict(python=sys.version.split()[0], numpy=numpy.__version__, scipy=scipy.__version__,
             sklearn=sklearn.__version__)
    try:
        import mne
        v["mne"] = mne.__version__
    except Exception:  # noqa: BLE001
        pass
    return v


# --------------------------------------------------------------------------- feature cache
def load_cache(sub):
    """Per-sequence arrays of the feature cache as lists indexed by sequence."""
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


# --------------------------------------------------------------------------- inputs u
def recall_curve(S):
    """p(k | subject, set size L): per-item recall probability from the MEMORY sequences' own
    per-item outcomes in events.tsv (1 = 'correctly recalled', 0 = 'forgotten').

    Returns {L: array of length L}.  This is the object the primary input u3 is built from:
        u3(k, L) = sum_{j<=k} p(j | L)     ('expected items held', inclusive of the current item)
    which saturates at the participant's own span at that set size.
    """
    out = {}
    for L in sorted(set(S["n_digits"])):
        acc = []
        for i in range(S["n"]):
            if S["cond"][i] == 1 and S["n_digits"][i] == L:
                ok = np.asarray(S["item_ok"][i], float)
                if len(ok) == L and np.all(ok >= 0):
                    acc.append(ok)
        out[L] = np.mean(acc, axis=0) if acc else np.full(L, np.nan)
    return out


def u_maps(S):
    """The three inputs, each as {L: array u[k-1], k = 1..L}.

    u1 = serial position k                      (linear alternative)
    u2 = min(k, 9)
    u3 = cumulative expected items held         (primary)

    The SAME map is applied to the control (listen-only) arm at matched serial position, so the
    memory-minus-control difference in differences is taken between identical reparametrisations of k.
    """
    p = recall_curve(S)
    m = {}
    for L in sorted(set(S["n_digits"])):
        k = np.arange(1, L + 1, dtype=float)
        m[L] = dict(u1=k, u2=np.minimum(k, 9.0), u3=np.cumsum(p[L]))
    return m, p


# --------------------------------------------------------------------------- state space
class RefSpace:
    """Robust z (median / raw IQR) + PCA fitted on the reference condition."""

    def __init__(self, d, seed=RANDOM_STATE):
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


def lda_axis(Z0, Z1):
    from sklearn.covariance import OAS
    dmu = Z1.mean(0) - Z0.mean(0)
    pooled = OAS(assume_centered=True).fit(np.vstack([Z0 - Z0.mean(0), Z1 - Z1.mean(0)])).covariance_
    w = np.linalg.solve(pooled, dmu)
    return w / np.linalg.norm(w)


def split_runs(mask):
    """Index runs of True in a boolean mask -> list of index arrays."""
    runs, cur = [], []
    for i, m in enumerate(mask):
        if m:
            cur.append(i)
        elif cur:
            runs.append(np.array(cur)); cur = []
    if cur:
        runs.append(np.array(cur))
    return runs


def detrend_within_condition(blocks) -> None:
    """Regress a degree-3 polynomial in session time out of F_dist, per condition.

    The fit is per feature and per condition, so it removes what drifts across
    the session inside each condition and leaves the between-condition
    difference, which is the contrast, untouched.
    """
    for cond in (0, 1):
        bs = [b for b in blocks if b["cond"] == cond]
        if not bs:
            continue
        t = np.concatenate([b["t_start"] for b in bs]).astype(float)
        F = np.vstack([b["F_dist"] for b in bs])
        tc = (t - t.mean()) / (t.std() + 1e-12)
        V = np.vander(tc, DEG + 1, increasing=True)        # 1, t, t^2, t^3
        coef, *_ = np.linalg.lstsq(V, F, rcond=None)
        R = F - V @ coef + F.mean(axis=0)                  # keep the condition mean
        i = 0
        for b in bs:
            n = b["F_dist"].shape[0]
            b["F_dist"] = R[i:i + n]
            i += n
        assert i == R.shape[0], (i, R.shape)


# --------------------------------------------------------------------------- VAR(1) / OU fit
@dataclass
class OUFit:
    M: np.ndarray            # discrete one-step propagator, exp(A dt)
    Sigma_eps: np.ndarray    # one-step residual covariance
    Sigma0: np.ndarray       # lag-0 (stationary) covariance
    Sigma1: np.ndarray       # lag-1 cross-covariance E[(x_t+1-mu)(x_t-mu)']
    dt: float
    n_pairs: int

    @property
    def spectral_radius(self):
        return float(np.max(np.abs(np.linalg.eigvals(self.M))))

    @property
    def lambda_recovery(self):
        """Slowest relaxation rate, -max Re eig(A) = -log(rho(M))/dt."""
        rho = self.spectral_radius
        if not (0 < rho < 1):
            return np.nan
        return float(-np.log(rho) / self.dt)

    @property
    def lambda_mean(self):
        """Geometric-mean relaxation rate, -(1/d) sum_k log|eig_k(M)| / dt (= -(1/d) tr(A)), units 1/s."""
        ev = np.abs(np.linalg.eigvals(self.M))
        ev = ev[ev > 0]
        if len(ev) == 0 or np.any(ev >= 1):
            return np.nan
        return float(-np.mean(np.log(ev)) / self.dt)


def fit_ou(segments, dt, lag=1, ridge=1e-6):
    """Fit a VAR(1) propagator from one or more continuous state segments.

    `segments` is a list of (n_t, d) arrays.  Each is centred on its own mean
    before pooling, so a pooled fit describes the shared relaxation dynamics
    rather than displacements between segments.  Lagged pairs are never formed
    across a segment boundary.
    """
    S0 = S1 = None
    npair = 0
    d = segments[0].shape[1]
    S0 = np.zeros((d, d))
    S1 = np.zeros((d, d))
    ntot = 0
    for X in segments:
        X = np.asarray(X, dtype=float)
        if len(X) < lag + 3:
            continue
        Xc = X - X.mean(axis=0)
        a, b = Xc[:-lag], Xc[lag:]
        S0 += Xc.T @ Xc
        S1 += b.T @ a
        ntot += len(Xc)
        npair += len(a)
    if npair < d + 2:
        return None
    S0 /= max(ntot - 1, 1)
    S1 /= max(npair - 1, 1)
    reg = ridge * np.trace(S0) / d * np.eye(d)
    M = S1 @ np.linalg.inv(S0 + reg)
    Seps = S0 - M @ S0 @ M.T
    return OUFit(M=M, Sigma_eps=0.5 * (Seps + Seps.T), Sigma0=S0, Sigma1=S1,
                 dt=dt * lag, n_pairs=npair)


def _simulate_var1_batch(M, L, n, B, rng, burn=200):
    """Vectorised simulation of B independent VAR(1) paths. Returns (B, n, d)."""
    d = M.shape[0]
    x = np.zeros((B, d))
    for _ in range(burn):
        x = x @ M.T + rng.standard_normal((B, d)) @ L.T
    X = np.empty((B, n, d))
    for t in range(n):
        x = x @ M.T + rng.standard_normal((B, d)) @ L.T
        X[:, t] = x
    return X


def _refit_M_batch(X, ridge=1e-6):
    """Least-squares VAR(1) propagator for each path in a (B, n, d) batch."""
    Xc = X - X.mean(axis=1, keepdims=True)
    a, b = Xc[:, :-1], Xc[:, 1:]
    S0 = np.einsum("bti,btj->bij", Xc, Xc) / (X.shape[1] - 1)
    S1 = np.einsum("bti,btj->bij", b, a) / (a.shape[1] - 1)
    d = X.shape[2]
    reg = ridge * (np.trace(S0, axis1=1, axis2=2) / d)[:, None, None] * np.eye(d)
    return S1 @ np.linalg.inv(S0 + reg)


def bias_correct(fit: OUFit, n_boot=200, rng=None):
    """Bootstrap bias correction of the VAR(1) propagator, and lambda from it.

    Least-squares estimates of autoregressive coefficients are biased toward
    zero (of order 1/n), which shrinks |eig(M)| and therefore inflates any
    relaxation rate computed as -log|eig|; the spectral radius, a maximum over
    d noisy eigenvalues, is inflated by their sampling spread.  Both originate
    in M-hat, so both are addressed by correcting M-hat itself: simulating from
    M-hat and re-estimating gives E[M*|M-hat]; the bootstrap bias estimate is
    E[M*] - M-hat, and the corrected estimate is M_bc = 2 M-hat - E[M*].
    """
    rng = np.random.default_rng(rng)
    d = fit.M.shape[0]
    Se = 0.5 * (fit.Sigma_eps + fit.Sigma_eps.T)
    w, Vv = np.linalg.eigh(Se)
    L = Vv @ np.diag(np.sqrt(np.maximum(w, 0))) @ Vv.T
    n = max(fit.n_pairs + 1, 4 * d)

    X = _simulate_var1_batch(fit.M, L, n, n_boot, rng)
    M_star = _refit_M_batch(X).mean(axis=0)
    M_bc = 2 * fit.M - M_star

    def rates(M):
        ev = np.abs(np.linalg.eigvals(M))
        ev_ok = ev[(ev > 0) & (ev < 1)]
        lam_mean = float(-np.mean(np.log(ev_ok)) / fit.dt) if len(ev_ok) == d else np.nan
        rho = float(np.max(ev))
        lam_max = float(-np.log(rho) / fit.dt) if 0 < rho < 1 else np.nan
        return lam_mean, lam_max

    lm_bc, lx_bc = rates(M_bc)
    lm_raw, lx_raw = rates(fit.M)
    return dict(M_bc=M_bc, lambda_mean_bc=lm_bc, lambda_max_bc=lx_bc,
                lambda_mean_raw=lm_raw, lambda_max_raw=lx_raw,
                bias_fro=float(np.linalg.norm(M_star - fit.M, "fro")))
