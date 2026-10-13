"""Shared functions of the landscape readouts (Fig. 2b,c,f) and the separation floors (Fig. 1f).

Contents
  ridgeline mode counting of Gaussian mixtures   ridgeline_modes_2, ridgeline_modes_K
                                                 (Ray and Lindsay 2005, Ann. Statist. 33:2042-2065)
  regime construction for the equilibrium count  seg_bounds, regimes_from_assignment, count_field
  Gaussian hidden Markov model (Baum-Welch)      fit_gaussian_hmm, GaussianHMM, log_gauss
  one-dimensional Kramers-Moyal drift            kramers_moyal_segments, count_downcrossings
  basin-count readouts of a pooled state cloud   ladder_basins (PCA, Gaussian mixture with K by BIC,
                                                 ridgeline mode count), ising_basins (pairwise maximum
                                                 entropy model, exact fit, basin graph; elapy_functions.py)

The reference state space (RefSpace: robust z on median/IQR and a PCA with random_state 20260728), the
regime fit (fit_linear_regime: VAR(1) -> A = log(M)/dt), the mixture drift field (multilinear_drift), the
equilibrium search (find_equilibria), ridgeline_points and split_runs are those of
analyses/cohort_states/states_lib.py and are re-exported here.

Every function computes exactly what the analysis that produced the stored outputs computed; only names,
docstrings and comments were edited.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from analyses.cohort_states.states_lib import (  # noqa: F401  (re-exported for the scripts of this directory)
    RANDOM_STATE, LinearRegime, RefSpace, find_equilibria, fit_linear_regime, multilinear_drift,
    ridgeline_points, split_runs)

#: dimension of the reference state space of the EEG and MEG cohorts
D = 8
#: one-dimensional Kramers-Moyal settings: number of bins and minimum count per resolved bin
N_BINS_KM, MIN_COUNT_KM = 48, 10


# ============================================================ ridgeline mode counting (Ray-Lindsay)
def _log_mix_density(X, mus, Sigmas, weights):
    lds = [np.linalg.slogdet(np.asarray(S, float))[1] for S in Sigmas]
    Ps = [np.linalg.inv(np.asarray(S, float)) for S in Sigmas]
    d = X.shape[1]
    L = np.empty((len(X), len(mus)))
    for k, (m, P, ld) in enumerate(zip(mus, Ps, lds)):
        dv = X - np.asarray(m, float)
        L[:, k] = (np.log(max(weights[k], 1e-300)) - 0.5 * d * np.log(2 * np.pi)
                   - 0.5 * ld - 0.5 * np.einsum("nk,kl,nl->n", dv, P, dv))
    mx = L.max(axis=1, keepdims=True)
    return mx[:, 0] + np.log(np.exp(L - mx).sum(axis=1))


@dataclass
class RidgelineResult:
    n_modes: int
    barrier_nats: float        # depth of the separating well, in nats
    separation: float          # Mahalanobis separation of the two centres
    alpha_grid: np.ndarray
    logp: np.ndarray
    path: np.ndarray


def ridgeline_modes_2(mu1, S1, mu2, S2, w1=0.5, n=4001) -> RidgelineResult:
    """Exact mode count of a two-component Gaussian mixture.

    Returns the number of modes and the barrier depth (the smaller peak height minus the intervening
    minimum of log p along the ridgeline; 0 for one mode) together with the Mahalanobis separation of
    the two centres under the mean covariance.  Two Gaussians with equal covariance and equal weight are
    bimodal if and only if their Mahalanobis separation exceeds 2 (Behboodian 1970, Technometrics
    12:131-139).
    """
    a = np.linspace(0.0, 1.0, n)
    alphas = np.column_stack([1 - a, a])
    X = ridgeline_points([mu1, mu2], [S1, S2], alphas)
    lp = _log_mix_density(X, [mu1, mu2], [S1, S2], [w1, 1 - w1])

    is_max = np.zeros(n, bool)
    is_max[1:-1] = (lp[1:-1] > lp[:-2]) & (lp[1:-1] > lp[2:])
    is_max[0], is_max[-1] = lp[0] > lp[1], lp[-1] > lp[-2]
    idx = np.where(is_max)[0]
    n_modes = int(len(idx))
    if n_modes >= 2:
        interior = lp[idx[0]:idx[-1] + 1]
        barrier = float(min(lp[idx[0]], lp[idx[-1]]) - interior.min())
    else:
        barrier = 0.0
    dmu = np.asarray(mu2, float) - np.asarray(mu1, float)
    sep = float(np.sqrt(dmu @ np.linalg.solve(0.5 * (np.asarray(S1, float)
                                                     + np.asarray(S2, float)), dmu)))
    return RidgelineResult(n_modes, barrier, sep, a, lp, X)


def _is_local_max(x, mus, Ps, lds, weights, tol=1e-8):
    """Hessian test for log p at a ridgeline fixed point."""
    l = np.array([np.log(max(w, 1e-300)) - 0.5 * ld - 0.5 * (x - m) @ P @ (x - m)
                  for w, m, P, ld in zip(weights, mus, Ps, lds)])
    l -= l.max()
    r = np.exp(l)
    r /= r.sum()
    g = [P @ (m - x) for P, m in zip(Ps, mus)]
    gbar = sum(rk * gk for rk, gk in zip(r, g))
    H = -sum(rk * P for rk, P in zip(r, Ps))
    H = H + sum(rk * np.outer(gk, gk) for rk, gk in zip(r, g)) - np.outer(gbar, gbar)
    return bool(np.linalg.eigvalsh(0.5 * (H + H.T)).max() < tol)


def ridgeline_modes_K(mus, Sigmas, weights, n_sample=4000, seed=0):
    """Mode count for K > 2 components by Dirichlet sampling of the simplex.

    All critical points lie on the (K-1)-dimensional ridgeline, which is sampled; each sample is driven
    to a fixed point of the EM-style ascent map, duplicates are merged, and only points with a
    negative-definite Hessian of log p are kept as modes.
    """
    rng = np.random.default_rng(seed)
    K = len(mus)
    A = np.vstack([rng.dirichlet(np.ones(K), size=n_sample), np.eye(K)])
    X = ridgeline_points(mus, Sigmas, A)
    Ps = [np.linalg.inv(np.asarray(S, float)) for S in Sigmas]
    lds = [np.linalg.slogdet(np.asarray(S, float))[1] for S in Sigmas]
    mus = [np.asarray(m, float) for m in mus]

    def resp(x):
        l = np.array([np.log(max(w, 1e-300)) - 0.5 * ld - 0.5 * (x - m) @ P @ (x - m)
                      for w, m, P, ld in zip(weights, mus, Ps, lds)])
        l -= l.max()
        e = np.exp(l)
        return e / e.sum()

    keep = []
    for x in X:
        for _ in range(80):
            r = resp(x)
            M = sum(rk * P for rk, P in zip(r, Ps))
            b = sum(rk * (P @ m) for rk, P, m in zip(r, Ps, mus))
            xn = np.linalg.solve(M, b)
            if np.linalg.norm(xn - x) < 1e-10:
                x = xn
                break
            x = xn
        if not any(np.linalg.norm(x - y) < 1e-3 * (1 + np.linalg.norm(y)) for y in keep):
            keep.append(x)
    modes = [x for x in keep if _is_local_max(x, mus, Ps, lds, weights)]
    return len(modes), (np.array(modes) if modes else np.zeros((0, len(mus[0]))))


# ============================================================ regime construction for the count
#: Newton starting points of the equilibrium count on every record
N_STARTS = 200
#: seed of the equilibrium count
COUNT_SEED = 20260921


def _runs_of(mask):
    out, cur = [], []
    for i, m in enumerate(mask):
        if m:
            cur.append(i)
        elif cur:
            out.append(np.array(cur)); cur = []
    if cur:
        out.append(np.array(cur))
    return out


def seg_bounds(seg):
    """(starts, lengths) of the segments, which the count cache stores contiguously and in order."""
    seg = np.asarray(seg)
    if len(seg) == 0:
        return np.zeros(0, int), np.zeros(0, int)
    brk = np.flatnonzero(np.diff(seg) != 0) + 1
    starts = np.concatenate([[0], brk])
    ends = np.concatenate([brk, [len(seg)]])
    return starts, ends - starts


def regimes_from_assignment(Z, seg, asg, dt, min_pairs=5):
    """fit_linear_regime per assigned regime, cutting segments at assignment changes.

    With a segment-level assignment (asg constant within every segment) no segment is cut, which is the
    construction of the reference count.  With a sample-level assignment segments are cut, which
    discards the increments that straddle a regime change; the kept fraction is returned.
    """
    regs, parts = [], {}
    starts, lens = seg_bounds(seg)
    head = asg[starts]
    const = bool(np.all(np.add.reduceat(asg.astype(np.int64), starts) == head.astype(np.int64) * lens))
    if const:
        # segment-level assignment: no segment is cut, so the parts are whole segments
        for st, L, k in zip(starts, lens, head):
            if L >= 2:
                parts.setdefault(int(k), []).append(Z[st:st + L])
    else:
        for st, L in zip(starts, lens):
            a = asg[st:st + L]
            for k in np.unique(a):
                for r in _runs_of(a == k):
                    if len(r) >= 2:
                        parts.setdefault(int(k), []).append(Z[st + r[0]: st + r[-1] + 1])
    n_pairs_total = int(np.sum(lens - 1))
    kept = {}
    for k in sorted(parts):
        n_pairs = sum(len(x) - 1 for x in parts[k])
        kept[k] = int(n_pairs)
        if n_pairs >= min_pairs:
            try:
                regs.append(fit_linear_regime(parts[k], f"r{k}", dt=dt))
            except Exception:  # noqa: BLE001
                pass
    info = dict(n_pairs_per_regime={str(k): v for k, v in kept.items()},
                frac_pairs_kept=(float(sum(kept.values()) / n_pairs_total) if n_pairs_total else None))
    return regs, info


def count_field(regs, n_starts=N_STARTS, seed=COUNT_SEED, softness=1.0):
    """Number of equilibria (and of stable ones) of the mixture drift field of the fitted regimes."""
    if len(regs) == 0:
        # no regime had enough within-regime increments to fit: the count is undefined, not zero
        return dict(n_equilibria=None, n_stable=None, n_unstable=None, K_eff=0,
                    degenerate_single_regime=False, undefined_no_regime=True)
    if len(regs) == 1:
        # a one-regime field is linear and has exactly one equilibrium by construction; flagged
        return dict(n_equilibria=1, n_stable=1, n_unstable=0, K_eff=1,
                    degenerate_single_regime=True, undefined_no_regime=False)
    eqs = find_equilibria(regs, softness=softness, n_starts=n_starts, seed=seed)
    return dict(n_equilibria=len(eqs), n_stable=int(sum(e["stable"] for e in eqs)),
                n_unstable=int(sum(not e["stable"] for e in eqs)), K_eff=len(regs),
                degenerate_single_regime=False, undefined_no_regime=False)


# ============================================================ Gaussian hidden Markov model
def _regularise(S, reg):
    d = S.shape[0]
    return S + reg * float(np.trace(S) / d + 1e-12) * np.eye(d)


def log_gauss(X, mu, S):
    """log N(x | mu, S) for every row of X, via Cholesky."""
    d = X.shape[1]
    L = np.linalg.cholesky(S)
    z = np.linalg.solve(L, (X - mu).T)
    return -0.5 * (d * np.log(2 * np.pi) + 2 * np.log(np.diag(L)).sum()
                   + (z ** 2).sum(axis=0))


def _kmeanspp(X, k, rng):
    n = len(X)
    idx = [int(rng.integers(n))]
    d2 = ((X - X[idx[0]]) ** 2).sum(axis=1)
    for _ in range(1, k):
        p = d2 / d2.sum() if d2.sum() > 0 else np.full(n, 1.0 / n)
        j = int(rng.choice(n, p=p))
        idx.append(j)
        d2 = np.minimum(d2, ((X - X[j]) ** 2).sum(axis=1))
    return X[idx].copy()


@dataclass
class GaussianHMM:
    startprob_: np.ndarray
    transmat_: np.ndarray
    means_: np.ndarray
    covars_: np.ndarray
    log_likelihood: float
    n_iter_run: int
    converged: bool

    @property
    def n_states(self):
        return len(self.means_)

    def _log_B(self, X):
        return np.column_stack([log_gauss(X, self.means_[k], self.covars_[k])
                                for k in range(self.n_states)])

    def predict(self, X):
        """Viterbi most-likely state path."""
        X = np.asarray(X, float)
        logA = np.log(self.transmat_ + 1e-300)
        lb = self._log_B(X)
        T, k = lb.shape
        delta = np.log(self.startprob_ + 1e-300) + lb[0]
        psi = np.zeros((T, k), dtype=int)
        for t in range(1, T):
            m = delta[:, None] + logA
            psi[t] = np.argmax(m, axis=0)
            delta = m[psi[t], np.arange(k)] + lb[t]
        path = np.zeros(T, dtype=int)
        path[-1] = int(np.argmax(delta))
        for t in range(T - 2, -1, -1):
            path[t] = psi[t + 1, path[t + 1]]
        return path


def _em_once(X, k, rng, n_iter, tol, reg):
    n, d = X.shape
    means = _kmeanspp(X, k, rng)
    S0 = _regularise(np.cov(X, rowvar=False, ddof=1), reg)
    covars = np.repeat(S0[None], k, axis=0)
    start = np.full(k, 1.0 / k)
    # mild self-transition bias of the initial transition matrix
    trans = 0.9 * np.eye(k) + 0.1 / max(k - 1, 1) * (1 - np.eye(k))

    prev, it, conv = -np.inf, 0, False
    for it in range(1, n_iter + 1):
        lb = np.column_stack([log_gauss(X, means[j], covars[j])
                              for j in range(k)])
        # scaled forward-backward; the per-step normaliser c[t] carries the log-likelihood
        lbmax = lb.max(axis=1)
        B = np.exp(lb - lbmax[:, None])
        A = trans
        alpha = np.empty((n, k))
        cscale = np.empty(n)
        a = start * B[0]
        cscale[0] = a.sum() + 1e-300
        alpha[0] = a / cscale[0]
        for t in range(1, n):
            a = (alpha[t - 1] @ A) * B[t]
            cscale[t] = a.sum() + 1e-300
            alpha[t] = a / cscale[t]
        ll = float(np.log(cscale).sum() + lbmax.sum())
        beta = np.empty((n, k))
        beta[-1] = 1.0
        for t in range(n - 2, -1, -1):
            beta[t] = (A @ (B[t + 1] * beta[t + 1])) / cscale[t + 1]
        g = alpha * beta
        g /= np.maximum(g.sum(axis=1, keepdims=True), 1e-300)
        # xi, vectorised over t
        Wt = (B[1:] * beta[1:]) / cscale[1:, None]           # (n-1, k)
        xi = (alpha[:-1].T @ Wt) * A
        # M-step
        start = g[0] / g[0].sum()
        trans = xi / np.maximum(xi.sum(axis=1, keepdims=True), 1e-300)
        w = g.sum(axis=0)
        for j in range(k):
            wj = max(w[j], 1e-10)
            means[j] = (g[:, j] @ X) / wj
            dv = X - means[j]
            covars[j] = _regularise((g[:, j][:, None] * dv).T @ dv / wj, reg)
        if ll - prev < tol * max(1.0, abs(prev)):
            conv = True
            prev = ll
            break
        prev = ll
    return GaussianHMM(start, trans, means, covars, prev, it, conv)


def fit_gaussian_hmm(X, n_states, n_iter=200, tol=1e-6, reg=1e-4,
                     n_init=5, seed=0):
    """Fit a full-covariance Gaussian HMM by Baum-Welch; best of `n_init` k-means++ starts."""
    X = np.asarray(X, float)
    if X.ndim != 2:
        raise ValueError("X must be (n_samples, n_features)")
    if int(n_states) == 1:
        # a one-state HMM is a single Gaussian
        mu = X.mean(axis=0)
        S = _regularise(np.cov(X, rowvar=False, ddof=1), reg)
        ll = float(log_gauss(X, mu, S).sum())
        return GaussianHMM(np.ones(1), np.ones((1, 1)), mu[None], S[None],
                           ll, 0, True)
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(max(1, n_init)):
        try:
            m = _em_once(X, int(n_states), rng, n_iter, tol, reg)
        except np.linalg.LinAlgError:
            continue
        if best is None or m.log_likelihood > best.log_likelihood:
            best = m
    if best is None:
        raise RuntimeError("every EM start failed (singular covariance)")
    return best


# ============================================================ one-dimensional Kramers-Moyal drift
def kramers_moyal_segments(segments, dt, lag, n_bins=N_BINS_KM):
    """Binned first and second Kramers-Moyal coefficients from increments within segments."""
    xc = np.concatenate([s[:-lag] for s in segments if len(s) > lag])
    dx = np.concatenate([(s[lag:] - s[:-lag]) for s in segments if len(s) > lag])
    edges = np.linspace(xc.min(), xc.max(), n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    idx = np.clip(np.digitize(xc, edges) - 1, 0, n_bins - 1)
    D1 = np.zeros(n_bins); D2 = np.zeros(n_bins); counts = np.zeros(n_bins, dtype=int)
    for i in range(n_bins):
        m = idx == i
        counts[i] = int(m.sum())
        if counts[i]:
            D1[i] = dx[m].mean(); D2[i] = (dx[m] ** 2).mean()
    return centers, D1 / (lag * dt), D2 / (lag * dt), counts


def count_downcrossings(D1, valid):
    """Number of positive-to-negative zero crossings of the smoothed drift between resolved bins (wells)."""
    D1s = np.convolve(np.asarray(D1, float), np.ones(7) / 7, mode="same")
    n = 0
    for i in range(len(D1s) - 1):
        if valid[i] and valid[i + 1] and D1s[i] > 0 and D1s[i + 1] < 0:
            n += 1
    return n


# ============================================================ basin-count readouts of a pooled cloud
def ladder_basins(F, d=8, k_max=6, n_sample=3000, seed=0):
    """Gaussian-mixture readout: PCA -> Gaussian mixture with K chosen by BIC -> ridgeline mode count.

    Returns (n_modes, k_bic), or (nan, nan) when the input has fewer than 20 samples.
    """
    from sklearn.decomposition import PCA
    from sklearn.mixture import GaussianMixture

    F = np.asarray(F, float)
    if len(F) < 20:
        return np.nan, np.nan
    Fz = (F - np.median(F, axis=0)) / np.maximum(np.ptp(F, axis=0), 1e-9)
    Z = PCA(n_components=min(d, F.shape[1]), random_state=seed).fit_transform(Fz)
    bics, models = [], []
    for k in range(1, k_max + 1):
        g = GaussianMixture(n_components=k, covariance_type="full",
                            reg_covar=1e-4, random_state=seed, n_init=3)
        g.fit(Z)
        bics.append(g.bic(Z))
        models.append(g)
    k = int(np.argmin(bics)) + 1
    g = models[k - 1]
    Sigmas = [np.asarray(S, float) for S in g.covariances_]
    n_modes, _ = ridgeline_modes_K(g.means_, Sigmas, g.weights_,
                                   n_sample=n_sample, seed=seed)
    return int(n_modes), k


def ising_basins(X, n_vars_max=15):
    """Pairwise maximum-entropy readout: number of local minima of the basin graph of an exact Ising fit.

    X is (n_vars, n_obs).  Each variable is binarised at its median ({0, 1} data pass unchanged), the
    model is fitted exactly (elapy.fit_exact) and the local minima of the basin graph
    (elapy.calc_basin_graph) are counted.  Returns None when n_vars exceeds n_vars_max.
    """
    import pandas as pd

    from analyses.landscape_readouts.elapy_functions import calc_basin_graph, fit_exact

    X = np.asarray(X)
    n_var = X.shape[0]
    if n_var > n_vars_max:
        return None
    if set(np.unique(X)).issubset({0, 1}):
        B = X.astype(int)
    else:
        B = (X > np.median(X, axis=1, keepdims=True)).astype(int)
    df = pd.DataFrame(B, index=[f"v{i}" for i in range(n_var)])
    h, W = fit_exact(df)
    graph = calc_basin_graph(h, W, df)
    return int((graph["source"] == graph["target"]).sum())
