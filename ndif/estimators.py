"""Estimators used by the gates: propagator, relaxation rates, white share.

All of these are standard; they are collected here so that a certificate can
name the exact estimator it was computed with, and so that the failure modes
documented in the NDIF paper attach to a specific implementation rather than
to "an AR(1) fit".

The observation model
---------------------
Throughout, the observed state series is modelled as a latent linear
(Ornstein-Uhlenbeck / VAR(1)) process plus an additive white component:

    z_t = y_t + e_t ,    y_t = a y_{t-1} + xi_t ,    e_t ~ white ,

so the observed autocorrelation is ``r(k) = (1 - nu) a^k`` for ``k >= 1``,
where ``nu`` is the share of the observed variance carried by the white
component.  Two consequences drive gates G1 and G2:

* ``nu`` is identified from two lags alone: ``a = r(2)/r(1)`` and
  ``1 - nu = r(1)^2 / r(2)``.
* the naive one-lag rate is inflated,
  ``lambda_eff = lambda_true - ln(1 - nu) / dt``,
  so a rate read off ``-ln r(1)/dt`` reports the instrument as much as the
  system.  This is the quantity gate G2 tests for.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.linalg import solve_lyapunov

__all__ = [
    "PropagatorFit",
    "as_segments",
    "fit_propagator",
    "relaxation_rates",
    "bootstrap_bias_correct",
    "white_share",
    "stationary_covariance",
    "lyapunov_residual",
    "directional_rate",
    "fit_ar1_kalman_batch",
    "kalman2_ll_batch",
    "lr_two_timescale_padded",
    "whittle_lr_two_timescale",
    "fit_abc",
    "fit_model",
    "fit_rates_padded",
    "simulate_eq6_padded",
    "pooled_acf_padded",
]


# --------------------------------------------------------------------------- io
def as_segments(X, segments=None) -> list:
    """Normalise input to a list of (n_t, d) contiguous blocks.

    A segment is a stretch of samples between which lag-1 pairs are legitimate.
    Trials, runs and sleep epochs are separate segments; forming pairs across a
    boundary would mix unrelated samples.
    """
    if isinstance(X, (list, tuple)) and not isinstance(X, np.ndarray):
        segs = [np.atleast_2d(np.asarray(s, float)) for s in X]
    else:
        X = np.asarray(X, float)
        if X.ndim == 1:
            X = X[:, None]
        if segments is None:
            segs = [X]
        else:
            segments = np.asarray(segments)
            if segments.ndim == 1 and segments.size == X.shape[0]:
                segs = [X[segments == u] for u in np.unique(segments)]
            else:
                segs = [X[int(a):int(b)] for a, b in segments]
    segs = [s for s in segs if s.shape[0] >= 2]
    if not segs:
        raise ValueError("no segment has at least two samples")
    d = segs[0].shape[1]
    if any(s.shape[1] != d for s in segs):
        raise ValueError("segments differ in dimension")
    return segs


@dataclass
class PropagatorFit:
    """One-step propagator fit of a linear stochastic process."""

    A: np.ndarray          # one-step propagator (dimensionless)
    Q: np.ndarray          # innovation covariance
    dt: float
    n_pairs: int
    n_segments: int
    demean: str
    d: int

    @property
    def M(self) -> np.ndarray:
        """Continuous-time drift matrix, ``A = expm(M dt)``."""
        from scipy.linalg import logm
        return np.real(logm(self.A)) / self.dt

    @property
    def D(self) -> np.ndarray:
        """Diffusion matrix implied by the innovation covariance.

        For ``dy = M y dt + sqrt(2 D) dW`` discretised at ``dt``,
        ``Q = Sigma - A Sigma A^T``; to first order in ``dt``,
        ``D ~= Q / (2 dt)``.
        """
        return self.Q / (2.0 * self.dt)

    def rates(self) -> np.ndarray:
        return relaxation_rates(self.A, self.dt)


def fit_propagator(X, segments=None, dt=1.0, demean="block", ridge=0.0) -> PropagatorFit:
    """Least-squares one-step propagator, pooled over segments.

    Parameters
    ----------
    demean : {"block", "global", "none"}
        ``"block"`` removes each segment's own mean.  This is what most
        pipelines do and it is *not* neutral: it attenuates the propagator by
        roughly ``1 - c/n_block`` and adds an upward bias to the implied rate
        that the bootstrap below does not remove, because the bootstrap
        regenerates a clean process whose long-run variance is already too
        small.  The size of that residual bias is a function of block length,
        which is why a certificate records the block length it was computed at.
    ridge : float
        Tikhonov term on the lag-0 moment; 0 disables.

    Notes
    -----
    Pairs are never formed across a segment boundary.
    """
    segs = as_segments(X, segments)
    d = segs[0].shape[1]
    if demean == "global":
        mu = np.concatenate(segs, axis=0).mean(axis=0)
        segs = [s - mu for s in segs]
    elif demean == "block":
        segs = [s - s.mean(axis=0) for s in segs]
    elif demean not in ("none", None):
        raise ValueError(f"unknown demean={demean!r}")

    C0 = np.zeros((d, d))
    C1 = np.zeros((d, d))
    n_pairs = 0
    for s in segs:
        a, b = s[:-1], s[1:]
        C0 += a.T @ a
        C1 += b.T @ a
        n_pairs += a.shape[0]
    C0 /= n_pairs
    C1 /= n_pairs
    if ridge:
        C0 = C0 + ridge * np.trace(C0) / d * np.eye(d)
    A = C1 @ np.linalg.pinv(C0)

    resid = []
    for s in segs:
        resid.append(s[1:] - s[:-1] @ A.T)
    R = np.concatenate(resid, axis=0)
    Q = R.T @ R / max(R.shape[0] - d, 1)
    return PropagatorFit(A=A, Q=Q, dt=float(dt), n_pairs=int(n_pairs),
                         n_segments=len(segs), demean=str(demean), d=d)


def relaxation_rates(A, dt=1.0) -> np.ndarray:
    """Relaxation rates ``lambda_k = -log|mu_k| / dt`` from propagator eigenvalues.

    Returned sorted ascending (slowest mode first).  Non-positive or complex
    eigenvalues are handled by taking the modulus, which is the usual
    convention and is exact for a real diagonalisable propagator.
    """
    mu = np.linalg.eigvals(np.asarray(A, float))
    m = np.abs(mu)
    m = np.clip(m, 1e-300, None)
    return np.sort(-np.log(m) / float(dt))


def bootstrap_bias_correct(fit: PropagatorFit, n_boot=200, rng=None, clip=0.999):
    """Bootstrap bias correction of the propagator.

    Simulates from the fitted process at the fitted segment structure, refits,
    and returns ``2 A_hat - mean(A_boot)``.

    Boundary of validity, which the certificate records.  This removes the
    finite-sample (Kendall-type) bias of the least-squares propagator.  It does
    **not** remove bias from block-mean centring or from an additive white
    component, because the bootstrap regenerates a process that has neither.
    Where those dominate, the corrected estimate is still biased and the
    correction can make it worse; G2 and G1 are what detect that.
    """
    rng = np.random.default_rng(rng)
    A, Q, d = fit.A, fit.Q, fit.d
    # segment lengths are not stored; use the average implied block length
    n_blk = max(2, int(round(fit.n_pairs / max(fit.n_segments, 1))) + 1)
    L = np.linalg.cholesky(Q + 1e-12 * np.trace(Q) / d * np.eye(d))
    acc = np.zeros((d, d))
    good = 0
    for _ in range(int(n_boot)):
        n = n_blk + 200
        Y = np.empty((n, fit.n_segments, d))
        Y[0] = rng.normal(size=(fit.n_segments, d)) @ L.T
        E = rng.normal(size=(n - 1, fit.n_segments, d)) @ L.T
        for t in range(1, n):
            Y[t] = Y[t - 1] @ A.T + E[t - 1]
        segs = [Y[200:, i, :] for i in range(fit.n_segments)]
        try:
            b = fit_propagator(segs, dt=fit.dt, demean=fit.demean)
        except Exception:
            continue
        acc += b.A
        good += 1
    if good == 0:
        return fit.A.copy()
    A_bc = 2.0 * A - acc / good
    # keep the corrected propagator inside the stable disc
    ev = np.abs(np.linalg.eigvals(A_bc)).max()
    if ev > clip:
        A_bc = A_bc * (clip / ev)
    return A_bc


def white_share(X, segments=None, max_lag=2, axis=None, tol_nu=0.02):
    """Share of observed variance carried by an additive white component.

    Uses the two-lag identity ``a = r(2)/r(1)``, ``1 - nu = r(1)^2 / r(2)``.

    Parameters
    ----------
    axis : (d,) array_like, optional
        Project onto this direction before estimating.  ``nu`` is
        direction-dependent; a task axis typically carries a *lower* white
        share than the orthogonal directions of the same state, so estimating
        on a task axis alone reports the most favourable value available.
        The certificate records which axis was used.

    Returns
    -------
    dict with ``nu``, ``a``, ``r1``, ``r2``, ``ok``.  ``ok`` is False when the
    two-lag solution leaves the admissible range (``r(2) <= 0`` or
    ``nu`` outside [0, 1)), which happens when the single-timescale model is
    wrong -- itself diagnostic, and reported rather than clipped away.
    """
    segs = as_segments(X, segments)
    if axis is not None:
        e = np.asarray(axis, float).ravel()
        e = e / np.linalg.norm(e)
        segs = [(s @ e)[:, None] for s in segs]
    segs = [s - s.mean(axis=0) for s in segs]

    def r_at(k):
        num = den = 0.0
        for s in segs:
            if s.shape[0] <= k:
                continue
            num += float(np.sum(s[k:] * s[:-k]))
            den += float(np.sum(s * s))
        return num / den if den > 0 else np.nan

    r1, r2 = r_at(1), r_at(2)
    out = dict(r1=float(r1), r2=float(r2), nu=np.nan, a=np.nan, ok=False)
    if not np.isfinite(r1) or not np.isfinite(r2) or r1 <= 0 or r2 <= 0:
        return out
    a = r2 / r1
    nu = 1.0 - r1 * r1 / r2
    # nu = 0 is a boundary of the admissible range, so a sampling fluctuation
    # puts a genuinely white-free recording slightly below it.  Clip a small
    # negative excursion rather than declaring the model violated; anything
    # beyond the tolerance is a real violation and is reported as one.
    if -tol_nu <= nu < 0.0:
        nu = 0.0
    out.update(a=float(a), nu=float(nu))
    out["ok"] = bool(0.0 <= nu < 1.0 and 0.0 < a < 1.0)
    return out


# ------------------------------------------------------------ rate estimators
def kalman_ll(Y, a, sw2, se2):
    """Exact Gaussian log-likelihood of ``y_t = x_t + e_t``, ``x_t = a x_{t-1} + w_t``.

    ``Y`` is (n_t, n_seg); segments are independent and advance together.
    """
    a = float(a)
    sw2 = max(float(sw2), 1e-300)
    se2 = max(float(se2), 1e-300)
    n_t, n_seg = Y.shape
    P0 = sw2 / (1 - a * a) if abs(a) < 1 else 1e6
    x = np.zeros(n_seg)
    P = np.full(n_seg, P0)
    ll = 0.0
    for t in range(n_t):
        if t > 0:
            x = a * x
            P = a * a * P + sw2
        S = P + se2
        v = Y[t] - x
        ll += float(np.sum(-0.5 * (np.log(2 * np.pi * S) + v * v / S)))
        K = P / S
        x = x + K * v
        P = P * (1.0 - K)
    return ll


def _moment_init(Y):
    """Two-lag method-of-moments start for the Kalman fit.

    ``a = r(2)/r(1)`` and ``1 - nu = r(1)^2/r(2)`` are closed form, so the
    optimiser starts near the answer instead of searching for it.  This is what
    makes the maximum-likelihood estimator cheap enough to run inside a gate.
    """
    Z = Y - Y.mean(axis=0, keepdims=True)
    den = float(np.sum(Z * Z))
    if den <= 0:
        return 0.8, 0.5, 0.5
    r1 = float(np.sum(Z[1:] * Z[:-1])) / den
    r2 = float(np.sum(Z[2:] * Z[:-2])) / den
    v = float(np.var(Z))
    if not (0 < r1 and 0 < r2 < 1):
        return 0.8, 0.5 * v, 0.5 * v
    a = float(np.clip(r2 / r1, 0.02, 0.995))
    nu = float(np.clip(1.0 - r1 * r1 / r2, 0.0, 0.95))
    # keep both variances off the floor: log-parameterised Nelder-Mead spends
    # thousands of evaluations crawling out of a corner at log(1e-9)
    se2 = max(nu * v, 1e-3 * v)
    sw2 = max((1 - nu) * v * (1 - a * a), 1e-3 * v)
    return a, sw2, se2


def fit_ar1_kalman(segments_1d, a_fixed=None, n_restart=2, seed=0, maxiter=400, equal_length="group"):
    """Maximum-likelihood AR(1)-plus-white fit, by Kalman filter.

    This is the *observation-model-aware* estimator.  Unlike least squares on
    lag-1 pairs it separates the latent autocorrelated component from an
    additive white one, so an instrument that injects white variance does not
    enter the rate.  Returns ``a``, ``sw2``, ``se2``, the implied ``nu`` and
    the log-likelihood.

    Every segment is centred on its own mean.  With ``equal_length="group"``
    each segment enters the likelihood at its full length: segments are grouped
    by length and the log-likelihoods of the groups are summed.  With
    ``equal_length="truncate"`` (the generator fit of the recovery check in the
    primary configuration) all segments are cut to the length of the shortest
    one.  With segments of equal length the two are the same computation.
    """
    from scipy.optimize import minimize
    rng = np.random.default_rng(seed)
    segs = [np.asarray(s, float).ravel() for s in segments_1d]
    if equal_length == "truncate":
        n = min(len(s) for s in segs)
        Y = np.column_stack([s[:n] - s[:n].mean() for s in segs])
        mats = [Y]
        v = float(np.var(Y))
    else:
        groups = {}
        for s in segs:
            groups.setdefault(len(s), []).append(s - s.mean())
        mats = [np.column_stack(g) for g in groups.values()]
        if len(mats) == 1:
            Y = mats[0]
            v = float(np.var(Y))
        else:
            # starting values only: moments of the segments cut to the shortest length
            n = min(groups)
            Y = np.column_stack([s[:n] - s[:n].mean() for s in segs])
            v = float(np.var(np.concatenate([m.ravel() for m in mats])))

    def unpack(z):
        if a_fixed is not None:
            return float(a_fixed), np.exp(z[0]), np.exp(z[1])
        return 1.0 / (1.0 + np.exp(-z[0])), np.exp(z[1]), np.exp(z[2])

    def nll(z):
        a, sw2, se2 = unpack(z)
        if not (0 < a < 0.99999):
            return 1e12
        val = -sum(kalman_ll(m, a, sw2, se2) for m in mats)
        return val if np.isfinite(val) else 1e12

    a_mm, sw2_mm, se2_mm = _moment_init(Y)
    starts = [[np.log(max(sw2_mm, 1e-9)), np.log(max(se2_mm, 1e-9))]]
    if a_fixed is None:
        starts[0] = [np.log(a_mm / (1 - a_mm))] + starts[0]
    for i in range(1, int(n_restart)):
        frac = (0.5, 0.2, 0.8)[i % 3]
        z0 = [np.log(max(v * frac, 1e-9)), np.log(max(v * (1 - frac), 1e-9))]
        if a_fixed is None:
            a0 = (0.8, 0.5, 0.95)[i % 3]
            z0 = [np.log(a0 / (1 - a0))] + z0
        starts.append(list(np.asarray(z0) + rng.normal(scale=0.05, size=len(z0))))

    best = None
    for z0 in starts:
        r = minimize(nll, np.asarray(z0, float), method="Nelder-Mead",
                     options=dict(maxiter=int(maxiter), xatol=1e-6, fatol=1e-8))
        if best is None or r.fun < best.fun:
            best = r
        if best.fun < 1e11 and len(starts) > 1 and z0 is starts[0]:
            # the moment start converged; a blind restart rarely improves it
            break
    a, sw2, se2 = unpack(best.x)
    stat_var = sw2 / max(1 - a * a, 1e-9)
    return dict(a=float(a), sw2=float(sw2), se2=float(se2),
                nu=float(se2 / max(se2 + stat_var, 1e-300)),
                ll=float(-best.fun))


def kalman_ll_batch(Y, a, sw2, se2, mask=None):
    """:func:`kalman_ll` for many datasets at once.

    ``Y`` is (n_t, n_data, n_seg); ``a``, ``sw2`` and ``se2`` are (n_data,).  Each dataset's log-likelihood is
    computed with the operations of :func:`kalman_ll` (the filter variance does not depend on the data, so it is
    one number per dataset and time step).  ``mask`` (n_t, n_seg) of 0/1 marks the samples that exist when
    segments of unequal length are padded to a common length: every segment starts at t = 0, so the filter
    variance is still shared, and padded samples contribute nothing.  Without a mask the result is bit-identical to
    :func:`kalman_ll`; with one it differs only by the order of summation.
    """
    a = np.asarray(a, float)
    sw2 = np.maximum(np.asarray(sw2, float), 1e-300)
    se2 = np.maximum(np.asarray(se2, float), 1e-300)
    n_t, n, n_seg = Y.shape
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        P = np.where(np.abs(a) < 1, sw2 / (1 - a * a), 1e6)
    x = np.zeros((n, n_seg))
    ll = np.zeros(n)
    two_pi = 2 * np.pi
    for t in range(n_t):
        if t > 0:
            x = a[:, None] * x
            P = a * a * P + sw2
        S = P + se2
        v = Y[t] - x
        term = -0.5 * (np.log(two_pi * S)[:, None] + v * v / S[:, None])
        if mask is not None:
            term = term * mask[t][None, :]
        ll += np.sum(term, axis=1)
        K = P / S
        x = x + K[:, None] * v
        P = P * (1.0 - K)
    return ll


def _nelder_mead_batch(f, X0, maxiter, xatol, fatol):
    """scipy's Nelder-Mead (non-adaptive, unbounded), run independently for every row of ``X0``.

    ``f(idx, Z)`` returns the objective of dataset ``idx[i]`` at ``Z[i]``.  Each row follows exactly the
    sequence of operations of ``scipy.optimize.minimize(method="Nelder-Mead")`` with the same initial simplex,
    coefficients, ordering and termination rule; only the function evaluations are grouped across rows.
    Returns the best vertex, its value and the number of iterations of every row.
    """
    rho, chi, psi, sigma = 1.0, 2.0, 0.5, 0.5
    nonzdelt, zdelt = 0.05, 0.00025
    X0 = np.asarray(X0, float)
    n, N = X0.shape
    sim = np.empty((n, N + 1, N))
    sim[:, 0] = X0
    for k in range(N):
        y = X0.copy()
        nz = y[:, k] != 0
        y[nz, k] = (1 + nonzdelt) * y[nz, k]
        y[~nz, k] = zdelt
        sim[:, k + 1] = y
    idx_all = np.arange(n)
    fsim = f(np.repeat(idx_all, N + 1), sim.reshape(n * (N + 1), N)).reshape(n, N + 1)

    def _sort(rows):
        ind = np.argsort(fsim[rows], axis=1)
        fsim[rows] = np.take_along_axis(fsim[rows], ind, axis=1)
        sim[rows] = np.take_along_axis(sim[rows], ind[:, :, None], axis=1)

    _sort(idx_all)
    it = np.ones(n, int)
    active = np.ones(n, bool)
    err = np.seterr(over="ignore", invalid="ignore")
    try:
        return _nm_loop(f, sim, fsim, it, active, N, maxiter, xatol, fatol, rho, chi, psi, sigma, _sort)
    finally:
        np.seterr(**err)


def _nm_loop(f, sim, fsim, it, active, N, maxiter, xatol, fatol, rho, chi, psi, sigma, _sort):
    """The iteration of :func:`_nelder_mead_batch` (separated so that floating-point warnings of diverging simplices
    are silenced; the penalty value keeps such simplices away from the optimum)."""
    while True:
        act = np.flatnonzero(active & (it < maxiter))
        if act.size == 0:
            break
        s, fs = sim[act], fsim[act]
        conv = ((np.max(np.abs(s[:, 1:] - s[:, :1]).reshape(act.size, -1), axis=1) <= xatol)
                & (np.max(np.abs(fs[:, :1] - fs[:, 1:]), axis=1) <= fatol))
        active[act[conv]] = False
        act = act[~conv]
        if act.size == 0:
            break
        s, fs = sim[act], fsim[act]
        xbar = np.add.reduce(s[:, :-1], 1) / N
        worst = s[:, -1]
        xr = (1 + rho) * xbar - rho * worst
        fxr = f(act, xr)
        new_x = xr.copy()
        new_f = fxr.copy()
        shrink = np.zeros(act.size, bool)
        m_e = fxr < fs[:, 0]
        if m_e.any():
            xe = (1 + rho * chi) * xbar[m_e] - rho * chi * worst[m_e]
            fxe = f(act[m_e], xe)
            take = fxe < fxr[m_e]
            ie = np.flatnonzero(m_e)[take]
            new_x[ie] = xe[take]
            new_f[ie] = fxe[take]
        m_rest = ~m_e
        m_r = m_rest & (fxr < fs[:, -2])
        m_c = m_rest & ~(fxr < fs[:, -2])
        m_oc = m_c & (fxr < fs[:, -1])
        m_ic = m_c & ~(fxr < fs[:, -1])
        if m_oc.any():
            xc = (1 + psi * rho) * xbar[m_oc] - psi * rho * worst[m_oc]
            fxc = f(act[m_oc], xc)
            ok = fxc <= fxr[m_oc]
            io = np.flatnonzero(m_oc)
            new_x[io[ok]] = xc[ok]
            new_f[io[ok]] = fxc[ok]
            shrink[io[~ok]] = True
        if m_ic.any():
            xcc = (1 - psi) * xbar[m_ic] + psi * worst[m_ic]
            fxcc = f(act[m_ic], xcc)
            ok = fxcc < fs[m_ic, -1]
            ii = np.flatnonzero(m_ic)
            new_x[ii[ok]] = xcc[ok]
            new_f[ii[ok]] = fxcc[ok]
            shrink[ii[~ok]] = True
        repl = ~shrink                       # rows whose worst vertex is replaced (reflection, expansion, contraction)
        rows = act[repl]
        sim[rows, -1] = new_x[repl]
        fsim[rows, -1] = new_f[repl]
        if shrink.any():
            rs = act[shrink]
            best = sim[rs, :1]
            sim[rs, 1:] = best + sigma * (sim[rs, 1:] - best)
            fv = f(np.repeat(rs, N), sim[rs, 1:].reshape(rs.size * N, N)).reshape(rs.size, N)
            fsim[rs, 1:] = fv
        it[act] += 1
        _sort(act)
    return sim[:, 0].copy(), np.min(fsim, axis=1), it


def padded_centred(datasets):
    """Datasets of one geometry as a (L_max, n_data, n_seg) array of segments centred on their own means.

    Samples beyond a segment's length are zero.  Returns the array and the segment lengths.
    """
    lens = [len(np.asarray(s).ravel()) for s in datasets[0]]
    Y = np.zeros((max(lens), len(datasets), len(lens)))
    for j, ds in enumerate(datasets):
        segs = [np.asarray(s, float).ravel() for s in ds]
        if [len(s) for s in segs] != lens:
            raise ValueError("every dataset must have the same sequence of segment lengths")
        for i, s in enumerate(segs):
            Y[:len(s), j, i] = s - s.mean()
    return Y, lens


def simulate_eq6_padded(a, nu, lens, n, rng):
    """``n`` recordings of equation (6) at the segment lengths ``lens``, as for the checks.

    The latent process has unit stationary variance and persistence ``a`` (a scalar or one value per recording);
    uncorrelated noise makes up the fraction ``nu`` of the total variance.  Every segment is generated at the longest
    length from a stationary start and cut to its own length (as in :func:`ndif.gates._simulate`), then centred on
    its own mean.  Returns the (L_max, n, n_seg) array, zero beyond each segment's length.
    """
    rng = np.random.default_rng(rng)
    lens = [int(L) for L in lens]
    L, m = max(lens), len(lens)
    a = np.broadcast_to(np.asarray(a, float), (n,))
    Y = np.empty((L, n, m))
    Y[0] = rng.normal(size=(n, m))
    sd_in = np.sqrt(np.maximum(1.0 - a * a, 1e-12))[:, None]
    E = rng.normal(size=(L, n, m))
    for t in range(1, L):
        Y[t] = a[:, None] * Y[t - 1] + sd_in * E[t]
    if nu > 0:
        Y = Y + rng.normal(scale=np.sqrt(nu / max(1.0 - nu, 1e-9)), size=Y.shape)
    mask = (np.arange(L)[:, None] < np.asarray(lens)[None, :]).astype(float)
    Y = Y * mask[:, None, :]
    mean = Y.sum(axis=0) / np.asarray(lens, float)[None, :]
    return (Y - mean[None]) * mask[:, None, :]


def ols_persistence_padded(Y, lens):
    """Least-squares lag-1 factor of every recording of a padded, centred array (as :func:`fit_rate` ``"ols"``)."""
    L = Y.shape[0]
    nxt = (np.arange(1, L + 1)[:, None] < np.asarray(lens)[None, :]).astype(float)   # sample t has a successor
    num = np.einsum("tns,tns->n", Y[1:], Y[:-1])
    den = np.einsum("tns,tns->n", Y * Y, np.broadcast_to(nxt[:, None, :], Y.shape))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def pooled_acf_padded(Y, max_lag):
    """Segment-pooled autocorrelation at lags 1..max_lag of every recording of a padded, centred array.

    Returns (n_data, max_lag) autocorrelations and the number of pooled pairs per lag (the same for every
    recording).  Matches the per-segment computation used by G8 and by approximate Bayesian computation.
    """
    L = Y.shape[0]
    c0 = np.einsum("tns,tns->n", Y, Y)
    K = int(max_lag)
    num = np.empty((Y.shape[1], K))
    for k in range(1, K + 1):
        num[:, k - 1] = np.einsum("tns,tns->n", Y[k:], Y[:L - k]) if k < L else 0.0
    with np.errstate(invalid="ignore", divide="ignore"):
        r = num / c0[:, None]
    return r


def _data_seed(segs):
    """A seed derived from the data, so that a stochastic estimator returns the same value for the same data."""
    import hashlib
    b = np.ascontiguousarray(np.concatenate([np.asarray(s, float).ravel() for s in segs])).tobytes()
    return int.from_bytes(hashlib.blake2b(b, digest_size=8).digest(), "little")


#: Draws simulated together by :func:`fit_abc` (a fixed part of its definition: the random stream is consumed in
#: blocks of this many draws).
_ABC_CHUNK = 100


def fit_abc(segments_1d, dt=1.0, n_sim=600, n_keep=60, rounds=2, max_lag=8, seed=None,
            a_range=(0.02, 0.995), nu_range=(0.0, 0.9)):
    """Approximate Bayesian computation for the relaxation rate under equation (6).

    The estimator of the comparison of established methods (Zeraati et al. 2022, with a noise term): parameter
    draws are simulated at the recording's own segment lengths, every simulated segment is centred on its own mean
    as the recording is, and the draws whose pooled autocorrelation at lags ``1..max_lag`` is closest to the
    recording's (mean squared difference) are retained, ``n_keep`` of ``n_sim`` in each of ``rounds`` rounds; the
    second round resamples around the retained set.  Priors are uniform on ``a_range`` (persistence) and
    ``nu_range`` (noise fraction).  Returns the median rate of the retained draws, per unit of ``dt``, together with
    the retained draws.  Unless ``seed`` is given, the random stream is derived from the data, so that the estimate is
    a function of the data.
    """
    segs = [np.asarray(s, float).ravel() for s in segments_1d]
    segs = [s for s in segs if len(s) >= 2]
    if not segs:
        return dict(lam=np.nan, a=np.nan, nu=np.nan)
    lens = [len(s) for s in segs]
    Yobs, _ = padded_centred([segs])
    obs = pooled_acf_padded(Yobs, max_lag)[0]
    if not np.all(np.isfinite(obs)):
        return dict(lam=np.nan, a=np.nan, nu=np.nan)
    rng = np.random.default_rng(_data_seed(segs) if seed is None else seed)
    lo_a, hi_a = a_range
    lo_n, hi_n = nu_range
    post = None
    for _ in range(int(rounds)):
        if post is None:
            A = rng.uniform(lo_a, hi_a, n_sim)
            NU = rng.uniform(lo_n, hi_n, n_sim)
        else:
            pa, pn = post
            A = np.clip(rng.choice(pa, n_sim) + rng.normal(0, max(pa.std(), 1e-3), n_sim), lo_a, hi_a)
            NU = np.clip(rng.choice(pn, n_sim) + rng.normal(0, max(pn.std(), 1e-3), n_sim), lo_n, hi_n)
        # simulate the draws in chunks (memory): unit latent variance, noise share NU of the total variance.
        # The random numbers are drawn draw-block by draw-block from one stream, so the chunking is part of the
        # definition of the estimator and the estimate does not depend on the memory available.
        L, m = max(lens), len(lens)
        mask = (np.arange(L)[:, None] < np.asarray(lens)[None, :]).astype(float)
        sim = np.empty((n_sim, int(max_lag)))
        for c0 in range(0, n_sim, _ABC_CHUNK):
            c1 = min(n_sim, c0 + _ABC_CHUNK)
            Ac, NUc = A[c0:c1], NU[c0:c1]
            k = c1 - c0
            Y = np.empty((L, k, m))
            Y[0] = rng.normal(size=(k, m))
            sd_in = np.sqrt(np.maximum(1.0 - Ac * Ac, 1e-12))[:, None]
            E = rng.normal(size=(L, k, m))
            for t in range(1, L):
                Y[t] = Ac[:, None] * Y[t - 1] + sd_in * E[t]
            del E
            Y += rng.normal(size=Y.shape) * np.sqrt(NUc / np.maximum(1.0 - NUc, 1e-9))[None, :, None]
            Y *= mask[:, None, :]
            Y = (Y - (Y.sum(axis=0) / np.asarray(lens, float)[None, :])[None]) * mask[:, None, :]
            sim[c0:c1] = pooled_acf_padded(Y, max_lag)
            del Y
        dist = np.nanmean((sim - obs[None, :]) ** 2, axis=1)
        dist = np.where(np.isfinite(dist), dist, np.inf)
        keep = np.argsort(dist, kind="stable")[:int(n_keep)]
        post = (A[keep], NU[keep])
    pa, pn = post
    lam = -np.log(np.clip(pa, 1e-9, 0.999999)) / float(dt)
    return dict(lam=float(np.median(lam)), a=float(np.median(pa)), nu=float(np.median(pn)),
                draws_a=pa, draws_nu=pn)


def fit_rates_padded(Y, lens, dt=1.0, method="ols"):
    """Relaxation rate of every recording of a padded, centred (L_max, n, n_seg) array by the named estimator.

    The array form of :func:`fit_rate`, used for the simulated recordings of a check.  Least squares is computed in
    closed form for all recordings at once, maximum likelihood by :func:`fit_ar1_kalman_batch`, and other estimators
    one recording at a time.
    """
    n = Y.shape[1]
    if method == "ols":
        a = ols_persistence_padded(Y, lens)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where((a > 0) & (a < 1), -np.log(np.where((a > 0) & (a < 1), a, 0.5)) / float(dt), np.nan)
    segs_of = lambda j: [Y[:L, j, i] for i, L in enumerate(lens)]                # noqa: E731
    if method == "kalman":
        res = _kalman_fit_padded(Y, lens)
        a = np.array([r["a"] for r in res])
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where((a > 0) & (a < 1), -np.log(np.where((a > 0) & (a < 1), a, 0.5)) / float(dt), np.nan)
    return np.array([fit_rate(segs_of(j), dt=dt, method=method) for j in range(n)])


def _kalman_fit_padded(Y, lens, maxiter=300, xatol=1e-6, fatol=1e-8):
    """Maximum-likelihood fit of equation (6) for every recording of a padded, centred array (see below)."""
    n = Y.shape[1]
    order = []
    for L in lens:
        if L not in order:
            order.append(L)
    one_group = len(order) == 1
    mask = None
    if not one_group:
        mask = (np.arange(Y.shape[0])[:, None] < np.asarray(lens)[None, :]).astype(float)
    Z0 = np.empty((n, 3))
    m = min(order)
    for j in range(n):
        if one_group:
            Yj = Y[:, j, :]
        else:
            Yc = Y[:m, j, :]
            Yj = Yc - Yc.mean(axis=0, keepdims=True)
        a_mm, sw2_mm, se2_mm = _moment_init(np.ascontiguousarray(Yj))
        Z0[j] = [np.log(a_mm / (1 - a_mm)), np.log(max(sw2_mm, 1e-9)), np.log(max(se2_mm, 1e-9))]

    def nll(idx, Z):
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            a = 1.0 / (1.0 + np.exp(-Z[:, 0]))
            sw2 = np.exp(Z[:, 1])
            se2 = np.exp(Z[:, 2])
        valid = (0 < a) & (a < 0.99999)
        out = np.full(len(idx), 1e12)
        if valid.any():
            iv = np.asarray(idx)[valid]
            val = np.zeros(int(valid.sum()))
            val = val + kalman_ll_batch(Y[:, iv, :], a[valid], sw2[valid], se2[valid], mask=mask)
            val = -val
            out[valid] = np.where(np.isfinite(val), val, 1e12)
        return out

    X, F, nit = _nelder_mead_batch(nll, Z0, maxiter=int(maxiter), xatol=xatol, fatol=fatol)
    res = []
    for j in range(n):
        with np.errstate(over="ignore"):
            a = float(1.0 / (1.0 + np.exp(-X[j, 0])))
        sw2, se2 = float(np.exp(X[j, 1])), float(np.exp(X[j, 2]))
        stat_var = sw2 / max(1 - a * a, 1e-9)
        res.append(dict(a=a, sw2=sw2, se2=se2, nu=float(se2 / max(se2 + stat_var, 1e-300)),
                        ll=float(-F[j]), nit=int(nit[j]), converged=bool(F[j] < 1e11)))
    return res


def kalman2_ll_batch(Y, a1, a2, q1, q2, r, mask=None):
    """Exact log-likelihood of the two-timescale model ``y = x1 + x2 + e`` for many recordings at once.

    ``x1`` and ``x2`` are independent first-order autoregressive processes with persistences ``a1`` and ``a2`` and
    innovation variances ``q1`` and ``q2``; ``e`` is uncorrelated noise of variance ``r``.  ``Y`` is a padded, centred
    (n_t, n_data, n_seg) array; parameters are (n_data,); ``mask`` as in :func:`kalman_ll_batch`.  The alternative
    model of the likelihood-ratio statistic of the model-adequacy check (G8).
    """
    n_t, n, m = Y.shape
    P11 = q1 / (1 - a1 * a1)
    P22 = q2 / (1 - a2 * a2)
    P12 = np.zeros(n)
    x1 = np.zeros((n, m))
    x2 = np.zeros((n, m))
    ll = np.zeros(n)
    c = np.log(2 * np.pi)
    for t in range(n_t):
        if t > 0:
            x1 = a1[:, None] * x1
            x2 = a2[:, None] * x2
            P11 = a1 * a1 * P11 + q1
            P22 = a2 * a2 * P22 + q2
            P12 = a1 * a2 * P12
        S = P11 + 2 * P12 + P22 + r
        v = Y[t] - x1 - x2
        term = -0.5 * ((c + np.log(S))[:, None] + v * v / S[:, None])
        if mask is not None:
            term = term * mask[t][None, :]
        ll += term.sum(axis=1)
        k1 = (P11 + P12) / S
        k2 = (P12 + P22) / S
        x1 = x1 + k1[:, None] * v
        x2 = x2 + k2[:, None] * v
        P11, P22, P12 = P11 - k1 * k1 * S, P22 - k2 * k2 * S, P12 - k1 * k2 * S
    return ll


def lr_two_timescale_padded(Y, lens, fits1, maxiter=600):
    """Likelihood-ratio statistic 2 (l2 - l1) of the two-timescale model against equation (6), per recording.

    ``fits1`` are the maximum-likelihood fits of equation (6) of the same recordings (with their log-likelihoods
    ``ll``).  The two-timescale model is maximised by Nelder-Mead from two starts derived from the equation (6) fit
    (the fit itself as the slow component with a small fast one; and a slower and a faster component sharing the
    variance), and the better of the two is kept.  The statistic is set to zero when the alternative's maximum is
    below the null's, which can happen only by incomplete optimisation.
    """
    n = Y.shape[1]
    mask = None
    if len(set(lens)) > 1:
        mask = (np.arange(Y.shape[0])[:, None] < np.asarray(lens)[None, :]).astype(float)
    a = np.array([f["a"] for f in fits1], float)
    sw2 = np.array([f["sw2"] for f in fits1], float)
    se2 = np.array([f["se2"] for f in fits1], float)
    ll1 = np.array([f["ll"] for f in fits1], float)

    def lg(p):
        p = np.clip(p, 1e-9, 1 - 1e-9)
        return np.log(p / (1 - p))

    def unpack(Z):
        with np.errstate(over="ignore"):
            a1 = 1 / (1 + np.exp(-Z[:, 0]))
            a2 = a1 / (1 + np.exp(-Z[:, 1]))
            return a1, a2, np.exp(Z[:, 2]), np.exp(Z[:, 3]), np.exp(Z[:, 4])

    def nll(idx, Z):
        a1, a2, q1, q2, r = unpack(Z)
        ok = (a1 > 0) & (a1 < 0.99999) & (a2 >= 0)
        out = np.full(len(idx), 1e12)
        if ok.any():
            iv = np.asarray(idx)[ok]
            v = -kalman2_ll_batch(Y[:, iv, :], a1[ok], a2[ok], q1[ok], q2[ok], r[ok], mask=mask)
            out[ok] = np.where(np.isfinite(v), v, 1e12)
        return out

    a1 = np.clip(a, 0.02, 0.995)
    s1 = np.column_stack([lg(a1), lg(np.full(n, 0.3)), np.log(sw2 + 1e-300), np.log(0.1 * sw2 + 1e-12),
                          np.log(se2 + 1e-12)])
    a1b = np.clip(1 - (1 - a) / 3, 0.05, 0.998)
    s2 = np.column_stack([lg(a1b), lg(np.full(n, 0.4)),
                          np.log(0.5 * sw2 * (1 - a1b ** 2) / np.maximum(1 - a ** 2, 1e-9) + 1e-300),
                          np.log(0.5 * sw2 + 1e-12), np.log(se2 + 1e-12)])
    best = None
    for Z0 in (s1, s2):
        _, F, _ = _nelder_mead_batch(nll, Z0, maxiter=int(maxiter), xatol=1e-5, fatol=1e-7)
        best = F if best is None else np.minimum(best, F)
    return np.maximum(2.0 * (-best - ll1), 0.0)


def _periodogram_groups(Y, lens):
    """Pooled periodograms of a padded, centred (L_max, n, n_seg) array, one entry per group of equal length.

    Returns a list of (omega (J,), pooled I (n, J), number of segments) over the Fourier frequencies strictly between
    0 and pi; I is |sum_t y_t e^{-i omega t}|^2 / (2 pi L), summed over the segments of the group.
    """
    out = []
    lens = np.asarray(lens)
    for L in sorted(set(lens.tolist())):
        cols = np.flatnonzero(lens == L)
        J = (int(L) - 1) // 2
        if J < 1:
            continue
        F = np.fft.rfft(Y[:int(L)][:, :, cols], axis=0)[1:J + 1]            # (J, n, k)
        I = (F.real ** 2 + F.imag ** 2).sum(axis=2).T / (2 * np.pi * L)     # (n, J)
        out.append((2 * np.pi * np.arange(1, J + 1) / L, I, len(cols)))
    return out


def whittle_lr_two_timescale(Y, lens, maxiter=800):
    """Whittle likelihood-ratio statistic of two latent timescales plus noise against equation (6), per recording.

    The spectral (Whittle) log-likelihood of each model is maximised over the pooled periodograms of the centred
    segments (Fourier frequencies strictly between 0 and pi), by Nelder-Mead for all recordings at once.  Equation (6)
    has the spectral density q / (2 pi |1 - a e^{-i w}|^2) + r / (2 pi); the alternative has two such autoregressive
    terms.  Returns 2 (l2 - l1), set to zero when negative.  Used as a test statistic calibrated by simulation, so the
    approximation of the exact likelihood affects its power, not its level.
    """
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        return _whittle_lr(Y, lens, maxiter)


def _whittle_lr(Y, lens, maxiter):
    groups = _periodogram_groups(Y, lens)
    n = Y.shape[1]
    if not groups:
        return np.full(n, np.nan)

    def loglik(params_fn, idx, Z):
        val = np.zeros(len(idx))
        for w, I, k in groups:
            f = params_fn(Z, w)                                    # (len(idx), J)
            val += -(k * np.log(f).sum(axis=1) + (I[idx] / f).sum(axis=1))
        return val

    def lg(p):
        p = np.clip(p, 1e-9, 1 - 1e-9)
        return np.log(p / (1 - p))

    def spec1(Z, w):
        with np.errstate(over="ignore"):
            a = 1 / (1 + np.exp(-Z[:, 0]))
        q, r = np.exp(Z[:, 1]), np.exp(Z[:, 2])
        c = np.cos(w)[None, :]
        return (q[:, None] / (1 - 2 * a[:, None] * c + a[:, None] ** 2) + r[:, None]) / (2 * np.pi)

    def spec2(Z, w):
        with np.errstate(over="ignore"):
            a1 = 1 / (1 + np.exp(-Z[:, 0]))
            a2 = a1 / (1 + np.exp(-Z[:, 1]))
        q1, q2, r = np.exp(Z[:, 2]), np.exp(Z[:, 3]), np.exp(Z[:, 4])
        c = np.cos(w)[None, :]
        return (q1[:, None] / (1 - 2 * a1[:, None] * c + a1[:, None] ** 2)
                + q2[:, None] / (1 - 2 * a2[:, None] * c + a2[:, None] ** 2) + r[:, None]) / (2 * np.pi)

    def nll1(idx, Z):
        v = -loglik(spec1, idx, Z)
        return np.where(np.isfinite(v), v, 1e12)

    def nll2(idx, Z):
        v = -loglik(spec2, idx, Z)
        return np.where(np.isfinite(v), v, 1e12)

    # starts from the lag-1 and lag-2 moments of the pooled segments
    r1 = pooled_acf_padded(Y, 2)
    var = np.einsum("tns,tns->n", Y, Y) / float(sum(lens))
    a0 = np.clip(np.where((r1[:, 0] > 0) & (r1[:, 1] > 0), r1[:, 1] / np.maximum(r1[:, 0], 1e-9), 0.5), 0.05, 0.98)
    nu0 = np.clip(np.where((r1[:, 0] > 0) & (r1[:, 1] > 0), 1 - r1[:, 0] ** 2 / np.maximum(r1[:, 1], 1e-9), 0.3),
                  0.02, 0.9)
    q0 = np.maximum((1 - nu0) * var * (1 - a0 ** 2), 1e-6 * var + 1e-300)
    rr0 = np.maximum(nu0 * var, 1e-6 * var + 1e-300)
    Z1, F1, _ = _nelder_mead_batch(nll1, np.column_stack([lg(a0), np.log(q0), np.log(rr0)]), maxiter=int(maxiter),
                                   xatol=1e-6, fatol=1e-8)
    a1 = 1 / (1 + np.exp(-Z1[:, 0]))
    q1, r_ = np.exp(Z1[:, 1]), np.exp(Z1[:, 2])
    s_a = np.column_stack([Z1[:, 0], lg(np.full(n, 0.3)), np.log(q1), np.log(0.1 * q1), np.log(r_)])
    a1b = np.clip(1 - (1 - a1) / 3, 0.05, 0.998)
    s_b = np.column_stack([lg(a1b), lg(np.full(n, 0.4)), np.log(0.5 * q1 * (1 - a1b ** 2) / np.maximum(1 - a1 ** 2, 1e-9)),
                           np.log(0.5 * q1), np.log(r_)])
    best = None
    for Z0 in (s_a, s_b):
        _, F2, _ = _nelder_mead_batch(nll2, Z0, maxiter=int(maxiter), xatol=1e-6, fatol=1e-8)
        best = F2 if best is None else np.minimum(best, F2)
    return np.maximum(2.0 * (F1 - best), 0.0)


def fit_ar1_kalman_batch(datasets, maxiter=300, xatol=1e-6, fatol=1e-8):
    """:func:`fit_ar1_kalman` with ``n_restart=1`` for many datasets of the same geometry.

    ``datasets`` is a list of datasets, each a list of one-dimensional segments; every dataset must have the same
    sequence of segment lengths (as the simulations of one check do).  The estimator is the one of
    :func:`fit_ar1_kalman`: segments centred on their own means, the exact likelihood maximised by Nelder-Mead
    (scipy's algorithm, reproduced step by step) from the two-lag moment start.  With segments of equal length the
    estimates are bit-identical to :func:`fit_ar1_kalman`; with unequal segments, which are padded and masked so that
    one filter pass serves all of them, they differ only by the order of summation (relative differences of the rate
    below 1e-6 in the package tests).  Returns a list of dicts as :func:`fit_ar1_kalman` does; ``converged`` is
    False when the single start ended at the penalty value (the persistence is then unusable).
    """
    if len(datasets) == 0:
        return []
    Y, lens = padded_centred(datasets)
    return _kalman_fit_padded(Y, lens, maxiter=maxiter, xatol=xatol, fatol=fatol)


#: Estimators a certificate can be issued *for*.  ``observation_aware`` records whether the estimator separates an
#: additive uncorrelated component from the latent process (reported by the noise diagnostic, G1).  ``model`` is the
#: single-timescale model the estimator assumes, which the model-adequacy check (G8) tests: ``"ar1"``, a first-order
#: autoregressive process, or ``"ar1_white"``, that process plus additive uncorrelated noise (equation (6) of the
#: paper).  Add your own with :func:`register_estimator`; the checks read this registry, so a registered estimator is
#: certified by exactly the same path as the built-in ones.
ESTIMATORS = {
    "ols": dict(observation_aware=False, model="ar1",
                label="least squares on lag-1 pairs"),
    "kalman": dict(observation_aware=True, model="ar1_white",
                   label="maximum likelihood, AR(1) plus additive white"),
    "abc": dict(observation_aware=True, model="ar1_white",
                label="approximate Bayesian computation on the autocorrelation, AR(1) plus additive white"),
}


def register_estimator(name, fn, observation_aware, label=None, overwrite=False, model=None):
    """Register a rate estimator so a certificate can be issued for it.

    ``fn(segments_1d, dt) -> float`` returns a relaxation rate per unit time, or
    NaN where it cannot fit. It is called exactly as the built-in estimators are,
    including inside G2's refits at each decimation level and inside G3's
    injection ladder, so whatever it does is what gets certified.

    ``observation_aware`` must be stated by the caller. It declares whether the
    estimator separates an additive white component from the latent process. It
    cannot be inferred: an estimator that happens to be robust to white noise on
    one recording is not the same as one that models it.  ``model`` is the
    single-timescale model the estimator assumes (``"ar1"`` or ``"ar1_white"``),
    which the model-adequacy check tests; by default ``"ar1_white"`` for an
    observation-aware estimator and ``"ar1"`` otherwise.

    >>> register_estimator("mine", lambda segs, dt: 0.1, observation_aware=False)
    >>> certify(X, design, estimator="mine")            # doctest: +SKIP

    Raises if the name is already taken, unless ``overwrite`` is set, so that a
    typo cannot silently redefine "ols" half way through a run.
    """
    if not callable(fn):
        raise TypeError(f"fn must be callable, got {type(fn).__name__}")
    if not isinstance(observation_aware, bool):
        raise TypeError("observation_aware must be declared as True or False; it "
                        "cannot be inferred")
    if model is None:
        model = "ar1_white" if observation_aware else "ar1"
    if model not in ("ar1", "ar1_white"):
        raise ValueError("model must be 'ar1' or 'ar1_white'")
    if name in ESTIMATORS and not overwrite:
        raise ValueError(f"estimator {name!r} is already registered; pass "
                         f"overwrite=True to replace it")
    ESTIMATORS[name] = dict(observation_aware=bool(observation_aware), model=model,
                            label=label or f"user estimator {name!r}", fn=fn)
    return name


def fit_rate(segments_1d, dt=1.0, method="ols", seed=0):
    """Scalar relaxation rate by the named estimator.

    ``"ols"`` is what is most often run and has no notion of an observation
    process.  ``"kalman"`` fits the observation process jointly by maximum
    likelihood (:func:`fit_ar1_kalman_batch`, one start, at most 300 iterations).
    ``"abc"`` is approximate Bayesian computation under the same model
    (:func:`fit_abc`).  The whole point of naming the estimator is that
    identifiability is a property of the pair (recording, estimator), not of the
    recording alone.  ``seed`` is accepted but not used by the built-in
    estimators.
    """
    segs = [np.asarray(s, float).ravel() for s in segments_1d]
    if method == "ols":
        num = den = 0.0
        for s in segs:
            s = s - s.mean()
            num += float(np.dot(s[1:], s[:-1]))
            den += float(np.dot(s[:-1], s[:-1]))
        if den <= 0:
            return np.nan
        a = num / den
    elif method == "kalman":
        a = fit_ar1_kalman_batch([segs], maxiter=300)[0]["a"]
    elif method == "abc":
        return fit_abc(segs, dt=float(dt))["lam"]
    elif method in ESTIMATORS and callable(ESTIMATORS[method].get("fn")):
        # a registered user estimator returns the RATE, not the lag-1 factor
        r = ESTIMATORS[method]["fn"](segs, float(dt))
        r = float(r) if r is not None else np.nan
        return r if np.isfinite(r) and r > 0 else np.nan
    else:
        raise ValueError(f"unknown estimator {method!r}; have {sorted(ESTIMATORS)}. "
                         f"Register your own with ndif.register_estimator(name, fn, "
                         f"observation_aware=...).")
    return -np.log(a) / float(dt) if 0 < a < 1 else np.nan


def fit_model(segments_1d, model="ar1_white"):
    """Persistence and noise fraction of the single-timescale ``model`` (``"ar1"`` or ``"ar1_white"``).

    ``"ar1_white"`` is the maximum-likelihood fit of equation (6); ``"ar1"`` is the least-squares lag-1 factor with a
    noise fraction of zero.  Used for the generators of the model-adequacy and recovery checks.
    """
    segs = [np.asarray(s, float).ravel() for s in segments_1d]
    if model == "ar1_white":
        g = fit_ar1_kalman_batch([segs], maxiter=300)[0]
        return float(g["a"]), float(g["nu"])
    lam = fit_rate(segs, method="ols")
    return (float(np.exp(-lam)) if np.isfinite(lam) else float("nan")), 0.0


def stationary_covariance(M, D):
    """Stationary covariance of ``dy = M y dt + sqrt(2 D) dW``.

    Solves ``M Sigma + Sigma M^T + 2 D = 0``.
    """
    M = np.asarray(M, float)
    D = np.asarray(D, float)
    if D.ndim == 0:
        D = float(D) * np.eye(M.shape[0])
    return solve_lyapunov(M, -2.0 * D)


def directional_rate(M, D, e):
    """The pair ``(lambda_e, D_e)`` along direction ``e``, and their ratio.

    The identity the framework rests on: for a linear system, the stationary
    variance along any direction satisfies

        e^T Sigma e = D_e / lambda_e ,

    so an observed variance fixes only the *ratio*.  Scaling ``(M, D) ->
    (cM, cD)`` leaves every functional of the density exactly unchanged while
    multiplying every rate by ``c``.
    """
    e = np.asarray(e, float).ravel()
    e = e / np.linalg.norm(e)
    M = np.asarray(M, float)
    D = np.asarray(D, float)
    if D.ndim == 0:
        D = float(D) * np.eye(M.shape[0])
    lam = float(-e @ M @ e)
    De = float(e @ D @ e)
    return dict(lambda_e=lam, D_e=De, ratio=De / lam if lam != 0 else np.inf)


def lyapunov_residual(M, D, e):
    """Absolute residual of ``e^T Sigma e - D_e / lambda_e``.

    Exact for a normal ``M``; for non-normal ``M`` the identity holds with
    ``lambda_e`` read as the decay rate of the projected process, and the
    residual returned here is the departure from the naive reading.  Used as a
    self-test of the linear algebra, not as a claim about data.
    """
    Sig = stationary_covariance(M, D)
    e = np.asarray(e, float).ravel()
    e = e / np.linalg.norm(e)
    dr = directional_rate(M, D, e)
    return abs(float(e @ Sig @ e) - dr["ratio"])
