"""Generators with known ground truth, and the injections the gates push through.

Nothing here is a model of the brain.  These are systems whose answer is known
in closed form, so that an estimator's behaviour can be measured rather than
assumed.  Gate G3 is built entirely out of them.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "simulate_var1", "matched_isotropic", "random_hurwitz_propagator",
    "scale_rate", "bistable_2d", "single_basin_2d", "add_white",
]


def simulate_var1(A, Q, n_segments=1, n_block=1000, rng=None, burn=200,
                  x0_stationary=True):
    """Simulate a VAR(1) process as a list of independent segments.

    Segments are independent, as trials or runs are; this is what makes the
    simulated geometry match a real recording's.
    """
    rng = np.random.default_rng(rng)
    A = np.asarray(A, float)
    Q = np.asarray(Q, float)
    d = A.shape[0]
    L = np.linalg.cholesky(Q + 1e-12 * max(np.trace(Q), 1e-12) / d * np.eye(d))
    S = int(n_segments)
    b = int(burn) if x0_stationary else 0
    n = int(n_block) + b
    # All segments advance together: the recursion is unavoidable in time but
    # not across segments, and the gates simulate hundreds of segments.
    Y = np.empty((n, S, d))
    Y[0] = rng.normal(size=(S, d)) @ L.T
    E = rng.normal(size=(n - 1, S, d)) @ L.T
    At = A.T
    for t in range(1, n):
        Y[t] = Y[t - 1] @ At + E[t - 1]
    Y = Y[b:]
    return [Y[:, i, :] for i in range(S)]


def stationary_cov_discrete(A, Q):
    """Stationary covariance of ``y_t = A y_{t-1} + xi``, ``cov(xi) = Q``."""
    from scipy.linalg import solve_discrete_lyapunov
    return solve_discrete_lyapunov(np.asarray(A, float), np.asarray(Q, float))


def matched_isotropic(A, Q, dt=1.0, rate=1.0, jitter=1e-10):
    """The uniform-decay surrogate: drift -> -rate * I, same stationary covariance.

    Keeps the density *exactly* and replaces the fitted dynamics by a uniform
    decay that is unrelated to how the recorded state moves.  A readout that cannot tell this apart from the fitted
    process is a functional of the density, i.e. a level-1 quantity however it
    is interpreted.

    The reference rate is **fixed** (``drift = -rate * I``, so
    ``A = exp(-rate*dt) I``) and is deliberately *not* read from the fit.  The
    checks pass the reference rate converted to the unit of ``dt`` (default 1
    per second), so the surrogate is the same process whatever unit time is
    written in.  A surrogate
    read from the fit's own geometric-mean eigenvalue would preserve the rate
    rather than delete it and would be exactly the identity map
    in one dimension -- making gates G4 and G7 compare the process against
    itself.  Callers must still check :func:`is_degenerate_surrogate`.
    """
    A = np.asarray(A, float)
    Q = np.asarray(Q, float)
    d = A.shape[0]
    Sig = stationary_cov_discrete(A, Q)
    mu = float(np.exp(-float(rate) * float(dt)))
    mu = float(np.clip(mu, 1e-6, 0.999))
    A_iso = mu * np.eye(d)
    Q_iso = Sig - A_iso @ Sig @ A_iso.T
    Q_iso = 0.5 * (Q_iso + Q_iso.T)
    w, V = np.linalg.eigh(Q_iso)
    w = np.clip(w, jitter * max(w.max(), 1.0), None)
    return A_iso, V @ np.diag(w) @ V.T


def is_degenerate_surrogate(A, A_sur, rtol=0.02):
    """True when the surrogate is indistinguishable from the fitted propagator.

    A surrogate that equals the fit deletes nothing, so any gate built on it is
    vacuous and must report *indeterminate* rather than a verdict.  Compared on
    the relaxation spectrum, which is what the gates read.
    """
    a = np.sort(np.abs(np.linalg.eigvals(np.asarray(A, float))))
    b = np.sort(np.abs(np.linalg.eigvals(np.asarray(A_sur, float))))
    a = np.clip(a, 1e-12, None)
    b = np.clip(b, 1e-12, None)
    return bool(np.all(np.abs(np.log(b) - np.log(a)) <= rtol))


def random_hurwitz_propagator(A, Q, rng=None, max_try=60, jitter=1e-10,
                              lo=0.05, hi=0.95):
    """A random stable propagator with the same stationary covariance.

    The second G4 surrogate.  The largest eigenvalue modulus is drawn
    log-uniformly in ``[lo, hi]`` (the other moduli follow from a random Gaussian
    matrix scaled to it) rather than matched to the fit's spectral radius, for the same
    reason as :func:`matched_isotropic`: a surrogate that keeps the fitted rate
    has not deleted the dynamics.

    Returns ``(A_rand, Q_rand)`` or ``(None, None)`` if no admissible pair was
    found, which is reported rather than retried forever.
    """
    rng = np.random.default_rng(rng)
    A = np.asarray(A, float)
    Q = np.asarray(Q, float)
    d = A.shape[0]
    Sig = stationary_cov_discrete(A, Q)
    for _ in range(int(max_try)):
        target = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        if d == 1:
            B = np.array([[target]])
        else:
            B = rng.normal(size=(d, d))
            B = B * (target / max(np.abs(np.linalg.eigvals(B)).max(), 1e-12))
        Qr = Sig - B @ Sig @ B.T
        Qr = 0.5 * (Qr + Qr.T)
        w, V = np.linalg.eigh(Qr)
        if w.min() > -1e-8 * max(abs(w).max(), 1.0):
            w = np.clip(w, jitter * max(w.max(), 1.0), None)
            return B, V @ np.diag(w) @ V.T
    return None, None


def scale_rate(A, factor):
    """Multiply every relaxation rate by ``factor`` (a matrix power).

    ``lambda_k = -log|mu_k| / dt``, so scaling the rates by ``c`` means raising
    the propagator to the power ``c``.  Implemented through the eigenbasis and
    returned real.
    """
    A = np.asarray(A, float)
    w, V = np.linalg.eig(A)
    w = np.where(np.abs(w) < 1e-12, 1e-12, w)
    wc = np.abs(w) ** float(factor) * np.exp(1j * np.angle(w) * float(factor))
    return np.real(V @ np.diag(wc) @ np.linalg.inv(V))


def add_white(segments, nu, rng=None):
    """Add an isotropic white component carrying variance share ``nu``.

    ``nu`` is the share of the *observed* variance, matching
    :func:`ndif.estimators.white_share`.
    """
    rng = np.random.default_rng(rng)
    nu = float(nu)
    if nu <= 0:
        return [np.array(s, float) for s in segments]
    out = []
    for s in segments:
        s = np.asarray(s, float)
        v = s.var(axis=0, ddof=1)
        sd = np.sqrt(v * nu / max(1.0 - nu, 1e-12))
        out.append(s + rng.normal(size=s.shape) * sd)
    return out


# ----------------------------------------------------------- nonlinear truths
def single_basin_2d(k=1.0, D=0.05):
    """Drift of a single-well system."""
    def drift(P):
        P = np.atleast_2d(np.asarray(P, float))
        return -k * P
    return drift, dict(kind="single_basin", n_basins=1, D=D, k=k)


def bistable_2d(barrier_scale=1.0, coupling=6.0, D=0.05):
    """Drift of a double well with a curved instanton.

    ``b = -grad U`` with ``U = s(x^4/4 - x^2/2) + (K/2)(y - (x^2-1))^2``.
    Minima at ``(+-1, 0)``, saddle at ``(0, -1)``; the exact quasipotential
    barrier is ``s/4 / D`` in the action convention of :mod:`ndif.action`.
    """
    s, K = float(barrier_scale), float(coupling)

    def drift(P):
        P = np.atleast_2d(np.asarray(P, float))
        x, y = P[:, 0], P[:, 1]
        g = y - (x ** 2 - 1.0)
        return -np.column_stack([s * (x ** 3 - x) + K * g * (-2.0 * x), K * g])

    exact = (s / 4.0) / float(D)
    return drift, dict(kind="bistable", n_basins=2, D=D, exact_barrier=exact,
                       minima=[(-1.0, 0.0), (1.0, 0.0)], saddle=(0.0, -1.0))


def simulate_sde(drift, D, n_block, dt=0.01, n_segments=1, rng=None, x0=None,
                 burn=2000, thin=1):
    """Euler-Maruyama for ``dx = b dt + sqrt(2 D) dW``."""
    rng = np.random.default_rng(rng)
    D = np.asarray(D, float)
    probe = np.atleast_2d(np.zeros(2) if x0 is None else np.asarray(x0, float))
    d = np.asarray(drift(probe), float).shape[1]
    if D.ndim == 0:
        D = float(D) * np.eye(d)
    L = np.linalg.cholesky(2.0 * D * dt)
    out = []
    for _ in range(int(n_segments)):
        n = burn + int(n_block) * int(thin)
        x = np.zeros(d) if x0 is None else np.asarray(x0, float).copy()
        keep = np.empty((int(n_block), d))
        j = 0
        for t in range(n):
            x = x + np.asarray(drift(x[None, :]), float)[0] * dt + rng.normal(size=d) @ L.T
            if t >= burn and (t - burn) % int(thin) == 0 and j < int(n_block):
                keep[j] = x
                j += 1
        out.append(keep[:j])
    return out
