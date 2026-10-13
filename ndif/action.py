"""Freidlin-Wentzell geometric action, and a minimum-action solver.

Convention
----------
For the Ito SDE

    dx = b(x) dt + sqrt(2 D) dW ,

the geometric (parameterisation-free) action of a curve phi joining two points
is (Heymann & Vanden-Eijnden 2008, Phys. Rev. Lett. 100:140601)

    S[phi] = int ( |phi'|_a |b(phi)|_a - <phi', b(phi)>_a ) ds ,
    <u,v>_a = u^T a^{-1} v ,   a = 2 D .

S vanishes on paths that follow the deterministic flow and is positive on paths
that oppose it, so its minimum over curves joining two equilibria is the
quasipotential barrier.  For a gradient system b = -grad U with isotropic
D = sigma^2 I the minimum is exactly dU / sigma^2; concretely, the 1-D double
well b = -x(x^2-1) from x=-1 to x=0 with D = 0.05 has dU = 1/4 and S = 5.0
exactly, which is the regression test in ``tests/test_action.py``.

Why the solver is built this way
--------------------------------
A translation-only descent is kept in this package as a regression case
(``tests/test_action.py::test_translation_only_descent_fails_where_ours_succeeds``).
Such a descent evaluates

    G[1:-1, k] = (S(P + h e_k on all interior nodes)
                  - S(P - h e_k on all interior nodes)) / (2h)

which is a *single* number, the directional derivative under a rigid
translation of the whole interior path, and assigns it to every interior
node.  The descent direction is therefore constant along the path and the
curve can only translate, never bend, so any problem whose instanton is curved
is unreachable.  On the two-dimensional curved-valley problem whose exact
action is 5.0, the descent of the regression test stalls at 12.9 after 300
iterations at 101 nodes (25.6 on the straight line), and the control of
``analyses/solver_validation`` stops at 69.178 after 600 iterations at 81 nodes
between the two minima (108.18 on the straight line).  It passes the one-dimensional
control (5.0016 against 5.0, indistinguishable from the straight line at
5.00025) because in one dimension a rigid translation is the only motion
available, which is why a one-dimensional control cannot detect this failure
mode.

``minimum_action`` below uses instead the exact analytic gradient of the
discretised action with respect to every interior node independently, plus the
arclength reparameterisation that defines gMAM and a coarse-to-fine node
schedule.  It recovers 5.0 on the same two-dimensional problem.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

__all__ = [
    "geometric_action",
    "action_gradient",
    "reparameterise",
    "minimum_action",
    "MinimumActionResult",
]


def _a_inv(d: int, D) -> np.ndarray:
    """Inverse of a = 2D, for D given as None (identity), scalar, or matrix."""
    if D is None:
        a = 2.0 * np.eye(d)
    else:
        D = np.asarray(D, float)
        if D.ndim == 0:
            a = 2.0 * float(D) * np.eye(d)
        elif D.ndim == 1:
            a = 2.0 * np.diag(D)
        else:
            a = 2.0 * D
    return np.linalg.inv(a)


def geometric_action(path, drift_fn, D=None) -> float:
    """Geometric action of a discretised curve.

    Parameters
    ----------
    path : (n_pts, d) array
        Curve nodes, ordered from start to end.
    drift_fn : callable
        ``drift_fn(X)`` for ``X`` of shape (m, d) returning (m, d).
    D : None, scalar, (d,) or (d, d)
        Diffusion.  ``a = 2 D``.

    Notes
    -----
    Midpoint rule on each segment.
    """
    P = np.asarray(path, float)
    ai = _a_inv(P.shape[1], D)
    dP = np.diff(P, axis=0)
    mid = 0.5 * (P[1:] + P[:-1])
    b = np.asarray(drift_fn(mid), float)
    nb = np.sqrt(np.maximum(np.einsum("nk,kl,nl->n", b, ai, b), 0.0))
    nd = np.sqrt(np.maximum(np.einsum("nk,kl,nl->n", dP, ai, dP), 0.0))
    dot = np.einsum("nk,kl,nl->n", dP, ai, b)
    return float(np.sum(nd * nb - dot))


def _drift_jacobian(drift_fn, X, h_rel=1e-6):
    """Central-difference Jacobian of the drift at every row of X.

    Returns (m, d, d) with ``J[n, i, k] = d b_i / d x_k`` at ``X[n]``.
    Costs 2*d vectorised drift evaluations for the whole path.
    """
    X = np.asarray(X, float)
    m, d = X.shape
    scale = h_rel * (1.0 + np.abs(X).max(axis=0))
    J = np.empty((m, d, d))
    for k in range(d):
        e = np.zeros(d)
        e[k] = scale[k]
        bp = np.asarray(drift_fn(X + e), float)
        bm = np.asarray(drift_fn(X - e), float)
        J[:, :, k] = (bp - bm) / (2.0 * scale[k])
    return J


def action_gradient(path, drift_fn, D=None, jac_fn=None):
    """Exact gradient of the discretised geometric action w.r.t. every node.

    Returns (n_pts, d).  Endpoint rows are returned as computed; callers that
    hold the endpoints fixed should ignore or zero them.

    The discretisation is the same midpoint rule as :func:`geometric_action`,
    so this is the exact gradient of *that* function, not of the continuum
    action -- which is what a descent method needs.
    """
    P = np.asarray(path, float)
    n, d = P.shape
    ai = _a_inv(d, D)
    dP = np.diff(P, axis=0)
    mid = 0.5 * (P[1:] + P[:-1])
    b = np.asarray(drift_fn(mid), float)

    ai_dP = dP @ ai.T
    ai_b = b @ ai.T
    nd = np.sqrt(np.maximum(np.einsum("nk,nk->n", dP, ai_dP), 0.0))
    nb = np.sqrt(np.maximum(np.einsum("nk,nk->n", b, ai_b), 0.0))

    eps = np.finfo(float).tiny
    u_d = ai_dP / np.maximum(nd, eps)[:, None]        # d|dP|_a / d dP
    u_b = ai_b / np.maximum(nb, eps)[:, None]         # d|b|_a  / d b

    J = _drift_jacobian(drift_fn, mid) if jac_fn is None else np.asarray(jac_fn(mid), float)

    # dS_n/d dP  and  dS_n/d b
    g_dP = nb[:, None] * u_d - ai_b                   # (n-1, d)
    g_b = nd[:, None] * u_b - ai_dP                   # (n-1, d)
    # push g_b through the Jacobian: (J^T g_b)_k = sum_i J[n,i,k] g_b[n,i]
    g_mid = np.einsum("nik,ni->nk", J, g_b)           # (n-1, d)

    G = np.zeros((n, d))
    # segment n couples nodes n and n+1:  dP = P[n+1]-P[n], mid = (P[n+1]+P[n])/2
    np.add.at(G, np.arange(0, n - 1), -g_dP + 0.5 * g_mid)
    np.add.at(G, np.arange(1, n), g_dP + 0.5 * g_mid)
    return G


def reparameterise(P):
    """Redistribute nodes uniformly in arclength -- the defining gMAM step."""
    P = np.asarray(P, float)
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] <= 0:
        return P
    s = s / s[-1]
    t = np.linspace(0.0, 1.0, len(P))
    return np.column_stack([np.interp(t, s, P[:, k]) for k in range(P.shape[1])])


def _resample(P, n_new):
    """Interpolate a path onto ``n_new`` nodes, uniformly in arclength."""
    P = np.asarray(P, float)
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] <= 0:
        t = np.linspace(0.0, 1.0, len(P))
        s = t.copy()
    else:
        s = s / s[-1]
    t_new = np.linspace(0.0, 1.0, n_new)
    return np.column_stack([np.interp(t_new, s, P[:, k]) for k in range(P.shape[1])])


@dataclass
class MinimumActionResult:
    """Outcome of a minimum-action computation."""

    action: float
    path: np.ndarray
    n_pts: int
    n_restarts: int
    converged: bool
    per_start: list = field(default_factory=list)

    def __repr__(self) -> str:  # pragma: no cover - display only
        return (f"MinimumActionResult(action={self.action:.6g}, n_pts={self.n_pts}, "
                f"n_restarts={self.n_restarts}, converged={self.converged})")


def _initial_paths(x0, x1, n_pts, n_random, rng):
    """Straight line plus transverse-bulge starts, in the plane-free way.

    The bulges are drawn in random directions orthogonal to the chord, with
    amplitude on the scale of the chord, so no initialisation encodes the
    answer to any particular problem.
    """
    d = len(x0)
    t = np.linspace(0.0, 1.0, n_pts)[:, None]
    chord = x1 - x0
    L = np.linalg.norm(chord)
    straight = x0 + t * chord
    paths = [straight]
    if L <= 0 or n_random <= 0:
        return paths
    u = chord / L
    bump = np.sin(np.pi * t)  # zero at both endpoints
    for _ in range(n_random):
        v = rng.normal(size=d)
        v = v - (v @ u) * u
        nv = np.linalg.norm(v)
        if nv < 1e-12:
            continue
        v = v / nv
        amp = L * rng.uniform(0.15, 0.75)
        paths.append(straight + amp * bump * v)
    return paths


def minimum_action(x0, x1, drift_fn, D=None, n_pts=101, n_random_starts=4,
                   schedule=(21, 51, None), n_outer=12, seed=0, jac_fn=None,
                   ftol=1e-13, gtol=1e-11, maxiter=300, keep=2, rtol=1e-11):
    """Minimise the geometric action over curves from ``x0`` to ``x1``.

    Three things distinguish it from the translation-only descent kept as a
    regression case (see module docstring), and all three are needed:

    1. **Per-node analytic gradient.**  Every interior node moves on its own
       gradient, so the curve can bend.  A numerical gradient that is constant
       along the path can only translate it.
    2. **Arclength reparameterisation between outer iterations**, which is what
       makes the geometric formulation well posed.
    3. **Coarse-to-fine node schedule and transverse-bulge restarts**, so a
       curved instanton is found without seeding the solver with its shape.

    Parameters
    ----------
    x0, x1 : (d,) array_like
        Fixed endpoints.
    drift_fn : callable
        ``drift_fn(X)`` for ``X`` of shape (m, d) returning (m, d).
    D : None, scalar, (d,) or (d, d)
        Diffusion; ``a = 2 D``.
    n_pts : int
        Node count of the final refinement.
    n_random_starts : int
        Transverse-bulge initialisations in addition to the straight line.
    schedule : tuple
        Node counts of the coarse-to-fine stages; ``None`` means ``n_pts``.
    n_outer : int
        Reparameterise-and-reminimise rounds per stage.
    jac_fn : callable, optional
        ``jac_fn(X) -> (m, d, d)`` analytic drift Jacobian.  Finite differences
        are used when omitted.
    keep : int
        Restarts carried from one refinement stage to the next, best action
        first.  All restarts are run at the coarsest stage only; this is what
        keeps the cost close to that of a single descent.
    rtol : float
        Outer loop stops when the relative change in the action between
        reparameterisations falls below this.

    Returns
    -------
    MinimumActionResult
    """
    x0 = np.asarray(x0, float).ravel()
    x1 = np.asarray(x1, float).ravel()
    d = x0.size
    rng = np.random.default_rng(seed)
    stages = [n_pts if s is None else int(s) for s in schedule]

    def S_of(v, P_shape, xa, xb):
        P = np.vstack([xa, v.reshape(P_shape), xb])
        return geometric_action(P, drift_fn, D)

    def SG_of(v, P_shape, xa, xb):
        P = np.vstack([xa, v.reshape(P_shape), xb])
        S = geometric_action(P, drift_fn, D)
        G = action_gradient(P, drift_fn, D, jac_fn=jac_fn)
        return S, G[1:-1].ravel()

    def descend(P, n_stage):
        """Reparameterise-and-reminimise until the action stops moving."""
        if P.shape[0] != n_stage:
            P = _resample(P, n_stage)
        shape = (n_stage - 2, d)
        S_prev, ok = np.inf, False
        for _ in range(n_outer):
            res = minimize(SG_of, P[1:-1].ravel(), args=(shape, x0, x1), jac=True,
                           method="L-BFGS-B",
                           options=dict(maxiter=maxiter, ftol=ftol, gtol=gtol))
            ok = bool(res.success) or res.status == 1   # 1 = maxiter, path still usable
            P = reparameterise(np.vstack([x0, res.x.reshape(shape), x1]))
            S = geometric_action(P, drift_fn, D)
            if abs(S_prev - S) <= rtol * max(1.0, abs(S)):
                S_prev = S
                break
            S_prev = S
        return P, float(S_prev), ok

    starts = _initial_paths(x0, x1, stages[0], n_random_starts, rng)
    converged_any = False

    # Coarsest stage: every restart.  Later stages: only the survivors.
    live = []
    for P0 in starts:
        P, S, ok = descend(P0.copy(), stages[0])
        converged_any = converged_any or ok
        live.append((S, P))
    per_start = [float(S) for S, _ in live]

    for n_stage in stages[1:]:
        live.sort(key=lambda t: t[0])
        survivors = live[:max(1, keep)]
        live = []
        for _, P in survivors:
            P2, S2, ok = descend(P, n_stage)
            converged_any = converged_any or ok
            live.append((S2, P2))

    live.sort(key=lambda t: t[0])
    best_S, best_P = live[0]

    return MinimumActionResult(action=float(best_S), path=best_P, n_pts=stages[-1],
                               n_restarts=len(starts), converged=converged_any,
                               per_start=per_start)
