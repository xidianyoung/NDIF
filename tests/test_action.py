"""The minimum-action solver, against closed-form answers.

Three families, chosen so that a solver cannot pass by accident:

1. one-dimensional double well -- the easy case, where a translation-only
   descent also succeeds, so this control alone cannot tell the two apart;
2. two-dimensional gradient system, minimum to saddle -- a *curved* instanton,
   where a translation-only descent cannot succeed;
3. linear non-normal Ornstein-Uhlenbeck in 2-4 dimensions, where the exact
   action is ``0.5 x^T Sigma^-1 x`` and the system is not a gradient flow.
"""

import numpy as np
import pytest
from scipy.linalg import solve_lyapunov

from ndif.action import (action_gradient, geometric_action, minimum_action,
                         reparameterise)

K, DS = 6.0, 0.05
DM = DS * np.eye(2)


def double_well_2d(P):
    """b = -grad U, U = x^4/4 - x^2/2 + (K/2)(y-(x^2-1))^2."""
    P = np.atleast_2d(np.asarray(P, float))
    x, y = P[:, 0], P[:, 1]
    g = y - (x ** 2 - 1.0)
    return -np.column_stack([x * (x ** 2 - 1.0) + K * g * (-2.0 * x), K * g])


def double_well_1d(P):
    P = np.atleast_2d(np.asarray(P, float))
    return -np.column_stack([P[:, 0] * (P[:, 0] ** 2 - 1.0)])


# --------------------------------------------------------------------- basics
def test_gradient_matches_finite_differences():
    """The analytic gradient is the exact gradient of the discretised action."""
    rng = np.random.default_rng(0)
    P = np.column_stack([np.linspace(-1, 1, 15), rng.normal(0, 0.3, 15)])
    G = action_gradient(P, double_well_2d, DM)
    Gn = np.zeros_like(P)
    h = 1e-7
    for i in range(P.shape[0]):
        for k in range(2):
            Pp, Pm = P.copy(), P.copy()
            Pp[i, k] += h
            Pm[i, k] -= h
            Gn[i, k] = (geometric_action(Pp, double_well_2d, DM)
                        - geometric_action(Pm, double_well_2d, DM)) / (2 * h)
    assert np.abs(G - Gn).max() / np.abs(Gn).max() < 1e-6


def test_action_is_zero_along_the_flow():
    """S vanishes on a path that follows the deterministic flow."""
    x = np.array([0.9, 0.0])
    P = [x.copy()]
    for _ in range(4000):
        x = x + double_well_2d(x[None, :])[0] * 1e-3
        P.append(x.copy())
    P = np.asarray(P)
    assert geometric_action(P, double_well_2d, DM) < 1e-6


def test_reparameterise_is_near_idempotent_and_converges():
    """Arclength redistribution is idempotent only to the discretisation order.

    It interpolates linearly between nodes, so on a curved path a second
    application moves the nodes by O(h^2), not by zero.  What must hold is that
    the movement shrinks with each application rather than drifting.
    """
    P = np.column_stack([np.linspace(0, 1, 40), np.linspace(0, 1, 40) ** 2])
    Q1 = reparameterise(P)
    Q2 = reparameterise(Q1)
    Q3 = reparameterise(Q2)
    d1 = np.abs(Q2 - Q1).max()
    d2 = np.abs(Q3 - Q2).max()
    assert d1 < 1e-4          # O(h^2) at h = 1/39
    assert d2 < d1            # and it settles rather than drifting


# ------------------------------------------------------ closed-form recoveries
def test_one_dimensional_double_well():
    """Exact action from the well to the saddle is dU/sigma^2 = 0.25/0.05 = 5."""
    r = minimum_action([-1.0], [0.0], double_well_1d, DS * np.eye(1),
                       n_pts=201, n_random_starts=0)
    assert abs(r.action - 5.0) / 5.0 < 1e-3


def test_two_dimensional_curved_instanton():
    """Minimum to saddle across a curved instanton; exact action 5.0.

    This is the case a translation-only descent fails: a numerical gradient
    that is the derivative under a rigid translation of the whole path cannot
    bend the curve.
    """
    r = minimum_action([-1.0, 0.0], [0.0, -1.0], double_well_2d, DM, n_pts=201)
    assert abs(r.action - 5.0) / 5.0 < 1e-3


def test_two_dimensional_min_to_min_converges_with_resolution():
    """Minimum to minimum: the descent branch contributes only discretisation
    error, so the estimate converges from above as nodes are added."""
    errs = [abs(minimum_action([-1.0, 0.0], [1.0, 0.0], double_well_2d, DM,
                               n_pts=n).action - 5.0) / 5.0
            for n in (101, 401)]
    assert errs[1] < errs[0] / 4
    assert errs[1] < 1e-3


@pytest.mark.parametrize("d,seed", [(2, 1), (2, 7), (3, 2), (4, 5)])
def test_linear_non_normal_ou(d, seed):
    """Non-gradient linear system: exact action is 0.5 x^T Sigma^-1 x."""
    rg = np.random.default_rng(seed)
    A = rg.normal(size=(d, d))
    A = A - (np.max(np.linalg.eigvals(A).real) + 0.6) * np.eye(d)
    L = rg.normal(size=(d, d)) * 0.4
    D = L @ L.T + 0.3 * np.eye(d)
    Sig = solve_lyapunov(A, -2 * D)
    xs = rg.normal(size=d)
    xs = xs / np.linalg.norm(xs) * 0.8
    exact = 0.5 * xs @ np.linalg.solve(Sig, xs)
    r = minimum_action(np.zeros(d), xs, lambda P: np.atleast_2d(P) @ A.T, D, n_pts=201)
    assert abs(r.action - exact) / exact < 1e-4


def test_translation_only_descent_fails_where_ours_succeeds():
    """Regression case: a translation-only descent cannot reach a curved instanton.

    A descent whose gradient is constant along the path -- which is what
    assigning one finite-difference number to every interior node produces --
    cannot reach a curved instanton.  Reproduced here so the failure is
    documented in the package rather than only in the paper.
    """
    x0, x1 = np.array([-1.0, 0.0]), np.array([0.0, -1.0])
    n = 101
    P = x0 + np.linspace(0, 1, n)[:, None] * (x1 - x0)
    step = 0.05 * np.linalg.norm(x1 - x0) / n
    best = geometric_action(P, double_well_2d, DM)
    for _ in range(300):
        G = np.zeros_like(P)
        h = 1e-6
        for k in range(2):
            Pp, Pm = P.copy(), P.copy()
            Pp[1:-1, k] += h
            Pm[1:-1, k] -= h
            # one number, broadcast to every interior node -- the defect
            G[1:-1, k] = (geometric_action(Pp, double_well_2d, DM)
                          - geometric_action(Pm, double_well_2d, DM)) / (2 * h)
        gn = np.linalg.norm(G)
        if gn < 1e-14:
            break
        P = P - step * G / gn * np.sqrt(n)
        P[0], P[-1] = x0, x1
        P = reparameterise(P)
        best = min(best, geometric_action(P, double_well_2d, DM))
    ours = minimum_action(x0, x1, double_well_2d, DM, n_pts=n).action
    assert abs(ours - 5.0) / 5.0 < 1e-2
    assert best > 1.5 * 5.0          # the translation-only descent does not get there
