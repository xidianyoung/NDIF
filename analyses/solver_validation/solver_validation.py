"""Validation of the minimum-action solver against problems with closed-form actions.

Computes every value of Extended Data Fig. 5 and the solver checks quoted in the Methods
(Software implementation):

  a  action error |S - 5| against node count on the curved-valley problem, for the path from
     a minimum to the saddle and for the path from one minimum to the other through the
     saddle (exact action 5.0 on both); seven solves per path at 11-641 nodes, and the
     least-squares slope of log|S - 5| on log(nodes) for the minimum-to-saddle path over
     41-641 nodes;
  b  computed against exact action 0.5 x' Sigma^-1 x for fifteen random non-normal linear
     systems, five each in two, three and four dimensions, at 201 nodes; the four systems
     of tests/test_action.py are flagged, and the non-normality index
     ||A A' - A' A|| / ||A||^2 (Frobenius norms) of each drift matrix is tabulated;
  c  the 641-node minimum-to-minimum path of the solver and the 81-node path of the
     translation-only control descent, with the action of the straight line between the
     minima;
  Methods: the analytic gradient against central finite differences on a 15-node path, and
     the minimum-to-saddle action at 401 nodes.

Curved-valley problem
---------------------
U(x, y) = x^4/4 - x^2/2 + 3 (y - x^2 + 1)^2, drift b = -grad U, diffusion D = 0.05 I
(a = 2D).  Minima (-1, 0) and (1, 0), saddle (0, -1), barrier dU = 1/4, so the minimum action
from a minimum to the saddle is dU / 0.05 = 5.0.  From one minimum to the other the minimum
is also 5.0: beyond the saddle the path follows the flow, where the action density is zero.
The minimum-action path follows the curved valley near y = x^2 - 1 and passes through the
saddle, where |y| = 1; a path that can only be translated cannot follow it.

Linear systems
--------------
For (d, seed), with rng = numpy.random.default_rng(seed): A ~ N(0, 1)^(d x d) shifted by
-(max Re eig(A) + 0.6) I; D = L L' + 0.3 I with L ~ 0.4 N(0, 1)^(d x d); Sigma solves
A Sigma + Sigma A' = -2D; the target x is a N(0, I) draw scaled to norm 0.8.  The drift is
b(x) = A x and the exact action from the origin to x is 0.5 x' Sigma^-1 x.  The generator is
the one of tests/test_action.py::test_linear_non_normal_ou.

Solver settings
---------------
ndif.action.minimum_action with its default arguments throughout (straight line plus four
transverse-bulge starts, seed 0; up to 12 reparameterise-and-minimise cycles per stage;
L-BFGS-B with maxiter 300, ftol 1e-13, gtol 1e-11; two starts kept between stages;
rtol 1e-11; drift Jacobian by central differences), with one exception in panel a: the node
schedule is the default stages (21, 51) that lie below the final node count, followed by the
final count,

    11 and 21 nodes: (n,)      41 nodes: (21, n)      81 to 641 nodes: (21, 51, n)

For 81 nodes and above this is the default schedule.  Below 51 nodes the default schedule
would refine to 51 nodes and then resample down to the final count; the schedule used here
never passes through more nodes than the final count.  The 641-node minimum-to-minimum solve
of panel a is also the path drawn in panel c.

Translation-only control
------------------------
translation_only_descent below.  Each iteration computes, for each coordinate k, one central
difference of the action under a rigid shift of all interior nodes along k, and moves every
interior node by the same vector; see its docstring for the step rule.  Run from the 81-node
straight line between the minima for 600 iterations.  The straight-line action quoted with it
is evaluated on 201 nodes (straight_line_action); the action of the control's own 81-node
initial path is reported as path_control_initial_action.

Run
---
    python analyses/solver_validation/solver_validation.py

Writes analyses/solver_validation/out/solver_validation.json (two to three minutes on one
core; the run time is recorded under "seconds").  The keys of the drawn quantities are those of
the Source Data file solver_validation.json; the values quoted in the text are compared
under "quoted".
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time

import numpy as np
import scipy
from scipy.linalg import solve_lyapunov

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))   # the repository root

from ndif.action import (action_gradient, geometric_action,      # noqa: E402
                         minimum_action, reparameterise)

# ------------------------------------------------------------------ curved-valley problem
K_VALLEY = 6.0                       # U = x^4/4 - x^2/2 + (K/2)(y - (x^2 - 1))^2
D_SCALAR = 0.05
D_MAT = D_SCALAR * np.eye(2)
X_MIN_L = np.array([-1.0, 0.0])
X_MIN_R = np.array([1.0, 0.0])
X_SADDLE = np.array([0.0, -1.0])
EXACT_ACTION = 0.25 / D_SCALAR       # dU / D = 5.0

NODES = [11, 21, 41, 81, 161, 321, 641]      # panel a
FIT_FROM = 41                                # first node count of the slope fit
DEFAULT_STAGES = (21, 51)                    # coarse stages of the default schedule
N_SADDLE_TEXT = 401                          # minimum-to-saddle solve quoted in the Methods
N_LINEAR = 201                               # panel b
N_CONTROL, ITER_CONTROL, LR_CONTROL = 81, 600, 0.05
N_STRAIGHT = 201                             # straight-line action quoted with the control

# ------------------------------------------------------------------ linear systems
LINEAR_SYSTEMS = [(2, 1), (2, 7), (2, 11), (2, 13), (2, 101),
                  (3, 2), (3, 3), (3, 17), (3, 19), (3, 103),
                  (4, 5), (4, 23), (4, 29), (4, 31), (4, 107)]
TEST_SUITE = {(2, 1), (2, 7), (3, 2), (4, 5)}    # tests/test_action.py::test_linear_non_normal_ou

# ------------------------------------------------------------------ values quoted in the text
# (quantity, quoted value, format applied to the computed value)
QUOTED = [
    ("log-log slope of |S - 5|, minimum to saddle, 41-641 nodes", "-2.025", "%.3f"),
    ("solves per curve in panel a", "7", "%d"),
    ("linear systems in panel b", "15", "%d"),
    ("relative error, minimum over the 15 linear systems", "2.41e-12", "%.2e"),
    ("relative error, maximum over the 15 linear systems", "6.84e-05", "%.2e"),
    ("relative error, minimum over the 4 test-suite systems", "5.19e-12", "%.2e"),
    ("relative error, maximum over the 4 test-suite systems", "8.75e-07", "%.2e"),
    ("action, minimum to minimum, 641 nodes", "5.000569", "%.6f"),
    ("peak |y|, minimum-to-minimum path, 641 nodes", "1.0011", "%.4f"),
    ("peak |y|, translation-only control", "0.379", "%.3f"),
    ("terminal action, translation-only control", "69.178", "%.3f"),
    ("action of the straight line between the minima (201 nodes)", "108.18", "%.2f"),
    ("analytic gradient against finite differences, max relative difference", "2.24e-09", "%.2e"),
    ("action, minimum to saddle, 401 nodes", "5.000021", "%.6f"),
]


def drift_valley(P):
    """b = -grad U of the curved valley (same expression as tests/test_action.py)."""
    P = np.atleast_2d(np.asarray(P, float))
    x, y = P[:, 0], P[:, 1]
    g = y - (x ** 2 - 1.0)
    return -np.column_stack([x * (x ** 2 - 1.0) + K_VALLEY * g * (-2.0 * x), K_VALLEY * g])


def schedule_below(n, stages=DEFAULT_STAGES):
    """Coarse-to-fine schedule: the stages below ``n`` nodes, then ``n`` (``None``)."""
    return tuple(s for s in stages if s < n) + (None,)


def linear_system(d, seed):
    """Random non-normal stable linear system: drift matrix, diffusion, target, exact action."""
    rg = np.random.default_rng(seed)
    A = rg.normal(size=(d, d))
    A = A - (np.max(np.linalg.eigvals(A).real) + 0.6) * np.eye(d)
    L = rg.normal(size=(d, d)) * 0.4
    Dm = L @ L.T + 0.3 * np.eye(d)
    Sig = solve_lyapunov(A, -2 * Dm)
    xs = rg.normal(size=d)
    xs = xs / np.linalg.norm(xs) * 0.8
    exact = float(0.5 * xs @ np.linalg.solve(Sig, xs))
    return A, Dm, xs, exact


def translation_only_descent(x0, x1, drift_fn, D, n_pts=81, n_iter=600, lr=0.05):
    """Descent in which all interior nodes receive the same coordinate update.

    Starts from the straight line on ``n_pts`` nodes.  Each iteration evaluates, for each
    coordinate k,

        G[1:-1, k] = (S(P + h e_k on all interior nodes) - S(P - h e_k on all interior nodes)) / (2h),
        h = 1e-6 (1 + max |P|),

    a single number per coordinate, and updates P <- P - step * G / |G| * sqrt(n_pts), with
    the initial step lr |x1 - x0| / n_pts.  The endpoints are then reset and the nodes
    redistributed uniformly in arclength.  Whenever the action of the new path is not below
    the lowest action so far, the step is multiplied by 0.85 (the path continues from the new
    iterate).  Stops after ``n_iter`` iterations, when |G| < 1e-14 or when the step falls
    below 1e-12.

    Returns
    -------
    (float, ndarray)
        The lowest action reached and its path.
    """
    x0, x1 = np.asarray(x0, float), np.asarray(x1, float)
    t = np.linspace(0, 1, n_pts)[:, None]
    P = x0 + t * (x1 - x0)
    best, best_S = P.copy(), geometric_action(P, drift_fn, D)
    step = lr * np.linalg.norm(x1 - x0) / n_pts
    for _ in range(n_iter):
        G = np.zeros_like(P)
        h = 1e-6 * (1 + np.abs(P).max())
        for k in range(P.shape[1]):
            Pp, Pm = P.copy(), P.copy()
            Pp[1:-1, k] += h
            Pm[1:-1, k] -= h
            G[1:-1, k] = (geometric_action(Pp, drift_fn, D)
                          - geometric_action(Pm, drift_fn, D)) / (2 * h)
        gn = np.linalg.norm(G)
        if gn < 1e-14:
            break
        P = P - step * G / gn * np.sqrt(n_pts)
        P[0], P[-1] = x0, x1
        P = reparameterise(P)
        S = geometric_action(P, drift_fn, D)
        if S < best_S:
            best, best_S = P.copy(), S
        else:
            step *= 0.85
            if step < 1e-12:
                break
    return best_S, best


def gradient_check(n=15, h=1e-7, seed=0):
    """Max |analytic - central-difference gradient| / max |central-difference gradient|.

    On the 15-node path x = linspace(-1, 1), y ~ N(0, 0.3^2) (seed 0) of
    tests/test_action.py::test_gradient_matches_finite_differences, every node and
    coordinate perturbed by +-h.
    """
    rg = np.random.default_rng(seed)
    P = np.column_stack([np.linspace(-1, 1, n), rg.normal(0, 0.3, n)])
    Ga = action_gradient(P, drift_valley, D_MAT)
    Gn = np.zeros_like(P)
    for i in range(P.shape[0]):
        for k in range(P.shape[1]):
            Pp, Pm = P.copy(), P.copy()
            Pp[i, k] += h
            Pm[i, k] -= h
            Gn[i, k] = (geometric_action(Pp, drift_valley, D_MAT)
                        - geometric_action(Pm, drift_valley, D_MAT)) / (2 * h)
    return float(np.abs(Ga - Gn).max() / np.abs(Gn).max())


def straight_line(x0, x1, n):
    return x0 + np.linspace(0, 1, n)[:, None] * (x1 - x0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "out", "solver_validation.json"))
    a = ap.parse_args()

    t0 = time.time()
    out = {"nodes": NODES}

    # a: action against node count on both paths
    paths = {"saddle": (X_MIN_L, X_SADDLE), "through": (X_MIN_L, X_MIN_R)}
    finest = {}
    for name, (xa, xb) in paths.items():
        vals = []
        for n in NODES:
            r = minimum_action(xa, xb, drift_valley, D_MAT, n_pts=n, schedule=schedule_below(n))
            vals.append(float(r.action))
            if n == NODES[-1]:
                finest[name] = r
            print(f"  [a] {name:8s} {n:4d} nodes  action {r.action:.9f}", flush=True)
        out[f"solver_{name}"] = vals

    # b: non-normal linear systems
    rows, nonnormality = [], []
    for d, seed in LINEAR_SYSTEMS:
        A, Dm, xs, exact = linear_system(d, seed)
        drift = lambda P, A=A: np.atleast_2d(np.asarray(P, float)) @ A.T      # noqa: E731
        r = minimum_action(np.zeros(d), xs, drift, Dm, n_pts=N_LINEAR)
        rows.append(dict(d=d, seed=seed, exact=exact, solver=float(r.action),
                         in_test_suite=(d, seed) in TEST_SUITE))
        nonnormality.append(float(np.linalg.norm(A @ A.T - A.T @ A) / np.linalg.norm(A) ** 2))
        print(f"  [b] d={d} seed={seed:3d}  exact {exact:.9f}  solver {r.action:.9f}", flush=True)
    out["linear"] = rows
    out["linear_nonnormality"] = nonnormality

    # c: the 641-node minimum-to-minimum path of panel a, the control, the straight line
    r_c = finest["through"]
    s_ctl, P_ctl = translation_only_descent(X_MIN_L, X_MIN_R, drift_valley, D_MAT,
                                            n_pts=N_CONTROL, n_iter=ITER_CONTROL, lr=LR_CONTROL)
    print(f"  [c] translation-only control {N_CONTROL} nodes, {ITER_CONTROL} iterations: "
          f"action {s_ctl:.6f}", flush=True)
    out["path_solver"] = np.asarray(r_c.path).tolist()
    out["path_solver_action"] = float(r_c.action)
    out["path_solver_max_abs_y"] = float(np.abs(np.asarray(r_c.path)[:, 1]).max())
    out["path_control"] = np.asarray(P_ctl).tolist()
    out["path_control_action"] = float(s_ctl)
    out["path_control_max_abs_y"] = float(np.abs(np.asarray(P_ctl)[:, 1]).max())
    out["straight_line_action"] = float(geometric_action(straight_line(X_MIN_L, X_MIN_R, N_STRAIGHT),
                                                         drift_valley, D_MAT))
    out["path_control_initial_action"] = float(geometric_action(
        straight_line(X_MIN_L, X_MIN_R, N_CONTROL), drift_valley, D_MAT))

    # Methods: gradient check and the 401-node minimum-to-saddle solve
    out["gradient_max_relative_difference"] = gradient_check()
    r401 = minimum_action(X_MIN_L, X_SADDLE, drift_valley, D_MAT, n_pts=N_SADDLE_TEXT)
    out["saddle_action_401"] = float(r401.action)
    print(f"  [methods] minimum to saddle, {N_SADDLE_TEXT} nodes: action {r401.action:.9f}", flush=True)

    # derived values
    nn = np.array(NODES, float)
    err_s = np.abs(np.array(out["solver_saddle"]) - EXACT_ACTION)
    i0 = NODES.index(FIT_FROM)
    out["slope_saddle"] = float(np.polyfit(np.log(nn[i0:]), np.log(err_s[i0:]), 1)[0])
    ex = np.array([r["exact"] for r in rows])
    sv = np.array([r["solver"] for r in rows])
    in_test = np.array([r["in_test_suite"] for r in rows])
    rel = np.abs(sv - ex) / ex
    out["linear_relative_error"] = rel.tolist()
    out["linear_relative_error_range"] = [float(rel.min()), float(rel.max())]
    out["linear_relative_error_range_test_suite"] = [float(rel[in_test].min()), float(rel[in_test].max())]

    computed = [out["slope_saddle"], len(NODES), len(rows),
                rel.min(), rel.max(), rel[in_test].min(), rel[in_test].max(),
                out["path_solver_action"], out["path_solver_max_abs_y"],
                out["path_control_max_abs_y"], out["path_control_action"],
                out["straight_line_action"], out["gradient_max_relative_difference"],
                out["saddle_action_401"]]
    quoted = []
    for (what, q, fmt), v in zip(QUOTED, computed):
        s = fmt % v
        quoted.append(dict(quantity=what, quoted=q, computed=s, value=float(v), agrees=s == q))
    out["quoted"] = quoted

    out["settings"] = dict(
        potential="U = x^4/4 - x^2/2 + %g (y - x^2 + 1)^2, drift -grad U" % (0.5 * K_VALLEY),
        D=D_SCALAR, minima=[X_MIN_L.tolist(), X_MIN_R.tolist()], saddle=X_SADDLE.tolist(),
        exact_action=EXACT_ACTION,
        panel_a_schedules={str(n): [s if s is not None else n for s in schedule_below(n)]
                           for n in NODES},
        slope_fit_nodes=NODES[i0:],
        linear_systems=[list(t) for t in LINEAR_SYSTEMS], linear_nodes=N_LINEAR,
        test_suite_systems=sorted(list(t) for t in TEST_SUITE),
        control=dict(n_pts=N_CONTROL, n_iter=ITER_CONTROL, lr=LR_CONTROL),
        straight_line_nodes=N_STRAIGHT,
        gradient_check=dict(n_nodes=15, h=1e-7, seed=0),
        saddle_text_nodes=N_SADDLE_TEXT,
        solver="ndif.action.minimum_action, default arguments except the panel a schedules",
    )
    out["environment"] = dict(python=sys.version.split()[0], numpy=np.__version__,
                              scipy=scipy.__version__, platform=platform.platform())
    out["seconds"] = round(time.time() - t0, 1)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)

    print()
    for q in quoted:
        print(f"  {'agrees' if q['agrees'] else 'DIFFERS':7s} {q['quantity']}: "
              f"computed {q['computed']}, quoted {q['quoted']}")
    print(f"  action of the control's {N_CONTROL}-node initial path: "
          f"{out['path_control_initial_action']:.5f}")
    try:
        shown = os.path.relpath(a.out, os.path.dirname(os.path.dirname(HERE)))
    except ValueError:                       # output on another drive
        shown = a.out
    print(f"  wrote {shown} ({out['seconds']} s)")


if __name__ == "__main__":
    main()
