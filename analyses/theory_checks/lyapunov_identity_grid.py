"""The directional variance identity and the scale invariance of the stationary covariance, on a declared grid.

For dy = M y dt + sqrt(2 D) dW the stationary covariance solves M Sigma + Sigma M' + 2 D = 0.  Contracting that
equation with a unit direction e gives e' Sigma e = D_e / lambda_e, with D_e = e' D e and lambda_e the decay rate of
the autocorrelation along e; scaling (M, D) -> (c M, c D) leaves Sigma unchanged.  Both statements are checked here
with the package functions ndif.stationary_covariance and ndif.lyapunov_residual.

Grid
  state dimensions 2, 3, 5, 8, 12 and 20, 200 random systems per dimension, numpy.random.default_rng(20260924);
  per system: Q orthogonal (QR of a standard normal matrix), drift M = -Q diag(lam) Q' (symmetrised) and
  diffusion D = Q diag(dd) Q' (symmetrised), with lam and dd log-uniform over one decade (10^U(-0.5, 0.5));
  the identity is evaluated along each eigenvector of M (the regime in which lambda_e is the eigenvalue: for a
  non-normal M the identity holds only with lambda_e read as the decay rate of the projected process, see the
  docstring of ndif.lyapunov_residual), and the scale factor c is log-uniform on [0.1, 10].
  Relative identity residual: lyapunov_residual(M, D, e) / e' Sigma e.
  Relative scale-invariance residual: max |Sigma(cM, cD) - Sigma(M, D)| / max |Sigma(M, D)|.

Input   none (analytic).
Output  analyses/theory_checks/out/lyapunov_identity_grid.json: maximum and median identity residual over the
        10,000 checks, per-dimension 5th percentile / median / maximum of both residuals, and the median and maximum
        condition number of Sigma over the 1,200 systems.
Reported in  Fig. 1c (identity residual below 1e-14; maximum 8.39e-15 over 10,000 checks).
Runtime  a few seconds on one core.

    python analyses/theory_checks/lyapunov_identity_grid.py
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import ndif                                                           # noqa: E402
from analyses.common import paths                                     # noqa: E402

DIMS = (2, 3, 5, 8, 12, 20)
N_SYS = 200
SEED = 20260924
LO, HI = -0.5, 0.5                       # one decade of eigenvalue spread (log10)


def grid(dims=DIMS, n_sys=N_SYS, seed=SEED):
    """Identity and scale-invariance residuals, and cond(Sigma), over the grid (draw order as declared above)."""
    rng = np.random.default_rng(seed)
    ident, invar, conds, n_checks = {}, {}, [], 0
    for d in dims:
        rel, inv = [], []
        for _ in range(n_sys):
            Q, _ = np.linalg.qr(rng.normal(size=(d, d)))
            lam = 10.0 ** rng.uniform(LO, HI, d)
            dd = 10.0 ** rng.uniform(LO, HI, d)
            M = -(Q * lam) @ Q.T
            M = 0.5 * (M + M.T)
            D = (Q * dd) @ Q.T
            D = 0.5 * (D + D.T)
            Sig = ndif.stationary_covariance(M, D)
            conds.append(float(np.linalg.cond(Sig)))
            for j in range(d):
                e = Q[:, j]
                rel.append(ndif.lyapunov_residual(M, D, e) / abs(float(e @ Sig @ e)))
            c = 10.0 ** rng.uniform(-1.0, 1.0)
            Sc = ndif.stationary_covariance(c * M, c * D)
            inv.append(float(np.max(np.abs(Sc - Sig)) / np.max(np.abs(Sig))))
        ident[d] = np.asarray(rel)
        invar[d] = np.asarray(inv)
        n_checks += len(rel)
    return ident, invar, conds, n_checks


def summary(v):
    return [float(np.percentile(v, 5)), float(np.median(v)), float(v.max())]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--n-sys", type=int, default=N_SYS)
    a = ap.parse_args()
    t0 = time.time()
    ident, invar, conds, n_checks = grid(DIMS, a.n_sys, a.seed)
    all_ident = np.concatenate([ident[d] for d in DIMS])
    all_invar = np.concatenate([invar[d] for d in DIMS])
    out = dict(
        settings=dict(dims=list(DIMS), n_systems_per_dim=a.n_sys, seed=a.seed,
                      eigenvalues="log-uniform over one decade, 10**U(-0.5, 0.5), drift and diffusion",
                      scale_factor="c = 10**U(-1, 1)", directions="eigenvectors of the symmetric drift M",
                      functions="ndif.stationary_covariance, ndif.lyapunov_residual",
                      ndif_version=ndif.__version__, numpy=np.__version__, python=platform.python_version()),
        identity_residual=dict(
            description="|e' Sigma e - D_e / lambda_e| / e' Sigma e",
            max=float(all_ident.max()), median=float(np.median(all_ident)), n_checks=int(n_checks),
            per_dim={str(d): dict(p5_median_max=summary(ident[d]), n=int(ident[d].size)) for d in DIMS}),
        scale_invariance_residual=dict(
            description="max |Sigma(cM, cD) - Sigma(M, D)| / max |Sigma(M, D)|",
            max=float(all_invar.max()), n=int(all_invar.size),
            per_dim={str(d): dict(p5_median_max=summary(invar[d]), n=int(invar[d].size)) for d in DIMS}),
        cond_sigma=dict(median=float(np.median(conds)), max=float(np.max(conds)), n=len(conds)),
        seconds=float(time.time() - t0))
    target = paths.out("analyses", "theory_checks", "out", "lyapunov_identity_grid.json")
    with open(target, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(out, indent=1) + "\n")
    print(f"identity residual: max {out['identity_residual']['max']:.6e} over {n_checks} checks, "
          f"median {out['identity_residual']['median']:.6e}")
    print(f"scale-invariance residual: max {out['scale_invariance_residual']['max']:.6e}")
    print(f"cond(Sigma): median {out['cond_sigma']['median']:.4f}, max {out['cond_sigma']['max']:.4f}")
    print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
