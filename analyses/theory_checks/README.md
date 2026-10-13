# Analytic checks: the directional variance identity

| Script | Output | Reported in |
|---|---|---|
| `lyapunov_identity_grid.py` | `out/lyapunov_identity_grid.json` | Fig. 1c: relative residual of e' Sigma e = D_e / lambda_e at most 8.39e-15 over 10,000 checks (median 7.95e-16; per dimension 2, 3, 5, 8, 12, 20: 5th percentile, median and maximum); scale-invariance residual of Sigma under (M, D) -> (cM, cD) at most 7.47e-15; cond(Sigma) median 11.30, maximum 75.14 |

```bash
python analyses/theory_checks/lyapunov_identity_grid.py      # a few seconds on one core
```

The grid is declared in the script: 200 random systems in each of the state dimensions 2, 3, 5, 8, 12 and 20
(seed 20260924), symmetric drift and diffusion with eigenvalues log-uniform over one decade, the identity
evaluated along each eigenvector of the drift, and a scale factor log-uniform on [0.1, 10]. The computation uses
the package functions `ndif.stationary_covariance` and `ndif.lyapunov_residual`. No input data are read. The
stored output equals the values of the Source Data file `figure_tables/Fig1/source_summary.csv` (panel c)
exactly; the run time is recorded under `seconds`.
