"""Estimators against closed-form answers and injected truth."""

import numpy as np
import pytest

from ndif.estimators import (as_segments, bootstrap_bias_correct, directional_rate,
                             fit_propagator, lyapunov_residual, relaxation_rates,
                             stationary_covariance, white_share)
from ndif.simulate import add_white, simulate_var1


@pytest.mark.parametrize("d", [2, 3, 5, 8])
def test_lyapunov_identity(d):
    """e^T Sigma e = D_e / lambda_e -- the identity the framework rests on."""
    worst = 0.0
    for s in range(6):
        rg = np.random.default_rng(100 * d + s)
        lam = rg.uniform(0.2, 2.0, d)
        U, _ = np.linalg.qr(rg.random((d, d)))
        M = -U @ np.diag(lam) @ U.T
        D0 = rg.uniform(0.1, 1.0)
        e = U[:, rg.integers(d)]
        worst = max(worst, lyapunov_residual(M, D0 * np.eye(d), e))
    assert worst < 1e-12


def test_scaling_drift_and_diffusion_together_leaves_the_density_fixed():
    """The equivalence class, in one assertion.

    (M, D) -> (cM, cD) multiplies every rate by c and leaves the stationary
    covariance -- and therefore every functional of the density -- unchanged.
    """
    rg = np.random.default_rng(0)
    d = 4
    U, _ = np.linalg.qr(rg.random((d, d)))
    M = -U @ np.diag(rg.uniform(0.3, 1.5, d)) @ U.T
    D = np.eye(d) * 0.4
    for c in (0.5, 2.0, 5.0):
        assert np.allclose(stationary_covariance(M, D),
                           stationary_covariance(c * M, c * D), rtol=1e-10)
        a = directional_rate(M, D, U[:, 0])
        b = directional_rate(c * M, c * D, U[:, 0])
        assert abs(b["lambda_e"] / a["lambda_e"] - c) < 1e-10
        assert abs(b["ratio"] - a["ratio"]) < 1e-12


@pytest.mark.parametrize("nu_true", [0.0, 0.2, 0.4, 0.6])
def test_white_share_recovers_injected_nu(nu_true):
    rng = np.random.default_rng(1)
    a, n = 0.8, 200_000
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = a * y[t - 1] + rng.normal() * np.sqrt(1 - a * a)
    sw = np.sqrt(nu_true / (1 - nu_true))
    w = white_share(y[:, None] + rng.normal(size=(n, 1)) * sw)
    assert w["ok"]
    assert abs(w["nu"] - nu_true) < 0.02
    assert abs(w["a"] - a) < 0.02


def test_white_share_reports_failure_rather_than_clipping():
    """A series that violates the model gives ok=False, not a wrong number."""
    rng = np.random.default_rng(2)
    w = white_share(rng.normal(size=(5000, 1)))        # pure white: r(1)~0
    assert w["ok"] is False


def test_propagator_recovers_a_known_var1():
    rg = np.random.default_rng(7)
    d = 4
    A = rg.normal(size=(d, d)) * 0.3
    A *= 0.85 / max(abs(np.linalg.eigvals(A)).max(), 1e-9)
    segs = simulate_var1(A, np.eye(d) * 0.5, 1, 40_000, rng=3)
    f = fit_propagator(segs, dt=1.0, demean="global")
    assert np.abs(f.A - A).max() < 0.03
    assert abs(relaxation_rates(f.A)[0] - relaxation_rates(A)[0]) < 0.02


def test_bias_correction_helps_at_short_blocks():
    rg = np.random.default_rng(7)
    d = 3
    A = np.diag([0.9, 0.8, 0.65])
    segs = simulate_var1(A, np.eye(d) * 0.5, 600, 10, rng=11)
    f = fit_propagator(segs, dt=1.0, demean="block")
    true = relaxation_rates(A)[0]
    raw = relaxation_rates(f.A)[0]
    bc = relaxation_rates(bootstrap_bias_correct(f, n_boot=60, rng=5))[0]
    assert abs(bc - true) < abs(raw - true)


def test_segments_never_pair_across_a_boundary():
    """Two segments with opposite trends must not produce a spurious lag-1 pair."""
    a = np.linspace(0, 1, 50)[:, None]
    b = np.linspace(1, 0, 50)[:, None]
    segs = as_segments([a, b])
    assert len(segs) == 2
    f = fit_propagator(segs, dt=1.0, demean="block")
    g = fit_propagator(np.vstack([a, b]), dt=1.0, demean="block")
    assert not np.allclose(f.A, g.A)


def test_as_segments_rejects_degenerate_input():
    import pytest as _p
    with _p.raises(ValueError):
        as_segments([np.zeros((1, 2))])


def test_ml_fit_uses_complete_segments_of_unequal_length():
    """Segments of different lengths enter the likelihood at their full length.

    A fit that cut every segment to the shortest one would, with 30 segments of
    12 samples and 30 of 240, see 720 of 7,560 samples; the fit on the complete
    segments must use the long ones and recover the generator.
    """
    from ndif.estimators import fit_ar1_kalman, kalman_ll
    rng = np.random.default_rng(5)
    a, nu = 0.8, 0.3

    def seg(n):
        x = np.zeros(n)
        x[0] = rng.normal()
        for t in range(1, n):
            x[t] = a * x[t - 1] + rng.normal() * np.sqrt(1 - a * a)
        return x + rng.normal(size=n) * np.sqrt(nu / (1 - nu))

    segs = [seg(12) for _ in range(30)] + [seg(240) for _ in range(30)]
    full = fit_ar1_kalman(segs, seed=0)
    cut = fit_ar1_kalman([s[:12] for s in segs], seed=0)
    assert abs(full["a"] - a) < 0.08 and abs(full["nu"] - nu) < 0.12
    # the reported log-likelihood is the sum over the two groups of complete segments
    groups = [np.column_stack([s - s.mean() for s in segs if len(s) == n]) for n in (12, 240)]
    assert abs(full["ll"] - sum(kalman_ll(g, full["a"], full["sw2"], full["se2"]) for g in groups)) < 1e-6
    assert full["ll"] < cut["ll"] - 100.0          # far more samples than the cut likelihood
    # segments of equal length: one group, the same computation as before
    same = fit_ar1_kalman([s[:12] for s in segs[:30]], seed=0)
    Y = np.column_stack([s[:12] - s[:12].mean() for s in segs[:30]])
    assert abs(same["ll"] - kalman_ll(Y, same["a"], same["sw2"], same["se2"])) < 1e-9
