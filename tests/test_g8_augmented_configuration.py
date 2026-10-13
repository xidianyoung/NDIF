"""Components of the G8-augmented configuration.

* The batched maximum-likelihood fit is the estimator of :func:`fit_ar1_kalman` (bit-identical for equal segments).
* Approximate Bayesian computation is a function of the data.
* The recovery check corrects its generator by indirect inference and decides only beyond Monte Carlo error.
* The noise check refuses when plausible noise fractions change the rate by more than the tolerance.
* The separation check uses a null that permutes condition labels across segments.
* A recording without temporal structure is not admitted (spread criterion).
"""
import numpy as np
import pytest

from ndif import Design, Recording, certify
from ndif.estimators import fit_abc, fit_ar1_kalman, fit_ar1_kalman_batch
from functools import partial

from ndif.gates import _corrected_generator, _mc_combine, _mc_criterion
from ndif.gates import g1_instrument as _g1, g3_recovery as _g3, g5_floor as _g5
# the forms of G1, G3 and G5 that the G8-augmented configuration prescribes
g1_instrument = partial(_g1, rule="profile")
g3_recovery = partial(_g3, generator="corrected")
g5_floor = partial(_g5, null="permutation")


def _ar1(rng, a, n, nu=0.0):
    y = np.zeros(n)
    y[0] = rng.normal()
    e = rng.normal(scale=np.sqrt(1 - a * a), size=n)
    for t in range(1, n):
        y[t] = a * y[t - 1] + e[t]
    if nu > 0:
        y = y + rng.normal(scale=np.sqrt(nu / (1 - nu)), size=n)
    return y


def _rec(segs, labels=None, dt=1.0):
    lens = [len(s) for s in segs]
    e = np.cumsum([0] + lens)
    X = np.concatenate([np.atleast_2d(np.asarray(s).T).T if np.ndim(s) == 1 else s for s in segs])
    if X.ndim == 1:
        X = X[:, None]
    return Recording(X, Design(dt=dt, window_s=dt, step_s=dt, labels=labels,
                               segments=[(int(e[i]), int(e[i + 1])) for i in range(len(lens))]))


def test_batched_ml_fit_equals_the_reference_for_equal_segments():
    rng = np.random.default_rng(1)
    ds = [[_ar1(rng, a, 60, nu) for _ in range(20)] for a, nu in ((0.5, 0.0), (0.85, 0.3), (0.95, 0.1))]
    for d, b in zip(ds, fit_ar1_kalman_batch(ds)):
        r = fit_ar1_kalman(d, n_restart=1, maxiter=300)
        assert r["a"] == b["a"] and r["sw2"] == b["sw2"] and r["se2"] == b["se2"] and r["ll"] == b["ll"]


def test_batched_ml_fit_agrees_for_unequal_segments():
    rng = np.random.default_rng(2)
    lens = list(rng.integers(20, 120, size=12))
    ds = [[_ar1(rng, 0.8, int(L), 0.2) for L in lens] for _ in range(3)]
    for d, b in zip(ds, fit_ar1_kalman_batch(ds)):
        r = fit_ar1_kalman(d, n_restart=1, maxiter=300)
        assert abs(np.log(-np.log(r["a"])) - np.log(-np.log(b["a"]))) < 1e-5


def test_abc_is_a_function_of_the_data():
    rng = np.random.default_rng(3)
    segs = [_ar1(rng, 0.85, 100, 0.2) for _ in range(10)]
    a = fit_abc(segs)["lam"]
    assert a == fit_abc(segs)["lam"]                         # same data, same estimate
    other = [s * 1.0001 for s in segs]
    assert np.isfinite(a) and fit_abc(other)["lam"] != a     # other data, its own random stream


def test_monte_carlo_decisions():
    assert _mc_criterion(1.05, 1.0, 0.2, 0.01, 2.0)[0] == "pass"
    assert _mc_criterion(1.35, 1.0, 0.2, 0.01, 2.0)[0] == "fail"
    assert _mc_criterion(1.19, 1.0, 0.2, 0.05, 2.0)[0] == "ambiguous"
    assert _mc_criterion(0.1, None, 0.2, 0.01, 2.0)[0] == "pass"            # spread: one-sided
    assert _mc_combine(["pass", "fail", "ambiguous"]) == "fail"
    assert _mc_combine(["pass", "ambiguous"]) == "ambiguous"
    assert _mc_combine(["pass", "pass"]) == "pass"


def test_generator_correction_reduces_the_bias_of_the_fit():
    """At persistence 0.95 in segments of 60 samples the fit is biased towards faster relaxation; the corrected
    generator is closer to the truth on average."""
    err_fit, err_cor = [], []
    for seed in range(4):
        rng = np.random.default_rng(10 + seed)
        segs = [_ar1(rng, 0.95, 60) for _ in range(20)]
        g = fit_ar1_kalman_batch([segs])[0]
        a_star, _, _, _ = _corrected_generator(g["a"], g["nu"], [60] * 20, 32, 2, True, np.random.default_rng(seed))
        truth = np.log(-np.log(0.95))
        err_fit.append(np.log(-np.log(g["a"])) - truth)
        err_cor.append(np.log(-np.log(a_star)) - truth)
    assert np.mean(err_fit) > 0.2 and abs(np.mean(err_cor)) < 0.5 * np.mean(err_fit)


def test_recovery_check_refuses_a_biased_estimator_at_a_slow_rate():
    rng = np.random.default_rng(4)
    rec = _rec([_ar1(rng, 0.95, 60) for _ in range(20)])
    r = g3_recovery(rec, estimator="kalman", n_rep=12, rng=1)
    assert r.passed is False and r.detail["bias_log"] > 0.2
    assert r.detail["rate"]["a_generator"] > r.detail["a_fit"]


def test_noise_check_refuses_when_the_noise_fraction_is_not_determined():
    rng = np.random.default_rng(5)
    fast_noisy = _rec([_ar1(rng, 0.45, 60, 0.3) for _ in range(12)])        # fast dynamics, noise, short
    slow = _rec([_ar1(rng, 0.85, 150, 0.2) for _ in range(40)])              # determined
    r1 = g1_instrument(fast_noisy, estimator="kalman")
    r2 = g1_instrument(slow, estimator="kalman")
    assert r1.passed is False and r1.value > 0.2
    assert r2.passed is True and r2.value <= 0.2


def test_separation_null_permutes_labels_across_segments():
    rng = np.random.default_rng(6)
    n_seg, L = 24, 50
    # segment means differ at random (blocks, drifts) but not between conditions
    segs = [(_ar1(rng, 0.5, L) + rng.normal(scale=1.0))[:, None] for _ in range(n_seg)]
    lab = np.concatenate([np.full(L, i % 2) for i in range(n_seg)])
    r = g5_floor(_rec(segs, labels=lab), n_perm=400, rng=1)
    assert r.passed is not True                     # the segment random effect is part of the null
    shifted = [s + (1.5 if i % 2 else 0.0) for i, s in enumerate(segs)]
    r2 = g5_floor(_rec(shifted, labels=lab), n_perm=400, rng=1)
    assert r2.passed is True and r2.detail["p_value"] <= 0.05
    mixed = lab.copy()
    mixed[:L // 2] = 1 - mixed[:L // 2]             # a segment that contains both conditions
    assert g5_floor(_rec(segs, labels=mixed), n_perm=50, rng=1).passed is None
    few = np.concatenate([np.full(L, i % 2) for i in range(4)])
    assert g5_floor(_rec(segs[:4], labels=few), n_perm=50, rng=1).passed is None   # 6 arrangements


def test_a_series_without_temporal_structure_is_not_admitted():
    rng = np.random.default_rng(7)
    segs = [rng.normal(size=60) for _ in range(30)]                         # white noise
    rec = _rec(segs)
    for est in ("ols", "kalman"):
        c = certify(rec.X, rec.design, which=["G0", "G2", "G8", "G1", "G3"], configuration="g8_augmented",
                    rng=1, estimator=est,
                    G3=dict(n_rep=8), G8=dict(n_rep=40))
        assert c.levels["L3"].status != "admitted", est
