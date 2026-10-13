"""G8, the model-adequacy check.

A single-timescale recording passes under both estimators; a recording with two timescales or with autocorrelated
noise is refused (the likelihood ratio against two timescales and the short-lag autocorrelation); segments shorter
than four relaxation times make the check indeterminate; a degenerate series is indeterminate; and the certificate
consults G8 after the sampling check and before the noise and recovery checks.
"""
import numpy as np
import pytest

from ndif import Design, certify
from ndif.configurations import CONFIGURATIONS

LEVEL_GATES = CONFIGURATIONS["g8_augmented"]["level_gates"]
from ndif.estimators import padded_centred, simulate_eq6_padded, whittle_lr_two_timescale
from ndif.gates import g8_adequacy
from ndif.types import Recording


def _rec(segs, dt=1.0):
    lens = [len(s) for s in segs]
    e = np.cumsum([0] + lens)
    X = np.concatenate(segs)[:, None]
    return Recording(X, Design(dt=dt, window_s=dt, step_s=dt,
                               segments=[(int(e[i]), int(e[i + 1])) for i in range(len(lens))]))


def _ar1(rng, a, n, nu=0.0):
    y = np.zeros(n)
    y[0] = rng.normal()
    e = rng.normal(scale=np.sqrt(1 - a * a), size=n)
    for t in range(1, n):
        y[t] = a * y[t - 1] + e[t]
    if nu > 0:
        y = y + rng.normal(scale=np.sqrt(nu / (1 - nu)), size=n)
    return y


@pytest.mark.parametrize("est,nu", [("ols", 0.0), ("kalman", 0.0), ("kalman", 0.4)])
def test_single_timescale_passes(est, nu):
    # a typical recording with a fixed seed; recordings whose combined P value lies within Monte Carlo error of
    # 0.05 are indeterminate
    rng = np.random.default_rng(6)
    rec = _rec([_ar1(rng, 0.85, 150, nu) for _ in range(40)])
    g = g8_adequacy(rec, estimator=est, rng=1, n_rep=60)
    assert g.gate == "G8" and g.passed is True
    assert 0.0 < g.value <= 1.0 and g.margin > 0
    assert g.detail["estimator"] == est and len(g.detail["lags_used"]) >= 2


@pytest.mark.parametrize("est", ["ols", "kalman"])
def test_two_timescales_refused(est):
    rng = np.random.default_rng(6)
    segs = [np.sqrt(0.5) * _ar1(rng, 0.95, 150) + np.sqrt(0.5) * _ar1(rng, 0.5, 150) for _ in range(40)]
    g = g8_adequacy(_rec(segs), estimator=est, rng=1, n_rep=60)
    assert g.passed is False and g.margin < 0
    assert "refused" in g.note


def test_autocorrelated_noise_refused():
    rng = np.random.default_rng(7)
    segs = [np.sqrt(0.6) * _ar1(rng, 0.85, 150) + np.sqrt(0.4) * _ar1(rng, 0.3, 150) for _ in range(40)]
    g = g8_adequacy(_rec(segs), estimator="kalman", rng=1, n_rep=60)
    assert g.passed is False


def test_segments_shorter_than_four_relaxation_times_are_indeterminate():
    rng = np.random.default_rng(9)
    rec = _rec([_ar1(rng, 0.98, 40, 0.1) for _ in range(30)])
    g = g8_adequacy(rec, estimator="kalman", rng=1, n_rep=40)
    assert g.passed is None and "relaxation times" in g.note


def test_whittle_likelihood_ratio_detects_a_second_timescale():
    rng = np.random.default_rng(3)
    one = [_ar1(rng, 0.85, 150, 0.2) for _ in range(40)]
    two = [np.sqrt(0.5) * _ar1(rng, 0.95, 150) + np.sqrt(0.5) * _ar1(rng, 0.4, 150) for _ in range(40)]
    lr1 = whittle_lr_two_timescale(*padded_centred([one]))[0]
    lr2 = whittle_lr_two_timescale(*padded_centred([two]))[0]
    assert lr1 >= 0 and lr2 > lr1 + 10


def test_degenerate_series_is_indeterminate():
    rec = _rec([np.ones(50) for _ in range(10)])
    g = g8_adequacy(rec, estimator="ols", rng=1)
    assert g.passed is None and g.note


def test_g8_order_in_the_certificate():
    L3 = LEVEL_GATES["L3"]
    assert L3.index("G8") == L3.index("G2") + 1 and L3.index("G8") < L3.index("G1") < L3.index("G3")
    rng = np.random.default_rng(8)
    segs = [np.sqrt(0.5) * _ar1(rng, 0.95, 150) + np.sqrt(0.5) * _ar1(rng, 0.5, 150) for _ in range(40)]
    rec = _rec(segs)
    cert = certify(rec.X, rec.design, which=L3, configuration="g8_augmented", rng=1, estimator="kalman",
                   G3=dict(n_rep=6), G8=dict(n_rep=60))
    assert cert.levels["L3"].status == "refused"
    assert cert.gate("G8").passed is False
    assert cert.levels["L3"].binding_gate in ("G2", "G8")
