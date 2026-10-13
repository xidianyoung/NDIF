# -*- coding: utf-8 -*-
"""A user's own rate estimator must be certifiable by the same path as the built-ins.

The framework's claim is that identifiability belongs to the pair (recording,
estimator). If only the two estimators shipped in the box can be certified, that
claim is untestable by anyone else, so these tests pin the registration path:
the estimator is really called, it really reaches the gates, and the declared
model, which the model-adequacy check tests, is really honoured.
"""

from __future__ import annotations

import numpy as np
import pytest

import ndif
from ndif import Design, certify
from ndif.estimators import ESTIMATORS, fit_rate


A_TRUE = 0.85
LAM_TRUE = -np.log(A_TRUE)


def _ar1(n_seg=30, n=150, a=A_TRUE, seed=0, d=1):
    rng = np.random.default_rng(seed)
    segs, bounds, off = [], [], 0
    for _ in range(n_seg):
        x = np.zeros((n, d))
        for t in range(1, n):
            x[t] = a * x[t - 1] + rng.normal(size=d)
        segs.append(x)
        bounds.append((off, off + n))
        off += n
    return np.concatenate(segs, axis=0), bounds


def _lag1(segments, dt):
    s = np.concatenate([np.asarray(z).ravel() for z in segments])
    s = s - s.mean()
    a = float(np.dot(s[1:], s[:-1]) / np.dot(s[:-1], s[:-1]))
    return -np.log(a) / dt if 0.0 < a < 1.0 else float("nan")


@pytest.fixture(autouse=True)
def _clean_registry():
    """Restore the estimator registry after every test."""
    before = dict(ESTIMATORS)
    yield
    ESTIMATORS.clear()
    ESTIMATORS.update(before)


def test_registered_estimator_is_called_and_returns_a_rate():
    calls = []

    def counting(segments, dt):
        calls.append(len(segments))
        return _lag1(segments, dt)

    ndif.register_estimator("t_count", counting, observation_aware=False)
    X, bounds = _ar1(seed=1)
    segs = [X[a:b, 0] for a, b in bounds]
    lam = fit_rate(segs, dt=1.0, method="t_count")
    assert calls, "the registered estimator was never called"
    assert abs(lam - LAM_TRUE) < 0.05, (lam, LAM_TRUE)


def test_registration_declares_what_the_checks_read():
    """The declaration, not the function, decides which model the model-adequacy check tests."""
    from ndif.estimators import ESTIMATORS
    ndif.register_estimator("t_naive", _lag1, observation_aware=False, overwrite=True)
    ndif.register_estimator("t_aware", _lag1, observation_aware=True, overwrite=True)
    assert ESTIMATORS["t_naive"]["observation_aware"] is False and ESTIMATORS["t_naive"]["model"] == "ar1"
    assert ESTIMATORS["t_aware"]["observation_aware"] is True and ESTIMATORS["t_aware"]["model"] == "ar1_white"
    ndif.register_estimator("t_explicit", _lag1, observation_aware=True, model="ar1", overwrite=True)
    assert ESTIMATORS["t_explicit"]["model"] == "ar1"
    X, bounds = _ar1(seed=2)
    d = Design(dt=1.0, window_s=1.0, step_s=1.0, segments=bounds, name="synthetic")
    from ndif.gates import g8_adequacy
    from ndif.types import Recording
    rec = Recording(X, d)
    assert g8_adequacy(rec, estimator="t_naive", rng=0, n_rep=40).detail["model"] == "ar1"
    assert g8_adequacy(rec, estimator="t_aware", rng=0, n_rep=40).detail["model"] == "ar1_white"
    with pytest.raises(ValueError):
        ndif.register_estimator("t_bad", _lag1, observation_aware=True, model="ar2", overwrite=True)


def test_certify_runs_a_registered_estimator_end_to_end():
    ndif.register_estimator("t_e2e", _lag1, observation_aware=False,
                            label="test lag-1 rate")
    X, bounds = _ar1(seed=3)
    d = Design(dt=1.0, window_s=1.0, step_s=1.0, segments=bounds, name="synthetic")
    cert = certify(X, d, which=["G0", "G1", "G2"], estimator="t_e2e", rng=1)
    assert cert.gate("G2").status in {"cleared", "binding", "indeterminate"}
    # and it agrees with the built-in least-squares estimator, which is what it is
    ref = certify(X, d, which=["G0", "G1", "G2"], estimator="ols", rng=1)
    assert cert.gate("G2").status == ref.gate("G2").status


def test_registration_refuses_a_silent_redefinition_and_a_missing_declaration():
    with pytest.raises(ValueError):
        ndif.register_estimator("ols", _lag1, observation_aware=False)
    ndif.register_estimator("ols", _lag1, observation_aware=False, overwrite=True)
    with pytest.raises(TypeError):
        ndif.register_estimator("t_bad", _lag1, observation_aware="yes")
    with pytest.raises(TypeError):
        ndif.register_estimator("t_bad2", "not callable", observation_aware=False)


def test_unknown_estimator_names_the_way_to_register_one():
    with pytest.raises(ValueError, match="register_estimator"):
        fit_rate([np.zeros(10)], method="no_such_estimator")
