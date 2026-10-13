"""Recovery check (G3) at the boundary of the noise model.

When the maximum-likelihood fit puts the variance of the autocorrelated component at zero (noise fraction at
its upper boundary), the likelihood no longer depends on the persistence, so the fitted rate is undefined and
a recovery simulation from that fit cannot test an estimator. The rate channel must then be indeterminate, not
binding, and the variance channel must still be evaluated.
"""
import numpy as np

import ndif.estimators as est
from functools import partial

from ndif import Design, Recording
from ndif import g3_recovery as _g3_recovery
g3_recovery = partial(_g3_recovery, generator="corrected")      # the recovery check of the G8-augmented configuration
from ndif.gates import NU_BOUNDARY
from ndif.simulate import simulate_var1


def _rec(X_segments):
    X = np.concatenate(X_segments, axis=0)
    edges = np.cumsum([0] + [len(s) for s in X_segments])
    bounds = [(edges[i], edges[i + 1]) for i in range(len(X_segments))]
    return Recording(X, Design(dt=1.0, window_s=1.0, step_s=1.0, segments=bounds))


def _segs():
    return simulate_var1(np.diag([0.9, 0.8, 0.7]), np.eye(3) * 0.3, 40, 150, rng=5)


def _at_boundary(real):
    def f(*a, **k):
        out = real(*a, **k)
        for g in out:
            g.update(sw2=0.0, nu=1.0 - 1e-15)      # what the fit returns for the MEG trial clock
        return out
    return f


def test_fit_at_the_noise_boundary_makes_the_rate_channel_indeterminate(monkeypatch):
    monkeypatch.setattr(est, "fit_ar1_kalman_batch", _at_boundary(est.fit_ar1_kalman_batch))
    r = g3_recovery(_rec(_segs()), generator="corrected", n_rep=6)
    assert r.detail["nu_fit"] >= NU_BOUNDARY
    assert r.detail["nu_at_boundary"] is True
    assert r.passed is None                         # indeterminate, never binding
    assert not np.isfinite(r.detail["rate_recovery"])
    assert np.isfinite(r.detail["variance_recovery"])   # the variance channel is still evaluated


def test_variance_channel_is_unchanged_by_the_boundary_rule(monkeypatch):
    clean = g3_recovery(_rec(_segs()), n_rep=6, include_white=False, rng=3)
    monkeypatch.setattr(est, "fit_ar1_kalman_batch", _at_boundary(est.fit_ar1_kalman_batch))
    skipped = g3_recovery(_rec(_segs()), n_rep=6, include_white=False, rng=3)
    # include_white=False never reaches the boundary branch
    assert skipped.detail["nu_at_boundary"] is False
    assert np.isfinite(clean.detail["variance_recovery"]) and np.isfinite(skipped.detail["variance_recovery"])


def test_autocorrelated_series_is_not_at_the_boundary():
    r = g3_recovery(_rec(_segs()), generator="corrected", n_rep=6)
    assert r.detail["nu_at_boundary"] is False
    assert r.passed in (True, False, None)
    assert r.detail["rate"]["decision"] in ("pass", "fail", "ambiguous")
