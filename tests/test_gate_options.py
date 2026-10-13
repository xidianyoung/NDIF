"""Optional forms of two gates: how G2 reads the rate at each decimation factor, and the white-share boundary rule of
the fitted-generator recovery check (G3).

By default, in every configuration, G2 refits the declared estimator on the decimated projected series and drops a
factor that returns no positive rate.  With ``rate="propagator"`` it reads the rate from the least-squares propagator
refitted on the decimated state, each eigenvalue by its modulus, for every estimator.  The annual discharge of the
Nile at Aswan, 1871-1970 (Cobb 1978), in four segments of 25 years, separates the two: the lag-1 factors of its
decimated segments are negative at factors 2, 3 and 4.

By default G3 leaves the rate channel indeterminate when the generator fit places the white share at its boundary;
``boundary_rule=False`` evaluates the rate channel from that fit.
"""
import numpy as np
import pytest

import ndif.estimators as est
from ndif import CONFIGURATIONS, G2_RATES, Design, Recording, certify, g2_aliasing, g3_recovery
from ndif.simulate import add_white, simulate_var1

NILE = np.array([
    1120, 1160, 963, 1210, 1160, 1160, 813, 1230, 1370, 1140, 995, 935, 1110, 994, 1020, 960, 1180, 799, 958, 1140,
    1100, 1210, 1150, 1250, 1260, 1220, 1030, 1100, 774, 840, 874, 694, 940, 833, 701, 916, 692, 1020, 1050, 969,
    831, 726, 456, 824, 702, 1120, 1100, 832, 764, 821, 768, 845, 864, 862, 698, 845, 744, 796, 1040, 759,
    781, 865, 845, 944, 984, 897, 822, 1010, 771, 676, 649, 846, 812, 742, 801, 1040, 860, 874, 848, 890,
    744, 749, 838, 1050, 918, 986, 797, 923, 975, 815, 1020, 906, 901, 1170, 912, 746, 919, 718, 714, 740], float)


def _nile():
    bounds = [(i, i + 25) for i in range(0, 100, 25)]
    return Recording(NILE[:, None], Design(dt=1.0, window_s=1.0, step_s=1.0, segments=bounds, name="nile"))


def _rec(segs):
    e = np.cumsum([0] + [len(s) for s in segs])
    return Recording(np.concatenate(segs, 0),
                     Design(dt=1.0, window_s=1.0, step_s=1.0, segments=[(e[i], e[i + 1]) for i in range(len(segs))]))


def _white(nu=0.4):
    return _rec(add_white(simulate_var1(np.array([[0.85]]), np.array([[1 - 0.85 ** 2]]), 40, 150, rng=5), nu, rng=6))


def _lag1(segs):
    """Pooled least-squares lag-1 factor of segments, each centred on its own mean."""
    num = den = 0.0
    for s in segs:
        s = s - s.mean()
        num += float(s[1:] @ s[:-1])
        den += float(s[:-1] @ s[:-1])
    return num / den


def test_g2_rate_default_and_choices():
    assert G2_RATES == ("estimator", "propagator")
    for conf in CONFIGURATIONS.values():                         # no configuration overrides the default
        assert "rate" not in conf["gate_kw"].get("G2", {})
    with pytest.raises(ValueError):
        g2_aliasing(_nile(), rate="lag1")


def test_estimator_rate_drops_a_factor_without_a_positive_rate():
    r = g2_aliasing(_nile(), rng=1)
    assert sorted(r.detail["per_factor"]) == [1]
    assert np.isnan(r.value) and r.passed is None and r.note == "too few usable decimation levels"


def test_propagator_rate_reads_a_negative_lag1_factor_by_its_modulus():
    rec = _nile()
    segs = [s[:, 0] for s in rec.segments()]
    r = g2_aliasing(rec, rate="propagator", rng=1)
    assert r.detail["rate"] == "propagator" and sorted(r.detail["per_factor"]) == [1, 2, 3, 4]
    for m in (1, 2, 3, 4):
        a = _lag1([s[::m] for s in segs])
        assert (a < 0) is (m > 1)
        assert r.detail["per_factor"][m] == pytest.approx(-np.log(abs(a)) / m, rel=1e-9)
    # the rate falls as the sampling interval grows: it is read off the sampling, so the check binds
    assert r.value == pytest.approx(1.0803, abs=1e-4) and r.passed is False


def test_propagator_rate_is_the_same_for_every_estimator():
    rec = _white()
    ols = g2_aliasing(rec, rate="propagator", estimator="ols", rng=0)
    kal = g2_aliasing(rec, rate="propagator", estimator="kalman", rng=0)
    assert kal.value == ols.value and kal.detail["ci"] == ols.detail["ci"]
    assert ols.value > 0.5 and ols.passed is False               # the white share inflates the rate at fine sampling
    # refitted with the declared estimator, the observation-aware fit removes that dependence
    assert abs(g2_aliasing(rec, estimator="kalman", rng=0).value) < 0.2


def test_certify_passes_the_g2_rate():
    rec = _nile()
    p = certify(rec.X, rec.design, which=["G0", "G1", "G2"], configuration="primary", rng=1)
    assert p.gate("G2").passed is None and p.gate("G2").note == "too few usable decimation levels"
    o = certify(rec.X, rec.design, which=["G0", "G1", "G2"], configuration="primary", rng=1,
                G2=dict(rate="propagator"))
    assert o.gate("G2").value == pytest.approx(1.0803, abs=1e-4) and o.gate("G2").passed is False
    assert o.levels["L3"].status == "refused" and o.levels["L3"].binding_gate == "G2"


def _kalman_at_boundary(real):
    def f(*a, **k):
        out = dict(real(*a, **k))
        out.update(sw2=0.0, nu=1.0 - 1e-15)      # a fit with no autocorrelated component
        return out
    return f


def test_g3_boundary_rule_of_the_fitted_generator(monkeypatch):
    rec = _rec(simulate_var1(np.diag([0.9, 0.8]), np.eye(2) * 0.3, 20, 150, rng=5))
    monkeypatch.setattr(est, "fit_ar1_kalman", _kalman_at_boundary(est.fit_ar1_kalman))
    on = g3_recovery(rec, n_rep=6, rng=2)
    assert on.detail["nu_at_boundary"] is True and on.passed is None and not np.isfinite(on.value)
    off = g3_recovery(rec, n_rep=6, rng=2, boundary_rule=False)
    assert off.detail["nu_at_boundary"] is False and np.isfinite(off.value) and off.passed in (True, False)
    # the variance channel keeps its random stream, so it does not depend on the setting
    assert on.detail["variance_recovery"] == off.detail["variance_recovery"]
