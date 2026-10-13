"""Gate behaviour and certificate scoring, against injected ground truth.

The point of these tests is not that the gates run.  It is that each gate
*binds on the thing it is supposed to detect and clears otherwise*, and that
the certificate discriminates rather than always answering no.
"""

import json

import numpy as np
import pytest

from ndif import (ADMITTED, INDETERMINATE, REFUSED, Design, Recording, certify,
                  g0_overlap, g1_instrument, g2_aliasing, g3_recovery,
                  g4_deletion, g5_floor, g6_counter, g7_modes)
from ndif.simulate import add_white, simulate_var1

A = np.diag([0.90, 0.80, 0.65])
Q = np.eye(3) * 0.3
FAST = dict(G3=dict(n_rep=8), G4=dict(n_rep=12), G6=dict(n_rep=15),
            G7=dict(n_rep=40), G8=dict(n_rep=40))


def make(nu=0.0, n_seg=60, n_blk=300, seed=7, window=1.0, step=1.0, labels=None):
    segs = simulate_var1(A, Q, n_seg, n_blk, rng=seed)
    if nu > 0:
        segs = add_white(segs, nu, rng=seed + 1)
    X = np.concatenate(segs, axis=0)
    edges = np.cumsum([0] + [len(s) for s in segs])
    bounds = [(edges[i], edges[i + 1]) for i in range(len(segs))]
    return Recording(X, Design(dt=1.0, window_s=window, step_s=step,
                               segments=bounds, labels=labels))


# --------------------------------------------------------------------- gates
def test_g0_detects_overlap():
    assert g0_overlap(make(window=1.0, step=1.0)).passed is True
    assert g0_overlap(make(window=2.0, step=1.0)).passed is False
    r = g0_overlap(Recording(np.zeros((100, 2)), Design(dt=1.0)))
    assert r.passed is None and r.status == "indeterminate"


@pytest.mark.parametrize("nu", [0.0, 0.25, 0.5])
def test_g1_reports_the_white_share_and_the_rate_interval(nu):
    r = g1_instrument(make(nu=nu), rule="profile")
    assert abs(r.detail["two_lag"]["nu"] - nu) < 0.05          # the two-lag moment estimate is also reported
    t = g1_instrument(make(nu=nu))                               # primary configuration: the two-lag rule
    assert abs(t.value - nu) < 0.05 and t.passed is (nu <= 0.20)
    lo, hi = r.detail["nu_set"]
    assert lo - 0.05 <= nu <= hi + 0.05                          # the profile set covers the injected fraction
    assert r.passed is True and r.value <= 0.20                  # long recordings: the rate is determined


def test_g2_theta_is_zero_when_clean_and_rises_with_white_noise():
    clean = g2_aliasing(make(nu=0.0))
    noisy = g2_aliasing(make(nu=0.5))
    assert abs(clean.value) < 0.2 and clean.passed is True
    assert noisy.value > clean.value + 0.2


def test_g3_recovers_known_truth_and_separates_the_two_channels():
    """The asymmetry that separates L2 from L3 (least squares, no noise model)."""
    clean = g3_recovery(make(nu=0.0), n_rep=10)
    noisy = g3_recovery(make(nu=0.5), n_rep=10)
    assert abs(clean.value - 1.0) < 0.2 and clean.passed is True
    assert noisy.passed is False and noisy.detail["bias_log"] > 0.2      # noise biases the rate channel
    for r in (clean, noisy):
        assert abs(r.detail["variance_recovery"] - 1.0) < 0.1           # variance is not affected
    assert np.isfinite(clean.detail["detection_margin_fold"])


def test_g4_readout_moves_for_a_dynamics_sensitive_readout():
    r = g4_deletion(make(), n_rep=12)
    assert r.passed is True
    assert r.detail["agreement_isotropic"] <= 1.0


def test_g4_density_functional_readout_survives_deletion():
    """A readout that is a functional of the density cannot tell the
    surrogate dynamics apart -- which is the whole point of G4."""
    def density_readout(sim, dt):
        return float(np.log(np.trace(np.cov(np.concatenate(sim, 0), rowvar=False))))
    r = g4_deletion(make(), readout=density_readout, n_rep=16)
    assert r.detail["agreement_isotropic"] > 0.5
    assert r.passed is False


def test_g5_indeterminate_without_labels_and_evaluable_with_them():
    assert g5_floor(make()).passed is None
    n_seg, n_blk = 40, 200
    lab = np.concatenate([np.full(n_blk, i % 2) for i in range(n_seg)])
    rec = make(n_seg=n_seg, n_blk=n_blk, labels=lab)
    rec.X = rec.X + np.outer(lab, np.array([1.2, 0.0, 0.0]))
    r = g5_floor(rec, null="permutation", n_perm=200)
    assert r.passed is True and r.value > r.detail["floor"] and r.detail["p_value"] <= 0.05
    s = g5_floor(rec, n_rep=50)                                   # primary configuration: simulated zero-truth floor
    assert s.passed is True and s.value > s.detail["floor"]


def test_g6_error_rates_under_known_truth():
    r = g6_counter(make(), n_rep=25)
    assert r.detail["tpr"] >= 0.8
    assert r.detail["fpr"] <= 0.2


def test_g7_share_is_a_fraction():
    r = g7_modes(make(), n_rep=40)
    assert 0.0 <= r.value <= 1.0
    assert r.detail["n_modes"] == 3


# --------------------------------------------------------------- certificate
def test_certificate_discriminates():
    """Same generator, one property changed: admit, refuse, indeterminate."""
    clean = certify(make(nu=0.0).X, make(nu=0.0).design, rng=1, **FAST)
    noisy = certify(make(nu=0.5).X, make(nu=0.5).design, rng=1, **FAST)
    lap = make(window=2.0, step=1.0)
    overlap = certify(lap.X, lap.design, rng=1, **FAST)

    assert clean.levels["L3"].status == ADMITTED
    assert noisy.levels["L3"].status == REFUSED
    assert noisy.levels["L3"].binding_gate in ("G2", "G8", "G1", "G3")
    assert overlap.levels["L3"].binding_gate == "G0"
    assert overlap.levels["L2"].status == REFUSED
    # L2 survives white noise where L3 does not
    assert noisy.levels["L2"].status == ADMITTED


def test_refusal_carries_a_detection_margin():
    rec = make(nu=0.5)
    cert = certify(rec.X, rec.design, rng=1, **FAST)
    assert cert.levels["L3"].status == REFUSED
    assert cert.levels["L3"].detection_margin
    assert "would have been detected" in cert.levels["L3"].detection_margin


def test_unevaluable_gate_is_indeterminate_not_a_pass():
    rec = Recording(np.random.default_rng(0).normal(size=(400, 2)), Design(dt=1.0))
    cert = certify(rec.X, rec.design, which=["G0"], rng=1)
    assert cert.levels["L3"].status == INDETERMINATE
    assert cert.levels["L3"].status != ADMITTED


def test_certificate_serialises():
    rec = make()
    cert = certify(rec.X, rec.design, which=["G0", "G1", "G2"], rng=1)
    d = json.loads(cert.to_json())
    assert set(d) == {"levels", "gates", "design", "provenance"}
    assert set(d["levels"]) == {"L1", "L2", "L3"}
    for g in d["gates"]:
        assert g["status"] in {"cleared", "binding", "indeterminate"}
    assert "ndif_version" in d["provenance"]


def test_render_is_non_empty_and_names_every_level():
    rec = make()
    cert = certify(rec.X, rec.design, which=["G0", "G1"], rng=1)
    s = str(cert)
    for k in ("L1", "L2", "L3", "identifiability certificate"):
        assert k in s


# ------------------------------------------------- the d=1 surrogate
def test_no_dynamics_surrogate_actually_deletes_the_dynamics():
    """The G4/G7 surrogate must not be built from the fit's own rate.

    A surrogate propagator set to the geometric mean of the fitted eigenvalues
    would, in one dimension, be the fitted propagator exactly, so both gates
    would compare the process against itself: G7's null would be drawn from the
    very spectrum it tests, and no mode could be distinguished from "no
    dynamics".  The surrogate therefore uses a fixed reference rate, and a
    surrogate that still coincides with the fit is reported as indeterminate
    rather than given a verdict.
    """
    from ndif.simulate import (is_degenerate_surrogate, matched_isotropic,
                               random_hurwitz_propagator)
    for a in (0.5, 0.85, 0.97):
        A, Q = np.array([[a]]), np.array([[1 - a * a]])
        A_iso, _ = matched_isotropic(A, Q, dt=1.0)
        assert not np.isclose(A_iso[0, 0], a, atol=1e-6), "surrogate kept the rate"
        assert not is_degenerate_surrogate(A, A_iso)
        A_rnd, _ = random_hurwitz_propagator(A, Q, rng=0)
        assert A_rnd is not None


def test_g7_is_indeterminate_when_the_surrogate_cannot_delete_anything():
    """A surrogate that coincides with the fit yields indeterminate, not a verdict."""
    from ndif.simulate import matched_isotropic
    # dt chosen so exp(-1*dt) equals the fitted propagator: nothing is deleted
    a = float(np.exp(-1.0))
    segs = simulate_var1(np.array([[a]]), np.array([[1 - a * a]]), 30, 200, rng=3)
    X = np.concatenate(segs, 0)
    edges = np.cumsum([0] + [len(s) for s in segs])
    rec = Recording(X, Design(dt=1.0, window_s=1.0, step_s=1.0,
                              segments=[(edges[i], edges[i + 1])
                                        for i in range(len(segs))]))
    A_iso, _ = matched_isotropic(np.array([[a]]), np.array([[1 - a * a]]), dt=1.0)
    assert np.isclose(A_iso[0, 0], a, atol=1e-9)      # the degenerate case, by construction
    r = g7_modes(rec, n_rep=30)
    assert r.passed is None and r.status == "indeterminate"


def test_one_dimensional_recording_can_still_be_admitted():
    """A one-dimensional recording with a well-identified rate must be admissible."""
    a = 0.85
    segs = simulate_var1(np.array([[a]]), np.array([[1 - a * a]]), 40, 400, rng=12)   # a typical recording
    X = np.concatenate(segs, 0)
    edges = np.cumsum([0] + [len(s) for s in segs])
    rec = Recording(X, Design(dt=1.0, window_s=1.0, step_s=1.0,
                              segments=[(edges[i], edges[i + 1])
                                        for i in range(len(segs))]))
    cert = certify(rec.X, rec.design, which=["G0", "G1", "G2", "G8", "G3", "G7", "G4"],
                   configuration="g8_augmented", rng=1, G3=dict(n_rep=8), G7=dict(n_rep=40), G4=dict(n_rep=12))
    assert cert.levels["L3"].status == ADMITTED
