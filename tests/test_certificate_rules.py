"""Rules of the complete certificate.

* A level is admitted only when every check that governs it was run and passed: a reduced set of checks cannot
  admit, and the recommendation uses the complete set.
* Diagnostics (the mode and surrogate checks) govern no level: their failure changes no classification.
* The certificate does not depend on the unit in which time is expressed.
* The simulations of the checks use the segment lengths of the recording.
"""
import numpy as np

import ndif.estimators as est
import ndif.gates as G
from ndif import Design, certify, recommend_estimator
from ndif.configurations import CONFIGURATIONS
from ndif.simulate import simulate_var1
from ndif.types import ADMITTED, INDETERMINATE, GateResult

AUG = CONFIGURATIONS["g8_augmented"]
LEVEL_GATES = AUG["level_gates"]
KW = dict(configuration="g8_augmented", rng=1, G3=dict(n_rep=8), G7=dict(n_rep=40), G4=dict(n_rep=12),
          G8=dict(n_rep=40))


def _bounds(segs):
    e = np.cumsum([0] + [len(s) for s in segs])
    return [(int(e[i]), int(e[i + 1])) for i in range(len(segs))]


def _clean(seed=12, a=0.8, n_seg=40, n=400):
    # a typical single-timescale recording with a fixed seed; at the nominal 5% level the model-adequacy check
    # refuses some single-timescale recordings by chance
    segs = simulate_var1(np.array([[a]]), np.array([[1 - a * a]]), n_seg, n, rng=seed)
    return np.concatenate(segs, 0), _bounds(segs)


def test_diagnostics_govern_no_level():
    for lv, gates in LEVEL_GATES.items():
        assert "G4" not in gates and "G7" not in gates, lv
    assert LEVEL_GATES["L3"] == ["G0", "G2", "G8", "G1", "G3"]
    assert set(G.DIAGNOSTICS) == {"G4", "G7"}
    assert "G4" not in AUG["gates"] and "G7" not in AUG["gates"]
    assert set(AUG["diagnostics"]) == {"G4", "G7"}


def test_a_reduced_set_of_checks_cannot_admit_a_level():
    X, b = _clean()
    d = Design(dt=1.0, window_s=1.0, step_s=1.0, segments=b)
    full = certify(X, d, which=LEVEL_GATES["L3"], **KW)
    assert full.levels["L3"].status == ADMITTED
    assert full.levels["L3"].not_run == [] and full.provenance["complete"]["L3"] is True
    reduced = certify(X, d, which=[g for g in LEVEL_GATES["L3"] if g != "G2"], **KW)
    assert reduced.levels["L3"].status == INDETERMINATE          # not run is not passed
    assert reduced.levels["L3"].not_run == ["G2"] and "G2" in reduced.levels["L3"].grounds
    assert "G2" in reduced.provenance["gates_not_run"] and reduced.provenance["complete"]["L3"] is False
    assert "G2" in str(reduced)                                   # the text certificate names what was not run


def test_a_failing_diagnostic_changes_nothing_and_a_failing_check_prevents_recommendation(monkeypatch):
    X, b = _clean()
    d = Design(dt=1.0, window_s=1.0, step_s=1.0, segments=b)
    ok = recommend_estimator(X, d, candidates=["ols"], **KW)
    assert ok["recommended"] == "ols"
    assert ok["candidates"][0]["complete"] is True

    def g4_deletion(rec, **kw):                                   # a diagnostic that fails
        return GateResult("G4", "Does the readout change when the dynamics are replaced?",
                          "agreement with surrogate dynamics", 1.0, "agreement <= 0.95", False, -0.05, ())

    monkeypatch.setattr(G, "GATES", [g4_deletion if g.__name__ == "g4_deletion" else g for g in G.GATES])
    diag = certify(X, d, which=list(LEVEL_GATES["L3"]) + ["G4"], **KW)
    assert diag.gate("G4").passed is False and diag.levels["L3"].status == ADMITTED
    assert "diagnostic" in str(diag)
    monkeypatch.undo()

    def g2_aliasing(rec, **kw):                                   # a deciding check that fails
        return GateResult("G2", "Is the rate an estimate or the sampling rate?", "theta", 0.9,
                          "|theta| <= 0.5 and CI excludes 1", False, -0.4, ("L3",))

    monkeypatch.setattr(G, "GATES", [g2_aliasing if g.__name__ == "g2_aliasing" else g for g in G.GATES])
    bad = recommend_estimator(X, d, candidates=["ols"], **KW)
    assert bad["recommended"] is None and bad["candidates"][0]["binding_gate"] == "G2"
    monkeypatch.undo()
    # a recommendation asked for on a reduced set of checks is not given either
    red = recommend_estimator(X, d, candidates=["ols"], which=[g for g in LEVEL_GATES["L3"] if g != "G2"], **KW)
    assert red["recommended"] is None and red["candidates"][0]["checks_not_run"] == ["G2"]
    assert red["candidates"][0]["complete"] is False


def test_certificate_does_not_depend_on_the_unit_of_time():
    """The same recording with time in seconds, milliseconds and minutes: one certificate, diagnostics included."""
    A = np.diag([0.9, 0.6])
    segs = simulate_var1(A, np.eye(2) - A @ A.T, 30, 200, rng=5)
    X, b = np.concatenate(segs, 0), _bounds(segs)
    which = list(AUG["gates"]) + ["G4", "G7"]
    kw = dict(configuration="g8_augmented", rng=1, which=which, G3=dict(n_rep=6), G4=dict(n_rep=12), G6=dict(n_rep=10), G7=dict(n_rep=40),
              G8=dict(n_rep=40))
    ref = certify(X, Design(dt=2.0, window_s=2.0, step_s=2.0, segments=b), **kw)
    for dt, unit in ((2000.0, "ms"), (2.0 / 60.0, "min")):
        other = certify(X, Design(dt=dt, window_s=dt, step_s=dt, segments=b, time_unit=unit), **kw)
        for lv in ("L1", "L2", "L3"):
            assert ref.levels[lv].status == other.levels[lv].status, (unit, lv)
            assert ref.levels[lv].binding_gate == other.levels[lv].binding_gate, (unit, lv)
        for g in ref.gates:
            h = other.gate(g.gate)
            assert g.passed == h.passed, (unit, g.gate)
            if g.gate in ("G4", "G7"):
                for u, v in ((g.value, h.value), (g.margin, h.margin)):
                    if np.isfinite(u) or np.isfinite(v):
                        assert abs(u - v) <= 1e-6 * max(1.0, abs(u)), (unit, g.gate, u, v)
        assert ref.gate("G7").detail["n_identified"] == other.gate("G7").detail["n_identified"], unit
        assert np.allclose(ref.gate("G7").detail["weights"], other.gate("G7").detail["weights"]), unit
    d = ref.gate("G7").detail
    assert d["reference_rate"] == 1.0
    assert np.allclose(d["weights"], np.abs(np.log(np.asarray(d["rates"]) / d["reference_rate"])))


def test_reference_rate_per_sampling_interval_is_a_conversion():
    """ref_unit="interval" is a rate per sampling interval: with dt = 2 s, 1 per interval = 0.5 per second."""
    A = np.diag([0.9, 0.6])
    segs = simulate_var1(A, np.eye(2) - A @ A.T, 30, 200, rng=6)
    X, b = np.concatenate(segs, 0), _bounds(segs)
    d = Design(dt=2.0, window_s=2.0, step_s=2.0, segments=b)
    a = certify(X, d, which=["G7", "G4"], rng=1, G4=dict(n_rep=12, ref_rate=1.0, ref_unit="interval"),
                G7=dict(n_rep=40, ref_rate=1.0, ref_unit="interval"))
    c = certify(X, d, which=["G7", "G4"], rng=1, G4=dict(n_rep=12, ref_rate=0.5), G7=dict(n_rep=40, ref_rate=0.5))
    for g in ("G7", "G4"):
        assert a.gate(g).passed == c.gate(g).passed and abs(a.gate(g).value - c.gate(g).value) < 1e-9, g
    assert a.gate("G7").detail["reference_rate"] == c.gate("G7").detail["reference_rate"] == 0.5


def test_simulations_use_the_segment_lengths_of_the_recording(monkeypatch):
    rng = np.random.default_rng(2)
    lens = [12] * 10 + [60] * 10 + [150] * 10
    segs = []
    for n in lens:
        x = np.zeros(n)
        x[0] = rng.normal()
        for t in range(1, n):
            x[t] = 0.8 * x[t - 1] + rng.normal() * 0.6
        segs.append(x[:, None])
    X, b = np.concatenate(segs, 0), _bounds(segs)
    seen = []
    real = est.simulate_eq6_padded

    def spy(a, nu, lens_, n, rng_):
        seen.append(sorted(int(L) for L in lens_))
        return real(a, nu, lens_, n, rng_)

    monkeypatch.setattr(est, "simulate_eq6_padded", spy)
    certify(X, Design(dt=1.0, window_s=1.0, step_s=1.0, segments=b), which=["G3"], configuration="g8_augmented",
            rng=1, G3=dict(n_rep=3))
    assert seen and all(s == sorted(lens) for s in seen)   # every simulation has the lengths of the recording
