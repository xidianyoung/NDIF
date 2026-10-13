"""``certify`` -- the entry point.  One recording in, one certificate out."""

from __future__ import annotations

import platform
import sys

import numpy as np

from .configurations import CONFIGURATIONS, DEFAULT_CONFIGURATION, resolve_configuration
from .gates import run_gates
from .types import (ADMITTED, INDETERMINATE, LEVELS, REFUSED, Certificate,
                    Design, GateResult, LevelVerdict, Recording)

__all__ = ["certify", "recommend_estimator", "LEVEL_GATES"]

#: Which gates govern which level in the primary configuration, in the order they are consulted.  The first
#: gate that fails is the first failed (binding) gate for that level.  The other configurations are defined in
#: :mod:`ndif.configurations`.
LEVEL_GATES = CONFIGURATIONS[DEFAULT_CONFIGURATION]["level_gates"]


def _g3_variance_view(g3):
    """G3's variance channel as a pseudo-gate, since it governs L2 not L3."""
    if g3 is None:
        return None
    d = g3.detail or {}
    v = d.get("variance_recovery", float("nan"))
    ok = d.get("variance_ok", None)
    if v is None or not np.isfinite(v):
        ok = None
    if "variance_margin" in d:
        # corrected generator (G8-augmented configuration): slope and spread, each beyond Monte Carlo error
        tol, spr = d.get("tol", 0.20), d.get("spread_tol", 0.20)
        crit = f"|slope - 1| <= {tol} and spread <= {spr} log units, each beyond Monte Carlo error"
        margin = float(d.get("variance_margin", float("nan")))
        detail = dict(source="G3 variance channel", **(d.get("variance") or {}))
    else:
        # fitted generator (primary and benchmark configurations): slope within 0.20 of 1
        crit = g3.criterion
        margin = float(0.20 - abs(v - 1.0)) if (v is not None and np.isfinite(v)) else float("nan")
        detail = dict(source="G3 variance channel")
    return GateResult("G3v", "Does a known change in variability come back?",
                      "recovered fraction (variance channel)", float(v) if v is not None else float("nan"),
                      crit, ok, margin, ("L2",), cost_s=0.0, detail=detail)


def _score_level(level, gates_by_name, level_gates):
    """Admitted / refused / indeterminate, with the gate that decides it."""
    order = level_gates[level]
    evaluated = [gates_by_name[n] for n in order if gates_by_name.get(n) is not None]
    # A check that was not run is not a check that passed: it is recorded, and the level cannot be admitted.
    not_run = [("G3" if n == "G3v" else n) for n in order if gates_by_name.get(n) is None]
    if not evaluated:
        return LevelVerdict(level, INDETERMINATE, None, None,
                            grounds="no gate governing this level could be evaluated", not_run=not_run)
    binding = next((g for g in evaluated if g.passed is False), None)
    if binding is not None:
        return LevelVerdict(level, REFUSED, binding.gate, float(binding.margin),
                            grounds=f"{binding.gate} binds: {binding.statistic} = "
                                    f"{binding.value:.4g}, criterion {binding.criterion}", not_run=not_run)
    unknown = [g for g in evaluated if g.passed is None]
    cleared = [g for g in evaluated if g.passed is True]
    if unknown:
        return LevelVerdict(level, INDETERMINATE, unknown[0].gate, None,
                            grounds="; ".join(f"{g.gate} could not be evaluated"
                                              + (f" ({g.note})" if g.note else "")
                                              for g in unknown)
                                    + (f"; not run: {', '.join(not_run)}" if not_run else ""),
                            not_run=not_run)
    if not_run:
        return LevelVerdict(level, INDETERMINATE, not_run[0], None,
                            grounds=f"not run: {', '.join(not_run)}; the checks that were run passed, but a "
                                    f"level is admitted only when every check that governs it has been run "
                                    f"and passed (reduced set of checks)",
                            not_run=not_run)
    tightest = min(cleared, key=lambda g: (g.margin if np.isfinite(g.margin) else np.inf))
    return LevelVerdict(level, ADMITTED, tightest.gate,
                        float(tightest.margin) if np.isfinite(tightest.margin) else None,
                        grounds=f"all governing gates cleared; tightest is "
                                f"{tightest.gate} at margin {tightest.margin:.4g}")


def _with_configuration_defaults(cfg, kw):
    """Per-gate keyword dicts with the configuration's gate settings unless the caller overrides them."""
    kw = dict(kw)
    for name, defaults in cfg["gate_kw"].items():
        merged = dict(defaults)
        user = kw.get(name)
        if isinstance(user, dict):
            merged.update(user)
        kw[name] = merged
    return kw


def certify(X, design=None, which=None, configuration=DEFAULT_CONFIGURATION, verbose=False, **kw) -> Certificate:
    """Certify what a recording can be read for.

    Parameters
    ----------
    X : (n_t, d) array_like, list of segments, or :class:`Recording`
        The state series.
    design : :class:`Design`, optional
        Sampling interval, window/step, labels, segments, axis.
    which : sequence of str, optional
        Subset of gate names to run, e.g. ``["G1", "G2"]``.  By default the
        gates of the configuration are run.  Levels governed by gates that
        were not run come back *indeterminate*.
    configuration : str or dict
        One of :data:`ndif.configurations.CONFIGURATIONS` (``"primary"``,
        ``"benchmark"`` or ``"g8_augmented"``): the gates run by default, the
        governing gates of each level and their order, the forms of G1, G3
        and G5 and the simulation geometry of the gates.
    **kw
        Per-gate keyword overrides, e.g. ``G3=dict(n_rep=64)``, plus a global
        ``rng`` and ``estimator``.

    Returns
    -------
    :class:`ndif.types.Certificate`
        With ``levels`` (admitted / refused / indeterminate per level, each
        carrying its binding gate and margin), ``gates`` (the full trace), and
        ``provenance``.

    Examples
    --------
    >>> import numpy as np
    >>> from ndif import certify, Design
    >>> rng = np.random.default_rng(0)
    >>> x = np.zeros((4000, 1))
    >>> for t in range(1, 4000):
    ...     x[t] = 0.9 * x[t-1] + rng.normal(scale=0.4)
    >>> cert = certify(x, Design(dt=1.0, window_s=1.0, step_s=1.0))
    >>> cert.levels["L3"].status in {"admitted", "refused", "indeterminate"}
    True
    """
    cfg = resolve_configuration(configuration)
    if isinstance(X, Recording):
        rec = X
        if design is not None:
            rec.design = design
    else:
        rec = Recording(np.asarray(X, float) if not isinstance(X, (list, tuple))
                        else np.concatenate([np.atleast_2d(s) for s in X], axis=0),
                        design if design is not None else Design())
        if isinstance(X, (list, tuple)) and rec.design.segments is None:
            lens = [len(np.atleast_2d(s)) for s in X]
            edges = np.cumsum([0] + lens)
            rec.design.segments = [(edges[i], edges[i + 1]) for i in range(len(lens))]

    if which is None:
        which = list(cfg["gates"])
    gates = run_gates(rec, which=which, verbose=verbose, **_with_configuration_defaults(cfg, kw))
    by_name = {g.gate: g for g in gates}
    by_name["G3v"] = _g3_variance_view(by_name.get("G3"))

    levels = {k: _score_level(k, by_name, cfg["level_gates"]) for k in ("L1", "L2", "L3")}

    # detection margin: what size of effect this recording could have shown
    g3 = by_name.get("G3")
    if g3 is not None and g3.detail:
        fold = g3.detail.get("detection_margin_fold")
        if fold is not None and np.isfinite(fold):
            txt = (f"a change in relaxation rate of at least x{fold:.2f} "
                   f"would have been detected at 80% power")
            levels["L3"].detection_margin = txt
        elif fold is not None and np.isinf(fold):
            levels["L3"].detection_margin = (
                "no change in relaxation rate of any size would have been "
                "detected through this chain: the recovered fraction is "
                "indistinguishable from zero")
        vfold = g3.detail.get("detection_margin_fold_variance")
        if vfold is not None and np.isfinite(vfold):
            levels["L2"].detection_margin = (
                f"a change in state variance of at least x{vfold:.2f} "
                f"would have been detected at 80% power")
    g5 = by_name.get("G5")
    if g5 is not None and g5.detail.get("floor") is not None:
        levels["L1"].detection_margin = (
            f"separations above {g5.detail['floor']:.3f} clear the zero-truth floor "
            f"at this geometry")

    run = {g.gate for g in gates}
    prov = dict(
        ndif_version=_version(),
        configuration=cfg["name"],
        python=sys.version.split()[0],
        platform=platform.platform(),
        numpy=np.__version__,
        n_samples=int(rec.n_t), d=int(rec.d),
        n_segments=len(rec.segments()),
        total_cost_s=float(sum(g.cost_s for g in gates)),
        gates_run=[g.gate for g in gates],
        gates_not_run=[n for n in ("G0", "G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8") if n not in run],
        diagnostics=[n for n in cfg.get("diagnostics", ()) if n in run],
        level_gates={k: list(v) for k, v in cfg["level_gates"].items()},
        complete={k: not v.not_run for k, v in levels.items()},
        estimator=str(kw.get("estimator", "ols")),
    )
    return Certificate(levels=levels, gates=gates, design=rec.design.summary(),
                       provenance=prov)


def recommend_estimator(X, design=None, candidates=None, which=None, configuration=DEFAULT_CONFIGURATION, **kw):
    """Which estimator, if any, this recording supports for a level-3 claim.

    Certifies the recording once per candidate estimator (by default the two
    estimators of the framework, ``"ols"`` and ``"kalman"``) and returns the
    cheapest one whose level-3 verdict is *admitted*, together with the full
    comparison.  By default every candidate is certified with every check that
    governs level 3 in the configuration, the complete set for the level the
    recommendation decides.  ``which`` restricts the checks; a level-3 check
    that is not run is recorded for each candidate (``checks_not_run``) and the
    level is then not admitted, so a reduced run never yields a recommendation.
    This is the operational output of the framework: not "your recording is
    inadequate" but "this estimator cannot recover a rate here and that one
    can, at this cost".  A cost-based ranking depends on the machine and does
    not rank scientific accuracy.

    Returns
    -------
    dict
        ``recommended`` -- the estimator name, or ``None`` if none is admitted;
        ``reason`` -- why, in one sentence;
        ``candidates`` -- one entry per estimator with its verdict, binding
        gate, recovered fraction, detection margin and cost.

    Examples
    --------
    >>> rec = recommend_estimator(X, Design(dt=1.0, window_s=1.0, step_s=1.0))
    >>> rec["recommended"]
    'kalman'
    """
    from .estimators import ESTIMATORS
    cfg = resolve_configuration(configuration)
    if candidates is None:
        candidates = ["ols", "kalman"]
    if which is None:
        which = [g for g in cfg["level_gates"]["L3"] if g != "G3v"]

    out = []
    for est in candidates:
        gkw = dict(kw)
        gkw["estimator"] = est          # forwarded to every gate by run_gates
        cert = certify(X, design, which=which, configuration=cfg, **gkw)
        lv = cert.levels["L3"]
        g = cert.gate("G3")
        out.append(dict(
            estimator=est, label=ESTIMATORS.get(est, {}).get("label", est),
            verdict=lv.status, binding_gate=lv.binding_gate,
            recovered_fraction=(float(g.value) if g is not None and g.value is not None and
                                np.isfinite(g.value) else None),
            detection_margin=lv.detection_margin,
            cost_s=float(cert.provenance["total_cost_s"]),
            checks_run=list(cert.provenance["gates_run"]),
            checks_not_run=list(lv.not_run),
            complete=bool(cert.provenance["complete"]["L3"]),
            certificate=cert))

    admitted = [c for c in out if c["verdict"] == ADMITTED]
    if admitted:
        best = min(admitted, key=lambda c: c["cost_s"])
        reason = (f"{best['estimator']} ({best['label']}) is admitted at level 3, "
                  f"recovering {best['recovered_fraction']:.2f} of an injected truth; "
                  + ("no other candidate is admitted."
                     if len(admitted) == 1 else
                     f"{len(admitted)} candidates are admitted and this is the cheapest."))
        return dict(recommended=best["estimator"], reason=reason, candidates=out)

    gates = {c["estimator"]: c["binding_gate"] for c in out}
    reduced = sorted({g for c in out for g in c["checks_not_run"]})
    return dict(recommended=None,
                reason=("no candidate estimator is admitted at level 3 on this "
                        "recording; binding gates were " +
                        ", ".join(f"{k}:{v}" for k, v in gates.items()) +
                        (f" (not run: {', '.join(reduced)}; a reduced set of checks cannot admit)"
                         if reduced else "") +
                        ". Report the level-3 quantity as a bound with its "
                        "detection margin, not as an estimate."),
                candidates=out)


def _version():
    try:
        from . import __version__
        return __version__
    except Exception:                                            # pragma: no cover
        return "unknown"
