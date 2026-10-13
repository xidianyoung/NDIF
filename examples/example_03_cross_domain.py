"""The gates carry no neural assumption: published non-neural series (Fig. 6).

Every gate takes a state series and the design that produced it.  Nothing in
the sequence refers to brains, so if the framework is what it claims to be it
must run unchanged on any field that reads a stationary observation for a
dynamical quantity -- and must reach *different* verdicts on series whose
dynamics differ.

The series are real, published, externally curated measurements, not
simulations:

==============  ====================================================  ======
series          domain and source                                      n
==============  ====================================================  ======
nile            annual Nile river discharge at Aswan, 1871-1970;       100
                the canonical local-level (AR(1) + white) dataset
                (Cobb 1978; Durbin & Koopman 2012)
sunspots        annual sunspot number, 1700-2008, Solar Influences     309
                Data Analysis Center
macro_unemp     quarterly US unemployment rate, 1959-2009,             203
                Federal Reserve / BEA
co2             weekly atmospheric CO2, Mauna Loa (Keeling et al.)     2284
==============  ====================================================  ======

What to expect, and why it is the point:

* **nile** obeys the observation model the gates assume almost exactly -- it is
  the textbook example of it -- so the white share is well identified and the
  rate should be certifiable with an observation-aware estimator.
* **sunspots** is quasi-periodic with an 11-year cycle.  A single relaxation
  timescale is the wrong model, and the two-lag solution should leave the
  admissible range, which G1 must report as *indeterminate* rather than as a
  number.
* **macro_unemp** is a persistent but genuinely stochastic series, the closest
  non-neural analogue of the recordings the framework was built on.
* **co2** carries a deterministic trend and an annual cycle.  Its lag-1
  autocorrelation is near 1 for reasons that have nothing to do with
  relaxation, which is the failure mode gate G2 exists to catch.

The three regularly indexed series (Fig. 6a) are certified under the primary
configuration with the settings of :data:`REGULAR_SERIES_KW`: G2 reads the rate
at each decimation factor from the least-squares propagator of the decimated
state (``G2=dict(rate="propagator")``), and the rate channel of G3 is decided
on the recovery slope (``G3=dict(bias_tol=inf)``).  The CO2 series (Fig. 6b) is
kept on its weekly calendar and its residual is certified under the
G8-augmented configuration for four segment lengths (:func:`co2_calendar_sweep`).

    python examples/example_03_cross_domain.py                # prints everything (a few minutes)
    python examples/example_03_cross_domain.py --write        # also writes the stored Fig. 6 inputs
    python examples/example_03_cross_domain.py --write OUTDIR # the same files below OUTDIR

``--write`` writes ``analyses/primary_configuration/out/cross_domain.log`` (the
gate statistics of the three regularly indexed series),
``figure_reproduction/inputs/nonneural_series_assessments.csv`` (their verdicts)
and ``figure_reproduction/inputs/co2_block_sweep.json`` / ``.csv`` (the CO2
sweep), the values plotted in Fig. 6.
"""

import argparse
import csv
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)   # the repository root

from ndif import CONFIGURATIONS, Design, Recording, certify, recommend_estimator   # noqa: E402
from ndif.report import render_matrix   # noqa: E402

#: The three regularly indexed series of Fig. 6a.
REGULAR = ("nile", "sunspots", "macro_unemp")

#: Settings of the Fig. 6a assessments: the primary configuration (the default of :func:`ndif.certify`) with the rate
#: of G2 read from the least-squares propagator of the decimated state and the G3 rate channel decided on the
#: recovery slope, and the numbers of repetitions of G3, G4, G6 and G7 below.
REGULAR_SERIES_KW = dict(rng=1, G2=dict(rate="propagator"), G3=dict(n_rep=10, bias_tol=np.inf),
                         G4=dict(n_rep=16), G6=dict(n_rep=25), G7=dict(n_rep=40))

#: Display names of the series in the Fig. 6a table.
DISPLAY = dict(nile="Nile", sunspots="Sunspots", macro_unemp="Unemployment")

#: Segment lengths (weeks) of the CO2 sweep (Fig. 6b).
CO2_BLOCKS = (100, 200, 550, 1100)


def load():
    """Four published series, segmented into blocks of a comparable length."""
    import statsmodels.api as sm

    out = []

    nile = sm.datasets.nile.load_pandas().data["volume"].to_numpy(float)
    out.append(("nile", "hydrology", nile, 1.0, 25))

    sun = sm.datasets.sunspots.load_pandas().data["SUNACTIVITY"].to_numpy(float)
    out.append(("sunspots", "solar physics", sun, 1.0, 50))

    co2 = sm.datasets.co2.load_pandas().data["co2"].to_numpy(float)
    co2 = co2[np.isfinite(co2)]
    out.append(("co2_raw", "atmospheric chem.", co2, 1.0 / 52.0, 200))

    # The same series after the preprocessing its own field applies as a
    # matter of course: remove the secular trend and the annual cycle, keep
    # the residual.  The framework treats preprocessing as measured rather
    # than upstream and neutral, and this pair is the demonstration -- the
    # certificate for the residual is not the certificate for the series.
    t = np.arange(len(co2), dtype=float)
    w = 2 * np.pi / 52.1775
    B = np.column_stack([np.ones_like(t), t, t ** 2,
                         np.cos(w * t), np.sin(w * t),
                         np.cos(2 * w * t), np.sin(2 * w * t)])
    resid = co2 - B @ np.linalg.lstsq(B, co2, rcond=None)[0]
    out.append(("co2_residual", "atmospheric chem.", resid, 1.0 / 52.0, 200))

    mac = sm.datasets.macrodata.load_pandas().data["unemp"].to_numpy(float)
    out.append(("macro_unemp", "macroeconomics", mac, 0.25, 40))

    recs = []
    for name, domain, x, dt, blk in out:
        n = (len(x) // blk) * blk
        x = x[:n]
        bounds = [(i, i + blk) for i in range(0, n, blk)]
        recs.append((name, domain,
                     Recording(x[:, None],
                               Design(dt=dt, window_s=dt, step_s=dt,
                                      segments=bounds, name=name,
                                      modality=domain))))
    return recs


def _verdict(level):
    """'Admitted', 'Refused G6' or 'Indet. G5', as in the Fig. 6a table."""
    if level.status == "admitted":
        return "Admitted"
    return {"refused": "Refused", "indeterminate": "Indet."}[level.status] + f" {level.binding_gate}"


def regular_series(recs):
    """Fig. 6a: the three regularly indexed series under :data:`REGULAR_SERIES_KW`.

    Returns the certificates, the console lines of the gate statistics and the rows of the verdict table.
    """
    lines = ["=" * 88,
             "NDIF on the three regularly indexed non-neural series -- primary configuration",
             "=" * 88]
    certs, table = {}, []
    for name, domain, rec in recs:
        if name not in REGULAR:
            continue
        cert = certify(rec.X, rec.design, **REGULAR_SERIES_KW)
        certs[name] = cert
        g1, g2, g3 = cert.gate("G1"), cert.gate("G2"), cert.gate("G3")
        lines += ["",
                  f"{name}  ({domain})   n = {rec.n_t} in {len(rec.segments())} segments, "
                  f"dt = {rec.design.dt:g}",
                  f"   L1 {cert.levels['L1'].status:<14} "
                  f"L2 {cert.levels['L2'].status:<14} "
                  f"L3 {cert.levels['L3'].status:<14} "
                  f"binding {cert.levels['L3'].binding_gate}",
                  f"   G1 nu    = {g1.value:>8.3f}   {g1.status}"
                  + (f"   [{g1.note}]" if g1.note and g1.passed is None else ""),
                  f"   G2 theta = {g2.value:>8.3f}   {g2.status}",
                  f"   G3 rate recovery = {g3.value:>6.3f}, "
                  f"variance recovery = {g3.detail['variance_recovery']:.3f}"]
        table.append(dict(series=DISPLAY[name], n_and_blocks=f"{rec.n_t} / {len(rec.segments())}",
                          L1=_verdict(cert.levels["L1"]), L2=_verdict(cert.levels["L2"]),
                          L3=_verdict(cert.levels["L3"])))
    print("\n".join(lines))
    return certs, lines, table


def co2_calendar_residual():
    """Residual of the weekly Mauna Loa CO2 series on its complete weekly calendar.

    The source has 2,284 weekly values, 59 of them missing.  The missing weeks are filled by linear interpolation
    between the neighbouring weeks, so that every lagged pair is one calendar week apart, and a quadratic trend and
    two annual harmonics are removed by least squares in calendar time.
    """
    import statsmodels.api as sm
    s = sm.datasets.co2.load_pandas().data["co2"]
    assert len(s) == 2284 and int(s.isna().sum()) == 59, (len(s), int(s.isna().sum()))
    t = np.arange(len(s), dtype=float)
    y = s.interpolate(method="linear", limit_direction="both").to_numpy(float)
    w = 2 * np.pi / 52.1775
    B = np.column_stack([np.ones_like(t), t, t ** 2, np.cos(w * t), np.sin(w * t),
                         np.cos(2 * w * t), np.sin(2 * w * t)])
    return y - B @ np.linalg.lstsq(B, y, rcond=None)[0]


def co2_calendar_sweep():
    """Fig. 6b: one real series, one property changed -- the length of the segments.

    The process is the same in every row; what changes is the recording.  Centring each segment on its own mean
    removes a larger share of a slow component the shorter the segment, so the fitted timescale depends on the
    segmentation.  Every row is certified for least squares under the G8-augmented configuration with the five
    gates that govern level 3 (G0, G2, G8, G1, G3) at their default numbers of repetitions; each segment length uses
    the first 2,200 calendar weeks in complete segments.  Returns the summary written to ``co2_block_sweep.json``.
    """
    from ndif.estimators import fit_ar1_kalman

    r = co2_calendar_residual()
    tref = -1.0 / np.log(fit_ar1_kalman([r])["a"])
    l3 = list(CONFIGURATIONS["g8_augmented"]["level_gates"]["L3"])

    def tri(p):
        return None if p is None else bool(p)

    def failed(cert):
        # the gates of level 3 that fail, in their order of evaluation; '?' marks a gate that is undecided
        out = [n + ("" if cert.gate(n).passed is False else "?") for n in l3
               if cert.gate(n) is not None and cert.gate(n).passed is not True]
        return " ".join(out) or "-"

    print()
    print("=" * 88)
    print(f"CO2 residual on its weekly calendar, segment length swept "
          f"(reference relaxation time {tref:.1f} weeks)")
    print("least squares, G8-augmented configuration, the five gates of level 3, default numbers of repetitions.")
    print("tau fit: relaxation time fitted to the segmented series (weeks); block/tau: segment length in reference")
    print("relaxation times; L3 failed: the gates of level 3 that fail, in their order of evaluation ('?': undecided)")
    print(f"{'block':>7} {'segs':>5} {'tau fit':>8} {'block/tau':>10} {'G3 recovery':>12} {'G2 theta':>9} "
          f"{'L3':>11}  {'L3 failed':<12}")
    print("-" * 81)
    rows = []
    for blk in CO2_BLOCKS:
        n = (len(r) // blk) * blk
        x = r[:n]
        bounds = [(i, i + blk) for i in range(0, n, blk)]
        g = fit_ar1_kalman([x[a:b] for a, b in bounds])
        tau = -1.0 / np.log(max(g["a"], 1e-9))
        cert = certify(x[:, None],
                       Design(dt=1 / 52, window_s=1 / 52, step_s=1 / 52,
                              segments=bounds, name=f"co2_blk{blk}", time_unit="year"),
                       which=l3, configuration="g8_augmented", rng=1)
        g2, g3 = cert.gate("G2"), cert.gate("G3")
        print(f"{blk:>7} {n // blk:>5} {tau:>8.1f} {blk / tref:>10.1f} "
              f"{g3.value:>12.3f} {g2.value:>9.4f} {cert.levels['L3'].status:>11}  "
              f"{failed(cert):<12}")
        rows.append(dict(block=blk, n_segments=n // blk, tau_fit_weeks=float(tau),
                         block_over_tau_fit=float(blk / tau), block_over_tau_reference=float(blk / tref),
                         G3_recovery=float(g3.value), G3_passed=tri(g3.passed),
                         G2_theta=float(g2.value), G2_ci=(g2.detail or {}).get("ci"), G2_passed=tri(g2.passed),
                         gates={q.gate: tri(q.passed) for q in cert.gates},
                         L3=cert.levels["L3"].status, first_failed=cert.levels["L3"].binding_gate))
    print("-" * 81)
    print("  The process is identical in every row; only the segmentation differs.  The")
    print("  fitted timescale changes with the segment length, and each certificate")
    print("  applies to the series as segmented.")
    return dict(gaps="interpolate", n_used=int(len(r)), reference_tau_weeks=float(tref),
                settings="G8-augmented configuration; least squares; the five checks of level 3 "
                         "(G0, G2, G8, G1, G3); default numbers of repetitions; time unit year; rng=1",
                rows=rows)


def write_outputs(base, log_lines, table, sweep):
    """Write the Fig. 6 inputs below ``base`` (the repository root by default)."""
    def target(*parts):
        p = os.path.join(base, *parts)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        return p

    with open(target("analyses", "primary_configuration", "out", "cross_domain.log"), "w",
              encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(log_lines) + "\n")
    with open(target("figure_reproduction", "inputs", "nonneural_series_assessments.csv"), "w",
              encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["series", "n_and_blocks", "L1", "L2", "L3"], lineterminator="\n")
        w.writeheader()
        w.writerows(table)
    with open(target("figure_reproduction", "inputs", "co2_block_sweep.json"), "w",
              encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(sweep, ensure_ascii=False, indent=1) + "\n")
    cols = ["block", "n_segments", "tau_fit_weeks", "G3_recovery", "G2_theta", "L3", "first_failed"]
    with open(target("figure_reproduction", "inputs", "co2_block_sweep.csv"), "w",
              encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(cols)
        for r in sweep["rows"]:
            w.writerow([r[c] for c in cols])
    print(f"\nwritten the Fig. 6 inputs below {base}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--write", nargs="?", const=REPO, default=None, metavar="OUTDIR",
                    help="write the stored Fig. 6 inputs (default: their locations in the repository)")
    a = ap.parse_args(argv)

    recs = load()
    certs, log_lines, table = regular_series(recs)

    print()
    print("=" * 88)
    print("certificate matrix")
    print(render_matrix([certs[n] for n in REGULAR], list(REGULAR)))

    print()
    print("=" * 88)
    print("Which estimator does each series support for a rate?")
    for name, domain, rec in recs:
        if name not in REGULAR:
            continue
        r = recommend_estimator(rec.X, rec.design, rng=1, G3=dict(n_rep=10),
                                G7=dict(n_rep=40))
        print(f"\n  {name:<14} -> {str(r['recommended']):<8}")
        print(f"     {r['reason']}")
        for c in r["candidates"]:
            rf = c["recovered_fraction"]
            print(f"       {c['estimator']:<8} {c['verdict']:<14} "
                  f"binding {str(c['binding_gate']):<4} "
                  f"recovery {rf if rf is None else round(rf, 3)}  "
                  f"{c['cost_s']:.1f}s")

    sweep = co2_calendar_sweep()

    print()
    print("The verdicts differ across the series, and they differ for the")
    print("reasons the gates name.  That is the claim: identifiability is a")
    print("property of a recording and an estimator, in any field that reads a")
    print("stationary observation for a dynamical quantity.")

    if a.write:
        write_outputs(os.path.abspath(a.write), log_lines, table, sweep)


if __name__ == "__main__":
    main()
