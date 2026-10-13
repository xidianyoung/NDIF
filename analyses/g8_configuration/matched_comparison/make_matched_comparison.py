"""Matched-admission comparison of the G8-augmented certificate with the estimator-matched parametric bootstrap.

Tabulates the stored outputs of the extended simulation evaluation; nothing is fitted or simulated here.  The unit is
the simulated recording that has both a certificate of the G8-augmented configuration
(``out/benchmark_09_single.jsonl``, ``out/benchmark_09_variants.jsonl``) and a stored estimator-matched parametric
bootstrap (``out/benchmark_09b_comparator.jsonl``): 384 single-timescale and ten Student t recordings (394).  The ten
unequal-length recordings of the 404-recording evaluation have no comparator record (the comparator requires equal
segment lengths) and are listed as absent.  Each estimator (least squares ``ols``; maximum likelihood ``kalman``) is
tabulated separately, one output per recording.

Rules
  accuracy            an output is inaccurate if the absolute relative error of its rate is at least 0.25; the
                      simulation truth labels accuracy only and never enters a ranking
  fixed_certificate   the recordings whose certificate admits the rate (L3 admitted)
  original_bootstrap  the unmatched comparator rule: accepted if the integer-truncated percent error of the bootstrap
                      timescale self-consistency is at most 20
  matched_bootstrap   the comparator at matched coverage: recordings ranked by the absolute bootstrap timescale
                      self-consistency error abs(err_exact), ascending, with exact ties ordered lexicographically by
                      record key; exactly as many recordings are accepted as the certificate admits
  uncertainty         5000 paired bootstrap draws stratified by simulation condition (cell): the recordings of each
                      condition are resampled with replacement (numpy.random.default_rng(20261009), conditions in
                      ascending cell order; the least-squares draws come first and the maximum-likelihood draws
                      continue the same stream).  On each draw the coverage is re-matched to the number of
                      certificate admissions in the draw and the comparator risk minus the certificate risk is
                      recorded; the interval is the 2.5th and 97.5th percentiles (numpy.percentile, linear).  It is
                      conditional on the simulation design, the fitted generators and the stored bootstrap scores.
  G3-margin sweep     exploratory thresholds of the stored G3 margin among the recordings that pass G0, G2, G8 and G1
                      (admitted if the margin is at least the threshold).  The margin omits the Monte Carlo
                      uncertainty band, so a zero threshold is not the admission rule.

Outputs (``analyses/g8_configuration/matched_comparison/``, or below ``NDIF_OUT`` if set)
  matched_coverage_results.json   rules, absent records and, per estimator, the three selections (n, admitted,
                                  inaccurate admitted, accurate, accurate not admitted, coverage, risk, accurate
                                  retention), the matched cutoff and its tie count, the comparator-minus-certificate
                                  risk difference with its paired stratified interval, and the largest coverage of
                                  the G3-margin sweep
  paired_records.csv              one row per recording and estimator (788): relative error, accuracy, certificate
                                  admission, bootstrap error, unmatched and matched comparator admission, G3 margin
                                  and whether G0, G2, G8 and G1 pass
  risk_coverage_curves.csv        risk-coverage curves: the bootstrap ranking at every coverage (394 thresholds per
                                  estimator) and the G3-margin sweep

Reported in: Extended Data Fig. 2b and Supplementary Table 5 (maximum likelihood, 179 admissions: 21 inaccurate for
the certificate and 16 for the matched bootstrap; risk difference -2.793 [-7.345, 1.156] percentage points);
Supplementary Fig. 3 (the least-squares bootstrap ranking, unmatched 277/365 and coverage-matched 63/72 against the
certificate's 5/72; panel b, the maximum-likelihood curves; the G3-margin sweeps of both panels); Supplementary Note 4.

Runtime: about 10 s on one desktop core.

    python analyses/g8_configuration/matched_comparison/make_matched_comparison.py --draws 5000 --seed 20261009
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from analyses.common import paths  # noqa: E402

PARTS = ("single", "variants")
ESTIMATORS = ("ols", "kalman")
OTHER_GATES = ("G0", "G2", "G8", "G1")
TOL = 0.25            # relative rate error at or above which an output is inaccurate
ACCEPT_PCT = 20       # unmatched comparator rule: integer-truncated percent error at most 20
DRAWS = 5000
SEED = 20261009

FIELDS = ["key", "estimator", "cell", "family", "relative_error", "inaccurate", "ndif_admitted",
          "bootstrap_error_abs", "original_bootstrap_admitted", "g3_margin", "other_four_gates_pass",
          "matched_bootstrap_admitted"]
CURVE_FIELDS = ["estimator", "method", "threshold", "n", "admitted", "inaccurate_admitted", "accurate",
                "accurate_not_admitted", "coverage", "risk", "accurate_retention"]


def read_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def summary(rows, admitted):
    """Selection performance of one admission mask over the paired recordings of one estimator."""
    n = len(rows)
    adm = [r for r, a in zip(rows, admitted) if a]
    accurate = sum(not r["inaccurate"] for r in rows)
    accurate_not_admitted = sum(not r["inaccurate"] and not a for r, a in zip(rows, admitted))
    bad = sum(r["inaccurate"] for r in adm)
    return dict(n=n, admitted=len(adm), inaccurate_admitted=bad, accurate=accurate,
                accurate_not_admitted=accurate_not_admitted, coverage=len(adm) / n,
                risk=bad / len(adm) if adm else None,
                accurate_retention=(accurate - accurate_not_admitted) / accurate if accurate else None)


def matched_difference(err, keys, bad, adm, idx):
    """Comparator risk minus certificate risk on the recordings ``idx``, at the certificate's admission count."""
    k = int(adm[idx].sum())
    if k == 0:
        return np.nan
    cert_bad = int(bad[idx][adm[idx]].sum())
    order = sorted(range(len(idx)), key=lambda j: (err[idx[j]], keys[idx[j]]))[:k]
    boot_bad = int(bad[idx[np.array(order)]].sum())
    return (boot_bad - cert_bad) / k


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--records", default=paths.repo("analyses", "g8_configuration", "out"),
                    help="directory of benchmark_09_{single,variants}.jsonl and benchmark_09b_comparator.jsonl")
    ap.add_argument("--draws", type=int, default=DRAWS, help="paired stratified bootstrap draws")
    ap.add_argument("--seed", type=int, default=SEED, help="seed of numpy.random.default_rng for the draws")
    a = ap.parse_args()
    records = Path(a.records)

    cert = {}
    for part in PARTS:
        for r in read_jsonl(records / f"benchmark_09_{part}.jsonl"):
            cert[f"{part}|{r['key'].split('|')[0]}|{r['rep']}"] = r
    comp = {c["key"]: c for c in read_jsonl(records / "benchmark_09b_comparator.jsonl")}
    absent = [k for k in cert if k not in comp]
    paired = sorted(comp)

    rows = {}
    for est in ESTIMATORS:
        rr = []
        for k in paired:
            ce = cert[k]["certs"][est]
            boot = comp[k][f"est_{est}"]
            gates = ce["gates"]
            rel = ce["rel_error"]["primary"]
            rr.append(dict(key=k, estimator=est, cell=cert[k]["cell"], family=cert[k]["family"], relative_error=rel,
                           inaccurate=rel >= TOL, ndif_admitted=ce["levels"]["L3"]["status"] == "admitted",
                           bootstrap_error_abs=abs(boot["err_exact"]),
                           original_bootstrap_admitted=abs(boot["err_pct"]) <= ACCEPT_PCT,
                           g3_margin=gates["G3"]["margin"],
                           other_four_gates_pass=all(gates[g]["passed"] == 1 for g in OTHER_GATES)))
        k_adm = sum(r["ndif_admitted"] for r in rr)
        ranked = sorted(rr, key=lambda r: (r["bootstrap_error_abs"], r["key"]))
        chosen = {r["key"] for r in ranked[:k_adm]}
        for r in rr:
            r["matched_bootstrap_admitted"] = r["key"] in chosen
        rows[est] = rr

    # paired condition-stratified bootstrap: one generator, least-squares draws first
    rng = np.random.default_rng(a.seed)
    intervals = {}
    for est in ESTIMATORS:
        rr = rows[est]
        keys = [r["key"] for r in rr]
        err = np.array([r["bootstrap_error_abs"] for r in rr])
        bad = np.array([r["inaccurate"] for r in rr])
        adm = np.array([r["ndif_admitted"] for r in rr])
        strata = defaultdict(list)
        for i, r in enumerate(rr):
            strata[r["cell"]].append(i)
        strata = [np.array(strata[c]) for c in sorted(strata)]
        diffs = np.array([matched_difference(err, keys, bad, adm,
                                             np.concatenate([rng.choice(s, size=len(s), replace=True) for s in strata]))
                          for _ in range(a.draws)])
        intervals[est] = [float(v) for v in np.percentile(diffs, [2.5, 97.5])]

    results, curves = {}, []
    for est in ESTIMATORS:
        rr = rows[est]
        fixed = summary(rr, [r["ndif_admitted"] for r in rr])
        original = summary(rr, [r["original_bootstrap_admitted"] for r in rr])
        matched = summary(rr, [r["matched_bootstrap_admitted"] for r in rr])
        cutoff = max(r["bootstrap_error_abs"] for r in rr if r["matched_bootstrap_admitted"])
        for t in sorted(r["bootstrap_error_abs"] for r in rr):
            curves.append(dict(estimator=est, method="bootstrap", threshold=t,
                               **summary(rr, [r["bootstrap_error_abs"] <= t for r in rr])))
        g3_rows = []
        for t in sorted((r["g3_margin"] for r in rr if r["other_four_gates_pass"]), reverse=True):
            g3_rows.append(dict(estimator=est, method="G3_margin_exploration", threshold=t,
                                **summary(rr, [r["other_four_gates_pass"] and r["g3_margin"] >= t for r in rr])))
        curves.extend(g3_rows)
        results[est] = dict(fixed_certificate=fixed, original_bootstrap=original, matched_bootstrap=matched,
                            bootstrap_abs_error_cutoff=cutoff,
                            cutoff_tie_total=sum(r["bootstrap_error_abs"] == cutoff for r in rr),
                            excess_bootstrap_risk=matched["risk"] - fixed["risk"],
                            paired_stratified_bootstrap_ci95=intervals[est], bootstrap_draws=a.draws,
                            g3_sweep_max_coverage=max(c["coverage"] for c in g3_rows))

    families = defaultdict(int)
    for k in paired:
        families[cert[k]["family"]] += 1
    absent_cells = sorted({cert[k]["cell"] for k in absent})
    out = {
        "analysis": "Post hoc comparison.",
        "scope": f"{families['single']} single-timescale recordings and {families['student_t']} Student t "
                 f"recordings with equal segments.",
        "expected_recordings": len(cert),
        "paired_recordings": len(paired),
        "absent_comparator_records": [dict(key=k, family=cert[k]["family"], cell=cert[k]["cell"], rep=cert[k]["rep"])
                                      for k in absent],
        "absent_reason": f"All {len(absent)} unequal-length recordings (cell {', '.join(map(str, absent_cells))}) are "
                         f"absent: the comparator requires equal segment lengths.",
        "accuracy_rule": f"Inaccurate iff absolute relative error >= {TOL}.",
        "coverage_rule": "For each estimator, rank abs(err_exact) ascending and accept exactly as many recordings as "
                         "the certificate admits. Exact ties are ordered lexicographically by key, never by truth. "
                         f"The unmatched comparator (integer-truncated percent error <= {ACCEPT_PCT}) is reported "
                         "separately as original_bootstrap.",
        "uncertainty": f"{a.draws} paired condition-stratified bootstrap draws, seed {a.seed}. Coverage is re-matched "
                       "on each draw; percentile CI for comparator risk minus certificate risk, conditional on the "
                       "simulation design.",
        "g3_curve": "Exploratory thresholds of the stored G3 margin among recordings that pass G0, G2, G8 and G1. The "
                    "margin omits the Monte Carlo uncertainty band, so a zero threshold is not the admission rule.",
        "limitations": "Compares the stored estimator-matched parametric bootstrap only.",
        "results": results,
    }

    target = paths.out("analyses", "g8_configuration", "matched_comparison", "matched_coverage_results.json")
    with open(target, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(out, indent=2) + "\n")
    for name, fields, table in (("paired_records.csv", FIELDS, rows["ols"] + rows["kalman"]),
                                ("risk_coverage_curves.csv", CURVE_FIELDS, curves)):
        with open(paths.out("analyses", "g8_configuration", "matched_comparison", name), "w",
                  encoding="utf-8-sig", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
            w.writeheader()
            w.writerows(table)

    for est in ESTIMATORS:
        r = results[est]
        print(f"{est}: certificate {r['fixed_certificate']['inaccurate_admitted']}/{r['fixed_certificate']['admitted']}"
              f", matched bootstrap {r['matched_bootstrap']['inaccurate_admitted']}/{r['matched_bootstrap']['admitted']}"
              f", unmatched bootstrap {r['original_bootstrap']['inaccurate_admitted']}/"
              f"{r['original_bootstrap']['admitted']}; risk difference {100 * r['excess_bootstrap_risk']:.3f} "
              f"[{100 * r['paired_stratified_bootstrap_ci95'][0]:.3f}, {100 * r['paired_stratified_bootstrap_ci95'][1]:.3f}]"
              f" percentage points")
    print("written:", Path(target).parent)


if __name__ == "__main__":
    main()
