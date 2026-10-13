"""Selection performance and leave-one-check-out exclusions of the certificates, tabulated from stored outputs.

Nothing is fitted or simulated.  An output is accurate if the absolute relative error of its rate is below 0.25.

Tables
  g8_outputs_truth_and_gates.csv  one row per estimator output of the G8-augmented extended evaluation (1,648):
                                  the 404 recordings where equation (6) holds (``single`` 384, ``variants`` 20), the
                                  100 misspecified recordings (``missp``) and the 320 sleep-matched simulations
                                  (``twins``), with the simulated rate, the estimate, the relative error, the stored
                                  L3 status and the outcome (pass / fail / unavailable) of G0, G1, G2, G3 and G8
  selection_performance.csv       admissions, inaccurate admissions, accurate estimates retained, admission coverage,
                                  admission risk and accurate-estimate retention for
                                    the 320-output benchmark of the benchmark configuration,
                                    the three G8-augmented evaluations (pooled and per estimator),
                                    the matched-admission comparisons on the 394 paired recordings (certificate
                                    against the noise-aware least-squares bootstrap ranked by latent-target bias, and
                                    against the estimator-matched maximum-likelihood bootstrap),
                                    the 602 real sleep cells under both configurations (statuses only; truth unknown)
  check_removal_ablation.csv      for each G8-augmented evaluation and each check of L3 (G0, G2, G8, G1, G3): the
                                  inaccurate and the accurate outputs that would be admitted if only that check were
                                  removed with every other rule fixed (failed and unavailable outcomes both count)
  selection_sensitivity.json      for the 808 outputs where equation (6) holds: inaccurate admissions at relative-error
                                  tolerances 0.10, 0.15, 0.20, 0.25 and 0.30; and the subtraction of each admitted
                                  estimate's G3 recovery offset from its log rate (inaccurate admissions and mean log
                                  error before and after); the subtraction is a sensitivity analysis and not part of
                                  the certificate

L3 status of an output from its check outcomes: refused if any of G0, G2, G8, G1, G3 fails, otherwise indeterminate if
any is unavailable, otherwise admitted.  The tabulation checks that this reproduces the stored L3 status of every
output and the leave-one-check-out counts of ``out/benchmark_09_results.json``.

Inputs (stored package files)
  analyses/primary_configuration/out/benchmark_01.json
  analyses/primary_configuration/out/corpus_full_summary.json
  analyses/g8_configuration/out/benchmark_09_{single,variants,missp,twins}.jsonl, benchmark_09_results.json,
      benchmark_10_results.json
  analyses/g8_configuration/noise_aware_bootstrap/paired_records.csv
  analyses/g8_configuration/matched_comparison/paired_records.csv

Outputs: analyses/g8_configuration/out/ (or below NDIF_OUT if set).

Reported in: Supplementary Table 6 and Supplementary Note 4 (261 of 808 admitted, 28 inaccurate, 233 of 339 accurate
admitted; misspecification 10 admitted, all inaccurate, 187 refused, 3 indeterminate; two-timescale family 40 of 40
refused; sleep-matched 16 of 210 accurate admitted, 633 scoreable outputs; check-alone exclusions G2 3/21, G3 85/27,
G8 2/16, G1 3/11 and, under misspecification, G8 32/1, G3 19/0; offset subtraction 28 -> 4 inaccurate admissions;
the tolerance sweep, 109 of 261 admissions inaccurate at the 10 % tolerance); Extended Data Fig. 2 and Supplementary
Tables 4 and 5 (72 least-squares admissions, 5 against 4 inaccurate; 179 maximum-likelihood admissions, 21 against
16); the real sleep counts of Supplementary Note 4 (G8-augmented least squares 11/583/8, maximum likelihood 6/590/6).

Runtime: a few seconds on one desktop core.

    python analyses/g8_configuration/tabulate_selection.py
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from analyses.common import paths  # noqa: E402

G8_OUT = Path(paths.repo("analyses", "g8_configuration", "out"))
PRIMARY_OUT = Path(paths.repo("analyses", "primary_configuration", "out"))
OLS_PAIRED = Path(paths.repo("analyses", "g8_configuration", "noise_aware_bootstrap", "paired_records.csv"))
ML_PAIRED = Path(paths.repo("analyses", "g8_configuration", "matched_comparison", "paired_records.csv"))

TOLERANCE = 0.25
GATES = ("G0", "G2", "G8", "G1", "G3")
G8_CONFIGURATION = "G8-augmented (G0, G2, G8, G1, G3)"
PARTS = ("single", "variants", "missp", "twins")
ESTIMATORS = ("ols", "kalman")
RECORD_FIELDS = ["part", "record_id", "cell", "rep", "family", "source", "estimator", "target_primary",
                 "estimated_rate", "relative_error", "stored_L3", "G0", "G1", "G2", "G3", "G8"]
SWEEP_TOLERANCES = (0.10, 0.15, 0.20, 0.25, 0.30)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def read_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_csv(name, rows, fields=None):
    with open(paths.out("analyses", "g8_configuration", "out", name), "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def outcome(gate):
    """pass / fail / unavailable from a stored check record."""
    passed = None if gate is None else gate.get("passed")
    return "unavailable" if passed is None else ("pass" if passed else "fail")


def output_rows(records):
    """The per-output table of the G8-augmented evaluation."""
    rows = []
    for part in PARTS:
        for r in records[part]:
            for est in ESTIMATORS:
                ce = r["certs"][est]
                rel = (ce.get("rel_error") or {}).get("primary")
                rows.append(dict(part=part, record_id=f"{part}|{r['key']}", cell=r.get("cell", ""), rep=r.get("rep", ""),
                                 family=r.get("family", "sleep_matched"), source=r.get("source", ""), estimator=est,
                                 target_primary=r["targets"]["primary"],
                                 estimated_rate="" if ce.get("lam") is None else ce["lam"],
                                 relative_error="" if rel is None else rel, stored_L3=ce["levels"]["L3"]["status"],
                                 **{g: outcome(ce["gates"].get(g)) for g in ("G0", "G1", "G2", "G3", "G8")}))
    return rows


def status(row, gates=GATES):
    values = [row[g] for g in gates]
    if "fail" in values:
        return "refused"
    if "unavailable" in values:
        return "indeterminate"
    return "admitted"


def accuracy_counts(rows):
    """Keep all outputs; truth-scoring excludes missing estimates explicitly."""
    scoreable = [r for r in rows if r["relative_error"] != ""]
    admitted = [r for r in rows if status(r) == "admitted"]
    assert all(r["relative_error"] != "" for r in admitted)
    accurate = [r for r in scoreable if float(r["relative_error"]) < TOLERANCE]
    inaccurate_admitted = sum(float(r["relative_error"]) >= TOLERANCE for r in admitted)
    accurate_not_admitted = sum(status(r) != "admitted" for r in accurate)
    return dict(
        outputs_total=len(rows), outputs_scoreable=len(scoreable),
        missing_rate_outputs=len(rows) - len(scoreable), admitted=len(admitted),
        inaccurate_admitted=inaccurate_admitted, accurate_total=len(accurate),
        accurate_admitted=len(accurate) - accurate_not_admitted,
        accurate_not_admitted=accurate_not_admitted,
        statuses=dict(Counter(status(r) for r in rows)),
    )


def performance_row(evaluation, configuration, estimator, method, counts, truth_known=True,
                    unit="estimator output", note=""):
    n = counts["outputs_total"]
    admitted = counts["admitted"]
    accurate = counts.get("accurate_total") if truth_known else None
    bad = counts.get("inaccurate_admitted") if truth_known else None
    retained = counts.get("accurate_admitted") if truth_known else None
    return dict(
        evaluation=evaluation, configuration=configuration, estimator=estimator,
        method=method, unit=unit, outputs_total=n,
        outputs_scoreable=counts.get("outputs_scoreable", "") if truth_known else "",
        missing_rate_outputs=counts.get("missing_rate_outputs", "") if truth_known else "",
        admitted=admitted, inaccurate_admitted=bad if bad is not None else "",
        accurate_total=accurate if accurate is not None else "",
        accurate_admitted=retained if retained is not None else "",
        accurate_not_admitted=counts.get("accurate_not_admitted", "") if truth_known else "",
        admission_coverage=admitted / n,
        inaccurate_admission_risk=bad / admitted if bad is not None and admitted else "",
        accurate_retention=retained / accurate if accurate else "",
        refused=counts.get("statuses", {}).get("refused", ""),
        indeterminate=counts.get("statuses", {}).get("indeterminate", ""),
        truth_known=truth_known, note=note,
    )


def certificate_status(gates, order=GATES):
    """L3 status from the stored check records of one certificate (a missing check counts as unavailable)."""
    out = "admitted"
    for k in order:
        if k not in gates:
            out = "indeterminate" if out == "admitted" else out
            continue
        p = gates[k]["passed"]
        if p is not None and not p:
            return "refused"
        if p is None:
            out = "indeterminate"
    return out


def score(recs, tol=TOLERANCE):
    """Admissions and inaccurate admissions over both estimators of the records ``recs`` at tolerance ``tol``."""
    c = Counter()
    for r in recs:
        for e in ESTIMATORS:
            ce = r["certs"].get(e)
            if ce is None:
                continue
            rel = (ce.get("rel_error") or {}).get("primary")
            if rel is None:
                continue
            adm = certificate_status(ce["gates"]) == "admitted"
            acc = rel < tol
            c["n"] += 1
            c["admitted"] += adm
            c["fp"] += adm and not acc
            c["accurate"] += acc
            c["fr"] += acc and not adm
            c["agree"] += adm == acc
    return dict(c)


def sensitivity(holds):
    """Tolerance sweep and subtraction of the G3 recovery offset, over the outputs where equation (6) holds."""
    sweep = {}
    for t in SWEEP_TOLERANCES:
        c = score(holds, tol=t)
        sweep[f"{t:.2f}"] = dict(admitted=c["admitted"], inaccurate_admitted=c["fp"],
                                 percent_inaccurate=round(100 * c["fp"] / c["admitted"], 1))
    corr = Counter()
    le0, le1 = [], []
    for r in holds:
        for e in ESTIMATORS:
            ce = r["certs"][e]
            if certificate_status(ce["gates"]) != "admitted":
                continue
            lt = math.log(ce["lam"] / r["targets"]["primary"])
            off = ce["gates"]["G3"]["offset"]
            le0.append(lt)
            le1.append(lt - off)
            corr["n"] += 1
            corr["fp"] += abs(math.exp(lt) - 1) >= TOLERANCE
            corr["fp_corrected"] += abs(math.exp(lt - off) - 1) >= TOLERANCE
    offset = dict(admitted=corr["n"], inaccurate=corr["fp"], inaccurate_after=corr["fp_corrected"],
                  mean_log_error=float(np.mean(le0)), mean_log_error_after=float(np.mean(le1)),
                  sd_log_error=float(np.std(le0, ddof=1)), sd_log_error_after=float(np.std(le1, ddof=1)))
    return dict(outputs=sum(len(r["certs"]) for r in holds), relative_error_tolerance=TOLERANCE,
                tolerance_sweep=sweep, offset_subtraction=offset)


def main():
    argparse.ArgumentParser(description=__doc__.split("\n")[0]).parse_args()
    primary = read_json(PRIMARY_OUT / "benchmark_01.json")
    pc = primary["cert_rows"]
    assert len(pc) == 320
    assert len({(r["nu"], r["geom"], r["rep"]) for r in pc}) == 160
    pcounts = dict(outputs_total=320, outputs_scoreable=320, missing_rate_outputs=0,
                   admitted=sum(r["verdict"] == "admitted" for r in pc),
                   inaccurate_admitted=sum(r["verdict"] == "admitted" and r["rel_error"] >= TOLERANCE for r in pc),
                   accurate_total=sum(r["rel_error"] < TOLERANCE for r in pc),
                   accurate_admitted=sum(r["verdict"] == "admitted" and r["rel_error"] < TOLERANCE for r in pc),
                   accurate_not_admitted=sum(r["verdict"] != "admitted" and r["rel_error"] < TOLERANCE for r in pc),
                   statuses=dict(Counter(r["verdict"] for r in pc)))
    assert (pcounts["admitted"], pcounts["inaccurate_admitted"], pcounts["accurate_total"], pcounts["accurate_not_admitted"]) == (114, 23, 115, 24)
    selection = [performance_row("primary_known_rate_benchmark", "G0,G1,G2,G3,G7; G4 omitted",
                                 "OLS+ML", "NDIF", pcounts,
                                 note="160 recordings, two dependent estimator outputs per recording; 25% relative-rate tolerance.")]

    records = {part: read_jsonl(G8_OUT / f"benchmark_09_{part}.jsonl") for part in PARTS}
    all_g8 = output_rows(records)
    assert len(all_g8) == 1648
    assert len({(r["record_id"], r["estimator"]) for r in all_g8}) == len(all_g8)
    assert all(status(r) == r["stored_L3"] for r in all_g8)
    g8_summary = read_json(G8_OUT / "benchmark_09_results.json")
    groups = {
        "g8_correct_model_and_variants": [r for r in all_g8 if r["part"] in ("single", "variants")],
        "g8_misspecified": [r for r in all_g8 if r["part"] == "missp"],
        "g8_sleep_matched": [r for r in all_g8 if r["part"] == "twins"],
    }
    expected = {
        "g8_correct_model_and_variants": (808, 808, 261, 28, 339, 106),
        "g8_misspecified": (200, 200, 10, 10, 8, 8),
        "g8_sleep_matched": (640, 633, 16, 0, 210, 194),
    }
    group_counts = {}
    for name, rows in groups.items():
        counts = accuracy_counts(rows)
        assert tuple(counts[k] for k in ("outputs_total", "outputs_scoreable", "admitted", "inaccurate_admitted", "accurate_total", "accurate_not_admitted")) == expected[name]
        group_counts[name] = counts
        for estimator in ("OLS+ML", "ols", "kalman"):
            selected = rows if estimator == "OLS+ML" else [r for r in rows if r["estimator"] == estimator]
            selection.append(performance_row(name, G8_CONFIGURATION, estimator, "NDIF", accuracy_counts(selected),
                note="Internal evaluation; do not pool configurations. Sleep matched has seven unavailable OLS rates, excluded only from accuracy scoring." if name == "g8_sleep_matched" else "Internal evaluation; dependent estimator outputs; no controlled false-admission risk."))
    assert g8_summary["model_holds"]["pooled"]["counts"]["n"] == group_counts["g8_correct_model_and_variants"]["outputs_total"]
    assert g8_summary["twins"]["pooled"]["counts"]["n"] == group_counts["g8_sleep_matched"]["outputs_scoreable"]
    two_mode = [r for r in groups["g8_misspecified"] if r["family"] == "two_timescales"]
    assert len(two_mode) == 40 and all(status(r) == "refused" for r in two_mode)

    ablation = []
    for name, rows in groups.items():
        source_key = {"g8_correct_model_and_variants": "holds", "g8_misspecified": "missp", "g8_sleep_matched": "twins"}[name]
        for gate in GATES:
            inaccurate_removed = accurate_vetoed = 0
            for r in rows:
                if r["relative_error"] == "":
                    continue
                if status(r) != "admitted" and status(r, tuple(g for g in GATES if g != gate)) == "admitted":
                    inaccurate_removed += float(r["relative_error"]) >= TOLERANCE
                    accurate_vetoed += float(r["relative_error"]) < TOLERANCE
            saved = g8_summary["remove_one_check"][source_key][gate]
            assert inaccurate_removed == saved["inaccurate_admissions_removed"]
            assert accurate_vetoed == saved["accurate_estimates_refused"]
            ablation.append(dict(evaluation=name, configuration=G8_CONFIGURATION, removed_gate=gate,
                outputs_total=len(rows), inaccurate_admissions_prevented=int(inaccurate_removed), accurate_estimates_not_admitted=int(accurate_vetoed),
                interpretation="Within-NDIF remove-one diagnostic yield and cost; not a strong-baseline comparison, causal effect, or risk guarantee."))

    ols = read_csv(OLS_PAIRED)
    ml = [r for r in read_csv(ML_PAIRED) if r["estimator"] == "kalman"]
    paired_counts = {}
    for est, rows, ndif_key, baseline_key, error_key, id_key in (
        ("ols", ols, "original_ndif_admitted", "matched_score_latent_bias", "inaccurate", "record_id"),
        ("kalman", ml, "ndif_admitted", "matched_bootstrap_admitted", "inaccurate", "key"),
    ):
        assert len(rows) == 394 and len({r[id_key] for r in rows}) == 394
        for method, key in (("NDIF", ndif_key), ("noise_aware_bootstrap" if est == "ols" else "estimator_matched_bootstrap", baseline_key)):
            admitted = [r for r in rows if r[key] == "True"]
            counts = dict(outputs_total=394, outputs_scoreable=394, missing_rate_outputs=0, admitted=len(admitted),
                inaccurate_admitted=sum(r[error_key] == "True" for r in admitted), accurate_total=sum(r[error_key] == "False" for r in rows),
                accurate_admitted=sum(r[error_key] == "False" for r in admitted),
                accurate_not_admitted=sum(r[error_key] == "False" and r[key] != "True" for r in rows))
            paired_counts[f"{est}|{method}"] = counts
            selection.append(performance_row("matched_admission_394", "paired G8 evaluation; post hoc matched ranking", est, method, counts, unit="recording for one estimator",
                note="384 Gaussian + ten Student-t recordings; ten unequal-segment recordings lack the comparator. Ranking never uses simulation truth."))
    assert (paired_counts["ols|NDIF"]["admitted"], paired_counts["ols|NDIF"]["inaccurate_admitted"], paired_counts["ols|noise_aware_bootstrap"]["inaccurate_admitted"]) == (72, 5, 4)
    assert (paired_counts["kalman|NDIF"]["admitted"], paired_counts["kalman|NDIF"]["inaccurate_admitted"], paired_counts["kalman|estimator_matched_bootstrap"]["inaccurate_admitted"]) == (179, 21, 16)

    primary_sleep = read_json(PRIMARY_OUT / "corpus_full_summary.json")
    real_g8 = read_json(G8_OUT / "benchmark_10_results.json")["sleep"]
    sleep_counts = {}
    for config in ("primary_G0_to_G7", "g8_augmented"):
        for est in ESTIMATORS:
            if config == "primary_G0_to_G7":
                rows = [r for r in primary_sleep["rows"] if r["estimator"] == est]
                statuses = dict(Counter(r["L3"] for r in rows))
                assert len(rows) == 602
            else:
                saved = real_g8[est]["L3"]
                statuses = {s: saved.get(s, 0) for s in ("admitted", "refused", "indeterminate")}
            assert sum(statuses.values()) == 602
            counts = dict(outputs_total=602, admitted=statuses.get("admitted", 0), statuses=statuses)
            sleep_counts[f"{config}|{est}"] = counts
            selection.append(performance_row("real_sleep_602_cells", config, est, "NDIF", counts, truth_known=False,
                unit="night-stage cell for one estimator", note="153 nights, 78 participants; 602 shared cells and 1204 dependent outputs across estimators. Latent truth unknown; no accuracy/risk calculated."))
    assert tuple(sleep_counts["primary_G0_to_G7|ols"]["statuses"].get(s, 0) for s in ("admitted", "refused", "indeterminate")) == (0, 590, 12)
    assert tuple(sleep_counts["primary_G0_to_G7|kalman"]["statuses"].get(s, 0) for s in ("admitted", "refused", "indeterminate")) == (87, 511, 4)
    assert tuple(sleep_counts["g8_augmented|ols"]["statuses"].get(s, 0) for s in ("admitted", "refused", "indeterminate")) == (11, 583, 8)
    assert tuple(sleep_counts["g8_augmented|kalman"]["statuses"].get(s, 0) for s in ("admitted", "refused", "indeterminate")) == (6, 590, 6)

    sens = sensitivity(records["single"] + records["variants"])

    write_csv("g8_outputs_truth_and_gates.csv", all_g8, RECORD_FIELDS)
    write_csv("selection_performance.csv", selection)
    write_csv("check_removal_ablation.csv", ablation)
    target = paths.out("analyses", "g8_configuration", "out", "selection_sensitivity.json")
    with open(target, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(sens, indent=1) + "\n")

    print(f"rows: outputs {len(all_g8)}, selection_performance {len(selection)}, check_removal_ablation {len(ablation)}")
    for t, v in sens["tolerance_sweep"].items():
        print(f"tolerance {t}: {v['inaccurate_admitted']} of {v['admitted']} admissions inaccurate ({v['percent_inaccurate']} %)")
    o = sens["offset_subtraction"]
    print(f"offset subtraction: inaccurate admissions {o['inaccurate']} -> {o['inaccurate_after']} of {o['admitted']}; "
          f"mean log error {o['mean_log_error']:.3f} -> {o['mean_log_error_after']:.3f}")
    print("two-timescale family: 40 of 40 outputs refused")
    print("written:", Path(target).parent)


if __name__ == "__main__":
    main()
