"""Recount the stored outputs behind the numbers reported in the manuscript.  Nothing is fitted or simulated.

    python verification/recount_stored_outputs.py [--output verification/recount_stored_outputs.json]

Reads the stored outputs of the three analysis configurations and reports the counts that the manuscript cites:
the 320-output benchmark (benchmark configuration), the 1,204 sleep cell assessments and the 336 adjacent-stage
pairs, the 83 HCP MEG and 56 COG-BCI assessments (primary configuration), and the extended simulation evaluation,
the matched comparisons and the sleep re-evaluation (G8-augmented configuration), and the drift-replacement and
separation-threshold counts of the 646 recordings (Fig. 2b,c).  Each count is compared with the
value stated in the manuscript; the script exits with status 1 if any differs.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PRIMARY = os.path.join(ROOT, "analyses", "primary_configuration", "out")
WORKLOAD = os.path.join(ROOT, "analyses", "workload_certificates")
G8 = os.path.join(ROOT, "analyses", "g8_configuration")
INPUTS = os.path.join(ROOT, "figure_reproduction", "inputs")

PRIMARY_L3 = ["G0", "G1", "G2", "G3", "G7", "G4"]
AUGMENTED_L3 = ["G0", "G2", "G8", "G1", "G3"]


def score(gates_by_name, order):
    """The verdict of a level from stored gate flags under a governing order (a failed gate takes precedence)."""
    present = [gates_by_name[n] for n in order if n in gates_by_name]
    if not present:
        return "indeterminate"
    if any(g["passed"] is not None and not bool(g["passed"]) for g in present):
        return "refused"
    if any(g["passed"] is None for g in present):
        return "indeterminate"
    return "admitted"


def jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def check(report, key, value, expected):
    report[key] = dict(value=value, expected=expected, ok=(value == expected))
    return value == expected


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", default=os.path.join(HERE, "recount_stored_outputs.json"))
    a = ap.parse_args()
    rep = {}
    ok = True

    # ---- benchmark configuration: 320 estimator outputs with known rates
    b = json.load(open(os.path.join(PRIMARY, "benchmark_01.json"), encoding="utf-8"))
    c = Counter()
    for row in b["cert_rows"]:
        admitted = row["verdict"] == "admitted"
        accurate = row["rel_error"] is not None and row["rel_error"] < 0.25
        c[("admitted" if admitted else "not admitted", "accurate" if accurate else "inaccurate")] += 1
    ok &= check(rep, "benchmark.outputs", len(b["cert_rows"]), 320)
    ok &= check(rep, "benchmark.accurate_admitted", c[("admitted", "accurate")], 91)
    ok &= check(rep, "benchmark.inaccurate_admitted", c[("admitted", "inaccurate")], 23)
    ok &= check(rep, "benchmark.accurate_not_admitted", c[("not admitted", "accurate")], 24)
    ok &= check(rep, "benchmark.inaccurate_not_admitted", c[("not admitted", "inaccurate")], 182)

    # ---- primary configuration: 602 sleep cells x 2 estimators, 336 adjacent-stage pairs
    cells = defaultdict(Counter)
    binding = defaultdict(Counter)
    rescored_mismatch = []
    pairs = Counter()
    for path in sorted(glob.glob(os.path.join(PRIMARY, "corpus_full", "*.json"))):
        if path.endswith("_run_manifest.json"):
            continue
        row = json.load(open(path, encoding="utf-8"))
        if row.get("kind") == "pair":
            pairs[str(row["passed"])] += 1
            continue
        est = row["estimator"]
        status = row["levels"]["L3"]["status"]
        cells[est][status] += 1
        if status == "refused":
            binding[est][row["levels"]["L3"]["binding_gate"]] += 1
        by = {g["gate"]: g for g in row["gates"]}
        if score(by, PRIMARY_L3) != status:
            rescored_mismatch.append(os.path.basename(path))
    ok &= check(rep, "sleep.cells_per_estimator", {k: sum(v.values()) for k, v in cells.items()}, {"ols": 602, "kalman": 602})
    ok &= check(rep, "sleep.ols_L3", dict(cells["ols"]), {"refused": 590, "indeterminate": 12})
    ok &= check(rep, "sleep.kalman_L3", dict(cells["kalman"]), {"refused": 511, "admitted": 87, "indeterminate": 4})
    ok &= check(rep, "sleep.ols_first_failed", dict(binding["ols"]), {"G1": 557, "G2": 11, "G3": 22})
    ok &= check(rep, "sleep.kalman_first_failed", dict(binding["kalman"]), {"G2": 263, "G3": 248})
    ok &= check(rep, "sleep.rescored_from_gate_flags_mismatches", rescored_mismatch, [])
    ok &= check(rep, "sleep.adjacent_stage_pairs", dict(pairs), {"True": 336})

    # ---- primary configuration: HCP MEG (83) and COG-BCI workload (56)
    def cohort(dirname):
        lv = defaultdict(Counter)
        first = defaultdict(Counter)
        n = 0
        for path in sorted(glob.glob(os.path.join(dirname, "*.json"))):
            row = json.load(open(path, encoding="utf-8"))
            n += 1
            for L in ("L1", "L2", "L3"):
                st = row["levels"][L]["status"]
                lv[L][st] += 1
                if st == "refused":
                    first[L][row["levels"][L]["binding_gate"]] += 1
        return n, {k: dict(v) for k, v in lv.items()}, {k: dict(v) for k, v in first.items()}
    n, lv, first = cohort(os.path.join(PRIMARY, "corpus_meg"))
    ok &= check(rep, "meg.n", n, 83)
    ok &= check(rep, "meg.L1", lv["L1"], {"admitted": 72, "refused": 11})
    ok &= check(rep, "meg.L2", lv["L2"], {"admitted": 83})
    ok &= check(rep, "meg.L3", lv["L3"], {"refused": 83})
    ok &= check(rep, "meg.L3_first_failed", first["L3"], {"G3": 82, "G2": 1})
    n, lv, first = cohort(os.path.join(WORKLOAD, "corpus_cog"))
    ok &= check(rep, "cog.n", n, 56)
    ok &= check(rep, "cog.L1", lv["L1"], {"indeterminate": 56})
    ok &= check(rep, "cog.L2", lv["L2"], {"admitted": 56})
    ok &= check(rep, "cog.L3", lv["L3"], {"refused": 55, "indeterminate": 1})
    ok &= check(rep, "cog.L3_first_failed", first["L3"], {"G1": 32, "G2": 10, "G3": 13})

    # ---- G8-augmented configuration: extended simulation evaluation
    out = os.path.join(G8, "out")
    holds = jsonl(os.path.join(out, "benchmark_09_single.jsonl")) + jsonl(os.path.join(out, "benchmark_09_variants.jsonl"))
    adm = inacc = acc = acc_adm = 0
    for row in holds:
        for est, cert in row["certs"].items():
            accurate = cert["rel_error"]["primary"] <= 0.25
            acc += accurate
            if cert["levels"]["L3"]["status"] == "admitted":
                adm += 1
                inacc += (not accurate)
                acc_adm += accurate
    ok &= check(rep, "g8.recordings", len(holds), 404)
    ok &= check(rep, "g8.outputs", 2 * len(holds), 808)
    ok &= check(rep, "g8.admitted", adm, 261)
    ok &= check(rep, "g8.inaccurate_admitted", inacc, 28)
    ok &= check(rep, "g8.accurate_outputs", acc, 339)
    ok &= check(rep, "g8.accurate_admitted", acc_adm, 233)
    missp = jsonl(os.path.join(out, "benchmark_09_missp.jsonl"))
    st = Counter(cert["levels"]["L3"]["status"] for row in missp for cert in row["certs"].values())
    inacc_m = sum(1 for row in missp for cert in row["certs"].values()
                  if cert["levels"]["L3"]["status"] == "admitted" and cert["rel_error"]["primary"] > 0.25)
    ok &= check(rep, "g8.misspecification_outputs", sum(st.values()), 200)
    ok &= check(rep, "g8.misspecification_L3", dict(st), {"admitted": 10, "refused": 187, "indeterminate": 3})
    ok &= check(rep, "g8.misspecification_inaccurate_admitted", inacc_m, 10)
    two = Counter(cert["levels"]["L3"]["status"] for row in missp if row["family"] == "two_timescales" for cert in row["certs"].values())
    ok &= check(rep, "g8.two_timescale_family", dict(two), {"refused": 40})
    twins = jsonl(os.path.join(out, "benchmark_09_twins.jsonl"))
    scored = [(cert["levels"]["L3"]["status"], cert["rel_error"]["primary"]) for row in twins for cert in row["certs"].values()
              if cert.get("rel_error", {}).get("primary") is not None]
    acc_t = [s for s, e in scored if e <= 0.25]
    ok &= check(rep, "g8.sleep_matched_recordings", len(twins), 320)
    ok &= check(rep, "g8.sleep_matched_outputs", 2 * len(twins), 640)
    ok &= check(rep, "g8.sleep_matched_scoreable", len(scored), 633)
    ok &= check(rep, "g8.sleep_matched_accurate", len(acc_t), 210)
    ok &= check(rep, "g8.sleep_matched_accurate_admitted", sum(1 for s in acc_t if s == "admitted"), 16)
    sleep = jsonl(os.path.join(out, "benchmark_10_sleep.jsonl"))
    sl = defaultdict(Counter)
    for row in sleep:
        sl[row["estimator"]][row["levels"]["L3"]["status"]] += 1
    ok &= check(rep, "g8.sleep_cells", len(sleep), 1204)
    ok &= check(rep, "g8.sleep_ols_L3", dict(sl["ols"]), {"admitted": 11, "refused": 583, "indeterminate": 8})
    ok &= check(rep, "g8.sleep_kalman_L3", dict(sl["kalman"]), {"admitted": 6, "refused": 590, "indeterminate": 6})
    co2 = [r for r in jsonl(os.path.join(out, "benchmark_10_co2.jsonl"))
           if r.get("prep") == "interp_calendar" and r["estimator"] == "ols"]
    ok &= check(rep, "g8.co2_calendar_ols_L3",
                dict(Counter(f"{r['levels']['L3']['status']}|{r['levels']['L3']['binding']}" for r in co2)),
                {"refused|G2": 4})
    na = json.load(open(os.path.join(G8, "noise_aware_bootstrap", "results.json"), encoding="utf-8"))
    ok &= check(rep, "g8.matched_ls_admissions", na["original_ndif"]["admitted"], 72)
    ok &= check(rep, "g8.matched_ls_ndif_inaccurate", na["original_ndif"]["inaccurate_admitted"], 5)
    comp = json.load(open(os.path.join(out, "benchmark_09b_comparator_results.json"), encoding="utf-8"))
    rep["g8.ml_comparator_results"] = comp

    # ---- landscape readouts: drift replacement, constant one-basin counter, separation threshold (Fig. 2b,c)
    with open(os.path.join(INPUTS, "drift_replacement_records.csv"), encoding="utf-8", newline="") as fh:
        recs = list(csv.DictReader(fh))
    awake = [r for r in recs if r["cohort"] != "sleep"]
    asleep = [r for r in recs if r["cohort"] == "sleep"]
    ok &= check(rep, "landscape.records", dict(Counter(r["cohort"] for r in recs)),
                {"ds003838": 64, "eegmmidb": 104, "meg": 83, "fmri": 198, "sleep": 197})
    ok &= check(rep, "landscape.awake_minusI_agree", sum(r["agree_minusI"] == "True" for r in awake), 445)
    ok &= check(rep, "landscape.awake_random_stable_agree", sum(r["agree_rand_all"] == "True" for r in awake), 445)
    ok &= check(rep, "landscape.sleep_minusI_agree", sum(r["agree_minusI"] == "True" for r in asleep), 165)
    ok &= check(rep, "landscape.sleep_random_stable_agree", sum(r["agree_rand_all"] == "True" for r in asleep), 131)
    ok &= check(rep, "landscape.constant_one_counter_awake", sum(r["n_stable_anchor"] == "1" for r in awake), 446)
    ok &= check(rep, "landscape.constant_one_counter_all", sum(r["n_stable_anchor"] == "1" for r in recs), 607)
    thr = json.load(open(os.path.join(INPUTS, "separation_threshold_summary.json"), encoding="utf-8"))
    ok &= check(rep, "landscape.threshold_best_agree", thr["best_agree"], 623)
    ok &= check(rep, "landscape.threshold_reference_agree", thr["agree_at_reference"], 557)

    rep["status"] = "PASS" if ok else "FAIL"
    os.makedirs(os.path.dirname(a.output), exist_ok=True)
    with open(a.output, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(rep, fh, indent=1, ensure_ascii=False)
    for k, v in rep.items():
        if isinstance(v, dict) and "ok" in v:
            print(f"{'ok ' if v['ok'] else '!! '} {k}: {v['value']}" + ("" if v["ok"] else f"  (expected {v['expected']})"))
    print(rep["status"])
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
