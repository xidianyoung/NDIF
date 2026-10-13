"""Summaries of the real recordings certified with the G8-augmented configuration (benchmark_10_real_recordings.py).

Output: out/benchmark_10_results.json.  Every number the article reports about the real recordings under the G8-augmented
configuration is read from this file.
"""
from __future__ import annotations

import collections
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))   # the repository root
import ndif                                                       # noqa: E402
OUT = os.path.join(HERE, "out")
STAGES = ["Wake", "N1", "N2", "REM", "N3", "N4"]
PAIRS = [("Wake", "N1"), ("N1", "N2"), ("N2", "N3"), ("N3", "N4")]
ENTROPY = 202610090200


def rows(part):
    p = os.path.join(OUT, f"benchmark_10_{part}.jsonl")
    if not os.path.exists(p):
        return []
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def counts(rs, level):
    c = collections.Counter(r["levels"][level]["status"] for r in rs)
    b = collections.Counter(f"{r['levels'][level]['status']}|{r['levels'][level]['binding']}" for r in rs
                            if r["levels"][level]["status"] != "admitted")
    return dict(n=len(rs), **{k: int(v) for k, v in c.items()}, binding={k: int(v) for k, v in sorted(b.items())})


def q(v, p):
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], float)
    return float(np.percentile(v, p)) if v.size else None


def g3_margin(r):
    """Smallest distance of the recovery check's three statistics from their criteria (0.20 each)."""
    d = (r["gates"].get("G3") or {}).get("detail") or {}
    s, o, sp = d.get("rate_recovery"), d.get("bias_log"), d.get("spread_log")
    if s is None or o is None or sp is None:
        return None
    return float(min(0.20 - abs(s - 1.0), 0.20 - abs(o), 0.20 - sp))


def kappa(x, y):
    x, y = np.asarray(x, bool), np.asarray(y, bool)
    po = np.mean(x == y)
    pe = np.mean(x) * np.mean(y) + np.mean(~x) * np.mean(~y)
    return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")


def sleep_summary():
    rs = rows("sleep")
    out = dict(n_certificates=len(rs))
    by_est = collections.defaultdict(list)
    for r in rs:
        by_est[r["estimator"]].append(r)
    for est, rr in by_est.items():
        e = dict(L2=counts(rr, "L2"), L3=counts(rr, "L3"))
        if est == "ols":
            e["L1"] = counts(rr, "L1")
        adm2 = [r for r in rr if r["levels"]["L2"]["status"] == "admitted"]
        mv = [((r["gates"].get("G3") or {}).get("detail") or {}).get("detection_margin_fold_variance") for r in adm2]
        e["L2_mdc_var"] = dict(n=len(adm2), median=q(mv, 50), q25=q(mv, 25), q75=q(mv, 75))
        e["L2_by_stage"] = {s: counts([r for r in rr if r["stage"] == s], "L2") for s in STAGES}
        e["L3_by_stage"] = {s: counts([r for r in rr if r["stage"] == s], "L3") for s in STAGES}
        adm = [r for r in rr if r["levels"]["L3"]["status"] == "admitted"]
        e["L3_admitted"] = dict(
            n=len(adm), participants=len({r["participant"] for r in adm}),
            by_stage={s: dict(n=len([r for r in adm if r["stage"] == s]),
                              median_rate=q([r["lam"] for r in adm if r["stage"] == s], 50)) for s in STAGES},
            mdc_rate=dict(median=q([((r["gates"]["G3"].get("detail") or {}).get("detection_margin_fold")) for r in adm], 50),
                          q25=q([((r["gates"]["G3"].get("detail") or {}).get("detection_margin_fold")) for r in adm], 25),
                          q75=q([((r["gates"]["G3"].get("detail") or {}).get("detection_margin_fold")) for r in adm], 75)),
            within_005_of_recovery_criterion=int(sum(1 for r in adm if (g3_margin(r) is not None and g3_margin(r) < 0.05))),
            recovery_margin=[g3_margin(r) for r in adm])
        both = {}
        for lo, hi in PAIRS:
            parts_lo = {r["participant"] for r in adm if r["stage"] == lo}
            parts_hi = {r["participant"] for r in adm if r["stage"] == hi}
            both[f"{lo}-{hi}"] = len(parts_lo & parts_hi)
        e["participants_admitted_in_both_stages"] = both
        # night-to-night agreement of the level-3 admission (same participant and stage, nights 1 and 2)
        idx = {(r["participant"], r["night"], r["stage"]): r["levels"]["L3"]["status"] == "admitted" for r in rr}
        pairs = [(p, s) for (p, n, s) in idx if n == "1" and (p, "2", s) in idx]
        x = [idx[(p, "1", s)] for p, s in pairs]
        y = [idx[(p, "2", s)] for p, s in pairs]
        kap = kappa(x, y) if pairs else None
        rng = np.random.default_rng(ENTROPY)
        parts = sorted({p for p, _ in pairs})
        bs = []
        for _ in range(10000):
            pick = rng.choice(parts, len(parts), replace=True)
            xs, ys = [], []
            for p in pick:
                for (pp, s) in pairs:
                    if pp == p:
                        xs.append(idx[(p, "1", s)])
                        ys.append(idx[(p, "2", s)])
            k = kappa(xs, ys)
            if np.isfinite(k):
                bs.append(k)
        e["nights"] = dict(n_pairs=len(pairs), n_participants=len(parts), admitted_night1=int(np.sum(x)),
                           admitted_night2=int(np.sum(y)), admitted_both=int(np.sum(np.array(x) & np.array(y))),
                           kappa=kap, ci95=[q(bs, 2.5), q(bs, 97.5)] if bs else None,
                           n_boot_finite=len(bs))
        out[est] = e
    return out


def pairs_summary():
    rs = rows("pairs")
    c = collections.Counter("pass" if r["passed"] == 1 else ("fail" if r["passed"] == 0 else "not evaluable")
                            for r in rs)
    by = {f"{lo}-{hi}": dict(collections.Counter("pass" if r["passed"] == 1 else ("fail" if r["passed"] == 0 else "ne")
                                                 for r in rs if (r["lo"], r["hi"]) == (lo, hi))) for lo, hi in PAIRS}
    notes = collections.Counter(r["note"].split(":")[0][:60] for r in rs if r["passed"] is None)
    return dict(n=len(rs), **{k: int(v) for k, v in c.items()}, by_pair=by, not_evaluable_reasons=dict(notes))


def series_summary():
    rs = rows("series")
    return {f"{r['recording']}|{r['estimator']}": dict(L1=r["levels"]["L1"], L2=r["levels"]["L2"], L3=r["levels"]["L3"],
                                                        lam=r["lam"]) for r in rs}


def co2_summary():
    rs = rows("co2")
    out = {}
    for r in sorted(rs, key=lambda r: (r["prep"], r["block"], r["estimator"])):
        g3 = (r["gates"].get("G3") or {}).get("detail") or {}
        g2 = r["gates"].get("G2") or {}
        g8 = (r["gates"].get("G8") or {}).get("detail") or {}
        a, nu = r["a_ml_segments"], r["nu_ml_segments"]
        pred = None
        if 0 < a < 1 and 0 <= nu < 1:
            pred = float(np.log(a) / np.log((1 - nu) * a))
        out[f"{r['prep']}|{r['block']}|{r['estimator']}"] = dict(
            L3=r["levels"]["L3"]["status"], binding=r["levels"]["L3"]["binding"], L2=r["levels"]["L2"]["status"],
            slope=g3.get("rate_recovery"), offset=g3.get("bias_log"), spread=g3.get("spread_log"),
            g2_value=g2.get("value"), g2_passed=g2.get("passed"), g8_p=g8.get("combined_p"),
            g8_passed=(r["gates"].get("G8") or {}).get("passed"), g1_passed=(r["gates"].get("G1") or {}).get("passed"),
            tau_ref_weeks=r["tau_ref_weeks"], nu_ref=r["nu_ref"], block_over_tau_ref=r["block_over_tau_ref"],
            a_ml_segments=a, nu_ml_segments=nu, ls_attenuation_predicted=pred, lam=r["lam"])
    return out


def main():
    res = dict(version=ndif.__version__, sleep=sleep_summary(), pairs=pairs_summary(), series=series_summary(),
               co2=co2_summary())
    json.dump(res, open(os.path.join(OUT, "benchmark_10_results.json"), "w", encoding="utf-8"), indent=1)
    s = res["sleep"]
    for est in ("ols", "kalman"):
        if est in s:
            print(est, "L2", s[est]["L2"], "\n   L3", s[est]["L3"], "\n   admitted", {k: v for k, v in s[est]["L3_admitted"].items() if k != "recovery_margin"},
                  "\n   nights", s[est]["nights"], "\n   both", s[est]["participants_admitted_in_both_stages"])
    print("pairs", res["pairs"])


if __name__ == "__main__":
    main()
