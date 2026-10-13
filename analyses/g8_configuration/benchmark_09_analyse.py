"""Benchmark 09, analysis of the extended simulation evaluation (G8-augmented configuration).

Reads out/benchmark_09_<part>.jsonl and writes out/benchmark_09_results.json.

Scoring: an estimate is accurate when its relative error with respect to the reference
rate is below 25%; refused and indeterminate certificates are pooled as not admitted.  Intervals of proportions are
percentiles of 10,000 bootstrap resamples of recordings (both estimates of a recording kept together); for zero
counts the exact (Clopper-Pearson) interval is given instead.  The remove-one-check analysis re-scores the stored
check outcomes with one deciding check of level 3 omitted.

    python analyses/g8_configuration/benchmark_09_analyse.py
"""
from __future__ import annotations

import json
import os
from collections import Counter, defaultdict

import numpy as np
from scipy.stats import beta, norm

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
TOL = 0.25
SEED = 20261008
L3 = ["G0", "G2", "G8", "G1", "G3"]
ESTS = ("ols", "kalman")


SMOKE = bool(os.environ.get("NDIF_SMOKE"))


def read(part):
    p = os.path.join(OUT, f"benchmark_09_{part}{'_smoke' if SMOKE else ''}.jsonl")
    if not os.path.exists(p):
        return []
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def status(gates, order=L3):
    ps = []
    for g in order:
        if g not in gates:
            ps.append("missing")
            continue
        p = gates[g]["passed"]
        ps.append(None if p is None else bool(p))
    if any(p == "missing" for p in ps):
        return "indeterminate"
    if any(p is False for p in ps):
        return "refused"
    if any(p is None for p in ps):
        return "indeterminate"
    return "admitted"


def cp(k, n):
    lo = 0.0 if k == 0 else float(beta.ppf(0.025, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(0.975, k + 1, n - k))
    return [lo, hi]


def table(recs, target="primary", ests=ESTS, order=L3, seed=SEED):
    """Counts and bootstrap intervals over recordings."""
    per = []
    for r in recs:
        items = []
        for e in ests:
            c = r["certs"].get(e)
            if c is None:
                continue
            rel = (c.get("rel_error") or {}).get(target)
            if rel is None:
                continue
            st = status(c["gates"], order)
            items.append((st, rel < TOL))
        per.append(items)
    cnt = Counter()
    for items in per:
        for st, acc in items:
            cnt["n"] += 1
            cnt["admitted"] += st == "admitted"
            cnt["indeterminate"] += st == "indeterminate"
            cnt["accurate"] += acc
            cnt["false_pass"] += st == "admitted" and not acc
            cnt["true_pass"] += st == "admitted" and acc
            cnt["false_rejection"] += st != "admitted" and acc
            cnt["true_rejection"] += st != "admitted" and not acc
    out = dict(counts=dict(cnt))
    if not per or cnt["n"] == 0:
        return out
    rng = np.random.default_rng(seed)
    R = len(per)
    adm = np.array([sum(st == "admitted" for st, _ in it) for it in per])
    fp = np.array([sum(st == "admitted" and not a for st, a in it) for it in per])
    acc = np.array([sum(a for _, a in it) for it in per])
    fr = np.array([sum(st != "admitted" and a for st, a in it) for it in per])
    agree = np.array([sum((st == "admitted") == a for st, a in it) for it in per])
    n = np.array([len(it) for it in per])
    idx = rng.integers(0, R, size=(10000, R))

    def prop(num, den, k_obs, n_obs):
        if n_obs == 0:
            return None
        v = k_obs / n_obs
        if k_obs == 0:
            return dict(value=0.0, lo=0.0, hi=cp(0, n_obs)[1], interval="exact")
        b = num[idx].sum(1) / np.maximum(den[idx].sum(1), 1)
        return dict(value=float(v), lo=float(np.percentile(b, 2.5)), hi=float(np.percentile(b, 97.5)),
                    interval="bootstrap")

    out["inaccurate_among_admitted"] = prop(fp, adm, cnt["false_pass"], cnt["admitted"])
    out["accurate_not_admitted"] = prop(fr, acc, cnt["false_rejection"], cnt["accurate"])
    out["agreement"] = float(agree.sum() / n.sum())
    out["agreement_admit_none"] = float((cnt["n"] - cnt["accurate"]) / cnt["n"])
    return out


def by(recs, keyf, **kw):
    g = defaultdict(list)
    for r in recs:
        g[keyf(r)].append(r)
    return {k: table(v, **kw) for k, v in sorted(g.items())}


def binding(recs, ests=ESTS):
    c = Counter()
    for r in recs:
        for e in ests:
            if e in r["certs"]:
                x = r["certs"][e]["levels"]["L3"]
                c[f"{e}|{x['status']}|{x['binding']}"] += 1
    return dict(sorted(c.items()))


def remove_one(recs, target="primary"):
    out = {}
    for chk in L3:
        order = [g for g in L3 if g != chk]
        removed_inacc = refused_acc = 0
        for r in recs:
            for e in ESTS:
                c = r["certs"].get(e)
                if c is None:
                    continue
                rel = (c.get("rel_error") or {}).get(target)
                if rel is None:
                    continue
                full = status(c["gates"]) == "admitted"
                wo = status(c["gates"], order) == "admitted"
                if wo and not full:
                    removed_inacc += rel >= TOL
                    refused_acc += rel < TOL
        out[chk] = dict(inaccurate_admissions_removed=int(removed_inacc), accurate_estimates_refused=int(refused_acc))
    return out


def recommendation(recs):
    """recommend_estimator: the least costly admitted estimator (least squares before maximum likelihood)."""
    n = rec = inacc = 0
    for r in recs:
        n += 1
        for e in ESTS:
            c = r["certs"][e]
            if status(c["gates"]) == "admitted":
                rec += 1
                rel = c["rel_error"].get("primary")
                inacc += rel is not None and rel >= TOL
                break
    return dict(recordings=n, recommended=rec, inaccurate=inacc)


def stability(stab, base_parts):
    base = {}
    for p in base_parts:
        for r in read(p):
            base[r["key"]] = r
    same = Counter()
    n = Counter()
    adm_n = adm_keep = 0
    changes = Counter()
    per_rec = defaultdict(lambda: [0, 0])
    for r in stab:
        k0 = r["key"].rsplit("|s", 1)[0]
        b = base.get(k0)
        if b is None:
            continue
        for e in ESTS:
            s0 = status(b["certs"][e]["gates"])
            s1 = status(r["certs"][e]["gates"])
            n[e] += 1
            same[e] += s0 == s1
            per_rec[k0][0] += s0 == s1
            per_rec[k0][1] += 1
            if s0 == "admitted":
                adm_n += 1
                adm_keep += s1 == "admitted"
            if s0 != s1:
                changes[f"{e}|{s0}->{s1}"] += 1
    keys = sorted(per_rec)
    rng = np.random.default_rng(SEED + 1)
    a = np.array([per_rec[k][0] for k in keys], float)
    m = np.array([per_rec[k][1] for k in keys], float)
    if len(keys):
        idx = rng.integers(0, len(keys), size=(10000, len(keys)))
        b = a[idx].sum(1) / m[idx].sum(1)
        ci = [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
    else:
        ci = None
    tot = sum(n.values())
    return dict(n=tot, identical=int(sum(same.values())), identical_share=(sum(same.values()) / tot if tot else None),
                ci95=ci, by_estimator={e: f"{same[e]}/{n[e]}" for e in n}, admissions=f"{adm_keep}/{adm_n}",
                changes=dict(sorted(changes.items())))


def mdc_power(single, mdc):
    truth = {(m["cell"], m["estimator"]): m for m in mdc}
    allp, admp = [], []
    rows = defaultdict(list)
    for r in single:
        for e in ESTS:
            m = truth.get((r["cell"], e))
            c = r["certs"][e]
            rp = c["gates"]["G3"].get("mdc")
            if m is None or rp is None or not np.isfinite(rp) or abs(m["slope"]) < 1e-6:
                continue
            z = abs(m["slope"]) * np.log(rp) / (np.sqrt(2) * m["sd0"])
            p = float(norm.cdf(z - 1.96) + norm.cdf(-z - 1.96))
            allp.append(p)
            rows[(r["a"], r["nu"], r["geom"], e)].append(p)
            if status(c["gates"]) == "admitted":
                admp.append(p)
    return dict(n=len(allp), median_power=(float(np.median(allp)) if allp else None),
                admitted_n=len(admp), admitted_median_power=(float(np.median(admp)) if admp else None),
                share_below_0_70=(float(np.mean(np.array(allp) < 0.7)) if allp else None),
                by_condition={"|".join(map(str, k)): float(np.median(v)) for k, v in sorted(rows.items())})


def main():
    single, variants, missp, neg = read("single"), read("variants"), read("missp"), read("neg")
    twins, stab, abc, mdc = read("twins"), read("stab"), read("abc"), read("mdc")
    holds = single + variants
    res = dict(n=dict(single=len(single), variants=len(variants), missp=len(missp), neg=len(neg), twins=len(twins),
                      stab=len(stab), abc=len(abc), mdc=len(mdc)))
    res["model_holds"] = dict(pooled=table(holds), ols=table(holds, ests=("ols",)), kalman=table(holds, ests=("kalman",)))
    res["single_by_condition"] = by(single, lambda r: f"a={r['a']}|nu={r['nu']}|{r['geom']}")
    res["single_by_persistence"] = by(single, lambda r: f"a={r['a']}")
    res["single_by_size"] = by(single, lambda r: r["geom"])
    res["single_by_noise"] = by(single, lambda r: f"nu={r['nu']}")
    res["variants_by_family"] = by(variants, lambda r: r["family"])
    res["misspecified"] = dict(pooled=table(missp), ols=table(missp, ests=("ols",)), kalman=table(missp, ests=("kalman",)),
                               by_family=by(missp, lambda r: f"{r['family']}|{r['geom']}"),
                               lag1=by([r for r in missp if "lag1" in r["targets"]], lambda r: r["family"], target="lag1"))
    res["negative"] = dict(admitted=sum(status(r["certs"][e]["gates"]) == "admitted" for r in neg for e in ESTS),
                           estimates=2 * len(neg),
                           by_family={f: sum(status(r["certs"][e]["gates"]) == "admitted" for r in neg
                                             if r["family"] == f for e in ESTS) for f in ("white_noise", "random_walk")})
    res["negative"]["ci95"] = cp(res["negative"]["admitted"], max(res["negative"]["estimates"], 1))
    res["twins"] = dict(pooled=table(twins), ols=table(twins, ests=("ols",)), kalman=table(twins, ests=("kalman",)),
                        by_stage=by(twins, lambda r: r["source"].split("/")[1]))
    res["binding"] = dict(holds=binding(holds), missp=binding(missp), twins=binding(twins), neg=binding(neg))
    res["remove_one_check"] = dict(holds=remove_one(holds), missp=remove_one(missp), twins=remove_one(twins))
    res["recommendation"] = dict(holds=recommendation(holds), missp=recommendation(missp))
    res["stability"] = stability(stab, ("single", "missp", "variants"))
    res["mdc"] = mdc_power(single, mdc) if mdc else None
    if abc:
        base = {r["key"]: r for r in single + missp}
        a_holds = [r for r in abc if r["family"] == "single"]
        a_missp = [r for r in abc if r["family"] != "single"]
        same_rec = [base[r["key"]] for r in a_holds if r["key"] in base]
        res["abc"] = dict(holds=table(a_holds, ests=("abc",)), missp=table(a_missp, ests=("abc",)),
                          holds_ml_same_recordings=table(same_rec, ests=("kalman",)),
                          holds_ls_same_recordings=table(same_rec, ests=("ols",)),
                          binding=binding(abc, ests=("abc",)),
                          median_abs_log_error=dict(
                              abc=float(np.median([abs(np.log(r["certs"]["abc"]["lam"] / r["targets"]["primary"]))
                                                   for r in a_holds if r["certs"]["abc"]["lam"]])),
                              kalman=float(np.median([abs(np.log(b["certs"]["kalman"]["lam"] / b["targets"]["primary"]))
                                                      for b in same_rec if b["certs"]["kalman"]["lam"]]))),
                          wall_s_median=float(np.median([r["certs"]["abc"]["wall_s"] for r in abc])))
    walls = defaultdict(list)
    for r in holds + missp:
        for e in ESTS:
            walls[e].append(r["certs"][e]["wall_s"])
    res["wall_s_median"] = {e: float(np.median(v)) for e, v in walls.items()}
    json.dump(res, open(os.path.join(OUT, f"benchmark_09_results{'_smoke' if SMOKE else ''}.json"), "w",
                        encoding="utf-8"), indent=1)
    h = res["model_holds"]["pooled"]
    print("model holds:", h["counts"], h.get("inaccurate_among_admitted"))
    print("misspecified:", res["misspecified"]["pooled"]["counts"])
    print("negative:", res["negative"])
    print("twins:", res["twins"]["pooled"]["counts"])
    print("stability:", res["stability"])
    print("mdc:", {k: v for k, v in (res["mdc"] or {}).items() if k != "by_condition"})


if __name__ == "__main__":
    main()
