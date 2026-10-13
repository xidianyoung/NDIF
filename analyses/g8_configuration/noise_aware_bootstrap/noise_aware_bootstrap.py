"""Noise-aware least-squares parametric bootstrap on the 394 paired records of the extended evaluation.

A post hoc, descriptive comparison; the analysis settings are in protocol.json.

    python analyses/g8_configuration/noise_aware_bootstrap/noise_aware_bootstrap.py plan    # write protocol.json if absent

    python analyses/g8_configuration/noise_aware_bootstrap/noise_aware_bootstrap.py test    # unit tests
    python analyses/g8_configuration/noise_aware_bootstrap/noise_aware_bootstrap.py pilot   # time the first eight records
    python analyses/g8_configuration/noise_aware_bootstrap/noise_aware_bootstrap.py run     # the full analysis
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import platform
from pathlib import Path
import re
import sys
import time
import unittest

sys.dont_write_bytecode = True
import numpy as np

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parents[2]                                   # the repository root
SOURCE = HERE.parent / "out"
sys.path.insert(0, str(PACKAGE))
from ndif.estimators import simulate_eq6_padded, fit_rates_padded

SEED = 20261009
N_BOOT = 200
N_CI = 5000


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_protocol():
    p = HERE / "protocol.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    protocol = dict(
        status="post hoc analysis; settings fixed before the full run",
        scope="The 394 equal-segment paired records, including 10 Student-t records; the unequal-length records have no comparator.",
        bootstrap_replicates=N_BOOT, seed=SEED,
        random_stream="Independent deterministic per-record SeedSequence [20261009, first 32 SHA256 bits of record_id]",
        observed_generator="The stored G3 a_fit and nu_fit: the Kalman maximum-likelihood fit of a latent stationary Gaussian AR(1) plus additive white observation noise to the observed recording, before the G3 indirect-inference correction. The true parameters and the corrected generator are not used. Stationary latent variance 1 is scale-normalized; OLS persistence is scale invariant.",
        primary_score="abs(mean(log(lambda_OLS_boot)) - log(-log(a_fit)/dt)), dt=1. This estimates OLS log-rate bias relative to the fitted latent rate, including observation-noise attenuation. It is not a posterior error probability.",
        secondary_scores=["abs(log(lambda_OLS_raw) - mean(log(lambda_OLS_boot))): plug-in self-consistency, no latent-target protection", "abs((tau_OLS_raw - mean(tau_OLS_boot))/mean(tau_OLS_boot)): original-style timescale self-consistency with noise-aware generator", "97.5th minus 2.5th percentile of bootstrap log-rate: uncertainty width"],
        validity_rule="All 394 records stay in the denominator. An invalid observed OLS rate, a missing or invalid generator (0<a_fit<1 and 0<=nu_fit<1), an invalid geometry, any nonpositive or nonfinite bootstrap rate, or a simulation exception gives every new score +infinity (ineligible). Each record is simulated once.",
        ranking="Ascending score; exact ties ordered by record_id; only finite scores are eligible. The comparator admits as many records as NDIF admitted (72; the NDIF admission mask is fixed). With fewer than 72 eligible records, matched coverage is reported as insufficient.",
        truth_evaluation="After all scores and masks are fixed, a record is inaccurate when its stored relative lambda error is >=0.25. The log-rate ranking score and the relative-lambda-error evaluation are distinct quantities.",
        curves="Every eligible rank k=1..eligible_count, coverage k/394; record_id breaks score ties. NDIF appears as its one fixed observed operating point.",
        uncertainty="5000 paired condition-stratified cluster resamples of recording IDs within the original cell. Both methods use identical draws, and the comparator coverage is re-matched to the number of NDIF admissions in each draw. Curve intervals are nominal pointwise percentile intervals at each k, not simultaneous bands. The saved per-record scores are reused; intervals are conditional on the fitted generators and the Monte Carlo draws, do not propagate parameter-fit or Monte Carlo uncertainty, and cover the simulated families only.",
        interpretation="Noise-aware plug-in parametric comparator: no Bayesian noise/scale marginalization, no joint nuisance-parameter bootstrap, no separate confirmation dataset; 200 replicates per record limit Monte Carlo precision.",
    )
    p.write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8")
    return protocol


def inputs():
    comparator = {r["key"]: r for r in read_jsonl(SOURCE / "benchmark_09b_comparator.jsonl")}
    certs = {f"{part}|{r['cell']}|{r['rep']}": r for part in ("single", "variants")
             for r in read_jsonl(SOURCE / f"benchmark_09_{part}.jsonl")}
    assert len(comparator) == 394 and len(certs) == 404 and set(comparator) <= set(certs)
    missing = set(certs) - set(comparator)
    assert len(missing) == 10 and all(certs[k]["family"] == "unequal" for k in missing)
    rows = [(k, certs[k]) for k in sorted(comparator)]
    assert sum(r["certs"]["ols"]["levels"]["L3"]["status"] == "admitted" for _, r in rows) == 72
    return rows


def observed_parameters(record_id, row):
    # Deliberately whitelist only observed-fit parameters and recording geometry.
    cert = row["certs"]["ols"]
    g = cert["gates"].get("G3", {})
    m = re.fullmatch(r"g(\d+)x(\d+)", row.get("geom", ""))
    return dict(record_id=record_id, raw_rate=cert.get("lam"), a_fit=g.get("a_fit"),
                nu_fit=g.get("nu_fit"), lens=[int(m[2])] * int(m[1]) if m else None)


def compute_one(observed):
    start = time.perf_counter()
    out = dict(record_id=observed["record_id"], status="failed", failure=None, n_boot=N_BOOT,
               n_valid=0, score_latent_bias=None, score_log_self_consistency=None,
               score_tau_self_consistency=None, score_log_width=None)
    values = np.full(N_BOOT, np.nan)
    try:
        a, nu, raw, lens = (observed[k] for k in ("a_fit", "nu_fit", "raw_rate", "lens"))
        if not all(isinstance(x, (int, float)) and np.isfinite(x) for x in (a, nu, raw)):
            raise ValueError("missing or nonfinite observed fit")
        if not (0 < a < 1 and 0 <= nu < 1 and raw > 0):
            raise ValueError("invalid observed fit domain")
        if not lens or any(L < 3 for L in lens):
            raise ValueError("invalid recording geometry")
        stream = int(hashlib.sha256(observed["record_id"].encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(np.random.SeedSequence([SEED, stream]))
        Y = simulate_eq6_padded(a, nu, lens, N_BOOT, rng)
        values = fit_rates_padded(Y, lens, dt=1.0, method="ols")
        valid = np.isfinite(values) & (values > 0)
        out["n_valid"] = int(valid.sum())
        if not valid.all():
            raise ValueError("at least one invalid OLS bootstrap rate")
        logv = np.log(values)
        fit_rate = -np.log(a)
        out.update(status="ok", lambda_latent_fit=float(fit_rate),
                   mean_log_rate=float(logv.mean()),
                   score_latent_bias=float(abs(logv.mean() - np.log(fit_rate))),
                   score_log_self_consistency=float(abs(np.log(raw) - logv.mean())),
                   score_tau_self_consistency=float(abs((1 / raw - np.mean(1 / values)) / np.mean(1 / values))),
                   score_log_width=float(np.quantile(logv, .975) - np.quantile(logv, .025)))
    except Exception as e:
        out["failure"] = f"{type(e).__name__}: {e}"
    out["wall_s"] = time.perf_counter() - start
    return out, values


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def metrics(mask, wrong):
    k = int(mask.sum())
    w = int(np.sum(mask & wrong))
    return dict(n=len(mask), admitted=k, inaccurate_admitted=w,
                coverage=k / len(mask), risk=w / k if k else None)


def ranked_mask(score, count, keys):
    eligible = np.flatnonzero(np.isfinite(score))
    order = eligible[np.lexsort((keys[eligible], score[eligible]))]
    m = np.zeros(len(score), bool)
    m[order[:count]] = True
    return m


def summarize(rows, fitted, all_boot):
    # Truth labels enter only here, after all simulations and scores are complete.
    keys = np.array([key for key, _ in rows])
    cell = np.array([r["cell"] for _, r in rows])
    wrong = np.array([r["certs"]["ols"]["rel_error"]["primary"] >= .25 for _, r in rows])
    original = np.array([r["certs"]["ols"]["levels"]["L3"]["status"] == "admitted" for _, r in rows])
    k = int(original.sum())
    groups = [np.flatnonzero(cell == v) for v in sorted(set(cell))]
    rng = np.random.default_rng(SEED)
    draws = np.stack([np.concatenate([rng.choice(g, len(g), replace=True) for g in groups]) for _ in range(N_CI)])
    oldk = original[draws].sum(axis=1)
    oldw = (original[draws] & wrong[draws]).sum(axis=1)
    paired = []
    curves = []
    methods = {}
    score_names = ("score_latent_bias", "score_log_self_consistency", "score_tau_self_consistency", "score_log_width")
    scores = {name: np.array([r[name] if r[name] is not None else np.inf for r in fitted]) for name in score_names}
    masks = {name: ranked_mask(s, k, keys) for name, s in scores.items()}
    for i, ((key, row), fit) in enumerate(zip(rows, fitted)):
        paired.append(dict(record_id=key, cell=int(cell[i]), family=row["family"], geom=row["geom"],
                           a_fit=observed_parameters(key, row)["a_fit"], nu_fit=observed_parameters(key, row)["nu_fit"],
                           raw_ols_rate=row["certs"]["ols"]["lam"],
                           relative_lambda_error=row["certs"]["ols"]["rel_error"]["primary"],
                           inaccurate=bool(wrong[i]), original_ndif_admitted=bool(original[i]),
                           **{n: fit[n] for n in score_names}, **{f"matched_{n}": bool(masks[n][i]) for n in score_names},
                           status=fit["status"], failure=fit["failure"], n_valid=fit["n_valid"], wall_s=fit["wall_s"]))
    for name, score in scores.items():
        eligible = np.flatnonzero(np.isfinite(score))
        order = eligible[np.lexsort((keys[eligible], score[eligible]))]
        cumulative = np.cumsum(wrong[order])
        risk_draw = np.full((N_CI, len(order)), np.nan)
        matched_differences = []
        insufficient = 0
        for b, idx in enumerate(draws):
            si = score[idx]
            ei = np.flatnonzero(np.isfinite(si))
            oi = ei[np.lexsort((keys[idx][ei], si[ei]))]
            cum = np.cumsum(wrong[idx][oi])
            cap = min(len(cum), len(order))
            risk_draw[b, :cap] = cum[:cap] / np.arange(1, cap + 1)
            kb = int(oldk[b])
            if kb and len(cum) >= kb:
                matched_differences.append(float((cum[kb - 1] - oldw[b]) / kb))
            else:
                insufficient += 1
        ci = np.nanquantile(risk_draw, [.025, .975], axis=0)
        for j, idx in enumerate(order):
            curves.append(dict(method=name, n=394, admitted=j + 1, coverage=(j + 1) / 394,
                               inaccurate_admitted=int(cumulative[j]), risk=float(cumulative[j] / (j + 1)),
                               score_cutoff=float(score[idx]), cutoff_record_id=str(keys[idx]),
                               pointwise_ci95_low=float(ci[0, j]), pointwise_ci95_high=float(ci[1, j]),
                               ci_valid_draws=int(np.isfinite(risk_draw[:, j]).sum())))
        matched = metrics(masks[name], wrong)
        adequate = len(eligible) >= k
        methods[name] = dict(eligible=len(eligible), failed=394 - len(eligible),
                             sufficient_for_matched_coverage=adequate, matched=matched,
                             excess_risk_vs_ndif=(matched["risk"] - metrics(original, wrong)["risk"]) if adequate else None,
                             paired_stratified_ci95=np.quantile(matched_differences, [.025, .975]).tolist() if matched_differences else None,
                             ci_valid_draws=len(matched_differences), ci_insufficient_draws=insufficient,
                             cutoff=float(np.max(score[masks[name]])) if masks[name].any() else None)
    summary = dict(status="completed; post hoc descriptive comparison",
                   completed_at_utc=datetime.now(timezone.utc).isoformat(), n=394, expected_single_model=404,
                   absent_comparator="10 unequal-segment records; no new denominator exclusions",
                   original_ndif=metrics(original, wrong), methods=methods,
                   failed_records=[r["record_id"] for r in fitted if r["status"] != "ok"],
                   failure_counts={reason: sum(r["failure"] == reason for r in fitted) for reason in sorted({r["failure"] for r in fitted if r["failure"]})},
                   bootstrap_replicates=N_BOOT, seed=SEED, paired_ci_draws=N_CI,
                   simulation_wall_sum_s=sum(r["wall_s"] for r in fitted),
                   environment=dict(python=sys.version, numpy=np.__version__, platform=platform.platform(),
                                    processor=platform.processor()),
                   interpretation="Score uses log-rate bias relative to fitted latent generator; accuracy is evaluated against stored truth using relative lambda error >=25%. CI is conditional on this fixed condition design, fitted generator, and saved Monte Carlo scores; neither fit uncertainty nor Monte Carlo uncertainty is propagated. Secondary self-consistency and width scores are descriptive.")
    write_csv(HERE / "paired_records.csv", paired)
    write_csv(HERE / "risk_coverage_curves.csv", curves)
    np.savez_compressed(HERE / "bootstrap_rates.npz", record_ids=keys, rates=all_boot)
    (HERE / "results.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 5))
    labels = dict(score_latent_bias="Noise-aware OLS: latent-target bias", score_log_self_consistency="Noise-aware OLS: log self-consistency", score_tau_self_consistency="Noise-aware OLS: timescale self-consistency", score_log_width="Noise-aware OLS: log-rate interval width")
    for name in score_names:
        rs = [r for r in curves if r["method"] == name]
        ax.plot([r["coverage"] for r in rs], [r["risk"] for r in rs], label=labels[name])
        if name == "score_latent_bias":
            ax.fill_between([r["coverage"] for r in rs], [r["pointwise_ci95_low"] for r in rs], [r["pointwise_ci95_high"] for r in rs], alpha=.15)
    old = summary["original_ndif"]
    ax.scatter([old["coverage"]], [old["risk"]], marker="*", s=110, color="black", label="Fixed original NDIF (72/394)", zorder=5)
    ax.set(xlabel="Coverage (admitted / 394 recordings)", ylabel="Risk (relative rate error ≥25% among admitted)", ylim=(0, 1), xlim=(0, 1))
    ax.legend(fontsize=7, loc="upper left")
    ax.set_title("Post-hoc noise-aware OLS comparison; 200 replicates / recording", fontsize=10)
    fig.tight_layout()
    fig.savefig(HERE / "risk_coverage.png", dpi=200)
    fig.savefig(HERE / "risk_coverage.pdf")
    plt.close(fig)
    return summary


class Tests(unittest.TestCase):
    def test_truth_not_in_parameter_whitelist(self):
        row = dict(geom="g3x20", certs={"ols": dict(lam=.3, gates={"G3":dict(a_fit=.8, nu_fit=.2)})}, a=.1, nu=.9, targets={"primary":9})
        one = observed_parameters("unit", row)
        row.update(a=.9, nu=0, targets={"primary":.01})
        self.assertEqual(one, observed_parameters("unit", row))
        self.assertEqual(set(one), {"record_id", "raw_rate", "a_fit", "nu_fit", "lens"})

    def test_invalid_generator_is_failed_not_excluded(self):
        result, rates = compute_one(dict(record_id="invalid", raw_rate=.3, a_fit=.8, nu_fit=1., lens=[20]*3))
        self.assertEqual(result["status"], "failed")
        self.assertIsNone(result["score_latent_bias"])
        self.assertEqual(len(rates), N_BOOT)

    def test_ties_and_failures(self):
        m = ranked_mask(np.array([.1, np.inf, .1]), 3, np.array(["z", "a", "b"]))
        self.assertEqual(m.tolist(), [True, False, True])
        self.assertEqual(ranked_mask(np.array([.1, .1]), 1, np.array(["z", "b"])).tolist(), [False, True])

    def test_deterministic_simulation(self):
        p = dict(record_id="unit", raw_rate=.3, a_fit=.8, nu_fit=.2, lens=[100]*8)
        a, ar = compute_one(p)
        b, br = compute_one(p)
        np.testing.assert_equal(ar, br)
        self.assertEqual(a["status"], "ok")
        self.assertEqual(a["score_latent_bias"], b["score_latent_bias"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["plan", "test", "pilot", "run"])
    args = parser.parse_args()
    if args.mode == "test":
        unittest.main(argv=[__file__], exit=False)
        return
    load_protocol()
    if args.mode == "plan":
        print("Protocol:", HERE / "protocol.json", flush=True)
        return
    rows = inputs()
    fitted, boot = [], []
    cache = HERE / "pilot_cache.json"
    if cache.exists():
        cached = json.loads(cache.read_text(encoding="utf-8"))
        fitted = cached["fitted"]
        boot = np.load(HERE / "pilot_rates.npy").tolist()
        assert [r["record_id"] for r in fitted] == [k for k, _ in rows[:8]]
    else:
        for key, row in rows[:8]:
            result, rates = compute_one(observed_parameters(key, row))
            fitted.append(result)
            boot.append(rates.tolist())
        cache.write_text(json.dumps(dict(fitted=fitted), ensure_ascii=False, indent=2), encoding="utf-8")
        np.save(HERE / "pilot_rates.npy", np.asarray(boot))
    estimate = max(r["wall_s"] for r in fitted[:8]) * 394 * 4
    timing = dict(pilot_record_ids=[k for k, _ in rows[:8]], pilot_wall_s=[r["wall_s"] for r in fitted[:8]],
                  conservative_full_estimate_s=estimate)
    (HERE / "timing.json").write_text(json.dumps(timing, indent=2), encoding="utf-8")
    print(json.dumps(timing), flush=True)
    if args.mode == "pilot":
        return
    start = time.perf_counter()
    for key, row in rows[len(fitted):]:
        result, rates = compute_one(observed_parameters(key, row))
        fitted.append(result)
        boot.append(rates.tolist())
        if len(fitted) % 50 == 0:
            print(f"Completed {len(fitted)}/394; remaining phase elapsed {time.perf_counter()-start:.1f}s", flush=True)
    (HERE / "fitted_scores.json").write_text(json.dumps(fitted, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = summarize(rows, fitted, np.asarray(boot))
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
