"""Read out the separation-floor benchmark: floors, detection crossings, bootstrap intervals (Fig. 1f).

Per cohort and generator (floor_benchmark.py):
  * the zero-separation floor (median measured separation when the true separation is 0) and the 95th
    percentile of that zero-truth distribution, the false detection rate at zero, and the 50 %, 5 %,
    80 % and 95 % detection crossings (true separation, and measured separation at that point);
  * participant-bootstrap 95 % intervals (4,000 resamples of the replicate participants, paired across
    separations; seed 424242 + generator index) of the floor and of the crossings;
  * the change in floor and 50 % crossing added by each generator ingredient;
  * the observed separations (extract_generators.py) against the floor of the cohort-matched generator
    (nonstationarity): participant median (median over a participant's records) and range, the numbers
    of participants at or below the floor and at or below its 95th percentile, and the record-level
    median.

Unit of analysis is the participant: every replicate in a cell comes from a different participant and the
bootstrap resamples participants.

Inputs   analyses/separation_floors/out/floor_benchmark.json; paths.cache("separation_floors")/
         floor_benchmark_rows.pkl and generators/<cohort>.pkl.
Outputs  analyses/separation_floors/out/separation_floor_summary.json, separation_floor_ladder.csv,
         separation_floor_report.txt.
Reported in  Fig. 1f: floors 0.571 [0.515, 0.720] (HCP fMRI), 0.225 [0.210, 0.249] (HCP MEG),
         0.178 [0.164, 0.197] (ds003838), 0.173 [0.159, 0.193] (eegmmidb), against observed participant
         medians 0.86, 0.48, 0.49 and 0.46; 8 of 99, 4 of 83, 0 of 64 and 1 of 106 participants at or below
         the floor; generator geometry d / dt / median segments per regime.  Summary keys
         /cohorts/<cohort>/rungs/nonstationarity/{floor, floor_p95, bootstrap/floor_ci} and
         /cohorts/<cohort>/observed_separation.

Runtime about 1 min on one core.

    python analyses/separation_floors/floor_report.py --n-boot 4000
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from analyses.common import paths                                     # noqa: E402

LABEL = dict(hcp_fmri="HCP WM fMRI", hcp_meg="HCP MEG", ds003838="digit-span EEG (ds003838)",
             eegmmidb="motor EEG (eegmmidb)")
RUNGS = ["isotropic_reference", "dimension", "anisotropy", "drift",
         "sample_size", "segments", "nonstationarity"]
INGREDIENT = {"dimension": "state dimension (and dt, which cannot move the count)",
              "anisotropy": "anisotropic per-regime covariance + the record's displacement direction",
              "drift": "per-regime one-step map (autocorrelation, rotation, non-normality)",
              "sample_size": "number of windows per regime",
              "segments": "segment count and segment lengths",
              "nonstationarity": "measured between-segment mean heterogeneity"}


def interp_cross(dd, fr, sp, level):
    dd, fr, sp = map(lambda x: np.asarray(x, float), (dd, fr, sp))
    for i in range(1, len(dd)):
        if fr[i] >= level and fr[i - 1] < level:
            if fr[i] == fr[i - 1]:
                return float(dd[i]), float(sp[i])
            t = (level - fr[i - 1]) / (fr[i] - fr[i - 1])
            return (float(dd[i - 1] + t * (dd[i] - dd[i - 1])),
                    float(sp[i - 1] + t * (sp[i] - sp[i - 1])))
    return None, None


def cross_from_cells(cells, level):
    dd = sorted((float(k) for k in cells), key=float)
    fr = [cells[f"{d}"]["frac_field_ge2"] for d in dd]
    sp = [cells[f"{d}"]["sep_measured_median"] for d in dd]
    return interp_cross(dd, fr, sp, level)


def boot_cell(rows, cohort, rung, deltas, n_boot, seed):
    """Participant bootstrap of the floor and of the two crossings, paired across separations."""
    R = [r for r in rows if r["ok"] and r["cohort"] == cohort and r["rung"] == rung]
    by = {}
    for r in R:
        by.setdefault(r["rep"], {})[round(r["delta"], 6)] = r
    reps = sorted(by)
    rng = np.random.default_rng(seed)
    fl, s50, s05, d50 = [], [], [], []
    for _ in range(n_boot):
        idx = rng.choice(len(reps), size=len(reps), replace=True)
        sel = [by[reps[i]] for i in idx]
        z = [c[round(deltas[0], 6)]["sep_measured"] for c in sel if round(deltas[0], 6) in c]
        fl.append(np.median(z))
        fr, sp = [], []
        for dl in deltas:
            k = round(dl, 6)
            v = [c[k] for c in sel if k in c]
            fr.append(np.mean([x["n_stable"] >= 2 for x in v]))
            sp.append(np.median([x["sep_measured"] for x in v]))
        a, b = interp_cross(deltas, fr, sp, 0.5)
        c_, e_ = interp_cross(deltas, fr, sp, 0.05)
        d50.append(np.nan if a is None else a)
        s50.append(np.nan if b is None else b)
        s05.append(np.nan if e_ is None else e_)

    def ci(v):
        v = np.asarray(v, float)
        v = v[np.isfinite(v)]
        if len(v) < 10:
            return [None, None]
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return dict(floor_ci=ci(fl), sep50_ci=ci(s50), sep05_ci=ci(s05), delta50_ci=ci(d50),
                sep50_resolved_frac=float(np.mean(np.isfinite(np.asarray(s50, float)))),
                sep05_resolved_frac=float(np.mean(np.isfinite(np.asarray(s05, float)))))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n-boot", type=int, default=4000)
    args = ap.parse_args()

    cache = paths.cache("separation_floors")
    out = lambda name: paths.out("analyses", "separation_floors", "out", name)  # noqa: E731
    with open(out("floor_benchmark.json"), encoding="utf-8") as fh:
        res = json.load(fh)
    with open(os.path.join(cache, "floor_benchmark_rows.pkl"), "rb") as fh:
        rows = pickle.load(fh)
    deltas = res["deltas"]
    head = dict(created=time.strftime("%Y-%m-%d %H:%M:%S"), reps=res["reps"],
                deltas=deltas, n_boot=args.n_boot, cohorts={})
    txt = []

    def P(s=""):
        txt.append(s)
        print(s, flush=True)

    P("=" * 108)
    P("Single-basin separation floors: simulation benchmark of the equilibrium count at each cohort's geometry")
    P("=" * 108)
    P(f"replicates per cell {res['reps']}; separations {deltas}")
    P("counter: fit_linear_regime -> find_equilibria(softness=1, n_starts=200) -> n_stable>=2;")
    P("ridgeline_modes_2 on the same fitted pair gives the measured separation.")
    P("")

    for cohort in res["cohorts"]:
        C = res["cohorts"][cohort]
        with open(os.path.join(cache, "generators", f"{cohort}.pkl"), "rb") as fh:
            recs = pickle.load(fh)
        obs = np.array([r["sep_real"] for r in recs])
        subs = sorted(set(r["subject"] for r in recs))
        obs_sub = np.array([np.median([r["sep_real"] for r in recs if r["subject"] == s]) for s in subs])

        P("-" * 108)
        P(f"{LABEL[cohort]}   d={C['d']}  dt={C['dt']}s  "
          f"records {C['n_records']} / subjects {C['n_subjects']}")
        P(f"  isotropic reference generator : n = {C['n_iso_per_regime']} per regime, one contiguous segment, "
          f"Sigma = I, lambda = 0.5 s^-1, d = 8")
        P(f"  this cohort has               : {C['windows_per_regime']} windows per regime in "
          f"{C['segments_per_regime']} segments, {C['increments_per_regime']} increments per regime")
        P("")
        P(f"  {'generator':24s} {'floor (sep at true 0)':>24s} {'fp@0':>6s} "
          f"{'50% crossing':>22s} {'5% crossing':>22s}")
        ladder = {}
        for rung in RUNGS:
            s = C["rungs"][rung]
            b = boot_cell(rows, cohort, rung, deltas, args.n_boot, 424242 + RUNGS.index(rung))
            s["bootstrap"] = b
            ladder[rung] = s
            f50 = s["sep_measured_at_50pct"]
            f05 = s["sep_measured_at_5pct"]
            P(f"  {rung:24s} {s['floor_sep_measured_median_at_zero']:>10.4f} "
              f"[{b['floor_ci'][0]:.4f},{b['floor_ci'][1]:.4f}] {s['false_detection_rate_at_zero']:>6.3f} "
              f"{('%8.3f' % f50) if f50 else '     n/a'} "
              f"[{'  n/a ' if b['sep50_ci'][0] is None else '%.3f' % b['sep50_ci'][0]},"
              f"{'  n/a ' if b['sep50_ci'][1] is None else '%.3f' % b['sep50_ci'][1]}] "
              f"{('%8.3f' % f05) if f05 else '     n/a'} "
              f"[{'  n/a ' if b['sep05_ci'][0] is None else '%.3f' % b['sep05_ci'][0]},"
              f"{'  n/a ' if b['sep05_ci'][1] is None else '%.3f' % b['sep05_ci'][1]}]")
        P("")
        P("  ingredient attribution (change in the floor and in the 50% crossing added by each generator)")
        prev = ladder["isotropic_reference"]
        attr = {}
        for rung in RUNGS[1:]:
            cur = ladder[rung]
            df = cur["floor_sep_measured_median_at_zero"] - prev["floor_sep_measured_median_at_zero"]
            a, b2 = cur["sep_measured_at_50pct"], prev["sep_measured_at_50pct"]
            dx = (a - b2) if (a is not None and b2 is not None) else None
            attr[rung] = dict(ingredient=INGREDIENT[rung], d_floor=float(df),
                              d_sep50=(None if dx is None else float(dx)))
            P(f"    {rung:22s} floor {df:+.4f}   50% crossing "
              f"{'n/a' if dx is None else '%+.4f' % dx}   <- {INGREDIENT[rung]}")
            prev = cur
        iso = ladder["isotropic_reference"]
        matched = ladder["nonstationarity"]
        P(f"    {'total':22s} floor "
          f"{matched['floor_sep_measured_median_at_zero'] - iso['floor_sep_measured_median_at_zero']:+.4f}   "
          f"50% crossing "
          f"{'n/a' if (matched['sep_measured_at_50pct'] is None or iso['sep_measured_at_50pct'] is None) else '%+.4f' % (matched['sep_measured_at_50pct'] - iso['sep_measured_at_50pct'])}")
        P("")
        fl6 = matched["floor_sep_measured_median_at_zero"]
        p95 = matched["floor_sep_measured_p95_at_zero"]
        P(f"  observed separations against the cohort-matched floor {fl6:.4f} (95th pct of the zero-truth "
          f"distribution {p95:.4f}):")
        P(f"    per participant  median {np.median(obs_sub):.4f}  range "
          f"[{obs_sub.min():.4f}, {obs_sub.max():.4f}]  "
          f"{int((obs_sub <= fl6).sum())}/{len(obs_sub)} at or below the floor, "
          f"{int((obs_sub <= p95).sum())}/{len(obs_sub)} at or below its 95th percentile")
        P(f"    per record       median {np.median(obs):.4f}  range [{obs.min():.4f}, {obs.max():.4f}]  "
          f"{int((obs <= fl6).sum())}/{len(obs)} at or below the floor  [record level]")
        d80, s80 = cross_from_cells(matched["cells"], 0.80)
        d95, s95 = cross_from_cells(matched["cells"], 0.95)
        P(f"  detection by the cohort-matched generator: a true separation of "
          f"{'n/a' if d80 is None else '%.3f' % d80} is detected in 80% of participants and "
          f"{'n/a' if d95 is None else '%.3f' % d95} in 95%; measured-separation equivalents "
          f"{'n/a' if s80 is None else '%.3f' % s80} and {'n/a' if s95 is None else '%.3f' % s95}.")
        P("")
        head["cohorts"][cohort] = dict(
            label=LABEL[cohort], d=C["d"], dt=C["dt"], n_records=C["n_records"],
            n_subjects=C["n_subjects"],
            windows_per_regime=C["windows_per_regime"],
            increments_per_regime=C["increments_per_regime"],
            segments_per_regime=C["segments_per_regime"],
            rungs={k: dict(floor=v["floor_sep_measured_median_at_zero"],
                           delta_80=cross_from_cells(v["cells"], 0.80)[0],
                           sep_80=cross_from_cells(v["cells"], 0.80)[1],
                           delta_95=cross_from_cells(v["cells"], 0.95)[0],
                           sep_95=cross_from_cells(v["cells"], 0.95)[1],
                           floor_p95=v["floor_sep_measured_p95_at_zero"],
                           fp_at_zero=v["false_detection_rate_at_zero"],
                           delta_50=v["delta_at_50pct"], sep_50=v["sep_measured_at_50pct"],
                           delta_05=v["delta_at_5pct"], sep_05=v["sep_measured_at_5pct"],
                           bootstrap=v["bootstrap"],
                           detection={k2: v2["frac_field_ge2"] for k2, v2 in v["cells"].items()},
                           density={k2: v2["frac_density_bimodal"] for k2, v2 in v["cells"].items()},
                           sep_median={k2: v2["sep_measured_median"] for k2, v2 in v["cells"].items()})
                   for k, v in ladder.items()},
            attribution=attr,
            observed_separation=dict(
                subject_median=float(np.median(obs_sub)),
                subject_range=[float(obs_sub.min()), float(obs_sub.max())],
                record_median=float(np.median(obs)),
                record_range=[float(obs.min()), float(obs.max())],
                n_subjects_at_or_below_floor=int((obs_sub <= fl6).sum()),
                n_subjects_at_or_below_floor_p95=int((obs_sub <= p95).sum()),
                n_records_at_or_below_floor=int((obs <= fl6).sum())))

    # ------------------------------------------------------------------ all cohorts
    P("=" * 108)
    P("All four cohorts, cohort-matched generator")
    P(f"  {'cohort':28s} {'floor':>10s} {'50% crossing':>14s} {'5% crossing':>14s}")
    fl_all, x50_all, x05_all = [], [], []
    for c in res["cohorts"]:
        h = head["cohorts"][c]
        matched = h["rungs"]["nonstationarity"]
        fl_all.append(matched["floor"])
        if matched["sep_50"]:
            x50_all.append(matched["sep_50"])
        if matched["sep_05"]:
            x05_all.append(matched["sep_05"])
        P(f"  {LABEL[c]:28s} {matched['floor']:>10.4f} "
          f"{('%14.3f' % matched['sep_50']) if matched['sep_50'] else '           n/a'} "
          f"{('%14.3f' % matched['sep_05']) if matched['sep_05'] else '           n/a'}")
    head["resolution_bound"] = dict(
        crossings_5pct={c: head["cohorts"][c]["rungs"]["nonstationarity"]["sep_05"]
                        for c in res["cohorts"]},
        crossing_5pct_min=(float(np.min(x05_all)) if x05_all else None),
        crossing_5pct_max=(float(np.max(x05_all)) if x05_all else None),
        crossing_50pct_min=(float(np.min(x50_all)) if x50_all else None),
        crossing_50pct_max=(float(np.max(x50_all)) if x50_all else None),
        floor_min=float(np.min(fl_all)), floor_max=float(np.max(fl_all)))
    P("")
    P(f"  5% crossings span {head['resolution_bound']['crossing_5pct_min']:.3f} to "
      f"{head['resolution_bound']['crossing_5pct_max']:.3f}")
    P("=" * 108)

    with open(out("separation_floor_summary.json"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(head, indent=1))
    with open(out("separation_floor_report.txt"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(txt))
    with open(out("separation_floor_ladder.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["cohort", "generator", "ingredient_added", "floor_sep_at_true_zero", "floor_ci_lo",
                    "floor_ci_hi", "floor_p95", "false_detection_rate_at_zero", "delta_at_50pct",
                    "sep_at_50pct", "sep50_ci_lo", "sep50_ci_hi", "delta_at_5pct", "sep_at_5pct"])
        for c in res["cohorts"]:
            h = head["cohorts"][c]
            for rung in RUNGS:
                v = h["rungs"][rung]
                b = v["bootstrap"]
                w.writerow([c, rung, INGREDIENT.get(rung, "isotropic reference generator"),
                            round(v["floor"], 6), b["floor_ci"][0], b["floor_ci"][1],
                            round(v["floor_p95"], 6), v["fp_at_zero"], v["delta_50"],
                            v["sep_50"], b["sep50_ci"][0], b["sep50_ci"][1],
                            v["delta_05"], v["sep_05"]])
    print("\nwrote separation_floor_summary.json, separation_floor_ladder.csv, separation_floor_report.txt")


if __name__ == "__main__":
    main()
