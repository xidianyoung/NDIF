"""Summary of the drift/diffusion scaling grid (network_scaling_grid.py): the exact law, the three scaling paths,
the collapse onto D/lambda, the offset construction, the one-well construction, the per-network endpoint folds
and Supplementary Table 14.

  A  the exact law: var_e * lambda_e / D_e = 1 in every cell; cells sharing s^2/g share the exact latent
     separation; lambda_e scales with g and D_e with s^2
  B  the three paths of the two-well construction -- joint (s^2 = g), diffusion only (g = 1) and drift only
     (s^2 = 1): per-x medians over networks, and per network the fold from the smallest to the largest x of the
     density barrier, the lag-1 rate (m_lam_axis), the diffusion, the separation and the escape hazard
  C  log(density barrier) regressed on log g and log s^2 per network
  C2 the offset construction (fixed displacement of 3 base-cell s.d.), in which the equilibrium count can move:
     number of networks whose count changes along each path
  D  the one-well construction (true separation zero)
  endpoint tables: per network, path and metric the ratio of the value at x = 2 to the value at x = 0.5
     (escape hazard not measurable when no chain escaped at either end), and their median / minimum / maximum
     over networks; Supplementary Table 14 is the same ratios to three decimals

Input   analyses/recurrent_networks/out/network_parameter_grid.csv (--rows; the stored rows are the Source Data
        file analysis_inputs/stored_results/network_parameter_grid.csv).
Outputs analyses/recurrent_networks/out/network_parameter_grid_report.txt (sections A-D),
        network_parameter_grid_numbers.json, all_endpoint_ratios.csv, endpoint_summary.csv,
        supplementary_table_14.csv, network_scaling_endpoints.json.
Reported in  Fig. 1d (joint path: barrier x0.912 [0.811, 1.032], lag-1 rate x2.535 [2.017, 3.433], escape hazard
        x4.745 [3.698, 8.263] over the 5 networks whose escapes resolve at both ends), Fig. 1e (drift only: barrier
        x5.033; diffusion only: x0.193; offset construction: count changes in 7/8 and 8/8 networks, joint 1/8),
        Supplementary Fig. 1 (600 combinations, 66 of 200 escape hazards censored), Supplementary Table 14.
Runtime  seconds.

    python analyses/recurrent_networks/network_scaling_summary.py
"""
from __future__ import annotations

import argparse
import csv as _csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from analyses.common import paths  # noqa: E402

SINK = []


def log(m):
    print(m, flush=True)
    SINK.append(str(m))


def num(r, k):
    v = r.get(k, "")
    if v in ("", "None", None):
        return np.nan
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def out(name):
    return paths.out("analyses", "recurrent_networks", "out", name)


def main():
    ap = argparse.ArgumentParser(description="summary of the drift/diffusion scaling grid")
    ap.add_argument("--rows", default=None,
                    help="grid rows (default: analyses/recurrent_networks/out/network_parameter_grid.csv)")
    args = ap.parse_args()
    rows_path = args.rows or out("network_parameter_grid.csv")
    rows = list(_csv.DictReader(open(rows_path, newline="")))
    seeds = sorted({r["anchor_seed"] for r in rows})
    log(f"networks: {seeds}  rows {len(rows)}")
    res = dict(n_networks=len(seeds), seeds=seeds)

    # ---------------------------------------------------------------- A: the exact law
    bad = [r for r in rows if not np.isfinite(num(r, "exact_ratio_identity"))]
    dev = max(abs(num(r, "exact_ratio_identity") - 1.0) for r in rows
              if np.isfinite(num(r, "exact_ratio_identity")))
    log("")
    log("=== A. THE EXACT LAW ===")
    log(f"  var_e * lambda_e / D_e = 1 to within {dev:.2e} over "
        f"{len(rows) - len(bad)} cells: the stationary spread along ANY direction is exactly "
        f"D_e / lambda_e, so a density reads the RATIO.")
    # exactness of the ratio collapse in the EXACT quantities
    worst = 0.0
    for s in seeds:
        rs = [r for r in rows if r["anchor_seed"] == s and r["arm"] == "two_wells"]
        byratio = {}
        for r in rs:
            byratio.setdefault(round(num(r, "ratio"), 6), []).append(num(r, "exact_delta"))
        for k, v in byratio.items():
            if len(v) > 1:
                worst = max(worst, float(np.max(v) - np.min(v)) / float(np.mean(v)))
    log(f"  cells sharing the same s^2/g have the SAME exact latent separation to within "
        f"{worst:.2e} relative, in {len(seeds)} of {len(seeds)} networks.")
    res["exact"] = dict(ratio_identity_max_dev=float(dev),
                        delta_collapse_max_rel_spread=float(worst))

    # exact lambda and D scaling
    lam_ok, D_ok = [], []
    for s in seeds:
        rs = [r for r in rows if r["anchor_seed"] == s and r["arm"] == "two_wells"]
        for r in rs:
            g, s2 = num(r, "g"), num(r, "s2")
            base = [x for x in rs if abs(num(x, "g") - 1) < 1e-9 and abs(num(x, "s2") - 1) < 1e-9]
            if not base:
                continue
            lam_ok.append(num(r, "exact_lam_e_per_index") / num(base[0],
                                                               "exact_lam_e_per_index") / g)
            D_ok.append(num(r, "exact_D_e_per_index") / num(base[0],
                                                            "exact_D_e_per_index") / s2)
    log(f"  lambda_e / (g * lambda_e(base)) = {np.mean(lam_ok):.6f} "
        f"[{np.min(lam_ok):.6f}, {np.max(lam_ok):.6f}];  "
        f"D_e / (s^2 * D_e(base)) = {np.mean(D_ok):.6f} "
        f"[{np.min(D_ok):.6f}, {np.max(D_ok):.6f}]")

    # ---------------------------------------------------------------- B: the three arms
    def arm_rows(arm, name):
        out = []
        for r in rows:
            if r["arm"] != arm:
                continue
            g, s2 = num(r, "g"), num(r, "s2")
            if name == "ratio" and abs(g - s2) > 1e-9:
                continue
            if name == "D_only" and abs(g - 1.0) > 1e-9:
                continue
            if name == "lam_only" and abs(s2 - 1.0) > 1e-9:
                continue
            out.append(r)
        return out

    log("")
    log("=== B. THE THREE ARMS, two-well construction ===")
    arms = {}
    for name, xcol in (("ratio", "g"), ("D_only", "s2"), ("lam_only", "g")):
        rs = arm_rows("two_wells", name)
        xs = sorted({round(num(r, xcol), 6) for r in rs})
        log("")
        log(f"  --- {name} arm (x = {xcol}) ---")
        log(f"  {'x':>7s} {'ratio':>7s} {'exact d2/8':>11s} {'barrier':>9s} {'nstable':>8s} "
            f"{'lam_hat':>9s} {'D_hat':>9s} {'esc hazard':>11s} {'clean':>7s} {'nets':>5s}")
        tab = []
        for x in xs:
            sel = [r for r in rs if abs(num(r, xcol) - x) < 1e-6]
            def agg(k):
                v = np.array([num(r, k) for r in sel])
                v = v[np.isfinite(v)]
                return (float(np.median(v)), float(np.min(v)), float(np.max(v)),
                        int(len(v))) if len(v) else (np.nan,) * 3 + (0,)
            b = agg("m_density_barrier_nats"); ns = agg("m_n_stable")
            lm = agg("m_lam_axis"); dh = agg("m_D_axis"); hz = agg("escape_hazard")
            d28 = agg("exact_delta2_over_8"); cf = agg("clean_frac_hi")
            rat = float(np.median([num(r, "ratio") for r in sel]))
            log(f"  {x:7.3f} {rat:7.3f} {d28[0]:11.2f} {b[0]:9.3f} {ns[0]:8.2f} "
                f"{lm[0]:9.4f} {dh[0]:9.4f} {hz[0]:11.3e} {cf[0]:7.3f} {b[3]:5d}")
            tab.append(dict(x=x, ratio=rat, exact_d2_8=d28[0], barrier=b[0],
                            barrier_min=b[1], barrier_max=b[2], n_stable=ns[0],
                            lam_hat=lm[0], D_hat=dh[0], hazard=hz[0], clean_frac=cf[0],
                            n_networks=b[3]))
        arms[name] = tab
    res["arms"] = arms

    # per-network fold across each arm, so the claim carries its own spread
    log("")
    log("=== B2. What each arm does, per network (fold from the smallest to the largest x) ===")
    folds = {}
    for name, xcol in (("ratio", "g"), ("D_only", "s2"), ("lam_only", "g")):
        rs = arm_rows("two_wells", name)
        per = []
        for s in seeds:
            a = sorted([r for r in rs if r["anchor_seed"] == s], key=lambda z: num(z, xcol))
            if len(a) < 2:
                continue
            def fold(k):
                lo, hi = num(a[0], k), num(a[-1], k)
                return float(hi / lo) if (np.isfinite(lo) and np.isfinite(hi) and lo > 0) else np.nan
            per.append(dict(seed=s, barrier=fold("m_density_barrier_nats"),
                            lam=fold("m_lam_axis"), D=fold("m_D_axis"),
                            hazard=fold("escape_hazard"),
                            n_stable_lo=num(a[0], "m_n_stable"),
                            n_stable_hi=num(a[-1], "m_n_stable"),
                            sep=fold("m_separation")))
        folds[name] = per

        def q(k):
            v = np.array([p[k] for p in per if np.isfinite(p[k])])
            return (f"x{np.median(v):.3f} [{np.min(v):.3f}, {np.max(v):.3f}] (n={len(v)})"
                    if len(v) else "n/a")
        nch = sum(1 for p in per if p["n_stable_lo"] != p["n_stable_hi"])
        log(f"  {name:9s} barrier {q('barrier'):34s} lambda {q('lam'):30s} D {q('D'):30s}")
        log(f"  {'':9s} separation {q('sep'):31s} escape hazard {q('hazard'):24s} "
            f"count changed in {nch}/{len(per)} networks")
    res["folds"] = folds

    # ---------------------------------------------------------------- C: the collapse
    log("")
    log("=== C. THE COLLAPSE: is the barrier a function of s^2/g alone? ===")
    coll = []
    for s in seeds:
        rs = [r for r in rows if r["anchor_seed"] == s and r["arm"] == "two_wells"
              and np.isfinite(num(r, "m_density_barrier_nats"))
              and num(r, "m_density_barrier_nats") > 0]
        if len(rs) < 8:
            continue
        X = np.array([[1.0, math.log(num(r, "g")), math.log(num(r, "s2"))] for r in rs])
        y = np.array([math.log(num(r, "m_density_barrier_nats")) for r in rs])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        pred = X @ beta
        r2 = 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
        # the one-parameter ratio law
        Xr = np.array([[1.0, math.log(num(r, "g") / num(r, "s2"))] for r in rs])
        br, *_ = np.linalg.lstsq(Xr, y, rcond=None)
        r2r = 1 - np.sum((y - Xr @ br) ** 2) / np.sum((y - y.mean()) ** 2)
        coll.append(dict(seed=s, beta_logg=float(beta[1]), beta_logs2=float(beta[2]),
                         sum=float(beta[1] + beta[2]), r2_two_param=float(r2),
                         beta_ratio=float(br[1]), r2_ratio_law=float(r2r), n=len(rs)))
    res["collapse"] = coll
    if coll:
        bg = np.array([c["beta_logg"] for c in coll])
        bs = np.array([c["beta_logs2"] for c in coll])
        sm = np.array([c["sum"] for c in coll])
        log(f"  log barrier = a + b_g log g + b_s log s^2 :  b_g = {np.median(bg):+.3f} "
            f"[{bg.min():+.3f}, {bg.max():+.3f}],  b_s = {np.median(bs):+.3f} "
            f"[{bs.min():+.3f}, {bs.max():+.3f}]")
        log(f"  b_g + b_s = {np.median(sm):+.3f} [{sm.min():+.3f}, {sm.max():+.3f}] "
            f"(0 = the barrier depends on g/s^2 ALONE); "
            f"|sum| < 0.15 in {int((np.abs(sm) < 0.15).sum())}/{len(sm)} networks")
        r2r = np.array([c["r2_ratio_law"] for c in coll])
        r22 = np.array([c["r2_two_param"] for c in coll])
        log(f"  R^2 of the one-parameter ratio law {np.median(r2r):.4f} "
            f"[{r2r.min():.4f}, {r2r.max():.4f}] against {np.median(r22):.4f} for the "
            f"two-parameter fit")

    # ---------------------------------------------------------------- C2: the offset arm
    log("")
    log("=== C2. THE ARM IN WHICH THE COUNT CAN MOVE: a FIXED physical displacement, "
        "3.0 base-cell s.d., with only g and s^2 changed ===")
    log(f"  {'g':>7s} {'s2':>7s} {'ratio':>7s} {'sep':>8s} {'n_stable':>9s} {'frac>=2':>8s} "
        f"{'barrier':>9s} {'modes':>7s}")
    offs = []
    rs = [r for r in rows if r["arm"] == "offset"]
    for name, xcol, sel in (("ratio", "g", lambda r: abs(num(r, "g") - num(r, "s2")) < 1e-9),
                            ("D_only", "s2", lambda r: abs(num(r, "g") - 1) < 1e-9),
                            ("lam_only", "g", lambda r: abs(num(r, "s2") - 1) < 1e-9)):
        log(f"  --- {name} arm ---")
        sub = [r for r in rs if sel(r)]
        for x in sorted({round(num(r, xcol), 6) for r in sub}):
            cell = [r for r in sub if abs(num(r, xcol) - x) < 1e-6]
            ns = np.array([num(r, "m_n_stable") for r in cell])
            ns = ns[np.isfinite(ns)]
            sep = np.array([num(r, "m_separation") for r in cell])
            ba = np.array([num(r, "m_density_barrier_nats") for r in cell])
            md = np.array([num(r, "m_density_modes") for r in cell])
            ent = dict(arm=name, x=x, g=float(np.median([num(r, "g") for r in cell])),
                       s2=float(np.median([num(r, "s2") for r in cell])),
                       ratio=float(np.median([num(r, "ratio") for r in cell])),
                       separation=float(np.median(sep)), n_stable=float(np.median(ns)),
                       frac_ge2=float((ns >= 2).mean()) if len(ns) else np.nan,
                       barrier=float(np.median(ba)), modes=float(np.median(md)),
                       n=len(cell))
            offs.append(ent)
            log(f"  {ent['g']:7.3f} {ent['s2']:7.3f} {ent['ratio']:7.3f} {ent['separation']:8.3f} "
                f"{ent['n_stable']:9.2f} {ent['frac_ge2']:8.3f} {ent['barrier']:9.3f} "
                f"{ent['modes']:7.2f}")
    res["offset_arm"] = offs
    # does the count change along each arm, per network?
    log("")
    for name, xcol, sel in (("ratio", "g", lambda r: abs(num(r, "g") - num(r, "s2")) < 1e-9),
                            ("D_only", "s2", lambda r: abs(num(r, "g") - 1) < 1e-9),
                            ("lam_only", "g", lambda r: abs(num(r, "s2") - 1) < 1e-9)):
        ch = 0; tot = 0
        for s in seeds:
            a = sorted([r for r in rs if sel(r) and r["anchor_seed"] == s],
                       key=lambda z: num(z, xcol))
            if len(a) < 2:
                continue
            tot += 1
            if num(a[0], "m_n_stable") != num(a[-1], "m_n_stable"):
                ch += 1
        log(f"  offset arm, {name:9s}: the equilibrium COUNT changes across the arm in "
            f"{ch}/{tot} networks")

    # ---------------------------------------------------------------- D: the cortical regime
    log("")
    log("=== D. THE REGIME EVERY CORTICAL RECORDING OCCUPIES (one_well, true separation 0) ===")
    ow = [r for r in rows if r["arm"] == "one_well"]
    bar = np.array([num(r, "m_density_barrier_nats") for r in ow])
    bar = bar[np.isfinite(bar)]
    ns = np.array([num(r, "m_n_stable") for r in ow])
    ns = ns[np.isfinite(ns)]
    sep = np.array([num(r, "m_separation") for r in ow])
    sep = sep[np.isfinite(sep)]
    log(f"  n = {len(ow)} cells over {len(seeds)} networks and the whole 5x5 (g, s^2) grid")
    log(f"  density barrier exactly 0.0000 nats in {float((bar == 0).mean()):.3f} of cells "
        f"(max {bar.max():.4f})")
    log(f"  equilibrium count = 1 in {float((ns == 1).mean()):.3f} of cells, >= 2 in "
        f"{float((ns >= 2).mean()):.3f}")
    log(f"  measured separation {np.median(sep):.3f} [{sep.min():.3f}, {sep.max():.3f}]")
    res["one_well"] = dict(n=len(ow), frac_barrier_zero=float((bar == 0).mean()),
                           max_barrier=float(bar.max()),
                           frac_count_one=float((ns == 1).mean()),
                           frac_count_ge2=float((ns >= 2).mean()),
                           separation_median=float(np.median(sep)))

    with open(out("network_parameter_grid_numbers.json"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(res, indent=1, default=float))
    with open(out("network_parameter_grid_report.txt"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(SINK))
    endpoint_tables(rows_path)
    return 0


# ---------------------------------------------------------------- endpoint tables
METRICS = ["exact_delta", "m_density_barrier_nats", "m_lam_axis", "escape_hazard"]
PATH_LABEL = {"joint": "Joint", "D_only": "Diffusion", "lam_only": "Drift"}


def endpoint_tables(rows_path):
    """Per network, path and metric: value at x = 2 over value at x = 0.5 on the two-well construction (x = g on
    the joint and drift-only paths, x = s^2 on the diffusion-only path).  The escape-hazard ratio is not
    measurable when the hazard is censored (no escape within the horizon) at either end."""
    dr = pd.read_csv(rows_path)
    seeds = sorted(dr.anchor_seed.unique())
    rl = []
    for arm in ["joint", "D_only", "lam_only"]:
        mask = np.isclose(dr.g, dr.s2) if arm == "joint" else np.isclose(dr.g if arm == "D_only" else dr.s2, 1.)
        sub = dr[(dr.arm == "two_wells") & mask]
        xx = "s2" if arm == "D_only" else "g"
        for seed in seeds:
            z = sub[sub.anchor_seed == seed].sort_values(xx)
            lo = z.iloc[0]
            hi = z.iloc[-1]
            assert np.isclose(lo[xx], .5) and np.isclose(hi[xx], 2)
            for m in METRICS:
                cens = m == "escape_hazard" and (bool(lo.escape_censored) or bool(hi.escape_censored))
                ratio = np.nan if cens else hi[m] / lo[m]
                rl.append(dict(arm=arm, anchor_seed=int(seed), metric=m, endpoint_low=.5, endpoint_high=2.,
                               value_low=lo[m], value_high=hi[m], ratio_high_over_low=ratio,
                               measurable=not cens,
                               reason="no events at one or both fixed-horizon endpoints" if cens else "",
                               escape_censored_low=bool(lo.escape_censored),
                               escape_censored_high=bool(hi.escape_censored)))
    rat = pd.DataFrame(rl)
    rat.to_csv(out("all_endpoint_ratios.csv"), index=False, lineterminator="\n")
    su = rat[rat.measurable].groupby(["arm", "metric"]).ratio_high_over_low.agg(
        ["count", "median", "min", "max"]).reset_index()
    su["interval_definition"] = "complete observed across-network minimum/maximum, not confidence interval"
    su.to_csv(out("endpoint_summary.csv"), index=False, lineterminator="\n")

    # Supplementary Table 14: the same ratios to three decimals, NA where not measurable
    with open(out("supplementary_table_14.csv"), "w", encoding="utf-8", newline="") as fh:
        w = _csv.writer(fh, lineterminator="\n")
        w.writerow(["Network", "Path", "Local sep.", "Barrier", "AR rate", "Crossing hazard"])
        for arm in ["joint", "D_only", "lam_only"]:
            for seed in seeds:
                cell = rat[(rat.arm == arm) & (rat.anchor_seed == seed)].set_index("metric")
                w.writerow([int(seed), PATH_LABEL[arm]] + [
                    ("%.3f" % cell.loc[m, "ratio_high_over_low"]) if cell.loc[m, "measurable"] else "NA"
                    for m in METRICS])

    # counts quoted with Fig. 1d,e and Supplementary Fig. 1
    tw = dr[dr.arm == "two_wells"]
    off = dr[dr.arm == "offset"]
    moved = {}
    for arm in ["joint", "D_only", "lam_only"]:
        mask = np.isclose(off.g, off.s2) if arm == "joint" else np.isclose(off.g if arm == "D_only" else off.s2, 1.)
        xx = "s2" if arm == "D_only" else "g"
        n_moved, n_net = 0, 0
        for seed in seeds:
            z = off[mask & (off.anchor_seed == seed)].sort_values(xx)
            if len(z) < 2:
                continue
            n_net += 1
            n_moved += int(z.iloc[0].m_n_stable != z.iloc[-1].m_n_stable)
        moved[arm] = dict(n_networks_count_changed=n_moved, n_networks=n_net,
                          n_stable_low_median=float(np.median(
                              [off[mask & (off.anchor_seed == s)].sort_values(xx).iloc[0].m_n_stable for s in seeds])),
                          n_stable_high_median=float(np.median(
                              [off[mask & (off.anchor_seed == s)].sort_values(xx).iloc[-1].m_n_stable for s in seeds])))
    summ = {f"{r['arm']}|{r['metric']}": dict(count=int(r["count"]), median=float(r["median"]),
                                               min=float(r["min"]), max=float(r["max"]))
            for r in su.to_dict("records")}
    res = dict(n_combinations=int(len(dr)), n_networks=int(len(seeds)), n_cells=int(len(tw)),
               n_escape_hazards_censored=int(tw.escape_censored.sum()),
               endpoint_folds=summ, offset_count_changes=moved)
    with open(out("network_scaling_endpoints.json"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(res, indent=1) + "\n")
    print("")
    print(f"combinations {res['n_combinations']}; escape hazards censored in "
          f"{res['n_escape_hazards_censored']} of {res['n_cells']} cells")
    for k, v in summ.items():
        print(f"  {k:32s} x{v['median']:.3f} [{v['min']:.3f}, {v['max']:.3f}] (n={v['count']})")
    for arm, v in moved.items():
        print(f"  offset construction, {arm:9s}: count {v['n_stable_low_median']:.0f} -> "
              f"{v['n_stable_high_median']:.0f}, changes in {v['n_networks_count_changed']}/{v['n_networks']} networks")


if __name__ == "__main__":
    raise SystemExit(main())
