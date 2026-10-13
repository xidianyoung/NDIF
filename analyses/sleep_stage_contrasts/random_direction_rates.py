# -*- coding: utf-8 -*-
r"""Random-direction two-lag rate contrasts of adjacent sleep stages at 1-s windows (153 Sleep-EDF
cassette nights).

Along a single direction u of the four-dimensional sleep state (sleep_common.state_space), the
projected bout-centred series y_t = x_t + e_t with e white has r_y(k) = (1 - s) rho^k for k >= 1, so

    rho = r(2) / r(1)            free of the white share s
    signal variance = r(1)^2 / r(2) * Var(y)
    lambda = -ln rho / dt,       D = sigma_signal^2 (1 - rho^2) / (2 dt).

For every night and adjacent stage pair (Wake>N1, N1>N2, N2>N3, N3>N4; at least 6 bouts in each
stage), both stages are read along the same directions: 200 random unit directions (contrast =
median over directions of log lambda_deeper - log lambda_shallower, without bias correction), the
in-sample axis between the two stage means and the axis estimated from every other night
(each with a matched bootstrap bias correction of rho, 200 draws, `_bc_rho`), and a within-stage
ceiling from two halves of the same stage (60 directions).  Nights are averaged within
participant and summarised by sleep_common.paired_summary (median, 4,000 participant resamples,
one generator seeded 20260922 in a fixed order).

Reported value: the random-direction rows of Fig. 5e / Supplementary Table 11,
contrasts/<pair>/random/dlog_lam: S1/Wake 1.446 [1.333, 1.663] (n 57) and S2/S1 1.445
[1.301, 1.654] (n 76); tabulated by rate_corrections.py.

Random streams.  The random directions and the bootstrap draws of a night use a generator seeded
Python's hash of (<stem>, "ax1d", 20260922), and the summary runs over the rows in the order in which the worker
processes return them.
`--rows` re-runs the summary on a stored row table in its stored order, which reproduces the
stored summary exactly.

Inputs   the 1-s sleep window cache (window_cache.py) in paths.cache("sleep_states")
         (NDIF_SLEEP_STATES_CACHE); or, with --rows, a stored row table.
Outputs  analyses/sleep_stage_contrasts/out/sleep_random_direction_two_lag.json   summary
         analyses/sleep_stage_contrasts/out/sleep_random_direction_two_lag_rows.csv   one row
         per night, pair and direction set (not written with --rows); below NDIF_OUT when set.
         The Source Data hold both under analysis_inputs/sleep/.
Runtime  about 1 min with 4 workers; --rows a few seconds.

    python analyses/sleep_stage_contrasts/random_direction_rates.py --win 1.0 --workers 4
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse                                                       # noqa: E402
import csv                                                            # noqa: E402
import json                                                           # noqa: E402
import sys                                                            # noqa: E402
import time                                                           # noqa: E402
import traceback                                                      # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed     # noqa: E402
from pathlib import Path                                              # noqa: E402

import numpy as np                                                    # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))
from analyses.common import paths  # noqa: E402
import sleep_common as C  # noqa: E402

LIBRARY = "cassette"
N_RANDOM = 200
N_BOOT = 4000
N_BC = 200

OUT_JSON = ("analyses", "sleep_stage_contrasts", "out", "sleep_random_direction_two_lag.json")
OUT_ROWS = ("analyses", "sleep_stage_contrasts", "out", "sleep_random_direction_two_lag_rows.csv")


def acf_pair(blocks, kmax=3):
    """r(1..kmax) and the pooled within-block variance (each block centred on its mean)."""
    num = np.zeros(kmax); da = np.zeros(kmax); db = np.zeros(kmax)
    ss = 0.0; n = 0
    for b in blocks:
        c = np.asarray(b, float)
        c = c - c.mean()
        ss += float(c @ c); n += len(c)
        for k in range(1, kmax + 1):
            if len(c) <= k + 1:
                continue
            x, y = c[:-k], c[k:]
            num[k - 1] += float(x @ y); da[k - 1] += float(x @ x); db[k - 1] += float(y @ y)
    den = np.sqrt(da * db)
    r = np.where(den > 0, num / np.maximum(den, 1e-300), np.nan)
    return r, (ss / max(n, 1))


def _bc_rho(rho, share, n_blocks, n_win, dt, rng, n_boot=N_BC):
    """Bootstrap bias correction of the lag-2 ratio on this exact geometry.

    Per-block mean removal biases every r(k) downward at order 1/n_win, and the
    ratio only partly cancels it, so the correction is measured rather than
    assumed: simulate an AR(1) of this rho plus a white share of this size on the
    same block count and length, re-estimate, and take rho_bc = 2 rho - E[rho*].
    """
    if not (0 < rho < 1):
        return np.nan
    s = float(np.clip(share, 0.0, 0.98))
    x = np.zeros((n_boot, n_blocks))
    sd = np.sqrt(1 - rho ** 2)
    for _ in range(40):
        x = rho * x + sd * rng.standard_normal(x.shape)
    P = np.empty((n_boot, n_blocks, n_win))
    for t in range(n_win):
        x = rho * x + sd * rng.standard_normal(x.shape)
        P[:, :, t] = x
    P = np.sqrt(1 - s) * P + np.sqrt(s) * rng.standard_normal(P.shape)
    P = P - P.mean(axis=2, keepdims=True)

    def rk(k):
        a, z = P[:, :, :-k], P[:, :, k:]
        return ((a * z).sum(axis=(1, 2))
                / np.maximum(np.sqrt((a * a).sum(axis=(1, 2))
                                     * (z * z).sum(axis=(1, 2))), 1e-300))
    r1s, r2s = rk(1), rk(2)
    ok = (r1s > 0) & (r2s > 0)
    if ok.sum() < n_boot // 4:
        return np.nan
    rho_star = float(np.mean((r2s[ok] / r1s[ok])))
    return 2 * rho - rho_star


def measure(blocks, u, dt, rng, bias_correct=True):
    p = [np.asarray(z) @ u for z in blocks]
    r, V = acf_pair(p, 3)
    out = dict(r1=float(r[0]), r2=float(r[1]), var=float(V),
               n_blocks=len(p), n_win=int(np.mean([len(x) for x in p])))
    if not (np.isfinite(r[0]) and np.isfinite(r[1]) and r[0] > 0 and r[1] > 0):
        return out
    rho = r[1] / r[0]
    share = float(np.clip(1 - r[0] ** 2 / r[1], 0.0, 0.995))
    out["white_share"] = share
    if bias_correct:
        rho_bc = _bc_rho(rho, share, out["n_blocks"], out["n_win"], dt, rng)
        if np.isfinite(rho_bc):
            rho = rho_bc
    if not (0 < rho < 1):
        return out
    sig2 = (1 - share) * V
    out.update(rho=float(rho), lam=float(-np.log(rho) / dt),
               sigma2=float(sig2), sigma=float(np.sqrt(sig2)),
               D=float(sig2 * (1 - rho ** 2) / (2 * dt)),
               lam_naive=float(-np.log(r[0]) / dt) if 0 < r[0] < 1 else np.nan,
               D_naive=float(V * (1 - r[0] ** 2) / (2 * dt)))
    return out


def log_contrast(ma, mb):
    o = {}
    for k, tag in (("lam", "dlog_lam"), ("D", "dlog_D"), ("sigma2", "dlog_var"),
                   ("lam_naive", "dlog_lam_naive"), ("D_naive", "dlog_D_naive")):
        if k in ma and k in mb and ma[k] > 0 and mb[k] > 0:
            o[tag] = float(np.log(mb[k]) - np.log(ma[k]))
    return o


def run_one(args):
    stem, win_s, group_axes, want_means = args
    feat, meta = C.load_cell(stem, win_s)
    if feat is None:
        return dict(stem=stem, error="no stages")
    Z, evr = C.state_space(feat)
    means = {g: v.reshape(-1, v.shape[-1]).mean(axis=0) for g, v in Z.items()}
    if want_means:
        return dict(stem=stem, means={g: means[g].tolist() for g in means})
    rng = np.random.default_rng(abs(hash((stem, "ax1d", C.SEED))) % (2 ** 31))
    d = next(iter(Z.values())).shape[-1]
    U = rng.standard_normal((N_RANDOM, d))
    U /= np.linalg.norm(U, axis=1, keepdims=True)
    rows = []
    for (a, b) in C.CONTRASTS:
        if a not in Z or b not in Z:
            continue
        Ba, Bb = list(Z[a]), list(Z[b])
        if len(Ba) < 6 or len(Bb) < 6:
            continue
        key = f"{a}>{b}"
        axes = {}
        ui = means[b] - means[a]
        axes["axis_in"] = ui / np.linalg.norm(ui)
        if key in group_axes:
            axes["axis_out"] = np.asarray(group_axes[key])
        for name, u in axes.items():
            ma = measure(Ba, u, win_s, rng)
            mb = measure(Bb, u, win_s, rng)
            row = dict(stem=stem, subject=stem[:5], contrast=key, direction=name,
                       n_a=len(Ba), n_b=len(Bb))
            row.update({f"a_{k}": v for k, v in ma.items()})
            row.update({f"b_{k}": v for k, v in mb.items()})
            row.update(log_contrast(ma, mb))
            rows.append(row)
        # random directions: distribution of the SAME pair
        dl, dd, dv = [], [], []
        for u in U:
            ma = measure(Ba, u, win_s, rng, bias_correct=False)
            mb = measure(Bb, u, win_s, rng, bias_correct=False)
            c_ = log_contrast(ma, mb)
            if "dlog_lam" in c_:
                dl.append(c_["dlog_lam"])
            if "dlog_D" in c_:
                dd.append(c_["dlog_D"])
            if "dlog_var" in c_:
                dv.append(c_["dlog_var"])
        # matched ceiling: two halves of the SAME stage, truth exactly zero
        cl, cd = [], []
        for B in (Ba, Bb):
            h1, h2 = B[0::2], B[1::2]
            if len(h1) < 3 or len(h2) < 3:
                continue
            for u in U[:60]:
                m1 = measure(h1, u, win_s, rng, bias_correct=False)
                m2 = measure(h2, u, win_s, rng, bias_correct=False)
                c_ = log_contrast(m1, m2)
                if "dlog_lam" in c_:
                    cl.append(c_["dlog_lam"])
                if "dlog_D" in c_:
                    cd.append(c_["dlog_D"])
        rows.append(dict(
            stem=stem, subject=stem[:5], contrast=key, direction="random",
            n_a=len(Ba), n_b=len(Bb),
            dlog_lam=float(np.median(dl)) if dl else np.nan,
            dlog_D=float(np.median(dd)) if dd else np.nan,
            dlog_var=float(np.median(dv)) if dv else np.nan,
            dlog_lam_sd=float(np.std(dl, ddof=1)) if len(dl) > 2 else np.nan,
            dlog_D_sd=float(np.std(dd, ddof=1)) if len(dd) > 2 else np.nan,
            n_random_ok=len(dl),
            ceiling_lam_sd=float(np.std(cl, ddof=1)) if len(cl) > 2 else np.nan,
            ceiling_D_sd=float(np.std(cd, ddof=1)) if len(cd) > 2 else np.nan,
            ceiling_lam_med=float(np.median(cl)) if cl else np.nan,
            ceiling_D_med=float(np.median(cd)) if cd else np.nan))
    return dict(stem=stem, rows=rows)


def _worker(a):
    try:
        return run_one(a)
    except Exception:  # noqa: BLE001
        return dict(stem=a[0], error=traceback.format_exc(limit=4))


def summarise(rows):
    """Participant-level summaries per pair and direction set, from the rows in their order."""
    rng = np.random.default_rng(C.SEED)
    res = {}
    for (x, y) in C.CONTRASTS:
        key = f"{x}>{y}"
        out = {}
        for direction in ("axis_out", "axis_in", "random"):
            rr = [r for r in rows if r["contrast"] == key and r["direction"] == direction]
            if not rr:
                continue
            bysub = {}
            for r in rr:
                bysub.setdefault(r["subject"], []).append(r)
            o = dict(n_nights=len(rr), n_subjects=len(bysub))
            for field in ("dlog_lam", "dlog_D", "dlog_var", "dlog_lam_naive",
                          "dlog_D_naive", "a_white_share", "b_white_share",
                          "a_lam", "b_lam", "dlog_lam_sd", "dlog_D_sd",
                          "ceiling_lam_sd", "ceiling_D_sd", "ceiling_lam_med",
                          "ceiling_D_med"):
                v = []
                for s, items in bysub.items():
                    z = [i[field] for i in items
                         if field in i and i[field] is not None
                         and np.isfinite(i[field])]
                    if z:
                        v.append(float(np.mean(z)))
                if v:
                    o[field] = C.paired_summary(v, n_boot=N_BOOT, rng=rng)
            out[direction] = o
        # axis minus random, paired within participant
        for field in ("dlog_lam", "dlog_D", "dlog_var"):
            pa, pr = {}, {}
            for r in rows:
                if r["contrast"] != key or field not in r or r[field] is None \
                        or not np.isfinite(r[field]):
                    continue
                (pa if r["direction"] == "axis_out" else
                 pr if r["direction"] == "random" else {}).setdefault(
                    r["subject"], []).append(r[field])
            v = [float(np.mean(pa[s])) - float(np.mean(pr[s]))
                 for s in pa if s in pr]
            out[f"axis_minus_random|{field}"] = C.paired_summary(v, n_boot=N_BOOT,
                                                                 rng=rng)
        res[key] = out
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--win", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None,
                    help="only the first N recordings (a check run)")
    ap.add_argument("--rows", default=None,
                    help="re-run the summary on this stored row table instead of measuring")
    a = ap.parse_args()

    t0 = time.time()
    if a.rows:
        rows = C.read_table(a.rows)
        print(f"{len(rows)} stored rows from {a.rows}", flush=True)
    else:
        stems = C.cache_stems("SC", a.win)
        if a.limit:
            stems = stems[: a.limit]
        print(f"random-direction two-lag rates: {len(stems)} {LIBRARY} recordings at "
              f"{a.win} s", flush=True)
        means = {}
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            futs = {ex.submit(_worker, (s, a.win, {}, True)): s for s in stems}
            for f in as_completed(futs):
                r = f.result()
                if "means" in r:
                    means[r["stem"]] = r["means"]
        print(f"  axes from {len(means)} recordings ({(time.time()-t0)/60:.1f} min)",
              flush=True)

        def loo(stem, x, y):
            v = [np.asarray(m[y]) - np.asarray(m[x]) for s, m in means.items()
                 if s != stem and x in m and y in m]
            if len(v) < 5:
                return None
            v = np.stack([u / np.linalg.norm(u) for u in v])
            u = v.mean(axis=0)
            return (u / np.linalg.norm(u)).tolist()

        rows = []
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            futs = {}
            for s in stems:
                ga = {f"{x}>{y}": loo(s, x, y) for (x, y) in C.CONTRASTS}
                ga = {k: v for k, v in ga.items() if v is not None}
                futs[ex.submit(_worker, (s, a.win, ga, False))] = s
            for k, f in enumerate(as_completed(futs), 1):
                r = f.result()
                if "error" in r:
                    continue
                rows += r["rows"]
                if k % 25 == 0 or k == len(stems):
                    print(f"  {k}/{len(stems)}  {(time.time()-t0)/60:.1f} min", flush=True)

    res = summarise(rows)
    dep = dict(win_s=a.win, library=LIBRARY,
               n_random=N_RANDOM, n_boot=N_BOOT, n_bias_boot=N_BC, seed=C.SEED,
               estimator=("rho = r(2)/r(1) with a matched bootstrap bias "
                          "correction; D = sigma_signal^2 (1-rho^2)/(2 dt); "
                          "immune to any additive component uncorrelated at dt"),
               contrasts=res)
    p = paths.out(*OUT_JSON)
    Path(p).write_text(json.dumps(dep, indent=1), encoding="utf-8", newline="\n")
    if not a.rows:
        cols = sorted({k for d in rows for k in d} - {"contrast"})
        cols.insert(cols.index("stem"), "contrast")      # column order of the stored table
        with open(paths.out(*OUT_ROWS), "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, lineterminator="\n")
            w.writeheader()
            w.writerows(rows)

    print("\n" + "=" * 112)
    for key, out in res.items():
        for direction in ("axis_out", "random"):
            o = out.get(direction)
            if not o:
                continue

            def g(f):
                s_ = o.get(f, {})
                return (s_.get("log_median", float("nan")),
                        np.log(s_["fold_lo"]) if "fold_lo" in s_ else float("nan"),
                        np.log(s_["fold_hi"]) if "fold_hi" in s_ else float("nan"),
                        s_.get("sign_p", float("nan")), s_.get("n", 0))
            L = g("dlog_lam"); D = g("dlog_D"); V = g("dlog_var")
            print(f"{key:>9} {direction:>9} n={L[4]:>3}  "
                  f"dlogL={L[0]:+.3f} [{L[1]:+.3f},{L[2]:+.3f}] P={L[3]:.2e}   "
                  f"dlogD={D[0]:+.3f} [{D[1]:+.3f},{D[2]:+.3f}] P={D[3]:.2e}   "
                  f"dlogVar={V[0]:+.3f}")
        for f in ("dlog_lam", "dlog_D"):
            s_ = out.get(f"axis_minus_random|{f}", {})
            if "log_median" in s_:
                print(f"{'':>9} axis-random {f}: {s_['log_median']:+.3f} "
                      f"[{np.log(s_['fold_lo']):+.3f},{np.log(s_['fold_hi']):+.3f}] "
                      f"P={s_['sign_p']:.3f}  (n={s_['n']})")
        o = out.get("random", {})
        for f, c_ in (("dlog_lam_sd", "ceiling_lam_sd"), ("dlog_D_sd", "ceiling_D_sd")):
            if "log_median" in o.get(f, {}) and "log_median" in o.get(c_, {}):
                print(f"{'':>9} across-direction SD {f}={o[f]['log_median']:.3f} "
                      f"vs same-stage ceiling {o[c_]['log_median']:.3f}")
    print(f"\nwrote {p} ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
