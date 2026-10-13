# -*- coding: utf-8 -*-
r"""Sleep-stage contrasts of directional spread and apparent relaxation rate in the standardized
four-dimensional sleep state (153 Sleep-EDF cassette nights, 78 participants; windows 2 s and 1 s).

Per recording and stage (at least 5 bouts), in the d = 4 state space of the recording
(sleep_common.state_space):
  * directional spread `width`: the median, over the directions from this stage's mean towards
    every other stage's mean, of the standard deviation of the bout-centred projections
    (sleep_common.escape_widths); `width_iso` = sqrt(tr C / d);
  * VAR(1) on the non-overlapping windows of each bout (no pair across bouts, each bout centred on
    its own mean): the apparent rate `lam_mean_1` = -(1/4) sum ln|eig M| / dt and the diffusion
    from the innovation covariance; the bootstrap bias-corrected fit (`_bc`, 200 draws); the
    lag-2 instrument (`_iv`); with the instrument-noise covariance of noise_cache.py, the
    noise-corrected fits (`_nc`) and their matched bootstrap correction (`_ncbc`, 120 draws);
  * a held-out discrete Lyapunov check: (M, Q) fitted on the odd bouts, the implied stationary
    covariance compared with the lag-0 covariance of the even bouts.

Contrasts (`contrast_stat`): for each adjacent pair Wake>N1, N1>N2, N2>N3, N3>N4 and each
quantity, nights are averaged within participant on the log scale, and the participant log
ratios (shallower minus deeper) give the fold exp(median), a percentile-bootstrap interval
(4,000 participant resamples), the sign test and the paired two-sided Wilcoxon test; for width,
lam_slow_ncbc and logD_ncbc also a within-night stage-label permutation null (5,000 draws;
two-sided P = (1 + #|null| >= |observed|) / (1 + draws)).  One generator seeded 20260922 per
window serves all contrasts in a fixed order.  Stage codes N1-N4 stand for the R&K stages S1-S4.

Reported values (2-s window unless stated): directional spread Wake/S1 1.19 [1.11, 1.27] (n 64),
S1/S2 0.99 [0.93, 1.06] (76), S2/S3 1.20 [1.15, 1.28] (51), S3/S4 1.09 [1.03, 1.16] (20), with the
label-permutation P (Fig. 4c, Supplementary Table 8); apparent rate deeper/shallower =
1 / fold of `lam_mean_1`: S1/Wake 1.25 [1.18, 1.33], S2/S1 1.24 [1.20, 1.28] (Fig. 5b); the 1-s
spread ratios 1.10, 0.96, 1.14, 1.09 (Methods); the two-lag IV and surrogate-subtraction rows of
Fig. 5e / Supplementary Table 11 (`lam_mean_iv`, `lam_mean_ncbc`; rate_corrections.py); and the
Benjamini-Hochberg adjustment over the eight primary 2-s contrasts (Methods; `primary_contrasts_bh`).

Random streams.  The per-recording bootstrap draws (`_bc`, `_ncbc`) use a generator seeded with
Python's hash of (<stem>, <win_s>, 20260922), and the contrasts are computed over the cells in the order in which
the worker processes return them.
`--cells` re-runs the contrast step on a stored per-cell table in its stored row order, which
reproduces the stored contrasts exactly.

Inputs   the sleep window and noise caches (window_cache.py, noise_cache.py) in
         paths.cache("sleep_states") (NDIF_SLEEP_STATES_CACHE); or, with --cells, a stored table.
Outputs  figure_reproduction/inputs/sleep_stage_contrasts.json   contrasts and per-stage levels
         figure_reproduction/inputs/sleep_stage_cells.csv        one row per night, stage, window
         analyses/sleep_stage_contrasts/out/stage_cell_covariances.npz   C0, Sigma_x, M per cell
         analyses/sleep_stage_contrasts/out/primary_contrasts_bh.json    Benjamini-Hochberg
         (below NDIF_OUT when it is set; --cells writes only the two JSON files)
Runtime  about 5 min with 4 workers for both windows; --cells about 4 min on one core
         (the permutation nulls).

    python analyses/sleep_stage_contrasts/stage_contrasts.py --win 2.0 1.0 --workers 4
    python analyses/sleep_stage_contrasts/stage_contrasts.py --cells figure_reproduction/inputs/sleep_stage_cells.csv
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
from scipy import linalg as sla                                       # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))
from analyses.common import paths  # noqa: E402
import sleep_common as C  # noqa: E402

LIBRARY = "cassette"
N_BOOT = 4000
N_PERM = 5000
P_BOUND_WILCOXON = 1e-4      # Wilcoxon P below this are reported as < 0.0001 (Supplementary Table 8)

OUT_CONTRASTS = ("figure_reproduction", "inputs", "sleep_stage_contrasts.json")
OUT_CELLS = ("figure_reproduction", "inputs", "sleep_stage_cells.csv")
OUT_COV = ("analyses", "sleep_stage_contrasts", "out", "stage_cell_covariances.npz")
OUT_BH = ("analyses", "sleep_stage_contrasts", "out", "primary_contrasts_bh.json")


def heldout_lyapunov(blocks, dt, d, R=None):
    """Fit (M, Q) on the odd bouts, solve Sigma = M Sigma M' + Q exactly, and
    compare the implied stationary covariance with the lag-0 covariance measured
    on the EVEN bouts.

    The in-sample version of this check is vacuous: with Q = C0 - M C0 M' the
    solution is C0 by construction.  Out of sample it is a real test of the
    discrete Lyapunov relation.  Reported for the one-step fit and, where R is
    available, for the noise-corrected fit against C0_even - R.
    """
    out = dict(lyap_oos_tr_ratio=np.nan, lyap_oos_logdet=np.nan,
               lyap_oos_tr_ratio_nc=np.nan)
    odd, even = blocks[0::2], blocks[1::2]
    if len(odd) < 3 or len(even) < 3:
        return out
    Ce = C.lagged_cov(even, kmax=0)[0][0]
    f = C.fit_ou(odd, dt=dt, lag=1)
    if f is None:
        return out
    Q = 0.5 * (f.Sigma_eps + f.Sigma_eps.T)
    try:
        Sig = sla.solve_discrete_lyapunov(f.M, Q)
        out["lyap_oos_tr_ratio"] = float(np.trace(Sig) / max(np.trace(Ce), 1e-300))
        out["lyap_oos_logdet"] = float((np.linalg.slogdet(Sig)[1]
                                        - np.linalg.slogdet(Ce)[1]) / d)
    except Exception:  # noqa: BLE001
        pass
    if R is not None:
        try:
            Co, C1o = C.lagged_cov(odd, kmax=1)[0][:2]
            Mn, Qn, Sx, alpha = C.nc_propagator(Co, C1o, R, dt, d)
            Sig2 = sla.solve_discrete_lyapunov(Mn, Qn)
            Sxe, _ = C._shrink_to_pd(Ce, alpha * R)
            out["lyap_oos_tr_ratio_nc"] = float(np.trace(Sig2)
                                                / max(np.trace(Sxe), 1e-300))
        except Exception:  # noqa: BLE001
            pass
    return out


def run_one(args):
    stem, win_s = args
    feat, meta = C.load_cell(stem, win_s)
    if feat is None:
        return dict(stem=stem, error="no usable stages")
    Z, evr, J = C.state_space(feat, with_jacobian=True)
    dt = win_s
    rng = np.random.default_rng((abs(hash((stem, win_s, C.SEED))) % (2 ** 31)))
    rows = []
    for g in Z:
        blocks = list(Z[g])
        w, dirs = C.escape_widths(Z, g)
        A = np.vstack(blocks)
        Cc = np.cov(A - A.mean(axis=0), rowvar=False)
        R = C.load_noise_R(stem, g, win_s, J)
        fit = C.fit_cell(blocks, dt, d=C.D_STATE, n_boot=200, rng=rng, R=R)
        if fit is None:
            continue
        fit.update(heldout_lyapunov(blocks, dt, C.D_STATE, R=R))
        fit.update(stem=stem, subject=stem[:5], win_s=win_s, stage=g,
                   width=w, width_iso=float(np.sqrt(np.trace(Cc) / Cc.shape[0])),
                   n_dirs=len(dirs), evr1=float(evr[0]))
        fit["dirs"] = dirs
        rows.append(fit)
    return dict(stem=stem, rows=rows)


def _worker(a):
    try:
        return run_one(a)
    except Exception:  # noqa: BLE001
        return dict(stem=a[0], error=traceback.format_exc(limit=4))


# ------------------------------------------------------------------ inference
def subject_table(cells, key, log=True):
    """{subject: {stage: value}} averaging nights on the log scale."""
    acc = {}
    for c in cells:
        v = c.get(key)
        if v is None or not np.isfinite(v):
            continue
        if log:
            if v <= 0:
                continue
            v = np.log(v)
        acc.setdefault(c["subject"], {}).setdefault(c["stage"], []).append(v)
    return {s: {g: float(np.mean(v)) for g, v in d.items()} for s, d in acc.items()}


def contrast_stat(cells, key, a, b, log=True, rng=None, n_perm=N_PERM):
    """Paired participant-level contrast a vs b, with a within-night stage-label
    permutation null size-matched to the observed sample."""
    tab = subject_table(cells, key, log=log)
    lr = [tab[s][a] - tab[s][b] for s in tab if a in tab[s] and b in tab[s]]
    out = C.paired_summary(lr, n_boot=N_BOOT, rng=rng)
    if not lr or n_perm <= 0:
        return out
    rng = np.random.default_rng(rng)
    bynight = {}
    for c in cells:
        v = c.get(key)
        if v is None or not np.isfinite(v) or (log and v <= 0):
            continue
        bynight.setdefault(c["stem"], []).append(
            (c["subject"], c["stage"], np.log(v) if log else v))
    null = []
    for _ in range(n_perm):
        perm = []
        for stem, items in bynight.items():
            stages = [x[1] for x in items]
            rng.shuffle(stages)
            perm += [dict(stem=stem, subject=items[i][0], stage=stages[i],
                          _v=items[i][2]) for i in range(len(items))]
        t = {}
        for p in perm:
            t.setdefault(p["subject"], {}).setdefault(p["stage"], []).append(p["_v"])
        t = {s: {g: float(np.mean(v)) for g, v in d.items()} for s, d in t.items()}
        d_ = [t[s][a] - t[s][b] for s in t if a in t[s] and b in t[s]]
        null.append(np.median(d_) if d_ else np.nan)
    null = np.asarray(null, float)
    null = null[np.isfinite(null)]
    obs = out.get("log_median", np.nan)
    if len(null) and np.isfinite(obs):
        out["perm_p_two_sided"] = float((1 + np.sum(np.abs(null) >= abs(obs)))
                                        / (1 + len(null)))
        out["perm_null_fold_mean"] = float(np.exp(np.mean(null)))
        out["perm_n"] = int(len(null))
    return out


def summarise(cells):
    """Contrasts and per-stage levels of one window, from its cells in their order."""
    rng = np.random.default_rng(C.SEED)
    res = dict(n_cells=len(cells), n_nights=len({c["stem"] for c in cells}),
               n_subjects=len({c["subject"] for c in cells}), contrasts={})
    keys = [("width", True), ("width_iso", True),
            ("lam_mean_1", True), ("lam_mean_bc", True), ("lam_mean_iv", True),
            ("lam_mean_nc", True), ("lam_mean_ncbc", True),
            ("lam_slow_ncbc", True), ("lam_slow_bc", True), ("trSx", True)]
    perm_keys = ("width", "lam_slow_ncbc", "logD_ncbc")
    for key, lg in keys:
        for (x, y) in C.CONTRASTS:
            res["contrasts"][f"{key}|{x}>{y}"] = contrast_stat(
                cells, key, x, y, log=lg, rng=rng,
                n_perm=N_PERM if key in perm_keys else 0)
    for key in ("logD_1", "logD_bc", "logD_iv", "logD_nc", "logD_ncbc",
                "logDtr_ncbc", "white_share_iv", "noise_share_measured",
                "nc_alpha"):
        for (x, y) in C.CONTRASTS:
            res["contrasts"][f"{key}|{x}>{y}"] = contrast_stat(
                cells, key, x, y, log=False, rng=rng,
                n_perm=N_PERM if key in perm_keys else 0)
    # per-stage levels
    res["per_stage"] = {}
    for g in C.STAGES:
        cc = [c for c in cells if c["stage"] == g]
        if not cc:
            continue

        def nm(k):
            v = [c.get(k) for c in cc]
            v = [x for x in v if x is not None and np.isfinite(x)]
            return float(np.median(v)) if v else float("nan")
        res["per_stage"][g] = dict(
            n_cells=len(cc), n_subjects=len({c["subject"] for c in cc}),
            width=nm("width"), width_iso=nm("width_iso"),
            lam_mean_1=nm("lam_mean_1"), lam_mean_bc=nm("lam_mean_bc"),
            lam_mean_iv=nm("lam_mean_iv"), lam_mean_nc=nm("lam_mean_nc"),
            lam_mean_ncbc=nm("lam_mean_ncbc"),
            lam_slow_ncbc=nm("lam_slow_ncbc"),
            tau_ncbc=float(1.0 / nm("lam_mean_ncbc")),
            tau_slow_ncbc=float(1.0 / nm("lam_slow_ncbc")),
            logD_1=nm("logD_1"), logD_bc=nm("logD_bc"), logD_iv=nm("logD_iv"),
            logD_nc=nm("logD_nc"), logD_ncbc=nm("logD_ncbc"),
            white_share_iv=nm("white_share_iv"),
            noise_share_measured=nm("noise_share_measured"),
            nc_alpha=nm("nc_alpha"), nc_bias_fro=nm("nc_bias_fro"),
            lyap_oos_tr_ratio=nm("lyap_oos_tr_ratio"),
            lyap_oos_tr_ratio_nc=nm("lyap_oos_tr_ratio_nc"),
            frac_iv_stable=float(np.mean([np.isfinite(c.get("lam_mean_iv",
                                                            np.nan))
                                          for c in cc])),
            frac_ncbc_stable=float(np.mean(
                [np.isfinite(c.get("lam_mean_ncbc", np.nan)) for c in cc])),
            frac_sx_pd=float(np.mean([bool(c.get("sx_pd")) for c in cc])))
    return res


def report(res, win_s):
    print(f"\n  --- directional spread at {win_s} s (participant-level fold, shallower/deeper) ---")
    for (x, y) in C.CONTRASTS:
        r = res["contrasts"][f"width|{x}>{y}"]
        print(f"   width {x:>4}>{y:<3} n={r.get('n',0):>3} fold={r.get('fold',float('nan')):.3f} "
              f"[{r.get('fold_lo',float('nan')):.3f},{r.get('fold_hi',float('nan')):.3f}] "
              f"sign_p={r.get('sign_p',float('nan')):.2e} "
              f"perm_p={r.get('perm_p_two_sided',float('nan')):.4f}")
    print(f"  --- lambda / D at {win_s} s (deeper vs shallower) ---")
    for (x, y) in C.CONTRASTS:
        for key in ("lam_mean_1", "lam_slow_ncbc", "lam_mean_ncbc", "lam_mean_bc",
                    "logD_ncbc", "logD_bc"):
            r = res["contrasts"][f"{key}|{x}>{y}"]
            if key.startswith("lam"):
                d = -r.get("log_median", np.nan)     # deeper - shallower
                lo, hi = -np.log(r.get("fold_hi", np.nan)), -np.log(r.get("fold_lo", np.nan))
            else:
                d = -r.get("log_median", np.nan)
                lo, hi = -r.get("hi", np.nan), -r.get("lo", np.nan)
            print(f"   {key:>13} {x:>4}->{y:<3} n={r.get('n',0):>3} "
                  f"dlog={d:+.3f} [{lo:+.3f},{hi:+.3f}] "
                  f"sign_p={r.get('sign_p',float('nan')):.2e}")
    print(f"  --- per-stage levels at {win_s} s ---")
    for g, v in res["per_stage"].items():
        print(f"   {g:>4} n={v['n_cells']:>3} width={v['width']:.3f} "
              f"lam_bc={v['lam_mean_bc']:.3f} lam_ncbc={v['lam_mean_ncbc']:.3f} "
              f"tau_ncbc={v['tau_ncbc']:.2f}s logD_bc={v['logD_bc']:+.2f} "
              f"logD_ncbc={v['logD_ncbc']:+.2f} "
              f"Rshare={v['noise_share_measured']:.2f} "
              f"wshare_iv={v['white_share_iv']:.2f} alpha={v['nc_alpha']:.2f} "
              f"lyapOOS={v['lyap_oos_tr_ratio']:.2f}/"
              f"{v['lyap_oos_tr_ratio_nc']:.2f} "
              f"ok={v['frac_ncbc_stable']:.2f}")


def primary_contrasts_bh(dep, win="2"):
    """Benjamini-Hochberg adjustment over the eight primary 2-s contrasts of Fig. 5b.

    Family: the four directional-spread contrasts (within-night stage-label permutation P) and the
    four apparent-rate contrasts (paired two-sided Wilcoxon P of `lam_mean_1`).  P values are
    entered at their reporting bounds: a permutation P cannot fall below 1/(draws + 1), and a
    Wilcoxon P below 1e-4 is entered as 1e-4.  The adjustment of the unbounded P values is
    recorded beside it.
    """
    from scipy.stats import false_discovery_control
    con = dep["by_window"][win]["contrasts"]
    fam = []
    for a, b in C.CONTRASTS:
        r = con[f"width|{a}>{b}"]
        fam.append(dict(contrast=f"{a}/{b}", key=f"width|{a}>{b}",
                        quantity="directional spread", test="within-night stage-label permutation",
                        p=r["perm_p_two_sided"], bound=1.0 / (r["perm_n"] + 1)))
    for a, b in C.CONTRASTS:
        r = con[f"lam_mean_1|{a}>{b}"]
        fam.append(dict(contrast=f"{a}/{b}", key=f"lam_mean_1|{a}>{b}",
                        quantity="apparent rate", test="paired two-sided Wilcoxon",
                        p=r["wilcoxon_p"], bound=P_BOUND_WILCOXON))
    p_raw = np.array([f["p"] for f in fam], float)
    p_in = np.maximum(p_raw, [f["bound"] for f in fam])
    q = false_discovery_control(p_in, method="bh")
    q_raw = false_discovery_control(p_raw, method="bh")
    for f, pe, qq, qr in zip(fam, p_in, q, q_raw):
        f.update(p_entered=float(pe), p_adjusted=float(qq), p_adjusted_unbounded=float(qr))
    below = [f for f in fam if f["p_adjusted"] < 0.05]
    return dict(
        window_s=float(win), method="Benjamini-Hochberg (scipy.stats.false_discovery_control)",
        family=("four directional-spread contrasts (label-permutation P) and four apparent-rate "
                "contrasts (Wilcoxon P), 2-s window"),
        p_entered=("at the reporting bounds: permutation P >= 1/(draws + 1); Wilcoxon P below "
                   "1e-4 entered as 1e-4"),
        n_contrasts=len(fam), n_adjusted_below_0_05=len(below),
        max_adjusted_p_below_0_05=float(max(f["p_adjusted"] for f in below)) if below else None,
        contrasts=fam)


def write_cells(all_cells):
    flat = []
    for c in all_cells:
        d = {k: v for k, v in c.items()
             if k not in ("dirs", "C0", "Sx", "M_ncbc", "modes_1", "modes_bc",
                          "modes_iv", "modes_nc", "modes_ncbc")}
        for j, m in enumerate(c.get("modes_ncbc", [])):
            d[f"mode_ncbc_{j}"] = m
        for j, m in enumerate(c.get("modes_bc", [])):
            d[f"mode_bc_{j}"] = m
        flat.append(d)
    if flat:
        cols = sorted({k for d in flat for k in d})
        with open(paths.out(*OUT_CELLS), "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, lineterminator="\n")
            w.writeheader()
            w.writerows(flat)
    payload = {}
    for c in all_cells:
        k = f"{c['stem']}|{c['stage']}|{c['win_s']:g}"
        payload[f"C0|{k}"] = np.asarray(c["C0"])
        if "Sx" in c:
            payload[f"Sx|{k}"] = np.asarray(c["Sx"])
        if "M_ncbc" in c:
            payload[f"M|{k}"] = np.asarray(c["M_ncbc"])
    np.savez_compressed(paths.out(*OUT_COV), **payload)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--win", type=float, nargs="+", default=[2.0, 1.0])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None,
                    help="only the first N recordings (a check run)")
    ap.add_argument("--cells", default=None,
                    help="re-run the contrasts on this stored per-cell table instead of fitting")
    a = ap.parse_args()

    dep = dict(library=LIBRARY, windows=a.win, n_boot=N_BOOT, n_perm=N_PERM, seed=C.SEED,
               by_window={})
    stored = C.read_table(a.cells) if a.cells else None
    all_cells = []
    t0 = time.time()
    for win_s in a.win:
        if stored is not None:
            cells = [c for c in stored if c["win_s"] == win_s]
            print(f"\n=== window {win_s} s : {len(cells)} stored cells ===", flush=True)
        else:
            stems = C.cache_stems("SC", win_s)
            if a.limit:
                stems = stems[: a.limit]
            print(f"\n=== window {win_s} s : {len(stems)} {LIBRARY} recordings ===",
                  flush=True)
            cells = []
            with ProcessPoolExecutor(max_workers=a.workers) as ex:
                futs = {ex.submit(_worker, (s, win_s)): s for s in stems}
                for k, f in enumerate(as_completed(futs), 1):
                    r = f.result()
                    if "error" in r:
                        if "no usable" not in str(r["error"]):
                            print(f"  {r['stem']} ERROR "
                                  f"{str(r['error']).splitlines()[-1][:90]}", flush=True)
                        continue
                    cells += r["rows"]
                    if k % 25 == 0 or k == len(stems):
                        print(f"  {k}/{len(stems)}  {(time.time()-t0)/60:.1f} min", flush=True)
            for c in cells:
                c["library"] = LIBRARY
        all_cells += cells
        res = summarise(cells)
        dep["by_window"][f"{win_s:g}"] = res
        report(res, win_s)

    p = paths.out(*OUT_CONTRASTS)
    Path(p).write_text(json.dumps(dep, indent=1) + "\n", encoding="utf-8", newline="\n")
    written = [p]
    if "2" in dep["by_window"]:
        pb = paths.out(*OUT_BH)
        Path(pb).write_text(json.dumps(primary_contrasts_bh(dep), indent=1) + "\n",
                            encoding="utf-8", newline="\n")
        written.append(pb)
    if stored is None:
        write_cells(all_cells)
        written += [paths.out(*OUT_CELLS), paths.out(*OUT_COV)]
    print("\nwrote " + ", ".join(written) + f"  ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
