"""Recount of the equilibria of the mixture drift field under label-free regime assignments, per record.

The reference count of a record is the number of stable zeros of

    b(x) = sum_k w_k(x) A_k (x - mu_k),

where w_k(x) are Gaussian responsibilities of the condition means and covariances (mu_k, Sigma_k)
(landscape_lib.multilinear_drift).  The count is recomputed on the same records, in the same state
spaces and with the same counter (landscape_lib.find_equilibria, 200 starts), changing only the regime
assignment:

  sup    the condition labels (the reference assignment), recomputed from the count cache;
  u1     a full-covariance Gaussian mixture fitted to x alone with K held at the number of conditions,
         segment-level assignment (each segment to the component with the largest summed
         log-density), so that only the assignment changes ("labels hidden" in Fig. 2f);
  u1w    the same mixture with a sample-level assignment (regimes may switch within a segment);
  u2     a Gaussian mixture with K chosen by BIC over 1..6, segment-level assignment;
  u3     a Gaussian hidden Markov model fitted to x alone (Baum-Welch, Viterbi path), K at the number of
         conditions;
  perm   condition labels permuted across segments (segments per condition preserved), 20 draws (10
         on records above the fitting cap), 60 starts;
  km     the label-free one-dimensional Kramers-Moyal drift on the first principal axis of the pooled
         state series (no mixture, no labels; "model-free 1-D drift" in Fig. 2f), and the same
         estimator on the condition-difference axis.  A record whose drift curve cannot be resolved is
         reported as unresolved.

Regimes are built by cutting each segment at assignment changes and fitting
landscape_lib.fit_linear_regime to the resulting sub-segments, so increments never cross a segment
boundary or a regime change; with the condition labels this is the reference construction.  Mixture
fits use at most 20,000 samples (whole segments, seeded) and the HMM at most 4,000; assignment and
counting always use every sample.

Input   the count cache (paths.cache("count"); NDIF_COUNT_CACHE): <cohort>/_meta.json with dt, K and the
        records, and <cohort>/<record>.npz with Z, seg, lab (state series, segment index, condition) and,
        for ds003838 and eegmmidb, Zdist, segdist, labdist (the overlapping-window series used by the
        Kramers-Moyal arm).
Output  one JSON row per record in paths.cache("count_rows")/<cohort>/<record>.json (NDIF_COUNT_ROWS_CACHE;
        records already written are skipped), read by readout_table.py.
Reported in  Fig. 2f (labels used, labels hidden, model-free 1-D drift on the 64 ds003838 recordings);
        the separation of the reference regimes (sup) is the separation of Fig. 2c.

Runtime about 25 min per cohort group on 6 workers (awake cohorts and sleep nights in two processes).

    python analyses/landscape_readouts/count_recount.py --cohort all --workers 6
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse                                                       # noqa: E402
import json                                                           # noqa: E402
import sys                                                            # noqa: E402
import time                                                           # noqa: E402
from pathlib import Path                                              # noqa: E402

import numpy as np                                                    # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from analyses.common import paths                                     # noqa: E402
from analyses.landscape_readouts import landscape_lib as LL           # noqa: E402
from analyses.landscape_readouts.landscape_lib import (               # noqa: E402
    MIN_COUNT_KM, N_BINS_KM, count_downcrossings, count_field, kramers_moyal_segments,
    regimes_from_assignment, seg_bounds)

SEED = 20260921
N_PERM = 20               # label permutations per record
HMM_MAX_N = 4000          # whole segments kept, seeded, for the HMM fit when a record is longer
FIT_MAX_N = 20000         # the same cap for the mixture fits; assignment and counting use every sample

# per-cohort one-dimensional Kramers-Moyal settings; "reference" marks the settings of the cohort's
# reference pipeline, the fMRI and sleep settings are declared here
KM_CFG = {
    "ds003838": dict(series="dist", dt=0.5, lag=2, reference=True),
    "eegmmidb": dict(series="dist", dt=0.5, lag=2, reference=True),
    "meg":      dict(series="dyn",  dt=1.0, lag=1, reference=True),
    "fmri":     dict(series="dyn",  dt=0.72, lag=1, reference=False),
    "sleep":    dict(series="dyn",  dt=1.0, lag=1, reference=False),
}


def log(m):
    print(time.strftime("[%H:%M:%S] ") + str(m), flush=True)


# --------------------------------------------------------------------------- regime assignments
def seg_assign_from_gmm(Z, seg, g):
    """Segment-level assignment: each segment goes to the component maximising its summed log-density."""
    lp = g._estimate_weighted_log_prob(Z) if hasattr(g, "_estimate_weighted_log_prob") else None
    if lp is None:
        lp = np.log(np.maximum(g.predict_proba(Z), 1e-300))
    starts, lens = seg_bounds(seg)
    tot = np.add.reduceat(lp, starts, axis=0)
    return np.repeat(np.argmax(tot, axis=1), lens)


def _sep_of(regs):
    """Mahalanobis separation of the two most distant regime centres under the mean regime covariance."""
    if len(regs) < 2:
        return None
    S = np.mean([r.Sigma for r in regs], axis=0)
    best = 0.0
    for i in range(len(regs)):
        for j in range(i + 1, len(regs)):
            d = regs[i].mu - regs[j].mu
            best = max(best, float(np.sqrt(max(d @ np.linalg.solve(S, d), 0.0))))
    return best


def fit_sample(Z, seg, seed=SEED, cap=FIT_MAX_N):
    """Seeded subsample of whole segments used to fit a mixture when a record is very long.

    Only the fitting sample is reduced; every segment is still assigned by the fitted model and every
    increment still enters the regime fits and the count (sleep nights only)."""
    if len(Z) <= cap:
        return np.arange(len(Z)), False
    starts, lens = seg_bounds(seg)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(starts))
    keep, n = [], 0
    for i in order:
        keep.append(np.arange(starts[i], starts[i] + lens[i])); n += lens[i]
        if n >= cap:
            break
    return np.sort(np.concatenate(keep)), True


def gmm_assign(Z, K, seed=SEED, n_init=10, fit_idx=None):
    from sklearn.mixture import GaussianMixture
    Zf = Z if fit_idx is None else Z[fit_idx]
    g = GaussianMixture(n_components=K, covariance_type="full", n_init=n_init,
                        reg_covar=1e-6, random_state=seed, max_iter=500).fit(Zf)
    return g.predict(Z), g


def gmm_bic_assign(Z, kmax=6, seed=SEED, n_init=5, fit_idx=None):
    from sklearn.mixture import GaussianMixture
    Zf = Z if fit_idx is None else Z[fit_idx]
    best, bk, bics = None, None, {}
    for k in range(1, kmax + 1):
        try:
            g = GaussianMixture(n_components=k, covariance_type="full", n_init=n_init,
                                reg_covar=1e-6, random_state=seed, max_iter=500).fit(Zf)
        except Exception:  # noqa: BLE001
            continue
        b = float(g.bic(Zf))
        bics[k] = b
        if best is None or b < bics[bk]:
            best, bk = g, k
    return best.predict(Z), bk, bics


def hmm_assign(Z, seg, K, seed=SEED):
    """Viterbi path of a Gaussian HMM fitted to x alone; whole segments are kept (seeded) when the record
    exceeds HMM_MAX_N samples, and the remaining samples are assigned by emission likelihood."""
    idx, sub = fit_sample(Z, seg, seed=seed, cap=HMM_MAX_N)
    m = LL.fit_gaussian_hmm(Z[idx], K, n_init=3, n_iter=120, seed=seed)
    path_sub = m.predict(Z[idx])
    if not sub:
        return path_sub, dict(subsampled=False, n_used=int(len(idx)))
    lb = np.column_stack([LL.log_gauss(Z, m.means_[k], m.covars_[k]) for k in range(K)])
    path = np.argmax(lb, axis=1)
    path[idx] = path_sub
    return path, dict(subsampled=True, n_used=int(len(idx)), n_total=int(len(Z)))


# --------------------------------------------------------------------------- model-free 1-D arm
def km_row(X1d_segments, dt, lag):
    segs = [s for s in X1d_segments if len(s) > lag]
    if not segs or sum(len(s) - lag for s in segs) < 5 * MIN_COUNT_KM:
        return dict(resolved=False, reason="no segment longer than the lag", n_wells=None)
    c, D1, D2, cnt = kramers_moyal_segments(segs, dt=dt, lag=lag, n_bins=N_BINS_KM)
    valid = cnt >= MIN_COUNT_KM
    nv = int(valid.sum())
    wells = int(count_downcrossings(D1, valid))
    slope = float(np.polyfit(c[valid], D1[valid], 1)[0]) if nv >= 5 else float("nan")
    return dict(resolved=bool(nv >= 5 and wells >= 1), n_wells=wells, n_valid_bins=nv,
                n_increments=int(sum(len(s) - lag for s in segs)),
                restoring_slope=(slope if np.isfinite(slope) else None),
                lambda_s=(-slope if np.isfinite(slope) else None),
                reason=(None if nv >= 5 and wells >= 1 else
                        ("too few resolvable bins" if nv < 5 else "no down-crossing of D1")))


# --------------------------------------------------------------------------- one record
def one_record(args):
    cache, cohort, rec, dt, K, kmcfg = args
    t0 = time.time()
    z = np.load(os.path.join(cache, cohort, f"{rec}.npz"))
    Z = z["Z"].astype(float); seg = z["seg"]; lab = z["lab"].astype(int)
    row = dict(cohort=cohort, record=rec, n=int(len(Z)), d=int(Z.shape[1]),
               n_seg=int(seg.max() + 1), K_reference_pipeline=int(K), dt=float(dt))

    fit_idx, fit_sub = fit_sample(Z, seg)
    row["fit_subsample"] = dict(used=bool(fit_sub), n_fit=int(len(fit_idx)), n_total=int(len(Z)))

    def arm(asg, extra=None):
        regs, info = regimes_from_assignment(Z, seg, asg, dt)
        out = count_field(regs)
        out.update(info)
        out["separation_mahalanobis"] = _sep_of(regs)
        out["occupancy"] = [float(np.mean(asg == k)) for k in sorted(set(asg.tolist()))]
        out["adjusted_rand_vs_labels"] = _ari(lab, asg)
        if extra:
            out.update(extra)
        return out

    # ---- sup: the condition labels, recomputed
    row["sup"] = arm(lab)

    # ---- u1: Gaussian mixture on x alone, K held at the number of conditions, segment-level assignment
    try:
        _, g1 = gmm_assign(Z, K, n_init=(5 if fit_sub else 10), fit_idx=fit_idx)
        a1 = seg_assign_from_gmm(Z, seg, g1)
        row["u1_gmm_Kfixed_segment"] = arm(a1, dict(level="segment"))
    except Exception as e:  # noqa: BLE001
        row["u1_gmm_Kfixed_segment"] = dict(error=repr(e))

    # ---- u1w: the same mixture, assignment allowed to switch within a segment
    try:
        a1w, _ = gmm_assign(Z, K, n_init=(5 if fit_sub else 10), fit_idx=fit_idx)
        row["u1w_gmm_Kfixed_sample"] = arm(a1w, dict(level="sample"))
    except Exception as e:  # noqa: BLE001
        row["u1w_gmm_Kfixed_sample"] = dict(error=repr(e))

    # ---- u2: Gaussian mixture with K by BIC on x alone, segment-level assignment
    try:
        _, k2, bics = gmm_bic_assign(Z, n_init=(3 if fit_sub else 5), fit_idx=fit_idx)
        from sklearn.mixture import GaussianMixture
        g2 = GaussianMixture(n_components=k2, covariance_type="full", n_init=5, reg_covar=1e-6,
                             random_state=SEED, max_iter=500).fit(Z[fit_idx])
        a2 = seg_assign_from_gmm(Z, seg, g2)
        row["u2_gmm_BIC"] = arm(a2, dict(level="segment", K_BIC=int(k2),
                                         bic={str(k): v for k, v in bics.items()}))
    except Exception as e:  # noqa: BLE001
        row["u2_gmm_BIC"] = dict(error=repr(e))

    # ---- u3: Gaussian HMM on x alone, K at the number of conditions (sample-level Viterbi path)
    try:
        a3, info = hmm_assign(Z, seg, K)
        row["u3_hmm_Kfixed"] = arm(a3, dict(level="sample", **info))
    except Exception as e:  # noqa: BLE001
        row["u3_hmm_Kfixed"] = dict(error=repr(e))

    # ---- perm: condition labels permuted across segments (segment counts preserved exactly)
    rng = np.random.default_rng(SEED)
    starts, lens = seg_bounds(seg)
    seg_lab = lab[starts]
    n_perm = N_PERM if not fit_sub else max(10, N_PERM // 2)
    counts = []
    for _ in range(n_perm):
        asg = np.repeat(rng.permutation(seg_lab), lens)
        regs, _ = regimes_from_assignment(Z, seg, asg, dt)
        counts.append(count_field(regs, n_starts=60)["n_stable"])
    row["perm_labels"] = dict(n_draws=n_perm, n_stable=counts,
                              frac_equal_to_sup=float(np.mean([c == row["sup"]["n_stable"] for c in counts])),
                              n_starts=60,
                              note="segment-label permutation; segments per condition preserved, so n is matched")

    # ---- km: label-free 1-D drift reconstruction on the principal axis of the pooled series
    if kmcfg["series"] == "dist" and "Zdist" in z.files:
        Xk = z["Zdist"].astype(float)
        segk = z["segdist"] if "segdist" in z.files else np.zeros(len(Xk), np.int32)
    else:
        Xk = Z
        segk = seg
    v = _pc1(Xk)
    st, ln = seg_bounds(segk)
    x1 = Xk @ v
    segs1d = [x1[a:a + b] for a, b in zip(st, ln)]
    row["km_labelfree"] = km_row(segs1d, dt=kmcfg["dt"], lag=kmcfg["lag"])
    row["km_labelfree"]["axis"] = "PC1 of the pooled state series (no labels)"
    row["km_labelfree"]["series"] = kmcfg["series"]
    row["km_labelfree"]["settings_from_reference_pipeline"] = bool(kmcfg["reference"])
    # the same estimator on the label-dependent condition-difference axis, for comparison
    if len(np.unique(lab)) >= 2:
        labk = z["labdist"].astype(int) if (kmcfg["series"] == "dist" and "labdist" in z.files) else lab
        w = Xk[labk == 0].mean(0) - Xk[labk == 1].mean(0)
        w /= (np.linalg.norm(w) + 1e-12)
        x2 = Xk @ w
        s1d = [x2[a:a + b] for a, b in zip(st, ln)]
        row["km_label_axis"] = km_row(s1d, dt=kmcfg["dt"], lag=kmcfg["lag"])
        row["km_label_axis"]["axis"] = "condition-difference axis (label dependent)"

    row["seconds"] = round(time.time() - t0, 1)
    return row


def _pc1(X):
    Xc = X - X.mean(0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Vt[0]


def _ari(a, b):
    from sklearn.metrics import adjusted_rand_score
    return float(adjusted_rand_score(a, b))


def _safe(args):
    try:
        return one_record(args)
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        return dict(cohort=args[1], record=args[2], error=repr(e))


def run_cohort(cache, out_dir, cohort, workers):
    with open(os.path.join(cache, cohort, "_meta.json"), encoding="utf-8") as fh:
        meta = json.load(fh)
    dt, K = meta["dt"], meta["K"]
    kmcfg = KM_CFG[cohort]
    od = os.path.join(out_dir, cohort)
    os.makedirs(od, exist_ok=True)
    recs = sorted(meta["records"])
    todo = [(cache, cohort, r, dt, K, kmcfg) for r in recs
            if not os.path.exists(os.path.join(od, f"{r}.json"))]
    log(f"{cohort}: {len(recs)} records, {len(todo)} to do")
    if not todo:
        return
    from concurrent.futures import ProcessPoolExecutor
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, row in enumerate(ex.map(_safe, todo), 1):
            with open(os.path.join(od, f"{row['record']}.json"), "w", encoding="utf-8") as fh:
                fh.write(json.dumps(row, indent=1, default=float))
            if i % 5 == 0 or "error" in row:
                s = row.get("sup", {}).get("n_stable")
                u = row.get("u1_gmm_Kfixed_segment", {}).get("n_stable")
                log(f"  [{i}/{len(todo)}] {row['record']} sup={s} u1={u} "
                    f"({(time.time()-t0)/60:.1f} min)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cohort", default="all", help="all, or a comma-separated list of cohorts")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    cache = paths.cache("count")
    out_dir = paths.cache("count_rows")
    cs = ["fmri", "meg", "ds003838", "eegmmidb", "sleep"] if a.cohort == "all" else a.cohort.split(",")
    for c in cs:
        if os.path.exists(os.path.join(cache, c, "_meta.json")):
            log(f"=== {c} ===")
            run_cohort(cache, out_dir, c, a.workers)
        else:
            log(f"=== {c} === skipped (no cache)")


if __name__ == "__main__":
    main()
