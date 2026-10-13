"""Assemble the (Delta log lambda, Delta log D) plane of drive_gain_plane.py and measure how well drive is separated
from gain.

  1  THE PLANE.  For every (modality, truth, size, nu): the generator truth, the sampled-lag truth and the measured
     pair, with the between-network spread, and the bias each stage introduces.
  2  SEPARATION OF DRIVE FROM GAIN, at the effect sizes cortex shows.  Two measures, both leave-one-network-out so
     nothing is fitted on the network it is scored on:
       - the Mahalanobis distance between the drive and gain clouds in the plane, standardised by the NULL cell's
         own per-participant covariance measured in the same run;
       - a nearest-centroid classification accuracy on the two-dimensional plane point, with chance MEASURED by
         permuting the truth labels within (network, size), 200 permutations, size-matched to the observed
         sample, and a matched within-network ceiling.
     The minimum manipulation size at which the accuracy reaches 0.80 is read off the size ladder by monotone
     interpolation, per modality.
  3  THE THIRD POSSIBILITY.  Where the count-changing manipulation lands, whether an eigenvalue of the drift
     crosses zero (exact enumeration), and whether the plane can tell it from a gain change of the same sign.
  4  CALIBRATION and the rate's identifiability at each sampling.

Input   drive_gain_rows/rows_<seed>.json in the network cache paths.cache("rnn") (--rows).
Outputs figure_reproduction/inputs/drive_gain_separation.csv (the values of Fig. 3e; 256 rows: modality x nu x
        size x axis rule x gain truth x diffusion readout), analyses/recurrent_networks/out/drive_gain_plane.csv,
        drive_gain_numbers.json, drive_gain_report.txt.
Reported in  Fig. 3e: rows nu = 0, size = 1, rule = fixed, dkind = D, gain_truth = gain_c -- EEG accuracy 0.898
        (chance 0.504, within-network ceiling 0.875), networks at or above 0.80: EEG 8, MEG 5, fMRI 2 of 8.
Runtime  about 1.5 minutes.

    python analyses/recurrent_networks/drive_gain_report.py
"""
from __future__ import annotations


import argparse
import csv as _csv
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from analyses.common import paths  # noqa: E402

ROWS = Path(paths.cache("rnn")) / "drive_gain_rows"


def out(name):
    return paths.out("analyses", "recurrent_networks", "out", name)


NPERM = 200

SINK = []


def log(m):
    print(m, flush=True)
    SINK.append(str(m))


# ==================================================================== load the circuit rows
def load_rows():
    data = []
    for f in sorted(ROWS.glob("rows_*.json")):
        data.append(json.loads(f.read_text(encoding="utf-8")))
    return data


def _slope3(y):
    y = np.asarray(y, float)
    if not np.all(np.isfinite(y)):
        return np.nan
    lv = np.array([-1.0, 0.0, 1.0])
    return float(lv @ (y - y.mean()) / 2.0)


def dlog_Dc(cell, dt):
    """The BIAS-FREE diffusion readout, from the same three numbers the arm already stores.

    The cortical arms report D_hat = var(innovation)/dt.  For an OU process sampled at dt the
    stationary variance is var(innovation)/(1 - a^2) and the diffusion is D = lambda * var_stat, so
        D  =  lambda * D_hat * dt / (1 - a^2) ,
    which equals D_hat only when lambda*dt is small.  At the COG-BCI arm's own numbers
    (lambda = 1.33 s^-1, dt = 1 s) it is not small, and the consequence is measured here: a pure
    GAIN change, which leaves D exactly unchanged, moves D_hat by -0.10 per level -- the raw
    readout is close to the stationary variance, not to the diffusion.  Both are reported.
    """
    lam = cell.get("lam"); D = cell.get("D"); a = cell.get("a")
    if lam is None or D is None or a is None:
        return np.nan
    out = []
    for l, d, aa in zip(lam, D, a):
        if l is None or d is None or aa is None:
            return np.nan
        if not (np.isfinite(l) and np.isfinite(d) and np.isfinite(aa)) or d <= 0 or l <= 0:
            return np.nan
        if not (0.0 < aa < 1.0):
            return np.nan
        out.append(math.log(l * d * dt / (1.0 - aa ** 2)))
    return _slope3(out)


def pair(r, rule="fixed", clean=False, dkind="D"):
    src = r.get("est_clean") if clean else r.get("est")
    if not src or not src.get(rule):
        return None
    cell = src[rule]
    a = cell.get("dlog_lam")
    b = cell.get("dlog_D") if dkind == "D" else dlog_Dc(cell, r["dt"])
    if a is None or b is None or not np.isfinite(a) or not np.isfinite(b):
        return None
    return np.array([a, b], float)


# ==================================================================== 1. the plane
def plane_table(data, rule="fixed", dkind="D"):
    rows = []
    keys = sorted({(r["modality"], r["truth"], r["size"], r["nu"])
                   for d in data for r in d["rows"]})
    for mod, truth, size, nu in keys:
        gen_l, gen_d, sam_l, sam_d, me_l, me_d, per_net = [], [], [], [], [], [], {}
        esc = []
        for d in data:
            sel = [r for r in d["rows"] if (r["modality"], r["truth"], r["size"], r["nu"])
                   == (mod, truth, size, nu)]
            if not sel:
                continue
            gen_l.append(sel[0]["gen"]["dlog_lam_e"]); gen_d.append(sel[0]["gen"]["dlog_D_e"])
            sam_l.append(sel[0]["sampled"]["dlog_lam"]); sam_d.append(sel[0]["sampled"]["dlog_D"])
            esc.append(sel[0].get("escape_frac"))
            P = [pair(r, rule, dkind=dkind) for r in sel]
            P = np.array([p for p in P if p is not None])
            if len(P):
                per_net[d["anchor_seed"]] = P.mean(0)
                me_l += list(P[:, 0]); me_d += list(P[:, 1])
        if not per_net:
            continue
        nm = np.array(list(per_net.values()))
        rows.append(dict(
            modality=mod, truth=truth, size=size, nu=nu, rule=rule, dkind=dkind,
            n_networks=len(per_net),
            gen_dlog_lam=float(np.mean(gen_l)), gen_dlog_D=float(np.mean(gen_d)),
            samp_dlog_lam=float(np.nanmean(sam_l)), samp_dlog_D=float(np.nanmean(sam_d)),
            meas_dlog_lam=float(np.mean(nm[:, 0])), meas_dlog_D=float(np.mean(nm[:, 1])),
            meas_dlog_lam_net_sd=float(np.std(nm[:, 0], ddof=1)) if len(nm) > 1 else np.nan,
            meas_dlog_D_net_sd=float(np.std(nm[:, 1], ddof=1)) if len(nm) > 1 else np.nan,
            meas_dlog_lam_rep_sd=float(np.std(me_l, ddof=1)),
            meas_dlog_D_rep_sd=float(np.std(me_d, ddof=1)),
            meas_dlog_lam_net_min=float(nm[:, 0].min()), meas_dlog_lam_net_max=float(nm[:, 0].max()),
            meas_dlog_D_net_min=float(nm[:, 1].min()), meas_dlog_D_net_max=float(nm[:, 1].max()),
            escape_frac=float(np.mean([e for e in esc if e is not None])) if esc else np.nan,
            angle_gen_deg=float(math.degrees(math.atan2(np.mean(gen_d), np.mean(gen_l)))),
            angle_meas_deg=float(math.degrees(math.atan2(np.mean(nm[:, 1]), np.mean(nm[:, 0])))),
            bias_sampling_lam=float(np.nanmean(sam_l) - np.mean(gen_l)),
            bias_sampling_D=float(np.nanmean(sam_d) - np.mean(gen_d)),
            bias_estimator_lam=float(np.mean(nm[:, 0]) - np.nanmean(sam_l)),
            bias_estimator_D=float(np.mean(nm[:, 1]) - np.nanmean(sam_d))))
    return rows


# ==================================================================== 2. separation
def collect(data, mod, truth, size, nu, rule, dkind="D", clean=False):
    out = []
    for d in data:
        for r in d["rows"]:
            if (r["modality"], r["truth"], r["size"], r["nu"]) == (mod, truth, size, nu):
                p = pair(r, rule, clean=clean, dkind=dkind)
                if p is not None:
                    out.append((d["anchor_seed"], p))
    return out


def loo_accuracy(A, B, labels=("a", "b")):
    """Leave-one-network-out nearest-centroid accuracy between two labelled clouds of 2-D points.
    Each element is (network, point).  Standardisation uses the TRAINING networks only."""
    nets = sorted({n for n, _ in A} | {n for n, _ in B})
    if len(nets) < 2:
        return np.nan, []
    correct, total, per_net = 0, 0, []
    for held in nets:
        tr_a = np.array([p for n, p in A if n != held])
        tr_b = np.array([p for n, p in B if n != held])
        te_a = np.array([p for n, p in A if n == held])
        te_b = np.array([p for n, p in B if n == held])
        if len(tr_a) < 2 or len(tr_b) < 2 or not len(te_a) or not len(te_b):
            continue
        mu = np.vstack([tr_a, tr_b]).mean(0)
        sd = np.vstack([tr_a, tr_b]).std(0)
        sd = np.where(sd > 1e-12, sd, 1.0)
        ca, cb = (tr_a - mu) / sd, (tr_b - mu) / sd
        ca, cb = ca.mean(0), cb.mean(0)
        c, t = 0, 0
        for X, truth_is_a in ((te_a, True), (te_b, False)):
            Z = (X - mu) / sd
            da = ((Z - ca) ** 2).sum(1)
            db = ((Z - cb) ** 2).sum(1)
            pred_a = da <= db
            c += int((pred_a == truth_is_a).sum()); t += len(X)
        per_net.append(c / t if t else np.nan)
        correct += c; total += t
    return (correct / total if total else np.nan), per_net


def mahalanobis_sep(A, B, N):
    """Distance between the two class means in units of the NULL cloud's own covariance."""
    a = np.array([p for _, p in A]); b = np.array([p for _, p in B])
    n = np.array([p for _, p in N])
    if len(n) < 5:
        return np.nan
    S = np.cov(n, rowvar=False) + 1e-12 * np.eye(2)
    d = a.mean(0) - b.mean(0)
    return float(np.sqrt(max(d @ np.linalg.solve(S, d), 0.0)))


def within_network_ceiling(A, B, seed=20260922):
    """The matched within-family ceiling: the SAME classifier trained on half of one
    network's own replicates and tested on the other half, so the cross-network accuracy has
    something to be compared with rather than only a chance level."""
    rng = np.random.default_rng(seed)
    byn = {}
    for n, p_ in A:
        byn.setdefault(n, [[], []])[0].append(p_)
    for n, p_ in B:
        byn.setdefault(n, [[], []])[1].append(p_)
    accs = []
    for n, (a, b) in byn.items():
        if len(a) < 4 or len(b) < 4:
            continue
        a = np.array(a); b = np.array(b)
        ia = rng.permutation(len(a)); ib = rng.permutation(len(b))
        ha, hb = len(a) // 2, len(b) // 2
        tr_a, te_a = a[ia[:ha]], a[ia[ha:]]
        tr_b, te_b = b[ib[:hb]], b[ib[hb:]]
        mu = np.vstack([tr_a, tr_b]).mean(0)
        sd = np.vstack([tr_a, tr_b]).std(0)
        sd = np.where(sd > 1e-12, sd, 1.0)
        ca = ((tr_a - mu) / sd).mean(0); cb = ((tr_b - mu) / sd).mean(0)
        c, tt = 0, 0
        for X, is_a in ((te_a, True), (te_b, False)):
            Z = (X - mu) / sd
            pred_a = ((Z - ca) ** 2).sum(1) <= ((Z - cb) ** 2).sum(1)
            c += int((pred_a == is_a).sum()); tt += len(X)
        if tt:
            accs.append(c / tt)
    return (float(np.mean(accs)) if accs else np.nan,
            float(np.min(accs)) if accs else np.nan,
            float(np.max(accs)) if accs else np.nan, len(accs))


def perm_chance(A, B, n_perm=NPERM, seed=20260922):
    rng = np.random.default_rng(seed)
    pool = A + B
    nA = len(A)
    accs = []
    for _ in range(n_perm):
        idx = rng.permutation(len(pool))
        pa = [pool[i] for i in idx[:nA]]
        pb = [pool[i] for i in idx[nA:]]
        # keep the network structure: relabel within network
        byn = {}
        for n, p in pool:
            byn.setdefault(n, []).append(p)
        pa, pb = [], []
        for n, ps in byn.items():
            k = rng.permutation(len(ps))
            half = len(ps) // 2
            pa += [(n, ps[i]) for i in k[:half]]
            pb += [(n, ps[i]) for i in k[half:]]
        a, _ = loo_accuracy(pa, pb)
        if np.isfinite(a):
            accs.append(a)
    if not accs:
        return np.nan, np.nan
    return float(np.mean(accs)), float(np.percentile(accs, 95))


def separation_table(data, rule="fixed", gain_truth="gain_c", dkind="D"):
    rows = []
    mods = sorted({r["modality"] for d in data for r in d["rows"]})
    sizes = sorted({r["size"] for d in data for r in d["rows"] if r["truth"] == "drive"})
    for mod in mods:
        for nu in sorted({r["nu"] for d in data for r in d["rows"]}):
            for size in sizes:
                A = collect(data, mod, "drive", size, nu, rule, dkind)
                B = collect(data, mod, gain_truth, size, nu, rule, dkind)
                N = collect(data, mod, "null", size, nu, rule, dkind)
                if not A or not B:
                    continue
                acc, per_net = loo_accuracy(A, B)
                ch, ch95 = perm_chance(A, B)
                cm, clo, chi, cn = within_network_ceiling(A, B)
                rows.append(dict(modality=mod, nu=nu, size=size, rule=rule,
                                 gain_truth=gain_truth, dkind=dkind,
                                 n_drive=len(A), n_gain=len(B),
                                 acc=acc, chance=ch, chance_p95=ch95,
                                 ceiling_within_network=cm, ceiling_lo=clo, ceiling_hi=chi,
                                 ceiling_n_networks=cn,
                                 acc_net_min=float(np.min(per_net)) if per_net else np.nan,
                                 acc_net_max=float(np.max(per_net)) if per_net else np.nan,
                                 n_networks_at_080=int(sum(a >= 0.80 for a in per_net)),
                                 n_networks=len(per_net),
                                 maha_null_units=mahalanobis_sep(A, B, N)))
    return rows


def min_size(rows, mod, nu, target=0.80):
    r = sorted([x for x in rows if x["modality"] == mod and x["nu"] == nu],
               key=lambda z: z["size"])
    if not r:
        return None
    xs = [x["size"] for x in r]; ys = [x["acc"] for x in r]
    if ys[0] >= target:
        return float(xs[0])
    for i in range(1, len(xs)):
        if ys[i] >= target > ys[i - 1]:
            f = (target - ys[i - 1]) / (ys[i] - ys[i - 1])
            return float(xs[i - 1] + f * (xs[i] - xs[i - 1]))
    return None


def main():
    global ROWS
    ap = argparse.ArgumentParser(description="the drive/gain plane and its separation")
    ap.add_argument("--rows", default=str(ROWS),
                    help="directory of rows_<seed>.json (default: drive_gain_rows in the network cache)")
    ROWS = Path(ap.parse_args().rows)
    data = load_rows()
    log(f"networks loaded: {[d['anchor_seed'] for d in data]}")
    res = {"n_networks": len(data)}

    # the operating point and the exact identity
    ident = [r["gen"]["ratio_identity_max_dev"] for d in data for r in d["rows"]]
    res["lyapunov_ratio_identity_max_dev"] = float(np.max(ident))
    log(f"exact identity  var_e * lambda_e / D_e = 1  to within "
        f"{np.max(ident):.2e} over every configuration of every network")

    # count arm: the enumerated number of stable states and the eigenvalue crossing
    cnt = []
    for d in data:
        cs = [c for c in d["configs"] if c["arm"] == "count"]
        cnt.append(dict(seed=d["anchor_seed"],
                        gammas=[c["gamma"] for c in cs],
                        n_stable=[c["n_stable"] for c in cs],
                        max_re=[c["max_re_eig_any_fp"] for c in cs],
                        sep=[c.get("sep_from_other") for c in cs],
                        lam_e=[c["true"]["lam_e"] for c in cs],
                        D_e=[c["true"]["D_e"] for c in cs]))
    res["count_arm"] = cnt
    ok = [c for c in cnt if c["n_stable"][0] == 1 and c["n_stable"][-1] >= 2]
    log(f"count arm: 1 -> >=2 stable fixed points in {len(ok)} of {len(cnt)} networks; "
        f"max Re eig over all fixed points goes from "
        f"{np.median([c['max_re'][0] for c in cnt]):+.4f} to "
        f"{np.median([c['max_re'][-1] for c in cnt]):+.4f}")

    combos = [("fixed", "D"), ("heldout", "D"), ("pc1", "D"), ("fixed", "Dc"), ("heldout", "Dc")]
    log("")
    log("=== the drive's subspace in the circuit, with its matched ceiling ===")
    cosv = [d["meta"]["cos_v_input_e_load"] for d in data]
    nd = int(np.median([8 for _ in data]))  # placeholder, replaced below
    rng = np.random.default_rng(20260922)
    R = rng.standard_normal((20000, 64))
    R /= np.linalg.norm(R, axis=1, keepdims=True)
    rc = np.abs(R @ (rng.standard_normal(64) / np.linalg.norm(rng.standard_normal(64))))
    log(f"  |cos| between the direction the input's VARIANCE enters (v = diag(1-tanh^2) W_in[:,0])"
        f" and the direction the equilibrium MOVES (e = -J^-1 W_in[:,0]):")
    log(f"    {np.median(cosv):.3f} [{min(cosv):.3f}, {max(cosv):.3f}] over "
        f"{len(cosv)} networks")
    log(f"  matched ceiling, two random unit directions in R^64: {np.median(rc):.3f}, "
        f"95th percentile {np.percentile(rc, 95):.3f}, max {rc.max():.3f}")
    log(f"  so the two directions are far from chance but also far from identical: in a recurrent "
        f"network the drive's 'same subspace' is an OVERLAP, not a coincidence.")
    res["drive_subspace"] = dict(cos=cosv, median=float(np.median(cosv)),
                              min=float(min(cosv)), max=float(max(cosv)),
                              random_median=float(np.median(rc)),
                              random_p95=float(np.percentile(rc, 95)))

    for rule, dk in combos:
        res[f"plane_{rule}_{dk}"] = plane_table(data, rule, dk)
    pt = res["plane_fixed_D"]
    with open(out("drive_gain_plane.csv"), "w", newline="") as fh:
        allr = sum([res[f"plane_{r}_{d}"] for r, d in combos], [])
        keys = sorted({k for r in allr for k in r})
        w = _csv.DictWriter(fh, fieldnames=keys); w.writeheader()
        for r in allr:
            w.writerow(r)

    log("")
    log("=== THE PLANE at the size cortex shows (k = 1), axis rule `fixed`, nu = 0 ===")
    log(f"{'modality':6s} {'truth':8s} {'gen dlogL':>10s} {'gen dlogD':>10s} "
        f"{'samp dlogL':>11s} {'samp dlogD':>11s} {'meas dlogL':>11s} {'meas dlogD':>11s} "
        f"{'net sd L':>9s} {'net sd D':>9s}")
    for r in pt:
        if r["size"] != 1.0 or r["nu"] != 0.0:
            continue
        log(f"{r['modality']:6s} {r['truth']:8s} {r['gen_dlog_lam']:+10.4f} "
            f"{r['gen_dlog_D']:+10.4f} {r['samp_dlog_lam']:+11.4f} {r['samp_dlog_D']:+11.4f} "
            f"{r['meas_dlog_lam']:+11.4f} {r['meas_dlog_D']:+11.4f} "
            f"{r['meas_dlog_lam_net_sd']:9.4f} {r['meas_dlog_D_net_sd']:9.4f}")

    log("")
    log("=== THE SAME PLANE with the BIAS-FREE diffusion readout "
        "D = lambda * var(innovation) / (1 - a^2), k = 1, rule `fixed`, nu = 0 ===")
    for r in res["plane_fixed_Dc"]:
        if r["size"] != 1.0 or r["nu"] != 0.0:
            continue
        log(f"{r['modality']:6s} {r['truth']:8s} {'':10s} {r['gen_dlog_D']:+10.4f} "
            f"{'':11s} {'':11s} {r['meas_dlog_lam']:+11.4f} {r['meas_dlog_D']:+11.4f} "
            f"{r['meas_dlog_lam_net_sd']:9.4f} {r['meas_dlog_D_net_sd']:9.4f}")

    log("")
    log("=== SEPARATION of drive from gain in the plane (leave-one-network-out) ===")
    sep_rows = []
    for gt in ("gain_c", "gain"):
        for rule in ("fixed", "heldout"):
            for dk in ("D", "Dc"):
                sep_rows += separation_table(data, rule, gt, dk)
    res["separation"] = sep_rows
    with open(paths.out("figure_reproduction", "inputs", "drive_gain_separation.csv"), "w", newline="") as fh:
        keys = sorted({k for r in sep_rows for k in r})
        w = _csv.DictWriter(fh, fieldnames=keys, lineterminator="\n"); w.writeheader()
        for r in sep_rows:
            w.writerow(r)
    prim = [r for r in sep_rows if r["rule"] == "fixed" and r["gain_truth"] == "gain_c"
            and r["dkind"] == "D"]
    log(f"{'modality':6s} {'nu':4s} {'size':5s} {'acc':>6s} {'ceiling':>8s} {'chance':>7s} "
        f"{'p95':>6s} {'nets>=.8':>9s} {'maha(null)':>11s}")
    for r in sorted(prim, key=lambda z: (z["modality"], z["nu"], z["size"])):
        log(f"{r['modality']:6s} {r['nu']:<4.1f} {r['size']:<5.1f} {r['acc']:6.3f} "
            f"{r['ceiling_within_network']:8.3f} "
            f"{r['chance']:7.3f} {r['chance_p95']:6.3f} "
            f"{r['n_networks_at_080']:>4d}/{r['n_networks']:<4d} {r['maha_null_units']:11.3f}")
    log("")
    log("=== the angle between the drive and gain directions in the plane (degrees) ===")
    ang = []
    for dk in ("D", "Dc"):
        for mod in sorted({r["modality"] for d in data for r in d["rows"]}):
            pl = res[f"plane_fixed_{dk}"]
            dr = [r for r in pl if r["modality"] == mod and r["truth"] == "drive"
                  and r["size"] == 1.0 and r["nu"] == 0.0]
            gn = [r for r in pl if r["modality"] == mod and r["truth"] == "gain_c"
                  and r["size"] == 1.0 and r["nu"] == 0.0]
            if not dr or not gn:
                continue
            a = math.degrees(math.atan2(dr[0]["meas_dlog_D"], dr[0]["meas_dlog_lam"]))
            b = math.degrees(math.atan2(gn[0]["meas_dlog_D"], gn[0]["meas_dlog_lam"]))
            gta = math.degrees(math.atan2(dr[0]["gen_dlog_D"], dr[0]["gen_dlog_lam"]))
            gtb = math.degrees(math.atan2(gn[0]["gen_dlog_D"], gn[0]["gen_dlog_lam"]))
            sep_ang = abs(((a - b) + 180) % 360 - 180)
            true_ang = abs(((gta - gtb) + 180) % 360 - 180)
            ang.append(dict(modality=mod, dkind=dk, drive_deg=a, gain_deg=b,
                            separation_deg=sep_ang, true_separation_deg=true_ang))
            log(f"  {mod:6s} {dk:3s}  drive {a:+7.1f}  gain {b:+7.1f}  apart {sep_ang:6.1f} "
                f"(true {true_ang:.1f})")
    res["angles"] = ang

    res["min_size"] = {}
    log("")
    log("minimum manipulation size for a stated drive-vs-gain accuracy (x the cortical size):")
    log(f"  {'cell':16s} {'0.70':>8s} {'0.80':>8s} {'0.90':>8s}")
    for mod in sorted({r["modality"] for r in prim}):
        for nu in sorted({r["nu"] for r in prim}):
            e_ = {f"{tgt:.2f}": min_size(prim, mod, nu, tgt) for tgt in (0.70, 0.80, 0.90)}
            res["min_size"][f"{mod}|nu={nu}"] = e_
            log(f"  {mod + '|nu=' + str(nu):16s} " + " ".join(
                f"{('>x4' if v is None else 'x%.2f' % v):>8s}" for v in e_.values()))

    log("")
    log("=== the bias and the spread the SAMPLING introduces (rule: measured - truth) ===")
    log(f"  {'mod':6s} {'truth':8s} {'k':>4s} {'samp-gen L':>11s} {'samp-gen D':>11s} "
        f"{'meas-samp L':>12s} {'meas-samp D':>12s} {'net sd L':>9s} {'net sd D':>9s} "
        f"{'rep sd L':>9s} {'rep sd D':>9s}")
    bias = []
    for r in res["plane_fixed_D"]:
        if r["nu"] != 0.0 or r["truth"] not in ("drive", "gain_c", "null") or r["size"] != 1.0:
            continue
        bias.append(r)
        log(f"  {r['modality']:6s} {r['truth']:8s} {r['size']:>4.1f} "
            f"{r['bias_sampling_lam']:+11.4f} {r['bias_sampling_D']:+11.4f} "
            f"{r['bias_estimator_lam']:+12.4f} {r['bias_estimator_D']:+12.4f} "
            f"{r['meas_dlog_lam_net_sd']:9.4f} {r['meas_dlog_D_net_sd']:9.4f} "
            f"{r['meas_dlog_lam_rep_sd']:9.4f} {r['meas_dlog_D_rep_sd']:9.4f}")
    res["bias_table"] = bias

    # the third possibility: is the saddle-node crossing marked ANYWHERE on the gain locus?
    log("")
    log("=== THE THIRD POSSIBILITY: is the crossing marked in the plane? ===")
    kink = []
    for d in data:
        cs = sorted([c for c in d["configs"] if c["arm"] == "count"], key=lambda z: z["level"])
        if len(cs) != 3:
            continue
        g0, g1, g2 = [c["gamma"] for c in cs]
        l0, l1, l2 = [c["true"]["lam_e"] for c in cs]
        s_below = math.log(l1 / l0) / (g1 - g0) if g1 != g0 else np.nan
        s_across = math.log(l2 / l1) / (g2 - g1) if g2 != g1 else np.nan
        kink.append(dict(seed=d["anchor_seed"],
                         n_stable=[c["n_stable"] for c in cs],
                         max_re=[c["max_re_eig_any_fp"] for c in cs],
                         dloglam_per_gamma_below=s_below,
                         dloglam_per_gamma_across=s_across,
                         ratio=s_across / s_below if s_below else np.nan,
                         dlogD_exact=[math.log(c["true"]["D_e"] / cs[0]["true"]["D_e"])
                                      for c in cs],
                         dloglam_total=math.log(l2 / l0)))
    res["count_kink"] = kink
    rr = [k["ratio"] for k in kink if np.isfinite(k["ratio"])]
    log(f"  d log lambda_e / d gamma just BELOW the saddle-node vs just ACROSS it: ratio "
        f"{np.median(rr):.3f} [{min(rr):.3f}, {max(rr):.3f}] over {len(rr)} networks "
        f"(1.000 = no kink; the crossing leaves no mark on the occupied well's rate)")
    log(f"  exact d log D over the whole count ladder: "
        f"{max(abs(v) for k in kink for v in k['dlogD_exact']):.2e} (zero by construction)")
    log(f"  total d log lambda_e traversed to reach the crossing: "
        f"{np.median([k['dloglam_total'] for k in kink]):+.4f} "
        f"[{min(k['dloglam_total'] for k in kink):+.4f}, "
        f"{max(k['dloglam_total'] for k in kink):+.4f}]")

    log("")
    log("  what a recording DOES see: completed escapes to the new state")
    esc_rows = []
    for mod in sorted({r["modality"] for d in data for r in d["rows"]}):
        ef, dif = [], []
        for d in data:
            sel = [r for r in d["rows"] if r["modality"] == mod and r["truth"] == "count"
                   and r["nu"] == 0.0]
            if not sel:
                continue
            # the row stores the escape fraction averaged over the THREE levels, and only the top
            # level has a second state to escape to, so the top level's own fraction is 3x it
            ef.append(float(np.mean([r["escape_frac"] for r in sel])) * 3.0)
            a = [pair(r, "fixed", clean=False) for r in sel]
            b = [pair(r, "fixed", clean=True) for r in sel]
            ok = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
            if ok:
                dif.append(np.mean([x - y for x, y in ok], axis=0))
        if not ef:
            continue
        dm = np.array(dif).mean(0) if dif else np.array([np.nan, np.nan])
        esc_rows.append(dict(modality=mod, level="bistable (top)",
                             escape_frac_mean=float(np.mean(ef)),
                             escape_frac_min=float(np.min(ef)), escape_frac_max=float(np.max(ef)),
                             raw_minus_clean_dlog_lam=float(dm[0]),
                             raw_minus_clean_dlog_D=float(dm[1]), n_networks=len(ef)))
        log(f"    {mod:6s} escape fraction in the BISTABLE level {np.mean(ef):.3f} "
            f"[{np.min(ef):.3f}, {np.max(ef):.3f}] over networks;  "
            f"raw minus escape-free plane point ({dm[0]:+.4f}, {dm[1]:+.4f})")
    res["count_escape"] = esc_rows

    log("")
    log("=== THE THIRD POSSIBILITY: count change against gain change, same classifier ===")
    three = []
    for mod in sorted({r["modality"] for d in data for r in d["rows"]}):
        for nu in (0.0, 1.0):
            C = collect(data, mod, "count", 1.0, nu, "fixed")
            if not C:
                continue
            for other, sz in (("gain_c", 0.5), ("gain_c", 1.0), ("drive", 1.0),
                              ("null", 1.0)):
                O = collect(data, mod, other, sz, nu, "fixed")
                if not O:
                    continue
                acc, per = loo_accuracy(C, O)
                ch, ch95 = perm_chance(C, O)
                cm, clo, chi, cn = within_network_ceiling(C, O)
                three.append(dict(modality=mod, nu=nu, against=f"{other}x{sz}",
                                  acc=acc, chance=ch, chance_p95=ch95,
                                 ceiling_within_network=cm, ceiling_lo=clo, ceiling_hi=chi,
                                 ceiling_n_networks=cn,
                                  n_networks=len(per)))
                log(f"  {mod:6s} nu={nu:.1f}  count vs {other}x{sz:<4.1f}  acc {acc:.3f} "
                    f"(ceiling {cm:.3f}, chance {ch:.3f}, p95 {ch95:.3f})")
    res["third_possibility"] = three

    # ---- calibration and the identifiability check that must travel with every rate
    log("")
    log("=== CALIBRATION: recovered fraction of a KNOWN knob, and the rate's identifiability ===")
    cal = []
    for dk in ("D", "Dc"):
        pl = res[f"plane_fixed_{dk}"]
        for mod in sorted({r["modality"] for r in pl}):
            for size in (1.0, 4.0):
                dr = [r for r in pl if r["modality"] == mod and r["truth"] == "drive"
                      and r["size"] == size and r["nu"] == 0.0]
                gn = [r for r in pl if r["modality"] == mod and r["truth"] == "gain_c"
                      and r["size"] == size and r["nu"] == 0.0]
                if not dr or not gn:
                    continue
                d, g = dr[0], gn[0]
                ent = dict(modality=mod, dkind=dk, size=size,
                           driveD_over_generator=d["meas_dlog_D"] / d["gen_dlog_D"],
                           driveD_over_sampled=(d["meas_dlog_D"] / d["samp_dlog_D"]
                                                if d["samp_dlog_D"] else np.nan),
                           gainlam_over_generator=g["meas_dlog_lam"] / g["gen_dlog_lam"],
                           gainlam_over_sampled=(g["meas_dlog_lam"] / g["samp_dlog_lam"]
                                                 if g["samp_dlog_lam"] else np.nan),
                           gain_leak_onto_D=g["meas_dlog_D"],
                           drive_leak_onto_lam=d["meas_dlog_lam"])
                cal.append(ent)
                if dk == "D":
                    log(f"  {mod:6s} x{size:<4.1f} drive log D recovered "
                        f"{ent['driveD_over_generator']:.3f} of the generator truth "
                        f"({ent['driveD_over_sampled']:.3f} of the sampled-lag truth); "
                        f"gain log lambda {ent['gainlam_over_generator']:.3f} "
                        f"({ent['gainlam_over_sampled']:.3f}); "
                        f"gain leak onto D {ent['gain_leak_onto_D']:+.4f}")
    res["calibration"] = cal
    log("  with the bias-free diffusion readout the gain leak onto D becomes:")
    for e_ in [c for c in cal if c["dkind"] == "Dc" and c["size"] == 1.0]:
        log(f"    {e_['modality']:6s} {e_['gain_leak_onto_D']:+.4f}  "
            f"(drive log D recovered {e_['driveD_over_generator']:.3f})")

    log("")
    log("=== the rate's identifiability at each sampling, measured in the same run ===")
    ident_rows = []
    for mod in sorted({r["modality"] for d in data for r in d["rows"]}):
        rr = []
        for d in data:
            for r in d["rows"]:
                if (r["modality"] == mod and r["truth"] == "null" and r["nu"] == 0.0
                        and r["size"] == 1.0 and r.get("acf0")):
                    rr.append(r["acf0"]["r"])
        if not rr:
            log(f"  {mod:6s} segments too short for an r(1..5) shape test "
                f"(seg_len < 7): not evaluated")
            continue
        R = np.array(rr)
        r1 = R[:, 0]
        pred = np.array([r1 ** (k + 1) for k in range(R.shape[1])]).T
        dev = (R - pred)
        ident_rows.append(dict(modality=mod, n=len(R),
                               r=[float(x) for x in R.mean(0)],
                               r1_powers=[float(x) for x in pred.mean(0)],
                               max_abs_dev=float(np.max(np.abs(dev.mean(0)))),
                               dev=[float(x) for x in dev.mean(0)]))
        log(f"  {mod:6s} observed r(1..5) " + " ".join(f"{x:+.3f}" for x in R.mean(0))
            + "   |   r(1)^k " + " ".join(f"{x:+.3f}" for x in pred.mean(0))
            + f"   max deviation {np.max(np.abs(dev.mean(0))):.3f}")
    res["identifiability"] = ident_rows

    with open(out("drive_gain_numbers.json"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(res, indent=1, default=float))
    with open(out("drive_gain_report.txt"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(SINK))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
