# -*- coding: utf-8 -*-
"""Task location in the workload EEG cohort (COG-BCI, 28 participants after the duplicate exclusion): position and
spread slopes of the graded task difficulty (n-back and MATB, levels 0/1/2) and of within-session sleepiness (KSS,
resting epochs), read along leave-one-participant-out (LOPO) group axes; the interaction, the cross-axis 2 x 2, the
angles between the axes with matched ceilings; the cross-task transfer of the load axis; and the per-level
restoring rate along the load axis.

State: the 248-feature log band-power state (62 channels x 4 bands) standardised on each participant's Flanker
windows (space S248), and its group PCA to d = 8 fitted on the pooled Flanker windows (space S8), from the package
module analyses/workload_certificates/features/cog_features.py, which also applies the duplicate exclusion
(sub-27; sub-28 session 1).  Group summaries: participant means with percentile bootstrap intervals (10,000
resamples) and two-sided sign-flip P values (10,000 draws), seed 20260921.

Inputs:  the COG-BCI feature cache (paths.cache("cogbci"), environment variable NDIF_COGBCI_CACHE) and the
         KSS.txt and RSME.txt tables of the COG-BCI release (NDIF_COGBCI_RAW).
Outputs: figure_reproduction/inputs/workload_position_slopes.json (settings, PCA, S248 cells and transfer);
         analyses/task_location/out/workload_axes_summary.json (all results, one row per participant),
         workload_axes_cells.csv, workload_axes_dynamics_rows.csv, workload_axes_S248.npz and
         workload_axes_S8.npz (participant axes); NDIF_OUT relocates them.
Reported in: Fig. 4a (other-participant axis 0.415 [0.337, 0.492], n = 28; key spaces/S248/cells/LP), Fig. 4b
         (n-back -> MATB 0.502, MATB -> n-back 0.072, own-task references 0.667 and 0.311; keys
         spaces/S248/transfer/*), Extended Data Fig. 4b,c, Supplementary Table 7b,c.
BLAS runs on one thread (OMP/MKL/OPENBLAS_NUM_THREADS = 1), the setting of the stored outputs; with several
threads the group-PCA explained-variance ratios change in the last one or two digits.
Runtime: about 30 s on one desktop.

    python analyses/task_location/workload_axes.py --boot 10000 --perm 10000
"""
from __future__ import annotations
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")       # one BLAS thread: the setting of the stored outputs
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse, json, sys, time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "analyses" / "workload_certificates" / "features"))
from analyses.common import paths                                   # noqa: E402
import cog_features as C                                            # noqa: E402
from cog_features import (SESSIONS, FAMILIES, REST_EO, REST_EC, log,  # noqa: E402
                          ar1_lambda, boot_mean, signflip_p, cos_unit,
                          gain_tilt_z, share_in)

LEVELS = [0, 1, 2]
EYES = {"EO": REST_EO, "EC": REST_EC}
DT = 1.0


# ------------------------------------------------------------------ per-participant data
def resid(Z, seg):
    """Remove the per-segment mean."""
    Z = np.asarray(Z, float).copy()
    for s in np.unique(seg):
        m = seg == s
        Z[m] -= Z[m].mean(axis=0, keepdims=True)
    return Z


class Data:
    """Everything one participant contributes, in S248 and S8."""

    def __init__(self, sub, pca, kss, rsme):
        self.sid = sub.sid
        self.sub = sub
        self.load = {}
        self.rest = {}
        self.reject = {}
        comp, mean_ = pca.components_, pca.mean_
        def to8(Z):
            return (Z - mean_) @ comp.T
        for fam, recs in FAMILIES.items():
            for ses_i, ses in enumerate(SESSIONS, start=1):
                ok = True
                cell = {}
                for lev, rec in enumerate(recs):
                    Zd, sd_ = sub.zrec(ses, rec, "dist")
                    Zy, sy_ = sub.zrec(ses, rec, "dyn")
                    if Zd is None or len(Zd) < 60 or Zy is None or len(Zy) < 30:
                        ok = False; break
                    cell[lev] = dict(Z=Zd.astype(np.float32), seg=sd_, Z8=to8(Zd).astype(np.float32),
                                     Y=Zy.astype(np.float32), yseg=sy_, Y8=to8(Zy).astype(np.float32),
                                     rsme=rsme.get((sub.sid, ses_i, C.RSME_CODE[rec])))
                    m = sub.meta["records"].get(f"{ses}|{rec}", {})
                    self.reject[(fam, ses_i, lev)] = 1.0 - float(m.get("keep_frac_dist", np.nan))
                if ok:
                    self.load[(fam, ses_i)] = cell
        for eye, recs in EYES.items():
            for ses_i, ses in enumerate(SESSIONS, start=1):
                k = kss.get((sub.sid, ses_i))
                if k is None or "beginning" not in k or "end" not in k:
                    continue
                for tp, rec in zip(("beg", "end"), recs):
                    Zd, sd_ = sub.zrec(ses, rec, "dist")
                    Zy, sy_ = sub.zrec(ses, rec, "dyn")
                    if Zd is None or len(Zd) < 40:
                        continue
                    self.rest[(eye, ses_i, tp)] = dict(
                        Z=Zd.astype(np.float32), seg=sd_, Z8=to8(Zd).astype(np.float32),
                        Y=Zy.astype(np.float32), yseg=sy_, Y8=to8(Zy).astype(np.float32),
                        kss=k["beginning"] if tp == "beg" else k["end"])
                    m = sub.meta["records"].get(f"{ses}|{rec}", {})
                    self.reject[(eye, ses_i, tp)] = 1.0 - float(m.get("keep_frac_dist", np.nan))

    # ---------------- axes (unit vectors), in a chosen space key 'Z' (S248) or 'Z8'
    def w_load(self, key="Z", family=None):
        ds = []
        for (fam, ses), cell in self.load.items():
            if family and fam != family:
                continue
            ds.append(cell[2][key].mean(axis=0) - cell[0][key].mean(axis=0))
        if not ds:
            return None
        v = np.mean(ds, axis=0)
        n = np.linalg.norm(v)
        return v / n if n > 0 else None

    def sleep_m(self, eye):
        """(list of keys, m_within, m_between) for the six rest epochs of one eye condition."""
        ks = [(eye, s, t) for s in (1, 2, 3) for t in ("beg", "end") if (eye, s, t) in self.rest]
        if len(ks) < 4:
            return None
        kss = np.array([self.rest[k]["kss"] for k in ks], float)
        ses = np.array([k[1] for k in ks])
        mw = kss.copy()
        for s in np.unique(ses):
            mw[ses == s] -= kss[ses == s].mean()
        mb = np.zeros_like(kss)
        for s in np.unique(ses):
            mb[ses == s] = kss[ses == s].mean()
        mb = mb - mb.mean()
        return ks, mw, mb

    def w_sleep(self, key="Z", comp="within"):
        vs = []
        for eye in EYES:
            got = self.sleep_m(eye)
            if got is None:
                continue
            ks, mw, mb = got
            m = mw if comp == "within" else mb
            if m.std() == 0:
                continue
            M = np.stack([self.rest[k][key].mean(axis=0) for k in ks])
            mc = m - m.mean()
            v = (mc @ (M - M.mean(axis=0))) / np.sum(mc ** 2)
            n = np.linalg.norm(v)
            if n > 0:
                vs.append(v / n)
        if not vs:
            return None
        v = np.mean(vs, axis=0)
        n = np.linalg.norm(v)
        return v / n if n > 0 else None

    def _axis_eye(self, eye, key="Z", comp="within"):
        """The sleepiness axis from one eye condition alone (split-half reliability)."""
        got = self.sleep_m(eye)
        if got is None:
            return None
        ks, mw, mb = got
        m = mw if comp == "within" else mb
        if m.std() == 0:
            return None
        M = np.stack([self.rest[k][key].mean(axis=0) for k in ks])
        mc = m - m.mean()
        v = (mc @ (M - M.mean(axis=0))) / np.sum(mc ** 2)
        n = np.linalg.norm(v)
        return v / n if n > 0 else None

    # ---------------- readouts
    def load_cells(self, w, key="Z", family=None, use_rsme=False):
        out = []
        for (fam, ses), cell in self.load.items():
            if family and fam != family:
                continue
            groups = [resid(cell[l][key], cell[l]["seg"]) for l in LEVELS]
            mu = [cell[l][key].mean(axis=0) for l in LEVELS]
            if use_rsme:
                m = [cell[l]["rsme"] for l in LEVELS]
                if any(v is None for v in m) or np.std(m) == 0:
                    continue
            else:
                m = list(LEVELS)
            r = _ps_from(groups, mu, w, m)
            r.update(family=fam, session=ses)
            out.append(r)
        return out

    def sleep_cells(self, w, key="Z", comp="within"):
        out = []
        for eye in EYES:
            got = self.sleep_m(eye)
            if got is None:
                continue
            ks, mw, mb = got
            m = mw if comp == "within" else mb
            if np.std(m) == 0:
                continue
            groups = [resid(self.rest[k][key], self.rest[k]["seg"]) for k in ks]
            mu = [self.rest[k][key].mean(axis=0) for k in ks]
            r = _ps_from(groups, mu, w, m)
            r.update(eye=eye)
            out.append(r)
        return out

    # ---------------- restoring rate / diffusion, per condition
    def load_dynamics(self, w, key="Y", equalize=True):
        rows = []
        w = np.asarray(w, float); w = w / np.linalg.norm(w)
        for (fam, ses), cell in self.load.items():
            ns = [len(cell[l][key]) for l in LEVELS]
            nmin = min(ns)
            for l in LEVELS:
                Y, seg = cell[l][key], cell[l]["yseg"]
                if equalize and len(Y) > nmin:
                    Y, seg = Y[:nmin], seg[:nmin]
                p = np.asarray(Y, float) @ w
                lam, a = ar1_lambda(p, seg, DT)
                r = resid(p[:, None], seg)[:, 0]
                if np.isfinite(a) and 0 < a < 1:
                    e = []
                    for s in np.unique(seg):
                        x = p[seg == s]
                        if len(x) < 10:
                            continue
                        x = x - x.mean()
                        e.append(x[1:] - a * x[:-1])
                    sig2 = float(np.var(np.concatenate(e), ddof=1)) if e else np.nan
                else:
                    sig2 = np.nan
                rows.append(dict(family=fam, session=ses, level=l, lam=lam, a=a,
                                 sig2=sig2, D=sig2 / DT if np.isfinite(sig2) else np.nan,
                                 sd=float(r.std(ddof=1)), n=int(len(p))))
        return rows


def _ps_from(groups, mu, w, m):
    w = np.asarray(w, float); w = w / np.linalg.norm(w)
    ms = np.asarray(m, float)
    ms = (ms - ms.mean()) / (ms.std(ddof=1) if ms.std(ddof=1) > 0 else 1.0)
    pm = np.array([float(v @ w) for v in mu])
    sd = np.array([float((np.asarray(g, float) @ w).std(ddof=1)) for g in groups])
    n = np.array([len(g) for g in groups])
    pooled = float(np.sqrt(np.sum((n - 1) * sd ** 2) / max(1, np.sum(n - 1))))
    den = float(np.sum((ms - ms.mean()) ** 2))
    b_pos = float(np.sum((ms - ms.mean()) * (pm - pm.mean())) / den) / (pooled if pooled > 0 else np.nan)
    ls = np.log(np.maximum(sd, 1e-12))
    b_spr = float(np.sum((ms - ms.mean()) * (ls - ls.mean())) / den)
    return dict(beta_pos=b_pos, beta_spr=b_spr, pooled_sd=pooled,
                mu=pm.tolist(), sd=sd.tolist(), n=n.tolist())


# ------------------------------------------------------------------ group machinery
def lopo(axes, i):
    """Leave-one-participant-out group axis.

    NO sign alignment: the sign of each participant's axis is fixed by the manipulation
    (level 2 minus level 0; increasing KSS), so flipping axes to agree with an arbitrary
    reference would manufacture agreement and invert the group direction.
    """
    vs = [v for j, v in axes.items() if j != i and v is not None]
    if not vs:
        return None
    v = np.mean(vs, axis=0)
    n = np.linalg.norm(v)
    return v / n if n > 0 else None


def group_axis(axes, keys=None):
    vs = [axes[j] for j in (keys if keys is not None else axes) if axes.get(j) is not None]
    if not vs:
        return None
    v = np.mean(vs, axis=0)
    n = np.linalg.norm(v)
    return v / n if n > 0 else None


def summarise(name, x, boot, perm):
    m, lo, hi, n = boot_mean(x, boot)
    p = signflip_p(x, perm)
    return dict(name=name, mean=m, lo=lo, hi=hi, n=n, p_signflip=p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--perm", type=int, default=10000)
    a = ap.parse_args()
    t0 = time.time()
    res = {"params": vars(a)}

    log("loading cache ...")
    subs = C.load_subjects()
    kss, rsme = C.load_kss(), C.load_rsme()
    log(f"{len(subs)} participants cached: {sorted(subs)}")
    pca = C.fit_group_pca(subs)
    res["pca"] = dict(d=int(pca.n_components_),
                      evr=[float(v) for v in pca.explained_variance_ratio_],
                      evr_total=float(pca.explained_variance_ratio_.sum()))
    log(f"group PCA d=8 on Flanker: total EVR {res['pca']['evr_total']:.3f}")

    D = {}
    for i, s in subs.items():
        D[i] = Data(s, pca, kss, rsme)
        log(f"  sub-{i:02d}: load cells {len(D[i].load)}  rest epochs {len(D[i].rest)}")
    ids = sorted(D)

    # ---- window rejection: condition dependence
    rej = {}
    for i in ids:
        for k, v in D[i].reject.items():
            rej.setdefault(str(k), []).append(v)
    res["rejection"] = {k: dict(mean=float(np.nanmean(v)), max=float(np.nanmax(v))) for k, v in rej.items()}

    out = {}
    for key, spacename in (("Z", "S248"), ("Z8", "S8")):
        log(f"==== space {spacename} ====")
        wl = {i: D[i].w_load(key) for i in ids}
        wl_nb = {i: D[i].w_load(key, "nback") for i in ids}
        wl_mb = {i: D[i].w_load(key, "matb") for i in ids}
        ws = {i: D[i].w_sleep(key, "within") for i in ids}
        ws_b = {i: D[i].w_sleep(key, "between") for i in ids}

        cells = {k: [] for k in ("LP", "LS", "SP", "SS", "LL", "LD", "SL", "SD",
                                 "LP_rsme", "LS_rsme", "SP_btw", "SS_btw")}
        per_sub = []
        for i in ids:
            WL, WS = lopo(wl, i), lopo(ws, i)
            row = {"sid": i}
            if WL is not None:
                r = D[i].load_cells(WL, key)
                if r:
                    row["LP"] = float(np.mean([x["beta_pos"] for x in r]))
                    row["LS"] = float(np.mean([x["beta_spr"] for x in r]))
                r2 = D[i].load_cells(WL, key, use_rsme=True)
                if r2:
                    row["LP_rsme"] = float(np.mean([x["beta_pos"] for x in r2]))
                    row["LS_rsme"] = float(np.mean([x["beta_spr"] for x in r2]))
            if WS is not None:
                r = D[i].sleep_cells(WS, key, "within")
                if r:
                    row["SP"] = float(np.mean([x["beta_pos"] for x in r]))
                    row["SS"] = float(np.mean([x["beta_spr"] for x in r]))
                r = D[i].sleep_cells(WS, key, "between")
                if r:
                    row["SP_btw"] = float(np.mean([x["beta_pos"] for x in r]))
                    row["SS_btw"] = float(np.mean([x["beta_spr"] for x in r]))
            # cross-axis 2x2, position readout
            if WL is not None:
                r = D[i].load_cells(WL, key)
                row["LL"] = float(np.mean([x["beta_pos"] for x in r])) if r else np.nan
                r = D[i].sleep_cells(WL, key, "within")
                row["SL"] = float(np.mean([x["beta_pos"] for x in r])) if r else np.nan
            if WS is not None:
                r = D[i].load_cells(WS, key)
                row["LD"] = float(np.mean([x["beta_pos"] for x in r])) if r else np.nan
                r = D[i].sleep_cells(WS, key, "within")
                row["SD"] = float(np.mean([x["beta_pos"] for x in r])) if r else np.nan
            per_sub.append(row)
            for k in cells:
                cells[k].append(row.get(k, np.nan))

        cellsum = {k: summarise(k, v, a.boot, a.perm) for k, v in cells.items()}
        # ---- primary interaction
        I_signed = np.array([r.get("LP", np.nan) - r.get("LS", np.nan)
                             - (r.get("SP", np.nan) - r.get("SS", np.nan)) for r in per_sub])
        I_x = np.array([r.get("LL", np.nan) - r.get("LD", np.nan)
                        - (r.get("SL", np.nan) - r.get("SD", np.nan)) for r in per_sub])
        I_abs = np.array([abs(r.get("LP", np.nan)) - abs(r.get("LS", np.nan))
                          - (abs(r.get("SP", np.nan)) - abs(r.get("SS", np.nan))) for r in per_sub])

        # ---- angles and matched ceilings
        ang = {}
        pc = [abs(cos_unit(wl[i], ws[i])) for i in ids if wl[i] is not None and ws[i] is not None]
        pc = [x for x in pc if np.isfinite(x)]
        ang["per_participant_abscos_load_sleep"] = summarise("cos", pc, a.boot, a.perm)
        GL, GS = group_axis(wl), group_axis(ws)
        ang["group_abscos_load_sleep"] = float(abs(cos_unit(GL, GS))) if GL is not None else None
        rng = np.random.default_rng(C.SEED)
        halves = {"load": [], "sleep": [], "cross": []}
        ok_l = [i for i in ids if wl[i] is not None]
        ok_s = [i for i in ids if ws[i] is not None]
        for _ in range(500):
            pl = rng.permutation(ok_l); h1, h2 = pl[:len(pl) // 2], pl[len(pl) // 2:]
            halves["load"].append(abs(cos_unit(group_axis(wl, h1), group_axis(wl, h2))))
            ps_ = rng.permutation(ok_s); g1, g2 = ps_[:len(ps_) // 2], ps_[len(ps_) // 2:]
            halves["sleep"].append(abs(cos_unit(group_axis(ws, g1), group_axis(ws, g2))))
            halves["cross"].append(abs(cos_unit(group_axis(wl, h1), group_axis(ws, g2))))
        for k, v in halves.items():
            v = [x for x in v if np.isfinite(x)]
            ang[f"halfsplit_{k}"] = dict(mean=float(np.mean(v)), lo=float(np.percentile(v, 2.5)),
                                         hi=float(np.percentile(v, 97.5)), n=len(v))
        # participant-level within-family ceilings
        for nm, ax in (("load", wl), ("sleep", ws)):
            vals = [abs(cos_unit(ax[i], ax[j])) for ii, i in enumerate(ids) for j in ids[ii + 1:]
                    if ax.get(i) is not None and ax.get(j) is not None]
            vals = [x for x in vals if np.isfinite(x)]
            ang[f"pairwise_within_{nm}"] = dict(mean=float(np.mean(vals)), lo=float(np.percentile(vals, 2.5)),
                                                hi=float(np.percentile(vals, 97.5)), n=len(vals))
        vals = [abs(cos_unit(wl[i], ws[j])) for i in ids for j in ids
                if i != j and wl.get(i) is not None and ws.get(j) is not None]
        vals = [x for x in vals if np.isfinite(x)]
        ang["pairwise_cross_load_sleep"] = dict(mean=float(np.mean(vals)), lo=float(np.percentile(vals, 2.5)),
                                                hi=float(np.percentile(vals, 97.5)), n=len(vals))
        dim = (248 if key == "Z" else 8)
        rr = rng.standard_normal((20000, dim))
        rr /= np.linalg.norm(rr, axis=1, keepdims=True)
        rc = np.abs((rr[:10000] * rr[10000:]).sum(axis=1))
        ang["random_direction_abscos"] = dict(mean=float(rc.mean()), p95=float(np.percentile(rc, 95)),
                                              dim=dim)
        # gain / tilt shares (S248 only: the directions live in the raw feature space)
        if key == "Z":
            sh = {"load": [], "sleep": []}
            for i in ids:
                Q = gain_tilt_z(subs[i])
                if wl[i] is not None:
                    sh["load"].append(share_in(Q, wl[i]))
                if ws[i] is not None:
                    sh["sleep"].append(share_in(Q, ws[i]))
            Qg = gain_tilt_z(subs[ids[0]])
            ang["gain_tilt_share"] = {k: dict(mean=float(np.mean(v)), lo=float(np.percentile(v, 2.5)),
                                              hi=float(np.percentile(v, 97.5))) for k, v in sh.items()}
            ang["gain_tilt_share_group"] = dict(load=share_in(Qg, GL), sleep=share_in(Qg, GS))

        # ---- cross-family transfer of the load axis
        tr = {"nb_on_matb": [], "mb_on_nback": [], "own_matb": [], "own_nback": []}
        for i in ids:
            Wnb, Wmb = lopo(wl_nb, i), lopo(wl_mb, i)
            if Wnb is not None:
                r = D[i].load_cells(Wnb, key, family="matb")
                if r:
                    tr["nb_on_matb"].append(float(np.mean([x["beta_pos"] for x in r])))
                r = D[i].load_cells(Wnb, key, family="nback")
                if r:
                    tr["own_nback"].append(float(np.mean([x["beta_pos"] for x in r])))
            if Wmb is not None:
                r = D[i].load_cells(Wmb, key, family="nback")
                if r:
                    tr["mb_on_nback"].append(float(np.mean([x["beta_pos"] for x in r])))
                r = D[i].load_cells(Wmb, key, family="matb")
                if r:
                    tr["own_matb"].append(float(np.mean([x["beta_pos"] for x in r])))
        transfer = {k: summarise(k, v, a.boot, a.perm) for k, v in tr.items()}
        transfer["abscos_nback_matb_group"] = float(abs(cos_unit(group_axis(wl_nb), group_axis(wl_mb))))
        transfer["abscos_nback_matb_per_participant"] = summarise(
            "cos", [abs(cos_unit(wl_nb[i], wl_mb[i])) for i in ids
                    if wl_nb.get(i) is not None and wl_mb.get(i) is not None], a.boot, a.perm)

        out[spacename] = dict(
            cells=cellsum, per_subject=per_sub,
            I_signed=summarise("I_signed", I_signed, a.boot, a.perm),
            I_abs=summarise("I_abs", I_abs, a.boot, a.perm),
            I_cross_axis=summarise("I_cross_axis", I_x, a.boot, a.perm),
            angles=ang, transfer=transfer,
            n_axis_load=int(sum(v is not None for v in wl.values())),
            n_axis_sleep=int(sum(v is not None for v in ws.values())))
        np.savez_compressed(paths.out("analyses", "task_location", "out", f"workload_axes_{spacename}.npz"),
                            **{f"load_{i}": wl[i] for i in ids if wl[i] is not None},
                            **{f"sleep_{i}": ws[i] for i in ids if ws[i] is not None},
                            **{f"loadnb_{i}": wl_nb[i] for i in ids if wl_nb[i] is not None},
                            **{f"loadmb_{i}": wl_mb[i] for i in ids if wl_mb[i] is not None})
        log(f"  {spacename}: LP={cellsum['LP']['mean']:.3f} LS={cellsum['LS']['mean']:.3f} "
            f"SP={cellsum['SP']['mean']:.3f} SS={cellsum['SS']['mean']:.3f} "
            f"I={out[spacename]['I_signed']['mean']:.3f}")

    # ---- restoring rate (S8 group load axis, LOPO), per load level
    log("==== restoring rate ====")
    wl8 = {i: D[i].w_load("Z8") for i in ids}
    dyn_rows, per_sub_lam = [], []
    for i in ids:
        W = lopo(wl8, i)
        if W is None:
            continue
        rows = D[i].load_dynamics(W, "Y8")
        for r in rows:
            r["sid"] = i
        dyn_rows += rows
        byfam = {}
        for r in rows:
            byfam.setdefault((r["family"], r["session"]), {})[r["level"]] = r
        slopes_lam, slopes_D, slopes_sd = [], [], []
        for k, cell in byfam.items():
            if len(cell) < 3:
                continue
            lv = np.array([0., 1., 2.]); lv = (lv - lv.mean())
            lam = np.array([cell[l]["lam"] for l in LEVELS])
            Dd = np.array([cell[l]["D"] for l in LEVELS])
            sd = np.array([cell[l]["sd"] for l in LEVELS])
            if np.all(np.isfinite(lam)):
                slopes_lam.append(float(lv @ (lam - lam.mean()) / (lv @ lv)))
            if np.all(np.isfinite(Dd)) and np.all(Dd > 0):
                slopes_D.append(float(lv @ (np.log(Dd) - np.log(Dd).mean()) / (lv @ lv)))
            if np.all(np.isfinite(sd)) and np.all(sd > 0):
                slopes_sd.append(float(lv @ (np.log(sd) - np.log(sd).mean()) / (lv @ lv)))
        per_sub_lam.append(dict(sid=i,
                                dlam=float(np.mean(slopes_lam)) if slopes_lam else np.nan,
                                dlogD=float(np.mean(slopes_D)) if slopes_D else np.nan,
                                dlogsd=float(np.mean(slopes_sd)) if slopes_sd else np.nan,
                                lam_mean=float(np.nanmean([r["lam"] for r in rows]))
                                if any(np.isfinite(r["lam"]) for r in rows) else np.nan))
    res["restoring"] = dict(
        dlam_per_level=summarise("dlam", [r["dlam"] for r in per_sub_lam], a.boot, a.perm),
        dlogD_per_level=summarise("dlogD", [r["dlogD"] for r in per_sub_lam], a.boot, a.perm),
        dlogsd_per_level=summarise("dlogsd", [r["dlogsd"] for r in per_sub_lam], a.boot, a.perm),
        lam_mean=summarise("lam_mean", [r["lam_mean"] for r in per_sub_lam], a.boot, a.perm),
        per_subject=per_sub_lam,
        meg_reference=dict(dlam=0.134, lo=0.065, hi=0.204),
        fmri_reference=dict(dlam=0.086, p=0.005))
    log(f"  dlam/level = {res['restoring']['dlam_per_level']['mean']:.4f} "
        f"[{res['restoring']['dlam_per_level']['lo']:.4f}, {res['restoring']['dlam_per_level']['hi']:.4f}]")

    res["spaces"] = out
    res["elapsed_s"] = time.time() - t0
    Path(paths.out("analyses", "task_location", "out", "workload_axes_summary.json")).write_text(
        json.dumps(res, indent=1, default=float), encoding="utf-8", newline="\n")
    # the figure input: settings, PCA and the S248 cells and transfer readouts
    s248 = out["S248"]
    fig = dict(params=res["params"], pca=res["pca"],
               spaces=dict(S248=dict(cells=s248["cells"], transfer=s248["transfer"],
                                     n_axis_load=s248.get("n_axis_load"))))
    Path(paths.out("figure_reproduction", "inputs", "workload_position_slopes.json")).write_text(
        json.dumps(fig, ensure_ascii=False, indent=1, default=float) + "\n", encoding="utf-8", newline="\n")
    import csv
    with open(paths.out("analyses", "task_location", "out", "workload_axes_cells.csv"), "w", newline="") as fh:
        wri = csv.writer(fh)
        wri.writerow(["space", "quantity", "mean", "lo95", "hi95", "n", "p_signflip"])
        for sp, v in out.items():
            for k, c in v["cells"].items():
                wri.writerow([sp, k, c["mean"], c["lo"], c["hi"], c["n"], c["p_signflip"]])
            for k in ("I_signed", "I_abs", "I_cross_axis"):
                c = v[k]
                wri.writerow([sp, k, c["mean"], c["lo"], c["hi"], c["n"], c["p_signflip"]])
    with open(paths.out("analyses", "task_location", "out", "workload_axes_dynamics_rows.csv"), "w",
              newline="") as fh:
        wri = csv.writer(fh)
        wri.writerow(["sid", "family", "session", "level", "lam", "a", "D", "sd", "n"])
        for r in dyn_rows:
            wri.writerow([r["sid"], r["family"], r["session"], r["level"], r["lam"], r["a"],
                          r["D"], r["sd"], r["n"]])
    log(f"DONE {time.time()-t0:.0f}s -> workload_axes_summary.json, workload_position_slopes.json")


if __name__ == "__main__":
    main()
