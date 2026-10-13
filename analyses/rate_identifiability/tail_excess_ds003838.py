# -*- coding: utf-8 -*-
r"""Tail excess of the state autocorrelation in ds003838 (digit span, n = 64), with the listen-only control arm and
with load as literally the number of items held (Extended Data Fig. 7a, ds003838 row).

Cohort and state space are those of injection_lib: the per-sequence feature cache (item slots from the TRUE
per-item onsets, inter-item interval 2.0330 s) and a robust z + PCA (d = 8) fitted on the LISTEN-ONLY arm, i.e.
blind to memory load.

WINDOWS.  1-s windows at a 1-s step, NON-OVERLAPPING.  A block is one contiguous run of kept windows inside ONE
sequence, spanning the item presentation (slots 1..L): 18 and 26 windows at set sizes 9 and 13.  The
autocorrelation therefore reaches lag 6, the maximum this design supports.

WHY THIS COHORT IS THE PARITY TEST.  One item arrives every 2.0330 s against a 1-s window, so the stimulus train is
at almost exactly period 2 in window units.  The listen-only arm has the identical timing and no memory load:
anything shared by the two arms is the stimulus train or the recording, and only the DIFFERENCE can be memory load.

Inputs   NDIF_DS003838_CACHE (sub-XXX.npz, sub-XXX_meta.json)
Outputs  figure_reproduction/inputs/tail_excess_ds003838.json;
         analyses/rate_identifiability/out/tail_excess_ds003838_rows.csv, tail_excess_ds003838_profile.csv,
         tail_excess_ds003838_calibration.csv
Reported Extended Data Fig. 7a (ds003838: memory minus listen-only tail excess, n = 50)
Runtime  about 5 min on one desktop core
Usage    python analyses/rate_identifiability/tail_excess_ds003838.py [--nrep 200] [--perm 2000]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyses.common import paths                                          # noqa: E402
import injection_lib as L                                                  # noqa: E402
from injection_lib import RefSpace, load_cache, log                        # noqa: E402
import tail_excess_lib as A                                                # noqa: E402

DT = 1.0
D_PCA = 8
SEED = 20260922
MAXLAG = 6
TAIL = (3, 6)
MIN_BLOCK = MAXLAG + 8
LSET = (9, 13)
BINS = {"A": (1, 4), "C": (9, 13)}
N_CTRL = 20


def prep(sub, inject=None, rng=None):
    """Per-arm block lists over the whole presentation, plus the per-bin means for the load axis."""
    S = load_cache(sub)
    Fc = [S["F_dyn"][i][S["keep_y"][i]] for i in range(S["n"]) if S["cond"][i] == 0]
    Fc = [f for f in Fc if len(f)]
    if not Fc:
        return None
    sp = RefSpace(D_PCA).fit(np.vstack(Fc))
    out = {arm: [] for arm in (0, 1)}
    binmu = {}
    for i in range(S["n"]):
        L = S["n_digits"][i]
        if L not in LSET:
            continue
        arm = S["cond"][i]
        k = S["slot_true_dyn"][i]
        m = S["keep_y"][i] & (k >= 1) & (k <= L)
        idx = np.flatnonzero(m)
        if len(idx) == 0:
            continue
        F = S["F_dyn"][i]
        if inject is not None:
            F = inject(F, np.zeros(len(F), int), rng)
        Z = sp.transform(F)
        out[arm] += A.split_contiguous(Z[idx], idx, MIN_BLOCK)
        kd = S["slot_true"][i]
        Fd = S["F_dist"][i]
        for bn, (lo, hi) in BINS.items():
            md = S["keep"][i] & (kd >= lo) & (kd <= hi)
            if md.sum() >= 4:
                binmu.setdefault((arm, bn), []).append(sp.transform(Fd[md]).mean(axis=0))
    ax = None
    if (1, "C") in binmu and (1, "A") in binmu:
        u = np.mean(binmu[(1, "C")], 0) - np.mean(binmu[(1, "A")], 0)
        ax = u / (np.linalg.norm(u) + 1e-12)
    # load-bin block lists inside the memory arm, for the item-count contrast
    bins = {}
    for i in range(S["n"]):
        L = S["n_digits"][i]
        if L != 13 or S["cond"][i] != 1:
            continue
        k = S["slot_true_dyn"][i]
        for bn, (lo, hi) in (("early", (1, 6)), ("late", (8, 13))):
            m = S["keep_y"][i] & (k >= lo) & (k <= hi)
            idx = np.flatnonzero(m)
            if len(idx) == 0:
                continue
            Z = sp.transform(S["F_dyn"][i])
            bins.setdefault(bn, []).extend(A.split_contiguous(Z[idx], idx, MAXLAG + 4))
    return out, ax, bins


def lopo(axes, sid):
    v = np.mean([a for s, a in axes.items() if s != sid], axis=0)
    return v / (np.linalg.norm(v) + 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nrep", type=int, default=200)
    ap.add_argument("--perm", type=int, default=2000)
    ap.add_argument("--subs", type=int, default=0)
    ap.add_argument("--skip-calib", action="store_true")
    a = ap.parse_args()
    t0 = time.time()

    subs = L.subjects()
    if a.subs:
        subs = subs[: a.subs]
    log(f"ds003838: {len(subs)} participants; non-overlapping 1-s windows, maxlag {MAXLAG}")

    P, axes, BINS_ = {}, {}, {}
    for s in subs:
        try:
            r = prep(s)
        except Exception as e:                                            # noqa: BLE001
            log(f"  {s}: {e}")
            continue
        if r is None:
            continue
        P[s], ax, BINS_[s] = r[0], r[1], r[2]
        if ax is not None:
            axes[s] = ax
    log(f"  {len(P)} usable, {len(axes)} with a load axis")

    RES = dict(cohort="ds003838", n_participants=len(P), dt=DT, d=D_PCA, maxlag=MAXLAG,
               tail=list(TAIL), window_s=1.0, step_s=1.0, overlap="none",
               set_sizes=list(LSET), r1_floor=A.R1_FLOOR,
               generated=time.strftime("%Y-%m-%d %H:%M:%S"))
    rows, prof = [], []

    for arm, name in ((1, "memory"), (0, "listen")):
        units, nulls, profiles, ceil = {}, [], [], []
        for sid in sorted(P):
            B = P[sid][arm]
            if len(B) < 4 or sid not in axes:
                continue
            w = lopo(axes, sid)
            u = A.analyse_unit(B, w, np.random.default_rng(SEED + L.seed_offset(sid) % 9999), a.nrep,
                               MAXLAG, TAIL)
            if u is None:
                continue
            units[sid] = u
            nulls.append(u["null_E"])
            profiles.append((np.array(u["ks"]), np.array(u["rho"])))
            ceil.append(A.half_ceiling(B, w, np.random.default_rng(SEED + 99 + L.seed_offset(sid) % 9999),
                                       MAXLAG, TAIL))
            rows.append(dict(cohort="ds003838", arm=name, subject=sid, n_blocks=u["n_blocks"],
                             n_win=u["n_win"], r1=u["r"][1], r2=u["r"][2], r3=u["r"][3],
                             exp_index=u["exp_index"], parity=u["parity_amp"], a=u["a"],
                             nu=u["nu"], E_tail=u["E_tail"], E_tail_raw=u["E_tail_raw"],
                             E_tail_corr=u["E_tail_corr"], slope=u["slope"], slope2=u["slope2"]))
        if not units:
            continue
        gE = A.group_stat([u["E_tail"] for u in units.values()], f"tail excess {name}", nulls)
        gS = A.group_stat([u["slope"] for u in units.values()], f"ratio slope {name}",
                          [u["null_slope"] for u in units.values()])
        lp = A.lagperm_null(profiles, a.perm, SEED)
        gS["p_lagperm"] = A.perm_p(lp, gS["mean"], "greater")
        RES[name] = dict(n=len(units), tail_excess=gE, ratio_slope=gS,
                         ceiling_halfdiff=A.boot_ci([c for c in ceil if np.isfinite(c)]),
                         parity=A.boot_ci([u["parity_amp"] for u in units.values()]),
                         exp_index=A.boot_ci([u["exp_index"] for u in units.values()]),
                         a_hat=A.boot_ci([u["a"] for u in units.values()]),
                         nu_hat=A.boot_ci([u["nu"] for u in units.values()]),
                         mean_r=np.mean([u["r"] for u in units.values()], axis=0).tolist(),
                         mean_pred=np.mean([u["pred"] for u in units.values()], axis=0).tolist(),
                         units={k: v["E_tail"] for k, v in units.items()},
                         units_parity={k: v["parity_amp"] for k, v in units.items()})
        mr, mp = np.array(RES[name]["mean_r"]), np.array(RES[name]["mean_pred"])
        for k in range(1, MAXLAG):
            prof.append(dict(cohort="ds003838", arm=name, k=k, lag_s=k * DT, r_obs=mr[k],
                             r_single_timescale=mp[k], rho_obs=mr[k + 1] / mr[k] if mr[k] > 1e-3 else np.nan,
                             rho_single_timescale=mp[k + 1] / mp[k] if mp[k] > 1e-3 else np.nan))
        c = RES[name]["ceiling_halfdiff"]
        log(f"  {name}: n={len(units)} exp_index={RES[name]['exp_index']['mean']:.3f} "
            f"parity={RES[name]['parity']['mean']:+.3f} a={RES[name]['a_hat']['mean']:.3f} "
            f"nu={RES[name]['nu_hat']['mean']:.3f}")
        log(f"        E_tail={gE['mean']:+.4f} [{gE['lo']:+.4f},{gE['hi']:+.4f}] vs single timescale "
            f"{gE.get('null_mean', float('nan')):+.4f} P={gE.get('p_single_timescale_model', float('nan')):.4f}; "
            f"ceiling {c['mean']:+.4f} [{c['lo']:+.4f},{c['hi']:+.4f}]")

    # ---- the arm difference: only this can be memory load
    if "memory" in RES and "listen" in RES:
        common = sorted(set(RES["memory"]["units"]) & set(RES["listen"]["units"]))
        dE = np.array([RES["memory"]["units"][s] - RES["listen"]["units"][s] for s in common])
        dP = np.array([RES["memory"]["units_parity"][s] - RES["listen"]["units_parity"][s]
                       for s in common])
        RES["arm_difference"] = dict(n=len(common), E_tail=A.boot_ci(dE),
                                     p=A.signflip_p(dE), parity=A.boot_ci(dP))
        g = RES["arm_difference"]["E_tail"]
        log(f"  memory - listen, paired in {len(common)}: E_tail {g['mean']:+.4f} "
            f"[{g['lo']:+.4f},{g['hi']:+.4f}] P={RES['arm_difference']['p']:.4f}; "
            f"parity difference {RES['arm_difference']['parity']['mean']:+.4f} "
            f"[{RES['arm_difference']['parity']['lo']:+.4f},"
            f"{RES['arm_difference']['parity']['hi']:+.4f}]")

    # ---- does the profile change with the NUMBER OF ITEMS HELD (set size 13, early vs late)
    de, dv = [], []
    for sid in sorted(BINS_):
        b = BINS_[sid]
        if "early" not in b or "late" not in b or sid not in axes:
            continue
        w = lopo(axes, sid)
        ue = A.analyse_unit(b["early"], w, None, 0, MAXLAG - 2, (3, MAXLAG - 2))
        ul = A.analyse_unit(b["late"], w, None, 0, MAXLAG - 2, (3, MAXLAG - 2))
        if ue and ul:
            de.append(ul["E_tail"] - ue["E_tail"])
            dv.append(np.log(max(ul["a"], 1e-6) / max(ue["a"], 1e-6)))
    if de:
        RES["items_held"] = dict(n=len(de), dE_tail=A.boot_ci(de), p=A.signflip_p(de),
                                 dlog_a=A.boot_ci(dv), p_a=A.signflip_p(dv),
                                 note="set size 13; slots 8-13 minus slots 1-6, same sequences")
        log(f"  items held (late minus early, L=13, n={len(de)}): "
            f"dE_tail={RES['items_held']['dE_tail']['mean']:+.4f} "
            f"[{RES['items_held']['dE_tail']['lo']:+.4f},"
            f"{RES['items_held']['dE_tail']['hi']:+.4f}] P={RES['items_held']['p']:.4f}; "
            f"dlog a={RES['items_held']['dlog_a']['mean']:+.4f} "
            f"[{RES['items_held']['dlog_a']['lo']:+.4f},"
            f"{RES['items_held']['dlog_a']['hi']:+.4f}] P={RES['items_held']['p_a']:.4f}")

    # ---- control directions
    Ea, Ec, nadm, natt = [], [], 0, 0
    for sid in sorted(P):
        B = P[sid][1]
        if len(B) < 4 or sid not in axes:
            continue
        r2 = np.random.default_rng(SEED + 500 + L.seed_offset(sid) % 9999)
        w = lopo(axes, sid)
        u = A.analyse_unit(B, w, None, 0, MAXLAG, TAIL)
        if u is None:
            continue
        cv = []
        for v in A.orth_directions(w, N_CTRL, r2):
            natt += 1
            uu = A.analyse_unit(B, v, None, 0, MAXLAG, TAIL)
            if uu:
                nadm += 1
                cv.append(uu["E_tail"])
        if cv:
            Ea.append(u["E_tail"]); Ec.append(float(np.mean(cv)))
    if Ea:
        RES["direction"] = dict(task_aligned=A.boot_ci(Ea), control_mean=A.boot_ci(Ec),
                                paired_difference=A.boot_ci(np.array(Ea) - np.array(Ec)),
                                n_control_directions=N_CTRL, control_admissible=nadm,
                                control_attempted=natt)
        d = RES["direction"]["paired_difference"]
        log(f"  direction: task-aligned {np.mean(Ea):+.4f} vs {nadm}/{natt} admissible controls "
            f"{np.mean(Ec):+.4f}; paired {d['mean']:+.4f} [{d['lo']:+.4f},{d['hi']:+.4f}]")

    # ---- end-to-end calibration
    calib = []
    if not a.skip_calib:
        for kind, sizes in (("white", (0.10, 0.25, 0.50)), ("slow", (0.02, 0.05, 0.10, 0.20))):
            for sz in sizes:
                E = []
                for sid in sorted(P):
                    if sid not in axes:
                        continue
                    rr = np.random.default_rng(SEED + 31 + L.seed_offset(sid) % 9999)
                    inj = ((lambda F, s, r, n=sz: A.inject_white(F, n, r)) if kind == "white"
                           else (lambda F, s, r, w_=sz: A.inject_slow(F, s, w_, 0.95, r)))
                    try:
                        q = prep(sid, inj, rr)
                    except Exception:                                     # noqa: BLE001
                        continue
                    if q is None:
                        continue
                    u = A.analyse_unit(q[0][1], lopo(axes, sid), None, 0, MAXLAG, TAIL)
                    if u:
                        E.append(u["E_tail"])
                if E:
                    g = A.boot_ci(E)
                    calib.append(dict(cohort="ds003838", injection=kind, size=sz,
                                      a_slow=0.95 if kind == "slow" else "",
                                      E_tail=g["mean"], lo=g["lo"], hi=g["hi"], n=g["n"]))
                    log(f"  CALIB {kind} size={sz}: E_tail={g['mean']:+.4f} "
                        f"[{g['lo']:+.4f},{g['hi']:+.4f}]")
    RES["calibration"] = calib

    for k in ("memory", "listen"):
        if k in RES:
            RES[k].pop("units", None)
            RES[k].pop("units_parity", None)
    target = paths.out("figure_reproduction", "inputs", "tail_excess_ds003838.json")
    Path(target).write_text(json.dumps(RES, indent=1, default=float) + "\n", encoding="utf-8")
    _csv(L.out("tail_excess_ds003838_rows.csv"), rows)
    _csv(L.out("tail_excess_ds003838_profile.csv"), prof)
    _csv(L.out("tail_excess_ds003838_calibration.csv"), calib)
    log(f"done in {time.time() - t0:.0f}s -> {target}")


def _csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, list(rows[0].keys()))
        wr.writeheader()
        for r in rows:
            wr.writerow(r)


if __name__ == "__main__":
    main()
