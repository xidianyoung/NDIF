# -*- coding: utf-8 -*-
r"""Tail excess of the state autocorrelation in HCP MEG, on the trial clock (observed points of the identifiability
map, HCP MEG).

WHAT THIS COHORT CANNOT DECIDE.  The per-trial MEG state is built WITHIN trials and never across them (the HCP
trials carry per-trial offsets and a fractional-sample resampling phase).  The non-overlapping series is exactly two
1-s windows per trial, [0,1) and [1,2) s after stimulus onset: ONE lag-1 pair per trial, so r(2) does not exist and
the successive ratio cannot be formed within a trial.  This is stated and no within-trial verdict is returned.

WHAT IT CAN DO: A SECOND, SLOWER CLOCK.  One window per trial at a fixed latency gives a NON-OVERLAPPING series at
the trial rate (about 2.5 s), whose autocorrelation reaches lag 5 inside a contiguous run of trials.  Caveats:
(i) the cache carries trial INDICES but not block identities, so a contiguous run can straddle one inter-block gap;
runs are centred on their own mean, and the run-length distribution is reported; (ii) the series is
stimulus-locked, so the matched control is the same series in the 0-back condition.

Inputs   NDIF_MEG_CACHE (<subject>.npz: X0/Y0/X1/Y1, trial0/trial1, load axis v; analyses/cohort_states/hcp_meg_states.py)
Outputs  analyses/rate_identifiability/out/tail_excess_meg.json, tail_excess_meg_rows.csv (read by
         identifiability_map.py: clock trial, level 0bk, columns a, nu, r1)
Reported Fig. 3a / Extended Data Fig. 1 (observed point of HCP MEG)
Runtime  about 2 min on one desktop core
Usage    python analyses/rate_identifiability/tail_excess_meg.py [--nrep 200] [--perm 2000]
"""
from __future__ import annotations

import argparse
import csv
import collections
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import injection_lib as L                                                  # noqa: E402
import tail_excess_lib as A                                                # noqa: E402

CACHE = L.MEG_CACHE
DT_TRIAL = 2.5                     # nominal trial SOA; the trial clock, not the sample clock
SEED = 20260922
MAXLAG = 5
TAIL = (3, 5)
MIN_BLOCK = MAXLAG + 4
N_CTRL = 20


def log(m):
    print(time.strftime("[%H:%M:%S] ") + str(m), flush=True)


def load():
    out = {}
    for f in sorted(CACHE.glob("*.npz")):
        z = np.load(f)
        out[f.stem] = {k: z[k].astype(float) if z[k].dtype.kind == "f" else z[k]
                       for k in z.files}
    return out


def trial_series(X, trials, min_len=MIN_BLOCK):
    """Contiguous runs of trial indices -> blocks of the across-trial series at one latency."""
    o = np.argsort(trials)
    return A.split_contiguous(X[o], np.asarray(trials)[o], min_len)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nrep", type=int, default=200)
    ap.add_argument("--perm", type=int, default=2000)
    a = ap.parse_args()
    t0 = time.time()
    D = load()
    log(f"HCP MEG: {len(D)} participants")

    # ---- the structural limit, measured from the cache rather than asserted
    runs0 = collections.Counter()
    npair = []
    for sid, z in D.items():
        idx = np.sort(z["trial0"])
        cuts = np.flatnonzero(np.diff(idx) != 1) + 1
        for p in np.split(idx, cuts):
            runs0[len(p)] += 1
        npair.append(len(z["X0"]))
    RES = dict(cohort="HCP MEG", n_participants=len(D), maxlag_within_trial=1,
               within_trial=dict(
                   windows_per_trial=2, step_s=1.0, overlap="none",
                   verdict="NOT DECIDABLE",
                   reason=("the dyn design gives exactly one lag-1 pair per trial, so r(2) does "
                           "not exist; the only longer series is the 50 %-overlap dist grid, "
                           "which is not used because its overlap would manufacture a second timescale"),
                   median_pairs_per_participant=float(np.median(npair))),
               trial_clock=dict(dt_s=DT_TRIAL, maxlag=MAXLAG, tail=list(TAIL),
                                run_length_distribution={str(k): int(v)
                                                         for k, v in sorted(runs0.items())}),
               r1_floor=A.R1_FLOOR, generated=time.strftime("%Y-%m-%d %H:%M:%S"))
    log(f"  within-trial: {RES['within_trial']['verdict']} -- "
        f"{RES['within_trial']['median_pairs_per_participant']:.0f} lag-1 pairs per participant, "
        f"no lag 2")
    log(f"  trial clock: contiguous trial-run lengths {dict(sorted(runs0.items()))}")

    rows = []
    # ---- the load axis, leave-one-participant-out
    V = {sid: z["v"] / (np.linalg.norm(z["v"]) + 1e-12) for sid, z in D.items()}

    def lopo(sid):
        v = np.mean([u for s, u in V.items() if s != sid], axis=0)
        return v / (np.linalg.norm(v) + 1e-12)

    for lev, name in ((0, "0bk"), (1, "2bk")):
        units, nulls, profiles, ceil = {}, [], [], []
        for sid, z in D.items():
            X = z["X0"] if lev == 0 else z["X1"]
            tr = z["trial0"] if lev == 0 else z["trial1"]
            B = trial_series(X, tr)
            if len(B) < 2:
                continue
            w = lopo(sid)
            u = A.analyse_unit(B, w, np.random.default_rng(SEED + L.seed_offset(sid) % 9999), a.nrep,
                               MAXLAG, TAIL)
            if u is None:
                continue
            units[sid] = u
            nulls.append(u["null_E"])
            profiles.append((np.array(u["ks"]), np.array(u["rho"])))
            ceil.append(A.half_ceiling(B, w, np.random.default_rng(SEED + 77 + L.seed_offset(sid) % 9999),
                                       MAXLAG, TAIL))
            rows.append(dict(cohort="MEG", clock="trial", level=name, subject=sid,
                             n_blocks=u["n_blocks"], n_win=u["n_win"], r1=u["r"][1], r2=u["r"][2],
                             r3=u["r"][3], exp_index=u["exp_index"], parity=u["parity_amp"],
                             a=u["a"], nu=u["nu"], E_tail=u["E_tail"],
                             E_tail_corr=u["E_tail_corr"], slope=u["slope"]))
        if not units:
            continue
        gE = A.group_stat([u["E_tail"] for u in units.values()], f"tail excess {name}", nulls)
        gS = A.group_stat([u["slope"] for u in units.values()], f"ratio slope {name}",
                          [u["null_slope"] for u in units.values()])
        lp = A.lagperm_null(profiles, a.perm, SEED)
        gS["p_lagperm"] = A.perm_p(lp, gS["mean"], "greater")
        RES[name] = dict(n=len(units), n_attempted=len(D), tail_excess=gE, ratio_slope=gS,
                         ceiling_halfdiff=A.boot_ci([c for c in ceil if np.isfinite(c)]),
                         exp_index=A.boot_ci([u["exp_index"] for u in units.values()]),
                         parity=A.boot_ci([u["parity_amp"] for u in units.values()]),
                         a_hat=A.boot_ci([u["a"] for u in units.values()]),
                         nu_hat=A.boot_ci([u["nu"] for u in units.values()]),
                         mean_r=np.mean([u["r"] for u in units.values()], axis=0).tolist(),
                         mean_pred=np.mean([u["pred"] for u in units.values()], axis=0).tolist(),
                         units={k: v["E_tail"] for k, v in units.items()})
        c = RES[name]["ceiling_halfdiff"]
        log(f"  trial clock {name}: n={len(units)}/{len(D)} "
            f"exp_index={RES[name]['exp_index']['mean']:.3f} "
            f"a={RES[name]['a_hat']['mean']:.3f} nu={RES[name]['nu_hat']['mean']:.3f}")
        log(f"        E_tail={gE['mean']:+.4f} [{gE['lo']:+.4f},{gE['hi']:+.4f}] vs single timescale "
            f"{gE.get('null_mean', float('nan')):+.4f} P={gE.get('p_single_timescale_model', float('nan')):.4f}; "
            f"ceiling {c['mean']:+.4f} [{c['lo']:+.4f},{c['hi']:+.4f}]")

    if "0bk" in RES and "2bk" in RES:
        common = sorted(set(RES["0bk"]["units"]) & set(RES["2bk"]["units"]))
        d = np.array([RES["2bk"]["units"][s] - RES["0bk"]["units"][s] for s in common])
        RES["load_difference"] = dict(n=len(common), E_tail=A.boot_ci(d), p=A.signflip_p(d))
        g = RES["load_difference"]["E_tail"]
        log(f"  2bk - 0bk paired in {len(common)}: E_tail {g['mean']:+.4f} "
            f"[{g['lo']:+.4f},{g['hi']:+.4f}] P={RES['load_difference']['p']:.4f}")

    # ---- control directions on the trial clock
    Ea, Ec, nadm, natt = [], [], 0, 0
    for sid, z in D.items():
        B = trial_series(z["X0"], z["trial0"])
        if len(B) < 2:
            continue
        r2 = np.random.default_rng(SEED + 500 + L.seed_offset(sid) % 9999)
        w = lopo(sid)
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

    for k in ("0bk", "2bk"):
        if k in RES:
            RES[k].pop("units", None)
    L.out("tail_excess_meg.json").write_text(json.dumps(RES, indent=1, default=float), encoding="utf-8")
    if rows:
        with open(L.out("tail_excess_meg_rows.csv"), "w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, list(rows[0].keys()))
            wr.writeheader()
            for r in rows:
                wr.writerow(r)
    log(f"done in {time.time() - t0:.0f}s -> tail_excess_meg.json")


if __name__ == "__main__":
    main()
