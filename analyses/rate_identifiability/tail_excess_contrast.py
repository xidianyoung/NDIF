# -*- coding: utf-8 -*-
r"""What cognitive load does to the temporal structure of the state: the paired within-participant load contrast of
the tail excess (Extended Data Fig. 7a), in COG-BCI (MATB and n-back) and HCP fMRI, in one invocation.

  1. THE PAIRED LOAD CONTRAST.  Delta E_tail = E_tail(high) - E_tail(low) per participant, in the same state
     space, the same direction and the MATCHED block structure: the two levels are truncated to the same number
     of blocks of the same lengths, so the estimator's finite-sample bias is common-mode and cancels in the
     difference.  Its matched CEILING is the same contrast between two halves of ONE level, where nothing
     changes.  Its null is a within-participant permutation of the load labels over blocks, size-matched.
  2. THE DRIFT CONTROL: every block is additionally detrended with its own straight line and the contrast is
     re-read; a contrast that SURVIVES it is not a drift.
  3. DIRECTION: the same paired contrast along the leave-one-participant-out load axis and along 20 orthogonal
     control directions, with the admitted fraction reported.
  4. WHICH PART OF THE SPECTRUM MOVES: a two-exponential-plus-white basis fitted through the estimator; the load
     change is decomposed into amplitude d log(1 - nu), weights d(w_slow / (w_slow + w_fast)) and timescales
     d log a_slow, d log a_fast.
  5. MODEL CLASS by OUT-OF-SAMPLE prediction across held-out lags and held-out blocks: one exponential (the
     single-timescale model), two, three, a stretched exponential and a power law, each with its own white
     component.

States: tail_excess_cog.py (COG-BCI, 1-s non-overlapping windows) and tail_excess_fmri.py (HCP fMRI, TR 0.72 s).
The two cohorts are computed independently (no random stream is shared), and the output records the start time of
each part.  The permutation null, the control directions and the held-out folds draw from per-participant
generators seeded with SEED + injection_lib.seed_offset(participant).

Inputs   NDIF_COGBCI_CACHE, NDIF_COUNT_CACHE/fmri
Outputs  figure_reproduction/inputs/tail_excess_contrasts.json;
         analyses/rate_identifiability/out/tail_excess_contrast_rows.csv, tail_excess_contrast_models.csv
Reported Extended Data Fig. 7a (COG-BCI MATB n = 27, within-level control 15; n-back 15 / 10; HCP fMRI 95 / 90)
Runtime  about 50 min on one desktop core (COG-BCI 12 min, HCP fMRI 36 min)
Usage    python analyses/rate_identifiability/tail_excess_contrast.py [--perm 200]
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
import tail_excess_lib as A                                                # noqa: E402

SEED = 20260922
N_CTRL = 20
FAMILIES = ("exp1", "exp2", "exp3", "stretch", "power")


def log(m):
    print(time.strftime("[%H:%M:%S] ") + str(m), flush=True)

NOTE = ("paired within-participant load contrasts on MATCHED block structure; "
        "the ceiling is the same contrast inside one level; the null permutes the "
        "load labels over blocks; the drift control detrends each block linearly")


def assemble(cog, fmri, n_cog):
    """One output from the two independently computed cohort parts (either may be None).

    Key order: generated, note, the COG-BCI families, HCP fMRI; `generated` records the start time of each part.
    """
    if cog is None or fmri is None:
        return cog if fmri is None else fmri
    out = dict(generated=f"{fmri['generated']} (fMRI part); COG-BCI part {cog['generated']} (dedup, n = {n_cog})",
               note=fmri["note"])
    for k in ("COG-BCI matb", "COG-BCI nback"):
        if k in cog:
            out[k] = cog[k]
    for k, v in fmri.items():
        if k not in ("generated", "note") and not k.startswith("COG-BCI"):
            out[k] = v
    return out


def match_blocks(Blo, Bhi):
    """Identical block structure in the two levels, so the estimator's bias cancels in the
    difference: the same number of blocks, paired longest-to-longest and truncated to equal length."""
    n = min(len(Blo), len(Bhi))
    if n == 0:
        return None, None
    a = sorted(Blo, key=lambda b: -len(b))[:n]
    b = sorted(Bhi, key=lambda b: -len(b))[:n]
    oa, ob = [], []
    for x, y in zip(a, b):
        m = min(len(x), len(y))
        oa.append(np.asarray(x[:m], float))
        ob.append(np.asarray(y[:m], float))
    return oa, ob


def paired(Blo, Bhi, w, maxlag, tail, center="block", r1_floor=A.R1_FLOOR):
    ulo = A.analyse_unit(Blo, w, None, 0, maxlag, tail, False, r1_floor, center)
    uhi = A.analyse_unit(Bhi, w, None, 0, maxlag, tail, False, r1_floor, center)
    if ulo is None or uhi is None:
        return None
    return dict(lo=ulo, hi=uhi, dE=uhi["E_tail"] - ulo["E_tail"],
                dslope=uhi["slope"] - ulo["slope"],
                dlog_a=float(np.log(max(uhi["a"], 1e-6) / max(ulo["a"], 1e-6))),
                dnu=uhi["nu"] - ulo["nu"],
                dlog_r1=float(np.log(max(uhi["r"][1], 1e-6) / max(ulo["r"][1], 1e-6))))


def label_perm_null(Blo, Bhi, w, maxlag, tail, n, rng):
    """Size-matched null: the load labels are permuted over the MATCHED blocks, so the block
    structure, the number of blocks and the participant are all preserved and only the assignment
    of blocks to levels is destroyed."""
    allB = list(Blo) + list(Bhi)
    nlo = len(Blo)
    out = []
    for _ in range(n):
        idx = rng.permutation(len(allB))
        a = [allB[i] for i in idx[:nlo]]
        b = [allB[i] for i in idx[nlo:]]
        p = paired(a, b, w, maxlag, tail)
        if p:
            out.append(p["dE"])
    return np.array(out)


def spectrum_decomposition(u, lens, maxlag, kmax, a_grid):
    """Two exponentials plus white, fitted through the estimator; returns the three channels."""
    from itertools import combinations
    best = None
    for al in combinations(a_grid, 2):
        f = A.fit_mixture(u["r"], lens, maxlag, list(al), kmax)
        if f and (best is None or f["sse"] < best["sse"]):
            best = f
    if best is None:
        return None
    w = np.array(best["weights"])
    al = np.array(best["a_list"])
    o = np.argsort(al)
    w, al = w[o], al[o]
    tot = float(w.sum())
    if tot <= 0:
        return None
    return dict(nu=best["nu"], a_fast=float(al[0]), a_slow=float(al[1]),
                w_fast=float(w[0]), w_slow=float(w[1]),
                frac_slow=float(w[1] / tot), amplitude=float(tot), sse=best["sse"])


def run_pair(units, name, maxlag, tail, kmax, perm, rng, RES, rows, a_grid):
    """Everything for one cohort x family: the paired contrast, its ceiling, its null, the drift
    control, the direction control, the spectrum decomposition and the model comparison."""
    dE, dEc, dSl, dnu, dloga, ceil, nulls = [], [], [], [], [], [], []
    dir_task, dir_ctrl = [], []
    spec = []
    cvs = {f: [] for f in FAMILIES}
    nadm = natt = 0
    for sid, (Blo, Bhi, w) in units.items():
        Blo, Bhi = match_blocks(Blo, Bhi)
        if Blo is None:
            continue
        p = paired(Blo, Bhi, w, maxlag, tail)
        if p is None:
            continue
        dE.append(p["dE"]); dSl.append(p["dslope"]); dnu.append(p["dnu"]); dloga.append(p["dlog_a"])
        rows.append(dict(cohort=name, subject=str(sid), n_blocks=len(Blo),
                         r1_lo=p["lo"]["r"][1], r1_hi=p["hi"]["r"][1],
                         E_lo=p["lo"]["E_tail"], E_hi=p["hi"]["E_tail"], dE=p["dE"],
                         a_lo=p["lo"]["a"], a_hi=p["hi"]["a"], nu_lo=p["lo"]["nu"],
                         nu_hi=p["hi"]["nu"], dslope=p["dslope"], dlog_r1=p["dlog_r1"]))
        # ---- drift control
        q = paired(Blo, Bhi, w, maxlag, tail, center="linear")
        if q:
            dEc.append(q["dE"])
        # ---- matched ceiling: the same contrast inside ONE level
        h = len(Blo) // 2
        if h >= 2:
            c = paired(Blo[:h], Blo[h:2 * h], w, maxlag, tail)
            if c:
                ceil.append(c["dE"])
        # ---- size-matched label permutation
        if perm:
            nulls.append(label_perm_null(Blo, Bhi, w, maxlag, tail, perm,
                                         np.random.default_rng(SEED + L.seed_offset(str(sid)) % 9999)))
        # ---- direction control
        r2 = np.random.default_rng(SEED + 31 + L.seed_offset(str(sid)) % 9999)
        cv = []
        for v in A.orth_directions(w, N_CTRL, r2):
            natt += 1
            pv = paired(Blo, Bhi, v, maxlag, tail)
            if pv:
                nadm += 1
                cv.append(pv["dE"])
        if cv:
            dir_task.append(p["dE"]); dir_ctrl.append(float(np.mean(cv)))
        # ---- spectrum decomposition
        _, lens = A.acf_blocks(Blo, w, maxlag)
        slo = spectrum_decomposition(p["lo"], lens, maxlag, kmax, a_grid)
        shi = spectrum_decomposition(p["hi"], lens, maxlag, kmax, a_grid)
        if slo and shi:
            spec.append((slo, shi))
        # ---- out-of-sample model comparison, at the HIGH level
        cv2 = A.cv_families(Bhi, w, maxlag, FAMILIES, kmax, maxlag, r2)
        if cv2:
            for f in FAMILIES:
                if np.isfinite(cv2.get(f, np.nan)):
                    cvs[f].append(cv2[f])

    if not dE:
        return
    g = A.boot_ci(dE)
    res = dict(n=len(dE), dE_tail=g, p_signflip=A.signflip_p(dE),
               dE_tail_detrended=A.boot_ci(dEc), p_detrended=A.signflip_p(dEc),
               ceiling_within_level=A.boot_ci(ceil), p_ceiling=A.signflip_p(ceil),
               dslope=A.boot_ci(dSl), dnu=A.boot_ci(dnu), dlog_a=A.boot_ci(dloga))
    if nulls:
        nb = min(len(x) for x in nulls)
        gm = np.array([x[:nb] for x in nulls]).mean(axis=0)
        res["label_perm"] = dict(mean=float(np.mean(gm)), lo=float(np.percentile(gm, 2.5)),
                                 hi=float(np.percentile(gm, 97.5)),
                                 p=A.perm_p(gm, g["mean"], "greater"), n_perm=int(nb))
    if dir_task:
        n = len(dir_task)
        res["direction"] = dict(task_aligned=A.boot_ci(dir_task), control=A.boot_ci(dir_ctrl),
                                paired=A.boot_ci(np.array(dir_task) - np.array(dir_ctrl)),
                                p=A.signflip_p(np.array(dir_task) - np.array(dir_ctrl)),
                                control_admissible=nadm, control_attempted=natt)
    if spec:
        ch = {}
        for k in ("amplitude", "frac_slow", "a_slow", "a_fast", "nu"):
            d = [(s1[k] - s0[k]) if k in ("frac_slow", "nu") else
                 float(np.log(max(s1[k], 1e-9) / max(s0[k], 1e-9))) for s0, s1 in spec]
            ch[("d_" if k in ("frac_slow", "nu") else "dlog_") + k] = A.boot_ci(d)
            ch[("p_" + k)] = A.signflip_p(d)
        ch["n"] = len(spec)
        ch["low"] = {k: float(np.mean([s0[k] for s0, _ in spec]))
                     for k in ("amplitude", "frac_slow", "a_slow", "a_fast", "nu")}
        ch["high"] = {k: float(np.mean([s1[k] for _, s1 in spec]))
                      for k in ("amplitude", "frac_slow", "a_slow", "a_fast", "nu")}
        res["spectrum"] = ch
    res["model_comparison_high"] = {f: A.boot_ci(v) for f, v in cvs.items() if v}
    RES[name] = res

    log(f"  {name}: n={res['n']}  dE_tail={g['mean']:+.4f} [{g['lo']:+.4f},{g['hi']:+.4f}] "
        f"P={res['p_signflip']:.4g}")
    log(f"        ceiling (same level) {res['ceiling_within_level']['mean']:+.4f} "
        f"[{res['ceiling_within_level']['lo']:+.4f},{res['ceiling_within_level']['hi']:+.4f}]; "
        f"detrended {res['dE_tail_detrended']['mean']:+.4f} "
        f"[{res['dE_tail_detrended']['lo']:+.4f},{res['dE_tail_detrended']['hi']:+.4f}] "
        f"P={res['p_detrended']:.4g}"
        + (f"; label-permutation P={res['label_perm']['p']:.4g}" if "label_perm" in res else ""))
    if "direction" in res:
        d = res["direction"]
        log(f"        task-aligned {d['task_aligned']['mean']:+.4f} vs "
            f"{d['control_admissible']}/{d['control_attempted']} admissible controls "
            f"{d['control']['mean']:+.4f}; paired {d['paired']['mean']:+.4f} "
            f"[{d['paired']['lo']:+.4f},{d['paired']['hi']:+.4f}] P={d['p']:.4g}")
    if "spectrum" in res:
        s = res["spectrum"]
        log(f"        spectrum (n={s['n']}): amplitude {s['dlog_amplitude']['mean']:+.4f} "
            f"[{s['dlog_amplitude']['lo']:+.4f},{s['dlog_amplitude']['hi']:+.4f}] "
            f"P={s['p_amplitude']:.3g} | slow WEIGHT {s['d_frac_slow']['mean']:+.4f} "
            f"[{s['d_frac_slow']['lo']:+.4f},{s['d_frac_slow']['hi']:+.4f}] "
            f"P={s['p_frac_slow']:.3g} | slow TIMESCALE {s['dlog_a_slow']['mean']:+.4f} "
            f"[{s['dlog_a_slow']['lo']:+.4f},{s['dlog_a_slow']['hi']:+.4f}] "
            f"P={s['p_a_slow']:.3g}")
    if res["model_comparison_high"]:
        o = sorted(res["model_comparison_high"].items(), key=lambda kv: kv[1]["mean"])
        log("        held-out model ranking (rms, lower is better): "
            + ", ".join(f"{k} {v['mean']:.4f}" for k, v in o))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perm", type=int, default=200)
    ap.add_argument("--subs", type=int, default=0)
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    RES_COG = RES_FMRI = None
    n_cog = 0
    rows, mrows = [], []
    A_GRID = np.concatenate([np.linspace(0.05, 0.88, 18), np.linspace(0.90, 0.985, 8)])

    if a.only in ("", "cog"):
        import cog_features as C
        import tail_excess_cog as G
        RES = RES_COG = dict(generated=time.strftime("%Y-%m-%d %H:%M:%S"), note=NOTE)
        subs = C.load_subjects()
        pca = C.fit_group_pca(subs, G.D_PCA, seed=SEED)
        data, axes = G.build(subs, pca)
        log(f"COG-BCI: {len(data)} participants")
        n_cog = len(data)
        for fam in ("matb", "nback"):
            units = {}
            for sid in sorted(data):
                if (fam, 0) not in data[sid] or (fam, 2) not in data[sid] or sid not in axes:
                    continue
                units[sid] = (data[sid][(fam, 0)], data[sid][(fam, 2)], G.lopo(axes, sid))
            run_pair(units, f"COG-BCI {fam}", G.MAXLAG, G.TAIL, 5, a.perm, rng, RES, rows, A_GRID)

    if a.only in ("", "fmri"):
        import tail_excess_fmri as F
        RES = RES_FMRI = dict(generated=time.strftime("%Y-%m-%d %H:%M:%S"), note=NOTE)
        subs = F.load_subjects(a.subs)
        ax = F.load_axis(subs)
        log(f"HCP fMRI: {len(subs)} participants")
        units = {sid: (subs[sid][0], subs[sid][1], F.lopo(ax, sid)) for sid in sorted(subs)}
        run_pair(units, "HCP fMRI", F.MAXLAG, (F.TAIL0, F.TAIL1), 5, a.perm, rng, RES, rows, A_GRID)

    RES = assemble(RES_COG, RES_FMRI, n_cog)
    for k, v in RES.items():
        if isinstance(v, dict) and "model_comparison_high" in v:
            for f, g in v["model_comparison_high"].items():
                mrows.append(dict(cohort=k, family=f, rms=g["mean"], lo=g["lo"], hi=g["hi"],
                                  n=g["n"]))
    target = paths.out("figure_reproduction", "inputs", "tail_excess_contrasts.json")
    Path(target).write_text(json.dumps(RES, indent=1, default=float), encoding="utf-8")
    _csv(L.out("tail_excess_contrast_rows.csv"), rows)
    _csv(L.out("tail_excess_contrast_models.csv"), mrows)
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
