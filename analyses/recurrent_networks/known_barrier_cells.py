"""The equilibrium counter and the density barrier on simulated recordings whose answer is known: a single basin
(one-well cells) and a known non-zero barrier (two-well cells).

Construction 1 -- a known single basin.  Each network is taken to its compensated monostable operating point
(network_lib.base_geometry) and its single-basin status is established without the counter, by exhaustive
fixed-point enumeration of the 64-unit network (600 starts and the completeness probe).  Segments are simulated
from that one basin with no manipulation, split into two "conditions" and handed to the counters.  Every cell has
ground truth: one basin, no condition difference (arm `single_basin`).

Construction 2 -- a known non-zero barrier.  The bistable network at its operating gain 0.86, whose
Freidlin-Wentzell actions are in the escape-ladder output (rnn_escape_simulation.csv).  Segments are drawn from
one well (chains that leave are detected and dropped, and the surviving fraction is recorded) and handed to the
density barrier: `known_barrier_one_well` (both conditions from the same well) and `known_barrier_two_wells`
(one condition per well).

Three counters run on the same fitted regimes:
    reference pipeline   find_equilibria(softness=1, n_starts=200), fsolve multi-start on the two-regime
                         mixture drift (columns n_eq_reference_pipeline, n_stable_reference_pipeline)
    reimplementation     scipy least_squares multi-start from draws of each regime's stationary law plus the
                         ridgeline, with independent de-duplication and finite-difference Jacobian
                         (n_eq_reimplementation_ls, n_stable_reimplementation_ls)
    line                 exact 1-D count: sign changes of the drift's component along the ridgeline between the two
                         regime centres (n_sign_changes_line, n_stable_line)
and two barrier readers: ridgeline_modes_2 (density_modes, density_barrier_nats; *_1d on the connecting
direction) and a Gaussian kernel density estimate of the projected data (kde_modes, kde_barrier_nats).
Segment structures: EEG, MEG, fMRI (with the haemodynamic response) and SLEEP (network_lib.MODALITIES).

Inputs   trained weights (NDIF_RNN_WEIGHTS), the escape-ladder table rnn_escape_simulation.csv in the network
         cache paths.cache("rnn") (NDIF_RNN_CACHE; written by escape_ladder.py), device NDIF_DEVICE.
Outputs  figure_reproduction/inputs/density_barrier_cells.csv (768 rows = 4 networks x 4 segment structures x
         16 repetitions x 3 arms; the cells of Fig. 2d,e, also read by the two-well/one-well consolidation),
         analyses/recurrent_networks/out/density_barrier_cells.json (rows, ground truth, true barriers) and
         density_barrier_cells.log.
Reported in  Fig. 2d (two-well cells) and Fig. 2e left (one-well cells; density barrier zero in every cell).
Runtime  about 13.5 minutes on one CUDA GPU (stored run: torch 2.9.0+cu130).

    python analyses/recurrent_networks/known_barrier_cells.py --seeds 20260922,20260935,20260948,20260961 --reps 16
"""
from __future__ import annotations

import argparse
import csv as _csv
import itertools
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.linalg import solve_continuous_lyapunov

HERE = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

import torch  # noqa: E402
import network_lib as NL  # noqa: E402
from analyses.common import paths  # noqa: E402

STAB = Path(paths.cache("rnn")) / "rnn_escape_simulation.csv"     # written by escape_ladder.py
SEED = 20260921
ALPHA = NL.ALPHA
SIGMA0 = NL.SIGMA0
ANCHORS = NL.ANCHORS
MODALITIES = NL.MODALITIES
BURN = 300
N_STARTS = 200
OVER = dict(EEG=8, MEG=4, fMRI=16, SLEEP=12)   # chains simulated per chain needed,
#                                              before the well-membership filter
# column order of density_barrier_cells.csv (alphabetical in the names under which the cells were computed)
COLUMNS = ["anchor_seed", "arm", "clean_frac_well_hi", "clean_frac_well_lo", "density_barrier_nats",
           "density_barrier_nats_1d", "density_modes", "density_modes_1d", "fp_separation", "fp_separation_in_sd",
           "gain", "kde_barrier_nats", "kde_modes", "modality", "n_eq_reimplementation_ls",
           "n_eq_reference_pipeline", "n_sign_changes_line", "n_stable_line", "n_stable_reimplementation_ls",
           "n_stable_reference_pipeline", "rep", "separation", "true_barrier_generative_nats", "true_barrier_nats",
           "true_fw_hi_to_saddle", "true_fw_lo_to_saddle", "true_n_basins"]


# ------------------------------------------------------------------ re-implemented counters
def _drift(regs, softness=1.0):
    return lambda z: NL.multilinear_drift(np.asarray(z, float), regs, softness)[0]


def count_reimplementation_ls(regs, n_starts=200, seed=0):
    """Re-implemented multi-start root find: least_squares (trust-region) from draws of each
    regime's stationary law plus the ridgeline, with independent de-duplication and stability test."""
    rng = np.random.default_rng(seed)
    f = _drift(regs)
    d = len(regs[0].mu)
    starts = [np.asarray(r.mu, float) for r in regs]
    a = np.linspace(0, 1, 11)
    starts += list(NL.ridgeline_points([regs[0].mu, regs[1].mu],
                                       [regs[0].Sigma, regs[1].Sigma],
                                       np.column_stack([1 - a, a])))
    for r in regs:
        starts += list(rng.multivariate_normal(r.mu, r.Sigma, size=n_starts // 2))
    eqs = []
    for s in starts:
        try:
            sol = least_squares(f, s, xtol=1e-14, ftol=1e-14, gtol=1e-14)
        except Exception:
            continue
        z = sol.x
        if np.linalg.norm(f(z)) > 1e-7:
            continue
        if not any(np.linalg.norm(z - w) < 1e-4 * (1 + np.linalg.norm(w)) for w in eqs):
            eqs.append(z)
    n_st = 0
    for z in eqs:
        J = np.empty((d, d))
        h = 1e-5 * (1 + np.abs(z))
        for k in range(d):
            ek = np.zeros(d); ek[k] = h[k]
            J[:, k] = (f(z + ek) - f(z - ek)) / (2 * h[k])
        if np.all(np.linalg.eigvals(J).real < 0):
            n_st += 1
    return len(eqs), n_st


def count_line(regs, n=2001):
    """Exact 1-D count along the ridgeline: the number of sign changes of the drift's
    component along the connecting curve.  A two-regime mixture field is linear inside each
    regime's domain of responsibility, so any additional zero must lie on this curve."""
    a = np.linspace(0, 1, n)
    X = NL.ridgeline_points([regs[0].mu, regs[1].mu], [regs[0].Sigma, regs[1].Sigma],
                            np.column_stack([1 - a, a]))
    f = _drift(regs)
    t = np.array([f(x) for x in X])
    tang = np.gradient(X, axis=0)
    tang /= np.maximum(np.linalg.norm(tang, axis=1, keepdims=True), 1e-15)
    g = np.einsum("ij,ij->i", t, tang)
    sgn = np.sign(g)
    ch = int(np.sum(sgn[:-1] * sgn[1:] < 0))
    down = int(np.sum((sgn[:-1] > 0) & (sgn[1:] < 0)))   # stable crossings along the curve
    return ch, down


def kde_modes(p, grid=1024):
    """Independent barrier reader: Gaussian KDE of the 1-D projection, Silverman bandwidth.
    Returns (n_modes, barrier in nats) with the barrier read off -log density."""
    p = np.asarray(p, float)
    p = p[np.isfinite(p)]
    if len(p) < 20:
        return 0, float("nan")
    s = np.std(p, ddof=1)
    if s <= 0:
        return 1, 0.0
    h = 0.9 * min(s, (np.percentile(p, 75) - np.percentile(p, 25)) / 1.349) * len(p) ** (-0.2)
    h = max(h, 1e-6 * s)
    x = np.linspace(p.min() - 3 * h, p.max() + 3 * h, grid)
    dens = np.exp(-0.5 * ((x[:, None] - p[None, :]) / h) ** 2).sum(1) / (len(p) * h
                                                                        * math.sqrt(2 * math.pi))
    lp = np.log(np.maximum(dens, 1e-300))
    is_max = np.zeros(grid, bool)
    is_max[1:-1] = (lp[1:-1] > lp[:-2]) & (lp[1:-1] > lp[2:])
    idx = np.where(is_max)[0]
    if len(idx) < 2:
        return int(max(len(idx), 1)), 0.0
    return int(len(idx)), float(min(lp[idx[0]], lp[idx[-1]]) - lp[idx[0]:idx[-1] + 1].min())


# ------------------------------------------------------------------ simulation helper
def _sim(cfgs, n_chains, n_keep, stride, seed, over=1.0):
    return NL.simulate(cfgs, n_chains, n_keep, stride, BURN, seed, sep=None, over=over)


def _mk_cfg(W, Win, b, u, sigma, x, S):
    return dict(W=W, b=b, u=u, sigma=float(sigma), Win_u=Win @ u,
                true=dict(x=x), _Scov=S)


def _calibrate(cell, e, lam_base, dt):
    rho_t = math.exp(-lam_base * dt)
    tau = NL.solve_tau(cell, e, rho_t)
    if tau is None:
        return None
    return max(1, int(round(tau / ALPHA)))


# ------------------------------------------------------------------ ARM 1
def arm_single(seed, reps, mods, log):
    """A network whose single-basin status is established without the counter."""
    geo = NL.base_geometry(seed, log)
    W, Win, b, x0, e = geo["W"], geo["Win"], geo["b"], geo["x0"], geo["e_load"]
    cell = NL.exact_cell(W, Win, b, np.zeros(4), x0, e, SIGMA0)
    # --- independent ground truth: exhaustive enumeration of the ACTUAL network
    net = NL._as_net(seed, W, Win, b)
    fps, _, _ = NL.fixed_points_hardened(net, np.zeros(4, np.float32), n_starts=600,
                                         seed=NL.ENUM_SEED)
    st = [np.asarray(f["point"], float) for f in fps if f["stable"]]
    comp = NL.completeness_check(net, np.zeros(4, np.float32),
                                 [s.astype(np.float32) for s in st], n_probe=500,
                                 seed=NL.ENUM_SEED)
    truth = dict(n_fixed=len(fps), n_stable=len(st),
                 completeness=float(comp["frac_accounted"]))
    log(f"  [{seed}] ARM1 ground truth (no counter used): n_stable={len(st)}, "
        f"completeness={comp['frac_accounted']:.3f}")
    unit_sd = float(np.sqrt(np.trace(cell["S"]) / len(x0)))
    rows = []
    for mname in mods:
        M = MODALITIES[mname]
        stride = _calibrate(cell, e, M["lam_base"], M["dt"])
        hrf = NL.spm_hrf(M["dt"]) if M["hrf"] else None
        n_lead = (len(hrf) - 1) if hrf is not None else 0
        cfg = [_mk_cfg(W, Win, b, np.zeros(4), SIGMA0, x0, cell["S"])]
        n_chains = M["n_seg"] * reps
        Wnd, simi = _sim(cfg, n_chains, M["seg_len"] + n_lead, stride, SEED + 31 * len(mods))
        if hrf is not None:
            Wc = np.empty((1, Wnd.shape[1], M["seg_len"], Wnd.shape[3]))
            for k in range(M["seg_len"]):
                Wc[:, :, k, :] = np.tensordot(Wnd[:, :, k:k + len(hrf), :], hrf[::-1],
                                              axes=([2], [0]))
            Wnd = Wc
        X = Wnd[0]                                        # (n_chains, seg_len, 64)
        for rep in range(reps):
            sl = slice(rep * M["n_seg"], (rep + 1) * M["n_seg"])
            seg = X[sl]
            h = M["n_seg"] // 2
            rng = np.random.default_rng(SEED + rep)
            Zl, _ = NL.state_space([seg[:h], seg[h:]], M["d"], 0.0, unit_sd, rng)
            regs = [NL.fit_linear_regime([s for s in Zl[i]], f"c{i}", dt=M["dt"])
                    for i in (0, 1)]
            dmu = regs[0].mu - regs[1].mu
            Sm = 0.5 * (regs[0].Sigma + regs[1].Sigma)
            sep = float(np.sqrt(max(dmu @ np.linalg.solve(Sm, dmu), 0.0)))
            eqs = NL.find_equilibria(regs, softness=1.0, n_starts=N_STARTS, seed=SEED + rep)
            n_pub, n_pub_st = len(eqs), int(sum(q["stable"] for q in eqs))
            n_ls, n_ls_st = count_reimplementation_ls(regs, n_starts=N_STARTS, seed=SEED + rep)
            n_ch, n_down = count_line(regs)
            r2 = NL.ridgeline_modes_2(regs[0].mu, regs[0].Sigma, regs[1].mu, regs[1].Sigma,
                                      w1=0.5)
            v = dmu / max(np.linalg.norm(dmu), 1e-12)
            proj = np.concatenate([(Zl[i].reshape(-1, M["d"]) @ v) for i in (0, 1)])
            km, kb = kde_modes(proj)
            rows.append(dict(arm="single_basin", anchor_seed=int(seed), modality=mname,
                             rep=int(rep), separation=sep,
                             n_eq_reference_pipeline=n_pub, n_stable_reference_pipeline=n_pub_st,
                             n_eq_reimplementation_ls=n_ls, n_stable_reimplementation_ls=n_ls_st,
                             n_sign_changes_line=n_ch, n_stable_line=n_down,
                             density_modes=int(r2.n_modes),
                             density_barrier_nats=float(r2.barrier_nats),
                             kde_modes=km, kde_barrier_nats=kb,
                             true_n_basins=len(st), true_barrier_nats=0.0))
        log(f"  [{seed}] ARM1 {mname}: {reps} cells, "
            f"reference-pipeline count>=2 in "
            f"{np.mean([r['n_stable_reference_pipeline'] >= 2 for r in rows if r['modality']==mname]):.3f}"
            f", reimplementation {np.mean([r['n_stable_reimplementation_ls'] >= 2 for r in rows if r['modality']==mname]):.3f}"
            f", median sep "
            f"{np.median([r['separation'] for r in rows if r['modality']==mname]):.3f}")
    return truth, rows


# ------------------------------------------------------------------ ARM 2
def arm_barrier(seed, reps, mods, log, gain=NL.BASE_GAIN):
    """The bistable network at its operating gain: a known non-zero barrier."""
    m, W, Win, b = NL.anchor_params(seed, gain)
    fps, _, _ = NL.fixed_points_hardened(m, np.zeros(4, np.float32), n_starts=300,
                                         seed=NL.ENUM_SEED)
    st = [np.asarray(f["point"], float) for f in fps if f["stable"]]
    if len(st) != 2:
        log(f"  [{seed}] ARM2 skipped: {len(st)} stable fixed points at gain {gain}")
        return []
    ax = st[1] - st[0]; ax /= np.linalg.norm(ax)
    xb = st[int(np.argmax([float(s @ ax) for s in st]))]
    xo = st[int(np.argmin([float(s @ ax) for s in st]))]
    J = NL.jac_c(xb, W, Win, b, np.zeros(4))
    D = SIGMA0 ** 2 / (2.0 * ALPHA)
    S = solve_continuous_lyapunov(J, -2.0 * D * np.eye(len(xb)))
    S = 0.5 * (S + S.T)
    ev, V = np.linalg.eig(J)
    cellb = dict(x=xb, J=J, ev=ev, V=V, S=S, D=D,
                 lam_slow=float(-np.max(ev.real)),
                 sd_load=float(np.sqrt(max(ax @ S @ ax, 1e-18))),
                 spread_rms=float(np.sqrt(np.trace(S) / len(xb))))
    unit_sd = float(np.sqrt(np.trace(S) / len(xb)))
    Jo = NL.jac_c(xo, W, Win, b, np.zeros(4))
    So = solve_continuous_lyapunov(Jo, -2.0 * D * np.eye(len(xo)))
    So = 0.5 * (So + So.T)
    rows = []
    for mname in mods:
        M = MODALITIES[mname]
        stride = _calibrate(cellb, ax, M["lam_base"], M["dt"])
        if stride is None:
            continue
        hrf = NL.spm_hrf(M["dt"]) if M["hrf"] else None
        n_lead = (len(hrf) - 1) if hrf is not None else 0
        cfgs = [_mk_cfg(W, Win, b, np.zeros(4), SIGMA0, xb, S),
                _mk_cfg(W, Win, b, np.zeros(4), SIGMA0, xo, So)]
        n_need = M["n_seg"] * reps
        # WELL MEMBERSHIP IS ENFORCED, NOT ASSUMED.  The network's two wells are
        # 7.1 apart while 4 x the RMS radius is 10.1, so the inherited excursion detector
        # cannot see a well change at all: a chain that has crossed to the other well is
        # inside its own 4-sigma ball.  Chains are therefore over-sampled and filtered on the
        # side of the wells-axis midpoint, on the RAW windows before any convolution, and the
        # surviving fraction is reported -- it is the escape contamination at that segment
        # length, which is a result in its own right.
        mid = 0.5 * (xb + xo)
        side = np.array([np.sign((xb - mid) @ ax), np.sign((xo - mid) @ ax)])
        ov = OVER[mname]
        Wnd, simi = _sim(cfgs, n_need * ov, M["seg_len"] + n_lead, stride, SEED + 77)
        clean_frac = []
        keep = []
        for k in (0, 1):
            pr = np.einsum("cij,j->ci", Wnd[k] - mid, ax)
            ok = np.all(np.sign(pr) == side[k], axis=1)
            clean_frac.append(float(ok.mean()))
            idx = np.flatnonzero(ok)[:n_need]
            if len(idx) < n_need:                      # not enough clean chains: pad and flag
                idx = np.concatenate([idx, np.flatnonzero(~ok)[:n_need - len(idx)]])
            keep.append(Wnd[k][idx])
        Wnd = np.stack(keep)
        log(f"  [{seed}] ARM2 {mname}: clean-chain fraction "
            f"{clean_frac[0]:.3f}/{clean_frac[1]:.3f} of {n_need*ov} at segment length "
            f"{M['seg_len']} x stride {stride}")
        if hrf is not None:
            Wc = np.empty((2, Wnd.shape[1], M["seg_len"], Wnd.shape[3]))
            for k in range(M["seg_len"]):
                Wc[:, :, k, :] = np.tensordot(Wnd[:, :, k:k + len(hrf), :], hrf[::-1],
                                              axes=([2], [0]))
        for rep in range(reps):
            sl = slice(rep * M["n_seg"], (rep + 1) * M["n_seg"])
            h = M["n_seg"] // 2
            for sub, groups in (("one_well", [Wnd[0][sl][:h], Wnd[0][sl][h:]]),
                                ("two_wells", [Wnd[0][sl][:h], Wnd[1][sl][:h]])):
                rng = np.random.default_rng(SEED + rep)
                Zl, _ = NL.state_space(groups, M["d"], 0.0, unit_sd, rng)
                regs = [NL.fit_linear_regime([s for s in Zl[i]], f"c{i}", dt=M["dt"])
                        for i in (0, 1)]
                dmu = regs[0].mu - regs[1].mu
                Sm = 0.5 * (regs[0].Sigma + regs[1].Sigma)
                sep = float(np.sqrt(max(dmu @ np.linalg.solve(Sm, dmu), 0.0)))
                r_full = NL.ridgeline_modes_2(regs[0].mu, regs[0].Sigma,
                                              regs[1].mu, regs[1].Sigma, w1=0.5)
                v = dmu / max(np.linalg.norm(dmu), 1e-12)
                r_1d = NL.ridgeline_modes_2(np.array([regs[0].mu @ v]),
                                            np.array([[float(v @ regs[0].Sigma @ v)]]),
                                            np.array([regs[1].mu @ v]),
                                            np.array([[float(v @ regs[1].Sigma @ v)]]),
                                            w1=0.5)
                proj = np.concatenate([(Zl[i].reshape(-1, M["d"]) @ v) for i in (0, 1)])
                km, kb = kde_modes(proj)
                eqs = NL.find_equilibria(regs, softness=1.0, n_starts=N_STARTS,
                                         seed=SEED + rep)
                rows.append(dict(arm=f"known_barrier_{sub}", anchor_seed=int(seed),
                                 modality=mname, rep=int(rep), gain=float(gain),
                                 separation=sep,
                                 n_eq_reference_pipeline=len(eqs),
                                 n_stable_reference_pipeline=int(sum(q["stable"] for q in eqs)),
                                 density_modes=int(r_full.n_modes),
                                 density_barrier_nats=float(r_full.barrier_nats),
                                 density_modes_1d=int(r_1d.n_modes),
                                 density_barrier_nats_1d=float(r_1d.barrier_nats),
                                 kde_modes=km, kde_barrier_nats=kb,
                                 clean_frac_well_hi=clean_frac[0],
                                 clean_frac_well_lo=clean_frac[1],
                                 fp_separation=float(np.linalg.norm(xb - xo)),
                                 fp_separation_in_sd=float(np.linalg.norm(xb - xo)
                                                           / cellb["sd_load"])))
        log(f"  [{seed}] ARM2 {mname}: one_well barrier median "
            f"{np.median([r['density_barrier_nats'] for r in rows if r['modality']==mname and r['arm'].endswith('one_well')]):.4f} nats"
            f" | two_wells "
            f"{np.median([r['density_barrier_nats'] for r in rows if r['modality']==mname and r['arm'].endswith('two_wells')]):.4f}")
    return rows


def main():
    ap = argparse.ArgumentParser(description="counters and density barriers on known single-basin and "
                                                 "known-barrier cells")
    ap.add_argument("--reps", type=int, default=16)
    ap.add_argument("--seeds", default="20260922,20260935,20260948,20260961")
    ap.add_argument("--mods", default="EEG,MEG,fMRI,SLEEP")
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    mods = [m for m in a.mods.split(",") if m.strip()]
    lines = []

    def log(msg):
        s = time.strftime("[%H:%M:%S] ") + str(msg)
        print(s, flush=True)
        lines.append(s)

    log(f"known_barrier_cells.py  python {platform.python_version()} torch {torch.__version__} "
        f"device {NL.DEV}  reps {a.reps}")
    # the true barriers, from the escape ladder
    true_bar = {}
    if STAB.exists():
        for r in _csv.DictReader(STAB.open(encoding="utf-8")):
            try:
                if abs(float(r["gain"]) - NL.BASE_GAIN) < 1e-9 and r["gb1_pass"] in ("True", "1"):
                    true_bar[int(r["anchor_seed"])] = dict(
                        fw_lo_to_saddle=float(r["fw_lo_to_saddle"]),
                        fw_hi_to_saddle=float(r["fw_hi_to_saddle"]),
                        barrier_generative_nats=float(r["barrier_generative_nats"]))
            except Exception:
                continue
    log(f"  true barriers for {len(true_bar)} networks at gain {NL.BASE_GAIN}")
    all_rows, truths = [], {}
    for s in seeds:
        t0 = time.time()
        tr, r1 = arm_single(s, a.reps, mods, log)
        truths[str(s)] = tr
        all_rows += r1
        r2 = arm_barrier(s, a.reps, mods, log)
        for r in r2:
            r.update({f"true_{k}": v for k, v in true_bar.get(s, {}).items()})
        all_rows += r2
        log(f"  [{s}] done in {time.time()-t0:.0f}s")
    out = lambda name: paths.out("analyses", "recurrent_networks", "out", name)  # noqa: E731
    with open(out("density_barrier_cells.json"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(NL._clean(
            dict(script="known_barrier_cells.py", seeds=seeds, mods=mods, reps=a.reps,
                 ground_truth=truths, true_barriers=true_bar, rows=all_rows))))
    present = set().union(*[set(r) for r in all_rows])
    keys = [k for k in COLUMNS if k in present] + sorted(present - set(COLUMNS))
    with open(paths.out("figure_reproduction", "inputs", "density_barrier_cells.csv"), "w", newline="",
              encoding="utf-8") as fh:
        w = _csv.DictWriter(fh, fieldnames=keys, lineterminator="\n")
        w.writeheader()
        for r in all_rows:
            w.writerow(r)
    with open(out("density_barrier_cells.log"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines))
    log(f"  wrote {len(all_rows)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
