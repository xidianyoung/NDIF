"""Escape ladder: independent dynamical measures of stability across a ladder of recurrent gains in the trained
flip-flop networks, and the Arrhenius test of two candidate barriers.

THE SYSTEM
    r_{t+1} = (1-alpha) r_t + alpha tanh(W_g r_t + W_in u_t + b) + sigma xi_t
    64 tanh units, sigma = 0.08 per step, alpha = 0.05, u = 0 (maintenance).  Networks are trained by
    network_lib.train_original_recipe(network_lib.flipflop_batch, tag, 1200, seed) over the candidate seeds
    SEED + 7 + 13k and kept only when the enumerated structure at u = 0 is exactly two stable fixed points with
    completeness >= 0.98 (8 of the 9 candidates; the trained weights are in weights/).  Gains (W_g = g W):
    0.76, 0.84, 0.86 (the operating point of the other network analyses), 0.90, 1.00.

THE TIME AND ENERGY CONVENTION
    The update is Euler-Maruyama for dr = f(r) dt + sqrt(2 D) dW, f(r) = -r + tanh(W_g r + W_in u + b), with
    dt = alpha, so D = sigma^2 / (2 alpha) = 0.064.  For p_ss ~ exp(-U/D) a dip of B nats in log p_ss is dU/D, and
    the Freidlin-Wentzell geometric action with a = 2D is likewise dU/D for a gradient field: both are the
    exponent of Kramers' law, so an Arrhenius regression of log MFPT on either has a predicted slope of 1 for the
    true dynamical barrier.  The continuous Jacobian is J_c = -I + diag(1 - tanh^2(W_g x + W_in u + b)) W_g.

THE FOUR MEASURES, none of which touches a density fit
    (1) fixed points of the noise-free system by root finding and the continuous-time Jacobian spectrum at every
        fixed point, with a completeness statement;
    (2) escape rate = 1 / mean first-passage time (128 chains per well, 80,000-sample horizon, exponential MLE
        under censoring).  Primary transition rule: crossing the hyperplane through the saddle whose normal is the
        LEFT eigenvector of J_c for the unstable eigenvalue (the tangent plane of the saddle's stable manifold);
        five alternative rules (wells-axis hyperplane through the saddle, three well-core fractions, the
        wells-axis midpoint) are reported beside it;
    (3) controlled perturbation recovery: the critical displacement that falls into the other well, and the
        noise-free time to recover to 1/e from a fixed displacement and from half the critical displacement;
    (4) the Freidlin-Wentzell quasipotential between the wells and to the saddle by geometric minimum action, both
        directions (full-path Adam on the autograd gradient of the geometric action with arclength
        reparameterisation, gmam_batch; Heymann & Vanden-Eijnden 2008, Phys. Rev. Lett. 100:140601).
    The comparison quantity is the fitted-density barrier of a long run labelled at the wells-axis midpoint
    (network_lib.latent_landscape).  The finite-difference minimiser network_lib.gmam, whose descent can only
    translate the interior of the path rigidly, is run beside the full-path minimiser on the first network and in
    the self-test for comparison.

THE DECIDING TEST
    The Arrhenius slope: log(measured MFPT) regressed on each candidate barrier across gains, per network, judged
    against 1 (calibrated end-to-end on a 1-D double-well ladder of known barrier, self-test block), never
    against 0.  The unit of analysis is the network; every group interval resamples networks.  Monotone ordering
    of the gains is judged against a within-network permutation of the gain labels (10,000 draws).

Inputs   trained weights weights/network_<seed>.pt (+ network_<seed>.json, qualification_<seed>.json;
         NDIF_RNN_WEIGHTS), device NDIF_DEVICE (default cuda if available).
Outputs  in the network cache paths.cache("rnn") (NDIF_RNN_CACHE, default analyses/_work/rnn):
         cells/cell_<seed>_<gain x 100>.json (one per network and gain), selftest.json,
         rnn_escape_simulation.json (everything: cells, group statistics, self-tests), rnn_escape_simulation.csv
         (one row per network and gain; read by known_barrier_cells.py), escape_ladder.log.  The stored output is
         the Source Data file analysis_inputs/stored_results/rnn_escape_simulation.json.
Reported in  Fig. 2e right and the escape simulation of the Methods, through arrhenius_summary.py (Arrhenius
         slopes 0.832 [0.796, 0.880] on the minimum action and 0.170 [0.104, 0.234] on the density barrier, five
         of the seven networks with complete ladders).
Runtime  40 cells of 28-131 s on one CUDA GPU (about 16 min wall with concurrent workers, about 1 h serial);
         the aggregation with --resume takes seconds.  The stored run used torch 2.9.0+cu130 on CUDA.

    python analyses/recurrent_networks/escape_ladder.py --qualify-only
    python analyses/recurrent_networks/escape_ladder.py --indices 0,4      # workers in parallel, one per
    python analyses/recurrent_networks/escape_ladder.py --indices 1,5      #   index set 0,4 / 1,5 / 2,6 / 3,7
    ...
    python analyses/recurrent_networks/escape_ladder.py --resume           # aggregation of the cached cells
    python analyses/recurrent_networks/escape_ladder.py                    # serial run without workers
    python analyses/recurrent_networks/escape_ladder.py --selftest         # calibration block only
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

import scipy  # noqa: E402
import sklearn  # noqa: E402  (version record)
import torch  # noqa: E402
from scipy import stats as sps  # noqa: E402

import network_lib as NL  # noqa: E402
from analyses.common import paths  # noqa: E402

OUTDIR = Path(paths.cache("rnn"))         # cells, self-test, ladder outputs
CACHE = NL.WEIGHTS                        # trained weights and their qualification records

# ------------------------------------------------------------------ constants
SEED = NL.SEED                        # 20260915
ENUM_SEED = SEED + 11                 # fixed-point enumeration seed
SIM_SEED = SEED + 20200               # stochastic seed of the escape and long runs
ALPHA = NL.ALPHA                      # 0.05
SIGMA = 0.08                          # operating noise; HELD FIXED
D_DIFF = SIGMA ** 2 / (2.0 * ALPHA)   # 0.064 state units^2 per unit time

GAINS = (0.76, 0.84, 0.86, 0.90, 1.00)
GAIN_LABEL = {0.76: "B0", 0.84: "B1", 0.86: "ANCHOR", 0.90: "B2", 1.00: "B3"}
REF_GAIN = 1.00                       # axis-orientation reference within a seed

N_ANCHORS_TARGET = 8
N_CANDIDATES_MAX = 24     # candidate seeds examined at most; the qualification rate over
                          # the candidate seeds is measured, not assumed

# --- escape measurement (pre-declared)
N_CHAINS = 128                        # per well per cell (both wells run in one batch)
MAX_STEPS = 80_000                    # first-passage horizon, samples
CHECK_EVERY = 500
NOISE_BLOCK = 250                     # pre-generated noise block, steps
CORE_FRACS = (0.50, 0.75, 0.90)
PRIMARY_RULE = "separatrix_saddle_lefteig"

# --- enumeration: 300 random starts, plus starts along the segment joining the two
#     wells so the saddle cannot be missed.
N_STARTS = 300
N_LINE_STARTS = 100

# --- long equilibrium run (for the DENSITY barrier, the comparison quantity)
N_LONG_CHAINS = 96
T_LONG_CHAIN = 6000
LONG_BURN = 500

# --- perturbation recovery
N_SCAN = 200                          # displacement grid per bracketing pass
T_RELAX = 2000                        # samples of noise-free flow per probe
D_FIXED = 1.0                         # the fixed displacement, state units
T_HETERO = 4000                       # samples of unstable-manifold integration

# --- gMAM
GM_PTS = 101
GM_ITERS = 1200
GM_LR = 0.02
GM_REPARAM_EVERY = 100
AL_GMAM_ITERS = 200                   # finite-difference gmam, first network only

N_PERM = 10_000
N_BOOT = 10_000

DEV = NL.DEV


def _log(msg, sink):
    print(msg, flush=True)
    sink.append(str(msg))


def _clean(o):
    """NaN/inf -> None, numpy scalars -> python, so the JSON is strict."""
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (np.integer, int)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        v = float(o)
        return v if math.isfinite(v) else None
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    return o


# ==========================================================================
# 1.  drift, geometric action, and a CORRECT minimum-action path
# ==========================================================================
def drift_np(X, W, bvec):
    """f(r) = -r + tanh(W r + b) at u = 0.  Rows of X are states."""
    X = np.atleast_2d(np.asarray(X, float))
    return -X + np.tanh(X @ W.T + bvec)


def _action_t(P, W, bvec, D):
    """FW geometric action of a batch of paths P (n_path, n_pts, d), a = 2D I.

    Identical to lib.attractor_lib.geometric_action for isotropic D:
        S = sum_i ( |dP_i|_a |b_i|_a - <dP_i, b_i>_a ),   <u,v>_a = u.v/(2D)
          = (1/(2D)) sum_i ( |dP_i| |b_i| - dP_i . b_i )
    """
    dP = P[:, 1:] - P[:, :-1]
    mid = 0.5 * (P[:, 1:] + P[:, :-1])
    b = -mid + torch.tanh(mid @ W.T + bvec)
    nd = torch.linalg.norm(dP, dim=2)
    nb = torch.linalg.norm(b, dim=2)
    return ((nd * nb - (dP * b).sum(dim=2)).sum(dim=1)) / (2.0 * D)


def _resample_arclength(E, n_pts):
    """Resample one polyline (m, d) to n_pts nodes, uniform in arclength."""
    dt, dev = E.dtype, E.device
    seg = torch.linalg.norm(E[1:] - E[:-1], dim=1)
    s = torch.cat([torch.zeros(1, dtype=dt, device=dev), torch.cumsum(seg, 0)])
    if float(s[-1]) <= 0:
        return E[:1].repeat(n_pts, 1)
    s = s / s[-1]
    tt = torch.linspace(0, 1, n_pts, dtype=dt, device=dev)
    idx = torch.searchsorted(s.contiguous(), tt.contiguous(), right=True).clamp(1, len(s) - 1)
    w = ((tt - s[idx - 1]) / (s[idx] - s[idx - 1]).clamp_min(1e-30)).unsqueeze(1)
    return E[idx - 1] * (1 - w) + E[idx] * w


def _reparam_batch(P):
    """Redistribute every path's nodes uniformly in arclength (the gMAM step)."""
    out = P.clone()
    for i in range(P.shape[0]):
        out[i] = _resample_arclength(P[i], P.shape[1])
    return out


def gmam_batch(pairs, W_t, b_t, D, inits_extra=None, n_pts=GM_PTS,
               n_iter=GM_ITERS, lr=GM_LR, reparam_every=GM_REPARAM_EVERY):
    """Minimum FW geometric action for several (x0, x1) pairs at once.

    For every pair, several initial paths are optimised simultaneously -- they are
    independent rows of one batch and the objective is separable -- and the lowest
    action is kept.  Interior nodes are free; endpoints are clamped.  This is the
    full-path minimisation lib/attractor_lib.py::gmam fails to perform.

    pairs        list of (x0, x1) arrays
    inits_extra  optional list, one entry per pair, each a list of (m, d) polylines
                 to use as additional initialisations (resampled to n_pts).
    Returns (actions, paths), one entry per pair.
    """
    dt, dev = W_t.dtype, W_t.device
    rows, owner = [], []
    for j, (x0, x1) in enumerate(pairs):
        t0 = torch.as_tensor(np.asarray(x0, float), dtype=dt, device=dev)
        t1 = torch.as_tensor(np.asarray(x1, float), dtype=dt, device=dev)
        tt = torch.linspace(0, 1, n_pts, dtype=dt, device=dev).unsqueeze(1)
        cand = [t0 * (1 - tt) + t1 * tt]                       # straight line
        for extra in (inits_extra[j] if inits_extra else []):
            E = torch.as_tensor(np.ascontiguousarray(extra, dtype=float),
                                dtype=dt, device=dev)
            if E.shape[0] < 2:
                continue
            R = _resample_arclength(E, n_pts).clone()
            R[0], R[-1] = t0, t1
            cand.append(R)
        for c in cand:
            rows.append(c)
            owner.append(j)
    P0 = torch.stack(rows, 0)
    owner = np.asarray(owner)
    e0, e1 = P0[:, 0].clone(), P0[:, -1].clone()

    inner = P0[:, 1:-1].clone().requires_grad_(True)
    opt = torch.optim.Adam([inner], lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_iter, eta_min=lr / 50)
    with torch.no_grad():
        best = _action_t(P0, W_t, b_t, D).clone()
    bestP = P0.clone()
    for it in range(n_iter):
        P = torch.cat([e0.unsqueeze(1), inner, e1.unsqueeze(1)], dim=1)
        S = _action_t(P, W_t, b_t, D)
        opt.zero_grad(set_to_none=True)
        S.sum().backward()
        opt.step()
        sch.step()
        if (it + 1) % reparam_every == 0 or it == n_iter - 1:
            with torch.no_grad():
                P = torch.cat([e0.unsqueeze(1), inner, e1.unsqueeze(1)], dim=1)
                P = _reparam_batch(P)
                P[:, 0], P[:, -1] = e0, e1
                Snew = _action_t(P, W_t, b_t, D)
                imp = Snew < best
                best = torch.where(imp, Snew, best)
                bestP[imp] = P[imp]
                inner.data.copy_(P[:, 1:-1])
    best = best.detach().cpu().numpy()
    bestP = bestP.detach().cpu().numpy()
    out_S, out_P = [], []
    for j in range(len(pairs)):
        sel = np.flatnonzero(owner == j)
        k = sel[int(np.argmin(best[sel]))]
        out_S.append(float(best[k]))
        out_P.append(bestP[k])
    return out_S, out_P


# ==========================================================================
# 2.  SELF-TESTS -- the estimator calibration block
# ==========================================================================
def _minimise_2d(act_fn, xA, xB, n_pts=201, n_iter=4000, lr=0.01):
    """The same minimiser as gmam_batch, for the 2-D self-test."""
    e0 = torch.as_tensor(xA, dtype=torch.float64)
    e1 = torch.as_tensor(xB, dtype=torch.float64)
    t = torch.linspace(0, 1, n_pts, dtype=torch.float64).unsqueeze(1)
    P0 = (e0 * (1 - t) + e1 * t).unsqueeze(0)
    inner = P0[:, 1:-1].clone().requires_grad_(True)
    opt = torch.optim.Adam([inner], lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_iter, eta_min=lr / 50)
    best = float(act_fn(P0)[0])
    for it in range(n_iter):
        P = torch.cat([e0.view(1, 1, -1), inner, e1.view(1, 1, -1)], dim=1)
        S = act_fn(P)
        opt.zero_grad(set_to_none=True)
        S.sum().backward()
        opt.step()
        sch.step()
        if (it + 1) % 100 == 0:
            with torch.no_grad():
                P = torch.cat([e0.view(1, 1, -1), inner, e1.view(1, 1, -1)], dim=1)
                P = _reparam_batch(P)
                P[:, 0], P[:, -1] = e0, e1
                best = min(best, float(act_fn(P)[0]))
                inner.data.copy_(P[:, 1:-1])
    return best


def _selftest_gmam(sink):
    """Three calibrations: two exactly-known actions and an end-to-end Arrhenius."""
    res = {}
    Dsc = 0.05

    # ---- (a) 1-D double well: U = (x^2-1)^2/4, D = 0.05 -> exact dU/D = 5.0
    def act_1d(P):
        dP = P[:, 1:] - P[:, :-1]
        mid = 0.5 * (P[:, 1:] + P[:, :-1])
        b = mid - mid ** 3
        nd = torch.linalg.norm(dP, dim=2)
        nb = torch.linalg.norm(b, dim=2)
        return ((nd * nb - (dP * b).sum(2)).sum(1)) / (2.0 * Dsc)

    t = np.linspace(0, 1, 201)[:, None]
    S_line = float(act_1d(torch.as_tensor((-1.0 + t * 2.0)[None], dtype=torch.float64))[0])

    def drift_1d(X):
        X = np.atleast_2d(np.asarray(X, float))
        return X - X ** 3

    S_al, _ = NL.gmam([-1.0], [1.0], drift_1d, D=np.eye(1) * Dsc, n_pts=81, n_iter=300)
    S_ok = _minimise_2d(act_1d, np.array([-1.0]), np.array([1.0]))
    res["double_well_1d"] = dict(
        exact=5.0, straight_line=S_line, attractor_lib_gmam=float(S_al),
        correct_minimiser=float(S_ok))
    # the straight line IS the minimiser in 1-D, so this test cannot discriminate the two minimisers

    # ---- (b) curved valley, 2-D gradient system whose MAP is a parabola
    #      U(x,y) = (x^2-1)^2/4 + K (y - (x^2-1))^2 / 2 ; saddle (0,-1); dU = 1/4
    K = 6.0

    def drift_curv_np(X):
        X = np.atleast_2d(np.asarray(X, float))
        x, y = X[:, 0], X[:, 1]
        return np.column_stack([-((x ** 3 - x) + K * (y - (x ** 2 - 1)) * (-2 * x)),
                                -(K * (y - (x ** 2 - 1)))])

    def act_curv(P):
        dP = P[:, 1:] - P[:, :-1]
        mid = 0.5 * (P[:, 1:] + P[:, :-1])
        x, y = mid[..., 0], mid[..., 1]
        b = torch.stack([-((x ** 3 - x) + K * (y - (x ** 2 - 1)) * (-2 * x)),
                         -(K * (y - (x ** 2 - 1)))], dim=-1)
        nd = torch.linalg.norm(dP, dim=2)
        nb = torch.linalg.norm(b, dim=2)
        return ((nd * nb - (dP * b).sum(2)).sum(1)) / (2.0 * Dsc)

    xA, xB = np.array([-1.0, 0.0]), np.array([1.0, 0.0])
    t2 = np.linspace(0, 1, 201)[:, None]
    S_line2 = float(act_curv(torch.as_tensor((xA + t2 * (xB - xA))[None],
                                             dtype=torch.float64))[0])
    S_al2, P_al2 = NL.gmam(xA, xB, drift_curv_np, D=np.eye(2) * Dsc,
                           n_pts=81, n_iter=600)
    S_ok2 = _minimise_2d(act_curv, xA, xB)
    res["curved_valley_2d"] = dict(
        exact=5.0, straight_line=S_line2, attractor_lib_gmam=float(S_al2),
        attractor_lib_path_max_abs_y=float(np.abs(P_al2[:, 1]).max()),
        true_path_max_abs_y=1.0, correct_minimiser=float(S_ok2),
        attractor_lib_descent_fraction=float((S_line2 - S_al2) / (S_line2 - 5.0)),
        correct_minimiser_rel_error=float(abs(S_ok2 - 5.0) / 5.0),
        attractor_lib_rel_error=float(abs(S_al2 - 5.0) / 5.0))
    # the finite-difference 'gradient' of gmam assigns one scalar per coordinate to every interior
    # node, so its path can only translate rigidly and does not reach the curved valley floor

    # ---- (c) end-to-end Arrhenius calibration with a KNOWN barrier, two ladders.
    #
    #  PRIMARY  vary D at fixed U(x) = (x^2-1)^2/4.  The barrier dU/D moves and the
    #           Kramers prefactor pi/sqrt(U''(min)|U''(0)|) = pi/sqrt(2) does NOT,
    #           so the Arrhenius slope must be exactly 1.  This is what calibrates
    #           the deciding test.
    #  SECONDARY vary c in U_c = c (x^2-1)^2/4.  The barrier AND both curvatures
    #           move, so the prefactor moves with the barrier and the slope is
    #           analytically 1 - 1/(5c), not 1.  This rung exists because the RNN
    #           gain ALSO moves the saddle curvature (lambda_u rises with gain), so
    #           a measured RNN slope slightly below 1 is expected on these grounds
    #           alone and must not be read as a failure of the barrier.
    def _ladder_1d(rows, rng):
        out = []
        dtstep, nch = 0.01, 2048
        for c, Dl in rows:
            Bk = (c * 0.25) / Dl
            x = -np.ones(nch)
            fp = np.zeros(nch)
            s = math.sqrt(2 * Dl * dtstep)
            step = 0
            for step in range(1, 200_001):
                x = x + (c * (x - x ** 3)) * dtstep + s * rng.standard_normal(nch)
                hit = (fp == 0) & (x >= 0.0)
                if hit.any():
                    fp[hit] = step * dtstep
                if step % 500 == 0 and np.mean(fp > 0) > 0.98:
                    break
            n_esc = int(np.sum(fp > 0))
            t_end = step * dtstep
            risk = float(np.sum(np.where(fp > 0, fp, t_end)))
            # analytic Kramers prefactor for first passage to the barrier top
            pref = math.pi / math.sqrt(2 * c * c)
            out.append(dict(c=float(c), D=float(Dl), barrier_exact=float(Bk),
                            mfpt=risk / max(n_esc, 1), n_escaped=n_esc, n_chains=nch,
                            censored_frac=float(1 - n_esc / nch),
                            kramers_prefactor=float(pref)))
            _log(f"    1-D ladder c={c:.2f} D={Dl:.4f}  barrier {Bk:.3f}  MFPT "
                 f"{out[-1]['mfpt']:.2f}  censored {out[-1]['censored_frac']:.3f}",
                 sink)
        return out

    def _fit(ladder, predicted):
        bb = np.array([r["barrier_exact"] for r in ladder])
        ll = np.log([r["mfpt"] for r in ladder])
        lr_ = sps.linregress(bb, ll)
        return dict(ladder=ladder, slope=float(lr_.slope), slope_se=float(lr_.stderr),
                    ci_lo=float(lr_.slope - 1.96 * lr_.stderr),
                    ci_hi=float(lr_.slope + 1.96 * lr_.stderr),
                    r2=float(lr_.rvalue ** 2), predicted_slope=float(predicted))

    rng = np.random.default_rng(SEED)
    lad_D = _ladder_1d([(1.0, 0.25 / B) for B in (2.0, 2.5, 3.0, 3.5, 4.0)], rng)
    # PRIMARY calibration: U fixed, D varied, so the Kramers prefactor is constant and the slope
    # must be exactly 1.  This is the reference the network slopes are judged against -- never zero.
    res["arrhenius_calibration_1d"] = _fit(lad_D, 1.0)
    lad_c = _ladder_1d([(B / 5.0, 0.05) for B in (2.0, 2.5, 3.0, 3.5, 4.0)], rng)
    pred_c = float(sps.linregress(
        np.array([r["barrier_exact"] for r in lad_c]),
        np.log([r["kramers_prefactor"] for r in lad_c])
        + np.array([r["barrier_exact"] for r in lad_c])).slope)
    # SECONDARY calibration: c scales the whole potential, so the barrier and both curvatures move
    # together and the analytic slope is 1 - 1/(5c), not 1.  Reported because the network gain likewise
    # moves the saddle curvature, so a network slope a little below 1 is expected from the prefactor alone.
    res["arrhenius_calibration_1d_moving_prefactor"] = _fit(lad_c, pred_c)
    return res


# ==========================================================================
# 3.  anchors and fixed-point enumeration
# ==========================================================================
def get_anchor(seed, sink):
    """Train (or load) one flip-flop network with network_lib.train_original_recipe."""
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"network_{seed}.pt"
    if f.exists():
        net = NL.FineRNN(n_in=4, seed=int(seed)).to(DEV)
        net.load_state_dict(torch.load(f, map_location=DEV))
        meta = json.loads((CACHE / f"network_{seed}.json").read_text(encoding="utf-8"))
        return net, float(meta["train_score"])
    t0 = time.time()
    net, score = NL.train_original_recipe(NL.flipflop_batch, f"flipflop-{seed}",
                                          1200, int(seed))
    torch.save(net.state_dict(), f)
    (CACHE / f"network_{seed}.json").write_text(
        json.dumps(dict(seed=int(seed), train_score=float(score), steps=1200,
                        recipe="network_lib.train_original_recipe(network_lib.flipflop_batch, "
                               "tag, 1200, seed)")), encoding="utf-8")
    _log(f"    trained seed {seed} in {time.time()-t0:.0f}s, score {score:+.3f}", sink)
    return net, float(score)


def jacobian_c(x, W, bvec):
    """Continuous-time Jacobian J_c = -I + diag(1 - tanh^2(W x + b)) W."""
    pre = W @ x + bvec
    return (1.0 - np.tanh(pre) ** 2)[:, None] * W - np.eye(len(x))


def enumerate_cell(m, seed_enum, line_starts=None, n_starts=1000):
    """All fixed points at u = 0, with the CONTINUOUS Jacobian spectrum of each."""
    u0 = np.zeros(4, np.float32)
    fps, n_conv, n_cand = NL.fixed_points_hardened(m, u0, n_starts=n_starts,
                                                   traj_starts=line_starts,
                                                   seed=seed_enum)
    W = m.W.detach().cpu().numpy().astype(float)
    bb = m.b.detach().cpu().numpy().astype(float)
    recs = []
    for f in fps:
        x = np.asarray(f["point"], float)
        ev = np.linalg.eigvals(jacobian_c(x, W, bb))
        re = np.sort(ev.real)[::-1]
        recs.append(dict(
            point=x.tolist(), norm=float(np.linalg.norm(x)),
            stable_discrete=bool(f["stable"]),
            max_abs_eig_discrete=float(f["max_abs_eig"]),
            Jc_max_re=float(re[0]),
            Jc_second_re=float(re[1]) if len(re) > 1 else None,
            Jc_min_re=float(re[-1]),
            Jc_max_abs_im=float(np.abs(ev.imag).max()),
            n_unstable=int(np.sum(ev.real > 0)),
            stable_continuous=bool(np.all(ev.real < 0))))
    comp = NL.completeness_check(m, u0, [np.asarray(f["point"], float)
                                         for f in fps if f["stable"]],
                                 n_probe=500, seed=seed_enum)
    return recs, dict(
        n_converged=int(n_conv), n_candidates=int(n_cand),
        frac_accounted=float(comp["frac_accounted"]),
        max_dist=float(comp["max_dist"]), median_dist=float(comp["median_dist"]),
        statement="every one of 500 states settled from random initial conditions "
                  "lies within 1e-2 of an enumerated stable fixed point when "
                  "frac_accounted = 1.0")


# ==========================================================================
# 4.  escape / first-passage
# ==========================================================================
def _step_batch(m, r, u, sigma, gen):
    """One noisy step of the network map (FineRNN.step), batched."""
    r = m.step(r, u)
    if sigma > 0:
        r = r + sigma * torch.randn(r.shape, device=r.device, generator=gen)
    return r


def fused_step_parts(m):
    """W^T and b for the u = 0 update, and the check that the fusion is exact.

    At u = 0 the network step reduces to
        r <- (1-alpha) r + alpha tanh(r W^T + b)
    because W_in u = 0.  Fusing it into one addmm removes two kernel launches per
    step, which matters when 60,000 steps are taken per cell.  The fusion is
    VERIFIED against NL.FineRNN.step rather than assumed: the returned dict carries
    the maximum absolute disagreement on a random batch, which must be at float32
    round-off.
    """
    Wt = m.W.detach().t().contiguous()
    bv = m.b.detach()
    g = torch.Generator(device=DEV).manual_seed(0)
    rr = torch.randn(64, m.n_units, device=DEV, generator=g)
    with torch.no_grad():
        ref = m.step(rr, torch.zeros(64, 4, device=DEV))
        got = (1 - ALPHA) * rr + ALPHA * torch.tanh(torch.addmm(bv, rr, Wt))
        err = float((ref - got).abs().max())
    return Wt, bv, err


def first_passage(m, starts, boundaries, sigma, seed, max_steps):
    """First-passage time of each chain to each boundary, in samples.

    `starts` is (n_chains, d); `boundaries` is a list of (name, normal, offset) and
    a chain has crossed when normal . x >= offset.  Chains start exactly at a fixed
    point.  The within-well relaxation time (~1/|Re lambda| ~ 3.4 time units = 68
    samples) is far below the shortest first-passage time measured here, so the
    deterministic start is not rate limiting.  -1 marks a censored chain.

    No per-step host synchronisation: the completion test runs once every
    CHECK_EVERY steps and the noise is pre-generated in blocks, so the loop issues
    device work without waiting on it.
    """
    Wt, bv, fuse_err = fused_step_parts(m)
    gen = torch.Generator(device=DEV).manual_seed(int(seed))
    r = torch.as_tensor(np.asarray(starts, np.float32), device=DEV)
    n_chains = r.shape[0]
    NV = torch.as_tensor(np.stack([b[1] for b in boundaries]).astype(np.float32).T,
                         device=DEV).contiguous()          # (d, n_bound)
    OF = torch.as_tensor(np.array([b[2] for b in boundaries], np.float32), device=DEV)
    fp = torch.full((n_chains, len(boundaries)), -1, dtype=torch.long, device=DEV)
    step = 0
    with torch.no_grad():
        while step < max_steps:
            nblk = min(NOISE_BLOCK, max_steps - step)
            noise = torch.randn(nblk, n_chains, r.shape[1], device=DEV,
                                generator=gen).mul_(sigma)
            for j in range(nblk):
                step += 1
                r = ((1 - ALPHA) * r
                     + ALPHA * torch.tanh(torch.addmm(bv, r, Wt))).add_(noise[j])
                newly = ((r @ NV) >= OF) & (fp < 0)
                fp = torch.where(newly, step, fp)
            if bool((fp >= 0).all()):
                break
    return fp.cpu().numpy(), int(step), fuse_err


def mfpt_from_fp(fp_col, t_end, rng):
    """Exponential MLE under type-I censoring: total time at risk / n events.

    The interval here is a bootstrap over CHAINS and is a within-cell precision
    statement only.  Every group-level interval in this script resamples anchor
    SEEDS, which is the unit of analysis.
    """
    fp_col = np.asarray(fp_col, float)
    esc = fp_col > 0
    n_esc = int(esc.sum())
    risk = float(np.where(esc, fp_col, t_end).sum())
    out = dict(n_escaped=n_esc, n_chains=int(len(fp_col)),
               censored_frac=float(1 - n_esc / len(fp_col)),
               total_time_at_risk=risk)
    if n_esc == 0:
        out.update(mfpt=None, ci_lo=None, ci_hi=None,
                   mfpt_lower_bound=risk)          # every chain censored
        return out
    idx = rng.integers(0, len(fp_col), size=(2000, len(fp_col)))
    nb = esc[idx].sum(1)
    tot = np.where(esc[idx], fp_col[idx], t_end).sum(1)
    boot = np.where(nb > 0, tot / np.maximum(nb, 1), np.nan)
    out.update(mfpt=risk / n_esc,
               ci_lo=float(np.nanpercentile(boot, 2.5)),
               ci_hi=float(np.nanpercentile(boot, 97.5)))
    return out


# ==========================================================================
# 5.  noise-free flow and perturbation recovery
# ==========================================================================
def flow_last(m, X0, n_steps):
    """Final state of the deterministic network map from a batch of starts."""
    Wt, bv, _ = fused_step_parts(m)
    r = torch.as_tensor(np.asarray(X0, np.float32), device=DEV)
    with torch.no_grad():
        for _ in range(n_steps):
            r = (1 - ALPHA) * r + ALPHA * torch.tanh(torch.addmm(bv, r, Wt))
    return r.cpu().numpy()


def flow_traj(m, X0, n_steps):
    """Full deterministic trajectory (n_steps+1, n, d); use only for small n.

    Accumulated on the device and transferred once -- a per-step .cpu() call costs
    one synchronisation per step and dominates the wall clock otherwise.
    """
    Wt, bv, _ = fused_step_parts(m)
    r = torch.as_tensor(np.asarray(X0, np.float32), device=DEV)
    out = torch.empty((n_steps + 1,) + tuple(r.shape), device=DEV)
    out[0] = r
    with torch.no_grad():
        for i in range(n_steps):
            r = (1 - ALPHA) * r + ALPHA * torch.tanh(torch.addmm(bv, r, Wt))
            out[i + 1] = r
    return out.cpu().numpy()


def flow_dist(m, X0, ref, n_steps):
    """Distance to `ref` along the deterministic trajectory: (n_steps+1, n)."""
    Wt, bv, _ = fused_step_parts(m)
    r = torch.as_tensor(np.asarray(X0, np.float32), device=DEV)
    rt = torch.as_tensor(np.asarray(ref, np.float32), device=DEV)
    d = torch.empty((n_steps + 1, r.shape[0]), device=DEV)
    d[0] = torch.linalg.norm(r - rt, dim=1)
    with torch.no_grad():
        for i in range(n_steps):
            r = (1 - ALPHA) * r + ALPHA * torch.tanh(torch.addmm(bv, r, Wt))
            d[i + 1] = torch.linalg.norm(r - rt, dim=1)
    return d.cpu().numpy()


def critical_displacement(m, x_from, x_to, direction, hi, n_scan, t_relax, passes=2):
    """Largest displacement along `direction` that still returns to x_from.

    Two batched scans (a coarse grid, then a refinement inside the bracketing
    interval) rather than a sequential bisection, so this is two GPU passes.  The
    outcome is decided by which stable fixed point the noise-free flow ends nearest.
    """
    lo, up = 0.0, float(hi)
    bracketed = False
    x_from = np.asarray(x_from, float)
    x_to = np.asarray(x_to, float)
    direction = np.asarray(direction, float)
    for _ in range(passes):
        ds = np.linspace(lo, up, n_scan)
        end = flow_last(m, x_from[None, :] + ds[:, None] * direction[None, :], t_relax)
        returned = (np.linalg.norm(end - x_from, axis=1)
                    < np.linalg.norm(end - x_to, axis=1))
        if returned.all():
            return float(ds[-1]), False
        k = int(np.flatnonzero(~returned)[0])
        if k == 0:
            return float(ds[0]), False
        lo, up, bracketed = float(ds[k - 1]), float(ds[k]), True
    return 0.5 * (lo + up), bracketed


def recovery_time(m, x_from, direction, d, t_relax):
    """Noise-free samples for ||x - x_from|| to fall to 1/e of its initial value."""
    X0 = np.asarray(x_from, float)[None, :] + float(d) * np.asarray(direction, float)[None, :]
    dist = flow_dist(m, X0, x_from, t_relax)[:, 0]
    hit = np.flatnonzero(dist <= dist[0] / math.e)
    return (int(hit[0]) if len(hit) else None), float(dist[0]), float(dist[-1])


# ==========================================================================
# 6.  one (anchor seed, gain) cell
# ==========================================================================
def escape_rules(ax, mid, pS, p_to, w, xS, forward):
    """The six transition rules.  `forward` = moving along +ax.

    A boundary is (name, normal, offset), crossed when normal . x >= offset.
    For a wells-axis threshold `thr` in the mid-centred projection the boundary is
    sgn*(x - mid).ax >= sgn*thr, i.e. normal = sgn*ax, offset = sgn*thr + (sgn*ax).mid.
    """
    sgn = 1.0 if forward else -1.0
    wv = w if forward else -w
    R = [("separatrix_saddle_lefteig", wv, float(wv @ xS)),
         ("separatrix_saddle_wellsaxis", sgn * ax, float(sgn * pS + (sgn * ax) @ mid))]
    for c in CORE_FRACS:
        thr = pS + c * (p_to - pS)
        R.append((f"core_{c:.2f}", sgn * ax, float(sgn * thr + (sgn * ax) @ mid)))
    R.append(("midpoint", sgn * ax, float((sgn * ax) @ mid)))
    return R


def measure_cell(net, gain, seed_idx, anchor_seed, ax_ref, well_seg, sink,
                 quick=False):
    m = NL.scaled_net(net, gain)
    W = m.W.detach().cpu().numpy().astype(float)
    bb = m.b.detach().cpu().numpy().astype(float)
    rng = np.random.default_rng(SEED + 7 + seed_idx)

    # Starts along the segment joining the reference-gain wells, so the saddle
    # cannot be missed.  The fixed points move continuously with the gain, so the
    # reference segment still passes close to the saddle at every gain; this costs
    # one enumeration per cell instead of two.
    line_starts = None
    if well_seg is not None:
        tg = np.linspace(-0.1, 1.1, N_LINE_STARTS)[:, None]
        line_starts = well_seg[0][None, :] * (1 - tg) + well_seg[1][None, :] * tg

    recs, comp = enumerate_cell(m, ENUM_SEED, line_starts,
                                n_starts=(150 if quick else N_STARTS))
    stable = [r for r in recs if r["stable_discrete"]]
    unstab = [r for r in recs if not r["stable_discrete"]]
    cell = dict(seed_index=int(seed_idx), anchor_seed=int(anchor_seed),
                gain=float(gain), gain_level=GAIN_LABEL[gain],
                n_fixed_points=len(recs), n_stable=len(stable),
                n_unstable_points=len(unstab), fixed_points=recs, completeness=comp,
                gb1_pass=bool(len(stable) == 2 and comp["frac_accounted"] >= 0.98),
                saddle_found=False)
    if not cell["gb1_pass"]:
        cell["excluded_reason"] = ("not two stable fixed points, or completeness "
                                   "< 0.98 -- a gain that lost bistability is a "
                                   "different system, not a lower barrier")
        return cell

    A0, B0 = np.asarray(stable[0]["point"]), np.asarray(stable[1]["point"])
    L = float(np.linalg.norm(B0 - A0))
    ax = (B0 - A0) / L
    if ax_ref is not None and float(ax @ ax_ref) < 0:
        ax, A0, B0 = -ax, B0, A0
        stable = stable[::-1]
    mid = 0.5 * (A0 + B0)
    cell["fp_separation"] = L
    cell["wells_axis"] = ax.tolist()

    if not unstab:
        cell["excluded_reason"] = "no unstable fixed point found; no separatrix"
        return cell
    cands = [s for s in unstab if s["n_unstable"] == 1] or unstab
    sad = cands[int(np.argmin([abs(float((np.asarray(s["point"]) - mid) @ ax))
                               for s in cands]))]
    xS = np.asarray(sad["point"], float)
    cell["saddle_found"] = True
    cell["saddle"] = dict(sad)
    pA, pB = float((A0 - mid) @ ax), float((B0 - mid) @ ax)
    pS = float((xS - mid) @ ax)
    cell["saddle_proj_on_wells_axis"] = pS
    cell["well_projections"] = dict(lo=pA, hi=pB)

    # ---- separatrix normal: LEFT eigenvector of J_c for the unstable eigenvalue
    JcS = jacobian_c(xS, W, bb)
    evS, VR = np.linalg.eig(JcS)
    _, VL = np.linalg.eig(JcS.T)
    ku = int(np.argmax(evS.real))
    w = np.real(VL[:, ku])
    w = w / np.linalg.norm(w)
    if float(w @ (B0 - xS)) < 0:
        w = -w
    vu = np.real(VR[:, ku])
    vu = vu / np.linalg.norm(vu)
    if float(vu @ (B0 - xS)) < 0:
        vu = -vu
    cell["separatrix_normal_unstable_lefteig"] = w.tolist()
    cell["separatrix_normal_vs_wells_axis_cos"] = float(abs(w @ ax))
    cell["saddle_unstable_eig"] = float(evS.real[ku])
    cell["saddle_n_unstable"] = int(np.sum(evS.real > 0))

    # ---- (4) Freidlin-Wentzell quasipotential, both directions --------------
    t_g = time.time()
    W_t = torch.as_tensor(W, dtype=torch.float64, device=DEV)
    b_t = torch.as_tensor(bb, dtype=torch.float64, device=DEV)
    het = flow_traj(m, np.stack([xS + 1e-3 * vu, xS - 1e-3 * vu]).astype(np.float32),
                    T_HETERO if not quick else 1500)
    br_b, br_a = het[:, 0, :].astype(float), het[:, 1, :].astype(float)
    if np.linalg.norm(br_b[-1] - B0) > np.linalg.norm(br_b[-1] - A0):
        br_b, br_a = br_a, br_b
    path_AB = np.vstack([br_a[::-1], br_b])
    cell["heteroclinic_endpoint_error"] = dict(
        to_hi=float(np.linalg.norm(br_b[-1] - B0)),
        to_lo=float(np.linalg.norm(br_a[-1] - A0)))
    pairs = [(A0, B0), (B0, A0), (A0, xS), (B0, xS), (xS, B0), (xS, A0)]
    extras = [[path_AB, np.vstack([A0, xS, B0])],
              [path_AB[::-1].copy(), np.vstack([B0, xS, A0])],
              [br_a[::-1].copy()], [br_b[::-1].copy()],
              [br_b.copy()], [br_a.copy()]]
    S, _P = gmam_batch(pairs, W_t, b_t, D_DIFF, inits_extra=extras,
                       n_iter=(400 if quick else GM_ITERS))
    tg = np.linspace(0, 1, GM_PTS)[:, None]
    S_line = float(NL.geometric_action(A0[None, :] * (1 - tg) + B0[None, :] * tg,
                                       lambda X: drift_np(X, W, bb),
                                       np.eye(len(A0)) * D_DIFF))
    cell["fw_action"] = dict(
        A_to_B=S[0], B_to_A=S[1], A_to_saddle=S[2], B_to_saddle=S[3],
        saddle_to_B=S[4], saddle_to_A=S[5], straight_line_A_to_B=S_line,
        units="nats = dU/D with a = 2D, D = %.6f; the same units as "
              "barrier_generative_nats" % D_DIFF,
        seconds=float(time.time() - t_g))
    if seed_idx == 0 and not quick:
        t_al = time.time()
        S_al, _ = NL.gmam(A0, B0, lambda X: drift_np(X, W, bb),
                          D=np.eye(len(A0)) * D_DIFF, n_pts=GM_PTS,
                          n_iter=AL_GMAM_ITERS)
        cell["fw_action"]["attractor_lib_gmam_A_to_B"] = float(S_al)
        cell["fw_action"]["attractor_lib_gmam_seconds"] = float(time.time() - t_al)
        cell["fw_action"]["attractor_lib_gmam_iters"] = AL_GMAM_ITERS

    # ---- (2) escape: first-passage under six transition rules ---------------
    # Both directions run in ONE batch of chains.  Each direction has its own
    # boundary set; a chain's own direction's boundaries are the ones scored, and
    # the other direction's are scored too but read only as a cross-check that the
    # two halves never interfere (they cannot: the chains are independent).
    t_e = time.time()
    nch = 32 if quick else N_CHAINS
    mx = 15_000 if quick else MAX_STEPS
    rl_f = escape_rules(ax, mid, pS, pB, w, xS, True)
    rl_b = escape_rules(ax, mid, pS, pA, w, xS, False)
    esc = {}
    fuse_err = None
    for lbl, x_from, rl, off in (("lo_to_hi", A0, rl_f, 0), ("hi_to_lo", B0, rl_b, 7)):
        viol = [nm for nm, nv, of in rl if float(nv @ x_from) >= of]
        starts = np.tile(np.asarray(x_from, np.float32), (nch, 1))
        fpm, t_end, fuse_err = first_passage(
            m, starts, rl, SIGMA,
            SIM_SEED + 131 * seed_idx + int(round(gain * 1000)) + off, max_steps=mx)
        d = {}
        for k, (nm, _nv, _of) in enumerate(rl):
            stt = mfpt_from_fp(fpm[:, k], t_end, rng)
            stt["escape_rate_per_sample"] = (1.0 / stt["mfpt"]) if stt["mfpt"] else None
            stt["escape_rate_per_time"] = ((1.0 / (stt["mfpt"] * ALPHA))
                                           if stt["mfpt"] else None)
            d[nm] = stt
        esc[lbl] = dict(rules=d, horizon_samples=int(t_end),
                        start_side_violations=viol, n_chains=int(nch))
    esc["seconds"] = float(time.time() - t_e)
    esc["primary_rule"] = PRIMARY_RULE
    esc["fused_step_max_abs_error_vs_reference_step"] = fuse_err
    cell["escape"] = esc

    # ---- (3) controlled perturbation recovery -------------------------------
    t_p = time.time()
    rec = {}
    for lbl, x_from, x_to in (("lo", A0, B0), ("hi", B0, A0)):
        direc = (x_to - x_from) / np.linalg.norm(x_to - x_from)
        dcrit, brk = critical_displacement(
            m, x_from, x_to, direc, hi=1.2 * L,
            n_scan=(100 if quick else N_SCAN), t_relax=(800 if quick else T_RELAX))
        t_fix, _, _ = recovery_time(m, x_from, direc, D_FIXED,
                                    t_relax=(800 if quick else T_RELAX))
        t_half, _, _ = recovery_time(m, x_from, direc, 0.5 * dcrit,
                                     t_relax=(800 if quick else T_RELAX))
        rec[lbl] = dict(critical_displacement=float(dcrit), bracketed=bool(brk),
                        critical_over_separation=float(dcrit / L),
                        fixed_displacement=D_FIXED,
                        t_recover_1e_fixed_samples=t_fix,
                        t_recover_1e_fixed_time=(t_fix * ALPHA) if t_fix else None,
                        halfcrit_displacement=float(0.5 * dcrit),
                        t_recover_1e_halfcrit_samples=t_half,
                        t_recover_1e_halfcrit_time=(t_half * ALPHA) if t_half else None)
    rec["seconds"] = float(time.time() - t_p)
    cell["perturbation"] = rec

    # ---- the COMPARISON quantity: the fitted-density barrier --------------
    t_l = time.time()
    gen = torch.Generator(device=DEV).manual_seed(
        int(SIM_SEED + 977 * seed_idx + int(round(gain * 1000))))
    nch = 16 if quick else N_LONG_CHAINS
    tlen = 3000 if quick else T_LONG_CHAIN
    start = np.vstack([np.tile(A0.astype(np.float32), (nch // 2, 1)),
                       np.tile(B0.astype(np.float32), (nch - nch // 2, 1))])
    r = torch.as_tensor(start, device=DEV)
    axt = torch.as_tensor(ax.astype(np.float32), device=DEV)
    midt = torch.as_tensor(mid.astype(np.float32), device=DEV)
    Wt, bv, _ = fused_step_parts(m)
    buf = torch.empty((tlen, nch), device=DEV)
    done = 0
    with torch.no_grad():
        while done < tlen:
            nblk = min(NOISE_BLOCK, tlen - done)
            noise = torch.randn(nblk, nch, r.shape[1], device=DEV,
                                generator=gen).mul_(SIGMA)
            for j in range(nblk):
                r = ((1 - ALPHA) * r
                     + ALPHA * torch.tanh(torch.addmm(bv, r, Wt))).add_(noise[j])
                buf[done + j] = (r - midt) @ axt
            done += nblk
    proj = buf.cpu().numpy().T[:, LONG_BURN:]
    lab = (proj > 0).astype(int)
    n_tr = int(sum(int(np.count_nonzero(np.diff(lab[i]))) for i in range(nch)))
    n_s = int(proj.shape[0] * proj.shape[1])
    lm, lb_, ls_ = NL.latent_landscape(proj.ravel(), lab.ravel())
    taus = [NL.tau_within_dwell(proj[i], lab[i]) for i in range(nch)]
    taus = [tv for tv in taus if tv is not None]
    cell["density_read"] = dict(
        barrier_generative_nats=lb_, latent_density_modes=lm, latent_separation=ls_,
        mean_dwell_long=(n_s / n_tr) if n_tr else None, n_transitions_long=n_tr,
        n_samples=n_s, n_chains=int(nch),
        tau_within_dwell=float(np.median(taus)) if taus else None,
        estimator="latent-landscape density barrier; midpoint labelling on the wells-axis projection",
        seconds=float(time.time() - t_l))

    cell["wells"] = {k: dict(Jc_max_re=stable[i]["Jc_max_re"],
                             Jc_second_re=stable[i]["Jc_second_re"],
                             Jc_min_re=stable[i]["Jc_min_re"],
                             Jc_max_abs_im=stable[i]["Jc_max_abs_im"],
                             n_unstable=stable[i]["n_unstable"])
                     for i, k in ((0, "lo"), (1, "hi"))}
    return cell


# ==========================================================================
# 7.  group statistics
# ==========================================================================
def _rho_rows(A, rg):
    """Pearson correlation of each row of A (ranks) with rg (ranks)."""
    a = A - A.mean(-1, keepdims=True)
    b = rg - rg.mean()
    na = np.linalg.norm(a, axis=-1)
    return (a @ b) / np.maximum(na * np.linalg.norm(b), 1e-300)


def spearman_perm(vals_by_seed, gains, n_perm=N_PERM, seed=SEED):
    """Mean within-seed Spearman rho with gain, against a within-seed label
    permutation null.  Gain labels are permuted INDEPENDENTLY within each seed,
    which preserves each seed's own marginal distribution of the measure and
    destroys only the gain ordering -- the structure-matched null this project's
    convention requires.  Nothing here is compared to zero by assumption: the
    reference distribution is simulated."""
    rows = [np.asarray(v, float) for v in vals_by_seed
            if v is not None and np.all(np.isfinite(np.asarray(v, float)))]
    n = len(gains)
    if len(rows) < 3:
        return None
    rng = np.random.default_rng(seed)
    M = np.stack(rows)
    rk = np.stack([sps.rankdata(r) for r in M])
    rg = sps.rankdata(np.asarray(gains, float))
    per = _rho_rows(rk, rg)
    obs = float(per.mean())
    n_mono = int(sum(1 for r in M if np.all(np.diff(r) > 0) or np.all(np.diff(r) < 0)))
    perms = np.argsort(rng.random((n_perm, len(M), n)), axis=2)
    null = _rho_rows(np.take_along_axis(np.broadcast_to(rk, (n_perm,) + rk.shape),
                                        perms, axis=2), rg).mean(axis=1)
    p = float((1 + np.sum(np.abs(null) >= abs(obs))) / (n_perm + 1))
    bs = rng.integers(0, len(M), size=(N_BOOT, len(M)))
    boot = per[bs].mean(1)
    return dict(n_seeds=int(len(M)), mean_spearman=obs,
                ci_lo=float(np.percentile(boot, 2.5)),
                ci_hi=float(np.percentile(boot, 97.5)),
                p_perm=p, n_perm=int(n_perm),
                n_seeds_strictly_monotone=n_mono,
                per_seed_spearman=per.tolist(),
                null_mean=float(null.mean()), null_sd=float(null.std()),
                null_abs_p95=float(np.percentile(np.abs(null), 95)),
                per_seed_null_prob_perfect_monotone=2.0 / math.factorial(n),
                group_mean=float(M.mean(0).tolist()[0]) if False else None,
                per_gain_mean=M.mean(0).tolist(), per_gain_sd=M.std(0, ddof=1).tolist(),
                per_seed_values=M.tolist())


def exclusion_bound(vals_by_seed, i_from, i_to, seed=SEED):
    """Paired change between two gain levels, with a seed-level bootstrap interval.

    Reported as an EXCLUSION BOUND rather than as "no effect": whatever the
    interval excludes is what the design rules out.  The unit resampled is the
    anchor seed.
    """
    rows = [np.asarray(v, float) for v in vals_by_seed
            if v is not None and np.all(np.isfinite(np.asarray(v, float)))]
    if len(rows) < 3:
        return None
    M = np.stack(rows)
    d = M[:, i_to] - M[:, i_from]
    rng = np.random.default_rng(seed + 11)
    bs = rng.integers(0, len(d), size=(N_BOOT, len(d)))
    boot = d[bs].mean(1)
    lo, hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
    base = float(np.abs(M[:, i_from]).mean())
    out = dict(n_seeds=int(len(d)), mean_from=float(M[:, i_from].mean()),
               mean_to=float(M[:, i_to].mean()), mean_change=float(d.mean()),
               ci_lo=lo, ci_hi=hi,
               largest_change_not_excluded=float(max(abs(lo), abs(hi))),
               relative_bound=(float(max(abs(lo), abs(hi)) / base)
                               if base > 0 else None),
               per_seed_change=d.tolist(),
               statement="changes larger than %.4g (absolute) between these two "
                         "gain levels are excluded at 95%%" % max(abs(lo), abs(hi)))
    if np.all(M > 0):
        f = M[:, i_to] / M[:, i_from]
        bf = f[bs].mean(1)
        out["fold_change"] = dict(mean=float(f.mean()),
                                  ci_lo=float(np.percentile(bf, 2.5)),
                                  ci_hi=float(np.percentile(bf, 97.5)),
                                  per_seed=f.tolist())
    return out


def slope_group(x_by_seed, y_by_seed, h0=1.0, seed=SEED):
    """Per-seed OLS slope of y on x, then a seed-level interval.  The unit of
    analysis is the anchor seed; nothing here resamples time points or chains."""
    rng = np.random.default_rng(seed + 3)
    sl, r2 = [], []
    for x, y in zip(x_by_seed, y_by_seed):
        x, y = np.asarray(x, float), np.asarray(y, float)
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 3 or np.ptp(x[ok]) < 1e-12:
            continue
        lr = sps.linregress(x[ok], y[ok])
        sl.append(float(lr.slope))
        r2.append(float(lr.rvalue ** 2))
    if len(sl) < 3:
        return None
    sl = np.asarray(sl)
    bs = rng.integers(0, len(sl), size=(N_BOOT, len(sl)))
    boot = sl[bs].mean(1)
    t1 = sps.ttest_1samp(sl, h0)
    t0 = sps.ttest_1samp(sl, 0.0)
    hw = sps.t.ppf(0.975, len(sl) - 1) * sl.std(ddof=1) / math.sqrt(len(sl))
    return dict(n_seeds=int(len(sl)), mean_slope=float(sl.mean()),
                sd_slope=float(sl.std(ddof=1)),
                se_slope=float(sl.std(ddof=1) / math.sqrt(len(sl))),
                ci_lo_boot=float(np.percentile(boot, 2.5)),
                ci_hi_boot=float(np.percentile(boot, 97.5)),
                ci_lo_t=float(sl.mean() - hw), ci_hi_t=float(sl.mean() + hw),
                h0=float(h0), t_vs_h0=float(t1.statistic), p_vs_h0=float(t1.pvalue),
                t_vs_zero=float(t0.statistic), p_vs_zero=float(t0.pvalue),
                mean_r2=float(np.mean(r2)), per_seed_slope=sl.tolist(),
                per_seed_r2=r2)



# ==========================================================================
# 8.  worker mode: cells of assigned network indices, written to the cell cache
# ==========================================================================
def qualification_file(sd):
    """The qualification record of candidate `sd`: the cached one if present, else the shipped one."""
    own = OUTDIR / f"qualification_{sd}.json"
    return own if own.exists() else CACHE / f"qualification_{sd}.json"


def select_anchors(sink):
    """The main anchor loop, with its qualification records reused when present.

    A single cell is launch-bound on the GPU, so several cells run concurrently on
    one device at close to the single-cell rate.  Worker mode measures an assigned
    subset of network INDICES and writes exactly the cell files the main loop would
    have written; the main loop then aggregates them with --resume.  Every random
    seed inside measure_cell is a function of (seed_index, gain) only, and the
    seed_index assigned here is the same position in the same candidate order.
    """
    anchors, recs = [], []
    for sd in [SEED + 7 + 13 * kk for kk in range(N_CANDIDATES_MAX)]:
        net, score = get_anchor(sd, sink)
        qc = qualification_file(sd)
        if qc.exists():
            rec = json.loads(qc.read_text(encoding="utf-8"))
        else:
            fps, _, _ = NL.fixed_points_hardened(net, np.zeros(4, np.float32),
                                                 n_starts=1000, seed=int(sd))
            stable = [np.asarray(f["point"], float) for f in fps if f["stable"]]
            comp = NL.completeness_check(net, np.zeros(4, np.float32), stable,
                                         n_probe=500, seed=int(sd))
            rec = dict(
                seed=int(sd), train_score=float(score), n_fixed_points=len(fps),
                n_stable=len(stable), completeness=float(comp["frac_accounted"]),
                fp_separation_gain1=(float(np.linalg.norm(stable[0] - stable[1]))
                                     if len(stable) == 2 else None),
                usable=bool(len(stable) == 2 and comp["frac_accounted"] >= 0.98))
            OUTDIR.mkdir(parents=True, exist_ok=True)
            (OUTDIR / f"qualification_{sd}.json").write_text(json.dumps(_clean(rec), indent=1),
                                                             encoding="utf-8")
        recs.append(rec)
        print(f"  candidate {sd}: score {rec['train_score']:+.3f}  "
              f"{rec['n_stable']} stable  completeness {rec['completeness']:.2f}  "
              f"{'SELECTED' if rec['usable'] else 'rejected'}", flush=True)
        if rec["usable"]:
            anchors.append((int(sd), net))
        if len(anchors) >= N_ANCHORS_TARGET:
            break
    return anchors, recs


def worker(args):
    sink = []
    anchors, _ = select_anchors(sink)
    print(f"  {len(anchors)} usable networks: {[s for s, _ in anchors]}", flush=True)
    if args.qualify_only:
        return 0

    want = {int(v) for v in args.indices.split(",") if v.strip() != ""}
    CELLS = OUTDIR / "cells"
    CELLS.mkdir(parents=True, exist_ok=True)

    for si, (sd, net) in enumerate(anchors):
        if want and si not in want:
            continue
        ref = {}

        def _ref(net=net):
            if "ax" in ref:
                return ref["ax"], ref["seg"]
            m_ref = NL.scaled_net(net, REF_GAIN)
            f_ref, _, _ = NL.fixed_points_hardened(
                m_ref, np.zeros(4, np.float32), n_starts=N_STARTS,
                seed=ENUM_SEED)
            s_ref = [np.asarray(f["point"], float) for f in f_ref if f["stable"]]
            ref["ax"], ref["seg"] = None, None
            if len(s_ref) == 2:
                ref["ax"] = ((s_ref[1] - s_ref[0])
                             / np.linalg.norm(s_ref[1] - s_ref[0]))
                ref["seg"] = (s_ref[0], s_ref[1])
            return ref["ax"], ref["seg"]

        for g in GAINS:
            cf = CELLS / f"cell_{sd}_{int(round(g*100)):03d}.json"
            if cf.exists():
                print(f"  seed {sd} gain {g:.2f}: cached", flush=True)
                continue
            tc = time.time()
            ax_ref, well_seg = _ref()
            cell = measure_cell(net, g, si, sd, ax_ref, well_seg, sink,
                                quick=False)
            cell["seconds"] = float(time.time() - tc)
            cf.write_text(json.dumps(_clean(cell), indent=1), encoding="utf-8")
            if cell["gb1_pass"] and cell["saddle_found"]:
                pr = cell["escape"]["lo_to_hi"]["rules"][PRIMARY_RULE]
                print(f"  seed {sd} gain {g:.2f} ({GAIN_LABEL[g]}): fp_sep "
                      f"{cell['fp_separation']:.3f}  saddle lam_u "
                      f"{cell['saddle_unstable_eig']:+.4f}  well lam_max "
                      f"{cell['wells']['lo']['Jc_max_re']:+.4f}/"
                      f"{cell['wells']['hi']['Jc_max_re']:+.4f}  S_FW "
                      f"{cell['fw_action']['A_to_saddle']:.3f}/"
                      f"{cell['fw_action']['B_to_saddle']:.3f}  B_dens "
                      f"{cell['density_read']['barrier_generative_nats']:.3f}  "
                      f"MFPT {pr['mfpt']}  d_crit "
                      f"{cell['perturbation']['lo']['critical_displacement']:.3f}"
                      f"  [{cell['seconds']:.0f}s]", flush=True)
            else:
                print(f"  seed {sd} gain {g:.2f}: EXCLUDED -- "
                      f"{cell.get('excluded_reason')}", flush=True)
    return 0


# ==========================================================================
# 9.  main
# ==========================================================================
def main():
    ap = argparse.ArgumentParser(description="escape ladder of the trained flip-flop networks")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--resume", action="store_true",
                    help="reuse the cached cells, qualification records and self-test")
    ap.add_argument("--indices", default="",
                    help="worker mode: measure only the cells of these network indices")
    ap.add_argument("--qualify-only", action="store_true",
                    help="worker mode: qualify the candidate networks and stop")
    args = ap.parse_args()
    if args.indices or args.qualify_only:
        return worker(args)

    t0 = time.time()
    sink = []
    OUTDIR.mkdir(parents=True, exist_ok=True)

    res = dict(
        date="2026-09-21",
        conventions=dict(
            alpha=ALPHA, sigma=SIGMA, D=D_DIFF, D_formula="D = sigma^2 / (2 alpha)",
            time_per_sample=ALPHA,
            action_units="nats, = dU/D, the same units as barrier_generative_nats",
            jacobian="continuous-time J_c = -I + diag(1-tanh^2(W_g x + b)) W_g "
                     "= (J_discrete - I)/alpha; eigenvalues are rates per unit time"),
        seeds=dict(project_seed=SEED, enum_seed=ENUM_SEED, sim_seed=SIM_SEED),
        gains=list(GAINS), gain_labels={str(k): v for k, v in GAIN_LABEL.items()},
        versions=dict(python=platform.python_version(), numpy=np.__version__,
                      scipy=scipy.__version__, sklearn=sklearn.__version__,
                      torch=torch.__version__, device=DEV,
                      gpu=(torch.cuda.get_device_name(0)
                           if torch.cuda.is_available() else None),
                      platform=platform.platform()),
        settings=dict(n_chains=N_CHAINS, max_steps=MAX_STEPS,
                      core_fracs=list(CORE_FRACS), primary_rule=PRIMARY_RULE,
                      n_long_chains=N_LONG_CHAINS, t_long_chain=T_LONG_CHAIN,
                      gmam_pts=GM_PTS, gmam_iters=GM_ITERS, n_perm=N_PERM,
                      n_boot=N_BOOT, n_scan=N_SCAN, t_relax=T_RELAX,
                      fixed_displacement=D_FIXED, quick=bool(args.quick)))

    _log("=" * 78, sink)
    _log("escape ladder -- independent dynamical stability measures in the recurrent circuit",
         sink)
    _log(f"  device {DEV}  alpha {ALPHA}  sigma {SIGMA}  D = {D_DIFF:.6f}", sink)
    _log("=" * 78, sink)

    # ---------------- self-tests / calibration
    _log("\n[SELF-TEST] minimum-action estimator and the Arrhenius machinery", sink)
    st_cache = OUTDIR / "selftest.json"
    if args.resume and st_cache.exists():
        st = json.loads(st_cache.read_text(encoding="utf-8"))
        _log("  (cached)", sink)
    else:
        st = _selftest_gmam(sink)
        st_cache.parent.mkdir(parents=True, exist_ok=True)
        st_cache.write_text(json.dumps(_clean(st), indent=1), encoding="utf-8")
    res["selftest"] = st
    a = st["double_well_1d"]
    _log(f"  1-D double well (exact 5.0): straight line {a['straight_line']:.4f}, "
         f"attractor_lib.gmam {a['attractor_lib_gmam']:.4f}, correct minimiser "
         f"{a['correct_minimiser']:.4f} -- passed by both; cannot discriminate", sink)
    c = st["curved_valley_2d"]
    _log(f"  curved valley (exact 5.0): straight line {c['straight_line']:.3f}, "
         f"attractor_lib.gmam {c['attractor_lib_gmam']:.3f} (descends "
         f"{100*c['attractor_lib_descent_fraction']:.1f}% of the gap, path |y|max "
         f"{c['attractor_lib_path_max_abs_y']:.3f} vs 1.0), correct minimiser "
         f"{c['correct_minimiser']:.4f}", sink)
    k = st["arrhenius_calibration_1d"]
    _log(f"  Arrhenius calibration (fixed prefactor, vary D): slope {k['slope']:.4f} "
         f"[{k['ci_lo']:.4f},{k['ci_hi']:.4f}] (predicted "
         f"{k['predicted_slope']:.4f}), R2 {k['r2']:.4f}", sink)
    k2 = st["arrhenius_calibration_1d_moving_prefactor"]
    _log(f"  Arrhenius calibration (prefactor moves with the barrier): slope "
         f"{k2['slope']:.4f} [{k2['ci_lo']:.4f},{k2['ci_hi']:.4f}] (analytic "
         f"{k2['predicted_slope']:.4f}), R2 {k2['r2']:.4f}", sink)
    if args.selftest:
        (OUTDIR / "escape_ladder_selftest.json").write_text(
            json.dumps(_clean(res), indent=1), encoding="utf-8")
        _log(f"\nwall {time.time()-t0:.0f}s", sink)
        return 0

    # ---------------- anchors
    _log(f"\n[ANCHORS] training candidates until {N_ANCHORS_TARGET} have the "
         "intended enumerated structure", sink)
    anchors, anchor_recs = [], []
    target = 2 if args.quick else N_ANCHORS_TARGET
    for sd in [SEED + 7 + 13 * kk for kk in range(N_CANDIDATES_MAX)]:
        net, score = get_anchor(sd, sink)
        qc = qualification_file(sd)
        if args.resume and qc.exists():
            rec = json.loads(qc.read_text(encoding="utf-8"))
            usable = bool(rec["usable"])
        else:
            fps, _, _ = NL.fixed_points_hardened(net, np.zeros(4, np.float32),
                                                 n_starts=1000, seed=int(sd))
            stable = [np.asarray(f["point"], float) for f in fps if f["stable"]]
            comp = NL.completeness_check(net, np.zeros(4, np.float32), stable,
                                         n_probe=500, seed=int(sd))
            usable = bool(len(stable) == 2 and comp["frac_accounted"] >= 0.98)
            rec = dict(
                seed=int(sd), train_score=float(score), n_fixed_points=len(fps),
                n_stable=len(stable), completeness=float(comp["frac_accounted"]),
                fp_separation_gain1=(float(np.linalg.norm(stable[0] - stable[1]))
                                     if len(stable) == 2 else None), usable=usable)
            qc = OUTDIR / f"qualification_{sd}.json"
            qc.write_text(json.dumps(_clean(rec), indent=1), encoding="utf-8")
        anchor_recs.append(rec)
        comp = {"frac_accounted": rec["completeness"]}
        stable = [None] * rec["n_stable"]
        _log(f"  seed {sd}: score {score:+.3f}  {len(stable)} stable  completeness "
             f"{comp['frac_accounted']:.2f}  {'SELECTED' if usable else 'rejected'}",
             sink)
        if usable:
            anchors.append((int(sd), net))
        if len(anchors) >= target:
            break
    res["anchors"] = dict(
        candidates=anchor_recs, n_examined=len(anchor_recs), n_usable=len(anchors),
        selected_seeds=[s for s, _ in anchors],
        recipe="TI.train_original_recipe(RT.flipflop_batch, tag, 1200, seed); usable "
               "iff exactly 2 stable fixed points at u=0 and completeness >= 0.98; "
               "candidates examined in the order SEED+7+13k until the target is met")
    if not anchors:
        raise SystemExit("no usable anchor")

    # ---------------- the ladder
    _log(f"\n[LADDER] {len(anchors)} anchors x {len(GAINS)} gains", sink)
    cells = []
    CELLS = OUTDIR / "cells"
    CELLS.mkdir(parents=True, exist_ok=True)
    for si, (sd, net) in enumerate(anchors):
        ref = {}

        def _ref(net=net):
            """Wells at the reference gain: the within-seed orientation reference
            and the segment that seeds the saddle search.  Computed lazily so a
            fully cached seed costs nothing on --resume."""
            if "ax" in ref:
                return ref["ax"], ref["seg"]
            m_ref = NL.scaled_net(net, REF_GAIN)
            f_ref, _, _ = NL.fixed_points_hardened(m_ref, np.zeros(4, np.float32),
                                                   n_starts=N_STARTS, seed=ENUM_SEED)
            s_ref = [np.asarray(f["point"], float) for f in f_ref if f["stable"]]
            ref["ax"], ref["seg"] = None, None
            if len(s_ref) == 2:
                ref["ax"] = ((s_ref[1] - s_ref[0])
                             / np.linalg.norm(s_ref[1] - s_ref[0]))
                ref["seg"] = (s_ref[0], s_ref[1])
            return ref["ax"], ref["seg"]

        for g in GAINS:
            tc = time.time()
            cf = CELLS / f"cell_{sd}_{int(round(g*100)):03d}{'_q' if args.quick else ''}.json"
            if args.resume and cf.exists():
                cell = json.loads(cf.read_text(encoding="utf-8"))
                cell["seed_index"] = si
                cells.append(cell)
                _log(f"  seed {sd} gain {g:.2f}: cached", sink)
                continue
            ax_ref, well_seg = _ref()
            cell = measure_cell(net, g, si, sd, ax_ref, well_seg, sink,
                                quick=args.quick)
            cell["seconds"] = float(time.time() - tc)
            cf.write_text(json.dumps(_clean(cell), indent=1), encoding="utf-8")
            cells.append(cell)
            if not (cell["gb1_pass"] and cell["saddle_found"]):
                _log(f"  seed {sd} gain {g:.2f}: EXCLUDED -- "
                     f"{cell.get('excluded_reason')}", sink)
                continue
            pr = cell["escape"]["lo_to_hi"]["rules"][PRIMARY_RULE]
            _log(f"  seed {sd} gain {g:.2f} ({GAIN_LABEL[g]}): fp_sep "
                 f"{cell['fp_separation']:.3f}  saddle lam_u "
                 f"{cell['saddle_unstable_eig']:+.4f}  well lam_max "
                 f"{cell['wells']['lo']['Jc_max_re']:+.4f}/"
                 f"{cell['wells']['hi']['Jc_max_re']:+.4f}  S_FW "
                 f"{cell['fw_action']['A_to_saddle']:.3f}/"
                 f"{cell['fw_action']['B_to_saddle']:.3f}  B_dens "
                 f"{cell['density_read']['barrier_generative_nats']:.3f}  MFPT "
                 f"{pr['mfpt']:.0f}  d_crit "
                 f"{cell['perturbation']['lo']['critical_displacement']:.3f}  "
                 f"[{cell['seconds']:.0f}s]", sink)
    res["cells"] = cells

    ok_idx = {i for i, c in enumerate(cells)
              if c.get("gb1_pass") and c.get("saddle_found")}
    by_seed = {}
    for i in sorted(ok_idx):
        c = cells[i]
        by_seed.setdefault(c["seed_index"], {})[c["gain"]] = c
    full = {k: v for k, v in by_seed.items() if len(v) == len(GAINS)}
    res["ladder_coverage"] = dict(
        n_cells=len(cells), n_cells_ok=len(ok_idx),
        n_seeds_with_full_ladder=len(full),
        seeds_with_full_ladder=[full[k][GAINS[0]]["anchor_seed"] for k in sorted(full)],
        excluded=[{kk: cells[i].get(kk) for kk in
                   ("anchor_seed", "gain", "n_stable", "saddle_found",
                    "excluded_reason")}
                  for i in range(len(cells)) if i not in ok_idx])
    _log(f"\n  {len(ok_idx)}/{len(cells)} cells usable; {len(full)} seeds with a "
         f"complete {len(GAINS)}-gain ladder", sink)
    if len(full) < 3:
        res["error"] = "fewer than 3 complete ladders; no group statistics"
        (OUTDIR / "rnn_escape_simulation.json").write_text(
            json.dumps(_clean(res), indent=1), encoding="utf-8")
        raise SystemExit("too few complete ladders")

    G = np.asarray(GAINS, float)
    keys = sorted(full)

    def series(fn):
        out = []
        for kk in keys:
            row = []
            for g in GAINS:
                try:
                    v = fn(full[kk][g])
                except (TypeError, KeyError):
                    v = None
                row.append(np.nan if v is None else float(v))
            out.append(row)
        return out

    def mfpt_mean(c, rn):
        a = c["escape"]["lo_to_hi"]["rules"][rn]["mfpt"]
        b = c["escape"]["hi_to_lo"]["rules"][rn]["mfpt"]
        return 0.5 * (a + b) if (a and b) else None

    measures = {
        "saddle_unstable_eigenvalue": series(lambda c: c["saddle_unstable_eig"]),
        "well_slowest_relaxation_rate_lo": series(lambda c: c["wells"]["lo"]["Jc_max_re"]),
        "well_slowest_relaxation_rate_hi": series(lambda c: c["wells"]["hi"]["Jc_max_re"]),
        "well_slowest_relaxation_rate_mean": series(
            lambda c: 0.5 * (c["wells"]["lo"]["Jc_max_re"] + c["wells"]["hi"]["Jc_max_re"])),
        "well_fastest_relaxation_rate_mean": series(
            lambda c: 0.5 * (c["wells"]["lo"]["Jc_min_re"] + c["wells"]["hi"]["Jc_min_re"])),
        "fw_action_lo_to_saddle": series(lambda c: c["fw_action"]["A_to_saddle"]),
        "fw_action_hi_to_saddle": series(lambda c: c["fw_action"]["B_to_saddle"]),
        "fw_action_mean": series(lambda c: 0.5 * (c["fw_action"]["A_to_saddle"]
                                                  + c["fw_action"]["B_to_saddle"])),
        "fw_action_lo_to_hi": series(lambda c: c["fw_action"]["A_to_B"]),
        "fw_action_hi_to_lo": series(lambda c: c["fw_action"]["B_to_A"]),
        "critical_displacement_lo": series(
            lambda c: c["perturbation"]["lo"]["critical_displacement"]),
        "critical_displacement_hi": series(
            lambda c: c["perturbation"]["hi"]["critical_displacement"]),
        "critical_displacement_mean": series(
            lambda c: 0.5 * (c["perturbation"]["lo"]["critical_displacement"]
                             + c["perturbation"]["hi"]["critical_displacement"])),
        "recovery_1e_fixed_mean": series(
            lambda c: 0.5 * (c["perturbation"]["lo"]["t_recover_1e_fixed_samples"]
                             + c["perturbation"]["hi"]["t_recover_1e_fixed_samples"])),
        "recovery_1e_halfcrit_mean": series(
            lambda c: 0.5 * (c["perturbation"]["lo"]["t_recover_1e_halfcrit_samples"]
                             + c["perturbation"]["hi"]["t_recover_1e_halfcrit_samples"])),
        "fp_separation": series(lambda c: c["fp_separation"]),
        "barrier_generative_nats_DENSITY": series(
            lambda c: c["density_read"]["barrier_generative_nats"]),
        "mean_dwell_midpoint_DENSITY_RUN": series(
            lambda c: c["density_read"]["mean_dwell_long"]),
    }
    rule_names = [r for r in cells[min(ok_idx)]["escape"]["lo_to_hi"]["rules"]]
    for rn in rule_names:
        measures[f"mfpt[{rn}]"] = series(lambda c, rn=rn: mfpt_mean(c, rn))
        measures[f"log_mfpt[{rn}]"] = series(
            lambda c, rn=rn: (math.log(mfpt_mean(c, rn))
                              if mfpt_mean(c, rn) else None))
        measures[f"escape_rate[{rn}]"] = series(
            lambda c, rn=rn: (1.0 / (mfpt_mean(c, rn) * ALPHA)
                              if mfpt_mean(c, rn) else None))
        # SECONDARY, outside the declared analysis: Kramers says
        # MFPT ~ (1/lambda_u) * C * exp(S), so log MFPT + log lambda_u removes the
        # part of the prefactor that the gain manipulation itself moves.  The
        # self-test's second ladder shows how large that correction can be.
        measures[f"log_mfpt_prefactor_corrected[{rn}]"] = series(
            lambda c, rn=rn: (math.log(mfpt_mean(c, rn))
                              + math.log(c["saddle_unstable_eig"])
                              if (mfpt_mean(c, rn) and c["saddle_unstable_eig"] > 0)
                              else None))

    # The declared decision rule is about the four gain levels B0/B1/B2/B3.  Gain 0.86
    # is the operating point of the other network analyses, between B1 and B2;
    # including it is a five-point secondary.  The four-level test is the declared one.
    LEVELS4 = (0.76, 0.84, 0.90, 1.00)
    i4 = [GAINS.index(g) for g in LEVELS4]

    def sub4(rows):
        return [[r[i] for i in i4] for r in rows]

    _log("\n[ORDERING, declared: the four gain levels 0.76/0.84/0.90/1.00] "
         "mean within-seed Spearman rho with gain, against a within-seed gain-label "
         "permutation null (10,000 draws)", sink)
    ordering4 = {}
    for name, vals in measures.items():
        r = spearman_perm(sub4(vals), np.asarray(LEVELS4, float))
        ordering4[name] = r
        if r:
            _log(f"  {name:42s} rho {r['mean_spearman']:+.3f} "
                 f"[{r['ci_lo']:+.3f},{r['ci_hi']:+.3f}]  P_perm {r['p_perm']:.4f}  "
                 f"monotone {r['n_seeds_strictly_monotone']}/{r['n_seeds']}", sink)

    _log("\n[ORDERING, secondary: all five gains including the 0.86 anchor]", sink)
    ordering = {}
    for name, vals in measures.items():
        r = spearman_perm(vals, G)
        ordering[name] = r
        if r:
            _log(f"  {name:42s} rho {r['mean_spearman']:+.3f} "
                 f"[{r['ci_lo']:+.3f},{r['ci_hi']:+.3f}]  P_perm {r['p_perm']:.4f}  "
                 f"monotone {r['n_seeds_strictly_monotone']}/{r['n_seeds']}", sink)
    res["group"] = dict(gains=list(GAINS), gain_levels_declared=list(LEVELS4),
                        seeds=[full[k][GAINS[0]]["anchor_seed"] for k in keys],
                        ordering_four_gain_levels=ordering4,
                        ordering=ordering, per_seed_series=measures)

    # ---------------- exclusion bounds
    _log("\n[EXCLUSION BOUNDS] paired change B0 (gain 0.76) -> B3 (gain 1.00), "
         "seed-level bootstrap", sink)
    i0b, i3b = GAINS.index(0.76), GAINS.index(1.00)
    bounds = {}
    for name, vals in measures.items():
        b = exclusion_bound(vals, i0b, i3b)
        bounds[name] = b
        if b and not name.startswith(("mfpt[", "escape_rate[", "log_mfpt")):
            fold = (f"  fold {b['fold_change']['mean']:.3f} "
                    f"[{b['fold_change']['ci_lo']:.3f},{b['fold_change']['ci_hi']:.3f}]"
                    if "fold_change" in b else "")
            _log(f"  {name:42s} {b['mean_from']:+.4f} -> {b['mean_to']:+.4f}  "
                 f"change {b['mean_change']:+.4f} "
                 f"[{b['ci_lo']:+.4f},{b['ci_hi']:+.4f}]{fold}", sink)
    res["group"]["exclusion_bounds_B0_to_B3"] = bounds

    # ---------------- the deciding test
    _log("\n[ARRHENIUS] slope of log(MFPT) on each candidate barrier "
         "(predicted 1 for the true dynamical barrier)", sink)
    arr = {}
    cand = {"fw_action_mean": measures["fw_action_mean"],
            "fw_action_lo_to_saddle": measures["fw_action_lo_to_saddle"],
            "barrier_generative_nats_DENSITY":
                measures["barrier_generative_nats_DENSITY"],
            "fp_separation": measures["fp_separation"]}
    # the same regression on the four declared gain levels only, for the two key
    # candidates
    for rn in rule_names:
        y4 = sub4(measures[f"log_mfpt[{rn}]"])
        for cn in ("fw_action_mean", "barrier_generative_nats_DENSITY"):
            s = slope_group(sub4(measures[cn]), y4, h0=1.0)
            if s:
                s["pre_specified"] = True
                s["gain_levels"] = list(LEVELS4)
                arr[f"[4 gain levels] {cn} -> log_mfpt[{rn}]"] = s
    for rn in rule_names:
        _log(f"  rule {rn}:", sink)
        for ylab in (f"log_mfpt[{rn}]", f"log_mfpt_prefactor_corrected[{rn}]"):
            y = measures[ylab]
            for cn, x in cand.items():
                s = slope_group(x, y, h0=1.0)
                if s is None:
                    continue
                s["pre_specified"] = ylab.startswith("log_mfpt[")
                arr[f"{cn} -> {ylab}"] = s
                if not s["pre_specified"] and cn not in (
                        "fw_action_mean", "barrier_generative_nats_DENSITY"):
                    continue
                tag = "" if s["pre_specified"] else " [prefactor-corrected, secondary]"
                _log(f"     {cn:34s} slope {s['mean_slope']:+.4f} "
                     f"[{s['ci_lo_boot']:+.4f},{s['ci_hi_boot']:+.4f}] "
                     f"(n={s['n_seeds']} seeds, R2 {s['mean_r2']:.3f}, "
                     f"P vs 1 = {s['p_vs_h0']:.2e}){tag}", sink)
    res["group"]["arrhenius"] = arr

    # ---------------- over-prediction of the change in escape time
    _log("\n[OVER-PREDICTION] predicted / measured fold change in escape time, "
         "B3 (gain 1.00) vs B0 (gain 0.76)", sink)
    i0, i3 = GAINS.index(0.76), GAINS.index(1.00)
    bd = np.asarray(measures["barrier_generative_nats_DENSITY"], float)
    fw = np.asarray(measures["fw_action_mean"], float)
    # CENSORING: networks whose ladder produces no escape at one or both ends inside
    # the 80,000-sample horizon have a censored MFPT there, so a median over all
    # complete ladders would be undefined.  The measured fold change is taken over
    # the networks in which BOTH ends resolve, with that number printed; the
    # unresolved networks bound the fold from below (censored, not zero).  The
    # deciding test (the Arrhenius slopes above) does not depend on this.
    over = {}
    for rn in rule_names:
        mm = np.asarray(measures[f"mfpt[{rn}]"], float)
        ok = np.isfinite(mm[:, i0]) & np.isfinite(mm[:, i3])
        meas = (mm[:, i3] / mm[:, i0])[ok]
        pd_ = np.exp(bd[:, i3] - bd[:, i0])[ok]
        pf_ = np.exp(fw[:, i3] - fw[:, i0])[ok]
        pd_all = np.exp(bd[:, i3] - bd[:, i0])
        pf_all = np.exp(fw[:, i3] - fw[:, i0])
        if meas.size == 0:
            over[rn] = dict(n_seeds_resolved=0, n_seeds_total=int(mm.shape[0]))
            continue
        over[rn] = dict(
            n_seeds_resolved=int(ok.sum()), n_seeds_total=int(mm.shape[0]),
            seeds_unresolved_at_one_or_both_ends=int((~ok).sum()),
            measured_fold=dict(median=float(np.median(meas)), lo=float(meas.min()),
                               hi=float(meas.max()), per_seed=meas.tolist()),
            fw_predicted_fold=dict(median=float(np.median(pf_)), lo=float(pf_.min()),
                                   hi=float(pf_.max()),
                                   median_all_seeds=float(np.median(pf_all))),
            density_predicted_fold=dict(median=float(np.median(pd_)),
                                        lo=float(pd_.min()), hi=float(pd_.max()),
                                        median_all_seeds=float(np.median(pd_all))),
            fw_over_prediction=dict(median=float(np.median(pf_ / meas)),
                                    lo=float((pf_ / meas).min()),
                                    hi=float((pf_ / meas).max()),
                                    per_seed=(pf_ / meas).tolist()),
            density_over_prediction=dict(median=float(np.median(pd_ / meas)),
                                         lo=float((pd_ / meas).min()),
                                         hi=float((pd_ / meas).max()),
                                         per_seed=(pd_ / meas).tolist()),
            note="fold changes are over the seeds in which the escape time "
                 "resolves at BOTH ends of the ladder; the rest are censored at "
                 "the 80,000-sample horizon and bound the fold from below, they "
                 "are not zero")
        _log(f"  {rn:30s} measured {np.median(meas):7.2f}x   FW predicts "
             f"{np.median(pf_):8.2f}x (x{np.median(pf_/meas):6.2f})   density predicts "
             f"{np.median(pd_):11.1f}x (x{np.median(pd_/meas):9.1f})   "
             f"[{int(ok.sum())}/{mm.shape[0]} seeds resolved at both ends]", sink)
    res["group"]["over_prediction"] = over

    # censoring ladder: what the 80,000-sample horizon can and cannot resolve
    cens = {}
    for rn in rule_names:
        rows_c, rows_b = [], []
        for c in cells:
            if not (c.get("gb1_pass") and c.get("saddle_found")):
                continue
            for d in ("lo_to_hi", "hi_to_lo"):
                st = c["escape"][d]["rules"][rn]
                rows_c.append(st["censored_frac"])
                rows_b.append(0.5 * (c["fw_action"]["A_to_saddle"]
                                     + c["fw_action"]["B_to_saddle"]))
        rows_c, rows_b = np.asarray(rows_c), np.asarray(rows_b)
        full = rows_b[rows_c >= 1.0]
        part = rows_b[rows_c < 1.0]
        cens[rn] = dict(
            n_cell_directions=int(len(rows_c)),
            frac_fully_censored=float(np.mean(rows_c >= 1.0)),
            mean_censored_frac=float(rows_c.mean()),
            max_fw_action_with_any_escape=(float(part.max()) if part.size else None),
            min_fw_action_with_no_escape=(float(full.min()) if full.size else None),
            statement="the horizon resolves an escape wherever the FW action is "
                      "below about the first number and never above the second; "
                      "an unresolved cell bounds its MFPT from below by the total "
                      "time at risk, it does not measure it as infinite")
    res["group"]["escape_observability_ceiling"] = cens
    _log("\n[OBSERVABILITY CEILING of the 80,000-sample escape horizon]", sink)
    for rn, v in cens.items():
        _log(f"  {rn:30s} fully censored {v['frac_fully_censored']:.3f} of "
             f"{v['n_cell_directions']} cell-directions; largest FW action that "
             f"still escaped {v['max_fw_action_with_any_escape']}; smallest that "
             f"never did {v['min_fw_action_with_no_escape']}", sink)

    # ---------------- the decision
    indep = {"saddle_unstable_eigenvalue": +1, "fw_action_mean": +1,
             "critical_displacement_mean": +1, f"log_mfpt[{PRIMARY_RULE}]": +1}
    fails = []
    for nm, sgn in indep.items():
        r = ordering4.get(nm)
        if r is None or r["p_perm"] >= 0.05 or np.sign(r["mean_spearman"]) != sgn:
            fails.append(nm)
    strict = {nm: (ordering4[nm]["n_seeds_strictly_monotone"],
                   ordering4[nm]["n_seeds"])
              for nm in indep if ordering4.get(nm)}
    strict5 = {nm: (ordering[nm]["n_seeds_strictly_monotone"], ordering[nm]["n_seeds"])
               for nm in indep if ordering.get(nm)}
    prim = arr.get(f"fw_action_mean -> log_mfpt[{PRIMARY_RULE}]")
    dens = arr.get(f"barrier_generative_nats_DENSITY -> log_mfpt[{PRIMARY_RULE}]")
    wells = ordering4.get("well_slowest_relaxation_rate_mean")
    res["group"]["decision"] = dict(
        independent_measures=list(indep), failures=fails,
        criterion="group-level ordering of the FOUR declared gain levels against the "
                  "within-seed gain-label permutation null (P < 0.05) with the "
                  "predicted sign",
        jacobian_split=dict(
            note="The declared estimand is 'Jacobian eigenvalues at each fixed "
                 "point'.  The two halves answer different questions and are "
                 "scored separately.  The SADDLE's unstable eigenvalue is the "
                 "barrier-side quantity and is the one entered into the decision; "
                 "the WELLS' slowest relaxation rate is the local-stiffness "
                 "quantity, and if it does NOT order the gains that is the "
                 "finding, not a failure -- it says the manipulation moves the "
                 "barrier and not the stiffness of either well.",
            saddle=ordering.get("saddle_unstable_eigenvalue"),
            wells=wells,
            wells_order_the_gains=bool(wells and wells["p_perm"] < 0.05)),
        all_order_gains=bool(not fails),
        per_measure_seeds_strictly_monotone={k: list(v) for k, v in strict.items()},
        per_measure_seeds_strictly_monotone_all_five_gains={
            k: list(v) for k, v in strict5.items()},
        fw_slope=prim["mean_slope"] if prim else None,
        fw_slope_ci=[prim["ci_lo_boot"], prim["ci_hi_boot"]] if prim else None,
        fw_slope_includes_one=bool(prim and prim["ci_lo_boot"] <= 1.0 <= prim["ci_hi_boot"]),
        density_slope=dens["mean_slope"] if dens else None,
        density_slope_ci=[dens["ci_lo_boot"], dens["ci_hi_boot"]] if dens else None,
        density_slope_includes_one=bool(dens and dens["ci_lo_boot"] <= 1.0
                                        <= dens["ci_hi_boot"]))
    _log(f"\n[DECISION] ordering failures: {fails or 'none'}", sink)
    _log(f"  strictly monotone per measure: {strict}", sink)

    # ---------------- outputs
    res["runtime_s"] = float(time.time() - t0)
    res["wall_time_min"] = res["runtime_s"] / 60.0
    (OUTDIR / "rnn_escape_simulation.json").write_text(
        json.dumps(_clean(res), indent=1), encoding="utf-8")

    cols = ["anchor_seed", "seed_index", "gain", "gain_level", "gb1_pass", "saddle_found",
            "n_fixed_points", "n_stable", "n_unstable_points", "completeness",
            "fp_separation", "saddle_unstable_eig", "saddle_Jc_second_re",
            "saddle_n_unstable", "well_lo_Jc_max_re", "well_hi_Jc_max_re",
            "well_lo_Jc_min_re", "well_hi_Jc_min_re", "separatrix_cos_wells_axis",
            "fw_lo_to_saddle", "fw_hi_to_saddle", "fw_lo_to_hi", "fw_hi_to_lo",
            "fw_saddle_to_hi", "fw_saddle_to_lo", "fw_straight_line_lo_to_hi",
            "attractor_lib_gmam_lo_to_hi", "barrier_generative_nats",
            "mean_dwell_midpoint", "tau_within_dwell", "d_crit_lo", "d_crit_hi",
            "t_rec_1e_fixed_lo", "t_rec_1e_fixed_hi", "t_rec_1e_halfcrit_lo",
            "t_rec_1e_halfcrit_hi"]
    cols += [f"mfpt_{d}_{r}" for r in rule_names for d in ("lo2hi", "hi2lo")]
    cols += [f"cens_{d}_{r}" for r in rule_names for d in ("lo2hi", "hi2lo")]
    with open(OUTDIR / "rnn_escape_simulation.csv", "w", newline="", encoding="utf-8") as fh:
        wtr = csv.writer(fh, lineterminator="\n")
        wtr.writerow(cols)
        for c in cells:
            row = {"anchor_seed": c["anchor_seed"], "seed_index": c["seed_index"],
                   "gain": c["gain"], "gain_level": c["gain_level"], "gb1_pass": c["gb1_pass"],
                   "saddle_found": c["saddle_found"],
                   "n_fixed_points": c["n_fixed_points"], "n_stable": c["n_stable"],
                   "n_unstable_points": c["n_unstable_points"],
                   "completeness": c["completeness"]["frac_accounted"]}
            if c["gb1_pass"] and c["saddle_found"]:
                p = c["perturbation"]
                fwd = c["fw_action"]
                row.update({
                    "fp_separation": c["fp_separation"],
                    "saddle_unstable_eig": c["saddle_unstable_eig"],
                    "saddle_Jc_second_re": c["saddle"]["Jc_second_re"],
                    "saddle_n_unstable": c["saddle_n_unstable"],
                    "well_lo_Jc_max_re": c["wells"]["lo"]["Jc_max_re"],
                    "well_hi_Jc_max_re": c["wells"]["hi"]["Jc_max_re"],
                    "well_lo_Jc_min_re": c["wells"]["lo"]["Jc_min_re"],
                    "well_hi_Jc_min_re": c["wells"]["hi"]["Jc_min_re"],
                    "separatrix_cos_wells_axis":
                        c["separatrix_normal_vs_wells_axis_cos"],
                    "fw_lo_to_saddle": fwd["A_to_saddle"],
                    "fw_hi_to_saddle": fwd["B_to_saddle"],
                    "fw_lo_to_hi": fwd["A_to_B"], "fw_hi_to_lo": fwd["B_to_A"],
                    "fw_saddle_to_hi": fwd["saddle_to_B"],
                    "fw_saddle_to_lo": fwd["saddle_to_A"],
                    "fw_straight_line_lo_to_hi": fwd["straight_line_A_to_B"],
                    "attractor_lib_gmam_lo_to_hi":
                        fwd.get("attractor_lib_gmam_A_to_B"),
                    "barrier_generative_nats":
                        c["density_read"]["barrier_generative_nats"],
                    "mean_dwell_midpoint": c["density_read"]["mean_dwell_long"],
                    "tau_within_dwell": c["density_read"]["tau_within_dwell"],
                    "d_crit_lo": p["lo"]["critical_displacement"],
                    "d_crit_hi": p["hi"]["critical_displacement"],
                    "t_rec_1e_fixed_lo": p["lo"]["t_recover_1e_fixed_samples"],
                    "t_rec_1e_fixed_hi": p["hi"]["t_recover_1e_fixed_samples"],
                    "t_rec_1e_halfcrit_lo": p["lo"]["t_recover_1e_halfcrit_samples"],
                    "t_rec_1e_halfcrit_hi": p["hi"]["t_recover_1e_halfcrit_samples"]})
                for r in rule_names:
                    row[f"mfpt_lo2hi_{r}"] = c["escape"]["lo_to_hi"]["rules"][r]["mfpt"]
                    row[f"mfpt_hi2lo_{r}"] = c["escape"]["hi_to_lo"]["rules"][r]["mfpt"]
                    row[f"cens_lo2hi_{r}"] = \
                        c["escape"]["lo_to_hi"]["rules"][r]["censored_frac"]
                    row[f"cens_hi2lo_{r}"] = \
                        c["escape"]["hi_to_lo"]["rules"][r]["censored_frac"]
            wtr.writerow([row.get(kk) for kk in cols])

    _log(f"\nwall time {res['runtime_s']/60:.1f} min", sink)
    (OUTDIR / "escape_ladder.log").write_text("\n".join(sink), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
