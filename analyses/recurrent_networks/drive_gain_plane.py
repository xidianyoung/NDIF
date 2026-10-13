"""The (Delta log lambda, Delta log D) plane in the trained 64-unit networks: drive and gain manipulations pushed
through the rate and diffusion estimators at each modality's own sampling.

Near its operating point the population state obeys dx = A (x - x*) dt + Sigma^(1/2) dW, D = Sigma/2,
x* = -A^{-1}(B u + c).  An additive structured INPUT (drive) translates x*, raises D anisotropically in the same
subspace and leaves A untouched; a recurrent GAIN W -> gamma W changes the relaxation rates, leaves D alone and
contracts the stationary spread.  In the plane (Delta log lambda, Delta log D) drive is vertical and gain is
horizontal.  Each manipulation is implemented in the circuit, estimated at each modality's sampling, and placed
in the plane beside its true value; a change in the NUMBER of stable states is placed there too.

THE FOUR MANIPULATIONS (three levels m = 0,1,2; a size multiplier k scales the per-level step)
  drive   additive structured drive on the drive channel, mean and fluctuation in one input:
                u_t = c(m) e_0 + q(m) zeta_t e_0 ,   zeta_t ~ N(0,1) white,
          so the extra diffusion is rank one along v = diag(1 - tanh^2(pre*)) W_in[:,0]:
                D(q) = sigma^2/(2 alpha) I + (alpha q^2 / 2) v v^T.
          c(m) sets the displacement to 0.064 k m s.d. (ds003838 held-out displacement per item) and q(m) sets
          log(e' D e) to rise by 0.109 k m (COG-BCI log diffusion per load level).  A changes only through the
          tanh curvature at x*(c); that leak is measured.
  gain    recurrent gain W -> gamma W with b unchanged; D exactly unchanged; gamma(m) set so log lambda_e rises
          by 0.109 k m, the same per-level magnitude as the drive step.  In a tanh network this also moves x*.
  gain_c  the same gain change with b -> arctanh(x*) - gamma W x*, which pins x* exactly (gain only).
  both    drive and gain together.
  count   a compensated-gain ladder across the network's own saddle-node: the enumerated number of stable fixed
          points goes 1 -> 1 -> 2 with the occupied well pinned; levels verified by exhaustive enumeration.

DERIVATION OF THE INPUT-DRIVEN DIFFUSION
    r_{t+1} = (1-a) r_t + a tanh(W r_t + W_in u_t + b) + sigma xi_t, a = ALPHA, dt = a.  With
    u_t = c e_0 + q zeta_t e_0 the update gains, to first order, a q zeta v; its per-step covariance a^2 q^2 v v^T
    divided by 2 dt = 2a is (a q^2 / 2) v v^T, and the intrinsic noise contributes D0 I.  Hence
    e' D(q) e = D0 + (a q^2 / 2) (v . e)^2, which inverts in closed form for a target log diffusion ratio.  The
    full nonlinearity is in the simulation; only the target-setting uses the first-order expression.

THREE TIERS OF TRUTH
    generator    lambda_e = -(e' J Sigma e)/(e' Sigma e) and D_e = e' D e (e' Sigma e = D_e / lambda_e exactly)
    sampled-lag  what an infinite-sample AR(1) at the modality's sampling returns: lag tau = stride * ALPHA,
                 rho = e' expm(J tau) Sigma e / (e' Sigma e), lambda = -ln rho / dt,
                 D = (e' Sigma e)(1 - rho^2)/dt; for fMRI also through the haemodynamic response
    measured     the estimators below on simulated recordings

ESTIMATORS
    lambda = -ln(a_hat)/dt with a_hat the pooled per-segment-centred lag-1 slope, D = var(innovations)/dt (the
    COG-BCI rate estimator); for 2-window segments (HCP MEG pair design) the pooled independent-pair map with
    grand centring.  Axis rules: heldout (leave-one-simulated-participant-out level-contrast direction in the
    pooled state space), fixed (the true load axis e_load in the raw 64-unit space), pc1 (first principal
    component of the pooled state space).

MODALITY SAMPLING
    EEG   d=8 dt=1.00 s  6 segments x 49 windows, base rate 1.3123 s^-1 (COG-BCI, 28 participants)
    MEG   d=8 dt=1.00 s  150 segments x 2 windows, 0.7350 s^-1 (HCP MEG pair design)
    fMRI  d=7 dt=0.72 s  4 blocks x 38 volumes, SPM canonical HRF, 0.5185 s^-1
    SLEEP d=4 dt=2.00 s  8 bouts x 30 windows, 0.4628 s^-1
    One second of recording is tau network time units, with tau solved on the base condition so that the base
    network's exact 1-D autocorrelation along e_load equals exp(-lambda_base dt).  The sleep row carries its
    own identifiability check (recovered fraction of a known knob; departure of r(1..5) from one exponential).

Inputs   trained weights (NDIF_RNN_WEIGHTS), device NDIF_DEVICE (default cuda if available).
Outputs  drive_gain_rows/rows_<seed>.json in the network cache paths.cache("rnn") (NDIF_RNN_CACHE), read by
         drive_gain_report.py.
Reported in  Fig. 3e (through drive_gain_report.py).
Runtime  about 12.5 minutes for the eight networks on one CUDA GPU (stored run: torch 2.9.0+cu130).

    python analyses/recurrent_networks/drive_gain_plane.py --reps 8
"""
from __future__ import annotations


import argparse
import json
import math
import platform
import sys
import time
import zlib
from pathlib import Path

import numpy as np
from scipy.linalg import expm, solve_continuous_lyapunov

HERE = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

import torch  # noqa: E402
import network_lib as NL  # noqa: E402
from analyses.common import paths  # noqa: E402

ROWS = Path(paths.cache("rnn")) / "drive_gain_rows"
ROWS.mkdir(parents=True, exist_ok=True)

SEED = 20260922
ALPHA = NL.ALPHA
SIGMA0 = NL.SIGMA0
D0 = NL.D0
DRIVE_CH = NL.DRIVE_CH
ANCHORS = NL.ANCHORS
DEV = NL.DEV

# ------------------------------------------------------------------ declared design
LEVELS = 3                                   # m = 0,1,2 -- COG-BCI's own three load levels
SIZES = (0.5, 1.0, 2.0, 4.0)                 # multiples of the size cortex shows
TRUTHS = ("null", "drive", "gain", "gain_c", "both")
NU_GRID = (0.0, 1.0)                         # observation noise, x per-unit state s.d.
N_REPS = 8                                   # simulated participants per cell
BURN = 400

# the two cortical effect sizes that set the k = 1 rung, both per level
POS_STEP_SD = 0.064      # ds003838 held-out displacement per expected item, s.d. units
DLOGD_STEP = 0.109       # COG-BCI d log D per load level (28 participants, +0.1089)
DLOGL_STEP = 0.109       # matched magnitude on the gain axis, so the knobs are equal in length

SOURCES = dict(
    POS_STEP_SD="ds003838 held-out displacement +0.064 s.d. per expected item held",
    DLOGD_STEP="COG-BCI workload analysis: mean d log D per load level = +0.1089 (n = 28 participants after "
               "the duplicate exclusion)",
    DLOGL_STEP="matched to DLOGD_STEP by design so the two knobs have equal length in the plane; "
               "the k ladder 0.5-4 brackets the MEG load rung (+0.173) and the sleep Wake>N1 "
               "rung (+0.195)",
    EEG_lam_base="COG-BCI workload analysis: median per-cell lambda = 1.3123 s^-1 (dt = 1 s; n = 28)",
    EEG_nseg="COG-BCI workload analysis: median 294 non-overlapping 1-s windows per cell (6 x 49 = 294)",
    MEG_lam_base="HCP MEG per-load analysis: median lag-1 autocorrelation at 0-back 0.4796 -> 0.7350 s^-1",
    fMRI_lam_base="HCP fMRI input-dependent analysis: median bias-corrected lambda_max at 0-back = 0.5185 s^-1",
    SLEEP_lam_base="Sleep-EDF stage decomposition: Wake one-step median lambda = 0.4628 s^-1")

MODALITIES = {
    "EEG":   dict(d=8, dt=1.0,  n_seg=6,   seg_len=49, lam_base=1.3123, hrf=False),
    "MEG":   dict(d=8, dt=1.0,  n_seg=150, seg_len=2,  lam_base=0.7350, hrf=False),
    "fMRI":  dict(d=7, dt=0.72, n_seg=4,   seg_len=38, lam_base=0.5185, hrf=True),
    "SLEEP": dict(d=4, dt=2.0,  n_seg=8,   seg_len=30, lam_base=0.4628, hrf=False),
}

LV = np.array([-1.0, 0.0, 1.0])              # centred level index; slope = (y2 - y0)/2


def log_factory():
    def log(m):
        print(time.strftime("[%H:%M:%S] ") + str(m), flush=True)
    return log


_clean = NL._clean


# ==================================================================== exact truth
def exact_cell_D(W, Win, b, u_mean, x_guess, e, sigma, q=0.0):
    """x*, J, Sigma and the 1-D truths for a configuration with a fluctuating input of s.d. q."""
    x, res = NL.newton_fp(x_guess, W, Win, b, u_mean)
    if x is None:
        return None
    J = NL.jac_c(x, W, Win, b, u_mean)
    ev = np.linalg.eigvals(J)
    if np.max(ev.real) >= 0:
        return None
    pre = W @ x + Win @ u_mean + b
    g = 1.0 - np.tanh(pre) ** 2
    v = g * Win[:, DRIVE_CH]
    n = len(x)
    Dm = D0 * np.eye(n) + (0.5 * ALPHA * q * q) * np.outer(v, v)
    S = solve_continuous_lyapunov(J, -2.0 * Dm)
    S = 0.5 * (S + S.T)
    se = float(e @ S @ e)
    De = float(e @ Dm @ e)
    lam_e = float(-(e @ J @ S @ e) / se) if se > 0 else np.nan
    return dict(x=x, J=J, S=S, Dm=Dm, v_input=v, ev=ev,
                lam_slow=float(-np.max(ev.real)), lam_mean=float(-ev.real.mean()),
                sd_load=float(np.sqrt(max(se, 0.0))),
                spread_rms=float(np.sqrt(np.trace(S) / n)),
                var_e=se, D_e=De, lam_e=lam_e,
                ratio_identity=float(se * lam_e / De) if De > 0 else np.nan)


def sampled_lag_truth(cr, e, tau_rnn, dt_s, hrf=None):
    """What an infinite-sample AR(1) at THIS modality's sampling would return along e.

    The lag must be given in RNN TIME UNITS (tau_rnn = stride * ALPHA), not in seconds: one
    recorded sample is tau_rnn units of the circuit's own time.  A first version passed the
    modality's dt in seconds into expm(J dt), which at the EEG calibration understated the lag
    three-fold and made the drive knob's sampled log-diffusion step read +0.012 where the
    generator step was +0.113.  Caught because the two disagreed by an order of magnitude on a
    knob whose construction fixes the generator value exactly.

    With an HRF the recorded series is y_k = sum_j h_j x_{k-j} at one-TR spacing, so
        Cy(l) = sum_{j,j'} h_j h_j' Cx(l + j' - j),   Cx(m) = e' expm(J |m| tau) Sigma e,
    which is what the fMRI arm's estimator actually sees.  Rates are then per second.
    """
    se = cr["var_e"]
    if hrf is None:
        c0, c1 = se, float(e @ expm(cr["J"] * tau_rnn) @ cr["S"] @ e)
    else:
        L = len(hrf)
        mmax = L + 2
        Cx = np.array([float(e @ expm(cr["J"] * (abs(m) * tau_rnn)) @ cr["S"] @ e)
                       for m in range(mmax + 1)])
        def cy(l):
            tot = 0.0
            for j in range(L):
                for jp in range(L):
                    tot += hrf[j] * hrf[jp] * Cx[abs(l + jp - j)]
            return float(tot)
        c0, c1 = cy(0), cy(1)
    if c0 <= 0:
        return dict(rho=np.nan, lam=np.nan, D=np.nan, var=c0)
    rho = c1 / c0
    if not (0.0 < rho < 1.0):
        return dict(rho=float(rho), lam=np.nan, D=np.nan, var=float(c0))
    return dict(rho=float(rho), lam=float(-math.log(rho) / dt_s),
                D=float(c0 * (1.0 - rho ** 2) / dt_s), var=float(c0))


# ==================================================================== configurations
def build_plane_configs(geo, log):
    """Every (truth, size, level): physical parameters plus the exact generator truth."""
    W, Win, b, x0, e = geo["W"], geo["Win"], geo["b"], geo["x0"], geo["e_load"]
    at = np.arctanh(np.clip(x0, -1 + 1e-14, 1 - 1e-14))
    base = exact_cell_D(W, Win, b, np.zeros(4), x0, e, SIGMA0, 0.0)

    # ---- drive: displacement in s.d. as a function of |c|, both signs
    cand = {}
    for sgn in (+1, -1):
        cs, ds, lam = [], [], []
        for mag in np.arange(0.0, 0.6001, 0.004):
            u = np.zeros(4); u[DRIVE_CH] = sgn * mag
            cr = exact_cell_D(W, Win, b, u, x0, e, SIGMA0, 0.0)
            if cr is None:
                break
            cs.append(mag); ds.append(float(np.linalg.norm(cr["x"] - x0) / cr["sd_load"]))
            lam.append(cr["lam_e"])
        cand[sgn] = (np.array(cs), np.array(ds), np.array(lam))

    def leak(sgn):
        cs, ds, lam = cand[sgn]
        if len(ds) < 5:
            return -np.inf
        return float((math.log(lam[-1]) - math.log(lam[0])) / max(ds[-1] - ds[0], 1e-9))

    # declared sign convention, as in the two-knob construction: the direction in which the 1-D rate
    # RISES with displacement, because the measured cortical load contrast speeds the return
    sgn_load = +1 if leak(+1) >= leak(-1) else -1
    cs, ds, _ = cand[sgn_load]
    leak_both = dict(plus=leak(+1), minus=leak(-1), chosen=int(sgn_load))

    # ---- gain (uncompensated) and gain_c (compensated).  The target is set on lambda_e, the
    #      1-D generator rate along e_load, because THAT is the plane's own lambda coordinate;
    #      a first version targeted lambda_slow and the realised lambda_e step came out 2.6x the
    #      nominal one, which would have decided the separation question by effect size.
    gs_u, gl_u = [], []
    xprev = x0.copy()
    for gam in np.arange(1.0, 0.1999, -0.002):
        cr = exact_cell_D(gam * W, Win, b, np.zeros(4), xprev, e, SIGMA0, 0.0)
        if cr is None:
            continue
        xprev = cr["x"]
        gs_u.append(gam); gl_u.append(cr["lam_e"])
    gs_c, gl_c = [], []
    for gam in np.arange(1.0, 0.1999, -0.002):
        cr = exact_cell_D(gam * W, Win, at - gam * W @ x0, np.zeros(4), x0, e, SIGMA0, 0.0)
        if cr is None:
            continue
        gs_c.append(gam); gl_c.append(cr["lam_e"])
    gs_u, gl_u = np.array(gs_u), np.array(gl_u)
    gs_c, gl_c = np.array(gs_c), np.array(gl_c)
    lam0 = base["lam_e"]

    # ---- q for a target log diffusion ratio along e (closed form)
    ve = float(base["v_input"] @ e)
    cos_v_e = float(abs(ve) / max(np.linalg.norm(base["v_input"]), 1e-12))

    def q_for(dlogD):
        if dlogD <= 0:
            return 0.0
        num = 2.0 * D0 * (math.exp(dlogD) - 1.0)
        return float(math.sqrt(num / max(ALPHA * ve * ve, 1e-18)))

    cfgs = []
    for k in SIZES:
        for m in range(LEVELS):
            disp_t = POS_STEP_SD * m * k
            dlogD_t = DLOGD_STEP * m * k
            dlogl_t = DLOGL_STEP * m * k
            c = sgn_load * NL.invert(cs, ds, disp_t, 0.6) if m > 0 else 0.0
            q = q_for(dlogD_t)
            gam_u = NL.invert(gs_u, gl_u, lam0 * math.exp(dlogl_t)) if m > 0 else 1.0
            gam_c = NL.invert(gs_c, gl_c, lam0 * math.exp(dlogl_t)) if m > 0 else 1.0
            for truth in TRUTHS:
                if truth == "null":
                    Wc, bc, uc, qc = W, b, np.zeros(4), 0.0
                elif truth == "drive":
                    uc = np.zeros(4); uc[DRIVE_CH] = c
                    Wc, bc, qc = W, b, q
                elif truth == "gain":
                    Wc, bc, uc, qc = gam_u * W, b, np.zeros(4), 0.0
                elif truth == "gain_c":
                    Wc = gam_c * W; bc = at - Wc @ x0; uc, qc = np.zeros(4), 0.0
                else:                                             # both
                    Wc = gam_c * W; bc = at - Wc @ x0
                    uc = np.zeros(4); uc[DRIVE_CH] = c; qc = q
                cr = exact_cell_D(Wc, Win, bc, uc, x0, e, SIGMA0, qc)
                cfgs.append(dict(truth=truth, size=float(k), level=int(m), arm="plane",
                                 c=float(uc[DRIVE_CH]), q=float(qc),
                                 gamma=float(gam_u if truth == "gain" else
                                             (gam_c if truth in ("gain_c", "both") else 1.0)),
                                 W=Wc, b=bc, u=uc, q_in=float(qc), sigma=SIGMA0, _cell=cr,
                                 n_stable=None))
    meta = dict(sgn_load=int(sgn_load), leak=leak_both, lam0_e=lam0, lam0_slow=base["lam_slow"],
                cos_v_input_e_load=cos_v_e, v_dot_e=ve,
                disp_max_available=float(ds[-1]) if len(ds) else None,
                gamma_u_range=[float(gs_u.min()), float(gs_u.max())],
                gamma_c_range=[float(gs_c.min()), float(gs_c.max())])
    return cfgs, base, meta


def build_count_configs(geo, base, log):
    """THE THIRD POSSIBILITY: a compensated-gain ladder that crosses the network's saddle-node.

    The occupied well is pinned exactly (b -> arctanh(x*) - gamma W x*), so nothing translates;
    what changes is that a SECOND stable fixed point appears, which requires an eigenvalue of the
    drift to cross zero at the emerging pair.  Level membership is established by exhaustive
    enumeration, not by the equilibrium counter.
    """
    W, Win, x0, e = geo["W_base_gain"], geo["Win"], geo["x0"], geo["e_load"]
    at = np.arctanh(np.clip(x0, -1 + 1e-14, 1 - 1e-14))
    g_mono = geo["opinfo"]["gamma_mono"]
    g_first = geo["opinfo"]["gamma_first_monostable"]
    gammas = [g_mono, g_first, round(g_first + 0.01, 4)]
    cfgs = []
    for m, gam in enumerate(gammas):
        Wc = gam * W
        bc = at - Wc @ x0
        net = NL._as_net(int(geo["seed"]), Wc, Win, bc)
        fps, _, _ = NL.fixed_points_hardened(net, np.zeros(4, np.float32), n_starts=400,
                                             seed=NL.ENUM_SEED)
        st = [np.asarray(f["point"], float) for f in fps if f["stable"]]
        ns = len(st)
        # the eigenvalue that must cross zero: the least stable real part over ALL fixed points
        worst = -np.inf
        for f in fps:
            Jf = NL.jac_c(np.asarray(f["point"], float), Wc, Win, bc, np.zeros(4))
            worst = max(worst, float(np.max(np.linalg.eigvals(Jf).real)))
        cr = exact_cell_D(Wc, Win, bc, np.zeros(4), x0, e, SIGMA0, 0.0)
        other = None
        if ns >= 2:
            dd = [float(np.linalg.norm(s - x0)) for s in st]
            other = st[int(np.argmax(dd))]
        cfgs.append(dict(truth="count", size=1.0, level=int(m), arm="count",
                         c=0.0, q=0.0, gamma=float(gam), W=Wc, b=bc, u=np.zeros(4),
                         q_in=0.0, sigma=SIGMA0, _cell=cr, n_stable=int(ns),
                         n_fixed=int(len(fps)), max_re_eig_any_fp=float(worst),
                         other_well=other,
                         sep_from_other=(float(np.linalg.norm(other - x0))
                                         if other is not None else None)))
        if log:
            log(f"    count level {m}: gamma {gam}  n_stable {ns}  n_fixed {len(fps)}  "
                f"max Re eig over fps {worst:+.4f}")
    return cfgs


# ==================================================================== simulation
def simulate_plane(cfgs, n_chains, n_keep, stride, burn, seed, dev=DEV):
    """(n_cfg, n_chains, n_keep, 64).  Each chain is an independent segment started at its own
    configuration's stable fixed point.  The fluctuating input enters INSIDE the tanh, on the
    drive channel, exactly as a real input would."""
    C = len(cfgs)
    n = cfgs[0]["W"].shape[0]
    Wt = torch.tensor(np.stack([c["W"] for c in cfgs]), dtype=torch.float32,
                      device=dev).transpose(1, 2).contiguous()
    beff = torch.tensor(np.stack([c["b"] + c["Win"] @ c["u"] for c in cfgs]),
                        dtype=torch.float32, device=dev)[:, None, :]
    win0 = torch.tensor(np.stack([c["Win"][:, DRIVE_CH] for c in cfgs]),
                        dtype=torch.float32, device=dev)[:, None, :]
    qin = torch.tensor(np.array([c["q_in"] for c in cfgs]), dtype=torch.float32,
                       device=dev)[:, None, None]
    sig = torch.tensor(np.array([c["sigma"] for c in cfgs]), dtype=torch.float32,
                       device=dev)[:, None, None]
    x0 = torch.tensor(np.stack([c["_cell"]["x"] for c in cfgs]), dtype=torch.float32,
                      device=dev)[:, None, :]
    any_q = bool(np.any([c["q_in"] > 0 for c in cfgs]))
    # escape diagnostic: distance past the midpoint toward a second well, where one exists
    has_other = np.array([c.get("other_well") is not None for c in cfgs])
    if has_other.any():
        axv = np.stack([(np.asarray(c["other_well"], float) - c["_cell"]["x"])
                        / max(np.linalg.norm(np.asarray(c["other_well"], float)
                                             - c["_cell"]["x"]), 1e-12)
                        if c.get("other_well") is not None else np.zeros(n) for c in cfgs])
        midv = np.stack([0.5 * (np.asarray(c["other_well"], float) + c["_cell"]["x"])
                         if c.get("other_well") is not None else np.zeros(n) for c in cfgs])
        axt = torch.tensor(axv, dtype=torch.float32, device=dev)[:, None, :]
        midt = torch.tensor(midv, dtype=torch.float32, device=dev)[:, None, :]
        hast = torch.tensor(has_other, device=dev)[:, None]
    g = torch.Generator(device=dev).manual_seed(int(seed))
    r = x0.expand(C, n_chains, n).contiguous().clone()
    esc = torch.zeros(C, n_chains, dtype=torch.bool, device=dev)

    def step(r):
        pre = torch.bmm(r, Wt) + beff
        if any_q:
            pre = pre + qin * win0 * torch.randn((C, n_chains, 1), device=dev, generator=g)
        return (1 - ALPHA) * r + ALPHA * torch.tanh(pre) \
            + sig * torch.randn(r.shape, device=dev, generator=g)

    def mark(rr):
        if has_other.any():
            esc.logical_or_(hast & (((rr - midt) * axt).sum(-1) > 0))

    with torch.no_grad():
        for i in range(burn):
            r = step(r)
            if (i % 25) == 0:
                mark(r)
        keep = torch.empty(C, n_chains, n_keep, n, device=dev)
        for k in range(n_keep):
            for _ in range(stride):
                r = step(r)
                mark(r)
            keep[:, :, k, :] = r
    return (keep.cpu().numpy().astype(np.float64),
            esc.cpu().numpy(),
            dict(escape_frac_per_config=esc.float().mean(1).cpu().numpy().tolist(),
                 escape_frac=float(esc.float().mean().item())))


# ==================================================================== estimators
def ar1_lam_D(segs, dt):
    """The COG-BCI arm's own estimator: pooled per-segment-centred lag-1 slope, lambda =
    -ln(a)/dt, D = var(innovations)/dt.  Segments shorter than 10 windows are dropped, exactly as
    the COG-BCI rate estimator does.  If EVERY segment is shorter than 10 but at least 2 long
    (the HCP MEG 2-window pair design), the pooled independent-pair map with grand centring is
    used instead -- that arm's own estimator.  Declared."""
    long = [np.asarray(s, float) for s in segs if len(s) >= 10]
    if long:
        cen = [x - x.mean() for x in long]
        num = float(sum(np.dot(x[:-1], x[1:]) for x in cen))
        den = float(sum(np.dot(x[:-1], x[:-1]) for x in cen))
        if den <= 0:
            return None
        a = num / den
        sd = float(np.std(np.concatenate(cen), ddof=1))
        if not (0.0 < a < 1.0):
            return dict(a=a, lam=np.nan, D=np.nan, sd=sd, mode="long", n=len(long))
        inn = np.concatenate([x[1:] - a * x[:-1] for x in cen])
        s2 = float(np.var(inn, ddof=1))
        return dict(a=float(a), lam=float(-math.log(a) / dt), D=float(s2 / dt), sd=sd,
                    mode="long", n=len(long))
    pairs = [np.asarray(s, float) for s in segs if len(s) >= 2]
    if not pairs:
        return None
    P = np.array([s[:2] for s in pairs])
    mu = float(P.mean())
    x, y = P[:, 0] - mu, P[:, 1] - mu
    den = float(np.dot(x, x))
    if den <= 0:
        return None
    a = float(np.dot(x, y) / den)
    sd = float(np.std(np.concatenate([x, y]), ddof=1))
    if not (0.0 < a < 1.0):
        return dict(a=a, lam=np.nan, D=np.nan, sd=sd, mode="pair", n=len(P))
    s2 = float(np.var(y - a * x, ddof=1))
    return dict(a=a, lam=float(-math.log(a) / dt), D=float(s2 / dt), sd=sd,
                mode="pair", n=len(P))


def _lg(v):
    """Element-wise natural log with a nan for any non-positive entry.  slope3 is applied to
    the LOGARITHM of the per-level lambda and D values (the slope of the level itself would read
    +0.0216 for the drive knob's sampled log-diffusion step, whose exact value is +0.1132)."""
    return [math.log(x) if (x is not None and np.isfinite(x) and x > 0) else np.nan for x in v]


def slope3(y):
    y = np.asarray(y, float)
    if not np.all(np.isfinite(y)):
        return np.nan
    return float(LV @ (y - y.mean()) / (LV @ LV))


def acf5(segs, nlag=5):
    r, mom = NL.acf_pooled(segs, nlag=nlag)
    if r is None:
        return None
    return dict(r=r.tolist(), sd=mom["sd"])


# ==================================================================== one network
def geometry(seed, log):
    geo = NL.base_geometry(seed, log)
    geo["seed"] = int(seed)
    return geo


def do_network(seed, reps, log):
    t0 = time.time()
    geo = geometry(seed, log)
    e = geo["e_load"]
    cfgs, base, meta = build_plane_configs(geo, log)
    cnt = build_count_configs(geo, base, log)
    for c in cfgs + cnt:
        c["Win"] = geo["Win"]
    all_cfgs = [c for c in (cfgs + cnt) if c["_cell"] is not None]
    lost = len(cfgs) + len(cnt) - len(all_cfgs)
    if lost:
        log(f"  [{seed}] WARNING {lost} configurations lost their stable branch")

    # time calibration on the BASE condition only
    cal = {}
    base_tk = NL.exact_cell(geo["W"], geo["Win"], geo["b"], np.zeros(4), geo["x0"], e, SIGMA0)
    for name, M in MODALITIES.items():
        rho_t = math.exp(-M["lam_base"] * M["dt"])
        tau = NL.solve_tau(base_tk, e, rho_t)
        if tau is None:
            raise SystemExit(f"{seed}/{name}: base rho never reaches {rho_t}")
        stride = max(1, int(round(tau / ALPHA)))
        cal[name] = dict(rho_target=rho_t, tau=float(tau), stride=int(stride),
                         tau_realised=float(stride * ALPHA),
                         rho_realised=float(NL.rho_at(base_tk, e, stride * ALPHA)),
                         seconds_per_time_unit=float(M["dt"] / tau))
    log(f"  [{seed}] strides " + " ".join(f"{k}:{v['stride']}" for k, v in cal.items()))

    unit_sd = float(np.sqrt(np.trace(base["S"]) / len(base["x"])))
    idx = {(c["truth"], c["size"], c["level"]): i for i, c in enumerate(all_cfgs)}
    keys = sorted({(c["truth"], c["size"]) for c in all_cfgs})

    rows, sim_meta = [], {}
    for mname, M in MODALITIES.items():
        stride = cal[mname]["stride"]
        dt = M["dt"]
        hrf = NL.spm_hrf(dt) if M["hrf"] else None
        n_lead = (len(hrf) - 1) if hrf is not None else 0
        n_keep = M["seg_len"] + n_lead
        n_chains = M["n_seg"] * reps
        t1 = time.time()
        Wnd, escm, simi = simulate_plane(all_cfgs, n_chains, n_keep, stride, BURN,
                                         SEED + zlib.crc32(mname.encode()) % 100000)
        if hrf is not None:
            Wc = np.empty((Wnd.shape[0], Wnd.shape[1], M["seg_len"], Wnd.shape[3]))
            for k in range(M["seg_len"]):
                Wc[:, :, k, :] = np.tensordot(Wnd[:, :, k:k + len(hrf), :], hrf[::-1],
                                              axes=([2], [0]))
            Wnd = Wc
        sim_meta[mname] = _clean(simi)
        log(f"  [{seed}] {mname}: simulated {Wnd.shape} in {time.time()-t1:.0f}s "
            f"escape_frac {simi['escape_frac']:.4f}")

        for truth, size in keys:
            kk = [idx.get((truth, size, m)) for m in range(LEVELS)]
            if any(x is None for x in kk):
                continue
            crs = [all_cfgs[i]["_cell"] for i in kk]
            # ---- the three tiers of truth for this (truth, size) at this modality
            gen = dict(
                dlog_lam_e=slope3([math.log(c["lam_e"]) for c in crs]),
                dlog_D_e=slope3([math.log(c["D_e"]) for c in crs]),
                dlog_lam_slow=slope3([math.log(c["lam_slow"]) for c in crs]),
                dlog_var_e=slope3([math.log(c["var_e"]) for c in crs]),
                disp_sd=float(np.linalg.norm(crs[-1]["x"] - crs[0]["x"]) / crs[0]["sd_load"]),
                ratio_identity_max_dev=float(np.max([abs(c["ratio_identity"] - 1.0)
                                                     for c in crs])))
            tau_rnn = stride * ALPHA
            sl = [sampled_lag_truth(c, e, tau_rnn, dt) for c in crs]
            samp = dict(dlog_lam=slope3(_lg([s["lam"] for s in sl])),
                        dlog_D=slope3(_lg([s["D"] for s in sl])),
                        lam=[s["lam"] for s in sl], D=[s["D"] for s in sl],
                        rho=[s["rho"] for s in sl], tau_rnn=float(tau_rnn))
            if hrf is not None:
                slh = [sampled_lag_truth(c, e, tau_rnn, dt, hrf=hrf) for c in crs]
                samp["hrf"] = dict(dlog_lam=slope3(_lg([s["lam"] for s in slh])),
                                   dlog_D=slope3(_lg([s["D"] for s in slh])),
                                   rho=[s["rho"] for s in slh])
            for nu in NU_GRID:
                rng = np.random.default_rng(
                    zlib.crc32(f"{seed}|{mname}|{truth}|{size}|{nu}".encode()) % (2 ** 32))
                Xlev = [Wnd[i] for i in kk]                      # (n_chains, n_keep, 64)
                Xn = [x + nu * unit_sd * rng.standard_normal(x.shape) for x in Xlev]
                # pooled state space across LEVELS and REPS (never per level, never per rep)
                P = np.vstack([x.reshape(-1, x.shape[-1]) for x in Xn])
                mu_, sd_ = P.mean(0), P.std(0)
                sd_ = np.where(sd_ > 1e-9, sd_, 1.0)
                Cv = np.cov((P - mu_) / sd_, rowvar=False)
                w_, V_ = np.linalg.eigh(Cv)
                V_ = V_[:, np.argsort(w_)[::-1][:M["d"]]]
                Zlev = [((x - mu_) / sd_) @ V_ for x in Xn]
                # per-replicate level means in the pooled space, for the held-out axis
                mus = np.zeros((LEVELS, reps, M["d"]))
                for m in range(LEVELS):
                    for rp in range(reps):
                        s = slice(rp * M["n_seg"], (rp + 1) * M["n_seg"])
                        mus[m, rp] = Zlev[m][s].reshape(-1, M["d"]).mean(0)
                for rp in range(reps):
                    s = slice(rp * M["n_seg"], (rp + 1) * M["n_seg"])
                    others = [j for j in range(reps) if j != rp]
                    wax = (mus[LEVELS - 1, others].mean(0) - mus[0, others].mean(0))
                    nw = float(np.linalg.norm(wax))
                    wax = wax / nw if nw > 1e-12 else None
                    est, est_clean = {}, {}
                    for rule in ("heldout", "fixed", "pc1"):
                        per, per_cl = [], []
                        for m in range(LEVELS):
                            if rule == "heldout":
                                if wax is None:
                                    per.append(None); per_cl.append(None); continue
                                segs = [z @ wax for z in Zlev[m][s]]
                            elif rule == "fixed":
                                segs = [z @ e for z in Xn[m][s]]
                            else:
                                segs = [z[:, 0] for z in Zlev[m][s]]
                            per.append(ar1_lam_D(segs, dt))
                            ok = ~escm[kk[m]][s]
                            per_cl.append(ar1_lam_D([sg for sg, o in zip(segs, ok) if o], dt)
                                          if ok.sum() >= max(2, len(segs) // 4) else None)
                        if not any(p is None for p in per_cl):
                            est_clean[rule] = dict(
                                dlog_lam=slope3([math.log(p["lam"]) if p["lam"] > 0 else np.nan
                                                 for p in per_cl]),
                                dlog_D=slope3([math.log(p["D"]) if p["D"] > 0 else np.nan
                                               for p in per_cl]),
                                n=[p["n"] for p in per_cl])
                        else:
                            est_clean[rule] = None
                        if any(p is None for p in per):
                            est[rule] = None
                            continue
                        est[rule] = dict(
                            lam=[p["lam"] for p in per], D=[p["D"] for p in per],
                            sd=[p["sd"] for p in per], a=[p["a"] for p in per],
                            mode=per[0]["mode"],
                            dlog_lam=slope3([math.log(p["lam"]) if p["lam"] > 0 else np.nan
                                             for p in per]),
                            dlog_D=slope3([math.log(p["D"]) if p["D"] > 0 else np.nan
                                           for p in per]),
                            dlog_sd=slope3([math.log(p["sd"]) if p["sd"] > 0 else np.nan
                                            for p in per]),
                            dlam=slope3([p["lam"] for p in per]))
                    # identifiability check attached to every rate: the shape of
                    # r(1..5) along the fixed axis at THIS sampling, level 0
                    ac = acf5([z @ e for z in Xn[0][s]]) if M["seg_len"] >= 7 else None
                    rows.append(dict(anchor_seed=int(seed), modality=mname, truth=truth,
                                     size=float(size), nu=float(nu), rep=int(rp),
                                     dt=dt, stride=stride,
                                     gen=_clean(gen), sampled=_clean(samp),
                                     est=_clean(est), est_clean=_clean(est_clean),
                                     acf0=_clean(ac),
                                     escape_frac=float(np.mean(
                                         [simi["escape_frac_per_config"][i] for i in kk]))))
        del Wnd

    out = dict(anchor_seed=int(seed), calibration=cal, meta=_clean(meta),
               operating_point=_clean(geo["opinfo"]), unit_sd=unit_sd,
               sim=sim_meta,
               configs=[{k: v for k, v in c.items()
                         if k in ("truth", "size", "level", "arm", "c", "q", "gamma",
                                  "n_stable", "n_fixed", "max_re_eig_any_fp",
                                  "sep_from_other")}
                        | dict(true=_clean({
                            "x_disp": float(np.linalg.norm(c["_cell"]["x"] - geo["x0"])),
                            "lam_slow": c["_cell"]["lam_slow"],
                            "lam_e": c["_cell"]["lam_e"],
                            "D_e": c["_cell"]["D_e"],
                            "var_e": c["_cell"]["var_e"],
                            "sd_load": c["_cell"]["sd_load"],
                            "spread_rms": c["_cell"]["spread_rms"],
                            "ratio_identity": c["_cell"]["ratio_identity"]}))
                        for c in all_cfgs],
               rows=rows, wall_s=time.time() - t0)
    (ROWS / f"rows_{seed}.json").write_text(json.dumps(_clean(out)), encoding="utf-8")
    log(f"  [{seed}] {len(rows)} rows in {time.time()-t0:.0f}s")
    return out


def main():
    ap = argparse.ArgumentParser(description="drive and gain manipulations in the (dlog lambda, dlog D) plane")
    ap.add_argument("--seeds", default=",".join(str(s) for s in ANCHORS))
    ap.add_argument("--reps", type=int, default=N_REPS)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    log = log_factory()
    log(f"drive_gain_plane.py  python {platform.python_version()} numpy {np.__version__} "
        f"torch {torch.__version__} device {DEV}")
    log(f"  truths {TRUTHS}+count  sizes {SIZES}  levels {LEVELS}  nu {NU_GRID} reps {a.reps}")
    for s in [int(x) for x in a.seeds.split(",") if x.strip()]:
        f = ROWS / f"rows_{s}.json"
        if a.resume and f.exists():
            log(f"  [{s}] exists, skipped")
            continue
        do_network(s, a.reps, log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
