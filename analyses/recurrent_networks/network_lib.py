"""Shared functions of the recurrent-network analyses: the trained flip-flop networks, their fixed points and
Jacobians, simulated recordings, and the landscape readouts applied to them.

The networks are 64-unit rate networks r_{t+1} = (1 - alpha) r_t + alpha tanh(W r_t + W_in u_t + b) + sigma xi_t
trained on a flip-flop task (a +-1 pulse on input channel 0 in the first five steps must be held).  Training runs
at alpha = 0.2 (train_rate_rnn); every analysis reads the trained weights at alpha = 0.05 (FineRNN), with
sigma = 0.08 per step and u = 0 unless stated otherwise.  Fixed points do not depend on alpha, and stability at
alpha = 0.2 implies stability at alpha = 0.05.

Sections
  networks               RateRNN, flipflop_batch, train_rate_rnn (training), FineRNN, train_original_recipe,
                         scaled_net, fixed_points_hardened, completeness_check, anchor_params (the trained
                         weights), f_cont, jac_c, newton_fp, rho_1d
  simulated recordings   _clean, spm_hrf, _as_net, base_geometry, exact_cell, rho_at, solve_tau, invert,
                         simulate, state_space, acf_pooled, MODALITIES
  landscape readouts     ridgeline_points, ridgeline_modes_2, fit_linear_regime, multilinear_drift,
                         find_equilibria (equilibrium counter of a two-regime mixture drift), geometric_action,
                         gmam (finite-difference minimum-action descent), tau_within_dwell, latent_landscape

Locations
  NDIF_RNN_WEIGHTS   directory of the trained weights network_<seed>.pt (default analyses/recurrent_networks/weights)
  NDIF_DEVICE        torch device (default "cuda" if available, else "cpu").  The stored outputs were computed on a
                     CUDA device.
"""

from __future__ import annotations

import itertools
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy import linalg as sla
from scipy.linalg import expm, solve_continuous_lyapunov
from scipy.optimize import least_squares, root

HERE = Path(__file__).resolve().parent

# ----------------------------------------------------------------------------------------------- constants
DEV = os.environ.get("NDIF_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
WEIGHTS = Path(os.environ.get("NDIF_RNN_WEIGHTS") or (HERE / "weights"))

SEED = 20260915                       # project seed of the network analyses
ALPHA = 0.05                          # Euler step at which every analysis reads the networks
N_UNITS = 64
ENUM_SEED = SEED + 11                 # fixed-point enumeration seed
SIGMA0 = 0.08                         # operating noise per step
D0 = SIGMA0 ** 2 / (2.0 * ALPHA)      # 0.064, diffusion per unit time
BASE_GAIN = 0.86                      # operating gain of the bistable networks
DRIVE_CH = 0                          # input channel of the drive
MONO_MARGIN = 0.06                    # distance below the first monostable compensated gain
#: the eight trained networks with two stable fixed points (seed 20261013 has none and is not used)
ANCHORS = (20260922, 20260935, 20260948, 20260961, 20260974, 20260987, 20261000, 20261026)

# training recipe (alpha 0.2, 40-step trials)
TRAIN_SEED = 20260819
TRAIN_ALPHA = 0.2
TRAIN_T_TRIAL = 40
TRAIN_STEPS = 4000
TRAIN_LR = 5e-3

#: segment structure of simulated recordings: state dimension d, sampling interval dt (s), segments per
#: condition, samples per segment, the cohort's base relaxation rate (1/s) and whether the haemodynamic
#: response is applied
MODALITIES = {
    "EEG":   dict(d=8, dt=1.0,  n_seg=54,  seg_len=18, lam_base=0.3580, hrf=False),
    "MEG":   dict(d=8, dt=1.0,  n_seg=150, seg_len=2,  lam_base=0.7350, hrf=False),
    "fMRI":  dict(d=7, dt=0.72, n_seg=4,   seg_len=38, lam_base=0.5185, hrf=True),
    "SLEEP": dict(d=4, dt=2.0,  n_seg=8,   seg_len=30, lam_base=0.4628, hrf=False),
}


def weight_file(seed):
    """The trained weights of network `seed`."""
    return WEIGHTS / f"network_{int(seed)}.pt"


# ====================================================================================================
# networks: training
# ====================================================================================================


class RateRNN(torch.nn.Module):
    def __init__(self, n_in, n_units=N_UNITS, n_out=1, seed=TRAIN_SEED):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.W = torch.nn.Parameter(
            torch.randn(n_units, n_units, generator=g) * (1.2 / n_units ** 0.5))
        self.Win = torch.nn.Parameter(
            torch.randn(n_units, n_in, generator=g) * 0.5)
        self.b = torch.nn.Parameter(torch.zeros(n_units))
        self.Wout = torch.nn.Parameter(
            torch.randn(n_out, n_units, generator=g) * (1.0 / n_units ** 0.5))
        self.n_units = n_units

    def step(self, r, u):
        return (1 - TRAIN_ALPHA) * r + TRAIN_ALPHA * torch.tanh(
            r @ self.W.T + u @ self.Win.T + self.b)

    def forward(self, U, r0=None, sigma=0.0, gen=None):
        """U: (batch, T, n_in) -> states (batch, T, n_units), out (batch, T)."""
        B, T, _ = U.shape
        r = torch.zeros(B, self.n_units, device=U.device) if r0 is None else r0
        states = []
        for t in range(T):
            r = self.step(r, U[:, t])
            if sigma > 0:
                r = r + sigma * torch.randn(r.shape, device=U.device,
                                            generator=gen)
            states.append(r)
        S = torch.stack(states, 1)
        return S, (S @ self.Wout.T).squeeze(-1)


def flipflop_batch(n, rng, T=TRAIN_T_TRIAL):
    """Bistable control: a +-1 pulse in the first 5 steps must be held."""
    s = rng.choice([-1.0, 1.0], n).astype(np.float32)
    U = np.zeros((n, T, 4), np.float32)
    U[:, :5, 0] = s[:, None]
    U[:, :, 1] = rng.normal(0, 0.1, (n, T))
    return U, s, np.zeros(n, int)


def train_rate_rnn(task_fn, tag, steps=TRAIN_STEPS, seed=TRAIN_SEED):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = RateRNN(n_in=4, seed=seed).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=TRAIN_LR)
    def loss_of(out, yt):
        if yt.ndim == 2:                 # per-timestep target (tracking)
            return ((out[:, 5:] - yt[:, 5:]) ** 2).mean()
        return ((out[:, -10:] - yt[:, None]) ** 2).mean()

    def score(net, n=512):
        with torch.no_grad():
            U, y, _ = task_fn(n, rng)
            _, out = net(torch.tensor(U, device=DEV))
            o = out.cpu().numpy()
            if y.ndim == 2:              # tracking: variance explained
                r = o[:, 5:].ravel() - y[:, 5:].ravel()
                return float(1 - r.var() / y[:, 5:].ravel().var())
            return float((np.sign(o[:, -10:].mean(1)) == y).mean())

    for it in range(steps):
        U, y, _ = task_fn(128, rng)
        _, out = net(torch.tensor(U, device=DEV))
        loss = loss_of(out, torch.tensor(y, device=DEV))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        if (it + 1) % 1000 == 0:
            print(f"  [{tag}] {it + 1}/{steps} loss={loss.item():.4f} "
                  f"score={score(net):.3f}", flush=True)
    return net, score(net, 1024)


# ====================================================================================================
# networks: fixed points
# ====================================================================================================


class FineRNN(RateRNN):
    """RateRNN with the Euler step overridden. Weights and task code unchanged."""

    def step(self, r, u):
        return (1 - ALPHA) * r + ALPHA * torch.tanh(
            r @ self.W.T + u @ self.Win.T + self.b)


def train_original_recipe(task_fn, tag, steps, seed):
    """Train with train_rate_rnn (alpha = 0.2, 40-step trials), then hand the
    weights to a FineRNN so the read happens at alpha = 0.05.

    Fixed points are alpha-independent and stability at 0.2 implies stability at
    0.05 (see the module docstring), so this changes the sampling rate of the
    read and not the structure being read.
    """
    net02, score = train_rate_rnn(task_fn, tag, steps=steps, seed=seed)
    fine = FineRNN(n_in=4, seed=seed).to(DEV)
    fine.load_state_dict(net02.state_dict())
    return fine, score


def scaled_net(net, g):
    """A copy of `net` with the recurrent weights scaled by g.

    Scaling W walks the flip-flop toward the pitchfork at which its two stable
    fixed points merge.  The ground truth is NOT assumed to survive the scaling:
    the fixed points are re-enumerated at the operating point and the arm is
    used only where two stable points are still found.
    """
    import copy
    m = copy.deepcopy(net)
    with torch.no_grad():
        m.W.mul_(float(g))
    return m


def fixed_points_hardened(net, u_const, n_starts=1000, traj_starts=None,
                          seed=SEED):
    """Adam on |F|^2, then Newton/LM refinement, then Jacobian classification."""
    from scipy.optimize import root
    torch.manual_seed(seed)
    u = torch.tensor(u_const, dtype=torch.float32, device=DEV)[None, :]

    starts = torch.randn(n_starts, net.n_units, device=DEV) * 0.8
    if traj_starts is not None and len(traj_starts):
        idx = np.random.default_rng(seed).choice(
            len(traj_starts), size=min(len(traj_starts), n_starts // 2),
            replace=False)
        extra = torch.tensor(np.asarray(traj_starts)[idx], dtype=torch.float32,
                             device=DEV)
        starts = torch.cat([starts, extra], 0)
    r = starts.clone().requires_grad_(True)
    opt = torch.optim.Adam([r], lr=0.05)
    for _ in range(2000):
        q = ((net.step(r, u.expand(len(r), -1)) - r) ** 2).sum(1).mean()
        opt.zero_grad()
        q.backward()
        opt.step()
    cand = r.detach().cpu().numpy()

    W = net.W.detach().cpu().numpy()
    Win = net.Win.detach().cpu().numpy()
    b = net.b.detach().cpu().numpy()
    uc = np.asarray(u_const, float)

    def F(x):
        return np.tanh(W @ x + Win @ uc + b) - x

    def J_of(x):
        pre = W @ x + Win @ uc + b
        return (1.0 - np.tanh(pre) ** 2)[:, None] * W - np.eye(len(x))

    uniq, n_conv = [], 0
    for x0 in cand:
        try:
            sol = root(F, x0, jac=J_of, method="hybr", tol=1e-13)
        except Exception:
            continue
        if not sol.success or np.linalg.norm(F(sol.x)) > 1e-10:
            continue
        n_conv += 1
        if not any(np.linalg.norm(sol.x - q) < 1e-2 for q in uniq):
            uniq.append(sol.x)

    out = []
    for x in uniq:
        # discrete map Jacobian: stability of the alpha-step iteration
        pre = W @ x + Win @ uc + b
        Jd = (1 - ALPHA) * np.eye(len(x)) + ALPHA * (
            (1.0 - np.tanh(pre) ** 2)[:, None] * W)
        ev = np.abs(np.linalg.eigvals(Jd))
        out.append(dict(point=x, max_abs_eig=float(ev.max()),
                        stable=bool(ev.max() < 1.0 - 1e-9)))
    return out, n_conv, len(cand)


def completeness_check(net, u_const, stable_pts, n_probe=500, n_settle=4000,
                       seed=SEED):
    """Forward-simulate from random inits; every settled state must sit on an
    enumerated stable fixed point.  This is the argument that nothing was missed."""
    if not stable_pts:
        return dict(n_probe=0, frac_accounted=0.0, max_dist=float("nan"))
    g = torch.Generator(device=DEV).manual_seed(seed + 99)
    u = torch.tensor(u_const, dtype=torch.float32, device=DEV)[None, :]
    r = torch.randn(n_probe, net.n_units, device=DEV, generator=g) * 0.9
    with torch.no_grad():
        for _ in range(n_settle):
            r = net.step(r, u.expand(n_probe, -1))
    S = r.cpu().numpy()
    P = np.asarray(stable_pts)
    d = np.linalg.norm(S[:, None, :] - P[None, :, :], axis=2).min(axis=1)
    return dict(n_probe=int(n_probe), frac_accounted=float((d < 1e-2).mean()),
                max_dist=float(d.max()), median_dist=float(np.median(d)))


# ====================================================================================================
# networks: the trained weights, drift and Jacobian
# ====================================================================================================


def anchor_params(seed, gain=BASE_GAIN):
    """Trained network `seed` with its recurrent weights scaled by `gain`: (net, W, W_in, b) in float64."""
    f = weight_file(seed)
    if not f.exists():
        raise SystemExit(f"missing trained weights {f}")
    net = FineRNN(n_in=4, seed=int(seed)).to(DEV)
    net.load_state_dict(torch.load(f, map_location=DEV))
    m = scaled_net(net, gain)
    return (m,
            m.W.detach().cpu().numpy().astype(np.float64),
            m.Win.detach().cpu().numpy().astype(np.float64),
            m.b.detach().cpu().numpy().astype(np.float64))


def f_cont(x, W, Win, b, u):
    return np.tanh(W @ x + Win @ u + b) - x


def jac_c(x, W, Win, b, u):
    pre = W @ x + Win @ u + b
    return (1.0 - np.tanh(pre) ** 2)[:, None] * W - np.eye(len(x))


def newton_fp(x0, W, Win, b, u, tol=1e-11):
    sol = root(lambda x: f_cont(x, W, Win, b, u), x0,
               jac=lambda x: jac_c(x, W, Win, b, u), method="hybr", tol=1e-13)
    if not sol.success:
        return None, np.inf
    res = float(np.linalg.norm(f_cont(sol.x, W, Win, b, u)))
    return (sol.x if res < tol else None), res


def rho_1d(J, S, e, lags, ev=None, V=None):
    """rho(t) = e' expm(J t) S e / (e' S e): the exact 1-D autocorrelation along e.

    Computed through the eigendecomposition J = V diag(ev) V^-1, so
        e' expm(Jt) S e = sum_k (e'V)_k exp(ev_k t) (V^-1 S e)_k ,
    which costs ONE decomposition and then 64 complex exponentials per lag instead
    of a 64x64 matrix exponential per lag.  Verified against expm in the self-test.
    """
    den = float(e @ S @ e)
    if den <= 0:
        return None
    if ev is None:
        ev, V = np.linalg.eig(J)
    try:
        a = np.linalg.solve(V, S @ e)
    except np.linalg.LinAlgError:
        return np.asarray([float(e @ (expm(J * t) @ (S @ e))) / den for t in lags])
    c = (e @ V) * a                      # (64,) complex weights
    E = np.exp(np.outer(np.asarray(lags, float), ev))       # (n_lags, 64)
    return np.real(E @ c) / den


# ====================================================================================================
# simulated recordings
# ====================================================================================================


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return f if math.isfinite(f) else None
    if isinstance(o, (np.integer, int)):
        return int(o)
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    return o


def spm_hrf(tr, length_s=32.0):
    t = np.arange(0.0, length_s + 1e-9, tr)
    a1, a2, b1, b2, c = 6.0, 16.0, 1.0, 1.0, 1.0 / 6.0
    from scipy.special import gammaln
    def g(t, a, b):
        with np.errstate(divide="ignore", invalid="ignore"):
            v = np.exp(a * np.log(np.maximum(t, 1e-12) / b) - np.maximum(t, 0) / b
                       - gammaln(a + 1))
        return np.where(t > 0, v, 0.0)
    h = g(t, a1, b1) - c * g(t, a2, b2)
    s = h.sum()
    return h / (s if abs(s) > 1e-12 else 1.0)


def _as_net(seed, W, Win, b):
    net = FineRNN(n_in=4, seed=int(seed)).to(DEV)
    with torch.no_grad():
        net.W.copy_(torch.tensor(W, dtype=torch.float32))
        net.b.copy_(torch.tensor(b, dtype=torch.float32))
        net.Win.copy_(torch.tensor(Win, dtype=torch.float32))
    return net


def base_geometry(seed, log=None):
    """The monostable operating point of a trained network, its drive direction and its geometry.

    At the bistable operating gain 0.86 the mean first-passage time between the two wells is of
    the order of a few thousand samples, so at each modality's own segment length 23% (EEG), 9%
    (MEG), 51% (fMRI) and 46% (SLEEP) of simulated segments contain a completed escape -- a
    regime no cortical recording occupies, and one that breaks the rate estimator: with escapes
    present the fitted spectral radius of the 8-dimensional one-step map reads 0.960 where its
    true value is 0.449.

    The operating point is therefore moved down the COMPENSATED gain (the recurrent weights
    scaled by gamma, with the bias adjusted so that the upper well stays a fixed point) to
    gamma_mono, the largest gamma on a 0.01 grid descending from 1.00 at which the enumeration at
    u = 0 returns exactly one stable fixed point, minus a further MONO_MARGIN of gain.  The base
    fixed point is pinned at the network's own upper well by the compensation, so the operating
    point is that well, made the only one.  The drive direction e_load is -J0^-1 W_in[:, DRIVE_CH].
    """
    m, W, Win, b = anchor_params(seed, BASE_GAIN)
    fps, _, _ = fixed_points_hardened(m, np.zeros(4, np.float32), n_starts=300,
                                         seed=ENUM_SEED)
    stable = [np.asarray(f["point"], float) for f in fps if f["stable"]]
    if len(stable) == 2:
        ax = stable[1] - stable[0]; ax /= np.linalg.norm(ax)
        xb = stable[int(np.argmax([float(s @ ax) for s in stable]))]
        xo = stable[int(np.argmin([float(s @ ax) for s in stable]))]
    else:
        xb, xo = stable[0], None
    at = np.arctanh(np.clip(xb, -1 + 1e-14, 1 - 1e-14))
    gam_first, scan = None, []
    for gam in np.round(np.arange(1.00, 0.499, -0.01), 4):
        Wg = gam * W; bg = at - Wg @ xb
        f2, _, _ = fixed_points_hardened(_as_net(seed, Wg, Win, bg),
                                            np.zeros(4, np.float32), n_starts=300,
                                            seed=ENUM_SEED)
        ns = int(sum(f["stable"] for f in f2))
        scan.append(dict(gamma=float(gam), n_fixed=len(f2), n_stable=ns))
        if ns == 1:
            gam_first = float(gam)
            break
    if gam_first is None:
        raise SystemExit(f"{seed}: no monostable compensated gain found down to 0.50")
    gam_mono = round(gam_first - MONO_MARGIN, 4)
    Wb = gam_mono * W
    bb = at - Wb @ xb
    netb = _as_net(seed, Wb, Win, bb)
    f3, _, _ = fixed_points_hardened(netb, np.zeros(4, np.float32), n_starts=600,
                                        seed=ENUM_SEED)
    st3 = [np.asarray(f["point"], float) for f in f3 if f["stable"]]
    comp = completeness_check(netb, np.zeros(4, np.float32),
                                 [s.astype(np.float32) for s in st3], n_probe=500,
                                 seed=ENUM_SEED)
    J0 = jac_c(xb, Wb, Win, bb, np.zeros(4))
    e = -np.linalg.solve(J0, Win[:, DRIVE_CH])
    e = e / np.linalg.norm(e)
    info = dict(gamma_first_monostable=gam_first, gamma_mono=gam_mono,
                n_stable_base_gain=len(stable),
                fp_separation_base_gain=(float(np.linalg.norm(xb - xo))
                                         if xo is not None else None),
                n_fixed_at_operating=len(f3), n_stable_at_operating=len(st3),
                completeness_at_operating=float(comp["frac_accounted"]),
                fp_pinned_residual=float(np.linalg.norm(f_cont(xb, Wb, Win, bb,
                                                                  np.zeros(4)))),
                scan=scan)
    if log:
        log(f"  [{seed}] operating point gamma_mono {gam_mono} "
            f"(first monostable {gam_first}); n_stable {len(st3)}, "
            f"completeness {comp['frac_accounted']:.3f}")
    return dict(W=Wb, Win=Win, b=bb, x0=xb, x_other=None, J0=J0, e_load=e,
                W_base_gain=W, b_base_gain=b, x_other_base_gain=xo, opinfo=info)


def exact_cell(W, Win, b, u, x_guess, e, sigma):
    """Ground truth of one configuration: x*, spectrum, Sigma, 1-D autocorrelation."""
    x, res = newton_fp(x_guess, W, Win, b, u)
    if x is None:
        return None
    J = jac_c(x, W, Win, b, u)
    ev, V = np.linalg.eig(J)
    if np.max(ev.real) >= 0:
        return None
    D = sigma ** 2 / (2.0 * ALPHA)
    S = solve_continuous_lyapunov(J, -2.0 * D * np.eye(len(x)))
    S = 0.5 * (S + S.T)
    return dict(x=x, J=J, ev=ev, V=V, S=S, D=D,
                lam_slow=float(-np.max(ev.real)), lam_mean=float(-ev.real.mean()),
                sd_load=float(np.sqrt(max(e @ S @ e, 0.0))),
                spread_rms=float(np.sqrt(np.trace(S) / len(x))))


def rho_at(cellrec, e, t):
    return float(rho_1d(cellrec["J"], cellrec["S"], e, [t],
                           ev=cellrec["ev"], V=cellrec["V"])[0])


def solve_tau(cellrec, e, rho_target, t_hi=30.0):
    """Smallest t > 0 with rho(t) = rho_target, by bisection on the first crossing."""
    lo, hi = 1e-3, None
    ts = np.arange(0.02, t_hi, 0.02)
    r = rho_1d(cellrec["J"], cellrec["S"], e, ts, ev=cellrec["ev"], V=cellrec["V"])
    idx = np.flatnonzero(r <= rho_target)
    if len(idx) == 0:
        return None
    i = int(idx[0])
    lo = ts[i - 1] if i > 0 else 1e-3
    hi = ts[i]
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if rho_1d(cellrec["J"], cellrec["S"], e, [mid],
                     ev=cellrec["ev"], V=cellrec["V"])[0] > rho_target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def invert(levels_x, levels_y, target, hi_guess=None):
    """Monotone interpolation: the x at which y reaches target.

    y may be increasing OR decreasing in x (lambda_slow decreases with the recurrent gain),
    so the sort is on y, not on x.  A first version sorted on x and handed a decreasing y to
    np.interp, which silently returned the wrong gain; the error was caught because the
    realised lambda_slow fold did not match the target and was identical at two scales.
    """
    x = np.asarray(levels_x, float); y = np.asarray(levels_y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    o = np.argsort(y); x, y = x[o], y[o]
    if target <= y[0]:
        return float(x[0])
    if target >= y[-1]:
        return float(x[-1])
    return float(np.interp(target, y, x))


def simulate(cfgs, n_chains, n_keep, stride, burn, seed, sep=None, over=1.0, dev=DEV):
    """(n_cfg, n_chains, n_keep, 64) windows, each chain an independent segment.

    Every chain starts at its own configuration's stable fixed point and burns in, so the
    sampled regime is the one the recordings occupy: inside one basin, with no completed
    escape.  That is not assumed -- it is ENFORCED and counted.  `over` times as many chains
    are run as are needed; a chain that ever crosses the wells-axis midpoint is marked and
    dropped, and the surviving fraction is returned.

    THIS FILTER IS NOT COSMETIC.  Without it, 21% of the MEG chains of a bistable network
    escape to the other well by the end of the burn-in, and those segments alone drive the
    fitted spectral radius of the 8-dimensional one-step map from its true value 0.449 to
    0.960 -- an escape inside a recording-length segment breaks the per-load rate estimator
    completely.  The escape fraction at each modality's own segment length is returned.
    """
    C = len(cfgs)
    n = cfgs[0]["W"].shape[0]
    B = int(math.ceil(n_chains * over))
    Wt = torch.tensor(np.stack([c["W"] for c in cfgs]), dtype=torch.float32,
                      device=dev).transpose(1, 2).contiguous()
    beff = torch.tensor(np.stack([c["b"] + c["Win_u"] for c in cfgs]),
                        dtype=torch.float32, device=dev)[:, None, :]
    sig = torch.tensor(np.array([c["sigma"] for c in cfgs]), dtype=torch.float32,
                       device=dev)[:, None, None]
    x0 = torch.tensor(np.stack([c["true"]["x"] for c in cfgs]), dtype=torch.float32,
                      device=dev)[:, None, :]
    g = torch.Generator(device=dev).manual_seed(int(seed))
    r = x0.expand(C, B, n).contiguous().clone()
    esc = torch.zeros(C, B, dtype=torch.bool, device=dev)
    # excursion diagnostic, per configuration and relative to ITS OWN fixed point: a state
    # further than 4 times the configuration's own RMS radius is an excursion.  (A crossing of
    # the base network's wells-axis midpoint would flag every uncompensated-gain cell, because
    # their fixed point sits on the other side of it.)
    rad = torch.tensor(np.array([4.0 * math.sqrt(max(np.trace(c["_Scov"]), 1e-12))
                                 for c in cfgs]), dtype=torch.float32, device=dev)[:, None]

    def mark(rr):
        d2 = ((rr - x0) ** 2).sum(-1)
        esc.logical_or_(d2 > rad ** 2)

    with torch.no_grad():
        for i in range(burn):
            r = (1 - ALPHA) * r + ALPHA * torch.tanh(torch.bmm(r, Wt) + beff) \
                + sig * torch.randn(r.shape, device=dev, generator=g)
            if (i % 25) == 0:
                mark(r)
        keep = torch.empty(C, B, n_keep, n, device=dev)
        for k in range(n_keep):
            for _ in range(stride):
                r = (1 - ALPHA) * r + ALPHA * torch.tanh(torch.bmm(r, Wt) + beff) \
                    + sig * torch.randn(r.shape, device=dev, generator=g)
                mark(r)
            keep[:, :, k, :] = r
        n_clean = (~esc).sum(1)
        order = torch.argsort(esc.to(torch.int8), dim=1, stable=True)[:, :n_chains]
        sel = keep.gather(1, order[:, :, None, None].expand(C, n_chains, n_keep, n))
    return (sel.cpu().numpy().astype(np.float64),
            dict(n_run=int(B), n_needed=int(n_chains),
                 escape_frac=float(esc.float().mean().item()),
                 min_clean=int(n_clean.min().item()),
                 frac_configs_short=float((n_clean < n_chains).float().mean().item()),
                 escape_frac_per_config=esc.float().mean(1).cpu().numpy().tolist()))


def state_space(Xlev, d, nu, unit_sd, rng):
    """State-space rule: observation noise on the units, pooled z-score, PCA fitted on the
    windows POOLED ACROSS LEVELS (never per level), then project every level into it."""
    Xn = [x + nu * unit_sd * rng.standard_normal(x.shape) for x in Xlev]
    P = np.vstack([x.reshape(-1, x.shape[-1]) for x in Xn])
    mu, sd = P.mean(0), P.std(0)
    sd = np.where(sd > 1e-9, sd, 1.0)
    Pz = (P - mu) / sd
    C = np.cov(Pz, rowvar=False)
    w, V = np.linalg.eigh(C)
    V = V[:, np.argsort(w)[::-1][:d]]
    return [(((x - mu) / sd) @ V) for x in Xn], V


def acf_pooled(segs, nlag=5):
    """Pooled per-segment-centred lag-k autocorrelation of a 1-D set of segments."""
    segs = [np.asarray(s, float) for s in segs if len(s) >= nlag + 2]
    if not segs:
        return None, None
    c = [s - s.mean() for s in segs]
    v = float(np.mean(np.concatenate([x ** 2 for x in c])))
    if v <= 1e-15:
        return None, None
    r = []
    for k in range(1, nlag + 1):
        num, cnt = 0.0, 0
        for x in c:
            if len(x) > k:
                num += float(np.dot(x[:-k], x[k:])); cnt += len(x) - k
        r.append(num / cnt / v if cnt else np.nan)
    m2 = float(np.mean(np.concatenate([np.diff(x) ** 2 for x in c if len(x) > 1])))
    return np.array(r), dict(var=v, m2=m2, sd=math.sqrt(v))


# ====================================================================================================
# landscape readouts
# ====================================================================================================


def ridgeline_points(mus, Sigmas, alphas):
    """Ray & Lindsay (2005) ridgeline manifold of a Gaussian mixture.

    For weights alpha in the (K-1)-simplex,

        x*(alpha) = [ sum_k alpha_k S_k^-1 ]^-1 [ sum_k alpha_k S_k^-1 mu_k ]

    THEOREM (Ray & Lindsay 2005, Thm 2.1): every critical point of the mixture
    density -- every mode, every saddle, every antimode -- lies on this set.
    The converse is false, so the ridgeline is a *sufficient search set*:
    scanning it cannot miss a mode.

    This is what makes the mode count exact in d = 8.  A grid search would need
    O(g^8) evaluations; the ridgeline needs O(g^(K-1)) -- O(g) for two
    attractors -- INDEPENDENT of the ambient dimension.
    """
    Ps = [np.linalg.inv(np.asarray(S, float)) for S in Sigmas]
    Pm = [P @ np.asarray(m, float) for P, m in zip(Ps, mus)]
    out = np.empty((len(alphas), len(np.asarray(mus[0]))))
    for i, a in enumerate(alphas):
        M = sum(ai * P for ai, P in zip(a, Ps))
        b = sum(ai * v for ai, v in zip(a, Pm))
        out[i] = np.linalg.solve(M, b)
    return out


def _log_mix_density(X, mus, Sigmas, weights):
    lds = [np.linalg.slogdet(np.asarray(S, float))[1] for S in Sigmas]
    Ps = [np.linalg.inv(np.asarray(S, float)) for S in Sigmas]
    d = X.shape[1]
    L = np.empty((len(X), len(mus)))
    for k, (m, P, ld) in enumerate(zip(mus, Ps, lds)):
        dv = X - np.asarray(m, float)
        L[:, k] = (np.log(max(weights[k], 1e-300)) - 0.5 * d * np.log(2 * np.pi)
                   - 0.5 * ld - 0.5 * np.einsum("nk,kl,nl->n", dv, P, dv))
    mx = L.max(axis=1, keepdims=True)
    return mx[:, 0] + np.log(np.exp(L - mx).sum(axis=1))


@dataclass
class RidgelineResult:
    n_modes: int
    barrier_nats: float        # depth of the separating well, in nats
    separation: float          # Mahalanobis separation of the two centres
    alpha_grid: np.ndarray
    logp: np.ndarray
    path: np.ndarray


def ridgeline_modes_2(mu1, S1, mu2, S2, w1=0.5, n=4001) -> RidgelineResult:
    """EXACT mode count for a two-component Gaussian mixture.

    Returns the number of modes and the barrier depth = min(peak heights) minus
    the intervening minimum of log p along the ridgeline.  A barrier of 0 means
    ONE basin: the two clouds are two regions of a single attractor, and no
    Freidlin-Wentzell inter-attractor quasipotential exists between them,
    because there is nothing to cross.

    Calibration: two Gaussians with equal covariance and equal weight are
    bimodal iff their Mahalanobis separation exceeds 2 exactly (Behboodian 1970,
    Technometrics 12:131-139).  The separation is reported alongside so the
    result reads as a distance from the bifurcation rather than a bare yes/no.
    """
    a = np.linspace(0.0, 1.0, n)
    alphas = np.column_stack([1 - a, a])
    X = ridgeline_points([mu1, mu2], [S1, S2], alphas)
    lp = _log_mix_density(X, [mu1, mu2], [S1, S2], [w1, 1 - w1])

    is_max = np.zeros(n, bool)
    is_max[1:-1] = (lp[1:-1] > lp[:-2]) & (lp[1:-1] > lp[2:])
    is_max[0], is_max[-1] = lp[0] > lp[1], lp[-1] > lp[-2]
    idx = np.where(is_max)[0]
    n_modes = int(len(idx))
    if n_modes >= 2:
        interior = lp[idx[0]:idx[-1] + 1]
        barrier = float(min(lp[idx[0]], lp[idx[-1]]) - interior.min())
    else:
        barrier = 0.0
    dmu = np.asarray(mu2, float) - np.asarray(mu1, float)
    sep = float(np.sqrt(dmu @ np.linalg.solve(0.5 * (np.asarray(S1, float)
                                                     + np.asarray(S2, float)), dmu)))
    return RidgelineResult(n_modes, barrier, sep, a, lp, X)


@dataclass
class LinearRegime:
    label: str
    mu: np.ndarray
    Sigma: np.ndarray
    A: np.ndarray             # continuous-time drift, A = log(M)/dt
    M: np.ndarray             # discrete one-step map
    Q: np.ndarray             # innovation covariance
    n: int
    logm_ok: bool


def fit_linear_regime(segments, label, dt=1.0, ridge=1e-6):
    """VAR(1) -> OU fit on a list of contiguous non-overlapping state segments.

    Increments are taken WITHIN segments only, so block boundaries never
    contribute a spurious jump.  A = log(M)/dt uses the principal matrix
    logarithm; where that fails to reproduce M the Euler surrogate is used and
    flagged, which is the 'system aliasing' obstruction of Yue, Thunberg &
    Goncalves (2016, arXiv:1605.06973) surfacing rather than being hidden.
    """
    X, Y = [], []
    for S in segments:
        S = np.asarray(S, float)
        if len(S) >= 2:
            X.append(S[:-1])
            Y.append(S[1:])
    X, Y = np.vstack(X), np.vstack(Y)
    Z = np.vstack([np.asarray(s, float) for s in segments])
    mu = Z.mean(axis=0)
    Xc, Yc = X - mu, Y - mu
    G = Xc.T @ Xc + ridge * np.eye(Xc.shape[1]) * np.trace(Xc.T @ Xc) / Xc.shape[1]
    M = np.linalg.solve(G, Xc.T @ Yc).T
    R = Yc - Xc @ M.T
    Q = np.cov(R, rowvar=False, ddof=1)
    Sigma = np.cov(Z, rowvar=False, ddof=1)
    ok = True
    try:
        A = np.real(sla.logm(M)) / dt
        if not np.allclose(sla.expm(A * dt), M, atol=1e-6, rtol=1e-4):
            A, ok = (M - np.eye(len(M))) / dt, False
    except Exception:
        A, ok = (M - np.eye(len(M))) / dt, False
    return LinearRegime(label, mu, Sigma, A, M, Q, int(len(X)), ok)


def multilinear_drift(x, regimes, softness=1.0):
    """Piecewise-linear drift field: b(x) = sum_k w_k(x) A_k (x - mu_k).

    Weights are Gaussian responsibilities tempered by `softness`, so softness
    -> 0 recovers a hard partition (a genuinely piecewise-linear, non-smooth
    field) and softness = 1 the smooth mixture-of-experts field.  The resulting
    field is NONLINEAR even though every regime is linear, which is exactly what
    makes multiple equilibria possible where a single linear model can have only
    one.

    TRAP -- A UNIFORM RESCALING OF EVERY A_k IS AN IDENTITY, NOT A PERTURBATION.
    The responsibilities `w_k(x)` computed below depend on `r.mu` and `r.Sigma`
    ONLY; `r.A` enters at the last line and nowhere else.  So `A_k -> c A_k` for
    one common c > 0 gives `b(x) -> c b(x)`, whose zero set is IDENTICAL and
    whose Jacobian is `c J`, preserving the sign of every eigenvalue.  The number
    and stability of the equilibria are therefore invariant BY CONSTRUCTION.

    This is worth stating here because a robustness check of exactly that form
    ("the count is unchanged when every drift matrix is halved") was run, passed
    trivially, and was read as evidence about the bifurcation.  It is not
    evidence about anything except the numerical stability of `find_equilibria`.
    A perturbation that can move the equilibria has to change the drifts
    RELATIVE TO EACH OTHER -- see the independent per-regime draw in
    `run_28_logm_robustness.py`.

    This construction is a recurrent switching linear dynamical system (rSLDS)
    field in all but name: Gaussian responsibilities over locally linear
    dynamics.  The difference is what selects the regime -- here an exogenous
    task context, in rSLDS the continuous state itself -- not the algebra.
    """
    x = np.atleast_2d(np.asarray(x, float))
    L = np.empty((len(x), len(regimes)))
    for k, r in enumerate(regimes):
        dv = x - r.mu
        P = np.linalg.inv(r.Sigma)
        L[:, k] = -0.5 * (np.linalg.slogdet(r.Sigma)[1]
                          + np.einsum("nk,kl,nl->n", dv, P, dv))
    L /= max(softness, 1e-6)
    L -= L.max(axis=1, keepdims=True)
    Wt = np.exp(L)
    Wt /= Wt.sum(axis=1, keepdims=True)
    b = np.zeros_like(x)
    for k, r in enumerate(regimes):
        b += Wt[:, [k]] * ((x - r.mu) @ r.A.T)
    return b


def find_equilibria(regimes, softness=1.0, n_starts=300, seed=0):
    """Newton search for zeros of the multi-linear drift field.

    Starting points are the regime centres, ridgeline points between every
    regime pair (where a merged pair of wells and its saddle must lie if they
    exist at all), and draws from each regime's stationary law.  Returns
    distinct equilibria with Jacobian spectra so stable nodes can be separated
    from saddles.
    """
    from scipy.optimize import fsolve
    rng = np.random.default_rng(seed)
    d = len(regimes[0].mu)
    starts = [np.asarray(r.mu, float) for r in regimes]
    for i, j in itertools.combinations(range(len(regimes)), 2):
        a = np.linspace(0, 1, 9)
        starts += list(ridgeline_points([regimes[i].mu, regimes[j].mu],
                                        [regimes[i].Sigma, regimes[j].Sigma],
                                        np.column_stack([1 - a, a])))
    for r in regimes:
        starts += list(rng.multivariate_normal(r.mu, r.Sigma,
                                               size=max(1, n_starts // len(regimes))))

    def f(z):
        return multilinear_drift(z, regimes, softness)[0]

    eqs = []
    for s in starts:
        try:
            z, _, ier, _ = fsolve(f, s, full_output=True, xtol=1e-12)
        except Exception:
            continue
        if ier != 1 or np.linalg.norm(f(z)) > 1e-6:
            continue
        if not any(np.linalg.norm(z - e) < 1e-4 * (1 + np.linalg.norm(e)) for e in eqs):
            eqs.append(z)
    out = []
    for z in eqs:
        J = np.empty((d, d))
        h = 1e-5 * (1 + np.abs(z))
        for k in range(d):
            e = np.zeros(d)
            e[k] = h[k]
            J[:, k] = (f(z + e) - f(z - e)) / (2 * h[k])
        ev = np.linalg.eigvals(J)
        out.append(dict(x=z, eig=ev, stable=bool(np.all(ev.real < 0)),
                        n_unstable=int(np.sum(ev.real > 0))))
    return out


def geometric_action(path, drift_fn, D=None):
    """Freidlin-Wentzell geometric action of a curve (Heymann & Vanden-Eijnden
    2008, Phys. Rev. Lett. 100:140601).

        S(phi) = int ( |phi'|_a |b(phi)|_a - <phi', b(phi)>_a ) ds ,
        <u,v>_a = u' a^{-1} v ,  a = 2D.

    The geometric form is parameterisation-free, which is what allows the
    minimisation in arbitrary dimension without the grid that limits
    ordered-upwind solvers to 2-3 dimensions (Cameron 2012, Physica D
    241:1532-1550).  The action vanishes on paths following the deterministic
    flow and is positive on paths opposing it, so its minimum over curves
    joining two equilibria is the quasipotential barrier between them.
    """
    P = np.asarray(path, float)
    d = P.shape[1]
    a = 2.0 * (np.eye(d) if D is None else np.asarray(D, float))
    ai = np.linalg.inv(a)
    dP = np.diff(P, axis=0)
    mid = 0.5 * (P[1:] + P[:-1])
    b = drift_fn(mid)
    nb = np.sqrt(np.maximum(np.einsum("nk,kl,nl->n", b, ai, b), 0))
    nd = np.sqrt(np.maximum(np.einsum("nk,kl,nl->n", dP, ai, dP), 0))
    dot = np.einsum("nk,kl,nl->n", dP, ai, b)
    return float(np.sum(nd * nb - dot))


def _reparameterise(P):
    """Redistribute nodes uniformly in arclength -- the defining gMAM step."""
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    if s[-1] <= 0:
        return P
    s = s / s[-1]
    t = np.linspace(0, 1, len(P))
    return np.column_stack([np.interp(t, s, P[:, k]) for k in range(P.shape[1])])


def gmam(x0, x1, drift_fn, D=None, n_pts=81, n_iter=300, lr=0.05):
    """Minimise the geometric action over curves from x0 to x1."""
    x0, x1 = np.asarray(x0, float), np.asarray(x1, float)
    t = np.linspace(0, 1, n_pts)[:, None]
    P = x0 + t * (x1 - x0)
    best, bestS = P.copy(), geometric_action(P, drift_fn, D)
    step = lr * np.linalg.norm(x1 - x0) / n_pts
    for _ in range(n_iter):
        G = np.zeros_like(P)
        h = 1e-6 * (1 + np.abs(P).max())
        for k in range(P.shape[1]):
            Pp, Pm = P.copy(), P.copy()
            Pp[1:-1, k] += h
            Pm[1:-1, k] -= h
            G[1:-1, k] = (geometric_action(Pp, drift_fn, D)
                          - geometric_action(Pm, drift_fn, D)) / (2 * h)
        gn = np.linalg.norm(G)
        if gn < 1e-14:
            break
        P = P - step * G / gn * np.sqrt(n_pts)
        P[0], P[-1] = x0, x1
        P = _reparameterise(P)
        S = geometric_action(P, drift_fn, D)
        if S < bestS:
            best, bestS = P.copy(), S
        else:
            step *= 0.85
            if step < 1e-12:
                break
    return bestS, best


# ====================================================================================================
# density barrier of a two-well run
# ====================================================================================================


def tau_within_dwell(proj, lab):
    """Relaxation time from lag-1 autocorrelation inside dwell segments.

    The same estimator transit_and_tau uses, on the latent projection with the
    ground-truth well label, so the barrier and the transit statistic share one
    clock.
    """
    taus, runs, cur, start = [], [], lab[0], 0
    for t in range(1, len(lab)):
        if lab[t] != cur:
            runs.append((start, t))
            cur, start = lab[t], t
    runs.append((start, len(lab)))
    for a, b in runs:
        seg = proj[a:b]
        if len(seg) < 20:
            continue
        seg = seg - seg.mean()
        v = float(seg.var())
        if v <= 1e-12:
            continue
        r1 = float(np.dot(seg[:-1], seg[1:]) / (len(seg) - 1) / v)
        if 0.01 < r1 < 0.999:
            taus.append(-1.0 / np.log(r1))
    return float(np.median(taus)) if taus else None


def latent_landscape(proj, lab):
    """The latent landscape, read with the SAME estimator the assay uses.

    TWO ESTIMATORS WERE TRIED AND REJECTED BEFORE THIS ONE.

    (1) log(mean dwell / tau_relax). The blind calibration showed tau RISING
        with gain (39.5 at 0.76, 58.2 at 0.80) where a deeper, stiffer well must
        relax faster. The cause is the estimator: tau comes from the lag-1
        autocorrelation of residuals about each dwell segment's own mean, and
        removing a segment mean biases r1 downward more for short segments.
        Deeper barriers give longer segments, less bias, larger tau -- so the
        ratio partly cancels the thing it measures. Kept as a secondary, with
        the bias recorded, because its disagreement with the density read is
        itself an instrument finding.

    (2) A smoothed density of the projection, read by modes_and_barrier. There
        is no usable bandwidth: at Silverman's rule a well-separated bimodal
        sample returns SIX modes and a barrier of 0.174 nats where the true
        value is 5.43, because ragged tails create spurious outer maxima; at
        twice Silverman the mode count is stable but the barrier is shrunk by
        smoothing, and the shrinkage grows with depth, which would bend exactly
        the identity line this run exists to draw.

    So the latent landscape is read the way the assay reads the observed one:
    ridgeline_modes_2 on the two labelled groups' moments, here in the 1-D
    latent projection with the GROUND-TRUTH well label. Same estimator, same
    definition of a barrier, no free parameter -- which is what makes the
    generative and the read barrier directly comparable.

    Returns (n_modes, barrier_nats, separation) in the latent projection.
    """
    proj = np.asarray(proj, float)
    a, b = proj[lab == 0], proj[lab == 1]
    if len(a) < 20 or len(b) < 20:
        return None, None, None
    r = ridgeline_modes_2(np.array([a.mean()]), np.array([[a.var()]]),
                             np.array([b.mean()]), np.array([[b.var()]]),
                             w1=0.5, n=4001)
    return int(r.n_modes), float(r.barrier_nats), float(r.separation)
