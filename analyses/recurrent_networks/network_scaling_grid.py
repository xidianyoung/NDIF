"""Joint and separate scaling of the drift and the diffusion in eight trained bistable networks.

The stationary covariance solves the Lyapunov equation A Sigma + Sigma A' + 2D = 0, so a density is a joint function
of A and D.  Contracting that equation with any unit direction e gives, exactly and in any dimension,
        e' D e = - e' A Sigma e      =>      e' Sigma e = D_e / lambda_e ,
with lambda_e = -(e' A Sigma e)/(e' Sigma e) the initial decay rate of the autocorrelation along e.  The spread a
density reports along any direction is therefore the ratio D/lambda and not either of them; the separation between
modes in units of the spread, the density barrier and the equilibrium count are functions of that ratio alone.

The networks.  The Euler map r_{t+1} = (1-a) r_t + a tanh(W r_t + b) + sigma xi_t of each trained network at its
operating gain 0.86 is, per sample index, a drift a*f(r) with f(r) = -r + tanh(Wr+b) and a diffusion sigma^2/2.
Two knobs:
    g   a -> g a          every relaxation rate is multiplied by g; the fixed points do not move
    s   sigma -> s sigma  the diffusion is multiplied by s^2; fixed points and Jacobians are unchanged
so lambda ~ g, D ~ s^2, Sigma ~ s^2/g, and the density is a function of s^2/g alone.  Three paths through one
5 x 5 grid (g, s^2 in 0.5, 2^-0.5, 1, 2^0.5, 2):
    joint (ratio)    s^2 = g : lambda and D both move by up to a factor of 2 with the ratio held
    diffusion only   g = 1, s varies
    drift only       s = 1, g varies

Per (network, g, s^2)
    exact     Sigma from the Lyapunov equation at each well, lambda_e and D_e along the wells axis, the ratio
              identity's residual, the latent separation delta and delta^2/8
    measured  fit_linear_regime per regime -> find_equilibria(softness=1, n_starts=200) and ridgeline_modes_2 on
              the connecting direction, on simulated recordings at one fixed sampling held identical across the
              grid (the stride at which the base cell's autocorrelation along the wells axis equals
              exp(-0.358 x 1 s)).  Three constructions: `two_wells` (regime 0 = lower well, regime 1 = upper
              well), `one_well` (both regimes from the same well; true separation zero) and `offset` (both
              regimes from the same well, one displaced by a fixed vector in state units set so that the base
              cell sits at a latent separation of 3.0, the construction in which the count can move)
    escape    128 chains started in the upper well over a fixed horizon; exponential hazard per sample index
              under right censoring (events / total exposure); censored when no chain escapes

Inputs   trained weights analyses/recurrent_networks/weights/network_<seed>.pt (NDIF_RNN_WEIGHTS), device
         NDIF_DEVICE (default cuda if available).
Outputs  analyses/recurrent_networks/out/network_parameter_grid.csv (600 rows = 8 networks x 25 cells x
         3 constructions), network_parameter_grid.json, network_parameter_grid.log.  The stored rows are the
         Source Data file analysis_inputs/stored_results/network_parameter_grid.csv.
Reported in  Fig. 1d,e; Supplementary Fig. 1; Supplementary Table 14 (through network_scaling_summary.py).
Runtime  about 4 minutes on one CUDA GPU (torch float32; the stored run used torch 2.9.0+cu130).

    python analyses/recurrent_networks/network_scaling_grid.py --reps 2 --horizon 20000
"""
from __future__ import annotations

import argparse
import csv as _csv
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import solve_continuous_lyapunov

HERE = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

import torch  # noqa: E402
import network_lib as NL  # noqa: E402
from analyses.common import paths  # noqa: E402

SEED = 20260922
ALPHA0 = NL.ALPHA
SIGMA0 = NL.SIGMA0
BASE_GAIN = NL.BASE_GAIN
ANCHORS = NL.ANCHORS
DEV = NL.DEV

G_GRID = (0.5, 2.0 ** -0.5, 1.0, 2.0 ** 0.5, 2.0)          # lambda multiplier
S2_GRID = (0.5, 2.0 ** -0.5, 1.0, 2.0 ** 0.5, 2.0)         # D multiplier (s = sqrt(s2))
D_STATE = 8
N_SEG = 54
SEG_LEN = 18
OVER = 12.0                     # chains run per chain needed, to survive the escape filter
BURN = 400
N_STARTS_EQ = 200
LAM_BASE = 0.3580               # ds003838 base rate, s^-1, for the ONE fixed sampling
DT_S = 1.0
ESC_CHAINS = 128
OFFSET_SD = 3.0                 # the `offset` arm's fixed physical displacement, in base-cell s.d.

_clean = NL._clean


def log_factory(sink):
    def log(m):
        s = time.strftime("[%H:%M:%S] ") + str(m)
        print(s, flush=True)
        sink.append(s)
    return log


# ==================================================================== geometry
def wells(seed):
    """The bistable network at the operating gain and its two stable fixed points."""
    net, W, Win, b = NL.anchor_params(seed, BASE_GAIN)
    fps, _, _ = NL.fixed_points_hardened(net, np.zeros(4, np.float32), n_starts=400,
                                         seed=NL.ENUM_SEED)
    st = [np.asarray(f["point"], float) for f in fps if f["stable"]]
    if len(st) != 2:
        return None
    ax = st[1] - st[0]
    ax = ax / np.linalg.norm(ax)
    hi = st[int(np.argmax([float(x @ ax) for x in st]))]
    lo = st[int(np.argmin([float(x @ ax) for x in st]))]
    return dict(W=W, Win=Win, b=b, hi=hi, lo=lo, axis=(hi - lo) / np.linalg.norm(hi - lo),
                mid=0.5 * (hi + lo), sep=float(np.linalg.norm(hi - lo)),
                n_fixed=len(fps), n_stable=len(st))


def exact_at(x, W, Win, b, g, s2):
    """Per-sample-index drift g*ALPHA0*Jc and diffusion s2*SIGMA0^2/2 at the well x."""
    Jc = NL.jac_c(x, W, Win, b, np.zeros(4))
    A = (g * ALPHA0) * Jc
    Dm = (s2 * SIGMA0 ** 2 / 2.0) * np.eye(len(x))
    S = solve_continuous_lyapunov(A, -2.0 * Dm)
    S = 0.5 * (S + S.T)
    return A, Dm, S


# ==================================================================== simulation
def simulate(cfgs, n_chains, n_keep, stride, burn, seed, mid, axis, side, dev=DEV):
    """Per-configuration alpha and sigma.  A chain that ever crosses the wells-axis midpoint
    away from its own starting well is marked; nothing is filtered here, the mask is returned."""
    C = len(cfgs)
    n = cfgs[0]["W"].shape[0]
    Wt = torch.tensor(np.stack([c["W"] for c in cfgs]), dtype=torch.float32,
                      device=dev).transpose(1, 2).contiguous()
    bt = torch.tensor(np.stack([c["b"] for c in cfgs]), dtype=torch.float32,
                      device=dev)[:, None, :]
    al = torch.tensor(np.array([c["alpha"] for c in cfgs]), dtype=torch.float32,
                      device=dev)[:, None, None]
    sg = torch.tensor(np.array([c["sigma"] for c in cfgs]), dtype=torch.float32,
                      device=dev)[:, None, None]
    x0 = torch.tensor(np.stack([c["x"] for c in cfgs]), dtype=torch.float32,
                      device=dev)[:, None, :]
    sd = torch.tensor(np.array([c["side"] for c in cfgs]), dtype=torch.float32,
                      device=dev)[:, None]
    midt = torch.tensor(mid, dtype=torch.float32, device=dev)[None, None, :]
    axt = torch.tensor(axis, dtype=torch.float32, device=dev)[None, None, :]
    g = torch.Generator(device=dev).manual_seed(int(seed))
    r = x0.expand(C, n_chains, n).contiguous().clone()
    esc = torch.zeros(C, n_chains, dtype=torch.bool, device=dev)

    def step(r):
        return (1 - al) * r + al * torch.tanh(torch.bmm(r, Wt) + bt) \
            + sg * torch.randn(r.shape, device=dev, generator=g)

    def mark(rr):
        esc.logical_or_(sd * ((rr - midt) * axt).sum(-1) < 0)

    with torch.no_grad():
        for i in range(burn):
            r = step(r)
            if (i % 20) == 0:
                mark(r)
        keep = torch.empty(C, n_chains, n_keep, n, device=dev)
        for k in range(n_keep):
            for _ in range(stride):
                r = step(r)
                mark(r)
            keep[:, :, k, :] = r
    return keep.cpu().numpy().astype(np.float64), esc.cpu().numpy()


def escape_hazard(cfgs, n_chains, horizon, seed, mid, axis, dev=DEV):
    """Fraction of chains started in the upper well that cross the midpoint within `horizon`
    sample indices, and the hazard -ln(1-frac)/horizon.  Censored when frac reaches 1."""
    C = len(cfgs)
    n = cfgs[0]["W"].shape[0]
    Wt = torch.tensor(np.stack([c["W"] for c in cfgs]), dtype=torch.float32,
                      device=dev).transpose(1, 2).contiguous()
    bt = torch.tensor(np.stack([c["b"] for c in cfgs]), dtype=torch.float32,
                      device=dev)[:, None, :]
    al = torch.tensor(np.array([c["alpha"] for c in cfgs]), dtype=torch.float32,
                      device=dev)[:, None, None]
    sg = torch.tensor(np.array([c["sigma"] for c in cfgs]), dtype=torch.float32,
                      device=dev)[:, None, None]
    x0 = torch.tensor(np.stack([c["x"] for c in cfgs]), dtype=torch.float32,
                      device=dev)[:, None, :]
    midt = torch.tensor(mid, dtype=torch.float32, device=dev)[None, None, :]
    axt = torch.tensor(axis, dtype=torch.float32, device=dev)[None, None, :]
    g = torch.Generator(device=dev).manual_seed(int(seed))
    r = x0.expand(C, n_chains, n).contiguous().clone()
    first = torch.full((C, n_chains), float("nan"), device=dev)
    with torch.no_grad():
        for i in range(horizon):
            r = (1 - al) * r + al * torch.tanh(torch.bmm(r, Wt) + bt) \
                + sg * torch.randn(r.shape, device=dev, generator=g)
            if (i % 10) == 0:
                crossed = ((r - midt) * axt).sum(-1) < 0
                new = crossed & torch.isnan(first)
                first = torch.where(new, torch.full_like(first, float(i)), first)
    f = first.cpu().numpy()
    out = []
    for c in range(C):
        ev = np.isfinite(f[c])
        frac = float(ev.mean())
        # exponential MLE under right censoring: events / total exposure.  A first version used
        # -ln(1-frac)/H, which is undefined once every chain has escaped; at the largest D on this
        # grid that is every cell, so the arm would have gone missing exactly where it is decisive.
        expo = float(np.sum(np.where(ev, f[c], horizon)))
        haz = (ev.sum() / expo) if expo > 0 and ev.sum() > 0 else np.nan
        out.append(dict(escape_frac=frac, hazard=haz, n_events=int(ev.sum()),
                        exposure=expo,
                        median_first=float(np.median(f[c][ev])) if ev.any() else None,
                        censored=bool(ev.sum() == 0)))
    return out


# ==================================================================== counter / barrier
def counter_and_barrier(segs0, segs1, dt, seed):
    """The landscape construction: fit_linear_regime per regime in a pooled PCA state space,
    find_equilibria on the two-regime field, ridgeline density barrier along the connecting
    direction."""
    P = np.vstack([s.reshape(-1, s.shape[-1]) for s in (segs0, segs1)])
    mu, sd = P.mean(0), P.std(0)
    sd = np.where(sd > 1e-9, sd, 1.0)
    C = np.cov((P - mu) / sd, rowvar=False)
    w, V = np.linalg.eigh(C)
    V = V[:, np.argsort(w)[::-1][:D_STATE]]
    Z = [((s - mu) / sd) @ V for s in (segs0, segs1)]
    try:
        regs = [NL.fit_linear_regime([x for x in Z[i]], f"r{i}", dt=dt) for i in (0, 1)]
    except Exception as ex:
        return dict(error=str(ex))
    S = np.mean([r.Sigma for r in regs], axis=0)
    dmu = regs[1].mu - regs[0].mu
    sep = float(np.sqrt(max(dmu @ np.linalg.solve(S, dmu), 0.0)))
    out = dict(separation=sep)
    try:
        eqs = NL.find_equilibria(regs, softness=1.0, n_starts=N_STARTS_EQ, seed=seed)
        out["n_equilibria"] = len(eqs)
        out["n_stable"] = int(sum(e["stable"] for e in eqs))
    except Exception as ex:
        out["counter_error"] = str(ex)
    try:
        v = dmu / max(np.linalg.norm(dmu), 1e-12)
        r = NL.ridgeline_modes_2(np.array([regs[0].mu @ v]),
                                 np.array([[float(v @ regs[0].Sigma @ v)]]),
                                 np.array([regs[1].mu @ v]),
                                 np.array([[float(v @ regs[1].Sigma @ v)]]),
                                 w1=0.5, n=4001)
        out.update(density_modes=int(r.n_modes), density_barrier_nats=float(r.barrier_nats),
                   latent_separation=float(r.separation))
    except Exception as ex:
        out["barrier_error"] = str(ex)
    return out


def ar1_lam_D(segs, dt):
    long = [np.asarray(s, float) for s in segs if len(s) >= 10]
    if not long:
        return None
    cen = [x - x.mean() for x in long]
    num = float(sum(np.dot(x[:-1], x[1:]) for x in cen))
    den = float(sum(np.dot(x[:-1], x[:-1]) for x in cen))
    if den <= 0:
        return None
    a = num / den
    sd = float(np.std(np.concatenate(cen), ddof=1))
    if not (0.0 < a < 1.0):
        return dict(a=float(a), lam=np.nan, D=np.nan, sd=sd)
    inn = np.concatenate([x[1:] - a * x[:-1] for x in cen])
    return dict(a=float(a), lam=float(-math.log(a) / dt),
                D=float(np.var(inn, ddof=1) / dt), sd=sd)


# ==================================================================== one network
def do_network(seed, reps, horizon, log):
    t0 = time.time()
    geo = wells(seed)
    if geo is None:
        log(f"  [{seed}] SKIPPED: the network is not bistable at gain {BASE_GAIN}")
        return []
    W, Win, b, e = geo["W"], geo["Win"], geo["b"], geo["axis"]
    A0, D0m, S0 = exact_at(geo["hi"], W, Win, b, 1.0, 1.0)
    # ONE fixed sampling for the whole grid: the stride at which the BASE cell's 1-D
    # autocorrelation along the wells axis equals exp(-LAM_BASE * DT_S)
    se0 = float(e @ S0 @ e)
    rho_t = math.exp(-LAM_BASE * DT_S)
    stride = None
    from scipy.linalg import expm
    for k in range(1, 4001):
        if float(e @ expm(A0 * k) @ S0 @ e) / se0 <= rho_t:
            stride = k
            break
    if stride is None:
        stride = 200
    log(f"  [{seed}] wells sep {geo['sep']:.3f}  stride {stride}  base sd_axis {math.sqrt(se0):.4f}")

    cfgs, meta = [], []
    for g in G_GRID:
        for s2 in S2_GRID:
            for which in ("hi", "lo"):
                x = geo[which]
                cfgs.append(dict(W=W, b=b, x=x, alpha=g * ALPHA0,
                                 sigma=math.sqrt(s2) * SIGMA0,
                                 side=(+1.0 if which == "hi" else -1.0)))
                meta.append((g, s2, which))
    n_chains = int(round(N_SEG * reps * OVER))
    Wnd, esc = simulate(cfgs, n_chains, SEG_LEN, stride, BURN,
                        SEED + 1, geo["mid"], geo["axis"], None)
    log(f"  [{seed}] simulated {Wnd.shape} escape_frac {esc.mean():.3f}")
    hz = escape_hazard([c for c, m in zip(cfgs, meta) if m[2] == "hi"], ESC_CHAINS, horizon,
                       SEED + 2, geo["mid"], geo["axis"])
    hz_by = {(m[0], m[1]): h for (m, h) in zip([m for m in meta if m[2] == "hi"], hz)}
    log(f"  [{seed}] escape horizon {horizon}: frac "
        f"{min(h['escape_frac'] for h in hz):.3f}-{max(h['escape_frac'] for h in hz):.3f}")

    rows = []
    for gi, g in enumerate(G_GRID):
        for si, s2 in enumerate(S2_GRID):
            ih = meta.index((g, s2, "hi")); il = meta.index((g, s2, "lo"))
            Ah, Dh, Sh = exact_at(geo["hi"], W, Win, b, g, s2)
            Al_, Dl, Sl = exact_at(geo["lo"], W, Win, b, g, s2)
            Sp = 0.5 * (Sh + Sl)
            se = float(e @ Sp @ e)
            lam_e = float(-(e @ Ah @ Sh @ e) / float(e @ Sh @ e))
            D_e = float(e @ Dh @ e)
            delta = geo["sep"] / math.sqrt(se)
            base = dict(anchor_seed=int(seed), g=float(g), s2=float(s2),
                        ratio=float(s2 / g), stride=int(stride),
                        exact_lam_e_per_index=lam_e, exact_D_e_per_index=D_e,
                        exact_var_e=se,
                        exact_ratio_identity=float(float(e @ Sh @ e) * lam_e / D_e),
                        exact_delta=float(delta), exact_delta2_over_8=float(delta ** 2 / 8.0),
                        escape_frac=float(hz_by[(g, s2)]["escape_frac"]),
                        escape_hazard=hz_by[(g, s2)]["hazard"],
                        escape_censored=hz_by[(g, s2)]["censored"])
            # keep only chains that never crossed, so each regime is a clean one-well sample
            oh = np.flatnonzero(~esc[ih])[: N_SEG * reps]
            ol = np.flatnonzero(~esc[il])[: N_SEG * reps]
            base["clean_frac_hi"] = float((~esc[ih]).mean())
            base["clean_frac_lo"] = float((~esc[il]).mean())
            base["n_clean_hi"] = int(len(oh)); base["n_clean_lo"] = int(len(ol))
            if len(oh) < N_SEG or len(ol) < N_SEG:
                base["error"] = "too few escape-free chains"
                rows.append(base)
                continue
            Xh, Xl = Wnd[ih][oh], Wnd[il][ol]
            # a FIXED physical offset, in state units, set once per network so that the base cell
            # (g = 1, s^2 = 1) sits at a latent separation of 3.0 -- just above the counter's own
            # detection edge.  Nothing about the dynamics differs between the two regimes; only the
            # geometry is set.  This is the arm in which the COUNT can move, because in the
            # two-well arm the separation is 6-25 and the counter is saturated at 2 everywhere.
            off = OFFSET_SD * math.sqrt(float(e @ exact_at(geo["hi"], W, Win, b, 1.0, 1.0)[2] @ e))
            for arm in ("two_wells", "one_well", "offset"):
                if arm == "two_wells":
                    s0, s1 = Xl, Xh
                elif arm == "one_well":
                    half = len(Xh) // 2
                    s0, s1 = Xh[:half], Xh[half:]
                else:
                    half = len(Xh) // 2
                    s0, s1 = Xh[:half], Xh[half:] + off * e
                cb = counter_and_barrier(s0, s1, DT_S, SEED + 3)
                ar = ar1_lam_D([x @ e for x in np.concatenate([s0, s1])], DT_S)
                r = dict(base); r["arm"] = arm
                for k, v in cb.items():
                    r[f"m_{k}"] = v
                if ar:
                    r["m_lam_axis"] = ar["lam"]; r["m_D_axis"] = ar["D"]; r["m_sd_axis"] = ar["sd"]
                rows.append(r)
    log(f"  [{seed}] {len(rows)} rows in {time.time()-t0:.0f}s")
    return rows


def main():
    ap = argparse.ArgumentParser(description="joint and separate drift/diffusion scaling grid")
    ap.add_argument("--seeds", default=",".join(str(s) for s in ANCHORS))
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--horizon", type=int, default=20000)
    a = ap.parse_args()
    sink = []
    log = log_factory(sink)
    log(f"network_scaling_grid.py  python {platform.python_version()} "
        f"numpy {np.__version__} torch {torch.__version__} device {DEV}")
    log(f"  g grid {G_GRID}  s2 grid {S2_GRID}  reps {a.reps}  horizon {a.horizon}")
    rows = []
    for s in [int(x) for x in a.seeds.split(",") if x.strip()]:
        rows += do_network(s, a.reps, a.horizon, log)
    out = lambda name: paths.out("analyses", "recurrent_networks", "out", name)  # noqa: E731
    with open(out("network_parameter_grid.json"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(_clean(rows)))
    keys = sorted({k for r in rows for k in r})
    with open(out("network_parameter_grid.csv"), "w", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=keys, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    with open(out("network_parameter_grid.log"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(sink))
    log(f"wrote {len(rows)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
