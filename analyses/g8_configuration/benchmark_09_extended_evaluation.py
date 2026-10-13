"""Benchmark 09: the extended simulation evaluation (G8-augmented configuration).

Seeds and conditions are fixed in this script before any recording is generated.  Every
recording is certified once with the package defaults (levels 2 and 3: checks G0, G1, G2, G3, G8) and the stored
results are scored by benchmark_09_analyse.py.  Parts (one JSON-lines file each, resumable):

  single     48 single-timescale conditions (persistence 0.5/0.75/0.9/0.96 x noise 0/0.25/0.5 x four sizes), 8 each
  variants   unequal segment lengths; Student t innovations and noise (model holds), 10 each
  missp      ten misspecified conditions (two timescales, autocorrelated noise, long memory, oscillation,
             three-dimensional state; two sizes each), 10 each
  neg        white noise and random walks at two sizes, 10 each
  twins      simulations matched to 160 sleep recordings (two each)
  stab       60 recordings certified again with three other random streams
  abc        approximate Bayesian computation certified on a subset of single and missp (and its estimate)
  mdc        slope and spread of each estimator at the true generator of every single-timescale condition

    python analyses/g8_configuration/benchmark_09_extended_evaluation.py --part single [--workers 6] [--budget s] [--limit N]
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse                                                      # noqa: E402
import hashlib                                                       # noqa: E402
import itertools                                                     # noqa: E402
import json                                                          # noqa: E402
import sys                                                           # noqa: E402
import time                                                          # noqa: E402

import numpy as np                                                   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))   # the repository root
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "primary_configuration"))   # the sleep-corpus loader
OUT = os.path.join(HERE, "out")

SMOKE = bool(os.environ.get("NDIF_SMOKE"))          # smoke runs use other seeds and other files
ENTROPY = 202610082015 + (1_000_003 if SMOKE else 0)   # seed material of this evaluation
WHICH = ("G0", "G1", "G2", "G3", "G8")
SIZES = {"g25x50": (25, 50), "g15x200": (15, 200), "g50x100": (50, 100), "g8x500": (8, 500)}
MSIZES = {"g25x50": (25, 50), "g50x100": (50, 100)}


# ================================================================ conditions
def cells_single():
    out = []
    for a, nu, g in itertools.product((0.5, 0.75, 0.9, 0.96), (0.0, 0.25, 0.5), SIZES):
        out.append(dict(family="single", a=a, nu=nu, geom=g))
    return out


def cells_variants():
    return [dict(family="unequal", a=0.8, nu=0.2, geom="u30_200x20"),
            dict(family="student_t", a=0.8, nu=0.2, geom="g30x100")]


def cells_missp():
    out = []
    for fam in ("two_timescales", "ar_noise", "long_memory", "oscillation", "three_rates"):
        for g in MSIZES:
            out.append(dict(family=fam, geom=g))
    return out


def cells_neg():
    return [dict(family=f, geom=g) for f in ("white_noise", "random_walk") for g in MSIZES]


def _with_ids(cells, base):
    for i, c in enumerate(cells):
        c["cell"] = base + i
    return cells


CELLS = dict(single=_with_ids(cells_single(), 0), variants=_with_ids(cells_variants(), 100),
             missp=_with_ids(cells_missp(), 200), neg=_with_ids(cells_neg(), 300))
REPS = dict(single=8, variants=10, missp=10, neg=10)


# ================================================================ generators
def _ar1(a, n, m, rng, df=None):
    """m independent stationary AR(1) segments of length n with unit variance, shape (n, m)."""
    draw = (lambda size: rng.standard_t(df, size=size) / np.sqrt(df / (df - 2.0))) if df else \
        (lambda size: rng.normal(size=size))
    Y = np.empty((n, m))
    Y[0] = draw(m)
    q = np.sqrt(1 - a * a)
    E = draw((n, m))
    for t in range(1, n):
        Y[t] = a * Y[t - 1] + q * E[t]
    return Y


def _noise(shape, nu, rng, df=None):
    if nu <= 0:
        return np.zeros(shape)
    z = rng.standard_t(df, size=shape) / np.sqrt(df / (df - 2.0)) if df else rng.normal(size=shape)
    return np.sqrt(nu / (1 - nu)) * z


def _fracdiff(d, K):
    psi = np.empty(K)
    psi[0] = 1.0
    for j in range(1, K):
        psi[j] = psi[j - 1] * (j - 1 + d) / j
    return psi


def make(cell, rng):
    """Segments (list of (n,) or (n, d) arrays) and reference rates per sampling interval."""
    fam = cell["family"]
    if fam == "single":
        m, n = SIZES[cell["geom"]]
        Y = _ar1(cell["a"], n, m, rng) + _noise((n, m), cell["nu"], rng)
        return [Y[:, j] for j in range(m)], dict(primary=-np.log(cell["a"]))
    if fam == "unequal":
        lens = rng.integers(30, 201, size=20)
        segs = [_ar1(0.8, int(L), 1, rng)[:, 0] + _noise(int(L), 0.2, rng) for L in lens]
        return segs, dict(primary=-np.log(0.8))
    if fam == "student_t":
        Y = _ar1(0.8, 100, 30, rng, df=3) + _noise((100, 30), 0.2, rng, df=3)
        return [Y[:, j] for j in range(30)], dict(primary=-np.log(0.8))
    m, n = MSIZES.get(cell["geom"], (None, None))
    if fam == "two_timescales":
        Y = np.sqrt(0.6) * _ar1(0.97, n, m, rng) + np.sqrt(0.4) * _ar1(0.6, n, m, rng)
        Y = Y + _noise((n, m), 0.2, rng)
        r1 = 0.6 * 0.97 + 0.4 * 0.6
        return [Y[:, j] for j in range(m)], dict(primary=-np.log(0.97), lag1=-np.log(r1))
    if fam == "ar_noise":
        Y = np.sqrt(0.7) * _ar1(0.85, n, m, rng) + np.sqrt(0.3) * _ar1(0.5, n, m, rng)
        return [Y[:, j] for j in range(m)], dict(primary=-np.log(0.85))
    if fam == "long_memory":
        d, K = 0.4, 3000
        psi = _fracdiff(d, K)
        sd = np.sqrt(np.sum(psi ** 2))
        segs = []
        for _ in range(m):
            e = rng.normal(size=n + K - 1)
            x = np.convolve(e, psi, mode="valid") / sd
            segs.append(x + _noise(n, 0.1, rng))
        return segs, dict(primary=-np.log(d / (1 - d)))
    if fam == "oscillation":
        r, w = 0.95, np.pi / 10
        R = r * np.array([[np.cos(w), -np.sin(w)], [np.sin(w), np.cos(w)]])
        q = np.sqrt(1 - r * r)
        segs = []
        for _ in range(m):
            z = np.empty((n, 2))
            z[0] = rng.normal(size=2)
            for t in range(1, n):
                z[t] = R @ z[t - 1] + q * rng.normal(size=2)
            segs.append(z[:, 0] + _noise(n, 0.1, rng))
        return segs, dict(primary=-np.log(r), lag1=-np.log(r * np.cos(w)))
    if fam == "three_rates":
        a = np.array([0.97, 0.8, 0.5])
        U, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        segs = []
        for _ in range(m):
            z = np.column_stack([_ar1(a[k], n, 1, rng)[:, 0] for k in range(3)])
            x = z @ U.T
            segs.append(x + _noise((n, 3), 0.1, rng))
        return segs, dict(primary=-np.log(a.max()), _U=U.tolist(), _a=a.tolist())
    if fam == "white_noise":
        return [rng.normal(size=n) for _ in range(m)], {}
    if fam == "random_walk":
        return [np.cumsum(rng.normal(size=n)) for _ in range(m)], {}
    raise ValueError(fam)


def _design(segs, dt=1.0):
    from ndif import Design
    segs2 = [s[:, None] if np.ndim(s) == 1 else s for s in segs]
    X = np.concatenate(segs2, axis=0)
    e = np.cumsum([0] + [len(s) for s in segs2])
    return X, Design(dt=dt, window_s=dt, step_s=dt, segments=[(int(e[i]), int(e[i + 1])) for i in range(len(segs2))])


# ================================================================ certificates
def compact(cert):
    d = cert.to_dict()
    gates = {}
    for g in d["gates"]:
        det = g.get("detail") or {}
        e = dict(passed=g["passed"], value=g["value"], margin=g["margin"], cost_s=g["cost_s"], note=g["note"])
        if g["gate"] == "G3":
            r, v = det.get("rate") or {}, det.get("variance") or {}
            e.update(slope=r.get("slope"), offset=r.get("offset"), spread=r.get("spread"), criteria=r.get("criteria"),
                     n_rep=r.get("n_rep"), a_generator=r.get("a_generator"), nu_generator=r.get("nu_generator"),
                     a_fit=det.get("a_fit"), nu_fit=det.get("nu_fit"), mdc=det.get("detection_margin_fold"),
                     mdc_var=det.get("detection_margin_fold_variance"), var_decision=v.get("decision"),
                     var_spread=v.get("spread"), nu_at_boundary=det.get("nu_at_boundary"))
        elif g["gate"] == "G8":
            e.update(P=det.get("combined_p"), p_acf=det.get("p_autocorrelation"), p_lr=det.get("p_likelihood_ratio"),
                     lr=det.get("likelihood_ratio"), tau=det.get("relaxation_time"), n_rep=det.get("n_rep"))
        elif g["gate"] == "G1":
            e.update(nu_set=det.get("nu_set"), rate_set=det.get("rate_set"), nu_two_lag=(det.get("two_lag") or {}).get("nu"))
        elif g["gate"] == "G2":
            e.update(ci=det.get("ci"))
        gates[g["gate"]] = e
    return dict(levels={k: dict(status=v["status"], binding=v["binding_gate"]) for k, v in d["levels"].items()},
                gates=gates, version=d["provenance"]["ndif_version"])


def certify_all(segs, targets, cert_rng, ests, dt=1.0):
    from ndif import certify
    from ndif.estimators import fit_rate
    from ndif.gates import _project
    from ndif.types import Recording
    X, design = _design(segs, dt)
    proj = _project(Recording(X, design))
    out = {}
    for est in ests:
        t0 = time.perf_counter()
        c = certify(X, design, configuration="g8_augmented", rng=cert_rng, estimator=est, which=WHICH)
        wall = time.perf_counter() - t0
        lam = fit_rate(proj, dt=dt, method=est)
        lam = float(lam) if np.isfinite(lam) else None
        e = compact(c)
        e.update(lam=lam, wall_s=wall,
                 rel_error={k: (abs(lam - t) / t if lam is not None else None)
                            for k, t in targets.items() if not k.startswith("_")})
        out[est] = e
    return out, proj


def _seeds(*key):
    ss = np.random.SeedSequence(ENTROPY, spawn_key=tuple(int(k) for k in key))
    s_data, s_cert = ss.spawn(2)
    return np.random.default_rng(s_data), int(s_cert.generate_state(1)[0])


def job_cell(part, cell, rep, ests=("ols", "kalman"), stream=0):
    rng, cert_rng = _seeds(cell["cell"], rep)
    segs, targets = make(cell, rng)
    if stream:
        cert_rng = int(np.random.SeedSequence([cert_rng, 7000 + int(stream)]).generate_state(1)[0])
    certs, proj = certify_all(segs, targets, cert_rng, ests)
    if cell["family"] == "three_rates":
        # lag-1 rate of the noise-free projected signal, from the true model and the axis the data give
        from ndif.gates import _project
        from ndif.types import Recording
        X, design = _design(segs)
        rec = Recording(X, design)
        Z = np.concatenate(rec.segments(), 0)
        Z = Z - Z.mean(0)
        e = np.linalg.svd(Z, full_matrices=False)[2][0]
        w = np.asarray(targets["_U"]).T @ e
        a = np.asarray(targets["_a"])
        rho = float(np.sum(w ** 2 * a) / np.sum(w ** 2))
        targets["lag1"] = -np.log(rho)
        for est, c in certs.items():
            lam = c["lam"]
            c["rel_error"]["lag1"] = abs(lam - targets["lag1"]) / targets["lag1"] if lam is not None else None
    tg = {k: v for k, v in targets.items() if not k.startswith("_")}
    return dict(part=part, key=f"{cell['cell']}|{rep}" + (f"|s{stream}" if stream else ""), rep=int(rep),
                stream=int(stream), targets=tg, certs=certs, n_windows=int(sum(len(s) for s in segs)),
                **{k: v for k, v in cell.items()})


# ---------------------------------------------------------------- sleep-matched simulations
def sleep_sources():
    """160 stage-specific sleep recordings drawn (with the seed material of this test) from the 602 certified ones."""
    import glob
    ids = sorted(f"{os.path.basename(f).split('_')[0]}/{os.path.basename(f).split('_')[1]}"
                 for f in glob.glob(os.path.join(os.path.dirname(HERE), "primary_configuration", "out", "corpus_full",
                                                 "SC4*_*_kalman.json")))
    rng = np.random.default_rng(np.random.SeedSequence(ENTROPY + 1))
    return [ids[i] for i in sorted(rng.choice(len(ids), size=160, replace=False))]


def job_twin(idx, ident, k):
    import benchmark_05_corpus_full as B5
    from ndif.estimators import fit_ar1_kalman_batch
    name, stage = ident.split("/")
    _, win, stages = B5.load_night(os.path.join(B5.DEFAULT_CACHE, f"{name}_win1.npz"))
    segs_rec = stages[stage]
    # the source's own maximum-likelihood fit of equation (6) on its projected series, with the G8-augmented configuration
    from ndif.gates import _project
    from ndif.types import Recording
    Xr, dr = _design(segs_rec, win)
    g = fit_ar1_kalman_batch([_project(Recording(Xr, dr))])[0]
    a_fit, nu_fit = float(g["a"]), float(min(g["nu"], 0.95))
    lens = [len(s) for s in segs_rec]
    rng, cert_rng = _seeds(1000 + idx, k)
    q = np.sqrt(1 - a_fit * a_fit)
    segs = []
    for L in lens:
        x = np.empty(L)
        x[0] = rng.normal()
        e = rng.normal(size=L)
        for t in range(1, L):
            x[t] = a_fit * x[t - 1] + q * e[t]
        segs.append(x + _noise(L, nu_fit, rng))
    lam_true = float(-np.log(a_fit) / win)
    certs, _ = certify_all(segs, dict(primary=lam_true), cert_rng, ("ols", "kalman"), dt=win)
    return dict(part="twins", key=f"{idx}|{k}", source=ident, a_true=a_fit, nu_true=nu_fit, win=win,
                n_windows=int(sum(lens)), targets=dict(primary=lam_true), certs=certs)


# ---------------------------------------------------------------- minimum detectable change at the truth
def job_mdc(cell, est, n_sim=250):
    from ndif.estimators import fit_rates_padded, simulate_eq6_padded
    m, n = SIZES[cell["geom"]]
    rng, _ = _seeds(5000 + cell["cell"], {"ols": 0, "kalman": 1}[est])
    lens = [n] * m
    ladder = (0.6, 0.8, 1.0, 1.25, 1.7)
    A = np.concatenate([np.full(n_sim, cell["a"] ** f) for f in ladder])
    X = np.concatenate([np.full(n_sim, np.log(f)) for f in ladder])
    Y = simulate_eq6_padded(A, cell["nu"], lens, A.size, rng)
    lam = fit_rates_padded(Y, lens, 1.0, est)
    ok = np.isfinite(lam) & (lam > 0)
    x, y = X[ok], np.log(lam[ok])
    slope = float(np.polyfit(x, y, 1)[0])
    sd0 = float(np.std(y[np.isclose(x, 0)], ddof=1))
    return dict(part="mdc", key=f"{cell['cell']}|{est}", cell=cell["cell"], estimator=est, slope=slope, sd0=sd0,
                n=int(ok.sum()), a=cell["a"], nu=cell["nu"], geom=cell["geom"])


# ================================================================ driver
def jobs_for(part):
    if part in ("single", "variants", "missp", "neg"):
        return [(part, c, r) for r in range(REPS[part]) for c in CELLS[part]]
    if part == "twins":
        src = sleep_sources()
        return [("twins", i, s, k) for k in range(2) for i, s in enumerate(src)]
    if part == "stab":
        base = [("single", c, 0) for c in CELLS["single"]] + [("missp", c, 0) for c in CELLS["missp"]] + \
               [("variants", c, 0) for c in CELLS["variants"]]
        return [(p, c, r, s) for s in (1, 2, 3) for (p, c, r) in base]
    if part == "abc":
        return [("single", c, r) for r in range(3) for c in CELLS["single"]] + \
               [("missp", c, r) for r in range(2) for c in CELLS["missp"]]
    if part == "mdc":
        return [(c, e) for c in CELLS["single"] for e in ("ols", "kalman")]
    raise ValueError(part)


def key_of(part, j):
    if part in ("single", "variants", "missp", "neg", "abc"):
        return f"{j[1]['cell']}|{j[2]}"
    if part == "stab":
        return f"{j[1]['cell']}|{j[2]}|s{j[3]}"
    if part == "twins":
        return f"{j[1]}|{j[3]}"
    if part == "mdc":
        return f"{j[0]['cell']}|{j[1]}"
    raise ValueError(part)


def run_one(args):
    part, job = args
    if part in ("single", "variants", "missp", "neg"):
        return job_cell(job[0], job[1], job[2])
    if part == "abc":
        r = job_cell(job[0], job[1], job[2], ests=("abc",))
        r["part"] = "abc"
        return r
    if part == "stab":
        r = job_cell(job[0], job[1], job[2], stream=job[3])
        r["part"] = "stab"
        return r
    if part == "twins":
        return job_twin(job[1], job[2], job[3])
    if part == "mdc":
        return job_mdc(job[0], job[1])
    raise ValueError(part)



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["single", "variants", "missp", "neg", "twins", "stab", "abc", "mdc"])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--budget", type=float, default=None, help="stop submitting jobs after this many seconds; resumable")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    jobs = jobs_for(a.part)
    path = os.path.join(OUT, f"benchmark_09_{a.part}{'_smoke' if SMOKE else ''}.jsonl")
    done = set()
    if os.path.exists(path):
        done = {json.loads(l)["key"] for l in open(path, encoding="utf-8") if l.strip()}
    todo = [j for j in jobs if key_of(a.part, j) not in done]
    if a.limit:
        todo = todo[:a.limit]
    print(f"[{a.part}] {len(jobs)} jobs, {len(done)} done, {len(todo)} to run", flush=True)
    from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
    t0 = time.time()
    n = 0
    with open(path, "a", encoding="utf-8") as fh, ProcessPoolExecutor(max_workers=a.workers) as ex:
        pend, it = set(), iter(todo)
        while True:
            while len(pend) < a.workers and (a.budget is None or time.time() - t0 < a.budget):
                j = next(it, None)
                if j is None:
                    break
                pend.add(ex.submit(run_one, (a.part, j)))
            if not pend:
                break
            fin, pend = wait(pend, return_when=FIRST_COMPLETED)
            for fu in fin:
                try:
                    r = fu.result()
                except Exception as exc:
                    print(f"  FAILED: {exc!r}", flush=True)
                    continue
                fh.write(json.dumps(r) + "\n")
                fh.flush()
                n += 1
    print(f"[{a.part}] wrote {n} in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
