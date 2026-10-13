# -*- coding: utf-8 -*-
r"""What the short-time band-power pipeline does to the autocorrelation of a single-timescale process.

Short-time band power is a SQUARED, SMOOTHED quantity, so its autocorrelation is not the
autocorrelation of the signal.  The pipeline does three things, separated here:

  (1) WINDOW AVERAGING.  A non-overlapping window average of a single-timescale process has
      r(k) = C a^k EXACTLY, with C > 1: constant successive ratio, amplitude EXCESS.  It pushes the
      exponential index ABOVE 1 and cannot produce a deficit.
  (2) SPECTRAL ESTIMATION NOISE.  A band power on a finite window is a chi-square variable about its
      true value, and log10 of it is the true log power plus a draw that is INDEPENDENT between
      non-overlapping windows.  That is an additive white component: r(k) = (1 - nu) a^k, with nu set
      by the number of FFT bins in the band.  Amplitude DEFICIT, constant successive ratio.
  (3) MIXING ACROSS FEATURES.  Robust z and the PCA form linear combinations of eight features.  If
      the features carry DIFFERENT timescales, every projection is a MIXTURE OF EXPONENTIALS and the
      successive ratio RISES.  This is measured here and bounded.

Design.  A two-channel 100 Hz signal is synthesised whose band-power envelope follows a KNOWN AR(1)
in log10 power, per band, with a known coefficient a and a known amplitude.  The band power is then
computed by the pipeline's own function (lag_lib.band_power: scipy spectrogram, Hann,
nperseg = win_s * sfreq, noverlap = 0, constant detrend, psd density), log10-ed, robust-z-ed and
pushed through the sleep state space (lag_lib.build_state_space, d = 4) - the chain of the sleep
analysis.  The calibrated single-timescale fit (lag_lib.fit_single_timescale) returns (a, nu) per
principal component; the matched single-timescale null at the fitted (a, nu) gives the 95th
percentile of the profile slope, and p_second_timescale_rule is the fraction of component slopes
above it.

Arms (each at 1.0-s windows x 60 and 2.0-s windows x 30, 12 blocks per synthetic recording):
  same_a08_step, same_a08_continuous, same_a05_step   all eight features share one a
  bandwise_spectrum                                  delta slow ... beta fast
  same_a08_overlap50, same_a08_overlap75             the same_a08_step arm with overlapping windows

Supports the white-share band of Fig. 3b: arm same_a08_step at 1.0-s windows, summary.state_pc
(nu_fit 0.643, exponential index E 0.613).

Inputs: none (synthetic; seed 20260922).
Outputs: figure_reproduction/inputs/white_share_control_simulation.json (per-arm summaries) and
         analyses/rate_identifiability/out/white_share_control_rows.csv (per-feature and per-component
         rows).
Runtime: about one minute on one core.

    python analyses/rate_identifiability/white_share_pipeline_control.py --reps 40 --n-blocks 12
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse                                                                   # noqa: E402
import csv                                                                        # noqa: E402
import json                                                                       # noqa: E402
import sys                                                                        # noqa: E402
import time                                                                       # noqa: E402
import warnings                                                                   # noqa: E402
from pathlib import Path                                                          # noqa: E402

import numpy as np                                                                # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)    # empty-slice means of all-NaN profiles
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyses.common import paths                                                 # noqa: E402
from lag_lib import (BANDS, SFREQ, acf_ou_white, build_state_space, chol_from_acf,  # noqa: E402
                     features_from_signal, fit_single_timescale, gen_blocks, n_bins_per_band,
                     pooled_acf_many, pooled_acf_ragged, profile_many, single_timescale_grid,
                     synth_recording)

SEED = 20260922
KMAX = 6


# ------------------------------------------------------------- measurement
def profile_of(series_blocks, kmax=KMAX):
    r = pooled_acf_ragged(series_blocks, kmax)
    P = profile_many(r[None, :])
    return r, {k: (float(v[0]) if np.ndim(v[0]) == 0 else np.asarray(v[0]).tolist())
               for k, v in P.items()}


def run_arm(name, a_per_band, sd_per_band, n_blocks, n_win, win_s, reps, rng,
            envelope="step", overlap=0.0, grid=None):
    """One arm: `reps` synthetic recordings, each with `n_blocks` bouts."""
    feat_rows, state_rows = [], []
    per_rec = []
    for rep in range(reps):
        blocks_F, blocks_U = [], []
        for _ in range(n_blocks):
            sig, U = synth_recording(a_per_band, sd_per_band, n_win, win_s, rng,
                                     envelope=envelope)
            F, nw = features_from_signal(sig, win_s, overlap=overlap)
            blocks_F.append(F)
            blocks_U.append(U.reshape(U.shape[0], -1))
        # ---- feature level: observed vs injected, per feature
        n_feat = blocks_F[0].shape[1]
        fr = []
        aligned = (overlap == 0.0)
        for j in range(n_feat):
            r_obs, p_obs = profile_of([b[:, j] for b in blocks_F])
            if aligned:
                r_inj, p_inj = profile_of([b[:, j] for b in blocks_U])
            else:
                r_inj = np.full(KMAX, np.nan)
                p_inj = dict(slope=np.nan, exp_index=np.nan)
            a_f, nu_f, res_f, _ = fit_single_timescale(r_obs, grid)
            a_i, nu_i, _, _ = fit_single_timescale(r_inj, grid)
            with np.errstate(invalid="ignore", divide="ignore"):
                C = np.where(np.abs(r_inj) > 0.02, r_obs / r_inj, np.nan)
            fr.append(dict(feature=j, r_obs=r_obs.tolist(), r_inj=r_inj.tolist(),
                           C_profile=np.asarray(C, float).tolist(),
                           C_mean=float(np.nanmean(C)), C_sd=float(np.nanstd(C)),
                           slope_obs=p_obs["slope"], slope_inj=p_inj["slope"],
                           E_obs=p_obs["exp_index"], E_inj=p_inj["exp_index"],
                           a_fit_obs=a_f, nu_fit_obs=nu_f, a_fit_inj=a_i,
                           nu_fit_inj=nu_i, single_timescale_resid_obs=res_f))
        # ---- state level: robust z + PCA(d = 4), then projections
        pooled = {"all": np.vstack(blocks_F)}
        transform, evr = build_state_space(pooled)
        Z = [transform(b) for b in blocks_F]
        d = Z[0].shape[1]
        sr = []
        for pc in range(d):
            r_obs, p_obs = profile_of([z[:, pc] for z in Z])
            a_f, nu_f, res_f, _ = fit_single_timescale(r_obs, grid)
            sr.append(dict(proj=f"PC{pc+1}", r=r_obs.tolist(), slope=p_obs["slope"],
                           E=p_obs["exp_index"], a_fit=a_f, nu_fit=nu_f,
                           single_timescale_resid=res_f, evr=float(evr[pc])))
        R = rng.standard_normal((64, d))
        R /= np.linalg.norm(R, axis=1, keepdims=True)
        sl, ee, af, nf = [], [], [], []
        for q in range(R.shape[0]):
            r_obs, p_obs = profile_of([z @ R[q] for z in Z])
            a_f, nu_f, _, _ = fit_single_timescale(r_obs, grid)
            sl.append(p_obs["slope"]); ee.append(p_obs["exp_index"])
            af.append(a_f); nf.append(nu_f)
        sr.append(dict(proj="random_mean", slope=float(np.nanmean(sl)),
                       E=float(np.nanmean(ee)), a_fit=float(np.nanmean(af)),
                       nu_fit=float(np.nanmean(nf)), n_dir=int(R.shape[0])))
        feat_rows.append(fr)
        state_rows.append(sr)
        per_rec.append(dict(rep=rep, evr=np.asarray(evr, float).tolist()))
    return dict(arm=name, envelope=envelope, overlap=overlap, win_s=win_s,
                n_blocks=n_blocks, n_win=n_win, reps=reps,
                a_per_band=a_per_band, sd_per_band=sd_per_band,
                bins=n_bins_per_band(SFREQ, win_s),
                features=feat_rows, states=state_rows, per_rec=per_rec)


def matched_null_state(a_h, nu_h, n_blocks, n_win, reps, rng, grid, kmax=KMAX):
    """The decision rule's matched single-timescale null, at the STATE geometry of this arm.

    One draw = one synthetic recording of `n_blocks` blocks of `n_win` windows,
    i.e. exactly the unit the arm's own slope is computed on.
    """
    a_h = float(np.clip(a_h, 0.02, 0.985))
    nu_h = float(np.clip(nu_h, 0.0, 0.92))
    L, _ = chol_from_acf(acf_ou_white(a_h, nu_h, n_win), n_win)
    X = gen_blocks(L, max(400, reps * 4), n_blocks, n_win, rng)
    P = profile_many(pooled_acf_many(X, kmax))
    sl = np.asarray(P["slope"], float)
    ei = np.asarray(P["exp_index"], float)
    return dict(a=a_h, nu=nu_h,
                slope_q95=float(np.nanpercentile(sl, 95)),
                slope_mean=float(np.nanmean(sl)),
                slope_lo=float(np.nanpercentile(sl, 2.5)),
                slope_hi=float(np.nanpercentile(sl, 97.5)),
                E_mean=float(np.nanmean(ei)),
                E_lo=float(np.nanpercentile(ei, 2.5)),
                E_hi=float(np.nanpercentile(ei, 97.5)), n_draw=int(len(sl)))


def agg(rows, key):
    v = np.array([[x[key] for x in rec] for rec in rows], float)
    with np.errstate(invalid="ignore"):
        return dict(mean=float(np.nanmean(v)), lo=float(np.nanpercentile(v, 2.5)),
                    hi=float(np.nanpercentile(v, 97.5)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--reps", type=int, default=40)
    ap.add_argument("--n-blocks", type=int, default=12)
    a = ap.parse_args()
    rng = np.random.default_rng(SEED)
    t0 = time.time()
    res = dict(generated=time.strftime("%Y-%m-%d %H:%M:%S"), seed=SEED,
               sfreq=SFREQ, bands={k: list(v) for k, v in BANDS.items()},
               reps=a.reps, arms=[])
    rows = []

    BN = list(BANDS)
    # injected amplitude: the sleep recordings' window-to-window log10 sd minus the
    # band-power estimator's own noise:
    #   delta 0.45 total / 0.368 estimator -> 0.26 true;  beta 0.21 / 0.168 -> 0.12
    SD_REAL = dict(delta=0.26, theta=0.22, alpha=0.22, beta=0.12)

    for win_s, n_win in ((1.0, 60), (2.0, 30)):
        for arm, a_map, env, ov in (
            ("same_a08_step",      {b: 0.80 for b in BN}, "step", 0.0),
            ("same_a08_continuous", {b: 0.80 for b in BN}, "continuous", 0.0),
            ("same_a05_step",      {b: 0.50 for b in BN}, "step", 0.0),
            ("bandwise_spectrum",  dict(delta=0.93, theta=0.85, alpha=0.70, beta=0.40),
             "step", 0.0),
            ("same_a08_overlap50", {b: 0.80 for b in BN}, "step", 0.5),
            ("same_a08_overlap75", {b: 0.80 for b in BN}, "step", 0.75),
        ):
            grid = single_timescale_grid(n_win if ov == 0 else int(n_win / (1 - ov)) - 1, KMAX)
            r = run_arm(arm, a_map, SD_REAL, a.n_blocks, n_win, win_s, a.reps, rng,
                        envelope=env, overlap=ov, grid=grid)
            # ---- summaries
            fmean = {}
            for k in ("C_mean", "C_sd", "slope_obs", "E_obs", "a_fit_obs",
                      "nu_fit_obs", "slope_inj", "E_inj", "a_fit_inj", "nu_fit_inj"):
                fmean[k] = agg(r["features"], k)
            smean = {}
            for k in ("slope", "E", "a_fit", "nu_fit"):
                smean[k] = agg([[x for x in rec if x["proj"].startswith("PC")]
                                for rec in r["states"]], k)
            rmean = {}
            for k in ("slope", "E", "a_fit", "nu_fit"):
                rmean[k] = agg([[x for x in rec if x["proj"] == "random_mean"]
                                for rec in r["states"]], k)
            nw_eff = n_win if ov == 0 else int(n_win / (1 - ov)) - 1
            nul = matched_null_state(smean["a_fit"]["mean"], smean["nu_fit"]["mean"],
                                     a.n_blocks, nw_eff, a.reps, rng, grid)
            pc_slopes = np.array([[x["slope"] for x in rec if x["proj"].startswith("PC")]
                                  for rec in r["states"]], float).ravel()
            p_second = float(np.mean(pc_slopes > nul["slope_q95"]))
            r["summary"] = dict(feature=fmean, state_pc=smean, state_random=rmean,
                                matched_null=nul, p_second_timescale_rule=p_second,
                                E_excess_over_matched_null=float(smean["E"]["mean"]
                                                                 - nul["E_mean"]))
            # per-unit rows go to the CSV, not the JSON
            for rep_i, (fr, sr) in enumerate(zip(r["features"], r["states"])):
                for x in fr:
                    rows.append(dict(win_s=win_s, arm=arm, level="feature", rep=rep_i,
                                     unit=f"f{x['feature']}",
                                     band=BN[x["feature"] % len(BN)],
                                     slope=x["slope_obs"], E=x["E_obs"],
                                     a_fit=x["a_fit_obs"], nu_fit=x["nu_fit_obs"],
                                     slope_inj=x["slope_inj"], E_inj=x["E_inj"],
                                     a_fit_inj=x["a_fit_inj"], nu_fit_inj=x["nu_fit_inj"],
                                     C_mean=x["C_mean"], C_sd=x["C_sd"],
                                     r_obs=";".join(f"{y:.4f}" for y in x["r_obs"]),
                                     r_inj=";".join(f"{y:.4f}" for y in x["r_inj"])))
                for x in sr:
                    rows.append(dict(win_s=win_s, arm=arm, level="state", rep=rep_i,
                                     unit=x["proj"], band="", slope=x["slope"],
                                     E=x["E"], a_fit=x["a_fit"], nu_fit=x["nu_fit"],
                                     slope_inj="", E_inj="", a_fit_inj="",
                                     nu_fit_inj="", C_mean="", C_sd="",
                                     r_obs=";".join(f"{y:.4f}" for y in x["r"])
                                     if "r" in x else "", r_inj=""))
            r.pop("features"); r.pop("states"); r.pop("per_rec")
            res["arms"].append(r)
            print(f"[{time.time()-t0:6.1f}s] win={win_s} {arm:22s} "
                  f"feat C={fmean['C_mean']['mean']:.3f}+-{fmean['C_sd']['mean']:.3f} "
                  f"nu={fmean['nu_fit_obs']['mean']:.3f} E={fmean['E_obs']['mean']:.3f} "
                  f"| state slope={smean['slope']['mean']:+.4f} "
                  f"E={smean['E']['mean']:.3f} a={smean['a_fit']['mean']:.3f} "
                  f"nu={smean['nu_fit']['mean']:.3f} | null slope q95={nul['slope_q95']:+.4f} "
                  f"P(second timescale)={p_second:.2f}", flush=True)

    out_json = paths.out("figure_reproduction", "inputs", "white_share_control_simulation.json")
    with open(out_json, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(res, indent=1, default=float, ensure_ascii=False) + "\n")
    out_rows = paths.out("analyses", "rate_identifiability", "out", "white_share_control_rows.csv")
    keys = sorted({k for r_ in rows for k in r_})
    with open(out_rows, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"done in {(time.time()-t0)/60:.1f} min -> {out_json}, {out_rows}", flush=True)


if __name__ == "__main__":
    main()
