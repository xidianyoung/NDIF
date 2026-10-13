# -*- coding: utf-8 -*-
r"""Sleep-EDF instrument-noise covariance: the covariance of the log band power of phase-randomised
copies of the same bouts, per recording, stage and window length.

If the observed state is y_t = x_t + v_t with v white and independent of x, then
C0_y = Sigma_x + R and C1_y = M Sigma_x, so with R measured independently of the dynamics the
propagator is identified from lag 0 and lag 1, M = C1_y (C0_y - R)^{-1}
(sleep_common.nc_propagator; the `lam_mean_nc` / `lam_mean_ncbc` rates of stage_contrasts.py).

For every bout used by the window cache, the same 80-s span of the same recording is phase-
randomised (one common phase draw for the two channels, so each channel keeps its spectrum and the
pair keeps its cross-spectrum; `--draws` copies per bout), pushed through the band-power call of
window_cache.py at each window length, and the 8 x 8 covariance of the log band power after the
first 20 s is accumulated per stage.  The surrogate is Gaussian; real EEG has heavier tails, so R
is a lower bound on the estimator noise.

Random stream: one generator per recording, seeded with Python's hash of ("noise", <stem>).

Inputs   Sleep-EDF Expanded EDF files and hypnograms in NDIF_SLEEPEDF_RAW.
Outputs  paths.cache("sleep_states")/<stem>_noise.npz: "<stage>|<w>" -> 8 x 8 covariance for
         w = 2, 1, 0.5 (NDIF_SLEEP_STATES_CACHE, default analyses/_work/sleep_states).
Used by  stage_contrasts.py (noise-corrected rates: Fig. 5e, Supplementary Table 11) and
         ../rate_identifiability.
Runtime  about 1.2 min with 4 workers (197 recordings); existing files are skipped.

    python analyses/sleep_stage_contrasts/noise_cache.py --workers 4 --draws 2
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse                                                       # noqa: E402
import sys                                                            # noqa: E402
import time                                                           # noqa: E402
import traceback                                                      # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed     # noqa: E402
from pathlib import Path                                              # noqa: E402

import numpy as np                                                    # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import window_cache as CA  # noqa: E402

CACHE = CA.CACHE
WINDOWS = CA.WINDOWS
SPAN_S = CA.SKIP_S + CA.USE_S


def mv_phase_randomise(x, rng):
    """Multivariate phase randomisation: one phase draw shared by both channels,
    so each channel keeps its own spectrum and the pair keeps its cross-spectrum."""
    n = x.shape[-1]
    F = np.fft.rfft(x, axis=-1)
    ph = rng.uniform(0, 2 * np.pi, size=F.shape[-1])
    ph[0] = 0.0
    if n % 2 == 0:
        ph[-1] = 0.0
    return np.fft.irfft(F * np.exp(1j * ph)[None, :], n=n, axis=-1)


def run_one(args):
    stem, psg, hyp, draws = args
    out_p = CACHE / f"{stem}_noise.npz"
    if out_p.exists():
        return dict(stem=stem, cached=True)
    rng = np.random.default_rng(abs(hash(("noise", stem))) % (2 ** 31))
    eeg, sfreq, chans, epochs = CA.load_eeg(psg, hyp)
    if chans[:2] != list(CA.EEG_CHANNELS):
        return dict(stem=stem, error="channel mismatch")
    bouts = CA.usable_bouts(epochs)
    acc = {}
    for onset, dur, g in bouts:
        i0 = int(round(onset * sfreq))
        i1 = i0 + int(round(SPAN_S * sfreq))
        if i1 > eeg.shape[1]:
            continue
        seg = eeg[:, i0:i1]
        for _ in range(draws):
            sur = mv_phase_randomise(seg, rng)
            for win_s in WINDOWS:
                P = CA.band_power(sur, sfreq, win_s)
                n_skip = int(round(CA.SKIP_S / win_s))
                if len(P) < n_skip + 8:
                    continue
                F = np.log10(np.maximum(P, 1e-30)).reshape(P.shape[0], -1)[n_skip:]
                if not np.isfinite(F).all():
                    continue
                Fc = F - F.mean(axis=0)
                key = (g, win_s)
                if key not in acc:
                    acc[key] = [np.zeros((8, 8)), 0]
                acc[key][0] += Fc.T @ Fc
                acc[key][1] += len(Fc) - 1
    if not acc:
        return dict(stem=stem, error="no bouts")
    payload, summary = {}, {}
    for (g, w), (S, n) in acc.items():
        R = S / max(n, 1)
        payload[f"{g}|{w:g}"] = R.astype(np.float64)
        summary[f"{g}|{w:g}"] = dict(trR=float(np.trace(R)), n=int(n))
    np.savez_compressed(out_p, **payload)
    return dict(stem=stem, summary=summary, draws=draws)


def _worker(a):
    try:
        return run_one(a)
    except Exception:  # noqa: BLE001
        return dict(stem=a[0], error=traceback.format_exc(limit=3))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--draws", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None,
                    help="only the first N recordings (a check run)")
    a = ap.parse_args()
    recs = CA.pair_recordings("cassette") + CA.pair_recordings("telemetry")
    if a.limit:
        recs = recs[: a.limit]
    print(f"instrument-noise covariance on {len(recs)} recordings, "
          f"{a.draws} surrogate draws per bout", flush=True)
    t0, done, errs = time.time(), 0, []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(_worker, (s, p, h, a.draws)): s for s, p, h in recs}
        for k, f in enumerate(as_completed(futs), 1):
            r = f.result()
            if "error" in r:
                errs.append(r["stem"])
            else:
                done += 1
            if k % 25 == 0 or k == len(recs):
                print(f"  {k}/{len(recs)}  {(time.time()-t0)/60:.1f} min", flush=True)
    print(f"done {done}, errors {len(errs)} {errs[:5]}  "
          f"({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
