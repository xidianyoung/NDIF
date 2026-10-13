"""``ndif-certify`` -- certify a state series stored in a .npy or .npz file.

    ndif-certify states.npy --dt 1.0 --window 1.0 --step 1.0 --json cert.json

For an ``.npz`` the array is taken from ``--key`` (default ``X``); optional
``segments`` and ``labels`` arrays are read from the same file when present.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from .certify import certify
from .configurations import CONFIGURATIONS, DEFAULT_CONFIGURATION
from .types import Design


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="ndif-certify", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("path", help=".npy or .npz holding the state series")
    p.add_argument("--key", default="X", help="array name inside an .npz (default: X)")
    p.add_argument("--dt", type=float, default=1.0,
                   help="sampling interval of the STATE SERIES in seconds "
                        "(for windowed features this is the window step)")
    p.add_argument("--window", type=float, default=None, help="feature window, s")
    p.add_argument("--step", type=float, default=None, help="feature step, s")
    p.add_argument("--name", default="", help="label for the certificate header")
    p.add_argument("--modality", default="", help="e.g. EEG, MEG, fMRI")
    p.add_argument("--gates", default=None,
                   help="comma-separated subset, e.g. G0,G1,G2")
    p.add_argument("--json", default=None, help="write the certificate here")
    p.add_argument("--seed", type=int, default=0, help="random stream of the gate simulations (default: 0)")
    p.add_argument("--configuration", default=DEFAULT_CONFIGURATION, choices=sorted(CONFIGURATIONS),
                   help="analysis configuration (gates run, governing order and gate settings)")
    p.add_argument("--quick", action="store_true",
                   help="fewer replicates; for a first look, not for a result")
    a = p.parse_args(argv)

    segments = labels = None
    if a.path.endswith(".npz"):
        z = np.load(a.path, allow_pickle=False)
        if a.key not in z:
            print(f"error: {a.path} has no array {a.key!r}; found {list(z)}",
                  file=sys.stderr)
            return 2
        X = z[a.key]
        segments = z["segments"] if "segments" in z else None
        labels = z["labels"] if "labels" in z else None
    else:
        X = np.load(a.path, allow_pickle=False)

    design = Design(dt=a.dt, window_s=a.window, step_s=a.step, segments=segments,
                    labels=labels, name=a.name or a.path, modality=a.modality)
    which = [g.strip().upper() for g in a.gates.split(",")] if a.gates else None
    kw = dict(rng=a.seed)
    if a.quick:
        kw.update(G3=dict(n_rep=6), G4=dict(n_rep=10), G5=dict(n_rep=60),
                  G6=dict(n_rep=15), G7=dict(n_rep=40))
    cert = certify(X, design, which=which, configuration=a.configuration, **kw)
    print(cert)
    if a.json:
        cert.to_json(a.json)
        print(f"\nwritten: {a.json}")
    return 0


if __name__ == "__main__":                                       # pragma: no cover
    raise SystemExit(main())
