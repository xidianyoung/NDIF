"""Does the framework discriminate, or does it refuse everything?

Five regimes from one generator, one property changed at a time.  Reproduces
the table in docs/PACKAGE_USAGE.md.  Runs in about a minute on one core.

    python examples/example_02_discrimination.py
"""

import time

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # the repository root

from ndif import Design, certify
from ndif.simulate import add_white, simulate_var1

A = np.diag([0.90, 0.80, 0.65])
Q = np.eye(3) * 0.3

CASES = [
    # name                     nu     n_seg n_blk window step
    ("clean, long blocks",     0.00,  60,   400,  1.0,   1.0),
    ("additive white nu=0.25", 0.25,  60,   400,  1.0,   1.0),
    ("additive white nu=0.50", 0.50,  60,   400,  1.0,   1.0),
    ("50% window overlap",     0.00,  60,   400,  2.0,   1.0),
    ("8-sample blocks",        0.00, 300,     8,  1.0,   1.0),
]


def build(nu, n_seg, n_blk, seed=7):
    segs = simulate_var1(A, Q, n_seg, n_blk, rng=seed)
    if nu > 0:
        segs = add_white(segs, nu, rng=seed + 1)
    X = np.concatenate(segs, axis=0)
    edges = np.cumsum([0] + [len(s) for s in segs])
    return X, [(edges[i], edges[i + 1]) for i in range(len(segs))]


def main():
    t0 = time.time()
    hdr = f"{'regime':24s} {'L2':>9s} {'L3':>14s} {'bind':>5s} {'nu':>7s} {'rate':>7s} {'var':>7s}"
    print(hdr)
    print("-" * len(hdr))
    for name, nu, n_seg, n_blk, win, step in CASES:
        X, segs = build(nu, n_seg, n_blk)
        cert = certify(X, Design(dt=1.0, window_s=win, step_s=step,
                                 segments=segs, name=name),
                       rng=1, G3=dict(n_rep=12), G4=dict(n_rep=20),
                       G6=dict(n_rep=30), G7=dict(n_rep=60))
        g3 = cert.gate("G3")
        g1 = cert.gate("G1")
        print(f"{name:24s} {cert.levels['L2'].status:>9s} "
              f"{cert.levels['L3'].status:>14s} "
              f"{str(cert.levels['L3'].binding_gate or '-'):>5s} "
              f"{g1.value:7.3f} {g3.value:7.3f} "
              f"{g3.detail['variance_recovery']:7.3f}")
    print("-" * len(hdr))
    print("The variance channel recovers an injected truth at ~1.00 in every")
    print("regime while the rate channel collapses under white noise.  L2 and")
    print("L3 are not the same kind of quantity.")
    print(f"\n{time.time() - t0:.0f} s total")


if __name__ == "__main__":
    main()
