"""Quickstart: one certificate, and one quasipotential barrier.

    python examples/example_01_quickstart.py

Part 1 certifies a short two-condition recording end to end, including the
level-1 gates, so the printed certificate has all three levels filled in.

Part 2 computes a Freidlin-Wentzell barrier on a system whose exact answer is
known; the translation-only descent kept as a regression case in
tests/test_action.py cannot reach it.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # the repository root

from ndif import Design, certify, minimum_action
from ndif.simulate import bistable_2d, simulate_var1


def part1_certificate():
    print("=" * 74)
    print("PART 1  -  a certificate for a two-condition recording")
    print("=" * 74)
    rng = np.random.default_rng(0)
    d, n_seg, n_blk = 3, 40, 150

    # two conditions that differ in where the state sits (a level-1 effect)
    # and not at all in how fast it relaxes (no level-3 effect)
    A = np.diag([0.88, 0.75, 0.60])
    Q = np.eye(d) * 0.25
    shift = np.array([0.9, 0.0, 0.0])

    segs, labels = [], []
    for i in range(n_seg):
        s = simulate_var1(A, Q, 1, n_blk, rng=int(rng.integers(1 << 30)))[0]
        cond = i % 2
        if cond == 1:
            s = s + shift
        segs.append(s)
        labels.append(np.full(n_blk, cond))

    X = np.concatenate(segs, axis=0)
    lab = np.concatenate(labels)
    edges = np.cumsum([0] + [len(s) for s in segs])
    bounds = [(edges[i], edges[i + 1]) for i in range(len(segs))]

    cert = certify(X, Design(dt=0.5, window_s=0.5, step_s=0.5, segments=bounds,
                             labels=lab, name="demo-2cond", modality="synthetic"),
                   rng=1, G3=dict(n_rep=12), G4=dict(n_rep=20),
                   G6=dict(n_rep=30), G7=dict(n_rep=60))
    print(cert)
    print()
    print("Read it as: the displacement between conditions is certified (L1),")
    print("and any claim about relaxation would be certified separately (L3).")
    print("The two are different questions and the certificate keeps them apart.")
    return cert


def part2_barrier():
    print()
    print("=" * 74)
    print("PART 2  -  a quasipotential barrier, against the exact answer")
    print("=" * 74)
    drift, meta = bistable_2d(barrier_scale=1.0, coupling=6.0, D=0.05)
    exact = meta["exact_barrier"]
    r = minimum_action(meta["minima"][0], meta["saddle"], drift, meta["D"] * np.eye(2),
                       n_pts=201)
    print(f"  system      : double well, curved instanton, D = {meta['D']}")
    print(f"  exact       : {exact:.6f}")
    print(f"  recovered   : {r.action:.6f}   (relative error {abs(r.action - exact) / exact:.2e})")
    print(f"  path max|y| : {abs(r.path[:, 1]).max():.4f}   (the instanton bends to y = -1)")
    print()
    print("  A descent whose gradient is constant along the path cannot bend,")
    print("  so it cannot reach this barrier at all; see tests/test_action.py")
    print("  ::test_translation_only_descent_fails_where_ours_succeeds.")
    return r


if __name__ == "__main__":
    part1_certificate()
    part2_barrier()
