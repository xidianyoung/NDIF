"""Summaries of the full-corpus assessment written by benchmark_05_corpus_full.py.

Reads every per-task JSON under ``out/corpus_full/`` (next to this script) and writes

  corpus_full.log            the stage x estimator table, plus the level-1 arms
  corpus_full_summary.json   per (stage, estimator): n, admitted/refused/indeterminate counts
                             per level, binding-gate tally for L3, median G3 rate recovery,
                             median detection margin fold, and the per-cell verdict rows
  corpus_full_L1.json        rows (G6 per cell) and stage_pairs (G5)

and, to ``figure_reproduction/inputs`` (``--figure-inputs``),

  sleep_detection_thresholds.json   per (estimator, stage): L3 verdicts, first failed gates of the refused cells,
                                    median G3 rate recovery and median detection thresholds (all cells and refused
                                    cells), and the number of refused cells with a finite or undefined threshold
                                    (Fig. 5c,d)
  adjacent_stage_pairs_index.json   the 336 adjacent-stage pair assessments (file name and SHA-256)
                                    (Supplementary Fig. 2)

    python analyses/primary_configuration/summarize_corpus_full.py [--in-dir ...]
"""

from __future__ import annotations

import argparse
import collections
import glob
import hashlib
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
FIGURE_INPUTS = os.path.join(REPO, "figure_reproduction", "inputs")
STAGES = ["Wake", "N1", "N2", "N3", "N4", "REM"]
PAIR_ORDER = ["Wake vs N1", "N1 vs N2", "N2 vs N3", "N3 vs N4"]


def load(in_dir):
    cells, pairs, pair_index = [], [], []
    for f in sorted(glob.glob(os.path.join(in_dir, "*.json"))):
        if os.path.basename(f).startswith("_"):
            continue
        with open(f, "rb") as fh:
            raw = fh.read()
        d = json.loads(raw.decode("utf-8"))
        (pairs if d.get("kind") == "pair" else cells).append(d)
        if d.get("kind") == "pair":
            pair_index.append(dict(name=os.path.basename(f), sha256=hashlib.sha256(raw).hexdigest()))
    return cells, pairs, pair_index


def gate(d, name):
    for g in d["gates"]:
        if g["gate"] == name:
            return g
    return None


def med(xs):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return float(np.median(xs)) if xs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default=os.path.join(HERE, "out", "corpus_full"))
    ap.add_argument("--out-dir", default=os.path.join(HERE, "out"))
    ap.add_argument("--figure-inputs", default=FIGURE_INPUTS)
    a = ap.parse_args()
    cells, pairs, pair_index = load(a.in_dir)
    ests = sorted({c["estimator"] for c in cells}, key=lambda e: (e != "ols", e))
    nights = sorted({c["recording"] for c in cells})
    subjects = sorted({r[:5] for r in nights})          # SC4ssN -> SC4ss (participant)

    source = os.path.relpath(os.path.abspath(a.in_dir), REPO).replace(os.sep, "/")
    summary = dict(source=source, n_nights=len(nights), n_participants=len(subjects),
                   n_cells=len({(c["recording"], c["stage"]) for c in cells}),
                   estimators=ests, by_stage={}, rows=[])
    lines = ["=" * 78, "NDIF certificates on Sleep-EDF cassette, all primary nights, by scored stage",
             f"{len(nights)} nights, {len(subjects)} participants, "
             f"{summary['n_cells']} night-stage cells (>= 6 epochs), package default replicates",
             "=" * 78,
             f"{'stage':>6} {'estimator':>10} {'n':>4} {'L2 adm':>8} {'L3 adm':>8} "
             f"{'median G3':>10} {'MDE fold':>9}  binding gates for L3 (refused cells)",
             "-" * 78]
    for st in STAGES:
        for est in ests:
            cc = [c for c in cells if c["stage"] == st and c["estimator"] == est]
            if not cc:
                continue
            n = len(cc)
            tally = {lvl: collections.Counter(c["levels"][lvl]["status"] for c in cc)
                     for lvl in ("L1", "L2", "L3")}
            binding = collections.Counter(c["levels"]["L3"]["binding_gate"] for c in cc
                                          if c["levels"]["L3"]["status"] == "refused")
            admitted_gate = collections.Counter(c["levels"]["L3"]["binding_gate"] for c in cc
                                                if c["levels"]["L3"]["status"] == "admitted")
            g3v = [gate(c, "G3")["value"] if gate(c, "G3") else None for c in cc]
            mde = [(gate(c, "G3")["detail"] or {}).get("detection_margin_fold") if gate(c, "G3") else None
                   for c in cc]
            mde_ref = [(gate(c, "G3")["detail"] or {}).get("detection_margin_fold")
                       for c in cc if c["levels"]["L3"]["status"] == "refused" and gate(c, "G3")]
            nu = [gate(c, "G1")["value"] if gate(c, "G1") else None for c in cc]
            rec = dict(stage=st, estimator=est, n=n,
                       L1=dict(tally["L1"]), L2=dict(tally["L2"]), L3=dict(tally["L3"]),
                       L3_admitted_frac=tally["L3"]["admitted"] / n,
                       L2_admitted_frac=tally["L2"]["admitted"] / n,
                       binding_gates_L3_refused=dict(binding),
                       tightest_gate_L3_admitted=dict(admitted_gate),
                       median_g3_recovery=med(g3v),
                       median_g3_recovery_admitted=med([g for g, c in zip(g3v, cc)
                                                        if c["levels"]["L3"]["status"] == "admitted"]),
                       median_detection_margin_fold=med(mde),
                       median_detection_margin_fold_refused=med(mde_ref),
                       median_white_share_G1=med(nu),
                       n_participants=len({c["recording"][:5] for c in cc}))
            summary["by_stage"][f"{st}|{est}"] = rec
            gates = ", ".join(f"{k} x{v}" for k, v in binding.most_common(3))
            lines.append(f"{st:>6} {est:>10} {n:>4} {rec['L2_admitted_frac']:>7.0%} "
                         f"{rec['L3_admitted_frac']:>7.0%} "
                         f"{(rec['median_g3_recovery'] if rec['median_g3_recovery'] is not None else float('nan')):>10.3f} "
                         f"{(rec['median_detection_margin_fold_refused'] if rec['median_detection_margin_fold_refused'] is not None else float('nan')):>9.2f}  {gates}")
    lines.append("-" * 78)
    # totals
    for est in ests:
        cc = [c for c in cells if c["estimator"] == est]
        n = len(cc)
        adm = sum(c["levels"]["L3"]["status"] == "admitted" for c in cc)
        l2 = sum(c["levels"]["L2"]["status"] == "admitted" for c in cc)
        binding = collections.Counter(c["levels"]["L3"]["binding_gate"] for c in cc
                                      if c["levels"]["L3"]["status"] == "refused")
        summary[f"total|{est}"] = dict(n=n, L3_admitted=adm, L2_admitted=l2,
                                       binding_gates_L3_refused=dict(binding))
        lines.append(f"{'all':>6} {est:>10} {n:>4} {l2/n:>7.0%} {adm/n:>7.0%}  "
                     + ", ".join(f"{k} x{v}" for k, v in binding.most_common(4)))
    for c in cells:
        g3 = gate(c, "G3"); g1 = gate(c, "G1"); g2 = gate(c, "G2")
        summary["rows"].append(dict(
            recording=c["recording"], participant=c["recording"][:5], stage=c["stage"],
            estimator=c["estimator"], n_segments=c["n_segments"],
            L1=c["levels"]["L1"]["status"], L2=c["levels"]["L2"]["status"],
            L3=c["levels"]["L3"]["status"], L3_binding=c["levels"]["L3"]["binding_gate"],
            g1_nu=(g1["value"] if g1 else None), g2_theta=(g2["value"] if g2 else None),
            g3_recovery=(g3["value"] if g3 else None),
            detection_margin_fold=((g3["detail"] or {}).get("detection_margin_fold") if g3 else None)))

    # ---- level 1: G6 per cell (ols pass), G5 on stage pairs
    l1_rows, by_st = [], collections.defaultdict(list)
    for c in cells:
        if c["estimator"] != "ols":
            continue
        g6, g5 = gate(c, "G6"), gate(c, "G5")
        row = dict(recording=c["recording"], stage=c["stage"], n_segments=c["n_segments"],
                   L1=c["levels"]["L1"]["status"], binding_gate=c["levels"]["L1"]["binding_gate"],
                   g6_tpr=((g6["detail"] or {}).get("tpr") if g6 else None),
                   g6_fpr=((g6["detail"] or {}).get("fpr") if g6 else None),
                   g6_passed=(g6["passed"] if g6 else None),
                   g5_passed=(g5["passed"] if g5 else None))
        l1_rows.append(row); by_st[c["stage"]].append(row)
    pair_rows = [dict(recording=p["recording"], contrast=p["contrast"], status=p["status"],
                      passed=p["passed"], separation=p["separation"], floor=p["floor"],
                      margin=p["margin"]) for p in pairs]
    l1_summary = {}
    lines += ["", "level 1, G6 per cell (estimator-free, from the ols pass):",
              f"{'stage':>6} {'n':>4} {'G6 cleared':>11} {'median TPR':>11} {'median FPR':>11}"]
    for st in STAGES:
        rs = by_st.get(st, [])
        if not rs:
            continue
        l1_summary[st] = dict(n=len(rs), g6_cleared=sum(bool(r["g6_passed"]) for r in rs),
                              L1=dict(collections.Counter(r["L1"] for r in rs)),
                              median_tpr=med([r["g6_tpr"] for r in rs]),
                              median_fpr=med([r["g6_fpr"] for r in rs]))
        s = l1_summary[st]
        lines.append(f"{st:>6} {s['n']:>4} {s['g6_cleared']:>11} "
                     f"{(s['median_tpr'] if s['median_tpr'] is not None else float('nan')):>11.3f} "
                     f"{(s['median_fpr'] if s['median_fpr'] is not None else float('nan')):>11.3f}")
    lines += ["", "level 1, G5 on adjacent stage pairs within a night:"]
    pair_summary = {}
    for k in PAIR_ORDER:
        ps = [p for p in pair_rows if p["contrast"] == k]
        if not ps:
            continue
        seps = [p["separation"] for p in ps if p["separation"] is not None]
        floors = [p["floor"] for p in ps if isinstance(p["floor"], (int, float))]
        pair_summary[k] = dict(n=len(ps), cleared=sum(p["status"] == "cleared" for p in ps),
                               median_separation=med(seps), median_floor=med(floors),
                               separation_range=[float(min(seps)), float(max(seps))] if seps else None,
                               floor_range=[float(min(floors)), float(max(floors))] if floors else None)
        s = pair_summary[k]
        lines.append(f"  {k:<10} n {s['n']:>4}  cleared {s['cleared']:>4}  "
                     f"median separation {s['median_separation']:.3f}  median floor {s['median_floor']:.3f}")
    n_pass = sum(p["status"] == "cleared" for p in pair_rows)
    lines.append(f"  all pairs: G5 cleared in {n_pass} of {len(pair_rows)} contrasts")

    # ---- detection thresholds per stage and estimator (Fig. 5c,d)
    thresholds = []
    for est in ests:
        for st in STAGES:
            cc = [c for c in cells if c["stage"] == st and c["estimator"] == est]
            if not cc:
                continue
            ref = [c for c in cc if c["levels"]["L3"]["status"] == "refused"]

            def mdf(c):
                return (gate(c, "G3")["detail"] or {}).get("detection_margin_fold") if gate(c, "G3") else None

            finite = [mdf(c) for c in ref if mdf(c) is not None and np.isfinite(mdf(c))]
            thresholds.append(dict(
                stage=st, estimator=est, n=len(cc),
                L3=dict(collections.Counter(c["levels"]["L3"]["status"] for c in cc)),
                first_refused=dict(collections.Counter(c["levels"]["L3"]["binding_gate"] for c in ref)),
                median_g3_recovery_all_cells=med([gate(c, "G3")["value"] if gate(c, "G3") else None for c in cc]),
                median_detection_threshold_all_cells=med([mdf(c) for c in cc]),
                median_detection_threshold_refused_cells=med([mdf(c) for c in ref]),
                n_threshold_finite_refused=len(finite),
                n_threshold_undefined_refused=len(ref) - len(finite)))

    os.makedirs(a.out_dir, exist_ok=True)
    os.makedirs(a.figure_inputs, exist_ok=True)
    with open(os.path.join(a.figure_inputs, "sleep_detection_thresholds.json"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(dict(stages=thresholds), indent=1) + "\n")
    with open(os.path.join(a.figure_inputs, "adjacent_stage_pairs_index.json"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(pair_index, indent=2))
    with open(os.path.join(a.out_dir, "corpus_full.log"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")
    with open(os.path.join(a.out_dir, "corpus_full_summary.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(summary, fh, indent=1)
    with open(os.path.join(a.out_dir, "corpus_full_L1.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(dict(rows=l1_rows, stage_pairs=pair_rows, summary=l1_summary,
                       pair_summary=pair_summary, n_cells=len(l1_rows)), fh, indent=1)
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
