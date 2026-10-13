"""Human-readable rendering of a certificate."""

from __future__ import annotations

import numpy as np

from .types import ADMITTED, INDETERMINATE, LEVELS, REFUSED

__all__ = ["render", "render_matrix"]

_MARK = {ADMITTED: "ADMITTED", REFUSED: "REFUSED", INDETERMINATE: "INDETERMINATE"}


#: Display names for statistics longer than the 44-character column of the report.
#: The name stored in the certificate is unchanged.
_SHORT_STAT = {"G1": "half-width of the rate interval (noise)",
               "G3": "recovered fraction (rate channel)",
               "G5": "separation (Mahalanobis)",
               "G8": "combined P value (ACF and likelihood ratio)"}


def _fmt(v, width=10):
    if v is None:
        return "-".rjust(width)
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v).rjust(width)
    if not np.isfinite(f):
        return "n/a".rjust(width)
    return f"{f:.4g}".rjust(width)


def render(cert) -> str:
    """The certificate as text: four fields per level, then the gate trace."""
    out = []
    name = cert.design.get("name") or "(unnamed)"
    mod = cert.design.get("modality") or "-"
    out.append("=" * 74)
    out.append(f"NDIF identifiability certificate   {name}   [{mod}]")
    out.append("=" * 74)
    p = cert.provenance
    out.append(f"  {p.get('n_samples')} samples x {p.get('d')} dims in "
               f"{p.get('n_segments')} segments; dt = {cert.design.get('dt')} s; "
               f"cost {p.get('total_cost_s', 0):.1f} s")
    out.append("")
    out.append(f"  {'level':<5} {'status':<14} {'binding gate':<13} {'margin':>10}")
    out.append("  " + "-" * 46)
    for k in ("L1", "L2", "L3"):
        v = cert.levels[k]
        out.append(f"  {k:<5} {_MARK[v.status]:<14} {str(v.binding_gate or '-'):<13} "
                   f"{_fmt(v.margin)}")
    out.append("")
    for k in ("L1", "L2", "L3"):
        v = cert.levels[k]
        out.append(f"  {k} ({LEVELS[k]})")
        out.append(f"      grounds: {v.grounds}")
        if getattr(v, "not_run", None):
            out.append(f"      not run: {', '.join(v.not_run)} (reduced set of checks; this level cannot be admitted)")
        if v.detection_margin:
            out.append(f"      detection margin: {v.detection_margin}")
    out.append("")
    out.append(f"  {'gate':<5} {'status':<13} {'statistic':<44} {'value':>10}")
    out.append("  " + "-" * 74)
    diagnostics = set((cert.provenance or {}).get("diagnostics", ()))
    for g in cert.gates:
        stat = g.statistic if len(g.statistic) <= 44 else _SHORT_STAT.get(g.gate, g.statistic[:44])
        status = "diagnostic" if g.gate in diagnostics else g.status
        out.append(f"  {g.gate:<5} {status:<13} {stat:<44} {_fmt(g.value)}")
        if g.note:
            out.append(f"        note: {g.note}")
    out.append("=" * 74)
    return "\n".join(out)


def render_matrix(certs, labels=None) -> str:
    """A levels x recordings matrix, the compact form used for a corpus.

    ``certs`` is a sequence of certificates; ``labels`` their column names.
    Each cell shows the verdict and the binding gate.
    """
    certs = list(certs)
    if labels is None:
        labels = [c.design.get("name") or f"#{i}" for i, c in enumerate(certs)]
    w = max(12, max(len(str(x)) for x in labels) + 2)
    head = "  " + "level".ljust(6) + "".join(str(x).ljust(w) for x in labels)
    lines = [head, "  " + "-" * (6 + w * len(certs))]
    short = {ADMITTED: "admit", REFUSED: "refuse", INDETERMINATE: "indet"}
    for k in ("L1", "L2", "L3"):
        row = "  " + k.ljust(6)
        for c in certs:
            v = c.levels[k]
            row += f"{short[v.status]}/{v.binding_gate or '-'}".ljust(w)
        lines.append(row)
    return "\n".join(lines)
