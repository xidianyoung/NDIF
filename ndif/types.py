"""Core objects: what goes into a certification, and what comes out."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Optional, Sequence

import numpy as np

__all__ = ["Design", "Recording", "GateResult", "LevelVerdict", "Certificate",
           "ADMITTED", "REFUSED", "INDETERMINATE", "LEVELS"]

ADMITTED = "admitted"
REFUSED = "refused"
INDETERMINATE = "indeterminate"

LEVELS = {
    "L1": "state geometry - where the system is",
    "L2": "state variability - how it explores",
    "L3": "intrinsic dynamics - how it evolves",
}


#: Seconds per unit of time, for :attr:`Design.time_unit` and the reference rate of the surrogate dynamics.
SECONDS_PER = {"us": 1e-6, "ms": 1e-3, "s": 1.0, "min": 60.0, "h": 3600.0, "day": 86400.0,
               "week": 604800.0, "year": 31557600.0}


def rate_in_unit(rate, rate_unit, time_unit, dt):
    """A rate given per ``rate_unit`` expressed per ``time_unit``.

    ``rate_unit="interval"`` means per sampling interval (``dt``).  A unit that
    is not in :data:`SECONDS_PER` is accepted only when it equals ``time_unit``.
    """
    rate = float(rate)
    if rate_unit == "interval":
        return rate / float(dt)
    if rate_unit == time_unit:
        return rate
    if rate_unit not in SECONDS_PER or time_unit not in SECONDS_PER:
        raise ValueError(f"cannot convert a rate per {rate_unit!r} to a rate per {time_unit!r}; "
                         f"known units: {sorted(SECONDS_PER)} and 'interval'")
    return rate / SECONDS_PER[rate_unit] * SECONDS_PER[time_unit]


@dataclass
class Design:
    """How the recording was turned into a state series.

    Every field here changes what is identifiable, which is why the certificate
    prints them rather than assuming them.

    Parameters
    ----------
    dt : float
        Sampling interval of the *state series*, in the unit ``time_unit``
        (seconds by default).  For windowed features this is the window step,
        not the raw sampling period -- the commonest source of a rate expressed
        on the wrong clock.  Rates are reported per ``time_unit``.
    window_s, step_s : float, optional
        Feature window and step.  Required by gate G0.
    labels : (n_t,) array_like, optional
        Condition label per sample, for level-1 contrasts.
    segments : optional
        Segment boundaries; see :func:`ndif.estimators.as_segments`.
    axis : (d,) array_like, optional
        Direction of interest (e.g. a task axis).  ``nu`` and rates are
        direction dependent, so the certificate records this.
    time_unit : str
        Unit of ``dt``, ``window_s`` and ``step_s``: one of :data:`SECONDS_PER`
        (default ``"s"``).  The reference rate of the surrogate dynamics of G4
        and G7 is converted to this unit, so the certificate does not depend on
        the unit in which time is written.
    """

    dt: float = 1.0
    window_s: Optional[float] = None
    step_s: Optional[float] = None
    labels: Optional[np.ndarray] = None
    segments: Optional[object] = None
    axis: Optional[np.ndarray] = None
    name: str = ""
    modality: str = ""
    time_unit: str = "s"

    def summary(self) -> dict:
        return dict(dt=self.dt, window_s=self.window_s, step_s=self.step_s,
                    time_unit=self.time_unit,
                    name=self.name, modality=self.modality,
                    has_labels=self.labels is not None,
                    has_axis=self.axis is not None)


@dataclass
class Recording:
    """A state series plus the design that produced it."""

    X: np.ndarray
    design: Design = field(default_factory=Design)

    def __post_init__(self):
        X = np.asarray(self.X, float)
        if X.ndim == 1:
            X = X[:, None]
        self.X = X

    @property
    def n_t(self) -> int:
        return self.X.shape[0]

    @property
    def d(self) -> int:
        return self.X.shape[1]

    def segments(self) -> list:
        from .estimators import as_segments
        return as_segments(self.X, self.design.segments)


@dataclass
class GateResult:
    """Outcome of one gate.

    ``passed`` is tri-state: ``True`` (the gate is cleared), ``False`` (the
    gate binds), or ``None`` (the gate could not be evaluated on this
    recording -- which is reported as *indeterminate* and never silently as a
    pass).
    """

    gate: str
    question: str
    statistic: str
    value: float
    criterion: str
    passed: Optional[bool]
    margin: float
    binds: Sequence[str]
    cost_s: float = 0.0
    detail: dict = field(default_factory=dict)
    note: str = ""

    @property
    def status(self) -> str:
        if self.passed is None:
            return INDETERMINATE
        return "cleared" if self.passed else "binding"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["status"] = self.status
        d["binds"] = list(self.binds)
        return _jsonable(d)


@dataclass
class LevelVerdict:
    """Verdict for one of the three levels."""

    level: str
    status: str                 # admitted / refused / indeterminate
    binding_gate: Optional[str]
    margin: Optional[float]
    detection_margin: Optional[str] = None
    grounds: str = ""
    #: Checks that govern this level and were not run.  A level with a check that was not run is never admitted.
    not_run: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return _jsonable(asdict(self))


@dataclass
class Certificate:
    """The object NDIF returns: what this recording can be read for.

    Four fields per the framework's specification -- admitted level, refused
    level, binding gate, detection margin -- plus the full gate trace.
    """

    levels: dict
    gates: list
    design: dict
    provenance: dict = field(default_factory=dict)

    # ------------------------------------------------------------- accessors
    @property
    def admitted(self) -> list:
        return [k for k, v in self.levels.items() if v.status == ADMITTED]

    @property
    def refused(self) -> list:
        return [k for k, v in self.levels.items() if v.status == REFUSED]

    @property
    def indeterminate(self) -> list:
        return [k for k, v in self.levels.items() if v.status == INDETERMINATE]

    def gate(self, name: str) -> Optional[GateResult]:
        for g in self.gates:
            if g.gate == name:
                return g
        return None

    def binding_gate(self, level: str) -> Optional[str]:
        return self.levels[level].binding_gate

    # --------------------------------------------------------------- output
    def to_dict(self) -> dict:
        return _jsonable(dict(
            levels={k: v.to_dict() for k, v in self.levels.items()},
            gates=[g.to_dict() for g in self.gates],
            design=self.design,
            provenance=self.provenance,
        ))

    def to_json(self, path=None, indent=2) -> str:
        s = json.dumps(self.to_dict(), indent=indent)
        if path is not None:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(s)
        return s

    def __str__(self) -> str:
        from .report import render
        return render(self)

    __repr__ = __str__


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, (np.floating, float)):
        v = float(o)
        return None if not np.isfinite(v) else v
    if isinstance(o, (np.integer, int)):
        return int(o)
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    return o
