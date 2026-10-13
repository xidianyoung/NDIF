"""The analysis configurations of the framework.

A configuration fixes which gates are run, which gates govern each claim class and in what order they are
consulted, and which rule G1 and G3 apply.  The three configurations are those defined in the Methods of the
manuscript (*Analysis configurations*):

``"primary"``
    Gates G0-G7.  L1 is governed by G6 then G5; L2 by G0 then the variance channel of G3; L3 by G0, G1, G2,
    G3, G7 then G4.  G1 is the two-lag white-share rule (``nu <= 0.20`` for an estimator without an
    observation model, not binding for an observation-aware estimator), G3 simulates from the fitted
    generator and G5 compares the separation with the 95th percentile of 200 simulations of the fitted
    generator without a condition effect.  Gate simulations use the recording's number of segments at their
    rounded mean length.  Used for the human recordings (the 1,204 sleep cell assessments, the 83 HCP MEG assessments
    and the 56 COG-BCI workload assessments) and for the three regularly indexed non-neural series.

``"benchmark"``
    Gates G0, G1, G2, G3 and G7 of the primary configuration, applied to simulated recordings with known rates.
    G4 is not run, so L3 is governed by G0, G1, G2, G3 and G7 only and the decisions assess the rate gates
    rather than complete L3 assessments; the L1 gates are not run.  Used for the 320-output benchmark.

``"g8_augmented"``
    Gates G0-G3, G5, G6 and G8, with G4 and G7 available as diagnostics that govern no level.  L3 is governed by
    G0, G2, G8, G1 then G3.  G1 bounds the variation of the log rate over the 95% profile-likelihood set of the
    white share at 0.20 log units, G3 uses an indirect-inference-corrected generator with slope, offset and
    spread criteria decided beyond Monte Carlo error, and G5 uses a permutation null (condition labels permuted
    across segments).  Gate simulations use the recording's own segment lengths.  Used for the extended simulation evaluation, the
    calendar-preserving CO2 analysis and the re-evaluation of the 602 sleep cells.

A claim is refused when any governing gate fails, indeterminate when no governing gate fails but at least one
could not be evaluated (or was not run), and admitted only when every governing gate was run and passed.
"""

from __future__ import annotations

__all__ = ["CONFIGURATIONS", "DEFAULT_CONFIGURATION", "resolve_configuration"]

_PRIMARY_LEVELS = {"L1": ["G6", "G5"], "L2": ["G0", "G3v"], "L3": ["G0", "G1", "G2", "G3", "G7", "G4"]}
_BENCHMARK_LEVELS = {"L1": ["G6", "G5"], "L2": ["G0", "G3v"], "L3": ["G0", "G1", "G2", "G3", "G7"]}
_AUGMENTED_LEVELS = {"L1": ["G6", "G5"], "L2": ["G0", "G3v"], "L3": ["G0", "G2", "G8", "G1", "G3"]}

CONFIGURATIONS = {
    "primary": dict(
        name="primary",
        description=("gates G0-G7; L1: G6, G5; L2: G0, G3 variance channel; L3: G0, G1, G2, G3, G7, G4; "
                     "G1 two-lag white share (nu <= 0.20 without an observation model); "
                     "G3 recovery from the fitted generator; G5 simulated zero-truth floor"),
        gates=("G0", "G1", "G2", "G3", "G4", "G5", "G6", "G7"),
        diagnostics=(),
        level_gates=_PRIMARY_LEVELS,
        gate_kw={"G1": dict(rule="two_lag"), "G3": dict(generator="fitted"), "G5": dict(null="simulated"),
                 "G4": dict(segment_lengths="mean"), "G6": dict(segment_lengths="mean"), "G7": dict(segment_lengths="mean")},
    ),
    "benchmark": dict(
        name="benchmark",
        description=("gates G0, G1, G2, G3, G7 of the primary configuration on simulated recordings with known "
                     "rates; G4 not run; L3: G0, G1, G2, G3, G7"),
        gates=("G0", "G1", "G2", "G3", "G7"),
        diagnostics=(),
        level_gates=_BENCHMARK_LEVELS,
        gate_kw={"G1": dict(rule="two_lag"), "G3": dict(generator="fitted"), "G5": dict(null="simulated"),
                 "G4": dict(segment_lengths="mean"), "G6": dict(segment_lengths="mean"), "G7": dict(segment_lengths="mean")},
    ),
    "g8_augmented": dict(
        name="g8_augmented",
        description=("gates G0-G3, G5, G6, G8 (G4 and G7 diagnostics); L3: G0, G2, G8, G1, G3; "
                     "G1 profile-likelihood bound on the log rate over the white-share set; "
                     "G3 recovery from the indirect-inference-corrected generator, decided beyond Monte Carlo error; "
                     "G5 permutation null"),
        gates=("G0", "G1", "G2", "G3", "G5", "G6", "G8"),
        diagnostics=("G4", "G7"),
        level_gates=_AUGMENTED_LEVELS,
        gate_kw={"G1": dict(rule="profile"), "G3": dict(generator="corrected"), "G5": dict(null="permutation"),
                 "G4": dict(segment_lengths="actual"), "G6": dict(segment_lengths="actual"), "G7": dict(segment_lengths="actual")},
    ),
}

#: The configuration used when none is named: the primary configuration of the manuscript.
DEFAULT_CONFIGURATION = "primary"


def resolve_configuration(configuration):
    """The configuration dict for a name (or a dict with the same keys, returned unchanged)."""
    if isinstance(configuration, dict):
        missing = {"name", "gates", "level_gates", "gate_kw"} - set(configuration)
        if missing:
            raise ValueError(f"a configuration dict needs the keys {sorted(missing)}")
        return configuration
    try:
        return CONFIGURATIONS[configuration]
    except KeyError:
        raise ValueError(f"unknown configuration {configuration!r}; choose one of {sorted(CONFIGURATIONS)}") from None
