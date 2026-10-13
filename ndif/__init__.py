"""NDIF -- the Neural Dynamics Identifiability Framework.

A recording is read for quantities of three different kinds, and they are not
one continuum:

===== ================================ ==========================================
level question                         status
===== ================================ ==========================================
L1    state geometry: where is it?     estimable from observation
L2    state variability: how does it   estimable, and *not* a stability measure
      explore?
L3    intrinsic dynamics: how does it  requires identifiability certification
      evolve?
===== ================================ ==========================================

The reason L3 is different is that a stationary density is a joint function of
drift and diffusion, so an observation constrains the pair only up to an
equivalence class.  L1 and L2 are functionals of the distribution and are
invariant within that class, which is why they are estimable.  L3 varies within
it, which is why it is not -- absent additional evidence.

``certify`` returns that evidence, per recording, as an **identifiability
certificate**: for each level, *admitted*, *refused* or *indeterminate*,
together with the gate that binds, the margin at that gate, and -- for a
refusal -- the size of effect that would have been detected.  A refusal is
never reported as "no effect".

Quickstart
----------
>>> import numpy as np
>>> from ndif import certify, Design
>>> rng = np.random.default_rng(0)
>>> x = np.zeros(5000)
>>> for t in range(1, 5000):
...     x[t] = 0.9 * x[t - 1] + rng.normal(scale=0.4)
>>> cert = certify(x[:, None], Design(dt=1.0, window_s=1.0, step_s=1.0,
...                                   name="demo"))
>>> cert.levels["L3"].status
'admitted'

See ``examples/`` for a worked case on a real recording and for the
minimum-action solver.
"""

__version__ = "1.0.0"

from .types import (ADMITTED, INDETERMINATE, LEVELS, REFUSED, Certificate,
                    Design, GateResult, LevelVerdict, Recording)
from .configurations import CONFIGURATIONS, DEFAULT_CONFIGURATION
from .certify import certify, recommend_estimator
from .gates import (GATES, g0_overlap, g1_instrument, g2_aliasing, g3_recovery,
                    g4_deletion, g5_floor, g6_counter, g7_modes, g8_adequacy, run_gates,
                    G1_RULES, G2_RATES, G3_GENERATORS, G5_NULLS)
from .estimators import (ESTIMATORS, register_estimator, PropagatorFit, bootstrap_bias_correct,
                         directional_rate, fit_ar1_kalman, fit_propagator, fit_rate,
                         kalman_ll, lyapunov_residual, relaxation_rates,
                         stationary_covariance, white_share)
from .action import (MinimumActionResult, action_gradient, geometric_action,
                     minimum_action)
from .report import render, render_matrix

__all__ = [
    "__version__",
    # entry point
    "certify", "recommend_estimator", "CONFIGURATIONS", "DEFAULT_CONFIGURATION",
    # types
    "Recording", "Design", "Certificate", "GateResult", "LevelVerdict",
    "ADMITTED", "REFUSED", "INDETERMINATE", "LEVELS",
    # gates
    "run_gates", "GATES", "g0_overlap", "g1_instrument", "g2_aliasing",
    "g3_recovery", "g4_deletion", "g5_floor", "g6_counter", "g7_modes", "g8_adequacy",
    "G1_RULES", "G2_RATES", "G3_GENERATORS", "G5_NULLS",
    # estimators
    "fit_propagator", "PropagatorFit", "relaxation_rates", "white_share",
    "fit_rate", "register_estimator", "fit_ar1_kalman", "kalman_ll", "ESTIMATORS",
    "bootstrap_bias_correct", "stationary_covariance", "directional_rate",
    "lyapunov_residual",
    # action
    "geometric_action", "action_gradient", "minimum_action",
    "MinimumActionResult",
    # reporting
    "render", "render_matrix",
]
