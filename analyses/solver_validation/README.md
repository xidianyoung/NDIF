# Minimum-action solver: validation against closed-form actions

| Script | Output | Reported in |
|---|---|---|
| `solver_validation.py` | `out/solver_validation.json` | Extended Data Fig. 5 (panels a–c) and the solver checks of the Methods (*Software implementation*): action error against node count with its log-log slope, the minimum-to-minimum path (5.000569, peak \|y\| 1.0011), fifteen non-normal linear systems (relative errors 2.41 × 10⁻¹² to 6.84 × 10⁻⁵; the four test-suite systems 5.19 × 10⁻¹² to 8.75 × 10⁻⁷), the translation-only control (69.178 after 600 iterations at 81 nodes, peak \|y\| 0.379; straight line 108.18), the analytic gradient against finite differences (2.24 × 10⁻⁹) and the minimum-to-saddle action at 401 nodes (5.000021) |

```bash
python analyses/solver_validation/solver_validation.py      # about three minutes on one core
```

The script uses `ndif.action.minimum_action` with its default arguments, except for the node schedule of the 11-, 21-
and 41-node solves of panel a, which never pass through more nodes than the final count (see the script's docstring).
The translation-only control moves every interior node by the same vector at each iteration, so the path can be
translated but not bent.  The output records every computed value, the settings and the agreement of each value with
the number quoted in the paper.
