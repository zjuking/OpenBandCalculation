# tests/dev — development scratch scripts

This directory holds **throwaway diagnostic scripts** written while debugging
the transport solver. They are **not** part of the test suite and are **not**
run by `python -m pytest` (pytest only collects `test_*.py` at the top level of
`tests/`, and these live one level down under `dev/`).

They exist for two reasons: to document *how* the solver was debugged (the
internal quantities that were inspected, the finite-difference Jacobian checks,
the trace of the Gummel loop), and so a maintainer can re-run one when chasing
a regression in the same code path. Each script is self-contained: it builds a
GaAs pn diode (or, in a couple of cases, a HEMT), calls the internal transport
routines directly, and prints the quantities of interest.

| Script | What it inspects |
| --- | --- |
| `debug_bias.py` | Quality of the Gummel initial guess across bias, both signs |
| `debug_cold.py` | Cold-start (analytic initial guess) across the bias range |
| `debug_cons.py` | Floor of the continuity residual and the flatness of `Jtot` |
| `debug_conv.py` | Why the continuity Newton reports non-convergence |
| `debug_dd.py` | Step through one drift-diffusion solve, inspecting link currents and recombination |
| `debug_fdjac.py` | Finite-difference check of the continuity Jacobian |
| `debug_gummel.py` | Drive the Gummel loop manually, printing each continuity solve |
| `debug_hole.py` | Trace one hole-continuity Newton solve with the live code |
| `debug_hole2.py` | Trace the hole Newton at a properly staged forward-bias state |
| `debug_hole3.py` | Step through the Gummel loop, dumping state at every stage |
| `debug_iv.py` | Forward/reverse I–V sweep with the ideality factor |
| `debug_iv2.py` | Compact I–V probe: a few bias points with timing and convergence |
| `debug_jac.py` | Assemble the continuity Jacobian and check entries against finite differences |
| `debug_jest.py` | Terminal-current estimator: depletion-region median vs the naive mean |
| `debug_profile.py` | Current / carrier profiles of a converged forward-bias point |
| `debug_trace.py` | Trace the Gummel loop at one bias point |
| `debug_zero.py` | Why the continuity Newton fails at exactly `V = 0` |

The formal verification lives one directory up, in the `validate_*.py` scripts
(run through `python -m pytest`), whose stdout is the reproducible
`paper/cpc/data/*.out` artefact. These `debug_*.py` scripts are complementary
and intentionally informal.
