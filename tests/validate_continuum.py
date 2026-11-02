"""
Quantify the error of the quantum continuum-compensation approximation.

``quantum.py`` models the itinerant (above-threshold) electron density as

    n_cont(x) = n_3D(x) * Q(3/2, delta(x)),   delta = max(0, E_cut - Ec)/kT

with ``Q(3/2, .)`` the regularised upper incomplete gamma.  The weight is
*exact in the Boltzmann limit*: there the full density is
``n_3D ~ Nc exp(eta)`` and the truncated integral is
``Nc exp(eta) Gamma(3/2, delta)``, so the ratio is ``Q(3/2, delta)`` exactly.
But ``quantum.py`` multiplies the *Fermi-Dirac* ``n_3D`` by the same weight,
which is an approximation -- the Fermi function no longer factorises into
``exp(eta)`` times a function of ``u = (E-Ec)/kT`` alone.

This suite quantifies that error two ways:

  1. abstractly, as a function of ``eta = (F - Ec)/kT`` (degeneracy) and
     ``delta`` (well depth), by comparing the exact truncated Fermi-Dirac
     integral to ``Q(3/2, delta)``;
  2. for the real AlGaAs/GaAs HEMT, by integrating the difference over the
     device and reporting its effect on the sheet density.

Run:  python tests/validate_continuum.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
from scipy.integrate import quad
from scipy.special import gammaincc

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from obpds import Layer, OhmicContact, TwoTerminalDevice  # noqa: E402
from obpds.fermi import KB_EV  # noqa: E402
from obpds.solver import carriers  # noqa: E402

OK = "\033[92mPASS\033[0m"
BAD = "\033[91mFAIL\033[0m"


def check(name, got, want, tol, unit=""):
    rel = abs(got - want) / (abs(want) if want else 1.0)
    good = rel <= tol
    print(f"  [{'PASS' if good else 'FAIL'}] {name:36s} "
          f"got {got: .4g}{unit}  want {want: .4g}{unit}  "
          f"(rel {rel:.2e}, tol {tol:.0e})")
    return good


def section(t):
    print(f"\n=== {t} " + "=" * (60 - len(t)))


def exact_ratio(eta, delta):
    """Exact ``n_cont/n_3D`` under Fermi-Dirac statistics.

    The truncated integral is ``int_delta^inf u^{1/2}/(1+e^{u-eta}) du`` over
    the full one ``int_0^inf ...``.  The upper limit is a fixed multiple of
    kT; ``u = eta + 40`` is far into the exponential tail.
    """
    hi = max(80.0, eta + 40.0)
    num = quad(lambda u: np.sqrt(u) / (1.0 + np.exp(u - eta)), delta, hi)[0]
    den = quad(lambda u: np.sqrt(u) / (1.0 + np.exp(u - eta)), 0.0, hi)[0]
    return num / den if den > 0 else 0.0


all_ok = True

# ------------------------------------------------------- 1. Boltzmann limit
section("Boltzmann limit: the Q(3/2, delta) weight is exact")
for delta in (0.5, 2.0, 5.0):
    # deep in the non-degenerate regime the exact ratio must collapse onto
    # Q(3/2, delta); any residual is only numerical integration error.
    ex = exact_ratio(-10.0, delta)
    ap = float(gammaincc(1.5, delta))
    all_ok &= check(f"Boltzmann exact at delta={delta}", ex, ap, 1e-3)

# ------------------------------------------------------- 2. abstract error
section("Fermi-Dirac error of the Q(3/2) weight vs (eta, delta)")
print(f"  {'eta':>6} {'delta=0.5':>12} {'delta=2':>12} {'delta=5':>12}")
worst_nondeg = 0.0
worst_deg = 0.0
for eta in (-5.0, -2.0, 0.0, 2.0, 5.0):
    row = f"  {eta:>6}"
    for delta in (0.5, 2.0, 5.0):
        ex = exact_ratio(eta, delta)
        ap = float(gammaincc(1.5, delta))
        err = abs(ap - ex) / ex if ex > 1e-300 else 0.0
        row += f"{err:>12.3e}"
        if eta < 0:
            worst_nondeg = max(worst_nondeg, err)
        else:
            worst_deg = max(worst_deg, err)
    print(row)
print(f"  worst error, non-degenerate (eta<0): {worst_nondeg:.3e}")
print(f"  worst error, degenerate     (eta>0): {worst_deg:.3e}")

# The approximation is good where carriers are non-degenerate and visibly
# degrades under degeneracy -- that is the whole point of quantifying it.
all_ok &= check("non-degenerate error stays small", worst_nondeg, 0.0, 0.06)
all_ok &= check("degenerate error is large (flagged)", worst_deg, 1.0, 0.5)

# ------------------------------------------------- 3. effect on a real HEMT
section("Effect on the real AlGaAs/GaAs HEMT sheet density")
hemt = TwoTerminalDevice(
    [Layer(30, "AlGaAs", composition=0.30, Nd=1e18),
     Layer(10, "AlGaAs", composition=0.30),
     Layer(500, "GaAs")],
    [OhmicContact(), OhmicContact()], T=300, N=800, approx="kane")
q = hemt.get_quantum_equilibrium(N=800, n_states=12)
m = hemt.model(300.0, 800)
kT = KB_EV * 300.0
Ec = m.CBO_Gamma - q.psi
E_cut = float(min(Ec[0], Ec[-1]))
delta = np.maximum(E_cut - Ec, 0.0) / kT
eta = (0.0 - Ec) / kT
# classical 3D density at equilibrium, the thing the weight multiplies
_, n3, *_ = carriers(m, q.psi, np.zeros(m.N), np.zeros(m.N), "kane")
weight = np.asarray(gammaincc(1.5, delta), dtype=float)

# exact continuum weight vs the approximate one, node by node.  Outside the
# well delta == 0 and Q == 1 exactly, so the weight carries no error there.
ex = np.ones(m.N)
inwell = delta > 1e-3
for i in np.nonzero(inwell)[0]:
    ex[i] = exact_ratio(float(eta[i]), float(delta[i]))

dx = m.x[1] - m.x[0]
cont_approx = float(np.sum(n3 * weight) * dx)
cont_exact = float(np.sum(n3 * ex) * dx)
bound_sheet = float(q.subbands["sheet_density_cm2"])
err = cont_approx - cont_exact
total_exact = bound_sheet + cont_exact
print(f"  bound sheet density = {bound_sheet:.4e} cm^-2")
print(f"  continuum (approx)  = {cont_approx:.4e} cm^-2")
print(f"  continuum (exact)   = {cont_exact:.4e} cm^-2")
print(f"  integrated error    = {err:.4e} cm^-2")
rel = err / total_exact
print(f"  relative error on the TOTAL sheet density = {rel:.3e}")
# The error is a real, non-negligible few percent in a degenerate well --
# exactly what this suite exists to surface -- but it must not be so large
# that the quantised density is meaningless.
all_ok &= check("HEMT total-density error stays below 10%", abs(rel), 0.0,
                0.10)

print("\n" + ("ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED"))
sys.exit(0 if all_ok else 1)
