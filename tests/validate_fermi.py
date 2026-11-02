"""
Validation of the tabulated Fermi--Dirac integral against the quadrature.

Run:  python tests/validate_fermi.py

The fast path (:func:`obpds.fermi.generalized_fermi_fast`) replaces the
128-node Gauss--Legendre quadrature with a Hermite/linear table lookup inside
the solver.  This suite pins that optimisation against the exact quadrature:
at the table nodes the two agree to round-off, inside the table the
interpolation error stays below a few 1e-4 relative, at and beyond the table
boundary the two agree to round-off (the fast path falls back to quadrature
there), and the identity ``H(phi,0) = F_{1/2}(phi)`` shared by both paths is
asserted independently.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from obpds.fermi import (generalized_fermi, generalized_fermi_fast,  # noqa: E402
                         fermi_half, _fermi_table, _TAB_PHI_MIN,
                         _TAB_PHI_MAX, _TAB_ALPHA_MAX)


def check(name, got, want, tol, unit=""):
    rel = abs(got - want) / (abs(want) if want else 1.0)
    good = rel <= tol
    print(f"  [{'PASS' if good else 'FAIL'}] {name:44s} "
          f"got {got: .4g}{unit}  want {want: .4g}{unit}  "
          f"(rel {rel:.2e}, tol {tol:.0e})")
    return good


def section(t):
    print(f"\n=== {t} " + "=" * (58 - len(t)))


all_ok = True

# ------------------------------------------------------------ exact nodes
section("Table nodes are exact")
# At a grid node the Hermite interpolant reproduces the tabulated value and
# derivative exactly (the endpoint basis coefficients are 1 there), and the
# tabulated values come straight from the quadrature, so node evaluations of
# the fast path must match the quadrature to round-off.
phi_grid, alpha_grid, Htab, dHtab = _fermi_table()
# sample a spread of nodes, not all of them (the table is large)
idx_p = np.linspace(0, phi_grid.size - 1, 200).astype(int)
idx_a = np.linspace(0, alpha_grid.size - 1, 20).astype(int)
P, A = np.meshgrid(phi_grid[idx_p], alpha_grid[idx_a])
Hf, dHf = generalized_fermi_fast(P.ravel(), A.ravel())
Hx, dHx = generalized_fermi(P.ravel(), A.ravel())
all_ok &= check("node values match to round-off",
                float(np.abs(Hf - Hx).max() / max(np.abs(Hx).max(), 1e-300)),
                0.0, 1e-10)
all_ok &= check("node derivatives match to round-off",
                float(np.abs(dHf - dHx).max() / max(np.abs(dHx).max(), 1e-300)),
                0.0, 1e-10)

# --------------------------------------------------- inside-table accuracy
section("Interpolation error inside the table")
rng = np.random.default_rng(2)
phi = rng.uniform(_TAB_PHI_MIN + 1.0, _TAB_PHI_MAX - 1.0, 100000)
alpha = rng.uniform(0.0, _TAB_ALPHA_MAX, 100000)
Hf, dHf = generalized_fermi_fast(phi, alpha)
Hx, dHx = generalized_fermi(phi, alpha)
relH = np.abs(Hf - Hx) / np.maximum(np.abs(Hx), 1e-300)
relD = np.abs(dHf - dHx) / np.maximum(np.abs(dHx), 1e-300)
all_ok &= check("max |dH/H| relative error (value)", relH.max(), 0.0, 5e-4)
all_ok &= check("max |ddH/dH| relative error (derivative)", relD.max(), 0.0,
                2e-3)
print(f"    median value error {np.median(relH):.2e}, "
      f"median derivative error {np.median(relD):.2e}")

# ------------------------------------------------------- boundary fallback
section("Fallback outside the table is exact")
# Outside [phi_min, phi_max] and above alpha_max the fast path must agree with
# the quadrature to round-off (it delegates to it), so no accuracy is lost.
rng = np.random.default_rng(3)
phi = np.concatenate([
    rng.uniform(_TAB_PHI_MIN - 20.0, _TAB_PHI_MIN, 20000),
    rng.uniform(_TAB_PHI_MAX, _TAB_PHI_MAX + 20.0, 20000),
])
alpha = np.full(phi.size, 0.05)
Hf, dHf = generalized_fermi_fast(phi, alpha)
Hx, dHx = generalized_fermi(phi, alpha)
all_ok &= check("phi beyond table matches quadrature",
                float(np.abs(Hf - Hx).max() / max(np.abs(Hx).max(), 1e-300)),
                0.0, 1e-12)
phi2 = rng.uniform(_TAB_PHI_MIN, _TAB_PHI_MAX, 20000)
alpha2 = rng.uniform(_TAB_ALPHA_MAX, _TAB_ALPHA_MAX * 3.0, 20000)
Hf2, dHf2 = generalized_fermi_fast(phi2, alpha2)
Hx2, dHx2 = generalized_fermi(phi2, alpha2)
all_ok &= check("alpha beyond table matches quadrature",
                float(np.abs(Hf2 - Hx2).max() / max(np.abs(Hx2).max(), 1e-300)),
                0.0, 1e-12)

# ------------------------------------------------------- alpha = 0 identity
section("H(phi, 0) = F_{1/2}(phi)")
# Both the quadrature and the table honour this identity; assert it on the
# exact path so it is pinned independently of the interpolation.
phi = rng.uniform(_TAB_PHI_MIN, _TAB_PHI_MAX, 10000)
H0, dH0 = generalized_fermi(phi, 0.0)
F, dF = fermi_half(phi)
all_ok &= check("H(phi,0) == F_{1/2}(phi)", float(np.abs(H0 - F).max()), 0.0,
                1e-12)
all_ok &= check("dH(phi,0) == F'_{1/2}(phi)", float(np.abs(dH0 - dF).max()),
                0.0, 1e-12)

# ------------------------------------------------------ scalar / 0-d inputs
section("Scalar and 0-d inputs")
ref = float(fermi_half(0.0)[0])
s, ds = generalized_fermi_fast(0.0, 0.0)
all_ok &= check("scalar returns the right value", float(s), ref, 1e-9)
ref1 = float(fermi_half(1.0)[0])
arr0d, _ = generalized_fermi_fast(np.array(1.0), np.array(0.0))
all_ok &= check("0-d arrays return the right value", float(arr0d), ref1, 1e-9)

print("\n" + ("ALL FERMI-TABLE CHECKS PASSED" if all_ok
              else "FERMI-TABLE CHECKS FAILED"))
sys.exit(0 if all_ok else 1)
