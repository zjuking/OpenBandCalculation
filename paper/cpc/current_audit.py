"""
Current-reporting audit for the CPC software paper.

The manuscript reports the terminal current as the median of ``Jn + Jp`` over
the links where ``max(n, p) <= 1e3 * ni`` (the depletion region), and reports
``J_spread`` as the relative deviation of the worst link.  This script checks
the two things the referee asked about:

  1. *Sensitivity of the median rule.*  How much does the reported current
     move when the ``1e3 * ni`` threshold is changed to ``1e2``, ``1e4`` or
     removed (median over the whole device)?  If the answer is stable, the
     choice of threshold is immaterial; if it is not, the low-bias number is
     not a well-defined prediction.

  2. *Current conservation.*  For each bias it reports the left- and
     right-contact currents, the device-integrated recombination as a
     discrete sum ``q * h * sum_i R[i]`` over the interior nodes, and the
     link-current distribution.  It then checks the three finite-volume
     telescoping identities that a conserved current must satisfy exactly:

        electrons:  Jn[-1] - Jn[0] = + q h sum_i R[i]
        holes:      Jp[-1] - Jp[0] = - q h sum_i R[i]
        total:      Jtot[-1] - Jtot[0] = 0

     The first two are exact consequences of the solved continuity equation;
     the third (total current) is what the floating-point noise in the
     quasi-neutral regions degrades, and its relative imbalance is the honest
     measure of how trustworthy the reported terminal current is.

Run:  python paper/cpc/current_audit.py
Writes paper/cpc/data/current_audit.json and prints the same table.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))

from obpds import Layer, OhmicContact, TwoTerminalDevice  # noqa: E402
from obpds.solver import Q  # noqa: E402


def pn(N=1000):
    return TwoTerminalDevice(
        [Layer(1000, "GaAs", Na=1e17), Layer(1000, "GaAs", Nd=1e17)],
        [OhmicContact(), OhmicContact()], T=300, N=N, approx="kane")


def audit(V, N=1000):
    """One bias point: current-distribution stats + conservation check."""
    dev = pn(N)
    s = dev.get_drift_diffusion(V, N=N)
    m = dev.model(300.0, N)

    Jtot = s.Jtotal                      # length N-1 link currents, A/cm^2
    n = s.n
    p = s.p
    ni = m.ni
    # A link sits between nodes k and k+1; a link is "in the depletion region"
    # when the larger of the two endpoint densities is below k * ni.  Take the
    # max over both endpoints so the mask matches the manuscript's wording.
    both = np.maximum(np.maximum(n[:-1], n[1:]),
                      np.maximum(p[:-1], p[1:]))
    ni_link = np.maximum(ni[:-1], ni[1:])

    def median_over(k):
        mask = both <= k * ni_link
        if not mask.any():
            return float("nan"), 0
        return float(np.median(Jtot[mask])), int(mask.sum())

    rows = []
    for k in (1e2, 1e3, 1e4, float("inf")):
        med, nlink = median_over(k)
        rows.append({"threshold_factor": None if np.isinf(k) else k,
                     "n_links": nlink, "J_median": med})

    # link-current distribution over the whole device
    Jmin, Jmax = float(Jtot.min()), float(Jtot.max())
    Jmean = float(np.mean(Jtot))
    Jstd = float(np.std(Jtot))

    # Discrete continuity conservation.  The finite-volume continuity
    # equations (transport.py) are applied at the *interior* nodes 1..N-2,
    # the two boundary nodes being Dirichlet contacts:
    #     electrons:  Jn[i] - Jn[i-1] = + q R[i] h
    #     holes:      Jp[i] - Jp[i-1] = - q R[i] h
    # Summing over i = 1..N-2 telescopes the interior fluxes away exactly:
    #     Jn[-1] - Jn[0] = + q h sum_i R[i]      (electrons)
    #     Jp[-1] - Jp[0] = - q h sum_i R[i]      (holes)
    #     Jtot[-1] - Jtot[0] = 0                 (total current, exact)
    # A plain discrete sum is the *only* consistent quadrature here: the
    # identity is exact for the nodal sum, whereas a trapezoid rule injects
    # its own O(h^2) truncation error and, worse, drags in R[0] and R[-1],
    # which sit at Dirichlet nodes and are not part of any continuity
    # equation.  The total current telescopes to zero on its own, so it must
    # NOT be compared against q*int R -- that is the single-carrier relation.
    Jn = s.Jn
    Jp = s.Jp
    R = s.R
    dx = m.x[1] - m.x[0]
    q_int_R = Q * float(np.sum(R[1:-1])) * dx       # A/cm^2, interior nodes only

    Jn_left, Jn_right = float(Jn[0]), float(Jn[-1])
    Jp_left, Jp_right = float(Jp[0]), float(Jp[-1])
    J_left = Jn_left + Jp_left
    J_right = Jn_right + Jp_right

    elec_imbalance = (Jn_right - Jn_left) - q_int_R        # electrons
    hole_imbalance = (Jp_right - Jp_left) + q_int_R        # holes
    total_imbalance = J_right - J_left                     # total current
    scale = max(abs(J_left), abs(J_right), abs(q_int_R), 1e-300)

    return {
        "V": V,
        "J_reported": float(s.J),
        "J_spread": float(s.J_spread),
        "converged": bool(s.converged),
        "gummel_iterations": int(s.gummel_iterations),
        "median_rule": rows,
        "link_currents": {
            "min": Jmin, "max": Jmax, "mean": Jmean, "std": Jstd,
            "n_links": int(Jtot.size),
        },
        "conservation": {
            "J_left": J_left, "J_right": J_right,
            "Jn_left": Jn_left, "Jn_right": Jn_right,
            "Jp_left": Jp_left, "Jp_right": Jp_right,
            "q_int_R": q_int_R,
            "electron_imbalance": elec_imbalance,
            "hole_imbalance": hole_imbalance,
            "total_imbalance": total_imbalance,
            "electron_imbalance_relative": float(elec_imbalance / scale),
            "hole_imbalance_relative": float(hole_imbalance / scale),
            "total_imbalance_relative": float(total_imbalance / scale),
        },
    }


def main():
    out = []
    for V in (-1.0, -0.5, 0.0, 0.1, 0.4, 0.6, 0.8):
        out.append(audit(V))
    path = os.path.join(HERE, "data", "current_audit.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    # human-readable table
    hdr = f"{'V':>6} {'J(rep)':>12} {'spread':>9} {'q∫R(Σ)':>12} " \
          f"{'e-imbal':>11} {'h-imbal':>11} {'tot-imbal':>11}"
    print(hdr)
    for r in out:
        c = r["conservation"]
        print(f"{r['V']:>6} {r['J_reported']:>12.4e} {r['J_spread']:>9.3g} "
              f"{c['q_int_R']:>12.4e} "
              f"{c['electron_imbalance_relative']:>11.3g} "
              f"{c['hole_imbalance_relative']:>11.3g} "
              f"{c['total_imbalance_relative']:>11.3g}")
    print("  (e-/h-imbal: relative deviation of the electron / hole telescoping\n"
          "   sum from q*sum R*h; tot-imbal: total-current conservation)")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
