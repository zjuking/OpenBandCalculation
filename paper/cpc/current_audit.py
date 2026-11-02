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

  2. *Current conservation.*  For each bias it reports
        - the left- and right-contact currents ``Jn + Jp``,
        - the device-integrated recombination ``q * int R dx``,
        - the link-current distribution (median, mean, max, std) of ``Jn+Jp``,
     and checks ``J_left - J_right ~ q * int R dx``, which is the discrete
     continuity sum that a conserved current must satisfy.

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

    # conservation: J_left - J_right should equal q * int R dx
    J_left = float(s.Jn[0] + s.Jp[0])
    J_right = float(s.Jn[-1] + s.Jp[-1])
    R = s.R
    dx = m.x[1] - m.x[0]
    gen = float(Q * np.trapezoid(R, dx=dx))     # net recombination, A/cm^2
    imbalance = J_left - J_right - gen
    scale = max(abs(J_left), abs(J_right), abs(gen), 1e-300)

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
            "q_int_R": gen, "imbalance": imbalance,
            "imbalance_relative": float(imbalance / scale),
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
    hdr = f"{'V':>6} {'J(rep)':>12} {'spread':>9} {'J_L':>12} {'J_R':>12} " \
          f"{'q∫R':>12} {'imbal':>12} {'imbal/scale':>12}"
    print(hdr)
    for r in out:
        c = r["conservation"]
        print(f"{r['V']:>6} {r['J_reported']:>12.4e} {r['J_spread']:>9.3g} "
              f"{c['J_left']:>12.4e} {c['J_right']:>12.4e} "
              f"{c['q_int_R']:>12.4e} {c['imbalance']:>12.4e} "
              f"{c['imbalance_relative']:>12.3g}")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
