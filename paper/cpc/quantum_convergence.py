"""
Quantum mesh-convergence study for the CPC software paper.

The manuscript reports the quantised sheet density of the modulation-doped
AlGaAs/GaAs heterojunction at N = 400 / 800 / 1600 and discloses that it
moves by ~12 % between the two coarsest meshes.  This script extends the
study to N = 3200, reports the ground-state energy and sheet density at each
mesh, and applies Richardson extrapolation to both, so the quoted figures
carry an explicit convergence estimate rather than an accidental number of
significant digits.

Run:  python paper/cpc/quantum_convergence.py
Writes paper/cpc/data/qconv.json  (supersedes the hand-made 3-point file)
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))

from obpds import Layer, OhmicContact, TwoTerminalDevice  # noqa: E402


def hemt(N):
    return TwoTerminalDevice(
        [Layer(30, "AlGaAs", composition=0.30, Nd=1e18),
         Layer(10, "AlGaAs", composition=0.30),
         Layer(500, "GaAs")],
        [OhmicContact(), OhmicContact()], T=300, N=N, approx="kane")


def richardson(h, y):
    """Richardson extrapolation of ``y`` sampled at spacings ``h``.

    Fits ``y = y_inf + c * h^p`` over the last three points by solving for
    ``p`` and ``c`` in a least-squares sense (log-spaced, since ``p`` appears
    as an exponent).  Returns ``(y_inf, p)``; ``p`` is the observed order of
    convergence.  Uses only the three finest meshes, where the asymptotic
    form is most credible.
    """
    h = np.asarray(h, dtype=float)
    y = np.asarray(y, dtype=float)
    # take the last three points (finest meshes)
    h3, y3 = h[-3:], y[-3:]
    # solve  y_i = y_inf + c h_i^p  via a coarse nonlinear fit on p
    def err(p):
        c = (y3[-1] - y3[0]) / (h3[-1] ** p - h3[0] ** p)
        yinf = y3[0] - c * h3[0] ** p
        return np.sum((y3 - (yinf + c * h3 ** p)) ** 2)

    from scipy.optimize import minimize_scalar
    r = minimize_scalar(err, bounds=(0.5, 4.0), method="bounded")
    p = float(r.x)
    c = (y3[-1] - y3[0]) / (h3[-1] ** p - h3[0] ** p)
    yinf = float(y3[0] - c * h3[0] ** p)
    return yinf, p


def main():
    rows = []
    for N in (400, 800, 1600, 3200):
        q = hemt(N).get_quantum_equilibrium(N=N, n_states=8)
        levels = q.subbands["levels_eV"]
        bound = q.subbands["bound"]
        rows.append({
            "N": N,
            "ns": float(q.subbands["sheet_density_cm2"]),
            "E0": float(levels[0]),
            "E_cut": float(q.subbands["E_cut_eV"]),
            "nbound": int(sum(bound)),
            "conv": bool(q.converged),
        })
        print(f"  N={N:>5}  ns={rows[-1]['ns']:.4e}  E0={rows[-1]['E0']:.6f} "
              f"  nbound={rows[-1]['nbound']}  conv={rows[-1]['conv']}")

    Ns = np.array([r["N"] for r in rows], dtype=float)
    h = 1.0 / Ns
    ns = np.array([r["ns"] for r in rows])
    e0 = np.array([r["E0"] for r in rows])

    ns_inf, p_ns = richardson(h, ns)
    e0_inf, p_e0 = richardson(h, e0)

    result = {
        "rows": rows,
        "richardson": {
            "ns_inf": ns_inf, "ns_order_p": p_ns,
            "E0_inf": e0_inf, "E0_order_p": p_e0,
            # relative change from the finest mesh to the extrapolated limit
            "ns_rel_change_finest": float((ns[-1] - ns_inf) / ns_inf),
            "E0_rel_change_finest": float((e0[-1] - e0_inf) / abs(e0_inf)),
        },
    }
    path = os.path.join(HERE, "data", "qconv.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("\nRichardson (finest three meshes):")
    print(f"  ns  -> {ns_inf:.4e}  (order p ~ {p_ns:.2f}), "
          f"finest mesh off by {100*(ns[-1]-ns_inf)/ns_inf:+.3f} %")
    print(f"  E0  -> {e0_inf:.6f} eV  (order p ~ {p_e0:.2f}), "
          f"finest mesh off by {100*(e0[-1]-e0_inf)/abs(e0_inf):+.3f} %")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
