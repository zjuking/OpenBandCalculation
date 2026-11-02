"""
Timing and scaling measurements for the CPC software paper.

Every number quoted in the "Running time" field of the Program Summary and in
the performance table of the manuscript comes from this script.

Run:  python paper/cpc/benchmark.py
Writes paper/cpc/data/bench.json.
"""
from __future__ import annotations

import json
import os
import platform
import sys
import time

import numpy as np
import scipy

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))

from obpds import Layer, OhmicContact, TwoTerminalDevice  # noqa: E402


def timed(fn, repeat=1):
    """Robust wall-time statistics over ``repeat`` runs, in seconds.

    Returns ``(median, (min, max), out)``.  The ``best-of`` timing that
    earlier versions of this script reported understates the typical runtime
    (it records the one run that happened to suffer the least OS jitter), so
    the headline number is now the *median* and the spread is reported
    alongside it.  For the cheap operating points ``repeat`` is small; the
    numbers are measurements of one core on the machine recorded in
    ``bench.json``, not claims about other hardware.
    """
    assert repeat >= 1
    samples = []
    out = None
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn()
        samples.append(time.perf_counter() - t0)
    samples.sort()
    med = samples[len(samples) // 2]
    return med, (samples[0], samples[-1]), out


def pn(N=1000):
    """GaAs pn junction, 1 um + 1 um, 1e17 cm^-3 on both sides."""
    return TwoTerminalDevice(
        [Layer(1000, "GaAs", Na=1e17), Layer(1000, "GaAs", Nd=1e17)],
        [OhmicContact(), OhmicContact()], T=300, N=N, approx="kane")


def hemt(N=1000):
    """Modulation-doped Al0.3Ga0.7As/GaAs single heterojunction."""
    return TwoTerminalDevice(
        [Layer(30, "AlGaAs", Nd=1e18, composition=0.30),
         Layer(10, "AlGaAs", composition=0.30),
         Layer(500, "GaAs")],
        [OhmicContact(), OhmicContact()], T=300, N=N, approx="kane")


def main():
    res = {"machine": {}, "timing": {}, "scaling": [], "spread": [],
           "timing_method": "median of repeats, min..max recorded"}

    res["machine"] = {
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }

    # --- representative single operating points, N = 1000 -----------------
    # A fresh device each time: the public API caches operating points, and
    # timing a cache hit measures nothing.  Repeat counts are chosen so the
    # cheap solves get several samples and the expensive ones stay tractable;
    # every timing reports the median and the min..max spread.
    t_eq, spread_eq, _ = timed(lambda: pn(1000).get_equilibrium(), repeat=5)
    res["timing"]["equilibrium_pn_N1000_s"] = t_eq
    res["timing"]["equilibrium_pn_N1000_minmax_s"] = list(spread_eq)

    t_dd, spread_dd, dd = timed(lambda: pn(1000).get_drift_diffusion(0.8),
                                repeat=3)
    res["timing"]["drift_diffusion_pn_0.8V_N1000_s"] = t_dd
    res["timing"]["drift_diffusion_pn_0.8V_N1000_minmax_s"] = list(spread_dd)
    res["timing"]["gummel_iterations_0.8V"] = dd.gummel_iterations
    res["timing"]["J_0.8V_A_cm2"] = dd.J
    res["timing"]["J_spread_0.8V"] = dd.J_spread

    t_iv, spread_iv, _ = timed(lambda: pn(1000).get_iv(-0.5, 1.0, 21), repeat=3)
    res["timing"]["iv_sweep_21_points_s"] = t_iv
    res["timing"]["iv_sweep_21_points_minmax_s"] = list(spread_iv)
    res["timing"]["iv_per_point_s"] = t_iv / 21.0

    t_q, spread_q, q = timed(lambda: hemt(1000).get_quantum_equilibrium(n_states=12),
                             repeat=3)
    res["timing"]["quantum_hemt_N1000_s"] = t_q
    res["timing"]["quantum_hemt_N1000_minmax_s"] = list(spread_q)
    res["timing"]["quantum_sheet_cm2"] = q.subbands["sheet_density_cm2"]
    res["timing"]["quantum_n_bound"] = int(sum(q.subbands["bound"]))
    res["timing"]["quantum_levels_eV"] = q.subbands["levels_eV"][:4]

    # --- Gummel iterations and current spread vs bias ---------------------
    dev = pn(1000)
    for V in (-1.0, -0.5, 0.1, 0.4, 0.6, 0.8):
        s = dev.get_drift_diffusion(V)
        res["spread"].append({
            "V": V,
            "J": s.J,
            "J_spread": s.J_spread,
            "gummel_iterations": s.gummel_iterations,
            "converged": bool(s.converged),
        })

    # --- mesh scaling -----------------------------------------------------
    for N in (250, 500, 1000, 2000, 4000):
        t, spread, sol = timed(lambda N=N: pn(N).get_equilibrium(), repeat=3)
        res["scaling"].append({
            "N": N,
            "equilibrium_s": t,
            "equilibrium_minmax_s": list(spread),
            "Vbi_V": float(sol.psi[-1] - sol.psi[0]),
        })

    out = os.path.join(HERE, "data")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "bench.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)

    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
