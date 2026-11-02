"""Trace one continuity Newton solve with the live production code."""
from __future__ import annotations
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from obpds import Layer, OhmicContact, TwoTerminalDevice
from obpds.transport import (_mu, _link_currents, _densities, _gummel)
from obpds.solver import charge_neutral_guess, solve_poisson, equilibrium_densities

dev = TwoTerminalDevice([Layer(1000, "GaAs", Na=1e17),
                         Layer(1000, "GaAs", Nd=1e17)],
                        [OhmicContact(), OhmicContact()], N=200)
m = dev.model(300.0, 200)
N = m.N
mu_n, mu_p = _mu(m)
ref_c, ref_v = float(m.Nc.max()), float(m.Nv.max())
equilibrium_densities(m, "kane")

V = float(sys.argv[1]) if len(sys.argv) > 1 else 0.3
which = sys.argv[2] if len(sys.argv) > 2 else "h"
electron = which == "e"

psi, Fn, Fp, ok, it = _gummel(m, V, "kane", 40, None, mu_n, mu_p,
                              ref_c, ref_v, 1e-7)
print(f"staged V={V}: ok={ok} iters={it}")
print(f"  Fn: {Fn[0]:+.4f} {Fn[N//4]:+.4f} {Fn[N//2]:+.4f} {Fn[-1]:+.4f}")
print(f"  Fp: {Fp[0]:+.4f} {Fp[N//4]:+.4f} {Fp[N//2]:+.4f} {Fp[-1]:+.4f}")

from obpds.transport import _solve_continuity
F_new, ok2 = _solve_continuity(m, psi, Fn, Fp, "kane", electron,
                               -V, 0.0, mu_n, mu_p, ref_c, ref_v)
print(f"\n{'electron' if electron else 'hole'} solve: converged={ok2}")
print("  result:", " ".join(f"{v:+.4f}" for v in F_new[::N // 10]))
