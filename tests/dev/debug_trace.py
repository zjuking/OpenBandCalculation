"""Trace the Gummel loop at one bias point."""
from __future__ import annotations
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from obpds import Layer, OhmicContact, TwoTerminalDevice
from obpds.transport import (_mu, _link_currents, _densities,
                             _solve_continuity, _majority_fermi)
from obpds.solver import (charge_neutral_guess, solve_poisson,
                          equilibrium_densities)

dev = TwoTerminalDevice([Layer(1000, "GaAs", Na=1e17),
                         Layer(1000, "GaAs", Nd=1e17)],
                        [OhmicContact(), OhmicContact()], N=200)
m = dev.model(300.0, 200)
N = m.N
mu_n, mu_p = _mu(m)
ref_c, ref_v = float(m.Nc.max()), float(m.Nv.max())
equilibrium_densities(m, "kane")

z = np.zeros(N)
psi = charge_neutral_guess(m, z, z, "kane")
psi, _ = solve_poisson(m, psi, z, z, 0.0, 0.0, "kane")

Vtarget = float(sys.argv[1]) if len(sys.argv) > 1 else 0.3
nstage = max(1, int(round(abs(Vtarget) / 0.05)))
print(f"target V={Vtarget}, {nstage} stages")
for Vs in np.linspace(0.0, Vtarget, nstage + 1)[:-1]:
    Fn = _majority_fermi(m, Vs); Fp = Fn.copy()
    psi, _ = solve_poisson(m, psi, Fp, Fn, Vs, 0.0, "kane")
    print(f"  stage V={Vs:.3f} done")

Fn = _majority_fermi(m, Vtarget); Fp = Fn.copy()
for it in range(1, 61):
    psi_new, pr = solve_poisson(m, psi, Fp, Fn, Vtarget, 0.0, "kane")
    Fn_new, c1 = _solve_continuity(m, psi_new, Fn, Fp, "kane", True,
                                   -Vtarget, 0.0, mu_n, mu_p, ref_c, ref_v)
    Fp_new, c2 = _solve_continuity(m, psi_new, Fn_new, Fp, "kane", False,
                                   -Vtarget, 0.0, mu_n, mu_p, ref_c, ref_v)
    dpsi = np.abs(psi_new - psi).max()
    dfn = np.abs(Fn_new - Fn).max()
    dfp = np.abs(Fp_new - Fp).max()
    psi, Fn, Fp = psi_new, Fn_new, Fp_new
    if it <= 10 or it % 10 == 0:
        p, n, *_r, g_n, g_p = _densities(m, psi, Fn, Fp, "kane")
        Jn, Jp, *_ = _link_currents(m, n, p, mu_n, mu_p, Fn, Fp)
        Jt = Jn + Jp
        print(f"  it {it:3d} conv={pr.converged}{int(c1)}{int(c2)} "
              f"d=({dpsi:.2e},{dfn:.2e},{dfp:.2e}) "
              f"J[N//2]={Jt[N//2]:+.5e} J[10]={Jt[10]:+.5e} "
              f"J[-10]={Jt[-10]:+.5e}  Fp[N//2]={Fp[N//2]:+.5f}")
    if max(dpsi, dfn, dfp) < 1e-7 and pr.converged and c1 and c2:
        print(f"  CONVERGED at {it}")
        break
