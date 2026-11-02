"""Development scratch: drive the Gummel loop manually and print the electron
and hole continuity solves at each sweep.  Not part of the test suite; see
tests/dev/README.
"""
from __future__ import annotations
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from obpds import Layer, OhmicContact, TwoTerminalDevice
from obpds.transport import (_mu, _link_currents, _densities,
                             _solve_continuity)
from obpds.solver import carriers, charge_neutral_guess, solve_poisson

dev = TwoTerminalDevice([Layer(1000, "GaAs", Na=1e17),
                         Layer(1000, "GaAs", Nd=1e17)],
                        [OhmicContact(), OhmicContact()],
                        Fp="left", Fn="right", N=200)
m = dev.model(300.0, 200)
N = m.N
mu_n, mu_p = _mu(m)
ref_c, ref_v = float(m.Nc.max()), float(m.Nv.max())

z = np.zeros(N)
psi = charge_neutral_guess(m, z, z, "kane")
psi, _ = solve_poisson(m, psi, z, z, 0.0, 0.0, "kane")
Fn = np.zeros(N)
Fp = np.zeros(N)

Vtarget = 0.9
steps = np.linspace(0.0, Vtarget, 19)
for V in steps[1:]:
    fl, fr = -V, 0.0
    for it in range(1, 41):
        psi, pr = solve_poisson(m, psi, Fp, Fn, V, 0.0, "kane")
        Fn_new, c1 = _solve_continuity(m, psi, Fn, Fp, "kane", True,
                                       fl, fr, mu_n, mu_p, ref_c, ref_v)
        Fp_new, c2 = _solve_continuity(m, psi, Fn_new, Fp, "kane", False,
                                       fl, fr, mu_n, mu_p, ref_c, ref_v)
        d = max(np.abs(psi - psi).max(), np.abs(Fn_new - Fn).max(),
                np.abs(Fp_new - Fp).max())
        dpsi = 0.0
        Fn, Fp = Fn_new, Fp_new
        dfn = np.abs(Fn_new - Fn).max() if False else 0.0
        if it in (1, 2, 3, 5, 10, 20, 40):
            p, n, *_r, g_n, g_p = _densities(m, psi, Fn, Fp, "kane")
            Jn, Jp, *_ = _link_currents(m, n, p, mu_n, mu_p, Fn, Fp)
            Jt = Jn + Jp
            print(f"V={V:.3f} it={it:3d} conv={pr.converged}{c1}{c2} "
                  f"Fn[0]={Fn[0]:+.4f} Fn[N/2]={Fn[N//2]:+.4f} "
                  f"Fp[0]={Fp[0]:+.4f} Fp[N/2]={Fp[N//2]:+.4f} "
                  f"J mid={Jt[N//2]:.4e}")
        if d < 1e-7:
            break
    p, n, *_r, g_n, g_p = _densities(m, psi, Fn, Fp, "kane")
    Jn, Jp, *_ = _link_currents(m, n, p, mu_n, mu_p, Fn, Fp)
    Jt = Jn + Jp
    print(f"== V={V:.3f}: {it} gummel iters  J={np.mean(Jt[N//4:3*N//4]):.6e} "
          f"spread {Jt.min():.3e}..{Jt.max():.3e}")
    print(f"   Fp: {Fp[0]:+.4f} {Fp[N//4]:+.4f} {Fp[N//2]:+.4f} "
          f"{Fp[3*N//4]:+.4f} {Fp[-1]:+.4f}")
    print(f"   Fn: {Fn[0]:+.4f} {Fn[N//4]:+.4f} {Fn[N//2]:+.4f} "
          f"{Fn[3*N//4]:+.4f} {Fn[-1]:+.4f}")
