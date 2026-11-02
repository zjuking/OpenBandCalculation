"""Check the quality of the Gummel *initial guess* across bias, both signs."""
import sys

import numpy as np

from obpds import TwoTerminalDevice, Layer, OhmicContact
from obpds import transport as T
from obpds.solver import (charge_neutral_guess, equilibrium_densities,
                          solve_poisson)

N = int(sys.argv[1]) if len(sys.argv) > 1 else 400
Vs = [float(v) for v in (sys.argv[2:] or
                         ["-1", "-0.5", "-0.1", "0.0", "0.1", "0.3", "0.5",
                          "0.7", "0.9"])]

dev = TwoTerminalDevice(
    [Layer(200, "GaAs", Nd=1e17), Layer(200, "GaAs", Na=1e17)],
    [OhmicContact(), OhmicContact()],
    N=N,
)
m = dev.model()
equilibrium_densities(m, "kane")
mu_n, mu_p = T._mu(m)
h = m.dx

print(f"{'V':>6}  {'pois':>5}  {'|res_n|':>10}  {'|res_p|':>10}  "
      f"{'Jn[0]':>12}  {'Jp[0]':>12}   Fn@p-contact  Fp@n-contact")
for V in Vs:
    Fn, Fp = T._majority_fermi(m, V)
    psi = charge_neutral_guess(m, Fp, Fn, "kane")
    psi, pres = solve_poisson(m, psi, Fp, Fn, float(V), 0.0, "kane")
    p, n, ng, nl, nx, dn, dp = T._densities(m, psi, Fn, Fp, "kane")
    R, *_r, dRdn, dRdp = T.recombination(m, n, p)
    Jn, Jp, *_ = T._link_currents(m, n, p, mu_n, mu_p, Fn, Fp)
    res_n = Jn[1:] - Jn[:-1] - T.Q * R[1:-1] * h
    res_p = Jp[1:] - Jp[:-1] + T.Q * R[1:-1] * h
    print(f"{V:+6.2f}  {str(pres.converged):>5}  {np.abs(res_n).max():10.3e}  "
          f"{np.abs(res_p).max():10.3e}  {Jn[0]:+12.4e}  {Jp[0]:+12.4e}   "
          f"{Fn[-1]:+.4f}        {Fp[0]:+.4f}")

# ---- shape of the initial guess at one forward and one reverse bias --------
for V in (-1.0, 0.5):
    Fn, Fp = T._majority_fermi(m, V)
    sel = np.linspace(0, m.N - 1, 11).astype(int)
    print(f"\nV={V:+.2f}   node: " + " ".join(f"{i:6d}" for i in sel))
    print("      Fn    : " + " ".join(f"{Fn[i]:+6.3f}" for i in sel))
    print("      Fp    : " + " ".join(f"{Fp[i]:+6.3f}" for i in sel))
    print("      Fn-Fp : " + " ".join(f"{Fn[i]-Fp[i]:+6.3f}" for i in sel))
