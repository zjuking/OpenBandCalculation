"""Print the current / carrier profiles of a converged forward bias point."""
from __future__ import annotations
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from obpds import Layer, OhmicContact, TwoTerminalDevice
from obpds.transport import (_mu, _link_currents, _densities, recombination,
                             solve_drift_diffusion)
from obpds.fermi import KB_EV

V = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
N = int(sys.argv[2]) if len(sys.argv) > 2 else 400

dev = TwoTerminalDevice([Layer(1000, "GaAs", Na=1e17),
                         Layer(1000, "GaAs", Nd=1e17)],
                        [OhmicContact(), OhmicContact()],
                        Fp="left", Fn="right", N=N)
m = dev.model(300.0, N)
s = solve_drift_diffusion(m, V, "kane", max_gummel=40)
mu_n, mu_p = _mu(m)
p, n, ng, nl, nx, dn, dp = _densities(m, s.psi, s.Fn, s.Fp, "kane")
Jn, Jp, *_ = _link_currents(m, n, p, mu_n, mu_p, s.Fn, s.Fp)
R, *_ = recombination(m, n, p)
Jt = Jn + Jp
h = m.dx
q = 1.602176634e-19

print(f"V={V}  conv={s.converged} iters={s.gummel_iterations}")
print(f"Jn[0]={Jn[0]:.4e}  Jp[0]={Jp[0]:.4e}  Jtot[0]={Jt[0]:.4e}")
print(f"Jn[-1]={Jn[-1]:.4e}  Jp[-1]={Jp[-1]:.4e}  Jtot[-1]={Jt[-1]:.4e}")
print(f"Jtot: min={Jt.min():.4e} max={Jt.max():.4e} mean(mid)={s.J:.4e}")
print("\n idx  x(um)     psi      Fn       Fp        n         p      "
      "Jn         Jp        Jtot      R")
for i in range(0, N, max(1, N // 20)):
    k = min(i, N - 2)
    print(f" {i:4d} {m.x[i]*1e4:7.4f} {s.psi[i]:+8.4f} {s.Fn[i]:+8.4f} "
          f"{s.Fp[i]:+8.4f} {n[i]:9.3e} {p[i]:9.3e} {Jn[k]:+10.3e} "
          f"{Jp[k]:+10.3e} {Jt[k]:+10.3e} {R[i]:+9.2e}")

# continuity residuals actually achieved
res_n = np.empty(N)
res_n[1:-1] = Jn[1:] - Jn[:-1] - q * R[1:-1] * h
res_p = np.empty(N)
res_p[1:-1] = Jp[1:] - Jp[:-1] + q * R[1:-1] * h
print(f"\nraw electron continuity residual: max |res| = {np.abs(res_n[1:-1]).max():.4e} A/cm2")
print(f"raw hole     continuity residual: max |res| = {np.abs(res_p[1:-1]).max():.4e} A/cm2")
print(f"integrated R over device: {np.trapezoid(R, dx=h):.4e} cm^-2 s^-1")
print(f"q*that = {q*np.trapezoid(R, dx=h):.4e} A/cm2   "
      f"(should equal Jn[-1]-Jn[0] = {Jn[-1]-Jn[0]:.4e})")
