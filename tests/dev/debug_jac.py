from __future__ import annotations
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from obpds import Layer, OhmicContact, TwoTerminalDevice
from obpds.transport import _mu, _link_currents, _densities, recombination
from obpds.solver import carriers, charge_neutral_guess, solve_poisson

dev = TwoTerminalDevice([Layer(1000, "GaAs", Na=1e17),
                         Layer(1000, "GaAs", Nd=1e17)],
                        [OhmicContact(), OhmicContact()],
                        Fp="left", Fn="right", N=200)
m = dev.model(300.0, 200)
N = m.N
z = np.zeros(N)
psi0 = charge_neutral_guess(m, z, z, "kane")
psi, _ = solve_poisson(m, psi0, z, z, 0.0, 0.0, "kane")

V = 0.3
Vl = V
Fn = np.linspace(-Vl, 0.0, N)
Fp = Fn.copy()
psi, res = solve_poisson(m, psi, Fp, Fn, Vl, 0.0, "kane")
print("poisson conv", res.converged, res.num_iter)
p, n, ng, nl, nx, g_n, g_p = _densities(m, psi, Fn, Fp, "kane")
print("n:", n.min(), n.max())
print("p:", p.min(), p.max())
print("g_n:", g_n.min(), g_n.max())
print("g_p:", g_p.min(), g_p.max())
mu_n, mu_p = _mu(m)
Jn, Jp, Bn_p, Bn_m, Bp_m, Bp_p, cn, cp = _link_currents(m, n, p, mu_n, mu_p, Fn, Fp)
print("Jn:", np.nanmin(Jn), np.nanmax(Jn))
print("Jp:", np.nanmin(Jp), np.nanmax(Jp))
print("Bn_p range", np.nanmin(Bn_p), np.nanmax(Bn_p))
print("Bn_m range", np.nanmin(Bn_m), np.nanmax(Bn_m))
print("Bp_m range", np.nanmin(Bp_m), np.nanmax(Bp_m))
print("Bp_p range", np.nanmin(Bp_p), np.nanmax(Bp_p))
print("cn", cn[0], "cp", cp[0])
R, rs, rr, dRdn, dRdp = recombination(m, n, p)
print("R:", np.nanmin(R), np.nanmax(R))

# assemble electron jacobian
h = m.dx
Q = 1.602176634e-19
g = g_n
dR = dRdn
main = np.zeros(N); up = np.zeros(N-1); lo = np.zeros(N-1)
main[1:-1] = (-cn[1:]*Bn_p[1:] - cn[:-1]*Bn_m[:-1] - Q*dR[1:-1]*h)*g[1:-1]
up[1:] = cn[1:]*Bn_m[1:]*g[2:]
lo[:-1] = -cn[:-1]*Bn_p[:-1]*g[:-2]
main[0] = 1.0; main[-1] = 1.0; up[0] = 0.0; lo[-1] = 0.0
print("main range:", np.nanmin(main), np.nanmax(main))
print("up range:", np.nanmin(up), np.nanmax(up))
print("lo range:", np.nanmin(lo), np.nanmax(lo))
nz = np.count_nonzero(main == 0)
print("zero main entries:", nz, np.where(main == 0)[0][:20])
print("main[1:6]", main[1:6], " main[N//2-2:N//2+2]", main[N//2-2:N//2+2])
print("main[-6:]", main[-6:])
print("diag ratio |main|/(|up|+|lo|) min:",
      np.nanmin(np.abs(main[1:-1])/(np.abs(up[1:])+np.abs(lo[:-1])+1e-300)))
