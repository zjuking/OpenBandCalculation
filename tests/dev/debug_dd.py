"""Development scratch: step through one drift-diffusion solve and inspect
the internal quantities (link currents, effective potentials, recombination)
at each Gummel iteration.  Not part of the test suite; see tests/dev/README.
"""
from __future__ import annotations
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from obpds import Layer, OhmicContact, TwoTerminalDevice
from obpds.transport import (_mu, _effective_potentials, _link_currents,
                             recombination, solve_drift_diffusion)
from obpds.solver import carriers, charge_neutral_guess, solve_poisson
from obpds.fermi import KB_EV

dev = TwoTerminalDevice([Layer(1000, "GaAs", Na=1e17),
                         Layer(1000, "GaAs", Nd=1e17)],
                        [OhmicContact(), OhmicContact()],
                        Fp="left", Fn="right", N=200)
m = dev.model(300.0, 200)
print("N =", m.N, " dx(um) =", m.dx * 1e4)
print("mu_n =", m.mu_n[0], " mu_p =", m.mu_p[0])
print("Nnet:", m.Nnet[0], m.Nnet[-1])

# ---- equilibrium Poisson
z = np.zeros(m.N)
psi0 = charge_neutral_guess(m, z, z, "kane")
t = time.time()
psi, res = solve_poisson(m, psi0, z, z, 0.0, 0.0, "kane")
print(f"equilibrium Poisson: {res.num_iter} iters, converged={res.converged}, {time.time()-t:.2f}s")
p, n, *_r, dp, dn = carriers(m, psi, z, z, "kane")
print("n range", n.min(), n.max(), " p range", p.min(), p.max())

mu_n, mu_p = _mu(m)
Jn, Jp, *_ = _link_currents(m, n, p, mu_n, mu_p, z, z)
print("equilibrium |Jn| max =", np.abs(Jn).max(), " |Jp| max =", np.abs(Jp).max())

# ---- equilibrium drift-diffusion (should be a no-op)
t = time.time()
s = solve_drift_diffusion(m, 0.0, "kane")
print(f"DD(0V): {s.gummel_iterations} gummel iters, converged={s.converged}, "
      f"J={s.J:.4e}, {time.time()-t:.2f}s")

# ---- forward bias
for V in (0.3, 0.6, 0.9):
    t = time.time()
    s = solve_drift_diffusion(m, V, "kane", max_gummel=60)
    print(f"DD({V}V): {s.gummel_iterations} iters, conv={s.converged}, "
          f"J={s.J:.6e}, Jn0={s.Jn_contact:.4e}, Jp0={s.Jp_contact:.4e}, "
          f"{time.time()-t:.2f}s")
    Jt = s.Jtotal
    print("   Jtotal spread: min %.4e max %.4e" % (Jt.min(), Jt.max()))
