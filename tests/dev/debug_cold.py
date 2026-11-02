"""Cold-start (analytic initial guess) across the bias range."""
import sys
import time

import numpy as np

from obpds import TwoTerminalDevice, Layer, OhmicContact
from obpds.transport import _gummel, _mu, _densities, _link_currents
from obpds.solver import equilibrium_densities

N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
Vs = [float(v) for v in (sys.argv[2:] or
                         [f"{v/10:.1f}" for v in range(-10, 11)])]

dev = TwoTerminalDevice(
    [Layer(200, "GaAs", Na=1e17), Layer(200, "GaAs", Nd=1e17)],
    [OhmicContact(), OhmicContact()], N=N)
m = dev.model()
equilibrium_densities(m, "kane")
mu_n, mu_p = _mu(m)
ref_c, ref_v = float(m.Nc.max()), float(m.Nv.max())

print(f"{'V':>6} {'Jn[0]':>12} {'Jp[0]':>12} {'Jtot(left)':>12} "
      f"{'Jtot(right)':>12} {'iters':>5} {'conv':>5} {'t(s)':>6}", flush=True)
for V in Vs:
    t = time.time()
    psi, Fn, Fp, ok, it = _gummel(m, V, "kane", 40, None,
                                  mu_n, mu_p, ref_c, ref_v, 1e-7)
    p, n, *_r, dn, dp = _densities(m, psi, Fn, Fp, "kane")
    Jn, Jp, *_ = _link_currents(m, n, p, mu_n, mu_p, Fn, Fp)
    print(f"{V:+6.2f} {Jn[0]:+12.4e} {Jp[0]:+12.4e} {Jn[0]+Jp[0]:+12.4e} "
          f"{Jn[-1]+Jp[-1]:+12.4e} {it:5d} {str(ok):>5} "
          f"{time.time()-t:6.1f}", flush=True)
