"""Where is the floor on the continuity residual, and how flat is Jtot?"""
import sys
import time

import numpy as np

from obpds import TwoTerminalDevice, Layer, OhmicContact
from obpds import transport as T
from obpds.solver import equilibrium_densities

N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
V = float(sys.argv[2]) if len(sys.argv) > 2 else 0.1
FACTORS = [float(f) for f in (sys.argv[3:] or ["1e-16", "1e-9", "1e-6"])]

dev = TwoTerminalDevice(
    [Layer(200, "GaAs", Na=1e17), Layer(200, "GaAs", Nd=1e17)],
    [OhmicContact(), OhmicContact()], N=N)
m = dev.model()
equilibrium_densities(m, "kane")
mu_n, mu_p = T._mu(m)
ref_c, ref_v = float(m.Nc.max()), float(m.Nv.max())
h, Q = m.dx, T.Q


def report(tag):
    psi, Fn, Fp, ok, it = T._gummel(m, V, "kane", 60, None,
                                    mu_n, mu_p, ref_c, ref_v, 1e-7)
    p, n, *_r, dn, dp = T._densities(m, psi, Fn, Fp, "kane")
    R, *_ = T.recombination(m, n, p)
    Jn, Jp, *_ = T._link_currents(m, n, p, mu_n, mu_p, Fn, Fp)
    Jt = Jn + Jp
    rn = np.abs(Jn[1:] - Jn[:-1] - Q * R[1:-1] * h).max()
    rp = np.abs(Jp[1:] - Jp[:-1] + Q * R[1:-1] * h).max()
    flat = (np.abs(Jt - np.mean(Jt)).max() / max(abs(np.mean(Jt)), 1e-300))
    print(f"  {tag:>8}: gummel={it:3d} ok={ok}  |res_n|={rn:.3e} "
          f"|res_p|={rp:.3e}  Jmean={np.mean(Jt):+.4e} "
          f"Jleft={Jt[0]:+.4e} flatness_err={flat:.2e}", flush=True)


# reference: the old fixed, unweighted tolerance
orig = T.GTOL_REL
print(f"V = {V}  N = {N}")
for f in FACTORS:
    T.GTOL_REL = f
    t = time.time()
    report(f"{f:g}")
    print(f"           ({time.time()-t:.1f}s)", flush=True)
T.GTOL_REL = orig
