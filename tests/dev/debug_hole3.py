"""Step through the Gummel loop, dumping the state at every stage."""
from __future__ import annotations
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from obpds import Layer, OhmicContact, TwoTerminalDevice
from obpds.transport import (_mu, _link_currents, _densities, _majority_fermi,
                             recombination, _solve_continuity)
from obpds.solver import (charge_neutral_guess, solve_poisson,
                          equilibrium_densities, Q)

dev = TwoTerminalDevice([Layer(1000, "GaAs", Na=1e17),
                         Layer(1000, "GaAs", Nd=1e17)],
                        [OhmicContact(), OhmicContact()], N=200)
m = dev.model(300.0, 200)
N, h = m.N, m.dx
mu_n, mu_p = _mu(m)
ref_c, ref_v = float(m.Nc.max()), float(m.Nv.max())
equilibrium_densities(m, "kane")
V = float(sys.argv[1]) if len(sys.argv) > 1 else 0.3

Fn, Fp = _majority_fermi(m, V)
psi = charge_neutral_guess(m, Fp, Fn, "kane")
print(f"init: psi[p]={psi[20]:+.4f} psi[n]={psi[180]:+.4f} "
      f"Fp={Fp[0]:+.3f}/{Fp[100]:+.3f}/{Fp[-1]:+.3f}")

for it in range(1, 6):
    psi, pr = solve_poisson(m, psi, Fp, Fn, V, 0.0, "kane")
    p, n, ng, nl, nx, g_n, g_p = _densities(m, psi, Fn, Fp, "kane")
    print(f"-- it {it}: Poisson conv={pr.converged} ({pr.num_iter} iters)")
    print(f"   psi: {psi[0]:+.4f} {psi[20]:+.4f} {psi[100]:+.4f} "
          f"{psi[180]:+.4f} {psi[-1]:+.4f}")
    print(f"   p  : {p[0]:.3e} {p[20]:.3e} {p[100]:.3e} {p[180]:.3e}")
    print(f"   n  : {n[0]:.3e} {n[20]:.3e} {n[100]:.3e} {n[180]:.3e}")
    R, *_x, dRdn, dRdp = recombination(m, n, p)
    _Jn, Jp, Bn_p, Bn_m, Bp_m, Bp_p, cn, cp = _link_currents(
        m, n, p, mu_n, mu_p, Fn, Fp)
    res = np.empty(N)
    res[1:-1] = Jp[1:] - Jp[:-1] + Q * R[1:-1] * h
    res[0] = Fp[0] + V
    res[-1] = Fp[-1]
    w = np.argmax(np.abs(res))
    print(f"   hole res: max={np.abs(res).max():.4e} at node {w}; "
          f"Jp[{w-1}]={Jp[min(w-1, N-2)]:+.3e} Jp[{w}]={Jp[min(w, N-2)]:+.3e} "
          f"p[{w}]={p[w]:.3e} Fp[{w}]={Fp[w]:+.5f}")
    if it == 1:
        from obpds.newton import row_scale
        from scipy.sparse import diags
        from scipy.sparse.linalg import spsolve
        g = -np.maximum(np.abs(g_p), np.max(np.abs(g_p)) * 1e-30)
        mn = np.zeros(N); uu = np.zeros(N - 1); ll = np.zeros(N - 1)
        mn[1:-1] = (cp[1:] * Bp_m[1:] + cp[:-1] * Bp_p[:-1]
                    + Q * dRdp[1:-1] * h) * g[1:-1]
        uu[1:] = -cp[1:] * Bp_p[1:] * g[2:]
        ll[:-1] = -cp[:-1] * Bp_m[:-1] * g[:-2]
        mn[0] = mn[-1] = 1.0; uu[0] = 0.0; ll[-1] = 0.0
        sc = row_scale(mn, uu, ll)
        du = spsolve(diags([ll / sc[1:], mn / sc, uu / sc[:-1]],
                           [-1, 0, 1], shape=(N, N), format="csc"), -res / sc)
        j = int(np.argmax(np.abs(du)))
        print(f"   hole du: max={np.abs(du).max():.4e} at node {j}")
        for i in (0, 1, 2, j - 1, j, j + 1, N - 1):
            i = min(max(i, 0), N - 1)
            print(f"     row {i:3d}: res/sc={res[i]/sc[i]:+.3e} "
                  f"main/sc={mn[i]/sc[i]:+.3e} "
                  f"up/sc={uu[i]/sc[i] if i < N-1 else 0:+.3e} "
                  f"lo/sc={ll[i-1]/sc[i] if i > 0 else 0:+.3e} "
                  f"du={du[i]:+.3e} p={p[i]:.3e} g={g[i]:+.3e}")
        print("     Bp_m:", " ".join(f"{v:.3e}" for v in Bp_m[:5]))
        print("     Bp_p:", " ".join(f"{v:.3e}" for v in Bp_p[:5]))
    Fn, c1 = _solve_continuity(m, psi, Fn, Fp, "kane", True, -V, 0.0,
                               mu_n, mu_p, ref_c, ref_v)
    Fp_new, c2 = _solve_continuity(m, psi, Fn, Fp, "kane", False, -V, 0.0,
                                   mu_n, mu_p, ref_c, ref_v)
    print(f"   electron conv={c1}  hole conv={c2}")
    print(f"   Fp after: {Fp_new[0]:+.4f} {Fp_new[20]:+.4f} "
          f"{Fp_new[100]:+.4f} {Fp_new[180]:+.4f} {Fp_new[-1]:+.4f}")
    Fp = Fp_new
