"""Why does the continuity Newton fail at V = 0 (exact equilibrium)?"""
from __future__ import annotations
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from obpds import Layer, OhmicContact, TwoTerminalDevice
from obpds.transport import (_mu, _link_currents, _densities,
                             _solve_continuity, _majority_fermi, recombination)
from obpds.solver import charge_neutral_guess, solve_poisson, Q

dev = TwoTerminalDevice([Layer(1000, "GaAs", Na=1e17),
                         Layer(1000, "GaAs", Nd=1e17)],
                        [OhmicContact(), OhmicContact()], N=200)
m = dev.model(300.0, 200)
N = m.N
mu_n, mu_p = _mu(m)
ref_c, ref_v = float(m.Nc.max()), float(m.Nv.max())
h = m.dx

z = np.zeros(N)
psi = charge_neutral_guess(m, z, z, "kane")
psi, _ = solve_poisson(m, psi, z, z, 0.0, 0.0, "kane")
Fn = np.zeros(N)
Fp = np.zeros(N)
p, n, *_r, g_n, g_p = _densities(m, psi, Fn, Fp, "kane")
R, rs, rr, dRdn, dRdp = recombination(m, n, p)
print("R at equilibrium: min %.4e max %.4e" % (R.min(), R.max()))
print("np - ni^2       : min %.4e max %.4e" % ((n * p - m.ni**2).min(),
                                               (n * p - m.ni**2).max()))
print("q*R*h           : %.4e" % (Q * np.abs(R).max() * h))
Jn, Jp, *_ = _link_currents(m, n, p, mu_n, mu_p, Fn, Fp)
print("|Jn|max %.3e   |Jp|max %.3e" % (np.abs(Jn).max(), np.abs(Jp).max()))

for electron in (True, False):
    from obpds.newton import newton, row_scale
    from scipy.sparse import diags
    from scipy.sparse.linalg import spsolve
    scale = np.ones(N); dscale = np.ones(N)
    x0 = Fn if electron else Fp

    def state(F):
        Fn_ = F if electron else Fn
        Fp_ = Fp if electron else F
        p, n, *_r, g_n, g_p = _densities(m, psi, Fn_, Fp_, "kane")
        return p, n, g_n, g_p

    def residual(F):
        p, n, g_n, g_p = state(F)
        R, *_x = recombination(m, n, p)
        Jn, Jp, *_ = _link_currents(m, n, p, mu_n, mu_p,
                                    F if electron else Fn,
                                    Fp if electron else F)
        res = np.empty(N)
        if electron:
            res[1:-1] = Jn[1:] - Jn[:-1] - Q * R[1:-1] * h
        else:
            res[1:-1] = Jp[1:] - Jp[:-1] + Q * R[1:-1] * h
        res[0] = F[0]; res[-1] = F[-1]
        return res / scale

    def jacobian(F):
        p, n, g_n, g_p = state(F)
        _R, *_x, dRdn, dRdp = recombination(m, n, p)
        _Jn, _Jp, Bn_p, Bn_m, Bp_m, Bp_p, cn, cp = _link_currents(
            m, n, p, mu_n, mu_p, F if electron else Fn, Fp if electron else F)
        main = np.zeros(N); up = np.zeros(N - 1); lo = np.zeros(N - 1)
        g = g_n if electron else g_p
        g_ref = max(float(np.max(g)), 1e-300)
        dscale[:] = g_ref / np.maximum(g, g_ref * 1e-200)
        g = np.full(N, g_ref)
        if electron:
            dR = dRdn
            main[1:-1] = (-cn[1:]*Bn_p[1:] - cn[:-1]*Bn_m[:-1]
                          - Q*dR[1:-1]*h) * g[1:-1]
            up[1:] = cn[1:]*Bn_m[1:]*g[2:]
            lo[:-1] = cn[:-1]*Bn_p[:-1]*g[:-2]
        else:
            dR = dRdp
            main[1:-1] = (cp[1:]*Bp_m[1:] + cp[:-1]*Bp_p[:-1]
                          + Q*dR[1:-1]*h) * g[1:-1]
            up[1:] = -cp[1:]*Bp_p[1:]*g[2:]
            lo[:-1] = -cp[:-1]*Bp_m[:-1]*g[:-2]
        main[0] = 1.0; main[-1] = 1.0; up[0] = 0.0; lo[-1] = 0.0
        scale[:] = row_scale(main, up, lo)
        return diags([lo/scale[1:], main/scale, up/scale[:-1]],
                     [-1, 0, 1], shape=(N, N), format="csc")

    jacobian(x0)
    F = x0.copy()
    print(f"\n--- {'electron' if electron else 'hole'} ---")
    for i in range(8):
        r = residual(F)
        A = jacobian(F)
        d = dscale * spsolve(A.tocsc(), -r)
        print(f"  it {i}  |res|={np.linalg.norm(r):.4e}  "
              f"|step|={np.abs(d).max():.4e}  "
              f"argmax={np.argmax(np.abs(d))}")
        if np.abs(d).max() < 1e-11:
            print("  -> converged"); break
        F = F + np.clip(d, -0.25, 0.25)
