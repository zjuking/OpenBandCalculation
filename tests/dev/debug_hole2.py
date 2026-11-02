"""Trace the hole continuity Newton at a properly staged V>0 state."""
from __future__ import annotations
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from obpds import Layer, OhmicContact, TwoTerminalDevice
from obpds.transport import (_mu, _link_currents, _densities,
                             _majority_fermi, recombination, _gummel)
from obpds.solver import (charge_neutral_guess, solve_poisson,
                          equilibrium_densities, Q)
from obpds.newton import row_scale, newton
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

dev = TwoTerminalDevice([Layer(1000, "GaAs", Na=1e17),
                         Layer(1000, "GaAs", Nd=1e17)],
                        [OhmicContact(), OhmicContact()], N=200)
m = dev.model(300.0, 200)
N = m.N
mu_n, mu_p = _mu(m)
ref_c, ref_v = float(m.Nc.max()), float(m.Nv.max())
h = m.dx
equilibrium_densities(m, "kane")

V = float(sys.argv[1]) if len(sys.argv) > 1 else 0.3
stages = np.linspace(0.0, V, 7)[:-1]
psi, Fn, Fp = None, None, None
for Vs in stages:
    psi, Fn, Fp, ok, it = _gummel(m, Vs, "kane", 40, None, mu_n, mu_p,
                                  ref_c, ref_v, 1e-7)
print("staged to", stages[-1], "ok", ok)

psi, _ = solve_poisson(m, psi, Fp, Fn, V, 0.0, "kane")
fl, fr = -V, 0.0

electron = (sys.argv[2] == "e") if len(sys.argv) > 2 else False
x0 = (Fn if electron else Fp).copy()
scale = np.ones(N); dscale = np.ones(N)


def state(F):
    return _densities(m, psi, F, Fp, "kane") if electron else \
        _densities(m, psi, Fn, F, "kane")


def residual(F):
    p, n, *_r, g_n, g_p = state(F)
    R, *_x = recombination(m, n, p)
    Jn, Jp, *_ = _link_currents(m, n, p, mu_n, mu_p,
                                F if electron else Fn,
                                Fp if electron else F)
    r = np.empty(N)
    if electron:
        r[1:-1] = Jn[1:] - Jn[:-1] - Q * R[1:-1] * h
    else:
        r[1:-1] = Jp[1:] - Jp[:-1] + Q * R[1:-1] * h
    r[0] = F[0] - fl; r[-1] = F[-1] - fr
    return r / scale


def jacobian(F):
    p, n, *_r, g_n, g_p = state(F)
    _R, *_x, dRdn, dRdp = recombination(m, n, p)
    _Jn, _Jp, Bn_p, Bn_m, Bp_m, Bp_p, cn, cp = _link_currents(
        m, n, p, mu_n, mu_p, F if electron else Fn, Fp if electron else F)
    main = np.zeros(N); up = np.zeros(N - 1); lo = np.zeros(N - 1)
    g = g_n if electron else g_p
    g_abs = np.abs(g)
    g_ref = max(float(np.max(g_abs)), 1e-300)
    dscale[:] = g_ref / np.maximum(g_abs, g_ref * 1e-200)
    g = np.full(N, g_ref if electron else -g_ref)
    if electron:
        dR = dRdn
        main[1:-1] = (-cn[1:]*Bn_p[1:] - cn[:-1]*Bn_m[:-1] - Q*dR[1:-1]*h)*g[1:-1]
        up[1:] = cn[1:]*Bn_m[1:]*g[2:]
        lo[:-1] = cn[:-1]*Bn_p[:-1]*g[:-2]
    else:
        dR = dRdp
        main[1:-1] = (cp[1:]*Bp_m[1:] + cp[:-1]*Bp_p[:-1] + Q*dR[1:-1]*h)*g[1:-1]
        up[1:] = -cp[1:]*Bp_p[1:]*g[2:]
        lo[:-1] = -cp[:-1]*Bp_m[:-1]*g[:-2]
    main[0] = 1.0; main[-1] = 1.0; up[0] = 0.0; lo[-1] = 0.0
    scale[:] = row_scale(main, up, lo)
    return diags([lo/scale[1:], main/scale, up/scale[:-1]],
                 [-1, 0, 1], shape=(N, N), format="csc")


jacobian(x0)
r = residual(x0)
print(f"\n{'electron' if electron else 'hole'}  |res0|={np.linalg.norm(r):.4e}")
arg = np.argmax(np.abs(r))
print(f"  worst node {arg}: res={r[arg]:.4e}")
p, n, *_r, g_n, g_p = state(x0)
R, *_x = recombination(m, n, p)
Jn, Jp, *_ = _link_currents(m, n, p, mu_n, mu_p,
                            x0 if electron else Fn, Fp if electron else x0)
for j in [arg - 1, arg, arg + 1]:
    print(f"   {j}: p={p[j]:.4e} n={n[j]:.4e} R={R[j]:.4e} "
          f"Jp[j-1]={Jp[j-1] if j>0 else 0:.4e} Jp[j]={Jp[j]:.4e}")

# Newton with damping, printing
F = x0.copy()
sk = 1.0
Gn = residual(F)
gn = np.linalg.norm(Gn)
for k in range(1, 41):
    A = jacobian(F)
    d = dscale * spsolve(A.tocsc(), -Gn)
    d = np.clip(d, -0.25, 0.25)
    if np.abs(d).max() < 1e-11:
        print(f"  converged at {k}"); break
    if k <= 6:
        xi1 = np.linalg.norm(residual(F + d)) / gn
        print(f"      (undamped xi = {xi1:.3e})")
    accepted = False
    for _ in range(60):
        Fk = F + sk * d
        Gk = residual(Fk)
        gk = np.linalg.norm(Gk)
        if not np.isfinite(gk):
            sk /= 2; continue
        xi = gk / gn
        if 1.0 - xi < 0.5 * sk:
            sk /= 2
            if sk < 1e-12:
                print(f"  it {k}: damping collapsed (sk<1e-12), xi={xi:.6f}")
                break
            continue
        sk = sk / (sk + (1 - sk) * xi / 2)
        accepted = True
        break
    if not accepted:
        print(f"  it {k}: NO STEP ACCEPTED, |G|={gn:.4e}")
        break
    if k <= 12:
        print(f"  it {k}: |G| {gn:.4e} -> {gk:.4e}  xi={xi:.5f} "
              f"|d|={np.abs(d).max():.3e} sk={sk:.4f}")
    F, Gn, gn = Fk, Gk, gk
print("final |G| =", gn)
