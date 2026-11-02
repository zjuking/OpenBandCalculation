"""Finite-difference check of the continuity Jacobian."""
from __future__ import annotations
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from obpds import Layer, OhmicContact, TwoTerminalDevice
from obpds.transport import (_mu, _link_currents, _densities,
                             _solve_continuity, _majority_fermi, recombination)
from obpds.solver import charge_neutral_guess, solve_poisson, equilibrium_densities
from obpds.solver import Q
from obpds.fermi import KB_EV

dev = TwoTerminalDevice([Layer(1000, "GaAs", Na=1e17),
                         Layer(1000, "GaAs", Nd=1e17)],
                        [OhmicContact(), OhmicContact()], N=60)
m = dev.model(300.0, 60)
N = m.N
mu_n, mu_p = _mu(m)
ref_c, ref_v = float(m.Nc.max()), float(m.Nv.max())
h = m.dx
kT = KB_EV * 300.0
equilibrium_densities(m, "kane")

z = np.zeros(N)
psi = charge_neutral_guess(m, z, z, "kane")
psi, _ = solve_poisson(m, psi, z, z, 0.0, 0.0, "kane")

V = 0.2
Fn = _majority_fermi(m, V)
Fp = Fn.copy()
psi, _ = solve_poisson(m, psi, Fp, Fn, V, 0.0, "kane")

# use boltzmann so we can compare against the analytic derivative cleanly
approx = "boltzmann"


def res_hole(F):
    p, n, *_r, g_n, g_p = _densities(m, psi, Fn, F, approx)
    R, *_x = recombination(m, n, p)
    Jn, Jp, *_ = _link_currents(m, n, p, mu_n, mu_p, Fn, F)
    r = np.empty(N)
    r[1:-1] = Jp[1:] - Jp[:-1] + Q * R[1:-1] * h
    r[0] = F[0] - (-V)
    r[-1] = F[-1] - 0.0
    return r


def res_elec(F):
    p, n, *_r, g_n, g_p = _densities(m, psi, F, Fp, approx)
    R, *_x = recombination(m, n, p)
    Jn, Jp, *_ = _link_currents(m, n, p, mu_n, mu_p, F, Fp)
    r = np.empty(N)
    r[1:-1] = Jn[1:] - Jn[:-1] - Q * R[1:-1] * h
    r[0] = F[0] - (-V)
    r[-1] = F[-1] - 0.0
    return r


def fd_jac(resfun, F0, rel=1e-6):
    r0 = resfun(F0)
    J = np.zeros((N, N))
    for j in range(N):
        d = rel * max(abs(F0[j]), 0.05)
        Fp1 = F0.copy(); Fp1[j] += d
        Fm1 = F0.copy(); Fm1[j] -= d
        J[:, j] = (resfun(Fp1) - resfun(Fm1)) / (2 * d)
    return J, r0


def analytic(resfun_kind, F0):
    # rebuild via _solve_continuity internals is awkward; redo here
    if resfun_kind == "hole":
        p, n, *_r, g_n, g_p = _densities(m, psi, Fn, F0, approx)
        g = g_p
        _R, *_x, dRdn, dRdp = recombination(m, n, p)
        dR = dRdp
        _Jn, _Jp, Bn_p, Bn_m, Bp_m, Bp_p, cn, cp = _link_currents(
            m, n, p, mu_n, mu_p, Fn, F0)
    else:
        p, n, *_r, g_n, g_p = _densities(m, psi, F0, Fp, approx)
        g = g_n
        _R, *_x, dRdn, dRdp = recombination(m, n, p)
        dR = dRdn
        _Jn, _Jp, Bn_p, Bn_m, Bp_m, Bp_p, cn, cp = _link_currents(
            m, n, p, mu_n, mu_p, F0, Fp)
    main = np.zeros(N); up = np.zeros(N - 1); lo = np.zeros(N - 1)
    if resfun_kind == "hole":
        main[1:-1] = (cp[1:]*Bp_m[1:] + cp[:-1]*Bp_p[:-1] + Q*dR[1:-1]*h)*g[1:-1]
        up[1:] = -cp[1:]*Bp_p[1:]*g[2:]
        lo[:-1] = -cp[:-1]*Bp_m[:-1]*g[:-2]
    else:
        main[1:-1] = (-cn[1:]*Bn_p[1:] - cn[:-1]*Bn_m[:-1] - Q*dR[1:-1]*h)*g[1:-1]
        up[1:] = cn[1:]*Bn_m[1:]*g[2:]
        lo[:-1] = cn[:-1]*Bn_p[:-1]*g[:-2]
    main[0] = 1.0; main[-1] = 1.0; up[0] = 0.0; lo[-1] = 0.0
    A = np.zeros((N, N))
    A[np.arange(N), np.arange(N)] = main
    A[np.arange(N - 1), np.arange(1, N)] = up
    A[np.arange(1, N), np.arange(N - 1)] = lo
    return A


for kind, resfun, F0 in (("hole", res_hole, Fp.copy()),
                         ("electron", res_elec, Fn.copy())):
    Jfd, r0 = fd_jac(resfun, F0)
    Jan = analytic(kind, F0)
    sel = np.abs(Jan) > 1e-30
    err = np.abs(Jan - Jfd) / np.maximum(np.abs(Jan), 1e-300)
    print(f"\n{kind}: median rel err {np.median(err[sel]):.3e}  "
          f"max {np.max(err[sel]):.3e}")
    bad = np.argwhere(err > 0.05)
    print(f"  entries with >5% error: {len(bad)}")
    for i, j in bad[:12]:
        print(f"    ({i},{j}) an={Jan[i,j]:.6e} fd={Jfd[i,j]:.6e}")
    # row sums
    print(f"  row sum (analytic) max |.| = {np.abs(Jan.sum(axis=1)).max():.4e}")
    print(f"  row sum (fd)        max |.| = {np.abs(Jfd.sum(axis=1)).max():.4e}")
