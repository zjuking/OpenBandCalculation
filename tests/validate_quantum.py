"""
Validation of the Schrodinger solver against closed-form spectra.

Run:  python tests/validate_quantum.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from obpds.quantum import (HBAR2_2M0, solve_schrodinger,  # noqa: E402
                           subband_population)

OK, BAD = "\033[92mPASS\033[0m", "\033[91mFAIL\033[0m"


def check(name, got, want, tol, unit=""):
    rel = abs(got - want) / (abs(want) if want else 1.0)
    good = rel <= tol
    print(f"  [{'PASS' if good else 'FAIL'}] {name:34s} "
          f"got {got: .6g}{unit}  want {want: .6g}{unit}  "
          f"(rel {rel:.2e}, tol {tol:.0e})")
    return good


def section(t):
    print(f"\n=== {t} " + "=" * (60 - len(t)))


all_ok = True
M = 0.067          # GaAs Gamma-valley effective mass
HBAR2_2M = HBAR2_2M0 / M

# ------------------------------------------------------------ square well
section("Infinite square well: E_n = hbar^2 pi^2 n^2 / (2 m* L^2)")
L = 20e-7          # 20 nm in cm
for N in (401, 801, 1601):
    x = np.linspace(0.0, L, N)
    V = np.zeros(N)
    E, phi = solve_schrodinger(x, V, np.full(N, M), n_states=4)
    # Ghost nodes are pinned to zero one step outside each end, so the box
    # the discretisation actually sees spans [-h, L + h]: width L + 2h.
    Leff = L + 2.0 * (x[1] - x[0])
    for n in (1, 2, 3):
        analytic = HBAR2_2M * (np.pi * n / Leff) ** 2
        all_ok &= check(f"N={N} E_{n}", E[n - 1], analytic, 2e-3, " eV")
    # orthonormality / normalisation
    h = x[1] - x[0]
    gram = phi.T @ phi * h
    err = float(np.abs(gram - np.eye(phi.shape[1])).max())
    all_ok &= check(f"N={N} orthonormal", err, 0.0, 1e-12)

# --------------------------------------------------------- triangular well
section("Triangular well: exact Airy spectrum")
# V(x) = q F x with a hard wall at x = 0.  The eigencondition is
# Ai(-E/eps) = 0 with eps = (hbar^2 F^2 / 2 m*)^{1/3}, so
#     E_n = eps * |a_n|,   a_n = Airy zeros
# The common textbook form ((3 pi /2) qF (n + 3/4))^{2/3} is only the
# large-n asymptotic of those zeros and is ~0.8 % low on the ground state --
# fine for hand estimates, but it would hide a solver error of the same size,
# so compare against the zeros themselves.
AIRY = (2.338107410459767, 4.087949444130970, 5.520559828095551)
F = 1.0e5          # V/cm  ->  1e5 eV/cm
eps = (HBAR2_2M * F * F) ** (1.0 / 3.0)
for N in (2001, 4001):
    L = 40e-7
    x = np.linspace(0.0, L, N)
    V = F * x
    E, phi = solve_schrodinger(x, V, np.full(N, M), n_states=3)
    h = x[1] - x[0]
    # The ghost wall sits one step *outside* x = 0, where the ramp has the
    # value -F h.  Energies are measured from V(0) = 0, so every level is
    # lower than the textbook figure by exactly F h -- a constant offset
    # across the whole spectrum, which is the giveaway that it is the wall
    # position and not a discretisation error.  Without this term the
    # comparison is off by a fixed 0.2 meV at N = 2001 and 0.1 meV at
    # N = 4001 (it halves with h, as an offset must).
    for n in (0, 1, 2):
        all_ok &= check(f"N={N} E_{n}", E[n], eps * AIRY[n] - F * h,
                        5e-4, " eV")
    print(f"    (N={N}) E = "
          + ", ".join(f"{e*1e3:.2f} meV" for e in E[:3])
          + "   [analytic " + ", ".join(f"{eps*a*1e3:.2f}" for a in AIRY)
          + "]")

# ------------------------------------------------------- BenDaniel-Duke BC
section("BenDaniel-Duke interface: analytic spectrum of a mass-step well")
# A symmetric tridiagonal assembly is symmetric whatever goes into it, so
# Hermiticity proves nothing about the interface.  What does is the exact
# spectrum of an infinite well of width L split at x = a into two masses.
# Matching phi and (1/m*) phi' at the step gives
#     (k1/m1) cot(k1 A1) + (k2/m2) cot(k2 A2) = 0
# which, cleared of the cotangent poles, is the smooth function
#     g(E) = (k1/m1) cos(k1 A1) sin(k2 A2) + (k2/m2) cos(k2 A2) sin(k1 A1)
# A face average other than the harmonic mean of 1/m* reproduces neither the
# phi' jump nor this spectrum.
from scipy.optimize import brentq  # noqa: E402

M1, M2 = 0.067, 0.15
Lw, N = 30e-7, 1201
x = np.linspace(0.0, Lw, N)
h = x[1] - x[0]
j = int(np.argmin(np.abs(0.5 * (x[:-1] + x[1:]) - 15e-7)))
a = 0.5 * (x[j] + x[j + 1])
A1, A2 = a + h, Lw + h - a
m_eff = np.where(np.arange(N) <= j, M1, M2)


def g(E):
    k1 = np.sqrt(E * M1 / HBAR2_2M0)
    k2 = np.sqrt(E * M2 / HBAR2_2M0)
    return ((k1 / M1) * np.cos(k1 * A1) * np.sin(k2 * A2)
            + (k2 / M2) * np.cos(k2 * A2) * np.sin(k1 * A1))


grid = np.linspace(1e-7, 0.06, 400001)
vals = g(grid)
roots = []
for i in np.nonzero(np.sign(vals[:-1]) * np.sign(vals[1:]) < 0)[0]:
    roots.append(brentq(g, grid[i], grid[i + 1], xtol=1e-14))
print(f"    analytic: " + ", ".join(f"{r*1e3:.3f} meV" for r in roots[:3]))

E, phi = solve_schrodinger(x, np.zeros(N), m_eff, n_states=3)
print(f"    solver:   " + ", ".join(f"{e*1e3:.3f} meV" for e in E[:3]))
for n in range(min(3, len(roots))):
    all_ok &= check(f"mass-step E_{n}", E[n], roots[n], 3e-3, " eV")

# ----------------------------------------------------------- 2D population
section("2D subband population")
# Non-degenerate limit: N = (m* kT / pi hbar^2) exp((F - E)/kT).
T = 300.0
kT = 8.617333262e-5 * T
E0 = 0.05
# Exact 2D statistics is ln(1 + e^eta).  It only *equals* the Boltzmann and
# degenerate limits asymptotically, so the reference has to be taken far
# enough into each regime that the next term is below the tolerance --
# otherwise the test measures the statistics, not the implementation.
F = -0.55        # eta = -23, so e^eta ~ 1e-10
got = float(subband_population(M, np.array([E0]), F, T)[0])
want = (3.59985e10 * T * M) * np.exp((F - E0) / kT)
all_ok &= check("Boltzmann limit", got, want, 1e-7, " cm^-2")
F2 = 0.30        # eta = +9.7, so the ln(1+e^eta) - eta gap is ~6e-6
got2 = float(subband_population(M, np.array([E0]), F2, T)[0])
want2 = (3.59985e10 * T * M) * (F2 - E0) / kT
all_ok &= check("degenerate limit", got2, want2, 1e-4, " cm^-2")

# ------------------------------------------------- Schrodinger-Poisson loop
section("Self-consistent Schrodinger-Poisson")
from obpds import Layer, OhmicContact, TwoTerminalDevice  # noqa: E402

# (a) Where nothing confines, the correction must vanish identically.  This
# is the check that the subband/continuum split is consistent -- if the two
# pieces double counted, or dropped density, a plain pn junction would show
# it immediately.
diode = TwoTerminalDevice([Layer(1000, "GaAs", Na=1e17),
                           Layer(1000, "GaAs", Nd=1e17)],
                          [OhmicContact(), OhmicContact()], N=400)
cl = diode.get_equilibrium(N=400)
qm = diode.get_quantum_equilibrium(N=400, n_states=6)
diff = float(np.abs(qm.n_quantum - cl.n).max() / cl.n.max())
all_ok &= check("pn diode: quantum == classical", diff, 0.0, 1e-9)
all_ok &= check("pn diode: no spurious bound states",
                sum(qm.subbands["bound"]), 0.0, 1e-9)

# (b) A modulation-doped heterojunction: the whole point of the correction.
hemt = TwoTerminalDevice(
    [Layer(30, "AlGaAs", composition=0.3, Nd=1e18),
     Layer(10, "AlGaAs", composition=0.3, doping=0.0),
     Layer(500, "GaAs", doping=0.0)],
    [OhmicContact(), OhmicContact()], N=800)
hq = hemt.get_quantum_equilibrium(N=800, n_states=8)
hcl = hemt.get_equilibrium(N=800)
mm = hemt.model(300.0, 800)
Ec = mm.CBO_Gamma - hq.psi
i_well = int(np.argmin(Ec))
all_ok &= check("HEMT: loop converged", float(hq.converged), 1.0, 1e-9)
ns = hq.subbands["sheet_density_cm2"]
print(f"    sheet density {ns:.4e} cm^-2, "
      f"{sum(hq.subbands['bound'])} bound subband(s)")
all_ok &= check("2DEG sheet density in range", ns, 6.5e11, 0.5, " cm^-2")

# The channel must sit *behind* the interface, not on it -- the classical
# solution piles the gas exactly onto the band-edge minimum at the junction.
psi0 = np.array(hq.subbands["psi"][0])
dens = psi0 * psi0
z_mean = float(np.sum(mm.x * dens) * mm.dx)
z_cl = float(np.sum(mm.x * hcl.n) * mm.dx) / float(np.sum(hcl.n) * mm.dx)
print(f"    quantum centroid {z_mean*1e7:.2f} nm, "
      f"classical centroid {z_cl*1e7:.2f} nm, well bottom "
      f"{mm.x[i_well]*1e7:.2f} nm")
all_ok &= check("channel is displaced behind the interface",
                (z_mean - mm.x[i_well]) * 1e7, 5.0, 0.4, " nm")
all_ok &= check("quantum centroid further in than classical",
                float(z_mean > z_cl), 1.0, 1e-9)

# Quantization raises the electron energy, so the *bound* channel holds
# fewer electrons than the classical solution piles onto the interface.
# The full quantum density n_quantum additionally carries the continuum
# compensation n3*Q(3/2,delta), whose Fermi-Dirac weight is an approximation
# that can push the total above the classical density near the well -- that
# is a separate, separately quantified approximation (validate_continuum.py),
# not what this check is about.  Compare the bound sheet density instead.
sel = mm.x > 41e-7
q_ch = float(ns)                                    # bound sheet density
c_ch = float(np.trapezoid(hcl.n[sel], dx=mm.dx))
print(f"    bound channel {q_ch:.4e} vs classical {c_ch:.4e} cm^-2 "
      f"(ratio {q_ch / c_ch:.4f})")
all_ok &= check("bound channel charge <= classical", q_ch / c_ch, 0.9,
                0.25)

# Subband energy must land in the right decade.  The bare-triangular Airy
# estimate with the field at the interface is an *upper* bound here: the
# self-consistent well is steeper near the interface and flatter further in,
# which pushes the ground state down.
F = abs(float((Ec[i_well + 6] - Ec[i_well]) / (6 * mm.dx)))
airy = (HBAR2_2M * F * F) ** (1.0 / 3.0) * 2.338107410459767
E0 = hq.subbands["levels_eV"][0] - float(Ec.min())
print(f"    E0 - Ec,min = {E0*1e3:.2f} meV vs bare-triangular "
      f"{airy*1e3:.2f} meV at F = {F:.3e} V/cm")
all_ok &= check("subband energy below the bare-triangular bound", E0,
                0.77 * airy, 0.20, " eV")

print("\n" + ("ALL QUANTUM CHECKS PASSED" if all_ok
              else "SOME QUANTUM CHECKS FAILED"))
sys.exit(0 if all_ok else 1)
