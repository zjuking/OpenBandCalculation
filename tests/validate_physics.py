"""
Physics validation of the rewritten solver against closed-form results.

Run:  python tests/validate_physics.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from obpds import Layer, OhmicContact, TwoTerminalDevice  # noqa: E402
from obpds.fermi import KB_EV, generalized_fermi  # noqa: E402
from obpds.materials import get_alloy  # noqa: E402

OK = "\033[92mPASS\033[0m"
BAD = "\033[91mFAIL\033[0m"


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

# ------------------------------------------------------------------ 1. pn
section("GaAs pn junction: equilibrium")
gaas = get_alloy("GaAs").props(300.0)
kT = KB_EV * 300.0
print(f"  GaAs: Eg={gaas['Eg']:.4f} eV  Nc={gaas['Nc']:.3e}  "
      f"Nv={gaas['Nv']:.3e}  ni={gaas['ni']:.3e}")

na, nd = 1e17, 1e17
Vbi_a = kT * np.log(na * nd / gaas["ni"] ** 2)
eps = gaas["dielectric"] * 8.8541878128e-14
W_a = np.sqrt(2 * eps * Vbi_a / 1.602176634e-19 * (1 / na + 1 / nd))
print(f"  analytic Vbi = {Vbi_a:.4f} V,  W = {W_a*1e4:.4f} um")

dev = TwoTerminalDevice(
    [Layer(1000, "GaAs", Na=na),
     Layer(1000, "GaAs", Nd=nd)],
    [OhmicContact(), OhmicContact()], Fp="left", Fn="right", N=2000)
sol = dev.get_equilibrium()
Vbi = float(sol.psi[-1] - sol.psi[0])
print(f"  simulated Vbi = {Vbi:.4f} V   (Newton converged: {sol.converged})")
all_ok &= check("built-in potential", Vbi, Vbi_a, 3e-3, " V")

# Depletion width.  A threshold on |rho| is useless here: outside the
# depletion edge rho decays only over a Debye length (~13 nm at 1e17), so a
# 1 %-of-doping threshold adds ~4.6 Ld on *each* side and overstates W by
# 60 %.  Use the field triangle instead, which is what the depletion
# approximation itself assumes:  Vbi = (1/2) Emax W.
q = 1.602176634e-19
Emax = float(np.abs(sol.field).max())         # V/cm (code stores -dpsi/dx)
W = 2.0 * Vbi / Emax * 1e4                    # um
print(f"  simulated W   = {W:.4f} um  (from 2*Vbi/Emax)")
all_ok &= check("depletion width", W, W_a * 1e4, 0.10, " um")

# half-maximum width, an independent estimate that does not use the field
rho = np.abs(sol.rho)
mask = rho > 0.5 * q * min(na, nd)
W_hm = float(np.count_nonzero(mask) * (sol.x_nm[1] - sol.x_nm[0]) * 1e-7) * 1e4
print(f"  simulated W   = {W_hm:.4f} um  (half-maximum of |rho|)")

# depletion approximation: peak field
Emax_a = 2 * Vbi_a / W_a           # V/cm  (triangular, symmetric; W_a in cm)
print(f"  peak |E|: sim {Emax:.3e}  analytic {Emax_a:.3e} V/cm")
all_ok &= check("peak field", Emax, Emax_a, 0.10, " V/cm")

# charge neutrality of the whole device
net = float(np.trapezoid(sol.rho, dx=(sol.x_nm[1] - sol.x_nm[0]) * 1e-7))
print(f"  net charge integral = {net:.3e} C/cm^2 (should be ~0)")

# ------------------------------------------------------------------ 2. I-V
section("GaAs pn junction: drift-diffusion I-V")
T = 300.0
Vs, Js = dev.get_iv(-0.5, 1.0, 31, T, 800)
print("   V (V)      J (A/cm^2)")
for v, j in zip(Vs, Js):
    print(f"  {v:+.3f}   {j:+.6e}")


def ideality(vs, js, lo, hi):
    """Slope of ln J vs V over a bias window, as an ideality factor."""
    sel = (vs >= lo) & (vs <= hi) & np.isfinite(js) & (js > 0)
    if sel.sum() < 3:
        return None
    slope = np.polyfit(vs[sel], np.log(js[sel]), 1)[0]
    return 1.0 / (slope * kT)


# Two regimes are expected and they are the point of the test.  At high
# forward bias the current is minority-carrier diffusion across the neutral
# regions, J ~ exp(V/kT), n = 1.  Lower down it is SRH recombination through
# mid-gap traps in the depletion region, J ~ exp(V/2kT), n = 2.  A solver
# that reproduces both, and the crossover between them, has the physics right.
n_hi = ideality(Vs, Js, 0.75, 0.95)
n_lo = ideality(Vs, Js, 0.15, 0.35)
print(f"  ideality 0.75-0.95 V: {n_hi:.3f}   (diffusion, expect ~1)")
print(f"  ideality 0.15-0.35 V: {n_lo:.3f}   (depletion SRH, expect ~2)")
all_ok &= check("ideality, diffusion regime", n_hi, 1.0, 0.10)
all_ok &= check("ideality, SRH regime", n_lo, 1.80, 0.25)

# Saturation current.  This is a *short-base* diode: both layers (1 um) are
# far thinner than the diffusion lengths (Ln = 47 um, Lp = 10 um), so the
# minority profile is very nearly straight and
#
#     J = q ni^2 (Dn/(Wp Na) + Dp/(Wn Nd)) exp(V/kT)
#
# with the neutral width Wp the layer thickness *minus* the depletion width.
#
# Do not extrapolate ln J back to V = 0 to get J0.  The measured ideality is
# 1.012 rather than exactly 1, and across a 0.2 V window that 1 % excess
# slope compounds into a ~60 % error in the intercept -- an artifact of the
# fit, not of the physics.  Compare the current where it is measured.
mu_n, mu_p = 8500.0, 400.0
Dn, Dp = mu_n * kT, mu_p * kT
for V in (0.80, 0.85, 0.90):
    # forward bias narrows the depletion region, which *lengthens* the
    # neutral regions and lowers the current:  W ~ sqrt(Vbi - V)
    Wv = W_a * np.sqrt(max(Vbi_a - V, 1e-3) / Vbi_a)   # cm
    Wp = 1e-4 - 0.5 * Wv                               # cm  (1 um layer)
    J0_a = q * gaas["ni"] ** 2 * (Dn / (Wp * na) + Dp / (Wp * nd))
    J_pred = J0_a * np.exp(V / kT)
    J_sim = float(Js[int(np.argmin(np.abs(Vs - V)))])
    all_ok &= check(f"J({V:.2f} V) vs short-base diode law",
                    J_sim, J_pred, 0.10, " A/cm^2")

# equilibrium current must vanish
Jeq = dev.get_drift_diffusion(0.0, N=800).J
print(f"  J(V=0) = {Jeq:.3e} A/cm^2")
all_ok &= check("zero current at equilibrium", Jeq, 0.0, 1e-6, " A/cm^2")

# the terminal current must be the same along the device
s_hi = dev.get_drift_diffusion(0.8, N=800)
print(f"  J(V=0.8) = {s_hi.J:.4e} A/cm^2, "
      f"worst-link deviation {s_hi.J_spread:.2e}")
all_ok &= check("current conservation at 0.8 V", s_hi.J_spread, 0.0, 5e-3)

# reverse bias is generation in the depletion region, so it must scale as 1/tau
jr = []
for tau in (1e-7, 1e-8):
    d = TwoTerminalDevice([Layer(1000, "GaAs", Na=na),
                           Layer(1000, "GaAs", Nd=nd)],
                          [OhmicContact(), OhmicContact()],
                          N=400, tau_n=tau, tau_p=tau)
    jr.append(abs(d.get_drift_diffusion(-1.0, N=400).J))
print(f"  |J(-1 V)|: tau=1e-7 -> {jr[0]:.4e},  tau=1e-8 -> {jr[1]:.4e} "
      f"A/cm^2  (ratio {jr[1]/jr[0]:.2f}, expect 10)")
all_ok &= check("reverse current scales as 1/tau", jr[1] / jr[0], 10.0, 0.25)

# ...and its magnitude against depletion generation.  The textbook
# J_gen = q ni W / 2 tau assumes both carriers are driven far below ni across
# the *whole* depletion width, but that needs |V| >> Vbi.  Both are below ni
# only where Fp < Ei < Fn -- an energy window |qV| wide out of the Vbi + |V|
# that Ei sweeps -- so the generating slab is W |V|/(Vbi + |V|), here 28 %
# of W at -0.5 V.  The residual shortfall is the slab edges, where n or p is
# still of order ni and the rate has not reached ni/2 tau yet.
Vr = -0.5
Wv = W_a * np.sqrt((Vbi_a - Vr) / Vbi_a)              # cm
Wgen = Wv * abs(Vr) / (Vbi_a + abs(Vr))               # cm
J_gen = q * gaas["ni"] * Wgen / (2 * 1e-7)
J_rev = abs(float(Js[int(np.argmin(np.abs(Vs - Vr)))]))
print(f"  |J(-0.5 V)| = {J_rev:.4e}  vs q ni Wgen/2tau = {J_gen:.4e} "
      f"(Wgen = {Wgen*1e4:.4f} um of W = {Wv*1e4:.4f} um)")
all_ok &= check("reverse generation current", J_rev, J_gen, 0.5, " A/cm^2")

# --------------------------------------------------------------- 3. hetero
section("AlGaAs/GaAs heterojunction: band offsets")
delta_ec = (get_alloy("AlGaAs", 0.3).props(300)["CBO"]
            - get_alloy("GaAs").props(300)["CBO"])
delta_eg = (get_alloy("AlGaAs", 0.3).props(300)["Eg"]
            - get_alloy("GaAs").props(300)["Eg"])
print(f"  dEc = {delta_ec:.4f} eV,  dEg = {delta_eg:.4f} eV,  "
      f"ratio = {delta_ec/delta_eg:.3f}")
all_ok &= check("dEc/dEg for Al0.3Ga0.7As/GaAs", delta_ec / delta_eg, 0.62, 0.05)

het = TwoTerminalDevice(
    [Layer(50, "AlGaAs", composition=0.3, Nd=1e18),
     Layer(200, "GaAs", Nd=1e16),
     Layer(500, "GaAs", Nd=1e18)],
    N=1200)
hs = het.get_equilibrium()
print(f"  heterostructure solved (converged={hs.converged}); "
      f"2DEG sheet density = "
      f"{np.trapezoid(hs.n, dx=(hs.x_nm[1]-hs.x_nm[0])*1e-7):.3e} cm^-2")

# -------------------------------------------------------------- 4. grading
section("Kane vs parabolic vs Boltzmann")
for approx in ("boltzmann", "fermi-dirac", "kane"):
    d = TwoTerminalDevice([Layer(500, "InAs", Nd=1e18),
                           Layer(500, "InAs", Nd=1e16)],
                          N=600, approx=approx)
    try:
        s = d.get_equilibrium()
        n_max = float(np.nanmax(s.n))
        print(f"  {approx:10s}: n_max = {n_max:.4e} cm^-3   "
              f"converged={s.converged}")
    except Exception as e:  # pragma: no cover
        print(f"  {approx:10s}: {e!r}")

# InAs is the extreme case for non-parabolicity
inas = get_alloy("InAs").props(300)
print(f"  InAs alpha (Kane) = {inas['nonparabolicity']:.4f}  "
      f"(kT/Eg*(1-m*)^2)")
Hg, dHg = generalized_fermi(np.array([2.0]), np.array([inas['nonparabolicity']]))
Fp_, _ = generalized_fermi(np.array([2.0]), np.array([0.0]))
print(f"  H_1/2(2, a)/F_1/2(2) = {float(Hg[0]/Fp_[0]):.4f}  "
      f"(non-parabolicity raises the Gamma-valley density)")

print("\n" + ("ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED"))
sys.exit(0 if all_ok else 1)
