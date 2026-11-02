"""
Validation of the pseudomorphic strain model.

    python tests/validate_strain.py

Everything here is checked against something closed-form or independently
measurable: the zero-strain limit, the tracelessness of the shear part of the
Bir-Pikus Hamiltonian, the Delta_SO -> infinity asymptote of the heavy/light
hole splitting, the hydrostatic pressure coefficient of the band gap, and the
Matthews-Blakeslee critical thickness.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from obpds import (Layer, OhmicContact, Substrate, TwoTerminalDevice,  # noqa: E402
                   get_alloy)
from obpds.materials.alloy import N_PREF  # noqa: E402
from obpds.strain import (biaxial_strain, critical_thickness, strained_nv,  # noqa: E402
                          valence_edges)


def check(name, got, want, tol, unit=""):
    rel = abs(got - want) / (abs(want) if want else 1.0)
    good = bool(rel <= tol) or (got == want == 0.0)
    print(f"  [{'PASS' if good else 'FAIL'}] {name:44s} "
          f"got {got: .6g}{unit}  want {want: .6g}{unit}  "
          f"(rel {rel:.2e}, tol {tol:.0e})")
    return good


def section(t):
    print(f"\n=== {t} " + "=" * (58 - len(t)))


all_ok = True

# ---------------------------------------------------------------- strain state
section("Biaxial strain state on a (001) substrate")
sub = Substrate("GaAs")
a_gaas = sub.lattice_constant(300.0)
all_ok &= check("GaAs substrate lattice constant", a_gaas, 5.65325, 1e-5, " A")

# In0.2Ga0.8As: GaInAs takes the *Ga* fraction, so x = 0.8
well = get_alloy("GaInAs", 0.8).props(300.0)
a_well = well["lattice_constant"]
eps_in, eps_out = biaxial_strain(a_well, a_gaas, well["C11"], well["C12"])
f = (a_well - a_gaas) / a_gaas
print(f"    a_layer = {a_well:.4f} A, mismatch f = {f * 100:+.3f} %")
all_ok &= check("in-plane strain is compressive", float(eps_in < 0), 1.0, 1e-9)
# eps_in = (a_sub - a_layer)/a_layer = -f/(1+f)
all_ok &= check("eps_in = -f/(1+f)", float(eps_in), -f / (1.0 + f), 1e-12)
r = well["C12"] / well["C11"]
all_ok &= check("eps_out = -2 C12/C11 eps_in", float(eps_out),
                -2.0 * r * float(eps_in), 1e-12)
tr = 2.0 * float(eps_in) + float(eps_out)
print(f"    eps_in {eps_in * 100:+.3f} %, eps_out {eps_out * 100:+.3f} %, "
      f"Tr(eps) {tr * 100:+.3f} %")

# ------------------------------------------------------- valence band algebra
section("Bir-Pikus valence edges: sum rule and limits")


def triplet(a_v, b, eps_in, eps_out, Ev, D):
    h = a_v * (2.0 * eps_in + eps_out)
    hh, lh, so = valence_edges(Ev, D, a_v, b, eps_in, eps_out)
    return h, float(hh), float(lh), float(so)


for name in ("GaAs", "InAs", "InP", "GaSb"):
    p = get_alloy(name).props(300.0)
    for strain in (-0.02, -0.005, 0.005, 0.02):
        ei, eo = biaxial_strain(p["lattice_constant"] * (1.0 + strain),
                                p["lattice_constant"], p["C11"], p["C12"])
        h, hh, lh, so = triplet(p["a_v"], p["b_shear"], ei, eo, p["VBO"],
                                p["Delta_SO"])
        # The shear part of the strain Hamiltonian is traceless, so however
        # far the three edges are pushed apart, their sum is fixed by the
        # hydrostatic part alone.
        resid = (hh - p["VBO"] - h) + (lh - p["VBO"] - h) + (so - p["VBO"] - h)
        all_ok &= check(f"{name} @ {strain * 100:+.1f}% : shear is traceless",
                        resid + p["Delta_SO"], 0.0, 1e-9, " eV")

# zero strain: HH and LH degenerate at the VBM, split-off exactly Delta below
p = get_alloy("GaAs").props(300.0)
_h, hh, lh, so = triplet(p["a_v"], p["b_shear"], 0.0, 0.0, p["VBO"],
                         p["Delta_SO"])
all_ok &= check("unstrained: HH = LH = VBM", hh - lh, 0.0, 1e-12, " eV")
all_ok &= check("unstrained: SO = VBM - Delta_SO", so, p["VBO"] - p["Delta_SO"],
                1e-12, " eV")

# Delta_SO -> infinity: the 2x2 light-hole/split-off mixing switches off and
# the splitting tends exactly to the shear parameter d = -2b(eps_out - eps_in).
ei, eo = biaxial_strain(a_well, a_gaas, well["C11"], well["C12"])
d = -2.0 * well["b_shear"] * float(eo - ei)


def split_at(D):
    hh, lh, _ = valence_edges(well["VBO"], D, well["a_v"], well["b_shear"],
                              ei, eo)
    return float(hh - lh)


# The light hole mixes with the split-off band through a term of order d; that
# mixing is what keeps the finite-Delta_SO splitting *below* d.  It switches
# off as 1/Delta_SO, so the error has to fall by two when D doubles -- that
# convergence order is a stronger statement than any single tolerance.
for D in (10.0, 100.0, 1000.0):
    err = d - split_at(D)
    all_ok &= check(f"HH-LH -> d as Delta_SO -> inf (D={D:g} eV)",
                    split_at(D), d, 1.2 * d / D, " eV")
err1, err2 = d - split_at(100.0), d - split_at(200.0)
all_ok &= check("mixing error falls as 1/Delta_SO", err1 / err2, 2.0, 5e-2)
print(f"    shear parameter d = {d * 1000:.1f} meV; "
      f"finite Delta_SO = {well['Delta_SO']:.3f} eV gives")

hh, lh, _so = valence_edges(well["VBO"], well["Delta_SO"], well["a_v"],
                            well["b_shear"], ei, eo)
split = float(hh - lh)
print(f"    HH-LH splitting = {split * 1000:.1f} meV for In0.2Ga0.8As/GaAs")
all_ok &= check("compressive: HH sits above LH", float(hh > lh), 1.0, 1e-9)
all_ok &= check("HH-LH splitting is tens of meV", split, 0.080, 0.35, " eV")

# tensile: the light hole comes to the top
a_inp = get_alloy("InP").props(300.0)["lattice_constant"]
ei_t, eo_t = biaxial_strain(a_well, a_inp, well["C11"], well["C12"])
hh_t, lh_t, _ = valence_edges(well["VBO"], well["Delta_SO"], well["a_v"],
                              well["b_shear"], ei_t, eo_t)
all_ok &= check("tensile: LH sits above HH", float(lh_t > hh_t), 1.0, 1e-9)

# ------------------------------------------------------ hydrostatic / pressure
section("Hydrostatic shift vs the measured pressure coefficient")
# dEg/dP = -(a_c - a_v)/B with B = (C11 + 2 C12)/3.  Measured coefficients for
# the direct gap cluster around 8-11 meV/kbar across the arsenides and
# phosphides; the tabulated deformation potentials are not good enough to
# pin anything tighter than that, except for GaAs where the value is one of
# the best known numbers in semiconductor physics.
for name in ("GaAs", "InAs", "InP", "AlAs", "GaP", "AlP", "GaSb", "InSb"):
    p = get_alloy(name).props(300.0)
    B = (p["C11"] + 2.0 * p["C12"]) / 3.0 * 100.0      # 10^11 dyn/cm^2 -> kbar
    dEdP = -(p["a_c"] - p["a_v"]) / B
    print(f"    {name}: a_c {p['a_c']:+.2f}, a_v {p['a_v']:+.2f}, "
          f"B {B:.0f} kbar -> dEg/dP {dEdP * 1e3:.2f} meV/kbar")
    all_ok &= check(f"{name}: dEg/dP is 5-20 meV/kbar",
                    float(5e-3 <= dEdP <= 20e-3), 1.0, 1e-9)
    all_ok &= check(f"{name}: compression widens the gap",
                    float(p["a_c"] - p["a_v"] < 0), 1.0, 1e-9)
p = get_alloy("GaAs").props(300.0)
B = (p["C11"] + 2.0 * p["C12"]) / 3.0 * 100.0
all_ok &= check("GaAs dEg/dP against the measured 10.8 meV/kbar",
                -(p["a_c"] - p["a_v"]) / B, 10.8e-3, 1.5e-1, " eV/kbar")

# -------------------------------------------------------------- density of states
section("Valence density of states under splitting")
T = 300.0
p = get_alloy("GaAs").props(T)
nv_un = N_PREF * T ** 1.5 * (p["meff_hh_100"] ** 1.5 + p["meff_lh_100"] ** 1.5)
all_ok &= check("unstrained Nv is reproduced",
                float(strained_nv(T, p["meff_hh_100"], p["meff_lh_100"],
                                  p["VBO"], p["VBO"])), float(p["Nv"]), 1e-12,
                " cm^-3")
all_ok &= check("splitting reduces Nv below the unstrained value",
                float(strained_nv(T, p["meff_hh_100"], p["meff_lh_100"],
                                  p["VBO"], p["VBO"] - 0.2)) < nv_un, 1.0, 1e-9)
# a splitting much larger than kT leaves only the heavy hole
all_ok &= check("large splitting keeps heavy holes only",
                float(strained_nv(T, p["meff_hh_100"], p["meff_lh_100"],
                                  p["VBO"], p["VBO"] - 1.0)),
                N_PREF * T ** 1.5 * p["meff_hh_100"] ** 1.5, 1e-3, " cm^-3")

# --------------------------------------------------------- critical thickness
section("Matthews-Blakeslee critical thickness")
hc = critical_thickness(f, a_gaas, well["C11"], well["C12"])
print(f"    In0.2Ga0.8As on GaAs: h_c = {hc:.1f} A = {hc / 10:.2f} nm")
all_ok &= check("In0.2Ga0.8As/GaAs h_c", hc / 10.0, 10.0, 0.5, " nm")
all_ok &= check("lattice matched: no critical thickness",
                float(math.isinf(critical_thickness(0.0, a_gaas, 118.79, 53.76))),
                1.0, 1e-9)
prev = math.inf
for mm in (0.005, 0.01, 0.02, 0.04):
    h = critical_thickness(mm, a_gaas, well["C11"], well["C12"])
    print(f"    f = {mm * 100:.1f} % -> h_c = {h / 10:.2f} nm")
    all_ok &= check(f"h_c decreases with mismatch ({mm * 100:.1f} %)",
                    float(h < prev), 1.0, 1e-9)
    prev = h

# Multi-system cross-check against reported MB critical thicknesses.  The
# MB force-balance value is a *lower bound* -- measured critical thicknesses
# are routinely 2-4x larger because it says nothing about how fast
# dislocations actually nucleate -- so these are order-of-magnitude checks,
# not pointwise ones.  Reference values: Photonica (buried-well MB estimate,
# ~14 nm at the 1.43% of In0.2Ga0.8As and ~22 nm at 1% mismatch) and an NLR
# report (~0.5 um at the 0.07% mismatch of 1% In).
for ga, want_nm in ((0.86, 22.0), (0.99, 500.0)):
    p = get_alloy("GaInAs", ga).props(300.0)
    fm = (p["lattice_constant"] - a_gaas) / a_gaas
    hc_i = critical_thickness(fm, a_gaas, p["C11"], p["C12"]) / 10.0
    print(f"    In{1 - ga:.2f}Ga{ga:.2f}As on GaAs (f={fm * 100:.2f}%): "
          f"h_c = {hc_i:.1f} nm  (lit ~{want_nm:g} nm)")
    all_ok &= check(f"In{1 - ga:.2f}Ga{ga:.2f}As/GaAs h_c ~ literature",
                    hc_i, want_nm, 0.6, " nm")

# --------------------------------------------------------------- device level
section("Device level: strained vs unstrained")

stack = lambda: [Layer(20, "GaAs"), Layer(8, "GaInAs", composition=0.8),
                 Layer(20, "GaAs")]
dev_s = TwoTerminalDevice(stack(), substrate="GaAs", N=300)
dev_u = TwoTerminalDevice(stack(), N=300)
ms, mu = dev_s.model(300.0, 300), dev_u.model(300.0, 300)
i = ms.N // 2                                  # middle of the well
print(f"    well: Eg {mu.Eg[i]:.4f} -> {ms.Eg[i]:.4f} eV, "
      f"Ec {mu.CBO_Gamma[i]:.4f} -> {ms.CBO_Gamma[i]:.4f} eV")
all_ok &= check("GaAs buffer is unstrained", float(ms.eps_in[5]), 0.0, 1e-12)
all_ok &= check("strained gap exceeds the unstrained one",
                float(ms.Eg[i] > mu.Eg[i]), 1.0, 1e-9)
all_ok &= check("strain raises the conduction band",
                float(ms.CBO_Gamma[i] - mu.CBO_Gamma[i]), 0.10, 0.35, " eV")
# Ec(HH) < Ec(LH) under compression: the heavy-hole transition is the low one
all_ok &= check("HH transition is the lower one (compressive)",
                float(ms.CBO_Gamma[i] - ms.E_hh[i] < ms.CBO_Gamma[i] - ms.E_lh[i]),
                1.0, 1e-9)

# A lattice-matched layer must come out unstrained.  GaInAs takes the Ga
# fraction, so In0.53Ga0.47As on InP is x = 0.47, not 0.53.
lm = get_alloy("GaInAs", 0.47).props(300.0)
a_inp = get_alloy("InP").props(300.0)["lattice_constant"]
f_lm = (lm["lattice_constant"] - a_inp) / a_inp
print(f"    GaInAs(0.47) on InP: a {lm['lattice_constant']:.4f} vs "
      f"{a_inp:.4f} A, f = {f_lm * 100:+.4f} %")
all_ok &= check("In0.53Ga0.47As is lattice matched to InP", abs(f_lm),
                0.0, 1e-3)
dev_lm = TwoTerminalDevice([Layer(100, "GaInAs", composition=0.47)],
                           substrate="InP", N=200)
mlm = dev_lm.model(300.0, 200)
mlm_u = TwoTerminalDevice([Layer(100, "GaInAs", composition=0.47)],
                          N=200).model(300.0, 200)
# The residual is not round-off: a 0.03 % mismatch is still a mismatch, and
# its hydrostatic shift is worth a few meV.  What must hold is that it is
# small -- a lattice-matched layer may not pick up a strain-sized shift.
all_ok &= check("lattice matched: only a few meV of shift",
                float(abs(mlm.Eg[100] - mlm_u.Eg[100])), 0.0, 6e-3, " eV")

# and it still solves
sol = dev_s.get_equilibrium(N=300)
all_ok &= check("strained stack converges", float(sol.converged), 1.0, 1e-9)
qm = dev_s.get_quantum_equilibrium(N=300)
all_ok &= check("strained stack converges with quantum correction",
                float(qm.converged), 1.0, 1e-9)

# Growing a stack on a substrate of its own material must change nothing:
# every layer is already at the reference lattice constant.
same = TwoTerminalDevice([Layer(500, "GaAs", Na=1e16),
                          Layer(500, "GaAs", Nd=1e16)],
                         substrate="GaAs", N=300).model(300.0, 300)
bare = TwoTerminalDevice([Layer(500, "GaAs", Na=1e16),
                          Layer(500, "GaAs", Nd=1e16)],
                         N=300).model(300.0, 300)
all_ok &= check("self-substrate leaves the model untouched",
                float(np.abs(same.Eg - bare.Eg).max()), 0.0, 1e-12, " eV")
all_ok &= check("self-substrate leaves VBO untouched",
                float(np.abs(same.VBO - bare.VBO).max()), 0.0, 1e-12, " eV")

# ------------------------------------------------ 7. calibration overrides
section("Layer-level parameter overrides")

# A layer may carry band-parameter overrides for calibration against
# experiment.  The point of the interface is that the derived quantities stay
# consistent: moving VBO must move every conduction edge with it, and must
# leave the gap alone.

def well(substrate=True, **kw):
    layers = [Layer(20, "GaAs", Nd=1e17),
              Layer(8, "GaInAs", composition=0.8, **kw),
              Layer(20, "GaAs", Nd=1e17)]
    return TwoTerminalDevice(layers, substrate="GaAs" if substrate else None,
                             N=400).model(300.0, 400)

i = 200
m_ref = well()
m_bare = well(substrate=False)
# The override replaces the *unstrained* value; strain is applied on top of
# it, so the strained result must move by exactly the requested amount.
vbo_bare = float(m_bare.VBO[i])
m_ovr = well(overrides={"VBO": vbo_bare - 0.10})
all_ok &= check("VBO override lands where asked",
                float(m_ovr.VBO[i] - m_ref.VBO[i]), -0.10, 1e-9, " eV")
all_ok &= check("VBO override carries the conduction edge",
                float(m_ovr.CBO_Gamma[i] - m_ref.CBO_Gamma[i]), -0.10, 1e-9, " eV")
all_ok &= check("VBO override leaves the gap alone",
                float(m_ovr.Eg[i] - m_ref.Eg[i]), 0.0, 1e-12, " eV")

# The strain parameters are read only by strain.py, so overriding one changes
# the strain shift and nothing else.
m_ac = well(overrides={"a_c": 0.0})
all_ok &= check("a_c override removes the hydrostatic shift",
                float(m_ac.Eg[i] - m_ref.Eg[i]), -0.1019, 1e-2, " eV")

# A misspelt key must fail loudly rather than silently do nothing.
try:
    Layer(10, "GaAs", overrides={"a_V": 1.0})
    raised = 0.0
except ValueError:
    raised = 1.0
all_ok &= check("unknown override key raises", raised, 1.0, 1e-9)

try:
    Layer(10, "GaAs", overrides={"Eg": 1.0})
    raised2 = 0.0
except ValueError:
    raised2 = 1.0
all_ok &= check("non-overridable key raises", raised2, 1.0, 1e-9)

print("\n" + ("ALL STRAIN CHECKS PASSED" if all_ok else "STRAIN CHECKS FAILED"))
sys.exit(0 if all_ok else 1)
