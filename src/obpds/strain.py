"""
Pseudomorphic strain on a (001) zinc-blende substrate.

An epitaxial layer thinner than its critical thickness cannot relax: it is
stretched or squeezed *in the plane* to the substrate's lattice constant and
pays for it by straining out of plane.  That does two things to the band
structure, and they pull in opposite directions:

* the **hydrostatic** part (a change in volume) shifts the whole band
  structure -- the conduction band up, the valence band down, so the gap
  widens under compression;
* the **shear** part (a change in shape at constant volume) splits the
  fourfold degenerate valence band into a heavy-hole and a light-hole
  doublet, by tens of meV at 1 % mismatch.

Both matter for device design: strain is how the band gap and the valence
band of a quantum well are engineered, and it is why a compressively
strained InGaAs channel is a better laser than a lattice-matched one.

Model
-----
Growth is taken to be along [001], the layer is commensurate with the
substrate (``eps_xx = eps_yy = eps_in`` locked to the substrate) and the free
surface carries no stress (``sigma_zz = 0``), which fixes

    eps_in  = (a_sub - a_layer) / a_layer
    eps_out = -2 (C12 / C11) eps_in.

Conduction band: only the hydrostatic term acts on the Gamma, L and X edges,

    dE_c = a_c * Tr(eps),      Tr(eps) = 2 eps_in + eps_out.

Valence band: the Bir-Pikus strain Hamiltonian plus spin-orbit coupling, in
the |3/2,+-3/2>, |3/2,+-1/2>, |1/2,+-1/2> basis.  For [001] biaxial strain
the +-3/2 block is already diagonal, and the +-1/2 blocks are a 2x2 that mixes
light hole with split-off.  With

    h  = a_v * Tr(eps)                 hydrostatic shift of the average
    d  = -2 b * (eps_out - eps_in)     shear splitting parameter
    s  = sqrt(D^2 - D*d + 9 d^2/4),    D = Delta_SO

the three edges are

    E_hh = E_v + h + d/2
    E_lh = E_v + h - D/2 - d/4 + s/2
    E_so = E_v + h - D/2 - d/4 - s/2

referenced to the *unstrained* valence band maximum ``E_v``.  At zero strain
this gives E_hh = E_lh = E_v and E_so = E_v - D, and the shear contribution
is traceless: E_hh + E_lh + E_so = 3(E_v + h) - D for any ``d``.

Deliberately not modelled
-------------------------
* Valley-specific hydrostatic potentials: L and X are shifted by the same
  ``a_c`` as Gamma, and the shear splitting of the four L and three X valleys
  (which needs the ``Xi_u`` deformation potentials) is ignored.  This is a
  good approximation for the *direct-gap* arsenides/antimonides (GaAs, InAs,
  InGaAs, ...), where Gamma is the conduction minimum and L/X sit well above
  it, so a shared ``a_c`` only shifts satellites that do not matter for the
  electrostatics.  It is *not* adequate for the indirect phosphides (GaP,
  AlP, AlAs) or for any structure whose conduction minimum is at X or L, where
  the valley-specific ``a_c`` and ``Xi_u`` decide the band alignment.
* Strain dependence of the effective masses.  The heavy-hole mass really is
  anisotropic under [001] strain; the tabulated [100] masses are used.
  What *is* captured is that a split-off band stops contributing to the
  density of states -- :func:`strained_nv` weights each band by how far below
  the top it sits, which is the effect that matters for the electrostatics.
* Relaxation.  :func:`critical_thickness` reports the Matthews-Blakeslee
  thickness but nothing relaxes automatically: past the critical thickness
  the layer is flagged, not quietly assumed to be partially relaxed.
* Piezo-electric charge, which only appears for growth on [111].
"""

from __future__ import annotations

import math

import numpy as np

from .fermi import KB_EV
from .materials.alloy import N_PREF, get_alloy

__all__ = ["Substrate", "biaxial_strain", "valence_edges", "strained_nv",
           "critical_thickness", "strain_bands"]

# Burgers vector magnitude of a 60-degree dislocation in zinc blende, in units
# of the lattice constant: |(a/2)[101]| = a/sqrt(2).
_BURGERS = 1.0 / math.sqrt(2.0)


class Substrate:
    """
    The wafer: an unstrained reference lattice constant, in Angstrom.

    Either name an alloy (``Substrate("GaAs")``, ``Substrate("GaInAs", 0.47)``)
    and its lattice constant is evaluated at the device temperature, or give
    ``a`` directly to pin a custom value.
    """

    __slots__ = ("material", "composition", "a_explicit", "_alloy")

    def __init__(self, material=None, composition=None, a=None):
        if material is None and a is None:
            raise ValueError("Give either a substrate material or a lattice "
                             "constant.")
        self.material = material
        self.composition = composition
        self.a_explicit = None if a is None else float(a)
        if self.a_explicit is not None or material is None:
            self._alloy = None
        else:
            self._alloy = get_alloy(material, composition)

    def __repr__(self):
        if self.material is None:
            return f"Substrate(a={self.a_explicit:g} A)"
        if self.composition is None:
            return f"Substrate({self.material})"
        return f"Substrate({self.material}, x={self.composition:g})"

    def lattice_constant(self, T=300.0) -> float:
        """Substrate lattice constant (Angstrom) at temperature ``T``."""
        if self.a_explicit is not None:
            return self.a_explicit
        return float(self._alloy.props(T)["lattice_constant"])


def biaxial_strain(a_layer, a_sub, C11=None, C12=None):
    """
    Strain of a (001) layer locked to a substrate of constant ``a_sub``.

    Returns ``(eps_in, eps_out)``.  Both are dimensionless and may be arrays;
    a positive ``eps_in`` is tensile (the substrate is larger than the layer).
    """
    a_layer = np.asarray(a_layer, dtype=float)
    eps_in = (a_sub - a_layer) / a_layer
    if C11 is None or C12 is None:
        ratio = 0.5
    else:
        c11 = np.asarray(C11, dtype=float)
        c12 = np.asarray(C12, dtype=float)
        ratio = np.where(c11 > 0.0, c12 / np.maximum(c11, 1e-30), 0.5)
    return eps_in, -2.0 * ratio * eps_in


def valence_edges(E_v, Delta_SO, a_v, b_shear, eps_in, eps_out):
    """
    Strained valence band edges, referenced to the unstrained VBM ``E_v``.

    Returns ``(E_hh, E_lh, E_so)`` in the same units as ``E_v`` (eV).  See the
    module docstring for the derivation; the sum rule
    ``E_hh + E_lh + E_so == 3 (E_v + a_v Tr(eps)) - Delta_SO`` holds exactly
    because the shear part of the Bir-Pikus Hamiltonian is traceless.
    """
    tr = 2.0 * eps_in + eps_out
    h = a_v * tr
    d = -2.0 * b_shear * (eps_out - eps_in)
    D = Delta_SO
    s = np.sqrt(np.maximum(D * D - D * d + 2.25 * d * d, 0.0))
    half = 0.5 * s
    e_hh = E_v + h + 0.5 * d
    e_lh = E_v + h - 0.5 * D - 0.25 * d + half
    e_so = E_v + h - 0.5 * D - 0.25 * d - half
    return e_hh, e_lh, e_so


def strained_nv(T, m_hh, m_lh, E_hh, E_lh):
    """
    Valence density of states of the *split* band structure.

    Each band contributes ``N_PREF (m T)^{3/2}`` weighted by how far below the
    valence band maximum its edge sits.  With no splitting both weights are 1
    and this reduces to the usual ``(m_hh^{3/2} + m_lh^{3/2})`` -- so a
    strained layer loses the light holes smoothly as the splitting grows past
    ``kT`` rather than discontinuously.
    """
    kT = KB_EV * T
    top = np.maximum(E_hh, E_lh)
    w_hh = np.exp(-np.maximum(top - E_hh, 0.0) / kT)
    w_lh = np.exp(-np.maximum(top - E_lh, 0.0) / kT)
    dos = (m_hh ** 1.5 * w_hh + m_lh ** 1.5 * w_lh)
    return N_PREF * T ** 1.5 * dos


def critical_thickness(mismatch, a_sub, C11=None, C12=None, theta_deg=60.0):
    """
    Matthews-Blakeslee critical thickness, in the same length unit as ``a_sub``.

    A layer thicker than this relaxes by misfit dislocations rather than
    staying commensurate.  The implicit equation

        h = A ln(alpha h / b + 1),
        A = b (1 - nu cos^2 theta) / (2 pi |f| (1 + nu))

    with ``b = a/sqrt(2)`` the Burgers magnitude of a 60-degree dislocation and
    ``nu = C12/(C11 + C12)``, is solved by fixed-point iteration.  It returns
    ``inf`` for a lattice-matched layer, and is an *estimate*: measured
    critical thicknesses are routinely a few times larger than this, because
    the equilibrium criterion says nothing about how fast dislocations
    actually nucleate.
    """
    f = abs(float(mismatch))
    if f < 1e-6:
        return math.inf
    if C11 is None or C12 is None or C11 <= 0.0:
        nu = 1.0 / 3.0
    else:
        nu = float(C12) / (float(C11) + float(C12))
    b = float(a_sub) * _BURGERS
    theta = math.radians(theta_deg)
    A = b * (1.0 - nu * math.cos(theta) ** 2) / (2.0 * math.pi * f * (1.0 + nu))
    h = 10.0 * b
    for _ in range(200):
        h_new = A * math.log(h / b + 1.0)
        if abs(h_new - h) <= 1e-12 * max(h_new, 1.0):
            h = h_new
            break
        h = h_new
    return h


def strain_bands(props_list, a_sub, T=300.0):
    """
    Apply pseudomorphic strain to a sequence of :meth:`Alloy.props` dicts.

    Returns ``(patched, info)``.  ``patched`` is a list of shallow copies with
    the band edges, gap, valence density of states and intrinsic density
    recomputed; ``info`` carries the strain state (in-plane and out-of-plane
    strain, the three valence edges, and a per-node flag for nodes whose
    material has no deformation potential data and were therefore left alone).

    ``a_sub`` is the substrate lattice constant in Angstrom at temperature
    ``T``, or ``None`` for no strain at all.
    """
    patched = [dict(p) for p in props_list]
    n = len(patched)
    info = {
        "eps_in": np.zeros(n), "eps_out": np.zeros(n),
        "E_hh": np.zeros(n), "E_lh": np.zeros(n), "E_so": np.zeros(n),
        "applied": np.zeros(n, dtype=bool),
    }
    if a_sub is None:
        info["E_hh"] = np.array([p["VBO"] for p in patched], dtype=float)
        info["E_lh"] = info["E_hh"].copy()
        info["E_so"] = info["E_hh"] - np.array(
            [p["Delta_SO"] or 0.0 for p in patched], dtype=float)
        return patched, info

    for i, p in enumerate(patched):
        if None in (p.get("a_c"), p.get("a_v"), p.get("b_shear")):
            continue                     # no deformation potential: skip
        eps_in, eps_out = biaxial_strain(p["lattice_constant"], a_sub,
                                         p.get("C11"), p.get("C12"))
        e_hh, e_lh, e_so = valence_edges(p["VBO"], p["Delta_SO"], p["a_v"],
                                         p["b_shear"], eps_in, eps_out)
        tr = 2.0 * eps_in + eps_out
        c_g = p["CBO_Gamma"] + p["a_c"] * tr
        c_l = p["CBO_L"] + p["a_c"] * tr
        c_x = p["CBO_X"] + p["a_c"] * tr
        # The valence band maximum is whichever of the two upper edges sits
        # higher; under compression that is the heavy hole, under tension the
        # light hole.  The split-off band can never be the top one: it is
        # always at least Delta_SO below the average.
        vbo = max(float(e_hh), float(e_lh))
        nv = strained_nv(T, p["meff_hh_100"], p["meff_lh_100"],
                         float(e_hh), float(e_lh))
        eg_g, eg_l, eg_x = c_g - vbo, c_l - vbo, c_x - vbo
        eg = min(eg_g, eg_l, eg_x)
        nc = p["Nc"]
        p.update(VBO=vbo, CBO_Gamma=c_g, CBO_L=c_l, CBO_X=c_x,
                 CBO=vbo + eg, Eg=eg, Eg_Gamma=eg_g, Eg_L=eg_l, Eg_X=eg_x,
                 Nv=nv,
                 ni=math.sqrt(max(nv, 1.0) * nc) * math.exp(-eg / (2 * KB_EV * T)),
                 nonparabolicity=(KB_EV * T / eg_g * (1.0 - p["meff_e_Gamma"]) ** 2
                                  if eg_g > 1e-6 and p["meff_e_Gamma"] else 0.0),
                 direct=bool(eg_g <= min(eg_l, eg_x)),
                 )
        info["eps_in"][i] = eps_in
        info["eps_out"][i] = eps_out
        info["E_hh"][i] = e_hh
        info["E_lh"][i] = e_lh
        info["E_so"][i] = e_so
        info["applied"][i] = True
    return patched, info
