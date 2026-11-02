"""
Alloy objects: interpolation of III-V parameters and derived quantities.

The interpolation order matters and follows the recommendation in
Vurgaftman *et al.*: interpolate the *raw* tabulated parameters (band gaps
at 0 K, Varshni coefficients, Luttinger parameters, ``Ep``, ``Delta_SO``,
the Kane ``F`` parameter) and only then evaluate the derived quantities
(band gaps at temperature ``T``, effective masses, non-parabolicity).
Interpolating derived quantities instead would double-count the
composition dependence.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.constants import h, k as k_SI, m_e, pi

from .data import BINARIES, TERNARIES, VACUUM_LEVEL
from ..fermi import KB_EV

__all__ = ["Alloy", "get_alloy", "N_PREF", "apply_overrides", "OVERRIDABLE"]

#: Parameters a layer may override for calibration against experiment.
#:
#: Deliberately narrow.  ``VBO`` is allowed because it cascades in a way that
#: can be carried exactly (every conduction-band edge is ``VBO + Eg``, so a
#: shift of the valence edge shifts all of them by the same amount).  The
#: strain parameters are allowed because nothing derived in :meth:`Alloy.props`
#: depends on them -- they are read only by :mod:`obpds.strain`.
#:
#: Everything else is excluded on purpose.  Overriding ``Eg`` or a band mass
#: would leave ``ni``, ``Nc``, ``Nv`` and the non-parabolicity stale, and a
#: half-updated parameter set is worse than none.  For those, edit
#: :mod:`obpds.materials.data`.
OVERRIDABLE = ("VBO", "a_c", "a_v", "b_shear", "C11", "C12")

#: The conduction-band edges that ride on ``VBO``.
_CBO_KEYS = ("CBO", "CBO_Gamma", "CBO_L", "CBO_X")

# 2 * (2*pi*m0*k/h^2)^(3/2), converted from m^-3 K^-3/2 to cm^-3 K^-3/2
N_PREF = 2.0 * (2.0 * pi * m_e * k_SI / h ** 2) ** 1.5 / 1.0e6


def _varshni(eg0, alpha, beta, T):
    """Varshni's Eg(T) = Eg(0) - alpha*T^2/(T+beta)."""
    if alpha is None or beta is None:
        return eg0  # no temperature dependence tabulated
    return eg0 - alpha * T ** 2 / (T + beta)


def _interp(va, vb, x, bow=0.0):
    """Ternary interpolation ``x*va + (1-x)*vb - bow*x*(1-x)``."""
    if va is None or vb is None:
        return None
    return x * va + (1.0 - x) * vb - bow * x * (1.0 - x)


def _binary_gap(name, prefix, T):
    """Band gap ``Eg_{prefix}(T)`` of a binary, including GaP's special form."""
    b = BINARIES[name]
    eg0 = b[f"Eg_{prefix}_0"]
    if name == "GaP" and prefix == "Gamma":
        # Vurgaftman's dedicated Gamma-gap temperature dependence for GaP
        return eg0 + 0.1081 * (1.0 - 1.0 / math.tanh(164.0 / T))
    return _varshni(eg0, b[f"alpha_{prefix}"], b[f"beta_{prefix}"], T)


def _f_kane(eg0, delta_so, ep, meff0):
    """
    Kane remote-band parameter ``F``, back-solved from the tabulated
    0 K Gamma effective mass (Vurgaftman *et al.*, Eq. for ``meff_e_Gamma``).
    """
    if None in (eg0, delta_so, ep, meff0):
        return None
    return (1.0 / meff0 - 1.0
            - (ep * (eg0 + 2.0 * delta_so / 3.0)) / (eg0 * (eg0 + delta_so))) / 2.0


class Alloy:
    """
    A III-V zinc-blende binary or ternary alloy.

    ``props(T)`` returns every parameter the device solver needs, evaluated
    at temperature ``T``.
    """

    def __init__(self, name: str, composition: float | None = None):
        self.name = name
        if name in BINARIES:
            self.kind = "binary"
            self._b = BINARIES[name]
            self.binaries = (name,)
            self.x = None
            self.x_label = None
        elif name in TERNARIES:
            self.kind = "ternary"
            t = TERNARIES[name]
            self._t = t
            self.binaries = t["binaries"]
            self.x_label = t["x_label"]
            x = 0.5 if composition is None else float(composition)
            self.x = min(max(x, 0.0), 1.0)
        else:
            raise KeyError(
                f"Unknown alloy {name!r}. Available: "
                f"{', '.join(sorted(list(BINARIES) + list(TERNARIES)))}"
            )

    def __repr__(self):
        if self.kind == "binary":
            return self.name
        return f"{self.name}(x={self.x:.3f})"

    # ------------------------------------------------------------------ raw
    def _raw(self, key):
        """Interpolated value of a raw tabulated parameter."""
        if self.kind == "binary":
            return self._b.get(key)
        t = self._t
        va = BINARIES[self.binaries[0]].get(key)
        vb = BINARIES[self.binaries[1]].get(key)
        bow = t["bowing"].get(key, 0.0)
        if key in t.get("bowing_fn", {}):
            bow = t["bowing_fn"][key](self.x)
        return _interp(va, vb, self.x, bow)

    def _bowing(self, key):
        """Alloy bowing parameter for ``key`` (0 for binaries / no bowing)."""
        if self.kind != "ternary":
            return 0.0
        t = self._t
        if key in t.get("bowing_fn", {}):
            return t["bowing_fn"][key](self.x)
        return t["bowing"].get(key, 0.0)

    # -------------------------------------------------------------- derived
    def props(self, T: float = 300.0) -> dict:
        T = float(T)
        kT = KB_EV * T
        g = self._raw

        delta_so = g("Delta_SO")
        ep = g("Ep")
        vbo = g("VBO")

        # --- band gaps -----------------------------------------------------
        # Bowing parameters tabulated by Vurgaftman act on the band gap
        # itself, so they are applied to the *temperature-corrected* gaps of
        # the two binaries.  Interpolating Eg(T) (rather than Eg(0) plus
        # interpolated Varshni coefficients) also handles GaP, whose Gamma
        # gap has a dedicated non-Varshni temperature dependence.
        def gap(prefix):
            if self.kind == "binary":
                return _binary_gap(self.name, prefix, T)
            va = _binary_gap(self.binaries[0], prefix, T)
            vb = _binary_gap(self.binaries[1], prefix, T)
            return _interp(va, vb, self.x, self._bowing(f"Eg_{prefix}"))

        eg_g, eg_l, eg_x = gap("Gamma"), gap("L"), gap("X")
        eg = min(eg_g, eg_l, eg_x)

        # --- Kane F and Gamma effective mass -------------------------------
        if self.kind == "binary":
            f_kane = _f_kane(g("Eg_Gamma_0"), delta_so, ep, g("meff_e_Gamma_0"))
        else:
            # Vurgaftman recommends interpolating F itself (it is a raw
            # band-structure parameter), then deriving the effective mass.
            fa = _f_kane(BINARIES[self.binaries[0]]["Eg_Gamma_0"],
                         BINARIES[self.binaries[0]]["Delta_SO"],
                         BINARIES[self.binaries[0]]["Ep"],
                         BINARIES[self.binaries[0]]["meff_e_Gamma_0"])
            fb = _f_kane(BINARIES[self.binaries[1]]["Eg_Gamma_0"],
                         BINARIES[self.binaries[1]]["Delta_SO"],
                         BINARIES[self.binaries[1]]["Ep"],
                         BINARIES[self.binaries[1]]["meff_e_Gamma_0"])
            f_kane = _interp(fa, fb, self.x, self._bowing("F"))

        if f_kane is not None and ep is not None and delta_so is not None:
            meff_g = 1.0 / ((1.0 + 2.0 * f_kane)
                            + (ep * (eg_g + 2.0 * delta_so / 3.0))
                            / (eg_g * (eg_g + delta_so)))
        else:
            meff_g = g("meff_e_Gamma_0")

        # --- satellite valley DOS masses -----------------------------------
        def dos_mass(prefix):
            dos = g(f"meff_e_{prefix}_DOS")
            if dos is not None:
                return dos
            lng = g(f"meff_e_{prefix}_long")
            trn = g(f"meff_e_{prefix}_trans")
            if lng is not None and trn is not None:
                return (trn * trn * lng) ** (1.0 / 3.0)
            return None

        meff_l, meff_x = dos_mass("L"), dos_mass("X")

        # --- valence band masses -------------------------------------------
        l1, l2, l3 = g("luttinger1"), g("luttinger2"), g("luttinger3")
        if None in (l1, l2):
            m_hh = m_lh = 0.5
        else:
            m_hh = 1.0 / (l1 - 2.0 * l2)
            m_lh = 1.0 / (l1 + 2.0 * l2)
        # Density-of-states mass combining heavy- and light-hole bands.
        # (The original OBPDS used the heavy-hole mass alone, which
        # underestimates Nv by roughly 25 %.)
        m_h_dos = (m_hh ** 1.5 + m_lh ** 1.5) ** (2.0 / 3.0)

        # --- effective densities of states ---------------------------------
        def n_eff(m):
            return N_PREF * (m * T) ** 1.5 if m is not None else 0.0

        nc_g, nc_l, nc_x = n_eff(meff_g), n_eff(meff_l), n_eff(meff_x)
        nv = n_eff(m_h_dos)
        # Nc used for scalar (single-valley) quantities such as Ei: the
        # valley with the lowest band edge, exactly as in the original code.
        if eg_g < eg_x:
            nc = nc_l if eg_g >= eg_l else nc_g
        else:
            nc = nc_l if eg_x >= eg_l else nc_x
        nc = max(nc, 1.0)

        # --- misc ----------------------------------------------------------
        cbo_g, cbo_l, cbo_x = vbo + eg_g, vbo + eg_l, vbo + eg_x
        cbo = vbo + eg
        alpha_np = kT / eg_g * (1.0 - meff_g) ** 2 if meff_g is not None else 0.0
        a_lat = (g("a_300K")
                 + (g("thermal_expansion") or 0.0) * (T - 300.0))
        ni = math.sqrt(nv * nc) * math.exp(-eg / (2.0 * kT))

        return dict(
            name=self.name,
            composition=self.x,
            T=T,
            # band edges on the Vurgaftman / openbandparams scale (InSb VBM = 0)
            VBO=vbo, CBO=cbo, CBO_Gamma=cbo_g, CBO_L=cbo_l, CBO_X=cbo_x,
            Eg=eg, Eg_Gamma=eg_g, Eg_L=eg_l, Eg_X=eg_x,
            # densities of states (cm^-3)
            Nc=nc, Nc_Gamma=nc_g, Nc_L=nc_l, Nc_X=nc_x, Nv=nv, ni=ni,
            # effective masses (m0)
            meff_e_Gamma=meff_g, meff_e_L_DOS=meff_l, meff_e_X_DOS=meff_x,
            meff_hh_100=m_hh, meff_lh_100=m_lh, meff_h_DOS=m_h_dos,
            Delta_SO=delta_so, Ep=ep, F=f_kane,
            nonparabolicity=alpha_np,
            dielectric=g("dielectric"),
            electron_affinity=VACUUM_LEVEL - cbo,
            lattice_constant=a_lat,
            # strain parameters (None for materials with no deformation
            # potential data -- strain is then skipped for that node)
            C11=g("C11"), C12=g("C12"), a_c=g("a_c"), a_v=g("a_v"),
            b_shear=g("b_shear"),
            mu_n=g("mu_n"), mu_p=g("mu_p"),
            direct=bool(eg_g <= min(eg_l, eg_x)),
        )


_CACHE: dict[tuple, Alloy] = {}


def get_alloy(name: str, composition: float | None = None) -> Alloy:
    """Look up (and memoise) an :class:`Alloy` by name."""
    key = (name, None if composition is None else round(float(composition), 6))
    a = _CACHE.get(key)
    if a is None:
        a = Alloy(name, composition)
        _CACHE[key] = a
    return a


def lattice_mismatch(a_layer, a_substrate):
    """Fractional lattice mismatch ``(a_layer - a_sub) / a_sub``."""
    return (a_layer - a_substrate) / a_substrate


def apply_overrides(props: dict, overrides: dict | None) -> dict:
    """
    Apply a layer's calibration overrides to its parameter dict.

    Returns a new dict; ``props`` is left untouched.  Only the keys listed in
    :data:`OVERRIDABLE` are accepted, and the dependent quantities are carried
    along: moving ``VBO`` moves every conduction-band edge with it, because
    each one is stored as ``VBO + Eg`` and a user who shifts the valence edge
    without shifting the conduction edges would silently change the band gap.

    Parameters
    ----------
    props : dict
        As returned by :meth:`Alloy.props`.
    overrides : dict or None
        ``{name: value}``.  ``None`` or ``{}`` returns ``props`` unchanged.
        The values replace the **unstrained** parameters; this function runs
        before :func:`obpds.strain.strain_bands`, so a strained layer gets its
        hydrostatic and shear shifts on top of the overridden value.

    Raises
    ------
    ValueError
        If a key is not overridable.  Silently ignoring a misspelt key would
        leave the user believing a calibration had been applied.
    """
    if not overrides:
        return props
    unknown = sorted(k for k in overrides if k not in OVERRIDABLE)
    if unknown:
        raise ValueError(
            f"cannot override {', '.join(unknown)}; "
            f"allowed: {', '.join(OVERRIDABLE)}. "
            "Other band parameters feed derived quantities (ni, Nc, Nv, the "
            "non-parabolicity) that would be left stale -- edit "
            "obpds/materials/data.py instead.")

    out = dict(props)
    if "VBO" in overrides:
        vbo = float(overrides["VBO"])
        shift = vbo - float(props["VBO"])
        out["VBO"] = vbo
        for k in _CBO_KEYS:
            out[k] = float(props[k]) + shift
        out["electron_affinity"] = VACUUM_LEVEL - out["CBO"]
    for k in ("a_c", "a_v", "b_shear", "C11", "C12"):
        if k in overrides:
            v = overrides[k]
            out[k] = None if v is None else float(v)
    return out
