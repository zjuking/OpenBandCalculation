"""
Layer stack geometry.

A device is a stack of :class:`Layer` objects.  Three kinds are provided:
uniform layers, composition-graded layers (the workhorse of
heterostructure design) and arbitrary trees built from :class:`CompoundLayer`.

Units
-----
Thicknesses are given in **nanometres** and dopant concentrations in
**cm^-3**; both are converted to cm internally.  Doping follows the same
sign convention as the original OBPDS: positive = acceptors (p-type),
negative = donors (n-type).
"""

from __future__ import annotations

import numpy as np

from .materials import OVERRIDABLE, get_alloy

__all__ = ["Layer", "GradedLayer", "CompoundLayer", "Contact",
           "OhmicContact", "SchottkyContact"]


def _check_overrides(overrides):
    """Validate an override dict early, so a typo fails at construction."""
    if not overrides:
        return {}
    unknown = sorted(k for k in overrides if k not in OVERRIDABLE)
    if unknown:
        raise ValueError(
            f"cannot override {', '.join(unknown)}; "
            f"allowed: {', '.join(OVERRIDABLE)}")
    return dict(overrides)


class Layer:
    """A uniform epitaxial layer."""

    def __init__(self, thickness: float, material: str, composition: float | None = None,
                 doping: float | None = None, Na: float | None = None,
                 Nd: float | None = None, mu_n: float | None = None,
                 mu_p: float | None = None, tau_n: float | None = None,
                 tau_p: float | None = None, B_rad: float | None = None,
                 overrides: dict | None = None):
        """
        Parameters
        ----------
        thickness : float
            Layer thickness in nm.
        material : str
            Alloy name, e.g. ``'GaAs'`` or ``'AlGaAs'``.
        composition : float, optional
            Composition ``x`` of the ternary (ignored for binaries).
        doping : float, optional
            Net ionised dopant concentration in cm^-3; positive = p-type.
        Na, Nd : float, optional
            Ionised acceptor / donor concentration in cm^-3, as an
            alternative to ``doping``.
        mu_n, mu_p : float, optional
            Override the material's default low-field mobility (cm^2/Vs).
        tau_n, tau_p : float, optional
            SRH lifetimes (s).  ``None`` falls back to the device default.
        B_rad : float, optional
            Radiative recombination coefficient (cm^3/s).
        overrides : dict, optional
            Calibration of band parameters for this layer, e.g.
            ``{"VBO": 0.35}`` or ``{"b_shear": -1.9}``.  Only the names in
            :data:`obpds.materials.OVERRIDABLE` are accepted: the valence-band
            offset (which shifts every conduction edge with it) and the strain
            parameters.  Anything else feeds quantities that would be left
            stale; edit ``obpds/materials/data.py`` for those.

            The value replaces the **unstrained** tabulated parameter; if the
            layer sits on a substrate, strain is applied to the overridden
            value afterwards, exactly as it would be to the tabulated one.
        """
        self.thickness = float(thickness) * 1e-7  # nm -> cm
        self.material = material
        self.composition = composition
        self.mu_n, self.mu_p = mu_n, mu_p
        self.tau_n, self.tau_p = tau_n, tau_p
        self.B_rad = B_rad
        self.overrides = _check_overrides(overrides)
        self.Na, self.Nd = _resolve_doping(doping, Na, Nd)
        self._graded = False

    def alloy(self, x: float):
        return get_alloy(self.material, self.composition)

    def __repr__(self):
        return f"Layer({self.thickness*1e7:g} nm, {self.material})"


class GradedLayer(Layer):
    """A layer whose ternary composition varies linearly with depth."""

    def __init__(self, thickness: float, material: str, composition,
                 **kwargs):
        """
        Parameters
        ----------
        composition : (float, float)
            Composition at the **top** (x = 0) and **bottom** of the layer.
        """
        super().__init__(thickness, material, None, **kwargs)
        self.composition = tuple(float(c) for c in composition)
        self._graded = True

    def alloy(self, x: float):
        """Alloy at fractional depth ``x`` (0 = top, 1 = bottom)."""
        c0, c1 = self.composition
        return get_alloy(self.material, c0 + (c1 - c0) * x)

    def __repr__(self):
        c0, c1 = self.composition
        return (f"GradedLayer({self.thickness*1e7:g} nm, {self.material}, "
                f"{c0:g}->{c1:g})")


class CompoundLayer(Layer):
    """A stack of layers that behaves as a single layer."""

    def __init__(self, layers):
        super().__init__(0.0, "compound")
        self.layers = list(layers)
        self.thickness = sum(l.thickness for l in self.layers)
        self._graded = any(getattr(l, "_graded", False) for l in self.layers)

    def __iter__(self):
        return iter(self.layers)

    def alloy(self, x: float):
        t = x * self.thickness
        acc = 0.0
        for layer in self.layers:
            if t <= acc + layer.thickness or layer is self.layers[-1]:
                local = (t - acc) / layer.thickness
                return layer.alloy(min(max(local, 0.0), 1.0))
            acc += layer.thickness
        return self.layers[-1].alloy(1.0)


class Contact:
    """Base class for the two terminal contacts."""

    kind = "contact"


class OhmicContact(Contact):
    """Charge-neutral, infinite-recombination-velocity contact."""

    kind = "ohmic"


class SchottkyContact(Contact):
    """
    A Schottky contact pinned at the 'universal pinning level'.

    The surface Fermi level is pinned at ``work_function`` eV below the
    vacuum level; the default of 4.9 eV reproduces the observed mid-gap
    pinning of the common III-V semiconductors.
    """

    kind = "schottky"

    def __init__(self, work_function: float = 4.9):
        self.work_function = float(work_function)


def _resolve_doping(doping, Na, Nd):
    if doping is not None and (Na is not None or Nd is not None):
        raise ValueError("Specify either `doping` or `Na`/`Nd`, not both.")
    if doping is not None:
        return (float(doping), 0.0) if doping >= 0 else (0.0, -float(doping))
    return (float(Na) if Na else 0.0, float(Nd) if Nd else 0.0)


def flatten(layers):
    """Yield (layer, start_fraction, end_fraction) over the whole stack."""
    total = sum(l.thickness for l in layers)
    acc = 0.0
    for layer in layers:
        yield layer, acc / total, (acc + layer.thickness) / total
        acc += layer.thickness
