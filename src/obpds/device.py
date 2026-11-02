"""
User-facing device object and the high level simulation API.
"""

from __future__ import annotations

import math

import numpy as np

from .layer import (CompoundLayer, Contact, Layer, OhmicContact,
                    SchottkyContact)
from .solver import (Model, build_model, charge_neutral_guess, evaluate,
                     solve_poisson, carriers)
from .materials import apply_overrides
from .strain import Substrate, critical_thickness, strain_bands
from .transport import solve_drift_diffusion, iv_curve, TransportSolution

__all__ = ["TwoTerminalDevice"]

BIG = 1.0e4  # eV, stands in for +/- infinity for unpinned quasi-Fermi levels


def _as_substrate(spec):
    """Normalise a substrate specification into a :class:`Substrate`."""
    if spec is None or isinstance(spec, Substrate):
        return spec
    if isinstance(spec, (int, float)):
        return Substrate(a=float(spec))
    if isinstance(spec, str):
        return Substrate(spec)
    if isinstance(spec, (tuple, list)) and len(spec) == 2:
        return Substrate(spec[0], spec[1])
    raise TypeError(f"Cannot use {spec!r} as a substrate.")


class TwoTerminalDevice:
    """
    A stack of epitaxial layers with two terminal contacts.

    Parameters
    ----------
    layers : list of Layer
        The epitaxial stack, ordered from the left/top contact downwards.
    contacts : list of Contact, optional
        Two contacts; defaults to two Ohmic contacts.
    Fp, Fn : str or list of str, optional
        Which contact pins the hole / electron quasi-Fermi level in each
        layer: ``'left'``, ``'right'``, or ``None`` (unpinned).  A scalar
        applies to every layer; a list is per layer.
    substrate : str, float, Substrate or None, optional
        The wafer, for pseudomorphic strain.  ``None`` (the default) leaves
        every layer at its natural lattice constant; a string names an alloy
        (``'GaAs'``, ``('GaInAs', 0.47)``); a number is a lattice constant in
        Angstrom.  Growth is assumed to be along [001].
    """

    def __init__(self, layers, contacts=None, Fp="left", Fn="right",
                 T=300.0, N=1000, approx="kane", tau_n=1e-7, tau_p=1e-7,
                 B_rad=1e-10, substrate=None):
        if not layers:
            raise ValueError("A device needs at least one layer.")
        self.substrate = _as_substrate(substrate)
        self.layers = list(layers)
        self._stack = CompoundLayer(self.layers)
        self.thickness = self._stack.thickness
        if contacts is None:
            contacts = [OhmicContact(), OhmicContact()]
        if len(contacts) != 2 or not all(isinstance(c, Contact) for c in contacts):
            raise ValueError("Exactly two Contact instances are required.")
        self.contacts = list(contacts)
        self.Fp, self.Fn = Fp, Fn
        self.T, self.N, self.approx = float(T), int(N), approx
        self.tau_n, self.tau_p, self.B_rad = tau_n, tau_p, B_rad
        self._models = {}
        self._cache = {}

    # ---------------------------------------------------------------- geometry
    def layer_at(self, fraction: float) -> Layer:
        """The layer containing the point at ``fraction`` of the total depth."""
        t = float(fraction) * self._stack.thickness
        acc = 0.0
        for layer in self.layers:
            if t < acc + layer.thickness or layer is self.layers[-1]:
                return layer
            acc += layer.thickness
        return self.layers[-1]

    def material_at(self, fraction: float):
        """The :class:`Alloy` at ``fraction`` of the total depth."""
        t = float(fraction) * self._stack.thickness
        acc = 0.0
        for layer in self.layers:
            if t < acc + layer.thickness or layer is self.layers[-1]:
                local = (t - acc) / layer.thickness
                return layer.alloy(min(max(local, 0.0), 1.0))
            acc += layer.thickness
        return self.layers[-1].alloy(1.0)

    def model(self, T=None, N=None) -> Model:
        T = self.T if T is None else float(T)
        N = self.N if N is None else int(N)
        key = (T, N)
        m = self._models.get(key)
        if m is None:
            m = build_model(self, T, N)
            self._models[key] = m
        return m

    # ---------------------------------------------------------- quasi-Fermi
    def quasi_fermi(self, spec, V, N):
        """Resolve a quasi-Fermi specification onto the grid."""
        if spec is None or isinstance(spec, str):
            specs = [spec] * len(self.layers)
        else:
            specs = list(spec)
        out = np.empty(N)
        for i in range(N):
            f = (i / (N - 1)) if N > 1 else 0.0
            layer = self.layer_at(min(f, 1.0))
            idx = self.layers.index(layer)
            s = specs[idx] if idx < len(specs) else specs[-1]
            if s == "left":
                out[i] = -V
            elif s == "right":
                out[i] = 0.0
            else:
                out[i] = 0.0
        return out

    def _unpinned(self, spec, N, sign):
        """Array of 'infinite' quasi-Fermi levels where unpinned."""
        if spec is None or isinstance(spec, str):
            specs = [spec] * len(self.layers)
        else:
            specs = list(spec)
        out = np.empty(N)
        for i in range(N):
            f = (i / (N - 1)) if N > 1 else 0.0
            idx = self.layers.index(self.layer_at(min(f, 1.0)))
            s = specs[idx] if idx < len(specs) else specs[-1]
            out[i] = 0.0 if s in ("left", "right") else sign * BIG
        return out

    # -------------------------------------------------------------- solvers
    def get_equilibrium(self, T=None, N=None, approx=None):
        """Self-consistent band profile at zero bias."""
        T = self.T if T is None else T
        N = self.N if N is None else N
        approx = self.approx if approx is None else approx
        key = ("eq", T, N, approx)
        if key in self._cache:
            return self._cache[key]
        m = self.model(T, N)
        Fp = np.zeros(N)
        Fn = np.zeros(N)
        psi0 = charge_neutral_guess(m, Fp, Fn, approx)
        psi, res = solve_poisson(m, psi0, Fp, Fn, 0.0, 0.0, approx)
        sol = evaluate(m, psi, Fp, Fn, 0.0, approx)
        sol.converged = res.converged
        self._cache[key] = sol
        return sol

    def get_quantum_equilibrium(self, T=None, N=None, approx=None,
                                n_states=12):
        """
        Self-consistent Schrodinger-Poisson band profile at zero bias.

        Same as :meth:`get_equilibrium` except the electron gas in the
        conduction band is treated quantum mechanically: the bound states of
        the well are solved for and populated as two-dimensional subbands,
        so the channel sits a few nanometres *behind* the interface rather
        than on it.  The returned solution carries the extra ``subbands``
        attribute (levels, wavefunctions, sheet density).
        """
        from .quantum import solve_quantum_poisson
        T = self.T if T is None else T
        N = self.N if N is None else N
        approx = self.approx if approx is None else approx
        key = ("qeq", T, N, approx, n_states)
        if key in self._cache:
            return self._cache[key]
        m = self.model(T, N)
        psi, info = solve_quantum_poisson(m, approx, n_states=n_states)
        sol = evaluate(m, psi, np.zeros(N), np.zeros(N), 0.0, approx)
        sol.converged = bool(info["converged"])
        sol.n_quantum = info["n_quantum"]
        sol.subbands = {
            "levels_eV": [float(e) for e in info["levels"]],
            "bound": [bool(b) for b in info["bound"]],
            "E_cut_eV": float(info["E_cut"]),
            "sheet_density_cm2": float(info["sheet_density"]),
            "population_cm2": [float(v) for v in info["population"]],
            "psi": [[float(v) for v in row]
                    for row in info["phi"][:, info["bound"]].T],
        }
        self._cache[key] = sol
        return sol

    def get_zero_current(self, V, T=None, N=None, approx=None):
        """
        Band profile at bias ``V`` with the quasi-Fermi levels pinned to the
        contacts (the zero-current approximation of the original OBPDS).

        This is *not* a transport solution: the quasi-Fermi levels are
        prescribed rather than solved for, so no current is obtained.
        """
        T = self.T if T is None else T
        N = self.N if N is None else N
        approx = self.approx if approx is None else approx
        key = ("zc", V, T, N, approx)
        if key in self._cache:
            return self._cache[key]
        m = self.model(T, N)
        Vl, Vr = float(V), 0.0
        Fp = self.quasi_fermi(self.Fp, Vl, N) + \
            np.where(self._unpinned(self.Fp, N, +1.0) != 0.0, BIG, 0.0)
        Fn = self.quasi_fermi(self.Fn, Vl, N) + \
            np.where(self._unpinned(self.Fn, N, -1.0) != 0.0, -BIG, 0.0)
        psi0 = charge_neutral_guess(m, Fp, Fn, approx)
        psi, res = solve_poisson(m, psi0, Fp, Fn, Vl, Vr, approx)
        sol = evaluate(m, psi, Fp, Fn, V, approx)
        sol.converged = res.converged
        self._cache[key] = sol
        return sol

    def get_drift_diffusion(self, V, T=None, N=None, approx=None,
                            max_gummel=200):
        """
        Full drift-diffusion solution at bias ``V``.

        Solves the electron and hole continuity equations together with
        Poisson, giving the terminal current density.
        """
        T = self.T if T is None else T
        N = self.N if N is None else N
        approx = self.approx if approx is None else approx
        key = ("dd", round(float(V), 9), T, N, approx)
        if key in self._cache:
            return self._cache[key]
        m = self.model(T, N)
        sol = solve_drift_diffusion(m, float(V), approx, max_gummel)
        self._cache[key] = sol
        return sol

    def get_iv(self, Vstart, Vstop, Vnum=61, T=None, N=None, approx=None):
        """Sweep the bias and return ``(V, J)`` with J in A/cm^2."""
        T = self.T if T is None else T
        N = self.N if N is None else N
        approx = self.approx if approx is None else approx
        return iv_curve(self, Vstart, Vstop, Vnum, T, N, approx)

    def get_capacitance(self, V, dV=1e-3, T=None, N=None, approx=None):
        """Small-signal capacitance per unit area (F/cm^2) at bias ``V``."""
        from .solver import Q
        T = self.T if T is None else T
        N = self.N if N is None else N
        approx = self.approx if approx is None else approx
        m = self.model(T, N)
        s1 = self.get_zero_current(V, T, N, approx)
        s2 = self.get_zero_current(V + dV, T, N, approx)
        drho = ((s2.p - s2.n) - (s1.p - s1.n)) * Q
        dx = m.x[1] - m.x[0]
        pos = np.where(drho > 0, drho, 0.0)
        neg = np.where(drho < 0, drho, 0.0)
        dQ = max(np.trapezoid(pos, dx=dx), -np.trapezoid(neg, dx=dx))
        return float(dQ / dV)

    def get_cv(self, Vstart, Vstop, Vnum=61, dV=1e-3, T=None, N=None,
               approx=None):
        V = np.linspace(Vstart, Vstop, Vnum)
        C = np.array([self.get_capacitance(v, dV, T, N, approx) for v in V])
        return V, C

    # ------------------------------------------------------------------ info
    def describe(self, T=None):
        """
        Per-layer summary.

        ``mismatch`` is against the first layer, which is what a band diagram
        is usually read against; ``mismatch_sub``, ``eps_in``, ``eps_out``,
        ``h_c_nm`` and ``pseudomorphic`` are against the substrate and are
        only meaningful when one is set.
        """
        T = self.T if T is None else T
        ref = self.material_at(0.0).props(T)
        a_sub = (None if self.substrate is None
                 else self.substrate.lattice_constant(T))
        rows = []
        for layer in self.layers:
            p, p1 = layer.alloy(0.0).props(T), layer.alloy(1.0).props(T)
            # report what the solver actually uses, overrides included
            p = apply_overrides(p, layer.overrides)
            p1 = apply_overrides(p1, layer.overrides)
            if a_sub is not None:
                # Report what the solver actually uses: a table of unstrained
                # gaps next to a strained band diagram is unreadable.
                p, p1 = strain_bands([p, p1], a_sub, T)[0]
            a = p["lattice_constant"]
            row = dict(
                material=str(layer), thickness_nm=layer.thickness * 1e7,
                Eg=p["Eg"], Eg_end=p1["Eg"],
                doping_cm3=layer.Nd - layer.Na,
                lattice_constant=a,
                mismatch=(a - ref["lattice_constant"]) / ref["lattice_constant"],
                dielectric=p["dielectric"],
            )
            if a_sub is not None:
                f = (a - a_sub) / a_sub
                # The critical thickness is a property of the *layer*, so it
                # is evaluated on the layer's own elastic constants even
                # though the mismatch is measured against the wafer.
                hc = critical_thickness(f, a_sub, p.get("C11"), p.get("C12"))
                eps_in, eps_out = _in_plane(f, p.get("C11"), p.get("C12"))
                row.update(
                    substrate=repr(self.substrate),
                    substrate_a=a_sub,
                    mismatch_sub=f,
                    eps_in=eps_in,
                    eps_out=eps_out,
                    h_c_nm=(math.inf if math.isinf(hc) else hc / 10.0),
                    pseudomorphic=bool(layer.thickness * 1e7 <= hc / 10.0),
                    # False for a material with no deformation-potential
                    # data: the strain is still reported, but no band edge
                    # moves, and saying so beats a silent no-op.
                    strain_applied=p.get("a_c") is not None,
                )
            rows.append(row)
        return rows


def _in_plane(mismatch, C11, C12):
    """In-plane strain implied by a mismatch ``(a_layer - a_sub)/a_sub``."""
    from .strain import biaxial_strain
    # biaxial_strain wants the substrate and layer constants; only the ratio
    # matters, so use a_sub = 1 and a_layer = 1 + f.
    eps_in, eps_out = biaxial_strain(1.0 + mismatch, 1.0, C11, C12)
    return float(eps_in), float(eps_out)
