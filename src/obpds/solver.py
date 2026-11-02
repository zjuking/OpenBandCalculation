"""
One-dimensional electrostatics of a layered III-V heterostructure.

Governing equation
------------------
With the electrostatic potential ``psi`` expressed in eV (numerically equal
to the potential in volts for a single electron charge) and band edges
bending as ``Ec(x) = CBO(x) - psi(x)``, the Poisson equation is

    d/dx ( eps(x) d psi / dx ) = -q * (p - n + Nd - Na)

Discretisation
--------------
A node-centred finite-volume scheme on a uniform grid.  The flux form
``div(eps grad psi)`` is discretised directly, with the face permittivity
taken as the harmonic mean of the two adjacent nodes:

    eps_{i+1/2} = 2 eps_i eps_{i+1} / (eps_i + eps_{i+1})

The harmonic mean is the exact face permeability for a piecewise-constant
eps whose discontinuity lies between the two nodes.

.. note::
   The original OBPDS discretised ``eps * psi'' = -rho``, i.e. it treated
   eps as locally constant and silently dropped the ``d(eps)/dx * dpsi/dx``
   term.  Across an abrupt heterojunction with a large permittivity
   contrast (InAs/GaSb, GaAs/AlAs, ...) that term is not negligible, so it
   is retained here.

Carrier statistics
------------------
Three approximations are available (``approx``):

``'boltzmann'``
    non-degenerate, parabolic -- fastest
``'parabolic'``
    Fermi-Dirac integral ``F_{1/2}``, parabolic bands
``'kane'``
    Fermi-Dirac with the Kane k.p non-parabolic Gamma valley and parabolic
    L and X satellite valleys (the default)

Bias convention
---------------
``V`` is the potential applied to the **left/top** contact, the right/bottom
contact being grounded.  The electrostatic potential at the contacts is
therefore ``V`` and ``0``, and the carrier Fermi levels are ``-V`` and
``0`` (an electron gains ``-qV`` of energy when the potential rises by
``V``).  With this convention a pn diode with p on the left has
``Fn - Fp = V``, so ``V > 0`` is forward bias.

.. note::
   The original OBPDS assigned ``+V`` to the quasi-Fermi level of the left
   contact, which reverses the sign of the applied bias: its
   ``show_zero_current(V=0.5)`` actually plots a reverse-biased diode.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import diags

from .fermi import KB_EV, generalized_fermi
from .materials import apply_overrides
from .newton import newton, scalar_newton, row_scale
from .strain import strain_bands

__all__ = ["Model", "build_model", "solve_poisson", "charge_neutral_guess",
           "carriers", "Solution"]

# electron charge (C) and vacuum permittivity (C V^-1 cm^-1)
Q = 1.602176634e-19
EPS0 = 8.8541878128e-14

C_F = 2.0 / np.sqrt(np.pi)  # 2/sqrt(pi) prefactor of the carrier integrals


class Model:
    """Flattened, grid-resolved description of a device at one temperature."""

    __slots__ = ("x", "dx", "N", "T", "VBO", "CBO", "CBO_Gamma", "CBO_L", "CBO_X",
                 "Eg", "Nc", "Nc_Gamma", "Nc_L", "Nc_X", "Nv", "ni", "alpha",
                 "eps", "Na", "Nd", "Nnet", "Ei", "mu_n", "mu_p", "tau_n",
                 "tau_p", "B_rad", "dielectric", "lattice_constant", "materials",
                 "meff_e_Gamma", "meff_h_DOS",
                 "eps_in", "eps_out", "E_hh", "E_lh", "E_so", "strain_applied",
                 "contacts", "approx", "n0", "p0")


def build_model(device, T=300.0, N=1000):
    """
    Sample a :class:`~obpds.device.TwoTerminalDevice` onto ``N`` grid nodes.
    """
    m = Model()
    m.N = int(N)
    m.T = float(T)
    total = device.thickness
    m.x = np.linspace(0.0, total, m.N)
    m.dx = m.x[1] - m.x[0]
    m.contacts = device.contacts
    m.approx = device.approx

    props = [device.material_at(x / total) for x in m.x]
    p0 = [p.props(T) for p in props]
    m.materials = props

    # A layer may carry calibration overrides for its band parameters.  They
    # belong here: after the alloy parameters have been evaluated and before
    # strain, which reads them.
    layers = [device.layer_at(x / total) for x in m.x]
    if any(l.overrides for l in layers):
        p0 = [apply_overrides(p, l.overrides) for p, l in zip(p0, layers)]

    # Pseudomorphic strain is a property of the layer *and* the wafer, so it
    # has to be applied after the alloy parameters are known and before the
    # band edges are sampled onto the grid.  Without a substrate the layers
    # are left at their natural lattice constants.
    a_sub = None
    if getattr(device, "substrate", None) is not None:
        a_sub = device.substrate.lattice_constant(T)
    if a_sub is not None:
        p0, sinfo = strain_bands(p0, a_sub, T)
    else:
        sinfo = None

    def arr(key):
        return np.array([p[key] for p in p0], dtype=float)

    m.VBO = arr("VBO")
    m.CBO = arr("CBO")
    m.CBO_Gamma = arr("CBO_Gamma")
    m.CBO_L = arr("CBO_L")
    m.CBO_X = arr("CBO_X")
    m.Eg = arr("Eg")
    m.Nc = arr("Nc")
    m.Nc_Gamma = arr("Nc_Gamma")
    m.Nc_L = arr("Nc_L")
    m.Nc_X = arr("Nc_X")
    m.Nv = arr("Nv")
    m.ni = arr("ni")
    m.alpha = arr("nonparabolicity")
    m.dielectric = arr("dielectric")
    m.lattice_constant = arr("lattice_constant")
    m.meff_e_Gamma = arr("meff_e_Gamma")
    m.meff_h_DOS = arr("meff_h_DOS")
    m.eps = EPS0 * m.dielectric

    # dopants and transport parameters come from the layer stack
    m.Na = np.array([l.Na for l in layers], dtype=float)
    m.Nd = np.array([l.Nd for l in layers], dtype=float)
    m.Nnet = m.Nd - m.Na
    m.mu_n = np.array([l.mu_n if l.mu_n else p["mu_n"] for l, p in zip(layers, p0)])
    m.mu_p = np.array([l.mu_p if l.mu_p else p["mu_p"] for l, p in zip(layers, p0)])
    m.tau_n = np.array([l.tau_n if l.tau_n else device.tau_n for l in layers])
    m.tau_p = np.array([l.tau_p if l.tau_p else device.tau_p for l in layers])
    m.B_rad = np.array([l.B_rad if l.B_rad is not None
                        else (device.B_rad if p["direct"] else 0.0)
                        for l, p in zip(layers, p0)])

    if sinfo is None:
        # Unstrained: heavy and light holes stay degenerate at the VBM, and
        # the split-off band sits Delta_SO below it.
        m.eps_in = np.zeros(m.N)
        m.eps_out = np.zeros(m.N)
        m.E_hh = m.VBO.copy()
        m.E_lh = m.VBO.copy()
        m.E_so = m.VBO - np.array([p["Delta_SO"] or 0.0 for p in p0],
                                  dtype=float)
        m.strain_applied = np.zeros(m.N, dtype=bool)
    else:
        m.eps_in = sinfo["eps_in"]
        m.eps_out = sinfo["eps_out"]
        m.E_hh = sinfo["E_hh"]
        m.E_lh = sinfo["E_lh"]
        m.E_so = sinfo["E_so"]
        m.strain_applied = sinfo["applied"]

    kT = KB_EV * T
    m.Ei = m.VBO + m.Eg / 2.0 + (kT / 2.0) * np.log(m.Nv / m.Nc)
    # Equilibrium carrier densities, filled in lazily by
    # :func:`equilibrium_densities`.  They are the reference for the
    # recombination model, because with Fermi-Dirac (and Kane) statistics
    # n*p is *not* exactly ni^2 at equilibrium -- the residual is a few
    # per cent, enough to fake a saturation current 100x too large.
    m.n0 = None
    m.p0 = None
    return m


def equilibrium_densities(m, approx="kane"):
    """
    Equilibrium carrier densities, used as the recombination reference.

    Solved once per model and cached.  Using these instead of the
    non-degenerate ``ni`` makes the net recombination vanish *identically*
    at equilibrium, which it must.
    """
    if m.n0 is None:
        z = np.zeros(m.N)
        psi0 = charge_neutral_guess(m, z, z, approx)
        psi, _ = solve_poisson(m, psi0, z, z, 0.0, 0.0, approx)
        p0, n0, *_ = carriers(m, psi, z, z, approx)
        m.n0, m.p0 = np.maximum(n0, 1e-30), np.maximum(p0, 1e-30)
    return m.n0, m.p0


def carriers(m, psi, Fp, Fn, approx="kane"):
    """
    Carrier densities and their derivatives w.r.t. ``psi``.

    Returns ``(p, n, n_gamma, n_L, n_X, dp_dpsi, dn_dpsi)``.
    """
    kT = KB_EV * m.T
    phi_p = (m.VBO - psi - Fp) / kT

    if approx == "boltzmann":
        p = m.Nv * np.exp(np.clip(phi_p, -700.0, 700.0))
        dp = -p / kT
    else:
        F, dF = generalized_fermi(phi_p, 0.0)
        p = C_F * m.Nv * F
        dp = -C_F * m.Nv * dF / kT

    alpha = m.alpha if approx == "kane" else 0.0
    phi_g = (Fn - m.CBO_Gamma + psi) / kT
    phi_l = (Fn - m.CBO_L + psi) / kT
    phi_x = (Fn - m.CBO_X + psi) / kT

    if approx == "boltzmann":
        ng = m.Nc_Gamma * np.exp(np.clip(phi_g, -700.0, 700.0))
        nl = m.Nc_L * np.exp(np.clip(phi_l, -700.0, 700.0))
        nx = m.Nc_X * np.exp(np.clip(phi_x, -700.0, 700.0))
        dn = (ng + nl + nx) / kT
    else:
        Hg, dHg = generalized_fermi(phi_g, alpha)
        Hl, dHl = generalized_fermi(phi_l, 0.0)
        Hx, dHx = generalized_fermi(phi_x, 0.0)
        ng = C_F * m.Nc_Gamma * Hg
        nl = C_F * m.Nc_L * Hl
        nx = C_F * m.Nc_X * Hx
        dn = C_F * (m.Nc_Gamma * dHg + m.Nc_L * dHl + m.Nc_X * dHx) / kT

    p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
    n = np.nan_to_num(ng + nl + nx, nan=0.0, posinf=0.0, neginf=0.0)
    dp = np.nan_to_num(dp, nan=1.0)
    dn = np.nan_to_num(dn, nan=1.0)
    return p, n, ng, nl, nx, dp, dn


def charge_neutral_guess(m, Fp, Fn, approx="kane"):
    """
    Initial guess for ``psi``: local charge neutrality, refined with a
    damped vector Newton iteration.

    Seeding with the Boltzmann result (majority carrier density equal to
    the net doping) and refining handles degeneracy, satellite valleys and
    band non-parabolicity.
    """
    kT = KB_EV * m.T
    psi = np.empty(m.N)
    ntype = m.Nnet > 0.0
    ptype = m.Nnet < 0.0
    intr = ~(ntype | ptype)

    psi[ntype] = m.CBO_Gamma[ntype] + kT * np.log(
        np.maximum(m.Nnet[ntype], 1.0) / np.maximum(m.Nc_Gamma[ntype], 1.0))
    psi[ptype] = m.VBO[ptype] - kT * np.log(
        np.maximum(-m.Nnet[ptype], 1.0) / np.maximum(m.Nv[ptype], 1.0))
    psi[intr] = m.Ei[intr]

    residual = np.zeros(m.N)
    d = np.empty(m.N)

    def G(psi_):
        p, n, *_ , dp, dn = carriers(m, psi_, Fp, Fn, approx)
        return p - n + m.Nnet - residual

    def A(psi_):
        p, n, *_ , dp, dn = carriers(m, psi_, Fp, Fn, approx)
        d[:] = dp - dn
        d[np.abs(d) < 1e-30] = 1.0
        return diags([d], [0], shape=(m.N, m.N), format="csc")

    # Damped, step-limited fixed-point Newton (scalar per node, vectorised).
    for _ in range(80):
        rho = G(psi)
        p, n, *_ , dp, dn = carriers(m, psi, Fp, Fn, approx)
        jac = dp - dn
        jac = np.where(np.abs(jac) < 1e-30, 1.0, jac)
        step = -rho / jac
        step = np.clip(step, -0.5, 0.5)
        psi = psi + step
        if np.abs(step).max() < 1e-10:
            break
    return psi


def _face_eps(m):
    e = m.eps
    return 2.0 * e[:-1] * e[1:] / (e[:-1] + e[1:])


def _boundary(m, slot, node, V_left, V_right, approx):
    """
    Dirichlet value of ``psi`` at the contact in slot ``slot`` (0 = left,
    1 = right), evaluated with the material of grid node ``node``.
    """
    contact = m.contacts[slot]
    Vc = V_left if slot == 0 else V_right
    F = -Vc
    if contact.kind == "ohmic":
        kT = KB_EV * m.T
        Nnet = m.Nnet[node]
        if Nnet > 0.0:
            psi = m.CBO_Gamma[node] + kT * np.log(Nnet / m.Nc_Gamma[node])
        elif Nnet < 0.0:
            psi = m.VBO[node] - kT * np.log(-Nnet / m.Nv[node])
        else:
            psi = m.Ei[node]

        # refine against the full statistics at the contact Fermi level
        def G(x):
            p, n, *_ , _, _ = carriers_scalar(m, node, x, F, approx)
            return p - n + Nnet

        def A(x):
            p, n, *_ , dp, dn = carriers_scalar(m, node, x, F, approx)
            j = dp - dn
            return j if abs(j) > 1e-30 else 1.0

        r = scalar_newton(G, A, psi, rtol=1e-10, atol=1e-12)
        return r.value
    elif contact.kind == "schottky":
        from .materials.data import VACUUM_LEVEL
        return (VACUUM_LEVEL - contact.work_function) + Vc
    raise ValueError(f"Unsupported contact type: {contact!r}")


def carriers_scalar(m, i, psi, F, approx):
    """Carrier densities at a single node (used for contact boundary values)."""
    kT = KB_EV * m.T
    phi_p = (m.VBO[i] - psi - F) / kT
    if approx == "boltzmann":
        p = m.Nv[i] * np.exp(np.clip(phi_p, -700.0, 700.0))
        dp = -p / kT
    else:
        Fv, dFv = generalized_fermi(np.array(phi_p), 0.0)
        p = float(C_F * m.Nv[i] * Fv)
        dp = float(-C_F * m.Nv[i] * dFv / kT)

    alpha = m.alpha[i] if approx == "kane" else 0.0
    phis = np.array([(F - m.CBO_Gamma[i] + psi) / kT,
                     (F - m.CBO_L[i] + psi) / kT,
                     (F - m.CBO_X[i] + psi) / kT])
    if approx == "boltzmann":
        n = float(np.sum(m.Nc_Gamma[i] * np.exp(np.clip(phis[0], -700, 700))
                         + m.Nc_L[i] * np.exp(np.clip(phis[1], -700, 700))
                         + m.Nc_X[i] * np.exp(np.clip(phis[2], -700, 700))))
        dn = n / kT
    else:
        Hg, dHg = generalized_fermi(phis[0], alpha)
        Hl, dHl = generalized_fermi(phis[1], 0.0)
        Hx, dHx = generalized_fermi(phis[2], 0.0)
        n = float(C_F * (m.Nc_Gamma[i] * Hg + m.Nc_L[i] * Hl + m.Nc_X[i] * Hx))
        dn = float(C_F * (m.Nc_Gamma[i] * dHg + m.Nc_L[i] * dHl
                          + m.Nc_X[i] * dHx) / kT)
    return p, n, 0.0, 0.0, 0.0, dp, dn


class Solution:
    """Container for a solved operating point."""

    def __init__(self, **kw):
        self.__dict__.update(kw)

    def as_dict(self, downsample=None):
        def pick(name):
            v = getattr(self, name)
            if v is None:
                return None
            v = np.asarray(v)
            if downsample and v.ndim == 1 and v.size > downsample:
                idx = np.linspace(0, v.size - 1, downsample).astype(int)
                v = v[idx]
            return np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0).tolist()

        return {k: pick(k) for k in ("x_nm", "Ev", "Ec", "Ec_Gamma", "Ec_L",
                                     "Ec_X", "Ei", "Fp", "Fn", "psi", "field",
                                     "n", "p", "n_Gamma", "n_L", "n_X",
                                     "Ev_HH", "Ev_LH", "Ev_SO",
                                     "rho", "Nd", "Na")
                if hasattr(self, k)}


def solve_poisson(m, psi0, Fp, Fn, V_left=0.0, V_right=0.0, approx="kane",
                  n_override=None):
    """
    Solve the nonlinear Poisson equation for fixed quasi-Fermi profiles.

    Parameters
    ----------
    m : Model
    psi0 : ndarray
        Initial guess.
    Fp, Fn : ndarray
        Hole / electron quasi-Fermi levels (eV) at every node.
    V_left, V_right : float
        Contact potentials (V).
    n_override : tuple, optional
        ``(n, dn_dpsi)`` to substitute for the electron density and its
        derivative, instead of taking them from :func:`carriers`.  This is
        how the Schrodinger-Poisson loop feeds the quantized density in:
        the density is held fixed *during* the Newton solve and refreshed
        outside it, which is a Gummel iteration.
    """
    N, dx = m.N, m.dx
    dx2 = dx * dx
    eps_f = _face_eps(m)  # length N-1
    a = _boundary(m, 0, 0, V_left, V_right, approx)
    b = _boundary(m, 1, N - 1, V_left, V_right, approx)

    main = np.zeros(N)
    upper = np.zeros(N - 1)
    lower = np.zeros(N - 1)
    scale = np.ones(N)
    F = np.empty(N)
    psi = np.array(psi0, dtype=float)
    psi[0] = a
    psi[-1] = b

    def densities(psi_):
        p, n, *_ , dp, dn = carriers(m, psi_, Fp, Fn, approx)
        if n_override is not None:
            n, dn = n_override[0], n_override[1]
        return p, n, dp, dn

    def G(psi_):
        p, n, dp, dn = densities(psi_)
        F[1:-1] = (eps_f[1:] * (psi_[2:] - psi_[1:-1])
                   - eps_f[:-1] * (psi_[1:-1] - psi_[:-2])
                   + Q * (p[1:-1] - n[1:-1] + m.Nnet[1:-1]) * dx2)
        F[0] = psi_[0] - a
        F[-1] = psi_[-1] - b
        return F / scale

    def A(psi_):
        p, n, dp, dn = densities(psi_)
        main[1:-1] = (-eps_f[1:] - eps_f[:-1]
                      + Q * (dp[1:-1] - dn[1:-1]) * dx2)
        main[0] = 1.0
        main[-1] = 1.0
        upper[1:] = eps_f[1:]       # coupling to psi_{i+1}
        lower[:-1] = eps_f[:-1]     # coupling to psi_{i-1}
        upper[0] = 0.0
        lower[-1] = 0.0
        scale[:] = row_scale(main, upper, lower)
        return diags([lower / scale[1:], main / scale, upper / scale[:-1]],
                     [-1, 0, 1], shape=(N, N), format="csc")

    res = newton(G, A, psi, rtol=0.0, atol=1e-9, max_iter=300)
    psi = res.value
    psi[0] = a
    psi[-1] = b
    return psi, res


def evaluate(m, psi, Fp, Fn, V, approx="kane"):
    """Turn a potential profile into a full :class:`Solution`."""
    p, n, ng, nl, nx, dp, dn = carriers(m, psi, Fp, Fn, approx)
    Ev = m.VBO - psi
    Ec_g = m.CBO_Gamma - psi
    Ec_l = m.CBO_L - psi
    Ec_x = m.CBO_X - psi
    Ec = m.CBO - psi
    Ei = m.Ei - psi
    field = -np.gradient(psi, m.dx)
    rho = Q * (p - n + m.Nnet)
    return Solution(
        x_nm=m.x * 1e7, psi=psi, Ev=Ev, Ec=Ec, Ec_Gamma=Ec_g, Ec_L=Ec_l,
        Ec_X=Ec_x, Ei=Ei, Fp=Fp, Fn=Fn, field=field, n=n, p=p,
        n_Gamma=ng, n_L=nl, n_X=nx, rho=rho, Nd=m.Nd, Na=m.Na,
        # The split valence edges coincide with Ev unless the stack is
        # strained; the front end draws them separately when they do not.
        Ev_HH=m.E_hh - psi, Ev_LH=m.E_lh - psi, Ev_SO=m.E_so - psi,
        V=V, T=m.T, converged=True,
    )
