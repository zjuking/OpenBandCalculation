"""
One-dimensional Schrodinger solver and the Schrodinger-Poisson loop.

Classical drift-diffusion puts the electron gas exactly *at* the band edge:
the density peaks where ``Ec`` is lowest, which for a heterojunction is the
interface plane itself.  Quantization says otherwise -- confining a state to
a slab of thickness ``L`` costs at least ``hbar^2 pi^2 / 2 m* L^2`` of kinetic
energy, so the ground state floats above the band edge and its wavefunction
has a node-free lobe pushed *away* from the interface.  That is the whole
point of the correction: the channel of a HEMT is not at the metallurgical
interface, it is a few nanometres behind it, and the sheet density does not
follow ``exp((F - Ec)/kT)`` but the subband populations.

Discretization
--------------
The envelope equation is solved in the BenDaniel-Duke form::

    -(hbar^2/2) d/dx (1/m* dphi/dx) + Ec(x) phi = E phi

which keeps the Hamiltonian Hermitian even where ``m*`` jumps.  The interface
value of ``1/m*`` is the *harmonic mean* of the two nodal values, for exactly
the reason the dielectric constant is treated that way in :mod:`.solver` --
both sit in the numerator of a flux, so two half-cells in series combine by
harmonic averaging.  (Taking the arithmetic mean of ``1/m*`` instead is the
usual mistake and it breaks Hermiticity of the assembled matrix.)

The result is a symmetric tridiagonal eigenproblem, handed to
``scipy.linalg.eigh_tridiagonal`` with ``select='i'`` so only the few lowest
levels are computed.

Density
-------
A bound subband is a *two-dimensional* gas -- free in the plane, quantized
along x -- so its sheet density is exact and closed form, because the 2D
Fermi-Dirac integral is just a logarithm::

    N_i = (m* kT / pi hbar^2) ln(1 + exp((F - E_i)/kT))

and the volume density follows from ``n(x) = sum_i N_i |phi_i(x)|^2`` with
``int |phi_i|^2 dx = 1``.

The continuum cannot be dropped or the density outside the well would vanish.
Electrons at position x are itinerant only if their total energy clears the
*continuum threshold* ``E_cut`` -- the lower of the two contact band edges,
since an electron above that can escape to a contact.  Restricting the
thermal integral to perpendicular kinetic energies above
``max(0, E_cut - Ec(x))`` gives, in the Boltzmann limit, a clean closed form
in terms of a regularized incomplete gamma::

    n_cont(x) = n_3D(x) * Q(3/2, max(0, E_cut - Ec(x))/kT)

which is 1 far from the well (all electrons there are itinerant) and decays
inside it.  Under Fermi-Dirac statistics this is applied as a multiplicative
weight on the degenerate density -- an approximation, but the 2D subbands
carry the physics and the weight only has to be right in the crossover.

Not modelled: band non-parabolicity in the confinement direction (the
two-band energy-dependent mass turns the eigenproblem nonlinear; subband
energies in narrow wells come out a few meV too high), and holes (which need
at least a 4x4 k.p treatment to get hh/lh right, not a single mass).
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import eigh_tridiagonal
from scipy.special import gammaincc

from .fermi import KB_EV

__all__ = ["HBAR2_2M0", "solve_schrodinger", "subband_population",
           "quantum_density", "solve_quantum_poisson"]

#: hbar^2 / (2 m0) in eV cm^2.
HBAR2_2M0 = 3.80998212e-16

#: m0 kT / (pi hbar^2) per kelvin, in cm^-2: the 2D density-of-states
#: prefactor for a particle of mass m0.
DOS2D_PER_K = 3.59985e10


def _face_inverse_mass(m_eff: np.ndarray) -> np.ndarray:
    """
    ``1/m*`` at the cell faces: harmonic mean of the nodal values.

    ``1/m*`` multiplies the derivative in the BenDaniel-Duke operator, so
    two half-cells in series combine the same way the dielectric constant
    does in Poisson's equation.
    """
    c = 1.0 / np.asarray(m_eff, dtype=float)
    return 2.0 * c[:-1] * c[1:] / (c[:-1] + c[1:])


def solve_schrodinger(x, V, m_eff, n_states=12):
    """
    Lowest eigenpairs of the 1D envelope equation on a uniform grid.

    Parameters
    ----------
    x : ndarray
        Node positions, cm, uniformly spaced.
    V : ndarray
        Potential energy, eV (the conduction band edge ``Ec``).
    m_eff : ndarray
        Effective mass in units of m0, one value per node.
    n_states : int
        How many of the lowest levels to return.

    Returns
    -------
    (E, phi) : (ndarray, ndarray)
        Energies in eV (ascending) and the wavefunctions as columns of
        ``phi`` (shape ``(N, k)``), normalized so ``int |phi|^2 dx = 1``.
        Dirichlet conditions (``phi = 0``) apply beyond both ends.
    """
    x = np.asarray(x, dtype=float)
    V = np.asarray(V, dtype=float)
    m_eff = np.asarray(m_eff, dtype=float)
    h = float(x[1] - x[0])
    kin = HBAR2_2M0 / (h * h)
    cf = _face_inverse_mass(m_eff)                  # length N-1

    c = 1.0 / m_eff
    d = np.empty(x.size)
    # Both end nodes need the face *outside* the grid as well: the ghost node
    # is held at zero (Dirichlet), but its flux term still lands on the
    # diagonal.  Omitting it is the easy mistake and it silently turns the
    # boundary condition into zero-flux (Neumann), which drops the spectrum
    # by one -- a spurious E ~ 0 state appears and every level shifts down a
    # notch, and the ground state of a triangular well comes out less than
    # half its true value because nothing confines it on the wall side.
    d[0] = kin * (cf[0] + c[0]) + V[0]
    d[-1] = kin * (cf[-1] + c[-1]) + V[-1]
    d[1:-1] = kin * (cf[1:] + cf[:-1]) + V[1:-1]
    e = -kin * cf                                    # off-diagonal, length N-1

    k = int(max(1, min(n_states, x.size)))
    w, v = eigh_tridiagonal(d, e, select="i", select_range=(0, k - 1))

    # eigh_tridiagonal returns unit 2-norm vectors.  The inner product the
    # discrete operator is self-adjoint under is the plain nodal sum, so
    # that -- not the trapezoid rule -- is the consistent quadrature here:
    # int |phi|^2 dx ~ h * sum_i phi_i^2.  Rescaling by 1/sqrt(h) then makes
    # the states orthonormal to machine precision.  (Trapezoid weights look
    # more "correct" for the integral but they belong to a different inner
    # product, and leave an O(h) residual in the Gram matrix.)
    norm = np.sqrt(np.sum(v * v, axis=0)) * np.sqrt(h)
    norm[norm <= 0.0] = 1.0
    return w, v / norm


def subband_population(m_eff, E, F, T):
    """
    Sheet density (cm^-2) of a 2D subband of energy ``E``.

    The 2D Fermi-Dirac integral is ``F_0(eta) = ln(1 + exp(eta))``, so this
    is exact for degenerate statistics too -- no quadrature needed.
    """
    kT = KB_EV * T
    eta = (F - E) / kT
    eta = np.clip(eta, -700.0, 700.0)
    # ln(1 + e^eta) written so large negative eta does not overflow exp
    log_occ = np.where(eta > 0.0,
                       eta + np.log1p(np.exp(-np.abs(eta))),
                       np.log1p(np.exp(eta)))
    return DOS2D_PER_K * T * m_eff * log_occ


def quantum_density(m, psi, Fn, approx="kane", n_states=12):
    """
    Electron density with the bound subbands treated quantum mechanically.

    Returns ``(n, dn_dpsi, info)``.  ``dn_dpsi`` keeps only the continuum
    part's dependence on ``psi``: the subband term is updated by the outer
    Schrodinger-Poisson loop instead, which is what a Gummel iteration is.
    """
    kT = KB_EV * m.T
    Ec = m.CBO_Gamma - psi

    # continuum threshold: an electron above the lower contact band edge can
    # escape to that contact, so anything below is a genuine bound state
    E_cut = float(min(Ec[0], Ec[-1]))

    levels, phi = solve_schrodinger(m.x, Ec, m.meff_e_Gamma, n_states)
    bound = levels < E_cut
    n_bound_levels = int(np.count_nonzero(bound))

    n_q = np.zeros(m.N)
    sheet = 0.0
    pop_bound = np.zeros(0)
    if n_bound_levels:
        E_b = levels[bound]
        # The DOS mass is the *in-plane* mass of a subband, and a state
        # straddling an interface samples both materials, so take the
        # |phi|^2-weighted average rather than the mass at whichever single
        # node happens to be the well bottom.
        phb = phi[:, bound]
        m_state = (np.sum(phb * phb * m.meff_e_Gamma[:, None], axis=0)
                   * m.dx)
        pop = subband_population(m_state, E_b, float(np.mean(Fn)), m.T)
        n_q += phb @ pop
        pop_bound = pop
        sheet = float(np.sum(pop))

    # classical 3D density, weighted down to its itinerant part
    from .solver import carriers
    p, n3, *_rest, dp, dn3 = carriers(m, psi, np.zeros(m.N), Fn, approx)
    delta = np.maximum(E_cut - Ec, 0.0) / kT
    cont = np.asarray(gammaincc(1.5, delta), dtype=float)
    cont = np.clip(np.nan_to_num(cont, nan=1.0), 0.0, 1.0)

    info = {"levels": levels, "phi": phi, "bound": bound,
            "E_cut": E_cut, "sheet_density": sheet,
            "population": pop_bound,
            "continuum_weight": cont,
            "n_bound_levels": n_bound_levels}
    return n_q + n3 * cont, dn3 * cont, info


def solve_quantum_poisson(m, approx="kane", max_iter=150, tol=1e-9,
                          omega=1.0, n_states=12):
    """
    Self-consistent Schrodinger-Poisson at equilibrium.

    Starts from the classical solution, then alternates: solve Schrodinger
    on the current band edge, rebuild the density, re-solve Poisson with
    that density held fixed.  Holding the density fixed is what makes this a
    Gummel iteration rather than a coupled Newton solve -- the price is
    linear convergence, the reward is that each step is a well-conditioned
    Poisson problem.

    Undamped, the map turns out to be a contraction: on an AlGaAs/GaAs
    heterojunction the potential change falls by a factor ~0.78 per sweep,
    so ~65 sweeps reach 1e-9.  The feedback loop is stable because it is
    self-limiting -- pushing electrons off the interface lowers the field
    there, which shallows the well, which pushes them back.  ``omega`` is
    kept as a safety net: if a sweep ever makes things worse the step is
    halved rather than trusted.
    """
    from .solver import charge_neutral_guess, solve_poisson

    Fn = np.zeros(m.N)
    Fp = np.zeros(m.N)
    psi, _ = solve_poisson(m, charge_neutral_guess(m, Fp, Fn, approx),
                           Fp, Fn, 0.0, 0.0, approx)

    info = {}
    converged = False
    prev = np.inf
    w = float(omega)
    for _ in range(int(max_iter)):
        n_q, dn_q, info = quantum_density(m, psi, Fn, approx, n_states)
        psi_new, _res = solve_poisson(m, psi, Fp, Fn, 0.0, 0.0, approx,
                                      n_override=(n_q, dn_q))
        err = float(np.abs(psi_new - psi).max())
        if err > prev:                      # a sweep made it worse: back off
            w = max(0.5 * w, 0.05)
        else:
            w = min(2.0 * w, float(omega))
        prev = err
        psi = psi + w * (psi_new - psi)
        if err < tol:
            converged = True
            break

    n_q, dn_q, info = quantum_density(m, psi, Fn, approx, n_states)
    info["converged"] = converged
    info["psi"] = psi
    info["n_quantum"] = n_q
    return psi, info
