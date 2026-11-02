"""
Fermi-Dirac integrals for semiconductor carrier statistics.

Definitions
-----------
The (unnormalised) Fermi-Dirac integral of order ``k`` is

    F_k(phi) = int_0^inf  t**k / (1 + exp(t - phi))  dt

so that the electron density in a parabolic band is

    n = (2 / sqrt(pi)) * Nc * F_{1/2}(phi),      phi = (Ef - Ec) / kT

which reduces to the Boltzmann result ``n = Nc * exp(phi)`` since
``F_{1/2}(phi) -> (sqrt(pi)/2) * exp(phi)`` for ``phi << 0``.

For the Kane non-parabolic (k.p) description of the Gamma valley the
generalised Fermi-Dirac integral is required:

    G_k(phi, alpha) = int_0^inf  t**k * sqrt(1 + alpha*t)
                                / (1 + exp(t - phi))  dt

and the non-parabolic electron density is

    n = (2 / sqrt(pi)) * Nc * H_{1/2}(phi, alpha)
    H_{1/2} = G_{1/2} + 2*alpha*G_1

with ``alpha = (kT / Eg) * (1 - meff/m0)**2`` the Kane non-parabolicity
parameter on the ``t = E / kT`` scale.  ``alpha = 0`` recovers ``F_{1/2}``.

Numerics
--------
Both integrands carry a ``sqrt(t)`` endpoint singularity in their derivative,
which destroys the spectral accuracy of Gauss-Legendre quadrature.  The
substitution ``t = u**2`` (``dt = 2u du``) removes it:

    F_{1/2} = 2 * int_0^inf  u**2 * sigma(u)  du
    G_{1/2} = 2 * int_0^inf  u**2 * sqrt(1 + alpha*u**2) * sigma(u) du
    G_1     = 2 * int_0^inf  u**3 * sqrt(1 + alpha*u**2) * sigma(u) du

with ``sigma(u) = 1 / (1 + exp(u**2 - phi))``.  All three integrands are
smooth in ``u``, so Gauss-Legendre converges rapidly.  The derivatives with
respect to ``phi`` needed for the Newton Jacobian follow from
``d(sigma)/d(phi) = sigma * (1 - sigma)``, and share the same quadrature
nodes, so values and derivatives are produced in a single pass.

The upper limit is ``u_max = sqrt(max(phi, 0) + CUT)``; beyond that the
integrand is bounded by ``exp(phi - u**2) <= exp(-CUT)``.
"""

from __future__ import annotations

import numpy as np
from scipy.special import expit

__all__ = ["KB_EV", "generalized_fermi", "fermi_half", "bernoulli"]


KB_EV = 8.617333262e-5  # Boltzmann constant, eV / K

_CUT = 60.0  # exponential cut-off in units of kT
_GL_ORDER = 128
_GL_X, _GL_W = np.polynomial.legendre.leggauss(_GL_ORDER)


def generalized_fermi(phi, alpha=0.0):
    """
    Non-parabolic (Kane) carrier integral and its derivative.

    Parameters
    ----------
    phi : array_like
        Normalised Fermi energy ``(Ef - Ec) / kT`` (electrons) or
        ``(Ev - Ef) / kT`` (holes).
    alpha : array_like, optional
        Kane non-parabolicity parameter on the ``E / kT`` scale.  Zero
        (the default) gives the parabolic result.

    Returns
    -------
    H : ndarray
        ``G_{1/2}(phi, alpha) + 2 * alpha * G_1(phi, alpha)``.  Equals
        ``F_{1/2}(phi)`` when ``alpha = 0``.
    dH : ndarray
        ``d H / d phi``.

    Notes
    -----
    The carrier density follows as ``n = 2 / sqrt(pi) * Nc * H`` and
    ``dn/d(phi) = 2 / sqrt(pi) * Nc * dH``.
    """
    phi_a = np.asarray(phi, dtype=np.float64)
    alpha_a = np.asarray(alpha, dtype=np.float64)
    phi_a, alpha_a = np.broadcast_arrays(phi_a, alpha_a)

    # integration limit in u = sqrt(t)
    umax = np.sqrt(np.maximum(phi_a, 0.0) + _CUT)

    # Gauss-Legendre nodes mapped onto [0, umax] (vectorised over grid)
    u = 0.5 * umax[..., None] * (_GL_X + 1.0)
    w = 0.5 * umax[..., None] * _GL_W

    sigma = expit(phi_a[..., None] - u * u)  # 1 / (1 + exp(u^2 - phi))
    root = np.sqrt(1.0 + alpha_a[..., None] * u * u)

    # weights for G_{1/2} and G_1 value / derivative integrands
    a_val = 2.0 * w * u * u * root
    b_val = a_val * u

    sv = sigma
    sd = sigma * (1.0 - sigma)

    g_half = np.einsum("...i,...i->...", a_val, sv)
    g_one = np.einsum("...i,...i->...", b_val, sv)
    dg_half = np.einsum("...i,...i->...", a_val, sd)
    dg_one = np.einsum("...i,...i->...", b_val, sd)

    two_a = 2.0 * alpha_a
    H = g_half + two_a * g_one
    dH = dg_half + two_a * dg_one
    return H, dH


def fermi_half(phi):
    """
    Parabolic Fermi-Dirac integral ``F_{1/2}(phi)`` and its derivative.

    Convenience wrapper around :func:`generalized_fermi` with ``alpha = 0``.
    """
    return generalized_fermi(phi, 0.0)


def bernoulli(x):
    """
    Bernoulli function ``B(x) = x / (exp(x) - 1)``, with ``B(0) = 1``.

    Central to the Scharfetter-Gummel discretisation of the drift-diffusion
    currents.  Evaluated stably for small and large ``|x|``.
    """
    x = np.asarray(x, dtype=np.float64)
    small = np.abs(x) < 1.0e-6
    xs = np.where(small, 1.0, x)  # avoid 0/0
    with np.errstate(over="ignore", invalid="ignore"):
        big = xs / np.expm1(xs)
    series = 1.0 - x / 2.0 + x * x / 12.0
    return np.where(small, series, big)
