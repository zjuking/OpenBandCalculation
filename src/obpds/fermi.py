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

__all__ = ["KB_EV", "generalized_fermi", "generalized_fermi_fast",
           "fermi_half", "bernoulli"]


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


# ---------------------------------------------------------------------------
# Tabulated evaluation.
#
# The Gauss--Legendre quadrature of :func:`generalized_fermi` is the accuracy
# reference, but it costs a 128-node reduction per node per call, and in the
# self-consistent Schrodinger--Poisson loop that single kernel absorbs most of
# the runtime (the paper's profile puts it at 82 %).  The function below
# trades that reduction for a lookup in a precomputed table: :math:`H_{1/2}`
# and its derivative are tabulated on a ``(phi, alpha)`` grid and recovered by
# cubic Hermite interpolation in ``phi`` and linear interpolation in ``alpha``.
# The Hermite interpolant uses the *tabulated derivative* as its slope, so the
# derivative it returns is the analytic derivative of the interpolant and the
# value is exact at the grid nodes.
#
# Accuracy.  The dominant interpolation error is the linear step in ``alpha``
# (``H`` is only mildly nonlinear there), giving a worst-case relative error of
# a few :math:`10^{-4}` over the table -- three orders below the mesh error the
# quantised results are quoted with.  Grid points *outside* the table are not
# interpolated at all: they fall back to the exact quadrature, so the fast
# path is a pure optimisation and never changes a result at the table boundary.
# ---------------------------------------------------------------------------

#: Table bounds and steps.  ``phi`` covers the range the carrier arguments
#: reach in practice (the conduction band in a device spans tens of kT), and
#: ``alpha`` covers every III--V alloy in the catalogue at room temperature
#: (the largest is InAsSb, ~0.3); narrower-gap materials at higher temperature
#: exceed it and simply fall back to quadrature.
_TAB_PHI_MIN = -80.0
_TAB_PHI_MAX = 40.0
_TAB_PHI_STEP = 0.02
_TAB_ALPHA_MAX = 0.32
_TAB_ALPHA_STEP = 0.005

_fermi_tab_cache = None


def _fermi_table():
    """Build (and memoise) the ``(phi_grid, alpha_grid, H, dH)`` table."""
    global _fermi_tab_cache
    if _fermi_tab_cache is not None:
        return _fermi_tab_cache
    phi_grid = np.arange(_TAB_PHI_MIN, _TAB_PHI_MAX + 0.5 * _TAB_PHI_STEP,
                         _TAB_PHI_STEP)
    alpha_grid = np.arange(0.0, _TAB_ALPHA_MAX + 0.5 * _TAB_ALPHA_STEP,
                           _TAB_ALPHA_STEP)
    H = np.empty((alpha_grid.size, phi_grid.size))
    dH = np.empty_like(H)
    for k, a in enumerate(alpha_grid):
        H[k], dH[k] = generalized_fermi(phi_grid, a)
    _fermi_tab_cache = (phi_grid, alpha_grid, H, dH)
    return _fermi_tab_cache


def _fermi_interp(phif, alphaf, phi_grid, alpha_grid, Htab, dHtab):
    """Table interpolation; all inputs must already lie inside the table."""
    fp = (phif - _TAB_PHI_MIN) / _TAB_PHI_STEP
    i = np.clip(np.floor(fp).astype(np.intp), 0, phi_grid.size - 2)
    t = fp - i

    ap = alphaf / _TAB_ALPHA_STEP
    j = np.clip(np.floor(ap).astype(np.intp), 0, alpha_grid.size - 2)
    s = ap - j

    H00 = Htab[j, i]
    H01 = Htab[j, i + 1]
    d00 = dHtab[j, i]
    d01 = dHtab[j, i + 1]
    H10 = Htab[j + 1, i]
    H11 = Htab[j + 1, i + 1]
    d10 = dHtab[j + 1, i]
    d11 = dHtab[j + 1, i + 1]

    h = _TAB_PHI_STEP
    # Cubic Hermite basis in phi, and its derivative (the two endpoint slopes
    # are the tabulated derivatives, so value and derivative are both exact at
    # the grid nodes and C1 across cells).
    b00 = 2 * t ** 3 - 3 * t ** 2 + 1
    b10 = t ** 3 - 2 * t ** 2 + t
    b01 = -2 * t ** 3 + 3 * t ** 2
    b11 = t ** 3 - t ** 2
    db00 = 6 * t ** 2 - 6 * t
    db10 = 3 * t ** 2 - 4 * t + 1
    db01 = -6 * t ** 2 + 6 * t
    db11 = 3 * t ** 2 - 2 * t

    Hj = b00 * H00 + b10 * (d00 * h) + b01 * H01 + b11 * (d01 * h)
    Hj1 = b00 * H10 + b10 * (d10 * h) + b01 * H11 + b11 * (d11 * h)
    dHj = (db00 * H00 + db10 * (d00 * h) + db01 * H01 + db11 * (d01 * h)) / h
    dHj1 = (db00 * H10 + db10 * (d10 * h) + db01 * H11 + db11 * (d11 * h)) / h

    H = (1.0 - s) * Hj + s * Hj1
    dH = (1.0 - s) * dHj + s * dHj1
    return H, dH


def generalized_fermi_fast(phi, alpha=0.0):
    """
    Tabulated :func:`generalized_fermi`, for the hot path in the solvers.

    Identical interface and semantics to :func:`generalized_fermi`: returns
    ``(H, dH)`` with ``H = G_{1/2} + 2 alpha G_1`` and ``dH = dH/dphi``.
    Values inside the table are interpolated (worst-case relative error a few
    :math:`10^{-4}`); values outside it -- a larger ``|phi|`` or a larger
    ``alpha`` than the table covers -- are evaluated by the exact quadrature,
    so the function is a drop-in replacement that never sacrifices
    correctness, only the speed of the common case.
    """
    phi_a = np.asarray(phi, dtype=np.float64)
    alpha_a = np.asarray(alpha, dtype=np.float64)
    phi_a, alpha_a = np.broadcast_arrays(phi_a, alpha_a)
    shape = phi_a.shape
    phif = phi_a.ravel()
    alphaf = alpha_a.ravel()

    phi_grid, alpha_grid, Htab, dHtab = _fermi_table()

    inside = ((phif >= _TAB_PHI_MIN) & (phif <= _TAB_PHI_MAX)
              & (alphaf >= 0.0) & (alphaf <= _TAB_ALPHA_MAX))

    H = np.empty_like(phif)
    dH = np.empty_like(phif)
    if inside.all():
        H, dH = _fermi_interp(phif, alphaf, phi_grid, alpha_grid, Htab, dHtab)
    else:
        H[inside], dH[inside] = _fermi_interp(
            phif[inside], alphaf[inside], phi_grid, alpha_grid, Htab, dHtab)
        H[~inside], dH[~inside] = generalized_fermi(
            phif[~inside], alphaf[~inside])
    return H.reshape(shape), dH.reshape(shape)


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
