"""
Damped Newton iteration for nonlinear systems.

The damping strategy is the one used by the original OBPDS (which follows
Bank & Rose): after computing the undamped step, the residual norm ratio
``xi = ||G(u + s*du)|| / ||G(u)||`` decides whether the step is accepted.
If the reduction is smaller than ``tau * s`` the step is halved; otherwise
it is accepted and the damping factor is increased for the next iteration.
This makes the iteration robust for the very stiff exponential
``n(psi)``, ``p(psi)`` relations of semiconductor electrostatics.
"""

from __future__ import annotations

import logging
import os

import numpy as np
from scipy.sparse.linalg import spsolve

logger = logging.getLogger(__name__)

# Set OBPDS_NEWTON_TRACE=1 to print every iteration (diagnostics only).
TRACE = bool(os.environ.get("OBPDS_NEWTON_TRACE"))

__all__ = ["NewtonResult", "newton", "scalar_newton", "row_scale"]


def row_scale(main, upper, lower):
    """
    Row-wise infinity norms of a tridiagonal matrix.

    Semiconductor Jacobians span absurd dynamic ranges: in the continuity
    equations the neutral-region rows are ``O(mu kT n / h kT) ~ 1e27`` while
    the Dirichlet rows are ``O(1)``, so the matrix is *numerically* singular
    even though it is perfectly well conditioned after row equilibration.
    Newton's step is invariant under left multiplication, so dividing row
    ``i`` of both the Jacobian and the residual by this scale is a free
    preconditioner.
    """
    s = np.abs(np.asarray(main, dtype=float)).copy()
    if upper is not None and len(upper):
        s[:-1] = np.maximum(s[:-1], np.abs(upper))   # upper[i] sits on row i
    if lower is not None and len(lower):
        s[1:] = np.maximum(s[1:], np.abs(lower))     # lower[k] sits on row k+1
    return np.where(s > 0.0, s, 1.0)


class NewtonResult:
    def __init__(self, value, num_iter, converged):
        self.value = value
        self.num_iter = num_iter
        self.converged = converged


def newton(G, A, u0, rtol=0.0, atol=1e-8, gtol=1e-12, tau=0.5, max_iter=200,
           max_step=None, col_scale=None, weight=None):
    """
    Solve ``G(U) = 0`` for the vector ``U`` given the Jacobian ``A(U)``.

    Parameters
    ----------
    atol : float
        Absolute tolerance on the *step*.  Reaching it means the iterate has
        stopped moving.
    gtol : float
        Absolute tolerance on the *residual*, measured in the norm selected
        by ``weight``.  This is the criterion that actually fires for the
        continuity equations: there the step is ``D (J D)^-1 G`` with a
        column preconditioner ``D`` spanning 20-odd decades, so ``|du|``
        stays ~1e-3 even when the residual has hit the round-off floor.
        Judging convergence by the step alone would then report failure
        forever.
    weight : ndarray, optional
        Per-component multiplier used when measuring the residual.  Pass the
        row-equilibration factors here and ``gtol`` is then expressed in
        *physical* units rather than in the equilibrated ones.  That
        distinction decides whether the current is conserved: a hole row
        carries ``cp*|dp/dFp| ~ 6e6`` while the current being solved for may
        be ``1e-15`` A/cm^2, so a tolerance that looks brutally tight in
        equilibrated units still permits an imbalance a billion times the
        answer.  The weighting costs nothing -- it only changes which norm
        the residual is measured in, not the step that is taken.
    max_step : float, optional
        Trust region on the *undamped* step.  Semiconductor unknowns are
        energies in eV, and a wild first step sends the carrier densities to
        ``exp(1e6)``; limiting the step to a fraction of a volt keeps every
        evaluation finite without slowing down the final quadratic
        convergence (the step is already tiny by then).
    col_scale : ndarray, optional
        Diagonal preconditioner ``D``.  The linear system is solved for the
        scaled step ``duhat`` (so ``A`` must be handed over already
        multiplied by ``D``) and the step actually taken is ``D * duhat``.
        ``D (J D)^-1 = J^-1`` for any invertible diagonal ``D``, so this is
        exact -- it only changes how well conditioned the solve is.  The
        continuity equations need it: their columns carry the local
        ``dn/dF``, which ranges over 20 decades between a neutral region and
        a depleted one.
    """
    def measure(v):
        if weight is None:
            return float(np.linalg.norm(v))
        return float(np.abs(v * weight).max())

    sk = 1.0  # damping parameter
    uk = np.array(u0, dtype=float)
    D = (np.ones_like(uk) if col_scale is None
         else np.asarray(col_scale, dtype=float))
    Ak = A(uk)
    Gk = G(uk)
    Gk_norm = measure(Gk)
    if Gk_norm == 0.0:
        return NewtonResult(uk, 0, True)

    for k in range(1, max_iter + 1):
        if Gk_norm <= gtol:
            return NewtonResult(uk, k - 1, True)

        duk = D * spsolve(Ak.tocsc(), -Gk)
        if not np.all(np.isfinite(duk)):
            logger.warning("Newton step is not finite at iteration %d", k)
            return NewtonResult(uk, k, False)
        if max_step is not None:
            duk = np.clip(duk, -max_step, max_step)

        aerr = np.abs(duk).max()
        if aerr < atol:
            return NewtonResult(uk + duk, k, True)

        accepted = False
        for _ in range(60):
            ukp = uk + sk * duk
            Gkp = G(ukp)
            Gkp_norm = measure(Gkp)
            if not np.isfinite(Gkp_norm):
                sk /= 2.0
                continue
            xi = Gkp_norm / Gk_norm
            if TRACE and sk == 1.0:
                print(f"      trial sk=1: xi={xi:.4e} finite="
                      f"{np.isfinite(Gkp_norm)}")
            if 1.0 - xi < tau * sk:
                # Not enough reduction.  If the damped step has already
                # become negligible, further halving cannot help; report
                # failure immediately instead of burning 60 residual
                # evaluations.  (A genuine solution is caught by the gtol
                # test at the top of the loop, so this branch really is a
                # stall, not convergence.)
                if sk * aerr < atol:
                    return NewtonResult(uk, k, False)
                sk /= 2.0
                if sk < 1e-14:
                    return NewtonResult(uk, k, False)
                continue
            sk = sk / (sk + (1.0 - sk) * xi / 2.0)
            accepted = True
            break
        if not accepted:
            if TRACE:
                print(f"    newton[{k}] no step accepted, |G|={Gk_norm:.3e}")
            return NewtonResult(uk, k, False)

        if TRACE:
            print(f"    newton[{k}] |G| {Gk_norm:.4e} -> {Gkp_norm:.4e} "
                  f"xi={xi:.4e} |du|={aerr:.3e} @{int(np.argmax(np.abs(duk)))}"
                  f" sk={sk:.4f}")

        rerr = np.abs(duk / np.where(ukp == 0.0, 1.0, ukp)).max()
        if rerr < rtol:
            return NewtonResult(ukp, k, True)

        uk = ukp
        Ak = A(uk)
        Gk = Gkp
        Gk_norm = Gkp_norm

    logger.warning("Newton failed to converge after %d iterations", max_iter)
    return NewtonResult(uk, max_iter, False)


def scalar_newton(G, A, u0, rtol=0.0, atol=1e-8, tau=0.5, max_iter=200):
    """Scalar version of :func:`newton`."""
    sk = 1.0
    uk = float(u0)
    Ak = A(uk)
    Gk = G(uk)
    Gk_norm = abs(Gk)
    if Gk_norm == 0.0:
        return NewtonResult(uk, 0, True)

    for k in range(1, max_iter + 1):
        if Ak == 0.0:
            break
        duk = -Gk / Ak
        if abs(duk) < atol:
            return NewtonResult(uk + duk, k, True)
        accepted = False
        for _ in range(60):
            ukp = uk + sk * duk
            Gkp = G(ukp)
            Gkp_norm = abs(Gkp)
            if not np.isfinite(Gkp_norm):
                sk /= 2.0
                continue
            xi = Gkp_norm / Gk_norm
            if 1.0 - xi < tau * sk:
                sk /= 2.0
                if sk < 1e-12:
                    return NewtonResult(uk, k, False)
                continue
            sk = sk / (sk + (1.0 - sk) * xi / 2.0)
            accepted = True
            break
        if not accepted:
            return NewtonResult(uk, k, False)
        if abs(duk / ukp) < rtol:
            return NewtonResult(ukp, k, True)
        uk = ukp
        Ak = A(uk)
        Gk = Gkp
        Gk_norm = Gkp_norm

    return NewtonResult(uk, max_iter, False)
