"""
Steady-state drift-diffusion transport.

This is the capability the original OBPDS never had: it solved Poisson with
*prescribed* quasi-Fermi levels ("zero-current approximation") and therefore
could not produce a terminal current at all.  Here the electron and hole
continuity equations are solved as well, coupled to Poisson through a Gummel
iteration.

Formulation
-----------
The currents are written in the thermodynamic (driving-force) form

    J_n = q mu_n n dPhi_n^F / dx        J_p = q mu_p p dPhi_p^F / dx

with ``F_n``, ``F_p`` the quasi-Fermi levels in eV (numerically equal to the
quasi-Fermi potentials in volts).  Both carriers obey the same compact rule,
which is the whole point of the thermodynamic form.

Scharfetter-Gummel and the generalised potential
------------------------------------------------
Exponential fitting requires the density to be exponential in a *potential*.
Rather than assume Boltzmann statistics we define, at every node, the
**generalised potential**

    Psi_n = F_n - kT ln(n / Nc_ref)          n = Nc_ref exp((F_n - Psi_n)/kT)
    Psi_p = F_p + kT ln(p / Nv_ref)          p = Nv_ref exp((Psi_p - F_p)/kT)

which is an identity for *any* statistics -- parabolic Fermi-Dirac, Kane
non-parabolic, multi-valley -- and which reduces to

    Psi_n -> Ec - kT ln(Nc/Nc_ref)           Psi_p -> Ev + kT ln(Nv/Nv_ref)

in the non-degenerate limit.  Because it is built from the solved ``n`` and
``p`` it carries the electrostatic potential, every heterojunction band-edge
step, the ``Nc``/``Nv`` gradients and the degeneracy correction at once, so
there is no separate "band-gap narrowing" or "effective field" term to get
wrong.

With ``u = dPsi/kT`` the textbook discretisation is

    J_n[i+1/2] = (mu_n kT/h) [ n_{i+1} B(-u_n) - n_i B(u_n) ]
    J_p[i+1/2] = (mu_p kT/h) [ p_i      B(-u_p) - p_{i+1} B(u_p) ]

``B(x) = x/(e^x - 1)`` is the Bernoulli function.

Cancellation-free evaluation
----------------------------
Written as above, the two terms of ``J_n`` are each ``O(mu kT n / h)`` and
cancel to ``O(mu n dF/dx)`` -- at equilibrium they differ by ~1e-16 relative,
which for ``mu = 8500``, ``n = 1e17`` and ``h = 10 nm`` is a spurious
10^10 A/cm^2.  Substituting ``Psi_n = F_n - kT ln(n/Nc_ref)`` (so that
``u_n = w_n - ln(n_{i+1}/n_i)`` with ``w_n = dF_n/kT``) collapses the bracket
analytically to

    J_n = (mu_n kT/h) n_{i+1} B(-u_n) (1 - exp(-w_n))

and likewise ``J_p = (mu_p kT/h) p_i B(-u_p) (1 - exp(-w_p))``.  Now the
equilibrium limit ``w -> 0`` gives *exactly* zero, because the factor that
vanishes is computed from the quasi-Fermi level itself.  This is the form
actually used below.

Continuity (steady state, finite volume over cell width ``h``):

    J_n[i+1/2] - J_n[i-1/2] = +q R h
    J_p[i+1/2] - J_p[i-1/2] = -q R h

Recombination
-------------
Shockley-Read-Hall through mid-gap traps plus band-to-band radiative
recombination:

    R_srh = (n p - n0 p0) / (tau_p (n + n1) + tau_n (p + p1))
    R_rad = B (n p - n0 p0)

with ``n1 = p1 = sqrt(n0 p0)`` for a mid-gap trap.  Both the reference state
and the trap densities are subtle -- see :func:`recombination`.

Auger recombination is not included.

Boundary conditions
-------------------
Both quasi-Fermi levels are pinned to the metal Fermi level at each contact
(the ideal ohmic / infinite-surface-recombination limit):

    F_n = F_p = -V at the left contact, 0 at the right.

Mobility
--------
The mobility model is deliberately simple: the tabulated room-temperature
low-field mobility of the alloy, scaled as ``mu ~ (T/300)^-1.5`` (optical
phonon scattering).  There is no field-velocity saturation and no
doping-dependent (Caughey-Thomas) term -- those would require parameters
that no III-V compilation provides consistently.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import diags

from .fermi import KB_EV, bernoulli
from .newton import row_scale
from .solver import Q, Model, evaluate, solve_poisson, charge_neutral_guess, carriers

__all__ = ["TransportSolution", "solve_drift_diffusion", "iv_curve"]

MU_T_EXPONENT = 1.5
_DENS_FLOOR = 1.0e-300  # keeps ln(n) finite in deep depletion


class TransportSolution:
    """A converged drift-diffusion operating point."""

    def __init__(self, **kw):
        self.__dict__.update(kw)

    def as_dict(self, downsample=None):
        base = self.solution.as_dict(downsample)

        def pick(name):
            v = getattr(self, name, None)
            if v is None:
                return None
            v = np.asarray(v)
            if downsample and v.ndim == 1 and v.size > downsample:
                v = v[np.linspace(0, v.size - 1, downsample).astype(int)]
            return np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0).tolist()

        base.update({k: pick(k) for k in ("Jn", "Jp", "Jtotal", "R_srh", "R_rad")
                     if getattr(self, k, None) is not None})
        base["J"] = self.J
        base["J_spread"] = getattr(self, "J_spread", None)
        base["Jn_contact"] = self.Jn_contact
        base["Jp_contact"] = self.Jp_contact
        base["converged"] = self.converged
        base["gummel_iterations"] = self.gummel_iterations
        return base


def _mu(m):
    """Temperature-scaled mobilities."""
    s = (300.0 / m.T) ** MU_T_EXPONENT
    return m.mu_n * s, m.mu_p * s


def _effective_potentials(m, n, p, Fn, Fp, ref_c, ref_v):
    """
    Generalised SG driving potentials ``Psi_n`` and ``Psi_p`` (eV).

    Defined by the identities ``n = Nc_ref exp((Fn - Psi_n)/kT)`` and
    ``p = Nv_ref exp((Psi_p - Fp)/kT)``, so they are exact for arbitrary
    carrier statistics and automatically contain psi, the band-edge offsets,
    the DOS gradients and the degeneracy correction.  Reported for
    diagnostics; the fluxes themselves never form them explicitly (see
    :func:`_link_currents`).
    """
    kT = KB_EV * m.T
    Psi_n = Fn - kT * np.log(np.maximum(n, _DENS_FLOOR) / ref_c)
    Psi_p = Fp + kT * np.log(np.maximum(p, _DENS_FLOOR) / ref_v)
    return Psi_n, Psi_p


def recombination(m, n, p):
    """
    Net recombination rate (cm^-3 s^-1) and its partial derivatives.

    Shockley-Read-Hall through mid-gap traps plus radiative::

        R = (n p - n0 p0) / (tau_p (n + n1) + tau_n (p + p1))  +  B (n p - n0 p0)

    Two details decide whether this term is right, and both are easy to get
    wrong.

    *Reference.*  The driving force is ``n p - n0 p0``, the departure from
    *equilibrium*, not ``n p - ni^2``.  Under Fermi-Dirac or Kane statistics
    ``n p`` differs from ``ni^2`` by a few per cent even at equilibrium, and
    that residual would otherwise appear as a spurious generation rate at
    zero bias and a saturation current orders of magnitude too large.
    Referencing the equilibrium state makes ``R == 0`` at equilibrium
    exactly, as thermodynamics demands.

    *Trap densities.*  ``n1`` and ``p1`` are the densities that would hold
    if the Fermi level sat *at the trap*::

        n1 = n_eff exp((Et - Ei)/kT),   p1 = n_eff exp((Ei - Et)/kT)

    so for a mid-gap trap ``n1 = p1 = n_eff``, with ``n_eff = sqrt(n0 p0)``
    generalising ``ni`` to whatever statistics are in force.  They are *not*
    the local equilibrium densities ``n0``, ``p0``.  Substituting those is
    the tempting shortcut -- it also makes ``R`` vanish at equilibrium --
    but it is equivalent to letting the trap energy slide along with the
    Fermi level, and inside the depletion region it inflates the denominator
    by the *majority* density (~1e17 against ~1e6) and suppresses generation
    by two orders of magnitude.  The mistake hides well: in the neutral
    regions both ``p0`` and ``ni`` drop out against ``p`` and either choice
    returns the correct minority lifetime, so the forward knee looks fine --
    only the reverse characteristic comes out far too small and almost
    independent of bias.
    """
    from .solver import equilibrium_densities
    n0, p0 = equilibrium_densities(m)
    ref = n0 * p0
    n_eff = np.sqrt(ref)
    excess = n * p - ref
    den = np.maximum(m.tau_p * (n + n_eff) + m.tau_n * (p + n_eff), 1e-30)
    r_srh = excess / den
    r_rad = m.B_rad * excess
    d_r_srh_dn = (p * den - excess * m.tau_p) / den ** 2
    d_r_srh_dp = (n * den - excess * m.tau_n) / den ** 2
    return (r_srh + r_rad, r_srh, r_rad,
            d_r_srh_dn + m.B_rad * p, d_r_srh_dp + m.B_rad * n)


_W_CLIP = 300.0  # keeps exp(-w) finite during wild Newton excursions


def _link_currents(m, n, p, mu_n, mu_p, Fn, Fp):
    """
    Scharfetter-Gummel link fluxes (length N-1), in A/cm^2.

    Uses the cancellation-free form derived in the module docstring,

        J_n[k] = cn[k] n[k+1] B(-u_n[k]) (1 - exp(-w_n[k]))
        J_p[k] = cp[k] p[k]   B(-u_p[k]) (1 - exp(-w_p[k]))

    with ``w = dF/kT`` and ``u_n = w_n - dln(n)``, ``u_p = w_p + dln(p)``.

    Returns ``(Jn, Jp, Bn_pos, Bn_neg, Bp_neg, Bp_pos, cn, cp)``, where
    ``B*_pos = B(+u)`` and ``B*_neg = B(-u)``.
    """
    kT = KB_EV * m.T
    h = m.dx
    # J = q mu n dF/dx; the q converts the particle flux mu kT n / h into a
    # current density in A/cm^2, which is what the continuity equations
    # ( J[i] - J[i-1] = +- q R h ) expect.
    cn = Q * mu_n[:-1] * kT / h
    cp = Q * mu_p[:-1] * kT / h

    wn = np.clip((Fn[1:] - Fn[:-1]) / kT, -_W_CLIP, _W_CLIP)
    wp = np.clip((Fp[1:] - Fp[:-1]) / kT, -_W_CLIP, _W_CLIP)
    ln_n = np.log(np.maximum(n, _DENS_FLOOR))
    ln_p = np.log(np.maximum(p, _DENS_FLOOR))
    un = wn - (ln_n[1:] - ln_n[:-1])
    up = wp + (ln_p[1:] - ln_p[:-1])

    Bn_p, Bn_m = bernoulli(un), bernoulli(-un)
    Bp_p, Bp_m = bernoulli(up), bernoulli(-up)
    sn = -np.expm1(-wn)          # 1 - exp(-w_n)
    sp = -np.expm1(-wp)          # 1 - exp(-w_p)

    Jn = cn * n[1:] * Bn_m * sn
    Jp = cp * p[:-1] * Bp_m * sp
    return Jn, Jp, Bn_p, Bn_m, Bp_m, Bp_p, cn, cp


def _densities(m, psi, Fn, Fp, approx):
    """
    Carrier densities plus ``dn/dFn`` and ``dp/dFp``.

    Note the signs: raising ``Fn`` *adds* electrons (``dn/dFn > 0``) while
    raising ``Fp`` *removes* holes (``dp/dFp < 0``, since
    ``p ~ exp((Ev - Fp)/kT)``).  Both derivatives coincide numerically with
    ``dn/dpsi`` and ``dp/dpsi`` because ``F`` and ``psi`` enter the band-edge
    arguments with the same sign.
    """
    p, n, ng, nl, nx, dp, dn = carriers(m, psi, Fp, Fn, approx)
    return p, n, ng, nl, nx, dn, dp


F_LIMIT = 10.0  # eV; quasi-Fermi levels outside +-10 eV are unphysical

def _solve_continuity(m, psi, Fn, Fp, approx, electron, bc_left, bc_right,
                      mu_n, mu_p, ref_c, ref_v, max_iter=100):
    """
    One Newton solve for a single continuity equation.

    ``electron=True`` solves for ``Fn`` with ``Fp`` frozen, and vice versa.
    """
    N, h = m.N, m.dx
    scale = np.ones(N)  # row equilibration, refreshed by `jacobian`

    def state(F):
        Fn_ = F if electron else Fn
        Fp_ = Fp if electron else F
        p, n, *_rest, g_n, g_p = _densities(m, psi, Fn_, Fp_, approx)
        return p, n, g_n, g_p

    # Impose the contact boundary conditions on the starting iterate: they
    # are linear, so satisfying them up front costs nothing and stops the
    # (much larger, unscaled) Dirichlet residual from dominating the Newton
    # damping criterion while the interior is still far from converged.
    x0 = np.array(Fn if electron else Fp, dtype=float)
    x0[0] = bc_left
    x0[-1] = bc_right

    def residual(F):
        p, n, g_n, g_p = state(F)
        R, *_r, dRdn, dRdp = recombination(m, n, p)
        Fn_ = F if electron else Fn
        Fp_ = Fp if electron else F
        Jn, Jp, *_ = _link_currents(m, n, p, mu_n, mu_p, Fn_, Fp_)
        res = np.empty(N)
        if electron:
            res[1:-1] = Jn[1:] - Jn[:-1] - Q * R[1:-1] * h
        else:
            res[1:-1] = Jp[1:] - Jp[:-1] + Q * R[1:-1] * h
        res[0] = F[0] - bc_left
        res[-1] = F[-1] - bc_right
        return res / scale

    def jacobian(F):
        p, n, g_n, g_p = state(F)
        _R, *_r, dRdn, dRdp = recombination(m, n, p)
        Fn_ = F if electron else Fn
        Fp_ = Fp if electron else F
        _Jn, _Jp, Bn_p, Bn_m, Bp_m, Bp_p, cn, cp = _link_currents(
            m, n, p, mu_n, mu_p, Fn_, Fp_)

        main = np.zeros(N)
        up = np.zeros(N - 1)
        lo = np.zeros(N - 1)
        # Row equilibration alone is enough here.  Every entry of row i
        # carries the local dn/dF (or dp/dF) of the node in its own column,
        # and that factor varies smoothly along the grid, so after dividing
        # row i by its own infinity norm the tridiagonal is O(1) throughout.
        # (An earlier version flattened the column factors to a uniform
        # +-g_ref and undid that with a diagonal preconditioner on the step.
        # It conditioned the matrix just as well but wrecked the residual
        # normalisation: the p-side row then carried the n-side's dn/dF,
        # 1e13 times too large, and a 5e-7 A/cm^2 current imbalance looked
        # like 2e-15 -- the solver declared convergence with the minority
        # carrier profile completely flat and the current missing.)
        g = g_n if electron else g_p
        g_ref = max(float(np.max(np.abs(g))), 1e-300)
        # Keep the sign and floor the magnitude: in deep depletion dn/dF is
        # genuinely ~0, and an all-zero row would make the matrix singular.
        g = (1.0 if electron else -1.0) * np.maximum(np.abs(g), g_ref * 1e-30)
        if electron:
            # Jn[k] = cn[k] n[k+1] B(-u_k) (1 - exp(-w_k)),  w = dFn/kT.
            # Differentiating with B(-u) lagged (exact for Boltzmann) gives
            #   dJn[k]/dFn[k]   = -cn[k] g[k]   B(+u_k)
            #   dJn[k]/dFn[k+1] = +cn[k] g[k+1] B(-u_k)
            # so row i of the Jacobian of  Jn[i] - Jn[i-1]  is
            #   main = -cn[i] B(+u_i) - cn[i-1] B(-u_{i-1})
            #   up   = +cn[i] B(-u_i)              (i -> i+1)
            #   lo   = +cn[i-1] B(+u_{i-1})        (i -> i-1)
            # The off-diagonals have the *same* sign here; the row sums to
            # zero for a uniform u, as a rigid shift of Fn must produce.
            dR = dRdn
            main[1:-1] = (-cn[1:] * Bn_p[1:] - cn[:-1] * Bn_m[:-1]
                          - Q * dR[1:-1] * h) * g[1:-1]
            up[1:] = cn[1:] * Bn_m[1:] * g[2:]
            lo[:-1] = cn[:-1] * Bn_p[:-1] * g[:-2]
        else:
            # Jp[k] = cp[k] p[k] B(-u_k) (1 - exp(-w_k)),  w = dFp/kT.
            #   dJp[k]/dFp[k]   = +cp[k] g[k]   B(-u_k)
            #   dJp[k]/dFp[k+1] = -cp[k] g[k+1] B(+u_k)
            #   main = +cp[i] B(-u_i) + cp[i-1] B(+u_{i-1})
            #   up   = -cp[i] B(+u_i)              (i -> i+1)
            #   lo   = -cp[i-1] B(-u_{i-1})        (i -> i-1)
            dR = dRdp
            main[1:-1] = (cp[1:] * Bp_m[1:] + cp[:-1] * Bp_p[:-1]
                          + Q * dR[1:-1] * h) * g[1:-1]
            up[1:] = -cp[1:] * Bp_p[1:] * g[2:]
            lo[:-1] = -cp[:-1] * Bp_m[:-1] * g[:-2]
        main[0] = 1.0
        main[-1] = 1.0
        up[0] = 0.0
        lo[-1] = 0.0
        scale[:] = row_scale(main, up, lo)
        return diags([lo / scale[1:], main / scale, up / scale[:-1]],
                     [-1, 0, 1], shape=(N, N), format="csc")

    from .newton import newton
    jacobian(x0)  # establish the row scaling before the first residual
    # The residual is measured after row equilibration, so `gtol` is *not* in
    # amperes.  It is tight because a hole row carries cp*|dp/dFp| ~ 6e6
    # while the physical current is ~1e-6 A/cm^2: a "relative" residual of
    # 1e-10 would still allow a 6e-4 A/cm^2 imbalance, six hundred times the
    # quantity being solved for.  1e-16 keeps the imbalance four decades
    # below the current and is still above the round-off floor.
    #
    # It cannot go further.  A neutral region carries current as q mu c dF/dx,
    # so with c ~ 1e17 the gradient belonging to a 1e-15 A/cm^2 current is
    # ~1e-22 eV per cell, five orders below the round-off of a stored level
    # of order 0.1 eV; the resulting spurious flux is c mu eps(F)/kT ~ 1e-10
    # A/cm^2 and pops out as isolated single-link spikes.  Those are removed
    # when the terminal current is read off (see `solve_drift_diffusion`),
    # not here.
    res = newton(residual, jacobian, x0, rtol=0.0, atol=1e-11,
                 gtol=1e-16, max_iter=max_iter, max_step=0.25)
    out = np.clip(res.value, -F_LIMIT, F_LIMIT)
    out[0] = bc_left
    out[-1] = bc_right
    return out, res.converged


def _majority_fermi(m, V):
    """
    Starting quasi-Fermi levels for the Gummel loop: the analytic
    Shockley minority-carrier profile.

    In a quasi-neutral region the minority density obeys
    ``D c'' = (c - c0)/tau``, and since the region is thin compared with the
    diffusion length the profile is very nearly *linear* between the
    injected value at the depletion edge, ``c0 exp(V/kT)``, and the
    equilibrium value ``c0`` at the contact.  With ``s`` the fractional
    distance from the junction (``s = 0``) to the contact (``s = 1``),

        c(s) = c0 ( exp(V/kT) (1 - s) + s )

    Writing ``c ~ exp(+-F/kT)`` turns that straight line into a
    *logarithmic* quasi-Fermi level:

        F(s) = F_majority + kT ln( exp(V/kT) (1 - s) + s )

    which is exact for *either* sign of V.  Under reverse bias the minority
    density is pulled *below* its equilibrium value at the depletion edge
    (``c -> c0 exp(V/kT) -> 0``) rather than pushed above it, and the
    logarithm stays finite because ``exp(V/kT)`` is positive however
    negative V gets.  (A formula built on ``1 - exp(-|V|/kT)`` instead --
    the first thing one writes down -- diverges to ``-inf`` as soon as V
    changes sign, which is what made every reverse-bias point stall.)

    Each level is therefore flat across the region where its carrier is the
    majority (and across the depletion region) and does almost all of its
    bending in the last few cells next to its own contact.

    Getting this shape right matters far more than it looks.  The current
    carried by a quasi-Fermi gradient is ``q mu c dF/dx``, so a *linear*
    ramp -- the obvious guess -- sits ~0.13 V away from the answer in the
    middle of the neutral region.  The continuity Jacobian there is a
    near-Laplacian whose row sum is only ``-du``, so that ~1e-4 relative
    residual is amplified into a ~1e2 V Newton step along a near-null
    direction; the trust region then clips it and the solve is destroyed.
    With the analytic profile the initial residual is at the round-off floor
    and the Newton step stays small.
    """
    kT = KB_EV * m.T
    Fl, Fr = -float(V), 0.0
    N = m.N
    idx = np.arange(N)
    nz = np.nonzero(m.Nnet)[0]
    ptype = m.Nnet < 0.0
    ntype = m.Nnet > 0.0
    p_left = bool(m.Nnet[nz[0]] < 0.0) if nz.size else True

    def frac(away_from_left):
        """Fractional distance from the junction to one end of the stack."""
        if away_from_left:
            return np.clip(idx / max(N - 1, 1), 0.0, 1.0)
        return np.clip((N - 1 - idx) / max(N - 1, 1), 0.0, 1.0)

    if ptype.any() and ntype.any():
        # The metallurgical junction is where the net doping *changes sign*,
        # not where it first becomes n-type: with the n layer on the left
        # ``argmax(ntype)`` is 0 and the "junction" lands on the contact,
        # bending every profile across the wrong half of the stack.
        s0 = float(np.sign(m.Nnet[nz[0]]))
        opp = np.nonzero(np.sign(m.Nnet) == -s0)[0]
        j = float(opp[0]) - 0.5 if opp.size else 0.5 * (N - 1)
        to_left = np.clip((j - idx) / max(j, 1.0), 0.0, 1.0)
        to_right = np.clip((idx - j) / max(N - 1 - j, 1.0), 0.0, 1.0)
        # distance from the junction towards the p contact / the n contact
        s_p = to_left if p_left else to_right
        s_n = to_right if p_left else to_left
    else:
        s_p = frac(False) if p_left else frac(True)
        s_n = frac(True) if p_left else frac(False)

    Fp_c = Fl if p_left else Fr      # hole level at the p-side contact
    Fn_c = Fr if p_left else Fl      # electron level at the n-side contact
    # Signed junction voltage in units of kT.  Both levels are flat through
    # the depletion region, so their splitting there is exactly the applied
    # bias, and its sign is what decides whether the minority carriers are
    # injected (forward) or extracted (reverse).
    A = (Fn_c - Fp_c) / kT

    def shape(s):
        """G(s): 1 at the junction, 0 at the contact; the log profile."""
        if abs(A) <= 1e-12:
            return 1.0 - s
        # exp(A) (1-s) + s is >= min(exp(A), 1) > 0, so the log cannot
        # blow up; the floor only guards against exp(A) underflowing for
        # absurd reverse biases (|V| > ~15 V at 300 K).
        arg = np.maximum(np.exp(A) * (1.0 - s) + s, 1e-300)
        return np.log(arg) / A

    # Each level equals *its own* contact value at the junction (it is flat
    # across the depletion region) and merges into the other carrier's level
    # only at the far contact, where the minority density is back to c0.
    Fn = Fp_c + (Fn_c - Fp_c) * shape(s_p)
    Fp = Fn_c + (Fp_c - Fn_c) * shape(s_n)
    Fn[0], Fn[-1] = Fl, Fr
    Fp[0], Fp[-1] = Fl, Fr
    return Fn, Fp


def _gummel(m, V, approx, max_gummel, init, mu_n, mu_p, ref_c, ref_v, tol):
    """One bias point of the Gummel loop; returns ``(psi, Fn, Fp, ok, iters)``."""
    N = m.N
    Vl, Vr = float(V), 0.0
    fl, fr = -Vl, -Vr
    if init is None:
        Fn, Fp = _majority_fermi(m, Vl)
        # Seed psi from local charge neutrality *at the biased quasi-Fermi
        # levels*, not from the equilibrium profile.  Shifting a neutral
        # region's quasi-Fermi level by dF shifts its electrostatic
        # potential by -dF (that is what keeps p and n at their equilibrium
        # values), so the equilibrium profile is a full V away from the
        # answer; handing it to the nonlinear Poisson makes that solver
        # absorb a V/kT ~ 12 kT jump in one go, and it stalls after two
        # iterations.  Re-centring first makes the first Poisson solve
        # nearly exact in the neutral regions.
        psi = charge_neutral_guess(m, Fp, Fn, approx)
    else:
        psi, Fn, Fp = (np.array(v, dtype=float) for v in init)

    converged = False
    iters = 0
    for it in range(1, max_gummel + 1):
        iters = it
        # ---- 1. nonlinear Poisson with the quasi-Fermi levels frozen -----
        psi_new, pres = solve_poisson(m, psi, Fp, Fn, Vl, Vr, approx)
        # ---- 2. electron continuity ---------------------------------------
        Fn_new, c1 = _solve_continuity(m, psi_new, Fn, Fp, approx, True,
                                       fl, fr, mu_n, mu_p, ref_c, ref_v)
        # ---- 3. hole continuity -------------------------------------------
        Fp_new, c2 = _solve_continuity(m, psi_new, Fn_new, Fp, approx, False,
                                       fl, fr, mu_n, mu_p, ref_c, ref_v)

        dpsi = np.abs(psi_new - psi).max()
        dfn = np.abs(Fn_new - Fn).max()
        dfp = np.abs(Fp_new - Fp).max()
        psi, Fn, Fp = psi_new, Fn_new, Fp_new
        if max(dpsi, dfn, dfp) < tol and pres.converged and c1 and c2:
            converged = True
            break
    return psi, Fn, Fp, converged, iters


def solve_drift_diffusion(m: Model, V: float, approx="kane",
                          max_gummel=200, init=None, tol=1e-7):
    """
    Gummel iteration to a self-consistent drift-diffusion solution.

    Parameters
    ----------
    m : Model
    V : float
        Bias on the left/top contact (V); the right/bottom contact is at 0.
    init : tuple, optional
        ``(psi, Fn, Fp)`` from a previous bias point.  Rarely useful and
        easy to get wrong, so it is off by default -- see the note below.
    """
    N = m.N
    mu_n, mu_p = _mu(m)
    ref_c, ref_v = float(m.Nc.max()), float(m.Nv.max())
    Vl = float(V)

    # No bias continuation.  Walking up from equilibrium *looks* like the
    # safe thing to do, but the previous operating point carries the
    # previous quasi-Fermi levels, and an ohmic contact pins the *densities*,
    # not the levels: reusing them at a new bias leaves the contact carrier
    # densities off by exp(dV/kT) and the first nonlinear Poisson solve
    # explodes.  Shifting the levels properly means re-deriving the whole
    # majority/minority profile anyway, which is what `_majority_fermi` does
    # analytically -- and with it the Gummel loop converges in two or three
    # iterations from cold, so there is nothing for continuation to save.
    psi, Fn, Fp, converged, iters = _gummel(
        m, Vl, approx, max_gummel, init, mu_n, mu_p, ref_c, ref_v, tol)

    p, n, ng, nl, nx, *_ = carriers(m, psi, Fp, Fn, approx)
    R, r_srh, r_rad, *_ = recombination(m, n, p)
    Jn, Jp, *_ = _link_currents(m, n, p, mu_n, mu_p, Fn, Fp)
    Jtot = Jn + Jp

    # Terminal current.  ``Jn + Jp`` is constant in exact arithmetic, but its
    # discrete value is not equally trustworthy everywhere: a *majority*
    # carrier in a quasi-neutral region carries the current as q mu c dF/dx
    # with c ~ 1e17, so the gradient belonging to a 1e-15 A/cm^2 current is
    # ~1e-22 eV per cell, five orders below the round-off of a stored level
    # of order 0.1 eV.  Those links report pure noise.  Inside the depletion
    # region both carriers are minorities, both densities are near ni and both
    # fluxes are clean, so read the current off there -- the constant is the
    # same, only the arithmetic is better.  `J_spread` reports how far the
    # worst link strays, which is the honest measure of how much of the
    # number is signal.
    both = np.maximum(n, p)[:-1]
    mask = both <= 1e3 * float(m.ni.max())
    J = float(np.median(Jtot[mask])) if mask.any() else float(np.median(Jtot))
    spread = float(np.abs(Jtot - J).max() / max(abs(J), 1e-300))

    sol = evaluate(m, psi, Fp, Fn, V, approx)
    return TransportSolution(
        solution=sol, psi=psi, Fn=Fn, Fp=Fp, n=n, p=p,
        Jn=Jn, Jp=Jp, Jtotal=Jtot,
        J=J, J_spread=spread,
        Jn_contact=float(Jn[0]), Jp_contact=float(Jp[0]),
        R_srh=r_srh, R_rad=r_rad, R=R,
        converged=converged, gummel_iterations=iters,
        V=V, T=m.T,
    )


def iv_curve(device, Vstart, Vstop, Vnum, T, N, approx):
    """
    Bias sweep.  Every point is solved from the analytic cold start
    (see :func:`solve_drift_diffusion`), so the points are independent and
    the sweep is trivially parallelisable.

    Returns ``(V, J)`` with ``J`` in A/cm^2.
    """
    m = device.model(T, N)
    Vs = np.linspace(Vstart, Vstop, int(Vnum))
    Js = np.empty(Vs.size)
    for i, V in enumerate(Vs):
        try:
            s = solve_drift_diffusion(m, float(V), approx)
        except Exception:
            Js[i] = np.nan
            continue
        Js[i] = s.J
    return Vs, Js
