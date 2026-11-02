"""
III-V zinc-blende semiconductor parameter tables.

Primary source
--------------
I. Vurgaftman, J. R. Meyer and L. R. Ram-Mohan, "Band parameters for III-V
compound semiconductors and their alloys", *J. Appl. Phys.* **89**, 5815
(2001).  Static dielectric constants are from the Adachi reviews
(Adachi, *J. Appl. Phys.* **53**, 8775 (1982) and Adachi, *J. Appl. Phys.*
**R1**, 1987).  The tables were transcribed from the ``openbandparams``
project (Scott J. Maddox, AGPL-3.0), which is itself a faithful
implementation of the same references.

Energy scale
------------
``VBO`` values follow Vurgaftman's table and are referenced to the valence
band maximum of InSb (VBO(InSb) = 0).  The vacuum level sits at
``VACUUM_LEVEL = 4.66 eV`` above the ``CBO = 0`` reference, which reproduces
the accepted electron affinities (GaAs: 4.66 - 0.62 = 4.04 eV).

Elastic constants and deformation potentials
--------------------------------------------
``C11`` / ``C12`` are in units of 10^11 dyn/cm^2 (= 10 GPa); ``a_c``,
``a_v`` and ``b_shear`` are in eV.  They come from Van de Walle's model-solid
compilation (Phys. Rev. B **39**, 1871 (1989)) as reproduced by Chuang,
*Physics of Optoelectronic Devices*, Appendix K, cross-checked against
Vurgaftman's table for the antimonides.

Sign convention (Van de Walle / Chuang, *not* the one Vurgaftman prints for
``a_v``): a hydrostatic strain ``Tr(eps)`` moves the edges by

    dE_c = a_c * Tr(eps),      dE_v = a_v * Tr(eps),
    dE_g = (a_c - a_v) * Tr(eps).

With ``a_c < 0`` and ``a_v > 0`` this gives ``dE_g/dP = -(a_c - a_v)/B > 0``:
GaAs, -(-8.33)/784 kbar = 10.6 meV/kbar against a measured 10.8.  Vurgaftman's
table prints ``a_v`` with the opposite sign and quotes the gap shift as
``(a_c + a_v)``; the two conventions agree numerically.  The nitrides carry
elastic constants but no deformation potentials (they are wurtzite in
practice, so the zinc-blende values are not trustworthy); strain is skipped
for them.

Mobility
--------
``mu_n`` / ``mu_p`` are room-temperature low-field drift mobilities in
cm^2 V^-1 s^-1.  These are *not* part of Vurgaftman's compilation; they are
order-of-magnitude reference values collected from standard tabulations
(Sze, and the Ioffe Institute semiconductor database) and are only used as
the default of the (deliberately simple) mobility model in
:mod:`obpds.transport`.  They can be overridden per layer.
"""

from __future__ import annotations

VACUUM_LEVEL = 4.66  # eV, above the CBO = 0 reference


BINARIES = {
    # ------------------------------------------------------------------ N
    "AlN": dict(elements=("Al", "N"), Delta_SO=0.019, Eg_Gamma_0=4.9, Eg_L_0=9.3,
                Eg_X_0=6.0, Ep=27.1, VBO=-3.44, a_300K=4.38, thermal_expansion=0.0,
                alpha_Gamma=5.93e-4, beta_Gamma=600., alpha_L=5.93e-4, beta_L=600.,
                alpha_X=5.93e-4, beta_X=600., luttinger1=1.92, luttinger2=0.47,
                luttinger3=0.85, meff_e_Gamma_0=0.25,
                meff_e_L_long=None, meff_e_L_trans=None, meff_e_L_DOS=None,
                meff_e_X_long=0.53, meff_e_X_trans=0.31, meff_e_X_DOS=None,
                C11=30.4, C12=16.0, a_c=None, a_v=None, b_shear=None,
                dielectric=8.5, mu_n=300., mu_p=50.),
    "GaN": dict(elements=("Ga", "N"), Delta_SO=0.017, Eg_Gamma_0=3.299, Eg_L_0=5.59,
                Eg_X_0=4.52, Ep=25.0, VBO=-2.64, a_300K=4.5, thermal_expansion=0.0,
                alpha_Gamma=5.93e-4, beta_Gamma=600., alpha_L=5.93e-4, beta_L=600.,
                alpha_X=5.93e-4, beta_X=600., luttinger1=2.67, luttinger2=0.75,
                luttinger3=1.1, meff_e_Gamma_0=0.15,
                meff_e_L_long=None, meff_e_L_trans=None, meff_e_L_DOS=None,
                meff_e_X_long=0.5, meff_e_X_trans=0.3, meff_e_X_DOS=None,
                C11=29.3, C12=15.9, a_c=None, a_v=None, b_shear=None,
                dielectric=8.9, mu_n=1200., mu_p=200.),
    "InN": dict(elements=("In", "N"), Delta_SO=0.006, Eg_Gamma_0=1.94, Eg_L_0=5.82,
                Eg_X_0=2.51, Ep=25.0, VBO=-2.38, a_300K=4.98, thermal_expansion=0.0,
                alpha_Gamma=2.45e-4, beta_Gamma=624., alpha_L=2.45e-4, beta_L=624.,
                alpha_X=2.45e-4, beta_X=624., luttinger1=3.72, luttinger2=1.26,
                luttinger3=1.63, meff_e_Gamma_0=0.12,
                meff_e_L_long=None, meff_e_L_trans=None, meff_e_L_DOS=None,
                meff_e_X_long=0.48, meff_e_X_trans=0.27, meff_e_X_DOS=None,
                C11=18.7, C12=12.5, a_c=None, a_v=None, b_shear=None,
                dielectric=15.3, mu_n=3000., mu_p=100.),
    # ------------------------------------------------------------------ P
    "AlP": dict(elements=("Al", "P"), Delta_SO=0.07, Eg_Gamma_0=3.63, Eg_L_0=3.57,
                Eg_X_0=2.52, Ep=17.7, VBO=-1.74, a_300K=5.4672,
                thermal_expansion=2.92e-5,
                alpha_Gamma=5.771e-4, beta_Gamma=372., alpha_L=3.18e-4, beta_L=588.,
                alpha_X=3.18e-4, beta_X=588., luttinger1=3.35, luttinger2=0.71,
                luttinger3=1.23, meff_e_Gamma_0=0.22,
                meff_e_L_long=None, meff_e_L_trans=None, meff_e_L_DOS=None,
                meff_e_X_long=2.68, meff_e_X_trans=0.155, meff_e_X_DOS=None,
                C11=13.3, C12=6.3, a_c=-5.7, a_v=1.5, b_shear=-1.5,
                dielectric=9.8, mu_n=80., mu_p=50.),
    "GaP": dict(elements=("Ga", "P"), Delta_SO=0.08, Eg_Gamma_0=2.886, Eg_L_0=2.72,
                Eg_X_0=2.35, Ep=31.4, VBO=-1.27, a_300K=5.4505,
                thermal_expansion=2.92e-5,
                alpha_Gamma=None, beta_Gamma=None,  # Eg_Gamma uses a special form
                alpha_L=5.771e-4, beta_L=372., alpha_X=5.771e-4, beta_X=372.,
                luttinger1=4.05, luttinger2=0.49, luttinger3=2.93,
                meff_e_Gamma_0=0.13,
                meff_e_L_long=1.2, meff_e_L_trans=0.15, meff_e_L_DOS=None,
                meff_e_X_long=2.0, meff_e_X_trans=0.253, meff_e_X_DOS=None,
                C11=14.05, C12=6.203, a_c=-7.14, a_v=1.7, b_shear=-1.8,
                dielectric=11.0, mu_n=250., mu_p=150.),
    "InP": dict(elements=("In", "P"), Delta_SO=0.108, Eg_Gamma_0=1.4236, Eg_L_0=2.014,
                Eg_X_0=2.384, Ep=20.7, VBO=-0.94, a_300K=5.8697,
                thermal_expansion=2.79e-5,
                alpha_Gamma=3.63e-4, beta_Gamma=162., alpha_L=3.63e-4, beta_L=162.,
                alpha_X=3.7e-4, beta_X=0., luttinger1=5.08, luttinger2=1.6,
                luttinger3=2.1, meff_e_Gamma_0=0.0795,
                meff_e_L_long=None, meff_e_L_trans=None, meff_e_L_DOS=0.47,
                meff_e_X_long=None, meff_e_X_trans=None, meff_e_X_DOS=0.88,
                C11=10.11, C12=5.61, a_c=-5.04, a_v=1.27, b_shear=-1.7,
                dielectric=12.3, mu_n=5400., mu_p=200.),
    # ----------------------------------------------------------------- As
    "AlAs": dict(elements=("Al", "As"), Delta_SO=0.28, Eg_Gamma_0=3.099, Eg_L_0=2.46,
                 Eg_X_0=2.24, Ep=21.1, VBO=-1.33, a_300K=5.6611,
                 thermal_expansion=2.9e-5,
                 alpha_Gamma=8.85e-4, beta_Gamma=530., alpha_L=6.05e-4, beta_L=204.,
                 alpha_X=7e-4, beta_X=530., luttinger1=3.76, luttinger2=0.82,
                 luttinger3=1.42, meff_e_Gamma_0=0.15,
                 meff_e_L_long=1.32, meff_e_L_trans=0.15, meff_e_L_DOS=None,
                 meff_e_X_long=0.97, meff_e_X_trans=0.22, meff_e_X_DOS=None,
                 C11=12.5, C12=5.34, a_c=-5.64, a_v=2.47, b_shear=-1.5,
                 dielectric=10.0, mu_n=280., mu_p=100.),
    "GaAs": dict(elements=("Ga", "As"), Delta_SO=0.341, Eg_Gamma_0=1.519, Eg_L_0=1.815,
                 Eg_X_0=1.981, Ep=28.8, VBO=-0.8, a_300K=5.65325,
                 thermal_expansion=3.88e-5,
                 alpha_Gamma=5.405e-4, beta_Gamma=204., alpha_L=6.05e-4, beta_L=204.,
                 alpha_X=4.6e-4, beta_X=204., luttinger1=6.98, luttinger2=2.06,
                 luttinger3=2.93, meff_e_Gamma_0=0.067,
                 meff_e_L_long=1.9, meff_e_L_trans=0.0754, meff_e_L_DOS=None,
                 meff_e_X_long=1.3, meff_e_X_trans=0.23, meff_e_X_DOS=None,
                 C11=11.879, C12=5.376, a_c=-7.17, a_v=1.16, b_shear=-1.7,
                 dielectric=13.0, mu_n=8500., mu_p=400.),
    "InAs": dict(elements=("In", "As"), Delta_SO=0.39, Eg_Gamma_0=0.417, Eg_L_0=1.133,
                 Eg_X_0=1.433, Ep=21.5, VBO=-0.59, a_300K=6.0583,
                 thermal_expansion=2.74e-5,
                 alpha_Gamma=2.76e-4, beta_Gamma=93., alpha_L=2.76e-4, beta_L=93.,
                 alpha_X=2.76e-4, beta_X=93., luttinger1=20.0, luttinger2=8.5,
                 luttinger3=9.2, meff_e_Gamma_0=0.026,
                 meff_e_L_long=0.64, meff_e_L_trans=0.05, meff_e_L_DOS=None,
                 meff_e_X_long=1.13, meff_e_X_trans=0.16, meff_e_X_DOS=None,
                 C11=8.329, C12=4.526, a_c=-5.08, a_v=1.0, b_shear=-1.8,
                 dielectric=14.5, mu_n=20000., mu_p=350.),
    # ----------------------------------------------------------------- Sb
    "AlSb": dict(elements=("Al", "Sb"), Delta_SO=0.676, Eg_Gamma_0=2.386,
                 Eg_L_0=2.329, Eg_X_0=1.696, Ep=18.7, VBO=-0.41, a_300K=6.1355,
                 thermal_expansion=2.6e-5,
                 alpha_Gamma=4.2e-4, beta_Gamma=140., alpha_L=3.9e-4, beta_L=140.,
                 alpha_X=5.8e-4, beta_X=140., luttinger1=5.18, luttinger2=1.19,
                 luttinger3=1.97, meff_e_Gamma_0=0.14,
                 meff_e_L_long=1.64, meff_e_L_trans=0.23, meff_e_L_DOS=None,
                 meff_e_X_long=1.357, meff_e_X_trans=0.123, meff_e_X_DOS=None,
                 C11=8.769, C12=4.341, a_c=-4.5, a_v=1.4, b_shear=-1.35,
                 dielectric=11.5, mu_n=200., mu_p=300.),
    "GaSb": dict(elements=("Ga", "Sb"), Delta_SO=0.76, Eg_Gamma_0=0.812,
                 Eg_L_0=0.875, Eg_X_0=1.141, Ep=27.0, VBO=-0.03, a_300K=6.0959,
                 thermal_expansion=4.72e-5,
                 alpha_Gamma=4.17e-4, beta_Gamma=140., alpha_L=5.97e-4, beta_L=140.,
                 alpha_X=4.75e-4, beta_X=94., luttinger1=13.4, luttinger2=4.7,
                 luttinger3=6.0, meff_e_Gamma_0=0.039,
                 meff_e_L_long=1.3, meff_e_L_trans=0.1, meff_e_L_DOS=None,
                 meff_e_X_long=1.51, meff_e_X_trans=0.22, meff_e_X_DOS=None,
                 C11=8.842, C12=4.026, a_c=-7.5, a_v=0.79, b_shear=-2.0,
                 dielectric=15.7, mu_n=3000., mu_p=1000.),
    "InSb": dict(elements=("In", "Sb"), Delta_SO=0.81, Eg_Gamma_0=0.235,
                 Eg_L_0=0.991, Eg_X_0=1.691, Ep=23.3, VBO=0.0, a_300K=6.4794,
                 thermal_expansion=3.48e-5,
                 alpha_Gamma=3.2e-4, beta_Gamma=170., alpha_L=3.2e-4, beta_L=170.,
                 alpha_X=3.2e-4, beta_X=170., luttinger1=34.8, luttinger2=15.5,
                 luttinger3=16.5, meff_e_Gamma_0=0.0135,
                 meff_e_L_long=None, meff_e_L_trans=None, meff_e_L_DOS=0.25,
                 meff_e_X_long=None, meff_e_X_trans=None, meff_e_X_DOS=0.5,
                 C11=6.847, C12=3.735, a_c=-6.94, a_v=0.36, b_shear=-2.0,
                 dielectric=17.8, mu_n=77000., mu_p=850.),
}


def _eg_gamma_gap(T):
    """GaP Gamma gap: Vurgaftman's dedicated temperature dependence."""
    import math
    return 0.1081 * (1.0 - 1.0 / math.tanh(164.0 / T))


TERNARIES = {
    "AlGaN": dict(binaries=("AlN", "GaN"), x_label="Al",
                  bowing=dict(Eg_Gamma=0.76, Eg_X=0.3)),
    "AlInN": dict(binaries=("AlN", "InN"), x_label="Al",
                  bowing=dict(Eg_Gamma=2.73, Eg_X=3.62)),
    "GaInN": dict(binaries=("GaN", "InN"), x_label="Ga",
                  bowing=dict(Eg_Gamma=1.38, Eg_X=1.67)),
    "AlGaP": dict(binaries=("AlP", "GaP"), x_label="Al", bowing=dict(Eg_X=0.13)),
    "AlInP": dict(binaries=("AlP", "InP"), x_label="Al",
                  bowing=dict(Delta_SO=-0.48, Eg_Gamma=0.38, Eg_X=-0.19)),
    "GaInP": dict(binaries=("GaP", "InP"), x_label="Ga",
                  bowing=dict(Eg_Gamma=0.65, Eg_L=1.03, Eg_X=0.2, F=0.78)),
    "AlGaAs": dict(binaries=("AlAs", "GaAs"), x_label="Al",
                   bowing=dict(Eg_X=0.055),
                   bowing_fn=dict(Eg_Gamma=lambda x: -0.127 + 1.310 * x)),
    "AlInAs": dict(binaries=("AlAs", "InAs"), x_label="Al",
                   bowing=dict(Delta_SO=0.15, Eg_Gamma=0.7, Ep=-4.81, F=-4.44,
                               VBO=-0.64)),
    "GaInAs": dict(binaries=("GaAs", "InAs"), x_label="Ga",
                   bowing=dict(Delta_SO=0.15, Eg_Gamma=0.477, Eg_L=0.33,
                               Eg_X=1.4, Ep=-1.48, F=1.77, VBO=-0.38)),
    "AlGaSb": dict(binaries=("AlSb", "GaSb"), x_label="Al",
                   bowing=dict(Delta_SO=0.3),
                   bowing_fn=dict(Eg_Gamma=lambda x: -0.044 + 1.22 * x)),
    "AlInSb": dict(binaries=("AlSb", "InSb"), x_label="Al",
                   bowing=dict(Delta_SO=0.25, Eg_Gamma=0.43)),
    "GaInSb": dict(binaries=("GaSb", "InSb"), x_label="Ga",
                   bowing=dict(Delta_SO=0.1, Eg_Gamma=0.415, Eg_L=0.4, Eg_X=0.33,
                               F=-6.84)),
    "AlPAs": dict(binaries=("AlP", "AlAs"), x_label="P",
                  bowing=dict(Eg_Gamma=0.22, Eg_L=0.22, Eg_X=0.22)),
    "GaPAs": dict(binaries=("GaP", "GaAs"), x_label="P",
                  bowing=dict(Eg_Gamma=0.19, Eg_L=0.16, Eg_X=0.24)),
    "InPAs": dict(binaries=("InP", "InAs"), x_label="P",
                  bowing=dict(Delta_SO=0.16, Eg_Gamma=0.10, Eg_L=0.27, Eg_X=0.27)),
    "AlAsSb": dict(binaries=("AlAs", "AlSb"), x_label="As",
                   bowing=dict(Delta_SO=0.15, Eg_Gamma=0.8, Eg_L=0.28, Eg_X=0.28,
                               VBO=-0.14)),
    "GaAsSb": dict(binaries=("GaAs", "GaSb"), x_label="As",
                   bowing=dict(Delta_SO=0.6, Eg_Gamma=1.43, Eg_L=1.2, Eg_X=1.2,
                               VBO=-1.06)),
    "InAsSb": dict(binaries=("InAs", "InSb"), x_label="As",
                   bowing=dict(Delta_SO=1.2, Eg_Gamma=0.67, Eg_L=0.6, Eg_X=0.6,
                               VBO=-0.402)),
}


def binary_names():
    return sorted(BINARIES)


def ternary_names():
    return sorted(TERNARIES)


def alloy_names():
    return sorted(list(BINARIES) + list(TERNARIES))
