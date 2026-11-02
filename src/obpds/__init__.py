"""
Open Band Parameters Device Simulator -- modern Python 3 rewrite.

A one-dimensional electrostatic / drift-diffusion simulator for III-V
compound-semiconductor heterostructures, rebuilt from the (Python 2, 2015)
``obpds`` of Scott J. Maddox.  The academic core is preserved:

* Vurgaftman et al. (2001) III-V alloy parameters,
* Kane k.p non-parabolic statistics for the Gamma valley with parabolic
  L and X satellite valleys,
* generalised Fermi-Dirac integrals,
* a damped Newton solve of the nonlinear Poisson equation.

and extended with

* a flux-form ``div(eps grad psi)`` discretisation (the original dropped the
  ``deps/dx * dpsi/dx`` term at heterointerfaces),
* full drift-diffusion transport with Scharfetter-Gummel discretisation,
  giving terminal currents and I-V curves that the original could not
  produce at all.
"""

from .version import __version__

from .layer import (CompoundLayer, Contact, GradedLayer, Layer, OhmicContact,
                    SchottkyContact)
from .device import TwoTerminalDevice
from .solver import Model, build_model, carriers, solve_poisson
from .strain import Substrate, biaxial_strain, critical_thickness
from .transport import iv_curve, solve_drift_diffusion
from .materials import Alloy, get_alloy, alloy_names

__all__ = ["__version__", "Alloy", "get_alloy", "alloy_names",
           "Layer", "GradedLayer", "CompoundLayer",
           "Contact", "OhmicContact", "SchottkyContact",
           "Substrate", "biaxial_strain", "critical_thickness",
           "TwoTerminalDevice", "Model", "build_model", "carriers",
           "solve_poisson", "solve_drift_diffusion", "iv_curve"]
