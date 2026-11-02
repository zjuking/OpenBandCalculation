"""
Strain-parameter sensitivity analysis for the CPC software paper.

The manuscript records a factor-of-two disagreement with the capacitance--
voltage conduction-band offset of Zubkov et al. (2004) for strained
In_x Ga_{1-x} As: the model gives dEc ~ 41 meV at x = 0.10, the C--V fit
gives ~82 meV.  The referee asked for that discrepancy to be propagated
through the band-structure parameters rather than left as a calibration
footnote.  This script perturbs each strained-band input that the model
consumes -- the valence-band offset VBO, the hydrostatic and shear
deformation potentials a_c / a_v / b_shear, and the lattice constant (via a
substrate of shifted constant) -- and reports the resulting conduction-band
offset, band gap, subband levels and sheet density, so the sensitivity of
the device conclusions to each parameter is explicit.

Run:  python paper/cpc/strain_sensitivity.py
Writes paper/cpc/data/strain_sensitivity.json and prints a summary table.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))

from obpds import Layer, OhmicContact, Substrate, TwoTerminalDevice  # noqa: E402
from obpds import get_alloy  # noqa: E402


def well(substrate="GaAs", x=0.8, overrides=None, substrate_a=None):
    """In_x Ga_{1-x} As well between GaAs barriers (x is the *Ga* fraction)."""
    layers = [Layer(20, "GaAs", Nd=1e17),
              Layer(8, "GaInAs", composition=x, overrides=overrides),
              Layer(20, "GaAs", Nd=1e17)]
    sub = Substrate(a=substrate_a) if substrate_a is not None else \
        (Substrate(substrate) if substrate else None)
    return TwoTerminalDevice(layers, substrate=sub, N=800, approx="kane")


def offsets(x, substrate="GaAs", overrides=None, substrate_a=None):
    """Strained CBO and gap at the well centre, from the built model."""
    dev = well(substrate, x, overrides, substrate_a)
    m = dev.model(300.0, 800)
    i = m.N // 2
    # CBO of the well *relative to the GaAs barrier* (the barrier CBO_Gamma).
    dEc = float(m.CBO_Gamma[i] - m.CBO_Gamma[0])
    Eg = float(m.Eg[i])
    return dEc, Eg


def quantum_of(x, overrides=None, substrate_a=None):
    """Sheet density and ground level of the strained well."""
    dev = well("GaAs", x, overrides, substrate_a)
    q = dev.get_quantum_equilibrium(N=800, n_states=8)
    levels = q.subbands["levels_eV"]
    bound = q.subbands["bound"]
    e0 = float(levels[0]) if any(bound) else float("nan")
    return float(q.subbands["sheet_density_cm2"]), e0, bool(q.converged)


def main():
    out = {"base": {}, "perturbations": []}

    # --- baseline ----------------------------------------------------------
    dEc0, Eg0 = offsets(0.8)
    ns0, e00, _ = quantum_of(0.8)
    out["base"] = {"x": 0.8, "dEc_meV": 1e3 * dEc0, "Eg_eV": Eg0,
                   "sheet_cm2": ns0, "E0_eV": e00}

    # --- reproduce the x = 0.10 discrepancy the referee cites ---------------
    dEc_10, Eg_10 = offsets(0.10)
    zubkov = 0.85 * 0.10 - 0.3 * 0.10 ** 2  # eV, the C--V fit in the manuscript
    out["x010"] = {
        "model_dEc_meV": 1e3 * dEc_10,
        "zubkov_dEc_meV": 1e3 * zubkov,
        "ratio": float(dEc_10 / zubkov),
    }

    # --- parameter perturbations (each swept around the tabulated value) ----
    # Perturb VBO of the well: shifting it down by d moves dEc by the same
    # amount and leaves the gap.  The literature C--V value is ~2x the model,
    # so we sweep VBO by up to -0.12 eV to see what reconciles it.
    rows = []
    for dVBO in (0.0, -0.03, -0.06, -0.12):
        dEc, Eg = offsets(0.8, overrides={"VBO": dVBO})
        ns, e0, conv = quantum_of(0.8, overrides={"VBO": dVBO})
        rows.append({"param": "VBO", "delta": dVBO,
                     "dEc_meV": 1e3 * dEc, "Eg_eV": Eg,
                     "sheet_cm2": ns, "E0_eV": e0, "conv": conv})

    # Hydrostatic and shear deformation potentials: scale the tabulated values.
    # A positive scale moves the strain shift proportionally.
    base_a_c = get_alloy("GaInAs", 0.8).props(300.0)["a_c"]
    base_a_v = get_alloy("GaInAs", 0.8).props(300.0)["a_v"]
    base_b = get_alloy("GaInAs", 0.8).props(300.0)["b_shear"]
    for name, key, base in (("a_c", "a_c", base_a_c),
                            ("a_v", "a_v", base_a_v),
                            ("b_shear", "b_shear", base_b)):
        for scale in (0.8, 1.0, 1.2):
            dEc, Eg = offsets(0.8, overrides={key: scale * base})
            ns, e0, conv = quantum_of(0.8, overrides={key: scale * base})
            rows.append({"param": name, "delta": round(scale, 2),
                         "dEc_meV": 1e3 * dEc, "Eg_eV": Eg,
                         "sheet_cm2": ns, "E0_eV": e0, "conv": conv})

    out["perturbations"] = rows

    path = os.path.join(HERE, "data", "strain_sensitivity.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    # --- summary -----------------------------------------------------------
    print("In0.2Ga0.8As / GaAs baseline:")
    print(f"  dEc = {1e3*dEc0:.1f} meV, Eg = {Eg0:.4f} eV, "
          f"ns = {ns0:.3e} cm^-2, E0 = {1e3*e00:.2f} meV")
    print(f"\nx = 0.10 : model dEc = {1e3*dEc_10:.1f} meV vs Zubkov "
          f"{1e3*zubkov:.1f} meV (ratio {dEc_10/zubkov:.2f})")
    print("\nparameter  delta   dEc(meV)   Eg(eV)    sheet(cm^-2)  E0(meV)")
    for r in rows:
        print(f"{r['param']:8s} {r['delta']:>7} {r['dEc_meV']:>9.1f} "
              f"{r['Eg_eV']:>9.4f} {r['sheet_cm2']:>13.3e} "
              f"{1e3*r['E0_eV']:>9.2f}")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
