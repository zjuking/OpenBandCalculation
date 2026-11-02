#!/usr/bin/env python3
"""
Comprehensive test run for the CPC submission.

Exercises every solver the program offers and prints one deterministic block
of text, so a referee can diff the output against examples/comprehensive_run.out:

    python examples/comprehensive_run.py

Takes about a minute on one core.  Nothing is written to disk.
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "src"))

import numpy as np                                          # noqa: E402

from obpds import Layer, OhmicContact, TwoTerminalDevice    # noqa: E402

T = 300.0
LINE = "-" * 66


def head(t):
    print(f"\n{LINE}\n{t}\n{LINE}")


def main():
    t_start = time.perf_counter()
    print("OBPDS 2.0 -- comprehensive test run")
    print(f"python {sys.version.split()[0]}, numpy {np.__version__}")

    # ---------------------------------------------------------- 1. diode
    head("1. GaAs pn junction: equilibrium")
    diode = TwoTerminalDevice(
        [Layer(1000, "GaAs", Na=1e17), Layer(1000, "GaAs", Nd=1e17)],
        [OhmicContact(), OhmicContact()], T=T, N=1000, approx="kane")
    sol = diode.get_equilibrium()
    vbi = float(sol.psi[-1] - sol.psi[0])
    field = np.abs(sol.field)
    print(f"  built-in potential   {vbi:.4f} V")
    print(f"  peak field           {field.max():.4e} V/cm at "
          f"{sol.x_nm[int(np.argmax(field))]:.1f} nm")
    print(f"  converged            {sol.converged}")

    # ------------------------------------------------- 2. drift-diffusion
    head("2. Drift-diffusion at 0.8 V")
    dd = diode.get_drift_diffusion(0.8)
    print(f"  J                    {dd.J:.6e} A/cm^2")
    print(f"  J_spread             {dd.J_spread:.2e}")
    print(f"  Gummel sweeps        {dd.gummel_iterations}")
    print(f"  converged            {dd.converged}")

    # ------------------------------------------------------------- 3. I-V
    head("3. I-V sweep")
    V, J = diode.get_iv(-0.5, 1.0, 5)
    for v, j in zip(V, J):
        print(f"  V = {v:+.2f} V    J = {j:+.6e} A/cm^2")

    # ------------------------------------------------------------- 4. C-V
    head("4. C-V sweep")
    V, C = diode.get_cv(-1.0, 0.0, 3)
    for v, c in zip(V, C):
        print(f"  V = {v:+.2f} V    C = {c:.6e} F/cm^2")

    # --------------------------------------------------------- 5. quantum
    head("5. Modulation-doped AlGaAs/GaAs heterojunction (quantum)")
    hemt = TwoTerminalDevice(
        [Layer(30, "AlGaAs", composition=0.30, Nd=1e18),
         Layer(10, "AlGaAs", composition=0.30),
         Layer(500, "GaAs")],
        [OhmicContact(), OhmicContact()], T=T, N=800, approx="kane")
    q = hemt.get_quantum_equilibrium(N=800, n_states=8)
    levels = q.subbands["levels_eV"]
    bound = q.subbands["bound"]
    print(f"  converged            {q.converged}")
    print(f"  bound subbands       {sum(bound)} of {len(levels)} levels")
    print(f"  E_0                  {levels[0]:.6f} eV")
    print(f"  E_cut                {q.subbands['E_cut_eV']:.6f} eV")
    print(f"  sheet density        {q.subbands['sheet_density_cm2']:.4e} cm^-2")

    # ---------------------------------------------------------- 6. strain
    head("6. Pseudomorphic In0.2Ga0.8As well on GaAs")
    well = TwoTerminalDevice(
        [Layer(20, "GaAs", Nd=1e17),
         Layer(8, "GaInAs", composition=0.8),
         Layer(20, "GaAs", Nd=1e17)],
        [OhmicContact(), OhmicContact()], T=T, N=800, substrate="GaAs")
    rows = well.describe()
    for r in rows:
        if abs(r.get("mismatch_sub", 0.0)) > 1e-6:
            print(f"  {r['material']:>10s}: f = {100*r['mismatch_sub']:+.3f} %"
                  f"  eps_in = {100*r['eps_in']:+.3f} %"
                  f"  eps_out = {100*r['eps_out']:+.3f} %"
                  f"  h_c = {r['h_c_nm']:.2f} nm"
                  f"  pseudomorphic = {r['pseudomorphic']}")
    sq = well.get_equilibrium(N=800)
    split = float(np.max(sq.Ev_HH - sq.Ev_LH))
    print(f"  HH-LH splitting      {1e3*split:.1f} meV")

    # ------------------------------------------------- 7. calibration
    head("7. Per-layer calibration of the valence-band offset")

    def well(substrate=True, overrides=None):
        dev = TwoTerminalDevice(
            [Layer(20, "GaAs", Nd=1e17),
             Layer(8, "GaInAs", composition=0.8, overrides=overrides),
             Layer(20, "GaAs", Nd=1e17)],
            [OhmicContact(), OhmicContact()], T=T, N=400,
            substrate="GaAs" if substrate else None)
        return dev.model(T, 400)

    i_well = int(np.argmin(np.abs(well().x * 1e7 - 24.0)))
    vbo_free = float(well(substrate=False).VBO[i_well])
    m_ref = well()
    m_cal = well(overrides={"VBO": vbo_free - 0.10})
    print(f"  VBO shift            {m_cal.VBO[i_well] - m_ref.VBO[i_well]:+.4f} eV")
    print(f"  gap shift            {m_cal.Eg[i_well] - m_ref.Eg[i_well]:+.2e} eV")
    print(f"  dEc, uncalibrated    "
          f"{1e3*(m_ref.CBO_Gamma[0] - m_ref.CBO_Gamma[i_well]):.1f} meV")
    print(f"  dEc, calibrated      "
          f"{1e3*(m_cal.CBO_Gamma[0] - m_cal.CBO_Gamma[i_well]):.1f} meV")

    print(f"\n{LINE}\nwall time {time.perf_counter()-t_start:.1f} s")


if __name__ == "__main__":
    main()
