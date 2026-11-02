"""Forward/reverse I-V sweep on the GaAs pn diode, with ideality factor."""
from __future__ import annotations
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from obpds import Layer, OhmicContact, TwoTerminalDevice
from obpds.fermi import KB_EV
from obpds.materials import get_alloy

kT = KB_EV * 300.0
gaas = get_alloy("GaAs").props(300.0)
print(f"GaAs: Eg={gaas['Eg']:.4f} eV  Nc={gaas['Nc']:.4e}  Nv={gaas['Nv']:.4e}"
      f"  ni={gaas['ni']:.4e}")
print(f"mu_n={gaas['mu_n']:.0f}  mu_p={gaas['mu_p']:.0f}  "
      f"eps_r={gaas['dielectric']:.3f}")

na = nd = 1e17
dev = TwoTerminalDevice([Layer(1000, "GaAs", Na=na),
                         Layer(1000, "GaAs", Nd=nd)],
                        [OhmicContact(), OhmicContact()],
                        Fp="left", Fn="right", N=400)
m = dev.model(300.0, 400)
from obpds.transport import solve_drift_diffusion

# analytic short-base diffusion saturation current (both neutral regions)
q = 1.602176634e-19
ni = gaas["ni"]
Dn = gaas["mu_n"] * kT
Dp = gaas["mu_p"] * kT
W = 0.9e-4   # quasi-neutral width on each side, cm
J0_diff = q * ni ** 2 * (Dp / (W * na) + Dn / (W * nd))
print(f"analytic short-base J0 (diffusion) = {J0_diff:.4e} A/cm^2")

Vs = np.concatenate([np.linspace(-1.0, 0.0, 11),
                     np.linspace(0.1, 1.0, 19)])
print("\n   V (V)     J (A/cm^2)      Jn_contact      Jp_contact   iters conv")
rows = []
init = None
for V in Vs:
    t0 = time.time()
    s = solve_drift_diffusion(m, float(V), "kane", max_gummel=40, init=init)
    if s.converged:
        init = (s.psi, s.Fn, s.Fp)
    rows.append((V, s.J))
    print(f"  {V:+.3f}   {s.J:+.5e}   {s.Jn_contact:+.3e}   "
          f"{s.Jp_contact:+.3e}   {s.gummel_iterations:3d}   "
          f"{int(s.converged)}   ({time.time()-t0:.1f}s)")
    sys.stdout.flush()

V = np.array([r[0] for r in rows])
J = np.array([r[1] for r in rows])
for lo, hi in ((0.15, 0.45), (0.50, 0.80), (0.85, 1.00)):
    sel = (V >= lo) & (V <= hi) & (J > 0) & np.isfinite(J)
    if sel.sum() >= 3:
        slope = np.polyfit(V[sel], np.log(J[sel]), 1)[0]
        print(f"ideality {lo:.2f}-{hi:.2f} V: {1.0/(slope*kT):.3f}")
