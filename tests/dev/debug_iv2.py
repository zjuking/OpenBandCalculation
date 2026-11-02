"""Compact I-V probe: a handful of bias points with timing and convergence."""
import sys
import time

import numpy as np

from obpds import TwoTerminalDevice, Layer, OhmicContact
from obpds.transport import solve_drift_diffusion

N = int(sys.argv[1]) if len(sys.argv) > 1 else 400
Vs = [float(v) for v in (sys.argv[2:] or
                         ["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7",
                          "0.8", "-0.1", "-0.5", "-1.0"])]

# p on the LEFT: this code follows the original OBPDS polarity convention,
# in which a positive bias on the left terminal is forward for a p-on-left
# stack (the default `Fp='left'` pinning).  With n on the left the sweep
# below runs the diode backwards and reads as a saturated reverse current.
dev = TwoTerminalDevice(
    [Layer(200, "GaAs", Na=1e17), Layer(200, "GaAs", Nd=1e17)],
    [OhmicContact(), OhmicContact()], N=N)
m = dev.model()
print(f"{'V':>6} {'J (A/cm2)':>13} {'Jn[0]':>12} {'Jp[0]':>12} "
      f"{'iters':>5} {'conv':>5} {'t(s)':>6}", flush=True)
for V in Vs:
    t = time.time()
    s = solve_drift_diffusion(m, V, "kane")
    print(f"{V:+6.2f} {s.J:13.5e} {s.Jn_contact:12.4e} {s.Jp_contact:12.4e} "
          f"{s.gummel_iterations:5d} {str(s.converged):>5} "
          f"{time.time()-t:6.1f}", flush=True)
