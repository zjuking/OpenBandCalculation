"""Terminal-current estimator: depletion-region median vs the naive mean."""
import sys
import time

import numpy as np

from obpds import TwoTerminalDevice, Layer, OhmicContact
from obpds.transport import solve_drift_diffusion

N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
Vs = [float(v) for v in (sys.argv[2:] or
                         ["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7",
                          "0.8", "0.9", "-0.1", "-0.5", "-1.0"])]

dev = TwoTerminalDevice(
    [Layer(200, "GaAs", Na=1e17), Layer(200, "GaAs", Nd=1e17)],
    [OhmicContact(), OhmicContact()], N=N)
m = dev.model()
kT = 8.617333262e-5 * m.T

print(f"{'V':>6} {'J (depletion)':>14} {'spread':>9} {'iters':>5} "
      f"{'conv':>5} {'n':>6} {'t(s)':>6}", flush=True)
prev = None
for V in Vs:
    t = time.time()
    s = solve_drift_diffusion(m, V, "kane")
    n = ""
    if prev is not None and prev[1] > 0 and s.J > 0:
        n = f"{np.log(s.J / prev[1]) * kT / (V - prev[0]):6.3f}"
    print(f"{V:+6.2f} {s.J:14.5e} {s.J_spread:9.1e} "
          f"{s.gummel_iterations:5d} {str(s.converged):>5} "
          f"{n:>6} {time.time()-t:6.1f}", flush=True)
    prev = (V, s.J)
