"""Diagnose why the continuity Newton reports non-convergence (real path)."""
from __future__ import annotations
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import obpds.newton as N
from obpds import Layer, OhmicContact, TwoTerminalDevice

_orig = N.newton
CALLS = [0]


def traced(G, A, u0, rtol=0.0, atol=1e-8, **kw):
    CALLS[0] += 1
    tag = CALLS[0]
    g0 = G(u0)
    t0 = time.time()
    r = _orig(G, A, u0, rtol=rtol, atol=atol, **kw)
    g = G(r.value)
    print(f"    newton#{tag}: n0={np.linalg.norm(g0):.3e} -> "
          f"|G|2={np.linalg.norm(g):.3e} |G|inf={np.abs(g).max():.3e} "
          f"iters={r.num_iter} conv={r.converged} {time.time()-t0:.2f}s")
    return r


N.newton = traced

dev = TwoTerminalDevice([Layer(1000, "GaAs", Na=1e17),
                         Layer(1000, "GaAs", Nd=1e17)],
                        [OhmicContact(), OhmicContact()],
                        Fp="left", Fn="right", N=200)

V = float(sys.argv[1]) if len(sys.argv) > 1 else 0.3
from obpds.transport import solve_drift_diffusion
m = dev.model(300.0, 200)
s = solve_drift_diffusion(m, V, "kane", max_gummel=8)
print(f"V={V}: gummel={s.gummel_iterations} conv={s.converged} J={s.J:.4e}")
print("   Jn_contact", s.Jn_contact, " Jp_contact", s.Jp_contact)
print("   Fn:", [f"{v:+.3f}" for v in s.Fn[::40]])
print("   Fp:", [f"{v:+.3f}" for v in s.Fp[::40]])
