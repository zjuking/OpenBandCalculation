"""
End-to-end smoke test of the web API against a running server.

    python tests/web_smoke.py [base_url]

Posts each preset through /api/solve and /api/sweep and checks that the
payload the front end actually draws on is present and finite.
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8123"


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=60) as r:
        return json.load(r)


def post(path, payload):
    req = urllib.request.Request(
        BASE + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def wait(jid, budget=300.0):
    t0 = time.time()
    while time.time() - t0 < budget:
        j = get(f"/api/job/{jid}")
        if j["status"] in ("done", "error"):
            return j
        time.sleep(0.4)
    raise TimeoutError(f"job {jid} did not finish in {budget}s")


def run(kind, payload, budget=300.0):
    j = post(f"/api/{kind}", payload)
    return wait(j["job_id"], budget)


ok = True


def check(name, cond, extra=""):
    global ok
    ok &= bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {extra}")


presets = get("/api/presets")["presets"]
print(f"=== presets: {len(presets)}")

NEEDED = ["x_nm", "Ev", "Ec", "Fn", "Fp", "psi", "field", "n", "p", "rho",
          "layers", "meta"]

# Every field the front end reads out of the payload.  A rename on either
# side fails *silently* -- the chart just comes up empty -- so the contract
# is asserted here rather than left to be noticed on screen.
UI_BAND = ["x_nm", "Ec_Gamma", "Ec_L", "Ec_X", "Ev", "Ei", "Fn", "Fp",
           "n", "p", "n_L", "n_X", "field", "converged", "total_nm",
           "meta", "layers", "V", "mode",
           # drawn as separate lines whenever strain splits them
           "Ev_HH", "Ev_LH", "Ev_SO"]
UI_DD = ["J", "Jn_contact", "Jp_contact", "gummel_iterations", "Jtotal"]
UI_ROWS = ["material", "thickness_nm", "Eg", "Eg_end", "doping_cm3",
           "lattice_constant", "mismatch", "dielectric"]
# Extra layer-table columns that only exist once a substrate is set.
UI_ROWS_STRAIN = ["substrate", "substrate_a", "mismatch_sub", "eps_in",
                  "eps_out", "h_c_nm", "pseudomorphic", "strain_applied"]
# The wavefunction tab reads everything below out of `subbands`; a rename
# there leaves the tab silently empty.
UI_QUANTUM = ["n_is_quantum", "subbands"]
UI_SUBBANDS = ["levels_eV", "bound", "E_cut_eV", "sheet_density_cm2",
               "population_cm2", "psi"]

for name, dev in presets.items():
    print(f"\n=== {name}")
    j = run("solve", {"device": dev, "mode": "equilibrium", "V": 0.0})
    if j["status"] != "done":
        check("equilibrium solve", False, (j.get("error") or ""))
        continue
    r = j["result"]
    check("equilibrium converged", bool(r.get("converged")),
          f"({j['elapsed']}s, N={r['meta']['N']})")
    missing = [k for k in NEEDED if k not in r]
    check("payload keys", not missing, f"missing={missing}")
    x = r["x_nm"]
    check("grid", len(x) >= 10 and x[-1] > x[0],
          f"N={len(x)} span={x[0]:.1f}..{x[-1]:.1f} nm")
    Ec = [v for v in r["Ec"] if v is not None]
    check("band edges finite", len(Ec) == len(r["Ec"]),
          f"Ec {min(Ec):.3f}..{max(Ec):.3f} eV")
    check("layer extents", len(r["layers"]) == len(dev["layers"]))
    miss = [k for k in UI_BAND if k not in r]
    check("front-end band fields", not miss, f"missing={miss}")

    rows = post("/api/describe", dev).get("rows") or []
    miss = sorted({k for k in UI_ROWS for row in rows if k not in row})
    check("front-end layer-table fields", bool(rows) and not miss,
          f"missing={miss}")

# biased solve + sweep on the diode
print("\n=== GaAs pn diode: drift-diffusion + sweep")
dev = dict(presets["GaAs pn diode"])
dev["N"] = 200
j = run("solve", {"device": dev, "mode": "drift_diffusion", "V": 0.6})
check("drift-diffusion solve", j["status"] == "done", j.get("error", ""))
if j["status"] == "done":
    r = j["result"]
    check("terminal current reported", r.get("J") is not None,
          f"J={r.get('J'):.4e} A/cm^2")
    miss = [k for k in UI_BAND + UI_DD if k not in r]
    check("front-end drift-diffusion fields", not miss, f"missing={miss}")

j = run("sweep", {"device": dev, "kind": "iv", "Vstart": -0.5,
                  "Vstop": 0.9, "Vnum": 15, "N": 200}, budget=400)
check("iv sweep", j["status"] == "done", j.get("error", ""))
if j["status"] == "done":
    r = j["result"]
    V, J = r["V"], r["J"]
    finite = [j for j in J if j is not None]
    check("all bias points finite", len(finite) == len(J),
          f"{len(finite)}/{len(J)}")
    check("monotone increasing", all(b >= a for a, b in zip(finite, finite[1:])),
          f"{finite[0]:.3e} -> {finite[-1]:.3e}")
    print(f"  ideality segments: {r.get('ideality')}")

j = run("sweep", {"device": dev, "kind": "cv", "Vstart": -1.0,
                  "Vstop": 0.3, "Vnum": 14, "N": 200}, budget=400)
check("cv sweep", j["status"] == "done", j.get("error", ""))
if j["status"] == "done":
    C = [c for c in j["result"]["C"] if c is not None]
    check("capacitance positive", len(C) > 0 and min(C) > 0,
          f"{min(C):.3e} .. {max(C):.3e} F/cm^2")

# Self-consistent Schrodinger-Poisson on the HEMT stack: this is the only
# preset with a channel narrow enough to actually quantise, and it is also
# the slowest path in the app, so it gets its own contract check.
print("\n=== AlGaAs/GaAs HEMT (2DEG): quantum correction")
dev = dict(presets["AlGaAs/GaAs HEMT (2DEG)"])
dev["N"] = 400
j = run("solve", {"device": dev, "mode": "equilibrium", "V": 0.0,
                  "quantum": True, "n_states": 8}, budget=400)
check("quantum solve finished", j["status"] == "done", j.get("error", ""))
if j["status"] == "done":
    r = j["result"]
    check("quantum converged", bool(r.get("converged")), f"({j['elapsed']}s)")
    miss = [k for k in UI_BAND + UI_QUANTUM if k not in r]
    check("front-end quantum fields", not miss, f"missing={miss}")
    sub = r.get("subbands") or {}
    miss = [k for k in UI_SUBBANDS if k not in sub]
    check("subband payload keys", not miss, f"missing={miss}")
    nb = sum(1 for b in (sub.get("bound") or []) if b)
    check("at least one bound subband", nb >= 1,
          f"{nb} of {len(sub.get('bound') or [])} levels, "
          f"E_cut={sub.get('E_cut_eV')}")
    check("one wavefunction per bound state", len(sub.get("psi") or []) == nb,
          f"{len(sub.get('psi') or [])} rows")
    widths = {len(row) for row in (sub.get("psi") or [])}
    check("wavefunctions on the payload grid",
          widths == {len(r["x_nm"])}, f"{sorted(widths)} vs {len(r['x_nm'])}")
    check("electron column is the quantum one", r.get("n_is_quantum") is True)
    check("quantum n on the payload grid", len(r["n"]) == len(r["x_nm"]),
          f"{len(r['n'])} vs {len(r['x_nm'])}")
    ns = sub.get("sheet_density_cm2")
    check("sheet density physical", ns is not None and 1e10 < ns < 1e14,
          f"{ns:.4e} cm^-2")
    pop = sub.get("population_cm2") or []
    check("per-subband populations sum to the sheet density",
          pop and abs(sum(pop) - ns) <= 1e-6 * max(ns, 1.0),
          " + ".join(f"{p:.3e}" for p in pop))

# Pseudomorphic strain: the substrate is a separate field from the stack, so
# it needs its own contract check on both endpoints.
print("\n=== 应变 InGaAs/GaAs 量子阱 (2DEG): pseudomorphic strain")
dev = dict(presets["应变 InGaAs/GaAs 量子阱 (2DEG)"])
check("preset carries a substrate", dev.get("substrate") == "GaAs",
      f"substrate={dev.get('substrate')}")
rows = post("/api/describe", dev).get("rows") or []
miss = sorted({k for k in UI_ROWS + UI_ROWS_STRAIN for r in rows if k not in r})
check("front-end strain columns", bool(rows) and not miss, f"missing={miss}")
well = next((r for r in rows if "GaInAs" in r["material"]), None)
if well:
    print(f"    well: f={well['mismatch_sub'] * 100:+.3f} %  "
          f"eps_in={well['eps_in'] * 100:+.3f} %  "
          f"eps_out={well['eps_out'] * 100:+.3f} %  "
          f"h_c={well['h_c_nm']} nm  pseudomorphic={well['pseudomorphic']}")
    check("well is compressively strained", well["eps_in"] < 0.0,
          f"{well['eps_in'] * 100:+.3f} %")
    check("out-of-plane strain is tensile", well["eps_out"] > 0.0,
          f"{well['eps_out'] * 100:+.3f} %")
    check("critical thickness is a few nm",
          2.0 < float(well["h_c_nm"]) < 40.0, f"{well['h_c_nm']} nm")
    check("8 nm well stays pseudomorphic", well["pseudomorphic"] is True)
    barrier = next(r for r in rows if r["material"].startswith("Layer(20"))
    check("GaAs barrier is unstrained", abs(barrier["eps_in"]) < 1e-12,
          f"{barrier['eps_in']:.2e}")

j = run("solve", {"device": dev, "mode": "equilibrium", "V": 0.0,
                  "downsample": 800})
check("strained solve finished", j["status"] == "done", j.get("error", ""))
if j["status"] == "done":
    r = j["result"]
    check("strained solve converged", bool(r.get("converged")))
    split = max(abs(a - b) for a, b in zip(r["Ev_HH"], r["Ev_LH"]))
    print(f"    HH-LH splitting in the payload: {split * 1000:.1f} meV")
    check("valence band is split", split > 0.05, f"{split * 1000:.1f} meV")
    check("heavy hole is the upper edge", r["Ev_HH"][0] >= r["Ev_LH"][0])
    check("split-off stays below both", max(r["Ev_SO"]) < min(r["Ev_LH"]))

# Turning the substrate off must take the splitting with it.
nostrain = dict(dev, substrate=None)
j = run("solve", {"device": nostrain, "mode": "equilibrium", "V": 0.0})
if j["status"] == "done":
    r = j["result"]
    split = max(abs(a - b) for a, b in zip(r["Ev_HH"], r["Ev_LH"]))
    check("no substrate: no splitting", split < 1e-12, f"{split:.2e} eV")

print("\n" + ("WEB SMOKE PASSED" if ok else "WEB SMOKE FAILED"))
sys.exit(0 if ok else 1)
