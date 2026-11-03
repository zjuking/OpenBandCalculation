"""
Shared web-API contract checks, parameterised over the HTTP transport.

The front end reads a fixed set of payload fields out of each endpoint.  A
rename on either side fails *silently* -- the chart just comes up empty -- so
the contract is asserted here rather than left to be noticed on screen.

Two drivers share this body:

* ``web_smoke.py`` posts against a real server over ``urllib`` (the
  reproducible ``paper/cpc/data/web_smoke.out`` artefact), and
* ``test_web_api.py`` drives ``web.app.app`` through ``fastapi``'s in-process
  ``TestClient`` so ``python -m pytest -m api`` needs no server or port.

Both pass two callables ``get(path)`` and ``post(path, payload)`` that return
the parsed JSON, and a ``check(name, cond, extra="")`` reporter.  ``run_all``
does not own the pass/fail bookkeeping -- it only calls ``check`` -- so the
drivers can print (script) or collect failures (pytest) as they like.
"""

from __future__ import annotations

import time

# Fields the front end reads out of every band-diagram payload.
NEEDED = ["x_nm", "Ev", "Ec", "Fn", "Fp", "psi", "field", "n", "p", "rho",
          "layers", "meta"]

UI_BAND = ["x_nm", "Ec_Gamma", "Ec_L", "Ec_X", "Ev", "Ei", "Fn", "Fp",
           "n", "p", "n_L", "n_X", "field", "converged", "total_nm",
           "meta", "layers", "V", "mode",
           # drawn as separate lines whenever strain splits them
           "Ev_HH", "Ev_LH", "Ev_SO"]
UI_DD = ["J", "Jn_contact", "Jp_contact", "gummel_iterations", "Jtotal"]
UI_ROWS = ["material", "label", "thickness_nm", "Eg", "Eg_end", "doping_cm3",
           "lattice_constant", "mismatch", "dielectric"]
UI_ROWS_STRAIN = ["substrate", "substrate_a", "mismatch_sub", "eps_in",
                  "eps_out", "h_c_nm", "pseudomorphic", "strain_applied"]
UI_QUANTUM = ["n_is_quantum", "subbands"]
UI_SUBBANDS = ["levels_eV", "bound", "E_cut_eV", "sheet_density_cm2",
               "population_cm2", "psi"]

# The layer editor labels the composition field with the element x counts, and
# the layer table shows a readable label instead of the Python repr -- both are
# read straight off the API.
UI_MATERIAL = ["name", "ternary", "x_label", "x_default"]
UI_ROW_LABEL = "label"
UI_PRESET_SWEEP = ["V", "iv", "cv"]
UI_SWEEP_WINDOW = ["start", "stop", "num"]


def run_all(get, post, check):
    """Drive every contract check through ``get``/``post``; report via ``check``."""

    def _spec(preset):
        """Preset minus its UI-only sweep block, i.e. a plain DeviceSpec."""
        return {k: v for k, v in preset.items() if k != "sweep"}

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

    presets = get("/api/presets")["presets"]
    print(f"=== presets: {len(presets)}")

    # The workbench fills the composition label from x_label ("Al" for AlGaAs,
    # "Ga" for GaInAs); without it the user has no way to know which way round
    # the fraction goes.
    mats = get("/api/materials")["materials"]
    miss = sorted({k for k in UI_MATERIAL for m in mats if k not in m})
    check("material catalogue fields", not miss, f"missing={miss}")
    ternary = [m for m in mats if m["ternary"]]
    check("ternaries name their composition element",
          all(m.get("x_label") for m in ternary),
          f"{sum(1 for m in ternary if not m.get('x_label'))} without x_label")
    check("binaries have no composition",
          all(m.get("x_label") is None and m.get("x_default") is None
              for m in mats if not m["ternary"]))

    for name, dev in presets.items():
        # A preset is also a bias window; the front end applies the sweep block
        # when the stack is loaded.
        sw = dev.get("sweep") or {}
        miss = [k for k in UI_PRESET_SWEEP if k not in sw]
        check(f"{name}: preset sweep block", not miss, f"missing={miss}")
        if not miss:
            for kind in ("iv", "cv"):
                w = sw[kind]
                miss = [k for k in UI_SWEEP_WINDOW if k not in w]
                ok = not miss and w["num"] >= 5 and w["stop"] > w["start"]
                check(f"{name}: {kind} window", ok, f"{w}")

    for name, dev in presets.items():
        print(f"\n=== {name}")
        dev = {k: v for k, v in dev.items() if k != "sweep"}   # DeviceSpec
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
    dev = _spec(presets["GaAs pn diode"])
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
        finite = [jv for jv in J if jv is not None]
        check("all bias points finite", len(finite) == len(J),
              f"{len(finite)}/{len(J)}")
        check("monotone increasing",
              all(b >= a for a, b in zip(finite, finite[1:])),
              f"{finite[0]:.3e} -> {finite[-1]:.3e}")
        print(f"  ideality segments: {r.get('ideality')}")

    j = run("sweep", {"device": dev, "kind": "cv", "Vstart": -1.0,
                      "Vstop": 0.3, "Vnum": 14, "N": 200}, budget=400)
    check("cv sweep", j["status"] == "done", j.get("error", ""))
    if j["status"] == "done":
        C = [c for c in j["result"]["C"] if c is not None]
        check("capacitance positive", len(C) > 0 and min(C) > 0,
              f"{min(C):.3e} .. {max(C):.3e} F/cm^2")

    # Self-consistent Schrodinger-Poisson on the HEMT stack
    print("\n=== AlGaAs/GaAs HEMT (2DEG): quantum correction")
    dev = _spec(presets["AlGaAs/GaAs HEMT (2DEG)"])
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
        check("one wavefunction per bound state",
              len(sub.get("psi") or []) == nb,
              f"{len(sub.get('psi') or [])} rows")
        widths = {len(row) for row in (sub.get("psi") or [])}
        check("wavefunctions on the payload grid",
              widths == {len(r["x_nm"])}, f"{sorted(widths)} vs {len(r['x_nm'])}")
        check("electron column is the quantum one",
              r.get("n_is_quantum") is True)
        check("quantum n on the payload grid", len(r["n"]) == len(r["x_nm"]),
              f"{len(r['n'])} vs {len(r['x_nm'])}")
        ns = sub.get("sheet_density_cm2")
        check("sheet density physical", ns is not None and 1e10 < ns < 1e14,
              f"{ns:.4e} cm^-2")
        pop = sub.get("population_cm2") or []
        check("per-subband populations sum to the sheet density",
              pop and abs(sum(pop) - ns) <= 1e-6 * max(ns, 1.0),
              " + ".join(f"{p:.3e}" for p in pop))

    # Pseudomorphic strain
    print("\n=== Strained InGaAs/GaAs quantum well (2DEG): pseudomorphic strain")
    dev = _spec(presets["Strained InGaAs/GaAs quantum well (2DEG)"])
    check("preset carries a substrate", dev.get("substrate") == "GaAs",
          f"substrate={dev.get('substrate')}")
    rows = post("/api/describe", dev).get("rows") or []
    miss = sorted({k for k in UI_ROWS + UI_ROWS_STRAIN
                   for r in rows if k not in r})
    check("front-end strain columns", bool(rows) and not miss,
          f"missing={miss}")
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
