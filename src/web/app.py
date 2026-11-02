"""
OBPDS web workbench: a FastAPI front end for the band-diagram solver.

The library itself is synchronous and a single drift-diffusion bias point can
take a second or two, so every simulation is dispatched to a worker thread and
polled through ``/api/job/{id}``.  That keeps the UI responsive and lets a
long bias sweep stream partial results back while it runs.

Run with::

    obpds-web                       # console script from pyproject.toml
    uvicorn web.app:app --reload    # development
"""

from __future__ import annotations

import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Literal

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from obpds import (GradedLayer, Layer, OhmicContact, SchottkyContact,
                   Substrate, TwoTerminalDevice)
from obpds.materials import BINARIES, TERNARIES, get_alloy

STATIC_DIR = str(Path(__file__).resolve().parent / "static")

app = FastAPI(title="OBPDS", version="2.0.0",
              description="Open Band Parameters Device Simulator")


# --------------------------------------------------------------------- schema
class LayerSpec(BaseModel):
    """One layer of the stack."""
    thickness_nm: float = 100.0
    material: str = "GaAs"
    composition: float | None = None
    composition_end: float | None = None
    doping: float = 0.0          # cm^-3, signed: + = acceptors, - = donors
    mu_n: float | None = None
    mu_p: float | None = None


class ContactSpec(BaseModel):
    kind: Literal["ohmic", "schottky"] = "ohmic"
    work_function: float = 4.9


class DeviceSpec(BaseModel):
    layers: list[LayerSpec] = Field(default_factory=list)
    contacts: list[ContactSpec] = Field(default_factory=list)
    T: float = 300.0
    N: int = 400
    approx: Literal["boltzmann", "fermi-dirac", "kane"] = "kane"
    tau_n: float = 1e-7
    tau_p: float = 1e-7
    B_rad: float = 1e-10
    # Which terminal the bias is referred to.  Only the zero-current and
    # C-V paths read these (drift-diffusion takes the polarity from the
    # doping profile itself), but a stack with its p layer on the right
    # needs them set or the C-V sweep comes out mirrored.
    Fp: Literal["left", "right"] = "left"
    Fn: Literal["left", "right"] = "right"
    # Suggested UI state only -- the solver reads the flag on SolveRequest.
    # A preset sets this so that loading a stack whose whole point is the
    # quantum correction arrives with the box already ticked.
    quantum: bool = False
    # The wafer.  ``None`` means every layer keeps its natural lattice
    # constant; a name picks an alloy from the catalogue, and substrate_a
    # overrides both with a lattice constant in Angstrom.
    substrate: str | None = None
    substrate_composition: float | None = None
    substrate_a: float | None = None


class SolveRequest(BaseModel):
    device: DeviceSpec
    mode: Literal["equilibrium", "zero_current", "drift_diffusion"] = \
        "equilibrium"
    V: float = 0.0
    downsample: int = 800
    quantum: bool = False
    n_states: int = 8


class SweepRequest(BaseModel):
    device: DeviceSpec
    kind: Literal["iv", "cv"] = "iv"
    Vstart: float = -1.0
    Vstop: float = 1.0
    Vnum: int = 41
    N: int | None = None


# ------------------------------------------------------------------- building
def build_device(spec: DeviceSpec, N: int | None = None) -> TwoTerminalDevice:
    layers = []
    for ls in spec.layers:
        kw = dict(mu_n=ls.mu_n, mu_p=ls.mu_p,
                  doping=float(ls.doping) if ls.doping else 0.0)
        if ls.composition_end is not None and ls.composition is not None:
            layers.append(GradedLayer(ls.thickness_nm, ls.material,
                                      (ls.composition, ls.composition_end),
                                      **kw))
        else:
            layers.append(Layer(ls.thickness_nm, ls.material,
                                ls.composition, **kw))
    if not layers:
        raise ValueError("The stack needs at least one layer.")

    contacts = []
    for cs in (spec.contacts or []):
        if cs.kind == "schottky":
            contacts.append(SchottkyContact(cs.work_function))
        else:
            contacts.append(OhmicContact())
    while len(contacts) < 2:
        contacts.append(OhmicContact())

    sub = None
    if spec.substrate_a:
        sub = Substrate(a=float(spec.substrate_a))
    elif spec.substrate:
        sub = Substrate(spec.substrate, spec.substrate_composition)

    return TwoTerminalDevice(
        layers, contacts[:2],
        T=spec.T, N=int(N if N is not None else spec.N), approx=spec.approx,
        tau_n=spec.tau_n, tau_p=spec.tau_p, B_rad=spec.B_rad,
        Fp=spec.Fp, Fn=spec.Fn, substrate=sub,
    )


def layer_extents(dev: TwoTerminalDevice):
    """``(label, x0_nm, x1_nm)`` for shading the band diagram."""
    out = []
    acc = 0.0
    for layer in dev.layers:
        t = layer.thickness * 1e7
        out.append({"label": str(layer.material), "x0_nm": acc,
                    "x1_nm": acc + t})
        acc += t
    return out, acc


def _clean(obj: Any):
    """Recursively replace non-finite floats so the JSON stays parseable."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return f if np.isfinite(f) else None
    # bool must be tested *before* int: it is a subclass, so the int branch
    # would otherwise turn every flag into 1 or 0.
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist())
    return obj


# ---------------------------------------------------------------------- jobs
class Job:
    def __init__(self, kind: str):
        self.id = uuid.uuid4().hex[:12]
        self.kind = kind
        self.status = "queued"
        self.progress = 0
        self.total = 1
        self.result: dict | None = None
        self.error: str | None = None
        self.started = time.time()

    def as_dict(self):
        return {"id": self.id, "kind": self.kind, "status": self.status,
                "progress": self.progress, "total": self.total,
                "result": self.result, "error": self.error,
                "elapsed": round(time.time() - self.started, 2)}


_JOBS: dict[str, Job] = {}
_LOCK = threading.Lock()
_POOL = ThreadPoolExecutor(max_workers=2)


def _submit(kind: str, fn) -> Job:
    job = Job(kind)
    with _LOCK:
        _JOBS[job.id] = job

    def run():
        job.status = "running"
        try:
            fn(job)
            job.status = "done"
        except Exception as exc:                       # pragma: no cover
            job.status = "error"
            job.error = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()
        finally:
            job.progress = job.total

    _POOL.submit(run)
    return job


# -------------------------------------------------------------------- solving
def _solve_job(req: SolveRequest):
    def work(job: Job):
        N = int(req.device.N)
        ds = req.downsample or None
        if req.mode == "equilibrium" and req.quantum:
            dev = build_device(req.device)
            sol = dev.get_quantum_equilibrium(N=N, n_states=int(
                req.n_states))
            payload = sol.as_dict(ds)
            payload["converged"] = bool(getattr(sol, "converged", True))
            payload["refined_to"] = N
            # The quantum density *is* the electron density once the
            # correction is on, so overwrite it rather than shipping two
            # columns and leaving the chart to pick.  It is downsampled to
            # match whatever the rest of the payload was thinned to.
            # Thin with the *same* index map Solution.as_dict uses, or the
            # quantum column and the wavefunctions come back a different
            # length from x_nm and the front end has nothing to plot against.
            def thin(v):
                v = np.asarray(v, dtype=float)
                if ds and v.ndim == 1 and v.size > ds:
                    v = v[np.linspace(0, v.size - 1, ds).astype(int)]
                return v

            payload["n"] = _clean(thin(sol.n_quantum).tolist())
            payload["n_is_quantum"] = True
            sub = dict(sol.subbands)
            sub["psi"] = _clean([thin(row).tolist() for row in sub["psi"]])
            payload["subbands"] = sub
        elif req.mode == "drift_diffusion":
            dev = build_device(req.device)
            sol = dev.get_drift_diffusion(req.V, N=N)
            payload = sol.as_dict(ds)
        else:
            solve_one = (lambda d: d.get_equilibrium(N=N)) \
                if req.mode == "equilibrium" \
                else (lambda d: d.get_zero_current(req.V, N=N))
            dev = build_device(req.device)
            sol = solve_one(dev)
            payload = sol.as_dict(ds)
            # A stack with a thin, heavily doped cap can out-run the mesh:
            # the Debye length there is a few nm and too-coarse cells leave
            # Newton stalling on a solution that is perfectly well behaved.
            # Refining is cheap next to a wrong band diagram.
            tries = 0
            while not getattr(sol, "converged", True) and N < 4000 and tries < 3:
                N *= 2
                tries += 1
                dev = build_device(req.device, N=N)
                sol = solve_one(dev)
                payload = sol.as_dict(ds)
            payload["converged"] = bool(getattr(sol, "converged", True))
            payload["refined_to"] = N
        extents, total_nm = layer_extents(dev)
        payload["V"] = float(req.V)
        payload["mode"] = req.mode
        payload["layers"] = extents
        payload["total_nm"] = total_nm
        payload["meta"] = {"T": dev.T, "N": N, "approx": dev.approx,
                           "kT": 8.617333262e-5 * dev.T}
        job.result = _clean(payload)
    return work


def _sweep_job(req: SweepRequest):
    def work(job: Job):
        N = req.N or min(req.device.N, 300)
        dev = build_device(req.device, N=N)
        V = np.linspace(req.Vstart, req.Vstop, int(req.Vnum))
        J = np.full(V.size, np.nan)
        job.total = V.size

        if req.kind == "cv":
            for i, v in enumerate(V):
                try:
                    J[i] = dev.get_capacitance(float(v), N=N)
                except Exception:
                    J[i] = np.nan
                job.progress = i + 1
                job.result = {"V": _clean(V), "C": _clean(J)}
        else:
            for i, v in enumerate(V):
                try:
                    s = dev.get_drift_diffusion(float(v), N=N)
                    J[i] = s.J
                    ok = s.converged
                except Exception:
                    ok = False
                job.progress = i + 1
                job.result = {"V": _clean(V), "J": _clean(J),
                              "converged": bool(ok),
                              "meta": {"N": N, "T": dev.T}}

        ok = np.isfinite(J)
        extra: dict[str, Any] = {}
        if req.kind == "iv" and ok.sum() >= 4:
            extra["ideality"] = _ideality(V[ok], np.abs(J[ok]))
        job.result = _clean({**(job.result or {}), **extra})
    return work


def _ideality(V, J, kT=0.025852, nseg=3):
    """
    Ideality factor over successive thirds of the *forward* branch.

    Fitting |J| across V = 0 would be meaningless: the magnitude falls
    through the reverse branch, bottoms out at zero and rises again, so a
    window straddling the origin measures the turn-around, not the slope.
    Only the forward branch has a defined ideality at all.
    """
    V = np.asarray(V, float)
    J = np.asarray(J, float)
    m = (V > 0.02) & np.isfinite(J) & (J > 0)
    if m.sum() < 6:
        return []
    x, y = V[m], np.log(J[m])
    lo, hi = float(x.min()), float(x.max())
    segs = []
    for i in range(nseg):
        a = lo + (hi - lo) * i / nseg
        b = lo + (hi - lo) * (i + 1) / nseg
        s = (x >= a) & (x <= b) if i == nseg - 1 else (x >= a) & (x < b)
        if s.sum() < 3:
            continue
        slope = float(np.polyfit(x[s], y[s], 1)[0])
        if not np.isfinite(slope) or abs(slope) < 1e-9:
            continue
        # ln J = V/(n kT) + const, so n = 1/(kT dlnJ/dV).  Returning kT*slope
        # gives 1/n instead -- which still looks plausible on screen, because
        # a well-behaved diode has n near 1.
        segs.append({"range": [float(x[s].min()), float(x[s].max())],
                     "ideality": float(1.0 / (kT * slope))})
    return segs


# -------------------------------------------------------------------- routing
@app.get("/api/materials")
def materials():
    """Alloy catalogue: name, whether it needs a composition, and properties."""
    out = []
    for name in sorted(set(BINARIES) | set(TERNARIES)):
        ternary = name in TERNARIES and name not in BINARIES
        row: dict[str, Any] = {"name": name, "ternary": ternary}
        try:
            p = get_alloy(name, 0.5 if ternary else None).props(300.0)
            for k in ("Eg", "Eg_Gamma", "Eg_L", "Eg_X", "VBO", "CBO",
                      "dielectric", "lattice_constant", "mu_n", "mu_p",
                      "Nc", "Nv", "ni", "electron_affinity", "direct",
                      "Delta_SO", "meff_hh_100", "meff_lh_100",
                      "C11", "C12", "a_c", "a_v", "b_shear"):
                if k in p:
                    row[k] = None if p[k] is None else float(p[k])
        except Exception as exc:
            row["error"] = str(exc)
        out.append(row)
    return {"materials": out}


@app.get("/api/properties")
def properties(material: str, composition: float | None = None, T: float = 300.0):
    try:
        p = get_alloy(material, composition).props(float(T))
    except Exception as exc:
        raise HTTPException(400, f"unknown alloy {material!r}: {exc}")
    return _clean({k: v for k, v in p.items()
                   if isinstance(v, (int, float, bool, str, type(None)))})


@app.get("/api/presets")
def presets():
    """A few stacks that show off what the simulator can do."""
    def L(t, m, d=0.0, c=None, c_end=None):
        return LayerSpec(thickness_nm=t, material=m, composition=c,
                         composition_end=c_end, doping=d)

    return {"presets": {
        "GaAs pn diode": DeviceSpec(
            layers=[L(200, "GaAs", +1e17), L(200, "GaAs", -1e17)],
            contacts=[ContactSpec(), ContactSpec()]),
        "AlGaAs/GaAs single heterojunction": DeviceSpec(
            layers=[L(50, "AlGaAs", +2e17, 0.3), L(20, "AlGaAs", 0.0, 0.3),
                    L(200, "GaAs", 0.0), L(300, "GaAs", -1e16)],
            contacts=[ContactSpec(), ContactSpec()]),
        "Graded AlGaAs/GaAs heterojunction": DeviceSpec(
            layers=[L(30, "GaAs", +1e17, None, None),
                    L(40, "AlGaAs", 0.0, 0.25, 0.25),
                    L(15, "AlGaAs", 0.0, 0.25),
                    L(400, "GaAs", 0.0)],
            contacts=[ContactSpec(), ContactSpec()]),
        # A modulation-doped heterojunction: donors in the wide-gap barrier
        # spill their electrons over into the undoped GaAs, where they pile
        # up in a notch narrow enough to quantise.  This is the stack the
        # quantum correction exists for -- run it with the box ticked and
        # the channel comes out ~5 nm *behind* the interface instead of on
        # it, with a sheet density of ~7e11 cm^-2.
        "AlGaAs/GaAs HEMT (2DEG)": DeviceSpec(
            layers=[L(30, "AlGaAs", -1e18, 0.30),
                    L(10, "AlGaAs", 0.0, 0.30),
                    L(500, "GaAs", 0.0)],
            contacts=[ContactSpec(), ContactSpec()], quantum=True),
        # The p layer is on the *right* here, so Fp/Fn have to be swapped or
        # the zero-current and C-V paths read the bias backwards.  It also
        # wants a finer mesh than the default: the 100 nm n+ InP cap sits on
        # a Debye length of ~4 nm, and at N = 400 the cells are 5 nm wide and
        # the Newton solve stalls.
        #
        # GaInAs takes the *Ga* fraction (Vurgaftman writes Ga_xIn_{1-x}As),
        # so lattice-matched In0.53Ga0.47As is composition 0.47.  Reading it
        # as the In fraction puts 0.53 % of mismatch on a layer that is meant
        # to be commensurate with the InP it grows on.
        "InGaAs/InP PIN photodiode": DeviceSpec(
            layers=[L(100, "InP", -1e18), L(1500, "GaInAs", 0.0, 0.47),
                    L(500, "InP", +1e17)],
            contacts=[ContactSpec(), ContactSpec()],
            N=800, Fp="right", Fn="left"),
        "GaAs Schottky diode": DeviceSpec(
            layers=[L(50, "GaAs", -1e17), L(450, "GaAs", -1e16)],
            contacts=[ContactSpec(kind="schottky", work_function=4.9),
                      ContactSpec()]),
        # 1.4 % compressive strain and a doped barrier, so the well fills:
        # the gap opens by ~70 meV, the heavy hole comes to the top ~80 meV
        # above the light hole, and the electrons that spill in from the GaAs
        # are quantised into a subband holding ~6e11 cm^-2.  Both ticks are
        # set because strain and quantisation are the two halves of what a
        # real quantum well does.
        "应变 InGaAs/GaAs 量子阱 (2DEG)": DeviceSpec(
            layers=[L(20, "GaAs", -1e17), L(8, "GaInAs", 0.0, 0.8),
                    L(20, "GaAs", -1e17)],
            contacts=[ContactSpec(), ContactSpec()],
            substrate="GaAs", quantum=True),
    }}


@app.post("/api/solve")
def solve(req: SolveRequest):
    job = _submit("solve", _solve_job(req))
    return {"job_id": job.id}


@app.post("/api/sweep")
def sweep(req: SweepRequest):
    job = _submit(f"sweep:{req.kind}", _sweep_job(req))
    return {"job_id": job.id}


@app.post("/api/describe")
def describe(req: DeviceSpec):
    dev = build_device(req)
    return {"rows": _clean(dev.describe()), "extents": layer_extents(dev)[0]}


@app.get("/api/job/{job_id}")
def job_status(job_id: str):
    with _LOCK:
        job = _JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "unknown job")
    return job.as_dict()


@app.exception_handler(Exception)
async def _unhandled(request, exc):                     # pragma: no cover
    traceback.print_exc()
    return JSONResponse(status_code=500,
                        content={"error": f"{type(exc).__name__}: {exc}"})


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")


def main():                                             # pragma: no cover
    import os
    import uvicorn
    uvicorn.run(app, host=os.environ.get("HOST", "127.0.0.1"),
                port=int(os.environ.get("PORT", 8000)), log_level="info")


if __name__ == "__main__":                              # pragma: no cover
    main()
