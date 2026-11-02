"""
Build the figures of the CPC software paper.

    python paper/cpc/make_figures.py            # everything (vector PDFs)
    python paper/cpc/make_figures.py --reuse    # reuse cached solve results
    python paper/cpc/make_figures.py --png      # also write bitmaps to look at

Writes vector PDFs into paper/cpc/figures/ and caches the solved data in
paper/cpc/data/figdata.json so the figures can be regenerated without
re-solving the devices.
"""
from __future__ import annotations

import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))

from obpds import Layer, OhmicContact, TwoTerminalDevice  # noqa: E402
from obpds.fermi import KB_EV  # noqa: E402

FIG = os.path.join(HERE, "figures")
DATA = os.path.join(HERE, "data")
os.makedirs(FIG, exist_ok=True)

MM = 1.0 / 25.4
W1 = 90 * MM      # single panel (included at \columnwidth)
W2 = 137 * MM     # full text width of the elsarticle preprint layout
W_ARCH = 165 * MM # module map: wider so the box labels fit at print size

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 8,
    "axes.linewidth": 0.6,
    "axes.labelsize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "savefig.bbox": "tight",
    "savefig.format": "pdf",
})

C_CLASS = "#1f4e79"
C_QUANT = "#b8860b"
C_STRAIN = "#2e7d32"
C_GREY = "#666666"


def save(fig, name):
    """Vector PDF always; bitmaps only on request, for eyeballing."""
    fig.savefig(os.path.join(FIG, f"{name}.pdf"))
    if "--png" in sys.argv:
        fig.savefig(os.path.join(FIG, f"{name}.png"), dpi=300)
    plt.close(fig)
    print(f"  wrote figures/{name}.pdf")


# --------------------------------------------------------------------------
def fig_architecture():
    """Module map and data flow."""
    fig, ax = plt.subplots(figsize=(W_ARCH, 78 * MM))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 62)
    ax.axis("off")

    def box(x, y, w, h, title, sub="", color=C_CLASS, fs=7.4):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.6,rounding_size=1.2",
            linewidth=0.8, edgecolor=color, facecolor=color + "14"))
        ax.text(x + w / 2, y + h - 2.2, title, ha="center", va="center",
                fontsize=fs, fontweight="bold", color=color)
        if sub:
            ax.text(x + w / 2, y + h / 2 - 1.4, sub, ha="center", va="center",
                    fontsize=6.6, color="#333333")

    def arrow(x0, y0, x1, y1, style="-|>", ls="-"):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style,
                                     mutation_scale=8, linewidth=0.7,
                                     color="#555555", linestyle=ls,
                                     shrinkA=0, shrinkB=0))

    # layer 1 -- user surface
    box(2, 50, 44, 9, "TwoTerminalDevice", "layer stack, contacts, substrate")
    box(54, 50, 44, 9, "web.app (FastAPI)", "/api/solve, /api/sweep, jobs")

    # layer 2 -- model construction
    box(2, 36, 20, 9, "layer.py", "Layer, GradedLayer")
    box(26, 36, 24, 9, "materials/", "Vurgaftman 2001\n+ alloys, Eg(T)")
    box(54, 36, 20, 9, "strain.py", "Bir-Pikus edges,\nMatthews-Blakeslee",
        C_STRAIN)
    box(78, 36, 20, 9, "build_model", "grid-resolved Model")

    # layer 3 -- kernels
    box(2, 20, 20, 9, "fermi.py", "H_1/2, Gauss-Leg", C_QUANT)
    box(26, 20, 22, 9, "newton.py", "damped Newton,\nrow scaling", C_QUANT)
    box(52, 20, 20, 9, "solver.py", "FV Poisson,\nmultivalley n, p")
    box(76, 20, 22, 9, "transport.py", "SG flux, Gummel")

    # layer 4 -- optional physics
    box(52, 4, 20, 9, "quantum.py", "BenDaniel-Duke",
        C_QUANT)
    box(76, 4, 22, 9, "output", "Solution -> JSON")

    # vertical flow
    arrow(24, 50, 12, 45)
    arrow(38, 50, 38, 45)
    arrow(64, 50, 64, 45)
    arrow(88, 50, 88, 45)
    arrow(12, 36, 12, 29)
    arrow(34, 36, 36, 29)
    arrow(62, 36, 62, 29)
    arrow(88, 36, 88, 29)
    arrow(62, 20, 62, 13)
    arrow(86, 20, 86, 13)

    # horizontal couplings between the kernels
    arrow(22, 24.5, 26, 24.5)
    arrow(48, 24.5, 52, 24.5)
    arrow(72, 24.5, 76, 24.5)

    # dashed feedback: quantum density -> Poisson (Gummel)
    ax.add_patch(FancyArrowPatch((55, 13.5), (55, 20), arrowstyle="-|>",
                                 mutation_scale=8, linewidth=0.7,
                                 color=C_QUANT, linestyle=(0, (2, 2)),
                                 shrinkA=0, shrinkB=0))
    ax.text(54, 16.6, "Gummel", fontsize=6.2, color=C_QUANT, ha="right",
            va="center")

    save(fig, "fig_architecture")


# --------------------------------------------------------------------------
CHECK = re.compile(
    r"\[PASS\]\s+(?P<name>.+?)\s+got\s+(?P<got>[-+0-9.eE]+).*?"
    r"\(rel\s+(?P<rel>[-+0-9.eE]+),\s*tol\s+(?P<tol>[-+0-9.eE]+)\)")


def fig_verification():
    """Relative error of every check against the tolerance it is held to."""
    files = [("classical", "validate_physics.out", C_CLASS, "o"),
             ("quantum", "validate_quantum.out", C_QUANT, "s"),
             ("strain", "validate_strain.out", C_STRAIN, "^")]
    fig, ax = plt.subplots(figsize=(W1, 68 * MM))
    total = 0
    for label, fn, color, marker in files:
        path = os.path.join(DATA, fn)
        if not os.path.exists(path):
            print(f"  !! missing {fn}")
            continue
        rels, tols = [], []
        for line in open(path, encoding="utf-8", errors="ignore"):
            m = CHECK.search(line)
            if not m:
                continue
            rel = float(m.group("rel"))
            rels.append(max(rel, 1e-17))
            tols.append(float(m.group("tol")))
        total += len(rels)
        ax.scatter(tols, rels, s=13, marker=marker, facecolor="none",
                   edgecolor=color, linewidth=0.7, label=f"{label} ({len(rels)})")

    lim = (1e-17, 3.0)
    ax.plot(lim, lim, color=C_GREY, linewidth=0.7, linestyle="-")
    ax.fill_between(lim, lim, [3.0, 3.0], color="#999999", alpha=0.08)
    ax.text(1.5e-9, 1.0, "outside tolerance", fontsize=6.4, color=C_GREY,
            ha="left", va="top", rotation=0)
    ax.text(3e-17, 3e-16, "exact to round-off", fontsize=6.2, color=C_GREY,
            ha="left", va="bottom")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("tolerance of the check")
    ax.set_ylabel("achieved relative error")
    ax.legend(loc="lower right", frameon=False, handletextpad=0.3,
              borderpad=0.2, labelspacing=0.3)
    ax.set_title(f"{total} closed-form checks, all satisfied", fontsize=7.6)
    save(fig, "fig_verification")
    return total


# --------------------------------------------------------------------------
def compute_devices(reuse=False):
    cache = os.path.join(DATA, "figdata.json")
    if reuse and os.path.exists(cache):
        return json.load(open(cache, encoding="utf-8"))

    out = {}

    # --- GaAs pn junction: I-V and local ideality factor ----------------
    dev = TwoTerminalDevice(
        [Layer(1000, "GaAs", Na=1e17), Layer(1000, "GaAs", Nd=1e17)],
        [OhmicContact(), OhmicContact()], T=300, N=1000, approx="kane")
    V, J = dev.get_iv(-0.5, 1.0, 31)
    out["iv"] = {"V": V.tolist(), "J": J.tolist()}

    # --- modulation-doped HEMT: quantum vs classical --------------------
    hemt = TwoTerminalDevice(
        [Layer(30, "AlGaAs", composition=0.30, Nd=1e18),
         Layer(10, "AlGaAs", composition=0.30),
         Layer(500, "GaAs")],
        [OhmicContact(), OhmicContact()], T=300, N=800, approx="kane")
    q = hemt.get_quantum_equilibrium(N=800, n_states=8)
    c = hemt.get_equilibrium(N=800)
    m = hemt.model(300.0, 800)
    out["hemt"] = {
        "x_nm": (m.x * 1e7).tolist(),
        "Ec": (m.CBO_Gamma - q.psi).tolist(),
        "Ev": (m.VBO - q.psi).tolist(),
        "n_class": c.n.tolist(),
        "n_quant": q.n_quantum.tolist(),
        "levels": q.subbands["levels_eV"],
        "bound": q.subbands["bound"],
        "E_cut": q.subbands["E_cut_eV"],
        "psi0": q.subbands["psi"][0] if q.subbands["psi"] else None,
        "sheet": q.subbands["sheet_density_cm2"],
    }
    json.dump(out, open(cache, "w", encoding="utf-8"))
    return out


def fig_iv(data):
    V = np.array(data["iv"]["V"])
    J = np.array(data["iv"]["J"])
    kT = KB_EV * 300.0
    pos = J > 0
    n_id = np.full_like(V, np.nan)
    n_id[pos] = (1.0 / kT) * np.gradient(V[pos], np.log(J[pos]))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W2, 62 * MM),
                                 gridspec_kw={"width_ratios": [1.25, 1]})
    a1.semilogy(V[V < 0], -J[V < 0], "o-", ms=2.2, lw=0.7, color=C_CLASS,
                mfc="none", label=r"reverse ($|J|$)")
    a1.semilogy(V[pos], J[pos], "s-", ms=2.2, lw=0.7, color="#c0392b",
                mfc="none", label="forward")
    a1.set_xlabel("bias (V)")
    a1.set_ylabel(r"current density (A cm$^{-2}$)")
    a1.set_ylim(1e-13, 1e2)
    a1.legend(frameon=False, loc="upper left")
    a1.set_title("(a) GaAs pn junction, 1 $\\mu$m + 1 $\\mu$m, 10$^{17}$ cm$^{-3}$",
                 fontsize=7.2, loc="left")

    band = (V > 0.15) & pos
    a2.plot(V[band], n_id[band], "-", lw=0.9, color=C_CLASS)
    a2.axhline(2.0, color=C_GREY, lw=0.6, ls=(0, (3, 2)))
    a2.axhline(1.0, color=C_GREY, lw=0.6, ls=(0, (3, 2)))
    a2.text(0.17, 2.05, "SRH limit $n=2$", fontsize=6.4, color=C_GREY)
    a2.text(0.17, 1.05, "diffusion limit $n=1$", fontsize=6.4, color=C_GREY)
    a2.set_xlabel("bias (V)")
    a2.set_ylabel("local ideality factor $n$")
    a2.set_ylim(0.8, 2.2)
    a2.set_title("(b) ideality factor", fontsize=7.2, loc="left")
    fig.tight_layout()
    save(fig, "fig_iv")


def fig_hemt(data):
    h = data["hemt"]
    x = np.array(h["x_nm"])
    Ec = np.array(h["Ec"])
    Ev = np.array(h["Ev"])
    nc = np.array(h["n_class"])
    nq = np.array(h["n_quant"])
    levels = np.array(h["levels"])
    bound = np.array(h["bound"])
    E_cut = h["E_cut"]
    i_well = int(np.argmin(Ec))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W2, 66 * MM))

    a1.plot(x, Ec, color=C_CLASS, lw=1.0, label=r"$E_c$")
    a1.plot(x, Ev, color="#c0392b", lw=1.0, label=r"$E_v$")
    for e in levels[bound]:
        a1.plot([37, 44], [e, e], color=C_QUANT, lw=1.6, solid_capstyle="butt")
        a1.annotate(r"$E_0$", xy=(44, e), xytext=(45.0, e - 0.035), fontsize=7,
                    color=C_QUANT, va="center")
    a1.axhline(E_cut, color=C_GREY, lw=0.6, ls=":")
    a1.text(66, E_cut + 0.012, r"$E_{\rm cut}$", fontsize=6.6, color=C_GREY)
    a1.annotate("well bottom\n(interface)", xy=(x[i_well], Ec[i_well]),
                xytext=(52, -0.28), fontsize=6.4, color=C_GREY,
                arrowprops=dict(arrowstyle="-|>", lw=0.6, color=C_GREY))
    a1.set_xlim(25, 80)
    a1.set_ylim(-0.34, 0.30)
    a1.set_xlabel("depth (nm)")
    a1.set_ylabel("energy (eV)")
    a1.legend(frameon=False, loc="upper right", handlelength=1.2)
    a1.set_title("(a) conduction- and valence-band edges", fontsize=7.2,
                 loc="left")

    a2.plot(x, nc, color=C_GREY, lw=1.0, label="classical")
    a2.plot(x, nq, color=C_QUANT, lw=1.2, label="Schrödinger–Poisson")
    a2.axvline(x[i_well], color=C_GREY, lw=0.5, ls=":")
    a2.set_xlim(30, 70)
    a2.set_ylim(0.0, 1.30e18)
    a2.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
    a2.set_xlabel("depth (nm)")
    a2.set_ylabel(r"electron density (cm$^{-3}$)")
    a2.legend(frameon=False, loc="upper right", handlelength=1.2)
    a2.set_title(f"(b) channel, $n_s$ = {h['sheet']:.2e} cm$^{{-2}}$",
                 fontsize=7.2, loc="left")
    fig.tight_layout()
    save(fig, "fig_hemt")


def fig_cost():
    b = json.load(open(os.path.join(DATA, "bench.json"), encoding="utf-8"))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W2, 58 * MM),
                                 gridspec_kw={"width_ratios": [1, 1.15]})

    N = np.array([r["N"] for r in b["scaling"]])
    t = np.array([r["equilibrium_s"] for r in b["scaling"]])
    a1.loglog(N, t, "o-", ms=3, lw=0.8, color=C_CLASS, mfc="white")
    a1.set_xticks(N)
    a1.set_xticklabels([f"{n}" for n in N])
    a1.minorticks_off()
    a1.set_xlabel("grid nodes $N$")
    a1.set_ylabel("equilibrium solve time (s)")
    a1.set_title("(a) cost of the nonlinear Poisson solve", fontsize=7.2,
                 loc="left")
    a1.grid(True, which="major", lw=0.3, color="#dddddd")

    labels = ["equilibrium\n$N$=1000", "drift--diff\n0.8 V",
              "I--V point\n(mean)", "quantum\nequilibrium"]
    vals = [b["timing"]["equilibrium_pn_N1000_s"],
            b["timing"]["drift_diffusion_pn_0.8V_N1000_s"],
            b["timing"]["iv_per_point_s"],
            b["timing"]["quantum_hemt_N1000_s"]]
    colors = [C_CLASS, "#c0392b", "#c0392b", C_QUANT]
    bars = a2.bar(range(len(vals)), vals, color=colors, width=0.55,
                  edgecolor="black", linewidth=0.4)
    for r, v in zip(bars, vals):
        # two decimals below 1 s so the bar matches the text (0.35, not 0.4)
        label = f"{v:.2f}" if v < 1.0 else f"{v:.1f}"
        a2.text(r.get_x() + r.get_width() / 2, v * 1.15, label,
                ha="center", fontsize=6.4)
    a2.set_yscale("log")
    a2.set_xticks(range(len(vals)))
    a2.set_xticklabels(labels, fontsize=5.8)
    a2.set_ylabel("wall time (s)")
    a2.set_ylim(0.1, 100)
    a2.set_title("(b) representative operating points", fontsize=7.2, loc="left")
    fig.tight_layout()
    save(fig, "fig_cost")


def fig_noise():
    """J_spread: how much of the reported current is arithmetic noise."""
    b = json.load(open(os.path.join(DATA, "bench.json"), encoding="utf-8"))
    V = np.array([r["V"] for r in b["spread"]])
    J = np.array([abs(r["J"]) for r in b["spread"]])
    s = np.array([r["J_spread"] for r in b["spread"]])
    it = np.array([r["gummel_iterations"] for r in b["spread"]])

    fig, ax = plt.subplots(figsize=(W1, 62 * MM))
    ax.semilogy(V, J, "o-", ms=3, lw=0.9, color="#c0392b", mfc="white",
                label="$|J|$ (reported)")
    ax.semilogy(V, J * s, "s--", ms=3, lw=0.8, color=C_GREY, mfc="white",
                label=r"worst link, $|J|\,(1+$spread$)$")
    for v, j, sp, n in zip(V, J, s, it):
        ax.annotate(f"{n}", (v, j), textcoords="offset points", xytext=(0, -11),
                    ha="center", fontsize=5.8, color=C_GREY)
    ax.axhline(0, color="none")
    ax.set_xlabel("bias (V)")
    ax.set_ylabel(r"current density (A cm$^{-2}$)")
    ax.set_ylim(1e-14, 1e-1)
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("discrete-current spread (labels: Gummel sweeps)",
                 fontsize=7.2)
    save(fig, "fig_noise")


if __name__ == "__main__":
    reuse = "--reuse" in sys.argv
    print("figures:")
    fig_architecture()
    n = fig_verification()
    print(f"  verification checks parsed: {n}")
    data = compute_devices(reuse)
    fig_iv(data)
    fig_hemt(data)
    fig_cost()
    fig_noise()
    print("done")
