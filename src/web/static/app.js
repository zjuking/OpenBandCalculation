/* OBPDS workbench front end.
 *
 * Self-contained: no CDN, no build step.  Charts are hand-rolled SVG so the
 * page works offline and inside any sandboxed preview.
 */
"use strict";

/* ───────────────────────────── helpers ─────────────────────────────────── */

const $ = (id) => document.getElementById(id);

function fmt(v, sig = 3) {
  if (v === null || v === undefined || !isFinite(v)) return "—";
  const a = Math.abs(v);
  if (a === 0) return "0";
  if (a >= 1e5 || a < 1e-3) {
    const e = Math.floor(Math.log10(a));
    return (v / 10 ** e).toFixed(sig - 1) + "e" + (e >= 0 ? "+" : "") + e;
  }
  return v.toFixed(Math.max(0, sig - 1 - Math.floor(Math.log10(a))));
}

async function api(path, body) {
  const opt = { headers: { "Content-Type": "application/json" } };
  if (body !== undefined) { opt.method = "POST"; opt.body = JSON.stringify(body); }
  const r = await fetch(path, opt);
  if (!r.ok) throw new Error(`${r.status}: ${(await r.text()).slice(0, 200)}`);
  return r.json();
}

function setStatus(text, cls = "") {
  const s = $("status");
  s.textContent = text;
  s.className = "status " + cls;
}

/* ──────────────────────────── chart engine ─────────────────────────────── */

const NS = "http://www.w3.org/2000/svg";
const el = (n, a = {}) => {
  const e = document.createElementNS(NS, n);
  for (const k in a) e.setAttribute(k, a[k]);
  return e;
};

function niceTicks(lo, hi, n = 6) {
  if (!(hi > lo)) return [lo];
  const raw = (hi - lo) / n;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const norm = raw / mag;
  const step = (norm < 1.5 ? 1 : norm < 3 ? 2 : norm < 7 ? 5 : 10) * mag;
  const out = [];
  for (let t = Math.ceil(lo / step) * step; t <= hi + step * 1e-9; t += step)
    out.push(Math.abs(t) < step * 1e-9 ? 0 : t);
  return out;
}

function decadeTicks(lo, hi) {
  const out = [];
  for (let d = Math.floor(lo); d <= Math.ceil(hi); d++) {
    out.push({ v: d, major: true });
    for (const m of [0.301, 0.699, 0.903]) if (d + m > lo && d + m < hi)
      out.push({ v: d + m, major: false });
  }
  return out;
}

/**
 * Draw a line chart into `host`.
 *
 * cfg: { x, series[], xlabel, ylabel, ylog, y2log, bands[], y2label,
 *        ymin, ymax, title }
 * series: { name, data, color, dash, width, axis ('y' | 'y2'), hidden }
 */
function drawChart(host, cfg) {
  host.innerHTML = "";
  const W = Math.max(host.clientWidth, 320);
  const H = Math.max(host.clientHeight, 200);
  const m = { l: 66, r: cfg.series.some(s => s.axis === "y2") ? 62 : 20,
              t: 14, b: 42 };
  const iw = W - m.l - m.r, ih = H - m.t - m.b;
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, width: W, height: H });
  host.appendChild(svg);

  const vis = cfg.series.filter(s => !s.hidden && s.data && s.data.length);
  if (!vis.length || !cfg.x || !cfg.x.length) {
    svg.appendChild(el("text", { x: W / 2, y: H / 2, fill: "#8b98b0",
      "text-anchor": "middle", "font-size": 12 }))
      .textContent = "No data — solve on the left first";
    return;
  }

  let x0 = Math.min(...cfg.x), x1 = Math.max(...cfg.x);
  if (x1 - x0 < 1e-12) { x0 -= 1; x1 += 1; }
  const sx = (v) => m.l + (v - x0) / (x1 - x0) * iw;

  const span = (axis) => {
    let lo = Infinity, hi = -Infinity;
    for (const s of vis) {
      if ((s.axis || "y") !== axis) continue;
      for (const v of s.data) {
        if (!isFinite(v)) continue;
        if (cfg.ylog || (axis === "y2" && cfg.y2log)) { if (v <= 0) continue; }
        lo = Math.min(lo, v); hi = Math.max(hi, v);
      }
    }
    if (!isFinite(lo)) return [0, 1];
    const log = cfg.ylog || (axis === "y2" && cfg.y2log);
    if (log) {
      const a = Math.log10(lo), b = Math.log10(hi);
      const pad = Math.max((b - a) * 0.08, 0.35);
      return [a - pad, b + pad];
    }
    const pad = Math.max((hi - lo) * 0.07, 1e-12);
    return [lo - pad, hi + pad];
  };

  const [ya, yb] = cfg.ymin !== undefined ? [cfg.ymin, cfg.ymax] : span("y");
  const [y2a, y2b] = span("y2");
  const sy = (v) => {
    if (cfg.ylog) {
      const t = (Math.log10(Math.max(v, 1e-300)) - ya) / (yb - ya);
      return m.t + ih - Math.max(-0.2, Math.min(1.2, t)) * ih;
    }
    return m.t + ih - (v - ya) / ((yb - ya) || 1) * ih;
  };
  const sy2 = (v) => {
    if (cfg.y2log) {
      const t = (Math.log10(Math.max(v, 1e-300)) - y2a) / (y2b - y2a);
      return m.t + ih - Math.max(-0.2, Math.min(1.2, t)) * ih;
    }
    return m.t + ih - (v - y2a) / ((y2b - y2a) || 1) * ih;
  };

  /* shaded layer bands */
  (cfg.bands || []).forEach((b, i) => {
    const a = sx(Math.max(b.x0, x0)), c = sx(Math.min(b.x1, x1));
    if (c <= a) return;
    svg.appendChild(el("rect", {
      x: a, y: m.t, width: c - a, height: ih,
      fill: i % 2 ? "#ffffff07" : "#4da3ff0c", stroke: "#26314a",
      "stroke-dasharray": "3 3",
    }));
    if (c - a > 46) {
      const t = el("text", { x: (a + c) / 2, y: m.t + 12, fill: "#7f8ca4",
        "font-size": 10, "text-anchor": "middle" });
      t.textContent = b.label;
      svg.appendChild(t);
    }
  });

  /* grid + y axis */
  const yTicks = cfg.ylog ? decadeTicks(ya, yb) : niceTicks(ya, yb, 6).map(v => ({ v, major: true }));
  for (const tk of yTicks) {
    const v = cfg.ylog ? 10 ** tk.v : tk.v;
    const y = sy(v);
    if (y < m.t - 2 || y > m.t + ih + 2) continue;
    svg.appendChild(el("line", { x1: m.l, x2: m.l + iw, y1: y, y2: y,
      stroke: tk.major ? "#1e2637" : "#161d2b", "stroke-width": 1 }));
    if (tk.major) {
      const t = el("text", { x: m.l - 8, y: y + 3.5, fill: "#8b98b0",
        "font-size": 10.5, "text-anchor": "end" });
      t.textContent = cfg.ylog
        ? (tk.v >= 0 ? "1e" + tk.v : "1e" + tk.v)
        : fmt(v, 3);
      svg.appendChild(t);
    }
  }
  /* x axis */
  for (const v of niceTicks(x0, x1, 7)) {
    const x = sx(v);
    svg.appendChild(el("line", { x1: x, x2: x, y1: m.t, y2: m.t + ih,
      stroke: "#161d2b" }));
    const t = el("text", { x, y: m.t + ih + 16, fill: "#8b98b0",
      "font-size": 10.5, "text-anchor": "middle" });
    t.textContent = fmt(v, 4);
    svg.appendChild(t);
  }
  /* right axis */
  if (vis.some(s => s.axis === "y2")) {
    for (const tk of (cfg.y2log ? decadeTicks(y2a, y2b) : niceTicks(y2a, y2b, 5).map(v => ({ v, major: true })))) {
      const v = cfg.y2log ? 10 ** tk.v : tk.v;
      const y = sy2(v);
      if (y < m.t - 2 || y > m.t + ih + 2 || !tk.major) continue;
      const t = el("text", { x: m.l + iw + 8, y: y + 3.5, fill: "#8b98b0",
        "font-size": 10.5 });
      t.textContent = cfg.y2log ? "1e" + tk.v : fmt(v, 3);
      svg.appendChild(t);
    }
  }

  /* frame + labels */
  svg.appendChild(el("rect", { x: m.l, y: m.t, width: iw, height: ih,
    fill: "none", stroke: "#26314a" }));
  const xl = el("text", { x: m.l + iw / 2, y: H - 6, fill: "#8b98b0",
    "font-size": 11, "text-anchor": "middle" });
  xl.textContent = cfg.xlabel || "";
  svg.appendChild(xl);
  const yl = el("text", { x: 14, y: m.t + ih / 2, fill: "#8b98b0",
    "font-size": 11, "text-anchor": "middle",
    transform: `rotate(-90 14 ${m.t + ih / 2})` });
  yl.textContent = cfg.ylabel || "";
  svg.appendChild(yl);

  /* series */
  for (const s of vis) {
    const right = s.axis === "y2";
    const f = right ? sy2 : sy;
    let d = "", pen = false;
    for (let i = 0; i < cfg.x.length; i++) {
      const v = s.data[i];
      if (!isFinite(v) || (cfg.ylog && !right && v <= 0) ||
          (cfg.y2log && right && v <= 0)) { pen = false; continue; }
      d += (pen ? "L" : "M") + sx(cfg.x[i]).toFixed(2) + " " + f(v).toFixed(2) + " ";
      pen = true;
    }
    svg.appendChild(el("path", { d, fill: "none", stroke: s.color,
      "stroke-width": s.width || 1.7,
      "stroke-dasharray": s.dash || "", "stroke-linejoin": "round" }));
  }

  /* hover crosshair */
  const cl = el("line", { y1: m.t, y2: m.t + ih, stroke: "#4da3ff",
    "stroke-dasharray": "2 3", opacity: 0 });
  svg.appendChild(cl);
  const hit = el("rect", { x: m.l, y: m.t, width: iw, height: ih,
    fill: "transparent" });
  svg.appendChild(hit);
  hit.addEventListener("mousemove", (ev) => {
    const r = svg.getBoundingClientRect();
    const px = (ev.clientX - r.left) / r.width * W;
    let best = 0, bd = Infinity;
    for (let i = 0; i < cfg.x.length; i++) {
      const d = Math.abs(sx(cfg.x[i]) - px);
      if (d < bd) { bd = d; best = i; }
    }
    cl.setAttribute("x1", sx(cfg.x[best]));
    cl.setAttribute("x2", sx(cfg.x[best]));
    cl.setAttribute("opacity", 1);
    const tip = $("tip");
    let html = `<div><span class="k">x = </span><b>${fmt(cfg.x[best], 4)}</b> nm</div>`;
    for (const s of vis) {
      const v = s.data[best];
      if (v === undefined || !isFinite(v)) continue;
      html += `<div><span class="k">${s.name} = </span>` +
        `<b style="color:${s.color}">${fmt(v, 4)}</b></div>`;
    }
    tip.innerHTML = html;
    tip.classList.add("show");
    const w = tip.offsetWidth;
    tip.style.left = Math.min(ev.clientX + 14, window.innerWidth - w - 8) + "px";
    tip.style.top = (ev.clientY + 14) + "px";
  });
  hit.addEventListener("mouseleave", () => {
    cl.setAttribute("opacity", 0);
    $("tip").classList.remove("show");
  });

  /* legend with toggles */
  const lg = $(cfg.legend);
  if (lg) {
    lg.innerHTML = "";
    cfg.series.forEach((s, i) => {
      const span = document.createElement("span");
      span.style.cursor = "pointer";
      span.style.opacity = s.hidden ? .35 : 1;
      span.innerHTML = `<i style="background:${s.color}"></i>${s.name}`;
      span.onclick = () => {
        s.hidden = !s.hidden;
        drawChart(host, cfg);
      };
      lg.appendChild(span);
    });
  }
}

/* ─────────────────────────── device state ──────────────────────────────── */

const state = {
  materials: [],          // catalogue from /api/materials
  layers: [],             // { thickness_nm, material, composition,
                          //   composition_end, doping }
  solution: null,
  sweep: null,            // { V, J } or { V, C }
  charts: { band: null, carrier: null, iv: null, cv: null, wf: null },
};

function readDevice() {
  return {
    layers: state.layers.map(l => ({ ...l })),
    contacts: [
      { kind: $("cLeft").value, work_function: +$("wfLeft").value || 4.9 },
      { kind: $("cRight").value, work_function: +$("wfRight").value || 4.9 },
    ],
    T: +$("T").value || 300,
    N: +$("N").value || 400,
    approx: $("approx").value,
    Fp: $("pSide").value === "right" ? "right" : "left",
    Fn: $("pSide").value === "right" ? "left" : "right",
    substrate: $("substrate").value || null,
    substrate_a: +$("substrateA").value || null,
    substrate_composition: null,
    tau_n: +$("tauN").value || 1e-7,
    tau_p: +$("tauP").value || 1e-7,
    B_rad: 10 ** (+$("brad").value),
  };
}

function writeDevice(spec) {
  state.layers = (spec.layers || []).map(l => ({ ...l }));
  const c = spec.contacts || [];
  $("cLeft").value = (c[0] && c[0].kind) || "ohmic";
  $("cRight").value = (c[1] && c[1].kind) || "ohmic";
  $("wfLeft").value = (c[0] && c[0].work_function) || 4.9;
  $("wfRight").value = (c[1] && c[1].work_function) || 4.9;
  $("T").value = spec.T ?? 300;
  $("N").value = spec.N ?? 400;
  $("approx").value = spec.approx || "kane";
  $("pSide").value = spec.Fp === "right" ? "right" : "left";
  $("quantum").checked = spec.quantum === true;
  $("substrate").value = spec.substrate || "";
  $("substrateA").value = spec.substrate_a || "";
  $("tauN").value = spec.tau_n ?? 1e-7;
  $("tauP").value = spec.tau_p ?? 1e-7;
  const e = Math.log10(spec.B_rad || 1e-10);
  $("brad").value = Math.max(-14, Math.min(-7, e));
  $("brad").dispatchEvent(new Event("input"));
  renderLayers();
}

/* ─────────────────────────── layer editor ──────────────────────────────── */

function materialOptions(sel, cur) {
  sel.innerHTML = state.materials.map(m =>
    `<option value="${m.name}"${m.name === cur ? " selected" : ""}>${m.name}` +
    `${m.ternary ? " (x)" : ""}</option>`).join("");
}

function renderLayers() {
  const host = $("layers");
  host.innerHTML = "";
  state.layers.forEach((L, i) => {
    const meta = state.materials.find(m => m.name === L.material) || {};
    const ternary = !!meta.ternary;
    const d = L.doping || 0;
    const type = d > 0 ? "p" : d < 0 ? "n" : "i";
    const exp = d ? Math.log10(Math.abs(d)) : 16;
    const box = document.createElement("div");
    box.className = "layer";
    box.innerHTML = `
      <div class="head">
        <span class="idx">Layer ${i + 1}</span>
        <span>
          <button class="btn tiny" data-act="up">↑</button>
          <button class="btn tiny" data-act="down">↓</button>
          <button class="btn tiny" data-act="del">✕</button>
        </span>
      </div>
      <div class="fields">
        <label class="span2">Material
          <select data-f="material"></select>
        </label>
        ${ternary ? `
        <label>x start <input type="number" step="0.01" min="0" max="1"
               value="${L.composition ?? 0.3}" data-f="composition"></label>
        <label>x end <input type="number" step="0.01" min="0" max="1"
               value="${L.composition_end ?? (L.composition ?? 0.3)}"
               data-f="composition_end"></label>` : ""}
        <label>Thickness (nm)
          <input type="number" step="5" min="0.5"
                 value="${L.thickness_nm}" data-f="thickness_nm"></label>
        <label>Doping
          <div class="doping">
            <select data-f="type">
              <option value="p"${type === "p" ? " selected" : ""}>p</option>
              <option value="n"${type === "n" ? " selected" : ""}>n</option>
              <option value="i"${type === "i" ? " selected" : ""}>i</option>
            </select>
            <div class="exp">
              <input type="range" min="13" max="20" step="0.1" value="${exp}"
                     data-f="exp">
              <span data-f="expv">${d ? fmt(Math.abs(d), 2) : "—"}</span>
            </div>
          </div>
        </label>
      </div>
      <div class="mini" data-f="info"></div>`;
    host.appendChild(box);

    const sel = box.querySelector('[data-f="material"]');
    materialOptions(sel, L.material);
    sel.onchange = () => {
      const m = state.materials.find(x => x.name === sel.value);
      L.material = sel.value;
      if (m && m.ternary) L.composition = L.composition ?? 0.3;
      else { delete L.composition; delete L.composition_end; }
      renderLayers();
    };
    box.querySelectorAll("[data-f]").forEach(inp => {
      const f = inp.dataset.f;
      if (["material", "type", "exp", "expv", "info"].includes(f)) return;
      inp.oninput = () => { L[f] = +inp.value; refreshInfo(box, L); };
    });
    const ts = box.querySelector('[data-f="type"]');
    const ex = box.querySelector('[data-f="exp"]');
    const sync = () => {
      const t = ts.value;
      const mag = 10 ** (+ex.value);
      L.doping = t === "p" ? mag : t === "n" ? -mag : 0;
      box.querySelector('[data-f="expv"]').textContent =
        t === "i" ? "—" : fmt(mag, 2);
    };
    ts.onchange = sync; ex.oninput = sync;

    box.querySelector('[data-act="del"]').onclick = () => {
      state.layers.splice(i, 1); renderLayers();
    };
    box.querySelector('[data-act="up"]').onclick = () => {
      if (i > 0) { [state.layers[i - 1], state.layers[i]] =
        [state.layers[i], state.layers[i - 1]]; renderLayers(); }
    };
    box.querySelector('[data-act="down"]').onclick = () => {
      if (i < state.layers.length - 1) { [state.layers[i + 1], state.layers[i]] =
        [state.layers[i], state.layers[i + 1]]; renderLayers(); }
    };
    refreshInfo(box, L);
  });
}

async function refreshInfo(box, L) {
  const meta = state.materials.find(m => m.name === L.material);
  const info = box.querySelector('[data-f="info"]');
  if (!meta) { info.textContent = ""; return; }
  try {
    const p = await api(`/api/properties?material=${encodeURIComponent(L.material)}` +
      `&composition=${L.composition ?? ""}&T=${+$("T").value || 300}`);
    info.textContent = `Eg ${fmt(p.Eg, 4)} eV · ${p.direct ? "direct" : "indirect"} gap` +
      ` · a ${fmt(p.lattice_constant, 5)} Å · εr ${fmt(p.dielectric, 3)}` +
      ` · μn ${fmt(p.mu_n, 3)} / μp ${fmt(p.mu_p, 3)} cm²/Vs`;
  } catch { info.textContent = `Eg ${fmt(meta.Eg, 4)} eV`; }
}

/* ──────────────────────────── job polling ──────────────────────────────── */

function busy(on, text = "Computing…", frac = null) {
  const b = $("busy");
  b.classList.toggle("show", on);
  $("busyText").textContent = text;
  b.querySelector("i").style.width = (frac === null ? 100 : frac * 100) + "%";
}

async function runJob(kind, payload, label) {
  busy(true, label, 0);
  setStatus(label, "warn");
  const { job_id } = await api(payload.url, payload.body);
  for (;;) {
    await new Promise(r => setTimeout(r, 250));
    const j = await api(`/api/job/${job_id}`);
    if (j.status === "error") {
      busy(false); setStatus("Solve failed", "err");
      throw new Error(j.error || "unknown solver error");
    }
    busy(true, `${label}  ${j.progress}/${j.total}`,
         j.total ? j.progress / j.total : 0);
    if (j.status === "done") { busy(false); return j.result; }
  }
}

/* ────────────────────────────── solving ────────────────────────────────── */

/* Quantum correction is a self-consistent equilibrium calculation, so it
   only has meaning for the band-diagram mode.  Read it once here and let
   the caller decide what to do when the mode cannot use it. */
function quantumWanted() {
  return $("quantum").checked;
}

async function doSolve(mode) {
  const V = +$("V").value || 0;
  const q = quantumWanted();
  if (q && mode !== "equilibrium") {
    setStatus("Quantum correction is only available in equilibrium band mode; solving classically this time", "warn");
  }
  try {
    const res = await runJob("solve",
      { url: "/api/solve", body: { device: readDevice(), mode, V,
                                   downsample: 800,
                                   quantum: q && mode === "equilibrium",
                                   n_states: +$("nStates").value || 8 } },
      mode === "equilibrium" ? (q ? "Equilibrium bands (quantum)" : "Equilibrium bands") :
      mode === "zero_current" ? `Zero current V=${V} V` : `Drift-diffusion V=${V} V`);
    state.solution = res;
    state.solution.mode = mode;
    drawSolution();
    const ok = res.converged !== false;
    const tag = res.n_is_quantum ? " · quantum" : "";
    setStatus(ok ? `Done · ${mode}${tag} · V=${V} V`
                 : `Not converged · ${mode} · V=${V} V`,
              ok ? "ok" : "warn");
  } catch (e) {
    setStatus("Error: " + e.message, "err");
  }
}

async function doSweep(kind) {
  const body = {
    device: readDevice(),
    kind,
    Vstart: +$("ivStart").value,
    Vstop: +$("ivStop").value,
    Vnum: +$("ivNum").value,
  };
  try {
    const res = await runJob("sweep", { url: "/api/sweep", body },
      kind === "iv" ? "I–V sweep" : "C–V sweep");
    state.sweep = { kind, ...res };
    if (kind === "iv") { drawIV(); switchTab("iv"); }
    else { drawCV(); switchTab("cv"); }
    setStatus(`${kind.toUpperCase()} sweep complete`, "ok");
  } catch (e) {
    setStatus("Error: " + e.message, "err");
  }
}

/* ───────────────────────────── rendering ───────────────────────────────── */

/* Strain splits the valence band.  Below a couple of meV the two edges are
   indistinguishable on screen and drawing them only adds clutter, so they
   replace the single Ev line only when the splitting is real. */
function valenceSplit(s) {
  if (!s.Ev_HH || !s.Ev_LH) return 0.0;
  let d = 0.0;
  for (let i = 0; i < s.Ev_HH.length; i++) {
    const v = Math.abs(s.Ev_HH[i] - s.Ev_LH[i]);
    if (v > d) d = v;
  }
  return d;
}

function bandSeries(s) {
  const split = valenceSplit(s);
  const hh_lh = split > 2e-3;              // 2 meV
  const out = [
    { name: "Ec (Γ)", data: s.Ec_Gamma, color: "#4da3ff", width: 2 },
    { name: "Ec (L)", data: s.Ec_L, color: "#a78bfa", width: 1.2, dash: "5 3" },
    { name: "Ec (X)", data: s.Ec_X, color: "#f472b6", width: 1.2, dash: "2 3" },
  ];
  if (hh_lh) {
    out.push({ name: "Ev (heavy hole)", data: s.Ev_HH, color: "#ff6b6b", width: 2 });
    out.push({ name: "Ev (light hole)", data: s.Ev_LH, color: "#ff9f7a",
               width: 1.6, dash: "6 3" });
  } else {
    out.push({ name: "Ev", data: s.Ev, color: "#ff6b6b", width: 2 });
  }
  out.push(
    { name: "Ei", data: s.Ei, color: "#7f8ca4", width: 1, dash: "1 3" },
    { name: "Fn", data: s.Fn, color: "#37d67a", width: 1.8 },
    { name: "Fp", data: s.Fp, color: "#ffb648", width: 1.8 },
  );
  return out.concat(quantSeries(s));
}

function quantSeries(s) {
  const out = [];
  // Quantized levels are flat across the device, so they go in as constant
  // series -- the chart needs no special case for horizontal lines.
  const sub = s.subbands;
  if (sub && sub.levels_eV) {
    const n = (s.x_nm || []).length;
    sub.levels_eV.forEach((E, i) => {
      if (!sub.bound || !sub.bound[i]) return;
      out.push({
        name: `E${i} subband`, data: new Array(n).fill(E),
        color: "#ffd166", width: 1.6, dash: "6 3",
      });
    });
  }
  return out;
}

function drawSolution() {
  const s = state.solution;
  if (!s) return;
  const bands = (s.layers || []).map(b =>
    ({ x0: b.x0_nm, x1: b.x1_nm, label: b.label }));

  state.charts.band = {
    x: s.x_nm, series: bandSeries(s), bands,
    xlabel: "Depth (nm)", ylabel: "Energy (eV)", legend: "legendBand",
  };
  drawChart($("chartBand"), state.charts.band);

  state.charts.carrier = {
    x: s.x_nm, ylog: true,
    series: [
      { name: "n", data: s.n, color: "#4da3ff", width: 2 },
      { name: "p", data: s.p, color: "#ff6b6b", width: 2 },
      { name: "n (L valley)", data: s.n_L, color: "#a78bfa", width: 1.1, dash: "4 3" },
      { name: "n (X valley)", data: s.n_X, color: "#f472b6", width: 1.1, dash: "2 3" },
      { name: "|E|", data: (s.field || []).map(Math.abs),
        color: "#37d67a", width: 1.4, axis: "y2" },
    ], bands, xlabel: "Depth (nm)", ylabel: "Concentration (cm⁻³)",
    y2label: "|E| (V/cm)", legend: "legendCarrier",
  };
  drawChart($("chartCarrier"), state.charts.carrier);

  /* readouts */
  const dE = s.Ec_Gamma ? Math.min(...s.Ec_Gamma) - Math.max(...s.Ev) : NaN;
  const Emax = s.field ? Math.max(...s.field.map(Math.abs)) : NaN;
  $("readBand").innerHTML =
    `<b>${s.mode || ""}</b> · V = ${fmt(s.V, 3)} V · T = ${s.meta.T} K · ` +
    `kT = ${fmt(s.meta.kT, 4)} eV · N = ${s.meta.N} · ${s.meta.approx} · ` +
    `Mesh step ${fmt((s.total_nm || 0) / s.meta.N, 3)} nm · ` +
    `min gap ${fmt(dE, 4)} eV · |E|max ${fmt(Emax, 4)} V/cm` +
    (s.converged === false ? ` · <span style="color:#ff6b6b">not converged</span>` : "");

  const split = valenceSplit(s);
  if (split > 2e-3) {
    const heavy = s.Ev_HH[0] >= s.Ev_LH[0];
    $("readBand").innerHTML +=
      ` · <b style="color:#ff9f7a">pseudomorphic strain</b>: HH–LH split ` +
      `<b>${fmt(split * 1000, 1)}</b> meV (${heavy ? "heavy hole" : "light hole"} on top)`;
  }

  const sub = s.subbands;
  if (sub && sub.levels_eV) {
    const lv = sub.levels_eV.filter((_, i) => !sub.bound || sub.bound[i]);
    $("readBand").innerHTML += lv.length
      ? ` · <b style="color:#ffd166">quantum correction</b>: ${lv.length} bound subbands ` +
        `(${lv.map(e => fmt(e, 4)).join(", ")} eV) · sheet density ` +
        `<b>${fmt(sub.sheet_density_cm2, 4)}</b> cm⁻²`
      : ` · <b style="color:#ffd166">quantum correction</b>: no bound states (this structure ` +
        `confines no carriers, so the density falls back to the classical result)`;
  }

  if (s.J !== undefined) {
    $("readCarrier").innerHTML =
      `J = <b>${fmt(s.J, 4)}</b> A/cm² · Jn(contact) ${fmt(s.Jn_contact, 3)} · ` +
      `Jp(contact) ${fmt(s.Jp_contact, 3)} · Gummel iterations ${s.gummel_iterations}`;
  } else if (s.Jtotal) {
    const j = s.Jtotal[Math.floor(s.Jtotal.length / 2)];
    $("readCarrier").innerHTML = `J(midpoint) = <b>${fmt(j, 4)}</b> A/cm²`;
  } else {
    $("readCarrier").innerHTML = "";
  }

  // Redraw only if the tab is on screen: drawChart measures its host, and a
  // display:none container measures zero.
  if ($("tab-wf").classList.contains("active")) drawWF();
  drawTable();
}

function drawIV() {
  const sw = state.sweep;
  if (!sw || sw.kind !== "iv") return;
  const V = sw.V, J = sw.J;
  const log = $("ivLog").checked;
  const data = log ? J.map(v => (isFinite(v) && v !== 0 ? Math.abs(v) : NaN)) : J;
  state.charts.iv = {
    x: V, ylog: log,
    series: [{ name: log ? "|J|" : "J", data, color: "#37d67a", width: 2 }],
    xlabel: "Bias V (V)", ylabel: log ? "|J| (A/cm²)" : "J (A/cm²)",
  };
  drawChart($("chartIV"), state.charts.iv);

  let txt = `${V.length} points · N=${sw.meta ? sw.meta.N : "—"} · ` +
    `V ∈ [${fmt(V[0], 3)}, ${fmt(V[V.length - 1], 3)}] V`;
  if (sw.ideality && sw.ideality.length) {
    txt += " · ideality factor " + sw.ideality.map(s =>
      `${fmt(s.range[0], 2)}–${fmt(s.range[1], 2)} V: <b>${fmt(s.ideality, 3)}</b>`
    ).join(", ");
  }
  $("readIV").innerHTML = txt;
}

function drawCV() {
  const sw = state.sweep;
  if (!sw || sw.kind !== "cv") return;
  const V = sw.V, C = sw.C;
  state.charts.cv = {
    x: V,
    series: [
      { name: "1/C²", data: C.map(c => (c > 0 ? 1 / (c * c) : NaN)),
        color: "#4da3ff", width: 2 },
      { name: "C", data: C, color: "#ffb648", width: 1.6, axis: "y2" },
    ],
    xlabel: "Bias V (V)", ylabel: "1/C² (cm⁴/F²)", y2label: "C (F/cm²)",
  };
  drawChart($("chartCV"), state.charts.cv);
  const ok = C.filter(isFinite);
  $("readCV").innerHTML = ok.length
    ? `C ∈ [${fmt(Math.min(...ok), 3)}, ${fmt(Math.max(...ok), 3)}] F/cm²`
    : "no finite results";
}

const WF_COLORS = ["#ffd166", "#4da3ff", "#37d67a", "#f472b6",
                   "#a78bfa", "#ff6b6b", "#7dd3fc", "#fb923c"];

/* |psi|^2 of the bound subbands.  Only meaningful after a quantum solve;
   the classical path ships no wavefunctions, so say so instead of drawing
   an empty axis. */
function drawWF() {
  const s = state.solution;
  const host = $("chartWF");
  if (!s) return;
  const sub = s.subbands;
  const idx = [];
  if (sub && sub.levels_eV) {
    sub.levels_eV.forEach((E, i) => { if (!sub.bound || sub.bound[i]) idx.push(i); });
  }
  const psi = (sub && sub.psi) || [];
  if (!psi.length || !(s.x_nm || []).length || psi[0].length !== s.x_nm.length) {
    state.charts.wf = null;
    host.innerHTML =
      `<div class="hint">Tick "quantum correction" and re-solve the equilibrium bands;<br>` +
      `the |ψ|² distribution of each bound subband will be drawn here.</div>`;
    $("legendWF").innerHTML = "";
    $("readWF").innerHTML = "";
    return;
  }
  const bands = (s.layers || []).map(b =>
    ({ x0: b.x0_nm, x1: b.x1_nm, label: b.label }));
  const series = psi.map((row, k) => {
    const sq = row.map(v => v * v);
    const mx = Math.max(...sq);
    return {
      name: `E${idx[k]} · ${fmt(sub.levels_eV[idx[k]], 4)} eV`,
      data: mx > 0 ? sq.map(v => v / mx) : sq,
      color: WF_COLORS[k % WF_COLORS.length], width: 1.8,
    };
  });
  state.charts.wf = { x: s.x_nm, series, bands,
                      xlabel: "Depth (nm)", ylabel: "|ψ|² (each normalised)",
                      legend: "legendWF" };
  drawChart(host, state.charts.wf);

  const pop = sub.population_cm2 || [];
  if (pop.length) {
    const tot = pop.reduce((a, b) => a + b, 0) || 1;
    $("readWF").innerHTML =
      `continuum threshold E<sub>cut</sub> = ${fmt(sub.E_cut_eV, 4)} eV · total sheet density ` +
      `<b>${fmt(sub.sheet_density_cm2, 4)}</b> cm⁻²<br>` +
      pop.map((p, k) =>
        `E${idx[k]}: ${fmt(sub.levels_eV[idx[k]], 4)} eV · ` +
        `<b>${fmt(p, 4)}</b> cm⁻² (${fmt(100 * p / tot, 1)} %)`
      ).join(" · ");
  }
}

async function drawTable() {
  try {
    const { rows } = await api("/api/describe", readDevice());
    const strained = rows.some(r => r.substrate_a !== undefined);
    const head = ["Layer", "Thickness (nm)", "Eg (eV)", "ΔEc/ΔEg", "Doping (cm⁻³)",
                  "a (Å)", "Mismatch %", "εr"]
      .concat(strained ? ["ε∥ %", "ε⊥ %", "h_c (nm)"] : []);
    const cell = (v, d = 4) => `<td>${fmt(v, d)}</td>`;
    const body = rows.map(r => `<tr><td>${r.material}</td>` +
      cell(r.thickness_nm) + cell(r.Eg) + cell(r.Eg_end - r.Eg, 3) +
      cell(r.doping_cm3, 3) + cell(r.lattice_constant, 5) +
      cell(r.mismatch * 100, 3) + cell(r.dielectric) +
      (strained
        ? cell(r.eps_in * 100, 3) + cell(r.eps_out * 100, 3) +
          // a lattice-matched layer has no critical thickness at all, and
          // Python's inf does not survive JSON, so it arrives as null
          `<td>${r.h_c_nm === null ? "∞" : fmt(r.h_c_nm, 2)}` +
          `${r.pseudomorphic ? "" : " ⚠"}</td>`
        : "") +
      `</tr>`).join("");
    $("layerTable").innerHTML =
      `<table><thead><tr>${head.map(h => `<th>${h}</th>`).join("")}</tr></thead>` +
      `<tbody>${body}</tbody></table>` +
      (strained
        ? `<p class="note">Strain computed relative to substrate ${rows[0].substrate}` +
          ` (a = ${fmt(rows[0].substrate_a, 4)} Å), growth direction [001]; ` +
          `⚠ marks a layer thicker than the Matthews–Blakeslee critical thickness, ` +
          `which in practice is likely to be partly relaxed.` +
          (rows.some(r => r.strain_applied === false)
            ? `<br>⚠ Some materials lack deformation-potential data (the nitrides, for example); ` +
              `strain is computed but no band shift is applied.`
            : "") +
          `</p>`
        : "");
  } catch (e) { $("layerTable").textContent = "Layer parameters unavailable: " + e.message; }
}

/* ────────────────────────────── tabs ───────────────────────────────────── */

function switchTab(name) {
  document.querySelectorAll(".tab").forEach(t =>
    t.classList.toggle("active", t.dataset.tab === name));
  document.querySelectorAll(".tabpage").forEach(p =>
    p.classList.toggle("active", p.id === "tab-" + name));
  if (name === "band" && state.charts.band) drawChart($("chartBand"), state.charts.band);
  if (name === "carrier" && state.charts.carrier) drawChart($("chartCarrier"), state.charts.carrier);
  if (name === "iv") drawIV();
  if (name === "cv") drawCV();
  if (name === "wf") drawWF();
  if (name === "table") drawTable();
}

/* ────────────────────────────── wiring ─────────────────────────────────── */

async function init() {
  const mat = await api("/api/materials");
  state.materials = mat.materials;

  // Anything in the catalogue can be a wafer.  Ternaries need a composition
  // the layer editor does not offer here, so they are listed but resolve at
  // the default x = 0.5 unless the user pins a lattice constant instead.
  $("substrate").innerHTML =
    `<option value="">None (each layer keeps its own lattice constant)</option>` +
    mat.materials.map(m =>
      `<option value="${m.name}">${m.name}` +
      `${m.ternary ? " (x=0.5)" : ""}</option>`).join("");

  const pre = await api("/api/presets").catch(() => null);
  const sel = $("preset");
  if (pre) {
    sel.innerHTML = Object.keys(pre.presets).map(k =>
      `<option value="${k}">${k}</option>`).join("");
    sel.onchange = () => {
      writeDevice(pre.presets[sel.value]);
      state.solution = null; state.sweep = null;
      setStatus("Loaded " + sel.value);
      drawTable();
    };
    writeDevice(pre.presets[sel.value]);
  } else {
    state.layers = [{ thickness_nm: 200, material: "GaAs", doping: 1e17 },
                    { thickness_nm: 200, material: "GaAs", doping: -1e17 }];
    renderLayers();
  }

  $("addLayer").onclick = () => {
    state.layers.push({ thickness_nm: 100, material: "GaAs", doping: 0 });
    renderLayers();
  };
  $("brad").oninput = () => {
    $("bradVal").textContent = fmt(10 ** (+$("brad").value), 2);
  };
  // The strain columns of the layer table are computed from the substrate,
  // so changing it has to refresh the table on its own.
  $("substrate").onchange = () => drawTable();
  $("substrateA").oninput = () => drawTable();
  $("brad").dispatchEvent(new Event("input"));

  $("runEq").onclick = () => doSolve("equilibrium");
  $("runZC").onclick = () => doSolve("zero_current");
  $("runDD").onclick = () => doSolve("drift_diffusion");
  $("runIV").onclick = () => doSweep("iv");
  $("runCV").onclick = () => doSweep("cv");

  document.querySelectorAll(".tab").forEach(t =>
    t.onclick = () => switchTab(t.dataset.tab));
  $("ivLog").onchange = () => drawIV();

  let rt;
  window.addEventListener("resize", () => {
    clearTimeout(rt);
    rt = setTimeout(() => switchTab(document.querySelector(".tab.active").dataset.tab), 120);
  });

  setStatus("Ready", "ok");
}

init().catch(e => setStatus("Initialisation failed: " + e.message, "err"));
