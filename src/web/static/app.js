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

/* Shared with styles.css: the chart has to sit on the same white card as the
   rest of the workbench, so grid and frame colours are the light-theme greys
   from the palette and the series are the saturated counterparts of them. */
const PALETTE = {
  grid: "#eef2f7",
  gridMajor: "#e2e8f0",
  frame: "#cbd5e0",
  tick: "#718096",
  band: "rgba(43, 108, 176, .035)",
  bandAlt: "rgba(43, 108, 176, .075)",
  bandLabel: "#a0aec0",
  crosshair: "#2b6cb0",
  empty: "#a0aec0",
  ec: "#2b6cb0",   // conduction band, Γ valley
  ecL: "#6b46c1",  // L valley
  ecX: "#b83280",  // X valley
  ev: "#c53030",   // valence band
  evLH: "#dd6b20", // light hole
  ei: "#a0aec0",   // intrinsic level
  fn: "#2f855a",   // electron quasi-Fermi level
  fp: "#b7791f",   // hole quasi-Fermi level
  sub: "#d69e2e",  // confined subband
  electron: "#2b6cb0",
  hole: "#c53030",
  field: "#2f855a",
  current: "#2f855a",
  cap: "#2b6cb0",
  invcap: "#b7791f",
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

/* Decades are written with the typographic minus, so "1e−3" sits on the
   same baseline as "−0.5" instead of mixing a hyphen with a minus sign. */
function expStr(e) {
  return String(e).replace("-", "−");
}

/* One number format per axis.  Mixing "1.20e+5" with "80000" on the same
   axis is the usual tell of an unformatted plot: pick the exponent from the
   largest tick and scale every label by it, or fall back to a fixed number
   of decimals when the axis is plainly linear. */
function makeTickLabel(ticks) {
  const nums = ticks.map(t => (typeof t === "number" ? t : t.v))
    .filter(v => isFinite(v));
  const mag = nums.length
    ? Math.max(...nums.map(Math.abs)) : 0;

  if (mag >= 1e4 || (mag > 0 && mag < 1e-2)) {
    const e = Math.floor(Math.log10(mag));
    const scale = 10 ** e;
    return (v) => {
      if (!v) return "0";
      return (v / scale).toFixed(1) + "e" + expStr(e);
    };
  }

  const sorted = nums.slice().sort((a, b) => a - b);
  let step = 0;
  for (let i = 1; i < sorted.length; i++) {
    const d = Math.abs(sorted[i] - sorted[i - 1]);
    if (d > 1e-12) { step = d; break; }
  }
  const dec = step > 0
    ? Math.max(0, Math.min(4, -Math.floor(Math.log10(step) + 1e-9)))
    : 2;
  return (v) => v.toFixed(dec).replace("-", "−");
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
    svg.appendChild(el("rect", { x: m.l, y: m.t, width: iw, height: ih,
      fill: PALETTE.grid, stroke: PALETTE.frame, "stroke-dasharray": "4 4" }));
    svg.appendChild(el("text", { x: W / 2, y: H / 2, fill: PALETTE.empty,
      "text-anchor": "middle", "font-size": 12 }))
      .textContent = "No data yet — solve on the left and the curve appears here";
    return;
  }

  let x0 = Math.min(...cfg.x), x1 = Math.max(...cfg.x);
  if (x1 - x0 < 1e-12) { x0 -= 1; x1 += 1; }
  const sx = (v) => m.l + (v - x0) / (x1 - x0) * iw;

  /* Log scaling is per axis: the carrier chart has a log left axis and a
     linear right axis, so asking cfg.ylog for the y2 span silently put the
     field series on a log scale it was never drawn on. */
  const isLog = (axis) => axis === "y2" ? !!cfg.y2log : !!cfg.ylog;

  const span = (axis) => {
    let lo = Infinity, hi = -Infinity;
    for (const s of vis) {
      if ((s.axis || "y") !== axis) continue;
      for (const v of s.data) {
        if (!isFinite(v)) continue;
        if (isLog(axis) && v <= 0) continue;
        lo = Math.min(lo, v); hi = Math.max(hi, v);
      }
    }
    if (!isFinite(lo)) return [0, 1];
    if (isLog(axis)) {
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
      fill: i % 2 ? PALETTE.bandAlt : PALETTE.band,
      stroke: PALETTE.frame, "stroke-dasharray": "3 3",
    }));
    if (c - a > 46) {
      const t = el("text", { x: (a + c) / 2, y: m.t + 12, fill: PALETTE.bandLabel,
        "font-size": 10, "text-anchor": "middle" });
      t.textContent = b.label;
      svg.appendChild(t);
    }
  });

  /* grid + y axis */
  const yTicks = cfg.ylog ? decadeTicks(ya, yb) : niceTicks(ya, yb, 6).map(v => ({ v, major: true }));
  const yLabel = cfg.ylog ? null : makeTickLabel(yTicks);
  for (const tk of yTicks) {
    const v = cfg.ylog ? 10 ** tk.v : tk.v;
    const y = sy(v);
    if (y < m.t - 2 || y > m.t + ih + 2) continue;
    svg.appendChild(el("line", { x1: m.l, x2: m.l + iw, y1: y, y2: y,
      stroke: tk.major ? PALETTE.gridMajor : PALETTE.grid, "stroke-width": 1 }));
    if (tk.major) {
      const t = el("text", { x: m.l - 8, y: y + 3.5, fill: PALETTE.tick,
        "font-size": 10.5, "text-anchor": "end" });
      t.textContent = cfg.ylog
        ? "1e" + expStr(tk.v)
        : yLabel(v);
      svg.appendChild(t);
    }
  }
  /* x axis — decimals follow the tick step, so 50 and 100 do not end up as
     "50.00" and "100.0" side by side */
  const xTicks = niceTicks(x0, x1, 7);
  const xLabel = makeTickLabel(xTicks);
  for (const v of xTicks) {
    const x = sx(v);
    svg.appendChild(el("line", { x1: x, x2: x, y1: m.t, y2: m.t + ih,
      stroke: PALETTE.grid }));
    const t = el("text", { x, y: m.t + ih + 16, fill: PALETTE.tick,
      "font-size": 10.5, "text-anchor": "middle" });
    t.textContent = xLabel(v);
    svg.appendChild(t);
  }
  /* right axis */
  if (vis.some(s => s.axis === "y2")) {
    const t2 = cfg.y2log ? decadeTicks(y2a, y2b)
      : niceTicks(y2a, y2b, 5).map(v => ({ v, major: true }));
    const l2 = cfg.y2log ? null : makeTickLabel(t2);
    for (const tk of t2) {
      const v = cfg.y2log ? 10 ** tk.v : tk.v;
      const y = sy2(v);
      if (y < m.t - 2 || y > m.t + ih + 2 || !tk.major) continue;
      const t = el("text", { x: m.l + iw + 8, y: y + 3.5, fill: PALETTE.tick,
        "font-size": 10.5 });
      t.textContent = cfg.y2log ? "1e" + expStr(tk.v) : l2(v);
      svg.appendChild(t);
    }
  }

  /* frame + labels */
  svg.appendChild(el("rect", { x: m.l, y: m.t, width: iw, height: ih,
    fill: "none", stroke: PALETTE.frame }));
  const xl = el("text", { x: m.l + iw / 2, y: H - 6, fill: PALETTE.tick,
    "font-size": 11, "text-anchor": "middle" });
  xl.textContent = cfg.xlabel || "";
  svg.appendChild(xl);
  const yl = el("text", { x: 14, y: m.t + ih / 2, fill: PALETTE.tick,
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
  const cl = el("line", { y1: m.t, y2: m.t + ih, stroke: PALETTE.crosshair,
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
  sweep: null,            // most recent sweep, either kind
  sweeps: { iv: null, cv: null },   // kept separately so one does not evict the other
  rows: null,             // last /api/describe table, kept for the CSV export
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
    `${m.ternary ? ` (x: ${m.x_label})` : ""}</option>`).join("");
}

function renderLayers() {
  const host = $("layers");
  host.innerHTML = "";
  state.layers.forEach((L, i) => {
    const meta = state.materials.find(m => m.name === L.material) || {};
    const ternary = !!meta.ternary;
    // x is not the same element in every alloy (Al_xGa_1-xAs vs Ga_xIn_1-xAs),
    // so the field says which one it is rather than a bare "x".
    const xTag = meta.x_label ? ` (${meta.x_label})` : "";
    const x0 = L.composition ?? meta.x_default ?? 0.5;
    const x1 = L.composition_end ?? x0;
    const graded = ternary && L.composition_end !== undefined &&
      Math.abs(x1 - x0) > 1e-9;
    const d = L.doping || 0;
    const type = d > 0 ? "p" : d < 0 ? "n" : "i";
    const exp = d ? Math.log10(Math.abs(d)) : 16;
    const box = document.createElement("div");
    box.className = "layer";
    box.innerHTML = `
      <div class="head">
        <span class="idx">Layer ${i + 1}${graded ? " · graded" : ""}</span>
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
        <label>x${xTag} start <input type="number" step="0.01" min="0" max="1"
               value="${x0}" data-f="composition"></label>
        <label>x${xTag} end <input type="number" step="0.01" min="0" max="1"
               value="${x1}" data-f="composition_end"
               title="Set the same value as the start to make the layer uniform"></label>` : ""}
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
      if (m && m.ternary) L.composition = L.composition ?? m.x_default ?? 0.5;
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
  const T = +$("T").value || 300;
  const url = (comp) =>
    `/api/properties?material=${encodeURIComponent(L.material)}` +
    (comp === null || comp === undefined ? "" : `&composition=${comp}`) +
    `&T=${T}`;
  try {
    // Sending composition= empty hits a 422 on the float parameter, so the
    // key is only added when the layer actually has a composition.
    const p = await api(url(L.composition));
    const line = (q, tag) =>
      `Eg ${fmt(q.Eg, 4)}${tag || ""} eV · ${q.direct ? "direct" : "indirect"}` +
      ` · a ${fmt(q.lattice_constant, 5)} Å · εr ${fmt(q.dielectric, 3)}` +
      ` · μn ${fmt(q.mu_n, 3)} / μp ${fmt(q.mu_p, 3)} cm²/Vs`;

    // A graded layer has two gaps, one at each end; showing only the start one
    // makes the layer look uniform.
    const graded = L.composition_end !== undefined &&
      Math.abs((L.composition_end ?? 0) - (L.composition ?? 0)) > 1e-9;
    if (graded) {
      const p1 = await api(url(L.composition_end));
      info.textContent = line(p) + ` → Eg ${fmt(p1.Eg, 4)} eV at x = ${L.composition_end}`;
    } else {
      info.textContent = line(p);
    }
  } catch { info.textContent = `Eg ${fmt(meta.Eg, 4)} eV`; }
}

/* ──────────────────────────── job polling ──────────────────────────────── */

/* A single solve has no step count to report, so the bar goes indeterminate
   and the label carries the elapsed time instead of a frozen "0/0". */
function busy(on, text = "Computing…", frac = null) {
  const b = $("busy");
  b.classList.toggle("show", on);
  b.classList.toggle("indet", on && frac === null);
  // Hiding the pill must not blank the label: the text is the last thing the
  // user reads about the run, and resetting it to "Computing…" looks like a
  // crash when the panel reappears a moment later.
  if (on) $("busyText").textContent = text;
  const bar = b.querySelector("i");
  if (frac === null) bar.style.width = "";
  else bar.style.width = (frac * 100) + "%";
}

async function runJob(kind, payload, label) {
  const t0 = performance.now();
  const secs = () => (performance.now() - t0) / 1000;
  busy(true, `${label} — 0.0 s`, null);
  setStatus(label, "warn");
  let job_id;
  try {
    ({ job_id } = await api(payload.url, payload.body));
  } catch (e) {
    // A rejected submission must not leave the bar spinning forever.
    busy(false);
    throw e;
  }
  try {
    for (;;) {
      await new Promise(r => setTimeout(r, 250));
      const j = await api(`/api/job/${job_id}`);
      if (j.status === "error") {
        setStatus("Solve failed", "err");
        throw new Error(j.error || "unknown solver error");
      }
      let text;
      // A single solve reports total = 1, which is not a progress bar: keep
      // the indeterminate sweep and show the clock instead of a frozen "0/1".
      const stepped = j.total > 1;
      if (stepped) {
        // Sweeps know how many points are left, so they get a real ETA.
        const done = j.progress || 0;
        const eta = done > 0 ? secs() * (j.total - done) / done : null;
        text = `${label} — ${done}/${j.total} · ${secs().toFixed(1)} s elapsed` +
          (eta !== null && eta > 1 && done < j.total
            ? ` · about ${Math.ceil(eta)} s left` : "");
      } else {
        text = `${label} — ${secs().toFixed(1)} s`;
      }
      busy(true, text, stepped ? j.progress / j.total : null);
      if (j.status === "done") return j.result;
    }
  } finally {
    busy(false);
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

/* An I–V sweep has a current in hand, so the strip should not keep saying
   "not solved for current" next to a finished curve. */
function updateCurrentMetric(sw) {
  const J = sw.J || [];
  let k = -1;
  for (let i = 0; i < J.length; i++) if (isFinite(J[i])) k = i;
  const cards = $("metrics").children;
  if (k < 0 || cards.length < 5) return;
  cards[4].className = "metric amber";
  cards[4].innerHTML =
    `<div class="m-label">Current density</div>` +
    `<div class="m-val">${fmt(J[k], 3)}<small>A/cm²</small></div>` +
    `<div class="m-sub">at V = ${fmt(sw.V[k], 3)} V</div>`;
}

async function doSweep(kind) {
  // I–V and C–V keep their own bias windows: a photodiode wants reverse bias
  // for both, a diode wants forward bias for I–V and a small reverse window
  // for C–V, and one shared range cannot serve both.
  const p = kind === "cv" ? "cv" : "iv";
  const body = {
    device: readDevice(),
    kind,
    Vstart: +$(p + "Start").value,
    Vstop: +$(p + "Stop").value,
    Vnum: +$(p + "Num").value,
  };
  try {
    const res = await runJob("sweep", { url: "/api/sweep", body },
      kind === "iv" ? "I–V sweep" : "C–V sweep");
    state.sweeps[kind] = { kind, ...res };
    state.sweep = state.sweeps[kind];
    if (kind === "iv") { drawIV(); updateCurrentMetric(res); switchTab("iv"); }
    else { drawCV(); switchTab("cv"); }
    setStatus(`${kind.toUpperCase()} sweep complete`, "ok");
  } catch (e) {
    setStatus("Error: " + e.message, "err");
  }
}

/* ──────────────────────── configuration in / out ───────────────────────── */

/* What the page currently has: the device the solver would receive right now,
   the bias windows, and the results on screen.  Kept in one place so the
   export, the import and the presets all agree on the shape. */
function readSweep() {
  return {
    V: +$("V").value || 0,
    iv: { start: +$("ivStart").value, stop: +$("ivStop").value,
          num: +$("ivNum").value },
    cv: { start: +$("cvStart").value, stop: +$("cvStop").value,
          num: +$("cvNum").value },
  };
}

function applySweep(sw) {
  if (!sw) return;
  const set = (id, v) => {
    if (v !== undefined && v !== null && $(id)) $(id).value = v;
  };
  set("V", sw.V);
  for (const k of ["iv", "cv"]) {
    if (!sw[k]) continue;
    set(k + "Start", sw[k].start);
    set(k + "Stop", sw[k].stop);
    set(k + "Num", sw[k].num);
  }
}

function fileStem() {
  const p = $("preset").value || "device";
  const s = state.solution;
  const v = s && s.V !== undefined ? `_V${Number(s.V).toFixed(2)}` : "";
  return "obpds_" + p.replace(/[^\w]+/g, "_").toLowerCase() + v;
}

function download(name, text, mime) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([text], { type: mime }));
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

function csvFrom(header, columns) {
  const n = Math.max(...columns.map(c => (c || []).length));
  const lines = [header.join(",")];
  for (let i = 0; i < n; i++) {
    lines.push(columns.map(col => {
      const v = col ? col[i] : "";
      return (v === null || v === undefined || (typeof v === "number" && !isFinite(v)))
        ? "" : String(v);
    }).join(","));
  }
  return lines.join("\n");
}

function exportJSON() {
  const s = state.solution;
  const pack = (sw) => sw ? {
    kind: sw.kind, V: sw.V,
    values: sw.kind === "iv" ? sw.J : sw.C,
    ideality: sw.ideality || null,
    meta: sw.meta || null,
  } : null;
  const payload = {
    format: "obpds-config",
    version: 2,
    saved: new Date().toISOString(),
    device: readDevice(),
    sweep: readSweep(),
    results: {
      solve: s ? {
        mode: s.mode, V: s.V, converged: s.converged !== false,
        T: s.meta && s.meta.T, N: s.meta && s.meta.N,
        approximation: s.meta && s.meta.approx,
        total_nm: s.total_nm,
        n_is_quantum: !!s.n_is_quantum,
        subbands: s.subbands
          ? { levels_eV: s.subbands.levels_eV,
              bound: s.subbands.bound,
              population_cm2: s.subbands.population_cm2,
              sheet_density_cm2: s.subbands.sheet_density_cm2,
              E_cut_eV: s.subbands.E_cut_eV }
          : null,
      } : null,
      // Both curves, not just whichever ran last: an I–V export that has
      // silently lost the C–V is worse than no export.
      sweeps: { iv: pack(state.sweeps.iv), cv: pack(state.sweeps.cv) },
      layers: state.rows || null,
    },
  };
  download(fileStem() + ".json", JSON.stringify(payload, null, 2),
           "application/json");
  setStatus("Configuration exported", "ok");
}

function exportCSV() {
  const tab = document.querySelector(".tab.active");
  const name = tab ? tab.dataset.tab : "band";
  const s = state.solution;

  if (name === "iv" || name === "cv") {
    const sw = state.sweeps[name];
    if (!sw) {
      setStatus("Run the " + name.toUpperCase() + " sweep first", "warn");
      return;
    }
    const vals = name === "iv" ? sw.J : sw.C;
    download(fileStem() + "_" + name + ".csv",
             csvFrom(["V", name === "iv" ? "J" : "C"], [sw.V, vals]),
             "text/csv");
    setStatus(name.toUpperCase() + " data exported", "ok");
    return;
  }

  if (name === "table") {
    if (!state.rows || !state.rows.length) {
      setStatus("Layer table not ready", "warn");
      return;
    }
    const keys = ["label", "thickness_nm", "Eg", "Eg_end", "doping_cm3",
                  "lattice_constant", "mismatch", "dielectric", "eps_in",
                  "eps_out", "h_c_nm"];
    const cols = keys.map(k => state.rows.map(r => r[k]));
    download(fileStem() + "_layers.csv",
             csvFrom(keys, cols), "text/csv");
    setStatus("Layer table exported", "ok");
    return;
  }

  if (!s) { setStatus("Solve first — there is no profile to export", "warn"); return; }
  const keys = [["x_nm", "x (nm)"], ["Ec_Gamma", "Ec_Gamma (eV)"],
                ["Ec_L", "Ec_L (eV)"], ["Ec_X", "Ec_X (eV)"],
                ["Ev", "Ev (eV)"], ["Fn", "Fn (eV)"], ["Fp", "Fp (eV)"],
                ["n", "n (cm^-3)"], ["p", "p (cm^-3)"],
                ["field", "field (V/cm)"]];
  const cols = keys.map(([k]) => s[k]);
  download(fileStem() + "_profile.csv",
           csvFrom(keys.map(([, h]) => h), cols), "text/csv");
  setStatus("Profile exported", "ok");
}

async function importConfig(file) {
  let obj;
  try {
    obj = JSON.parse(await file.text());
  } catch (e) {
    setStatus("Not a JSON file: " + e.message, "err");
    return;
  }
  const dev = obj.device || (obj.layers ? obj : null);
  if (!dev || !dev.layers || !dev.layers.length) {
    setStatus("This JSON has no device block", "err");
    return;
  }
  try {
    writeDevice(dev);
    applySweep(obj.sweep);
    state.solution = null;
    state.sweep = null;
    state.sweeps = { iv: null, cv: null };
    state.rows = null;
    resetViews();
    $("preset").value = "";
    setStatus("Imported " + file.name, "ok");
    drawTable();
  } catch (e) {
    setStatus("Import failed: " + e.message, "err");
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
    { name: "Ec (Γ)", data: s.Ec_Gamma, color: PALETTE.ec, width: 2 },
    { name: "Ec (L)", data: s.Ec_L, color: PALETTE.ecL, width: 1.2, dash: "5 3" },
    { name: "Ec (X)", data: s.Ec_X, color: PALETTE.ecX, width: 1.2, dash: "2 3" },
  ];
  if (hh_lh) {
    out.push({ name: "Ev (heavy hole)", data: s.Ev_HH, color: PALETTE.ev, width: 2 });
    out.push({ name: "Ev (light hole)", data: s.Ev_LH, color: PALETTE.evLH,
               width: 1.6, dash: "6 3" });
  } else {
    out.push({ name: "Ev", data: s.Ev, color: PALETTE.ev, width: 2 });
  }
  out.push(
    { name: "Ei", data: s.Ei, color: PALETTE.ei, width: 1, dash: "1 3" },
    { name: "Fn", data: s.Fn, color: PALETTE.fn, width: 1.8 },
    { name: "Fp", data: s.Fp, color: PALETTE.fp, width: 1.8 },
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
        color: PALETTE.sub, width: 1.6, dash: "6 3",
      });
    });
  }
  return out;
}

/* ── metric strip ────────────────────────────────────────────────────────
   Six cards across the top of the results panel.  They stay in place with a
   dash placeholder before the first solve, so the panel does not jump when
   numbers arrive. */

const APPROX_LABEL = {
  "kane": "Kane",
  "fermi-dirac": "F–D",
  "boltzmann": "Boltz.",
};

function metric(cls, label, value, unit, sub, idle) {
  return `<div class="metric ${cls}${idle ? " idle" : ""}">` +
    `<div class="m-label">${label}</div>` +
    `<div class="m-val">${value}${unit ? `<small>${unit}</small>` : ""}</div>` +
    `<div class="m-sub">${sub || "&nbsp;"}</div></div>`;
}

function renderMetrics(s) {
  const host = $("metrics");
  if (!s) {
    host.innerHTML =
      metric("blue", "Temperature", "—", "", "K", true) +
      metric("violet", "Statistics", "—", "", "carrier model", true) +
      metric("green", "Min band gap", "—", "", "E<sub>c</sub> − E<sub>v</sub>", true) +
      metric("red", "Peak field", "—", "", "V/cm", true) +
      metric("amber", "Current density", "—", "", "A/cm²", true) +
      metric("slate", "Mesh", "—", "", "finite-volume cells", true);
    return;
  }

  const meta = s.meta || {};
  const kT = meta.kT ? meta.kT * 1000 : NaN;

  const Ec = s.Ec_Gamma || [];
  const Ev = s.Ev || [];
  // The smallest local separation, not min(Ec) - max(Ev): the latter mixes two
  // different depths and reports the band bending rather than a gap.
  let gap = NaN;
  if (Ec.length && Ev.length && Ec.length === Ev.length) {
    gap = Infinity;
    for (let i = 0; i < Ec.length; i++) {
      const d = Ec[i] - Ev[i];
      if (isFinite(d) && d < gap) gap = d;
    }
    if (!isFinite(gap)) gap = NaN;
  }

  let fmax = NaN, fpos = NaN;
  if (s.field && s.field.length) {
    const a = s.field.map(Math.abs);
    let k = 0;
    for (let i = 1; i < a.length; i++) if (a[i] > a[k]) k = i;
    fmax = a[k];
    fpos = (s.x_nm || [])[k];
  }

  let J = NaN, Jsub = "not solved for current";
  if (s.J !== undefined && s.J !== null) {
    J = s.J;
    Jsub = `Jn ${fmt(s.Jn_contact, 3)} · Jp ${fmt(s.Jp_contact, 3)}`;
  } else if (s.Jtotal && s.Jtotal.length) {
    J = s.Jtotal[Math.floor(s.Jtotal.length / 2)];
    Jsub = "midpoint of the device";
  }

  const step = (s.total_nm && meta.N) ? s.total_nm / meta.N : NaN;

  host.innerHTML =
    metric("blue", "Temperature", fmt(meta.T, 4), "K",
      `kT = ${fmt(kT, 3)} meV`) +
    metric("violet", "Statistics", APPROX_LABEL[meta.approx] || "—", "",
      meta.approx === "kane" ? "non-parabolic + F–D" : "carrier model") +
    metric("green", "Min band gap", fmt(gap, 3), "eV", "E<sub>c</sub> − E<sub>v</sub>") +
    metric("red", "Peak field", fmt(fmax, 3), "V/cm",
      isFinite(fpos) ? `at ${fmt(fpos, 1)} nm` : "&nbsp;") +
    metric("amber", "Current density", fmt(J, 3), "A/cm²", Jsub) +
    metric("slate", "Mesh", meta.N ? String(meta.N) : "—", "",
      isFinite(step) ? `step ${fmt(step, 3)} nm` : "&nbsp;");
}


function drawSolution() {
  const s = state.solution;
  if (!s) return;
  renderMetrics(s);
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
      { name: "n", data: s.n, color: PALETTE.electron, width: 2 },
      { name: "p", data: s.p, color: PALETTE.hole, width: 2 },
      { name: "n (L valley)", data: s.n_L, color: PALETTE.ecL, width: 1.1, dash: "4 3" },
      { name: "n (X valley)", data: s.n_X, color: PALETTE.ecX, width: 1.1, dash: "2 3" },
      { name: "|E|", data: (s.field || []).map(Math.abs),
        color: PALETTE.field, width: 1.4, axis: "y2" },
    ], bands, xlabel: "Depth (nm)", ylabel: "Concentration (cm⁻³)",
    y2label: "|E| (V/cm)", legend: "legendCarrier",
  };
  drawChart($("chartCarrier"), state.charts.carrier);

  /* readouts */
  // Smallest local separation between the two edges along the stack; taking
  // min(Ec) - max(Ev) would subtract two different depths and report the band
  // bending as if it were a gap.
  let dE = NaN;
  if (s.Ec_Gamma && s.Ev && s.Ec_Gamma.length === s.Ev.length) {
    dE = Infinity;
    for (let i = 0; i < s.Ec_Gamma.length; i++) {
      const d = s.Ec_Gamma[i] - s.Ev[i];
      if (isFinite(d) && d < dE) dE = d;
    }
    if (!isFinite(dE)) dE = NaN;
  }
  const Emax = s.field ? Math.max(...s.field.map(Math.abs)) : NaN;
  $("readBand").innerHTML =
    `<b>${s.mode || ""}</b> · V = ${fmt(s.V, 3)} V · T = ${s.meta.T} K · ` +
    `kT = ${fmt(s.meta.kT, 4)} eV · N = ${s.meta.N} · ${s.meta.approx} · ` +
    `Mesh step ${fmt((s.total_nm || 0) / s.meta.N, 3)} nm · ` +
    `min gap ${fmt(dE, 4)} eV · |E|max ${fmt(Emax, 4)} V/cm` +
    (s.converged === false ? ` · <span style="color:#c53030">not converged</span>` : "");

  const split = valenceSplit(s);
  if (split > 2e-3) {
    const heavy = s.Ev_HH[0] >= s.Ev_LH[0];
    $("readBand").innerHTML +=
      ` · <b style="color:#dd6b20">pseudomorphic strain</b>: HH–LH split ` +
      `<b>${fmt(split * 1000, 1)}</b> meV (${heavy ? "heavy hole" : "light hole"} on top)`;
  }

  const sub = s.subbands;
  if (sub && sub.levels_eV) {
    const lv = sub.levels_eV.filter((_, i) => !sub.bound || sub.bound[i]);
    $("readBand").innerHTML += lv.length
      ? ` · <b style="color:#b7791f">quantum correction</b>: ${lv.length} bound subbands ` +
        `(${lv.map(e => fmt(e, 4)).join(", ")} eV) · sheet density ` +
        `<b>${fmt(sub.sheet_density_cm2, 4)}</b> cm⁻²`
      : ` · <b style="color:#b7791f">quantum correction</b>: no bound states (this structure ` +
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
  const sw = state.sweeps.iv;
  if (!sw) return;
  const V = sw.V, J = sw.J;
  const log = $("ivLog").checked;
  const data = log ? J.map(v => (isFinite(v) && v !== 0 ? Math.abs(v) : NaN)) : J;
  state.charts.iv = {
    x: V, ylog: log,
    series: [{ name: log ? "|J|" : "J", data, color: PALETTE.current, width: 2 }],
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
  const sw = state.sweeps.cv;
  if (!sw) return;
  const V = sw.V, C = sw.C;
  state.charts.cv = {
    x: V,
    series: [
      { name: "1/C²", data: C.map(c => (c > 0 ? 1 / (c * c) : NaN)),
        color: PALETTE.cap, width: 2 },
      { name: "C", data: C, color: PALETTE.invcap, width: 1.6, axis: "y2" },
    ],
    xlabel: "Bias V (V)", ylabel: "1/C² (cm⁴/F²)", y2label: "C (F/cm²)",
  };
  drawChart($("chartCV"), state.charts.cv);
  const ok = C.filter(isFinite);
  $("readCV").innerHTML = ok.length
    ? `C ∈ [${fmt(Math.min(...ok), 3)}, ${fmt(Math.max(...ok), 3)}] F/cm²`
    : "no finite results";
}

const WF_COLORS = ["#d69e2e", "#2b6cb0", "#2f855a", "#b83280",
                   "#6b46c1", "#c53030", "#3182ce", "#dd6b20"];

/* |psi|^2 of the bound subbands.  Only meaningful after a quantum solve;
   the classical path ships no wavefunctions, so say so instead of drawing
   an empty axis. */
function drawWF() {
  const s = state.solution;
  const host = $("chartWF");
  if (!s) {
    host.innerHTML =
      `<div class="chart-empty"><div>Solve the equilibrium bands first — the confined ` +
      `states are only known once the potential is.</div></div>`;
    $("legendWF").innerHTML = "";
    $("readWF").innerHTML = "";
    return;
  }
  const sub = s.subbands;
  const idx = [];
  if (sub && sub.levels_eV) {
    sub.levels_eV.forEach((E, i) => { if (!sub.bound || sub.bound[i]) idx.push(i); });
  }
  const psi = (sub && sub.psi) || [];
  if (!psi.length || !(s.x_nm || []).length || psi[0].length !== s.x_nm.length) {
    state.charts.wf = null;
    // Two different reasons for an empty panel: the quantum branch never ran,
    // or it ran and this stack simply confines nothing.  Say which.
    host.innerHTML = s.n_is_quantum
      ? `<div class="chart-empty"><div>The quantum solve ran, but this stack confines no ` +
        `bound states, so the electron density stays classical. Try a structure with a ` +
        `type-I well such as the AlGaAs/GaAs HEMT preset.</div></div>`
      : `<div class="chart-empty"><div>Tick the <b>quantum correction</b> box and ` +
        `re-solve the equilibrium bands — the |ψ|² distribution of each bound subband ` +
        `is drawn here.</div></div>`;
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
    state.rows = rows;
    const strained = rows.some(r => r.substrate_a !== undefined);
    const head = ["Layer", "Thickness (nm)", "Eg (eV)", "ΔEc/ΔEg", "Doping (cm⁻³)",
                  "a (Å)", "Mismatch %", "εr"]
      .concat(strained ? ["ε∥ %", "ε⊥ %", "h_c (nm)"] : []);
    const cell = (v, d = 4) => `<td>${fmt(v, d)}</td>`;
    const body = rows.map(r => `<tr><td>${r.label || r.material}</td>` +
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

/* An empty chart still has to look deliberate: one dashed frame and a line of
   instruction, otherwise the panel reads as broken. */
const EMPTY_CFG = { x: [], series: [], bands: [], legend: null };

/* Blank every result view.  Loading a preset or importing a file replaces the
   device, and curves drawn for the previous stack are worse than no curves. */
function resetViews() {
  state.charts = { band: null, carrier: null, iv: null, cv: null, wf: null };
  renderMetrics(null);
  ["chartCarrier", "chartIV", "chartCV", "chartWF"].forEach(id => {
    $(id).innerHTML = "";
  });
  ["legendBand", "legendCarrier", "legendWF"].forEach(id => { $(id).innerHTML = ""; });
  ["readBand", "readCarrier", "readIV", "readCV", "readWF"].forEach(id => {
    $(id).innerHTML = "";
  });
  drawChart($("chartBand"), EMPTY_CFG);
}

function switchTab(name) {
  document.querySelectorAll(".tab").forEach(t =>
    t.classList.toggle("active", t.dataset.tab === name));
  document.querySelectorAll(".tabpage").forEach(p =>
    p.classList.toggle("active", p.id === "tab-" + name));
  if (name === "band") drawChart($("chartBand"), state.charts.band || EMPTY_CFG);
  if (name === "carrier") drawChart($("chartCarrier"), state.charts.carrier || EMPTY_CFG);
  if (name === "iv") {
    if (state.charts.iv) drawIV(); else drawChart($("chartIV"), EMPTY_CFG);
  }
  if (name === "cv") {
    if (state.charts.cv) drawCV(); else drawChart($("chartCV"), EMPTY_CFG);
  }
  if (name === "wf") drawWF();
  if (name === "table") drawTable();
}

/* ────────────────────────────── wiring ─────────────────────────────────── */

async function init() {
  const mat = await api("/api/materials");
  state.materials = mat.materials;
  renderMetrics(null);

  // Anything in the catalogue can be a wafer.  Ternaries need a composition
  // the layer editor does not offer here, so they are listed but resolve at
  // the default x = 0.5 unless the user pins a lattice constant instead.
  $("substrate").innerHTML =
    `<option value="">None (per layer)</option>` +
    mat.materials.map(m =>
      `<option value="${m.name}">${m.name}` +
      `${m.ternary ? " (x=0.5)" : ""}</option>`).join("");

  const pre = await api("/api/presets").catch(() => null);
  const sel = $("preset");
  if (pre) {
    sel.innerHTML = `<option value="">— custom / imported —</option>` +
      Object.keys(pre.presets).map(k =>
        `<option value="${k}">${k}</option>`).join("");

    const loadPreset = (name) => {
      const spec = pre.presets[name];
      if (!spec) return;
      writeDevice(spec);
      // A preset is a starting point, not just a stack: it also knows which
      // bias window shows it off.
      applySweep(spec.sweep);
      state.solution = null; state.sweep = null;
      state.sweeps = { iv: null, cv: null };
      state.rows = null;
      resetViews();
      setStatus("Loaded " + name);
      drawTable();
    };
    sel.onchange = () => { if (sel.value) loadPreset(sel.value); };

    const first = Object.keys(pre.presets)[0];
    sel.value = first;
    writeDevice(pre.presets[first]);
    applySweep(pre.presets[first].sweep);
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

  $("importBtn").onclick = () => $("importFile").click();
  $("importFile").onchange = async (ev) => {
    const f = ev.target.files && ev.target.files[0];
    if (f) await importConfig(f);
    ev.target.value = "";          // so the same file can be re-imported
  };
  $("exportJson").onclick = exportJSON;
  $("exportCsv").onclick = exportCSV;

  document.querySelectorAll(".tab").forEach(t =>
    t.onclick = () => switchTab(t.dataset.tab));
  $("ivLog").onchange = () => drawIV();

  let rt;
  window.addEventListener("resize", () => {
    clearTimeout(rt);
    rt = setTimeout(() => switchTab(document.querySelector(".tab.active").dataset.tab), 120);
  });

  setStatus("Ready", "ok");
  // The band tab is visible from the first frame, so give it the dashed
  // "no data yet" frame rather than an empty white box.
  drawChart($("chartBand"), EMPTY_CFG);
}

init().catch(e => setStatus("Initialisation failed: " + e.message, "err"));
