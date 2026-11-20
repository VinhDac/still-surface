/* Still surface: the page's three drawings. DATA is written by build_page.py from results.json. */
(function () {
  "use strict";
  const reduce = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const css = getComputedStyle(document.documentElement);
  const col = (n) => css.getPropertyValue(n).trim();
  const fmt = (x, d) => (x < 0 ? "−" : "") + Math.abs(x).toFixed(d);
  const pct = (x) => Math.round(100 * x) + "%";
  const NS = "http://www.w3.org/2000/svg";
  function el(tag, attrs, parent) {
    const e = document.createElementNS(NS, tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }

  /* 1. The smiles: the fitted surface of the example, still, or shifting by each maturity's loading
        on a VIX-sized move (an Ornstein-Uhlenbeck path whose spread is the half day's), or alike. */
  const canvas = document.getElementById("smiles");
  const ctx = canvas.getContext("2d");
  const [a, b, c, d] = DATA.surface;
  const taus = [0.25, 0.5, 1, 3];
  const names = ["3m", "6m", "1y", "3y"];
  const vol = (m, tau) => { const z = Math.log(m) / Math.sqrt(tau); return a + b * z + c * z * z + d * z * z * z; };
  let mode = "still", shift = 0, last = 0;
  const LO = 0.9, HI = 1.1;
  function draw() {
    const w = canvas.clientWidth, h = canvas.clientHeight, dpr = window.devicePixelRatio || 1;
    if (canvas.width !== Math.round(w * dpr)) { canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr); }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    const padL = 44, padR = 40, padT = 12, padB = 26;
    const vmin = 0.12, vmax = 0.27;
    const X = (m) => padL + (m - LO) / (HI - LO) * (w - padL - padR);
    const Y = (v) => padT + (vmax - v) / (vmax - vmin) * (h - padT - padB);
    ctx.font = "11px " + col("--mono");
    ctx.fillStyle = col("--dim");
    ctx.strokeStyle = col("--rule");
    ctx.lineWidth = 1;
    for (const v of [0.15, 0.2, 0.25]) {
      ctx.beginPath(); ctx.moveTo(padL, Y(v)); ctx.lineTo(w - padR, Y(v)); ctx.stroke();
      ctx.fillText(Math.round(100 * v) + "%", 4, Y(v) + 4);
    }
    for (const m of [0.9, 1, 1.1]) ctx.fillText(Math.round(100 * m) + "%", X(m) - 12, h - 8);
    taus.forEach((tau, i) => {
      const load = mode === "term" ? DATA.loading[i] : mode === "parallel" ? 1 : 0;
      const tone = 0.95 - 0.18 * i;
      if (load) {
        ctx.strokeStyle = col("--rule"); ctx.lineWidth = 1;
        ctx.beginPath();
        for (let k = 0; k <= 60; k++) { const m = LO + (HI - LO) * k / 60; const p = [X(m), Y(vol(m, tau))]; k ? ctx.lineTo(...p) : ctx.moveTo(...p); }
        ctx.stroke();
      }
      ctx.strokeStyle = `rgba(27,36,48,${tone})`; ctx.lineWidth = 1.6;
      ctx.beginPath();
      for (let k = 0; k <= 60; k++) { const m = LO + (HI - LO) * k / 60; const p = [X(m), Y(vol(m, tau) + load * shift)]; k ? ctx.lineTo(...p) : ctx.moveTo(...p); }
      ctx.stroke();
      ctx.fillStyle = `rgba(27,36,48,${tone})`;
      ctx.fillText(names[i], X(LO) + 4, Y(vol(LO, tau) + load * shift) - 5);  // at 90%, where the smiles stand apart
    });
  }
  function tick(t) {
    const dt = Math.min(0.05, (t - last) / 1000 || 0); last = t;
    const theta = 0.8, sd = DATA.half_std;  // the path's spread is the half day's, in vol
    shift += -theta * shift * dt + sd * Math.sqrt(2 * theta * dt) * gauss();
    draw();
    if (mode !== "still" && !reduce) requestAnimationFrame(tick);
  }
  function gauss() { let u = 0, v = 0; while (!u) u = Math.random(); while (!v) v = Math.random(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }
  document.querySelectorAll(".modes button").forEach((btn) => btn.addEventListener("click", () => {
    document.querySelectorAll(".modes button").forEach((x) => x.setAttribute("aria-pressed", String(x === btn)));
    const was = mode; mode = btn.dataset.mode;
    if (mode === "still") { shift = 0; draw(); }
    else if (reduce) { shift = DATA.half_std; draw(); }
    else if (was === "still") { last = performance.now(); requestAnimationFrame(tick); }
  }));
  window.addEventListener("resize", draw);
  draw();

  /* 2. The ribbon: the book's vega through the session, the picked penalty against the paper's. */
  const rib = document.getElementById("ribbon");
  const RW = 900, RH = 300, rx0 = 50, rx1 = 880, mid = 150;
  const frac = [0].concat(DATA.fraction);
  const vmaxR = Math.max(...Object.values(DATA.vega).flat()) * 1.08;
  const rx = (f) => rx0 + f * (rx1 - rx0);
  const ry = (v) => v / vmaxR * 128;
  function smooth(pts) {
    let dd = `M${pts[0][0]},${pts[0][1]}`;
    for (let i = 0; i < pts.length - 1; i++) {
      const p0 = pts[Math.max(0, i - 1)], p1 = pts[i], p2 = pts[i + 1], p3 = pts[Math.min(pts.length - 1, i + 2)];
      dd += ` C${p1[0] + (p2[0] - p0[0]) / 6},${p1[1] + (p2[1] - p0[1]) / 6} ${p2[0] - (p3[0] - p1[0]) / 6},${p2[1] - (p3[1] - p1[1]) / 6} ${p2[0]},${p2[1]}`;
    }
    return dd;
  }
  function band(values) {
    const v = [0].concat(values);
    const top = frac.map((f, i) => [rx(f), mid - ry(v[i])]);
    const bot = frac.map((f, i) => [rx(f), mid + ry(v[i])]).reverse();
    return smooth(top) + " L" + bot[0][0] + "," + bot[0][1] + smooth(bot).slice(smooth(bot).indexOf(" C")) + " Z";
  }
  function drawRibbon(key) {
    rib.innerHTML = "";
    el("line", { x1: rx0, x2: rx1, y1: mid, y2: mid, stroke: col("--rule") }, rib);
    [["open", 0], ["mid-session", 0.5], ["close", 1]].forEach(([t, f]) => {
      const tx = el("text", { x: rx(f), y: RH - 10, "text-anchor": f === 0 ? "start" : f === 1 ? "end" : "middle",
        "font-family": col("--mono"), "font-size": 15, fill: col("--dim") }, rib);
      tx.textContent = t;
    });
    const pick = DATA.picks.find((p) => p.key === key);
    const color = pick.kind === "terminal" ? col("--terminal") : pick.kind === "running" ? col("--running") : col("--still");
    el("path", { d: band(DATA.vega[key]), fill: color, "fill-opacity": key === "optimal baseline" ? 0.35 : 0.55, stroke: "none" }, rib);
    el("path", { d: band(DATA.vega["optimal baseline"]), fill: "none", stroke: col("--ink"), "stroke-opacity": 0.45, "stroke-width": 1.2, "stroke-dasharray": "4 3" }, rib);
    const v = DATA.vega[key], v0 = DATA.vega["optimal baseline"];
    const m = Math.round(v.length / 2) - 1;
    document.getElementById("ribbon-readout").textContent =
      `${pick.label}: ${fmt(v[m], 1)} of percentage-vega at mid-session, ${fmt(v[v.length - 1], 1)} at the close` +
      (key === "optimal baseline" ? "." : ` (the paper's: ${fmt(v0[m], 1)} and ${fmt(v0[v0.length - 1], 1)}).`);
  }
  const picks = document.querySelectorAll(".picks button");
  picks.forEach((btn) => btn.addEventListener("click", () => {
    picks.forEach((x) => x.setAttribute("aria-pressed", String(x === btn)));
    drawRibbon(btn.dataset.key);
  }));
  drawRibbon(DATA.picks[1].key);

  /* 3. Cost against in-session risk, one line a kind of penalty. */
  const fr = document.getElementById("frontier");
  const FW = 900, FH = 440, fx0 = 78, fx1 = 870, fy0 = 380, fy1 = 20;
  const pts = DATA.frontier;
  const xmin = Math.min(0, ...pts.map((p) => p.cost)) - 0.2, xmax = Math.max(...pts.map((p) => p.cost)) * 1.06;
  const ymax = Math.max(...pts.map((p) => p.risk)) * 1.1;
  const fxs = (x) => fx0 + (x - xmin) / (xmax - xmin) * (fx1 - fx0);
  const fys = (y) => fy0 - y / ymax * (fy0 - fy1);
  el("line", { x1: fx0, x2: fx1, y1: fy0, y2: fy0, stroke: col("--dim") }, fr);
  el("line", { x1: fx0, x2: fx0, y1: fy0, y2: fy1, stroke: col("--dim") }, fr);
  const step = (xmax - xmin) > 4 ? 1 : 0.5;
  for (let x = Math.ceil(xmin / step) * step; x <= xmax; x += step) {
    el("line", { x1: fxs(x), x2: fxs(x), y1: fy0, y2: fy0 + 5, stroke: col("--dim") }, fr);
    const t = el("text", { x: fxs(x), y: fy0 + 24, "text-anchor": "middle", "font-family": col("--mono"), "font-size": 16, fill: col("--dim") }, fr);
    t.textContent = fmt(x, step < 1 ? 1 : 0);
  }
  for (let y = 0; y <= ymax; y += ymax > 6 ? 2 : 1) {
    el("line", { x1: fx0 - 5, x2: fx0, y1: fys(y), y2: fys(y), stroke: col("--dim") }, fr);
    const t = el("text", { x: fx0 - 10, y: fys(y) + 5, "text-anchor": "end", "font-family": col("--mono"), "font-size": 16, fill: col("--dim") }, fr);
    t.textContent = y.toFixed(0);
  }
  const xl = el("text", { x: (fx0 + fx1) / 2, y: FH - 6, "text-anchor": "middle", "font-size": 18, fill: col("--soft") }, fr);
  xl.textContent = "cost: mean value given up on a still surface";
  const yl = el("text", { x: 18, y: (fy0 + fy1) / 2, "text-anchor": "middle", "font-size": 18, fill: col("--soft"), transform: `rotate(-90 18 ${(fy0 + fy1) / 2})` }, fr);
  yl.textContent = "in-session risk left (std)";
  const famColor = { terminal: col("--terminal"), total: col("--light"), term: col("--running"), buckets: col("--still") };
  for (const fam of DATA.families) {
    const line = pts.filter((p) => p.families.includes(fam.key)).sort((p, q) => p.order[fam.key] - q.order[fam.key]);
    el("path", { d: "M" + line.map((p) => `${fxs(p.cost)},${fys(p.risk)}`).join(" L"), fill: "none", stroke: famColor[fam.key], "stroke-width": 2 }, fr);
    const lastp = line[line.length - 1];
    const right = fxs(lastp.cost) > fx0 + 0.6 * (fx1 - fx0);  // near the right edge: end the label at the point
    const below = fam.key === "total" || fam.key === "term";
    const t = el("text", { x: fxs(lastp.cost) + (right ? 0 : 10), y: fys(lastp.risk) + (below ? 24 : -14),
      "text-anchor": right ? "end" : "start", "font-size": 17, fill: famColor[fam.key] }, fr);
    t.textContent = fam.label;
  }
  const readout = document.getElementById("frontier-readout");
  for (const p of pts) {
    const dot = el("circle", { cx: fxs(p.cost), cy: fys(p.risk), r: p.key === "optimal baseline" ? 6 : 4.5,
      fill: p.key === "optimal baseline" ? col("--ink") : famColor[p.families[0]], stroke: col("--paper2"), "stroke-width": 1.5, tabindex: 0 }, fr);
    const say = () => { readout.textContent = `${p.label}: costs ${fmt(p.cost, 2)}, leaves ${fmt(p.risk, 2)} of in-session risk (${pct(p.cut)} less than the paper's), carried risk ${pct(p.carried)} less.`; };
    dot.addEventListener("mouseenter", say); dot.addEventListener("focus", say); dot.addEventListener("click", say);
  }
  const base = pts.find((p) => p.key === "optimal baseline");
  const bt = el("text", { x: fxs(base.cost) + 12, y: fys(base.risk) - 12, "font-size": 17, fill: col("--ink") }, fr);
  bt.textContent = "the paper's example";
})();
