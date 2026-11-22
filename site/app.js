const NS = "http://www.w3.org/2000/svg";
const tooltip = document.getElementById("tooltip");
const CATS = {
  kept: { label: "Right both times", color: "var(--s1)" },
  fixed: { label: "Fixed by the new build", color: "var(--s3)" },
  broken: { label: "Broken by the new build", color: "var(--s2)" },
  missed: { label: "Wrong both times", color: "var(--neutral)", hollow: true },
};

function el(tag, attrs, parent) {
  const node = document.createElementNS(NS, tag);
  for (const k in attrs) node.setAttribute(k, attrs[k]);
  if (parent) parent.appendChild(node);
  return node;
}

function esc(s) {
  return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

function showTip(html, x, y) {
  tooltip.innerHTML = html;
  tooltip.classList.add("on");
  const w = tooltip.offsetWidth;
  const h = tooltip.offsetHeight;
  let left = x + 14;
  if (left + w > window.innerWidth - 8) left = x - w - 14;
  let top = y - h - 12;
  if (top < 8) top = y + 16;
  tooltip.style.left = Math.max(8, left) + "px";
  tooltip.style.top = top + "px";
}

function hideTip() {
  tooltip.classList.remove("on");
}

function tipRow(color, label, value) {
  const key = color ? `<span class="key" style="background:${color}"></span>` : "";
  return `<div class="t-row">${key}${label}<b>${value}</b></div>`;
}

function category(q) {
  if (q.new === q.gold) return q.old === q.gold ? "kept" : "fixed";
  return q.old === q.gold ? "broken" : "missed";
}

function keyDot(cat) {
  const c = CATS[cat];
  return `<span class="key${c.hollow ? " hollow" : ""}" style="background:${c.color}"></span>`;
}

function tissue() {
  const svg = document.getElementById("tissue");
  const w = 1200;
  const h = 700;
  svg.setAttribute("viewBox", `0 0 ${w} ${h}`);
  svg.setAttribute("preserveAspectRatio", "xMidYMid slice");
  let seed = 7;
  const rnd = () => {
    seed = (seed * 16807) % 2147483647;
    return seed / 2147483647;
  };
  for (let i = 0; i < 90; i++) {
    const x = w * (0.28 + 0.78 * rnd());
    const y = h * rnd();
    const r = 12 + rnd() * 26;
    el("circle", { cx: x, cy: y, r, fill: "#e7a1bb", opacity: 0.08 + rnd() * 0.12 }, svg);
    el("ellipse", { cx: x + (rnd() - 0.5) * r * 0.5, cy: y + (rnd() - 0.5) * r * 0.5, rx: r * (0.22 + rnd() * 0.14), ry: r * (0.18 + rnd() * 0.12), fill: "#3a2370", opacity: 0.3 + rnd() * 0.3 }, svg);
  }
}

function hero(data) {
  const old = data.setups[1];
  const now = data.setups[2];
  document.getElementById("hero-new").textContent = now.correct;
  document.getElementById("before-after").innerHTML = [["As first built", old.correct, "var(--s2)"], ["Now", now.correct, "var(--s1)"]]
    .map(([name, v, color]) => `<div class="ba-row"><span>${name}</span><span class="ba-track"><span class="ba-fill" style="width:${v}%;background:${color}"></span></span><b>${v}%</b></div>`)
    .join("");
}

function renderAnswer(text) {
  if (!window.marked) return `<p style="white-space:pre-wrap">${esc(text)}</p>`;
  const html = window.marked.parse(text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;"));
  return html.replace(/<table>/g, '<div class="table-wrap"><table>').replace(/<\/table>/g, "</table></div>");
}

function demo(ex) {
  document.getElementById("demo-model").textContent = ex.model.split("/").pop();
  const list = document.getElementById("questions");
  const chat = document.getElementById("chat");
  const show = (i) => {
    const e = ex.examples[i];
    list.querySelectorAll("button").forEach((b, j) => b.setAttribute("aria-selected", j === i ? "true" : "false"));
    chat.innerHTML = `<div class="fade">
      <div class="chat-meta"><span class="tag">${esc(ex.model.split("/").pop())}</span><span class="tag">${ex.passages} passages retrieved</span><span class="tag">saved ${esc(ex.generated)}</span></div>
      <p class="bubble-q">${esc(e.question)}</p>
      <div class="answer">${renderAnswer(e.answer)}</div>
      <details class="sources"><summary>The ${e.sources.length} passages it was given</summary>
        ${e.sources.map((s) => `<div class="source"><div class="source-head">${esc(s.book)} · PDF page ${s.page}</div><p>“${esc(s.excerpt)}…”</p></div>`).join("")}
      </details></div>`;
  };
  list.innerHTML = ex.examples.map((e, i) => `<button class="q-button" type="button" role="tab" data-i="${i}">${esc(e.question)}</button>`).join("");
  list.addEventListener("click", (ev) => {
    const b = ev.target.closest("button");
    if (b) show(Number(b.dataset.i));
  });
  show(0);
}

function tiles(data) {
  const [, old, now, plain] = data.setups;
  const items = [
    ["First build", old.correct + "%", "", "Mistral-7B with a context-only prompt"],
    ["Current build", now.correct + "%", "", "Qwen3.8-27B with the new prompt"],
    ["Questions fixed", data.paired.only_new, "of 100", `and ${data.paired.only_old} broken, ${data.paired.p < 0.0001 ? "p < 0.0001" : "p = " + data.paired.p}`],
    ["Same model, no retrieval", plain.correct + "%", "", `the passages cost it ${plain.correct - now.correct} points`],
  ];
  document.getElementById("tiles").innerHTML = items.map((t) => `<div class="tile"><div class="tile-label">${t[0]}</div><div class="tile-value">${t[1]}<small>${t[2]}</small></div><div class="tile-foot">${t[3]}</div></div>`).join("");
}

function drawSetups(data) {
  const host = document.getElementById("chart-setups");
  host.innerHTML = "";
  const width = Math.max(280, host.clientWidth);
  const row = width > 460 ? 84 : 62;
  const height = data.setups.length * row + 34;
  const svg = el("svg", { viewBox: `0 0 ${width} ${height}`, width, height, role: "img", "aria-label": "Accuracy of four setups on the test questions" }, host);
  const x0 = 2;
  const x1 = width - 52;
  const top = 20;
  const x = (v) => x0 + (v / 100) * (x1 - x0);
  [0, 25, 50, 75, 100].forEach((t) => {
    el("line", { x1: x(t), x2: x(t), y1: top, y2: height - 22, stroke: t === 0 ? "var(--axis)" : "var(--grid)", "stroke-width": 1 }, svg);
    el("text", { x: x(t), y: height - 6, "text-anchor": t === 0 ? "start" : "middle" }, svg).textContent = t + "%";
  });
  el("text", { x: x(25) + 5, y: top - 6, class: "label" }, svg).textContent = "guessing";
  data.setups.forEach((s, i) => {
    const y = top + i * row + 8;
    const pct = (100 * s.correct) / s.n;
    const name = el("text", { x: x0 + 6, y: y + 12, class: "value" }, svg);
    name.textContent = s.name;
    if (width > 460) el("text", { x: x0 + 6 + name.getComputedTextLength() + 12, y: y + 12, class: "label" }, svg).textContent = s.note;
    const by = y + 22;
    const w = x(pct) - x0;
    const bar = el("path", { d: `M${x0} ${by}h${w - 4}a4 4 0 0 1 4 4v8a4 4 0 0 1 -4 4h${-(w - 4)}z`, fill: "var(--s1)" }, svg);
    el("line", { x1: x(s.low), x2: x(s.high), y1: by + 8, y2: by + 8, stroke: "var(--ink)", "stroke-width": 1.5, opacity: 0.6 }, svg);
    [s.low, s.high].forEach((v) => el("line", { x1: x(v), x2: x(v), y1: by + 4, y2: by + 12, stroke: "var(--ink)", "stroke-width": 1.5, opacity: 0.6 }, svg));
    el("text", { x: x(s.high) + 8, y: by + 13, class: "value" }, svg).textContent = pct.toFixed(0) + "%";
    const hit = el("rect", { x: x0, y: y - 4, width: width - x0, height: row - 6, fill: "transparent" }, svg);
    hit.addEventListener("pointermove", (ev) => {
      bar.setAttribute("opacity", 0.84);
      showTip(`<div class="t-head">${esc(s.note)}</div>` + tipRow("var(--s1)", esc(s.name), `${s.correct} of ${s.n}`) + tipRow("", "95% interval", `${s.low.toFixed(0)}% to ${s.high.toFixed(0)}%`), ev.clientX, ev.clientY);
    });
    hit.addEventListener("pointerleave", () => {
      bar.setAttribute("opacity", 1);
      hideTip();
    });
  });
  document.getElementById("table-setups").innerHTML =
    "<table><thead><tr><th>Setup</th><th>Details</th><th class='num'>Correct</th><th class='num'>95% interval</th></tr></thead><tbody>" +
    data.setups.map((s) => `<tr><td>${esc(s.name)}</td><td>${esc(s.note)}</td><td class='num'>${s.correct} of ${s.n}</td><td class='num'>${s.low.toFixed(0)}% to ${s.high.toFixed(0)}%</td></tr>`).join("") +
    "</tbody></table>";
}

function drawUnits(data, order) {
  const host = document.getElementById("chart-units");
  host.innerHTML = "";
  const width = Math.max(240, host.clientWidth);
  const cols = 10;
  const cell = Math.min(40, width / cols);
  const rows = Math.ceil(order.length / cols);
  const svg = el("svg", { viewBox: `0 0 ${width} ${rows * cell}`, width, height: rows * cell, role: "img", "aria-label": "Outcome of each test question" }, host);
  order.forEach((qi, k) => {
    const q = data.questions[qi];
    const cat = category(q);
    const c = CATS[cat];
    const cx = (k % cols) * cell + cell / 2;
    const cy = Math.floor(k / cols) * cell + cell / 2;
    const r = cell * 0.34;
    const dot = el("circle", Object.assign({ cx, cy, r: c.hollow ? r - 1 : r, class: "dot", tabindex: 0, role: "button", "aria-label": `Question ${qi + 1}: ${c.label}` },
      c.hollow ? { fill: "transparent", stroke: c.color, "stroke-width": 2 } : { fill: c.color }), svg);
    dot.addEventListener("pointermove", (ev) => showTip(`<div class="t-head">Question ${qi + 1}</div><div>${esc(q.question.slice(0, 120))}${q.question.length > 120 ? "…" : ""}</div>` + tipRow(c.hollow ? "" : c.color, c.label, ""), ev.clientX, ev.clientY));
    dot.addEventListener("pointerleave", hideTip);
    const open = () => openQuestion(qi);
    dot.addEventListener("click", open);
    dot.addEventListener("keydown", (ev) => {
      if (ev.key === "Enter" || ev.key === " ") {
        ev.preventDefault();
        open();
      }
    });
  });
}

function openQuestion(i) {
  setFilter("all", true);
  const item = document.getElementById("q-" + i);
  item.open = true;
  item.scrollIntoView({ block: "center" });
}

let setFilter = () => {};

function explorer(data) {
  const qs = data.questions;
  const hurt = (q) => q.new !== q.gold && q.plain === q.gold;
  const filters = [
    ["all", "All", () => true],
    ["fixed", "Fixed", (q) => category(q) === "fixed"],
    ["broken", "Broken", (q) => category(q) === "broken"],
    ["missed", "Wrong both times", (q) => category(q) === "missed"],
    ["hurt", "Retrieval hurt", hurt],
  ];
  const bar = document.getElementById("filters");
  bar.innerHTML = filters.map(([id, label, fn]) => `<button class="filter" type="button" data-f="${id}" aria-pressed="${id === "all"}">${label}<b>${qs.filter(fn).length}</b></button>`).join("");
  const pick = (name, cat) => `<span class="pick">${cat ? `<span class="key" style="background:${cat}"></span>` : ""}${name}</span>`;
  document.getElementById("qlist").innerHTML = qs.map((q, i) => {
    const cat = category(q);
    const opts = q.options.map((o, j) => {
      const letter = "ABCD"[j];
      const picks = [];
      if (q.gold === letter) picks.push(pick("keyed answer", "var(--good)"));
      if (q.old === letter) picks.push(pick("first build", "var(--s2)"));
      if (q.new === letter) picks.push(pick("now", "var(--s1)"));
      if (q.plain === letter) picks.push(pick("no retrieval", ""));
      return `<li class="${q.gold === letter ? "gold" : ""}"><span class="opt-letter">${letter}</span><span>${esc(o)}</span><span class="picks">${picks.join("")}</span></li>`;
    }).join("");
    const silent = "ABCD".includes(q.old) ? "" : `<p class="reason">The first build gave no usable answer here.</p>`;
    return `<details class="qitem" id="q-${i}" data-cat="${cat}" data-hurt="${hurt(q)}">
      <summary><span class="q-num">${String(i + 1).padStart(3, "0")}</span><span class="q-text">${esc(q.question)}</span><span class="status">${keyDot(cat)}${CATS[cat].label}</span></summary>
      <div class="q-body"><ul class="options">${opts}</ul>${silent}<p class="reason"><b>Why the current build chose ${q.new}:</b> ${esc(q.reason)}</p></div>
    </details>`;
  }).join("");
  const more = document.getElementById("more");
  let expanded = false;
  let current = "all";
  setFilter = (id, showAll) => {
    current = id;
    if (showAll) expanded = true;
    bar.querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", b.dataset.f === id ? "true" : "false"));
    let shown = 0;
    let total = 0;
    document.querySelectorAll(".qitem").forEach((d) => {
      const on = id === "all" || (id === "hurt" ? d.dataset.hurt === "true" : d.dataset.cat === id);
      if (on) total++;
      const visible = on && (expanded || shown < 12);
      if (visible) shown++;
      d.style.display = visible ? "" : "none";
    });
    more.hidden = shown >= total;
    more.textContent = `Show all ${total} questions`;
  };
  more.addEventListener("click", () => setFilter(current, true));
  setFilter("all");
  bar.addEventListener("click", (ev) => {
    const b = ev.target.closest("button");
    if (b) setFilter(b.dataset.f);
  });
}

function theme() {
  const root = document.documentElement;
  try {
    const saved = localStorage.getItem("theme");
    if (saved) root.dataset.theme = saved;
  } catch (e) {}
  document.getElementById("theme").addEventListener("click", () => {
    const dark = root.dataset.theme ? root.dataset.theme === "dark" : window.matchMedia("(prefers-color-scheme: dark)").matches;
    root.dataset.theme = dark ? "light" : "dark";
    try {
      localStorage.setItem("theme", root.dataset.theme);
    } catch (e) {}
  });
}

theme();
tissue();
Promise.all([fetch("data/evaluation.json").then((r) => r.json()), fetch("data/examples.json").then((r) => r.json())])
  .then(([data, ex]) => {
    hero(data);
    demo(ex);
    tiles(data);
    explorer(data);
    const rank = { kept: 0, fixed: 1, broken: 2, missed: 3 };
    const order = data.questions.map((q, i) => i).sort((a, b) => rank[category(data.questions[a])] - rank[category(data.questions[b])] || a - b);
    const counts = {};
    data.questions.forEach((q) => (counts[category(q)] = (counts[category(q)] || 0) + 1));
    document.getElementById("legend-units").innerHTML = Object.keys(CATS).map((c) => `<li>${keyDot(c)}${CATS[c].label}<b>${counts[c] || 0}</b></li>`).join("");
    const draw = () => {
      drawSetups(data);
      drawUnits(data, order);
    };
    draw();
    let timer;
    new ResizeObserver(() => {
      clearTimeout(timer);
      timer = setTimeout(draw, 120);
    }).observe(document.querySelector("main"));
  })
  .catch(() => {
    document.querySelector("main").insertAdjacentHTML("afterbegin", '<section class="section"><h2>The page data did not load</h2><p class="sub">Reload in a moment.</p></section>');
  });
