const NS = "http://www.w3.org/2000/svg";
const tooltip = document.getElementById("tooltip");
const CATS = {
  kept: { label: "Right both times", short: "Right twice", color: "var(--s1)" },
  fixed: { label: "Fixed by the new build", short: "Fixed", color: "var(--s3)" },
  broken: { label: "Broken by the new build", short: "Broken", color: "var(--s2)" },
  missed: { label: "Wrong both times", short: "Wrong twice", color: "var(--neutral)", hollow: true },
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

function tissue(id, start, left) {
  const svg = document.getElementById(id);
  if (!svg) return;
  const w = 1200;
  const h = 700;
  svg.setAttribute("viewBox", `0 0 ${w} ${h}`);
  svg.setAttribute("preserveAspectRatio", "xMidYMid slice");
  let seed = start;
  const rnd = () => {
    seed = (seed * 16807) % 2147483647;
    return seed / 2147483647;
  };
  for (let i = 0; i < 90; i++) {
    const x = w * (left + (1.06 - left) * rnd());
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

const API = (new URLSearchParams(location.search).get("api") || window.PATHOCHAT_API || "").replace(/\/$/, "");
const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const wait = (ms) => new Promise((done) => setTimeout(done, ms));

function renderAnswer(text) {
  if (!window.marked) return `<p style="white-space:pre-wrap">${esc(text)}</p>`;
  const html = window.marked.parse(text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;"));
  return html.replace(/<table>/g, '<div class="table-wrap"><table>').replace(/<\/table>/g, "</table></div>");
}

function spoken(text) {
  return text.replace(/\|[\s:|-]+\|/g, " ").replace(/[#*_`|>]/g, " ").replace(/\s+/g, " ").trim();
}

function chat(ex) {
  const box = document.getElementById("messages");
  const input = document.getElementById("input");
  const form = document.getElementById("composer");
  const mic = document.getElementById("mic");
  const hint = document.getElementById("hint");
  const api = document.getElementById("api");
  const apiText = document.getElementById("api-text");
  const saved = new Map(ex.examples.map((e) => [e.question.toLowerCase(), e]));
  const Speech = window.SpeechRecognition || window.webkitSpeechRecognition;
  let live = false;
  let busy = false;
  let rec = null;
  let recorder = null;

  const status = (state, text) => {
    api.dataset.state = state;
    apiText.textContent = text;
  };

  const grow = () => {
    input.style.height = "auto";
    input.style.height = Math.min(160, input.scrollHeight) + "px";
  };

  const add = (role, html) => {
    const node = document.createElement("div");
    node.className = "msg " + role + " fade";
    node.innerHTML = html;
    box.appendChild(node);
    box.scrollTop = box.scrollHeight;
    return node;
  };

  const greet = () => {
    if (window.speechSynthesis) window.speechSynthesis.cancel();
    box.innerHTML = "";
    add("bot", '<div class="answer"><p>Ask me about a disease, a cell or a mechanism. I search Robbins &amp; Cotran and Pathoma, answer from what I find, and show you the passages I was given.</p></div>');
  };

  const finish = (node, data, note) => {
    const sources = data.sources.map((s) => `<div class="source"><div class="source-head">${esc(s.book)} · PDF page ${s.page}</div><p>“${esc(s.excerpt)}…”</p></div>`).join("");
    node.innerHTML = `<div class="chat-meta"><span class="tag">${esc(data.model.split("/").pop())}</span><span class="tag">${data.sources.length} passages retrieved</span><span class="tag">${note}</span></div>
      <div class="answer"></div>
      <details class="sources"><summary>The ${data.sources.length} passages it was given</summary>${sources}</details>
      ${window.speechSynthesis ? '<div class="msg-actions"><button type="button" class="textbutton listen">Listen to this answer</button></div>' : ""}`;
    const target = node.querySelector(".answer");
    const text = data.answer;
    const steps = reduce ? 1 : 22;
    let i = 0;
    const tick = () => {
      i++;
      const cut = i >= steps ? text.length : Math.max(0, text.lastIndexOf(" ", Math.floor((text.length * i) / steps)));
      target.innerHTML = renderAnswer(text.slice(0, cut));
      if (i < steps) setTimeout(tick, 28);
    };
    tick();
    const listen = node.querySelector(".listen");
    if (listen) {
      listen.addEventListener("click", () => {
        const synth = window.speechSynthesis;
        if (synth.speaking) {
          synth.cancel();
          listen.textContent = "Listen to this answer";
          return;
        }
        const voice = new SpeechSynthesisUtterance(spoken(text));
        voice.lang = "en-US";
        voice.rate = 1.02;
        voice.onend = () => (listen.textContent = "Listen to this answer");
        listen.textContent = "Stop";
        synth.speak(voice);
      });
    }
  };

  const send = async (raw) => {
    const question = String(raw).trim().replace(/\s+/g, " ");
    if (!question || busy) return;
    busy = true;
    form.classList.add("busy");
    input.value = "";
    grow();
    hint.textContent = "";
    const asked = add("user", `<p class="bubble-q">${esc(question)}</p>`);
    const node = add("bot", '<div class="thinking"><span></span><span></span><span></span><em>Searching 23,814 passages</em></div>');
    const stage = setTimeout(() => {
      const em = node.querySelector("em");
      if (em) em.textContent = "Writing the answer";
    }, 1300);
    const hit = saved.get(question.toLowerCase());
    const fromSaved = () => finish(node, { answer: hit.answer, sources: hit.sources, model: ex.model }, "saved answer");
    try {
      if (live) {
        const r = await fetch(API + "/ask", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ question }) });
        const data = await r.json();
        if (!r.ok) throw new Error(data.detail || "The server did not answer.");
        finish(node, data, "live");
      } else if (hit) {
        await wait(reduce ? 0 : 650);
        fromSaved();
      } else {
        throw new Error("The live server is not connected right now, so I can only answer the suggested questions below.");
      }
    } catch (err) {
      if (hit) fromSaved();
      else node.innerHTML = `<div class="answer"><p class="problem">${esc(err instanceof TypeError ? "The server could not be reached. Try again in a moment." : err.message)}</p></div>`;
    }
    clearTimeout(stage);
    busy = false;
    form.classList.remove("busy");
    box.scrollTop = asked.offsetTop - 10;
  };

  const check = async (tries) => {
    if (!API) {
      status("off", "Saved answers only");
      return;
    }
    try {
      const r = await fetch(API + "/health", { signal: AbortSignal.timeout(25000) });
      if (!r.ok) throw new Error("down");
      const d = await r.json();
      live = true;
      status("live", "Live · " + d.model.split("/").pop());
    } catch (e) {
      if (tries > 0) {
        status("wait", "Waking the server");
        await wait(3000);
        return check(tries - 1);
      }
      status("off", "Server offline · saved answers only");
    }
  };

  const micOn = (on) => {
    mic.setAttribute("aria-pressed", on ? "true" : "false");
  };

  const dictate = () => {
    rec = new Speech();
    rec.lang = "en-US";
    rec.interimResults = true;
    rec.onresult = (ev) => {
      input.value = Array.from(ev.results).map((r) => r[0].transcript).join("");
      grow();
    };
    rec.onerror = (ev) => {
      hint.textContent = ev.error === "not-allowed" ? "Allow the microphone in your browser to speak a question." : "I could not hear that. Try again.";
    };
    rec.onend = () => {
      micOn(false);
      rec = null;
      if (input.value.trim()) send(input.value);
    };
    micOn(true);
    hint.textContent = "Listening. Ask your question.";
    rec.start();
  };

  const record = async () => {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const chunks = [];
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const analyser = ctx.createAnalyser();
    analyser.fftSize = 512;
    ctx.createMediaStreamSource(stream).connect(analyser);
    const wave = new Uint8Array(analyser.fftSize);
    let spoke = false;
    let quiet = 0;
    recorder = new MediaRecorder(stream);
    const watch = setInterval(() => {
      analyser.getByteTimeDomainData(wave);
      let peak = 0;
      for (const v of wave) peak = Math.max(peak, Math.abs(v - 128));
      if (peak > 10) {
        spoke = true;
        quiet = 0;
      } else if (spoke && (quiet += 100) >= 1500 && recorder && recorder.state === "recording") {
        recorder.stop();
      }
    }, 100);
    const limit = setTimeout(() => recorder && recorder.state === "recording" && recorder.stop(), 25000);
    recorder.ondataavailable = (e) => chunks.push(e.data);
    recorder.onstop = async () => {
      clearInterval(watch);
      clearTimeout(limit);
      stream.getTracks().forEach((t) => t.stop());
      ctx.close();
      const blob = new Blob(chunks, { type: recorder.mimeType || "audio/webm" });
      recorder = null;
      micOn(false);
      if (!spoke) {
        hint.textContent = "I could not hear a question. Try again.";
        return;
      }
      hint.textContent = "Transcribing your question.";
      try {
        const r = await fetch(API + "/transcribe", { method: "POST", headers: { "Content-Type": blob.type }, body: blob });
        const d = await r.json();
        if (!r.ok) throw new Error(d.detail);
        hint.textContent = "";
        if (d.text) send(d.text);
        else hint.textContent = "I could not hear a question. Try again.";
      } catch (e) {
        hint.textContent = e.message || "The recording could not be transcribed. Try again.";
      }
    };
    recorder.start();
    micOn(true);
    hint.textContent = "Listening. Ask your question, then pause.";
  };

  mic.addEventListener("click", () => {
    if (rec) return rec.stop();
    if (recorder) return recorder.state === "recording" && recorder.stop();
    if (busy) return;
    if (live && navigator.mediaDevices && window.MediaRecorder) {
      record().catch(() => {
        micOn(false);
        hint.textContent = "Allow the microphone in your browser to speak a question.";
      });
    } else if (Speech) {
      dictate();
    } else {
      hint.textContent = "Voice input needs the live server, or Chrome, Edge or Safari.";
    }
  });

  form.addEventListener("submit", (ev) => {
    ev.preventDefault();
    send(input.value);
  });
  input.addEventListener("input", grow);
  input.addEventListener("keydown", (ev) => {
    if (ev.key === "Enter" && !ev.shiftKey) {
      ev.preventDefault();
      send(input.value);
    }
  });
  document.getElementById("clear").addEventListener("click", greet);
  const chips = document.getElementById("suggest");
  chips.innerHTML = ex.examples.map((e, i) => `<button class="chipbutton" type="button" data-i="${i}">${esc(e.question)}</button>`).join("");
  chips.addEventListener("click", (ev) => {
    const b = ev.target.closest("button");
    if (b) send(ex.examples[Number(b.dataset.i)].question);
  });

  greet();
  status("wait", "Checking the server");
  check(8).then(() => {
    const m = location.hash.match(/^#ask=(.+)/);
    if (m) send(decodeURIComponent(m[1]));
  });
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

let selectQuestion = () => {};

function openQuestion(i) {
  selectQuestion(i);
  document.getElementById("explorer").scrollIntoView({ block: "start" });
}

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
  const tray = document.getElementById("tray");
  const view = document.getElementById("specimen");
  let active = "all";
  let current = 0;

  const matches = () => {
    const test = filters.find((f) => f[0] === active)[2];
    return qs.map((q, i) => i).filter((i) => test(qs[i]));
  };

  const pick = (name, color) => `<span class="pick">${color ? `<span class="key" style="background:${color}"></span>` : ""}${name}</span>`;

  const render = () => {
    const q = qs[current];
    const cat = category(q);
    const list = matches();
    const pos = list.indexOf(current);
    const opts = q.options.map((o, j) => {
      const letter = "ABCD"[j];
      const picks = [];
      if (q.gold === letter) picks.push(pick("keyed answer", "var(--good)"));
      if (q.old === letter) picks.push(pick("first build", "var(--s2)"));
      if (q.new === letter) picks.push(pick("now", "var(--s1)"));
      if (q.plain === letter) picks.push(pick("no retrieval", ""));
      return `<li class="${q.gold === letter ? "gold" : ""}"><span class="opt-letter">${letter}</span><span>${esc(o)}</span><span class="picks">${picks.join("")}</span></li>`;
    }).join("");
    const silent = "ABCD".includes(q.old) ? "" : '<p class="reason">The first build gave no usable answer here.</p>';
    view.innerHTML = `<div class="slide-label"><span>Specimen ${String(current + 1).padStart(3, "0")}</span><span>${CATS[cat].short}</span><span>MedMCQA</span></div>
      <div class="slide-body fade">
        <p class="status">${keyDot(cat)}${CATS[cat].label}${hurt(q) ? " · retrieval hurt" : ""}</p>
        <h3 class="specimen-q">${esc(q.question)}</h3>
        <ul class="options">${opts}</ul>
        ${silent}
        <p class="reason"><b>Why the current build chose ${esc(q.new)}:</b> ${esc(q.reason)}</p>
        <div class="stepper">
          <button type="button" class="step" data-d="-1" aria-label="Previous question">←</button>
          <span>${pos >= 0 ? pos + 1 : 1} of ${list.length}</span>
          <button type="button" class="step" data-d="1" aria-label="Next question">→</button>
        </div>
      </div>`;
    tray.querySelectorAll(".well").forEach((w) => {
      const i = Number(w.dataset.i);
      w.setAttribute("aria-current", i === current ? "true" : "false");
      w.classList.toggle("dim", !list.includes(i));
    });
  };

  const step = (d) => {
    const list = matches();
    if (!list.length) return;
    const pos = list.indexOf(current);
    current = list[(pos + d + list.length) % list.length];
    render();
  };

  const setFilter = (id) => {
    active = id;
    bar.querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", b.dataset.f === id ? "true" : "false"));
    const list = matches();
    if (!list.includes(current)) current = list[0];
    render();
  };

  selectQuestion = (i) => {
    active = "all";
    bar.querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", b.dataset.f === "all" ? "true" : "false"));
    current = i;
    render();
  };

  bar.innerHTML = filters.map(([id, label, fn]) => `<button class="filter" type="button" data-f="${id}" aria-pressed="${id === "all"}">${label}<b>${qs.filter(fn).length}</b></button>`).join("");
  tray.innerHTML = qs.map((q, i) => {
    const cat = category(q);
    return `<button class="well" type="button" data-i="${i}" aria-label="Question ${i + 1}: ${CATS[cat].label}"><span class="nucleus${CATS[cat].hollow ? " hollow" : ""}" style="--c:${CATS[cat].color}"></span></button>`;
  }).join("");

  bar.addEventListener("click", (ev) => {
    const b = ev.target.closest("button");
    if (b) setFilter(b.dataset.f);
  });
  tray.addEventListener("click", (ev) => {
    const w = ev.target.closest(".well");
    if (!w) return;
    current = Number(w.dataset.i);
    render();
  });
  tray.addEventListener("pointermove", (ev) => {
    const w = ev.target.closest(".well");
    if (!w) return hideTip();
    const i = Number(w.dataset.i);
    const q = qs[i];
    showTip(`<div class="t-head">Specimen ${String(i + 1).padStart(3, "0")}</div><div>${esc(q.question.slice(0, 110))}${q.question.length > 110 ? "…" : ""}</div>`, ev.clientX, ev.clientY);
  });
  tray.addEventListener("pointerleave", hideTip);
  tray.addEventListener("keydown", (ev) => {
    if (ev.key === "ArrowRight" || ev.key === "ArrowLeft") {
      ev.preventDefault();
      step(ev.key === "ArrowRight" ? 1 : -1);
      const now = tray.querySelector('[aria-current="true"]');
      if (now) now.focus();
    }
  });
  view.addEventListener("click", (ev) => {
    const b = ev.target.closest(".step");
    if (b) step(Number(b.dataset.d));
  });

  const counts = {};
  qs.forEach((q) => (counts[category(q)] = (counts[category(q)] || 0) + 1));
  document.getElementById("legend-tray").innerHTML = Object.keys(CATS).map((c) => `<li>${keyDot(c)}${CATS[c].label}<b>${counts[c] || 0}</b></li>`).join("");
  render();
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
tissue("tissue", 7, 0.28);
tissue("tissue2", 23, -0.04);
Promise.all([fetch("data/evaluation.json").then((r) => r.json()), fetch("data/examples.json").then((r) => r.json())])
  .then(([data, ex]) => {
    hero(data);
    chat(ex);
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
