/* BLM-SGAN Studio front-end (no dependencies) */
(() => {
  "use strict";

  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

  function h(tag, attrs = {}, ...children) {
    const el = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (v === false || v == null) continue;
      if (k === "class") el.className = v;
      else if (k === "text") el.textContent = v;
      else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
      else el.setAttribute(k, v === true ? "" : v);
    }
    for (const c of children.flat()) {
      if (c == null || c === false) continue;
      el.append(c.nodeType ? c : document.createTextNode(String(c)));
    }
    return el;
  }

  const store = {
    get(k, d) { try { const v = localStorage.getItem(k); return v == null ? d : JSON.parse(v); } catch { return d; } },
    set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch { /* ignore */ } },
  };

  // ------------------------------------------------------------------ state
  const S = {
    status: null,
    vocab: null,          // Set of words in SSA-GAN's vocabulary
    vocabList: [],
    wordsNum: 18,
    history: [],
    activeHistory: -1,
    count: 4,
    cmpCount: 3,
    busy: false,
  };

  const EXAMPLES = [
    "this small bird has a bright red crown and a white belly with black wings",
    "a yellow bird with black wings and a short pointed beak",
    "this bird has a blue head a white breast and grey wings",
    "a small brown bird with white stripes on its wings and a short beak",
    "this bird is black with a long tail and an orange beak",
    "a green bird with a yellow belly and a black crown",
  ];

  const PRESETS = {
    // a colour sweep: the same bird, one colour word changed
    colors: ["blue", "red", "yellow", "green", "black", "white"].map(
      (c) => `a colorful ${c} bird has wings with dark stripes and small eyes`),
    parts: [
      "this bird has a red head and a white belly",
      "this bird has a red head and a black belly",
      "this bird has a yellow head and a white belly",
      "this bird has a yellow head and a black belly",
    ],
    size: [
      "a small bird with a short beak and brown feathers",
      "a large bird with a long beak and brown feathers",
      "a small bird with a long tail and grey feathers",
      "a large bird with a short tail and grey feathers",
    ],
  };

  // ------------------------------------------------------------------ utils
  async function api(path, body) {
    const opt = body === undefined ? {} : {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
    };
    const res = await fetch(path, opt);
    let data = null;
    try { data = await res.json(); } catch { /* empty */ }
    if (!res.ok) {
      const d = data && data.detail;
      const msg = typeof d === "string" ? d : Array.isArray(d) ? d.map((x) => x.msg).join("; ") : `Request failed (${res.status})`;
      throw new Error(msg);
    }
    return data;
  }

  function toast(msg, kind = "") {
    const t = h("div", { class: `toast ${kind}`, text: msg });
    $("#toasts").append(t);
    setTimeout(() => t.remove(), kind === "error" ? 6500 : 3200);
  }

  const randSeed = () => Math.floor(Math.random() * 1_000_000);
  const fmtInt = (n) => (n == null ? "–" : Number(n).toLocaleString("en-US"));
  const fmtMs = (ms) => (ms < 1000 ? `${ms} ms` : `${(ms / 1000).toFixed(1)} s`);
  const timeNow = () => new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });

  // Same tokenisation idea as datasets.py: lowercase, \w+ runs, drop non-ASCII.
  function tokenize(text) {
    const m = (text || "").replace(/��/g, " ").toLowerCase().match(/[\p{L}\p{N}\p{M}_]+/gu) || [];
    return m.map((t) => t.replace(/[^\x00-\x7F]/g, "")).filter(Boolean);
  }

  function levenshtein(a, b, max) {
    if (Math.abs(a.length - b.length) > max) return max + 1;
    let prev = Array.from({ length: b.length + 1 }, (_, i) => i);
    for (let i = 1; i <= a.length; i++) {
      const cur = [i];
      let rowMin = i;
      for (let j = 1; j <= b.length; j++) {
        cur[j] = Math.min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1));
        if (cur[j] < rowMin) rowMin = cur[j];
      }
      if (rowMin > max) return max + 1;
      prev = cur;
    }
    return prev[b.length];
  }

  function suggest(word) {
    if (!S.vocabList.length || word.length < 3) return null;
    const max = word.length <= 4 ? 1 : 2;
    let best = null, bestD = max + 1;
    for (const w of S.vocabList) {
      const d = levenshtein(word, w, max);
      if (d < bestD) { best = w; bestD = d; if (d === 1) break; }
    }
    return best;
  }

  // ------------------------------------------------------------------ theme
  function initTheme() {
    const saved = store.get("blm-theme", null);
    if (saved) document.documentElement.dataset.theme = saved;
    $("#themeBtn").addEventListener("click", () => {
      const cur = document.documentElement.dataset.theme ||
        (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
      const next = cur === "dark" ? "light" : "dark";
      document.documentElement.dataset.theme = next;
      store.set("blm-theme", next);
    });
  }

  // ------------------------------------------------------------------ tabs
  function showTab(name) {
    for (const t of $$(".tab")) {
      const on = t.id === `tab-${name}`;
      t.setAttribute("aria-selected", on);
      $(`#${t.getAttribute("aria-controls")}`).hidden = !on;
    }
    if (location.hash !== `#${name}`) history.replaceState(null, "", name === "generate" ? location.pathname : `#${name}`);
  }
  function initTabs() {
    for (const t of $$(".tab")) t.addEventListener("click", () => showTab(t.id.replace("tab-", "")));
    const fromHash = location.hash.replace("#", "");
    if (["compare", "about"].includes(fromHash) && $(`#tab-${fromHash}`)) showTab(fromHash);
    $("#statusPill").addEventListener("click", () => { if ($("#tab-about")) showTab("about"); });
  }

  // ------------------------------------------------------------------ status
  let pollTimer = null;
  async function pollStatus() {
    clearTimeout(pollTimer);
    try {
      const st = await api("/api/status");
      const prevState = S.status && S.status.state;
      S.status = st;
      S.wordsNum = st.words_num || 18;
      if (st.state === "ready" && prevState !== "ready") renderModelPickers();
      renderStatus();
      if (st.state === "ready" && prevState !== "ready") { await loadVocab(); loadMetrics(); }
      if (st.state === "loading" || st.state === "idle") pollTimer = setTimeout(pollStatus, 1200);
    } catch {
      S.status = { state: "offline", message: "Cannot reach the server" };
      renderStatus();
      pollTimer = setTimeout(pollStatus, 3000);
    }
  }

  async function loadVocab() {
    try {
      const v = await api("/api/vocab");
      S.vocabList = v.words || [];
      S.vocab = S.vocabList.length ? new Set(S.vocabList) : null;
    } catch { S.vocab = null; S.vocabList = []; }
    renderTokens();
  }

  function renderStatus() {
    const st = S.status, dot = $("#statusDot"), txt = $("#statusText");
    dot.className = "dot";
    const info = st.info || {};
    if (st.state === "ready") {
      dot.classList.add("ready");
      const dev = info.device === "cuda" ? "GPU" : "CPU";
      const n = (st.models || []).length;
      txt.textContent = st.public ? "Ready" : `${dev} · ${n} model${n === 1 ? "" : "s"} · ready`;
    } else if (st.state === "loading" || st.state === "idle") {
      dot.classList.add("loading"); txt.textContent = `${st.message || "Loading"}…`;
    } else {
      dot.classList.add("error"); txt.textContent = st.state === "offline" ? "Server offline" : "Model error";
    }
    $("#genBtn").disabled = $("#cmpBtn").disabled = st.state !== "ready";
    renderBanner();
    renderModelCard();
  }

  function renderBanner() {
    const st = S.status, b = $("#banner");
    b.replaceChildren(); b.className = "banner"; b.hidden = true;
    const reloadBtn = () => h("button", { class: "ghost-btn", type: "button", onclick: reloadModel }, "Reload model");
    if (st.state === "error") {
      b.classList.add("error"); b.hidden = false;
      b.append(h("div", { class: "banner-body" }, h("strong", {}, "The model failed to load. "), st.message || ""), reloadBtn());
      return;
    }
    if (st.state === "offline") {
      b.classList.add("error"); b.hidden = false;
      b.append(h("div", { class: "banner-body" }, h("strong", {}, "Server offline. "),
        "Start it with ", h("code", {}, "start_app.bat"), " and this page reconnects automatically."));
      return;
    }
    if (st.state !== "ready" || st.vocab) return;
    b.hidden = false;
    b.append(h("div", { class: "banner-body" }, h("strong", {}, "Vocabulary file missing. "),
      "Captions go straight to BERT, unlike in training, and the SSA-GAN baseline is unavailable. Copy ",
      h("code", {}, "captions.pickle"), " into ", h("code", {}, "data/birds/"), "."), reloadBtn());
  }

  // ------------------------------------------------------------------ models
  const models = () => (S.status && S.status.models) || [];
  const modelById = (id) => models().find((m) => m.id === id);
  const isBaseline = (m) => m && m.kind === "ssagan";
  const isBLM = (m) => m && m.kind === "blm";
  const encoderLabel = (m) => (isBaseline(m) ? "DAMSM LSTM" : "BERT-DAMSM");
  const modelLabel = (m) => (m ? m.name : "");

  function renderModelPickers() {
    const list = models();
    const saved = store.get("blm-model", null);
    const def = list.some((m) => m.id === saved) ? saved : S.status.default_model;
    for (const sel of [$("#model"), $("#cmpModel")]) {
      const keep = sel.value;
      sel.replaceChildren(...list.map((m) => h("option", { value: m.id }, m.name)));
      sel.value = list.some((m) => m.id === keep) ? keep : def;
    }
    const checks = $("#cmpModels"), prev = new Set($$("input:checked", checks).map((i) => i.value));
    checks.replaceChildren(...list.map((m) => h("label", {},
      h("input", { type: "checkbox", value: m.id, checked: prev.size ? prev.has(m.id) : true }),
      h("span", {}, h("b", {}, m.name, h("span", { class: `tag${isBaseline(m) ? " base" : ""}` }, isBaseline(m) ? "baseline" : "ours")),
        h("small", {}, m.description)))));
    // comparing models needs at least two of them
    const multi = list.length > 1;
    $("#cmpBySeg").hidden = !multi;
    $("#cmpBySeg").previousElementSibling.hidden = !multi;
    if (!multi && S.cmpBy === "models") setCmpBy("captions");
    syncModelUI();
  }

  function syncModelUI() {
    const m = modelById($("#model").value);
    $("#modelDesc").textContent = m ? m.description : "";
    $("#modeHint").textContent = isBaseline(m) ? "SSA-GAN reads at most 18 words from the CUB vocabulary."
      : "BLM-SGAN reads the caption with its own BERT text encoder; any words work.";
    renderTokens();
  }

  function renderModelCard() {
    const st = S.status, info = st.info || {}, dl = $("#modelCard");
    if (!dl) return;   // the public site has no How-it-works page
    const rows = [["Status", st.state === "ready" ? "Ready" : (st.message || st.state)]];
    if (st.state === "ready") {
      for (const m of models()) rows.push([m.name, `${m.checkpoint} · ${encoderLabel(m)}`]);
      rows.push(
        ["Device", info.device_name || info.device],
        ["Generator", `${fmtInt(info.params_generator)} parameters each`],
        ["BERT encoder", `${info.bert} · ${fmtInt(info.params_text)} parameters`],
        ["Vocabulary", st.vocab ? `${fmtInt(info.vocab_size)}-word CUB vocabulary` : "missing"],
        ["PyTorch", info.torch],
      );
    }
    dl.replaceChildren(...rows.flatMap(([k, v]) => [h("dt", {}, k), h("dd", {}, v)]));
  }

  function renderLanguageTests(tests) {
    // the test of the reported model (newest file if several): one SSA-GAN row, two BLM-SGAN rows
    const t = tests.find((x) => x.tag === "blm_sgan_555") || tests[tests.length - 1];
    if (!t) return;
    const rw = t.rare_words.models, lc = t.long_captions.runs;
    const ep = (t.blm_sgan.match(/netG_(\d+)/) || [])[1];
    const pct = (v) => `${(100 * v).toFixed(0)}%`;
    const rp = (r) => `${r.orig[0].toFixed(1)}% → ${r.rare[0].toFixed(1)}%`;
    const tail = (k) => (lc[k] ? [`${lc[k].clip_score_tail[0].toFixed(4)}`, `${lc[k].r_precision_tail[0].toFixed(1)}%`] : ["–", "–"]);
    const best = (better, v) => h("td", { class: `num${better ? " best" : ""}` }, v);
    const ours = rw.blm_sgan, wins = ours.kept > rw.ssagan.kept;
    const [sC, sR] = tail("ssagan (first 18 words)"), [wC, wR] = tail("blm_sgan (whole caption)"),
      [fC, fR] = tail("blm_sgan (first 18 words)");
    const whole = lc["blm_sgan (whole caption)"], s18 = lc["ssagan (first 18 words)"];
    const tailWin = whole && whole.clip_score_tail[0] > s18.clip_score_tail[0];
    const tailRWin = whole && whole.r_precision_tail[0] > s18.r_precision_tail[0];
    const rareRWin = ours.r_precision_clip.rare[0] > rw.ssagan.r_precision_clip.rare[0];
    $("#langTable tbody").replaceChildren(
      h("tr", {}, h("td", {}, "SSA-GAN 590", h("br"), h("small", {}, "first 18 words")),
        best(!wins, pct(rw.ssagan.kept)), best(!rareRWin, rp(rw.ssagan.r_precision_clip)), best(!tailWin, sC),
        best(!tailRWin, sR)),
      h("tr", {}, h("td", {}, `BLM-SGAN${ep ? ` · epoch ${Number(ep)}` : ""}`, h("br"), h("small", {}, "whole caption")),
        best(wins, pct(ours.kept)), best(rareRWin, rp(ours.r_precision_clip)), best(tailWin, wC), best(tailRWin, wR)),
      h("tr", { class: "ref" }, h("td", {}, "BLM-SGAN, cut to the first 18 words"), h("td", { class: "num" }, "–"),
        h("td", { class: "num" }, "–"), h("td", { class: "num" }, fC), h("td", { class: "num" }, fR)));
  }

  function renderPaperScale(rows) {
    const by = (tag) => rows.find((r) => r.tag === tag);
    const fid = (r) => (r ? h("td", { class: "num" }, r.fid.toFixed(2)) : h("td", { class: "num muted" }, "–"));
    const pub = (v) => h("td", { class: "num" }, v == null ? "–" : v.toFixed(2));
    const measured = [
      ["DF-GAN, released model", "dfgan_released", null, 12.10],
      ["SSA-GAN 590", "ssagan_590_dfgan", "ssagan_590_galip", null],
      ["BLM-SGAN · epoch 555", "blm_sgan_555_dfgan", "blm_sgan_555_galip", null],
    ].filter(([, a, b]) => by(a) || (b && by(b)));
    if (!measured.length) return;
    const published = [["GALIP (CVPR 2023)", 10.08], ["TIGER (2025)", 9.38]];
    $("#paperTable tbody").replaceChildren(
      ...measured.map(([n, a, b, p]) => h("tr", {}, h("td", {}, n), fid(by(a)), fid(b && by(b)), pub(p))),
      ...published.map(([n, p]) => h("tr", { class: "ref" }, h("td", {}, n, h("br"), h("small", {}, "published")),
        h("td", { class: "num" }, "–"), h("td", { class: "num" }, "–"), pub(p))));
  }

  async function loadMetrics() {
    if (!$("#metricsTable")) return;   // the public site has no How-it-works page
    let all = {};
    try { all = await api("/api/metrics"); } catch { return; }
    renderLanguageTests(all.language_tests || []);
    renderPaperScale(all.paper_scale || []);
    const res = all.results || [];
    const tbody = $("#metricsTable tbody");
    const gen = res.filter((r) => r.tag !== "real_images" && r.fid != null);
    if (!gen.length) return;
    const real = res.find((r) => r.tag === "real_images");
    // the CUB-paper IS (StackGAN's CUB Inception) when it was computed, else the older ImageNet IS
    const isv = (r) => r.inception_score_cub || r.inception_score;
    const fidS = (r) => r.fid_uncropped ?? null;          // SSA-GAN's FID: 2,933 images vs the whole test photos
    const best = {
      fidS: Math.min(...gen.map((r) => fidS(r) ?? Infinity)),
      fid: Math.min(...gen.map((r) => r.fid)),
      is: Math.max(...gen.map((r) => (isv(r) ? isv(r)[0] : -1))),
      cs: Math.max(...gen.map((r) => r.clip_score ?? -1)),
      d: Math.max(...gen.map((r) => r.r_precision_damsm[0])),
      c: Math.max(...gen.map((r) => r.r_precision_clip[0])),
    };
    const num = (v, b, digits) => (v == null ? h("td", { class: "num" }, "–")
      : h("td", { class: `num${b != null && Math.abs(v - b) < 1e-9 ? " best" : ""}` }, v.toFixed(digits)));
    const name = (r) => {
      const m = models().find((x) => r.ckpt && r.ckpt.replace(/\\/g, "/").endsWith(`${x.id}/${x.checkpoint}`));
      return r.label || (m ? m.name : r.tag);
    };
    const enc = (r) => (r.model === "ssagan" ? "DAMSM LSTM" : "BERT-DAMSM");
    const rp = (v, b) => h("td", { class: `num${Math.abs(v[0] - b) < 1e-9 ? " best" : ""}` }, `${v[0].toFixed(1)}% `, h("small", {}, `±${v[1].toFixed(1)}`));
    const is = (r, b) => (isv(r)
      ? h("td", { class: `num${b != null && Math.abs(isv(r)[0] - b) < 1e-9 ? " best" : ""}` },
        `${isv(r)[0].toFixed(2)} `, h("small", {}, `±${isv(r)[1].toFixed(2)}`))
      : h("td", { class: "num" }, "–"));
    const rows = gen.sort((a, b) => (fidS(a) ?? Infinity) - (fidS(b) ?? Infinity) || a.fid - b.fid).map((r) => h("tr", {},
      h("td", {}, name(r), h("br"), h("small", {}, r.tag)), h("td", {}, enc(r)),
      num(fidS(r), best.fidS, 2), num(r.fid, best.fid, 2), is(r, best.is), num(r.clip_score, best.cs, 4),
      rp(r.r_precision_damsm, best.d), rp(r.r_precision_clip, best.c)));
    if (real) rows.push(h("tr", { class: "ref" }, h("td", {}, "Real test images"), h("td", {}, "–"),
      h("td", { class: "num" }, "–"), h("td", { class: "num" }, "–"), is(real, null), num(real.clip_score, null, 4),
      h("td", { class: "num" }, `${real.r_precision_damsm[0].toFixed(1)}%`), h("td", { class: "num" }, `${real.r_precision_clip[0].toFixed(1)}%`)));
    tbody.replaceChildren(...rows);
    $("#evalIntro").textContent = `CUB test split, ${fmtInt(gen[0].n_images)} generated images per model (one per test image), same captions and noise for every model.`;
  }

  async function reloadModel() {
    try { await api("/api/reload", {}); toast("Reloading model…"); } catch (e) { toast(e.message, "error"); }
    S.status = { state: "loading", message: "Reloading" }; renderStatus();
    pollTimer = setTimeout(pollStatus, 600);
  }

  // ------------------------------------------------------------------ prompt
  const cap = () => $("#caption");

  function renderTokens() {
    const box = $("#tokens"), toks = tokenize(cap().value);
    box.replaceChildren();
    let known = 0;
    const unknown = [];
    const open = isBLM(modelById($("#model").value));     // BLM-SGAN reads any words, no 18-word limit
    const vocab = open ? null : S.vocab;
    for (const t of toks) {
      const isKnown = !vocab || vocab.has(t);
      const chip = h("span", { class: "tok" }, t);
      if (vocab) {
        if (isKnown) {
          known++; chip.classList.add("known");
          if (known > S.wordsNum) { chip.classList.add("cut"); chip.title = `Only the first ${S.wordsNum} words are used`; }
        } else { chip.classList.add("unknown"); chip.title = "Not in the bird vocabulary, so it will be ignored"; unknown.push(t); }
      } else known++;
      box.append(chip);
    }
    const wc = $("#wordCount");
    wc.textContent = open ? `${known} word${known === 1 ? "" : "s"}` : `${known} / ${S.wordsNum} words`;
    wc.classList.toggle("over", !open && known > S.wordsNum);
    if (unknown.length) {
      const note = h("div", { class: "tok-note" }, `Not in the bird vocabulary: ${[...new Set(unknown)].join(", ")}.`);
      for (const u of [...new Set(unknown)].slice(0, 4)) {
        const s = suggest(u);
        if (s) note.append(" ", h("button", { type: "button", onclick: () => replaceWord(u, s) }, `${u} → ${s}`));
      }
      box.append(note);
    }
  }

  function replaceWord(from, to) {
    const re = new RegExp(`\\b${from.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}\\b`, "gi");
    cap().value = cap().value.replace(re, to);
    renderTokens(); cap().focus();
  }

  function initPrompt() {
    cap().value = store.get("blm-caption", EXAMPLES[0]);
    cap().addEventListener("input", () => { renderTokens(); store.set("blm-caption", cap().value); });
    cap().addEventListener("keydown", (e) => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); $("#genForm").requestSubmit(); } });
    $("#clearBtn").addEventListener("click", () => { cap().value = ""; renderTokens(); cap().focus(); });
    $("#examples").append(...EXAMPLES.map((ex) => h("button", {
      type: "button", class: "chip", title: ex,
      onclick: () => { cap().value = ex; renderTokens(); store.set("blm-caption", ex); cap().focus(); },
    }, ex.length > 38 ? `${ex.slice(0, 36)}…` : ex)));
    renderTokens();
  }

  function initSegmented(sel, key) {
    const seg = $(sel);
    seg.addEventListener("click", (e) => {
      const b = e.target.closest("button"); if (!b) return;
      for (const x of $$("button", seg)) x.setAttribute("aria-checked", x === b);
      S[key] = Number(b.dataset.v);
    });
  }

  function initControls() {
    initSegmented("#countSeg", "count");
    initSegmented("#cmpCountSeg", "cmpCount");
    $("#seed").value = randSeed();
    $("#diceBtn").addEventListener("click", () => { $("#seed").value = randSeed(); });
    $("#cmpDice").addEventListener("click", () => { $("#cmpSeed").value = randSeed(); });
    const sc = $("#scale"), sv = $("#scaleVal");
    sc.addEventListener("input", () => { sv.textContent = Number(sc.value).toFixed(2); });
  }

  // ------------------------------------------------------------------ generate
  function setBusy(btn, on) {
    S.busy = on; btn.classList.toggle("busy", on); btn.disabled = on || !(S.status && S.status.state === "ready");
    $(".btn-label", btn).textContent = on ? "Generating" : btn.id === "cmpBtn" ? "Compare" : "Generate";
  }

  async function onGenerate(e) {
    e.preventDefault();
    if (S.busy) return;
    const caption = cap().value.trim();
    if (!caption) { toast("Write a caption first.", "error"); cap().focus(); return; }
    if ($("#autoSeed").checked) $("#seed").value = randSeed();
    const body = {
      caption, n: S.count, seed: Math.max(0, parseInt($("#seed").value, 10) || 0),
      scale: Number($("#scale").value), save: $("#saveOut").checked, model: $("#model").value,
    };
    $("#genEmpty").hidden = true; $("#genResult").hidden = false;
    $("#resCaption").textContent = caption; $("#resMeta").replaceChildren(h("span", {}, "Generating…"));
    $("#resDropped").hidden = true; $("#resSaved").hidden = true;
    $("#resGrid").replaceChildren(...Array.from({ length: body.n }, () => h("div", { class: "skeleton" })));
    setBusy($("#genBtn"), true);
    try {
      const res = await api("/api/generate", body);
      res.time = timeNow();
      S.history.unshift(res); S.activeHistory = 0;
      if (S.history.length > 30) S.history.pop();
      renderResult(res); renderHistory();
    } catch (err) {
      toast(err.message, "error");
      if (S.activeHistory >= 0) renderResult(S.history[S.activeHistory]);
      else { $("#genResult").hidden = true; $("#genEmpty").hidden = false; }
    } finally { setBusy($("#genBtn"), false); }
  }

  const modeLabel = (m) => (m === "lstm" ? "DAMSM LSTM" : m === "bert" ? "BERT-DAMSM text encoder" : m || "");

  function renderResult(res) {
    $("#genEmpty").hidden = true; $("#genResult").hidden = false;
    $("#resCaption").textContent = res.caption;
    const seeds = res.images.map((i) => i.seed);
    const meta = [
      h("span", { class: "model-chip" }, res.model ? res.model.name : ""),
      h("span", {}, h("b", {}, res.images.length), res.images.length === 1 ? " image" : " images"),
      h("span", {}, "seed ", h("b", { class: "mono" }, seeds.length > 1 ? `${seeds[0]}–${seeds[seeds.length - 1]}` : seeds[0])),
      h("span", {}, fmtMs(res.ms)),
      h("span", {}, modeLabel(res.mode)),
    ];
    if (res.scale !== 1) meta.push(h("span", {}, "noise ×", h("b", { class: "mono" }, res.scale.toFixed(2))));
    $("#resMeta").replaceChildren(...meta);
    const saved = $("#resSaved"), path = (res.saved_to || "").replace(/\\/g, "/");
    saved.hidden = !path; saved.title = path;
    saved.replaceChildren(...(path ? ["Saved to ", h("span", { class: "mono" }, path)] : []));
    const a = res.analysis || {};
    const dropped = $("#resDropped");
    const notes = [];
    if (a.dropped && a.dropped.length) notes.push(`Ignored (not in vocabulary): ${a.dropped.join(", ")}.`);
    if (a.truncated) notes.push(`Only the first ${S.wordsNum} known words were used.`);
    dropped.hidden = !notes.length; dropped.textContent = notes.join(" ");
    $("#resGrid").replaceChildren(...res.images.map((img, i) => imageCard(img, res, i)));
  }

  function imageCard(img, ctx, i) {
    const minis = img.masks.slice(-3).map((m) => h("img", { src: m.png, alt: "", title: `${m.size}×${m.size} mask` }));
    return h("button", { type: "button", class: "card-img appear", style: `animation-delay:${i * 45}ms`,
      "aria-label": `Open image with seed ${img.seed}`, onclick: () => openDetail(img, ctx) },
      h("img", { src: img.image, alt: `Generated image for “${ctx.caption}”, seed ${img.seed}`, width: 256, height: 256 }),
      h("div", { class: "cap" }, h("span", {}, "seed ", h("span", { class: "mono" }, img.seed)), h("span", { class: "masks-mini" }, minis)));
  }

  function renderHistory() {
    const list = $("#historyList");
    list.replaceChildren(...S.history.map((r, i) => h("li", {},
      h("button", { type: "button", class: `history-item${i === S.activeHistory ? " active" : ""}`,
        onclick: () => { S.activeHistory = i; renderResult(r); renderHistory(); } },
        h("img", { src: r.images[0].image, alt: "" }),
        h("span", {}, h("span", { class: "h-cap" }, r.caption),
          h("span", { class: "h-meta" }, `${r.time} · ${r.model ? r.model.name : ""} · #${r.images[0].seed}`))))));
    $("#historyEmpty").hidden = S.history.length > 0;
    $("#clearHistory").hidden = S.history.length === 0;
  }

  // ------------------------------------------------------------------ detail dialog
  let detailCtx = null;
  function openDetail(img, ctx) {
    detailCtx = { img, ctx, stage: img.masks.length - 1 };
    $("#dImg").src = img.image;
    $("#dImg").alt = `Generated image, seed ${img.seed}`;
    $("#dCaption").textContent = ctx.caption;
    $("#dSeed").textContent = img.seed;
    $("#dMode").textContent = modeLabel(ctx.mode);
    $("#dModel").textContent = ctx.model ? modelLabel(ctx.model) : "";
    $("#dScale").textContent = (ctx.scale ?? 1).toFixed(2);
    const slug = ctx.caption.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "").slice(0, 40);
    $("#dDownload").href = img.image;
    $("#dDownload").download = `${ctx.model ? ctx.model.id : "blm-sgan"}_${slug}_seed-${img.seed}.png`;
    const strip = $("#maskStrip");
    strip.replaceChildren(...img.masks.map((m, k) => h("button", {
      type: "button", class: "mask-tile", "aria-pressed": k === detailCtx.stage, title: `Show the ${m.size}×${m.size} mask on the image`,
      onclick: () => { detailCtx.stage = k; $("#ovOn").checked = true; syncOverlay(); },
    }, h("img", { src: m.png, alt: `${m.size}×${m.size} mask` }), h("span", {}, `${m.size}²`))));
    strip.after(maskScale());
    syncOverlay();
    const dlg = $("#detail");
    if (!dlg.open) dlg.showModal();
  }

  function maskScale() {
    $$(".mask-scale").forEach((n) => n.remove());
    return h("div", { class: "mask-scale" }, h("span", {}, "text has no effect"), h("i"), h("span", {}, "strongest effect"));
  }

  function syncOverlay() {
    if (!detailCtx) return;
    const on = $("#ovOn").checked, m = detailCtx.img.masks[detailCtx.stage];
    const layer = $("#dMask");
    layer.src = m.png;
    layer.style.opacity = on ? $("#ovOpacity").value : 0;
    $$(".mask-tile").forEach((t, k) => t.setAttribute("aria-pressed", on && k === detailCtx.stage));
  }

  function initDetail() {
    const dlg = $("#detail");
    $("#dClose").addEventListener("click", () => dlg.close());
    dlg.addEventListener("click", (e) => { if (e.target === dlg) dlg.close(); });
    $("#ovOn").addEventListener("change", syncOverlay);
    $("#ovOpacity").addEventListener("input", () => { if (!$("#ovOn").checked) $("#ovOn").checked = true; syncOverlay(); });
    $("#dReuse").addEventListener("click", () => {
      if (!detailCtx) return;
      $("#seed").value = detailCtx.img.seed; $("#autoSeed").checked = false;
      $("#cmpSeed").value = detailCtx.img.seed;
      if (detailCtx.ctx.caption) { cap().value = detailCtx.ctx.caption; renderTokens(); }
      if (detailCtx.ctx.model && modelById(detailCtx.ctx.model.id)) { $("#model").value = detailCtx.ctx.model.id; syncModelUI(); }
      dlg.close(); showTab("generate");
      toast(`Seed ${detailCtx.img.seed} set. Generate 1 image to reproduce it, or change the caption.`);
    });
  }

  // ------------------------------------------------------------------ compare
  function cmpRow(text = "") {
    const ta = h("textarea", { rows: 1, maxlength: 300, "aria-label": "Caption", placeholder: "describe a bird…" });
    ta.value = text;
    ta.addEventListener("keydown", (e) => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); $("#cmpForm").requestSubmit(); } });
    ta.addEventListener("input", saveCmp);
    const rm = h("button", { type: "button", class: "icon-btn", "aria-label": "Remove caption", title: "Remove",
      onclick: () => { if ($$("#cmpList li").length > 1) { li.remove(); saveCmp(); } } });
    rm.innerHTML = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6 6l12 12M18 6 6 18"/></svg>';
    const li = h("li", { class: "cmp-row" }, ta, rm);
    return li;
  }
  const cmpCaptions = () => $$("#cmpList textarea").map((t) => t.value.trim()).filter(Boolean);
  function saveCmp() { store.set("blm-compare", $$("#cmpList textarea").map((t) => t.value)); }
  function setCmp(list) { $("#cmpList").replaceChildren(...list.map((c) => cmpRow(c))); saveCmp(); }

  function setCmpBy(v) {
    S.cmpBy = v;
    for (const b of $$("#cmpBySeg button")) b.setAttribute("aria-checked", b.dataset.v === v);
    $("#cmpCaptionsMode").hidden = v !== "captions";
    $("#cmpModelsMode").hidden = v !== "models";
    store.set("blm-compare-by", v);
  }

  function initCompare() {
    setCmp(store.get("blm-compare", null) || PRESETS.colors.slice(0, 4));
    $("#cmpAdd").addEventListener("click", () => {
      if ($$("#cmpList li").length >= 8) { toast("Up to 8 captions."); return; }
      const last = $$("#cmpList textarea").pop();
      const li = cmpRow(last ? last.value : ""); $("#cmpList").append(li); $("textarea", li).focus(); saveCmp();
    });
    $("#cmpPreset").addEventListener("change", (e) => { const p = PRESETS[e.target.value]; if (p) setCmp(p); e.target.value = ""; });
    $("#cmpBySeg").addEventListener("click", (e) => { const b = e.target.closest("button"); if (b) setCmpBy(b.dataset.v); });
    setCmpBy(store.get("blm-compare-by", "captions"));
    $("#cmpCaption").addEventListener("keydown", (e) => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); $("#cmpForm").requestSubmit(); } });
    $("#cmpForm").addEventListener("submit", onCompare);
  }

  function highlightDiff(caption, all) {
    const sets = all.map((c) => new Set(tokenize(c)));
    const common = sets.length > 1 ? [...sets[0]].filter((w) => sets.every((s) => s.has(w))) : [];
    const commonSet = new Set(common);
    const frag = [];
    for (const part of caption.split(/(\s+)/)) {
      const t = tokenize(part)[0];
      frag.push(t && sets.length > 1 && !commonSet.has(t) ? h("mark", {}, part) : part);
    }
    return frag;
  }

  async function onCompare(e) {
    e.preventDefault();
    if (S.busy) return;
    const byModels = S.cmpBy === "models";
    const seed = Math.max(0, parseInt($("#cmpSeed").value, 10) || 0), scale = Number($("#scale").value);
    let url, body, labels;
    if (byModels) {
      const caption = $("#cmpCaption").value.trim();
      const ids = $$("#cmpModels input:checked").map((i) => i.value);
      if (!caption) { toast("Write a caption first.", "error"); return; }
      if (!ids.length) { toast("Choose at least one model.", "error"); return; }
      url = "/api/compare_models"; body = { caption, models: ids, n_noise: S.cmpCount, seed, scale };
      labels = ids.map((id) => (modelById(id) || {}).name || id);
    } else {
      const captions = cmpCaptions();
      if (!captions.length) { toast("Add at least one caption.", "error"); return; }
      url = "/api/compare"; body = { captions, n_noise: S.cmpCount, seed, scale, model: $("#cmpModel").value };
      labels = captions;
    }
    $("#cmpEmpty").hidden = true; $("#cmpResult").hidden = false;
    $("#cmpMeta").replaceChildren(h("span", {}, "Generating…"));
    const table = $("#cmpTable");
    table.style.setProperty("--cols", body.n_noise);
    table.replaceChildren(...labels.map((c) => h("div", { class: "cmp-line" }, h("div", { class: "cmp-cap" }, c),
      ...Array.from({ length: body.n_noise }, () => h("div", { class: "skeleton" })))));
    setBusy($("#cmpBtn"), true);
    try {
      const res = await api(url, body);
      const head = h("div", { class: "cmp-head" }, h("span", {}, byModels ? "Model" : "Caption"), ...res.seeds.map((s) => h("span", {}, `noise #${s}`)));
      if (byModels) {
        $("#cmpMeta").replaceChildren(
          h("span", {}, h("b", {}, res.rows.length), " models × ", h("b", {}, res.seeds.length), " noise samples"),
          h("span", {}, fmtMs(res.ms)), h("span", { class: "cmp-caption-meta" }, `“${res.caption}”`));
        table.replaceChildren(head, ...res.rows.map((row) => h("div", { class: "cmp-line" },
          h("div", { class: "cmp-cap" }, row.model.name, h("span", { class: `tag${isBaseline(row.model) ? " base" : ""}` }, isBaseline(row.model) ? "baseline" : "ours"),
            h("small", { class: "cmp-sub" }, modeLabel(row.mode))),
          ...row.images.map((img, i) => imageCard(img, { caption: row.caption, mode: row.mode, scale, model: row.model }, i)))));
      } else {
        $("#cmpMeta").replaceChildren(
          h("span", { class: "model-chip" }, res.model.name),
          h("span", {}, h("b", {}, res.rows.length), " captions × ", h("b", {}, res.seeds.length), " noise samples"),
          h("span", {}, fmtMs(res.ms)), h("span", {}, modeLabel(res.mode)));
        const all = res.rows.map((r) => r.caption);
        table.replaceChildren(head, ...res.rows.map((row) => h("div", { class: "cmp-line" },
          h("div", { class: "cmp-cap" }, highlightDiff(row.caption, all)),
          ...row.images.map((img, i) => imageCard(img, { caption: row.caption, mode: res.mode, scale, model: res.model }, i)))));
      }
    } catch (err) {
      toast(err.message, "error");
      $("#cmpResult").hidden = true; $("#cmpEmpty").hidden = false;
    } finally { setBusy($("#cmpBtn"), false); }
  }

  // ------------------------------------------------------------------ boot
  function init() {
    initTheme();
    initTabs();
    initPrompt();
    initControls();
    initDetail();
    initCompare();
    $("#model").addEventListener("change", () => { store.set("blm-model", $("#model").value); syncModelUI(); });
    $("#genForm").addEventListener("submit", onGenerate);
    $("#clearHistory").addEventListener("click", () => { S.history = []; S.activeHistory = -1; renderHistory();
      $("#genResult").hidden = true; $("#genEmpty").hidden = false; });
    renderHistory();
    pollStatus();
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
