"""The built-in query page: streaming answer, clickable citations, evidence graph, stage timings.

Everything the page renders comes from the network (model output, retrieved text, entity
names) and is treated as untrusted: the script builds DOM with textContent/createElementNS
only, never innerHTML, and only turns retrieved http(s) source IDs into links, never anything the
model writes. The inline script and stylesheet are pinned by SHA-256 in the CSP header, so
injected markup could not run script even if an escaping bug slipped in.
"""

from __future__ import annotations

import base64
import hashlib

from .brand import THEME_COLOR

EXAMPLES = (
    "Which institution is the paper written by Lluís Garcia-Pueyo affiliated with?",
    "How is A General Language Assistant as a Laboratory for Alignment classified?",
    "Which institution is associated with A Mathematical Framework for Transformer Circuits?",
    "Where is Alice Johnson's company based?",
)

STYLE = r"""
:root { color-scheme: light; --ink: #17221f; --paper: #eef2f0; --green: #17624f; --mint: #5cd6a6;
  --muted: #52635e; --line: #c4d0cb; --card: #f8faf9; --soft: #dfe7e3; }
* { box-sizing: border-box; }
body { margin: 0; background: var(--paper); color: var(--ink); font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1240px; margin: 0 auto; padding: 5vh 24px 64px; }
.brand { display: flex; align-items: center; gap: 12px; margin-bottom: 20px; }
.brand img { width: 40px; height: 40px; }
.kicker { margin: 0; color: #687a73; letter-spacing: .04em; font-size: .85rem; }
h1 { font: 700 clamp(2.4rem, 6vw, 4.6rem)/.95 Georgia, serif; margin: 0 0 10px; }
h2 { font: 700 1.15rem Georgia, serif; margin: 0 0 12px; }
.lede { color: var(--muted); margin: 0; }
form { display: flex; gap: 10px; margin: 26px 0 12px; }
input { flex: 1; min-width: 0; padding: 14px 16px; border: 1px solid #aabbb4; border-radius: 8px; font: inherit; background: #fff; }
button { font: inherit; cursor: pointer; }
#submit { padding: 14px 24px; border: 0; border-radius: 8px; background: var(--green); color: #fff; font-weight: 700; }
#submit:disabled { opacity: .55; cursor: wait; }
.examples { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 22px; }
.examples button { border: 1px solid var(--line); background: var(--card); border-radius: 999px; padding: 5px 12px; color: var(--muted); font-size: .85rem; }
.examples button:hover { border-color: var(--green); color: var(--green); }
.timeline { display: flex; gap: 2px; height: 30px; border-radius: 8px; overflow: hidden; background: var(--soft); margin-bottom: 6px; }
.timeline div { min-width: 2px; display: flex; align-items: center; justify-content: center; font-size: .72rem; color: #fff; white-space: nowrap; overflow: hidden; }
.legend { display: flex; flex-wrap: wrap; gap: 14px; font-size: .78rem; color: var(--muted); margin-bottom: 22px; min-height: 1.2em; }
.legend span::before { content: ""; display: inline-block; width: 9px; height: 9px; border-radius: 2px; margin-right: 5px; background: var(--c); }
.grid { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 22px; }
@media (max-width: 900px) { .grid { grid-template-columns: 1fr; } }
.panel { min-width: 0; background: #fff; border: 1px solid var(--line); border-radius: 12px; padding: 18px 20px; }
#answer { font-size: 1.05rem; line-height: 1.6; white-space: pre-wrap; min-height: 3em; }
#answer.muted, .muted { color: #687a73; }
.cite.fact { background: var(--ink); color: var(--mint); border-color: var(--ink); }
.cite { display: inline-block; border: 1px solid var(--green); color: var(--green); background: #eaf6f1; border-radius: 6px; padding: 0 6px; margin: 0 2px; font-size: .78rem; line-height: 1.5; vertical-align: 1px; max-width: 22ch; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.sources { list-style: none; padding: 0; margin: 18px 0 0; display: grid; grid-template-columns: minmax(0, 1fr); gap: 10px; }
.source { min-width: 0; border: 1px solid var(--line); border-radius: 10px; padding: 10px 12px; background: var(--card); transition: border-color .2s, box-shadow .2s, opacity .2s; }
.source.flash, .source.lit { border-color: var(--green); box-shadow: 0 0 0 3px rgba(23, 98, 79, .15); }
.source.dim { opacity: .45; }
.source header { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; margin-bottom: 4px; font-size: .82rem; }
.source a, .source .id { color: var(--green); font-weight: 600; overflow-wrap: anywhere; }
.badge { border-radius: 999px; padding: 0 8px; font-size: .72rem; font-weight: 600; background: var(--soft); color: var(--muted); }
.badge.graph { background: var(--ink); color: var(--mint); }
.badge.both { background: var(--green); color: #fff; }
.score { margin-left: auto; color: var(--muted); font-variant-numeric: tabular-nums; }
.source p { margin: 0; overflow-wrap: anywhere; font-size: .86rem; color: #33443f; display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical; overflow: hidden; }
.source.open p { -webkit-line-clamp: unset; }
.source .more { border: 0; background: none; color: var(--green); padding: 0; font-size: .8rem; }
#graph { width: 100%; height: auto; display: block; background: var(--card); border-radius: 10px; border: 1px solid var(--soft); }
#graph .edge { stroke: #9fb3ab; stroke-width: 1.6; fill: none; }
#graph .edge.hop2 { stroke-dasharray: 5 4; }
#graph .edge.lit { stroke: var(--green); stroke-width: 2.6; }
#graph .edge.dim, #graph .node.dim { opacity: .18; }
#graph .node { cursor: pointer; outline: none; }
#graph .node:focus-visible circle { stroke: var(--ink); stroke-width: 3; }
#graph .node circle { stroke: #fff; stroke-width: 2; }
#graph .node.linked circle { fill: var(--mint); stroke: var(--ink); }
#graph .node.seed circle { fill: var(--green); }
#graph .node.hop1 circle { fill: #6f8a81; }
#graph .node.hop2 circle { fill: #b7c7c1; }
#graph .node.lit circle { stroke: var(--ink); stroke-width: 3; }
#graph .halo { fill: none; stroke: var(--mint); stroke-width: 2; opacity: .5; }
#graph text { font-size: 12.5px; fill: var(--ink); paint-order: stroke; stroke: var(--card); stroke-width: 3px; pointer-events: none; }
#graph .pred { font-size: 10.5px; fill: var(--muted); }
.graph-legend { display: flex; flex-wrap: wrap; gap: 12px; font-size: .78rem; color: var(--muted); margin: 10px 0 0; }
.graph-legend span::before { content: ""; display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 5px; vertical-align: -1px; }
.k-linked::before { background: var(--mint); } .k-seed::before { background: var(--green); }
.k-hop1::before { background: #6f8a81; } .k-hop2::before { background: #b7c7c1; }
#selection { font-size: .85rem; color: var(--muted); margin: 8px 0 0; min-height: 1.3em; overflow-wrap: anywhere; }
.facts { margin: 14px 0 0; padding-left: 18px; font-size: .84rem; color: #33443f; }
.facts li { margin-bottom: 3px; overflow-wrap: anywhere; }
.facts code { font: 600 .74rem ui-monospace, monospace; color: var(--green); }
.error { color: #9b2c2c; }
.verdict { margin: 8px 0 0; font-size: .82rem; line-height: 1.4; }
.verdict.ok { color: var(--green); }
.verdict.warn { color: #9b2c2c; font-weight: 600; }
"""

BODY = """
<main>
  <div class="brand"><img src="/favicon.svg" alt="Graph RAG logo"><p class="kicker">EVIDENCE-GROUNDED KNOWLEDGE SEARCH</p></div>
  <h1>Ask the graph.</h1>
  <p class="lede">Vector search finds relevant passages, the knowledge graph adds what they're connected to, and the answer cites both.</p>
  <form id="query-form">
    <input id="question" required maxlength="2000" placeholder="Ask about the indexed documents" autocomplete="off">
    <button id="submit" type="submit">Ask</button>
  </form>
  <div class="examples" id="examples"></div>
  <div class="timeline" id="timeline" aria-label="Pipeline stage timings"></div>
  <div class="legend" id="legend"></div>
  <div class="grid">
    <section class="panel">
      <h2>Answer</h2>
      <div id="answer" class="muted">Ask a question to see a grounded, cited answer.</div>
      <p id="verdict" class="verdict" role="status" hidden></p>
      <ul class="sources" id="sources"></ul>
    </section>
    <section class="panel">
      <h2>Evidence graph</h2>
      <svg id="graph" viewBox="0 0 640 500" role="img" aria-label="Entities and relationships used as evidence"></svg>
      <div class="graph-legend">
        <span class="k-linked">named in question</span><span class="k-seed">from evidence</span>
        <span class="k-hop1">1 hop</span><span class="k-hop2">2 hops</span>
      </div>
      <p id="selection">Click a node to trace its facts and sources.</p>
      <ol class="facts" id="facts"></ol>
    </section>
  </div>
</main>
"""

SCRIPT = r"""
(() => {
'use strict';
const $ = id => document.getElementById(id);
const SVG = 'http://www.w3.org/2000/svg';
const form = $('query-form'), input = $('question'), button = $('submit'), answerEl = $('answer');
const verdictEl = $('verdict');
const sourcesEl = $('sources'), factsEl = $('facts'), graphEl = $('graph'), selectionEl = $('selection');
const STAGES = [
  ['understand', 'embed + NER + link', '#17624f', t => Math.max(t.embedding_ms || 0, t.entity_extraction_ms || 0, t.entity_linking_ms || 0)],
  ['vector', 'vector search', '#2f7d68', t => t.vector_search_ms],
  ['graphdocs', 'graph retrieval', '#17221f', t => t.graph_retrieval_ms],
  ['rerank', 'rerank', '#4f6b62', t => t.reranking_ms],
  ['expand', 'graph expansion', '#33473f', t => t.graph_expansion_ms],
  ['first', 'first token', '#7c948c', t => t.first_token_ms],
  ['gen', 'generation', '#a7bab3', t => t.first_token_ms != null ? t.generation_ms - t.first_token_ms : t.generation_ms],
];
let state = { sources: [], graph: { nodes: [], edges: [] }, answer: '' };
let selected = { ids: [], node: null };   // a click selection survives hovering over source cards

function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text != null) node.textContent = text;
  return node;
}
function svg(tag, attrs) {
  const node = document.createElementNS(SVG, tag);
  for (const [k, v] of Object.entries(attrs || {})) node.setAttribute(k, String(v));
  return node;
}
function safeHttpUrl(value) {
  try { const url = new URL(value); return url.protocol === 'https:' || url.protocol === 'http:' ? url.href : null; }
  catch { return null; }
}
const shorten = (text, n) => text.length > n ? text.slice(0, n - 1) + '\u2026' : text;

// ---------------------------------------------------------------- timings
function renderTimings(t) {
  const bar = $('timeline'), legend = $('legend');
  bar.replaceChildren(); legend.replaceChildren();
  if (!t) return;
  for (const [key, label, colour, pick] of STAGES) {
    const ms = Number(pick(t));
    if (!isFinite(ms) || ms <= 0) continue;
    const seg = el('div', null, ms >= 120 ? Math.round(ms) + ' ms' : '');
    seg.style.flexGrow = String(ms); seg.style.background = colour; seg.title = label + ': ' + ms.toFixed(1) + ' ms';
    bar.appendChild(seg);
    const item = el('span', null, label + ' ' + ms.toFixed(0) + ' ms');
    item.style.setProperty('--c', colour);
    legend.appendChild(item);
  }
  if (t.total_ms != null) legend.appendChild(el('strong', null, 'total ' + (t.total_ms / 1000).toFixed(2) + ' s'));
}

// ---------------------------------------------------------------- answer with citations
function renderAnswer() {
  answerEl.className = '';
  answerEl.replaceChildren();
  const ids = new Set(state.sources.map(s => s.document_id));
  const edgeKey = (a, p, b) => a + '\u0000' + p + '\u0000' + b;
  const edges = new Map(state.graph.edges.map(e => [edgeKey(e.source, e.predicate, e.target), e]));
  // [subject -[PREDICATE]-> object] is a cited graph fact; [id] or [id, id] cites passages.
  const pattern = /\[([^\[\]\n]{1,300}) -\[([^\[\]\n]{1,60})\]-> ([^\[\]\n]{1,300})\]|\[([^\[\]\n]{1,400})\]/g;
  let last = 0, match;
  const flush = upto => answerEl.appendChild(document.createTextNode(state.answer.slice(last, upto)));
  while ((match = pattern.exec(state.answer)) !== null) {
    if (match[4] === undefined) {                       // only facts that are really in the evidence graph
      const edge = edges.get(edgeKey(match[1].trim(), match[2].trim(), match[3].trim()));
      if (!edge) continue;
      flush(match.index);
      const chip = el('button', 'cite fact', edge.predicate.toLowerCase().replace(/_/g, ' '));
      chip.type = 'button'; chip.title = 'Graph fact: ' + edge.source + ' \u2014' + edge.predicate + '\u2192 ' + edge.target;
      chip.addEventListener('click', () => { selected = { ids: [edge.source, edge.target], node: null }; highlight(selected.ids, null); graphEl.scrollIntoView({ behavior: 'smooth', block: 'center' }); });
      answerEl.appendChild(chip);
    } else {
      const parts = match[4].split(/\s*[,;]\s*/).map(p => p.trim());
      if (!parts.length || !parts.every(p => ids.has(p))) continue;   // only cite IDs we actually retrieved
      flush(match.index);
      for (const id of parts) {
        const chip = el('button', 'cite', shorten(id.replace(/^https?:\/\/(www\.)?/, ''), 28));
        chip.type = 'button'; chip.title = 'Show source ' + id;
        chip.addEventListener('click', () => focusSource(id));
        answerEl.appendChild(chip);
      }
    }
    last = match.index + match[0].length;
  }
  flush(state.answer.length);
}
// The server checks citations after generation; its verdict replaces the streamed draft.
const VERDICTS = {
  supported: ['ok', 'Citations checked: each claim appears in a cited source.'],
  repaired: ['ok', 'Citations corrected: the model cited the wrong source, so it was replaced with the one that supports the answer.'],
  refusal: ['muted', 'Not enough evidence in the index to answer.'],
  unsupported: ['warn', 'its citations did not support it.'],
  uncited: ['warn', 'it cited no retrieved source.'],
};
function renderVerdict(data) {
  let [kind, text] = VERDICTS[data.verdict] || ['warn', 'its citations could not be checked.'];
  if (kind === 'warn') text = (data.withdrawn ? 'Answer withdrawn: ' : 'Unverified answer: ') + text;
  const removed = (data.invalid || []).length + (data.facts_invalid || []).length;
  if (removed) text += ' Removed ' + removed + ' citation' + (removed > 1 ? 's' : '') + ' to sources that were never retrieved.';
  verdictEl.className = 'verdict ' + kind;
  verdictEl.textContent = text;
  verdictEl.hidden = false;
}
function focusSource(id) {
  const card = [...sourcesEl.children].find(li => li.dataset.id === id);
  if (!card) return;
  card.scrollIntoView({ behavior: 'smooth', block: 'center' });
  card.classList.add('flash'); setTimeout(() => card.classList.remove('flash'), 1400);
  selected = { ids: state.graph.nodes.filter(n => n.sources.includes(id)).map(n => n.id), node: null };
  highlight(selected.ids, null);
}

// ---------------------------------------------------------------- sources
function renderSources() {
  sourcesEl.replaceChildren();
  state.sources.forEach((source, index) => {
    const li = el('li', 'source'); li.dataset.id = source.document_id;
    const head = el('header');
    head.appendChild(el('strong', null, '#' + (index + 1)));
    const url = safeHttpUrl(source.document_id);
    if (url) {
      const a = el('a', null, shorten(source.document_id.replace(/^https?:\/\/(www\.)?/, ''), 60));
      a.href = url; a.target = '_blank'; a.rel = 'noopener noreferrer nofollow'; a.title = source.document_id;
      head.appendChild(a);
    } else head.appendChild(el('span', 'id', source.document_id));
    const via = source.retrieval || 'vector';
    head.appendChild(el('span', 'badge ' + (via === 'graph' ? 'graph' : via === 'vector+graph' ? 'both' : ''), via));
    head.appendChild(el('span', 'score', 'score ' + Number(source.score).toFixed(3) + ' \u00b7 cos ' + Number(source.similarity).toFixed(3)));
    li.appendChild(head);
    li.appendChild(el('p', null, source.text));
    const more = el('button', 'more', 'show more'); more.type = 'button';
    more.addEventListener('click', () => { li.classList.toggle('open'); more.textContent = li.classList.contains('open') ? 'show less' : 'show more'; });
    li.appendChild(more);
    li.addEventListener('mouseenter', () => highlight(state.graph.nodes.filter(n => n.sources.includes(source.document_id)).map(n => n.id), null, true));
    li.addEventListener('mouseleave', () => highlight(selected.ids, selected.node, true));
    sourcesEl.appendChild(li);
  });
}

// ---------------------------------------------------------------- graph
function layout(nodes, edges) {
  const W = 640, H = 500, cx = W / 2, cy = H / 2;
  const rings = { center: [], hop1: [], hop2: [] };
  for (const n of nodes) (n.role === 'hop1' ? rings.hop1 : n.role === 'hop2' ? rings.hop2 : rings.center).push(n);
  const pos = new Map();
  const neighbours = new Map(nodes.map(n => [n.id, []]));
  for (const e of edges) { neighbours.get(e.source)?.push(e.target); neighbours.get(e.target)?.push(e.source); }
  const place = (ring, radius, offset) => {
    const items = ring.map(n => {   // put each node near the mean angle of already-placed neighbours
      const angles = neighbours.get(n.id).filter(id => pos.has(id)).map(id => pos.get(id).a);
      const mean = angles.length ? Math.atan2(angles.reduce((s, a) => s + Math.sin(a), 0), angles.reduce((s, a) => s + Math.cos(a), 0)) : null;
      return { n, mean };
    });
    items.sort((x, y) => (x.mean ?? 99) - (y.mean ?? 99));
    items.forEach(({ n }, i) => {
      const a = offset + (2 * Math.PI * i) / Math.max(items.length, 1);
      const r = items.length === 1 && radius < 60 ? 0 : radius;
      pos.set(n.id, { a, x: cx + r * Math.cos(a), y: cy + r * Math.sin(a) * 0.86 });
    });
  };
  place(rings.center, rings.center.length > 1 ? 70 : 0, -Math.PI / 2);
  place(rings.hop1, 160, -Math.PI / 2 + 0.3);
  place(rings.hop2, 230, -Math.PI / 2 + 0.15);
  relax(nodes, edges, pos, W, H, rings.center.length === 1 ? rings.center[0].id : null);
  return pos;
}
// Deterministic force relaxation seeded by the radial layout: springs along edges, repulsion
// between all nodes, a weak pull to the centre. Small graphs spread out instead of lining up.
function relax(nodes, edges, pos, W, H, pinned) {
  const ids = nodes.map(n => n.id), ideal = Math.min(115, 250 / Math.sqrt(Math.max(ids.length, 1)) + 40);
  for (let iter = 0, heat = 18; iter < 260; iter++, heat *= 0.985) {
    const force = new Map(ids.map(id => [id, { x: 0, y: 0 }]));
    for (let i = 0; i < ids.length; i++) for (let j = i + 1; j < ids.length; j++) {
      const a = pos.get(ids[i]), b = pos.get(ids[j]);
      let dx = a.x - b.x, dy = a.y - b.y, d = Math.hypot(dx, dy);
      if (d < 0.01) { dx = (i - j) * 0.1 || 0.1; dy = 0.1; d = 0.15; }
      const push = (ideal * ideal) / d / d;
      force.get(ids[i]).x += dx * push; force.get(ids[i]).y += dy * push;
      force.get(ids[j]).x -= dx * push; force.get(ids[j]).y -= dy * push;
    }
    for (const e of edges) {
      const a = pos.get(e.source), b = pos.get(e.target);
      if (!a || !b) continue;
      const dx = b.x - a.x, dy = b.y - a.y, d = Math.hypot(dx, dy) || 0.01, pull = (d - ideal) / d * 0.5;
      force.get(e.source).x += dx * pull; force.get(e.source).y += dy * pull;
      force.get(e.target).x -= dx * pull; force.get(e.target).y -= dy * pull;
    }
    for (const id of ids) {
      if (id === pinned) continue;
      const p = pos.get(id), f = force.get(id);
      f.x += (W / 2 - p.x) * 0.05; f.y += (H / 2 - p.y) * 0.05;
      const len = Math.hypot(f.x, f.y) || 1, step = Math.min(len, heat);
      p.x = Math.min(W - 95, Math.max(95, p.x + f.x / len * step));   // room for half a label
      p.y = Math.min(H - 30, Math.max(24, p.y + f.y / len * step));
    }
  }
}
function renderGraph() {
  graphEl.replaceChildren();
  const { nodes, edges } = state.graph;
  if (!nodes.length) {
    const t = svg('text', { x: 320, y: 250, 'text-anchor': 'middle' }); t.textContent = 'No graph facts for this question.';
    graphEl.appendChild(t); return;
  }
  const defs = svg('defs');
  const marker = svg('marker', { id: 'arrow', viewBox: '0 0 10 10', refX: 9, refY: 5, markerWidth: 7, markerHeight: 7, orient: 'auto-start-reverse' });
  marker.appendChild(svg('path', { d: 'M0 0L10 5L0 10z', fill: '#9fb3ab' }));
  defs.appendChild(marker); graphEl.appendChild(defs);
  const pos = layout(nodes, edges);
  const edgeLayer = svg('g'), labelLayer = svg('g'), nodeLayer = svg('g');
  const radius = n => n.role === 'linked' ? 13 : n.role === 'seed' ? 10 : n.role === 'hop1' ? 8 : 6;
  edges.forEach((e, i) => {
    const s = pos.get(e.source), t = pos.get(e.target);
    if (!s || !t) return;
    const dx = t.x - s.x, dy = t.y - s.y, len = Math.hypot(dx, dy) || 1;
    const rt = radius(nodes.find(n => n.id === e.target)) + 3;
    const line = svg('line', { x1: s.x, y1: s.y, x2: t.x - dx / len * rt, y2: t.y - dy / len * rt, class: 'edge' + (e.hop > 1 ? ' hop2' : ''), 'marker-end': 'url(#arrow)' });
    line.dataset.i = i;
    const tip = svg('title'); tip.textContent = e.source + ' \u2014' + e.predicate + '\u2192 ' + e.target; line.appendChild(tip);
    edgeLayer.appendChild(line);
    if (edges.length <= 14) {
      const label = svg('text', { x: (s.x + t.x) / 2, y: (s.y + t.y) / 2 - 3, 'text-anchor': 'middle', class: 'pred' });
      label.textContent = shorten(e.predicate, 22); label.dataset.i = i; labelLayer.appendChild(label);
    }
  });
  for (const n of nodes) {
    const p = pos.get(n.id), g = svg('g', { class: 'node ' + n.role, tabindex: 0 });
    g.dataset.id = n.id;
    if (n.role === 'linked') g.appendChild(svg('circle', { cx: p.x, cy: p.y, r: 20, class: 'halo' }));
    g.appendChild(svg('circle', { cx: p.x, cy: p.y, r: radius(n) }));
    const label = svg('text', { x: p.x, y: p.y + radius(n) + 13, 'text-anchor': 'middle' });
    const name = n.label || n.id;
    label.textContent = shorten(name, n.role === 'linked' ? 34 : 26);
    const tip = svg('title'); tip.textContent = name === n.id ? n.id : name + '\n' + n.id; g.appendChild(tip);
    g.appendChild(label);
    const select = () => { selected = { ids: [n.id], node: n }; highlight([n.id], n); };
    g.addEventListener('click', select);
    g.addEventListener('keydown', ev => { if (ev.key === 'Enter' || ev.key === ' ') { ev.preventDefault(); select(); } });
    nodeLayer.appendChild(g);
  }
  graphEl.append(edgeLayer, labelLayer, nodeLayer);
}
function highlight(ids, node, hoverOnly) {
  const set = new Set(ids), { edges } = state.graph;
  // Trace up to two steps out from the selection: enough to follow author -> paper -> institution.
  const litEdges = new Set(), litNodes = new Set(set);
  for (let step = 0; step < (set.size > 1 ? 1 : 2); step++) {
    const frontier = new Set(litNodes);
    edges.forEach((e, i) => { if (frontier.has(e.source) || frontier.has(e.target)) { litEdges.add(i); litNodes.add(e.source); litNodes.add(e.target); } });
  }
  const active = set.size > 0;
  graphEl.querySelectorAll('.node').forEach(g => { g.classList.toggle('lit', set.has(g.dataset.id)); g.classList.toggle('dim', active && !litNodes.has(g.dataset.id)); });
  graphEl.querySelectorAll('.edge').forEach(l => { const on = litEdges.has(Number(l.dataset.i)); l.classList.toggle('lit', on); l.classList.toggle('dim', active && !on); });
  if (hoverOnly) return;
  const docs = new Set(state.graph.nodes.filter(n => set.has(n.id)).flatMap(n => n.sources));
  [...sourcesEl.children].forEach(li => { li.classList.toggle('lit', docs.has(li.dataset.id)); li.classList.toggle('dim', active && docs.size > 0 && !docs.has(li.dataset.id)); });
  [...factsEl.children].forEach((li, i) => li.classList.toggle('muted', active && !litEdges.has(i)));
  if (node) selectionEl.textContent = (node.label || node.id) + ' \u2014 ' + litEdges.size + ' fact(s), ' + docs.size + ' source(s)' + (node.role === 'linked' ? ' \u00b7 named in your question' : '');
}
function renderFacts() {
  factsEl.replaceChildren();
  for (const e of state.graph.edges) {
    const li = el('li');
    li.append(document.createTextNode(e.source + ' '), el('code', null, e.predicate), document.createTextNode(' \u2192 ' + e.target + (e.hop > 1 ? '  (2 hops)' : '')));
    factsEl.appendChild(li);
  }
}

// ---------------------------------------------------------------- query
async function ask(question) {
  button.disabled = true;
  state = { sources: [], graph: { nodes: [], edges: [] }, answer: '' };
  selected = { ids: [], node: null };
  answerEl.className = 'muted'; answerEl.textContent = 'Retrieving evidence\u2026';
  verdictEl.hidden = true; verdictEl.textContent = '';
  renderSources(); renderFacts(); renderGraph(); renderTimings(null);
  selectionEl.textContent = 'Click a node to trace its facts and sources.';
  try {
    const response = await fetch('/query/stream', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ question, limit: 8 }) });
    if (!response.ok) throw new Error(response.status === 422 ? 'Please enter a question (up to 2000 characters).' : 'Query failed (' + response.status + ')');
    const reader = response.body.getReader(), decoder = new TextDecoder();
    let buffer = '';
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const events = buffer.split('\n\n'); buffer = events.pop();
      for (const raw of events) {
        if (!raw.startsWith('data: ')) continue;
        const data = JSON.parse(raw.slice(6));
        if (data.type === 'evidence') {
          state.sources = data.sources || []; state.graph = data.graph || { nodes: [], edges: [] };
          renderSources(); renderFacts(); renderGraph(); renderTimings(data.timings);
          answerEl.textContent = '';
        } else if (data.type === 'token') { state.answer += data.text; renderAnswer(); }
        else if (data.type === 'citations') {
          state.answer = data.answer; renderAnswer(); renderVerdict(data);
          if (data.withdrawn) answerEl.className = 'muted';
        }
        else if (data.type === 'done') renderTimings(data.timings);
        else if (data.type === 'error') throw new Error(data.detail || 'Query failed');
      }
    }
    if (!state.answer) { answerEl.className = 'muted'; answerEl.textContent = 'No answer was generated.'; }
  } catch (error) {
    answerEl.className = 'error'; answerEl.textContent = error.message;
  } finally { button.disabled = false; }
}
form.addEventListener('submit', event => {
  event.preventDefault();
  const q = input.value.trim(); if (!q) return;
  history.replaceState(null, '', '?q=' + encodeURIComponent(q));
  ask(q);
});
for (const example of JSON.parse(document.getElementById('examples-data').textContent)) {
  const chip = el('button', null, example); chip.type = 'button';
  chip.addEventListener('click', () => { input.value = example; form.requestSubmit(); });
  $('examples').appendChild(chip);
}
const initial = new URLSearchParams(location.search).get('q');
if (initial) { input.value = initial.slice(0, 2000); ask(input.value); }
})();
"""


def _sha256(text: str) -> str:
    return "sha256-" + base64.b64encode(hashlib.sha256(text.encode("utf-8")).digest()).decode()


def _examples_json() -> str:
    import json

    # `<` is escaped so an example can never close the <script> data block.
    return json.dumps(list(EXAMPLES), ensure_ascii=False).replace("<", "\\u003c")


PAGE = (
    "<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
    "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n<title>Graph RAG</title>\n"
    "<meta name=\"description\" content=\"Ask questions answered from indexed documents and the relationships between them, with sources and graph facts.\">\n"
    f"<meta name=\"theme-color\" content=\"{THEME_COLOR}\">\n"
    "<link rel=\"icon\" type=\"image/svg+xml\" href=\"/favicon.svg\">\n"
    f"<style>{STYLE}</style>\n</head>\n<body>{BODY}"
    f"<script type=\"application/json\" id=\"examples-data\">{_examples_json()}</script>\n"
    f"<script>{SCRIPT}</script>\n</body>\n</html>\n"
)

CONTENT_SECURITY_POLICY = "; ".join((
    "default-src 'none'",
    f"script-src '{_sha256(SCRIPT)}'",
    # No style attributes in the markup; the script only touches styles through the CSSOM
    # (element.style.*), which CSP doesn't block, so the stylesheet hash is all that's needed.
    f"style-src '{_sha256(STYLE)}'",
    "img-src 'self'",
    "connect-src 'self'",
    "base-uri 'none'",
    "form-action 'none'",
    "frame-ancestors 'none'",
))

SECURITY_HEADERS = {
    "Content-Security-Policy": CONTENT_SECURITY_POLICY,
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "X-Frame-Options": "DENY",
}
