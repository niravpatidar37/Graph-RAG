# /// script
# requires-python = ">=3.11"
# dependencies = ["fonttools==4.66.1", "uharfbuzz==0.56.2", "pillow==12.3.0"]
# ///
"""Build every Graph RAG brand asset from source, reproducibly.

    uv run docs/brand/tools/build_brand.py

PNG exports need `resvg` on PATH (`cargo install resvg --locked --version 0.48.1`).

All text is shaped with HarfBuzz and converted to outlines, so the SVGs render
identically on GitHub, in browsers, and in resvg without any font installed.
Fonts are fetched from pinned upstream releases, verified by SHA-256, cached in
tools/.cache (git-ignored), and never vendored:

  * Source Serif 4 (Adobe, SIL OFL 1.1) - the "Graph" wordmark, echoing the
    serif headline of the built-in query page.
  * Inter (Rasmus Andersson, SIL OFL 1.1) - "RAG", taglines, diagram labels.

The OFL permits using glyph outlines in artwork such as logos.

The palette is taken from the built-in page in src/graph_rag/api.py
(#17221f ink, #eef2f0 paper, #17624f green) plus a mint accent for dark surfaces.
"""

from __future__ import annotations

import hashlib
import io
import pathlib
import re
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from dataclasses import dataclass

import uharfbuzz as hb
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.ttLib import TTFont
from PIL import Image

BRAND = pathlib.Path(__file__).resolve().parents[1]          # docs/brand
REPO = BRAND.parents[1]
CACHE = BRAND / "tools" / ".cache"
PACKAGE_BRAND_MODULE = REPO / "src" / "graph_rag" / "brand.py"

FONT_ARCHIVES = {
    "serif": (
        "https://github.com/adobe-fonts/source-serif/releases/download/4.005R/source-serif-4.005_Desktop.zip",
        "549fdb8f9a682bd06944298621404969f6de77c2e422ff3b8244a1dcd6a0c425",
    ),
    "sans": (
        "https://github.com/rsms/inter/releases/download/v4.1/Inter-4.1.zip",
        "9883fdd4a49d4fb66bd8177ba6625ef9a64aa45899767dde3d36aa425756b11e",
    ),
}
FONT_FILES = {
    "serif-bold": ("serif", "source-serif-4.005_Desktop/TTF/SourceSerif4Display-Bold.ttf"),
    "sans-medium": ("sans", "extras/ttf/Inter-Medium.ttf"),
    "sans-semibold": ("sans", "extras/ttf/Inter-SemiBold.ttf"),
    "sans-bold": ("sans", "extras/ttf/Inter-Bold.ttf"),
}

# ----------------------------------------------------------------- palette
INK = "#17221F"            # built-in page text colour; brand tile
INK_DEEP = "#111A17"       # banner / social background
PAPER = "#EEF2F0"          # built-in page background; nodes and text on dark
GREEN = "#17624F"          # built-in page button; accent on light surfaces
MINT = "#5CD6A6"           # accent on dark surfaces (the "retrieved" node)
MUTED_ON_DARK = "#9DB1AA"
EDGE_ON_DARK = "#7C948C"
FAINT_ON_DARK = "#2C3F39"
CARD_ON_DARK = "#1B2925"
LINE_ON_DARK = "#33473F"
GITHUB_LIGHT_BG = "#FFFFFF"
GITHUB_DARK_BG = "#0D1117"


def _lum(hex_color: str) -> float:
    rgb = [int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def contrast(a: str, b: str) -> float:
    hi, lo = sorted((_lum(a), _lum(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


# ----------------------------------------------------------------- fonts
@dataclass
class Font:
    tt: TTFont
    hb_font: hb.Font
    upem: int

    @property
    def cap_height(self) -> float:
        return self.tt["OS/2"].sCapHeight / self.upem

    @property
    def x_height(self) -> float:
        return self.tt["OS/2"].sxHeight / self.upem


def _archive(key: str) -> zipfile.ZipFile:
    url, digest = FONT_ARCHIVES[key]
    path = CACHE / url.rsplit("/", 1)[1]
    if not path.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=120) as response:  # noqa: S310 - pinned https URL
            path.write_bytes(response.read())
    data = path.read_bytes()
    actual = hashlib.sha256(data).hexdigest()
    if actual != digest:
        path.unlink()
        sys.exit(f"font archive checksum mismatch for {path.name}: {actual}")
    return zipfile.ZipFile(io.BytesIO(data))


def load_fonts() -> dict[str, Font]:
    archives = {key: _archive(key) for key in FONT_ARCHIVES}
    fonts: dict[str, Font] = {}
    for name, (archive, member) in FONT_FILES.items():
        raw = archives[archive].read(member)
        tt = TTFont(io.BytesIO(raw))
        fonts[name] = Font(tt, hb.Font(hb.Face(hb.Blob(raw))), tt["head"].unitsPerEm)
    return fonts


def n(value: float) -> str:
    text = f"{value:.2f}".rstrip("0").rstrip(".")
    return "0" if text in ("-0", "") else text


def n1(value: float) -> str:
    """One decimal is sub-pixel at every export size and keeps outlined text compact."""
    text = f"{value:.1f}".rstrip("0").rstrip(".")
    return "0" if text in ("-0", "") else text


def text_path(font: Font, text: str, size: float, x: float, baseline: float, tracking: float = 0.0) -> tuple[str, float]:
    """Shape `text` with HarfBuzz (kerning + ligatures) and outline it. Returns (d, advance)."""
    buf = hb.Buffer()
    buf.add_str(text)
    buf.guess_segment_properties()
    hb.shape(font.hb_font, buf, {"kern": True, "liga": True, "calt": False})  # calt off: keep "->" literal
    order = font.tt.getGlyphOrder()
    glyphs = font.tt.getGlyphSet()
    scale = size / font.upem
    pen = SVGPathPen(glyphs, ntos=n1)
    cursor = x
    for info, pos in zip(buf.glyph_infos, buf.glyph_positions, strict=True):
        gx = cursor + pos.x_offset * scale
        gy = baseline - pos.y_offset * scale
        glyphs[order[info.codepoint]].draw(TransformPen(pen, (scale, 0, 0, -scale, gx, gy)))
        cursor += pos.x_advance * scale + tracking
    width = cursor - x - (tracking if text else 0)
    return pen.getCommands(), width


def measure(font: Font, text: str, size: float, tracking: float = 0.0) -> float:
    return text_path(font, text, size, 0, 0, tracking)[1]


def label(font: Font, text: str, size: float, x: float, baseline: float, fill: str, *, tracking: float = 0.0,
          anchor: str = "start", max_width: float | None = None) -> str:
    """Outlined text element; shrinks to `max_width` rather than overflowing a box."""
    width = measure(font, text, size, tracking)
    if max_width is not None and width > max_width:
        size *= max_width / width
        width = max_width
    if anchor == "middle":
        x -= width / 2
    elif anchor == "end":
        x -= width
    d, _ = text_path(font, text, size, x, baseline, tracking)
    return f'<path fill="{fill}" d="{d}"/>'


def svg(width: float, height: float, body: str, title: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{n(width)}" height="{n(height)}" '
        f'viewBox="0 0 {n(width)} {n(height)}" role="img" aria-label="{title}">'
        f"<title>{title}</title>{body}</svg>\n"
    )


# ----------------------------------------------------------------- the mark
# A retrieved node (mint, with a similarity halo) and its graph neighbourhood:
# vector search finds the hit, the graph supplies the connections around it.
MARK_NODES = {"hit": (24.0, 40.0), "a": (43.0, 20.0), "b": (46.0, 44.0), "c": (17.0, 19.0)}
MARK_EDGES = [("hit", "a"), ("hit", "b"), ("a", "b"), ("hit", "c")]


def mark(*, x: float = 0, y: float = 0, size: float = 64, tile: str | None = INK, rx: float = 14,
         node: str = PAPER, hit: str = MINT, edge: str = EDGE_ON_DARK) -> str:
    s = size / 64
    p = MARK_NODES
    parts = [f'<g transform="translate({n(x)} {n(y)}) scale({n(s)})">']
    if tile:
        parts.append(f'<rect width="64" height="64" rx="{n(rx)}" fill="{tile}"/>')
    parts.append(f'<g stroke="{edge}" stroke-width="4" stroke-linecap="round" fill="none">')
    for u, v in MARK_EDGES:
        parts.append(f'<path d="M{n(p[u][0])} {n(p[u][1])}L{n(p[v][0])} {n(p[v][1])}"/>')
    parts.append("</g>")
    hx, hy = p["hit"]
    parts.append(f'<circle cx="{n(hx)}" cy="{n(hy)}" r="14" fill="none" stroke="{hit}" stroke-width="2.5" stroke-opacity="0.45"/>')
    parts.append(f'<circle cx="{n(hx)}" cy="{n(hy)}" r="9" fill="{hit}"/>')
    for key, r in (("a", 6.5), ("b", 6), ("c", 4.75)):
        parts.append(f'<circle cx="{n(p[key][0])}" cy="{n(p[key][1])}" r="{r}" fill="{node}"/>')
    parts.append("</g>")
    return "".join(parts)


def favicon16() -> str:
    """Pixel-tuned 16 px variant: no halo, three nodes, thicker relative strokes."""
    return (
        f'<rect width="16" height="16" rx="3.5" fill="{INK}"/>'
        f'<g stroke="{EDGE_ON_DARK}" stroke-width="1.3" stroke-linecap="round">'
        '<path d="M6 10L11.5 4.5"/><path d="M6 10L12 11.5"/><path d="M11.5 4.5L12 11.5"/></g>'
        f'<circle cx="6" cy="10" r="2.75" fill="{MINT}"/>'
        f'<circle cx="11.5" cy="4.5" r="2" fill="{PAPER}"/>'
        f'<circle cx="12" cy="11.5" r="1.75" fill="{PAPER}"/>'
    )


def lockup(fonts: dict[str, Font], *, word: str, accent: str) -> str:
    size, gap = 40.0, 16.0
    serif, sans = fonts["serif-bold"], fonts["sans-bold"]
    baseline = 32 + serif.cap_height * size / 2
    x0 = 64 + gap
    d1, w1 = text_path(serif, "Graph", size, x0, baseline, tracking=-0.4)
    rag_size = size * 0.86
    d2, w2 = text_path(sans, "RAG", rag_size, x0 + w1 + 11, baseline, tracking=1.6)
    width = x0 + w1 + 11 + w2 + 2
    body = mark() + f'<path fill="{word}" d="{d1}"/><path fill="{accent}" d="{d2}"/>'
    return svg(width, 64, body, "Graph RAG")


# ----------------------------------------------------------------- hero art
NET_NODES = [
    (0, 150), (72, 62), (86, 236), (168, 140), (214, 30), (240, 262),
    (292, 178), (338, 84), (378, 238), (420, 140), (132, 312), (452, 34),
]
NET_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 3), (1, 4), (3, 6), (2, 10), (5, 10), (5, 6),
    (6, 7), (4, 7), (6, 8), (7, 9), (8, 9), (7, 11), (9, 11), (3, 5),
]
NET_HIT = 3
NET_PATH = {(1, 3), (3, 6), (6, 7)}   # one hop either side, then a second hop


def network(x: float, y: float, scale: float = 1.0) -> str:
    def pt(i: int) -> tuple[float, float]:
        px, py = NET_NODES[i]
        return x + px * scale, y + py * scale

    parts = ['<g stroke-linecap="round" fill="none">']
    for u, v in NET_EDGES:
        lit = (u, v) in NET_PATH or (v, u) in NET_PATH
        (x1, y1), (x2, y2) = pt(u), pt(v)
        stroke, width = (MINT, 3.2) if lit else (FAINT_ON_DARK, 2.2)
        parts.append(f'<path stroke="{stroke}" stroke-width="{n(width * scale)}" d="M{n(x1)} {n(y1)}L{n(x2)} {n(y2)}"/>')
    parts.append("</g>")
    lit_nodes = {i for edge in NET_PATH for i in edge}
    for i in range(len(NET_NODES)):
        cx, cy = pt(i)
        if i == NET_HIT:
            parts.append(f'<circle cx="{n(cx)}" cy="{n(cy)}" r="{n(30 * scale)}" fill="none" stroke="{MINT}" stroke-width="{n(2 * scale)}" stroke-opacity="0.35"/>')
            parts.append(f'<circle cx="{n(cx)}" cy="{n(cy)}" r="{n(14 * scale)}" fill="{MINT}"/>')
        elif i in lit_nodes:
            parts.append(f'<circle cx="{n(cx)}" cy="{n(cy)}" r="{n(9 * scale)}" fill="{PAPER}"/>')
        else:
            parts.append(f'<circle cx="{n(cx)}" cy="{n(cy)}" r="{n(6.5 * scale)}" fill="{LINE_ON_DARK}"/>')
    return "".join(parts)


def doc_card(x: float, y: float, w: float = 92, h: float = 66) -> str:
    rows = "".join(
        f'<rect x="{n(x + 14)}" y="{n(y + 16 + i * 13)}" width="{n((w - 28) * f)}" height="5" rx="2.5" fill="{c}"/>'
        for i, (f, c) in enumerate(((1, MUTED_ON_DARK), (0.78, LINE_ON_DARK), (0.9, LINE_ON_DARK)))
    )
    return (f'<rect x="{n(x)}" y="{n(y)}" width="{n(w)}" height="{n(h)}" rx="9" fill="{CARD_ON_DARK}" '
            f'stroke="{LINE_ON_DARK}" stroke-width="1.5"/>' + rows)


def chip(fonts: dict[str, Font], text: str, x: float, baseline: float, size: float = 17) -> tuple[str, float]:
    font = fonts["sans-medium"]
    pad_x, height = 14.0, size * 1.95
    width = measure(font, text, size) + 2 * pad_x
    top = baseline - size * 0.36 - height / 2
    body = (f'<rect x="{n(x)}" y="{n(top)}" width="{n(width)}" height="{n(height)}" rx="{n(height / 2)}" '
            f'fill="{CARD_ON_DARK}" stroke="{LINE_ON_DARK}" stroke-width="1.5"/>'
            + label(font, text, size, x + pad_x, baseline, MUTED_ON_DARK))
    return body, width


def hero(fonts: dict[str, Font], *, width: int, height: int, rounded: bool, social: bool) -> str:
    serif, sans_b, sans_m = fonts["serif-bold"], fonts["sans-bold"], fonts["sans-medium"]
    left = 72 if not social else 88
    parts = []
    if rounded:
        parts.append(f'<rect width="{width}" height="{height}" rx="24" fill="{INK_DEEP}"/>')
    else:
        parts.append(f'<rect width="{width}" height="{height}" fill="{INK_DEEP}"/>')

    if social:
        mark_size, word_size, top = 104, 96, 112
        tag_size, tag_top, chip_base = 40, 330, 498
    else:
        mark_size, word_size, top = 84, 76, 70
        tag_size, tag_top, chip_base = 30, 230, 330

    # graph illustration on the right, drawn first so text always sits on top
    art_scale = 1.0 if social else 0.82
    art_x = width - 72 - 452 * art_scale
    art_y = (height - 312 * art_scale) / 2 + (0 if social else 6)
    parts.append(network(art_x, art_y, art_scale))
    hx, hy = NET_NODES[NET_HIT]
    card_x, card_y = art_x + 100 * art_scale, art_y - 44 * art_scale
    hit_x, hit_y = art_x + hx * art_scale, art_y + hy * art_scale
    parts.append(f'<path d="M{n(card_x + 52 * art_scale)} {n(card_y + 66 * art_scale)}L{n(hit_x - 3 * art_scale)} {n(hit_y - 32 * art_scale)}" '
                 f'stroke="{MINT}" stroke-width="2" stroke-dasharray="5 6" stroke-linecap="round" fill="none"/>')
    parts.append(f'<g transform="translate({n(card_x)} {n(card_y)}) scale({n(art_scale)}) translate({n(-card_x)} {n(-card_y)})">'
                 + doc_card(card_x, card_y) + "</g>")

    # mark + wordmark
    parts.append(mark(x=left, y=top, size=mark_size, rx=14))
    baseline = top + mark_size / 2 + serif.cap_height * word_size / 2
    wx = left + mark_size + 28
    d1, w1 = text_path(serif, "Graph", word_size, wx, baseline, tracking=-0.8)
    d2, _ = text_path(sans_b, "RAG", word_size * 0.86, wx + w1 + 20, baseline, tracking=3)
    parts.append(f'<path fill="{PAPER}" d="{d1}"/><path fill="{MINT}" d="{d2}"/>')

    lines = ("Answers grounded in your documents", "and the connections between them.")
    for i, line in enumerate(lines):
        colour = PAPER if i == 0 else MUTED_ON_DARK
        parts.append(label(sans_m, line, tag_size, left, tag_top + i * tag_size * 1.32, colour))

    cx = left
    for text in ("Qdrant vectors", "Neo4j graph", "FastAPI + SSE", "Ollama or Hugging Face"):
        body, w = chip(fonts, text, cx, chip_base, 17 if not social else 20)
        parts.append(body)
        cx += w + 12

    if social:
        parts.append(label(sans_m, "github.com/niravpatidar37/Graph-RAG", 24, left, height - 64, MINT))
    return svg(width, height, "".join(parts), "Graph RAG: answers grounded in documents and the connections between them")


# ----------------------------------------------------------------- architecture
@dataclass
class Box:
    x: float
    y: float
    w: float
    h: float
    title: str
    lines: tuple[str, ...]
    accent: str = MINT

    @property
    def cx(self) -> float:
        return self.x + self.w / 2

    @property
    def bottom(self) -> float:
        return self.y + self.h


def box(fonts: dict[str, Font], b: Box) -> str:
    parts = [
        f'<rect x="{n(b.x)}" y="{n(b.y)}" width="{n(b.w)}" height="{n(b.h)}" rx="10" fill="{CARD_ON_DARK}" stroke="{LINE_ON_DARK}" stroke-width="1.5"/>',
        f'<rect x="{n(b.x)}" y="{n(b.y + 14)}" width="3" height="{n(b.h - 28)}" rx="1.5" fill="{b.accent}"/>',
        label(fonts["sans-semibold"], b.title, 16, b.x + 18, b.y + 30, PAPER, max_width=b.w - 30),
    ]
    for i, line in enumerate(b.lines):
        parts.append(label(fonts["sans-medium"], line, 13, b.x + 18, b.y + 54 + i * 20, MUTED_ON_DARK, max_width=b.w - 30))
    return "".join(parts)


def arrow(x1: float, y1: float, x2: float, y2: float, *, colour: str = EDGE_ON_DARK, dashed: bool = False) -> str:
    import math

    angle = math.atan2(y2 - y1, x2 - x1)
    head, spread = 9.0, 0.45
    ex, ey = x2 - math.cos(angle) * 2, y2 - math.sin(angle) * 2
    lx, ly = ex - head * math.cos(angle - spread), ey - head * math.sin(angle - spread)
    rx_, ry_ = ex - head * math.cos(angle + spread), ey - head * math.sin(angle + spread)
    sx, sy = ex - math.cos(angle) * head * 0.8, ey - math.sin(angle) * head * 0.8
    dash = ' stroke-dasharray="5 5"' if dashed else ""
    return (f'<path d="M{n(x1)} {n(y1)}L{n(sx)} {n(sy)}" stroke="{colour}" stroke-width="1.8" fill="none"{dash}/>'
            f'<path d="M{n(ex)} {n(ey)}L{n(lx)} {n(ly)}L{n(rx_)} {n(ry_)}Z" fill="{colour}"/>')


def lane(fonts: dict[str, Font], x: float, y: float, w: float, h: float, title: str, sub: str) -> str:
    return (f'<rect x="{n(x)}" y="{n(y)}" width="{n(w)}" height="{n(h)}" rx="16" fill="none" stroke="{LINE_ON_DARK}" '
            f'stroke-width="1.5" stroke-dasharray="7 6"/>'
            + label(fonts["sans-bold"], title, 13, x + 24, y + 30, MINT, tracking=1.6)
            + label(fonts["sans-medium"], sub, 13, x + 24 + measure(fonts["sans-bold"], title, 13, 1.6) + 14, y + 30, MUTED_ON_DARK))


def architecture(fonts: dict[str, Font]) -> str:
    W, H = 1280, 900
    parts = [f'<rect width="{W}" height="{H}" rx="24" fill="{INK_DEEP}"/>']
    parts.append(mark(x=56, y=40, size=44, rx=10))
    parts.append(label(fonts["serif-bold"], "How Graph RAG answers a question", 30, 116, 72, PAPER))
    parts.append(label(fonts["sans-medium"], "Vectors find relevant text; the graph adds the facts connected to it. Both feed one grounded, cited answer.",
                       15, 116, 98, MUTED_ON_DARK))

    # ingestion plane
    parts.append(lane(fonts, 40, 128, 1200, 196, "INGESTION PLANE", "graph-rag-ingest <dir>  ·  graph-rag-dataset --limit N"))
    iw, igap, iy, ih = 260, (1152 - 4 * 260) / 3, 178, 124
    ing = [
        Box(64 + 0 * (iw + igap), iy, iw, ih, "Sources", ("data/*.md, *.txt", "AI Safety dataset (~9.4k records)", "structured authors / institutions")),
        Box(64 + 1 * (iw + igap), iy, iw, ih, "Chunk + extract", ("entities: dslim/bert-base-NER", "relations: chat model, JSON *", "dataset: fields mapped directly")),
        Box(64 + 2 * (iw + igap), iy, iw, ih, "Embed", ("BAAI/bge-base-en-v1.5", "batched per file", "local GPU or HF Inference")),
        Box(64 + 3 * (iw + igap), iy, iw, ih, "Upsert", ("batched UNWIND MERGE, uuid5 IDs", "re-runs are idempotent", "checkpoint per batch + resume")),
    ]
    for a, b in zip(ing, ing[1:]):
        parts.append(arrow(a.x + a.w + 4, a.y + a.h / 2, b.x - 4, b.y + b.h / 2))

    # stores
    sy, sh, sw = 372, 112, 330
    neo = Box(W / 2 - 24 - sw, sy, sw, sh, "Neo4j  ·  graph store", ("(:Chunk)-[:MENTIONS]->(:Entity)", "(:Entity)-[:RELATED {predicate}]->(:Entity)", "unique constraints + full-text on Entity.name"), PAPER)
    qd = Box(W / 2 + 24, sy, sw, sh, "Qdrant  ·  vector store", ("cosine similarity over chunk vectors", "payload: document, text, entities", "dimension fixed per collection"), PAPER)
    up = ing[3]
    parts.append(arrow(up.cx - 40, up.bottom + 4, neo.x + neo.w - 40, neo.y - 4))
    parts.append(arrow(up.cx, up.bottom + 4, qd.x + qd.w - 60, qd.y - 4))

    # query plane
    parts.append(lane(fonts, 40, 532, 1200, 330, "QUERY PLANE", "graph-rag-api  ·  stateless FastAPI workers"))
    qw, qgap, qy, qh = 200, (1152 - 5 * 200) / 4, 582, 124
    q = [
        Box(64 + 0 * (qw + qgap), qy, qw, qh, "Client", ("GET /  evidence-graph UI", "POST /query", "POST /query/stream (SSE)")),
        Box(64 + 1 * (qw + qgap), qy, qw, qh, "1  Understand", ("embed + NER the question", "link entities (full-text)", "all three in parallel")),
        Box(64 + 2 * (qw + qgap), qy, qw, qh, "2  Retrieve", ("Qdrant top 3k + graph chunks", "bounded rerank, graph bonus", "Neo4j: 2-hop facts, hub-capped")),
        Box(64 + 3 * (qw + qgap), qy, qw, qh, "3  Answer", ("facts first, 16k-char cap", "chat model answers only", "from that context")),
        Box(64 + 4 * (qw + qgap), qy, qw, qh, "4  Respond", ("answer + cited sources", "evidence graph + timings", "refuse when unsupported")),
    ]
    for a, b in zip(q, q[1:]):
        parts.append(arrow(a.x + a.w + 4, a.y + a.h / 2, b.x - 4, b.y + b.h / 2))
    ret = q[2]
    parts.append(arrow(ret.cx - 30, ret.y - 4, neo.cx + 40, neo.bottom + 4, colour=MINT, dashed=True))
    parts.append(arrow(ret.cx + 30, ret.y - 4, qd.cx - 40, qd.bottom + 4, colour=MINT, dashed=True))

    # model + observability strip
    my, mh = 742, 92
    models = Box(64 + qw + qgap, my, 3 * qw + 2 * qgap, mh, "Models  ·  swappable per piece, no code changes",
                 ("chat: Ollama or any OpenAI-compatible /v1 server (LLM_BASE_URL), else HF Inference router",
                  "embeddings + NER: local sentence-transformers / transformers (LOCAL_MODELS=1), else HF Inference"), MUTED_ON_DARK)
    obs = Box(64 + 4 * (qw + qgap), my, qw, mh, "Observability", ("Langfuse @observe spans", "GET /metrics stage timings"), MUTED_ON_DARK)
    for b in (*ing, neo, qd, *q, models, obs):
        parts.append(box(fonts, b))
    parts.append(arrow(q[3].cx, q[3].bottom + 4, q[3].cx, my - 4, dashed=True))
    parts.append(arrow(q[1].cx, q[1].bottom + 4, q[1].cx, my - 4, dashed=True))

    parts.append(label(fonts["sans-medium"],
                       "* relation extraction runs on every file ingest; the dataset importer calls the chat model only with --llm-relations.",
                       12.5, 56, 884, MUTED_ON_DARK))
    return svg(W, H, "".join(parts), "Graph RAG architecture: ingestion plane, Neo4j and Qdrant stores, query plane")


# ----------------------------------------------------------------- safety gate
FORBIDDEN = [
    (re.compile(r"<\s*script", re.I), "script element"),
    (re.compile(r"\son[a-z]+\s*=", re.I), "event handler attribute"),
    (re.compile(r"<\s*foreignObject", re.I), "foreignObject"),
    (re.compile(r"(?:xlink:)?href\s*=", re.I), "href reference"),
    (re.compile(r"url\s*\(", re.I), "url() reference"),
    (re.compile(r"javascript:", re.I), "javascript: URL"),
    (re.compile(r"<\s*(?:image|use|style|iframe|embed|object)\b", re.I), "disallowed element"),
]


def check_safe(name: str, text: str) -> None:
    for pattern, what in FORBIDDEN:
        if pattern.search(text):
            sys.exit(f"{name}: unsafe SVG content ({what})")


# ----------------------------------------------------------------- outputs
def write_package_module(favicon_svg: str) -> None:
    """Embed the favicon in the package so the API serves it without data files."""
    body = (
        '"""Brand assets served by the API. Generated by docs/brand/tools/build_brand.py; do not edit."""\n\n'
        f"FAVICON_SVG = {favicon_svg.strip()!r}\n"
        f'THEME_COLOR = "{INK}"\n'
    )
    PACKAGE_BRAND_MODULE.write_text(body, encoding="utf-8", newline="\n")
    print(f"wrote {PACKAGE_BRAND_MODULE.relative_to(REPO)}")


def render(resvg: str, src: str, dst: str, width: int, opaque: bool) -> None:
    out = BRAND / dst
    before = out.stat().st_mtime_ns if out.exists() else None
    # list-form argv, no shell; every argument is a constant or the resolved resvg path
    subprocess.run([resvg, "-w", str(width), str(BRAND / src), str(out)], check=True, shell=False)
    if before is not None and out.stat().st_mtime_ns == before:
        sys.exit(f"{dst} was not rewritten")
    with Image.open(out) as image:
        image.load()
        if opaque:
            flat = image.convert("RGB")
            flat.save(out, optimize=True)
            image = flat
        else:
            image.save(out, optimize=True)
        corner = image.getpixel((0, 0))
        size, mode = image.size, image.mode
    print(f"rendered {dst} {size} {mode} corner={corner} ({out.stat().st_size} bytes)")


def main() -> None:
    checks = {
        "mint on GitHub dark": contrast(MINT, GITHUB_DARK_BG),
        "green on GitHub light": contrast(GREEN, GITHUB_LIGHT_BG),
        "ink on GitHub light": contrast(INK, GITHUB_LIGHT_BG),
        "paper on GitHub dark": contrast(PAPER, GITHUB_DARK_BG),
        "mint node on tile": contrast(MINT, INK),
        "paper node on tile": contrast(PAPER, INK),
        "muted text on hero": contrast(MUTED_ON_DARK, INK_DEEP),
        "muted text on card": contrast(MUTED_ON_DARK, CARD_ON_DARK),
    }
    for name, ratio in checks.items():
        print(f"contrast {name}: {ratio:.2f}:1")
        if ratio < 4.5 and "text" in name or ratio < 3.0:
            sys.exit(f"contrast too low: {name}")

    fonts = load_fonts()
    favicon = svg(64, 64, mark(), "Graph RAG")
    files = {
        "logo-mark.svg": favicon,
        "favicon.svg": favicon,
        "favicon-16.svg": svg(16, 16, favicon16(), "Graph RAG"),
        "app-icon-square.svg": svg(64, 64, mark(rx=0), "Graph RAG"),
        "logo-horizontal-light.svg": lockup(fonts, word=PAPER, accent=MINT),
        "logo-horizontal-dark.svg": lockup(fonts, word=INK, accent=GREEN),
        "banner.svg": hero(fonts, width=1280, height=400, rounded=True, social=False),
        "social-preview.svg": hero(fonts, width=1280, height=640, rounded=False, social=True),
        "architecture.svg": architecture(fonts),
    }
    for name, text in files.items():
        check_safe(name, text)
        (BRAND / name).write_text(text, encoding="utf-8", newline="\n")
        print(f"wrote {name} ({len(text)} bytes)")
    write_package_module(favicon)

    resvg = shutil.which("resvg")
    if not resvg:
        sys.exit("resvg not found on PATH; install it to export PNGs")
    for src, dst, width, opaque in (
        ("logo-mark.svg", "logo-mark-512.png", 512, False),
        ("favicon.svg", "favicon-32.png", 32, False),
        ("favicon-16.svg", "favicon-16.png", 16, False),
        ("app-icon-square.svg", "apple-touch-icon.png", 180, True),
        ("social-preview.svg", "social-preview.png", 1280, True),
        ("banner.svg", "banner.png", 1280, False),
        ("architecture.svg", "architecture.png", 1920, False),
    ):
        render(resvg, src, dst, width, opaque)


if __name__ == "__main__":
    main()
