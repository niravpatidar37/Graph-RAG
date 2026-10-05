import base64
import hashlib
import re

from fastapi.testclient import TestClient

from graph_rag.api import app
from graph_rag.brand import FAVICON_SVG, THEME_COLOR

client = TestClient(app)

UNSAFE_SVG = [
    r"<\s*script",
    r"\son[a-z]+\s*=",
    r"<\s*foreignObject",
    r"href\s*=",
    r"url\s*\(",
    r"javascript:",
]


def test_favicon_is_served_as_inert_svg() -> None:
    response = client.get("/favicon.svg")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/svg+xml")
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.text == FAVICON_SVG
    for pattern in UNSAFE_SVG:
        assert not re.search(pattern, response.text, re.IGNORECASE), pattern


def test_home_page_links_brand_assets() -> None:
    page = client.get("/").text

    assert '<link rel="icon" type="image/svg+xml" href="/favicon.svg">' in page
    assert f'<meta name="theme-color" content="{THEME_COLOR}">' in page
    assert "THEME_COLOR_PLACEHOLDER" not in page


def test_favicon_is_not_in_openapi_schema() -> None:
    assert "/favicon.svg" not in client.get("/openapi.json").json()["paths"]


def _csp_hash(text: str) -> str:
    return "'sha256-" + base64.b64encode(hashlib.sha256(text.encode()).digest()).decode() + "'"


def test_home_page_csp_pins_the_exact_inline_script_and_style() -> None:
    response = client.get("/")
    csp = response.headers["content-security-policy"]
    page = response.text
    scripts = re.findall(r"<script>(.*?)</script>", page, re.S)
    styles = re.findall(r"<style>(.*?)</style>", page, re.S)

    assert len(scripts) == 1 and len(styles) == 1
    assert f"script-src {_csp_hash(scripts[0])}" in csp
    assert f"style-src {_csp_hash(styles[0])}" in csp
    assert "unsafe-inline" not in csp and "unsafe-eval" not in csp
    assert "default-src 'none'" in csp and "frame-ancestors 'none'" in csp
    assert response.headers["x-frame-options"] == "DENY"
    assert response.headers["referrer-policy"] == "no-referrer"


def test_home_page_never_parses_strings_as_html() -> None:
    page = client.get("/").text

    for sink in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval(", "new Function"):
        assert sink not in page, sink
    assert ' style="' not in page                 # CSP has no 'unsafe-inline' for style attributes


def test_question_length_is_capped() -> None:
    assert client.post("/query", json={"question": "x" * 2001}).status_code == 422
    assert client.post("/query/stream", json={"question": ""}).status_code == 422
