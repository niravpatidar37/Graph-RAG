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
