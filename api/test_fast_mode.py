"""Checks for the depth=basic fast tier (#14). Run: python api/test_fast_mode.py"""

import asyncio

from pydantic import ValidationError

import app

CANDIDATES = [
    {"url": "https://a.example/", "title": "A", "snippet": "A snippet about probes."},
    {"url": "https://b.example/", "title": "B", "snippet": "Another snippet."},
]


def test_depth_defaults_to_advanced():
    assert app.SearchRequest(query="q").depth == "advanced"
    assert app.SearchRequest(query="q", depth="basic").depth == "basic"


def test_unknown_depth_is_rejected():
    try:
        app.SearchRequest(query="q", depth="fast")
    except ValidationError:
        return
    raise AssertionError("depth must be limited to basic|advanced")


def test_get_route_forwards_depth():
    seen = {}
    original = app.search

    async def fake_search(request):
        seen["depth"] = request.depth
        return app.search_response(request.query, [])

    app.search = fake_search
    try:
        asyncio.run(app.search_get(q="q", max_results=5, candidates=10, depth="basic"))
    finally:
        app.search = original
    assert seen["depth"] == "basic", seen


def test_snippet_pages_use_the_snippet_as_content():
    pages = app.snippet_pages(CANDIDATES)
    assert pages == [
        {"url": "https://a.example/", "title": "A", "snippet": "A snippet about probes.", "content": "A snippet about probes."},
        {"url": "https://b.example/", "title": "B", "snippet": "Another snippet.", "content": "Another snippet."},
    ]


def test_snippetless_results_fall_back_to_the_title():
    # An empty passage scores nothing, so discovery results with no snippet still
    # reach the reranker carrying their title. The snippet field stays truthful.
    pages = app.snippet_pages([{"url": "https://c.example/", "title": "C title", "snippet": ""}])
    assert pages == [{"url": "https://c.example/", "title": "C title", "snippet": "", "content": "C title"}]


def test_short_snippets_survive_to_the_reranker():
    # Basic mode must not run page-boilerplate filtering over snippets: it drops
    # short lines, which is most snippets, including "Another snippet." here.
    chunks = app.snippet_pages(CANDIDATES)
    assert [chunk["url"] for chunk in chunks] == ["https://a.example/", "https://b.example/"]
    assert [chunk["content"] for chunk in chunks] == [item["snippet"] for item in CANDIDATES]
    assert app.chunk_documents(chunks) != chunks, "the page path does filter; basic must not use it"


def demo():
    for name, check in sorted(globals().items()):
        if name.startswith("test_"):
            check()
            print(f"ok {name}")
    print("all checks passed")


if __name__ == "__main__":
    demo()
