"""Packaged API compatibility checks for the retained dependency upgrade."""

from fastapi.testclient import TestClient

import app


def check_auth_and_validation_envelopes(client):
    response = client.post("/search", json={"query": "test"})
    assert response.status_code == 401, response.text
    assert response.json()["ok"] is False, response.text
    assert response.json()["error"]["code"] == "unauthorized", response.text

    response = client.post(
        "/search", json={"query": None}, headers={"X-API-Key": app.WEBSEARCH_API_KEY}
    )
    assert response.status_code == 422, response.text
    assert response.json()["ok"] is False, response.text
    assert response.json()["error"]["code"] == "validation_error", response.text


def check_get_defaults_and_openapi_surfaces(client):
    seen = {}
    original = app.search

    async def fake_search(request):
        seen.update(
            query=request.query,
            max_results=request.max_results,
            candidates=request.candidates,
            depth=request.depth,
        )
        return app.search_response(request.query, [])

    app.search = fake_search
    try:
        response = client.get(
            "/search?q=contract",
            headers={"X-API-Key": app.WEBSEARCH_API_KEY},
        )
    finally:
        app.search = original
    assert response.status_code == 200, response.text
    assert response.json()["ok"] is True, response.text
    assert seen == {
        "query": "contract",
        "max_results": app.MAX_RESULTS,
        "candidates": app.SEARCH_CANDIDATES,
        "depth": "advanced",
    }, seen

    for host, paths, title in (
        ("websearch.sparkfn.io", {"/health", "/search"}, "SparkFN Web Search API"),
        ("webcrawl.sparkfn.io", {"/health", "/crawl"}, "SparkFN Web Crawl API"),
    ):
        response = client.get("/openapi.json", headers={"Host": host})
        assert response.status_code == 200, response.text
        schema = response.json()
        assert set(schema["paths"]) == paths, schema["paths"]
        assert schema["info"]["title"] == title, schema["info"]


def check_request_id_and_search_filter_contract(client):
    response = client.get("/openapi.json", headers={"X-Request-ID": "req-contract-1"})
    assert response.headers["X-Request-ID"] == "req-contract-1", response.headers
    response = client.get("/openapi.json", headers={"X-Request-ID": "bad id"})
    assert response.headers["X-Request-ID"] != "bad id", response.headers

    headers = {"X-API-Key": app.WEBSEARCH_API_KEY}
    response = client.post(
        "/search",
        json={
            "query": "q",
            "include_domains": ["a.example"],
            "exclude_domains": ["b.example"],
        },
        headers=headers,
    )
    assert response.status_code == 422, response.text
    response = client.post(
        "/search",
        json={"query": "q", "categories": ["gov"], "time_range": "week"},
        headers=headers,
    )
    assert response.status_code == 422, response.text
    response = client.post(
        "/search",
        json={"query": "q", "scrape_options": {"zeroDataRetention": True}},
        headers=headers,
    )
    assert response.status_code == 422, response.text
    response = client.post(
        "/search",
        json={
            "query": "q",
            "scrape_options": {"actions": [{"type": "wait", "milliseconds": 1}]},
        },
        headers=headers,
    )
    assert response.status_code == 422, response.text


def check_free_surface_contract(client):
    headers = {"X-API-Key": app.WEBSEARCH_API_KEY}
    response = client.post(
        "/search",
        json={"query": None},
        headers={**headers, "X-Request-ID": "req-error-1"},
    )
    assert response.status_code == 422, response.text
    assert response.json()["error"]["request_id"] == "req-error-1", response.text
    assert response.headers["X-Request-ID"] == "req-error-1", response.headers

    for payload in (
        {"query": "q", "scrape_options": {"formats": 3}},
        {"query": "q", "scrape_options": {"formats": ["not-a-format"]}},
        {"query": "q", "scrape_options": {"formats": [{"type": "attributes"}]}},
        {"query": "q", "scrape_options": {"formats": [{"type": "question"}]}},
        {"query": "q", "scrape_options": {"formats": [{"type": "highlights"}]}},
        {"query": "q", "categories": ["gov"], "safe": True},
    ):
        response = client.post("/search", json=payload, headers=headers)
        assert response.status_code == 422, (payload, response.text)

    filters = app.search_filters(
        app.SearchRequest(query="q", scrape_options={"formats": ["markdown"]})
    )
    assert filters["scrapeOptions"]["storeInCache"] is False, filters
    filters = app.search_filters(
        app.SearchRequest(
            query="q",
            scrape_options={"formats": ["markdown"], "storeInCache": True},
        )
    )
    assert filters["scrapeOptions"]["storeInCache"] is True, filters

    assert app.SEARCH_CRAWLER_CONFIG["pdf"] is True
    capped = app.cap_inline_extras(
        {
            "links": ["https://a.example/"],
            "rawBase64": "x" * (app.MAX_EXTRAS_BYTES + 1),
            "summary": "y" * (app.MAX_EXTRAS_BYTES + 1),
        }
    )
    assert "rawBase64" not in capped, capped
    assert capped["omitted"] == ["rawBase64"], capped
    assert capped["links"] == ["https://a.example/"], capped
    assert capped["summary"].endswith("[truncated]"), capped["summary"][-20:]

    results = [
        app.SearchResult(
            title="a",
            url="https://a.example/",
            content="c",
            extras={"links": ["https://a.example/"], "rawBase64": "y" * 200},
        ),
        app.SearchResult(
            title="b",
            url="https://b.example/",
            content="c",
            extras={"summary": "z" * 200},
        ),
    ]
    bounded, truncated = app.cap_result_extras(results, per_field=100, total=250)
    assert truncated is True
    assert bounded[0].extras["links"] == ["https://a.example/"], bounded[0].extras
    assert bounded[0].extras["rawBase64"].endswith("[truncated]"), bounded[0].extras
    assert bounded[1].extras["summary"].endswith("[truncated]"), bounded[1].extras


def demo():
    app.WEBSEARCH_API_KEY = "dependency-contract-test-key"
    with TestClient(app.app) as client:
        check_auth_and_validation_envelopes(client)
        check_get_defaults_and_openapi_surfaces(client)
        check_request_id_and_search_filter_contract(client)
        check_free_surface_contract(client)
    print("API dependency contracts: ok")


if __name__ == "__main__":
    demo()
