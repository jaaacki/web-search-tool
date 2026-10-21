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


def demo():
    app.WEBSEARCH_API_KEY = "dependency-contract-test-key"
    with TestClient(app.app) as client:
        check_auth_and_validation_envelopes(client)
        check_get_defaults_and_openapi_surfaces(client)
    print("API dependency contracts: ok")


if __name__ == "__main__":
    demo()
