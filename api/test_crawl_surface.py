"""Crawl-side free Firecrawl surface: scrape_options, map, site crawl, batch."""

import asyncio

import httpx
from fastapi.testclient import TestClient

import app

HEADERS = {"X-API-Key": "crawl-surface-test-key"}


def _restore(saved):
    for name, value in saved.items():
        setattr(app, name, value)
    app.health_cache_reset()


def _setup(**overrides):
    saved = {
        name: getattr(app, name)
        for name in (
            "WEBSEARCH_API_KEY",
            "get_client",
            "is_allowed_public_url",
            "firecrawl_url_allowed",
            "firecrawl_request",
            "firecrawl_status",
            "call_crawl4ai",
        )
    }
    app.WEBSEARCH_API_KEY = HEADERS["X-API-Key"]
    app.get_client = lambda: asyncio.sleep(0, result=object())
    app.is_allowed_public_url = lambda _url: asyncio.sleep(0, result=True)
    app.firecrawl_url_allowed = lambda _url: True
    app.health_cache_reset()
    for name, value in overrides.items():
        setattr(app, name, value)
    return saved


def test_crawl_scrape_options_reach_firecrawl_and_return_extras():
    calls = []

    async def firecrawl(_client, endpoint, payload, *, timeout):
        calls.append((endpoint, payload))
        return httpx.Response(
            200,
            json={
                "success": True,
                "data": {
                    "markdown": "# Page",
                    "links": ["https://example.com/a"],
                    "metadata": {
                        "statusCode": 200,
                        "sourceURL": "https://example.com/",
                        "title": "Example",
                    },
                },
            },
        )

    async def no_crawl4ai(*_args, **_kwargs):
        raise AssertionError("Firecrawl success must not fall back")

    saved = _setup(firecrawl_request=firecrawl, call_crawl4ai=no_crawl4ai)
    try:
        with TestClient(app.app) as client:
            response = client.post(
                "/crawl",
                headers=HEADERS,
                json={
                    "url": "https://example.com/",
                    "scrape_options": {
                        "formats": ["markdown", "links"],
                        "onlyMainContent": False,
                    },
                },
            )
        assert response.status_code == 200, response.text
        data = response.json()["data"]
        assert data["provider"] == "firecrawl", data
        assert data["content"] == "# Page", data
        assert data["extras"] == {"links": ["https://example.com/a"]}, data
        assert data["metadata"]["title"] == "Example", data
        assert data["warnings"] == [], data
        assert calls == [
            (
                "/v2/scrape",
                {
                    "url": "https://example.com/",
                    "formats": ["markdown", "links"],
                    "onlyMainContent": False,
                    "maxAge": 0,
                    "skipTlsVerification": False,
                    "parsers": [],
                    "storeInCache": False,
                },
            )
        ], calls
    finally:
        _restore(saved)


def test_crawl_scrape_options_reject_mixing_and_enterprise_controls():
    saved = _setup()
    try:
        with TestClient(app.app) as client:
            mixed = client.post(
                "/crawl",
                headers=HEADERS,
                json={
                    "url": "https://example.com/",
                    "scrape_options": {"formats": ["markdown"]},
                    "cache_mode": "bypass",
                },
            )
            assert mixed.status_code == 422, mixed.text
            assert mixed.json()["error"]["code"] == "validation_error", mixed.text

            enterprise = client.post(
                "/crawl",
                headers=HEADERS,
                json={
                    "url": "https://example.com/",
                    "scrape_options": {"threatProtection": True},
                },
            )
            assert enterprise.status_code == 422, enterprise.text
    finally:
        _restore(saved)


def test_crawl_strict_and_degraded_fallback_for_unsupported_formats():
    payloads = []

    async def unavailable(*_args, **_kwargs):
        raise app.FirecrawlUnavailable(503, "unavailable", None, True)

    async def crawl4ai(_client, payload):
        payloads.append(payload)
        return {"success": True, "url": payload["urls"][0], "markdown": "# Local"}

    saved = _setup(firecrawl_request=unavailable, call_crawl4ai=crawl4ai)
    try:
        with TestClient(app.app) as client:
            strict = client.post(
                "/crawl",
                headers=HEADERS,
                json={
                    "url": "https://example.com/",
                    "scrape_options": {"formats": ["links"]},
                },
            )
            assert strict.status_code == 503, strict.text
            assert strict.json()["error"]["code"] == "fallback_feature_unavailable", strict.text
            assert payloads == [], payloads

            degraded = client.post(
                "/crawl",
                headers=HEADERS,
                json={
                    "url": "https://example.com/doc.pdf",
                    "scrape_options": {"formats": ["links"]},
                    "allow_degraded": True,
                },
            )
        assert degraded.status_code == 200, degraded.text
        data = degraded.json()["data"]
        assert data["provider"] == "crawl4ai", data
        assert data["content"] == "# Local", data
        assert data["degraded"] == {
            "provider": "crawl4ai",
            "unsupported": ["format:links"],
        }, data
        assert data["warnings"] == ["crawl4ai could not honor: format:links"], data
        assert payloads[0]["crawler_config"]["scraping_strategy"]["type"] == (
            "PDFContentScrapingStrategy"
        ), payloads[0]
    finally:
        _restore(saved)


def test_map_payload_and_response_normalization():
    calls = []

    async def firecrawl(_client, endpoint, payload, *, timeout):
        calls.append((endpoint, payload))
        return httpx.Response(
            200,
            json={
                "success": True,
                "links": [
                    "https://example.com/plain",
                    {"url": "https://example.com/titled", "title": "T", "description": "D"},
                    {"nope": True},
                    "http://127.0.0.1:80@example.com/",
                    "javascript:alert(1)",
                    {"url": "ftp://example.com/file"},
                ],
            },
        )

    saved = _setup(firecrawl_request=firecrawl)
    try:
        with TestClient(app.app) as client:
            response = client.post(
                "/map",
                headers=HEADERS,
                json={
                    "url": "https://example.com/",
                    "search": "docs",
                    "sitemap": "only",
                    "limit": 50,
                    "timeout_ms": 5000,
                },
            )
        assert response.status_code == 200, response.text
        data = response.json()["data"]
        assert data["provider"] == "firecrawl", data
        assert data["urls"] == [
            {"url": "https://example.com/plain"},
            {
                "url": "https://example.com/titled",
                "title": "T",
                "description": "D",
            },
        ], data
        assert calls == [
            (
                "/v2/map",
                {
                    "url": "https://example.com/",
                    "limit": 50,
                    "includeSubdomains": True,
                    "ignoreQueryParameters": True,
                    "search": "docs",
                    "sitemap": "only",
                    "timeout": 5000,
                },
            )
        ], calls
    finally:
        _restore(saved)


def test_map_rejects_non_public_seed():
    saved = _setup(is_allowed_public_url=lambda _url: asyncio.sleep(0, result=False))
    try:
        with TestClient(app.app) as client:
            response = client.post(
                "/map", headers=HEADERS, json={"url": "http://127.0.0.1/"}
            )
        assert response.status_code == 422, response.text
        assert response.json()["error"]["code"] == "invalid_url", response.text
    finally:
        _restore(saved)


def test_site_crawl_start_and_paged_status():
    start_calls = []
    status_calls = []

    async def firecrawl(_client, endpoint, payload, *, timeout):
        start_calls.append((endpoint, payload))
        return httpx.Response(200, json={"success": True, "id": "job-1"})

    async def status(_client, path, *, timeout):
        status_calls.append(path)
        return httpx.Response(
            200,
            json={
                "success": True,
                "status": "completed",
                "completed": 3,
                "total": 3,
                "data": [
                    {
                        "markdown": f"# {index}",
                        "metadata": {"sourceURL": f"https://example.com/{index}"},
                    }
                    for index in range(3)
                ],
            },
        )

    saved = _setup(firecrawl_request=firecrawl, firecrawl_status=status)
    try:
        with TestClient(app.app) as client:
            started = client.post(
                "/crawl-site",
                headers=HEADERS,
                json={
                    "url": "https://example.com/",
                    "limit": 2,
                    "include_paths": ["/docs/*"],
                    "scrape_options": {"formats": ["markdown"]},
                },
            )
            assert started.status_code == 200, started.text
            assert started.json() == {
                "ok": True,
                "data": {
                    "id": "job-1",
                    "url": "https://example.com/",
                    "status": "scraping",
                    "completed": 0,
                    "total": 0,
                    "items": [],
                    "offset": 0,
                    "limit": 0,
                    "truncated": False,
                    "provider": "firecrawl",
                },
            }, started.json()
            assert start_calls[0][0] == "/v2/crawl", start_calls
            assert start_calls[0][1]["scrapeOptions"] == {
                "formats": ["markdown"],
                "storeInCache": False,
            }, start_calls

            polled = client.get(
                "/crawl-site/job-1?offset=1&limit=1", headers=HEADERS
            )
        assert polled.status_code == 200, polled.text
        data = polled.json()["data"]
        assert data["id"] == "job-1", data
        assert data["status"] == "completed", data
        assert data["completed"] == 3 and data["total"] == 3, data
        assert data["offset"] == 1 and data["limit"] == 1, data
        assert data["truncated"] is True, data
        assert [item["content"] for item in data["items"]] == ["# 1"], data
        assert data["items"][0]["metadata"]["sourceURL"] == "https://example.com/1", data
        assert status_calls == ["/v2/crawl/job-1"], status_calls
    finally:
        _restore(saved)


def test_job_status_maps_missing_and_unavailable_jobs():
    async def missing(_client, _path, *, timeout):
        return httpx.Response(404, json={"success": False})

    saved = _setup(firecrawl_status=missing)
    try:
        with TestClient(app.app) as client:
            response = client.get("/crawl-site/missing", headers=HEADERS)
        assert response.status_code == 404, response.text
        assert response.json()["error"]["code"] == "job_not_found", response.text
    finally:
        _restore(saved)


def test_batch_crawl_filters_non_public_urls_and_polls_batch_endpoint():
    calls = []
    status_calls = []

    async def firecrawl(_client, endpoint, payload, *, timeout):
        calls.append((endpoint, payload))
        return httpx.Response(200, json={"success": True, "id": "batch-1"})

    async def status(_client, path, *, timeout):
        status_calls.append(path)
        return httpx.Response(
            200,
            json={"success": True, "status": "scraping", "completed": 0, "total": 1, "data": []},
        )

    async def public(url):
        return "private" not in url

    saved = _setup(
        firecrawl_request=firecrawl,
        firecrawl_status=status,
        is_allowed_public_url=public,
    )
    try:
        with TestClient(app.app) as client:
            response = client.post(
                "/batch-crawl",
                headers=HEADERS,
                json={
                    "urls": ["https://example.com/a", "http://private.internal/b"],
                    "scrape_options": {"formats": ["markdown"]},
                },
            )
        assert response.status_code == 200, response.text
        assert response.json()["data"]["id"] == "batch-1", response.text
        assert calls == [
            (
                "/v2/batch/scrape",
                {
                    "urls": ["https://example.com/a"],
                    "ignoreInvalidURLs": True,
                    "scrapeOptions": {"formats": ["markdown"], "storeInCache": False},
                },
            )
        ], calls

        with TestClient(app.app) as client:
            strict = client.post(
                "/batch-crawl",
                headers=HEADERS,
                json={
                    "urls": ["https://example.com/a", "http://private.internal/b"],
                    "ignore_invalid_urls": False,
                },
            )
            assert strict.status_code == 422, strict.text
            assert strict.json()["error"]["code"] == "invalid_url", strict.text

            polled = client.get("/batch-crawl/batch-1", headers=HEADERS)
        assert polled.status_code == 200, polled.text
        assert status_calls == ["/v2/batch/scrape/batch-1"], status_calls
    finally:
        _restore(saved)


def demo():
    test_crawl_scrape_options_reach_firecrawl_and_return_extras()
    test_crawl_scrape_options_reject_mixing_and_enterprise_controls()
    test_crawl_strict_and_degraded_fallback_for_unsupported_formats()
    test_map_payload_and_response_normalization()
    test_map_rejects_non_public_seed()
    test_site_crawl_start_and_paged_status()
    test_job_status_maps_missing_and_unavailable_jobs()
    test_batch_crawl_filters_non_public_urls_and_polls_batch_endpoint()
    print("crawl free surface: ok")


if __name__ == "__main__":
    demo()
