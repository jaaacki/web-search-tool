"""Regression checks for the hostile-review crawl fixes."""

import asyncio
import json

import httpx
from fastapi.testclient import TestClient

import app

HEADERS = {"X-API-Key": "crawl-fixes-key"}


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
            "_run_health_checks",
            "firecrawl_usage_summary",
        )
    }
    app.WEBSEARCH_API_KEY = HEADERS["X-API-Key"]
    app.get_client = lambda: asyncio.sleep(0, result=object())
    app.is_allowed_public_url = lambda _url: asyncio.sleep(0, result=True)
    app.firecrawl_url_allowed = lambda _url, **_kwargs: True
    app.health_cache_reset()
    for name, value in overrides.items():
        setattr(app, name, value)
    return saved


def test_job_status_follows_next_pages():
    calls = []

    async def status(_client, path, *, timeout):
        calls.append(path)
        if path.startswith("https://api.firecrawl.dev"):
            return httpx.Response(
                200,
                json={
                    "success": True,
                    "status": "completed",
                    "completed": 3,
                    "total": 3,
                    "data": [
                        {"markdown": "# b", "metadata": {"sourceURL": "https://example.com/b"}},
                        {"markdown": "# c", "metadata": {"sourceURL": "https://example.com/c"}},
                    ],
                    "next": None,
                },
            )
        return httpx.Response(
            200,
            json={
                "success": True,
                "status": "scraping",
                "completed": 1,
                "total": 3,
                "data": [
                    {"markdown": "# a", "metadata": {"sourceURL": "https://example.com/a"}}
                ],
                "next": "https://api.firecrawl.dev/v2/crawl/job-1?skip=1",
            },
        )

    saved = _setup(firecrawl_status=status)
    try:
        with TestClient(app.app) as client:
            response = client.get("/crawl-site/job-1?offset=0&limit=5", headers=HEADERS)
        assert response.status_code == 200, response.text
        data = response.json()["data"]
        assert [item["content"] for item in data["items"]] == ["# a", "# b", "# c"], data
        assert data["status"] == "completed", data
        assert data["truncated"] is False, data
        assert calls == [
            "/v2/crawl/job-1",
            "https://api.firecrawl.dev/v2/crawl/job-1?skip=1",
        ], calls
    finally:
        _restore(saved)


def test_malformed_firecrawl_2xx_is_502_not_500():
    async def firecrawl(_client, _endpoint, _payload, *, timeout):
        return httpx.Response(200, text="<html>edge error</html>")

    saved = _setup(firecrawl_request=firecrawl)
    try:
        with TestClient(app.app) as client:
            response = client.post("/map", headers=HEADERS, json={"url": "https://example.com"})
        assert response.status_code == 502, response.text
        assert response.json()["error"]["code"] == "upstream_error", response.text
    finally:
        _restore(saved)


def test_crawl_missing_requested_format_is_strict_or_degraded():
    async def firecrawl(_client, _endpoint, _payload, *, timeout):
        return httpx.Response(
            200,
            json={
                "success": True,
                "data": {
                    "markdown": "# Page",
                    "metadata": {"statusCode": 200, "sourceURL": "https://example.com/"},
                },
            },
        )

    saved = _setup(firecrawl_request=firecrawl)
    try:
        with TestClient(app.app) as client:
            strict = client.post(
                "/crawl",
                headers=HEADERS,
                json={
                    "url": "https://example.com/",
                    "scrape_options": {"formats": ["markdown", "links"]},
                },
            )
            assert strict.status_code == 503, strict.text
            assert strict.json()["error"]["code"] == "fallback_feature_unavailable", strict.text
            assert strict.json()["error"]["details"]["unsupported"] == ["format:links"], strict.text

            degraded = client.post(
                "/crawl",
                headers=HEADERS,
                json={
                    "url": "https://example.com/",
                    "scrape_options": {"formats": ["markdown", "links"]},
                    "allow_degraded": True,
                },
            )
        assert degraded.status_code == 200, degraded.text
        data = degraded.json()["data"]
        assert data["degraded"] == {"provider": "firecrawl", "unsupported": ["format:links"]}, data
        assert any("links" in warning for warning in data["warnings"]), data
    finally:
        _restore(saved)


def test_job_page_budget_counts_extras_and_metadata():
    saved_budget = app.MAX_JOB_ITEMS_BYTES
    app.MAX_JOB_ITEMS_BYTES = 40_000
    try:
        payload = {
            "success": True,
            "status": "completed",
            "completed": 2,
            "total": 2,
            "data": [
                {
                    "markdown": "m" * 20_000,
                    "html": "h" * 20_000,
                    "metadata": {"title": "t" * 5_000, "sourceURL": "https://example.com/1"},
                },
                {
                    "markdown": "n" * 20_000,
                    "html": "i" * 20_000,
                    "metadata": {"title": "u" * 5_000, "sourceURL": "https://example.com/2"},
                },
            ],
        }
        data = app._normalize_job_payload(payload, "job-1", offset=0, limit=2)
        assert data is not None
        assert data.truncated is True, data
        size = len(json.dumps(data.model_dump(), ensure_ascii=False).encode("utf-8"))
        assert size < app.MAX_JOB_ITEMS_BYTES + 5_000, size
    finally:
        app.MAX_JOB_ITEMS_BYTES = saved_budget


def test_job_items_drop_private_and_credentialed_urls():
    payload = {
        "success": True,
        "status": "completed",
        "completed": 1,
        "total": 1,
        "data": [
            {
                "markdown": "# x",
                "url": "http://169.254.169.254/latest/meta-data/",
                "metadata": {"sourceURL": "https://user:pw@evil.example/x"},
            }
        ],
    }
    data = app._normalize_job_payload(payload, "job-1", offset=0, limit=1)
    item = data.items[0]
    assert "url" not in item, item
    assert "sourceURL" not in item["metadata"], item
    assert "non-public" in item["error"], item


def test_map_link_sieve_rejects_private_ip_literals():
    assert app._public_http_url("http://127.0.0.1/") is False
    assert app._public_http_url("http://10.0.0.1/") is False
    assert app._public_http_url("http://[::1]/") is False
    assert app._public_http_url("https://user:pw@example.com/") is False
    assert app._public_http_url("https://example.com/") is True


def test_health_usage_requires_api_key():
    async def checks():
        return {
            "firecrawl": "ok",
            "brave": "ok",
            "searxng": "ok",
            "searxng_service": "ok",
            "crawl4ai": "ok",
            "reranker": "ok",
        }

    usage = {
        "date": "2026-10-25",
        "requests": 2,
        "credits": 3.0,
        "errors": 0,
        "by_endpoint": {"/v2/search": {"requests": 2, "credits": 3.0}},
        "last_request_at": None,
    }
    saved = _setup(_run_health_checks=checks, firecrawl_usage_summary=lambda: usage)
    try:
        app.health_cache_reset()
        with TestClient(app.app) as client:
            public = client.get("/health")
            assert public.status_code == 200, public.text
            assert public.json()["data"]["usage"] is None, public.text
            app.health_cache_reset()
            private = client.get("/health", headers=HEADERS)
        assert private.status_code == 200, private.text
        assert private.json()["data"]["usage"] == usage, private.text
    finally:
        _restore(saved)


def test_cost_sensitive_site_and_batch_targets_are_rejected():
    def allowed(url, **_kwargs):
        return "linkedin.com" not in url

    saved = _setup(firecrawl_url_allowed=allowed)
    calls = []

    async def firecrawl(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("routed target reached Firecrawl")

    app.firecrawl_request = firecrawl
    try:
        with TestClient(app.app) as client:
            site = client.post(
                "/crawl-site",
                headers=HEADERS,
                json={"url": "https://linkedin.com/in/x"},
            )
            assert site.status_code == 422, site.text
            assert site.json()["error"]["code"] == "unsupported_feature", site.text

            batch = client.post(
                "/batch-crawl",
                headers=HEADERS,
                json={
                    "urls": ["https://example.com/a", "https://linkedin.com/in/x"],
                    "ignore_invalid_urls": False,
                },
            )
            assert batch.status_code == 422, batch.text
            assert batch.json()["error"]["details"]["routed"] == [
                "https://linkedin.com/in/x"
            ], batch.text
        assert calls == [], calls
    finally:
        _restore(saved)


def test_parsers_are_reported_as_crawl4ai_losses():
    assert app.crawl4ai_feature_loss(
        {"formats": ["markdown"], "parsers": ["image"]}, "markdown", "https://example.com/a"
    ) == ["parser:image"]
    assert app.crawl4ai_feature_loss(
        {"formats": ["markdown"], "parsers": ["pdf"]}, "markdown", "https://example.com/a"
    ) == ["parser:pdf"]
    assert app.crawl4ai_feature_loss(
        {"formats": ["markdown"], "parsers": ["pdf"]}, "markdown", "https://example.com/a.pdf"
    ) == []


def test_map_timeout_is_forwarded_to_the_local_deadline():
    seen = []

    async def firecrawl(_client, endpoint, payload, *, timeout):
        seen.append((endpoint, timeout))
        return httpx.Response(200, json={"success": True, "links": []})

    saved = _setup(firecrawl_request=firecrawl)
    try:
        with TestClient(app.app) as client:
            response = client.post(
                "/map",
                headers=HEADERS,
                json={"url": "https://example.com/", "timeout_ms": 120000},
            )
        assert response.status_code == 200, response.text
        assert seen == [("/v2/map", 125.0)], seen
    finally:
        _restore(saved)


def demo():
    test_job_status_follows_next_pages()
    test_malformed_firecrawl_2xx_is_502_not_500()
    test_crawl_missing_requested_format_is_strict_or_degraded()
    test_job_page_budget_counts_extras_and_metadata()
    test_job_items_drop_private_and_credentialed_urls()
    test_map_link_sieve_rejects_private_ip_literals()
    test_health_usage_requires_api_key()
    test_cost_sensitive_site_and_batch_targets_are_rejected()
    test_parsers_are_reported_as_crawl4ai_losses()
    test_map_timeout_is_forwarded_to_the_local_deadline()
    print("crawl review fixes: ok")


if __name__ == "__main__":
    demo()
