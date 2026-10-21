"""Firecrawl v2 adapter payload and normalization checks for #46."""

import asyncio

import httpx

import app
import firecrawl_adapter as adapter


def test_search_payload_is_web_only_fresh_and_bounded():
    payload = adapter.search_payload("q" * 500, 50)
    assert payload == {"query": "q" * 500, "limit": 50, "sources": ["web"]}
    assert "scrapeOptions" not in payload
    assert adapter.search_payload("q" * 501, 50) is None
    assert adapter.search_payload("q", 51) is None


def test_search_uses_grouped_web_results_without_embedded_scraping():
    response = httpx.Response(
        200,
        json={
            "success": True,
            "data": {
                "web": [
                    {
                        "url": "https://a.example/",
                        "title": "A",
                        "description": "A snippet",
                        "markdown": "# Full A",
                        "metadata": {
                            "sourceURL": "https://a.example",
                            "url": "https://redirected.example/",
                            "statusCode": 200,
                        },
                    },
                    {
                        "url": "https://b.example/",
                        "title": "B",
                        "description": "B snippet",
                        "markdown": "# Failed B",
                        "metadata": {"sourceURL": "https://b.example/", "statusCode": 404},
                    },
                    {"url": "https://c.example/", "title": "C", "description": "No scrape data"},
                    {"url": "not a url", "title": "D", "description": "D"},
                    {"url": "https://reports.example/report.pdf", "title": "PDF", "description": "PDF result", "category": "pdf"},
                    {"title": "No identity", "description": "Not a candidate"},
                ],
                "news": [{"url": "https://news.example/", "title": "Not web"}],
            },
        },
    )

    results = adapter.search_results(response)

    assert results == [
        {"url": "https://a.example/", "title": "A", "description": "A snippet"},
        {
            "url": "https://b.example/",
            "title": "B",
            "description": "B snippet",
        },
        {
            "url": "https://c.example/",
            "title": "C",
            "description": "No scrape data",
        },
        {"url": "not a url", "title": "D", "description": "D"},
        {
            "url": "https://reports.example/report.pdf",
            "title": "PDF",
            "description": "PDF result",
            "category": "pdf",
        },
    ]
    assert adapter.is_pdf_target("https://reports.example/report.pdf")
    assert adapter.is_pdf_target("https://reports.example/download", "pdf")
    assert not adapter.is_pdf_target("https://reports.example/article")


def test_search_rejects_provider_and_response_failures_but_keeps_empty_success():
    assert adapter.search_results(httpx.Response(503, json={"success": False})) is None
    assert adapter.search_results(httpx.Response(200, json={"success": False})) is None
    assert adapter.search_results(httpx.Response(200, json={"success": True, "data": {"news": []}})) is None
    assert adapter.search_results(httpx.Response(200, json={"success": True, "data": {"web": []}})) == []


def test_only_plain_markdown_crawl_requests_are_firecrawl_compatible():
    ordinary = app.CrawlRequest(url="https://example.com")
    assert adapter.supports_markdown_crawl(ordinary)

    unsupported = (
        app.CrawlRequest(url="https://example.com", content_format="text"),
        app.CrawlRequest(url="https://example.com", cache_mode="bypass"),
        app.CrawlRequest(url="https://example.com", browser_config={"headless": True}),
        app.CrawlRequest(url="https://example.com", extraction_config={"unused": True}),
        app.CrawlRequest(url="https://example.com", crawler_config={"wait_for": "css:main"}),
        app.CrawlRequest(url="https://example.com", crawl_options={"word_count_threshold": 10}),
    )
    assert all(not adapter.supports_markdown_crawl(request) for request in unsupported)


def test_crawl_policy_denial_is_public_422_without_local_retry():
    saved_key = app.WEBSEARCH_API_KEY
    saved_get_client = app.get_client
    saved_public_check = app.is_allowed_public_url
    saved_firecrawl_check = app.firecrawl_url_allowed
    saved_firecrawl_request = app.firecrawl_request
    saved_crawl4ai = app.call_crawl4ai
    calls = []
    app.WEBSEARCH_API_KEY = "test-key"
    app.get_client = lambda: asyncio.sleep(0, result=object())
    app.is_allowed_public_url = lambda _: asyncio.sleep(0, result=True)
    app.firecrawl_url_allowed = lambda _: True

    async def denied(_client, endpoint, payload, *, timeout):
        calls.append(("firecrawl", endpoint))
        raise app.FirecrawlUnavailable(403, "policy_denied", None, False)

    async def local_crawl(_client, payload):
        calls.append(("crawl4ai", payload))
        raise AssertionError("policy denial must not retry this URL through Crawl4AI")

    app.firecrawl_request = denied
    app.call_crawl4ai = local_crawl

    async def request():
        transport = httpx.ASGITransport(app=app.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as api:
            return await api.post(
                "/crawl", headers={"X-API-Key": "test-key"}, json={"url": "https://example.com"}
            )

    try:
        response = asyncio.run(request())
        assert response.status_code == 422, response.text
        error = response.json()["error"]
        assert error["code"] == "firecrawl_policy_denied", error
        assert error["retryable"] is False, error
        assert error["details"] == {"provider": "firecrawl"}, error
        assert calls == [("firecrawl", "/v2/scrape")], calls
    finally:
        app.WEBSEARCH_API_KEY = saved_key
        app.get_client = saved_get_client
        app.is_allowed_public_url = saved_public_check
        app.firecrawl_url_allowed = saved_firecrawl_check
        app.firecrawl_request = saved_firecrawl_request
        app.call_crawl4ai = saved_crawl4ai
        app.health_cache_reset()


def test_provider_routed_direct_crawl_uses_local_crawl4ai():
    saved_public_check = app.is_allowed_public_url
    saved_firecrawl_check = app.firecrawl_url_allowed
    saved_firecrawl_request = app.firecrawl_request
    saved_crawl4ai = app.call_crawl4ai
    calls = []
    app.is_allowed_public_url = lambda _: asyncio.sleep(0, result=True)
    app.firecrawl_url_allowed = lambda _: False

    async def no_firecrawl(*_args, **_kwargs):
        calls.append("firecrawl")
        raise AssertionError("provider-routed target must be screened before Firecrawl")

    async def local_crawl(_client, payload):
        calls.append("crawl4ai")
        return {"success": True, "url": payload["urls"][0], "markdown": "# Local page"}

    app.firecrawl_request = no_firecrawl
    app.call_crawl4ai = local_crawl
    try:
        result = asyncio.run(app.crawl_direct_url(object(), app.CrawlRequest(url="https://linkedin.com/in/x")))
        assert result == {"url": "https://linkedin.com/in/x", "content": "# Local page"}, result
        assert calls == ["crawl4ai"], calls
    finally:
        app.is_allowed_public_url = saved_public_check
        app.firecrawl_url_allowed = saved_firecrawl_check
        app.firecrawl_request = saved_firecrawl_request
        app.call_crawl4ai = saved_crawl4ai
        app.health_cache_reset()


def test_public_policy_denial_is_422_without_crawl4ai_retry():
    saved_key = app.WEBSEARCH_API_KEY
    saved_get_client = app.get_client
    saved_public_check = app.is_allowed_public_url
    saved_firecrawl_check = app.firecrawl_url_allowed
    saved_firecrawl_request = app.firecrawl_request
    saved_crawl4ai = app.call_crawl4ai
    calls = []
    app.WEBSEARCH_API_KEY = "test-key"
    app.get_client = lambda: asyncio.sleep(0, result=object())
    app.is_allowed_public_url = lambda _: asyncio.sleep(0, result=True)
    app.firecrawl_url_allowed = lambda _: True

    async def denied(_client, endpoint, payload, *, timeout):
        calls.append(("firecrawl", endpoint))
        raise app.FirecrawlUnavailable(403, "policy_denied", None, False)

    async def local_crawl(_client, payload):
        calls.append(("crawl4ai", payload))
        raise AssertionError("policy denial must not retry through Crawl4AI")

    app.firecrawl_request = denied
    app.call_crawl4ai = local_crawl

    async def request():
        transport = httpx.ASGITransport(app=app.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as api:
            return await api.post(
                "/crawl", headers={"X-API-Key": "test-key"}, json={"url": "https://example.com"}
            )

    try:
        response = asyncio.run(request())
        assert response.status_code == 422, response.text
        error = response.json()["error"]
        assert error["code"] == "firecrawl_policy_denied", error
        assert error["retryable"] is False, error
        assert error["details"] == {"provider": "firecrawl"}, error
        assert calls == [("firecrawl", "/v2/scrape")], calls
    finally:
        app.WEBSEARCH_API_KEY = saved_key
        app.get_client = saved_get_client
        app.is_allowed_public_url = saved_public_check
        app.firecrawl_url_allowed = saved_firecrawl_check
        app.firecrawl_request = saved_firecrawl_request
        app.call_crawl4ai = saved_crawl4ai
        app.health_cache_reset()


def test_provider_routed_direct_crawl_uses_local_extractor():
    saved_public_check = app.is_allowed_public_url
    saved_firecrawl_check = app.firecrawl_url_allowed
    saved_firecrawl_request = app.firecrawl_request
    saved_crawl4ai = app.call_crawl4ai
    calls = []
    app.is_allowed_public_url = lambda _: asyncio.sleep(0, result=True)
    app.firecrawl_url_allowed = lambda _: False

    async def no_firecrawl(*_args, **_kwargs):
        calls.append("firecrawl")
        raise AssertionError("provider-routed target must be screened before Firecrawl")

    async def local_crawl(_client, payload):
        calls.append("crawl4ai")
        url = payload["urls"][0]
        return {"success": True, "url": url, "markdown": "# Local page"}

    app.firecrawl_request = no_firecrawl
    app.call_crawl4ai = local_crawl
    try:
        result = asyncio.run(
            app.crawl_direct_url(object(), app.CrawlRequest(url="https://linkedin.com/in/x"))
        )
        assert result == {"url": "https://linkedin.com/in/x", "content": "# Local page"}, result
        assert calls == ["crawl4ai"], calls
    finally:
        app.is_allowed_public_url = saved_public_check
        app.firecrawl_url_allowed = saved_firecrawl_check
        app.firecrawl_request = saved_firecrawl_request
        app.call_crawl4ai = saved_crawl4ai
        app.health_cache_reset()


def test_explicit_crawl4ai_options_keep_the_crawl4ai_route():
    saved_public_check = app.is_allowed_public_url
    saved_firecrawl_check = app.firecrawl_url_allowed
    saved_firecrawl_request = app.firecrawl_request
    saved_crawl4ai = app.call_crawl4ai
    calls = []
    app.is_allowed_public_url = lambda _: asyncio.sleep(0, result=True)
    app.firecrawl_url_allowed = lambda _: True

    async def no_firecrawl(*_args, **_kwargs):
        calls.append("firecrawl")
        raise AssertionError("explicit Crawl4AI options must not be silently ignored")

    async def local_crawl(_client, payload):
        calls.append("crawl4ai")
        assert payload["crawler_config"]["wait_for"] == "css:main", payload
        return {"success": True, "url": payload["urls"][0], "markdown": "# Local page"}

    app.firecrawl_request = no_firecrawl
    app.call_crawl4ai = local_crawl
    try:
        result = asyncio.run(
            app.crawl_direct_url(
                object(), app.CrawlRequest(url="https://example.com", crawler_config={"wait_for": "css:main"})
            )
        )
        assert result == {"url": "https://example.com", "content": "# Local page"}, result
        assert calls == ["crawl4ai"], calls
    finally:
        app.is_allowed_public_url = saved_public_check
        app.firecrawl_url_allowed = saved_firecrawl_check
        app.firecrawl_request = saved_firecrawl_request
        app.call_crawl4ai = saved_crawl4ai
        app.health_cache_reset()


def test_scrape_payload_pins_ordinary_main_markdown_and_freshness():
    assert adapter.scrape_payload("https://example.com") == {
        "url": "https://example.com",
        "formats": ["markdown"],
        "onlyMainContent": True,
        "maxAge": 0,
        "skipTlsVerification": False,
        "parsers": [],
    }


def test_scrape_requires_success_status_identity_and_nonempty_markdown():
    requested = "https://example.com"
    valid = httpx.Response(
        200,
        json={
            "success": True,
            "data": {
                "markdown": "# Article",
                "metadata": {
                    "sourceURL": requested + "/",
                    "url": "https://final.example/article",
                    "statusCode": 200,
                },
            },
        },
    )
    assert adapter.scrape_result(valid, requested) == {"url": requested, "content": "# Article"}
    for api_status, target_status in ((201, 203), (200, 304)):
        response = httpx.Response(
            api_status,
            json={
                "success": True,
                "data": {
                    "markdown": "# Cached or redirected article",
                    "metadata": {"sourceURL": requested, "statusCode": target_status},
                },
            },
        )
        assert adapter.scrape_result(response, requested) == {
            "url": requested,
            "content": "# Cached or redirected article",
        }

    for data in (
        {"markdown": "# Article", "metadata": {"sourceURL": requested, "statusCode": 404}},
        {"markdown": "   ", "metadata": {"sourceURL": requested, "statusCode": 200}},
        {"markdown": "", "metadata": {"sourceURL": requested, "statusCode": 304}},
        {"markdown": "# Wrong page", "metadata": {"sourceURL": "https://other.example/", "statusCode": 200}},
        {"markdown": "# Failed", "metadata": {"sourceURL": requested, "statusCode": 200, "error": "blocked"}},
        {
            "markdown": "# PDF text",
            "rawBase64": "AA==",
            "metadata": {"sourceURL": requested, "statusCode": 200, "contentType": "application/pdf"},
        },
    ):
        response = httpx.Response(200, json={"success": True, "data": data})
        assert adapter.scrape_result(response, requested) is None, data

    assert adapter.scrape_result(httpx.Response(200, json={"success": False}), requested) is None
    assert adapter.scrape_result(httpx.Response(200, json={"success": True, "data": {"markdown": "# no metadata"}}), requested) is None


def demo():
    for name, check in sorted(globals().items()):
        if name.startswith("test_"):
            check()
            print(f"ok {name}")
    print("all checks passed")


if __name__ == "__main__":
    demo()
