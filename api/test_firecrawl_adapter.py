"""Firecrawl v2 adapter payload and normalization checks for #46."""

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

    for data in (
        {"markdown": "# Article", "metadata": {"sourceURL": requested, "statusCode": 404}},
        {"markdown": "   ", "metadata": {"sourceURL": requested, "statusCode": 200}},
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
