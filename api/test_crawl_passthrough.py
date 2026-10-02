"""Self-check for the /crawl passthrough allowlist (#10).

Run: python api/test_crawl_passthrough.py
Fails loudly if a forbidden shape ever becomes forwardable again.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import AppError, CrawlRequest, validate_crawl_passthrough

ALLOWED_KEYS = {
    "wait_until": "networkidle",
    "css_selector": "main",
    "page_timeout": 30000,
    "screenshot": False,
    "excluded_tags": ["script", "style"],
    "exclude_domains": ["example.com"],
    "cache_mode": "BYPASS",
}

FORBIDDEN_KEYS = {
    "crawler_config": {"check_robots_txt": True},
    "crawler_config.link_preview": {"link_preview_config": {"url": "http://169.254.169.254/"}},
    "crawler_config.typed_wrapper": {"x": {"type": "dict", "value": {}}},
    "crawler_config.typed_under_allowed": {"css_selector": {"type": "dict", "value": "h1"}},
    "crawler_config.typed_in_list": {"excluded_tags": [{"type": "dict", "value": {}}]},
    "crawler_config.llm_config": {"llm_config": {}},
    "crawler_config.js_code": {"js_code": "fetch('/')"},
    "crawler_config.browser_user_data_dir": {"user_data_dir": "/tmp/x"},
    "crawled_options.browser_prefix": {"browser_type": "chromium"},
    "extraction_config.proxy_config": {"proxy_config": {"server": "http://evil"}},
}


def rejects(request: CrawlRequest) -> dict:
    try:
        validate_crawl_passthrough(request)
    except AppError as exc:
        assert (exc.status_code, exc.code) == (422, "validation_error"), (exc.status_code, exc.code)
        return exc.details[0]
    raise AssertionError("expected 422 validation_error")


def demo() -> None:
    for key, value in ALLOWED_KEYS.items():
        validate_crawl_passthrough(CrawlRequest(url="https://example.com", crawler_config={key: value}))

    for label, config in FORBIDDEN_KEYS.items():
        detail = rejects(CrawlRequest(url="https://example.com", crawler_config=config))
        assert detail["loc"][0] == "crawler_config", (label, detail)
        assert detail["type"] == "value_error", (label, detail)

    assert rejects(CrawlRequest(url="https://example.com", browser_config={"headless": True}))["loc"] == ["browser_config"]
    assert rejects(CrawlRequest(url="https://example.com", crawl_options={"llm_config": {}}))["loc"] == [
        "crawl_options",
        "llm_config",
    ]
    assert rejects(CrawlRequest(url="https://example.com", crawl_options={"urls": ["http://x"]}))["loc"][1] == "urls"
    assert rejects(CrawlRequest(url="https://example.com", crawler_config={"nope": 1}))["loc"][1] == "nope"
    assert "allowed keys:" in rejects(CrawlRequest(url="https://example.com", crawler_config={"nope": 1}))["msg"]

    assert CrawlRequest(url="https://example.com").browser_config == {}
    print("crawl passthrough allowlist: ok")


if __name__ == "__main__":
    demo()
