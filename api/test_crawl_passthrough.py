"""Self-check for the /crawl passthrough allowlist (#10, #17).

Run: python api/test_crawl_passthrough.py
Fails loudly if a forbidden shape ever becomes forwardable again.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pydantic import ValidationError

from app import AppError, CrawlRequest, build_crawl_payload, validate_crawl_passthrough

ALLOWED = [
    ("crawler_config", {"wait_until": "networkidle"}),
    ("crawler_config", {"css_selector": "main"}),
    ("crawler_config", {"page_timeout": 30000}),
    ("crawler_config", {"screenshot": False}),
    ("crawler_config", {"excluded_tags": ["script", "style"]}),
    ("crawler_config", {"exclude_domains": ["example.com"]}),
    ("crawler_config", {"cache_mode": "BYPASS"}),
    ("crawler_config", {"wait_for": "css:main article"}),
    ("crawler_config", {"max_retries": 2}),
    ("crawl_options", {"screenshot": True}),
    ("crawl_options", {"css_selector": "h1"}),
    ("crawl_options", {"word_count_threshold": 10}),
]

# (field, config) -> the request must be refused with 422 validation_error.
FORBIDDEN = [
    ("crawler_config", {"check_robots_txt": True}),
    ("crawler_config", {"link_preview_config": {"url": "http://169.254.169.254/"}}),
    ("crawler_config", {"x": {"type": "dict", "value": {}}}),
    ("crawler_config", {"css_selector": {"type": "dict", "value": "h1"}}),
    ("crawler_config", {"excluded_tags": [{"type": "dict", "value": {}}]}),
    ("crawler_config", {"llm_config": {}}),
    ("crawler_config", {"js_code": "fetch('/')"}),
    ("crawler_config", {"user_data_dir": "/tmp/x"}),
    ("crawler_config", {"wait_for": "js:() => fetch('http://169.254.169.254/')"}),
    ("crawler_config", {"wait_for": "() => true"}),
    ("crawler_config", {"wait_for": "fetch('http://169.254.169.254/')"}),
    ("crawler_config", {"max_retries": 3}),
    ("crawler_config", {"cache_mode": "nope"}),
    ("crawl_options", {"llm_config": {}}),
    ("crawl_options", {"urls": ["http://169.254.169.254/"]}),
    ("crawl_options", {"browser_type": "chromium"}),
    ("crawl_options", {"user_data_dir": "/tmp/x"}),
    ("crawl_options", {"wait_for": "js:1"}),
]

# Removed in #17: the field is gone, so pydantic must refuse it outright rather
# than ignore it.
REMOVED_FIELDS = [
    ("extraction_config", {"proxy_config": {"server": "http://evil"}}),
    ("extraction_config", {}),
    ("crawl_option", {"screenshot": True}),
]


def request(field: str, config: dict) -> CrawlRequest:
    return CrawlRequest(url="https://example.com", **{field: config})


def rejects(field: str, config: dict) -> dict:
    try:
        validate_crawl_passthrough(request(field, config))
    except AppError as exc:
        assert (exc.status_code, exc.code) == (422, "validation_error"), (exc.status_code, exc.code)
        detail = exc.details[0]
        assert detail["loc"][0] == field, detail
        assert detail["type"] == "value_error", detail
        return detail
    raise AssertionError(f"expected 422 for {field}: {config}")


def payload(**kwargs) -> dict:
    return build_crawl_payload(CrawlRequest(url="https://example.com", **kwargs))


def demo() -> None:
    for field, config in ALLOWED:
        validate_crawl_passthrough(request(field, config))

    for field, config in FORBIDDEN:
        rejects(field, config)

    for field, config in REMOVED_FIELDS:
        try:
            request(field, config)
        except ValidationError as exc:
            assert exc.errors()[0]["type"] == "extra_forbidden", exc.errors()
        else:
            raise AssertionError(f"expected the request model to refuse {field}")

    assert rejects("crawler_config", {"wait_for": "js:1"})["loc"] == ["crawler_config", "wait_for"]
    assert rejects("crawler_config", {"max_retries": 3})["loc"] == ["crawler_config", "max_retries"]
    assert rejects("crawler_config", {"browser_config": {"headless": True}})["loc"][1] == "browser_config"
    assert "allowed keys:" in rejects("crawler_config", {"nope": 1})["msg"]
    assert rejects("browser_config", {"headless": True})["loc"] == ["browser_config"]

    try:
        validate_crawl_passthrough(CrawlRequest(url="https://example.com", cache_mode="nope"))
    except AppError as exc:
        assert exc.details[0]["loc"] == ["cache_mode"], exc.details
    else:
        raise AssertionError("expected 422 for the top-level cache_mode field")
    validate_crawl_passthrough(CrawlRequest(url="https://example.com", cache_mode="bypass"))

    # #17: crawl_options and cache_mode land inside crawler_config, and an
    # explicit crawler_config value wins. cache_mode has to go out typed, or
    # Crawl4AI's CacheMode enum silently ignores it.
    merged = payload(crawl_options={"screenshot": True, "css_selector": "h1"}, crawler_config={"css_selector": "main"})
    assert merged == {
        "urls": ["https://example.com"],
        "crawler_config": {"screenshot": True, "css_selector": "main"},
    }, merged

    assert payload(cache_mode="bypass")["crawler_config"]["cache_mode"] == {"type": "CacheMode", "params": "bypass"}
    assert payload(cache_mode="BYPASS")["crawler_config"]["cache_mode"] == {"type": "CacheMode", "params": "bypass"}
    assert payload(crawler_config={"cache_mode": "enabled"})["crawler_config"]["cache_mode"] == {
        "type": "CacheMode",
        "params": "enabled",
    }
    assert payload(cache_mode="read_only", crawler_config={"cache_mode": "disabled"})["crawler_config"]["cache_mode"] == {
        "type": "CacheMode",
        "params": "disabled",
    }

    assert payload() == {"urls": ["https://example.com"]}, payload()
    assert CrawlRequest(url="https://example.com").browser_config == {}
    print("crawl passthrough allowlist: ok")


if __name__ == "__main__":
    demo()
