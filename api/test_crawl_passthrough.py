"""Self-check for the /crawl passthrough allowlist (#10).

Run: python api/test_crawl_passthrough.py
Fails loudly if a forbidden shape ever becomes forwardable again.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import AppError, CrawlRequest, validate_crawl_passthrough

ALLOWED = {
    "crawler_config": [
        {"wait_until": "networkidle"},
        {"css_selector": "main"},
        {"page_timeout": 30000},
        {"screenshot": False},
        {"excluded_tags": ["script", "style"]},
        {"exclude_domains": ["example.com"]},
        {"cache_mode": "BYPASS"},
        {"wait_for": "css:main article"},
        {"max_retries": 2},
    ],
    "crawl_options": [{"screenshot": True}, {"css_selector": "h1"}],
    "extraction_config": [{"word_count_threshold": 10}],
}

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
    ("crawl_options", {"llm_config": {}}),
    ("crawl_options", {"urls": ["http://169.254.169.254/"]}),
    ("crawl_options", {"browser_type": "chromium"}),
    ("crawl_options", {"user_data_dir": "/tmp/x"}),
    ("extraction_config", {"proxy_config": {"server": "http://evil"}}),
    ("extraction_config", {"wait_for": "js:1"}),
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


def demo() -> None:
    for field, configs in ALLOWED.items():
        for config in configs:
            validate_crawl_passthrough(request(field, config))

    for field, config in FORBIDDEN:
        rejects(field, config)

    assert rejects("crawler_config", {"wait_for": "js:1"})["loc"] == ["crawler_config", "wait_for"]
    assert rejects("crawler_config", {"max_retries": 3})["loc"] == ["crawler_config", "max_retries"]
    assert rejects("crawler_config", {"browser_config": {"headless": True}})["loc"][1] == "browser_config"
    assert rejects("crawler_config", {"browser_type": "chromium"})["loc"][1] == "browser_type"
    assert "allowed keys:" in rejects("crawler_config", {"nope": 1})["msg"]
    assert rejects("browser_config", {"headless": True})["loc"] == ["browser_config"]

    assert CrawlRequest(url="https://example.com").browser_config == {}
    print("crawl passthrough allowlist: ok")


if __name__ == "__main__":
    demo()
