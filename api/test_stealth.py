"""Checks for the server-owned crawl defaults. Run: python api/test_stealth.py"""

import app


def test_fills_every_missing_key():
    payload = app.with_stealth_defaults({"urls": ["https://example.com"]})
    browser, crawler = payload["browser_config"], payload["crawler_config"]

    assert browser["enable_stealth"] is True
    assert "HeadlessChrome" not in browser["user_agent"], "headless token is the giveaway this avoids"
    assert browser["user_agent"].endswith(f"Chrome/{app.CRAWL4AI_CHROMIUM_VERSION} Safari/537.36"), browser["user_agent"]
    assert crawler["remove_overlay_elements"] is True
    assert crawler["locale"] == "en-US" and crawler["timezone_id"] == "America/New_York"
    assert payload["urls"] == ["https://example.com"], "unrelated payload keys are untouched"


def test_caller_values_win_and_are_not_pruned():
    """#12 adds a pruning markdown_generator and excluded_tags; neither may be dropped."""
    payload = app.with_stealth_defaults({
        "urls": ["https://example.com"],
        "crawler_config": {
            "excluded_tags": ["nav", "footer"],
            "markdown_generator": {"type": "pruning"},
            "max_retries": 0,
            "locale": "de-DE",
        },
        "browser_config": {"headless": False},
    })

    assert payload["crawler_config"]["excluded_tags"] == ["nav", "footer"]
    assert payload["crawler_config"]["markdown_generator"] == {"type": "pruning"}
    assert payload["crawler_config"]["max_retries"] == 0
    assert payload["crawler_config"]["locale"] == "de-DE", "caller override survives"
    assert payload["browser_config"]["headless"] is False
    # Keys the caller left out still get the default.
    assert payload["crawler_config"]["remove_overlay_elements"] is True
    assert payload["browser_config"]["enable_stealth"] is True


def test_never_sends_keys_the_untrusted_gate_rejects():
    """`magic` and `override_navigator` 400 the whole crawl on crawl4ai 0.9.4."""
    assert "magic" not in app.STEALTH_CRAWLER_CONFIG
    assert "override_navigator" not in app.STEALTH_CRAWLER_CONFIG
    payload = app.with_stealth_defaults({"urls": ["https://example.com"]})
    assert not {"magic", "override_navigator"} & set(payload["crawler_config"])


def test_defaults_are_not_mutated_between_calls():
    app.with_stealth_defaults({"urls": ["https://a.example"], "crawler_config": {"max_retries": 2}})
    assert app.STEALTH_CRAWLER_CONFIG["max_retries"] == 1, "module defaults must not be shared into payloads"


if __name__ == "__main__":
    for name, test in sorted(globals().items()):
        if name.startswith("test_"):
            test()
            print(f"ok {name}")
    print("all checks passed")
