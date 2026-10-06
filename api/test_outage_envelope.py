"""Checks for the outage contract in #986: a dead engine pool is an error, not empty.

A full search-backend outage used to answer `ok: true, results: []`, which is
indistinguishable from "the web had nothing on this topic". An agent that trusted it
gave up instead of retrying, and `/health` echoed its configured URLs, so it stayed
green through the same outage. Run: python api/test_outage_envelope.py"""

import asyncio

import httpx

import app

DEAD_ENGINES = ["bing", "brave", "duckduckgo", "google cse"]


def client(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def use(handler):
    """Point every outbound call at a mock and clear the health verdict cache.

    The cache reset goes through getattr on purpose: against a pre-fix app.py the
    helper does not exist, and a test that dies on a missing test helper proves
    nothing. These checks must fail on their own behavioural assertion instead.
    """
    saved = app.get_client
    app.get_client = lambda: asyncio.sleep(0, result=client(handler))
    getattr(app, "health_cache_reset", lambda: None)()

    def restore():
        app.get_client = saved
        getattr(app, "health_cache_reset", lambda: None)()

    return restore


def dead_pool(request: httpx.Request) -> httpx.Response:
    """Every configured engine dead: searxng answers, but with nothing behind it."""
    return httpx.Response(200, json={"results": [], "unresponsive_engines": DEAD_ENGINES})


def empty_pool(request: httpx.Request) -> httpx.Response:
    """Engines healthy, genuinely nothing on the topic. Not an outage."""
    return httpx.Response(200, json={"results": [], "unresponsive_engines": []})


def partial_pool(request: httpx.Request) -> httpx.Response:
    """One engine dead, the rest answered with real rows."""
    return httpx.Response(
        200,
        json={
            "results": [{"url": "https://a.example/", "title": "A", "snippet": "s"}],
            "unresponsive_engines": ["bing"],
        },
    )


def answered_with_results(request: httpx.Request) -> httpx.Response:
    if httpx.URL(str(request.url)).path == "/search":
        return httpx.Response(
            200,
            json={
                "results": [{"url": "https://a.example/", "title": "A", "snippet": "s"}],
                "unresponsive_engines": [],
            },
        )
    return httpx.Response(200, json={})


def run(coro):
    return asyncio.run(coro)


def raises_app_error(coro):
    try:
        run(coro)
    except app.AppError as exc:
        return exc
    raise AssertionError("expected an AppError; the outage was reported as a success")


# ---------------------------------------------------------------- outage grading


def test_all_engines_dead_is_an_error_envelope():
    exc = raises_app_error(app.search_searxng(client(dead_pool), "anything", 5))
    assert exc.status_code == 503, exc.status_code
    assert exc.code == "upstream_unavailable", exc.code
    assert exc.retryable is True, "retryable belongs on the envelope, not only in details"


def test_retryable_is_top_level_on_the_wire():
    # The canonical envelope puts `retryable` beside `code` (sparkfn/pc-tools CLAUDE.md,
    # "Error envelope"). An agent branches on that field to retry-or-give-up, so it
    # cannot live inside `details` where nothing reads it.
    exc = raises_app_error(app.search_searxng(client(dead_pool), "anything", 5))
    body = app.error_response(exc.status_code, exc.code, exc.message, exc.details, retryable=exc.retryable).body
    import json

    error = json.loads(body)["error"]
    assert error["retryable"] is True, error
    assert "retryable" not in (error.get("details") or {}), "retryable must not be nested in details"


def test_a_live_but_empty_engine_is_not_an_outage():
    # One engine dead, the other engines alive and simply had no matches. searxng reports
    # only the FAILED engines, so an empty result here still means "a pool answered".
    # Grading this 503 turns every partial outage into a fake retry loop.
    def half_dead(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": [], "unresponsive_engines": ["bing"]})

    assert run(app.search_searxng(client(half_dead), "anything", 5)) == []


def test_genuinely_empty_results_are_not_an_outage():
    # Engines answered; the web simply had nothing. This must stay a success, or every
    # rare query with no hits becomes a fake retry loop.
    assert run(app.search_searxng(client(empty_pool), "anything", 5)) == []


def test_partial_outage_still_returns_results():
    # One dead engine must not fail a search the other engines answered.
    exc = None
    try:
        run(app.search_searxng(client(partial_pool), "anything", 5))
    except app.AppError as caught:
        exc = caught
    assert exc is None, f"a partial outage was escalated: {exc}"


def test_results_all_filtered_out_is_not_an_outage():
    # shape_candidates drops non-public URLs. A healthy engine whose rows were all
    # filtered out has still answered, so this is an empty result, never a 503: grading
    # it as an outage would retry a working backend forever.
    def filtered(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "results": [{"url": "http://localhost:8080/admin", "title": "x", "snippet": "s"}],
                "unresponsive_engines": DEAD_ENGINES,
            },
        )

    try:
        run(app.search_searxng(client(filtered), "anything", 5))
    except app.AppError as exc:
        raise AssertionError(f"filtered-but-answered was graded as an outage: {exc}")


def test_search_endpoint_returns_the_error_envelope():
    # Through the real request path, not the helper: the agent sees this envelope.
    saved_pool = app.BRAVE_POOL
    restore = use(dead_pool)
    app.BRAVE_POOL = None  # force the searxng-only fallback
    try:
        exc = raises_app_error(app.search(app.SearchRequest(query="anything", candidates=5, max_results=3)))
        assert exc.code == "upstream_unavailable", exc.code
        body = app.error_response(exc.status_code, exc.code, exc.message, exc.details, retryable=exc.retryable).body.decode()
        assert '"ok":false' in body or '"ok": false' in body, body
        assert "upstream_unavailable" in body, body
        assert '"retryable":true' in body or '"retryable": true' in body, body
    finally:
        app.BRAVE_POOL = saved_pool
        restore()


# ---------------------------------------------------------------- health


def test_health_fails_during_a_total_outage():
    # Health used to echo its configured URLs, so it stayed green while search was dead.
    restore = use(dead_pool)
    try:
        exc = raises_app_error(app.health())
        assert exc.status_code == 503, exc.status_code
        assert exc.code == "upstream_unavailable", exc.code
    finally:
        restore()


def test_health_actually_issues_a_query():
    # The probe must reach searxng's /search, not just read a config value.
    seen: list[str] = []

    def watcher(request: httpx.Request) -> httpx.Response:
        seen.append(httpx.URL(str(request.url)).path)
        return answered_with_results(request)

    restore = use(watcher)
    try:
        run(app.health())
        assert "/search" in seen, f"health never queried an engine; saw {seen}"
    finally:
        restore()


def test_health_is_green_when_engines_answer():
    restore = use(answered_with_results)
    try:
        envelope = run(app.health())
        assert envelope.ok is True
        assert envelope.data.checks["searxng"] == "ok", envelope.data.checks
    finally:
        restore()


def test_health_canary_is_bounded_by_health_timeout():
    # The canary runs through search_searxng, which hardcoded a 10s search timeout.
    # Health is polled, so it must be bounded by HEALTH_TIMEOUT instead — otherwise a
    # hung backend makes every poll block for the full search timeout. Asserted through
    # the signature so the failure is about the missing bound, not a missing test hook.
    import inspect

    params = inspect.signature(app.search_searxng).parameters
    assert "timeout" in params, f"search_searxng takes no timeout; health cannot be bounded: {params}"
    assert getattr(app, "HEALTH_TIMEOUT", None) is not None, "HEALTH_TIMEOUT is gone"
    assert app.HEALTH_TIMEOUT < app.SEARCH_TIMEOUT, (
        f"health canary must be bounded below the search timeout "
        f"(health={app.HEALTH_TIMEOUT}, search={app.SEARCH_TIMEOUT})"
    )


def test_search_still_gets_the_search_timeout():
    # Threading the timeout through must not quietly shorten a real search: the default
    # for a search caller stays SEARCH_TIMEOUT.
    import inspect

    default = inspect.signature(app.search_searxng).parameters["timeout"].default
    assert default == app.SEARCH_TIMEOUT, f"search default timeout drifted: {default}"


def test_health_is_cached_between_polls():
    # A poll loop must not become engine load of its own.
    calls: list[int] = []

    def counting(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return answered_with_results(request)

    restore = use(counting)
    try:
        run(app.health())
        after_first = sum(calls)
        run(app.health())
        run(app.health())
        assert sum(calls) == after_first, f"health re-queried on every poll ({sum(calls)} calls)"
    finally:
        restore()


def test_health_outage_body_reports_every_service():
    # The failing body must be as informative as the healthy one: an operator reading a
    # red health check should learn WHICH dependency broke, not just that one did.
    restore = use(dead_pool)
    try:
        exc = raises_app_error(app.health())
        checks = (exc.details or {}).get("checks")
        assert isinstance(checks, dict), exc.details
        assert set(checks) == {"searxng", "crawl4ai", "reranker"}, checks
        assert checks["searxng"].startswith("unavailable"), checks
    finally:
        restore()


def test_retry_after_is_top_level_on_the_wire():
    # retry_after belongs beside retryable, not buried in details: a retrying client
    # reads the wait without parsing prose out of a details blob.
    exc = raises_app_error(app.search_searxng(client(dead_pool), "anything", 5))
    import json

    error = json.loads(
        app.error_response(
            exc.status_code, exc.code, exc.message, exc.details,
            retryable=exc.retryable, retry_after=exc.retry_after, hint=exc.hint,
        ).body
    )["error"]
    assert error["retry_after"] == app.OUTAGE_RETRY_AFTER, error
    assert "retry_after" not in (error.get("details") or {}), "retry_after must not be nested in details"


# ------------------------------------------------------- zero configured engines
#
# With no engine configured, searxng returns nothing AND reports nothing dead, so the
# "every engine failed" test cannot fire. That made search answer ok:true with zero
# results and health report searxng GREEN: #986's silent-empty failure, reached a
# different way.


def _no_engines():
    return use(lambda request: httpx.Response(200, json={"results": [], "unresponsive_engines": []}))


def test_zero_configured_engines_is_an_error_not_an_empty_search():
    saved = app.SEARXNG_ENGINES
    app.SEARXNG_ENGINES = "   "
    restore = _no_engines()
    try:
        exc = raises_app_error(app.search_searxng(client(lambda r: httpx.Response(200, json={})), "anything", 5))
        assert exc.code == "upstream_unavailable", exc.code
        # A misconfiguration is not transient: retrying cannot fix it, and an agent that
        # believes it can will retry the same nothing forever.
        assert exc.retryable is False, exc.retryable
        # ...and the operator is told exactly which setting to change.
        assert "SEARXNG_ENGINES" in (exc.hint or ""), exc.hint
    finally:
        app.SEARXNG_ENGINES = saved
        restore()


def test_zero_configured_engines_fails_the_health_searxng_check():
    # Health must not report a green searxng while search is structurally unable to run.
    saved = app.SEARXNG_ENGINES
    app.SEARXNG_ENGINES = ""
    restore = _no_engines()
    try:
        exc = raises_app_error(app.health())
        assert exc.status_code == 503, exc.status_code
        checks = (exc.details or {}).get("checks")
        assert isinstance(checks, dict), exc.details
        assert checks["searxng"].startswith("unavailable"), checks
    finally:
        app.SEARXNG_ENGINES = saved
        restore()


def test_engines_configured_is_still_a_normal_empty_search():
    # The guard above must not fire when engines ARE configured but simply had no
    # matches — that is a real answer, not a fault.
    restore = use(empty_pool)
    try:
        assert run(app.search_searxng(client(empty_pool), "anything", 5)) == []
    finally:
        restore()


def demo():
    for name, check in sorted(globals().items()):
        if name.startswith("test_"):
            check()
            print(f"ok {name}")
    print("all checks passed")


if __name__ == "__main__":
    demo()