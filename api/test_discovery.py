"""Mocked discovery pagination, outage, and provider-health checks for #44."""

import asyncio
import logging
import time
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime

import httpx

import app


def run(coro):
    return asyncio.run(coro)


def client(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def result(index):
    return {"url": f"https://{index}.example/", "title": str(index), "description": "snippet"}


def brave_response(items, more=True):
    return httpx.Response(200, json={"query": {"more_results_available": more}, "web": {"results": items}})


def test_brave_uses_constant_page_size_page_index_and_trims():
    calls = []

    def handler(request):
        calls.append((int(request.url.params["count"]), int(request.url.params["offset"])))
        offset = int(request.url.params["offset"])
        return brave_response([result(i) for i in range(offset * 20, (offset + 1) * 20)])

    pool = app.BraveKeyPool(("test-key",))
    deadline = time.monotonic() + 10
    results = run(pool._search_with_key(client(handler), 0, "q", 50, deadline))

    assert calls == [(20, 0), (20, 1), (20, 2)], calls
    assert len(results) == 50, len(results)
    assert len({item["url"] for item in results}) == 50


def test_brave_deduplicates_overlapping_pages_and_honors_no_more_results():
    calls = []
    pages = [
        ([result(i) for i in range(20)], True),
        ([result(i) for i in range(10, 30)], True),
        ([result(i) for i in range(20, 40)], False),
    ]

    def handler(request):
        offset = int(request.url.params["offset"])
        calls.append(offset)
        items, more = pages[offset]
        return brave_response(items, more)

    pool = app.BraveKeyPool(("test-key",))
    deadline = time.monotonic() + 10
    results = run(pool._search_with_key(client(handler), 0, "q", 40, deadline))

    assert calls == [0, 1, 2], calls
    assert len(results) == 40, len(results)
    assert len({item["url"] for item in results}) == 40


def test_brave_page_cap_stops_no_progress_pages():
    calls = []
    page = [result(i) for i in range(20)]

    def handler(request):
        calls.append(int(request.url.params["offset"]))
        return brave_response(page)

    pool = app.BraveKeyPool(("test-key",))
    deadline = time.monotonic() + 10
    results = run(pool._search_with_key(client(handler), 0, "q", 50, deadline))

    assert calls == list(range(app.BRAVE_PAGE_LIMIT)), calls
    assert len(results) == 20


def test_retry_after_accepts_seconds_and_http_date():
    response = httpx.Response(429, headers={"Retry-After": "12"})
    assert app._retry_after(response, 300) == 12

    retry_at = datetime.now(timezone.utc) + timedelta(seconds=5)
    response = httpx.Response(429, headers={"Retry-After": format_datetime(retry_at, usegmt=True)})
    assert 0 <= app._retry_after(response, 300) <= 5

    response = httpx.Response(429, headers={"Retry-After": "NaN"})
    assert app._retry_after(response, 300) == 300


def test_discovery_outage_returns_retryable_error_and_does_not_retry_searxng():
    paths = []
    saved_pool = app.BRAVE_POOL
    app.BRAVE_POOL = app.BraveKeyPool(())

    def handler(request):
        paths.append(request.url.path)
        return httpx.Response(503)

    try:
        try:
            run(app.discover_candidates(client(handler), "q", 5))
        except app.AppError as exc:
            assert exc.status_code == 503
            assert exc.code == "upstream_unavailable"
            assert exc.retryable is True
            assert exc.retry_after == app.OUTAGE_RETRY_AFTER
            assert exc.details["providers"]["searxng"] == "searxng_error"
        else:
            raise AssertionError("all failed providers returned an empty success")
        assert paths == ["/search"], paths
    finally:
        app.BRAVE_POOL = saved_pool
        app.health_cache_reset()


def test_discovery_deadline_returns_retryable_error_without_restarting_fallback():
    saved_pool = app.BRAVE_POOL
    saved_timeout = app.DISCOVERY_TIMEOUT
    calls = []

    class SlowPool:
        _last_retry_after = None

        def __bool__(self):
            return True

        async def search(self, client, query, limit, deadline):
            remaining = deadline - asyncio.get_running_loop().time()
            await asyncio.wait_for(asyncio.sleep(1), timeout=remaining)

    app.BRAVE_POOL = SlowPool()
    app.DISCOVERY_TIMEOUT = 0.01

    def handler(request):
        calls.append(request.url.path)
        return httpx.Response(503)

    try:
        try:
            run(app.discover_candidates(client(handler), "q", 5))
        except app.AppError as exc:
            assert exc.retryable is True
            assert exc.code == "upstream_unavailable"
            assert exc.details["providers"] == {"brave": "timeout", "searxng": "deadline_exceeded"}
        else:
            raise AssertionError("discovery deadline became an empty success")
        assert calls == [], calls
    finally:
        app.BRAVE_POOL = saved_pool
        app.DISCOVERY_TIMEOUT = saved_timeout
        app.health_cache_reset()


def test_retry_after_from_brave_is_top_level_on_exhausted_discovery():
    saved_pool = app.BRAVE_POOL
    saved_url = app.BRAVE_SEARCH_URL
    app.BRAVE_POOL = app.BraveKeyPool(("test-key",))
    app.BRAVE_SEARCH_URL = "https://brave.example/search"

    def handler(request):
        if request.url.host == "brave.example":
            return httpx.Response(429, headers={"Retry-After": "12"})
        return httpx.Response(503)

    try:
        try:
            run(app.discover_candidates(client(handler), "q", 5))
        except app.AppError as exc:
            assert exc.retryable is True
            assert exc.retry_after == 12, exc.retry_after
            error = app.error_response(
                exc.status_code, exc.code, exc.message, exc.details,
                retryable=exc.retryable, retry_after=exc.retry_after,
            ).body
            import json

            assert json.loads(error)["error"]["retry_after"] == 12
        else:
            raise AssertionError("exhausted providers returned an empty success")
    finally:
        app.BRAVE_POOL = saved_pool
        app.BRAVE_SEARCH_URL = saved_url
        app.health_cache_reset()


def test_successful_empty_brave_result_stays_success_when_searxng_fails():
    saved_pool = app.BRAVE_POOL
    saved_url = app.BRAVE_SEARCH_URL
    app.BRAVE_POOL = app.BraveKeyPool(("test-key",))
    app.BRAVE_SEARCH_URL = "https://brave.example/search"
    calls = []

    def handler(request):
        calls.append(request.url.host)
        if request.url.host == "brave.example":
            return brave_response([], False)
        return httpx.Response(503)

    try:
        assert run(app.discover_candidates(client(handler), "q", 5)) == []
        assert calls.count("brave.example") == 1, calls
        assert calls.count("searxng") == 1, calls
    finally:
        app.BRAVE_POOL = saved_pool
        app.BRAVE_SEARCH_URL = saved_url
        app.health_cache_reset()


def test_brave_partial_candidates_survive_searxng_failure():
    saved_pool = app.BRAVE_POOL
    saved_url = app.BRAVE_SEARCH_URL
    saved_allow = app.is_allowed_public_url
    app.BRAVE_POOL = app.BraveKeyPool(("test-key",))
    app.BRAVE_SEARCH_URL = "https://brave.example/search"
    app.is_allowed_public_url = lambda _: asyncio.sleep(0, result=True)

    def handler(request):
        if request.url.host == "brave.example":
            return brave_response([result(1)], False)
        return httpx.Response(503)

    try:
        results = run(app.discover_candidates(client(handler), "q", 5))
        assert [item["url"] for item in results] == ["https://1.example/"]
        assert app._provider_health("brave", True) == "ok"
        assert app._provider_health("searxng", True) == "unavailable"
    finally:
        app.BRAVE_POOL = saved_pool
        app.BRAVE_SEARCH_URL = saved_url
        app.is_allowed_public_url = saved_allow
        app.health_cache_reset()


def test_health_accepts_healthy_brave_when_searxng_is_down_without_searching():
    saved_pool = app.BRAVE_POOL
    app.BRAVE_POOL = app.BraveKeyPool(("test-key",))
    app.health_cache_reset()
    app._record_provider_health("brave", "ok")
    app._record_provider_health("searxng", "unavailable")
    paths = []

    def handler(request):
        paths.append(request.url.path)
        return httpx.Response(200)

    saved_get_client = app.get_client
    app.get_client = lambda: asyncio.sleep(0, result=client(handler))
    try:
        health = run(app.health())
        assert health.data.checks["brave"] == "ok", health.data.checks
        assert health.data.checks["searxng"] == "unavailable", health.data.checks
        assert "/search" not in paths, paths
        assert "/res/v1/web/search" not in paths, paths
    finally:
        app.get_client = saved_get_client
        app.BRAVE_POOL = saved_pool
        app.health_cache_reset()


def test_health_is_red_when_all_configured_discovery_providers_are_unavailable():
    saved_pool = app.BRAVE_POOL
    app.BRAVE_POOL = app.BraveKeyPool(("test-key",))
    app.health_cache_reset()
    app._record_provider_health("brave", "unavailable")
    app._record_provider_health("searxng", "unavailable")
    saved_get_client = app.get_client
    app.get_client = lambda: asyncio.sleep(0, result=client(lambda _: httpx.Response(200)))
    try:
        try:
            run(app.health())
        except app.AppError as exc:
            assert exc.status_code == 503
            assert exc.details["checks"]["brave"] == "unavailable"
            assert exc.details["checks"]["searxng"] == "unavailable"
        else:
            raise AssertionError("health passed with all discovery providers unavailable")
    finally:
        app.get_client = saved_get_client
        app.BRAVE_POOL = saved_pool
        app.health_cache_reset()


def test_health_does_not_upgrade_stale_provider_failures_from_service_readiness():
    saved_pool = app.BRAVE_POOL
    app.BRAVE_POOL = app.BraveKeyPool(("test-key",))
    app.health_cache_reset()
    app._record_provider_health("brave", "unavailable")
    app._record_provider_health("searxng", "unavailable")
    for result in app._PROVIDER_HEALTH.values():
        result["at"] = time.monotonic() - app.HEALTH_CACHE_TTL - 1
    saved_get_client = app.get_client
    app.get_client = lambda: asyncio.sleep(0, result=client(lambda _: httpx.Response(200)))
    try:
        try:
            run(app.health())
        except app.AppError as exc:
            checks = exc.details["checks"]
            assert checks["brave"] == "stale:unavailable", checks
            assert checks["searxng"] == "stale:unavailable", checks
            assert checks["searxng_service"] == "not_probed", checks
        else:
            raise AssertionError("stale failed provider outcomes became a healthy search verdict")
    finally:
        app.get_client = saved_get_client
        app.BRAVE_POOL = saved_pool
        app.health_cache_reset()


def test_get_post_search_auth_and_success_envelopes_are_unchanged():
    saved_pool = app.BRAVE_POOL
    saved_key = app.WEBSEARCH_API_KEY
    saved_get_client = app.get_client
    app.BRAVE_POOL = app.BraveKeyPool(())
    app.WEBSEARCH_API_KEY = "test-api-key"

    state = {"available": True}

    def handler(_):
        if state["available"]:
            return httpx.Response(200, json={"results": [], "unresponsive_engines": []})
        return httpx.Response(503)

    app.get_client = lambda: asyncio.sleep(0, result=client(handler))

    async def exercise():
        transport = httpx.ASGITransport(app=app.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as api:
            denied = await api.post("/search", json={"query": "q", "depth": "basic"})
            post = await api.post(
                "/search", headers={"X-API-Key": "test-api-key"}, json={"query": "q", "depth": "basic"}
            )
            get = await api.get(
                "/search", headers={"X-API-Key": "test-api-key"}, params={"q": "q", "depth": "basic"}
            )
            state["available"] = False
            post_error = await api.post(
                "/search", headers={"X-API-Key": "test-api-key"}, json={"query": "q", "depth": "basic"}
            )
            get_error = await api.get(
                "/search", headers={"X-API-Key": "test-api-key"}, params={"q": "q", "depth": "basic"}
            )
        return denied, post, get, post_error, get_error

    try:
        denied, post, get, post_error, get_error = run(exercise())
        assert denied.status_code == 401, denied.text
        assert denied.json()["error"]["code"] == "unauthorized", denied.text
        for response in (post, get):
            assert response.status_code == 200, response.text
            assert response.json()["ok"] is True, response.text
        for response in (post_error, get_error):
            assert response.status_code == 503, response.text
            error = response.json()["error"]
            assert error["code"] == "upstream_unavailable", error
            assert error["retryable"] is True, error
            assert error["retry_after"] == app.OUTAGE_RETRY_AFTER, error
    finally:
        app.get_client = saved_get_client
        app.BRAVE_POOL = saved_pool
        app.WEBSEARCH_API_KEY = saved_key
        app.health_cache_reset()


def test_brave_error_logs_do_not_expose_key_or_response_body():
    secret_key = "sentinel-key-must-not-log"
    secret_body = "sentinel-response-must-not-log"
    messages = []

    class Capture(logging.Handler):
        def emit(self, record):
            messages.append(record.getMessage())

    handler = Capture()
    app.logger.addHandler(handler)
    try:
        pool = app.BraveKeyPool((secret_key,))
        run(pool.search(
            client(lambda _: httpx.Response(403, text=secret_body)),
            "q",
            5,
            time.monotonic() + 10,
        ))
        engines = app._engine_names()
        try:
            run(app.search_searxng(
                client(lambda _: httpx.Response(200, json={
                    "results": [],
                    "unresponsive_engines": [[engine, secret_body] for engine in engines],
                })),
                "q",
                5,
            ))
        except app.AppError:
            pass
    finally:
        app.logger.removeHandler(handler)

    output = "\n".join(messages)
    assert secret_key not in output, output
    assert secret_body not in output, output


def demo():
    for name, check in sorted(globals().items()):
        if name.startswith("test_"):
            check()
            print(f"ok {name}")
    print("all checks passed")


if __name__ == "__main__":
    demo()
