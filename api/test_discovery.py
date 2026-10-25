"""Mocked discovery pagination, outage, and provider-health checks for #44."""

import asyncio
import logging
import tempfile
import time
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from pathlib import Path
from unittest.mock import patch

import httpx

import app
import firecrawl_controls as fc


def run(coro):
    return asyncio.run(coro)


def client(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def result(index):
    return {"url": f"https://{index}.example/", "title": str(index), "description": "snippet"}


def brave_response(items, more=True):
    return httpx.Response(200, json={"query": {"more_results_available": more}, "web": {"results": items}})


def use_url_check(check):
    saved = app.is_allowed_public_url
    app.is_allowed_public_url = check
    return lambda: setattr(app, "is_allowed_public_url", saved)


def allow_public_urls():
    async def allow(_):
        return True

    return use_url_check(allow)


class MockDiscoveryClient:
    def __init__(self, rows, brave_page=None):
        self.rows = rows
        self.brave_page = brave_page
        self.calls = []

    async def get(self, url, params=None, headers=None, timeout=None):
        if params and "offset" in params:
            offset = int(params["offset"])
            self.calls.append(offset)
            response = await self.brave_page(offset)
        else:
            self.calls.append("searxng")
            response = httpx.Response(200, json={"results": self.rows, "unresponsive_engines": []})
        return httpx.Response(
            response.status_code,
            headers=response.headers,
            content=response.content,
            request=httpx.Request("GET", url),
        )


def run_discovery_with(provider, http_client, url_check, timeout, limit=10):
    saved_pool = app.BRAVE_POOL
    saved_timeout = app.DISCOVERY_TIMEOUT
    saved_check = app.is_allowed_public_url
    app.BRAVE_POOL = app.BraveKeyPool(("test-key",)) if provider == "brave" else app.BraveKeyPool(())
    app.DISCOVERY_TIMEOUT = timeout
    app.is_allowed_public_url = url_check
    app.health_cache_reset()
    try:
        return run(app.discover_candidates(http_client, "q", limit))
    finally:
        app.BRAVE_POOL = saved_pool
        app.DISCOVERY_TIMEOUT = saved_timeout
        app.is_allowed_public_url = saved_check
        app.health_cache_reset()


def test_brave_uses_constant_page_size_page_index_and_trims():
    calls = []

    def handler(request):
        calls.append((int(request.url.params["count"]), int(request.url.params["offset"])))
        offset = int(request.url.params["offset"])
        return brave_response([result(i) for i in range(offset * 20, (offset + 1) * 20)])

    restore = allow_public_urls()
    try:
        pool = app.BraveKeyPool(("test-key",))
        deadline = time.monotonic() + 10
        results = run(pool._search_with_key(client(handler), 0, "q", 50, deadline))
        assert calls == [(20, 0), (20, 1), (20, 2)], calls
        assert len(results) == 50, len(results)
        assert len({item["url"] for item in results}) == 50
    finally:
        restore()


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

    restore = allow_public_urls()
    try:
        pool = app.BraveKeyPool(("test-key",))
        deadline = time.monotonic() + 10
        results = run(pool._search_with_key(client(handler), 0, "q", 40, deadline))
        assert calls == [0, 1, 2], calls
        assert len(results) == 40, len(results)
        assert len({item["url"] for item in results}) == 40
    finally:
        restore()


def test_brave_page_cap_stops_no_progress_pages():
    calls = []
    page = [result(i) for i in range(20)]

    def handler(request):
        calls.append(int(request.url.params["offset"]))
        return brave_response(page)

    restore = allow_public_urls()
    try:
        pool = app.BraveKeyPool(("test-key",))
        deadline = time.monotonic() + 10
        results = run(pool._search_with_key(client(handler), 0, "q", 50, deadline))
        assert calls == list(range(app.BRAVE_PAGE_LIMIT)), calls
        assert len(results) == 20
    finally:
        restore()


def test_retry_after_accepts_seconds_and_http_date():
    response = httpx.Response(429, headers={"Retry-After": "12"})
    assert app._retry_after(response, 300) == 12

    retry_at = datetime.now(timezone.utc) + timedelta(seconds=5)
    response = httpx.Response(429, headers={"Retry-After": format_datetime(retry_at, usegmt=True)})
    assert 0 <= app._retry_after(response, 300) <= 5

    response = httpx.Response(429, headers={"Retry-After": "NaN"})
    assert app._retry_after(response, 300) == 300


def test_concurrent_brave_calls_keep_retry_after_per_request():
    async def exercise():
        pool = app.BraveKeyPool(("test-key",))
        long_started = asyncio.Event()
        short_finished = asyncio.Event()
        long_finished = asyncio.Event()

        class ConcurrentClient:
            async def get(self, url, params, headers, timeout):
                if params["q"] == "long":
                    long_started.set()
                    await short_finished.wait()
                    return httpx.Response(429, headers={"Retry-After": "300"})
                await long_started.wait()
                return httpx.Response(429, headers={"Retry-After": "20"})

        client = ConcurrentClient()
        deadline = asyncio.get_running_loop().time() + 5

        async def long_request():
            result = await pool.search(client, "long", 5, deadline)
            long_finished.set()
            return result

        async def short_request():
            result = await pool.search(client, "short", 5, deadline)
            short_finished.set()
            await long_finished.wait()
            return result

        long, short = await asyncio.gather(long_request(), short_request())
        return long, short

    long, short = run(exercise())
    assert long == (None, 300), long
    assert short == (None, 20), short


def test_local_validation_timeout_does_not_cool_brave_key():
    saved_check = app.is_allowed_public_url
    pool = app.BraveKeyPool(("test-key",))

    async def check(url):
        if url == "https://stall.example/":
            await asyncio.sleep(1)
        return True

    class Client:
        def __init__(self):
            self.calls = 0

        async def get(self, url, **_kwargs):
            self.calls += 1
            rows = (
                [result(0), {"url": "https://stall.example/", "title": "stall"}]
                if self.calls == 1
                else []
            )
            return httpx.Response(
                200,
                json={"query": {"more_results_available": False}, "web": {"results": rows}},
                request=httpx.Request("GET", url),
            )

    app.is_allowed_public_url = check

    async def exercise():
        client = Client()
        loop = asyncio.get_running_loop()
        first = await pool.search(client, "q", 5, loop.time() + 0.05)
        cooled = pool._cooled_until[0]
        second = await pool.search(client, "next", 5, loop.time() + 5)
        return first, cooled, second, client.calls

    try:
        first, cooled, second, calls = run(exercise())
        assert [item["url"] for item in first[0]] == [result(0)["url"]], first
        assert first[1] is None, first
        assert cooled == 0.0, cooled
        assert second == ([], None), second
        assert calls == 2, calls
    finally:
        app.is_allowed_public_url = saved_check
        app.health_cache_reset()


def test_real_brave_rate_limit_cooldown_is_preserved():
    pool = app.BraveKeyPool(("test-key",))
    calls = []

    class RateLimitedClient:
        async def get(self, url, **_kwargs):
            calls.append(url)
            return httpx.Response(429, headers={"Retry-After": "20"}, request=httpx.Request("GET", url))

    async def exercise():
        deadline = asyncio.get_running_loop().time() + 5
        first = await pool.search(RateLimitedClient(), "q", 5, deadline)
        cooled_until = pool._cooled_until[0]
        second = await pool.search(RateLimitedClient(), "q", 5, deadline)
        return first, cooled_until, second

    first, cooled_until, second = run(exercise())
    assert first == (None, 20), first
    assert cooled_until > time.monotonic() + 19, cooled_until
    assert second == (None, None), second
    assert len(calls) == 1, calls


def test_shorter_concurrent_cooldown_never_shortens_key_cooldown():
    async def exercise():
        pool = app.BraveKeyPool(("test-key",))
        pool._cool(0, 300)
        long_cooldown = pool._cooled_until[0]
        pool._cool(0, 20)
        assert pool._cooled_until[0] >= long_cooldown

    run(exercise())


def test_firecrawl_web_search_is_first_and_discovery_only():
    saved_pool = app.BRAVE_POOL
    saved_request = app.firecrawl_request
    saved_url_check = app.is_allowed_public_url
    app.BRAVE_POOL = app.BraveKeyPool(())
    calls = []

    async def firecrawl(_client, endpoint, payload, *, timeout):
        calls.append((endpoint, payload))
        return httpx.Response(
            200,
            json={
                "success": True,
                "data": {"web": [{"url": "https://fc.example/", "title": "FC", "description": "snippet"}]},
            },
        )

    async def allow_public(_url):
        return True

    app.firecrawl_request = firecrawl
    app.is_allowed_public_url = allow_public
    app.health_cache_reset()
    try:
        candidates = run(app.discover_candidates(object(), "q", 1))
        assert [item["url"] for item in candidates] == ["https://fc.example/"]
        assert calls[0][0] == "/v2/search", calls
        assert calls[0][1]["sources"] == ["web"], calls
        assert "scrapeOptions" not in calls[0][1], calls
        assert len(calls) == 1, calls
    finally:
        app.BRAVE_POOL = saved_pool
        app.firecrawl_request = saved_request
        app.is_allowed_public_url = saved_url_check
        app.health_cache_reset()


def test_firecrawl_skips_oversized_query_without_truncation():
    saved_pool = app.BRAVE_POOL
    saved_request = app.firecrawl_request
    app.BRAVE_POOL = app.BraveKeyPool(())
    firecrawl_calls = []
    searx_queries = []

    async def firecrawl(*args, **kwargs):
        firecrawl_calls.append((args, kwargs))
        raise AssertionError("oversized query reached Firecrawl")

    class SeaxClient:
        async def get(self, url, params=None, headers=None, timeout=None):
            searx_queries.append(params["q"])
            return httpx.Response(
                200,
                json={"results": [], "unresponsive_engines": []},
                request=httpx.Request("GET", url),
            )

    app.firecrawl_request = firecrawl
    app.health_cache_reset()
    try:
        assert run(app.discover_candidates(SeaxClient(), "q" * 501, 1)) == []
        assert firecrawl_calls == [], firecrawl_calls
        assert searx_queries == ["q" * 501], len(searx_queries)
    finally:
        app.BRAVE_POOL = saved_pool
        app.firecrawl_request = saved_request
        app.health_cache_reset()


def test_unconfigured_firecrawl_falls_back_to_brave_before_searxng():
    saved_pool = app.BRAVE_POOL
    saved_url = app.BRAVE_SEARCH_URL
    saved_request = app.firecrawl_request
    saved_check = app.is_allowed_public_url
    app.BRAVE_POOL = app.BraveKeyPool(("test-key",))
    app.BRAVE_SEARCH_URL = "https://brave.example/search"
    calls = []

    async def firecrawl(*_args, **_kwargs):
        calls.append("firecrawl")
        raise app.FirecrawlUnavailable(None, "not_configured", None, False)

    async def allow_public(_url):
        return True

    app.firecrawl_request = firecrawl
    app.is_allowed_public_url = allow_public

    def handler(request):
        calls.append("brave")
        return brave_response([result(1)], False)

    try:
        results = run(app.discover_candidates(client(handler), "q", 1))
        assert [item["url"] for item in results] == ["https://1.example/"]
        assert calls == ["firecrawl", "brave"], calls
    finally:
        app.BRAVE_POOL = saved_pool
        app.BRAVE_SEARCH_URL = saved_url
        app.firecrawl_request = saved_request
        app.is_allowed_public_url = saved_check
        app.health_cache_reset()


def test_busy_or_slow_firecrawl_reserves_budget_for_brave_and_searxng():
    async def exercise(total, busy):
        now = 0.0
        calls = []
        real_sleep = asyncio.sleep
        with tempfile.TemporaryDirectory() as tmp:
            controls = fc.FirecrawlControls(
                (fc._Budget("team", ("test-key", "unused-key")),),
                state_path=Path(tmp) / "state.json", lock_path=Path(tmp) / "lock",
            )
            held_lock = None

            async def sleep(delay):
                nonlocal now, held_lock
                now += delay
                if not busy and now >= total / 4 and held_lock is not None:
                    controls._release_lock(held_lock)
                    held_lock = None
                await real_sleep(0)

            async def handler(request):
                nonlocal now
                if request.method == "POST":
                    calls.append("firecrawl")
                    now += request.extensions["timeout"]["read"]
                    raise httpx.ReadTimeout("mock primary stall", request=request)
                provider = "brave" if "offset" in request.url.params else "searxng"
                calls.append(provider)
                assert abs(now - total / 2) < 1e-8, (total, busy, now)
                if provider == "brave":
                    return brave_response([result(1)], False)
                return httpx.Response(200, json={"results": [result(2)], "unresponsive_engines": []})

            with patch.object(asyncio, "get_running_loop") as get_loop, patch.object(
                asyncio, "sleep", sleep
            ), patch.object(app, "firecrawl_request", controls.request), patch.object(
                app, "DISCOVERY_TIMEOUT", total
            ), patch.object(app, "BRAVE_POOL", app.BraveKeyPool(("test-key",))), patch.object(
                app, "is_allowed_public_url", lambda _: real_sleep(0, result=True)
            ):
                get_loop.return_value.time.side_effect = lambda: now
                held_lock = await controls._acquire_lock(total)
                try:
                    async with client(handler) as http_client:
                        rows = await app.discover_candidates(http_client, "q", 2)
                    assert [row["url"] for row in rows] == [result(1)["url"], result(2)["url"]]
                    assert calls == (["brave", "searxng"] if busy else ["firecrawl", "brave", "searxng"]), calls
                finally:
                    if held_lock is not None:
                        controls._release_lock(held_lock)

    try:
        for total in (20.0, 6.0):
            for busy in (True, False):
                run(exercise(total, busy))
    finally:
        app.health_cache_reset()


def test_firecrawl_validation_uses_one_subdeadline_and_preserves_answer_semantics():
    async def exercise(mode):
        now = 0.0
        visited = []
        legacy = []
        validation_timeouts = []
        prefix = result("prefix")
        stalled = result("stalled")
        tail = result("tail")

        async def firecrawl(*_args, **_kwargs):
            nonlocal now
            now += 5.0  # HTTP already consumed part of the default ten-second share.
            rows = [] if mode == "complete_empty" else [stalled, tail]
            if mode == "prefix":
                rows.insert(0, prefix)
            return httpx.Response(200, json={"success": True, "data": {"web": rows}})

        async def allow(url):
            nonlocal now
            visited.append(url)
            if url == stalled["url"]:
                remaining = wait_for.call_args.kwargs["timeout"]
                validation_timeouts.append(remaining)
                now += remaining
                raise asyncio.TimeoutError
            return True

        def handler(request):
            legacy.append("brave" if "offset" in request.url.params else "searxng")
            assert now < 20.0, now
            if mode == "fallback":
                return brave_response([result(1), result(2)], False)
            return httpx.Response(500)

        with patch.object(asyncio, "get_running_loop") as get_loop, patch.object(
            asyncio, "wait_for", wraps=asyncio.wait_for
        ) as wait_for, patch.object(app, "firecrawl_request", firecrawl), patch.object(
            app, "DISCOVERY_TIMEOUT", 20.0
        ), patch.object(app, "BRAVE_POOL", app.BraveKeyPool(("test-key",))), patch.object(
            app, "is_allowed_public_url", allow
        ):
            get_loop.return_value.time.side_effect = lambda: now
            async with client(handler) as http_client:
                try:
                    rows = await app.discover_candidates(http_client, "q", 2)
                except app.AppError as exc:
                    assert mode == "incomplete_empty", mode
                    assert exc.status_code == 503 and exc.retryable
                    assert exc.details["providers"]["firecrawl"] == "candidate_validation_timeout"
                else:
                    assert mode != "incomplete_empty", "unfinished validation became a successful no-match"
                    expected = [result(1)["url"], result(2)["url"]] if mode == "fallback" else (
                        [prefix["url"]] if mode == "prefix" else []
                    )
                    assert [row["url"] for row in rows] == expected, (mode, rows)
            assert legacy == (["brave"] if mode == "fallback" else ["brave", "searxng"]), legacy
            assert tail["url"] not in visited, visited
            assert validation_timeouts == ([] if mode == "complete_empty" else [5.0]), validation_timeouts

    try:
        for mode in ("fallback", "prefix", "complete_empty", "incomplete_empty"):
            run(exercise(mode))
    finally:
        app.health_cache_reset()


def test_firecrawl_search_is_web_first_and_basic_mode_does_not_scrape():
    saved = {
        "pool": app.BRAVE_POOL,
        "brave_url": app.BRAVE_SEARCH_URL,
        "request": app.firecrawl_request,
        "get_client": app.get_client,
        "url_check": app.is_allowed_public_url,
        "rerank": app.rerank,
    }
    app.BRAVE_POOL = app.BraveKeyPool(("test-key",))
    app.BRAVE_SEARCH_URL = "https://brave.example/search"
    app.is_allowed_public_url = lambda _: asyncio.sleep(0, result=True)
    calls = []
    seen = {}

    async def firecrawl(_client, endpoint, payload, *, timeout):
        calls.append(("firecrawl", endpoint, payload))
        return httpx.Response(
            200,
            json={
                "success": True,
                "data": {"web": [{"url": "https://fc.example/", "title": "FC", "description": "snippet"}]},
            },
        )

    class DiscoveryClient:
        async def get(self, url, params=None, headers=None, timeout=None):
            calls.append(("brave", params["offset"], params["count"]))
            return httpx.Response(
                200,
                json={"query": {"more_results_available": False}, "web": {"results": [result(1)]}},
                request=httpx.Request("GET", url),
            )

    async def rerank(_client, _query, chunks, top_k):
        seen["chunks"] = chunks
        return [
            app.SearchResult(
                title=chunks[0]["title"],
                url=chunks[0]["url"],
                snippet=chunks[0]["snippet"],
                content=chunks[0]["content"],
                score=None,
            )
        ], "mock"

    app.firecrawl_request = firecrawl
    app.get_client = lambda: asyncio.sleep(0, result=DiscoveryClient())
    app.rerank = rerank
    app.health_cache_reset()
    try:
        response = run(app.search(app.SearchRequest(query="q", candidates=2, depth="basic")))
        assert response.ok is True
        assert response.data.results[0].url == "https://fc.example/"
        assert response.data.results[0].content == "snippet"
        assert [item["url"] for item in seen["chunks"]] == ["https://fc.example/", "https://1.example/"]
        assert calls == [
            ("firecrawl", "/v2/search", {"query": "q", "limit": 2, "sources": ["web"]}),
            ("brave", 0, 20),
        ], calls
    finally:
        app.BRAVE_POOL = saved["pool"]
        app.BRAVE_SEARCH_URL = saved["brave_url"]
        app.firecrawl_request = saved["request"]
        app.get_client = saved["get_client"]
        app.is_allowed_public_url = saved["url_check"]
        app.rerank = saved["rerank"]
        app.health_cache_reset()


def test_advanced_search_scrapes_firecrawl_once_and_skips_policy_denied_target():
    saved = {
        "pool": app.BRAVE_POOL,
        "brave_url": app.BRAVE_SEARCH_URL,
        "request": app.firecrawl_request,
        "get_client": app.get_client,
        "url_check": app.is_allowed_public_url,
        "firecrawl_allowed": app.firecrawl_url_allowed,
        "crawl_all": app.crawl_all,
        "rerank": app.rerank,
    }
    app.BRAVE_POOL = app.BraveKeyPool(("test-key",))
    app.BRAVE_SEARCH_URL = "https://brave.example/search"
    app.is_allowed_public_url = lambda url: asyncio.sleep(0, result=url != "http://127.0.0.1/")
    app.firecrawl_url_allowed = lambda url: "linkedin.com" not in url
    linked = "https://www.linkedin.com/in/person"
    denied = "https://denied.example/"
    allowed = "https://allowed.example/"
    unsafe = "https://unsafe.example/"
    pdf = "https://reports.example/download"
    calls = []
    crawled = []

    async def firecrawl(_client, endpoint, payload, *, timeout):
        calls.append((endpoint, payload.get("url")))
        if endpoint == "/v2/search":
            rows = [
                {"url": allowed, "title": "Allowed", "description": "A"},
                {"url": linked, "title": "LinkedIn", "description": "B"},
                {"url": denied, "title": "Denied", "description": "C"},
                {"url": pdf, "title": "PDF", "description": "D", "category": "pdf"},
                {"url": unsafe, "title": "Unsafe", "description": "E"},
            ]
            return httpx.Response(200, json={"success": True, "data": {"web": rows}})
        if payload["url"] == denied:
            raise app.FirecrawlUnavailable(403, "policy_denied", None, False)
        reported_url = "http://127.0.0.1/" if payload["url"] == unsafe else "https://final.example/"
        return httpx.Response(
            200,
            json={
                "success": True,
                "data": {
                    "markdown": "# Scraped page",
                    "metadata": {"sourceURL": payload["url"], "statusCode": 203, "url": reported_url},
                },
            },
        )

    brave_calls = []

    class DiscoveryClient:
        async def get(self, url, params=None, headers=None, timeout=None):
            brave_calls.append((params["offset"], params["count"]))
            return httpx.Response(
                200,
                json={
                    "query": {"more_results_available": False},
                    "web": {"results": [{"url": "https://allowed.example", "title": "Allowed alias", "description": "dup"}]},
                },
                request=httpx.Request("GET", url),
            )

    async def crawl_all(_client, candidates, *, deadline=None):
        crawled.extend(item["url"] for item in candidates)
        return [
            {"url": item["url"], "title": item["title"], "snippet": item["snippet"], "content": "# Local page"}
            for item in candidates
        ]

    async def rerank(_client, _query, chunks, top_k):
        return [
            app.SearchResult(
                title=item["title"], url=item["url"], snippet=item["snippet"], content=item["content"], score=None
            )
            for item in chunks[:top_k]
        ], "mock"

    app.firecrawl_request = firecrawl
    app.get_client = lambda: asyncio.sleep(0, result=DiscoveryClient())
    app.crawl_all = crawl_all
    app.rerank = rerank
    app.health_cache_reset()
    try:
        response = run(app.search(app.SearchRequest(query="q", candidates=6, depth="advanced")))
        assert {item.url for item in response.data.results} == {allowed, linked, pdf}, response.data.results
        assert calls == [
            ("/v2/search", None),
            ("/v2/scrape", allowed),
            ("/v2/scrape", denied),
            ("/v2/scrape", unsafe),
        ], calls
        assert brave_calls == [(0, app.BRAVE_PAGE_SIZE)], brave_calls
        assert crawled == [linked, pdf], crawled
        assert "https://allowed.example" not in crawled, crawled
        assert unsafe not in crawled, crawled
        assert denied not in {item.url for item in response.data.results}
    finally:
        app.BRAVE_POOL = saved["pool"]
        app.firecrawl_request = saved["request"]
        app.get_client = saved["get_client"]
        app.is_allowed_public_url = saved["url_check"]
        app.firecrawl_url_allowed = saved["firecrawl_allowed"]
        app.crawl_all = saved["crawl_all"]
        app.rerank = saved["rerank"]
        app.health_cache_reset()


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


def test_brave_response_is_bounded_by_the_discovery_deadline():
    saved_pool = app.BRAVE_POOL
    saved_timeout = app.DISCOVERY_TIMEOUT
    app.BRAVE_POOL = app.BraveKeyPool(("test-key",))
    app.DISCOVERY_TIMEOUT = 0.02

    class SlowClient:
        async def get(self, *args, **kwargs):
            await asyncio.sleep(0.2)
            return httpx.Response(200, json={"web": {"results": []}})

    started = time.monotonic()
    try:
        try:
            run(app.discover_candidates(SlowClient(), "q", 5))
        except app.AppError as exc:
            assert exc.retryable is True
            assert exc.code == "upstream_unavailable"
            assert exc.details["providers"] == {
                "firecrawl": "not_configured",
                "brave": "unavailable",
                "searxng": "deadline_exceeded",
            }
        else:
            raise AssertionError("discovery deadline became an empty success")
        assert time.monotonic() - started < 0.15
    finally:
        app.BRAVE_POOL = saved_pool
        app.DISCOVERY_TIMEOUT = saved_timeout
        app.health_cache_reset()


def test_brave_page_timeout_preserves_normalized_page_prefix():
    validated = []
    page_calls = []
    page_zero = [result(i) for i in range(20)]

    async def check_public(url):
        validated.append(url)
        return True

    async def brave_page(offset):
        page_calls.append(offset)
        if offset == 0:
            return brave_response(page_zero)
        assert len(validated) == 20, "page 0 must be validated before page 1 starts"
        await asyncio.sleep(0.5)
        return brave_response([result(i) for i in range(20, 40)])

    mock = MockDiscoveryClient([], brave_page)
    started = time.monotonic()
    results = run_discovery_with("brave", mock, check_public, 0.05, limit=50)

    assert [item["url"] for item in results] == [result(i)["url"] for i in range(20)]
    assert page_calls == [0, 1], page_calls
    assert mock.calls == [0, 1], mock.calls
    assert time.monotonic() - started < 0.3


def test_each_provider_returns_only_validated_prefix_when_later_validation_stalls():
    public = "https://public.example/"
    private = "http://127.0.0.1/private"
    stalled = "https://stall.example/"
    tail = "https://tail.example/"
    rows = [{"url": url, "title": url, "description": "snippet"} for url in (public, private, stalled, tail)]

    for provider in ("brave", "searxng"):
        visited = []
        cancelled = []

        async def check(url):
            visited.append(url)
            if url == private:
                return False
            if url == stalled:
                try:
                    await asyncio.sleep(1)
                finally:
                    cancelled.append(url)
            return True

        async def brave_page(_offset):
            return brave_response(rows)

        mock = MockDiscoveryClient(rows, brave_page)
        results = run_discovery_with(provider, mock, check, 0.05)

        assert [item["url"] for item in results] == [public], (provider, results)
        assert visited == [public, private, stalled], (provider, visited)
        assert cancelled == [stalled], (provider, cancelled)
        assert tail not in visited, (provider, visited)


def test_each_provider_first_validation_timeout_is_a_retryable_outage():
    stalled = "https://stall.example/"
    rows = [{"url": stalled, "title": "stall", "description": "snippet"}]

    for provider in ("brave", "searxng"):
        cancelled = []

        async def check(url):
            try:
                await asyncio.sleep(1)
            finally:
                cancelled.append(url)
            return True

        async def brave_page(_offset):
            return brave_response(rows, False)

        mock = MockDiscoveryClient(rows, brave_page)
        try:
            run_discovery_with(provider, mock, check, 0.05)
        except app.AppError as exc:
            assert exc.status_code == 503
            assert exc.code == "upstream_unavailable"
            assert exc.retryable is True
        else:
            raise AssertionError(f"{provider} incomplete-empty validation became a successful empty result")
        assert cancelled == [stalled], (provider, cancelled)


def test_candidate_validation_preserves_external_cancellation():
    saved_check = app.is_allowed_public_url
    started = None

    async def slow_check(_url):
        started.set()
        await asyncio.sleep(1)

    app.is_allowed_public_url = slow_check

    async def exercise():
        nonlocal started
        started = asyncio.Event()
        task = asyncio.create_task(
            app.shape_candidates(
                [result(1)], 5, deadline=asyncio.get_running_loop().time() + 5
            )
        )
        await started.wait()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            return
        raise AssertionError("shape_candidates swallowed external cancellation")

    try:
        run(exercise())
    finally:
        app.is_allowed_public_url = saved_check


def test_brave_page_timeout_preserves_normalized_prefix_without_fallback():
    validated = []
    page_calls = []
    first_page = [result(i) for i in range(20)]

    async def check_public(url):
        validated.append(url)
        return True

    async def brave_page(offset):
        page_calls.append(offset)
        if offset == 0:
            return brave_response(first_page)
        assert len(validated) == 20, "page zero must be validated before page one starts"
        await asyncio.sleep(1)
        return brave_response([result(i) for i in range(20, 40)])

    mock = MockDiscoveryClient([], brave_page)
    started = time.monotonic()
    results = run_discovery_with("brave", mock, check_public, 0.05, limit=50)

    assert [item["url"] for item in results] == [result(i)["url"] for i in range(20)]
    assert page_calls == [0, 1], page_calls
    assert mock.calls == [0, 1], mock.calls
    assert time.monotonic() - started < 0.3


def test_both_providers_return_only_accepted_prefix_on_stalled_validation():
    public = "https://public.example/"
    private = "http://127.0.0.1/private"
    stalled = "https://stall.example/"
    tail = "https://tail.example/"
    rows = [{"url": url, "title": url, "description": "snippet"} for url in (public, private, stalled, tail)]

    for provider in ("brave", "searxng"):
        visited = []
        cancelled = []

        async def check_url(url):
            visited.append(url)
            if url == private:
                return False
            if url == stalled:
                try:
                    await asyncio.sleep(1)
                finally:
                    cancelled.append(url)
            return True

        async def brave_page(_offset):
            return brave_response(rows)

        mock = MockDiscoveryClient(rows, brave_page)
        results = run_discovery_with(provider, mock, check_url, 0.05)

        assert [item["url"] for item in results] == [public], (provider, results)
        assert visited == [public, private, stalled], (provider, visited)
        assert cancelled == [stalled], (provider, cancelled)
        assert tail not in visited, (provider, visited)


def test_both_providers_first_validation_timeout_is_a_retryable_outage():
    stalled = "https://stall.example/"
    rows = [{"url": stalled, "title": "stall", "description": "snippet"}]

    for provider in ("brave", "searxng"):
        cancelled = []

        async def check_url(_url):
            try:
                await asyncio.sleep(1)
            finally:
                cancelled.append(stalled)
            return True

        async def brave_page(_offset):
            return brave_response(rows, False)

        mock = MockDiscoveryClient(rows, brave_page)
        try:
            run_discovery_with(provider, mock, check_url, 0.05)
        except app.AppError as exc:
            assert exc.status_code == 503
            assert exc.code == "upstream_unavailable"
            assert exc.retryable is True
        else:
            raise AssertionError(f"{provider} incomplete-empty validation became a successful empty result")
        assert cancelled == [stalled], (provider, cancelled)


def test_brave_page_timeout_preserves_normalized_prefix_without_fallback():
    validated = []
    page_calls = []
    first_page = [result(i) for i in range(20)]

    async def check_public(url):
        validated.append(url)
        return True

    async def brave_page(offset):
        page_calls.append(offset)
        if offset == 0:
            return brave_response(first_page)
        assert len(validated) == 20, "page zero must be validated before page one starts"
        await asyncio.sleep(1)
        return brave_response([result(i) for i in range(20, 40)])

    mock = MockDiscoveryClient([], brave_page)
    started = time.monotonic()
    results = run_discovery_with("brave", mock, check_public, 0.05, limit=50)

    assert [item["url"] for item in results] == [result(i)["url"] for i in range(20)]
    assert page_calls == [0, 1], page_calls
    assert mock.calls == [0, 1], mock.calls
    assert time.monotonic() - started < 0.3


def test_both_providers_return_only_accepted_prefix_on_stalled_validation():
    public = "https://public.example/"
    private = "http://127.0.0.1/private"
    stalled = "https://stall.example/"
    tail = "https://tail.example/"
    rows = [{"url": url, "title": url, "description": "snippet"} for url in (public, private, stalled, tail)]

    for provider in ("brave", "searxng"):
        visited = []
        cancelled = []

        async def check_url(url):
            visited.append(url)
            if url == private:
                return False
            if url == stalled:
                try:
                    await asyncio.sleep(1)
                finally:
                    cancelled.append(url)
            return True

        async def brave_page(_offset):
            return brave_response(rows)

        mock = MockDiscoveryClient(rows, brave_page)
        results = run_discovery_with(provider, mock, check_url, 0.05)

        assert [item["url"] for item in results] == [public], (provider, results)
        assert visited == [public, private, stalled], (provider, visited)
        assert cancelled == [stalled], (provider, cancelled)
        assert tail not in visited, (provider, visited)


def test_both_providers_first_validation_timeout_is_a_retryable_outage():
    stalled = "https://stall.example/"
    rows = [{"url": stalled, "title": "stall", "description": "snippet"}]

    for provider in ("brave", "searxng"):
        cancelled = []

        async def check_url(url):
            try:
                await asyncio.sleep(1)
            finally:
                cancelled.append(url)
            return True

        async def brave_page(_offset):
            return brave_response(rows, False)

        mock = MockDiscoveryClient(rows, brave_page)
        try:
            run_discovery_with(provider, mock, check_url, 0.05)
        except app.AppError as exc:
            assert exc.status_code == 503
            assert exc.code == "upstream_unavailable"
            assert exc.retryable is True
        else:
            raise AssertionError(f"{provider} incomplete-empty validation became a successful empty result")
        assert cancelled == [stalled], (provider, cancelled)


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


def test_health_accepts_healthy_firecrawl_when_searxng_is_down_without_searching():
    saved_pool = app.BRAVE_POOL
    saved_get_client = app.get_client
    app.BRAVE_POOL = app.BraveKeyPool(())
    app.health_cache_reset()
    app._record_provider_health("firecrawl", "ok")
    app._record_provider_health("searxng", "unavailable")
    paths = []

    def handler(request):
        paths.append(request.url.path)
        return httpx.Response(200)

    app.get_client = lambda: asyncio.sleep(0, result=client(handler))
    try:
        health = run(app.health())
        assert health.data.checks["firecrawl"] == "ok", health.data.checks
        assert health.data.checks["searxng"] == "unavailable", health.data.checks
        assert "/search" not in paths, paths
        assert "/v2/search" not in paths, paths
    finally:
        app.get_client = saved_get_client
        app.BRAVE_POOL = saved_pool
        app.health_cache_reset()


def test_health_keeps_stale_success_available_without_provider_requests():
    saved_pool = app.BRAVE_POOL
    saved_engines = app.SEARXNG_ENGINES
    saved_get_client = app.get_client
    app.BRAVE_POOL = app.BraveKeyPool(("test-key",))
    app.SEARXNG_ENGINES = "bing"
    requests = []

    def handler(request):
        requests.append((request.method, request.url.path))
        assert request.method == "GET" and request.url.path == "/health", requests
        return httpx.Response(200)

    app.get_client = lambda: asyncio.sleep(0, result=client(handler))
    try:
        for provider in ("firecrawl", "brave", "searxng"):
            requests.clear()
            app.health_cache_reset()
            for name in ("firecrawl", "brave", "searxng"):
                app._record_provider_health(name, "ok" if name == provider else "unavailable")
            assert run(app.health()).data.checks[provider] == "ok"
            stale_at = time.monotonic() - app.HEALTH_CACHE_TTL - 1
            for result in app._PROVIDER_HEALTH.values():
                result["at"] = stale_at
            app._HEALTH_CACHE["at"] = stale_at
            health = run(app.health())
            assert health.data.checks[provider] == "stale:ok", health.data.checks
            assert health.data.checks["searxng_service"] == "not_probed", health.data.checks
            assert requests == [("GET", "/health")] * 4, requests

            app._record_provider_health(provider, "unavailable")
            for _ in range(2):
                try:
                    run(app.health())
                except app.AppError as exc:
                    assert exc.status_code == 503
                    assert exc.details["checks"][provider] == "unavailable"
                else:
                    raise AssertionError("last-known success masked a newer provider failure")
            assert requests == [("GET", "/health")] * 6, requests
            app._record_provider_health(provider, "ok")
            assert run(app.health()).data.checks[provider] == "ok"
            assert requests == [("GET", "/health")] * 8, requests
    finally:
        app.get_client = saved_get_client
        app.BRAVE_POOL = saved_pool
        app.SEARXNG_ENGINES = saved_engines
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
    app._record_provider_health("firecrawl", "unavailable")
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
            assert checks["firecrawl"] == "stale:unavailable", checks
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


def test_strict_search_features_fail_when_no_provider_can_honor_them():
    saved_pool = app.BRAVE_POOL
    saved_request = app.firecrawl_request
    saved_searxng = app.search_searxng
    app.BRAVE_POOL = app.BraveKeyPool(())
    calls = []

    async def firecrawl(*_args, **_kwargs):
        raise app.FirecrawlUnavailable(503, "provider_unavailable", 1.0, True)

    async def searxng(*_args, **_kwargs):
        calls.append("searxng")
        return [{"url": "https://fallback.example/", "title": "F", "snippet": "s"}]

    app.firecrawl_request = firecrawl
    app.search_searxng = searxng
    try:
        request = app.SearchRequest(query="q", categories=["developer"])
        try:
            run(app.discover_candidates(object(), "q", 1, request))
        except app.AppError as exc:
            assert exc.status_code == 503, vars(exc)
            assert exc.code == "fallback_feature_unavailable", exc.code
            assert exc.details["unsupported"] == ["categories"], exc.details
            assert exc.details["unsupported_by_provider"]["searxng"] == ["categories"], exc.details
            assert exc.retryable is True, vars(exc)
        else:
            raise AssertionError("strict feature loss did not raise")
        assert calls == [], calls
    finally:
        app.BRAVE_POOL = saved_pool
        app.firecrawl_request = saved_request
        app.search_searxng = saved_searxng
        app.health_cache_reset()


def test_allow_degraded_returns_fallback_results_with_named_feature_loss():
    saved_pool = app.BRAVE_POOL
    saved_request = app.firecrawl_request
    saved_searxng = app.search_searxng
    saved_check = app.is_allowed_public_url
    app.BRAVE_POOL = app.BraveKeyPool(())

    async def firecrawl(*_args, **_kwargs):
        raise app.FirecrawlUnavailable(503, "provider_unavailable", 1.0, True)

    async def searxng(*_args, **_kwargs):
        return [{"url": "https://fallback.example/", "title": "F", "snippet": "s"}]

    async def allow(_url):
        return True

    app.firecrawl_request = firecrawl
    app.search_searxng = searxng
    app.is_allowed_public_url = allow
    degraded = {}
    try:
        request = app.SearchRequest(
            query="q", categories=["developer"], allow_degraded=True
        )
        candidates = run(app.discover_candidates(object(), "q", 1, request, degraded))
        assert [item["url"] for item in candidates] == ["https://fallback.example/"], candidates
        assert degraded == {
            "providers": [{"provider": "searxng", "unsupported": ["categories"]}]
        }, degraded
    finally:
        app.BRAVE_POOL = saved_pool
        app.firecrawl_request = saved_request
        app.search_searxng = saved_searxng
        app.is_allowed_public_url = saved_check
        app.health_cache_reset()


def test_inline_search_scrape_is_used_without_a_second_firecrawl_request():
    saved_request = app.firecrawl_request
    saved_check = app.is_allowed_public_url
    calls = []

    async def firecrawl(_client, endpoint, payload, *, timeout):
        calls.append((endpoint, payload))
        return httpx.Response(
            200,
            json={
                "success": True,
                "creditsUsed": 2,
                "data": {
                    "web": [
                        {
                            "url": "https://inline.example/",
                            "title": "Inline",
                            "description": "snippet",
                            "markdown": "# Inline page",
                            "links": ["https://inline.example/next"],
                            "metadata": {
                                "statusCode": 200,
                                "sourceURL": "https://inline.example/",
                            },
                        }
                    ]
                },
            },
        )

    async def allow(_url):
        return True

    app.firecrawl_request = firecrawl
    app.is_allowed_public_url = allow
    try:
        request = app.SearchRequest(
            query="q", scrape_options={"formats": ["markdown", "links"]}
        )
        candidates = run(app.discover_candidates(object(), "q", 1, request))
        assert candidates[0]["_inline"]["markdown"] == "# Inline page", candidates
        assert candidates[0]["_inline"]["links"] == ["https://inline.example/next"], candidates
        loss = set()
        pages, local, _ = run(
            app.scrape_firecrawl_candidates(
                object(), candidates, time.monotonic() + 5, request, loss
            )
        )
        assert [page["content"] for page in pages] == ["# Inline page"], pages
        assert pages[0]["extras"]["links"] == ["https://inline.example/next"], pages
        assert local == [], local
        assert loss == set(), loss
        assert [endpoint for endpoint, _ in calls] == ["/v2/search"], calls
    finally:
        app.firecrawl_request = saved_request
        app.is_allowed_public_url = saved_check


def test_inline_extras_only_results_keep_content_and_requested_formats():
    saved_request = app.firecrawl_request
    saved_check = app.is_allowed_public_url

    async def firecrawl(*_args, **_kwargs):
        raise AssertionError("extras-only inline content must not trigger a scrape")

    async def allow(_url):
        return True

    app.firecrawl_request = firecrawl
    app.is_allowed_public_url = allow
    try:
        request = app.SearchRequest(
            query="q",
            scrape_options={"formats": ["links"]},
        )
        candidates = [
            {
                "url": "https://inline.example/",
                "title": "Inline",
                "snippet": "fallback passage",
                "_firecrawl_source": True,
                "_inline": {
                    "links": ["https://inline.example/next"],
                    "metadata": {"statusCode": 200, "sourceURL": "https://inline.example/"},
                },
            }
        ]
        loss = set()
        pages, local, _ = run(
            app.scrape_firecrawl_candidates(
                object(), candidates, time.monotonic() + 5, request, loss
            )
        )
        assert pages == [
            {
                "url": "https://inline.example/",
                "title": "Inline",
                "snippet": "fallback passage",
                "content": "fallback passage",
                "metadata": {"statusCode": 200, "sourceURL": "https://inline.example/"},
                "extras": {"links": ["https://inline.example/next"]},
            }
        ], pages
        assert local == [], local
        assert loss == set(), loss
    finally:
        app.firecrawl_request = saved_request
        app.is_allowed_public_url = saved_check


def test_inline_missing_requested_formats_are_reported_not_silently_dropped():
    saved_request = app.firecrawl_request
    saved_check = app.is_allowed_public_url

    async def firecrawl(*_args, **_kwargs):
        raise AssertionError("inline content must not trigger a second scrape")

    async def allow(_url):
        return True

    app.firecrawl_request = firecrawl
    app.is_allowed_public_url = allow
    try:
        request = app.SearchRequest(
            query="q",
            scrape_options={"formats": ["markdown", "links"]},
        )
        candidates = [
            {
                "url": "https://inline.example/",
                "title": "Inline",
                "snippet": "s",
                "_firecrawl_source": True,
                "_inline": {
                    "markdown": "# Page",
                    "metadata": {"statusCode": 200, "sourceURL": "https://inline.example/"},
                },
            }
        ]
        loss = set()
        pages, _, _ = run(
            app.scrape_firecrawl_candidates(
                object(), candidates, time.monotonic() + 5, request, loss
            )
        )
        assert [page["content"] for page in pages] == ["# Page"], pages
        assert loss == {"format:links"}, loss
    finally:
        app.firecrawl_request = saved_request
        app.is_allowed_public_url = saved_check


def test_image_results_do_not_attempt_page_extraction():
    saved_request = app.firecrawl_request
    saved_check = app.is_allowed_public_url

    async def firecrawl(*_args, **_kwargs):
        raise AssertionError("image results must not trigger a page scrape")

    async def allow(_url):
        return True

    app.firecrawl_request = firecrawl
    app.is_allowed_public_url = allow
    try:
        request = app.SearchRequest(query="q")
        candidates = [
            {
                "url": "https://img.example/photo.png",
                "title": "Photo",
                "snippet": "",
                "source": "images",
                "imageUrl": "https://img.example/photo.png",
                "position": 1,
            }
        ]
        loss = set()
        pages, local, _ = run(
            app.scrape_firecrawl_candidates(
                object(), candidates, time.monotonic() + 5, request, loss
            )
        )
        assert pages == [
            {
                "url": "https://img.example/photo.png",
                "title": "Photo",
                "snippet": "",
                "content": "Photo",
                "source": "images",
                "imageUrl": "https://img.example/photo.png",
                "position": 1,
            }
        ], pages
        assert local == [], local
        assert loss == set(), loss

        image_request = app.SearchRequest(
            query="q", scrape_options={"formats": ["images"]}
        )
        loss = set()
        pages, _, _ = run(
            app.scrape_firecrawl_candidates(
                object(), candidates, time.monotonic() + 5, image_request, loss
            )
        )
        assert pages and loss == set(), (pages, loss)
        assert pages[0]["extras"]["images"] == ["https://img.example/photo.png"], pages

        markdown_request = app.SearchRequest(
            query="q", scrape_options={"formats": ["markdown"]}
        )
        loss = set()
        run(
            app.scrape_firecrawl_candidates(
                object(), candidates, time.monotonic() + 5, markdown_request, loss
            )
        )
        assert loss == {"format:markdown"}, loss
    finally:
        app.firecrawl_request = saved_request
        app.is_allowed_public_url = saved_check


def test_inline_reported_url_must_stay_public():
    saved_request = app.firecrawl_request
    saved_check = app.is_allowed_public_url

    async def firecrawl(*_args, **_kwargs):
        raise AssertionError("inline content must not trigger a second scrape")

    async def allow(url):
        return "127.0.0.1" not in url

    app.firecrawl_request = firecrawl
    app.is_allowed_public_url = allow
    try:
        request = app.SearchRequest(query="q")
        candidates = [
            {
                "url": "https://inline.example/",
                "title": "Inline",
                "snippet": "s",
                "_firecrawl_source": True,
                "_inline": {
                    "markdown": "# Page",
                    "metadata": {
                        "statusCode": 200,
                        "sourceURL": "https://inline.example/",
                        "url": "http://127.0.0.1/private",
                    },
                },
            }
        ]
        loss = set()
        pages, local, _ = run(
            app.scrape_firecrawl_candidates(
                object(), candidates, time.monotonic() + 5, request, loss
            )
        )
        assert pages == [], pages
        assert local == candidates, local
        assert loss == set(), loss
    finally:
        app.firecrawl_request = saved_request
        app.is_allowed_public_url = saved_check


def test_strict_not_configured_firecrawl_is_unsupported_not_retryable():
    saved_pool = app.BRAVE_POOL
    saved_request = app.firecrawl_request
    saved_searxng = app.search_searxng
    app.BRAVE_POOL = app.BraveKeyPool(())

    async def firecrawl(*_args, **_kwargs):
        raise app.FirecrawlUnavailable(None, "not_configured", None, False)

    async def searxng(*_args, **_kwargs):
        raise AssertionError("strict mode must not call a fallback that loses features")

    app.firecrawl_request = firecrawl
    app.search_searxng = searxng
    try:
        request = app.SearchRequest(query="q", categories=["developer"])
        try:
            run(app.discover_candidates(object(), "q", 1, request))
        except app.AppError as exc:
            assert exc.status_code == 422, vars(exc)
            assert exc.code == "unsupported_feature", exc.code
            assert exc.retryable is False, vars(exc)
            assert exc.details["unsupported"] == ["categories"], exc.details
        else:
            raise AssertionError("unconfigured Firecrawl did not raise unsupported_feature")
    finally:
        app.BRAVE_POOL = saved_pool
        app.firecrawl_request = saved_request
        app.search_searxng = saved_searxng
        app.health_cache_reset()


def test_custom_tbs_is_firecrawl_only():
    request = app.SearchRequest(query="q", tbs="qdr:w")
    assert app.unsupported_search_features(request, "brave") == ["tbs"]
    assert app.unsupported_search_features(request, "searxng") == ["tbs"]
    assert app.search_filters(request)["tbs"] == "qdr:w"


def test_fallback_query_and_params_translate_free_filters():
    request = app.SearchRequest(
        query="q",
        include_domains=["a.example"],
        time_range="week",
        safe=True,
        country="us",
    )
    query, params = app.fallback_query_and_params("q", request, "brave")
    assert query == "q site:a.example", query
    assert params == {
        "freshness": "pw",
        "safesearch": "strict",
        "country": "US",
    }, params
    query, params = app.fallback_query_and_params("q", request, "searxng")
    assert query == "q site:a.example", query
    assert params == {"time_range": "month", "safesearch": 2}, params


def test_strict_firecrawl_invalid_request_is_unsupported_not_retryable():
    saved_pool = app.BRAVE_POOL
    saved_request = app.firecrawl_request
    saved_searxng = app.search_searxng
    app.BRAVE_POOL = app.BraveKeyPool(())

    async def firecrawl(*_args, **_kwargs):
        return httpx.Response(422, json={"success": False, "error": "invalid request"})

    async def searxng(*_args, **_kwargs):
        raise AssertionError("strict mode must not fall back")

    app.firecrawl_request = firecrawl
    app.search_searxng = searxng
    try:
        request = app.SearchRequest(query="q", categories=["developer"])
        try:
            run(app.discover_candidates(object(), "q", 1, request))
        except app.AppError as exc:
            assert exc.status_code == 422, vars(exc)
            assert exc.code == "unsupported_feature", exc.code
            assert exc.retryable is False, vars(exc)
        else:
            raise AssertionError("permanent Firecrawl 4xx did not raise unsupported_feature")
    finally:
        app.BRAVE_POOL = saved_pool
        app.firecrawl_request = saved_request
        app.search_searxng = saved_searxng
        app.health_cache_reset()


def test_search_strict_and_degraded_extraction_loss():
    saved = (
        app.get_client,
        app.discover_candidates,
        app.scrape_firecrawl_candidates,
        app.crawl_all,
        app.rerank,
    )

    async def fake_discover(*_args, **_kwargs):
        return [
            {
                "url": "https://a.example/",
                "title": "A",
                "snippet": "s",
                "_firecrawl_source": True,
            }
        ]

    async def fake_scrape(_client, candidates, _deadline, _request, loss):
        loss.add("format:links")
        return [], candidates, set()

    async def fake_crawl(_client, candidates, *, deadline=None):
        return [
            {
                "url": candidate["url"],
                "title": candidate["title"],
                "snippet": "",
                "content": "A sufficiently long page passage for the extraction test.",
            }
            for candidate in candidates
        ]

    async def fake_rerank(_client, _query, chunks, _top_k):
        return (
            [
                app.SearchResult(
                    title=chunks[0]["title"],
                    url=chunks[0]["url"],
                    content=chunks[0]["content"],
                    score=None,
                )
            ],
            "test",
        )

    app.get_client = lambda: asyncio.sleep(0, result=object())
    app.discover_candidates = fake_discover
    app.scrape_firecrawl_candidates = fake_scrape
    app.crawl_all = fake_crawl
    app.rerank = fake_rerank
    try:
        request = app.SearchRequest(
            query="q", scrape_options={"formats": ["markdown", "links"]}
        )
        try:
            run(app.search(request))
        except app.AppError as exc:
            assert exc.status_code == 503, vars(exc)
            assert exc.code == "fallback_feature_unavailable", exc.code
            assert exc.details["unsupported"] == ["format:links"], exc.details
        else:
            raise AssertionError("strict extraction loss did not raise")

        envelope = run(
            app.search(
                app.SearchRequest(
                    query="q",
                    scrape_options={"formats": ["markdown", "links"]},
                    allow_degraded=True,
                )
            )
        )
        assert envelope.data.warnings, envelope
        assert envelope.data.degraded["providers"][0]["unsupported"] == [
            "format:links"
        ], envelope.data.degraded
    finally:
        (
            app.get_client,
            app.discover_candidates,
            app.scrape_firecrawl_candidates,
            app.crawl_all,
            app.rerank,
        ) = saved


def demo():
    for name, check in sorted(globals().items()):
        if name.startswith("test_"):
            check()
            print(f"ok {name}")
    print("all checks passed")


if __name__ == "__main__":
    demo()
