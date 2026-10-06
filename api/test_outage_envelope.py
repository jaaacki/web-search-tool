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


def run(coro):
    return asyncio.run(coro)


def raises_app_error(coro):
    try:
        run(coro)
    except app.AppError as exc:
        return exc
    raise AssertionError("expected an AppError; the outage was reported as a success")


def test_all_engines_dead_is_an_error_envelope():
    exc = raises_app_error(app.search_searxng(client(dead_pool), "anything", 5))
    assert exc.status_code == 503, exc.status_code
    assert exc.code == "upstream_unavailable", exc.code
    # retryable is what turns "give up" into "try again", so it must survive.
    assert exc.details["retryable"] is True, exc.details
    assert exc.details["unresponsive_engines"] == DEAD_ENGINES, exc.details


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
    saved_client = app.get_client
    app.BRAVE_POOL = None  # force the searxng-only fallback
    app.get_client = lambda: asyncio.sleep(0, result=client(dead_pool))
    try:
        exc = raises_app_error(app.search(app.SearchRequest(query="anything", candidates=5, max_results=3)))
        assert exc.code == "upstream_unavailable", exc.code
        body = app.error_response(exc.status_code, exc.code, exc.message, exc.details).body.decode()
        assert '"ok": false' in body or '"ok":false' in body, body
        assert "upstream_unavailable" in body, body
    finally:
        app.BRAVE_POOL = saved_pool
        app.get_client = saved_client


def test_health_fails_during_a_total_outage():
    # Health used to echo its configured URLs, so it stayed green while search was dead.
    saved = app.get_client
    app.get_client = lambda: asyncio.sleep(0, result=client(dead_pool))
    try:
        exc = raises_app_error(app.health())
        assert exc.status_code == 503, exc.status_code
        assert exc.code == "upstream_unavailable", exc.code
    finally:
        app.get_client = saved


def test_health_is_green_when_engines_answer():
    saved = app.get_client

    def healthy(request: httpx.Request) -> httpx.Response:
        if httpx.URL(str(request.url)).path == "/search":
            return httpx.Response(
                200,
                json={
                    "results": [{"url": "https://a.example/", "title": "A", "snippet": "s"}],
                    "unresponsive_engines": [],
                },
            )
        return httpx.Response(200, json={})

    app.get_client = lambda: asyncio.sleep(0, result=client(healthy))
    try:
        envelope = run(app.health())
        assert envelope.ok is True
        assert envelope.data.checks["searxng"] == "ok", envelope.data.checks
    finally:
        app.get_client = saved


def test_health_actually_issues_a_query():
    # The probe must reach searxng's /search, not just read a config value.
    seen: list[str] = []

    def watcher(request: httpx.Request) -> httpx.Response:
        seen.append(httpx.URL(str(request.url)).path)
        if seen[-1] == "/search":
            return httpx.Response(
                200,
                json={
                    "results": [{"url": "https://a.example/", "title": "A", "snippet": "s"}],
                    "unresponsive_engines": [],
                },
            )
        return httpx.Response(200, json={})

    saved = app.get_client
    app.get_client = lambda: asyncio.sleep(0, result=client(watcher))
    try:
        run(app.health())
        assert "/search" in seen, f"health never queried an engine; saw {seen}"
    finally:
        app.get_client = saved


def demo():
    for name, check in sorted(globals().items()):
        if name.startswith("test_"):
            check()
            print(f"ok {name}")
    print("all checks passed")


if __name__ == "__main__":
    demo()