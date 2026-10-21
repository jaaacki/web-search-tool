"""Mocked identity, recovery, and result-validity checks for #45."""

import asyncio

import app


def run(coro):
    return asyncio.run(coro)


def candidate(url, title):
    return {"url": url, "title": title, "snippet": f"snippet {title}"}


def crawl_item(url, content, **kwargs):
    return {
        "url": url,
        "success": True,
        "status_code": 200,
        "response_headers": {"content-type": "text/html; charset=utf-8"},
        "metadata": {"title": "result"},
        "markdown": content,
        **kwargs,
    }


async def allow_public(_url):
    return True


def test_bulk_reordered_redirects_keep_original_candidate_identity():
    a = candidate("https://a.example/start", "A")
    b = candidate("https://b.example/start", "B")
    rows = [
        crawl_item(b["url"], "content B", redirected_url="https://b.example/final"),
        crawl_item(a["url"], "content A", redirected_url="https://a.example/final"),
    ]

    async def fake_call(_client, _payload):
        return {"success": True, "results": rows}

    async def check():
        saved_call, saved_allow = app.call_crawl4ai, app.is_allowed_public_url
        app.call_crawl4ai, app.is_allowed_public_url = fake_call, allow_public
        try:
            pages, retry = await app._bulk_crawl(None, [a, b])
        finally:
            app.call_crawl4ai, app.is_allowed_public_url = saved_call, saved_allow
        by_title = {page["title"]: page for page in pages}
        assert not retry, retry
        assert by_title["A"]["content"] == "content A", by_title
        assert by_title["A"]["url"] == "https://a.example/final", by_title
        assert by_title["B"]["content"] == "content B", by_title
        assert by_title["B"]["url"] == "https://b.example/final", by_title

    run(check())


def test_ambiguous_bulk_rows_are_not_guessed_and_missing_targets_recover():
    candidates = [candidate("https://a.example/", "A"), candidate("https://b.example/", "B")]
    calls = []

    async def fake_call(_client, payload):
        urls = payload["urls"]
        calls.append(urls)
        if len(urls) > 1:
            return {"success": True, "results": [crawl_item("https://unclaimed.example/", "wrong page")]}
        return {"success": True, "results": [crawl_item(urls[0], f"right {urls[0]}")]}

    async def check():
        saved_call, saved_allow = app.call_crawl4ai, app.is_allowed_public_url
        app.call_crawl4ai, app.is_allowed_public_url = fake_call, allow_public
        try:
            pages = await app.crawl_all(None, candidates)
        finally:
            app.call_crawl4ai, app.is_allowed_public_url = saved_call, saved_allow
        by_title = {page["title"]: page for page in pages}
        assert calls == [[item["url"] for item in candidates], [candidates[0]["url"]], [candidates[1]["url"]]], calls
        assert by_title["A"]["content"] == f"right {candidates[0]['url']}", by_title
        assert by_title["B"]["content"] == f"right {candidates[1]['url']}", by_title
        assert [page["title"] for page in pages] == ["A", "B"], pages
        assert all(page["content"] != "wrong page" for page in pages), pages

    run(check())


def test_partial_bulk_recovers_only_the_missing_candidate():
    a = candidate("https://a.example/", "A")
    b = candidate("https://b.example/", "B")
    calls = []

    async def fake_call(_client, payload):
        urls = payload["urls"]
        calls.append(urls)
        if len(urls) > 1:
            return {"success": True, "results": [crawl_item(b["url"], "content B")]}
        return {"success": True, "results": [crawl_item(urls[0], "content A")]}

    async def check():
        saved_call, saved_allow = app.call_crawl4ai, app.is_allowed_public_url
        app.call_crawl4ai, app.is_allowed_public_url = fake_call, allow_public
        try:
            pages = await app.crawl_all(None, [a, b])
        finally:
            app.call_crawl4ai, app.is_allowed_public_url = saved_call, saved_allow
        assert calls == [[a["url"], b["url"]], [a["url"]]], calls
        assert [page["title"] for page in pages] == ["A", "B"], pages
        assert {page["title"]: page["content"] for page in pages} == {"A": "content A", "B": "content B"}

    run(check())


def test_partial_results_survive_deadline_and_hanging_sibling_is_cancelled():
    a = candidate("https://a.example/", "A")
    b = candidate("https://b.example/", "B")
    cancelled = []

    async def fake_call(_client, payload):
        urls = payload["urls"]
        if len(urls) > 1:
            return {"success": True, "results": []}
        if urls[0] == a["url"]:
            return {"success": True, "results": [crawl_item(a["url"], "finished A")]}
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.append(urls[0])

    async def check():
        saved_call, saved_allow, saved_timeout = (
            app.call_crawl4ai,
            app.is_allowed_public_url,
            app.CRAWL_TIMEOUT,
        )
        app.call_crawl4ai, app.is_allowed_public_url, app.CRAWL_TIMEOUT = fake_call, allow_public, 0.03
        try:
            pages = await app.crawl_all(None, [a, b])
        finally:
            app.call_crawl4ai, app.is_allowed_public_url, app.CRAWL_TIMEOUT = (
                saved_call,
                saved_allow,
                saved_timeout,
            )
        assert [page["content"] for page in pages] == ["finished A"], pages
        assert cancelled == [b["url"]], cancelled

    run(check())


def test_bulk_normalization_deadline_keeps_accepted_prefix():
    a = candidate("https://a.example/", "A")
    b = candidate("https://b.example/", "B")
    final_url = "https://b.example/final"
    calls = []
    final_started = asyncio.Event()
    cancelled = []

    async def fake_call(_client, payload):
        calls.append(payload["urls"])
        return {
            "success": True,
            "results": [
                crawl_item(a["url"], "finished A"),
                crawl_item(b["url"], "finished B", redirected_url=final_url),
            ],
        }

    async def allow(url):
        if url == final_url:
            final_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.append(url)
        return True

    async def check():
        saved_call, saved_allow = app.call_crawl4ai, app.is_allowed_public_url
        app.call_crawl4ai, app.is_allowed_public_url = fake_call, allow
        try:
            deadline = asyncio.get_running_loop().time() + 0.1
            pages = await app.crawl_all(None, [a, b], deadline=deadline)
        finally:
            app.call_crawl4ai, app.is_allowed_public_url = saved_call, saved_allow
        assert final_started.is_set(), "bulk final-URL validation did not start"
        assert cancelled == [final_url], cancelled
        assert calls == [[a["url"], b["url"]]], calls
        assert [page["content"] for page in pages] == ["finished A"], pages

    run(check())


def test_incomplete_empty_bulk_normalization_is_retryable_error():
    a = candidate("https://a.example/", "A")
    b = candidate("https://b.example/", "B")
    final_url = "https://b.example/final"
    calls = []
    final_started = asyncio.Event()
    cancelled = []

    async def fake_call(_client, payload):
        calls.append(payload["urls"])
        return {
            "success": True,
            "results": [
                crawl_item(a["url"], ""),
                crawl_item(b["url"], "finished B", redirected_url=final_url),
            ],
        }

    async def allow(url):
        if url == final_url:
            final_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.append(url)
        return True

    async def check():
        saved_call, saved_allow = app.call_crawl4ai, app.is_allowed_public_url
        app.call_crawl4ai, app.is_allowed_public_url = fake_call, allow
        try:
            deadline = asyncio.get_running_loop().time() + 0.1
            try:
                await app.crawl_all(None, [a, b], deadline=deadline)
            except app.AppError as exc:
                assert exc.status_code == 502, exc.status_code
                assert exc.code == "crawl4ai_error", exc.code
                assert exc.retryable is True, exc.retryable
            else:
                raise AssertionError("incomplete empty bulk normalization became a successful empty result")
        finally:
            app.call_crawl4ai, app.is_allowed_public_url = saved_call, saved_allow
        assert final_started.is_set(), "bulk final-URL validation did not start"
        assert cancelled == [final_url], cancelled
        assert calls == [[a["url"], b["url"]]], calls

    run(check())


def test_external_cancellation_propagates_during_bulk_normalization():
    a = candidate("https://a.example/", "A")
    b = candidate("https://b.example/", "B")
    final_url = "https://b.example/final"
    final_started = asyncio.Event()
    cancelled = []

    async def fake_call(_client, _payload):
        return {
            "success": True,
            "results": [
                crawl_item(a["url"], "finished A"),
                crawl_item(b["url"], "finished B", redirected_url=final_url),
            ],
        }

    async def allow(url):
        if url == final_url:
            final_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.append(url)
        return True

    async def check():
        saved_call, saved_allow = app.call_crawl4ai, app.is_allowed_public_url
        app.call_crawl4ai, app.is_allowed_public_url = fake_call, allow
        try:
            task = asyncio.create_task(
                app.crawl_all(None, [a, b], deadline=asyncio.get_running_loop().time() + 1)
            )
            await asyncio.wait_for(final_started.wait(), timeout=0.2)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            else:
                raise AssertionError("external cancellation was swallowed")
        finally:
            app.call_crawl4ai, app.is_allowed_public_url = saved_call, saved_allow
        assert cancelled == [final_url], cancelled

    run(check())


def test_failed_or_blocked_content_is_rejected_but_short_answers_survive():
    content = "Useful content despite the provider status."
    assert app.extract_crawl_content({"success": False, "results": [crawl_item("https://a.example/", content)]}) == ""
    assert app.extract_crawl_content(crawl_item("https://a.example/", content, status_code=404)) == ""
    assert app.extract_crawl_content(
        crawl_item("https://a.example/", content, response_headers={"Content-Type": "image/png"})
    ) == ""
    assert app.extract_crawl_content(
        crawl_item("https://a.example/", content, response_headers={"Content-Type": "application/ld+json"})
    ) == content
    assert app.extract_crawl_content(crawl_item("https://a.example/", content, status_code=None)) == content
    assert app.extract_crawl_content(
        crawl_item("https://a.example/", content, response_headers={"cf-mitigated": "challenge"})
    ) == ""
    assert app.extract_crawl_content(
        crawl_item(
            "https://a.example/",
            "A real article explains how captcha and Cloudflare challenges work.",
            metadata={"title": "Anti-bot systems explained"},
        )
    ) == "A real article explains how captcha and Cloudflare challenges work."
    assert app.extract_crawl_content(crawl_item("https://a.example/", "42")) == "42"


def test_direct_crawl_rejects_failed_result_and_private_final_url():
    request = app.CrawlRequest(url="https://public.example/")

    async def run_case(payload, allow):
        saved_call, saved_allow = app.call_crawl4ai, app.is_allowed_public_url

        async def fake_call(_client, _payload):
            return payload

        app.call_crawl4ai, app.is_allowed_public_url = fake_call, allow
        try:
            return await app.crawl_direct_url(None, request)
        finally:
            app.call_crawl4ai, app.is_allowed_public_url = saved_call, saved_allow

    async def check():
        failed = {"success": True, "results": [crawl_item(request.url, "failed body", success=False)]}
        try:
            await run_case(failed, allow_public)
        except app.AppError as exc:
            assert exc.status_code == 502, exc.status_code
        else:
            raise AssertionError("failed CrawlResult content was accepted")

        private = {"success": True, "results": [crawl_item(
            request.url, "private redirect body", redirected_url="http://127.0.0.1/admin"
        )]}

        async def public_request_only(url):
            return url == request.url

        try:
            await run_case(private, public_request_only)
        except app.AppError as exc:
            assert exc.code == "invalid_crawled_url", exc.code
        else:
            raise AssertionError("private final URL was accepted")

    run(check())


def test_search_paths_reject_private_final_urls_without_recrawling():
    item = candidate("https://public.example/", "A")
    payload = {
        "success": True,
        "results": [crawl_item(item["url"], "private redirect body", redirected_url="http://127.0.0.1/admin")],
    }
    calls = []

    async def fake_call(_client, request):
        calls.append(request["urls"])
        return payload

    async def allow_request_only(url):
        return url == item["url"]

    async def check():
        saved_call, saved_allow = app.call_crawl4ai, app.is_allowed_public_url
        app.call_crawl4ai, app.is_allowed_public_url = fake_call, allow_request_only
        try:
            try:
                await app.crawl_all(None, [item])
            except app.AppError as exc:
                assert exc.status_code == 502, exc.status_code
            else:
                raise AssertionError("search accepted a private final URL")
            assert calls == [[item["url"]]], calls
            assert await app.crawl_url(None, item) is None
        finally:
            app.call_crawl4ai, app.is_allowed_public_url = saved_call, saved_allow

    run(check())


def test_failed_crawls_are_retryable_but_no_candidates_remain_empty():
    only = candidate("https://a.example/", "A")

    async def fake_call(_client, payload):
        row = crawl_item(payload["urls"][0], "body from failed crawl", success=False)
        return {"success": True, "results": [row]}

    async def check():
        saved_call, saved_allow = app.call_crawl4ai, app.is_allowed_public_url
        app.call_crawl4ai, app.is_allowed_public_url = fake_call, allow_public
        try:
            try:
                await app.crawl_all(None, [only])
            except app.AppError as exc:
                assert exc.status_code == 502, exc.status_code
                assert exc.code == "crawl4ai_error", exc.code
                assert exc.retryable is True, exc.retryable
                assert exc.retry_after == app.OUTAGE_RETRY_AFTER, exc.retry_after
            else:
                raise AssertionError("failed target became a successful empty crawl")
        finally:
            app.call_crawl4ai, app.is_allowed_public_url = saved_call, saved_allow

        saved_client, saved_discover = app.get_client, app.discover_candidates
        app.get_client = lambda: asyncio.sleep(0, result=None)
        app.discover_candidates = lambda *_args: asyncio.sleep(0, result=[])
        try:
            result = await app.search(app.SearchRequest(query="no matches"))
        finally:
            app.get_client, app.discover_candidates = saved_client, saved_discover
        assert result.data.results == [], result

    run(check())


def demo():
    for name, check in sorted(globals().items()):
        if name.startswith("test_"):
            check()
            print(f"ok {name}")
    print("all checks passed")


if __name__ == "__main__":
    demo()
