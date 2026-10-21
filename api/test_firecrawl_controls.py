"""Mocked shared-budget, policy, and concurrency checks for #47."""

import asyncio
import json
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import httpx

import firecrawl_controls as fc


def run(coro):
    return asyncio.run(coro)


def controls(root, groups):
    budgets = tuple(fc._Budget(name, tuple(keys)) for name, keys in groups)
    root = Path(root)
    return fc.FirecrawlControls(
        budgets,
        state_path=root / "state.json",
        lock_path=root / "lock",
    )


def response(request, status=200, body=None, headers=None):
    return httpx.Response(status, json=body or {"success": True}, headers=headers, request=request)


def test_missing_budget_short_circuits_without_network():
    calls = []

    def handler(request):
        calls.append(request)
        raise AssertionError("unconfigured Firecrawl must not make a network request")

    async def check():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            try:
                await fc.FirecrawlControls(()).request(client, "/v2/search", {"query": "q"}, timeout=1)
            except fc.FirecrawlUnavailable as exc:
                assert (exc.status, exc.category, exc.retry_after, exc.retryable) == (
                    None, "not_configured", None, False
                )
            else:
                raise AssertionError("unconfigured Firecrawl request was allowed")

    run(check())
    assert calls == [], calls


def test_budget_parser_requires_explicit_groups_and_rejects_duplicate_keys():
    groups = fc._parse_budgets('[{"id":"primary","keys":["key-a","key-b"]}]')
    assert groups[0].id == "primary" and groups[0].keys == ("key-a", "key-b"), groups
    try:
        fc._parse_budgets('[{"id":"a","keys":["same"]},{"id":"b","keys":["same"]}]')
    except ValueError:
        pass
    else:
        raise AssertionError("same credential cannot be declared as two independent budgets")


def test_search_policy_merges_domain_exclusions_without_mutating_payload():
    seen = []
    payload = {"query": "q", "excludeDomains": ["example.org"]}

    def handler(request):
        seen.append(request)
        return response(request)

    async def check():
        with tempfile.TemporaryDirectory() as tmp:
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                result = await controls(tmp, [("team", ["fc-test"])]).request(
                    client, "/v2/search", payload, timeout=1
                )
                assert result.status_code == 200

    run(check())
    assert payload == {"query": "q", "excludeDomains": ["example.org"]}, payload
    assert len(seen) == 1
    body = json.loads(seen[0].content)
    assert "example.org" in body["excludeDomains"], body
    assert {"linkedin.com", "x.com", "twitter.com", "t.co"}.issubset(body["excludeDomains"]), body


def test_search_never_embeds_scraping_and_scrape_stays_markdown_only():
    seen = []

    def handler(request):
        seen.append(request)
        return response(request)

    async def check():
        with tempfile.TemporaryDirectory() as tmp:
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                instance = controls(tmp, [("team", ["fc-test"])])
                try:
                    await instance.request(
                        client,
                        "/v2/search",
                        {"query": "q", "scrapeOptions": {"formats": ["markdown"]}},
                        timeout=1,
                    )
                except fc.FirecrawlUnavailable as exc:
                    assert exc.category == "embedded_scrape_forbidden", exc.category
                else:
                    raise AssertionError("search call embedded billable scraping")

                result = await instance.request(
                    client, "/v2/scrape", {"url": "https://example.org"}, timeout=1
                )
                assert result.status_code == 200

    run(check())
    assert len(seen) == 1
    body = json.loads(seen[0].content)
    assert body == {
        "url": "https://example.org",
        "formats": ["markdown"],
        "parsers": [],
        "storeInCache": False,
    }, body


def test_expensive_scrape_options_fail_before_network():
    calls = []

    def handler(request):
        calls.append(request)
        return response(request)

    async def check():
        with tempfile.TemporaryDirectory() as tmp:
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                instance = controls(tmp, [("team", ["fc-test"])])
                for extra in (
                    {"formats": ["json"]},
                    {"parsers": ["pdf"]},
                    {"zeroDataRetention": True},
                    {"lockdown": True},
                    {"storeInCache": True},
                ):
                    try:
                        await instance.request(
                            client, "/v2/scrape", {"url": "https://example.org", **extra}, timeout=1
                        )
                    except fc.FirecrawlUnavailable as exc:
                        assert exc.category in {"unsupported_cost_options", "unsupported_privacy_option"}, exc.category
                    else:
                        raise AssertionError(f"cost option was not rejected: {extra}")

    run(check())
    assert calls == [], calls


def test_provider_routed_hosts_and_obvious_pdfs_are_blocked_before_scrape():
    calls = []

    def handler(request):
        calls.append(request)
        return response(request)

    async def denied(url):
        with tempfile.TemporaryDirectory() as tmp:
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                try:
                    await controls(tmp, [("team", ["fc-test"])]).request(
                        client, "/v2/scrape", {"url": url}, timeout=1
                    )
                except fc.FirecrawlUnavailable as exc:
                    assert exc.category == "provider_routed_domain", exc.category
                else:
                    raise AssertionError(f"Firecrawl accepted a blocked target: {url}")

    assert not fc.firecrawl_url_allowed("https://www.linkedin.com/in/person")
    assert not fc.firecrawl_url_allowed("https://x.com/post/1")
    assert not fc.firecrawl_url_allowed("https://example.com/report.PDF?download=1")
    assert not fc.firecrawl_url_allowed("file:///etc/passwd")
    assert fc.firecrawl_url_allowed("https://docs.example.com/guide")
    run(denied("https://www.linkedin.com/in/person"))
    run(denied("https://example.com/report.pdf"))
    assert calls == [], calls


def test_idna_provider_host_aliases_are_blocked_before_scrape():
    calls = []

    def handler(request):
        calls.append(request)
        return response(request)

    async def check():
        with tempfile.TemporaryDirectory() as tmp:
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                instance = controls(tmp, [("team", ["fc-test"])])
                for url in (
                    "https://linkedin。com/in/person",
                    "https://linkedin．com/in/person",
                    "https://ｌｉｎｋｅｄｉｎ.com/in/person",
                    "https://linkedin.com./in/person",
                    "https://bad\ud800.linkedin.com/in/person",
                ):
                    try:
                        await instance.request(client, "/v2/scrape", {"url": url}, timeout=1)
                    except fc.FirecrawlUnavailable as exc:
                        assert exc.category == "provider_routed_domain", exc.category
                    else:
                        raise AssertionError(f"Firecrawl accepted a restricted IDNA host: {url!r}")
                assert calls == [], calls

                result = await instance.request(
                    client, "/v2/scrape", {"url": "https://docs.example.com/guide"}, timeout=1
                )
                assert result.status_code == 200, result.status_code

    run(check())
    assert len(calls) == 1 and calls[0].url.host == "api.firecrawl.dev", calls


def test_team_exhaustion_fails_over_only_to_an_independent_budget():
    seen = []

    def handler(request):
        key = request.headers["Authorization"]
        seen.append((request.url.path, key))
        if key == "Bearer same-team-primary" and request.url.path == "/v2/search":
            return response(request, 402, {"success": False, "error": "Insufficient credits"})
        if key == "Bearer same-team-primary" and request.url.path == fc.CREDIT_USAGE_PATH:
            return response(request, 200, {"success": True, "data": {"remainingCredits": 0, "billingPeriodEnd": None}})
        if key == "Bearer independent" and request.url.path == "/v2/search":
            return response(request, 200, {"success": True, "data": {"web": {"results": []}}})
        raise AssertionError(f"unexpected request {request.url} {key}")

    async def check():
        with tempfile.TemporaryDirectory() as tmp:
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                result = await controls(
                    tmp,
                    [("team-a", ["same-team-primary", "same-team-sibling"]), ("team-b", ["independent"])],
                ).request(client, "/v2/search", {"query": "q"}, timeout=2)
                assert result.status_code == 200

    run(check())
    assert seen == [
        ("/v2/search", "Bearer same-team-primary"),
        (fc.CREDIT_USAGE_PATH, "Bearer same-team-primary"),
        ("/v2/search", "Bearer independent"),
    ], seen


def test_invalid_auth_fails_over_with_shared_cooldown_and_same_key_recovery():
    for endpoint, payload in (
        ("/v2/search", {"query": "q"}),
        ("/v2/scrape", {"url": "https://example.org"}),
    ):
        seen = []
        primary_status = 401
        groups = [("team-a", ["primary", "sibling"]), ("team-b", ["independent"])]

        def handler(request):
            key = request.headers["Authorization"]
            seen.append((request.url.path, key))
            assert request.method == "POST", "invalid auth must not trigger usage probes"
            return response(request, primary_status if key == "Bearer primary" else 200)

        async def check():
            nonlocal primary_status
            with tempfile.TemporaryDirectory() as tmp:
                async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                    instance = controls(tmp, groups)
                    for _ in range(2):
                        # A separate worker must also skip the cached bad primary.
                        result = await controls(tmp, groups).request(client, endpoint, payload, timeout=2)
                        assert result.status_code == 200
                    state = instance._read_state()
                    assert state["budgets"]["team-a"]["blocked"] == "invalid_auth", state
                    state["budgets"]["team-a"]["next_check"] = time.monotonic() - 1
                    instance._write_state(state)
                    # Still invalid: one same-key attempt re-blocks, then callers skip it again.
                    for _ in range(2):
                        assert (await instance.request(client, endpoint, payload, timeout=2)).status_code == 200
                    state = instance._read_state()
                    assert state["budgets"]["team-a"]["next_check"] > time.monotonic(), state
                    state["budgets"]["team-a"]["next_check"] = time.monotonic() - 1
                    instance._write_state(state)
                    primary_status = 200
                    assert (await instance.request(client, endpoint, payload, timeout=2)).status_code == 200
                    assert instance._read_state()["budgets"]["team-a"] == {}

        run(check())
        assert seen == [(endpoint, "Bearer " + key) for key in (
            "primary", "independent", "independent", "primary", "independent", "independent", "primary",
        )], seen


def test_all_skipped_budgets_preserve_the_last_failure_and_safe_metadata():
    secrets = ("primary-secret-61", "backup-secret-61", "body-secret-61", "header-secret-61", "url-secret-61")
    for statuses in ((401, 401), (402, 401), (401, 402), (402, 402)):
        seen = []

        def handler(request):
            key = request.headers["Authorization"]
            seen.append((request.url.path, key))
            if request.url.path == fc.CREDIT_USAGE_PATH:
                return response(request, body={"success": True, "data": {"remainingCredits": 0}})
            status = statuses[0] if key == "Bearer " + secrets[0] else statuses[1]
            return response(request, status, {"error": secrets[2]}, {"X-Private": secrets[3], "Retry-After": "Infinity"})

        async def check():
            with tempfile.TemporaryDirectory() as tmp:
                groups = [("a", [secrets[0], "sibling"]), ("b", [secrets[1]])]
                async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                    for _ in range(2):
                        try:
                            await controls(tmp, groups).request(
                                client, "/v2/scrape", {"url": "https://example.org/" + secrets[4]}, timeout=2
                            )
                        except fc.FirecrawlUnavailable as exc:
                            expected = "invalid_auth" if statuses[-1] == 401 else "team_exhausted"
                            assert (exc.status, exc.category) == (statuses[-1], expected), vars(exc)
                            assert str(exc) == expected
                            assert exc.retryable is True
                            assert 1 <= exc.retry_after <= fc.USAGE_RECHECK_SECONDS, exc.retry_after
                            if statuses[-1] == 402:
                                assert exc.retry_after == fc.USAGE_RECHECK_SECONDS, exc.retry_after
                        else:
                            raise AssertionError("all unavailable budgets returned success")
                    state = json.dumps(controls(tmp, groups)._read_state())
                    assert not any(secret in state for secret in secrets), state

        with patch.object(fc, "logger") as logs:
            run(check())
        logged = str(logs.mock_calls)
        assert "category=invalid_auth" in logged or statuses == (402, 402), logged
        assert not any(secret in logged for secret in (*secrets, "Authorization", "Bearer", "https://")), logged
        expected_calls = []
        for key, status in zip(secrets[:2], statuses):
            expected_calls.append(("/v2/scrape", "Bearer " + key))
            if status == 402:
                expected_calls.append((fc.CREDIT_USAGE_PATH, "Bearer " + key))
        assert seen == expected_calls, seen


def test_invalid_auth_does_not_bypass_later_cap_policy_rate_or_provider_errors():
    for status, category in ((402, "key_cap_or_ambiguous"), (403, "policy_denied"),
                             (429, "rate_limited"), (500, "provider_unavailable")):
        seen = []

        def handler(request):
            key = request.headers["Authorization"]
            seen.append((request.url.path, key))
            if request.url.path == fc.CREDIT_USAGE_PATH:
                return response(request, body={"success": True, "data": {"remainingCredits": 3}})
            return response(request, 401 if key == "Bearer primary" else status)

        async def check():
            with tempfile.TemporaryDirectory() as tmp:
                async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                    try:
                        await controls(tmp, [("a", ["primary", "sibling"]), ("b", ["backup", "other"]),
                                             ("c", ["unused"])]).request(
                            client, "/v2/search", {"query": "q"}, timeout=2
                        )
                    except fc.FirecrawlUnavailable as exc:
                        assert (exc.status, exc.category) == (status, category), vars(exc)
                    else:
                        raise AssertionError("failover bypassed a non-rotating failure")

        run(check())
        expected = [("/v2/search", "Bearer primary"), ("/v2/search", "Bearer backup")]
        if status == 402:
            expected.append((fc.CREDIT_USAGE_PATH, "Bearer backup"))
        assert seen == expected, seen


def test_auth_recovery_does_not_fail_over_after_policy_or_provider_errors():
    for status, category in ((403, "policy_denied"), (500, "provider_unavailable")):
        seen = []
        primary_status = 401

        def handler(request):
            key = request.headers["Authorization"]
            seen.append(key)
            return response(request, primary_status if key == "Bearer primary" else 200)

        async def check():
            nonlocal primary_status
            with tempfile.TemporaryDirectory() as tmp:
                async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                    instance = controls(tmp, [("a", ["primary"]), ("b", ["backup"])])
                    assert (await instance.request(client, "/v2/search", {"query": "q"}, timeout=2)).status_code == 200
                    state = instance._read_state()
                    state["budgets"]["a"]["next_check"] = time.monotonic() - 1
                    instance._write_state(state)
                    primary_status = status
                    for _ in range(2):
                        try:
                            await instance.request(client, "/v2/search", {"query": "q"}, timeout=2)
                        except fc.FirecrawlUnavailable as exc:
                            assert (exc.status, exc.category) == (status, category), vars(exc)
                        else:
                            raise AssertionError("stale auth guard bypassed a later non-rotating failure")

        run(check())
        assert seen == ["Bearer primary", "Bearer backup", "Bearer primary", "Bearer primary"], seen


def test_auth_failover_uses_the_remaining_operation_deadline():
    clock = [0.0]
    seen = []

    def handler(request):
        key = request.headers["Authorization"]
        seen.append(key)
        if key == "Bearer primary":
            clock[0] = 1.0
            return response(request, 401)
        return response(request)

    async def check():
        with tempfile.TemporaryDirectory() as tmp:
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                instance = controls(tmp, [("a", ["primary"]), ("b", ["backup"])])
                with patch.object(instance, "_acquire_lock", wraps=instance._acquire_lock) as acquire, patch.object(
                    instance, "_release_lock", wraps=instance._release_lock
                ) as release:
                    assert (await instance.request(client, "/v2/search", {"query": "q"}, timeout=2)).status_code == 200
                    assert acquire.call_count == release.call_count == 1
                    assert acquire.call_args.args == (2.0,)

    with patch.object(asyncio, "get_running_loop") as get_loop, patch.object(
        asyncio, "wait_for", wraps=asyncio.wait_for
    ) as wait_for:
        get_loop.return_value.time.side_effect = lambda: clock[0]
        run(check())
    assert [call.kwargs["timeout"] for call in wait_for.call_args_list] == [2.0, 1.0]
    assert seen == ["Bearer primary", "Bearer backup"], seen


def test_positive_balance_blocks_during_cooldown_then_retries_same_key():
    seen = []
    posts = 0

    def handler(request):
        nonlocal posts
        key = request.headers["Authorization"]
        seen.append((request.url.path, key))
        if request.url.path == "/v2/search":
            posts += 1
            if posts == 1:
                return response(request, 402, {"success": False, "error": "Insufficient credits"})
            return response(request, 200)
        if request.url.path == fc.CREDIT_USAGE_PATH:
            return response(request, 200, {"success": True, "data": {"remainingCredits": 3}})
        raise AssertionError(f"unexpected request {request.url}")

    async def check():
        with tempfile.TemporaryDirectory() as tmp:
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                instance = controls(
                    tmp,
                    [("team-a", ["primary", "sibling"]), ("team-b", ["independent"])],
                )
                try:
                    await instance.request(client, "/v2/search", {"query": "q"}, timeout=2)
                except fc.FirecrawlUnavailable as exc:
                    assert exc.category == "key_cap_or_ambiguous", exc.category
                else:
                    raise AssertionError("positive-balance 402 was treated as team exhaustion")

                try:
                    await instance.request(client, "/v2/search", {"query": "q"}, timeout=2)
                except fc.FirecrawlUnavailable as exc:
                    assert exc.category == "key_cap_or_ambiguous", exc.category
                else:
                    raise AssertionError("request bypassed the positive-balance cooldown")

                state = instance._read_state()
                state["budgets"]["team-a"]["next_check"] = time.monotonic() - 1
                instance._write_state(state)
                result = await instance.request(client, "/v2/search", {"query": "q"}, timeout=2)
                assert result.status_code == 200, result.status_code

    run(check())
    assert seen == [
        ("/v2/search", "Bearer primary"),
        (fc.CREDIT_USAGE_PATH, "Bearer primary"),
        (fc.CREDIT_USAGE_PATH, "Bearer primary"),
        ("/v2/search", "Bearer primary"),
    ], seen


def test_unavailable_usage_is_ambiguous_and_fails_closed():
    seen = []

    def handler(request):
        seen.append((request.url.path, request.headers["Authorization"]))
        if request.url.path == "/v2/search":
            return response(request, 402, {"success": False, "error": "Insufficient credits"})
        return response(request, 500, {"success": False, "error": "private diagnostic"})

    async def check():
        with tempfile.TemporaryDirectory() as tmp:
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                try:
                    await controls(tmp, [("a", ["key-a"]), ("b", ["key-b"]) ]).request(
                        client, "/v2/search", {"query": "q"}, timeout=2
                    )
                except fc.FirecrawlUnavailable as exc:
                    assert exc.category == "key_cap_or_ambiguous", exc.category
                    assert "private diagnostic" not in str(exc)
                else:
                    raise AssertionError("unverified 402 was treated as exhausted-team failover")

    run(check())
    assert len(seen) == 2, seen
    assert all(key != "Bearer key-b" for _, key in seen), seen


def test_malformed_usage_never_confirms_team_exhaustion():
    calls = []

    def handler(request):
        calls.append(request.url.path)
        if request.url.path == "/v2/search":
            return response(request, 402, {"success": False, "error": "Insufficient credits"})
        return response(request, 200, {"success": True, "data": {"remainingCredits": "unknown"}})

    async def check():
        with tempfile.TemporaryDirectory() as tmp:
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                try:
                    await controls(tmp, [("team-a", ["key-a"]), ("team-b", ["key-b"]) ]).request(
                        client, "/v2/search", {"query": "q"}, timeout=2
                    )
                except fc.FirecrawlUnavailable as exc:
                    assert exc.category == "key_cap_or_ambiguous", exc.category
                else:
                    raise AssertionError("malformed usage was treated as an exhausted team")

    run(check())
    assert calls == ["/v2/search", fc.CREDIT_USAGE_PATH], calls


def test_top_up_recovers_exhausted_budget():
    seen = []
    usage_remaining = [0, 50]

    def handler(request):
        seen.append(request.url.path)
        if request.url.path == "/v2/search":
            if len([path for path in seen if path == "/v2/search"]) == 1:
                return response(request, 402, {"success": False, "error": "Insufficient credits"})
            return response(request, 200)
        remaining = usage_remaining.pop(0)
        return response(request, 200, {"success": True, "data": {"remainingCredits": remaining, "billingPeriodEnd": None}})

    async def check():
        with tempfile.TemporaryDirectory() as tmp:
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                instance = controls(tmp, [("team", ["key"])])
                for _ in range(2):
                    try:
                        await instance.request(client, "/v2/search", {"query": "q"}, timeout=2)
                    except fc.FirecrawlUnavailable as exc:
                        assert exc.category == "team_exhausted", exc.category
                    else:
                        raise AssertionError("zero-credit team was not blocked")
                state = instance._read_state()
                state["budgets"]["team"]["next_check"] = time.monotonic() - 1
                instance._write_state(state)
                result = await instance.request(client, "/v2/search", {"query": "q"}, timeout=2)
                assert result.status_code == 200

    run(check())
    assert seen == ["/v2/search", fc.CREDIT_USAGE_PATH, fc.CREDIT_USAGE_PATH, "/v2/search"], seen


class SlowTransport(httpx.AsyncBaseTransport):
    def __init__(self):
        self.active = 0
        self.max_active = 0
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def handle_async_request(self, request):
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        self.entered.set()
        await self.release.wait()
        self.active -= 1
        return response(request)


def test_provider_call_is_cancelled_at_the_shared_deadline():
    class DelayedTransport(httpx.AsyncBaseTransport):
        def __init__(self):
            self.cancelled = asyncio.Event()

        async def handle_async_request(self, request):
            try:
                await asyncio.sleep(0.2)
            except asyncio.CancelledError:
                self.cancelled.set()
                raise
            return response(request)

    async def check():
        with tempfile.TemporaryDirectory() as tmp:
            transport = DelayedTransport()
            async with httpx.AsyncClient(transport=transport) as client:
                try:
                    await controls(tmp, [("team", ["key"]) ]).request(
                        client, "/v2/search", {"query": "q"}, timeout=0.02
                    )
                except fc.FirecrawlUnavailable as exc:
                    assert exc.category == "transport_timeout", exc.category
                    assert exc.retryable is True
                else:
                    raise AssertionError("progressing response exceeded its total deadline")
                assert transport.cancelled.is_set(), "deadline did not cancel the provider request"

    # Freeze budget accounting, not wait_for's real event-loop cancellation.
    with patch.object(asyncio, "get_running_loop") as get_loop, patch.object(
        asyncio, "wait_for", wraps=asyncio.wait_for
    ) as wait_for:
        get_loop.return_value.time.return_value = 0.0
        run(check())
    timeouts = [call.kwargs["timeout"] for call in wait_for.call_args_list]
    assert timeouts == [0.02], timeouts


def test_credit_usage_check_is_bounded_by_remaining_deadline():
    usage_cancelled = []
    clock = [0.0]

    class DelayedUsageTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            if request.url.path == "/v2/search":
                clock[0] = 0.02
                return response(request, 402, {"success": False, "error": "Insufficient credits"})
            try:
                await asyncio.sleep(0.2)
            except asyncio.CancelledError:
                usage_cancelled.append(True)
                raise
            return response(request, 200, {"success": True, "data": {"remainingCredits": 0}})

    async def check():
        with tempfile.TemporaryDirectory() as tmp:
            async with httpx.AsyncClient(transport=DelayedUsageTransport()) as client:
                try:
                    await controls(tmp, [("team", ["key"]) ]).request(
                        client, "/v2/search", {"query": "q"}, timeout=0.03
                    )
                except fc.FirecrawlUnavailable as exc:
                    assert exc.category == "key_cap_or_ambiguous", exc.category
                    assert exc.retryable is True
                else:
                    raise AssertionError("timed-out usage check was treated as balance evidence")

    with patch.object(asyncio, "get_running_loop") as get_loop, patch.object(
        asyncio, "wait_for", wraps=asyncio.wait_for
    ) as wait_for:
        get_loop.return_value.time.side_effect = lambda: clock[0]
        run(check())
    timeouts = [call.kwargs["timeout"] for call in wait_for.call_args_list]
    assert len(timeouts) == 2, timeouts
    assert abs(timeouts[0] - 0.03) < 1e-9, timeouts
    assert abs(timeouts[1] - 0.01) < 1e-9, timeouts
    assert usage_cancelled == [True], usage_cancelled


def test_nonfinite_retry_after_uses_bounded_cooldown():
    for value in ("NaN", "Infinity", "-Infinity"):
        seen = []

        def handler(request):
            seen.append(request)
            return response(request, 429, {"success": False}, {"Retry-After": value})

        async def check():
            with tempfile.TemporaryDirectory() as tmp:
                async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                    instance = controls(tmp, [("team", ["primary", "sibling"]), ("other", ["independent"])])
                    for attempt in range(2):
                        try:
                            await instance.request(client, "/v2/search", {"query": "q"}, timeout=2)
                        except fc.FirecrawlUnavailable as exc:
                            assert exc.category == "rate_limited", exc.category
                            if attempt == 0:
                                assert exc.retry_after == 60, exc.retry_after
                            else:
                                assert 1 <= exc.retry_after <= 60, exc.retry_after
                        else:
                            raise AssertionError(f"Retry-After {value} disabled the shared cooldown")

        run(check())
        assert len(seen) == 1, (value, len(seen))


def test_two_controls_share_one_inflight_limit_and_timeout_waiting_requests():
    async def check():
        with tempfile.TemporaryDirectory() as tmp:
            transport = SlowTransport()
            clients = [httpx.AsyncClient(transport=transport) for _ in range(2)]
            instances = [controls(tmp, [("team", ["key"])]) for _ in range(2)]
            first = asyncio.create_task(instances[0].request(clients[0], "/v2/search", {"query": "q"}, timeout=1))
            try:
                await transport.entered.wait()
                try:
                    await instances[1].request(clients[1], "/v2/search", {"query": "q"}, timeout=0.01)
                except fc.FirecrawlUnavailable as exc:
                    assert exc.category == "team_concurrency", exc.category
                else:
                    raise AssertionError("waiting request exceeded its deadline")
                transport.release.set()
                assert (await first).status_code == 200
            finally:
                transport.release.set()
                await asyncio.gather(first, return_exceptions=True)
                await asyncio.gather(*(client.aclose() for client in clients))
            assert transport.max_active == 1, transport.max_active

    run(check())


def test_forbidden_policy_status_does_not_fail_over():
    seen = []

    def handler(request):
        seen.append(request.headers["Authorization"])
        return response(request, 403, {"success": False, "error": "Forbidden"})

    async def check():
        with tempfile.TemporaryDirectory() as tmp:
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                try:
                    await controls(tmp, [("a", ["primary"]), ("b", ["backup"]) ]).request(
                        client, "/v2/scrape", {"url": "https://example.org"}, timeout=2
                    )
                except fc.FirecrawlUnavailable as exc:
                    assert exc.category == "policy_denied", exc.category
                    assert exc.status == 403
                else:
                    raise AssertionError("provider policy response was not surfaced")

    run(check())
    assert seen == ["Bearer primary"], seen


def demo():
    for name, check in sorted(globals().items()):
        if name.startswith("test_"):
            check()
            print(f"ok {name}")
    print("all checks passed")


if __name__ == "__main__":
    demo()
