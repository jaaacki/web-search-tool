"""Mocked key-pool, shared guard, policy, and concurrency checks."""

import asyncio
import json
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import httpx

import firecrawl_controls as fc
import observability as obs


def run(coro):
    return asyncio.run(coro)


def controls(root, entries, **kwargs):
    budgets = fc._parse_keys(json.dumps([{"label": label, "key": key} for label, key in entries]))
    root = Path(root)
    return fc.FirecrawlControls(
        budgets,
        state_path=root / "state.json",
        lock_path=root / "lock",
        **kwargs,
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


def test_labeled_and_singleton_legacy_inputs_preserve_order_and_credential_identity():
    slots = fc._parse_keys('[{"label":" first@example.test ","key":"secret-key-a"},'
                           '{"label":"secondary@example.test","key":"secret-key-b"}]')
    legacy = fc._parse_budgets('[{"id":"first@example.test","keys":["secret-key-a"]},'
                               '{"id":"secondary@example.test","keys":["secret-key-b"]}]')
    assert slots == legacy
    assert [slot.label for slot in slots] == ["first@example.test", "secondary@example.test"]
    assert [slot.keys for slot in slots] == [("secret-key-a",), ("secret-key-b",)]
    reordered = fc._parse_keys('[{"label":"renamed@example.test","key":"secret-key-b"},'
                               '{"label":"first@example.test","key":"replacement-key"}]')
    assert reordered[0].id == slots[1].id and reordered[0].label != slots[1].label
    assert reordered[1].id not in {slot.id for slot in slots}
    assert all(len(slot.id) == 64 and set(slot.id) <= set("0123456789abcdef") for slot in slots)
    assert "secret-key" not in repr(slots) and "first@example.test" in repr(slots)


def test_required_labels_and_unique_credentials_reject_unsafe_or_ambiguous_input():
    valid = {"label": "first@example.test", "key": "secret-key"}
    invalid = (None, {}, ["secret-key"], [{"key": "secret-key"}],
               *[[{"label": label, "key": "secret-key"}] for label in (None, 3, "", " ")])
    for value in (*invalid,
                  *[[{**valid, "label": label}] for label in (
                      "\nfirst@example.test", "primary\t@example.test", "primary\u001b@example.test",
                      "primary\u007f@example.test", "primary\u0085@example.test", "first @example.test")],
                  [valid, {"label": " first@example.test ", "key": "other-secret"}],
                  [valid, {"label": "secondary@example.test", "key": "secret-key"}],
                  [{"label": "first@example.test"}], [{**valid, "extra": True}],
                  *[[{**valid, "key": key}] for key in (None, 1, "", "key a", " key", "key\u0000a", "kéy")],
                  [{**valid, "label": "secret-key@example.test"}],
                  [valid, {"label": "secret-key@example.test", "key": "other-secret"}]):
        try:
            fc._parse_keys(json.dumps(value))
        except ValueError as exc:
            assert "secret-key" not in str(exc) and "example.test" not in str(exc), str(exc)
        else:
            raise AssertionError("invalid labeled key list accepted")
    for raw in ('secret-key,other-secret', '"secret-key"'):
        try:
            fc._parse_keys(raw)
        except ValueError:
            pass
        else:
            raise AssertionError("unlabeled credentials were accepted")
    for raw in ('[]', '{}', 'not-json', '[{"id":"first@example.test","keys":[]}]',
                '[{"id":"first@example.test","keys":["secret-key","other-secret"]}]',
                '[{"id":"","keys":["secret-key"]}]'):
        try:
            fc._parse_budgets(raw)
        except ValueError:
            pass
        else:
            raise AssertionError("ambiguous or unlabeled legacy input accepted")


def test_unknown_or_removed_credential_shaped_labels_fail_before_logging_or_network():
    credentials = (
        "fc-01234567-89ab-4cde-8fab-0123456789ab",
        "fc-" + "0123456789abcdef" * 2,
        "FC-" + "q7R2x9Z4" * 4,
        "0123456789abcdef" * 2,
        "01234567-89ab-4cde-8fab-0123456789ab",
    )
    seen = []

    def handler(request):
        seen.append(request)
        return response(request, 401)

    async def check():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            for index, credential in enumerate(credentials):
                # A formerly configured credential is still valid as a key, never as a label.
                previous = fc._parse_keys(json.dumps([{"label": "previous@example.test", "key": credential}]))
                assert previous[0].keys == (credential,)
                label = credential if index == 0 else f"retired {credential} @example.test"
                raw = json.dumps([{"label": label, "key": "different-secret"}])
                try:
                    fc._parse_keys(raw)
                except ValueError as exc:
                    assert str(exc) == "labels must not contain credentials"
                else:
                    raise AssertionError("credential-shaped label accepted")
                with patch.dict("os.environ", {"FIRECRAWL_KEYS_JSON": raw}, clear=True), patch.object(fc, "logger") as logs:
                    instance = fc._load_controls()
                    assert instance._config_error and not instance._budgets
                    try:
                        await instance.request(client, "/v2/search", {"query": "q"}, timeout=1)
                    except fc.FirecrawlUnavailable as exc:
                        assert str(exc) == "invalid_configuration" and exc.status is None
                    else:
                        raise AssertionError("credential-shaped label reached the provider")
                logged = str(logs.mock_calls)
                assert credential not in logged and label not in logged and "different-secret" not in logged
                assert logs.warning.call_count == 1
            for label in ("first.last+api@example.test", "fc-primary@example.test", "fc-primary"):
                accepted = fc._parse_keys(json.dumps([{"label": label, "key": "different-secret"}]))
                assert accepted[0].label == label

    run(check())
    assert seen == [], seen


def test_environment_precedence_empty_disable_and_sanitized_invalid_configuration():
    legacy = '[{"id":"legacy@example.test","keys":["legacy-secret"]}]'
    labeled = '[{"label":"first@example.test","key":"labeled-secret"}]'
    for raw, expected, invalid in ((None, [("legacy-secret",)], False),
                                   (labeled, [("labeled-secret",)], False),
                                   ("", [], False), ("  ", [], False), ("[]", [], False),
                                   ('["unlabeled-secret"]', [], True)):
        environment = {"FIRECRAWL_BUDGETS_JSON": legacy}
        if raw is not None:
            environment["FIRECRAWL_KEYS_JSON"] = raw
        with patch.dict("os.environ", environment, clear=True), patch.object(fc, "logger") as logs:
            instance = fc._load_controls()
            assert instance._budgets == fc.FirecrawlControls.from_env()._budgets
            assert instance._config_error == invalid
        assert [slot.keys for slot in instance._budgets] == expected
        assert not any(secret in str(logs.mock_calls) for secret in ("secret", "example.test"))
    with patch.dict("os.environ", {"FIRECRAWL_KEYS_JSON": "[]", "FIRECRAWL_BUDGETS_JSON": "invalid"}, clear=True):
        assert not fc._load_controls()._config_error
    with patch.dict("os.environ", {"FIRECRAWL_API_KEYS": "unlabeled-secret"}, clear=True):
        assert not fc._load_controls()._budgets  # The shelved CSV interface is not supported.
    with patch.dict("os.environ", {"FIRECRAWL_BUDGETS_JSON": "invalid"}, clear=True), patch.object(fc, "logger"):
        instance = fc._load_controls()
        assert instance._config_error and not instance._budgets

    def handler(request):
        raise AssertionError("invalid config must not use the network")

    async def check():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            try:
                await instance.request(client, "/v2/search", {"query": "q"}, timeout=1)
            except fc.FirecrawlUnavailable as exc:
                assert (exc.status, str(exc), exc.retryable) == (None, "invalid_configuration", False)
            else:
                raise AssertionError("invalid config was accepted")

    run(check())


def test_search_policy_merges_domain_exclusions_without_mutating_payload():
    seen = []
    payload = {"query": "q", "excludeDomains": ["example.org"]}

    def handler(request):
        seen.append(request)
        return response(request)

    async def check():
        with tempfile.TemporaryDirectory() as tmp:
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                result = await controls(tmp, [("first@example.test", "fc-test")]).request(
                    client, "/v2/search", payload, timeout=1
                )
                assert result.status_code == 200

    run(check())
    assert payload == {"query": "q", "excludeDomains": ["example.org"]}, payload
    assert len(seen) == 1
    body = json.loads(seen[0].content)
    assert "example.org" in body["excludeDomains"], body
    assert {"linkedin.com", "x.com", "twitter.com", "t.co"}.issubset(body["excludeDomains"]), body


def test_search_allows_free_inline_scrape_but_scrape_stays_markdown_only():
    seen = []

    def handler(request):
        seen.append(request)
        return response(request)

    async def check():
        with tempfile.TemporaryDirectory() as tmp:
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                instance = controls(tmp, [("first@example.test", "fc-test")])
                result = await instance.request(
                    client,
                    "/v2/search",
                    {
                        "query": "q",
                        "scrapeOptions": {"formats": ["markdown"], "onlyMainContent": True},
                    },
                    timeout=1,
                )
                assert result.status_code == 200

                try:
                    await instance.request(
                        client,
                        "/v2/search",
                        {"query": "q", "scrapeOptions": {"zeroDataRetention": True}},
                        timeout=1,
                    )
                except fc.FirecrawlUnavailable as exc:
                    assert exc.category == "unsupported_enterprise_option", exc.category
                else:
                    raise AssertionError("enterprise scrape option reached the provider")

                result = await instance.request(
                    client, "/v2/scrape", {"url": "https://example.org"}, timeout=1
                )
                assert result.status_code == 200

    run(check())
    assert len(seen) == 2
    search_body = json.loads(seen[0].content)
    assert search_body["scrapeOptions"] == {"formats": ["markdown"], "onlyMainContent": True}
    body = json.loads(seen[1].content)
    assert body == {
        "url": "https://example.org",
        "formats": ["markdown"],
        "parsers": [],
        "storeInCache": False,
    }, body


def test_free_scrape_options_reach_provider_and_enterprise_options_fail_before_network():
    calls = []

    def handler(request):
        calls.append(json.loads(request.content))
        return response(request)

    async def check():
        with tempfile.TemporaryDirectory() as tmp:
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                instance = controls(tmp, [("first@example.test", "fc-test")])
                allowed = (
                    {"formats": ["json"]},
                    {"parsers": ["pdf"]},
                    {"lockdown": True},
                    {"storeInCache": True},
                    {"redactPII": True},
                    {"checkPromptInjection": True},
                )
                for extra in allowed:
                    await instance.request(
                        client, "/v2/scrape", {"url": "https://example.org", **extra}, timeout=1
                    )
                for extra in ({"zeroDataRetention": True}, {"threatProtection": {"mode": "normal"}}):
                    try:
                        await instance.request(
                            client, "/v2/scrape", {"url": "https://example.org", **extra}, timeout=1
                        )
                    except fc.FirecrawlUnavailable as exc:
                        assert exc.category == "unsupported_enterprise_option", exc.category
                    else:
                        raise AssertionError(f"enterprise option was not rejected: {extra}")

    run(check())
    assert len(calls) == 6, calls
    assert calls[0]["formats"] == ["json"], calls
    assert calls[1]["parsers"] == ["pdf"], calls
    assert calls[2]["lockdown"] is True, calls
    assert calls[3]["storeInCache"] is True, calls
    assert calls[4]["redactPII"] is True, calls
    assert calls[5]["checkPromptInjection"] is True, calls


def test_provider_routed_hosts_and_obvious_pdfs_are_blocked_before_scrape():
    calls = []

    def handler(request):
        calls.append(request)
        return response(request)

    async def denied(url):
        with tempfile.TemporaryDirectory() as tmp:
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                try:
                    await controls(tmp, [("first@example.test", "fc-test")]).request(
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
                instance = controls(tmp, [("first@example.test", "fc-test")])
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


def test_verified_zero_or_negative_usage_tries_all_keys_including_same_team_siblings():
    seen = []

    def handler(request):
        key = request.headers["Authorization"]
        seen.append((request.url.path, key))
        if key in ("Bearer same-team-primary", "Bearer same-team-sibling"):
            if request.url.path == "/v2/search":
                return response(request, 402, {"success": False, "error": "Insufficient credits"})
            if request.url.path == fc.CREDIT_USAGE_PATH:
                # Both credentials see the same depleted team, not fresh funds.
                remaining = 0 if key.endswith("primary") else -1
                return response(request, body={"success": True, "data": {"remainingCredits": remaining}})
        if key == "Bearer independent" and request.url.path == "/v2/search":
            return response(request, 200, {"success": True, "data": {"web": {"results": []}}})
        raise AssertionError(f"unexpected request {request.url} {key}")

    async def check():
        with tempfile.TemporaryDirectory() as tmp:
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                result = await controls(
                    tmp,
                    [("first@example.test", "same-team-primary"), ("secondary@example.test", "same-team-sibling"),
                     ("other@example.test", "independent")],
                ).request(client, "/v2/search", {"query": "q"}, timeout=2)
                assert result.status_code == 200

    run(check())
    assert seen == [
        ("/v2/search", "Bearer same-team-primary"),
        (fc.CREDIT_USAGE_PATH, "Bearer same-team-primary"),
        ("/v2/search", "Bearer same-team-sibling"),
        (fc.CREDIT_USAGE_PATH, "Bearer same-team-sibling"),
        ("/v2/search", "Bearer independent"),
    ], seen


def test_invalid_auth_fails_over_with_shared_cooldown_and_same_key_recovery():
    for endpoint, payload in (
        ("/v2/search", {"query": "q"}),
        ("/v2/scrape", {"url": "https://example.org"}),
    ):
        seen = []
        primary_status = 401
        groups = [("first@example.test", "primary"), ("secondary@example.test", "sibling"), ("other@example.test", "independent")]

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
                    assert state["budgets"][instance._budgets[0].id]["blocked"] == "invalid_auth", state
                    state["budgets"][instance._budgets[0].id]["next_check"] = time.monotonic() - 1
                    instance._write_state(state)
                    # Still invalid: one same-key attempt re-blocks, then callers skip it again.
                    for _ in range(2):
                        assert (await instance.request(client, endpoint, payload, timeout=2)).status_code == 200
                    state = instance._read_state()
                    assert state["budgets"][instance._budgets[0].id]["next_check"] > time.monotonic(), state
                    state["budgets"][instance._budgets[0].id]["next_check"] = time.monotonic() - 1
                    instance._write_state(state)
                    primary_status = 200
                    assert (await instance.request(client, endpoint, payload, timeout=2)).status_code == 200
                    assert instance._read_state()["budgets"][instance._budgets[0].id] == {}

        run(check())
        assert seen == [(endpoint, "Bearer " + key) for key in (
            "primary", "sibling", "sibling", "primary", "sibling", "sibling", "primary",
        )], seen


def test_reordered_and_replaced_keys_keep_their_own_sanitized_guards():
    seen = []
    secrets = ("invalid-secret", "capped-secret", "fresh-secret", "replacement-secret")

    def handler(request):
        seen.append(request.headers["Authorization"])
        return response(request)

    async def check():
        with tempfile.TemporaryDirectory() as tmp:
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                original = controls(tmp, [("first@example.test", secrets[0]), ("capped@example.test", secrets[1])])
                bad_id, cap_id = [slot.id for slot in original._budgets]
                state = {"budgets": {"old@example.test": {"blocked": "invalid_auth"}}}
                original._block(original._group_state(state, bad_id), "invalid_auth", None)
                original._block(original._group_state(state, cap_id), "key_cap_or_ambiguous", {"remaining": 3})
                original._write_state(state)
                reordered = controls(tmp, [("renamed@example.test", secrets[1]),
                                            ("first@example.test", secrets[0]), ("fresh@example.test", secrets[2])])
                try:
                    await reordered.request(client, "/v2/search", {"query": "q"}, timeout=1)
                except fc.FirecrawlUnavailable as exc:
                    assert exc.category == "key_cap_or_ambiguous"
                else:
                    raise AssertionError("reordering lost the cap guard")
                assert seen == []
                reordered = controls(tmp, [("renamed@example.test", secrets[0]),
                                            ("fresh@example.test", secrets[2]), ("capped@example.test", secrets[1])])
                assert (await reordered.request(client, "/v2/search", {"query": "q"}, timeout=1)).status_code == 200
                assert reordered._read_state()["budgets"][bad_id]["blocked"] == "invalid_auth"
                assert reordered._read_state()["budgets"][cap_id]["blocked"] == "key_cap_or_ambiguous"
                replaced = controls(tmp, [("first@example.test", secrets[3]), ("capped@example.test", secrets[1])])
                assert (await replaced.request(client, "/v2/search", {"query": "q"}, timeout=1)).status_code == 200
                assert bad_id not in replaced._read_state()["budgets"]
                persisted = replaced._state_path.read_text()
                assert not any(value in persisted for value in (*secrets, "example.test"))

    with patch.object(fc, "logger") as logs:
        run(check())
    assert seen == ["Bearer " + secrets[2], "Bearer " + secrets[3]], seen
    logged = str(logs.mock_calls)
    assert "fresh@example.test" in logged and "first@example.test" in logged, logged
    assert "label=%s" in logged and not any(secret in logged for secret in secrets), logged


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
                groups = [("first@example.test", secrets[0]), ("secondary@example.test", secrets[1])]
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
        assert "first@example.test" in logged and "secondary@example.test" in logged, logged
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
                        await controls(tmp, [("first@example.test", "primary"), ("secondary@example.test", "backup"), ("reserve@example.test", "unused")]).request(
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
                    instance = controls(tmp, [("first@example.test", "primary"), ("secondary@example.test", "backup")])
                    assert (await instance.request(client, "/v2/search", {"query": "q"}, timeout=2)).status_code == 200
                    state = instance._read_state()
                    state["budgets"][instance._budgets[0].id]["next_check"] = time.monotonic() - 1
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
                instance = controls(tmp, [("first@example.test", "primary"), ("secondary@example.test", "backup")])
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
                    [("first@example.test", "primary"), ("secondary@example.test", "sibling"), ("other@example.test", "independent")],
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
                state["budgets"][instance._budgets[0].id]["next_check"] = time.monotonic() - 1
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
                    await controls(tmp, [("first@example.test", "key-a"), ("secondary@example.test", "key-b") ]).request(
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
                    await controls(tmp, [("first@example.test", "key-a"), ("secondary@example.test", "key-b") ]).request(
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
                instance = controls(tmp, [("first@example.test", "key")])
                for _ in range(2):
                    try:
                        await instance.request(client, "/v2/search", {"query": "q"}, timeout=2)
                    except fc.FirecrawlUnavailable as exc:
                        assert exc.category == "team_exhausted", exc.category
                    else:
                        raise AssertionError("zero-credit team was not blocked")
                state = instance._read_state()
                state["budgets"][instance._budgets[0].id]["next_check"] = time.monotonic() - 1
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
                    await controls(tmp, [("first@example.test", "key") ]).request(
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
                    await controls(tmp, [("first@example.test", "key") ]).request(
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
                    instance = controls(tmp, [("first@example.test", "primary"), ("secondary@example.test", "sibling"), ("other@example.test", "independent")])
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
            instances = [controls(tmp, [("first@example.test", "key")]) for _ in range(2)]
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
                    await controls(tmp, [("first@example.test", "primary"), ("secondary@example.test", "backup") ]).request(
                        client, "/v2/scrape", {"url": "https://example.org"}, timeout=2
                    )
                except fc.FirecrawlUnavailable as exc:
                    assert exc.category == "policy_denied", exc.category
                    assert exc.status == 403
                else:
                    raise AssertionError("provider policy response was not surfaced")

    run(check())
    assert seen == ["Bearer primary"], seen


def test_gov_category_skips_automatic_cost_exclusions():
    gov = fc._search_payload({"query": "q", "limit": 10, "categories": ["gov"]})
    assert "excludeDomains" not in gov, gov
    developer = fc._search_payload({"query": "q", "limit": 10, "categories": ["developer"]})
    assert "excludeDomains" in developer, developer
    assert "linkedin.com" in developer["excludeDomains"], developer


def test_usage_ledger_records_endpoint_credits_and_request_id():
    def handler(request):
        return response(request, body={"success": True, "creditsUsed": 2, "data": {"web": []}})

    async def check():
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Path(tmp) / "usage.jsonl"
            token = obs.request_id_var.set("req-ledger")
            try:
                async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                    instance = controls(
                        tmp, [("first@example.test", "key")], usage_path=ledger
                    )
                    result = await instance.request(
                        client, "/v2/search", {"query": "q"}, timeout=1
                    )
                    assert result.status_code == 200
            finally:
                obs.request_id_var.reset(token)
            lines = ledger.read_text().splitlines()
            assert len(lines) == 1, lines
            entry = json.loads(lines[0])
            assert entry["endpoint"] == "/v2/search", entry
            assert entry["label"] == "first@example.test", entry
            assert entry["status"] == 200, entry
            assert entry["credits"] == 2.0, entry
            assert entry["request_id"] == "req-ledger", entry
            assert entry["category"] is None, entry
            assert entry["elapsed_ms"] >= 0, entry

    run(check())


def test_status_probes_each_budget_for_the_key_that_owns_the_job():
    calls = []

    def handler(request):
        calls.append(request.headers["authorization"])
        if request.headers["authorization"] == "Bearer key-one":
            return response(request, status=404, body={"success": False})
        return response(request, status=200, body={"success": True, "status": "completed"})

    async def check():
        with tempfile.TemporaryDirectory() as tmp:
            instance = controls(
                tmp, [("first@example.test", "key-one"), ("second@example.test", "key-two")]
            )
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                result = await instance.status(client, "/v2/crawl/job-1", timeout=1)
                assert result.status_code == 200
            assert calls == ["Bearer key-one", "Bearer key-two"], calls

    run(check())


def test_status_returns_last_404_when_no_budget_owns_the_job():
    calls = []

    def handler(request):
        calls.append(request.headers["authorization"])
        return response(request, status=404, body={"success": False})

    async def check():
        with tempfile.TemporaryDirectory() as tmp:
            instance = controls(
                tmp, [("first@example.test", "key-one"), ("second@example.test", "key-two")]
            )
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                result = await instance.status(client, "/v2/crawl/job-1", timeout=1)
                assert result.status_code == 404
            assert calls == ["Bearer key-one", "Bearer key-two"], calls

    run(check())


def test_usage_summary_totals_today_and_maps_endpoints():
    today = datetime.now(timezone.utc).date().isoformat()
    with tempfile.TemporaryDirectory() as tmp:
        ledger = Path(tmp) / "usage.jsonl"
        ledger.write_text(
            "\n".join(
                [
                    json.dumps({"ts": f"{today}T01:00:00Z", "endpoint": "/v2/search", "status": 200, "credits": 2}),
                    json.dumps({"ts": f"{today}T02:00:00Z", "endpoint": "/v2/scrape", "status": 200, "credits": 1}),
                    json.dumps({"ts": f"{today}T03:00:00Z", "endpoint": "/v2/map", "status": 503, "credits": 0}),
                    json.dumps({"ts": "2020-01-01T00:00:00Z", "endpoint": "/v2/crawl", "status": 200, "credits": 5}),
                    "not json",
                ]
            )
            + "\n"
        )
        instance = controls(tmp, [("a@example.test", "key")], usage_path=ledger)
        summary = instance.usage_summary()
        assert summary == {
            "date": today,
            "requests": 3,
            "credits": 3.0,
            "errors": 1,
            "by_endpoint": {
                "/v2/search": {"requests": 1, "credits": 2.0},
                "/v2/scrape": {"requests": 1, "credits": 1.0},
                "/v2/map": {"requests": 1, "credits": 0.0},
            },
            "last_request_at": f"{today}T03:00:00Z",
        }, summary
        assert controls(tmp, []).usage_summary() is None


def demo():
    for name, check in sorted(globals().items()):
        if name.startswith("test_"):
            check()
            print(f"ok {name}")
    print("all checks passed")


def test_status_probes_past_auth_and_credit_errors_to_the_owning_key():
    calls = []

    def handler(request):
        auth = request.headers["authorization"]
        calls.append(auth)
        if auth == "Bearer key-one":
            return response(request, status=401, body={"success": False})
        if auth == "Bearer key-two":
            return response(request, status=402, body={"success": False})
        return response(request, status=200, body={"success": True, "status": "completed"})

    async def check():
        with tempfile.TemporaryDirectory() as tmp:
            instance = controls(
                tmp,
                [
                    ("first@example.test", "key-one"),
                    ("second@example.test", "key-two"),
                    ("third@example.test", "key-three"),
                ],
            )
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                result = await instance.status(client, "/v2/crawl/job-1", timeout=1)
                assert result.status_code == 200
            assert calls == ["Bearer key-one", "Bearer key-two", "Bearer key-three"], calls

    run(check())


def test_upstream_response_body_is_capped_before_json():
    def handler(request):
        return response(
            request,
            body={"success": True, "data": {"web": [{"url": "https://example.com/"}]}},
        )

    async def check():
        with tempfile.TemporaryDirectory() as tmp:
            instance = controls(tmp, [("first@example.test", "key")])
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                with patch.object(fc, "MAX_UPSTREAM_BYTES", 16):
                    try:
                        await instance.request(client, "/v2/search", {"query": "q"}, timeout=1)
                    except fc.FirecrawlUnavailable as exc:
                        assert exc.category == "response_too_large", exc.category
                    else:
                        raise AssertionError("oversized response was accepted")

    run(check())


def test_completed_job_credits_are_recorded_once_and_not_counted_as_requests():
    async def check():
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Path(tmp) / "usage.jsonl"
            instance = controls(tmp, [("first@example.test", "key")], usage_path=ledger)
            await instance.record_job_credits("/v2/crawl", "job-1", 7)
            await instance.record_job_credits("/v2/crawl", "job-1", 7)
            summary = instance.usage_summary()
            assert summary["requests"] == 0, summary
            assert summary["credits"] == 7.0, summary
            assert summary["by_endpoint"]["/v2/crawl"] == {
                "requests": 0,
                "credits": 7.0,
            }, summary

    run(check())


if __name__ == "__main__":
    demo()
