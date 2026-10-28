"""Secret-safe shared controls for Firecrawl search and scrape requests."""

import asyncio
import fcntl
import hashlib
import json
import logging
import math
import os
import re
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

import httpx

from observability import current_request_id

logger = logging.getLogger("websearch.firecrawl")

API_ROOT = "https://api.firecrawl.dev"
CREDIT_USAGE_PATH = "/v2/team/credit-usage"
STATE_PATH = Path("/tmp/websearch-firecrawl-controls.json")
LOCK_PATH = Path("/tmp/websearch-firecrawl-controls.lock")
USAGE_RECHECK_SECONDS = 900.0
USAGE_SCAN_CHUNK_BYTES = 1 << 20
USAGE_SCAN_MAX_BYTES = 32 << 20
USAGE_SUMMARY_TTL_SECONDS = 60.0
MAX_RETRY_AFTER = 86_400.0
# Firecrawl Free plan: 2 concurrent browsers and 10 requests/minute per team.
# Keys from separate teams each get their own allowance, so these are per-key
# ceilings until the provider says otherwise with a 429.
DEFAULT_KEY_CONCURRENCY = 2
DEFAULT_KEY_RPM = 10
RATE_WINDOW_SECONDS = 60.0
# A crashed worker must not strand a slot forever; leases outlive the longest
# provider timeout (45 s) with slack, and a lost release frees the slot in 60 s.
MAX_LEASE_SECONDS = 60.0
JOB_START_ENDPOINTS = ("/v2/crawl", "/v2/batch/scrape")
JOB_OWNER_TTL_SECONDS = 86_400.0
MAX_JOB_OWNERS = 500
JOB_PATH_RE = re.compile(r"/v2/(batch/scrape|crawl)/([A-Za-z0-9_-]{1,200})(?:[/?#]|$)")
# Firecrawl status pages are chunked at ~10 MB and formats such as screenshot
# or rawBase64 can be large. Refuse anything bigger instead of materialising it.
MAX_UPSTREAM_BYTES = int(os.getenv("FIRECRAWL_MAX_RESPONSE_BYTES", str(16 * 1024 * 1024)))
COST_SENSITIVE_DOMAINS = (
    "linkedin.com",
    "lnkd.in",
    "x.com",
    "twitter.com",
    "mobile.twitter.com",
    "t.co",
)


async def _send_bounded(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    *,
    headers: dict[str, str],
    timeout: float,
    json_body: dict[str, Any] | None = None,
) -> httpx.Response:
    """One Firecrawl exchange with a hard response-body cap.

    Error bodies are not needed by any caller (status codes and Retry-After
    headers are), so they are left unread and returned empty. Successful bodies
    are streamed and refused before they can exceed MAX_UPSTREAM_BYTES.
    """
    if not math.isfinite(timeout) or timeout <= 0:
        raise FirecrawlUnavailable(None, "invalid_timeout", None, False)
    request = client.build_request(method, url, json=json_body, headers=headers, timeout=timeout)
    response = None
    try:
        async with asyncio.timeout(timeout):
            response = await asyncio.wait_for(
                client.send(request, stream=True), timeout=timeout
            )
            status = response.status_code
            response_headers = response.headers
            if status < 300:
                chunks = []
                total = 0
                async for chunk in response.aiter_bytes():
                    total += len(chunk)
                    if total > MAX_UPSTREAM_BYTES:
                        raise FirecrawlUnavailable(status, "response_too_large", None, False)
                    chunks.append(chunk)
                body = b"".join(chunks)
            else:
                body = b""
    except (TimeoutError, httpx.TimeoutException) as exc:
        raise FirecrawlUnavailable(None, "transport_timeout", 1.0, True) from exc
    except httpx.HTTPError as exc:
        raise FirecrawlUnavailable(None, "transport_error", 1.0, True) from exc
    finally:
        if response is not None:
            try:
                await response.aclose()
            except Exception:
                # Closing must never mask the provider's answer.
                pass
    return httpx.Response(status, headers=response_headers, content=body, request=request)


class FirecrawlUnavailable(Exception):
    def __init__(
        self,
        status: int | None,
        category: str,
        retry_after: float | None,
        retryable: bool,
    ):
        super().__init__(category)
        self.status = status
        self.category = category
        self.retry_after = retry_after
        self.retryable = retryable


@dataclass(frozen=True)
class _Budget:
    id: str
    keys: tuple[str, ...] = field(repr=False)
    label: str = ""


@dataclass(frozen=True)
class _Lease:
    budget: _Budget
    lease_id: str = field(repr=False)
    recheck: bool = False
    granted_at: float = 0.0


def _env_positive_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning("%s is not an integer; using %d", name, default)
        return default
    if value <= 0:
        logger.warning("%s must be positive; using %d", name, default)
        return default
    return value


def _earliest(current: float | None, candidate: float) -> float:
    return candidate if current is None else min(current, candidate)


def _key_slots(value: Any) -> tuple[_Budget, ...]:
    if not isinstance(value, list):
        raise ValueError("labeled key list required")
    labels, keys = set(), set()
    slots = []
    for item in value:
        if not isinstance(item, dict) or set(item) != {"label", "key"}:
            raise ValueError("each key requires only label and key")
        label, key = item["label"], item["key"]
        if not isinstance(label, str) or not label.isprintable() or not label.strip():
            raise ValueError("nonempty printable label required")
        label = label.strip()
        # Removed or unknown credentials must not become readable labels either.
        if re.search(
            r"[a-f0-9]{32}|[a-f0-9]{8}(?:-[a-f0-9]{4}){3}-[a-f0-9]{12}|fc-[a-z0-9_-]{20,}",
            label,
            re.IGNORECASE,
        ):
            raise ValueError("labels must not contain credentials")
        if label in labels:
            raise ValueError("duplicate label")
        if not isinstance(key, str) or not re.fullmatch(r"[!-~]+", key) or "," in key or key in keys:
            raise ValueError("invalid or duplicate key")
        labels.add(label)
        keys.add(key)
        # State follows the credential, not its mutable display label or list position.
        slots.append(_Budget(hashlib.sha256(key.encode("ascii")).hexdigest(), (key,), label))
    if any(key in label for label in labels for key in keys):
        raise ValueError("labels must not contain credentials")
    return tuple(slots)


def _parse_keys(raw: str) -> tuple[_Budget, ...]:
    return _key_slots(json.loads(raw)) if raw.strip() else ()


def _parse_budgets(raw: str) -> tuple[_Budget, ...]:
    """Legacy singleton entries already have an unambiguous required label."""
    value = json.loads(raw)
    if not isinstance(value, list) or not value:
        raise ValueError("legacy labeled key list required")
    entries = []
    for item in value:
        if not isinstance(item, dict) or set(item) != {"id", "keys"}:
            raise ValueError("legacy entry requires only id and keys")
        raw_keys = item["keys"]
        if not isinstance(raw_keys, list) or len(raw_keys) != 1:
            raise ValueError("legacy entry requires exactly one labeled key")
        entries.append({"label": item["id"], "key": raw_keys[0]})
    return _key_slots(entries)


def firecrawl_url_allowed(url: str, *, allow_pdf: bool = False) -> bool:
    """Whether a target avoids Firecrawl-routed domains and unexpected document costs."""
    if not isinstance(url, str):
        return False
    try:
        parsed = urlsplit(url)
        host = (parsed.hostname or "").encode("idna").decode("ascii").rstrip(".").lower()
    except (UnicodeError, ValueError):
        return False
    if parsed.scheme not in {"http", "https"} or parsed.username or parsed.password:
        return False
    path = parsed.path.lower()
    if not allow_pdf and path.endswith(".pdf"):
        return False
    return bool(host) and not any(
        host == domain or host.endswith(f".{domain}") for domain in COST_SENSITIVE_DOMAINS
    )


def _search_payload(payload: dict[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    scrape = result.get("scrapeOptions")
    if scrape is not None:
        if not isinstance(scrape, dict):
            raise FirecrawlUnavailable(None, "invalid_request", None, False)
        # Enterprise/team-gated controls are unavailable on the free plan.
        if any(scrape.get(option) for option in ("zeroDataRetention", "threatProtection")):
            raise FirecrawlUnavailable(None, "unsupported_enterprise_option", None, False)
    categories = result.get("categories")
    if isinstance(categories, list) and "gov" in categories:
        # Firecrawl forbids gov with any other filter; the automatic cost
        # exclusions are filters too, so leave the payload untouched.
        return result
    included = result.get("includeDomains")
    if isinstance(included, list) and included:
        # Firecrawl rejects include+exclude together; the caller's allowlist wins.
        if any(result.get("excludeDomains") or ()):
            raise FirecrawlUnavailable(None, "invalid_request", None, False)
        return result
    excluded = result.get("excludeDomains", [])
    if not isinstance(excluded, list) or any(not isinstance(item, str) for item in excluded):
        raise FirecrawlUnavailable(None, "invalid_request", None, False)
    seen = {item.lower().rstrip(".") for item in excluded}
    result["excludeDomains"] = list(excluded) + [
        domain for domain in COST_SENSITIVE_DOMAINS if domain not in seen
    ]
    return result


def _parsers_include_pdf(parsers: Any) -> bool:
    if not isinstance(parsers, list):
        return False
    for parser in parsers:
        if parser == "pdf" or (isinstance(parser, dict) and parser.get("type") == "pdf"):
            return True
    return False


def _scrape_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload.get("url"), str) or not payload["url"]:
        raise FirecrawlUnavailable(None, "invalid_request", None, False)
    # Enterprise/team-gated controls remain unavailable on the free plan.
    if any(payload.get(option) for option in ("zeroDataRetention", "threatProtection")):
        raise FirecrawlUnavailable(None, "unsupported_enterprise_option", None, False)
    result = dict(payload)
    result.setdefault("formats", ["markdown"])
    result.setdefault("parsers", [])
    result.setdefault("storeInCache", False)
    return result


def _response_credits(endpoint: str, response: httpx.Response) -> float:
    """Credits for the ledger. Only search/scrape report a usable count."""
    if endpoint == "/v2/scrape":
        return 1.0
    if endpoint == "/v2/map":
        return 1.0
    if endpoint in ("/v2/crawl", "/v2/batch/scrape"):
        return 0.0
    try:
        payload = response.json()
    except ValueError:
        return 2.0
    value = payload.get("creditsUsed") if isinstance(payload, dict) else None
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0:
        return float(value)
    return 2.0


def _retry_after(response: httpx.Response) -> float:
    value = response.headers.get("Retry-After")
    if value:
        try:
            delay = float(value)
        except ValueError:
            try:
                date = parsedate_to_datetime(value)
                if date.tzinfo is None:
                    date = date.replace(tzinfo=timezone.utc)
                delay = date.timestamp() - time.time()
            except (TypeError, ValueError, OverflowError):
                return 60.0
        if math.isfinite(delay):
            return min(max(delay, 1.0), MAX_RETRY_AFTER)
    return 60.0


def _period_end(value: Any) -> float | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.timestamp()
    except (TypeError, ValueError, OverflowError):
        return None


class FirecrawlControls:
    def __init__(
        self,
        budgets: tuple[_Budget, ...],
        *,
        config_error: bool = False,
        state_path: Path = STATE_PATH,
        lock_path: Path = LOCK_PATH,
        usage_path: Path | None = None,
        concurrency: int | None = None,
        rpm: int | None = None,
        rate_window: float | None = None,
        lease_seconds: float | None = None,
    ):
        self._budgets = budgets
        self._config_error = config_error
        self._state_path = Path(state_path)
        self._lock_path = Path(lock_path)
        self._usage_path = Path(usage_path) if usage_path else None
        self._usage_cache: tuple[float, dict[str, Any] | None] | None = None
        self._concurrency = max(
            1,
            concurrency
            if concurrency is not None
            else _env_positive_int("FIRECRAWL_KEY_CONCURRENCY", DEFAULT_KEY_CONCURRENCY),
        )
        self._rpm = max(
            1,
            rpm if rpm is not None else _env_positive_int("FIRECRAWL_KEY_RPM", DEFAULT_KEY_RPM),
        )
        self._rate_window = float(
            rate_window if rate_window is not None else RATE_WINDOW_SECONDS
        )
        self._lease_seconds = float(
            lease_seconds if lease_seconds is not None else MAX_LEASE_SECONDS
        )

    @classmethod
    def from_env(cls) -> "FirecrawlControls":
        raw = os.getenv("FIRECRAWL_KEYS_JSON")
        usage_raw = os.getenv("FIRECRAWL_USAGE_LOG", "").strip()
        usage_path = Path(usage_raw) if usage_raw else None
        try:
            if raw is not None:
                return cls(_parse_keys(raw), usage_path=usage_path)
            legacy = os.getenv("FIRECRAWL_BUDGETS_JSON", "").strip()
            return cls(_parse_budgets(legacy) if legacy else (), usage_path=usage_path)
        except (TypeError, ValueError):
            logger.warning("Firecrawl key configuration is invalid; Firecrawl is disabled")
            return cls((), config_error=True, usage_path=usage_path)

    async def request(
        self,
        client: httpx.AsyncClient,
        endpoint: str,
        payload: dict[str, Any],
        *,
        timeout: float,
    ) -> httpx.Response:
        if not math.isfinite(timeout) or timeout <= 0:
            raise FirecrawlUnavailable(None, "invalid_timeout", None, False)
        if endpoint not in (
            "/v2/search",
            "/v2/scrape",
            "/v2/map",
            "/v2/crawl",
            "/v2/batch/scrape",
        ) or not isinstance(payload, dict):
            raise FirecrawlUnavailable(None, "invalid_request", None, False)
        if self._config_error:
            raise FirecrawlUnavailable(None, "invalid_configuration", None, False)
        if not self._budgets:
            raise FirecrawlUnavailable(None, "not_configured", None, False)

        if endpoint == "/v2/search":
            request_payload = _search_payload(payload)
        elif endpoint == "/v2/scrape":
            request_payload = _scrape_payload(payload)
            if not firecrawl_url_allowed(
                request_payload.get("url"),
                allow_pdf=_parsers_include_pdf(request_payload.get("parsers")),
            ):
                raise FirecrawlUnavailable(None, "provider_routed_domain", None, False)
        else:
            request_payload = dict(payload)

        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        tried: set[str] = set()
        earliest_retry: float | None = None
        last_error: FirecrawlUnavailable | None = None

        def record(
            label: str,
            status: int | None,
            credits: float,
            category: str | None,
            attempt_started: float,
        ) -> None:
            self._append_usage(
                {
                    "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                    "request_id": current_request_id(),
                    "endpoint": endpoint,
                    "label": label,
                    "status": status,
                    "credits": credits,
                    "category": category,
                    "elapsed_ms": int((time.perf_counter() - attempt_started) * 1000),
                }
            )

        while True:
            try:
                lease = await self._admit(deadline, exclude=tried)
            except FirecrawlUnavailable as exc:
                # Capacity or deadline exhaustion should not hide the provider
                # failure that caused the wait.
                if last_error is not None and exc.category in ("team_concurrency", "request_timeout"):
                    if last_error.category == "rate_limited" and earliest_retry is not None:
                        last_error = FirecrawlUnavailable(
                            429, "rate_limited", max(1.0, earliest_retry), True
                        )
                    raise last_error
                raise
            tried.add(lease.budget.id)
            settled = False
            try:
                if lease.recheck:
                    blocked_error = await self._recheck_guard(client, lease, deadline)
                    if blocked_error is not None:
                        settled = True
                        last_error = blocked_error
                        continue
                remaining = deadline - loop.time()
                if remaining <= 0:
                    raise last_error or FirecrawlUnavailable(None, "request_timeout", 1.0, True)
                attempt_started = time.perf_counter()
                try:
                    response = await _send_bounded(
                        client,
                        "POST",
                        f"{API_ROOT}{endpoint}",
                        json_body=request_payload,
                        headers={
                            "Authorization": "Bearer " + lease.budget.keys[0],
                            "Accept": "application/json",
                        },
                        timeout=remaining,
                    )
                except FirecrawlUnavailable as exc:
                    record(lease.budget.label, None, 0.0, exc.category, attempt_started)
                    raise

                status = response.status_code

                if 200 <= status < 300:
                    credits = _response_credits(endpoint, response)
                    record(lease.budget.label, status, credits, None, attempt_started)
                    await self._settle(
                        lease, clear_block=True, job=self._job_owner(endpoint, response)
                    )
                    settled = True
                    logger.info(
                        "firecrawl request ok label=%s endpoint=%s elapsed_ms=%d credits=%.1f",
                        lease.budget.label,
                        endpoint,
                        int((time.perf_counter() - attempt_started) * 1000),
                        credits,
                    )
                    return response

                if status == 402:
                    usage = await self._credit_usage(
                        client, lease.budget.keys[0], deadline, lease.budget.label
                    )
                    if usage is not None and usage["remaining"] <= 0:
                        record(lease.budget.label, status, 0.0, "team_exhausted", attempt_started)
                        await self._settle(lease, block="team_exhausted", usage=usage)
                        logger.warning(
                            "firecrawl key exhausted label=%s remaining=0", lease.budget.label
                        )
                        last_error = FirecrawlUnavailable(
                            402, "team_exhausted", USAGE_RECHECK_SECONDS, True
                        )
                    else:
                        record(
                            lease.budget.label,
                            status,
                            0.0,
                            "key_cap_or_ambiguous",
                            attempt_started,
                        )
                        await self._settle(lease, block="key_cap_or_ambiguous", usage=usage)
                        logger.warning(
                            "firecrawl key blocked label=%s category=key_cap_or_ambiguous",
                            lease.budget.label,
                        )
                        last_error = FirecrawlUnavailable(
                            402, "key_cap_or_ambiguous", USAGE_RECHECK_SECONDS, True
                        )
                    settled = True
                    continue

                if status == 429:
                    delay = _retry_after(response)
                    record(lease.budget.label, status, 0.0, "rate_limited", attempt_started)
                    await self._settle(lease, block="rate_limited", retry=delay)
                    settled = True
                    logger.warning(
                        "firecrawl key rate limited label=%s retry_after=%.0f",
                        lease.budget.label,
                        delay,
                    )
                    earliest_retry = _earliest(earliest_retry, delay)
                    last_error = FirecrawlUnavailable(429, "rate_limited", delay, True)
                    continue

                if status == 401:
                    record(lease.budget.label, status, 0.0, "invalid_auth", attempt_started)
                    await self._settle(lease, block="invalid_auth")
                    settled = True
                    logger.warning(
                        "firecrawl key blocked label=%s category=invalid_auth",
                        lease.budget.label,
                    )
                    last_error = FirecrawlUnavailable(
                        401, "invalid_auth", USAGE_RECHECK_SECONDS, True
                    )
                    continue

                credits = 1.0 if endpoint == "/v2/scrape" and status < 500 else 0.0
                category = (
                    "policy_denied"
                    if status == 403
                    else "provider_unavailable"
                    if status >= 500
                    else None
                )
                record(lease.budget.label, status, credits, category, attempt_started)
                await self._settle(lease)
                settled = True
                if status == 403:
                    raise FirecrawlUnavailable(403, "policy_denied", None, False)
                if status >= 500:
                    raise FirecrawlUnavailable(status, "provider_unavailable", 1.0, True)

                # The adapter owns target-level errors (invalid URLs, unsupported
                # content, no record, and page status); do not rotate a key for them.
                return response
            finally:
                if not settled:
                    # Cancellation, timeout, and transport errors must not strand
                    # a key slot until its lease expires.
                    try:
                        await self._settle(lease)
                    except asyncio.CancelledError:
                        pass

    async def _admit(
        self, deadline: float, *, exclude: set[str] | None = None
    ) -> _Lease:
        """Reserve the least-loaded healthy key, waiting only when every key is busy."""
        loop = asyncio.get_running_loop()
        poll = 0.02
        while True:
            if deadline - loop.time() <= 0:
                raise FirecrawlUnavailable(None, "request_timeout", 1.0, True)
            fd = await self._acquire_lock(deadline)
            wait_seconds: float | None = None
            rate_wait: float | None = None
            capacity_limited = False
            blocked: list[FirecrawlUnavailable] = []
            try:
                now = time.time()
                remaining = deadline - loop.time()
                if remaining <= 0:
                    raise FirecrawlUnavailable(None, "request_timeout", 1.0, True)
                state = self._read_state()
                candidates: list[tuple[int, float, int, _Budget, bool]] = []
                for position, budget in enumerate(self._budgets):
                    if exclude and budget.id in exclude:
                        # One attempt per key per request: a short Retry-After
                        # must not turn one call into a provider-call loop.
                        continue
                    group = self._group_state(state, budget.id)
                    self._prune_group(group, now)
                    reason = group.get("blocked")
                    recheck = False
                    if reason:
                        next_check = group.get("next_check")
                        granted = group.get("blocked_for")
                        skewed = (
                            isinstance(next_check, (int, float))
                            and isinstance(granted, (int, float))
                            and next_check - now > granted + 1.0
                        )
                        if isinstance(next_check, (int, float)) and now < next_check and not skewed:
                            retry = next_check - now
                            blocked.append(self._block_error(reason, retry))
                            wait_seconds = _earliest(wait_seconds, retry)
                            continue
                        # Expired, or a backwards clock step stretched the wall
                        # deadline past the granted duration: let the key recover.
                        recheck = reason in ("team_exhausted", "key_cap_or_ambiguous")
                    leases = group.get("leases")
                    if isinstance(leases, list) and len(leases) >= self._concurrency:
                        # A slot frees when its request settles, not when its
                        # crash-backstop lease expires, so poll instead of
                        # waiting for the expiry timestamp.
                        capacity_limited = True
                        continue
                    rate = group.get("rate")
                    if isinstance(rate, list) and len(rate) >= self._rpm:
                        rate_wait = rate[0] + self._rate_window - now
                        wait_seconds = _earliest(wait_seconds, rate_wait)
                        continue
                    last_used = group.get("last_used")
                    if not isinstance(last_used, (int, float)):
                        last_used = 0.0
                    candidates.append(
                        (
                            len(leases) if isinstance(leases, list) else 0,
                            last_used,
                            position,
                            budget,
                            recheck,
                        )
                    )
                if candidates:
                    _, _, _, budget, recheck = min(candidates)
                    group = self._group_state(state, budget.id)
                    if group.get("blocked") and not recheck:
                        self._clear_block(group)
                    lease_id = os.urandom(8).hex()
                    granted = max(self._lease_seconds, remaining + 5.0)
                    group.setdefault("leases", []).append(
                        {"id": lease_id, "expires": now + granted, "for": granted}
                    )
                    group.setdefault("rate", []).append(now)
                    group["last_used"] = now
                    self._write_state(state)
                    return _Lease(budget, lease_id, recheck, now)
                available = len(self._budgets) - (len(exclude) if exclude else 0)
                if blocked and len(blocked) == available:
                    # Every usable key has an active cooldown. Waiting inside the
                    # request would hold it to its deadline and herd every caller
                    # onto the same expiry. Report the class that recovers first;
                    # ties keep the last configured key, matching old behaviour.
                    def order(pair: tuple[int, FirecrawlUnavailable]) -> tuple[float, int]:
                        index, item = pair
                        retry = item.retry_after
                        return (
                            round(retry) if isinstance(retry, (int, float)) else 1e9,
                            -index,
                        )

                    _, error = min(enumerate(blocked), key=order)
                    if wait_seconds is not None:
                        error = FirecrawlUnavailable(
                            error.status, error.category, max(1.0, wait_seconds), error.retryable
                        )
                    raise error
            finally:
                self._release_lock(fd)
            wait = remaining if wait_seconds is None else wait_seconds
            if capacity_limited:
                # Poll for a released slot with backoff.
                wait = min(wait, poll)
                poll = min(poll * 2, 0.1)
            elif wait >= remaining:
                if rate_wait is not None and wait <= rate_wait + 1e-6:
                    # Local pacing, not a provider outage: this is a rate limit.
                    raise FirecrawlUnavailable(429, "rate_limited", max(1.0, wait), True)
                raise FirecrawlUnavailable(429, "team_concurrency", max(0.5, wait), True)
            await asyncio.sleep(max(0.01, min(wait, 0.25)))

    @staticmethod
    def _block_error(reason: str, retry_after: float) -> FirecrawlUnavailable:
        if reason == "rate_limited":
            return FirecrawlUnavailable(429, reason, max(1.0, retry_after), True)
        if reason == "invalid_auth":
            return FirecrawlUnavailable(
                401, reason, max(1.0, min(retry_after, USAGE_RECHECK_SECONDS)), True
            )
        if reason == "team_exhausted":
            return FirecrawlUnavailable(402, reason, USAGE_RECHECK_SECONDS, True)
        return FirecrawlUnavailable(
            402, "key_cap_or_ambiguous", max(1.0, min(retry_after, USAGE_RECHECK_SECONDS)), True
        )

    def _prune_group(self, group: dict[str, Any], now: float) -> None:
        leases = group.get("leases")
        if not isinstance(leases, list):
            group.pop("leases", None)
            leases = None
        if isinstance(leases, list):
            live = []
            for lease in leases:
                if not isinstance(lease, dict):
                    continue
                expires = lease.get("expires")
                if not isinstance(expires, (int, float)):
                    continue
                remaining = expires - now
                if remaining <= 0:
                    continue
                # A backwards wall-clock step must not stretch a lease past the
                # duration it was granted for; shorten it to that duration now.
                granted = lease.get("for")
                if isinstance(granted, (int, float)) and remaining > granted + 5.0:
                    lease["expires"] = now + granted
                live.append(lease)
            if live:
                group["leases"] = live
            else:
                group.pop("leases", None)
        rate = group.get("rate")
        if not isinstance(rate, list):
            group.pop("rate", None)
            rate = None
        if isinstance(rate, list):
            live_events = [
                stamp
                for stamp in rate
                if isinstance(stamp, (int, float))
                and now - stamp < self._rate_window
                and stamp <= now + 5.0
            ]
            if live_events:
                group["rate"] = live_events
            else:
                group.pop("rate", None)

    @staticmethod
    def _clear_block(group: dict[str, Any]) -> None:
        for key in ("blocked", "next_check", "blocked_for", "remaining", "billing_period_end"):
            group.pop(key, None)

    async def _recheck_guard(
        self, client: httpx.AsyncClient, lease: _Lease, deadline: float
    ) -> FirecrawlUnavailable | None:
        """Verify an expired exhaustion guard without holding the shared lock."""
        usage = await self._credit_usage(
            client, lease.budget.keys[0], deadline, lease.budget.label
        )
        if usage is None:
            await self._settle(lease, block="key_cap_or_ambiguous")
            return FirecrawlUnavailable(
                402, "key_cap_or_ambiguous", USAGE_RECHECK_SECONDS, True
            )
        if usage["remaining"] <= 0:
            await self._settle(lease, block="team_exhausted", usage=usage)
            return FirecrawlUnavailable(402, "team_exhausted", USAGE_RECHECK_SECONDS, True)
        await self._settle(lease, clear_block=True, release=False)
        logger.info(
            "firecrawl key recovered label=%s remaining=%.0f",
            lease.budget.label,
            usage["remaining"],
        )
        return None

    async def _settle(
        self,
        lease: _Lease,
        *,
        block: str | None = None,
        usage: dict[str, Any] | None = None,
        retry: float = USAGE_RECHECK_SECONDS,
        clear_block: bool = False,
        release: bool = True,
        job: tuple[str, str] | None = None,
    ) -> None:
        """Release a slot and record the outcome in one short locked write."""
        loop = asyncio.get_running_loop()
        try:
            fd = await self._acquire_lock(loop.time() + 2.0)
        except FirecrawlUnavailable:
            logger.warning(
                "firecrawl lease not settled label=%s; it will expire", lease.budget.label
            )
            return
        try:
            try:
                state = self._read_state()
                group = self._group_state(state, lease.budget.id)
                if release:
                    current = group.get("leases")
                    leases = [
                        item
                        for item in (current if isinstance(current, list) else [])
                        if isinstance(item, dict) and item.get("id") != lease.lease_id
                    ]
                    if leases:
                        group["leases"] = leases
                    else:
                        group.pop("leases", None)
                if clear_block and self._block_is_not_newer(group, lease.granted_at):
                    self._clear_block(group)
                if block:
                    self._block(group, block, usage, retry=retry)
                if job is not None:
                    self._remember_job_owner(state, lease.budget.id, *job)
                self._write_state(state)
            except FirecrawlUnavailable:
                # The provider outcome is authoritative; a lost state update only
                # costs one cooldown or one lease that expires on its own.
                return
        finally:
            self._release_lock(fd)

    @staticmethod
    def _job_owner(
        endpoint: str, response: httpx.Response
    ) -> tuple[str, str] | None:
        if endpoint not in JOB_START_ENDPOINTS:
            return None
        try:
            payload = response.json()
        except ValueError:
            return None
        job_id = payload.get("id") if isinstance(payload, dict) else None
        if not isinstance(job_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,200}", job_id):
            return None
        return endpoint, job_id

    @staticmethod
    def _owner_stamp(value: Any) -> float:
        at = value.get("at") if isinstance(value, dict) else None
        if isinstance(at, (int, float)) and not isinstance(at, bool) and math.isfinite(at):
            return float(at)
        return 0.0

    @classmethod
    def _remember_job_owner(
        cls, state: dict[str, Any], budget_id: str, endpoint: str, job_id: str
    ) -> None:
        owners = state.setdefault("job_owners", {})
        if not isinstance(owners, dict):
            owners = {}
            state["job_owners"] = owners
        now = time.time()
        owners[f"{endpoint}/{job_id}"] = {"budget": budget_id, "at": now}
        cutoff = now - JOB_OWNER_TTL_SECONDS
        for key in [key for key, value in owners.items() if cls._owner_stamp(value) < cutoff]:
            owners.pop(key, None)
        if len(owners) > MAX_JOB_OWNERS:
            ordered = sorted(owners.items(), key=lambda item: cls._owner_stamp(item[1]))
            for key, _ in ordered[: len(ordered) - MAX_JOB_OWNERS]:
                owners.pop(key, None)

    def _job_key(self, path: str) -> str | None:
        match = JOB_PATH_RE.search(path)
        if match is None:
            return None
        return f"/v2/{match.group(1)}/{match.group(2)}"

    def _ordered_budgets(self, path: str) -> tuple[tuple[_Budget, ...], bool]:
        """The key that started a job is tried first; status reads are free."""
        job_key = self._job_key(path)
        if job_key is None:
            return self._budgets, False
        try:
            state = self._read_state()
        except FirecrawlUnavailable:
            return self._budgets, False
        owner = state.get("job_owners", {}).get(job_key)
        budget_id = owner.get("budget") if isinstance(owner, dict) else None
        if not isinstance(budget_id, str):
            return self._budgets, False
        ordered = tuple(sorted(self._budgets, key=lambda budget: budget.id != budget_id))
        return ordered, ordered[0].id == budget_id

    async def _credit_usage(
        self, client: httpx.AsyncClient, key: str, deadline: float, label: str
    ) -> dict[str, Any] | None:
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            return None
        try:
            response = await _send_bounded(
                client,
                "GET",
                f"{API_ROOT}{CREDIT_USAGE_PATH}",
                headers={"Authorization": "Bearer " + key, "Accept": "application/json"},
                timeout=remaining,
            )
            if response.status_code != 200:
                return None
            body = response.json()
        except (FirecrawlUnavailable, ValueError):
            return None

        data = body.get("data") if isinstance(body, dict) and body.get("success") is True else None
        remaining = data.get("remainingCredits") if isinstance(data, dict) else None
        if not isinstance(remaining, (int, float)) or isinstance(remaining, bool) or not math.isfinite(remaining):
            return None
        period_end = _period_end(data.get("billingPeriodEnd"))
        logger.info("firecrawl usage checked label=%s remaining=%.0f", label, remaining)
        return {"remaining": float(remaining), "period_end": period_end}

    def _group_state(self, state: dict[str, Any], budget_id: str) -> dict[str, Any]:
        groups = state.setdefault("budgets", {})
        if not isinstance(groups, dict):
            raise FirecrawlUnavailable(None, "control_state_error", None, False)
        group = groups.setdefault(budget_id, {})
        if not isinstance(group, dict):
            raise FirecrawlUnavailable(None, "control_state_error", None, False)
        return group

    @staticmethod
    def _block(
        group: dict[str, Any],
        reason: str,
        usage: dict[str, Any] | None,
        *,
        retry: float = USAGE_RECHECK_SECONDS,
    ) -> None:
        # Keep live leases, rate history, and recency: other callers may still be
        # in flight on this key, and blocking one key must not erase their slots.
        # Wall-clock time, not monotonic: this file outlives a host reboot.
        now = time.time()
        existing = group.get("next_check")
        if isinstance(existing, (int, float)) and existing > now + retry:
            # A concurrent request found a longer or stronger guard; do not
            # downgrade it with this outcome.
            return
        group["blocked"] = reason
        group["next_check"] = now + retry
        # The duration bounds the cooldown if the wall clock steps backwards.
        group["blocked_for"] = retry
        group["remaining"] = usage.get("remaining") if usage else None
        group["billing_period_end"] = usage.get("period_end") if usage else None

    @staticmethod
    def _block_is_not_newer(group: dict[str, Any], granted_at: float) -> bool:
        next_check = group.get("next_check")
        blocked_for = group.get("blocked_for")
        if not isinstance(next_check, (int, float)) or not isinstance(blocked_for, (int, float)):
            return True
        # A guard set after this lease was granted belongs to a newer outcome.
        return next_check - blocked_for <= granted_at

    async def _acquire_lock(self, deadline: float) -> int:
        try:
            fd = os.open(
                self._lock_path,
                os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0),
                0o600,
            )
        except OSError as exc:
            raise FirecrawlUnavailable(None, "control_state_error", None, False) from exc
        try:
            waited = 0.0
            while True:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    return fd
                except BlockingIOError:
                    remaining = deadline - asyncio.get_running_loop().time()
                    if remaining <= 0:
                        raise FirecrawlUnavailable(None, "team_concurrency", 1.0, True)
                    # Short critical sections: poll quickly at first instead of
                    # adding 50 ms to every admission under contention.
                    delay = 0.001 if waited < 0.05 else 0.05
                    await asyncio.sleep(min(delay, remaining))
                    waited += delay
        except BaseException:
            try:
                os.close(fd)
            except OSError:
                pass
            raise

    @staticmethod
    def _release_lock(fd: int) -> None:
        # Closing the descriptor releases the flock anyway; a release error must
        # never turn a successful, already-billed provider response into a 500.
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            pass
        finally:
            try:
                os.close(fd)
            except OSError:
                pass

    def _read_state(self) -> dict[str, Any]:
        try:
            fd = os.open(self._state_path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        except FileNotFoundError:
            return {}
        except OSError as exc:
            raise FirecrawlUnavailable(None, "control_state_error", None, False) from exc
        try:
            with os.fdopen(fd, "r", encoding="utf-8") as state_file:
                try:
                    state = json.load(state_file)
                except (UnicodeDecodeError, json.JSONDecodeError):
                    # The file is disposable and the next write rebuilds it. A
                    # damaged guard must not take Firecrawl down with it.
                    logger.warning("firecrawl control state is unreadable; rebuilding it")
                    return {}
        except OSError as exc:
            raise FirecrawlUnavailable(None, "control_state_error", None, False) from exc
        if not isinstance(state, dict):
            logger.warning("firecrawl control state is not an object; rebuilding it")
            return {}
        groups = state.get("budgets", {})
        if not isinstance(groups, dict):
            logger.warning("firecrawl control state has invalid budgets; rebuilding it")
            return {}
        # Discard removed credentials and pre-pool operator labels, never reuse their guards.
        cleaned = {
            "budgets": {
                budget.id: groups[budget.id]
                for budget in self._budgets
                if isinstance(groups.get(budget.id), dict)
            }
        }
        jobs = state.get("jobs")
        if isinstance(jobs, dict):
            cleaned["jobs"] = jobs
        owners = state.get("job_owners")
        if isinstance(owners, dict):
            cleaned["job_owners"] = owners
        return cleaned

    async def status(
        self,
        client: httpx.AsyncClient,
        path: str,
        *,
        timeout: float,
    ) -> httpx.Response:
        """Poll a Firecrawl job. Status reads are not billable; keys are probed."""
        if not math.isfinite(timeout) or timeout <= 0:
            raise FirecrawlUnavailable(None, "invalid_timeout", None, False)
        if path.startswith("https://") or path.startswith("http://"):
            parsed = urlsplit(path)
            if (
                parsed.scheme != "https"
                or parsed.hostname != "api.firecrawl.dev"
                or parsed.username is not None
                or parsed.password is not None
                or ".." in parsed.path
            ):
                raise FirecrawlUnavailable(None, "invalid_request", None, False)
            request_url = path
        else:
            if not path.startswith("/v2/") or ".." in path:
                raise FirecrawlUnavailable(None, "invalid_request", None, False)
            request_url = API_ROOT + path
        if self._config_error:
            raise FirecrawlUnavailable(None, "invalid_configuration", None, False)
        if not self._budgets:
            raise FirecrawlUnavailable(None, "not_configured", None, False)
        # A job belongs to the key that created it, and request admission may
        # have rotated past the first budget. Status reads are free, so probe
        # each budget's first key until one knows the job. Auth/throttle errors
        # from a key that does not own the job are also skipped, not returned.
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        last = None
        owner_response = None
        budgets, pinned = self._ordered_budgets(path)

        def fallback() -> httpx.Response | None:
            # A foreign key's 404 must not mask the owning key's real error.
            if pinned and owner_response is not None:
                return owner_response
            return last

        for index, budget in enumerate(budgets):
            remaining = deadline - loop.time()
            if remaining <= 0:
                if fallback() is not None:
                    return fallback()
                raise FirecrawlUnavailable(None, "transport_timeout", 1.0, True)
            key = budget.keys[0]
            try:
                response = await _send_bounded(
                    client,
                    "GET",
                    request_url,
                    headers={"Authorization": "Bearer " + key, "Accept": "application/json"},
                    timeout=remaining,
                )
            except FirecrawlUnavailable as exc:
                if exc.category == "transport_timeout":
                    if fallback() is not None:
                        return fallback()
                    raise
                # A fast transport error on one key must not stop the scan for
                # the key that actually owns the job.
                if index < len(budgets) - 1:
                    continue
                ending = fallback()
                if ending is not None:
                    return ending
                raise
            if response.status_code == 429 and pinned and index == 0:
                # Only the owning key can read this job; a 404 from another key
                # would misreport a throttled poll as a missing job.
                return response
            if response.status_code >= 300:
                last = response
                if pinned and index == 0:
                    owner_response = response
                if index < len(budgets) - 1:
                    continue
                ending = fallback()
                return ending if ending is not None else response
            return response
        ending = fallback()
        if ending is None:
            raise FirecrawlUnavailable(None, "transport_timeout", 1.0, True)
        return ending

    async def record_job_credits(self, endpoint: str, job_id: str, credits: Any) -> None:
        """Record a completed job's credits once, keyed by job id.

        Job starts report 0 because Firecrawl only reveals `creditsUsed` on the
        status response. Status polling can repeat, so the state file keeps a
        small completed-job map and the ledger entry is written only on the
        first terminal observation.
        """
        if self._usage_path is None:
            return
        if not isinstance(credits, (int, float)) or isinstance(credits, bool):
            return
        if not math.isfinite(credits) or credits <= 0:
            return
        loop = asyncio.get_running_loop()
        try:
            fd = await self._acquire_lock(loop.time() + 2)
        except FirecrawlUnavailable:
            return
        try:
            try:
                state = self._read_state()
                jobs = state.get("jobs")
                if not isinstance(jobs, dict):
                    jobs = {}
                    state["jobs"] = jobs
                key = f"{endpoint}:{job_id}"
                if key in jobs:
                    return
                jobs[key] = float(credits)
                if len(jobs) > 2000:
                    for old in list(jobs)[:-1000]:
                        jobs.pop(old, None)
                self._write_state(state)
            except FirecrawlUnavailable:
                return
        finally:
            self._release_lock(fd)
        self._append_usage(
            {
                "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                "request_id": current_request_id(),
                "endpoint": endpoint,
                "label": None,
                "status": 200,
                "credits": float(credits),
                "category": None,
                "elapsed_ms": 0,
                "kind": "job_credits",
                "job_id": job_id,
            }
        )

    def usage_summary(self) -> dict[str, Any] | None:
        """Today's (UTC) ledger totals, scanned backwards only as far as today."""
        if self._usage_path is None:
            return None
        now = time.monotonic()
        cached = self._usage_cache
        if cached is not None and now - cached[0] < USAGE_SUMMARY_TTL_SECONDS:
            return cached[1]
        result = self._scan_usage()
        self._usage_cache = (now, result)
        return result

    def _scan_usage(self) -> dict[str, Any] | None:
        if self._usage_path is None:
            return None
        today = datetime.now(timezone.utc).date().isoformat()
        requests = 0
        credits = 0.0
        errors = 0
        by_endpoint: dict[str, dict[str, float]] = {}
        last_request_at: str | None = None

        seen_today = False

        def add(entry: dict[str, Any]) -> None:
            nonlocal requests, credits, errors, last_request_at, seen_today
            seen_today = True
            endpoint = entry.get("endpoint")
            if not isinstance(endpoint, str):
                return
            amount = entry.get("credits")
            amount = (
                float(amount)
                if isinstance(amount, (int, float)) and not isinstance(amount, bool)
                else 0.0
            )
            ts = entry.get("ts")
            if entry.get("kind") == "job_credits":
                credits += amount
                by_endpoint.setdefault(endpoint, {"requests": 0, "credits": 0.0})[
                    "credits"
                ] += amount
                return
            requests += 1
            credits += amount
            status = entry.get("status")
            if not isinstance(status, int) or not 200 <= status < 300:
                errors += 1
            bucket = by_endpoint.setdefault(endpoint, {"requests": 0, "credits": 0.0})
            bucket["requests"] += 1
            bucket["credits"] += amount
            if isinstance(ts, str) and (last_request_at is None or ts > last_request_at):
                last_request_at = ts

        try:
            with self._usage_path.open("rb") as ledger:
                ledger.seek(0, os.SEEK_END)
                position = ledger.tell()
                buffer = b""
                stopped = False
                scanned = 0
                while position > 0 and not stopped:
                    if not seen_today and scanned >= USAGE_SCAN_MAX_BYTES:
                        # No today entry in the tail (all-old file): bound the scan.
                        stopped = True
                        break
                    size = min(USAGE_SCAN_CHUNK_BYTES, position)
                    position -= size
                    scanned += size
                    ledger.seek(position)
                    buffer = ledger.read(size) + buffer
                    lines = buffer.split(b"\n")
                    # The first piece may continue into the unread earlier chunk.
                    buffer = lines.pop(0) if lines else b""
                    for raw in reversed(lines):
                        try:
                            entry = json.loads(raw)
                        except (TypeError, ValueError):
                            continue
                        if not isinstance(entry, dict):
                            continue
                        ts = entry.get("ts")
                        if not isinstance(ts, str):
                            continue
                        if ts.startswith(today):
                            add(entry)
                        elif ts[:10] < today:
                            if seen_today:
                                # The ledger is append-ordered; everything earlier
                                # is old. A line before any today entry is a
                                # midnight straddler appended out of order.
                                stopped = True
                                break
                        # A future-dated line (clock step) is skipped, not a boundary.
                if not stopped and buffer:
                    try:
                        entry = json.loads(buffer)
                    except (TypeError, ValueError):
                        entry = None
                    if (
                        isinstance(entry, dict)
                        and isinstance(entry.get("ts"), str)
                        and entry["ts"].startswith(today)
                    ):
                        add(entry)
        except OSError:
            return None
        return {
            "date": today,
            "requests": requests,
            "credits": round(credits, 1),
            "errors": errors,
            "by_endpoint": by_endpoint,
            "last_request_at": last_request_at,
        }

    def _append_usage(self, entry: dict[str, Any]) -> None:
        """Best-effort append-only ledger; a write failure never fails the request."""
        if self._usage_path is None:
            return
        try:
            self._usage_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            fd = os.open(
                self._usage_path,
                os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0),
                0o600,
            )
            try:
                os.write(fd, (json.dumps(entry, separators=(",", ":")) + "\n").encode("utf-8"))
            finally:
                os.close(fd)
        except OSError as exc:
            logger.warning("firecrawl usage log not written error=%s", type(exc).__name__)

    def _write_state(self, state: dict[str, Any]) -> None:
        path = self._state_path
        try:
            path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            fd, temporary = tempfile.mkstemp(prefix=".firecrawl-", dir=path.parent)
            try:
                os.fchmod(fd, 0o600)
                with os.fdopen(fd, "w", encoding="utf-8") as state_file:
                    json.dump(state, state_file, separators=(",", ":"))
                    state_file.flush()
                    os.fsync(state_file.fileno())
                os.replace(temporary, path)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
        except OSError as exc:
            raise FirecrawlUnavailable(None, "control_state_error", None, False) from exc


def _load_controls() -> FirecrawlControls:
    return FirecrawlControls.from_env()


CONTROLS = _load_controls()


async def firecrawl_request(
    client: httpx.AsyncClient,
    endpoint: str,
    payload: dict[str, Any],
    *,
    timeout: float,
) -> httpx.Response:
    return await CONTROLS.request(client, endpoint, payload, timeout=timeout)


async def firecrawl_status(
    client: httpx.AsyncClient,
    path: str,
    *,
    timeout: float,
) -> httpx.Response:
    return await CONTROLS.status(client, path, timeout=timeout)


def firecrawl_usage_summary() -> dict[str, Any] | None:
    return CONTROLS.usage_summary()


async def firecrawl_record_job_credits(endpoint: str, job_id: str, credits: Any) -> None:
    await CONTROLS.record_job_credits(endpoint, job_id, credits)
