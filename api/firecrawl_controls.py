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
USAGE_TAIL_BYTES = 512 * 1024
MAX_RETRY_AFTER = 86_400.0
COST_SENSITIVE_DOMAINS = (
    "linkedin.com",
    "lnkd.in",
    "x.com",
    "twitter.com",
    "mobile.twitter.com",
    "t.co",
)


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
    ):
        self._budgets = budgets
        self._config_error = config_error
        self._state_path = Path(state_path)
        self._lock_path = Path(lock_path)
        self._usage_path = Path(usage_path) if usage_path else None

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
        started = time.perf_counter()
        outcome: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "request_id": current_request_id(),
            "endpoint": endpoint,
            "label": None,
            "status": None,
            "credits": 0.0,
            "category": None,
            "elapsed_ms": None,
        }
        attempted = False
        fd = await self._acquire_lock(deadline)
        try:
            state = self._read_state()
            exhausted_error = FirecrawlUnavailable(402, "team_exhausted", USAGE_RECHECK_SECONDS, True)
            last_error = exhausted_error
            for budget in self._budgets:
                group = self._group_state(state, budget.id)
                blocked = await self._refresh_block(client, budget, group, state, deadline)
                if blocked is not None:
                    if blocked.category in ("team_exhausted", "invalid_auth"):
                        last_error = blocked if blocked.category == "invalid_auth" else exhausted_error
                        continue
                    raise blocked

                remaining = deadline - loop.time()
                if remaining <= 0:
                    raise FirecrawlUnavailable(None, "request_timeout", 1.0, True)
                key = budget.keys[0]
                outcome.update(label=budget.label, status=None, credits=0.0, category=None)
                attempted = True
                try:
                    request = client.post(
                        f"{API_ROOT}{endpoint}",
                        json=request_payload,
                        headers={"Authorization": "Bearer " + key, "Accept": "application/json"},
                        timeout=remaining,
                    )
                    response = await asyncio.wait_for(request, timeout=max(0.0, deadline - loop.time()))
                except (asyncio.TimeoutError, httpx.TimeoutException) as exc:
                    outcome["category"] = "transport_timeout"
                    raise FirecrawlUnavailable(None, "transport_timeout", 1.0, True) from exc
                except httpx.HTTPError as exc:
                    outcome["category"] = "transport_error"
                    raise FirecrawlUnavailable(None, "transport_error", 1.0, True) from exc

                outcome["status"] = response.status_code

                if 200 <= response.status_code < 300:
                    outcome["credits"] = _response_credits(endpoint, response)
                    group.clear()
                    try:
                        self._write_state(state)
                    except FirecrawlUnavailable:
                        # The provider response is already billable; do not discard it.
                        logger.warning("firecrawl control state not persisted label=%s", budget.label)
                    logger.info(
                        "firecrawl request ok label=%s endpoint=%s elapsed_ms=%d credits=%.1f",
                        budget.label,
                        endpoint,
                        int((timeout - max(0, deadline - loop.time())) * 1000),
                        outcome["credits"],
                    )
                    return response

                if response.status_code == 402:
                    usage = await self._credit_usage(client, key, deadline, budget.label)
                    if usage is not None and usage["remaining"] <= 0:
                        self._block(group, "team_exhausted", usage)
                        self._write_state(state)
                        logger.warning("firecrawl key exhausted label=%s remaining=0", budget.label)
                        last_error = exhausted_error
                        continue
                    self._block(group, "key_cap_or_ambiguous", usage)
                    self._write_state(state)
                    logger.warning("firecrawl key blocked label=%s category=key_cap_or_ambiguous", budget.label)
                    raise FirecrawlUnavailable(402, "key_cap_or_ambiguous", USAGE_RECHECK_SECONDS, True)

                if response.status_code == 429:
                    delay = _retry_after(response)
                    group.clear()
                    group.update(blocked="rate_limited", next_check=time.monotonic() + delay)
                    self._write_state(state)
                    logger.warning("firecrawl key rate limited label=%s retry_after=%.0f", budget.label, delay)
                    raise FirecrawlUnavailable(429, "rate_limited", delay, True)

                if response.status_code == 401:
                    self._block(group, "invalid_auth", None)
                    self._write_state(state)
                    logger.warning("firecrawl key blocked label=%s category=invalid_auth", budget.label)
                    last_error = FirecrawlUnavailable(401, "invalid_auth", USAGE_RECHECK_SECONDS, True)
                    continue
                if response.status_code == 403:
                    raise FirecrawlUnavailable(403, "policy_denied", None, False)
                if response.status_code >= 500:
                    raise FirecrawlUnavailable(response.status_code, "provider_unavailable", 1.0, True)

                # The adapter owns target-level errors (invalid URLs, unsupported
                # content, no record, and page status); do not rotate a key for them.
                if endpoint == "/v2/scrape":
                    outcome["credits"] = 1.0
                return response

            raise last_error
        except FirecrawlUnavailable as exc:
            if outcome["status"] is None:
                outcome["status"] = exc.status
            outcome["category"] = exc.category
            raise
        finally:
            if attempted:
                outcome["elapsed_ms"] = int((time.perf_counter() - started) * 1000)
                self._append_usage(outcome)
            self._release_lock(fd)

    async def _refresh_block(
        self,
        client: httpx.AsyncClient,
        budget: _Budget,
        group: dict[str, Any],
        state: dict[str, Any],
        deadline: float,
    ) -> FirecrawlUnavailable | None:
        blocked = group.get("blocked")
        if not blocked:
            return None
        now = time.monotonic()
        next_check = group.get("next_check")
        if isinstance(next_check, (int, float)) and now < next_check:
            retry_after = max(1.0, next_check - now)
            if blocked == "team_exhausted":
                return FirecrawlUnavailable(402, blocked, retry_after, True)
            if blocked == "rate_limited":
                return FirecrawlUnavailable(429, blocked, retry_after, True)
            if blocked == "invalid_auth":
                return FirecrawlUnavailable(401, blocked, min(retry_after, USAGE_RECHECK_SECONDS), True)
            return FirecrawlUnavailable(402, "key_cap_or_ambiguous", retry_after, True)

        if blocked == "rate_limited":
            group.clear()
            return None
        if blocked == "invalid_auth":
            # Only a caller retries the same key; another 401 re-arms the guard.
            group.clear()
            self._write_state(state)
            logger.info("firecrawl same-key retry allowed after cooldown label=%s", budget.label)
            return None

        key = budget.keys[0]
        usage = await self._credit_usage(client, key, deadline, budget.label)
        if usage is None:
            self._block(group, "key_cap_or_ambiguous", None)
            self._write_state(state)
            return FirecrawlUnavailable(402, "key_cap_or_ambiguous", USAGE_RECHECK_SECONDS, True)
        if usage["remaining"] <= 0:
            self._block(group, "team_exhausted", usage)
            self._write_state(state)
            return FirecrawlUnavailable(402, "team_exhausted", USAGE_RECHECK_SECONDS, True)

        if blocked == "team_exhausted":
            group.clear()
            self._write_state(state)
            logger.info("firecrawl key recovered label=%s remaining=%.0f", budget.label, usage["remaining"])
            return None

        # After this cached usage check, allow one caller-driven request on the same
        # key; success clears the block, while another 402 re-blocks its slot.
        self._block(group, "key_cap_or_ambiguous", usage)
        self._write_state(state)
        logger.info("firecrawl same-key retry allowed after cooldown label=%s", budget.label)
        return None

    async def _credit_usage(
        self, client: httpx.AsyncClient, key: str, deadline: float, label: str
    ) -> dict[str, Any] | None:
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            return None
        try:
            request = client.get(
                f"{API_ROOT}{CREDIT_USAGE_PATH}",
                headers={"Authorization": "Bearer " + key, "Accept": "application/json"},
                timeout=remaining,
            )
            response = await asyncio.wait_for(
                request, timeout=max(0.0, deadline - asyncio.get_running_loop().time())
            )
            if response.status_code != 200:
                return None
            body = response.json()
        except (asyncio.TimeoutError, httpx.HTTPError, ValueError):
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
    def _block(group: dict[str, Any], reason: str, usage: dict[str, Any] | None) -> None:
        group.clear()
        group.update(
            blocked=reason,
            next_check=time.monotonic() + USAGE_RECHECK_SECONDS,
            remaining=usage.get("remaining") if usage else None,
            billing_period_end=usage.get("period_end") if usage else None,
        )

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
            while True:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    return fd
                except BlockingIOError:
                    remaining = deadline - asyncio.get_running_loop().time()
                    if remaining <= 0:
                        raise FirecrawlUnavailable(None, "team_concurrency", 1.0, True)
                    await asyncio.sleep(min(0.05, remaining))
        except BaseException:
            os.close(fd)
            raise

    @staticmethod
    def _release_lock(fd: int) -> None:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)

    def _read_state(self) -> dict[str, Any]:
        try:
            fd = os.open(self._state_path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        except FileNotFoundError:
            return {}
        except OSError as exc:
            raise FirecrawlUnavailable(None, "control_state_error", None, False) from exc
        try:
            with os.fdopen(fd, "r", encoding="utf-8") as state_file:
                state = json.load(state_file)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise FirecrawlUnavailable(None, "control_state_error", None, False) from exc
        if not isinstance(state, dict):
            raise FirecrawlUnavailable(None, "control_state_error", None, False)
        groups = state.get("budgets", {})
        if not isinstance(groups, dict):
            raise FirecrawlUnavailable(None, "control_state_error", None, False)
        # Discard removed credentials and pre-pool operator labels, never reuse their guards.
        return {"budgets": {budget.id: groups[budget.id] for budget in self._budgets if budget.id in groups}}

    async def status(
        self,
        client: httpx.AsyncClient,
        path: str,
        *,
        timeout: float,
    ) -> httpx.Response:
        """Poll a Firecrawl job. Status reads are not billable; no key rotation."""
        if not math.isfinite(timeout) or timeout <= 0:
            raise FirecrawlUnavailable(None, "invalid_timeout", None, False)
        if not path.startswith("/v2/") or ".." in path:
            raise FirecrawlUnavailable(None, "invalid_request", None, False)
        if self._config_error:
            raise FirecrawlUnavailable(None, "invalid_configuration", None, False)
        if not self._budgets:
            raise FirecrawlUnavailable(None, "not_configured", None, False)
        # A job belongs to the key that created it, and request admission may
        # have rotated past the first budget. Status reads are free, so probe
        # each budget's first key until one knows the job.
        last = None
        for budget in self._budgets:
            key = budget.keys[0]
            try:
                response = await asyncio.wait_for(
                    client.get(
                        f"{API_ROOT}{path}",
                        headers={"Authorization": "Bearer " + key, "Accept": "application/json"},
                        timeout=timeout,
                    ),
                    timeout=timeout,
                )
            except (asyncio.TimeoutError, httpx.TimeoutException) as exc:
                raise FirecrawlUnavailable(None, "transport_timeout", 1.0, True) from exc
            except httpx.HTTPError as exc:
                raise FirecrawlUnavailable(None, "transport_error", 1.0, True) from exc
            if response.status_code == 404:
                last = response
                continue
            return response
        return last

    def usage_summary(self) -> dict[str, Any] | None:
        """Today's (UTC) ledger totals, read from the tail only."""
        if self._usage_path is None:
            return None
        try:
            with self._usage_path.open("rb") as ledger:
                ledger.seek(0, os.SEEK_END)
                size = ledger.tell()
                ledger.seek(max(0, size - USAGE_TAIL_BYTES))
                raw = ledger.read()
        except OSError:
            return None
        lines = raw.splitlines()
        if size > len(raw):
            lines = lines[1:]  # the first line may be a partial record.
        today = datetime.now(timezone.utc).date().isoformat()
        requests = 0
        credits = 0.0
        errors = 0
        by_endpoint: dict[str, dict[str, float]] = {}
        last_request_at = None
        for line in lines:
            try:
                entry = json.loads(line)
            except (TypeError, ValueError):
                continue
            if not isinstance(entry, dict) or not str(entry.get("ts", "")).startswith(today):
                continue
            endpoint = entry.get("endpoint")
            if not isinstance(endpoint, str):
                continue
            amount = entry.get("credits")
            amount = float(amount) if isinstance(amount, (int, float)) and not isinstance(amount, bool) else 0.0
            requests += 1
            credits += amount
            status = entry.get("status")
            if not isinstance(status, int) or not 200 <= status < 300:
                errors += 1
            bucket = by_endpoint.setdefault(endpoint, {"requests": 0, "credits": 0.0})
            bucket["requests"] += 1
            bucket["credits"] += amount
            ts = entry.get("ts")
            if isinstance(ts, str):
                last_request_at = ts
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
