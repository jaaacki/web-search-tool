"""Secret-safe shared controls for Firecrawl search and scrape requests."""

import asyncio
import fcntl
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

logger = logging.getLogger(__name__)

API_ROOT = "https://api.firecrawl.dev"
CREDIT_USAGE_PATH = "/v2/team/credit-usage"
STATE_PATH = Path("/tmp/websearch-firecrawl-controls.json")
LOCK_PATH = Path("/tmp/websearch-firecrawl-controls.lock")
USAGE_RECHECK_SECONDS = 900.0
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


def _parse_budgets(raw: str) -> tuple[_Budget, ...]:
    value = json.loads(raw)
    if not isinstance(value, list) or not value:
        raise ValueError("budget list required")

    budgets = []
    ids = set()
    keys = set()
    for item in value:
        if not isinstance(item, dict) or set(item) != {"id", "keys"}:
            raise ValueError("each budget requires only id and keys")
        name, raw_keys = item["id"], item["keys"]
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,31}", name) or name in ids:
            raise ValueError("invalid or duplicate budget id")
        if not isinstance(raw_keys, list) or not raw_keys:
            raise ValueError("each budget requires at least one key")
        group_keys = []
        for key in raw_keys:
            if not isinstance(key, str) or not key.strip() or key != key.strip() or key in keys:
                raise ValueError("invalid or duplicate key")
            group_keys.append(key)
            keys.add(key)
        ids.add(name)
        budgets.append(_Budget(name, tuple(group_keys)))
    return tuple(budgets)


def firecrawl_url_allowed(url: str) -> bool:
    """Whether a target avoids Firecrawl-routed domains and document parsing costs."""
    if not isinstance(url, str):
        return False
    try:
        parsed = urlsplit(url)
        host = (parsed.hostname or "").rstrip(".").lower()
    except ValueError:
        return False
    if parsed.scheme not in {"http", "https"} or parsed.username or parsed.password:
        return False
    path = parsed.path.lower()
    return bool(host) and not path.endswith(".pdf") and not any(
        host == domain or host.endswith(f".{domain}") for domain in COST_SENSITIVE_DOMAINS
    )


def _search_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if "scrapeOptions" in payload:
        raise FirecrawlUnavailable(None, "embedded_scrape_forbidden", None, False)
    result = dict(payload)
    excluded = result.get("excludeDomains", [])
    if not isinstance(excluded, list) or any(not isinstance(item, str) for item in excluded):
        raise FirecrawlUnavailable(None, "invalid_request", None, False)
    seen = {item.lower().rstrip(".") for item in excluded}
    result["excludeDomains"] = list(excluded) + [
        domain for domain in COST_SENSITIVE_DOMAINS if domain not in seen
    ]
    return result


def _scrape_payload(payload: dict[str, Any]) -> dict[str, Any]:
    formats = payload.get("formats", ["markdown"])
    parsers = payload.get("parsers", [])
    if formats != ["markdown"] or parsers != []:
        raise FirecrawlUnavailable(None, "unsupported_cost_options", None, False)
    if any(payload.get(option) for option in ("zeroDataRetention", "lockdown", "redactPII", "checkPromptInjection")):
        raise FirecrawlUnavailable(None, "unsupported_cost_options", None, False)
    if payload.get("storeInCache") not in (None, False):
        raise FirecrawlUnavailable(None, "unsupported_privacy_option", None, False)
    result = dict(payload)
    result["formats"] = ["markdown"]
    result["parsers"] = []
    result["storeInCache"] = False
    return result


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
    ):
        self._budgets = budgets
        self._config_error = config_error
        self._state_path = Path(state_path)
        self._lock_path = Path(lock_path)

    @classmethod
    def from_env(cls) -> "FirecrawlControls":
        raw = os.getenv("FIRECRAWL_BUDGETS_JSON", "").strip()
        if not raw:
            return cls(())
        try:
            return cls(_parse_budgets(raw))
        except (TypeError, ValueError, json.JSONDecodeError):
            logger.warning("Firecrawl budget configuration is invalid; Firecrawl is disabled")
            return cls((), config_error=True)

    async def request(
        self,
        client: httpx.AsyncClient,
        endpoint: Literal["/v2/search", "/v2/scrape"],
        payload: dict[str, Any],
        *,
        timeout: float,
    ) -> httpx.Response:
        if not math.isfinite(timeout) or timeout <= 0:
            raise FirecrawlUnavailable(None, "invalid_timeout", None, False)
        if endpoint not in ("/v2/search", "/v2/scrape") or not isinstance(payload, dict):
            raise FirecrawlUnavailable(None, "invalid_request", None, False)
        if self._config_error:
            raise FirecrawlUnavailable(None, "invalid_configuration", None, False)
        if not self._budgets:
            raise FirecrawlUnavailable(None, "not_configured", None, False)

        if endpoint == "/v2/search":
            request_payload = _search_payload(payload)
        else:
            if not firecrawl_url_allowed(payload.get("url")):
                raise FirecrawlUnavailable(None, "provider_routed_domain", None, False)
            request_payload = _scrape_payload(payload)

        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        fd = await self._acquire_lock(deadline)
        try:
            state = self._read_state()
            for budget in self._budgets:
                group = self._group_state(state, budget.id)
                blocked = await self._refresh_block(client, budget, group, deadline)
                if blocked is not None:
                    if blocked.category == "team_exhausted":
                        continue
                    raise blocked

                remaining = deadline - loop.time()
                if remaining <= 0:
                    raise FirecrawlUnavailable(None, "request_timeout", 1.0, True)
                # ponytail: one active key per team; rotate it only by an operator
                # config change after checking the team's key-spend policy.
                key = budget.keys[0]
                try:
                    request = client.post(
                        f"{API_ROOT}{endpoint}",
                        json=request_payload,
                        headers={"Authorization": "Bearer " + key, "Accept": "application/json"},
                        timeout=remaining,
                    )
                    response = await asyncio.wait_for(request, timeout=max(0.0, deadline - loop.time()))
                except (asyncio.TimeoutError, httpx.TimeoutException) as exc:
                    raise FirecrawlUnavailable(None, "transport_timeout", 1.0, True) from exc
                except httpx.HTTPError as exc:
                    raise FirecrawlUnavailable(None, "transport_error", 1.0, True) from exc

                if 200 <= response.status_code < 300:
                    group.clear()
                    try:
                        self._write_state(state)
                    except FirecrawlUnavailable:
                        # The provider response is already billable; do not discard it.
                        logger.warning("firecrawl control state not persisted budget=%s", budget.id)
                    logger.info(
                        "firecrawl request ok budget=%s endpoint=%s elapsed_ms=%d",
                        budget.id,
                        endpoint,
                        int((timeout - max(0, deadline - loop.time())) * 1000),
                    )
                    return response

                if response.status_code == 402:
                    usage = await self._credit_usage(client, key, deadline, budget.id)
                    if usage is not None and usage["remaining"] <= 0:
                        self._block(group, "team_exhausted", usage)
                        self._write_state(state)
                        logger.warning("firecrawl budget exhausted budget=%s remaining=0", budget.id)
                        continue
                    self._block(group, "key_cap_or_ambiguous", usage)
                    self._write_state(state)
                    logger.warning("firecrawl budget blocked budget=%s category=key_cap_or_ambiguous", budget.id)
                    raise FirecrawlUnavailable(402, "key_cap_or_ambiguous", USAGE_RECHECK_SECONDS, True)

                if response.status_code == 429:
                    delay = _retry_after(response)
                    group.clear()
                    group.update(blocked="rate_limited", next_check=time.monotonic() + delay)
                    self._write_state(state)
                    logger.warning("firecrawl budget rate limited budget=%s retry_after=%.0f", budget.id, delay)
                    raise FirecrawlUnavailable(429, "rate_limited", delay, True)

                if response.status_code == 401:
                    raise FirecrawlUnavailable(401, "invalid_auth", None, False)
                if response.status_code == 403:
                    raise FirecrawlUnavailable(403, "policy_denied", None, False)
                if response.status_code >= 500:
                    raise FirecrawlUnavailable(response.status_code, "provider_unavailable", 1.0, True)

                # The adapter owns target-level errors (invalid URLs, unsupported
                # content, no record, and page status); do not rotate a key for them.
                return response

            raise FirecrawlUnavailable(402, "team_exhausted", USAGE_RECHECK_SECONDS, True)
        finally:
            self._release_lock(fd)

    async def _refresh_block(
        self,
        client: httpx.AsyncClient,
        budget: _Budget,
        group: dict[str, Any],
        deadline: float,
    ) -> FirecrawlUnavailable | None:
        blocked = group.get("blocked")
        if not blocked:
            return None
        now = time.monotonic()
        wall_now = time.time()
        next_check = group.get("next_check")
        if isinstance(next_check, (int, float)) and now < next_check:
            retry_after = max(1.0, next_check - now)
            if blocked == "team_exhausted":
                return FirecrawlUnavailable(402, blocked, retry_after, True)
            if blocked == "rate_limited":
                return FirecrawlUnavailable(429, blocked, retry_after, True)
            return FirecrawlUnavailable(402, "key_cap_or_ambiguous", retry_after, True)

        if blocked == "rate_limited":
            group.clear()
            return None

        key = budget.keys[int(group.get("next_key", 0)) % len(budget.keys)]
        usage = await self._credit_usage(client, key, deadline, budget.id)
        if usage is None:
            self._block(group, "key_cap_or_ambiguous", None)
            return FirecrawlUnavailable(402, "key_cap_or_ambiguous", USAGE_RECHECK_SECONDS, True)
        remaining = usage["remaining"]
        if remaining <= 0:
            self._block(group, "team_exhausted", usage)
            return FirecrawlUnavailable(402, "team_exhausted", USAGE_RECHECK_SECONDS, True)

        previous = group.get("remaining")
        previous_period = group.get("billing_period_end")
        period_reset = (
            isinstance(previous_period, (int, float))
            and previous_period <= wall_now
            and isinstance(usage["period_end"], (int, float))
            and usage["period_end"] > wall_now
            and usage["period_end"] != previous_period
        )
        top_up = isinstance(previous, (int, float)) and remaining > previous
        if blocked == "team_exhausted" or top_up or period_reset:
            group.clear()
            logger.info("firecrawl budget recovered budget=%s remaining=%.0f", budget.id, remaining)
            return None

        self._block(group, "key_cap_or_ambiguous", usage)
        return FirecrawlUnavailable(402, "key_cap_or_ambiguous", USAGE_RECHECK_SECONDS, True)

    async def _credit_usage(
        self, client: httpx.AsyncClient, key: str, deadline: float, budget_id: str
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
        logger.info("firecrawl usage budget_checked budget=%s remaining=%.0f", budget_id, remaining)
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
        return state

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
    raw = os.getenv("FIRECRAWL_BUDGETS_JSON", "").strip()
    if not raw:
        return FirecrawlControls(())
    try:
        return FirecrawlControls(_parse_budgets(raw))
    except (TypeError, ValueError, json.JSONDecodeError):
        logger.warning("Firecrawl budget configuration is invalid; Firecrawl is disabled")
        return FirecrawlControls((), config_error=True)


CONTROLS = _load_controls()


async def firecrawl_request(
    client: httpx.AsyncClient,
    endpoint: Literal["/v2/search", "/v2/scrape"],
    payload: dict[str, Any],
    *,
    timeout: float,
) -> httpx.Response:
    return await CONTROLS.request(client, endpoint, payload, timeout=timeout)
