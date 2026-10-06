import asyncio
import ipaddress
import logging
import math
import os
import re
import secrets
import socket
import time
from collections import Counter
from contextlib import asynccontextmanager
from copy import deepcopy
from typing import Any, Literal
from urllib.parse import urlparse

import httpx
from fastapi import Depends, FastAPI, HTTPException, Query, Request, Security
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, ConfigDict, Field

SEARXNG_URL = os.getenv("SEARXNG_URL", "http://searxng:8080").rstrip("/")
CRAWL4AI_URL = os.getenv("CRAWL4AI_URL", "http://crawl4ai:11235").rstrip("/")
# crawl4ai >=0.9 refuses to bind non-loopback without an API token; send it as a bearer header.
CRAWL4AI_API_TOKEN = os.getenv("CRAWL4AI_API_TOKEN", "")
CRAWL4AI_HEADERS = {"Authorization": "Bearer " + CRAWL4AI_API_TOKEN} if CRAWL4AI_API_TOKEN else {}
RERANKER_URL = os.getenv("RERANKER_URL", "http://reranker:7997").rstrip("/")
SEARCH_CANDIDATES = int(os.getenv("SEARCH_CANDIDATES", "10"))
MAX_RESULTS = int(os.getenv("MAX_RESULTS", "5"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1800"))
# Whole-page chunking budget per crawled page. The old first-N-chunks limit kept
# roughly the nav bar of each page; this caps what one page can cost instead.
MAX_PAGE_CHARS = int(os.getenv("MAX_PAGE_CHARS", "60000"))
# `fit_markdown` below this length means pruning ate the page, so fall back to raw.
MIN_FIT_MARKDOWN_CHARS = 300
CRAWL_CONCURRENCY = int(os.getenv("CRAWL_CONCURRENCY", "8"))
# Chrome version the Crawl4AI image bundles (Chrome for Testing 153.0.8010.12 in 0.9.4).
# The UA we send has to match the engine that renders the page, so bump this with the image.
CRAWL4AI_CHROMIUM_VERSION = os.getenv("CRAWL4AI_CHROMIUM_VERSION", "153.0.8010.12")
WEBSEARCH_API_KEY = os.getenv("WEBSEARCH_API_KEY", "")
# Optional Brave Search API keys (comma-separated) for primary discovery.
# Empty = SearXNG-only discovery, exactly today's behavior.
BRAVE_API_KEYS = tuple(key.strip() for key in os.getenv("BRAVE_API_KEYS", "").split(",") if key.strip())
BRAVE_SEARCH_URL = os.getenv("BRAVE_SEARCH_URL", "https://api.search.brave.com/res/v1/web/search").rstrip("/")
BRAVE_TIMEOUT = float(os.getenv("BRAVE_TIMEOUT", "8"))
# Fences SearXNG discovery per request: settings.yml merges with defaults, so
# the allowlist lives here where every search routes through. Every engine named
# here must be active in the pinned image - an engine the image marks `inactive`
# is absent from /config and can never be re-enabled by config, which silently
# collapses discovery onto the remaining engines.
SEARXNG_ENGINES = os.getenv("SEARXNG_ENGINES", "bing,brave,duckduckgo,google cse")
# The canary query health sends through the real engine pool. It must be a term every
# configured engine can answer, because an engine set that cannot answer it is exactly
# the outage this probe exists to catch.
HEALTH_CANARY_QUERY = os.getenv("HEALTH_CANARY_QUERY", "open source web search engines")
# Bounded separately from SEARCH_TIMEOUT: health is polled, so it must not hold a
# caller for as long as a real search would. Both are per-request ceilings, not
# aggregate budgets.
HEALTH_TIMEOUT = float(os.getenv("HEALTH_TIMEOUT", "5"))
SEARCH_TIMEOUT = float(os.getenv("SEARCH_TIMEOUT", "10"))
# A poll interval shorter than this reuses the last health verdict instead of firing a
# real engine query again, so /health can be polled aggressively without turning into
# engine load of its own.
HEALTH_CACHE_TTL = float(os.getenv("HEALTH_CACHE_TTL", "15"))
# How long a caller should wait before retrying a downed pool. Engine outages clear in
# seconds-to-minutes, not milliseconds, and a hot retry loop from every agent is how a
# recovering backend gets knocked over again.
OUTAGE_RETRY_AFTER = float(os.getenv("OUTAGE_RETRY_AFTER", "30"))
DISCOVERY_TIMEOUT = float(os.getenv("DISCOVERY_TIMEOUT", "20"))
CRAWL_TIMEOUT = float(os.getenv("CRAWL_TIMEOUT", "45"))
RERANK_TIMEOUT = float(os.getenv("RERANK_TIMEOUT", "4"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "300"))
# Cheapest chunks the cross-encoder sees: lexical prefilter keeps this many. Sized so a
# CPU-only box finishes inside RERANK_TIMEOUT -- see the measurements on #27.
RERANK_PREFILTER = int(os.getenv("RERANK_PREFILTER", "10"))
# Characters per passage sent to the cross-encoder. Its cost scales with tokens, and the
# head of a chunk carries the match; results still return the whole chunk.
RERANK_MAX_CHARS = int(os.getenv("RERANK_MAX_CHARS", "512"))
# Documents per cross-encoder request; must not exceed the server's max client batch size.
RERANK_BATCH = int(os.getenv("RERANK_BATCH", "32"))
# After a cross-encoder failure the breaker stays open this long, ranking lexically:
# otherwise slow hardware pays the timeout on every single search.
RERANK_COOLDOWN = float(os.getenv("RERANK_COOLDOWN", "300"))
# How much the page's discovery position counts against the cross-encoder score.
RERANK_WEIGHT = float(os.getenv("RERANK_WEIGHT", "0.2"))
# Reciprocal-rank smoothing for the discovery term: rank 0 -> 1.0, rank 60 -> 0.5.
RERANK_RRF_K = 60
TOKEN_RE = re.compile(r"[\w-]+", re.UNICODE)

logger = logging.getLogger("websearch")
if not logger.handlers:  # uvicorn only configures its own loggers; without this our INFO is swallowed
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(_handler)
logger.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())


def _ms(since: float, until: float | None = None) -> float:
    """Milliseconds since `since`, or between `since` and `until`."""
    return ((time.perf_counter() if until is None else until) - since) * 1000


_shared_client: httpx.AsyncClient | None = None
_client_lock = asyncio.Lock()


async def get_client() -> httpx.AsyncClient:
    global _shared_client
    if _shared_client is None:
        async with _client_lock:
            if _shared_client is None:
                _shared_client = httpx.AsyncClient(
                    timeout=60,
                    follow_redirects=False,
                    limits=httpx.Limits(max_connections=50, max_keepalive_connections=20),
                )
    return _shared_client


@asynccontextmanager
async def lifespan(_: FastAPI):
    global _shared_client
    await get_client()  # pre-warm the upstream connection pool
    yield
    if _shared_client is not None:
        await _shared_client.aclose()
        _shared_client = None


app = FastAPI(
    title="SparkFN Web Search and Crawl API",
    version="0.1.0",
    lifespan=lifespan,
    description=(
        "Authenticated APIs for AI agents and MCP/CLI tools. "
        "Use the host-specific OpenAPI documents: websearch.sparkfn.io exposes search, "
        "and webcrawl.sparkfn.io exposes crawl."
    ),
    docs_url=None,
    openapi_url=None,
)
api_key_header = APIKeyHeader(
    name="X-API-Key",
    auto_error=False,
    description="Required for content endpoints. Send the provisioned WEBSEARCH_API_KEY value exactly as this header.",
)


class ErrorDetail(BaseModel):
    code: str = Field(
        description="Stable machine-readable error code. Agents should branch on this value instead of parsing message text.",
        examples=["unauthorized"],
    )
    message: str = Field(
        description="Human-readable error summary suitable for logs and end-user display.",
        examples=["A valid X-API-Key header is required"],
    )
    details: Any | None = Field(
        default=None,
        description="Optional structured details from validation or upstream services. May be null.",
        examples=[None],
    )
    retryable: bool | None = Field(
        default=None,
        description=(
            "Whether the same request can succeed later without the caller changing anything. "
            "True for a transient upstream fault (5xx, 429), false for a request the caller must "
            "fix — including a server misconfiguration, which retrying cannot repair. "
            "Defaults to the status class when a handler does not state it."
        ),
        examples=[True],
    )
    retry_after: float | None = Field(
        default=None,
        description=(
            "Seconds the caller should wait before retrying. Top-level beside `retryable`, not "
            "buried in `details`, so a retrying client can read it without parsing prose."
        ),
        examples=[30],
    )
    hint: str | None = Field(
        default=None,
        description="The next action to take, when one is knowable: a command to run, or the setting to change.",
        examples=["Set SEARXNG_ENGINES to a comma-separated engine list and restart the API."],
    )


class ErrorEnvelope(BaseModel):
    ok: Literal[False] = Field(default=False, description="Always false for failed requests.")
    error: ErrorDetail = Field(description="Error object with stable code, readable message, and optional details.")


class HealthData(BaseModel):
    services: dict[str, str] = Field(
        description="Internal service base URLs used by the stack. This is diagnostic metadata, not public crawl/search targets.",
        examples=[{"searxng": "http://searxng:8080", "crawl4ai": "http://crawl4ai:11235", "reranker": "http://reranker:7997"}],
    )
    checks: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Live per-service probe results. `searxng` is decided by a real canary query through the engine pool, so a total "
            "search-backend outage is reported here instead of passing as a healthy process."
        ),
        examples=[{"searxng": "ok", "crawl4ai": "ok", "reranker": "ok"}],
    )


class HealthEnvelope(BaseModel):
    ok: Literal[True] = Field(default=True, description="Always true when the API process can answer health checks.")
    data: HealthData


class SearchRequest(BaseModel):
    query: str = Field(
        min_length=1,
        description=(
            "Natural-language web search query. Use concise terms with enough context; do not pass a URL here. "
            "For a known URL, use the crawl API instead."
        ),
        examples=["open source web search engines", "latest Crawl4AI documentation cache_mode"],
    )
    max_results: int = Field(
        default=MAX_RESULTS,
        ge=1,
        le=20,
        description="Maximum number of final reranked results to return. For AI agents, 3-5 is usually enough; use more only when broad coverage is required.",
        examples=[5],
    )
    candidates: int = Field(
        default=SEARCH_CANDIDATES,
        ge=1,
        le=50,
        description=(
            "Number of search candidates to fetch before crawling and reranking. Higher values may improve recall but increase latency "
            "because more pages may be crawled."
        ),
        examples=[10],
    )
    depth: Literal["basic", "advanced"] = Field(
        default="advanced",
        description=(
            "How much work each result costs. `advanced` (default) crawls every candidate page and returns a cleaned passage from the page "
            "itself. `basic` skips crawling and returns the discovery snippet as `content`, so it answers in about a second instead of "
            "5-45; use it when you only need to choose URLs, and re-query with `advanced` (or crawl the URL) once you need the text."
        ),
        examples=["basic"],
    )


class SearchResult(BaseModel):
    title: str = Field(description="Best available page title from search discovery or the URL host.", examples=["SearXNG documentation"])
    url: str = Field(description="Public URL that was crawled and used for the returned content.", examples=["https://docs.searxng.org/"])
    snippet: str = Field(default="", description="Short discovery snippet from SearXNG. May be empty.", examples=["SearXNG is a free internet metasearch engine..."])
    content: str = Field(
        description="Extracted page text chunk suitable for LLM grounding, citations, or user-facing summaries.",
        examples=["SearXNG is a free internet metasearch engine which aggregates results from various search services..."],
    )
    score: float | None = Field(
        default=None,
        ge=0,
        le=1,
        description="Normalized relevance score from the reranker when available. Higher is better; null means no numeric score was produced.",
        examples=[0.87],
    )


class SearchData(BaseModel):
    query: str = Field(description="Original query string that was searched.", examples=["open source web search engines"])
    results: list[SearchResult] = Field(
        description="Reranked extracted results. Empty list is a successful no-results response, not an error.",
    )


class SearchEnvelope(BaseModel):
    ok: Literal[True] = Field(default=True, description="Always true for successful search responses, including empty result sets.")
    data: SearchData


CrawlOptionValue = str | int | float | bool | None | list[Any] | dict[str, Any]


class CrawlRequest(BaseModel):
    # Unknown fields are refused rather than dropped: `extraction_config` was
    # removed here, and silently ignoring it (or a typo) is the same failure
    # mode #10 fixed upstream.
    model_config = ConfigDict(extra="forbid")

    url: str = Field(
        min_length=1,
        description=(
            "Single public http(s) URL to crawl. Private/internal addresses, localhost, link-local/reserved IPs, "
            "and URLs containing username/password credentials are rejected before reaching Crawl4AI."
        ),
        examples=["https://example.com"],
    )
    content_format: Literal["markdown", "cleaned_html", "text", "html"] = Field(
        default="markdown",
        description=(
            "Preferred content field to return from Crawl4AI. The API falls back through other available extracted fields "
            "if the preferred format is absent. Use markdown for most LLM/MCP consumers."
        ),
        examples=["markdown"],
    )
    cache_mode: str | None = Field(
        default=None,
        description=(
            "Optional Crawl4AI cache mode, folded into `crawler_config.cache_mode` where Crawl4AI reads it. One of "
            "`enabled`, `disabled`, `read_only`, `write_only`, `bypass` (case-insensitive); an explicit `crawler_config.cache_mode` wins over "
            "this field. Leave null for the Crawl4AI default."
        ),
        examples=["bypass"],
    )
    browser_config: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Not accepted from the public API. The server owns browser and stealth settings, so any non-empty value is rejected with "
            "422 `validation_error`. Kept in the schema only so the rejection is explicit rather than a silent drop."
        ),
        examples=[{}],
    )
    crawler_config: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Crawl4AI crawler/run options, where Crawl4AI actually reads them. Each key must be in the server allowlist *and* each value a "
            "plain scalar or list of scalars; anything else is rejected with 422 `validation_error` naming the offending key. Nested objects "
            "(including `{\"type\": ...}` typed-object wrappers) and LLM/proxy/browser/JS keys such as `llm_config`, `proxy_config`, "
            "`check_robots_txt`, `link_preview_config`, `js_code` and `user_data_dir` are never forwarded. `wait_for` must be a CSS selector "
            "prefixed with `css:`, and `max_retries` is capped at 2. Wins over `crawl_options` and `cache_mode` on conflict."
        ),
        examples=[{"wait_until": "networkidle", "css_selector": "main"}],
    )
    extraction_config: dict[str, Any] = Field(
        default_factory=dict,
        deprecated=True,
        description=(
            "Deprecated and unsupported. Crawl4AI takes extraction settings in `crawler_config`, and this field was never honoured, so it is "
            "accepted only when empty: a non-empty value is rejected with 422 `validation_error`. Use `crawler_config`."
        ),
        examples=[{}],
    )
    crawl_options: dict[str, CrawlOptionValue] = Field(
        default_factory=dict,
        description=(
            "Crawler options merged into `crawler_config` before it is sent (Crawl4AI 0.9.x ignores top-level options, so passing them here "
            "as before would silently do nothing). Validated against the same scalar allowlist; `url` and `urls` are rejected so callers "
            "cannot bypass URL validation. `crawler_config` wins on conflict."
        ),
        examples=[{"screenshot": False, "word_count_threshold": 10}],
    )


class CrawlData(BaseModel):
    url: str = Field(description="Final public URL reported by Crawl4AI, or the requested URL when Crawl4AI does not provide one.", examples=["https://example.com"])
    content: str = Field(description="Extracted page content in the best available requested format. Empty extraction returns an error instead of an empty success.", examples=["# Example Domain\n\nThis domain is for use in illustrative examples in documents..."])


class CrawlEnvelope(BaseModel):
    ok: Literal[True] = Field(default=True, description="Always true for successful crawl responses.")
    data: CrawlData


SEARCH_SUCCESS_EXAMPLE = {
    "ok": True,
    "data": {
        "query": "open source web search engines",
        "results": [
            {
                "title": "SearXNG documentation",
                "url": "https://docs.searxng.org/",
                "snippet": "SearXNG is a free internet metasearch engine...",
                "content": "SearXNG is a free internet metasearch engine which aggregates results from various search services and databases.",
                "score": 0.87,
            }
        ],
    },
}

CRAWL_SUCCESS_EXAMPLE = {
    "ok": True,
    "data": {
        "url": "https://example.com",
        "content": "# Example Domain\n\nThis domain is for use in illustrative examples in documents.",
    },
}

ERROR_EXAMPLES = {
    "unauthorized": {
        "summary": "Missing or invalid API key",
        "value": {"ok": False, "error": {"code": "unauthorized", "message": "A valid X-API-Key header is required", "details": None}},
    },
    "validation_error": {
        "summary": "Request body or query parameters are invalid",
        "value": {"ok": False, "error": {"code": "validation_error", "message": "Request validation failed", "details": []}},
    },
    "invalid_url": {
        "summary": "Crawl URL is not a public http(s) URL",
        "value": {"ok": False, "error": {"code": "invalid_url", "message": "URL must be a public http(s) URL", "details": None}},
    },
    "crawl4ai_error": {
        "summary": "Internal Crawl4AI service failed",
        "value": {"ok": False, "error": {"code": "crawl4ai_error", "message": "Crawl4AI crawl failed", "details": {"error": "upstream error details"}}},
    },
    "empty_crawl_result": {
        "summary": "Crawl succeeded upstream but no extractable content was returned",
        "value": {"ok": False, "error": {"code": "empty_crawl_result", "message": "Crawl4AI did not return extractable content", "details": None}},
    },
    "searxng_error": {
        "summary": "Internal SearXNG service failed",
        "value": {"ok": False, "error": {"code": "searxng_error", "message": "SearXNG search failed", "details": {"error": "upstream error details"}}},
    },
    "reranker_error": {
        "summary": "Internal reranker service failed",
        "value": {"ok": False, "error": {"code": "reranker_error", "message": "Reranker failed", "details": {"error": "upstream error details"}}},
    },
}

SEARCH_ERROR_RESPONSES = {
    401: {
        "model": ErrorEnvelope,
        "description": "Missing or invalid `X-API-Key`. Add the provisioned API key and retry.",
        "content": {"application/json": {"examples": {"unauthorized": ERROR_EXAMPLES["unauthorized"]}}},
    },
    422: {
        "model": ErrorEnvelope,
        "description": "Invalid request. Fix the query/body values before retrying.",
        "content": {"application/json": {"examples": {"validation_error": ERROR_EXAMPLES["validation_error"]}}},
    },
    502: {
        "model": ErrorEnvelope,
        "description": "A search, crawl, or rerank upstream service failed. Retry later or reduce `candidates` if timeouts persist.",
        "content": {"application/json": {"examples": {"searxng_error": ERROR_EXAMPLES["searxng_error"], "crawl4ai_error": ERROR_EXAMPLES["crawl4ai_error"], "reranker_error": ERROR_EXAMPLES["reranker_error"]}}},
    },
    500: {"model": ErrorEnvelope, "description": "Unexpected server error. Retry later or contact the service owner."},
}

CRAWL_ERROR_RESPONSES = {
    401: {
        "model": ErrorEnvelope,
        "description": "Missing or invalid `X-API-Key`. Add the provisioned API key and retry.",
        "content": {"application/json": {"examples": {"unauthorized": ERROR_EXAMPLES["unauthorized"]}}},
    },
    422: {
        "model": ErrorEnvelope,
        "description": "Invalid request. Most often this is a non-public URL, unsupported URL scheme, credentialed URL, or malformed body.",
        "content": {"application/json": {"examples": {"validation_error": ERROR_EXAMPLES["validation_error"], "invalid_url": ERROR_EXAMPLES["invalid_url"]}}},
    },
    502: {
        "model": ErrorEnvelope,
        "description": "Crawl4AI failed or returned no extractable content. Retry later, use a simpler URL, or adjust advanced crawl options.",
        "content": {"application/json": {"examples": {"crawl4ai_error": ERROR_EXAMPLES["crawl4ai_error"], "empty_crawl_result": ERROR_EXAMPLES["empty_crawl_result"]}}},
    },
    500: {"model": ErrorEnvelope, "description": "Unexpected server error. Retry later or contact the service owner."},
}

HEALTH_ERROR_RESPONSES = {
    503: {
        "model": ErrorEnvelope,
        "description": "The search backend is unavailable: no engine responded to a canary query. Retry later.",
        "content": {"application/json": {"examples": {"upstream_unavailable": ERROR_EXAMPLES["searxng_error"]}}},
    },
    500: {"model": ErrorEnvelope, "description": "Unexpected server error. Retry later or contact the service owner."},
}


class AppError(Exception):
    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        details: Any | None = None,
        retryable: bool | None = None,
        retry_after: float | None = None,
        hint: str | None = None,
    ):
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details = details
        self.retryable = retryable
        self.retry_after = retry_after
        self.hint = hint


async def require_api_key(x_api_key: str | None = Security(api_key_header)):
    if not WEBSEARCH_API_KEY:
        raise AppError(500, "api_key_not_configured", "WEBSEARCH_API_KEY is not configured")
    if not x_api_key or not secrets.compare_digest(x_api_key, WEBSEARCH_API_KEY):
        raise AppError(401, "unauthorized", "A valid X-API-Key header is required")


@app.exception_handler(AppError)
async def app_error_handler(_: Request, exc: AppError):
    return error_response(
        exc.status_code,
        exc.code,
        exc.message,
        exc.details,
        retryable=exc.retryable,
        retry_after=exc.retry_after,
        hint=exc.hint,
    )


@app.exception_handler(RequestValidationError)
async def validation_error_handler(_: Request, exc: RequestValidationError):
    return error_response(422, "validation_error", "Request validation failed", exc.errors())


@app.exception_handler(HTTPException)
async def http_error_handler(_: Request, exc: HTTPException):
    message = exc.detail if isinstance(exc.detail, str) else "HTTP error"
    return error_response(exc.status_code, "http_error", message, exc.detail if not isinstance(exc.detail, str) else None)


@app.exception_handler(Exception)
async def unhandled_error_handler(_: Request, exc: Exception):
    return error_response(500, "internal_error", "Internal server error", {"type": type(exc).__name__})


async def probe_service(base_url: str, path: str, headers: dict[str, str] | None = None) -> str:
    """Reachability probe for a supporting service. Reported, never fatal on its own."""
    client = await get_client()
    try:
        response = await client.get(f"{base_url}{path}", headers=headers, timeout=HEALTH_TIMEOUT)
        return "ok" if response.status_code < 500 else f"unavailable: http_{response.status_code}"
    except (httpx.HTTPError, ValueError) as exc:
        return f"unavailable: {type(exc).__name__}"


async def _run_health_checks() -> dict[str, str]:
    """Probe every service. Returns the checks map; raises AppError when search is down.

    Every service is probed even after searxng fails, so the 503 body carries the same
    information as the healthy one: an operator reading a failing health check learns
    which dependency broke, not merely that something did.
    """
    client = await get_client()
    checks: dict[str, str] = {}
    search_failure: AppError | None = None
    try:
        # Reuses search_searxng, so health and search cannot disagree about an outage: a
        # total engine failure raises upstream_unavailable here exactly as it does there.
        # Bounded by HEALTH_TIMEOUT, not SEARCH_TIMEOUT — health is polled.
        await search_searxng(client, HEALTH_CANARY_QUERY, 1, timeout=HEALTH_TIMEOUT)
        checks["searxng"] = "ok"
    except AppError as exc:
        search_failure = exc
        checks["searxng"] = f"unavailable: {exc.code}"
    # Supporting services are reported, never fatal: search degrades to its lexical path
    # without the reranker, so failing health on it would report an outage we do not have.
    checks["crawl4ai"] = await probe_service(CRAWL4AI_URL, "/health", CRAWL4AI_HEADERS)
    checks["reranker"] = await probe_service(RERANKER_URL, "/health")
    if search_failure is not None:
        raise AppError(
            503,
            "upstream_unavailable",
            "No search engine responded to a canary query; search is unavailable",
            {"checks": checks, **(search_failure.details if isinstance(search_failure.details, dict) else {})},
            retryable=True,
        )
    return checks


# Last verdict, reused for HEALTH_CACHE_TTL so a poll loop cannot become engine load.
# Caches the failure too: a dead backend should not be hammered by its own health check.
_HEALTH_CACHE: dict[str, object] = {"at": 0.0, "result": None}


def health_cache_reset() -> None:
    _HEALTH_CACHE["at"] = 0.0
    _HEALTH_CACHE["result"] = None


@app.get(
    "/health",
    response_model=HealthEnvelope,
    responses=HEALTH_ERROR_RESPONSES,
    summary="Check service health",
    description=(
        "Public diagnostic endpoint for the search and crawl API hosts. It confirms the API process is running, reports the "
        "internal service URLs configured for SearXNG, Crawl4AI, and the reranker, and probes each service. The SearXNG probe "
        "is a real canary query through the engine pool, so a search-backend outage is reported as `upstream_unavailable` "
        "instead of a healthy process. Results are cached for a few seconds so the endpoint can be polled cheaply. "
        "Do not use this endpoint for search or crawl work."
    ),
    operation_id="check_health",
    tags=["Diagnostics"],
)
async def health():
    now = time.monotonic()
    cached_at = _HEALTH_CACHE["at"]
    if isinstance(cached_at, float) and now - cached_at < HEALTH_CACHE_TTL:
        cached = _HEALTH_CACHE["result"]
        if isinstance(cached, AppError):
            raise cached
        if isinstance(cached, dict):
            return HealthEnvelope(data=HealthData(services=_health_services(), checks=cached))
    try:
        checks = await _run_health_checks()
    except AppError as exc:
        _HEALTH_CACHE["at"] = time.monotonic()
        _HEALTH_CACHE["result"] = exc
        raise
    _HEALTH_CACHE["at"] = time.monotonic()
    _HEALTH_CACHE["result"] = checks
    return HealthEnvelope(data=HealthData(services=_health_services(), checks=checks))


def _health_services() -> dict[str, str]:
    return {"searxng": SEARXNG_URL, "crawl4ai": CRAWL4AI_URL, "reranker": RERANKER_URL}


@app.get("/docs", include_in_schema=False)
def docs(request: Request):
    title = "AI Crawl API" if api_surface(request) == "crawl" else "AI Search API"
    return get_swagger_ui_html(openapi_url="/openapi.json", title=f"{title} - Swagger UI")


@app.get("/openapi.json", include_in_schema=False)
def openapi(request: Request):
    surface = api_surface(request)
    return filtered_openapi(surface)


@app.post(
    "/search",
    response_model=SearchEnvelope,
    responses=SEARCH_ERROR_RESPONSES,
    dependencies=[Depends(require_api_key)],
    summary="Search the web and return extracted, reranked page content",
    description=(
        "Use this endpoint when the caller has a question or topic and needs current web evidence. The service discovers candidate "
        "URLs with SearXNG, crawls public pages through internal Crawl4AI, chunks extracted text, reranks chunks against the query, "
        "and returns normalized results. Prefer this POST endpoint for MCP/CLI tools because the request body is explicit and easy to validate. "
        "If `data.results` is empty, the request succeeded but no usable pages were found."
    ),
    operation_id="search_web",
    tags=["Search"],
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "examples": {
                        "agent_default": {
                            "summary": "Recommended agent search",
                            "value": {"query": "open source web search engines", "max_results": 5, "candidates": 10},
                        },
                        "higher_recall": {
                            "summary": "Broader recall with more crawling",
                            "value": {"query": "Crawl4AI cache_mode documentation", "max_results": 8, "candidates": 20},
                        },
                    }
                }
            }
        },
        "responses": {"200": {"content": {"application/json": {"examples": {"success": {"summary": "Search results", "value": SEARCH_SUCCESS_EXAMPLE}}}}}},
        "x-ai-guidance": "Call this when you need discovery. Use concise natural-language queries. Start with max_results=5 and candidates=10; increase candidates only when recall matters more than latency.",
    },
)
async def search(request: SearchRequest):
    client = await get_client()
    started = time.perf_counter()
    try:
        candidates = await asyncio.wait_for(
            discover_candidates(client, request.query, request.candidates),
            timeout=DISCOVERY_TIMEOUT,
        )
    except asyncio.TimeoutError:
        logger.warning("discovery timed out after %.0fs; trying searxng directly", DISCOVERY_TIMEOUT)
        try:
            candidates = await asyncio.wait_for(
                search_searxng(client, request.query, request.candidates), timeout=10
            )
        except (asyncio.TimeoutError, AppError) as exc:
            logger.warning("searxng fallback also failed (%r)", exc)
            candidates = []
    if not candidates:
        logger.info("search q=%r results=0 elapsed=%.0fms stage=discovery", request.query, _ms(started))
        return search_response(request.query, [])

    discovered = time.perf_counter()
    if request.depth == "basic":
        pages = snippet_pages(candidates)
    else:
        try:
            pages = await asyncio.wait_for(crawl_all(client, candidates), timeout=CRAWL_TIMEOUT)
        except asyncio.TimeoutError:
            logger.warning("crawl timed out after %.0fs", CRAWL_TIMEOUT)
            pages = []
    if not pages:
        logger.info(
            "search q=%r depth=%s candidates=%d results=0 discovery=%.0fms crawl=%.0fms elapsed=%.0fms",
            request.query, request.depth, len(candidates), _ms(started, discovered), _ms(discovered), _ms(started),
        )
        return search_response(request.query, [])

    crawled = time.perf_counter()
    # Basic pages are already the passages: a snippet is a clean summary, and the
    # boilerplate filter is tuned for page markdown (it drops short lines, which is
    # most snippets), so they go to the reranker as-is.
    chunks = pages if request.depth == "basic" else chunk_documents(pages)
    ranked, rerank_path = await rerank(client, request.query, chunks, request.max_results)
    logger.info(
        "search q=%r depth=%s candidates=%d pages=%d chunks=%d results=%d discovery=%.0fms crawl=%.0fms rerank=%.0fms elapsed=%.0fms rerank_path=%s",
        request.query, request.depth, len(candidates), len(pages), len(chunks), len(ranked),
        _ms(started, discovered), _ms(discovered, crawled), _ms(crawled), _ms(started), rerank_path,
    )

    return search_response(request.query, ranked)


@app.get(
    "/search",
    response_model=SearchEnvelope,
    responses=SEARCH_ERROR_RESPONSES,
    dependencies=[Depends(require_api_key)],
    summary="Search shortcut for simple clients",
    description=(
        "Query-string shortcut with the same behavior as POST /search. This is useful for manual curl calls and simple HTTP clients. "
        "Structured MCP/CLI integrations should prefer POST /search so all inputs are in a JSON body."
    ),
    operation_id="search_web_get",
    tags=["Search"],
    openapi_extra={
        "responses": {"200": {"content": {"application/json": {"examples": {"success": {"summary": "Search results", "value": SEARCH_SUCCESS_EXAMPLE}}}}}},
        "x-ai-guidance": "Prefer POST /search for tool calls. Use this GET shortcut only when a client cannot send JSON bodies.",
    },
)
async def search_get(
    q: str = Query(
        min_length=1,
        description="Natural-language web search query. Same as SearchRequest.query.",
        examples=["open source web search engines"],
    ),
    max_results: int = Query(
        default=MAX_RESULTS,
        ge=1,
        le=20,
        description="Maximum final reranked results to return. Same as SearchRequest.max_results.",
        examples=[5],
    ),
    candidates: int = Query(
        default=SEARCH_CANDIDATES,
        ge=1,
        le=50,
        description="Candidate discovery count before crawling/reranking. Same as SearchRequest.candidates.",
        examples=[10],
    ),
    depth: Literal["basic", "advanced"] = Query(
        default="advanced",
        description="Skip page crawling and return discovery snippets. Same as SearchRequest.depth.",
        examples=["basic"],
    ),
):
    return await search(SearchRequest(query=q, max_results=max_results, candidates=candidates, depth=depth))


@app.post(
    "/crawl",
    response_model=CrawlEnvelope,
    responses=CRAWL_ERROR_RESPONSES,
    dependencies=[Depends(require_api_key)],
    summary="Crawl one public URL and return extracted content",
    description=(
        "Use this endpoint when the caller already has a URL and needs clean page content. This is a safe public facade over internal Crawl4AI: "
        "it validates that the URL is public http(s), rejects localhost/private/internal targets, calls Crawl4AI on the private Docker network, "
        "and returns only normalized `url` and `content`. Do not use this endpoint for discovery; call the search API first when you do not already know the URL."
    ),
    operation_id="crawl_url",
    tags=["Crawl"],
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "examples": {
                        "basic_markdown": {
                            "summary": "Recommended simple crawl",
                            "value": {"url": "https://example.com", "content_format": "markdown"},
                        },
                        "advanced_options": {
                            "summary": "Advanced Crawl4AI pass-through options",
                            "value": {
                                "url": "https://example.com",
                                "content_format": "markdown",
                                "cache_mode": "BYPASS",
                                "browser_config": {"headless": True},
                                "crawler_config": {"wait_until": "networkidle"},
                                "crawl_options": {"word_count_threshold": 10},
                            },
                        },
                    }
                }
            }
        },
        "responses": {"200": {"content": {"application/json": {"examples": {"success": {"summary": "Crawled content", "value": CRAWL_SUCCESS_EXAMPLE}}}}}},
        "x-ai-guidance": "Call this only for known URLs. Always send a public http(s) URL. Leave advanced config fields empty unless you specifically need Crawl4AI behavior.",
    },
)
async def crawl(request: CrawlRequest):
    client = await get_client()
    result = await crawl_direct_url(client, request)

    return CrawlEnvelope(data=CrawlData(url=result["url"], content=result["content"]))


class _BraveKeyError(Exception):
    def __init__(self, reason: str, cooldown: float):
        super().__init__(reason)
        self.reason = reason
        self.cooldown = cooldown


class BraveKeyPool:
    """Round-robin Brave keys with per-key cooldowns. One sick key never blocks the healthy ones."""

    def __init__(self, keys: tuple[str, ...]):
        self._keys = list(keys)
        self._index = 0
        self._lock = asyncio.Lock()
        self._cooled_until = [0.0] * len(self._keys)
        # ponytail: in-memory counters reset on restart; persist when real metrics exist
        self._calls = [0] * len(self._keys)

    def __bool__(self):
        return bool(self._keys)

    async def search(self, client: httpx.AsyncClient, query: str, limit: int):
        """Candidate dicts, or None when the whole pool is unusable (caller falls back)."""
        tried = 0
        while tried < len(self._keys):
            key_idx = await self._take_key()
            if key_idx is None:
                logger.warning("brave pool exhausted (all %d keys cooling)", len(self._keys))
                return None
            tried += 1
            try:
                results = await self._search_with_key(client, key_idx, query, limit)
            except _BraveKeyError as exc:
                self._cool(key_idx, exc.cooldown)
                logger.warning("brave key %d failed (%s), cooling %.0fs", key_idx, exc.reason, exc.cooldown)
                continue
            self._calls[key_idx] += 1
            logger.info("brave key %d ok (%d candidates, %d calls)", key_idx, len(results), self._calls[key_idx])
            return results
        return None

    async def _take_key(self):
        async with self._lock:
            now = asyncio.get_running_loop().time()
            for _ in self._keys:
                idx = self._index
                self._index = (self._index + 1) % len(self._keys)
                if self._cooled_until[idx] <= now:
                    return idx
        return None

    def _cool(self, key_idx: int, cooldown: float):
        self._cooled_until[key_idx] = asyncio.get_running_loop().time() + max(cooldown, 0.0)

    async def _search_with_key(self, client: httpx.AsyncClient, key_idx: int, query: str, limit: int):
        items: list[dict] = []
        offset = 0
        try:
            while len(items) < limit and offset < limit:
                count = min(limit - len(items), 20)  # Brave page cap
                response = await client.get(
                    BRAVE_SEARCH_URL,
                    params={"q": query, "count": count, "offset": offset, "text_decorations": False},
                    headers={"X-Subscription-Token": self._keys[key_idx], "Accept": "application/json"},
                    timeout=BRAVE_TIMEOUT,
                )
                if response.status_code == 429:
                    raise _BraveKeyError("rate_limited", _retry_after(response, 300.0))
                if response.status_code in (401, 403):
                    raise _BraveKeyError(f"http_{response.status_code}", 86400.0)
                if 400 <= response.status_code < 500:
                    # Bad request, not a sick key (e.g. paging past Brave's
                    # offset cap): keep what we have instead of burning the pool.
                    logger.warning("brave key %d bad request (http_%d), stopping paging", key_idx, response.status_code)
                    break
                response.raise_for_status()
                page = (response.json().get("web") or {}).get("results") or []
                if not page:
                    break
                items.extend(page)
                offset += len(page)
        except _BraveKeyError:
            raise
        except (httpx.HTTPError, ValueError) as exc:
            raise _BraveKeyError(str(exc), 60.0) from exc
        return items


def _retry_after(response: httpx.Response, default: float) -> float:
    try:
        return max(float(response.headers.get("Retry-After", default)), 0.0)
    except (TypeError, ValueError):
        return default


BRAVE_POOL = BraveKeyPool(BRAVE_API_KEYS)


async def discover_candidates(client: httpx.AsyncClient, query: str, limit: int):
    if BRAVE_POOL:
        try:
            brave = await BRAVE_POOL.search(client, query, limit)
        except Exception as exc:  # never let the primary provider break search; SearXNG is the fallback
            logger.warning("brave pool error (%r), falling back to searxng", exc)
            brave = None
        if brave:
            shaped = await shape_candidates(brave, limit, snippet_key="description")
            if len(shaped) < limit:  # Brave pages cap out; top up the shortfall from SearXNG
                try:
                    extra = await search_searxng(client, query, limit)
                except AppError as exc:
                    logger.warning("searxng top-up failed (%s)", exc.code)
                    extra = []
                seen_urls = {item["url"] for item in shaped}
                shaped.extend(item for item in extra if item["url"] not in seen_urls)
                shaped = shaped[:limit]
            if shaped:
                return shaped
    return await search_searxng(client, query, limit)


async def shape_candidates(items, limit: int, snippet_key: str = "content"):
    seen = set()
    results = []
    for item in items:
        if not isinstance(item, dict):
            continue
        url = item.get("url")
        if not isinstance(url, str) or url in seen:
            continue
        seen.add(url)
        if not await is_allowed_public_url(url):
            continue
        results.append(
            {
                "title": item.get("title") or urlparse(url).netloc or url,
                "url": url,
                "snippet": item.get(snippet_key) or "",
            }
        )
        if len(results) >= limit:
            break
    return results


def _engine_names() -> list[str]:
    return [name.strip().lower() for name in SEARXNG_ENGINES.split(",") if name.strip()]


# Raised before any query is sent: with no engine configured there is nothing to ask,
# so this is a misconfiguration rather than an upstream fault.
class _NoEnginesError(AppError):
    def __init__(self) -> None:
        super().__init__(
            503,
            "upstream_unavailable",
            "No search engine is configured, so search cannot run",
            {"configured_engines": SEARXNG_ENGINES, "engine_count": 0},
            # Retrying cannot fix this: the operator has to configure the pool.
            retryable=False,
            hint=(
                "Set SEARXNG_ENGINES to a comma-separated engine list "
                f"(currently {SEARXNG_ENGINES!r}) and restart the API."
            ),
        )


async def search_searxng(client: httpx.AsyncClient, query: str, limit: int, timeout: float = SEARCH_TIMEOUT):
    configured = _engine_names()
    # Zero configured engines must never look like an empty web. With no engine list,
    # searxng returns nothing and reports nothing dead, so the "all engines failed"
    # test below cannot fire and the caller is handed ok:true with zero results — the
    # exact silent-empty failure #986 exists to remove (#986 follow-up).
    if not configured:
        raise _NoEnginesError()

    try:
        response = await client.get(
            f"{SEARXNG_URL}/search",
            params={"q": query, "format": "json", "engines": SEARXNG_ENGINES},
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise AppError(502, "searxng_error", "SearXNG search failed", {"error": str(exc)}) from exc

    unresponsive = payload.get("unresponsive_engines") or []
    if unresponsive:
        # A dead engine otherwise only shows up as fewer or zero results.
        logger.warning("searxng engines unresponsive: %s", ", ".join(str(entry) for entry in unresponsive))

    raw_results = payload.get("results", [])
    candidates = await shape_candidates(raw_results, limit)
    # An outage is EVERY configured engine failing, not one of them. searxng reports the
    # engines that failed, so a single answerer is proof the pool is partly alive: an
    # empty result set from a live engine is a real answer ("nothing matched") and must
    # stay `ok: true, results: []`, or a partial outage becomes a fake retry loop (#986).
    #
    # The test is on the RAW payload, not the shaped candidates: shape_candidates drops
    # non-public and duplicate URLs, so a healthy engine whose rows were all filtered
    # out would otherwise be misreported as an outage and retried forever.
    dead = {str(entry).strip().lower() for entry in unresponsive}
    answered = bool(raw_results) or bool(set(configured) - dead)
    if not answered and dead:
        raise AppError(
            503,
            "upstream_unavailable",
            "No search engine responded; the search backend is unavailable",
            {"unresponsive_engines": sorted(dead)},
            retryable=True,
            retry_after=OUTAGE_RETRY_AFTER,
        )
    return candidates


def snippet_pages(candidates: list[dict]) -> list[dict]:
    """Basic-depth passages: the discovery snippet stands in for page content.

    No crawl happens, which is the whole point of the tier - the caller is picking
    URLs to follow up, not reading pages. Some discovery results carry no snippet
    at all, and an empty passage gives the reranker nothing to score, so the title
    is the fallback.
    """
    return [
        {
            "url": item["url"],
            "title": item["title"],
            "snippet": item["snippet"],
            "content": item["snippet"] or item["title"],
        }
        for item in candidates
    ]


async def crawl_all(client: httpx.AsyncClient, candidates: list[dict]):
    """One bulk Crawl4AI call for the search path; per-URL fallback if bulk fails."""
    flags = await asyncio.gather(*(is_allowed_public_url(item["url"]) for item in candidates))
    allowed = [item for item, ok in zip(candidates, flags) if ok]
    if not allowed:
        return []

    pages = await _bulk_crawl(client, allowed)
    if not pages:  # transport failure OR zero parsed content: per-URL is the known-good shape
        if pages is None:
            logger.warning("bulk crawl failed, falling back to per-URL")
        else:
            logger.warning("bulk crawl returned no usable pages, falling back to per-URL")
        semaphore = asyncio.Semaphore(CRAWL_CONCURRENCY)
        results = await asyncio.gather(
            *(crawl_with_limit(semaphore, client, item) for item in allowed),
            return_exceptions=True,
        )
        pages = [page for page in results if isinstance(page, dict) and page.get("content")]
    return pages


# Search-path crawler config. Without a markdown generator the response carries no
# `fit_markdown` at all and every caller gets the raw page, nav and cookie banners
# included. The typed objects are the server's own, not caller input, so they are
# allowed to be more than the scalar shapes the /crawl allowlist accepts.
SEARCH_CRAWLER_CONFIG = {
    "excluded_tags": ["nav", "footer", "header", "aside", "form"],
    "remove_overlay_elements": True,
    "markdown_generator": {
        "type": "DefaultMarkdownGenerator",
        "params": {
            "content_filter": {"type": "PruningContentFilter", "params": {}},
            "options": {"ignore_links": True},
        },
    },
}


async def _bulk_crawl(client: httpx.AsyncClient, candidates: list[dict]):
    """Bulk crawl, mapped back to candidates. [] = no content; None = transport failure."""
    try:
        payload = await call_crawl4ai(
            client, {"urls": [item["url"] for item in candidates], "crawler_config": SEARCH_CRAWLER_CONFIG}
        )
    except AppError:
        return None

    items = payload.get("results", payload) if isinstance(payload, dict) else payload
    if not isinstance(items, list):
        items = [items]

    by_url = {item["url"]: item for item in candidates}
    claimed = set(by_url)
    crawled_urls = [extract_crawled_url(item) for item in items]
    contents = [extract_crawl_content(item) for item in items]
    used = [False] * len(items)
    pages = []
    for candidate in candidates:
        idx = next((i for i, url in enumerate(crawled_urls) if not used[i] and url == candidate["url"]), None)
        if idx is None:  # redirected/normalized URL no other candidate claims
            idx = next(
                (i for i, url in enumerate(crawled_urls) if not used[i] and url and url not in claimed and contents[i]),
                None,
            )
        if idx is None or not contents[idx]:
            continue
        used[idx] = True
        url = crawled_urls[idx] or candidate["url"]
        if url != candidate["url"] and not await is_allowed_public_url(url):
            continue
        pages.append({**candidate, "url": url, "content": contents[idx]})
    return pages


async def crawl_with_limit(semaphore: asyncio.Semaphore, client: httpx.AsyncClient, result: dict):
    async with semaphore:
        return await crawl_url(client, result)


async def crawl_url(client: httpx.AsyncClient, result: dict):
    if not await is_allowed_public_url(result["url"]):
        return None

    try:
        # Same cleaned extraction as the bulk call: this is the per-URL fallback,
        # and without the config it would quietly hand back raw nav markdown.
        payload = await call_crawl4ai(
            client, {"urls": [result["url"]], "crawler_config": SEARCH_CRAWLER_CONFIG}
        )
    except AppError:
        return None

    crawled_url = extract_crawled_url(payload)
    if crawled_url and not await is_allowed_public_url(crawled_url):
        return None

    content = extract_crawl_content(payload)
    if not content:
        return None
    return {**result, "content": content}


async def crawl_direct_url(client: httpx.AsyncClient, request: CrawlRequest):
    if not await is_allowed_public_url(request.url):
        raise AppError(422, "invalid_url", "URL must be a public http(s) URL")

    payload = await call_crawl4ai(client, build_crawl_payload(request))
    crawled_url = extract_crawled_url(payload) or request.url
    if not await is_allowed_public_url(crawled_url):
        raise AppError(422, "invalid_crawled_url", "Crawled URL must be a public http(s) URL")

    content = extract_crawl_content(payload, request.content_format)
    if not content:
        raise AppError(502, "empty_crawl_result", "Crawl4AI did not return extractable content")

    return {"url": crawled_url, "content": content}


# Server-owned crawl defaults, applied to every payload in call_crawl4ai().
#
# These cannot come from the server's own config.yml: /crawl builds its browser from
# the request (`BrowserConfig.load(request["browser_config"])`) and never reads
# `crawler.browser.kwargs`, so a mounted config changes nothing for the crawl path.
# They cannot come from a caller either -- /crawl may not send browser_config at all,
# and the stealth crawler keys are absent from CRAWL_ALLOWED_KEYS.
#
# Worth having regardless of anti-bot effect: crawl4ai's own defaults send Chrome/116
# while the image renders with Chrome 153, and leave the "HeadlessChrome" token in
# the UA untouched by enable_stealth.
STEALTH_BROWSER_CONFIG = {
    "enable_stealth": True,
    "user_agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
        f"Chrome/{CRAWL4AI_CHROMIUM_VERSION} Safari/537.36"
    ),
    "viewport_width": 1920,
    "viewport_height": 1080,
}
STEALTH_CRAWLER_CONFIG = {
    "remove_overlay_elements": True,
    "wait_until": "domcontentloaded",
    "delay_before_return_html": 0.2,
    "max_retries": 1,
    # One locale/timezone for every crawl: inconsistent values are themselves a signal.
    "locale": "en-US",
    "timezone_id": "America/New_York",
    # Deliberately absent: `magic` and `override_navigator`. The server loads this
    # dict as Provenance.UNTRUSTED and rejects both with a 400 that fails the whole
    # crawl, and the server-side base_config cannot set them either (they default to
    # False, and base_config only fills values that are None or ""). They are
    # unreachable in crawl4ai 0.9.4 unless the image itself sets them.
}


def with_stealth_defaults(payload: dict[str, Any]) -> dict[str, Any]:
    """Add the server's crawl defaults; anything the caller set wins.

    Fill-only, matching crawl4ai's own `crawler.base_config` semantics, so the scalars
    /crawl is allowed to tune (wait_until, max_retries, locale, ...) keep working.
    """
    for key, defaults in (
        ("browser_config", STEALTH_BROWSER_CONFIG),
        ("crawler_config", STEALTH_CRAWLER_CONFIG),
    ):
        payload[key] = {**defaults, **payload.get(key, {})}
    return payload


async def call_crawl4ai(client: httpx.AsyncClient, payload: dict[str, Any]):
    payload = with_stealth_defaults(payload)
    try:
        response = await client.post(f"{CRAWL4AI_URL}/crawl", json=payload, timeout=90, headers=CRAWL4AI_HEADERS)
        response.raise_for_status()
        return response.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise AppError(502, "crawl4ai_error", "Crawl4AI crawl failed", {"error": str(exc)}) from exc


# Public /crawl passthrough hardening (#10). Crawl4AI's Docker server loads our
# configs as Provenance.UNTRUSTED regardless of the bearer token, but that gate
# silently *drops* unknown fields; the API edge must reject loudly instead, and
# never forward a key that can reach an LLM, proxy, browser or JS sink.
CRAWL_PASSTHROUGH_FIELDS = ("crawler_config", "crawl_options")

# Crawl4AI's CacheMode enum, which is what the typed form below must carry.
CRAWL_CACHE_MODES = ("enabled", "disabled", "read_only", "write_only", "bypass")

CRAWL_ALLOWED_KEYS = frozenset({
    # timing / waiting
    "wait_until", "wait_for", "wait_for_timeout", "page_timeout",
    "delay_before_return_html", "body_visibility_timeout", "mean_delay", "max_range",
    # content selection / cleaning
    "word_count_threshold", "css_selector", "excluded_tags", "excluded_selector",
    "target_elements", "only_text", "keep_data_attributes", "keep_attrs",
    "remove_forms", "prettiify", "locale", "timezone_id",
    # rendering / capture
    "screenshot", "screenshot_wait_for", "scan_full_page", "scroll_delay",
    "max_scroll_steps", "process_iframes", "flatten_shadow_dom",
    "remove_overlay_elements", "adjust_viewport_to_content", "pdf", "capture_mhtml",
    # cache
    "cache_mode", "bypass_cache", "disable_cache", "no_cache_read", "no_cache_write",
    # link / image filtering
    "exclude_external_links", "exclude_internal_links", "exclude_domains",
    "exclude_social_media_links", "exclude_external_images", "exclude_all_images",
    "score_links", "preserve_https_for_internal_links",
    # misc scalars
    "verbose", "log_console", "method", "max_retries",
})

CRAWL_FORBIDDEN_KEYS = frozenset({
    "llm_config", "proxy_config", "proxy_rotation_strategy", "check_robots_txt",
    "link_preview_config", "js_code", "js_code_before_wait", "c4a_script",
    "fallback_fetch_function", "user_data_dir", "cdp_url", "storage_state",
    "extra_args", "init_scripts", "session_id", "deep_crawl_strategy",
    "extraction_strategy", "scraping_strategy", "markdown_generator",
    "chunking_strategy", "table_extraction", "virtual_scroll_config",
    "geolocation", "hooks", "crawler_configs", "url", "urls",
})

_CRAWL_SCALARS = (str, bool, int, float)
CRAWL_ALLOWED_KEYS_HINT = ", ".join(sorted(CRAWL_ALLOWED_KEYS))
CRAWL_MAX_RETRIES = 2


def _crawl_violation(loc: list[str], message: str) -> dict[str, Any]:
    """One entry in the 422 `details` list, shaped like a pydantic error."""
    return {"loc": loc, "msg": message, "type": "value_error"}


def _is_plain_crawl_value(value: Any) -> bool:
    if value is None or isinstance(value, _CRAWL_SCALARS):
        return True
    return isinstance(value, list) and all(
        item is None or isinstance(item, _CRAWL_SCALARS) for item in value
    )


def _crawl_value_violation(key: str, value: Any) -> str | None:
    """Extra rules for values that are plain scalars but still dangerous."""
    if key == "wait_for" and not (isinstance(value, str) and value.startswith("css:")):
        # smart_wait() runs anything that is not a `css:` selector as JavaScript
        # (`js:` prefix, a bare `() =>`/`function`, or any other value wrapped in
        # `() => { ... }`), so `wait_for` is a JS sink like `js_code`.
        return "'wait_for' must be a CSS selector prefixed with 'css:'; JavaScript wait conditions are not accepted"
    if key == "max_retries" and not (
        isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= CRAWL_MAX_RETRIES
    ):
        return f"'max_retries' must be an integer between 0 and {CRAWL_MAX_RETRIES}"
    if key == "cache_mode" and not (isinstance(value, str) and value.lower() in CRAWL_CACHE_MODES):
        return f"'cache_mode' must be one of: {', '.join(CRAWL_CACHE_MODES)}"
    return None


def validate_crawl_passthrough(request: CrawlRequest) -> None:
    """Reject any /crawl config the public surface may not forward.

    Values must be plain scalars (or lists of scalars): a nested object can carry
    a `{"type": ...}` typed-object wrapper, so any dict at any depth is refused.
    """
    problems: list[dict[str, Any]] = []
    if request.cache_mode is not None and request.cache_mode.lower() not in CRAWL_CACHE_MODES:
        problems.append(
            _crawl_violation(["cache_mode"], f"'cache_mode' must be one of: {', '.join(CRAWL_CACHE_MODES)}")
        )
    if request.extraction_config:
        problems.append(
            _crawl_violation(
                ["extraction_config"],
                "extraction_config is deprecated and unsupported; leave it empty and use crawler_config",
            )
        )
    if request.browser_config:
        problems.append(
            _crawl_violation(
                ["browser_config"],
                "browser_config is not accepted from the public API; the server owns browser settings",
            )
        )
    for field in CRAWL_PASSTHROUGH_FIELDS:
        for key, value in getattr(request, field).items():
            loc = [field, key]
            if key in CRAWL_FORBIDDEN_KEYS or key.startswith("browser_"):
                problems.append(_crawl_violation(loc, f"'{key}' is not accepted from the public API"))
            elif key not in CRAWL_ALLOWED_KEYS:
                problems.append(
                    _crawl_violation(loc, f"'{key}' is not an allowed key; allowed keys: {CRAWL_ALLOWED_KEYS_HINT}")
                )
            elif isinstance(value, dict):
                problems.append(
                    _crawl_violation(
                        loc,
                        f"'{key}' must be a scalar or list of scalars; nested objects are not accepted (typed-object injection)",
                    )
                )
            elif not _is_plain_crawl_value(value):
                problems.append(_crawl_violation(loc, f"'{key}' must be a scalar or list of scalars"))
            elif (violation := _crawl_value_violation(key, value)) is not None:
                problems.append(_crawl_violation(loc, violation))
    if problems:
        raise AppError(422, "validation_error", "Request validation failed", problems)


def build_crawl_payload(request: CrawlRequest):
    validate_crawl_passthrough(request)

    # Crawl4AI 0.9.x reads crawler settings only from `crawler_config` and drops
    # any other top-level key, so crawl_options and cache_mode are folded in
    # here. An explicit crawler_config value still wins over both.
    crawler_config: dict[str, Any] = dict(request.crawl_options)
    if request.cache_mode is not None:
        crawler_config["cache_mode"] = request.cache_mode
    crawler_config.update(request.crawler_config)

    if "cache_mode" in crawler_config:
        # CacheMode is an Enum and the server does not coerce a bare string, so
        # "bypass" sent as a plain scalar would silently never engage the cache.
        # Only the typed form works (verified against the 0.9.4 server:
        # cache_status goes "miss" -> "hit" for the typed form, stays "miss" for
        # the scalar).
        mode = crawler_config["cache_mode"]
        crawler_config["cache_mode"] = {"type": "CacheMode", "params": mode.lower() if isinstance(mode, str) else mode}

    payload: dict[str, Any] = {"urls": [request.url]}
    if crawler_config:
        payload["crawler_config"] = crawler_config
    return payload


def api_surface(request: Request):
    host = request.headers.get("host", "").split(":", 1)[0].lower()
    if host == "webcrawl.sparkfn.io":
        return "crawl"
    return "search"


def filtered_openapi(surface: Literal["search", "crawl"]):
    schema = deepcopy(app.openapi())
    allowed_paths = {"crawl": {"/crawl", "/health"}, "search": {"/health", "/search"}}[surface]
    allowed_components = {
        "crawl": {"CrawlData", "CrawlEnvelope", "CrawlRequest", "ErrorDetail", "ErrorEnvelope", "HTTPValidationError", "HealthData", "HealthEnvelope", "ValidationError"},
        "search": {"ErrorDetail", "ErrorEnvelope", "HTTPValidationError", "HealthData", "HealthEnvelope", "SearchData", "SearchEnvelope", "SearchRequest", "SearchResult", "ValidationError"},
    }[surface]
    schema["paths"] = {path: value for path, value in schema["paths"].items() if path in allowed_paths}
    schemas = schema.get("components", {}).get("schemas", {})
    schema.get("components", {})["schemas"] = {name: value for name, value in schemas.items() if name in allowed_components}
    if surface == "crawl":
        schema["info"]["title"] = "SparkFN Web Crawl API"
        schema["info"]["description"] = (
            "Tool-facing API for extracting content from a known public URL. Use this service when an agent, CLI, or MCP tool "
            "already has a URL and needs normalized page content for summarization, citation, or downstream reasoning. Do not use this "
            "API for search or discovery; use https://websearch.sparkfn.io for query-based discovery.\n\n"
            "How to use: send POST /crawl with JSON body {\"url\": \"https://example.com\", \"content_format\": \"markdown\"} and an "
            "X-API-Key header. The URL must be public http(s). Localhost, private networks, reserved/link-local IPs, and credentialed URLs "
            "are rejected before Crawl4AI is called. Advanced Crawl4AI config fields are optional pass-through controls; leave them empty for normal use."
        )
        schema["servers"] = [{"url": "https://webcrawl.sparkfn.io", "description": "Public crawl API"}]
        schema["tags"] = [{"name": "Crawl", "description": "Extract content from one known public URL."}]
        schema["x-ai-usage"] = {
            "when_to_use": "Use when you already know the exact URL and need extracted page content.",
            "when_not_to_use": "Do not use for web search, discovery, private network probing, or arbitrary Crawl4AI administration.",
            "authentication": "Send X-API-Key with every /crawl request. /health is public.",
            "recommended_request": {"url": "https://example.com", "content_format": "markdown"},
            "recovery": {
                "401": "Add or correct X-API-Key.",
                "422": "Use a valid public http(s) URL and valid enum/body values.",
                "502": "Retry later, try a simpler URL, or reduce advanced crawl options.",
            },
        }
    else:
        schema["info"]["title"] = "SparkFN Web Search API"
        schema["info"]["description"] = (
            "Tool-facing API for web discovery plus extracted, reranked page content. Use this service when an agent, CLI, or MCP tool "
            "has a natural-language question/topic and needs current web evidence. The pipeline is: SearXNG discovers candidate URLs, "
            "internal Crawl4AI extracts content from public pages, and the reranker returns the most relevant extracted chunks.\n\n"
            "How to use: send POST /search with JSON body {\"query\": \"your concise query\", \"max_results\": 5, \"candidates\": 10} "
            "and an X-API-Key header. Start with max_results 3-5 and candidates 10. Increase candidates only when recall matters more than latency. "
            "An empty results array is a successful no-results response, not a failure."
        )
        schema["servers"] = [{"url": "https://websearch.sparkfn.io", "description": "Public search API"}]
        schema["tags"] = [
            {"name": "Search", "description": "Discover web pages and return extracted, reranked content."},
            {"name": "Diagnostics", "description": "Public service diagnostics for availability checks."},
        ]
        schema["x-ai-usage"] = {
            "when_to_use": "Use for natural-language web discovery when you do not already know the target URL.",
            "when_not_to_use": "Do not use to crawl a single known URL; use https://webcrawl.sparkfn.io/crawl instead.",
            "authentication": "Send X-API-Key with every /search request. /health is public.",
            "recommended_request": {"query": "open source web search engines", "max_results": 5, "candidates": 10},
            "parameter_guidance": {
                "query": "Keep concise but specific. Include product/project/version terms when relevant.",
                "max_results": "Use 3-5 for most agent tasks; up to 20 for broad surveys.",
                "candidates": "Use 10 by default; increase toward 20-50 only for recall-heavy tasks and expect more latency.",
            },
            "recovery": {
                "401": "Add or correct X-API-Key.",
                "422": "Fix query/body/query parameters.",
                "502": "Retry later or reduce candidates if the request is too expensive.",
            },
        }
    return schema


def extract_crawl_content(payload, preferred_format: str = "markdown"):
    if isinstance(payload, list):
        for item in payload:
            content = extract_crawl_content(item, preferred_format)
            if content:
                return content
        return ""

    if not isinstance(payload, dict):
        return ""

    format_keys = [preferred_format, "markdown", "fit_markdown", "cleaned_html", "text", "content", "html"]
    for key in dict.fromkeys(format_keys):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, dict):
            # Prefer the pruned markdown, but a fit result shorter than
            # MIN_FIT_MARKDOWN_CHARS means the filter ate the page: raw prose
            # beats a two-line "fit".
            fit = value.get("fit_markdown")
            if isinstance(fit, str) and len(fit.strip()) >= MIN_FIT_MARKDOWN_CHARS:
                return fit.strip()
            nested = value.get("raw_markdown") or fit or value.get("content")
            if isinstance(nested, str) and nested.strip():
                return nested.strip()

    for key in ("result", "results", "data"):
        content = extract_crawl_content(payload.get(key), preferred_format)
        if content:
            return content

    return ""


def extract_crawled_url(payload):
    if isinstance(payload, list):
        for item in payload:
            url = extract_crawled_url(item)
            if url:
                return url
        return None

    if not isinstance(payload, dict):
        return None

    value = payload.get("url")
    if isinstance(value, str):
        return value

    for key in ("result", "results", "data"):
        url = extract_crawled_url(payload.get(key))
        if url:
            return url

    return None


MARKDOWN_LINK_RE = re.compile(r"!?\[[^\]]*\]\([^)\s]*\)")
URL_RE = re.compile(r"https?://\S+")


def strip_boilerplate_lines(text: str) -> str:
    """Drop lines that carry no prose: markdown link/image runs and short menu stubs.

    ponytail: word-count heuristic, not a parser. It can drop a genuine two-word
    line, so headings and anything with a digit (answer-bearing numbers) are kept,
    and the eval set is what decides whether it over-prunes.
    """
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        # Links nest ("[ ![logo](img) ](page)"), so strip targets as well and ask
        # only whether any word character survives.
        if not any(character.isalnum() for character in URL_RE.sub("", MARKDOWN_LINK_RE.sub("", stripped))):
            continue  # links/images only
        if (
            len(stripped) <= 24
            and len(stripped.split()) <= 2
            and not stripped.startswith("#")
            and not any(character.isdigit() for character in stripped)
        ):
            continue  # menu stub: "Docs", "Pricing", "Skip to content"
        lines.append(stripped)
    return "\n".join(lines)


def chunk_documents(documents: list[dict]):
    """Chunk the whole cleaned page, capped per page, not just its first chunks.

    The old limit kept the first CHUNKS_PER_PAGE chunks, which on a typical page is
    the nav bar and cookie banner, so relevant text further down never reached the
    reranker at all.
    """
    chunks = []
    for document in documents:
        text = strip_boilerplate_lines(document["content"])[:MAX_PAGE_CHARS]
        if not text:
            continue

        start = 0
        while start < len(text):
            end = min(start + CHUNK_SIZE, len(text))
            if end < len(text):  # snap to a sentence/line boundary past the midpoint
                snap = max(text.rfind(". ", start, end), text.rfind("\n", start, end))
                if snap > start + CHUNK_SIZE // 2:
                    end = snap + 1
            chunk = text[start:end].strip()
            if chunk:
                chunks.append({**document, "content": chunk})
            if end <= start:
                break
            start = end if end >= len(text) else max(end - CHUNK_OVERLAP, start + 1)
    return chunks


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def lexical_scores(query: str, documents: list[str]) -> list[float]:
    """BM25-ish word-overlap score per document, normalised so the best one is 1.0.

    Only has to be a good enough prefilter for the cross-encoder stage below.
    """
    query_tokens = tokenize(query)
    if not query_tokens:
        return [0.0 for _ in documents]

    counts = [Counter(tokenize(document)) for document in documents]
    document_frequencies = Counter(token for count in counts for token in count)
    document_count = max(len(documents), 1)
    query_counts = Counter(query_tokens)

    raw_scores = []
    for count in counts:
        length_norm = math.sqrt(max(sum(count.values()), 1))
        score = 0.0
        for token, query_count in query_counts.items():
            term_frequency = count[token]
            if not term_frequency:
                continue
            idf = math.log((document_count + 1) / (document_frequencies[token] + 0.5)) + 1
            score += query_count * math.log1p(term_frequency) * idf
        raw_scores.append(score / length_norm)

    max_score = max(raw_scores, default=0.0)
    if max_score <= 0:
        return raw_scores
    return [score / max_score for score in raw_scores]


def discovery_ranks(chunks: list[dict]) -> dict[str, int]:
    """Page position in the discovery order, per URL.

    Candidates are crawled in discovery-rank order and chunk_documents() preserves
    that order, so a URL's first chunk is its discovery position.
    """
    ranks: dict[str, int] = {}
    for chunk in chunks:
        ranks.setdefault(chunk["url"], len(ranks))
    return ranks


def fused_score(cross_encoder_score: float, discovery_rank: int, weight: float) -> float:
    """Blend the cross-encoder score with a reciprocal rank of the discovery position.

    RRF_K / (RRF_K + rank) is the usual 1/(60+rank) decay rescaled to [0, 1], so both
    terms are comparable and `weight` reads directly as "how much does engine rank buy".
    """
    discovery = RERANK_RRF_K / (RERANK_RRF_K + discovery_rank)
    return (1 - weight) * cross_encoder_score + weight * discovery


# ponytail: breaker state is per process, so each uvicorn worker trips and recovers on
# its own and a restart clears it. Fine for one failing upstream; share the state (redis
# or a sidecar) only if the workers ever disagreeing actually matters.
_rerank_skip_until = 0.0


def rerank_available() -> bool:
    """False while the breaker is open; logs the single transition back to available."""
    global _rerank_skip_until
    if _rerank_skip_until and time.monotonic() >= _rerank_skip_until:
        _rerank_skip_until = 0.0
        logger.info("rerank circuit breaker closed; trying the cross-encoder again")
    return not _rerank_skip_until


def trip_rerank_breaker(reason) -> None:
    """Stop calling the cross-encoder for RERANK_COOLDOWN; logs once per opening."""
    global _rerank_skip_until
    if not _rerank_skip_until:
        logger.warning(
            "rerank circuit breaker opened (%r); skipping the cross-encoder for %.0fs",
            reason, RERANK_COOLDOWN,
        )
    _rerank_skip_until = time.monotonic() + RERANK_COOLDOWN


async def cross_encode(client: httpx.AsyncClient, query: str, documents: list[str]):
    """Cross-encoder score for every document, in input order; None if unavailable.

    Returning None (rather than zeros) is what sends the caller to the lexical-order
    fallback instead of a silently-wrong all-equal ranking.
    """
    if not rerank_available():
        return None

    batches = [documents[i : i + RERANK_BATCH] for i in range(0, len(documents), RERANK_BATCH)]
    try:
        responses = await asyncio.gather(
            *(
                client.post(
                    f"{RERANKER_URL}/rerank",
                    json={"query": query, "texts": batch},
                    timeout=RERANK_TIMEOUT,
                )
                for batch in batches
            )
        )
        scores: list[float] = []
        for response, batch in zip(responses, batches):
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, list):
                raise ValueError(f"unexpected rerank payload: {payload!r}")
            batch_scores = [0.0] * len(batch)
            for item in payload:
                index, score = item.get("index"), item.get("score")
                if isinstance(index, int) and 0 <= index < len(batch_scores) and isinstance(score, (int, float)):
                    batch_scores[index] = float(score)
            scores.extend(batch_scores)
        return scores
    except (httpx.HTTPError, ValueError, TypeError, AttributeError, asyncio.TimeoutError) as exc:
        logger.warning("cross-encoder rerank failed (%r); using lexical order", exc)
        trip_rerank_breaker(exc)
        return None


async def rerank(client: httpx.AsyncClient, query: str, chunks: list[dict], top_k: int):
    """Two-stage rerank. Returns (results, path) where path names the ranking that won."""
    if not chunks:
        return [], "empty"

    # Stage 1: cheap lexical pass decides which chunks the cross-encoder sees at all.
    lexical = lexical_scores(query, [chunk["content"] for chunk in chunks])
    by_lexical = sorted(range(len(chunks)), key=lambda index: lexical[index], reverse=True)
    head, tail = by_lexical[:RERANK_PREFILTER], by_lexical[RERANK_PREFILTER:]

    # Stage 2: cross-encoder over the prefiltered head, blended with discovery rank.
    passages = [chunks[index]["content"][:RERANK_MAX_CHARS] for index in head]
    cross_encoder = await cross_encode(client, query, passages)
    if cross_encoder is None:
        order = [(index, None) for index in by_lexical]
        path = "lexical-fallback"
    else:
        path = "cross-encoder"
        ranks = discovery_ranks(chunks)
        scored = sorted(
            (
                (index, round(fused_score(score, ranks[chunks[index]["url"]], RERANK_WEIGHT), 6))
                for index, score in zip(head, cross_encoder)
            ),
            key=lambda item: item[1],
            reverse=True,
        )
        # Chunks the prefilter dropped rank behind every scored chunk, best-lexical first.
        order = scored + [(index, None) for index in tail]

    results = []
    seen_urls = set()
    for index, score in order:
        chunk = chunks[index]
        if chunk["url"] in seen_urls:
            continue
        seen_urls.add(chunk["url"])
        results.append(
            SearchResult(
                title=chunk["title"],
                url=chunk["url"],
                snippet=chunk.get("snippet", ""),
                content=chunk["content"],
                score=score,
            )
        )
        if len(results) >= top_k:
            break

    return results, path


_dns_cache: dict[str, tuple[float, bool]] = {}
_dns_lock = asyncio.Lock()
DNS_TTL = 60.0


async def is_allowed_public_url(url: str):
    parsed = urlparse(url)
    if not _url_prefix_ok(parsed):
        return False
    hostname = (parsed.hostname or "").rstrip(".").lower()
    async with _dns_lock:
        hit = _dns_cache.get(hostname)
    now = asyncio.get_running_loop().time()
    if hit is None or hit[0] <= now:
        verdict = await asyncio.to_thread(_host_resolves_public, hostname, parsed.port)
        async with _dns_lock:
            _dns_cache[hostname] = (asyncio.get_running_loop().time() + DNS_TTL, verdict)
        return verdict
    return hit[1]


def _url_prefix_ok(parsed) -> bool:
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return False
    if parsed.username or parsed.password:
        return False
    hostname = parsed.hostname.rstrip(".").lower()
    return not (hostname in {"localhost", "localhost.localdomain"} or hostname.endswith(".localhost"))


def _host_resolves_public(hostname: str, port) -> bool:
    try:
        addresses = socket.getaddrinfo(hostname, port, type=socket.SOCK_STREAM)
    except socket.gaierror:
        return False
    return all(is_public_ip(ipaddress.ip_address(address[4][0])) for address in addresses)


def is_public_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address):
    return not any(
        (
            ip.is_private,
            ip.is_loopback,
            ip.is_link_local,
            ip.is_multicast,
            ip.is_reserved,
            ip.is_unspecified,
        )
    )


def search_response(query: str, results: list[SearchResult]):
    return SearchEnvelope(data=SearchData(query=query, results=results))


def error_response(
    status_code: int,
    code: str,
    message: str,
    details: Any | None = None,
    retryable: bool | None = None,
    retry_after: float | None = None,
    hint: str | None = None,
):
    # `retryable`, `retry_after` and `hint` are top-level envelope fields, not details
    # buried in `details`: the canonical contract (sparkfn/pc-tools CLAUDE.md, "Error
    # envelope") puts them beside `code`, because together they decide retry-vs-fix and
    # how long to wait. When a handler does not state `retryable`, derive it from the
    # status class so the field is never a lie: a 5xx or a 429 is the upstream being
    # transient, everything else needs the caller (or an operator) to change something.
    if retryable is None:
        retryable = status_code >= 500 or status_code == 429
    return JSONResponse(
        status_code=status_code,
        content=jsonable_encoder(
            ErrorEnvelope(
                error=ErrorDetail(
                    code=code,
                    message=message,
                    details=details,
                    retryable=retryable,
                    retry_after=retry_after,
                    hint=hint,
                )
            )
        ),
    )
