import asyncio
import hashlib
import ipaddress
import json
import logging
import math
import os
import re
import secrets
import socket
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
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
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from firecrawl_adapter import (
    FORMAT_RESPONSE_KEYS,
    inline_scrape_result as firecrawl_inline_result,
    is_pdf_target,
    missing_inline_formats,
    normalize_formats,
    scrape_payload as firecrawl_scrape_payload,
    scrape_target_url as firecrawl_scrape_target_url,
    scrape_result as firecrawl_scrape_result,
    search_payload as firecrawl_search_payload,
    search_results as firecrawl_search_results,
    same_url_identity,
    supports_markdown_crawl,
)
from firecrawl_controls import FirecrawlUnavailable, firecrawl_request, firecrawl_url_allowed
from observability import ContextFormatter, JsonFormatter, RequestIdFilter, request_id_var

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
# Chrome version the Crawl4AI image bundles (Chrome for Testing 156.0.8078.4 in 0.9.5).
# The UA we send has to match the engine that renders the page, so bump this with the image.
CRAWL4AI_CHROMIUM_VERSION = os.getenv("CRAWL4AI_CHROMIUM_VERSION", "156.0.8078.4")
WEBSEARCH_API_KEY = os.getenv("WEBSEARCH_API_KEY", "")
# Firecrawl inline scrape can run browser actions, send caller headers, or reuse a
# persistent profile. Keep those off unless the operator opts in explicitly.
ALLOW_FIRECRAWL_ACTIONS = os.getenv("ALLOW_FIRECRAWL_ACTIONS", "").lower() in {"1", "true", "yes"}
ALLOW_FIRECRAWL_HEADERS = os.getenv("ALLOW_FIRECRAWL_HEADERS", "").lower() in {"1", "true", "yes"}
ALLOW_FIRECRAWL_PROFILE = os.getenv("ALLOW_FIRECRAWL_PROFILE", "").lower() in {"1", "true", "yes"}
LOG_QUERIES = os.getenv("LOG_QUERIES", "").lower() in {"1", "true", "yes"}
FIRECRAWL_FORMAT_TYPES = frozenset(FORMAT_RESPONSE_KEYS)
# Format objects that Firecrawl requires a companion field for.
FIRECRAWL_FORMAT_REQUIREMENTS = {
    "attributes": "selectors",
    "question": "question",
    "highlights": "query",
}
FIRECRAWL_SCRAPE_OPTION_KEYS = frozenset(
    {
        "formats",
        "onlyMainContent",
        "onlyCleanContent",
        "includeTags",
        "excludeTags",
        "maxAge",
        "minAge",
        "headers",
        "waitFor",
        "mobile",
        "skipTlsVerification",
        "timeout",
        "parsers",
        "actions",
        "location",
        "removeBase64Images",
        "blockAds",
        "proxy",
        "storeInCache",
        "lockdown",
        "checkPromptInjection",
        "redactPII",
        "profile",
        "auditMetadata",
    }
)
FIRECRAWL_ACTION_TYPES = frozenset(
    {
        "wait",
        "screenshot",
        "click",
        "write",
        "press",
        "scroll",
        "scrape",
        "executeJavascript",
        "pdf",
    }
)


def validate_scrape_options(options: dict[str, Any]) -> None:
    unknown = set(options) - FIRECRAWL_SCRAPE_OPTION_KEYS
    if unknown:
        raise ValueError(f"unsupported scrape_options keys: {sorted(unknown)}")
    formats = options.get("formats")
    if formats is not None:
        if not isinstance(formats, list) or not formats:
            raise ValueError("scrape_options.formats must be a non-empty array")
        for item in formats:
            fmt = item if isinstance(item, str) else item.get("type") if isinstance(item, dict) else None
            if fmt not in FIRECRAWL_FORMAT_TYPES:
                raise ValueError(f"unsupported scrape format: {fmt!r}")
            if fmt in ("json", "changeTracking", "attributes", "question", "highlights") and not isinstance(
                item, dict
            ):
                raise ValueError(f"{fmt} requires a format object")
            required = FIRECRAWL_FORMAT_REQUIREMENTS.get(fmt)
            if required:
                value = item.get(required)
                if required == "selectors":
                    if not isinstance(value, list) or not value:
                        raise ValueError(f"{fmt} requires a non-empty selectors array")
                    for selector in value:
                        if (
                            not isinstance(selector, dict)
                            or not isinstance(selector.get("selector"), str)
                            or not selector["selector"].strip()
                            or not isinstance(selector.get("attribute"), str)
                            or not selector["attribute"].strip()
                        ):
                            raise ValueError(
                                "attributes.selectors entries require selector and attribute strings"
                            )
                elif not isinstance(value, str) or not value.strip():
                    raise ValueError(f"{fmt} requires a non-empty {required} string")
            if fmt == "json":
                schema = item.get("schema")
                prompt = item.get("prompt")
                if not isinstance(schema, dict) and not (
                    isinstance(prompt, str) and prompt.strip()
                ):
                    raise ValueError("json requires a schema object or a prompt string")
            if fmt == "changeTracking":
                modes = item.get("modes")
                if (
                    not isinstance(modes, list)
                    or not modes
                    or any(mode not in ("git-diff", "json") for mode in modes)
                ):
                    raise ValueError("changeTracking requires modes from git-diff or json")
            if fmt == "screenshot" and isinstance(item, dict):
                quality = item.get("quality")
                if quality is not None and (
                    not isinstance(quality, int)
                    or isinstance(quality, bool)
                    or not 1 <= quality <= 100
                ):
                    raise ValueError("screenshot quality must be an integer from 1 to 100")
                full_page = item.get("fullPage")
                if full_page is not None and not isinstance(full_page, bool):
                    raise ValueError("screenshot fullPage must be boolean")
    actions = options.get("actions")
    if actions is not None:
        if not isinstance(actions, list) or not actions:
            raise ValueError("actions must be a non-empty array")
        if any(
            not isinstance(action, dict) or action.get("type") not in FIRECRAWL_ACTION_TYPES
            for action in actions
        ):
            raise ValueError("actions entries require a supported type")
    headers = options.get("headers")
    if headers is not None:
        if not isinstance(headers, dict) or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in headers.items()
        ):
            raise ValueError("headers must be an object of string values")
    profile = options.get("profile")
    if profile is not None:
        if (
            not isinstance(profile, dict)
            or not isinstance(profile.get("name"), str)
            or not profile["name"].strip()
        ):
            raise ValueError("profile requires a non-empty name")
    parsers = options.get("parsers")
    if parsers is not None:
        if not isinstance(parsers, list):
            raise ValueError("parsers must be an array")
        for parser in parsers:
            parser_type = (
                parser
                if isinstance(parser, str)
                else parser.get("type")
                if isinstance(parser, dict)
                else None
            )
            if parser_type not in ("pdf", "image"):
                raise ValueError("parsers entries must be pdf or image")
    proxy = options.get("proxy")
    if proxy is not None and proxy not in ("basic", "enhanced", "auto"):
        raise ValueError("proxy must be basic, enhanced, or auto")
    wait_for = options.get("waitFor")
    if wait_for is not None and (
        not isinstance(wait_for, int)
        or isinstance(wait_for, bool)
        or not 0 <= wait_for <= 60_000
    ):
        raise ValueError("waitFor must be an integer from 0 to 60000")
    timeout = options.get("timeout")
    if timeout is not None and (
        not isinstance(timeout, int)
        or isinstance(timeout, bool)
        or not 1_000 <= timeout <= 300_000
    ):
        raise ValueError("timeout must be an integer from 1000 to 300000")
# Keep inline extras bounded in bytes; rawBase64 above this is omitted rather than corrupted.
MAX_EXTRAS_BYTES = int(os.getenv("SEARCH_MAX_EXTRAS_BYTES", "200000"))
# Keep the whole inline-extras payload below the MCP client's 2 MB response cap.
MAX_TOTAL_EXTRAS_BYTES = int(os.getenv("SEARCH_MAX_TOTAL_EXTRAS_BYTES", "1000000"))
MAX_METADATA_BYTES = int(os.getenv("SEARCH_MAX_METADATA_BYTES", "20000"))
# Optional Brave Search API keys (comma-separated) for primary discovery.
# Empty = SearXNG-only discovery, exactly today's behavior.
BRAVE_API_KEYS = tuple(key.strip() for key in os.getenv("BRAVE_API_KEYS", "").split(",") if key.strip())
BRAVE_SEARCH_URL = os.getenv("BRAVE_SEARCH_URL", "https://api.search.brave.com/res/v1/web/search").rstrip("/")
BRAVE_TIMEOUT = float(os.getenv("BRAVE_TIMEOUT", "8"))
BRAVE_PAGE_SIZE = 20
BRAVE_PAGE_LIMIT = 10
# Fences SearXNG discovery per request: settings.yml merges with defaults, so
# the allowlist lives here where every search routes through. Every engine named
# here must be active in the pinned image - an engine the image marks `inactive`
# is absent from /config and can never be re-enabled by config, which silently
# collapses discovery onto the remaining engines.
SEARXNG_ENGINES = os.getenv("SEARXNG_ENGINES", "bing,brave,duckduckgo,google cse")
# Bounded separately from SEARCH_TIMEOUT: health is polled, so it must not hold a
# caller for as long as a real search would. Both are per-request ceilings, not
# aggregate budgets.
HEALTH_TIMEOUT = float(os.getenv("HEALTH_TIMEOUT", "5"))
SEARCH_TIMEOUT = float(os.getenv("SEARCH_TIMEOUT", "10"))
# Reuse per-provider outcomes and internal probes so frequent /health polls do not
# trigger additional network checks or paid searches.
HEALTH_CACHE_TTL = float(os.getenv("HEALTH_CACHE_TTL", "60"))
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
    _handler.setFormatter(
        JsonFormatter()
        if os.getenv("LOG_FORMAT", "text").lower() == "json"
        else ContextFormatter("%(levelname)s: %(message)s")
    )
    _handler.addFilter(RequestIdFilter())
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
    version="0.2.0",
    lifespan=lifespan,
    description=(
        "Authenticated APIs for AI agents and MCP/CLI tools. "
        "Use the host-specific OpenAPI documents: websearch.sparkfn.io exposes search, "
        "and webcrawl.sparkfn.io exposes crawl."
    ),
    docs_url=None,
    openapi_url=None,
)
_REQUEST_ID_RE = re.compile(r"[A-Za-z0-9._-]{1,64}")


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    incoming = request.headers.get("x-request-id", "")
    request_id = incoming if _REQUEST_ID_RE.fullmatch(incoming) else uuid.uuid4().hex
    token = request_id_var.set(request_id)
    try:
        response = await call_next(request)
    finally:
        request_id_var.reset(token)
    response.headers["X-Request-ID"] = request_id
    return response


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
    request_id: str | None = Field(
        default=None,
        description="Correlation id also returned in the X-Request-ID response header and written to the usage ledger.",
        examples=["9f2c1d4e5a6b7c8d"],
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
            "Last-known discovery outcomes plus bounded internal service probes. A stale: prefix reports age; "
            "last-known success remains available until a newer outcome supersedes it. Public health polling never "
            "issues paid searches or refreshes provider timestamps; recorded total discovery failure stays unavailable."
        ),
        examples=[{"firecrawl": "unknown", "brave": "unknown", "searxng": "unknown", "searxng_service": "ok", "crawl4ai": "ok", "reranker": "ok"}],
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
        le=100,
        description=(
            "Number of search candidates to fetch before crawling and reranking. Higher values may improve recall but increase latency "
            "because more pages may be crawled; very high values can exhaust the discovery validation budget on slow DNS."
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
    sources: list[Literal["web", "news", "images"]] = Field(
        default_factory=lambda: ["web"],
        min_length=1,
        max_length=3,
        description=(
            "Firecrawl result sources. `web` is the default; `news` and `images` add their result arrays. `images` results are "
            "returned as image metadata (`image_url`, `position`) without page extraction. Non-web sources are Firecrawl-only: "
            "strict mode returns `unsupported_feature` or `fallback_feature_unavailable` when a fallback would have to drop them, "
            "and `allow_degraded` returns the web subset with a warning."
        ),
        examples=[["web"], ["web", "news"]],
    )
    categories: list[Literal["developer", "gov", "research", "pdf"]] = Field(
        default_factory=list,
        max_length=4,
        description=(
            "Firecrawl category filters. `developer` targets repositories/docs, `gov` government material, `research` academic "
            "sites, `pdf` PDF results. `gov` cannot be combined with other categories or filters. Firecrawl-only."
        ),
        examples=[["developer"], ["pdf"]],
    )
    include_domains: list[str] = Field(
        default_factory=list,
        max_length=20,
        description=(
            "Hostnames to include. Cannot be combined with `exclude_domains`. Firecrawl uses `includeDomains`; Brave and "
            "SearXNG use `site:` operators. When set, the automatic LinkedIn/X/shortener exclusions are not applied."
        ),
        examples=[["docs.python.org"]],
    )
    exclude_domains: list[str] = Field(
        default_factory=list,
        max_length=20,
        description="Hostnames to exclude. Cannot be combined with `include_domains`. Brave and SearXNG use `-site:` operators.",
        examples=[["pinterest.com"]],
    )
    time_range: Literal["day", "week", "month", "year"] | None = Field(
        default=None,
        description="Recency filter. Firecrawl maps it to `tbs`; Brave to `freshness`; SearXNG to `time_range`.",
        examples=["week"],
    )
    tbs: str | None = Field(
        default=None,
        max_length=100,
        description="Raw Firecrawl `tbs` filter (for example `qdr:w` or a custom `cdr:` range). Cannot be combined with `time_range`.",
        examples=["qdr:w"],
    )
    country: str | None = Field(
        default=None,
        pattern=r"^[A-Za-z]{2}$",
        description="ISO alpha-2 country for Firecrawl/Brave geo targeting. SearXNG has no country parameter, so strict mode reports the loss.",
        examples=["US"],
    )
    location: str | None = Field(
        default=None,
        max_length=200,
        description="Firecrawl location string such as `city,state,country`. Firecrawl-only; strict mode reports the loss on fallback.",
        examples=["San Francisco,California,United States"],
    )
    safe: bool | None = Field(
        default=None,
        description="Filter explicit results. Supported by Firecrawl, Brave, and SearXNG.",
        examples=[True],
    )
    highlights: bool | None = Field(
        default=None,
        description="Ask Firecrawl for query-relevant highlights instead of provider snippets. Firecrawl-only.",
        examples=[True],
    )
    ignore_invalid_urls: bool = Field(
        default=False,
        description="Ask Firecrawl to drop URLs unsuitable for its other endpoints. Firecrawl-only; local validation still checks every returned URL.",
        examples=[True],
    )
    scrape_options: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Inline Firecrawl scrape options for every search result: `formats` plus the normal scrape options. Markdown is used as the "
            "reranked content; every other returned field is exposed as `extras`. `storeInCache` defaults to false; set true to opt into "
            "Firecrawl caching. `rawBase64` above SEARCH_MAX_EXTRAS_BYTES is omitted. Enterprise-only controls (`zeroDataRetention`, "
            "`threatProtection`, `alexandria`, `domainTools`) are rejected."
        ),
        examples=[{"formats": ["markdown", "links"], "onlyMainContent": True}],
    )
    max_content_chars: int | None = Field(
        default=None,
        ge=200,
        le=200_000,
        description=(
            "Truncate each returned `content` after ranking. Snippets are unchanged; inline extras are bounded separately by "
            "SEARCH_MAX_EXTRAS_BYTES."
        ),
        examples=[4000],
    )
    allow_degraded: bool = Field(
        default=False,
        description=(
            "Strict mode by default. When a fallback provider cannot honor a requested feature, the request fails with "
            "`fallback_feature_unavailable`. Set this true to accept the subset, with `data.degraded` naming every unsupported feature."
        ),
        examples=[True],
    )

    @field_validator("include_domains", "exclude_domains")
    @classmethod
    def _validate_domains(cls, values: list[str]) -> list[str]:
        cleaned = []
        for value in values:
            if not isinstance(value, str) or not value.strip():
                raise ValueError("domains must be non-empty hostnames")
            host = value.strip().lower().rstrip(".")
            if any(token in host for token in ("/", ":", " ", "@")):
                raise ValueError("domains must be hostnames without scheme, port, path, or credentials")
            cleaned.append(host)
        return cleaned

    @model_validator(mode="after")
    def _validate_combinations(self) -> "SearchRequest":
        if self.include_domains and self.exclude_domains:
            raise ValueError("include_domains and exclude_domains cannot be combined")
        if self.time_range and self.tbs:
            raise ValueError("time_range and tbs cannot be combined")
        if len(self.sources) != len(set(self.sources)):
            raise ValueError("sources must be unique")
        if "gov" in self.categories and (
            len(self.categories) > 1
            or self.sources != ["web"]
            or self.include_domains
            or self.exclude_domains
            or self.time_range
            or self.tbs
            or self.country
            or self.location
            or self.safe is not None
            or self.highlights is not None
        ):
            raise ValueError("gov cannot be combined with other categories or filters")
        if self.scrape_options is not None:
            if not isinstance(self.scrape_options, dict):
                raise ValueError("scrape_options must be an object")
            if self.depth != "advanced":
                raise ValueError("scrape_options requires depth=advanced")
            unavailable = {
                "zeroDataRetention",
                "threatProtection",
                "alexandria",
                "domainTools",
            } & set(self.scrape_options)
            if unavailable:
                raise ValueError(f"unavailable on the free plan: {sorted(unavailable)}")
            validate_scrape_options(self.scrape_options)
            gated = {
                "actions": ALLOW_FIRECRAWL_ACTIONS,
                "headers": ALLOW_FIRECRAWL_HEADERS,
                "profile": ALLOW_FIRECRAWL_PROFILE,
            }
            disabled = [key for key, allowed in gated.items() if key in self.scrape_options and not allowed]
            if disabled:
                raise ValueError(
                    f"disabled by operator: {', '.join(disabled)} "
                    "(set ALLOW_FIRECRAWL_ACTIONS/HEADERS/PROFILE to enable)"
                )
        return self


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
    source: Literal["web", "news", "images"] | None = Field(
        default=None,
        description="Discovery source that produced this result. Null when the fallback provider does not label it.",
        examples=["web"],
    )
    published_date: str | None = Field(
        default=None,
        description="Provider-published date for news results, when Firecrawl returns one.",
        examples=["2026-10-24T12:00:00.000Z"],
    )
    image_url: str | None = Field(
        default=None,
        description="Associated image URL for news or image results.",
        examples=["https://example.com/cover.png"],
    )
    position: int | None = Field(
        default=None,
        description="Provider result position for news or image sources.",
        examples=[1],
    )
    metadata: dict[str, Any] | None = Field(
        default=None,
        description="Inline scrape metadata returned with `scrape_options` (title, description, language, status, final URL).",
    )
    extras: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Inline Firecrawl fields returned by `scrape_options` that are not the reranked markdown: html, rawHtml, links, images, "
            "screenshot, summary, json, answer, highlights, and similar format payloads."
        ),
    )


class SearchData(BaseModel):
    query: str = Field(description="Original query string that was searched.", examples=["open source web search engines"])
    results: list[SearchResult] = Field(
        description="Reranked extracted results. Empty list is a successful no-results response, not an error.",
    )
    degraded: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Present only when `allow_degraded` accepted a provider that could not honor every requested feature. "
            "Names the provider and the unsupported features."
        ),
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Human-readable notices mirroring `degraded`, suitable for surfacing to the caller.",
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
    "firecrawl_policy_denied": {
        "summary": "The provider policy denied this target",
        "value": {
            "ok": False,
            "error": {
                "code": "firecrawl_policy_denied",
                "message": "The requested target is not available under provider policy",
                "details": {"provider": "firecrawl"},
                "retryable": False,
            },
        },
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
    "upstream_unavailable": {
        "summary": "All configured discovery providers failed",
        "value": {
            "ok": False,
            "error": {
                "code": "upstream_unavailable",
                "message": "All configured discovery providers failed",
                "details": {"providers": {"brave": "unavailable", "searxng": "searxng_error"}},
                "retryable": True,
                "retry_after": 30,
            },
        },
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
        "description": "Invalid request or permanent provider-policy denial. Do not retry unchanged values or targets.",
        "content": {"application/json": {"examples": {"validation_error": ERROR_EXAMPLES["validation_error"], "firecrawl_policy_denied": ERROR_EXAMPLES["firecrawl_policy_denied"]}}},
    },
    502: {
        "model": ErrorEnvelope,
        "description": "A search, crawl, or rerank upstream service failed. Retry later or reduce `candidates` if timeouts persist.",
        "content": {"application/json": {"examples": {"searxng_error": ERROR_EXAMPLES["searxng_error"], "crawl4ai_error": ERROR_EXAMPLES["crawl4ai_error"], "reranker_error": ERROR_EXAMPLES["reranker_error"]}}},
    },
    503: {
        "model": ErrorEnvelope,
        "description": "All configured discovery providers failed. Retry after the indicated delay.",
        "content": {"application/json": {"examples": {"upstream_unavailable": ERROR_EXAMPLES["upstream_unavailable"]}}},
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
        "description": "Invalid request, non-public URL, unsupported option, or permanent provider-policy denial.",
        "content": {"application/json": {"examples": {"validation_error": ERROR_EXAMPLES["validation_error"], "invalid_url": ERROR_EXAMPLES["invalid_url"], "firecrawl_policy_denied": ERROR_EXAMPLES["firecrawl_policy_denied"]}}},
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
        "description": "No configured discovery provider has a recent healthy outcome. Retry later.",
        "content": {"application/json": {"examples": {"upstream_unavailable": ERROR_EXAMPLES["upstream_unavailable"]}}},
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
    # Model-validator errors carry the original ValueError in ctx; encode it as text
    # instead of letting jsonable_encoder fail on an unknown type.
    details = jsonable_encoder(exc.errors(), custom_encoder={ValueError: str})
    return error_response(422, "validation_error", "Request validation failed", details)


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
        return "ok" if response.status_code < 400 else f"unavailable: http_{response.status_code}"
    except (httpx.HTTPError, ValueError) as exc:
        return f"unavailable: {type(exc).__name__}"


# Recent outcomes from real discovery requests. Health reads these instead of issuing
# searches that may be paid or rate-limited.
_PROVIDER_HEALTH: dict[str, dict[str, object]] = {
    "firecrawl": {"at": 0.0, "status": "unknown"},
    "brave": {"at": 0.0, "status": "unknown"},
    "searxng": {"at": 0.0, "status": "unknown"},
}
# Last verdict, reused for HEALTH_CACHE_TTL so public polling cannot become engine load.
# Caches failures too; an outage is not hammered by its own health check.
_HEALTH_CACHE: dict[str, object] = {"at": 0.0, "result": None}


def _record_provider_health(provider: str, status: str) -> None:
    _PROVIDER_HEALTH[provider] = {"at": time.monotonic(), "status": status}
    _HEALTH_CACHE["at"] = 0.0


def _provider_health(provider: str, configured: bool) -> str:
    if not configured:
        return "not_configured"
    result = _PROVIDER_HEALTH[provider]
    if result["at"] == 0.0:
        return "unknown"
    if time.monotonic() - result["at"] >= HEALTH_CACHE_TTL:
        return f"stale:{result['status']}"
    return str(result["status"])


def health_cache_reset() -> None:
    _HEALTH_CACHE["at"] = 0.0
    _HEALTH_CACHE["result"] = None
    for result in _PROVIDER_HEALTH.values():
        result.update(at=0.0, status="unknown")


async def _run_health_checks() -> dict[str, str]:
    """Report last-known provider outcomes and probe supporting services without searching."""
    checks = {
        "firecrawl": _provider_health("firecrawl", True),
        "brave": _provider_health("brave", bool(BRAVE_POOL)),
        "searxng": _provider_health("searxng", bool(_engine_names())),
    }
    if checks["searxng"] == "unknown" and checks["searxng"] != "not_configured":
        checks["searxng_service"] = await probe_service(SEARXNG_URL, "/healthz")
    else:
        checks["searxng_service"] = "not_probed"
    checks["crawl4ai"] = await probe_service(CRAWL4AI_URL, "/health", CRAWL4AI_HEADERS)
    checks["reranker"] = await probe_service(RERANKER_URL, "/health")

    # A live SearXNG process is the cold-start fallback until a real search records
    # whether its engine pool answered. Known provider failures always take precedence.
    # Freshness labels do not turn a recorded success into an observed failure.
    searxng_available = checks["searxng"] in ("ok", "stale:ok") or (
        checks["searxng"] == "unknown" and checks["searxng_service"] == "ok"
    )
    if (
        checks["firecrawl"] not in ("ok", "stale:ok")
        and checks["brave"] not in ("ok", "stale:ok")
        and not searxng_available
    ):
        raise AppError(
            503,
            "upstream_unavailable",
            "No configured discovery provider is available",
            {"checks": checks},
            retryable=True,
            retry_after=OUTAGE_RETRY_AFTER,
        )
    return checks


@app.get(
    "/health",
    response_model=HealthEnvelope,
    responses=HEALTH_ERROR_RESPONSES,
    summary="Check service health",
    description=(
        "Public diagnostic endpoint for the search and crawl API hosts. It reports configured internal service URLs and "
        "last-known discovery-provider outcomes; stale: marks age rather than a new failure. Last-known success remains "
        "available until a newer outcome supersedes it. SearXNG readiness is used only before an outcome is observed. "
        "Health polling never performs a paid search or refreshes provider timestamps. Verdicts are cached for "
        "HEALTH_CACHE_TTL. This is passive availability metadata, not a live-provider guarantee or search/crawl endpoint."
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
        "Use this endpoint when the caller has a question or topic and needs current web evidence. Firecrawl is the primary discovery "
        "provider when configured, with Brave and SearXNG as fallbacks/top-up. Accepts the free Firecrawl v2 search surface: sources, "
        "categories, domain/time/geo filters, safe/highlights, and inline `scrape_options`. Advanced search uses Firecrawl Markdown for "
        "eligible Firecrawl results and Crawl4AI for remaining pages, then chunks and reranks the content. Basic search returns discovery "
        "snippets without scraping. Requested features that a fallback cannot honor fail loudly by default (`unsupported_feature` / "
        "`fallback_feature_unavailable`); `allow_degraded` accepts the subset with `data.degraded` and warnings. Prefer this POST endpoint "
        "for MCP/CLI tools because the request body is explicit and easy to validate. If `data.results` is empty, the request succeeded "
        "but no usable pages were found."
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
    query_id = hashlib.sha256(request.query.encode("utf-8")).hexdigest()[:12]
    if LOG_QUERIES:
        logger.debug("search query=%r", request.query)
    degraded: dict[str, Any] = {}
    candidates = await discover_candidates(
        client, request.query, request.candidates, request, degraded
    )
    if not candidates:
        logger.info("search q_id=%s results=0 elapsed=%.0fms stage=discovery", query_id, _ms(started))
        return search_response(request.query, [], degraded or None)

    discovered = time.perf_counter()
    if request.depth == "basic":
        pages = snippet_pages(candidates)
    else:
        crawl_started = asyncio.get_running_loop().time()
        crawl_deadline = crawl_started + CRAWL_TIMEOUT
        firecrawl_deadline = crawl_started + CRAWL_TIMEOUT / 2
        extraction_loss: set[str] = set()
        firecrawl_pages, crawl_candidates, policy_denied = await scrape_firecrawl_candidates(
            client, candidates, firecrawl_deadline, request, extraction_loss
        )
        if extraction_loss and not request.allow_degraded:
            raise AppError(
                503,
                "fallback_feature_unavailable",
                "Firecrawl could not return the requested scrape features and the fallback cannot honor them",
                {
                    "provider": "firecrawl",
                    "fallback_provider": "crawl4ai",
                    "unsupported": sorted(extraction_loss),
                },
                retryable=True,
                retry_after=OUTAGE_RETRY_AFTER,
                hint=(
                    "Retry when Firecrawl recovers, or set allow_degraded=true "
                    "to accept the formats the fallback can return."
                ),
            )
        if extraction_loss:
            degraded.setdefault("providers", []).append(
                {"provider": "crawl4ai", "unsupported": sorted(extraction_loss)}
            )
        pages = firecrawl_pages
        if crawl_candidates:
            try:
                pages.extend(await crawl_all(client, crawl_candidates, deadline=crawl_deadline))
            except AppError as exc:
                if not pages:
                    raise
                logger.warning("crawl fallback failed after Firecrawl content (%s)", exc.code)
        if not pages and policy_denied and not crawl_candidates:
            raise AppError(
                422,
                "firecrawl_policy_denied",
                "The requested targets are not available under provider policy",
                {"provider": "firecrawl"},
                retryable=False,
            )
    if not pages:
        logger.info(
            "search q_id=%s depth=%s candidates=%d results=0 discovery=%.0fms crawl=%.0fms elapsed=%.0fms",
            query_id, request.depth, len(candidates), _ms(started, discovered), _ms(discovered), _ms(started),
        )
        return search_response(request.query, [], degraded or None)

    crawled = time.perf_counter()
    # Basic pages are already the passages: a snippet is a clean summary, and the
    # boilerplate filter is tuned for page markdown (it drops short lines, which is
    # most snippets), so they go to the reranker as-is.
    chunks = pages if request.depth == "basic" else chunk_documents(pages)
    ranked, rerank_path = await rerank(client, request.query, chunks, request.max_results)
    if request.max_content_chars:
        ranked = [
            result.model_copy(update={"content": truncate_content(result.content, request.max_content_chars)})
            for result in ranked
        ]
    ranked, extras_truncated = cap_result_extras(ranked)
    omitted_extras = sorted(
        {
            str(key)
            for result in ranked
            if isinstance(result.extras, dict)
            for key in (result.extras.get("omitted") or [])
        }
    )
    logger.info(
        "search q_id=%s depth=%s candidates=%d pages=%d chunks=%d results=%d discovery=%.0fms crawl=%.0fms rerank=%.0fms elapsed=%.0fms rerank_path=%s",
        query_id, request.depth, len(candidates), len(pages), len(chunks), len(ranked),
        _ms(started, discovered), _ms(discovered, crawled), _ms(crawled), _ms(started), rerank_path,
    )

    warnings = []
    if extras_truncated or omitted_extras:
        message = "inline extras were truncated to the response budget"
        if omitted_extras:
            message += f"; omitted: {', '.join(omitted_extras)}"
        warnings.append(message)
    return search_response(
        request.query,
        ranked,
        degraded or None,
        warnings or None,
    )


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
        "Use this endpoint when the caller already has a URL and needs Markdown. A plain, compatible Markdown request uses Firecrawl "
        "when the target is allowed; explicit Crawl4AI formats/cache/options and Firecrawl-unavailable cases use internal Crawl4AI. "
        "It validates public http(s) URLs and returns only normalized `url` and `content`. Do not use this endpoint for discovery; call "
        "the search API first when you do not already know the URL."
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
    def __init__(self, reason: str, cooldown: float, partial: list[dict] | None = None):
        super().__init__(reason)
        self.reason = reason
        self.cooldown = cooldown
        self.partial = partial or []


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

    async def search(
        self,
        client: httpx.AsyncClient,
        query: str,
        limit: int,
        deadline: float,
        extra_params: dict[str, Any] | None = None,
    ) -> tuple[list[dict] | None, float | None]:
        """Candidates and this request's Retry-After, or None when no key can answer."""
        retry_after: float | None = None
        tried = 0
        while tried < len(self._keys):
            if deadline <= asyncio.get_running_loop().time():
                _record_provider_health("brave", "unavailable")
                return None, retry_after
            key_idx = await self._take_key()
            if key_idx is None:
                logger.warning("brave pool exhausted (all %d keys cooling)", len(self._keys))
                _record_provider_health("brave", "unavailable")
                return None, retry_after
            tried += 1
            try:
                results = await self._search_with_key(client, key_idx, query, limit, deadline, extra_params)
            except _BraveKeyError as exc:
                if exc.cooldown > 0:
                    self._cool(key_idx, exc.cooldown)
                    logger.warning("brave key %d failed (%s), cooling %.0fs", key_idx, exc.reason, exc.cooldown)
                else:
                    logger.warning("brave key %d failed (%s)", key_idx, exc.reason)
                if exc.reason == "rate_limited":
                    retry_after = max(retry_after or 0.0, exc.cooldown)
                if exc.partial:
                    self._calls[key_idx] += 1
                    _record_provider_health("brave", "ok")
                    return exc.partial[:limit], retry_after
                continue
            self._calls[key_idx] += 1
            _record_provider_health("brave", "ok")
            logger.info("brave key %d ok (%d candidates, %d calls)", key_idx, len(results), self._calls[key_idx])
            return results[:limit], None
        _record_provider_health("brave", "unavailable")
        return None, retry_after

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
        until = asyncio.get_running_loop().time() + max(cooldown, 0.0)
        self._cooled_until[key_idx] = max(self._cooled_until[key_idx], until)

    async def _search_with_key(
        self,
        client: httpx.AsyncClient,
        key_idx: int,
        query: str,
        limit: int,
        deadline: float,
        extra_params: dict[str, Any] | None = None,
    ):
        items: list[dict] = []
        seen: set[str] = set()
        try:
            for page_index in range(BRAVE_PAGE_LIMIT):
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    if items:
                        break
                    raise _BraveKeyError("deadline_exceeded", 0.0)
                response = await asyncio.wait_for(
                    client.get(
                        BRAVE_SEARCH_URL,
                        params={
                            "q": query,
                            "count": BRAVE_PAGE_SIZE,
                            "offset": page_index,
                            "text_decorations": False,
                            **(extra_params or {}),
                        },
                        headers={"X-Subscription-Token": self._keys[key_idx], "Accept": "application/json"},
                        timeout=min(BRAVE_TIMEOUT, remaining),
                    ),
                    timeout=remaining,
                )
                if response.status_code == 429:
                    raise _BraveKeyError("rate_limited", _retry_after(response, 300.0))
                if response.status_code in (401, 403):
                    raise _BraveKeyError(f"http_{response.status_code}", 86400.0)
                if 400 <= response.status_code < 500:
                    # Invalid paging/config is not retried at later page indices.
                    if items:
                        break
                    raise _BraveKeyError(f"http_{response.status_code}", 0.0)
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, dict) or not isinstance(payload.get("web"), dict):
                    raise ValueError("invalid Brave response shape")
                web = payload["web"]
                page = web.get("results")
                if not isinstance(page, list):
                    raise ValueError("invalid Brave results")
                if not page:
                    break
                page_candidates, complete = await shape_candidates(
                    page, limit - len(items), "description", deadline=deadline, seen_urls=seen
                )
                items.extend(page_candidates)
                if not complete:
                    raise _BraveKeyError("normalization_timeout", 0.0, items)
                query_info = payload.get("query")
                if not isinstance(query_info, dict):
                    query_info = {}
                more_results = query_info.get(
                    "more_results_available",
                    payload.get("more_results_available", web.get("more_results_available", True)),
                )
                if more_results is False or len(items) >= limit:
                    break
        except _BraveKeyError as exc:
            exc.partial = items
            raise
        except asyncio.TimeoutError as exc:
            raise _BraveKeyError("request_timeout", 60.0, items) from exc
        except (httpx.HTTPError, ValueError) as exc:
            raise _BraveKeyError("request_failed", 60.0, items) from exc
        return items[:limit]


def _retry_after(response: httpx.Response, default: float) -> float:
    value = response.headers.get("Retry-After")
    if value is None:
        return default
    try:
        seconds = float(value)
        if not math.isfinite(seconds):
            raise ValueError("Retry-After is not finite")
        return max(seconds, 0.0)
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(value)
            if retry_at.tzinfo is None:
                retry_at = retry_at.replace(tzinfo=timezone.utc)
            return max((retry_at - datetime.now(timezone.utc)).total_seconds(), 0.0)
        except (TypeError, ValueError, OverflowError):
            return default


BRAVE_POOL = BraveKeyPool(BRAVE_API_KEYS)

TBS_BY_TIME_RANGE = {"day": "qdr:d", "week": "qdr:w", "month": "qdr:m", "year": "qdr:y"}
BRAVE_FRESHNESS = {"day": "pd", "week": "pw", "month": "pm", "year": "py"}
SEARCH_FEATURE_NAMES = (
    "sources",
    "categories",
    "include_domains",
    "exclude_domains",
    "time_range",
    "tbs",
    "country",
    "location",
    "safe",
    "highlights",
    "ignore_invalid_urls",
    "scrape_options",
)
# What each discovery provider can actually honor. Anything else is a feature
# loss, not a silent downgrade: strict callers get fallback_feature_unavailable.
BACKEND_SEARCH_FEATURES = {
    "firecrawl": set(SEARCH_FEATURE_NAMES),
    "brave": {"include_domains", "exclude_domains", "time_range", "country", "safe"},
    "searxng": {"include_domains", "exclude_domains", "time_range", "safe"},
}


def requested_search_features(request: SearchRequest | None) -> dict[str, Any]:
    if request is None:
        return {}
    features: dict[str, Any] = {}
    if request.sources != ["web"]:
        features["sources"] = request.sources
    if request.categories:
        features["categories"] = request.categories
    if request.include_domains:
        features["include_domains"] = request.include_domains
    if request.exclude_domains:
        features["exclude_domains"] = request.exclude_domains
    if request.time_range:
        features["time_range"] = request.time_range
    if request.tbs:
        features["tbs"] = request.tbs
    if request.country:
        features["country"] = request.country
    if request.location:
        features["location"] = request.location
    if request.safe is not None:
        features["safe"] = request.safe
    if request.highlights is not None:
        features["highlights"] = request.highlights
    if request.ignore_invalid_urls:
        features["ignore_invalid_urls"] = True
    if request.scrape_options:
        features["scrape_options"] = request.scrape_options
    return features


def unsupported_search_features(request: SearchRequest | None, provider: str) -> list[str]:
    unsupported = set(requested_search_features(request)) - BACKEND_SEARCH_FEATURES[provider]
    # SearXNG accepts day/month/year, not week.
    if provider == "searxng" and request is not None and request.time_range == "week":
        unsupported.add("time_range")
    return sorted(unsupported)


def search_filters(request: SearchRequest | None) -> dict[str, Any]:
    """Firecrawl v2 search body fields for the filters the caller actually set."""
    if request is None:
        return {}
    filters: dict[str, Any] = {}
    if request.sources != ["web"]:
        filters["sources"] = request.sources
    if request.categories:
        filters["categories"] = request.categories
    if request.include_domains:
        filters["includeDomains"] = request.include_domains
    elif request.exclude_domains:
        filters["excludeDomains"] = request.exclude_domains
    tbs = request.tbs or TBS_BY_TIME_RANGE.get(request.time_range or "")
    if tbs:
        filters["tbs"] = tbs
    if request.country:
        filters["country"] = request.country.upper()
    if request.location:
        filters["location"] = request.location
    if request.safe is not None:
        filters["safe"] = request.safe
    if request.highlights is not None:
        filters["highlights"] = request.highlights
    if request.ignore_invalid_urls:
        filters["ignoreInvalidURLs"] = True
    if request.scrape_options:
        # Match the direct /crawl privacy default; callers can opt in explicitly.
        scrape = dict(request.scrape_options)
        scrape.setdefault("storeInCache", False)
        filters["scrapeOptions"] = scrape
    return filters


def fallback_query_and_params(
    query: str, request: SearchRequest | None, provider: str
) -> tuple[str, dict[str, Any]]:
    """Translate the free filters into Brave/SearXNG terms; the rest is feature loss."""
    params: dict[str, Any] = {}
    if request is not None:
        if request.include_domains:
            query = " ".join([query, *(f"site:{domain}" for domain in request.include_domains)])
        if request.exclude_domains:
            query = " ".join([query, *(f"-site:{domain}" for domain in request.exclude_domains)])
        if request.time_range:
            if provider == "brave":
                params["freshness"] = BRAVE_FRESHNESS[request.time_range]
            else:
                # SearXNG has day/month/year; week degrades to month only when allow_degraded.
                params["time_range"] = "month" if request.time_range == "week" else request.time_range
        if request.safe is not None:
            if provider == "brave":
                params["safesearch"] = "strict" if request.safe else "off"
            else:
                params["safesearch"] = 2 if request.safe else 0
        if request.country and provider == "brave":
            params["country"] = request.country.upper()
    return query, params


async def discover_candidates(
    client: httpx.AsyncClient,
    query: str,
    limit: int,
    search_request: SearchRequest | None = None,
    degraded_out: dict[str, Any] | None = None,
):
    loop = asyncio.get_running_loop()
    started = loop.time()
    deadline = started + max(DISCOVERY_TIMEOUT, 0.0)
    # ponytail: fixed half-stage primary ceiling; adaptive shares only if measured latency needs them.
    firecrawl_deadline = started + max(DISCOVERY_TIMEOUT, 0.0) / 2
    failures: dict[str, str] = {}
    retry_afters: list[float] = []
    candidates: list[dict] = []
    answered = False

    payload = firecrawl_search_payload(query, limit, search_filters(search_request))
    if payload is None:
        failures["firecrawl"] = "query_too_long"
    else:
        remaining = firecrawl_deadline - loop.time()
        if remaining > 0:
            try:
                response = await firecrawl_request(client, "/v2/search", payload, timeout=remaining)
                if response.status_code in (400, 404, 422):
                    failures["firecrawl"] = "invalid_request"
                    _record_provider_health("firecrawl", "unavailable")
                    rows = None
                else:
                    rows = firecrawl_search_results(response)
                if rows is None and failures.get("firecrawl") != "invalid_request":
                    failures["firecrawl"] = "invalid_response"
                    _record_provider_health("firecrawl", "unavailable")
                elif rows is not None:
                    firecrawl_candidates, complete = await shape_candidates(
                        rows, limit, "description", deadline=firecrawl_deadline
                    )
                    if complete or firecrawl_candidates:
                        pdf_urls = {
                            row["url"]
                            for row in rows
                            if isinstance(row.get("category"), str)
                            and row["category"].lower() == "pdf"
                        }
                        for candidate in firecrawl_candidates:
                            candidate["_firecrawl_source"] = True
                            if candidate["url"] in pdf_urls:
                                candidate["_firecrawl_category"] = "pdf"
                        candidates = firecrawl_candidates
                        answered = True
                        _record_provider_health("firecrawl", "ok")
                    else:
                        failures["firecrawl"] = "candidate_validation_timeout"
                        _record_provider_health("firecrawl", "unavailable")
            except FirecrawlUnavailable as exc:
                failures["firecrawl"] = exc.category
                if exc.retry_after is not None:
                    retry_afters.append(exc.retry_after)
                _record_provider_health(
                    "firecrawl", "not_configured" if exc.category == "not_configured" else "unavailable"
                )
                if exc.category != "not_configured":
                    logger.warning("firecrawl discovery failed (%s)", exc.category)
            except asyncio.TimeoutError:
                failures["firecrawl"] = "timeout"
                _record_provider_health("firecrawl", "unavailable")
            except Exception as exc:
                failures["firecrawl"] = type(exc).__name__
                _record_provider_health("firecrawl", "unavailable")
                logger.warning("firecrawl discovery failed (%s)", type(exc).__name__)
        else:
            failures["firecrawl"] = "deadline_exceeded"
            _record_provider_health("firecrawl", "unavailable")

    blocked_features: dict[str, list[str]] = {}

    if len(candidates) < limit and BRAVE_POOL:
        brave_loss = unsupported_search_features(search_request, "brave")
        if brave_loss and not (search_request and search_request.allow_degraded):
            failures["brave"] = "unsupported_features"
            blocked_features["brave"] = brave_loss
        else:
            if brave_loss and degraded_out is not None:
                degraded_out.setdefault("providers", []).append(
                    {"provider": "brave", "unsupported": brave_loss}
                )
            brave_retry_after: float | None = None
            remaining = deadline - loop.time()
            if remaining > 0:
                try:
                    brave_query, brave_params = fallback_query_and_params(query, search_request, "brave")
                    brave, brave_retry_after = await BRAVE_POOL.search(
                        client, brave_query, limit - len(candidates), deadline, brave_params
                    )
                    if brave is None:
                        failures["brave"] = "unavailable"
                    else:
                        seen_urls = {item["url"] for item in candidates}
                        candidates.extend(item for item in brave if item["url"] not in seen_urls)
                        answered = True
                except asyncio.TimeoutError:
                    failures["brave"] = "timeout"
                    _record_provider_health("brave", "unavailable")
                except Exception as exc:
                    failures["brave"] = type(exc).__name__
                    _record_provider_health("brave", "unavailable")
                    logger.warning("brave discovery failed (%s); continuing with SearXNG", type(exc).__name__)
            else:
                failures["brave"] = "deadline_exceeded"
                _record_provider_health("brave", "unavailable")
            if brave_retry_after is not None:
                retry_afters.append(brave_retry_after)

    if len(candidates) < limit:
        searxng_loss = unsupported_search_features(search_request, "searxng")
        if searxng_loss and not (search_request and search_request.allow_degraded):
            failures["searxng"] = "unsupported_features"
            blocked_features["searxng"] = searxng_loss
        else:
            if searxng_loss and degraded_out is not None:
                degraded_out.setdefault("providers", []).append(
                    {"provider": "searxng", "unsupported": searxng_loss}
                )
            remaining = deadline - loop.time()
            if remaining > 0:
                try:
                    searxng_query, searxng_params = fallback_query_and_params(query, search_request, "searxng")
                    extra = await search_searxng(
                        client,
                        searxng_query,
                        limit - len(candidates),
                        timeout=remaining,
                        deadline=deadline,
                        params=searxng_params,
                    )
                    answered = True
                    seen_urls = {item["url"] for item in candidates}
                    candidates.extend(item for item in extra if item["url"] not in seen_urls)
                except asyncio.TimeoutError:
                    failures["searxng"] = "timeout"
                    _record_provider_health("searxng", "unavailable")
                except AppError as exc:
                    failures["searxng"] = exc.code
                    if exc.retry_after is not None:
                        retry_afters.append(exc.retry_after)
                    logger.warning("searxng discovery failed (%s)", exc.code)
            else:
                failures["searxng"] = "deadline_exceeded"
                _record_provider_health("searxng", "unavailable")

    if answered:
        return candidates[:limit]
    if blocked_features:
        first_provider = sorted(blocked_features)[0]
        details = {
            "providers": failures,
            "provider": first_provider,
            "unsupported": blocked_features[first_provider],
            "unsupported_by_provider": blocked_features,
        }
        if failures.get("firecrawl") in (
            "not_configured",
            "invalid_configuration",
            "unsupported_enterprise_option",
            "invalid_request",
        ):
            raise AppError(
                422,
                "unsupported_feature",
                "No configured provider can honor the requested search features",
                details,
                retryable=False,
                hint="Remove the unsupported features or configure Firecrawl and retry.",
            )
        raise AppError(
            503,
            "fallback_feature_unavailable",
            "Firecrawl is unavailable and the fallback cannot honor the requested search features",
            details,
            retryable=True,
            retry_after=max(retry_afters, default=OUTAGE_RETRY_AFTER),
            hint="Retry when Firecrawl recovers, or set allow_degraded=true to accept the reduced result set.",
        )
    if not BRAVE_POOL and not _engine_names() and failures.get("firecrawl") == "not_configured":
        raise _NoEnginesError()
    raise AppError(
        503,
        "upstream_unavailable",
        "All configured discovery providers failed",
        {"providers": failures},
        retryable=True,
        retry_after=max(retry_afters, default=OUTAGE_RETRY_AFTER),
    )


async def firecrawl_metadata_url_is_public(response: httpx.Response, deadline: float) -> bool:
    present, url = firecrawl_scrape_target_url(response)
    if not present:
        return True
    if not isinstance(url, str):
        return False
    remaining = deadline - asyncio.get_running_loop().time()
    if remaining <= 0:
        raise asyncio.TimeoutError
    return await asyncio.wait_for(is_allowed_public_url(url), timeout=remaining)


def truncate_bytes(text: str, limit: int) -> str:
    encoded = text.encode("utf-8")
    if len(encoded) <= limit:
        return text
    return encoded[: max(limit, 0)].decode("utf-8", "ignore") + "\n\n[truncated]"


def cap_metadata(metadata: Any) -> Any:
    if not isinstance(metadata, dict):
        return metadata
    return {
        key: truncate_bytes(value, MAX_METADATA_BYTES)
        if isinstance(value, str) and len(value.encode("utf-8")) > MAX_METADATA_BYTES
        else value
        for key, value in metadata.items()
    }


def cap_result_extras(
    results: list[SearchResult],
    per_field: int = MAX_EXTRAS_BYTES,
    total: int = MAX_TOTAL_EXTRAS_BYTES,
) -> tuple[list[SearchResult], bool]:
    """Bound extras in UTF-8 bytes across the whole response, not just per field."""
    budget = max(total, 0)
    truncated = False
    capped_results = []
    for result in results:
        extras = result.extras
        if not isinstance(extras, dict) or not extras:
            capped_results.append(result)
            continue
        capped: dict[str, Any] = {}
        omitted: list[str] = []
        for key, value in extras.items():
            if isinstance(value, str):
                limit = min(per_field, budget)
                if limit <= 0:
                    omitted.append(key)
                    truncated = True
                    continue
                if len(value.encode("utf-8")) > limit:
                    value = truncate_bytes(value, limit)
                    truncated = True
                capped[key] = value
                budget -= len(value.encode("utf-8"))
            else:
                encoded = json.dumps(value, ensure_ascii=False, default=str).encode("utf-8")
                if len(encoded) > budget:
                    omitted.append(key)
                    truncated = True
                    continue
                capped[key] = value
                budget -= len(encoded)
        if omitted:
            capped["omitted"] = omitted
        capped_results.append(result.model_copy(update={"extras": capped or None}))
    return capped_results, truncated


def cap_inline_extras(extras: Any) -> Any:
    if not isinstance(extras, dict):
        return extras
    capped: dict[str, Any] = {}
    omitted = []
    for key, value in extras.items():
        if isinstance(value, str) and len(value.encode("utf-8")) > MAX_EXTRAS_BYTES:
            if key == "rawBase64":
                omitted.append(key)
                continue
            capped[key] = truncate_bytes(value, MAX_EXTRAS_BYTES)
        else:
            capped[key] = value
    if omitted:
        capped["omitted"] = omitted
    return capped


async def scrape_firecrawl_candidates(
    client: httpx.AsyncClient,
    candidates: list[dict],
    deadline: float,
    search_request: SearchRequest | None = None,
    extraction_loss: set[str] | None = None,
) -> tuple[list[dict], list[dict], set[str]]:
    loop = asyncio.get_running_loop()
    pages = []
    local_candidates = []
    policy_denied: set[str] = set()
    rejected_metadata_urls: set[str] = set()
    seen_candidates: list[str] = []
    scrape_options = (search_request.scrape_options if search_request else None) or {}
    requested_formats = normalize_formats(scrape_options.get("formats"))
    # The separate Firecrawl scrape already uses these defaults; only options that
    # actually differ from them are lost when that path is used.
    fallback_defaults = {
        "storeInCache": False,
        "onlyMainContent": True,
        "maxAge": 0,
        "skipTlsVerification": False,
        "parsers": [],
    }
    requested_option_keys = {
        key
        for key in set(scrape_options) - {"formats"}
        if scrape_options.get(key) != fallback_defaults.get(key)
    }

    def mark_missing(formats: list[str]) -> None:
        if extraction_loss is not None:
            extraction_loss.update(f"format:{fmt}" for fmt in formats)

    def mark_option_loss() -> None:
        if requested_option_keys and extraction_loss is not None:
            extraction_loss.add("options")

    def defer(candidate: dict) -> None:
        local_candidates.append(candidate)
        # Crawl4AI returns Markdown; every other requested format is lost.
        mark_missing([fmt for fmt in requested_formats if fmt != "markdown"])
        mark_option_loss()

    for candidate in candidates:
        url = candidate["url"]
        if any(same_url_identity(url, seen) for seen in seen_candidates):
            continue
        seen_candidates.append(url)
        inline = candidate.get("_inline")
        if isinstance(inline, dict):
            page = firecrawl_inline_result(inline, url)
            if page is not None:
                reported = page.get("reported_url")
                if isinstance(reported, str):
                    remaining = deadline - loop.time()
                    if remaining <= 0:
                        defer(candidate)
                        continue
                    try:
                        allowed = await asyncio.wait_for(is_allowed_public_url(reported), timeout=remaining)
                    except asyncio.TimeoutError:
                        defer(candidate)
                        continue
                    if not allowed:
                        defer(candidate)
                        continue
                mark_missing(missing_inline_formats(requested_formats, inline))
                extras = cap_inline_extras(page.get("extras") or {})
                metadata = cap_metadata(page.get("metadata"))
                pages.append(
                    {
                        "url": page["url"],
                        "title": candidate["title"],
                        "snippet": candidate.get("snippet", ""),
                        "content": page["content"] or candidate.get("snippet") or candidate["title"],
                        **{
                            key: candidate[key]
                            for key in ("source", "date", "imageUrl", "position")
                            if candidate.get(key) is not None
                        },
                        **({"metadata": metadata} if isinstance(metadata, dict) else {}),
                        **({"extras": extras} if extras else {}),
                    }
                )
                continue
            mark_missing(missing_inline_formats(requested_formats, inline))
            if candidate.get("source") != "images":
                defer(candidate)
                continue
            # Image results still have their own metadata path below.

        if candidate.get("source") == "images":
            image_fields = {"images": [candidate["imageUrl"]]} if candidate.get("imageUrl") else {}
            image_extras = (
                cap_inline_extras(image_fields)
                if "images" in requested_formats and image_fields
                else {}
            )
            mark_missing(missing_inline_formats(requested_formats, image_fields))
            pages.append(
                {
                    "url": url,
                    "title": candidate["title"],
                    "snippet": candidate.get("snippet", ""),
                    "content": candidate.get("snippet") or candidate["title"],
                    **{
                        key: candidate[key]
                        for key in ("source", "date", "imageUrl", "position")
                        if candidate.get(key) is not None
                    },
                    **({"extras": image_extras} if image_extras else {}),
                }
            )
            continue

        if not candidate.get("_firecrawl_source"):
            defer(candidate)
            continue
        url = candidate["url"]
        if is_pdf_target(url, candidate.get("_firecrawl_category", "")) or not firecrawl_url_allowed(url):
            defer(candidate)
            continue
        remaining = deadline - loop.time()
        if remaining <= 0:
            defer(candidate)
            continue
        try:
            allowed = await asyncio.wait_for(is_allowed_public_url(url), timeout=remaining)
        except asyncio.TimeoutError:
            defer(candidate)
            continue
        if not allowed:
            defer(candidate)
            continue
        remaining = deadline - loop.time()
        if remaining <= 0:
            defer(candidate)
            continue
        # The separate scrape is Markdown-only and does not carry scrapeOptions.
        mark_missing([fmt for fmt in requested_formats if fmt != "markdown"])
        mark_option_loss()
        try:
            response = await firecrawl_request(
                client, "/v2/scrape", firecrawl_scrape_payload(url), timeout=remaining
            )
        except FirecrawlUnavailable as exc:
            if exc.category == "policy_denied":
                policy_denied.add(url)
                _record_provider_health("firecrawl", "ok")
                continue
            if exc.category == "provider_routed_domain":
                defer(candidate)
                continue
            _record_provider_health(
                "firecrawl", "not_configured" if exc.category == "not_configured" else "unavailable"
            )
            if exc.category != "not_configured":
                logger.warning("firecrawl scrape failed (%s)", exc.category)
            defer(candidate)
            continue
        except asyncio.TimeoutError:
            _record_provider_health("firecrawl", "unavailable")
            defer(candidate)
            continue

        if response.status_code < 500:
            _record_provider_health("firecrawl", "ok")
        try:
            reported_url_allowed = await firecrawl_metadata_url_is_public(response, deadline)
        except asyncio.TimeoutError:
            rejected_metadata_urls.add(url)
            continue
        if not reported_url_allowed:
            rejected_metadata_urls.add(url)
            continue
        page = firecrawl_scrape_result(response, url)
        if page is None:
            defer(candidate)
        else:
            pages.append(
                {
                    "url": page["url"],
                    "title": candidate["title"],
                    "snippet": candidate.get("snippet", ""),
                    "content": page["content"],
                    **{
                        key: candidate[key]
                        for key in ("source", "date", "imageUrl", "position")
                        if candidate.get(key) is not None
                    },
                }
            )

    seen_urls = [page["url"] for page in pages] + list(policy_denied | rejected_metadata_urls)
    unique_local_candidates = []
    for item in local_candidates:
        url = item["url"]
        if any(same_url_identity(url, seen) for seen in seen_urls):
            continue
        seen_urls.append(url)
        unique_local_candidates.append(item)
    return pages, unique_local_candidates, policy_denied


async def shape_candidates(
    items,
    limit: int,
    snippet_key: str = "content",
    *,
    deadline: float,
    seen_urls: set[str] | None = None,
) -> tuple[list[dict], bool]:
    loop = asyncio.get_running_loop()
    seen = seen_urls if seen_urls is not None else set()
    results = []
    for item in items:
        if loop.time() >= deadline:
            return results, False
        if not isinstance(item, dict):
            continue
        url = item.get("url")
        if not isinstance(url, str) or url in seen:
            continue
        seen.add(url)
        remaining = deadline - loop.time()
        if remaining <= 0:
            return results, False
        try:
            allowed = await asyncio.wait_for(is_allowed_public_url(url), timeout=remaining)
        except asyncio.TimeoutError:
            return results, False
        if loop.time() >= deadline:
            return results, False
        if not allowed:
            continue
        candidate = {
            "title": item.get("title") or urlparse(url).netloc or url,
            "url": url,
            "snippet": item.get(snippet_key) or "",
        }
        for key in ("source", "category", "date", "imageUrl", "imageWidth", "imageHeight", "position"):
            if item.get(key) is not None:
                candidate[key] = item[key]
        if isinstance(item.get("inline"), dict):
            candidate["_inline"] = item["inline"]
        results.append(candidate)
        if len(results) >= limit:
            return results, True
    return results, True


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


async def search_searxng(
    client: httpx.AsyncClient,
    query: str,
    limit: int,
    timeout: float = SEARCH_TIMEOUT,
    *,
    deadline: float | None = None,
    params: dict[str, Any] | None = None,
):
    loop = asyncio.get_running_loop()
    if deadline is None:
        deadline = loop.time() + max(timeout, 0.0)
    configured = _engine_names()
    # Zero configured engines must never look like an empty web. With no engine list,
    # searxng returns nothing and reports nothing dead, so the "all engines failed"
    # test below cannot fire and the caller is handed ok:true with zero results — the
    # exact silent-empty failure #986 exists to remove (#986 follow-up).
    if not configured:
        _record_provider_health("searxng", "unavailable")
        raise _NoEnginesError()

    remaining = deadline - loop.time()
    if remaining <= 0:
        _record_provider_health("searxng", "unavailable")
        raise AppError(
            503,
            "upstream_unavailable",
            "SearXNG discovery deadline expired",
            {"provider": "searxng", "stage": "request"},
            retryable=True,
            retry_after=OUTAGE_RETRY_AFTER,
        )
    request_timeout = min(timeout, remaining)
    try:
        response = await asyncio.wait_for(
            client.get(
                f"{SEARXNG_URL}/search",
                # timeout_limit makes searxng answer with whatever engines finish inside the
                # remaining discovery budget; without it one hung engine can stall the fallback
                # even after other engines answered (#36).
                params={
                    "q": query,
                    "format": "json",
                    "engines": SEARXNG_ENGINES,
                    "timeout_limit": max(request_timeout - 1, 1),
                    **(params or {}),
                },
                timeout=request_timeout,
            ),
            timeout=remaining,
        )
        response.raise_for_status()
        payload = response.json()
    except asyncio.TimeoutError as exc:
        _record_provider_health("searxng", "unavailable")
        raise AppError(
            503,
            "upstream_unavailable",
            "SearXNG discovery request timed out",
            {"provider": "searxng", "stage": "request"},
            retryable=True,
            retry_after=OUTAGE_RETRY_AFTER,
        ) from exc
    except (httpx.HTTPError, ValueError) as exc:
        _record_provider_health("searxng", "unavailable")
        raise AppError(502, "searxng_error", "SearXNG search failed", {"error": type(exc).__name__}) from exc

    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        _record_provider_health("searxng", "unavailable")
        raise AppError(502, "searxng_error", "SearXNG returned an invalid response")
    unresponsive = payload.get("unresponsive_engines") or []
    if not isinstance(unresponsive, list):
        _record_provider_health("searxng", "unavailable")
        raise AppError(502, "searxng_error", "SearXNG returned invalid engine status")
    if unresponsive:
        # A dead engine otherwise only shows up as fewer or zero results. Log names only;
        # upstream error reasons are response data and can contain sensitive text.
        names = sorted({str(entry[0] if isinstance(entry, (list, tuple)) else entry) for entry in unresponsive})
        logger.warning("searxng engines unresponsive: %s", ", ".join(names))

    raw_results = payload.get("results", [])
    # Grade raw results, before URL filtering: a healthy empty engine answer is not an outage.
    dead = {str(entry[0] if isinstance(entry, (list, tuple)) else entry).strip().lower() for entry in unresponsive}
    answered = bool(raw_results) or bool(set(configured) - dead)
    if not answered and dead:
        _record_provider_health("searxng", "unavailable")
        raise AppError(
            503,
            "upstream_unavailable",
            "No search engine responded; the search backend is unavailable",
            {"unresponsive_engines": sorted(dead)},
            retryable=True,
            retry_after=OUTAGE_RETRY_AFTER,
        )

    candidates, complete = await shape_candidates(raw_results, limit, deadline=deadline)
    if not complete and not candidates:
        _record_provider_health("searxng", "unavailable")
        raise AppError(
            503,
            "upstream_unavailable",
            "SearXNG candidate validation deadline expired",
            {"provider": "searxng", "stage": "candidate_validation"},
            retryable=True,
            retry_after=OUTAGE_RETRY_AFTER,
        )
    _record_provider_health("searxng", "ok")
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
            **{
                key: item[key]
                for key in ("source", "date", "imageUrl", "position", "metadata", "extras")
                if item.get(key) is not None
            },
        }
        for item in candidates
    ]


async def crawl_all(
    client: httpx.AsyncClient, candidates: list[dict], *, deadline: float | None = None
):
    """Bulk crawl, recover missing candidates, and keep successes inside one deadline."""
    loop = asyncio.get_running_loop()
    if deadline is None:
        deadline = loop.time() + CRAWL_TIMEOUT
    try:
        flags = await asyncio.wait_for(
            asyncio.gather(*(is_allowed_public_url(item["url"]) for item in candidates)),
            timeout=max(0, deadline - loop.time()),
        )
    except asyncio.TimeoutError as exc:
        raise AppError(
            502,
            "crawl4ai_error",
            "Crawl timed out while validating target URLs",
            retryable=True,
            retry_after=OUTAGE_RETRY_AFTER,
        ) from exc
    allowed = [item for item, ok in zip(candidates, flags) if ok]
    if not allowed:
        return []

    bulk = await _bulk_crawl(client, allowed, deadline=deadline)
    if bulk is None:
        pages, retry_candidates = [], allowed
    else:
        pages, retry_candidates = bulk

    if retry_candidates and loop.time() < deadline:
        semaphore = asyncio.Semaphore(CRAWL_CONCURRENCY)
        tasks = [
            asyncio.create_task(crawl_with_limit(semaphore, client, item))
            for item in retry_candidates
        ]
        done = set()
        try:
            done, _ = await asyncio.wait(tasks, timeout=max(0, deadline - loop.time()))
        finally:
            pending = [task for task in tasks if not task.done()]
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
        for task in tasks:
            if task not in done or task.cancelled():
                continue
            try:
                page = task.result()
            except Exception:
                continue
            if isinstance(page, dict) and page.get("content"):
                pages.append(page)

    if not pages:
        raise AppError(
            502,
            "crawl4ai_error",
            "Crawl4AI returned no usable page content",
            retryable=True,
            retry_after=OUTAGE_RETRY_AFTER,
        )
    order = {item["url"]: index for index, item in enumerate(allowed)}
    pages.sort(key=lambda page: order.get(page["_crawl_request_url"], len(order)))
    return [{key: value for key, value in page.items() if key != "_crawl_request_url"} for page in pages]


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

# Crawl4AI's `pdf` run option generates a PDF of the page; PDF *input* needs the
# typed scraping strategy. Verified against the pinned 0.9.5 image.
PDF_CRAWLER_CONFIG = {
    **SEARCH_CRAWLER_CONFIG,
    "scraping_strategy": {"type": "PDFContentScrapingStrategy", "params": {}},
}


async def _bulk_crawl_group(
    client: httpx.AsyncClient,
    candidates: list[dict],
    crawler_config: dict,
    *,
    deadline: float,
):
    loop = asyncio.get_running_loop()
    remaining = deadline - loop.time()
    if remaining <= 0:
        return None
    try:
        payload = await asyncio.wait_for(
            call_crawl4ai(
                client,
                {"urls": [item["url"] for item in candidates], "crawler_config": crawler_config},
            ),
            timeout=remaining,
        )
    except (AppError, asyncio.TimeoutError):
        return None
    if isinstance(payload, dict) and not crawl_result_is_usable(payload):
        return None

    items = payload.get("results", payload) if isinstance(payload, dict) else payload
    if not isinstance(items, list):
        items = [items]

    requested_urls = [extract_requested_url(item) for item in items]
    used = [False] * len(items)
    pages = []
    retry_candidates = []
    for candidate_index, candidate in enumerate(candidates):
        if loop.time() >= deadline:
            retry_candidates.extend(candidates[candidate_index:])
            break
        idx = next(
            (i for i, url in enumerate(requested_urls) if not used[i] and url == candidate["url"]),
            None,
        )
        if idx is None:
            retry_candidates.append(candidate)
            continue

        used[idx] = True
        crawled_url = extract_crawled_url(items[idx]) or candidate["url"]
        if crawled_url != candidate["url"]:
            remaining = deadline - loop.time()
            if remaining <= 0:
                retry_candidates.extend(candidates[candidate_index:])
                break
            try:
                allowed = await asyncio.wait_for(is_allowed_public_url(crawled_url), timeout=remaining)
            except asyncio.TimeoutError:
                retry_candidates.extend(candidates[candidate_index:])
                break
            if not allowed:
                continue
        content = extract_crawl_content(items[idx])
        if not content:
            retry_candidates.append(candidate)
            continue
        pages.append({
            **candidate,
            "url": crawled_url,
            "content": content,
            "_crawl_request_url": candidate["url"],
        })
    return pages, retry_candidates


async def _bulk_crawl(
    client: httpx.AsyncClient, candidates: list[dict], *, deadline: float | None = None
):
    """Return mapped pages and candidates needing recovery; None means provider failure."""
    loop = asyncio.get_running_loop()
    if deadline is None:
        deadline = loop.time() + CRAWL_TIMEOUT
    if deadline - loop.time() <= 0:
        return None

    pdf_urls = {
        candidate["url"]
        for candidate in candidates
        if is_pdf_target(candidate["url"], candidate.get("_firecrawl_category", ""))
    }
    groups = []
    other_candidates = [item for item in candidates if item["url"] not in pdf_urls]
    pdf_candidates = [item for item in candidates if item["url"] in pdf_urls]
    if other_candidates:
        groups.append((other_candidates, SEARCH_CRAWLER_CONFIG))
    if pdf_candidates:
        groups.append((pdf_candidates, PDF_CRAWLER_CONFIG))

    pages = []
    retry_candidates = []
    for group, crawler_config in groups:
        result = await _bulk_crawl_group(
            client, group, crawler_config, deadline=deadline
        )
        if result is None:
            retry_candidates.extend(group)
            continue
        group_pages, group_retry = result
        pages.extend(group_pages)
        retry_candidates.extend(group_retry)
    if not pages and retry_candidates:
        return None
    return pages, retry_candidates


async def crawl_with_limit(semaphore: asyncio.Semaphore, client: httpx.AsyncClient, result: dict):
    async with semaphore:
        return await crawl_url(client, result)


async def crawl_url(client: httpx.AsyncClient, result: dict):
    if not await is_allowed_public_url(result["url"]):
        return None

    crawler_config = (
        PDF_CRAWLER_CONFIG
        if is_pdf_target(result["url"], result.get("_firecrawl_category", ""))
        else SEARCH_CRAWLER_CONFIG
    )
    payload = await call_crawl4ai(
        client, {"urls": [result["url"]], "crawler_config": crawler_config}
    )
    crawled_url = extract_crawled_url(payload) or result["url"]
    if crawled_url != result["url"] and not await is_allowed_public_url(crawled_url):
        return None

    content = extract_crawl_content(payload)
    if not content:
        return None
    return {
        **result,
        "url": crawled_url,
        "content": content,
        "_crawl_request_url": result["url"],
    }


async def crawl_direct_url(client: httpx.AsyncClient, request: CrawlRequest):
    if not await is_allowed_public_url(request.url):
        raise AppError(422, "invalid_url", "URL must be a public http(s) URL")

    if (
        supports_markdown_crawl(request)
        and not is_pdf_target(request.url)
        and firecrawl_url_allowed(request.url)
    ):
        firecrawl_deadline = asyncio.get_running_loop().time() + CRAWL_TIMEOUT
        try:
            response = await firecrawl_request(
                client,
                "/v2/scrape",
                firecrawl_scrape_payload(request.url),
                timeout=max(firecrawl_deadline - asyncio.get_running_loop().time(), 0.0),
            )
        except FirecrawlUnavailable as exc:
            if exc.category == "policy_denied":
                _record_provider_health("firecrawl", "ok")
                raise AppError(
                    422,
                    "firecrawl_policy_denied",
                    "The requested target is not available under provider policy",
                    {"provider": "firecrawl"},
                    retryable=False,
                ) from exc
            if exc.category != "provider_routed_domain":
                _record_provider_health(
                    "firecrawl", "not_configured" if exc.category == "not_configured" else "unavailable"
                )
                if exc.category != "not_configured":
                    logger.warning("firecrawl scrape failed (%s)", exc.category)
        else:
            if response.status_code < 500:
                _record_provider_health("firecrawl", "ok")
            try:
                reported_url_allowed = await firecrawl_metadata_url_is_public(response, firecrawl_deadline)
            except asyncio.TimeoutError as exc:
                raise AppError(
                    503,
                    "upstream_unavailable",
                    "Firecrawl returned a URL that could not be validated before the deadline",
                    retryable=True,
                ) from exc
            if not reported_url_allowed:
                raise AppError(422, "invalid_crawled_url", "Crawled URL must be a public http(s) URL")
            result = firecrawl_scrape_result(response, request.url)
            if result is not None:
                return result

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
CRAWL_MAX_DELAY_SECONDS = 60
CRAWL_DELAY_KEYS = frozenset({"delay_before_return_html", "mean_delay", "max_range"})


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
    if key in CRAWL_DELAY_KEYS and not (
        (
            isinstance(value, int) and not isinstance(value, bool)
            and 0 <= value <= CRAWL_MAX_DELAY_SECONDS
        )
        or (
            isinstance(value, float) and math.isfinite(value)
            and 0 <= value <= CRAWL_MAX_DELAY_SECONDS
        )
    ):
        return f"'{key}' must be a finite number between 0 and {CRAWL_MAX_DELAY_SECONDS} seconds"
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
            "has a natural-language question/topic and needs current web evidence. The pipeline is: Firecrawl discovers web candidates "
            "when configured, Brave and SearXNG fill fallbacks, eligible Firecrawl results are scraped once through Firecrawl, other pages "
            "use internal Crawl4AI, and the reranker returns the most relevant extracted chunks.\n\n"
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


def crawl_result_is_usable(payload) -> bool:
    if not isinstance(payload, dict):
        return False
    if "success" in payload and payload["success"] is not True:
        return False
    if payload.get("error_message"):
        return False

    status = payload.get("status_code")
    if status is not None and (
        not isinstance(status, int) or isinstance(status, bool) or not 200 <= status < 300
    ):
        return False

    raw_headers = payload.get("response_headers")
    if raw_headers is not None and not isinstance(raw_headers, dict):
        return False
    headers = {str(key).lower(): str(value).lower() for key, value in (raw_headers or {}).items()}
    content_type = headers.get("content-type", "").split(";", 1)[0].strip()
    if content_type and not (
        content_type.startswith("text/")
        or content_type in {"application/json", "application/pdf", "application/xhtml+xml", "application/xml"}
        or content_type.endswith(("+json", "+xml"))
    ):
        return False

    metadata = payload.get("metadata")
    title = str(metadata.get("title", "")).strip().lower() if isinstance(metadata, dict) else ""
    cloudflare = "cloudflare" in headers.get("server", "") or "cf-ray" in headers
    html = payload.get("html") if isinstance(payload.get("html"), str) else ""
    if headers.get("cf-mitigated") == "challenge":
        return False
    if title in {"just a moment...", "attention required! | cloudflare"} and (
        cloudflare or "/cdn-cgi/challenge-platform/" in html.lower()
    ):
        return False
    return True


def extract_crawl_content(payload, preferred_format: str = "markdown"):
    if isinstance(payload, list):
        for item in payload:
            content = extract_crawl_content(item, preferred_format)
            if content:
                return content
        return ""

    if not isinstance(payload, dict) or not crawl_result_is_usable(payload):
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


def extract_requested_url(payload):
    if isinstance(payload, list):
        for item in payload:
            url = extract_requested_url(item)
            if url:
                return url
        return None

    if not isinstance(payload, dict):
        return None

    for key in ("requested_url", "original_url", "url"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
    for key in ("result", "results", "data"):
        url = extract_requested_url(payload.get(key))
        if url:
            return url
    return None


def extract_crawled_url(payload):
    if isinstance(payload, list):
        for item in payload:
            url = extract_crawled_url(item)
            if url:
                return url
        return None

    if not isinstance(payload, dict):
        return None

    for key in ("redirected_url", "final_url", "url"):
        value = payload.get(key)
        if isinstance(value, str) and value:
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
        raw = document["content"]
        # Image and extras-only results can carry a title/snippet as the passage;
        # the menu-stub filter would drop those short lines entirely.
        keep_short = document.get("source") == "images" or bool(document.get("extras"))
        text = (raw if keep_short else strip_boilerplate_lines(raw))[:MAX_PAGE_CHARS]
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


def query_aligned_passage(query: str, content: str) -> str:
    width = RERANK_MAX_CHARS
    if width <= 0:
        return ""
    if len(content) <= width:
        return content

    last = len(content) - width
    starts = list(range(0, last + 1, max(width // 2, 1)))
    if starts[-1] != last:
        starts.append(last)
    windows = [content[start : start + width] for start in starts]
    scores = lexical_scores(query, windows)
    return windows[max(range(len(windows)), key=scores.__getitem__)]


async def rerank(client: httpx.AsyncClient, query: str, chunks: list[dict], top_k: int):
    """Two-stage rerank. Returns (results, path) where path names the ranking that won."""
    if not chunks:
        return [], "empty"

    # Stage 1: reserve scored slots for distinct sources before filling by lexical rank.
    lexical = lexical_scores(query, [chunk["content"] for chunk in chunks])
    by_lexical = sorted(range(len(chunks)), key=lambda index: lexical[index], reverse=True)
    head = []
    seen_urls = set()
    for index in by_lexical:
        if len(head) >= RERANK_PREFILTER:
            break
        url = chunks[index]["url"]
        if url not in seen_urls:
            head.append(index)
            seen_urls.add(url)
    selected = set(head)
    for index in by_lexical:
        if len(head) >= RERANK_PREFILTER:
            break
        if index not in selected:
            head.append(index)
            selected.add(index)
    tail = [index for index in by_lexical if index not in selected]

    # Stage 2: cross-encoder over the source-diverse head, blended with discovery rank.
    passages = [query_aligned_passage(query, chunks[index]["content"]) for index in head]
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
                source=chunk.get("source"),
                published_date=chunk.get("date"),
                image_url=chunk.get("imageUrl"),
                position=chunk.get("position"),
                metadata=chunk.get("metadata"),
                extras=chunk.get("extras"),
            )
        )
        if len(results) >= top_k:
            break

    return results, path


_dns_cache: dict[str, tuple[float, bool]] = {}
_dns_lock = asyncio.Lock()
DNS_TTL = 60.0


async def is_allowed_public_url(url: str):
    try:
        parsed = urlparse(url)
        if not _url_prefix_ok(parsed):
            return False
        port = parsed.port
        hostname = (parsed.hostname or "").rstrip(".").lower()
    except (TypeError, ValueError):
        return False

    async with _dns_lock:
        hit = _dns_cache.get(hostname)
    now = asyncio.get_running_loop().time()
    if hit is None or hit[0] <= now:
        try:
            verdict = await asyncio.to_thread(_host_resolves_public, hostname, port)
        except UnicodeError:
            return False
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
    return ip.is_global and not any(
        (
            ip.is_private,
            ip.is_loopback,
            ip.is_link_local,
            ip.is_multicast,
            ip.is_reserved,
            ip.is_unspecified,
            getattr(ip, "is_site_local", False),
        )
    )


def truncate_content(content: str, limit: int | None) -> str:
    if limit is None or len(content) <= limit:
        return content
    return content[:limit].rstrip() + "\n\n[content truncated]"


def search_response(
    query: str,
    results: list[SearchResult],
    degraded: dict[str, Any] | None = None,
    extra_warnings: list[str] | None = None,
):
    warnings = list(extra_warnings or [])
    if degraded:
        for item in degraded.get("providers", []):
            unsupported = item.get("unsupported") or []
            warnings.append(
                f"{item.get('provider', 'fallback')} could not honor: {', '.join(map(str, unsupported))}"
            )
    return SearchEnvelope(
        data=SearchData(query=query, results=results, degraded=degraded, warnings=warnings)
    )


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
    correlation_id = request_id_var.get()
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
                    request_id=None if correlation_id == "-" else correlation_id,
                )
            )
        ),
    )
