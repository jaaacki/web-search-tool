import asyncio
import ipaddress
import logging
import os
import secrets
import socket
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
from pydantic import BaseModel, Field

SEARXNG_URL = os.getenv("SEARXNG_URL", "http://searxng:8080").rstrip("/")
CRAWL4AI_URL = os.getenv("CRAWL4AI_URL", "http://crawl4ai:11235").rstrip("/")
# crawl4ai >=0.9 refuses to bind non-loopback without an API token; send it as a bearer header.
CRAWL4AI_API_TOKEN = os.getenv("CRAWL4AI_API_TOKEN", "")
CRAWL4AI_HEADERS = {"Authorization": "Bearer " + CRAWL4AI_API_TOKEN} if CRAWL4AI_API_TOKEN else {}
RERANKER_URL = os.getenv("RERANKER_URL", "http://reranker:7997").rstrip("/")
SEARCH_CANDIDATES = int(os.getenv("SEARCH_CANDIDATES", "10"))
MAX_RESULTS = int(os.getenv("MAX_RESULTS", "5"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1800"))
CHUNKS_PER_PAGE = int(os.getenv("CHUNKS_PER_PAGE", "3"))
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
# the allowlist lives here where every search routes through.
SEARXNG_ENGINES = os.getenv("SEARXNG_ENGINES", "bing,mojeek,marginalia")
DISCOVERY_TIMEOUT = float(os.getenv("DISCOVERY_TIMEOUT", "20"))
CRAWL_TIMEOUT = float(os.getenv("CRAWL_TIMEOUT", "45"))
RERANK_TIMEOUT = float(os.getenv("RERANK_TIMEOUT", "8"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "300"))

logger = logging.getLogger("websearch")
if not logger.handlers:  # uvicorn only configures its own loggers; without this our INFO is swallowed
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(_handler)
logger.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())

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


class ErrorEnvelope(BaseModel):
    ok: Literal[False] = Field(default=False, description="Always false for failed requests.")
    error: ErrorDetail = Field(description="Error object with stable code, readable message, and optional details.")


class HealthData(BaseModel):
    services: dict[str, str] = Field(
        description="Internal service base URLs used by the stack. This is diagnostic metadata, not public crawl/search targets.",
        examples=[{"searxng": "http://searxng:8080", "crawl4ai": "http://crawl4ai:11235", "reranker": "http://reranker:7997"}],
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
        description="Optional Crawl4AI cache mode value passed through as `cache_mode`. Leave null unless you know the Crawl4AI cache semantics you need.",
        examples=["BYPASS"],
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
            "Crawl4AI crawler/run options. Each key must be in the server allowlist *and* each value a plain scalar or list of scalars; "
            "anything else is rejected with 422 `validation_error` naming the offending key. Nested objects (including `{\"type\": ...}` "
            "typed-object wrappers) and LLM/proxy/browser/JS keys such as `llm_config`, `proxy_config`, `check_robots_txt`, "
            "`link_preview_config`, `js_code` and `user_data_dir` are never forwarded. `wait_for` must be a CSS selector prefixed with "
            "`css:`, and `max_retries` is capped at 2."
        ),
        examples=[{"wait_until": "networkidle", "css_selector": "main"}],
    )
    extraction_config: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Extraction options, validated against the same scalar allowlist as `crawler_config`. Crawl4AI 0.9.4's `/crawl` takes extraction "
            "settings inside `crawler_config`, so this field is validated and forwarded but not honoured upstream; prefer `crawler_config`."
        ),
        examples=[{}],
    )
    crawl_options: dict[str, CrawlOptionValue] = Field(
        default_factory=dict,
        description=(
            "Top-level Crawl4AI options, validated against the same scalar allowlist as `crawler_config`. `url` and `urls` are rejected "
            "so callers cannot bypass URL validation. Prefer the named fields above."
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
    500: {"model": ErrorEnvelope, "description": "Unexpected server error. Retry later or contact the service owner."},
}


class AppError(Exception):
    def __init__(self, status_code: int, code: str, message: str, details: Any | None = None):
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details = details


async def require_api_key(x_api_key: str | None = Security(api_key_header)):
    if not WEBSEARCH_API_KEY:
        raise AppError(500, "api_key_not_configured", "WEBSEARCH_API_KEY is not configured")
    if not x_api_key or not secrets.compare_digest(x_api_key, WEBSEARCH_API_KEY):
        raise AppError(401, "unauthorized", "A valid X-API-Key header is required")


@app.exception_handler(AppError)
async def app_error_handler(_: Request, exc: AppError):
    return error_response(exc.status_code, exc.code, exc.message, exc.details)


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


@app.get(
    "/health",
    response_model=HealthEnvelope,
    responses=HEALTH_ERROR_RESPONSES,
    summary="Check service health",
    description=(
        "Public diagnostic endpoint for the search and crawl API hosts. It confirms the API process is running and reports the internal "
        "service URLs configured for SearXNG, Crawl4AI, and the reranker. Do not use this endpoint for search or crawl work."
    ),
    operation_id="check_health",
    tags=["Diagnostics"],
)
def health():
    return HealthEnvelope(
        data=HealthData(
            services={
                "searxng": SEARXNG_URL,
                "crawl4ai": CRAWL4AI_URL,
                "reranker": RERANKER_URL,
            }
        )
    )


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
        return search_response(request.query, [])

    try:
        pages = await asyncio.wait_for(crawl_all(client, candidates), timeout=CRAWL_TIMEOUT)
    except asyncio.TimeoutError:
        logger.warning("crawl timed out after %.0fs", CRAWL_TIMEOUT)
        pages = []
    if not pages:
        return search_response(request.query, [])

    chunks = chunk_documents(pages)
    ranked = await rerank(client, request.query, chunks, request.max_results)

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
):
    return await search(SearchRequest(query=q, max_results=max_results, candidates=candidates))


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


async def search_searxng(client: httpx.AsyncClient, query: str, limit: int):
    try:
        response = await client.get(
            f"{SEARXNG_URL}/search",
            params={"q": query, "format": "json", "engines": SEARXNG_ENGINES},
            timeout=10,
        )
        response.raise_for_status()
        payload = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise AppError(502, "searxng_error", "SearXNG search failed", {"error": str(exc)}) from exc

    return await shape_candidates(payload.get("results", []), limit)


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


async def _bulk_crawl(client: httpx.AsyncClient, candidates: list[dict]):
    """Bulk crawl, mapped back to candidates. [] = no content; None = transport failure."""
    try:
        payload = await call_crawl4ai(client, {"urls": [item["url"] for item in candidates]})
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
        payload = await call_crawl4ai(client, {"urls": [result["url"]]})
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
CRAWL_PASSTHROUGH_FIELDS = ("crawler_config", "crawl_options", "extraction_config")

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
    return None


def validate_crawl_passthrough(request: CrawlRequest) -> None:
    """Reject any /crawl config the public surface may not forward.

    Values must be plain scalars (or lists of scalars): a nested object can carry
    a `{"type": ...}` typed-object wrapper, so any dict at any depth is refused.
    """
    problems: list[dict[str, Any]] = []
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
    payload: dict[str, Any] = {"urls": [request.url]}
    for key, value in request.crawl_options.items():
        if key not in {"url", "urls"}:
            payload[key] = value
    if request.cache_mode is not None:
        payload["cache_mode"] = request.cache_mode
    if request.browser_config:
        payload["browser_config"] = request.browser_config
    if request.crawler_config:
        payload["crawler_config"] = request.crawler_config
    if request.extraction_config:
        payload["extraction_config"] = request.extraction_config
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
            nested = value.get("fit_markdown") or value.get("raw_markdown") or value.get("content")
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


def chunk_documents(documents: list[dict]):
    chunks = []
    for document in documents:
        text = "\n".join(line.strip() for line in document["content"].splitlines() if line.strip())
        if not text:
            continue

        start = 0
        page_chunks = 0
        while start < len(text) and page_chunks < CHUNKS_PER_PAGE:
            end = min(start + CHUNK_SIZE, len(text))
            if end < len(text):  # snap to a sentence/line boundary past the midpoint
                snap = max(text.rfind(". ", start, end), text.rfind("\n", start, end))
                if snap > start + CHUNK_SIZE // 2:
                    end = snap + 1
            chunk = text[start:end].strip()
            if chunk:
                chunks.append({**document, "content": chunk})
                page_chunks += 1
            if end <= start:
                break
            start = end if end >= len(text) else max(end - CHUNK_OVERLAP, start + 1)
    return chunks


async def rerank(client: httpx.AsyncClient, query: str, chunks: list[dict], top_k: int):
    if not chunks:
        return []
    # Over-fetch: URL-dedupe below keeps one chunk per page, so top_k chunks
    # can collapse to fewer than top_k results without this.
    fetch_k = min(len(chunks), max(top_k * 3, top_k + 5))
    try:
        response = await client.post(
            f"{RERANKER_URL}/rerank",
            json={
                "query": query,
                "documents": [chunk["content"] for chunk in chunks],
                "top_k": fetch_k,
            },
            timeout=RERANK_TIMEOUT,
        )
        response.raise_for_status()
        ranked = response.json().get("results", [])
    except (httpx.HTTPError, ValueError, TypeError, AttributeError, asyncio.TimeoutError):
        ranked = [{"index": index, "score": None} for index in range(len(chunks))]

    results = []
    seen_urls = set()
    for item in ranked:
        if not isinstance(item, dict):
            continue
        index = item.get("index")
        if not isinstance(index, int) or index < 0 or index >= len(chunks):
            continue
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
                score=item.get("score") if isinstance(item.get("score"), (int, float)) else None,
            )
        )
        if len(results) >= top_k:
            break

    return results


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


def error_response(status_code: int, code: str, message: str, details: Any | None = None):
    return JSONResponse(
        status_code=status_code,
        content=jsonable_encoder(ErrorEnvelope(error=ErrorDetail(code=code, message=message, details=details))),
    )
