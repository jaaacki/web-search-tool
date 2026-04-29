import asyncio
import ipaddress
import os
import secrets
import socket
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
RERANKER_URL = os.getenv("RERANKER_URL", "http://reranker:7997").rstrip("/")
SEARCH_CANDIDATES = int(os.getenv("SEARCH_CANDIDATES", "10"))
MAX_RESULTS = int(os.getenv("MAX_RESULTS", "5"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1800"))
CHUNKS_PER_PAGE = int(os.getenv("CHUNKS_PER_PAGE", "3"))
CRAWL_CONCURRENCY = int(os.getenv("CRAWL_CONCURRENCY", "3"))
WEBSEARCH_API_KEY = os.getenv("WEBSEARCH_API_KEY", "")

app = FastAPI(
    title="SparkFN Web Search and Crawl API",
    version="0.1.0",
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
        description="Advanced Crawl4AI browser configuration object passed through unchanged. Leave empty for normal headless crawling.",
        examples=[{"headless": True}],
    )
    crawler_config: dict[str, Any] = Field(
        default_factory=dict,
        description="Advanced Crawl4AI crawler/run configuration object passed through unchanged. Leave empty unless a specific Crawl4AI option is required.",
        examples=[{"wait_until": "networkidle"}],
    )
    extraction_config: dict[str, Any] = Field(
        default_factory=dict,
        description="Advanced Crawl4AI extraction strategy/config object passed through unchanged. Leave empty for general page text extraction.",
        examples=[{}],
    )
    crawl_options: dict[str, CrawlOptionValue] = Field(
        default_factory=dict,
        description=(
            "Optional top-level Crawl4AI options passed through unchanged, except `url` and `urls` are ignored so callers cannot bypass URL validation. "
            "Prefer the named fields above when available."
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
        "Public diagnostic endpoint for the search API host. It confirms the API process is running and reports the internal "
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
    async with httpx.AsyncClient(timeout=60, follow_redirects=False) as client:
        candidates = await search_searxng(client, request.query, request.candidates)
        if not candidates:
            return search_response(request.query, [])

        semaphore = asyncio.Semaphore(CRAWL_CONCURRENCY)
        pages = await asyncio.gather(
            *(crawl_with_limit(semaphore, client, result) for result in candidates),
            return_exceptions=True,
        )

        documents = [page for page in pages if isinstance(page, dict) and page.get("content")]
        if not documents:
            return search_response(request.query, [])

        chunks = chunk_documents(documents)
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
    async with httpx.AsyncClient(timeout=90, follow_redirects=False) as client:
        result = await crawl_direct_url(client, request)

    return CrawlEnvelope(data=CrawlData(url=result["url"], content=result["content"]))


async def search_searxng(client: httpx.AsyncClient, query: str, limit: int):
    try:
        response = await client.get(
            f"{SEARXNG_URL}/search",
            params={"q": query, "format": "json"},
        )
        response.raise_for_status()
        payload = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise AppError(502, "searxng_error", "SearXNG search failed", {"error": str(exc)}) from exc

    seen = set()
    results = []
    for item in payload.get("results", []):
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
                "snippet": item.get("content") or "",
            }
        )
        if len(results) >= limit:
            break
    return results


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


async def call_crawl4ai(client: httpx.AsyncClient, payload: dict[str, Any]):
    try:
        response = await client.post(f"{CRAWL4AI_URL}/crawl", json=payload, timeout=90)
        response.raise_for_status()
        return response.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise AppError(502, "crawl4ai_error", "Crawl4AI crawl failed", {"error": str(exc)}) from exc


def build_crawl_payload(request: CrawlRequest):
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
    allowed_paths = {"crawl": {"/crawl"}, "search": {"/health", "/search"}}[surface]
    allowed_components = {
        "crawl": {"CrawlData", "CrawlEnvelope", "CrawlRequest", "ErrorDetail", "ErrorEnvelope", "HTTPValidationError", "ValidationError"},
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
            "authentication": "Send X-API-Key with every /crawl request.",
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

    format_keys = [preferred_format, "markdown", "cleaned_html", "text", "content", "html"]
    for key in dict.fromkeys(format_keys):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, dict):
            nested = value.get("raw_markdown") or value.get("fit_markdown") or value.get("content")
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
            chunk = text[start : start + CHUNK_SIZE].strip()
            if chunk:
                chunks.append({**document, "content": chunk})
                page_chunks += 1
            start += CHUNK_SIZE
    return chunks


async def rerank(client: httpx.AsyncClient, query: str, chunks: list[dict], top_k: int):
    try:
        response = await client.post(
            f"{RERANKER_URL}/rerank",
            json={
                "query": query,
                "documents": [chunk["content"] for chunk in chunks],
                "top_k": top_k,
            },
            timeout=30,
        )
        response.raise_for_status()
        ranked = response.json().get("results", [])
    except (httpx.HTTPError, ValueError, TypeError, AttributeError):
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


async def is_allowed_public_url(url: str):
    return await asyncio.to_thread(is_public_http_url, url)


def is_public_http_url(url: str):
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return False
    if parsed.username or parsed.password:
        return False

    hostname = parsed.hostname.rstrip(".").lower()
    if hostname in {"localhost", "localhost.localdomain"} or hostname.endswith(".localhost"):
        return False

    try:
        addresses = socket.getaddrinfo(hostname, parsed.port, type=socket.SOCK_STREAM)
    except socket.gaierror:
        return False

    for address in addresses:
        ip = ipaddress.ip_address(address[4][0])
        if not is_public_ip(ip):
            return False
    return True


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
