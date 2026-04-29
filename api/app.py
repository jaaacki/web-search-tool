import asyncio
import ipaddress
import os
import secrets
import socket
from typing import Any, Literal
from urllib.parse import urlparse

import httpx
from fastapi import Depends, FastAPI, HTTPException, Query, Request, Security
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
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
    title="AI Search API",
    version="0.1.0",
    description="Searches SearXNG, extracts candidate pages through Crawl4AI, and reranks extracted content.",
)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


class ErrorDetail(BaseModel):
    code: str
    message: str
    details: Any | None = None


class ErrorEnvelope(BaseModel):
    ok: Literal[False] = False
    error: ErrorDetail


class HealthData(BaseModel):
    services: dict[str, str]


class HealthEnvelope(BaseModel):
    ok: Literal[True] = True
    data: HealthData


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, examples=["open source web search engines"])
    max_results: int = Field(default=MAX_RESULTS, ge=1, le=20)
    candidates: int = Field(default=SEARCH_CANDIDATES, ge=1, le=50)


class SearchResult(BaseModel):
    title: str
    url: str
    snippet: str = ""
    content: str
    score: float | None = Field(default=None, ge=0, le=1)


class SearchData(BaseModel):
    query: str
    results: list[SearchResult]


class SearchEnvelope(BaseModel):
    ok: Literal[True] = True
    data: SearchData


ERROR_RESPONSES = {
    401: {"model": ErrorEnvelope},
    400: {"model": ErrorEnvelope},
    422: {"model": ErrorEnvelope},
    502: {"model": ErrorEnvelope},
    500: {"model": ErrorEnvelope},
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


@app.get("/health", response_model=HealthEnvelope, responses=ERROR_RESPONSES)
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


@app.post("/search", response_model=SearchEnvelope, responses=ERROR_RESPONSES, dependencies=[Depends(require_api_key)])
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


@app.get("/search", response_model=SearchEnvelope, responses=ERROR_RESPONSES, dependencies=[Depends(require_api_key)])
async def search_get(
    q: str = Query(min_length=1),
    max_results: int = Query(default=MAX_RESULTS, ge=1, le=20),
    candidates: int = Query(default=SEARCH_CANDIDATES, ge=1, le=50),
):
    return await search(SearchRequest(query=q, max_results=max_results, candidates=candidates))


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
        response = await client.post(
            f"{CRAWL4AI_URL}/crawl",
            json={"urls": [result["url"]]},
            timeout=90,
        )
        response.raise_for_status()
        payload = response.json()
    except (httpx.HTTPError, ValueError):
        return None

    crawled_url = extract_crawled_url(payload)
    if crawled_url and not await is_allowed_public_url(crawled_url):
        return None

    content = extract_crawl_content(payload)
    if not content:
        return None
    return {**result, "content": content}


def extract_crawl_content(payload):
    if isinstance(payload, list):
        for item in payload:
            content = extract_crawl_content(item)
            if content:
                return content
        return ""

    if not isinstance(payload, dict):
        return ""

    for key in ("markdown", "cleaned_html", "text", "content", "html"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, dict):
            nested = value.get("raw_markdown") or value.get("fit_markdown") or value.get("content")
            if isinstance(nested, str) and nested.strip():
                return nested.strip()

    for key in ("result", "results", "data"):
        content = extract_crawl_content(payload.get(key))
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
