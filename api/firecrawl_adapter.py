"""Firecrawl v2 payload and response normalization for #46."""

from typing import Any
from urllib.parse import urlsplit


def _url_identity(value: str) -> tuple[str, str, int, str, str] | None:
    try:
        parsed = urlsplit(value)
        scheme = parsed.scheme.lower()
        if scheme not in ("http", "https") or not parsed.hostname:
            return None
        if parsed.username is not None or parsed.password is not None:
            return None
        port = parsed.port
        if port is None:
            port = 443 if scheme == "https" else 80
        return scheme, parsed.hostname.lower(), port, parsed.path or "/", parsed.query
    except ValueError:
        return None


def same_url_identity(first: str, second: str) -> bool:
    identity = _url_identity(first)
    return identity is not None and identity == _url_identity(second)


SEARCH_SOURCES = ("web", "news", "images")
INLINE_FIELDS = (
    "markdown",
    "html",
    "rawHtml",
    "links",
    "images",
    "screenshot",
    "summary",
    "json",
    "answer",
    "highlights",
    "metadata",
)


def search_payload(
    query: str, limit: int, filters: dict[str, Any] | None = None
) -> dict[str, Any] | None:
    if len(query) > 500 or not 1 <= limit <= 100:
        return None
    payload: dict[str, Any] = {"query": query, "limit": limit, "sources": ["web"]}
    if isinstance(filters, dict):
        for key, value in filters.items():
            if value is not None:
                payload[key] = value
    return payload


def search_results(response: Any) -> list[dict[str, Any]] | None:
    if response.status_code != 200:
        return None
    try:
        payload = response.json()
    except (TypeError, ValueError):
        return None
    if not isinstance(payload, dict) or payload.get("success") is not True:
        return None
    data = payload.get("data")
    if not isinstance(data, dict):
        return None
    if not any(isinstance(data.get(source), list) for source in SEARCH_SOURCES):
        return None

    results = []
    for source in SEARCH_SOURCES:
        rows = data.get(source)
        if not isinstance(rows, list):
            continue
        for item in rows:
            if not isinstance(item, dict) or not isinstance(item.get("url"), str):
                continue
            result = {
                "url": item["url"],
                "title": item.get("title") if isinstance(item.get("title"), str) else "",
                "description": (
                    item.get("description")
                    if isinstance(item.get("description"), str)
                    else item.get("snippet")
                    if isinstance(item.get("snippet"), str)
                    else ""
                ),
                "source": source,
            }
            for key in ("category", "date", "imageUrl", "imageWidth", "imageHeight", "position"):
                if item.get(key) is not None:
                    result[key] = item[key]
            inline = {key: item[key] for key in INLINE_FIELDS if item.get(key) is not None}
            if inline:
                result["inline"] = inline
            results.append(result)
    return results


def inline_scrape_result(item: dict[str, Any], requested_url: str) -> dict[str, Any] | None:
    """Page content returned by search `scrapeOptions`, without a second request."""
    markdown = item.get("markdown")
    if not isinstance(markdown, str) or not markdown.strip():
        return None
    metadata = item.get("metadata")
    if isinstance(metadata, dict):
        status = metadata.get("statusCode")
        if isinstance(status, int) and not isinstance(status, bool) and not (200 <= status < 300 or status == 304):
            return None
        if metadata.get("error"):
            return None
    result: dict[str, Any] = {"url": requested_url, "content": markdown}
    extras = {key: item[key] for key in INLINE_FIELDS if key != "markdown" and item.get(key) is not None}
    if extras:
        result["extras"] = extras
    return result


def supports_markdown_crawl(request: Any) -> bool:
    return (
        request.content_format == "markdown"
        and request.cache_mode is None
        and not request.browser_config
        and not request.crawler_config
        and not request.extraction_config
        and not request.crawl_options
    )


def is_pdf_target(url: str, category: str = "") -> bool:
    return category.lower() == "pdf" or urlsplit(url).path.lower().endswith(".pdf")


def scrape_payload(url: str) -> dict[str, Any]:
    return {
        "url": url,
        "formats": ["markdown"],
        "onlyMainContent": True,
        "maxAge": 0,
        "skipTlsVerification": False,
        "parsers": [],
    }


def scrape_target_url(response: Any) -> tuple[bool, Any]:
    try:
        payload = response.json()
    except (TypeError, ValueError):
        return False, None
    data = payload.get("data") if isinstance(payload, dict) else None
    metadata = data.get("metadata") if isinstance(data, dict) else None
    if not isinstance(metadata, dict) or "url" not in metadata:
        return False, None
    return True, metadata["url"]


def scrape_result(response: Any, requested_url: str) -> dict[str, str] | None:
    if not 200 <= response.status_code < 300:
        return None
    try:
        payload = response.json()
    except (TypeError, ValueError):
        return None
    if not isinstance(payload, dict) or payload.get("success") is not True:
        return None
    data = payload.get("data")
    if not isinstance(data, dict):
        return None
    metadata = data.get("metadata")
    markdown = data.get("markdown")
    target_status = metadata.get("statusCode") if isinstance(metadata, dict) else None
    if (
        not isinstance(metadata, dict)
        or not isinstance(target_status, int)
        or isinstance(target_status, bool)
        or not (200 <= target_status < 300 or target_status == 304)
        or not isinstance(metadata.get("sourceURL"), str)
        or not same_url_identity(metadata["sourceURL"], requested_url)
        or metadata.get("error")
        or (isinstance(metadata.get("contentType"), str) and "pdf" in metadata["contentType"].lower())
        or data.get("rawBase64")
        or not isinstance(markdown, str)
        or not markdown.strip()
    ):
        return None
    # Firecrawl's metadata.url does not establish final-browser-URL semantics; keep the requested identity.
    return {"url": requested_url, "content": markdown}
