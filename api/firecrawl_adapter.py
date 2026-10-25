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
    "rawBase64",
    "links",
    "images",
    "screenshot",
    "summary",
    "json",
    "changeTracking",
    "branding",
    "product",
    "menu",
    "audio",
    "video",
    "answer",
    "highlights",
    "attributes",
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


def _normalize_result(item: Any, source: str) -> dict[str, Any] | None:
    if not isinstance(item, dict) or not isinstance(item.get("url"), str):
        return None
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
    for key in ("category", "date"):
        if isinstance(item.get(key), str):
            result[key] = item[key]
    if isinstance(item.get("imageUrl"), str):
        result["imageUrl"] = item["imageUrl"]
    position = item.get("position")
    if isinstance(position, int) and not isinstance(position, bool):
        result["position"] = position
    inline = {key: item[key] for key in INLINE_FIELDS if item.get(key) is not None}
    if inline:
        result["inline"] = inline
    return result


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

    buckets: dict[str, list[dict[str, Any]]] = {source: [] for source in SEARCH_SOURCES}
    for source in SEARCH_SOURCES:
        rows = data.get(source)
        if not isinstance(rows, list):
            continue
        for item in rows:
            result = _normalize_result(item, source)
            if result is not None:
                buckets[source].append(result)

    # Interleave sources so a full web bucket cannot starve news/images when
    # shape_candidates later truncates to the requested candidate count.
    results: list[dict[str, Any]] = []
    while any(buckets.values()):
        for source in SEARCH_SOURCES:
            if buckets[source]:
                results.append(buckets[source].pop(0))
    return results


FORMAT_RESPONSE_KEYS = {
    "markdown": "markdown",
    "summary": "summary",
    "html": "html",
    "rawHtml": "rawHtml",
    "rawBase64": "rawBase64",
    "links": "links",
    "images": "images",
    "screenshot": "screenshot",
    "json": "json",
    "changeTracking": "changeTracking",
    "branding": "branding",
    "product": "product",
    "menu": "menu",
    "audio": "audio",
    "video": "video",
    "question": "answer",
    "highlights": "highlights",
    "attributes": "attributes",
}


def normalize_formats(formats: Any) -> list[str]:
    if not isinstance(formats, list):
        return []
    normalized = []
    for item in formats:
        if isinstance(item, str):
            normalized.append(item)
        elif isinstance(item, dict) and isinstance(item.get("type"), str):
            normalized.append(item["type"])
    return normalized


def missing_inline_formats(formats: list[str], item: dict[str, Any]) -> list[str]:
    """Requested formats the inline result did not actually return."""
    missing = []
    for fmt in formats:
        key = FORMAT_RESPONSE_KEYS.get(fmt)
        value = item.get(key) if key else None
        if key == "markdown":
            if not isinstance(value, str) or not value.strip():
                missing.append(fmt)
        elif value is None:
            missing.append(fmt)
    return missing


def inline_scrape_result(item: dict[str, Any], requested_url: str) -> dict[str, Any] | None:
    """Inline search content/extras, without a second request.

    Accepts extras-only results (for example `formats: ["links"]`); the caller
    supplies snippet/title as the passage when Markdown is absent.
    """
    if not isinstance(item, dict):
        return None
    metadata = item.get("metadata")
    reported_url = None
    if isinstance(metadata, dict):
        status = metadata.get("statusCode")
        if isinstance(status, int) and not isinstance(status, bool) and not (200 <= status < 300 or status == 304):
            return None
        if metadata.get("error"):
            return None
        content_type = metadata.get("contentType")
        if isinstance(content_type, str) and "pdf" in content_type.lower():
            return None
        source_url = metadata.get("sourceURL")
        if isinstance(source_url, str) and not same_url_identity(source_url, requested_url):
            return None
        reported = metadata.get("url")
        if isinstance(reported, str):
            reported_url = reported

    markdown = item.get("markdown")
    content = markdown if isinstance(markdown, str) and markdown.strip() else ""
    extras = {
        key: item[key]
        for key in INLINE_FIELDS
        if key not in ("markdown", "metadata") and item.get(key) is not None
    }
    if not content and not extras:
        return None
    result: dict[str, Any] = {"url": requested_url, "content": content}
    if extras:
        result["extras"] = extras
    if isinstance(metadata, dict):
        result["metadata"] = metadata
    if reported_url:
        result["reported_url"] = reported_url
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
