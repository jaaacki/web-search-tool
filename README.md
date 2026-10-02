# AI web search stack

Self-hosted stack for `websearch.sparkfn.io` and controlled headless crawling at `webcrawl.sparkfn.io`:

- `api`: OpenAPI/Swagger-compatible search and crawl API exposed through Traefik.
- `searxng`: internal URL discovery service.
- `crawl4ai`: internal page extraction service used by the API.
- `reranker`: internal cross-encoder reranking service (Hugging Face TEI). The API's lexical
  prefilter lives in `api/app.py`; this container only scores query/document pairs.

`api` is routed by Traefik through `traefik/websearch.sparkfn.io.yml`. `webcrawl.sparkfn.io` only routes `POST /crawl` to the API; Crawl4AI itself remains internal-only. Internal services use Docker `expose` only and are not host-published.

## Required files

Create `.env`:

```bash
cp .env.example .env
```

Edit `SEARXNG_SECRET` and `WEBSEARCH_API_KEY` to long random values.

Create runtime data/config directories:

```bash
mkdir -p data/searxng data/crawl4ai
cp data.crawl4ai.llm.env.example data/crawl4ai/.llm.env
```

All persistent/runtime data belongs under `./data`. The Compose file does not use named Docker volumes.

## Deploy

Install/update the Traefik dynamic config on the server:

```bash
cp traefik/websearch.sparkfn.io.yml /home/docker/traefik/dynamic/websearch.sparkfn.io.yml
```

Then start the stack:

```bash
docker compose up -d --build
```

## OpenAPI / Swagger

- Swagger UI: `https://websearch.sparkfn.io/docs` shows search endpoints; `https://webcrawl.sparkfn.io/docs` shows crawl endpoints.
- OpenAPI JSON: `https://websearch.sparkfn.io/openapi.json` shows search endpoints; `https://webcrawl.sparkfn.io/openapi.json` shows crawl endpoints.

All API responses use a consistent envelope.

Success:

```json
{
  "ok": true,
  "data": {}
}
```

Error:

```json
{
  "ok": false,
  "error": {
    "code": "validation_error",
    "message": "Request validation failed",
    "details": []
  }
}
```

## Authentication

`/health`, `/docs`, and `/openapi.json` are public on both `websearch.sparkfn.io` and `webcrawl.sparkfn.io`. `/search` and `/crawl` require:

```http
X-API-Key: <WEBSEARCH_API_KEY>
```

Invalid or missing API keys return the same error envelope format with `code: "unauthorized"`.

## API endpoints

```bash
curl 'https://websearch.sparkfn.io/health'
curl 'https://webcrawl.sparkfn.io/health'
```

```bash
curl -X POST 'https://websearch.sparkfn.io/search' \
  -H 'content-type: application/json' \
  -H 'X-API-Key: <WEBSEARCH_API_KEY>' \
  -d '{"query":"open source web search engines", "max_results": 5}'
```

Search shortcut:

```bash
curl 'https://websearch.sparkfn.io/search?q=open%20source%20web%20search%20engines' \
  -H 'X-API-Key: <WEBSEARCH_API_KEY>'
```

Headless crawl:

```bash
curl -X POST 'https://webcrawl.sparkfn.io/crawl' \
  -H 'content-type: application/json' \
  -H 'X-API-Key: <WEBSEARCH_API_KEY>' \
  -d '{"url":"https://example.com", "content_format":"markdown"}'
```

`/crawl` supports these Crawl4AI pass-through fields while keeping Crawl4AI private:

- `content_format`: `markdown`, `cleaned_html`, `text`, or `html`.
- `cache_mode`: optional Crawl4AI cache mode value.
- `crawler_config`: optional Crawl4AI crawler/run options. Only keys on the server allowlist are
  forwarded, and only as plain scalars or lists of scalars.
- `crawl_options`: top-level Crawl4AI options, merged into `crawler_config` before it is sent
  (`crawler_config` wins on conflict), under the same allowlist.
- `browser_config` is **not** accepted from the public API: the server owns browser and stealth
  settings, so any value returns 422.
- `extraction_config` is deprecated and ignored. It is accepted only when empty (`{}` or omitted);
  a non-empty value returns 422. Put extraction settings in `crawler_config`.

## Crawl stealth defaults

The API fills every crawl — search-path and `/crawl` — with server-owned browser settings:
`enable_stealth`, a user agent matching the Chrome the Crawl4AI image actually renders with,
a fixed viewport/locale/timezone, `remove_overlay_elements`, and a small post-load delay. Callers
cannot change or disable these, and the fill only adds keys a caller left unset.

`CRAWL4AI_CHROMIUM_VERSION` in `.env` must be bumped together with the Crawl4AI image tag: it is the
Chrome version in the UA string, and a UA that disagrees with the engine is itself a signal.

These defaults do not defeat a Cloudflare JS challenge or a DataDome captcha — see #15.

## Local component debugging

Internal services are not published to host ports. To debug them on the server, exec through Compose:

```bash
docker compose exec searxng wget -qO- 'http://localhost:8080/search?q=test&format=json'
docker compose exec crawl4ai wget -qO- 'http://localhost:11235/monitor/health'
docker compose exec reranker curl -fsS http://localhost:7997/health
```

## Crawl4AI config

This stack uses `data/crawl4ai/.llm.env`. Add provider keys there only if you use Crawl4AI features that require LLM credentials.
