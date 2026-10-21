# AI web search stack

Self-hosted stack for `websearch.sparkfn.io` and controlled headless crawling at `webcrawl.sparkfn.io`:

- `api`: OpenAPI/Swagger-compatible search and crawl API exposed through Traefik.
- `searxng`: internal URL discovery service.
- `crawl4ai`: internal page extraction service used by the API.
- `reranker`: internal cross-encoder reranking service (Hugging Face TEI). The API's lexical
  prefilter lives in `api/app.py`; this container only scores query/document pairs.

`api` is routed by the lab-001 Traefik through [`sparkfn/lab-001-cluster:traefik/dynamic/websearch.sparkfn.io.yml`](https://github.com/sparkfn/lab-001-cluster/blob/main/traefik/dynamic/websearch.sparkfn.io.yml); route changes go to that repo. `webcrawl.sparkfn.io` only routes `POST /crawl` to the API; Crawl4AI itself remains internal-only. Internal services use Docker `expose` only and are not host-published.

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

Production runs on lab-001 from a clean `main` checkout at `/home/docker/websearch.sparkfn.io`. PRCI checks pull requests targeting `main`; deployment remains manual. After a merge:

```bash
git pull --ff-only
docker compose up -d --build          # or name one service, e.g. `api`
curl -fsS https://websearch.sparkfn.io/health
```

`/health` reuses recent discovery-provider outcomes and uses SearXNG's `/healthz` only before a search outcome is known. Polling never launches a search. It returns 503 when no configured discovery provider is available; Crawl4AI and the reranker remain diagnostic checks.

Crawl4AI is pinned to the 0.9.5 multi-platform image digest in `docker-compose.yml`; its bundled Chrome for Testing is 156.0.8078.4. To roll back, restore `unclecode/crawl4ai:0.9.4` and set `CRAWL4AI_CHROMIUM_VERSION=153.0.8010.12` in `.env`. Crawl4AI 0.9.5 now verifies TLS certificates, refuses JavaScript `wait_for` conditions, and caps its delay fields at 60 seconds; the public API rejects negative, non-finite, or over-limit values for `delay_before_return_html`, `mean_delay`, and `max_range`.

## Tests

No framework; PRCI installs the pinned dependencies in Python 3.14, runs `pip check`, and executes every assert script fail-fast on pull requests targeting `main`. To test the packaged API image locally:

```bash
cd api && docker build --target test -t websearch-api-test .
```

`eval/run.py` is the search-quality harness (golden queries against a running API; see `--help`).

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

`CRAWL4AI_CHROMIUM_VERSION` in `.env` must be bumped together with the Crawl4AI image pin: it is the
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
