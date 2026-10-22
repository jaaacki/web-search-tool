# AI web search stack

Self-hosted stack for `websearch.sparkfn.io` and controlled headless crawling at `webcrawl.sparkfn.io`:

- `api`: OpenAPI/Swagger-compatible search and crawl API exposed through Traefik.
- `firecrawl`: optional managed web discovery and Markdown extraction when an authorized budget is configured.
- `searxng`: internal discovery fallback and top-up service.
- `crawl4ai`: internal page extraction fallback and service for explicit Crawl4AI options.
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

Persistent service data belongs under `./data`. The Compose file does not use named Docker volumes; the API's temporary controls use a bounded 16 MiB `/tmp` tmpfs.

## Deploy

Production runs on lab-001 from a clean `main` checkout at `/home/docker/websearch.sparkfn.io`. PRCI on `prci.sparkfn.io` checks pull requests targeting `main`; CI on `ci.sparkfn.io` runs the same tests on pushes to `main`. Both are test-only; deployment remains manual and requires the owner's go-ahead. After a merge:

```bash
git pull --ff-only
docker compose up -d --build          # or name one service, e.g. `api`
curl -fsS https://websearch.sparkfn.io/health
```

The API waits only for local service startup, not their healthchecks. Unhealthy SearXNG/Crawl4AI must not prevent usable Firecrawl or Brave/basic search; an unavailable reranker falls back to lexical ranking. Local extraction and explicit Crawl4AI options still need Crawl4AI, and discovery fallback/top-up is reduced without SearXNG.

Docker checks API process readiness through local `/openapi.json`, so traffic can reach cloud providers before their availability is known. This is not a provider-health verdict. Public `/health` reuses last-known discovery-provider outcomes and uses SearXNG's `/healthz` only before a search outcome is known. A `stale:` prefix reports age, not a new failure: last-known success stays available until a newer outcome supersedes it, while recorded failures stay unavailable. This is passive availability metadata, not a live-provider guarantee. Polling never launches a search or renews provider timestamps. It returns 503 when no configured discovery provider has last-known availability; Crawl4AI and the reranker remain diagnostic checks. A cold API with unhealthy SearXNG stays 503 until an actual successful cloud/Brave search records availability; startup readiness does not fabricate health.

Crawl4AI is pinned to the 0.9.5 multi-platform image digest in `docker-compose.yml`; its bundled Chrome for Testing is 156.0.8078.4. To roll back, restore `unclecode/crawl4ai:0.9.4` and set `CRAWL4AI_CHROMIUM_VERSION=153.0.8010.12` in `.env`. Crawl4AI 0.9.5 now verifies TLS certificates, refuses JavaScript `wait_for` conditions, and caps its delay fields at 60 seconds; the public API rejects negative, non-finite, or over-limit values for `delay_before_return_html`, `mean_delay`, and `max_range`.

## Tests

No framework; PRCI on pull requests targeting `main` and CI on pushes to `main` install the pinned dependencies in Python 3.15, run `pip check`, and execute every assert script fail-fast. To test the packaged API image locally:

```bash
cd api && docker build --target test -t websearch-api-test .
```

The offline startup/restart regression uses Docker and Compose with private temporary env/config files, unhealthy local stubs and mocked cloud providers (no published ports or provider traffic):

```bash
docker build --target runtime -t websearch-api-runtime ./api
python3 test_startup_safety.py websearch-api-runtime
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

## Firecrawl controls

Firecrawl is disabled unless `FIRECRAWL_BUDGETS_JSON` is set in private `.env`. Group keys
from one Firecrawl team together; only the first key in each group is active, and sibling keys are
never rotated around a per-key cap. Separate groups only for independently funded and authorized
balances. An invalid-auth 401 can fail over to the next declared group; the bad primary is skipped
for 15 minutes before one caller-requested operation may retry that same key, without a usage probe.
A team-exhausted 402 can fail over only after the non-billable team-usage check confirms zero remaining
credits. Ambiguous/key-cap 402s, policy 403s and rate 429s do not rotate keys or balances; 429 cooldowns
are shared across keys in a group. No pay-as-you-go or billing setting is
enabled automatically, and no universal per-request credit cap is claimed; configure vendor key
spend limits separately. After a 15-minute ambiguous-402 cooldown and cached usage refresh, one
caller-requested operation may retry only the same primary key; success clears the guard and another
402 restarts it. There are no background or dummy probes. The provider exposes no per-key reset date;
a verified daily/weekly cap reset can be retried by that same-key operation. Container stop/restart
or recreation clears the local `/tmp` guard, not the provider's spending limits.

Firecrawl search is discovery-only; page extraction uses a separate policy-checked scrape. LinkedIn,
X/Twitter and known shortener domains are excluded from Firecrawl and may use the local Crawl4AI
path instead. An explicit Firecrawl policy denial is not retried through another scraper. Scrapes
are Markdown-only with PDF/image parsers disabled and Firecrawl cache storage disabled; obvious PDF
URLs stay local, and any unexpected base64/binary result is discarded. `storeInCache: false` is not a
zero-data-retention guarantee. Firecrawl document parsing can incur per-page charges, so do not add
PDF/LLM formats without an explicit budget decision. The shared limiter allows one in-flight request
per API container across both Uvicorn workers. State and locks share the API's `/tmp` tmpfs: worker
restarts preserve sharing, but container/host restarts discard monotonic-clock deadlines before a
new clock epoch can reuse them. Outside this Compose configuration, use equivalent ephemeral `/tmp`
storage. It does not coordinate multiple API containers.

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
