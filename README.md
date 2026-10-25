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
mkdir -p data/searxng data/crawl4ai data/api
# The API image runs as the unprivileged `app` user; let it append the usage ledger.
chmod 0777 data/api
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

`POST /search` exposes the free Firecrawl v2 surface; the GET shortcut remains the simple subset.

- `sources`: `web` (default), `news`, `images`. `images` results come back as image metadata
  (`image_url`, `position`) without page extraction. Non-web sources are Firecrawl-only.
- `categories`: `developer`, `gov`, `research`, `pdf`. `gov` cannot be combined with other categories
  or filters; the automatic LinkedIn/X/shortener exclusions are skipped for it. `pdf` results are
  routed to Crawl4AI with its PDF parser enabled.
- `include_domains` / `exclude_domains`: hostnames only; the two cannot be combined. Brave/SearXNG
  use `site:` / `-site:`. Setting `include_domains` disables the automatic cost-sensitive exclusions.
- `time_range` (`day`, `week`, `month`, `year`) or raw `tbs`; `country`, `location`, `safe`,
  `highlights`, `ignore_invalid_urls`.
- `scrape_options`: inline Firecrawl extraction. Markdown becomes `content`; every other returned
  field lands in `extras` (extras strings are capped by `SEARCH_MAX_EXTRAS_CHARS`; oversized
  `rawBase64` is omitted). `storeInCache` defaults to false; set true to opt into Firecrawl caching.
  Requires `depth=advanced`. `actions`, `headers`, and `profile` are rejected unless
  `ALLOW_FIRECRAWL_ACTIONS`, `ALLOW_FIRECRAWL_HEADERS`, or `ALLOW_FIRECRAWL_PROFILE` is enabled.
- `max_content_chars`: truncate returned `content` after ranking.
- `candidates` is 1..100; `max_results` is 1..20.

**Fallback contract.** Strict by default. If Firecrawl is unavailable and Brave/SearXNG/Crawl4AI
cannot honor a requested filter or inline format, the request returns `422 unsupported_feature`
(nothing can honor it, `retryable: false`) or `503 fallback_feature_unavailable` (Firecrawl is down
and the fallback is a subset, `retryable: true`). Both carry `details.provider`,
`details.unsupported` (a list such as `["format:links", "options"]`), and, for discovery,
`details.unsupported_by_provider`. Set `allow_degraded: true` to accept the subset; the response then
carries `data.degraded` and human-readable `warnings`. Every error also carries `request_id` when a
request id is available. Nothing is silently dropped.

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

Set `FIRECRAWL_KEYS_JSON` in private `.env` to an ordered list of labeled keys:

```dotenv
FIRECRAWL_KEYS_JSON='[{"label":"primary@example.test","key":"fc-REPLACE-ONE"},{"label":"secondary@example.test","key":"fc-REPLACE-TWO"}]'
```

Every entry requires exactly `label` and `key`. Labels are human-readable identifiers, not separate
balances; email-style labels are allowed. Labels are trimmed and must be nonempty and printable,
without control characters or credentials. Duplicate labels (after trimming), duplicate credentials,
and unlabeled/bare keys are rejected with a sanitized configuration error. No labels are invented,
and no account/funding or active/spare decisions are needed.

An explicitly empty value or `[]` disables Firecrawl. Only when this variable is absent, legacy
`FIRECRAWL_BUDGETS_JSON` is accepted: each entry must contain exactly one key, with its existing
`id` used as the required label. Multi-key legacy entries are ambiguous and must be migrated to
explicit labeled entries; they are not auto-labeled. Compose forwards the new variable through its
existing `env_file`; no unlabeled CSV interface is supported.

An invalid-auth 401 tries the next key; the bad key is skipped for 15 minutes before one
caller-requested operation may retry it, without a usage probe. A 402 tries the next key only after
the non-billable team-usage check confirms remaining credits are zero or negative. Keys from the
same Firecrawl team share credits: another key does not create funds, and the same depleted team
will still deny requests. Balances are not summed or assumed independent. Positive, malformed or
unavailable usage leaves a 402 ambiguous/key-cap-like and stops rotation, as do policy 403s and
rate 429s. Failover never bypasses spending caps, rate limits or policy restrictions. Cooldowns
are shared across workers by a credential hash, not by list position or label. Renaming a label or
reordering entries preserves that credential's guard; replacing a credential does not inherit it.
Logs show the readable label, never the key; control state stores only the credential hash.

No pay-as-you-go or billing setting is enabled automatically, and no universal per-request credit
cap is claimed; configure vendor key spend limits separately. After a 15-minute ambiguous-402
cooldown and cached usage refresh, one caller-requested operation may retry only the same key;
success clears the guard and another 402 restarts it. There are no background or dummy probes.
The provider exposes no per-key reset date; a verified daily/weekly cap reset can be retried by
that same-key operation. Container stop/restart or recreation clears the local `/tmp` guard,
not the provider's spending limits.

Search gives Firecrawl at most half of each stage's existing budget, including shared-lock waits,
HTTP and URL validation: 10 of 20 seconds for discovery and 22.5 of 45 seconds for extraction by default.
Custom totals scale the same way; Brave/SearXNG and Crawl4AI keep the original absolute stage deadlines.
Accepted partial results survive the Firecrawl cutoff; a timeout does not rotate credentials.

Firecrawl search performs discovery and, when `scrape_options` is set, inline page extraction in the
same call. Without inline options, page extraction uses a separate policy-checked scrape. LinkedIn,
X/Twitter and known shortener domains are excluded from Firecrawl and may use the local Crawl4AI
path instead. An explicit Firecrawl policy denial is not retried through another scraper. Direct
`/crawl` Firecrawl extraction is Markdown-only with PDF/image parsers disabled and Firecrawl cache
storage disabled; obvious PDF URLs stay local, and any unexpected base64/binary result is discarded.
`storeInCache: false` is not a zero-data-retention guarantee. Firecrawl document parsing can incur
per-page charges, so do not add PDF/LLM formats without an explicit budget decision. The shared
limiter allows one in-flight request per API container across both Uvicorn workers. State and locks
share the API's `/tmp` tmpfs: worker restarts preserve sharing, but container/host restarts discard
monotonic-clock deadlines before a new clock epoch can reuse them. Outside this Compose
configuration, use equivalent ephemeral `/tmp` storage. It does not coordinate multiple API
containers.

## Usage log and request IDs

Every HTTP response carries `X-Request-ID` (an incoming valid `X-Request-ID` is echoed; otherwise a
UUID is generated). The same id appears in log lines and in the Firecrawl usage ledger.

Each Firecrawl search/scrape attempt appends one JSON line to `FIRECRAWL_USAGE_LOG`
(default `/data/firecrawl-usage.jsonl`, bind-mounted from `./data/api`):

```json
{"ts":"2026-10-25T12:00:00.000+00:00","request_id":"…","endpoint":"/v2/search","label":"primary@example.test","status":200,"credits":2.0,"category":null,"elapsed_ms":612}
```

Search credits come from Firecrawl's `creditsUsed`; if the provider omits it the ledger records an
estimate of 2. Scrapes count one page. A write failure never fails the request. Set
`LOG_FORMAT=json` to make stdout one JSON object per line. There is no hard credit or RPM cap: the
existing reactive 429/402 guards still apply.

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
