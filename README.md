# AI web search stack

Self-hosted stack for `websearch.sparkfn.io` and controlled headless crawling at `webcrawl.sparkfn.io`:

- `api`: OpenAPI/Swagger-compatible search and crawl API exposed through Traefik.
- `searxng`: internal URL discovery service.
- `crawl4ai`: internal page extraction service used by the API.
- `reranker`: internal lightweight lexical reranker API.

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

- Swagger UI: `https://websearch.sparkfn.io/docs` or `https://webcrawl.sparkfn.io/docs`
- OpenAPI JSON: `https://websearch.sparkfn.io/openapi.json` or `https://webcrawl.sparkfn.io/openapi.json`

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

`/health`, `/docs`, and `/openapi.json` are public. `/search` requires:

```http
X-API-Key: <WEBSEARCH_API_KEY>
```

Invalid or missing API keys return the same error envelope format with `code: "unauthorized"`.

## API endpoints

```bash
curl 'https://websearch.sparkfn.io/health'
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
- `browser_config`: optional Crawl4AI browser config object.
- `crawler_config`: optional Crawl4AI crawler config object.
- `extraction_config`: optional Crawl4AI extraction config object.
- `crawl_options`: optional top-level Crawl4AI options object.

## Local component debugging

Internal services are not published to host ports. To debug them on the server, exec through Compose:

```bash
docker compose exec searxng wget -qO- 'http://localhost:8080/search?q=test&format=json'
docker compose exec crawl4ai wget -qO- 'http://localhost:11235/monitor/health'
docker compose exec reranker python -c 'print("reranker container ok")'
```

## Crawl4AI config

This stack uses `data/crawl4ai/.llm.env`. Add provider keys there only if you use Crawl4AI features that require LLM credentials.
