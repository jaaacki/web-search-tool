# AI web search stack

Self-hosted stack for `websearch.sparkfn.io`:

- `api`: OpenAPI/Swagger-compatible search API exposed through Traefik.
- `searxng`: internal URL discovery service.
- `crawl4ai`: internal page extraction service.
- `reranker`: internal lightweight lexical reranker API.

Only `api` is routed by Traefik. Internal services use Docker `expose` only and are not host-published.

## Required files

Create `.env`:

```bash
cp .env.example .env
```

Edit `SEARXNG_SECRET` to a long random value.

Create runtime data/config directories:

```bash
mkdir -p data/searxng data/crawl4ai
cp data.crawl4ai.llm.env.example data/crawl4ai/.llm.env
```

All persistent/runtime data belongs under `./data`. The Compose file does not use named Docker volumes.

## Deploy

```bash
docker compose up -d --build
```

## OpenAPI / Swagger

- Swagger UI: `https://websearch.sparkfn.io/docs`
- OpenAPI JSON: `https://websearch.sparkfn.io/openapi.json`

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

## API endpoint

```bash
curl 'https://websearch.sparkfn.io/health'
```

```bash
curl -X POST 'https://websearch.sparkfn.io/search' \
  -H 'content-type: application/json' \
  -d '{"query":"open source web search engines", "max_results": 5}'
```

Shortcut:

```bash
curl 'https://websearch.sparkfn.io/search?q=open%20source%20web%20search%20engines'
```

## Local component debugging

Internal services are not published to host ports. To debug them on the server, exec through Compose:

```bash
docker compose exec searxng wget -qO- 'http://localhost:8080/search?q=test&format=json'
docker compose exec crawl4ai wget -qO- 'http://localhost:11235/monitor/health'
docker compose exec reranker python -c 'print("reranker container ok")'
```

## Crawl4AI config

This stack uses `data/crawl4ai/.llm.env`. Add provider keys there only if you use Crawl4AI features that require LLM credentials.
