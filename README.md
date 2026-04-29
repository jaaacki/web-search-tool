# AI web search stack

Self-hosted stack:

- `searxng`: finds candidate URLs.
- `crawl4ai`: dedicated Crawl4AI container for this stack.
- `reranker`: lightweight lexical reranker API.
- `api`: OpenAPI/Swagger-compatible search API exposed on `127.0.0.1:8000`.

All published ports bind to `127.0.0.1` by default. Do not expose these services directly to the internet without authentication and a reverse proxy.

## Start

Create `.env` first:

```bash
cp .env.example .env
```

Edit `SEARXNG_SECRET` to a long random value, then start:

```bash
docker compose up --build
```

## OpenAPI / Swagger

- Swagger UI: `http://localhost:8000/docs`
- OpenAPI JSON: `http://localhost:8000/openapi.json`

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
curl 'http://localhost:8000/health'
```

```bash
curl -X POST 'http://localhost:8000/search' \
  -H 'content-type: application/json' \
  -d '{"query":"open source web search engines", "max_results": 5}'
```

Shortcut:

```bash
curl 'http://localhost:8000/search?q=open%20source%20web%20search%20engines'
```

## Direct component endpoints

These are bound to localhost for debugging only.

### SearXNG

```bash
curl 'http://localhost:8080/search?q=open%20source%20search&format=json'
```

### Crawl4AI

```bash
curl -X POST 'http://localhost:11235/crawl' \
  -H 'content-type: application/json' \
  -d '{"urls":["https://example.com"]}'
```

### Reranker

```bash
curl -X POST 'http://localhost:7997/rerank' \
  -H 'content-type: application/json' \
  -d '{"query":"web search", "documents":["SearXNG is a metasearch engine", "Bananas are yellow"], "top_k": 1}'
```

## Crawl4AI config

This stack has its own `crawl4ai/.llm.env`. Add provider keys there only if you use Crawl4AI features that require LLM credentials.
