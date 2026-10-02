#!/usr/bin/env python3
"""Search-quality eval harness (#11). Stdlib only.

    python eval/run.py --base http://localhost:18000 --key $KEY [--depth basic] [--out results.json]
    python eval/run.py --compare base.json after.json

Judged metrics per run (rates over queries):
    domain_hit@5   expected domain appears in the top 5 result URLs
    term_hit@1     an expected term appears in the top-1 result content
    term_hit@5     an expected term appears in any of the top-5 result contents
    empty_rate     queries that returned no results at all
Plus p50/p95 latency, mean/p95 content length. Results JSON is not committed.
"""

import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

TIMEOUT = 90  # seconds per search
WORKERS = 3  # modest parallelism: more just hammers SearXNG's upstream engines
TOP_N = 5
DEFAULT_QUERIES = Path(__file__).with_name("queries.jsonl")

METRICS = ("domain_hit@5", "term_hit@1", "term_hit@5", "empty_rate")


def search(base, key, query, depth=None):
    params = {"q": query}
    if depth:
        params["depth"] = depth
    request = urllib.request.Request(
        f"{base.rstrip('/')}/search?{urllib.parse.urlencode(params)}",
        headers={"X-API-Key": key},
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            payload = json.load(response)
    except (urllib.error.URLError, OSError, ValueError) as exc:
        return {"query": query, "error": str(exc), "latency": time.perf_counter() - started, "results": []}
    return {
        "query": query,
        "latency": time.perf_counter() - started,
        "error": None if payload.get("ok") else (payload.get("error") or {}).get("code", "error"),
        "results": (payload.get("data") or {}).get("results") or [],
    }


def normalize_domain(value):
    """Reduce a URL or a bare domain to a comparable host: lower case, no www."""
    host = urllib.parse.urlparse(value if "//" in value else f"//{value}").netloc or value
    host = host.split("@")[-1].split(":")[0].lower()
    return host[4:] if host.startswith("www.") else host


def domain_hit(url, expected):
    """Exact host, or a subdomain of an expected domain (docs.x.org counts for x.org)."""
    host = normalize_domain(url)
    return any(host == want or host.endswith(f".{want}") for want in map(normalize_domain, expected))


def score(batch, query):
    expected = query["expect_domains"]
    top = batch["results"][:TOP_N]
    terms = [term.lower() for term in query["expect_terms"]]

    def hit(index):
        content = (top[index].get("content") or "").lower()
        return any(term in content for term in terms)

    return {
        "query": batch["query"],
        "latency": batch["latency"],
        "error": batch["error"],
        "results": len(batch["results"]),
        "domain_hit@5": any(domain_hit(result.get("url", ""), expected) for result in top),
        "term_hit@1": bool(top) and hit(0),
        "term_hit@5": any(hit(index) for index in range(len(top))),
        "empty": not batch["results"],
        "content_lengths": [len(result.get("content") or "") for result in top],
        "urls": [result.get("url", "") for result in top],
    }


def summarize(rows):
    latencies = [row["latency"] for row in rows]
    lengths = [length for row in rows for length in row["content_lengths"]]
    summary = {metric: sum(row[metric] for row in rows) / len(rows) for metric in METRICS if metric != "empty_rate"}
    summary["empty_rate"] = sum(row["empty"] for row in rows) / len(rows)
    summary.update(
        {
            "queries": len(rows),
            "errors": sum(1 for row in rows if row["error"]),
            "p50_latency": statistics.median(latencies),
            "p95_latency": percentile(latencies, 0.95),
            "mean_content_chars": statistics.fmean(lengths) if lengths else 0,
            "p95_content_chars": percentile(lengths, 0.95) if lengths else 0,
        }
    )
    return summary


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]


def report(summary):
    print(f"\n  {'metric':<20} {'value':>10}")
    for metric in METRICS:
        print(f"  {metric:<20} {summary[metric]:>9.1%}")
    print(f"  {'errors':<20} {summary['errors']:>10}")
    print(f"  {'p50 latency':<20} {summary['p50_latency']:>9.2f}s")
    print(f"  {'p95 latency':<20} {summary['p95_latency']:>9.2f}s")
    print(f"  {'mean content':<20} {summary['mean_content_chars']:>9.0f}c")
    print(f"  {'p95 content':<20} {summary['p95_content_chars']:>9.0f}c")


def compare(before, after):
    print(f"\n  {'metric':<20} {'before':>10} {'after':>10} {'delta':>10}")
    for metric in METRICS:
        delta = after[metric] - before[metric]
        print(f"  {metric:<20} {before[metric]:>9.1%} {after[metric]:>9.1%} {delta:>+9.1%}")
    for metric in ("p50_latency", "p95_latency", "mean_content_chars"):
        delta = after[metric] - before[metric]
        suffix = "s" if "latency" in metric else "c"
        print(f"  {metric:<20} {before[metric]:>9.2f} {after[metric]:>9.2f} {delta:>+9.2f}{suffix}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", help="API base URL, e.g. http://localhost:18000")
    parser.add_argument("--key", help="X-API-Key value")
    parser.add_argument("--depth", help="optional depth field (see #14)")
    parser.add_argument("--workers", type=int, default=WORKERS, help=f"parallel searches (default {WORKERS})")
    parser.add_argument("--queries", default=DEFAULT_QUERIES, type=Path)
    parser.add_argument("--out", type=Path, help="write per-query results here (not committed)")
    parser.add_argument("--compare", nargs=2, metavar=("BEFORE", "AFTER"), type=Path)
    args = parser.parse_args()

    if args.compare:
        before, after = (json.loads(path.read_text()) for path in args.compare)
        compare(before["summary"], after["summary"])
        return 0
    if not args.base or not args.key:
        parser.error("--base and --key are required unless --compare is used")

    queries = [json.loads(line) for line in args.queries.read_text().splitlines() if line.strip()]
    for query in queries:
        # A term already in the query is matched by nav junk and by any page that
        # echoes the question, so it cannot show whether an answer was retrieved.
        for term in query["expect_terms"]:
            if term.lower() in query["q"].lower():
                raise SystemExit(f"expect_term {term!r} appears in its own query {query['q']!r}: terms must be answer-bearing")
    print(f"{len(queries)} queries against {args.base}" + (f" depth={args.depth}" if args.depth else ""))
    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        batches = list(pool.map(lambda query: search(args.base, args.key, query["q"], args.depth), queries))
    rows = [score(batch, query) for batch, query in zip(batches, queries)]
    summary = summarize(rows)
    report(summary)
    if summary["empty_rate"] > 0.25:
        # All-zero runs look identical to a real regression; discovery here is
        # effectively single-engine (bing), which rate-limits under bursts.
        print(f"\n  WARNING: {summary['empty_rate']:.0%} of queries returned nothing - discovery looks degraded, discard this run")
    if args.out:
        args.out.write_text(json.dumps({"started": started, "base": args.base, "depth": args.depth, "summary": summary, "per_query": rows}, indent=2))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
