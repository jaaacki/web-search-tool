"""Stealth before/after check: crawl a handful of protected URLs and report pass/fail.

Talks to the Crawl4AI server directly so the result reflects its browser defaults
rather than the API's payload construction. Stdlib only.

    CRAWL4AI_URL=http://localhost:18002 CRAWL4AI_API_TOKEN=... python eval/stealth_check.py
"""

import json
import os
import time
import urllib.error
import urllib.request

CRAWL4AI_URL = os.getenv("CRAWL4AI_URL", "http://localhost:11235").rstrip("/")
CRAWL4AI_API_TOKEN = os.getenv("CRAWL4AI_API_TOKEN", "")
MIN_CHARS = 800  # below this the page is a challenge/blank shell, not content

URLS = [
    ("fingerprint", "https://bot.sannysoft.com/"),
    ("cloudflare", "https://nowsecure.nl/"),
    ("medium", "https://medium.com/@addyosmani/the-cost-of-javascript-in-2018-7d8950fbb5d4"),
    ("stackoverflow", "https://stackoverflow.com/questions/11227809/why-is-processing-a-sorted-array-faster-than-processing-an-unsorted-array"),
    ("datadome", "https://www.g2.com/categories/project-management"),
]

# Phrases that mean we got a challenge page, not the article.
CHALLENGE_MARKERS = (
    "just a moment", "attention required", "checking your browser", "enable javascript and cookies",
    "verify you are human", "are you a robot", "access denied", "captcha", "request blocked",
    "unusual traffic", "please enable js",
)


def crawl(url: str) -> dict:
    payload = json.dumps({"urls": [url]}).encode()
    headers = {"Content-Type": "application/json"}
    if CRAWL4AI_API_TOKEN:
        headers["Authorization"] = "Bearer " + CRAWL4AI_API_TOKEN
    request = urllib.request.Request(f"{CRAWL4AI_URL}/crawl", data=payload, headers=headers)
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            body = json.loads(response.read())
    except (urllib.error.HTTPError, urllib.error.URLError, ValueError) as exc:
        return {"ok": False, "seconds": time.perf_counter() - started, "reason": f"{exc!r}"[:120]}
    seconds = time.perf_counter() - started

    results = body.get("results") or ([] if body.get("success") else [body])
    if not results:
        return {"ok": False, "seconds": seconds, "reason": "no results in payload"}
    result = results[0]
    markdown = result.get("markdown") or ""
    if isinstance(markdown, dict):
        markdown = markdown.get("raw_markdown") or markdown.get("fit_markdown") or ""
    text = markdown if isinstance(markdown, str) else str(markdown)

    if result.get("success") is False:
        return {"ok": False, "seconds": seconds, "chars": len(text),
                "reason": str(result.get("error_message") or "crawl reported failure")[:120]}
    hit = next((m for m in CHALLENGE_MARKERS if m in text.lower()[:4000]), None)
    if hit:
        return {"ok": False, "seconds": seconds, "chars": len(text), "reason": f"challenge marker: {hit!r}"}
    if len(text) < MIN_CHARS:
        return {"ok": False, "seconds": seconds, "chars": len(text), "reason": f"content under {MIN_CHARS} chars"}
    return {"ok": True, "seconds": seconds, "chars": len(text), "reason": "content"}


def main():
    label = os.getenv("LABEL", "run")
    print(f"{label}: {CRAWL4AI_URL}, {len(URLS)} urls\n")
    print(f"{'site':<16}{'result':<8}{'chars':>7}{'secs':>7}  reason")
    outcomes = []
    for name, url in URLS:
        outcome = crawl(url)
        outcomes.append(outcome["ok"])
        print(f"{name:<16}{'PASS' if outcome['ok'] else 'FAIL':<8}{outcome.get('chars', 0):>7}"
              f"{outcome['seconds']:>7.1f}  {outcome['reason']}")
    print(f"\n{sum(outcomes)}/{len(outcomes)} passed")


if __name__ == "__main__":
    main()
