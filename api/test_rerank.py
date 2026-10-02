"""Checks for the two-stage rerank path. Run: python api/test_rerank.py"""

import asyncio
import json
import time

import httpx

import app


def chunk(url, content, title="t"):
    return {"url": url, "title": title, "snippet": "", "content": content}


def fake_reranker(scores, status=200):
    """Stand-in for the cross-encoder: `scores` maps document text to a score."""

    def handler(request):
        if status != 200:
            return httpx.Response(status)
        texts = json.loads(request.content)["texts"]
        return httpx.Response(
            200, json=[{"index": i, "score": scores.get(text, 0.0)} for i, text in enumerate(texts)]
        )

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def run(query, chunks, top_k, client):
    results, _ = asyncio.run(app.rerank(client, query, chunks, top_k))
    return results


def run_with_path(query, chunks, top_k, client):
    return asyncio.run(app.rerank(client, query, chunks, top_k))


def clear_breaker():
    """The breaker is module state, so keep it from leaking between checks."""
    app._rerank_skip_until = 0.0


def test_lexical_scores_normalise_to_one():
    scores = app.lexical_scores("bm25 ranking function", ["bm25 ranking", "unrelated text"])
    assert scores[0] == 1.0, scores
    assert scores[1] == 0.0, scores
    assert app.lexical_scores("", ["anything"]) == [0.0]


def test_fused_score_spans_weight():
    assert app.fused_score(0.4, 0, 0.0) == 0.4  # weight 0 -> pure cross-encoder
    assert app.fused_score(0.4, 0, 1.0) == 1.0  # weight 1 -> pure discovery rank
    # Deeper discovery ranks score lower for the same cross-encoder score.
    assert app.fused_score(0.5, 0, 0.3) > app.fused_score(0.5, 20, 0.3)
    for rank in (0, 1, 39, 500):
        assert 0.0 <= app.fused_score(0.7, rank, app.RERANK_WEIGHT) <= 1.0


def test_cross_encoder_overrides_lexical_order():
    clear_breaker()
    chunks = [
        chunk("https://a.example", "bm25 ranking function"),
        chunk("https://b.example", "bm25 ranking function details"),
    ]
    client = fake_reranker({"bm25 ranking function": 0.1, "bm25 ranking function details": 0.9})
    results = run("bm25 ranking function", chunks, 2, client)
    assert [r.url for r in results] == ["https://b.example", "https://a.example"]
    assert results[0].score > results[1].score


def test_breaker_opens_on_failure_and_skips_the_cross_encoder():
    clear_breaker()
    calls = []

    def handler(request):
        calls.append(request)
        raise httpx.ReadTimeout("cross-encoder too slow")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    chunks = [chunk("https://a.example", "bm25 ranking function")]

    results, path = run_with_path("bm25 ranking function", chunks, 1, client)
    assert path == "lexical-fallback"
    assert len(results) == 1, "a failing cross-encoder must not cost the search its results"
    assert len(calls) == 1
    assert app.rerank_available() is False, "a timeout must open the breaker"

    run_with_path("bm25 ranking function", chunks, 1, client)
    assert len(calls) == 1, "while the breaker is open the cross-encoder is not called again"
    clear_breaker()


def test_breaker_closes_after_the_cooldown():
    clear_breaker()
    app.trip_rerank_breaker("check")
    assert app.rerank_available() is False

    app._rerank_skip_until = time.monotonic() - 1  # as if RERANK_COOLDOWN had elapsed
    assert app.rerank_available() is True, "the breaker must reopen the cross-encoder after the cooldown"
    assert app._rerank_skip_until == 0.0


def test_breaker_survives_a_real_cooldown():
    """End to end with a tiny cooldown rather than poking the internals."""
    clear_breaker()
    original, app.RERANK_COOLDOWN = app.RERANK_COOLDOWN, 0.2
    try:
        app.trip_rerank_breaker("check")
        assert app.rerank_available() is False
        time.sleep(0.25)
        assert app.rerank_available() is True
    finally:
        app.RERANK_COOLDOWN = original
        clear_breaker()


def test_fallback_keeps_lexical_order_when_reranker_is_down():
    chunks = [
        chunk("https://a.example", "unrelated text"),
        chunk("https://b.example", "bm25 ranking function"),
    ]
    results = run("bm25 ranking function", chunks, 2, fake_reranker({}, status=503))
    clear_breaker()  # that 503 opens the breaker; don't leak it into the next check
    assert [r.url for r in results] == ["https://b.example", "https://a.example"], "lexical, not crawl order"
    assert [r.score for r in results] == [None, None]


def test_one_chunk_per_url_and_top_k():
    clear_breaker()
    chunks = [chunk("https://a.example", "bm25 ranking function"), chunk("https://a.example", "bm25 again")]
    results = run("bm25 ranking function", chunks, 5, fake_reranker({}))
    assert len(results) == 1
    assert run("bm25", [], 5, fake_reranker({})) == []
    assert run_with_path("bm25", [], 5, fake_reranker({})) == ([], "empty")


def test_passages_are_truncated_and_path_is_reported():
    clear_breaker()
    seen = []

    def handler(request):
        texts = json.loads(request.content)["texts"]
        seen.extend(texts)
        return httpx.Response(200, json=[{"index": i, "score": 0.5} for i in range(len(texts))])

    long_chunk = chunk("https://a.example", "bm25 " + "x" * (app.RERANK_MAX_CHARS * 3))
    results, path = run_with_path("bm25", [long_chunk], 1, httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    assert path == "cross-encoder"
    assert seen == [long_chunk["content"][: app.RERANK_MAX_CHARS]], "cross-encoder sees the truncated passage"
    assert results[0].content == long_chunk["content"], "caller still gets the whole chunk"
    assert results[0].score is not None


if __name__ == "__main__":
    for name, test in sorted(globals().items()):
        if name.startswith("test_"):
            test()
            print(f"ok {name}")
    print("all checks passed")
