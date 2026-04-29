import math
import os
import re
from collections import Counter
from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field

MODEL_ID = os.getenv("MODEL_ID", "lexical-idf")
TOKEN_RE = re.compile(r"[\w-]+", re.UNICODE)

app = FastAPI(title="Reranker", version="0.1.0")


class RerankRequest(BaseModel):
    query: str = Field(min_length=1)
    documents: list[str] = Field(min_length=1)
    top_k: int | None = Field(default=None, ge=1)
    order: Literal["desc", "input"] = "desc"


class RerankResult(BaseModel):
    index: int
    score: float
    document: str


class RerankResponse(BaseModel):
    model: str
    results: list[RerankResult]


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID}


@app.post("/rerank", response_model=RerankResponse)
def rerank(request: RerankRequest):
    query_tokens = tokenize(request.query)
    document_tokens = [tokenize(document) for document in request.documents]
    scores = score_documents(query_tokens, document_tokens)

    results = [
        RerankResult(index=index, score=score, document=document)
        for index, (score, document) in enumerate(zip(scores, request.documents))
    ]

    if request.order == "desc":
        results.sort(key=lambda result: result.score, reverse=True)

    if request.top_k is not None:
        results = results[: request.top_k]

    return RerankResponse(model=MODEL_ID, results=results)


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def score_documents(query_tokens: list[str], documents: list[list[str]]) -> list[float]:
    if not query_tokens:
        return [0.0 for _ in documents]

    query_counts = Counter(query_tokens)
    document_frequencies = Counter(
        token
        for document in documents
        for token in set(document)
    )
    document_count = max(len(documents), 1)

    raw_scores = []
    for document in documents:
        counts = Counter(document)
        length_norm = math.sqrt(max(len(document), 1))
        score = 0.0

        for token, query_count in query_counts.items():
            term_frequency = counts[token]
            if term_frequency == 0:
                continue
            idf = math.log((document_count + 1) / (document_frequencies[token] + 0.5)) + 1
            score += query_count * math.log1p(term_frequency) * idf

        raw_scores.append(score / length_norm)

    max_score = max(raw_scores, default=0.0)
    if max_score <= 0:
        return [0.0 for _ in raw_scores]

    return [round(score / max_score, 6) for score in raw_scores]
