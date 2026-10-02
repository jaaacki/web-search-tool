"""Checks for the search-path content pipeline (#12). Run: python api/test_content_quality.py"""

import app


def page(content, url="https://example.com/a"):
    return {"url": url, "title": "t", "snippet": "", "content": content}


def test_boilerplate_lines_are_dropped():
    text = "\n".join(
        [
            "[ ![Python logo](https://docs.python.org/3/_static/py.svg) ](https://www.python.org/)",
            "Docs",
            "Pricing",
            "Read more",
            "# urllib.request - Extensible library for opening URLs",
            "The boiling point of ethanol is 78.37 C at sea level.",
            "78 °C",
        ]
    )
    assert app.strip_boilerplate_lines(text).splitlines() == [
        "# urllib.request - Extensible library for opening URLs",
        "The boiling point of ethanol is 78.37 C at sea level.",
        "78 °C",
    ]


def test_boilerplate_lines_keep_prose_and_numbers():
    # A two-word line is dropped, but a heading, a digit line, and a longer
    # sentence all survive - the filter must not eat answers.
    kept = app.strip_boilerplate_lines("Read more\n# API reference\n206 bones\nA long enough prose line.").splitlines()
    assert kept == ["# API reference", "206 bones", "A long enough prose line."], kept


def test_chunking_covers_the_whole_page():
    # 20 x CHUNK_SIZE: the old first-CHUNKS_PER_PAGE limit would return 3 chunks.
    text = "\n".join(f"Sentence number {i} with enough words to matter here." for i in range(1000))
    chunks = app.chunk_documents([page(text)])
    assert len(chunks) > 10, len(chunks)
    assert sum(len(chunk["content"]) for chunk in chunks) > app.CHUNK_SIZE * 10


def test_chunking_respects_the_page_budget():
    text = "A" * app.MAX_PAGE_CHARS + "B" * 100_000
    chunks = app.chunk_documents([page(text)])
    assert chunks[-1]["content"].endswith("A"), "page budget should be cut at MAX_PAGE_CHARS"
    assert not any("B" in chunk["content"] for chunk in chunks), "content past the budget leaked into a chunk"


def test_fit_markdown_preferred_until_it_is_too_short():
    long_fit = "x" * (app.MIN_FIT_MARKDOWN_CHARS + 1)
    payload = {"markdown": {"raw_markdown": "r" * 5000, "fit_markdown": long_fit}}
    assert app.extract_crawl_content(payload) == long_fit

    payload["markdown"]["fit_markdown"] = "too short"
    assert app.extract_crawl_content(payload) == "r" * 5000

    payload["markdown"]["fit_markdown"] = ""
    assert app.extract_crawl_content(payload) == "r" * 5000


def test_search_crawl_request_asks_for_pruned_markdown():
    config = app.SEARCH_CRAWLER_CONFIG
    assert config["excluded_tags"] == ["nav", "footer", "header", "aside", "form"]
    assert config["remove_overlay_elements"] is True
    generator = config["markdown_generator"]
    assert generator["type"] == "DefaultMarkdownGenerator"
    assert generator["params"]["content_filter"]["type"] == "PruningContentFilter"
    assert generator["params"]["options"]["ignore_links"] is True


def demo():
    for name, check in sorted(globals().items()):
        if name.startswith("test_"):
            check()
            print(f"ok {name}")
    print("all checks passed")


if __name__ == "__main__":
    demo()
