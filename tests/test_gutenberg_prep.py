from scripts.fetch_gutenberg import chunk_paragraphs, strip_gutenberg


def test_strip_keeps_the_book_and_drops_the_banner():
    raw = (
        "Title: Example\n\n*** START OF THE PROJECT GUTENBERG EBOOK EXAMPLE ***\n\n"
        "Chapter 1\n\nThe lamp was copper.\n\n"
        "*** END OF THE PROJECT GUTENBERG EBOOK EXAMPLE ***\n\nlicense"
    )
    body = strip_gutenberg(raw)
    assert body.startswith("Chapter 1")
    assert "The lamp was copper." in body
    assert "license" not in body
    assert "START OF" not in body


def test_chunks_stay_under_the_cleaner_cap():
    paragraph = "The harbor bell is bronze. " * 50
    text = "\n\n".join([paragraph] * 80)
    parts = chunk_paragraphs(text, limit=5_000)
    assert len(parts) > 1
    assert all(len(part) <= 5_000 for part in parts)
    assert "harbor bell" in parts[0]
