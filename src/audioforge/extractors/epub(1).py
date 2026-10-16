"""
EPUB extraction via ebooklib.

EPUBs already have real structure: a spine (reading order) of XHTML
documents, each usually one chapter. We walk the spine in order, strip
markup with BeautifulSoup, and use the first heading tag (h1/h2/title) in
each document as the chapter title.
"""
from __future__ import annotations

from pathlib import Path

from audioforge import Chapter


def extract_epub(path: Path) -> list[Chapter]:
    import ebooklib
    from ebooklib import epub
    from bs4 import BeautifulSoup

    book = epub.read_epub(str(path))
    chapters: list[Chapter] = []

    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        soup = BeautifulSoup(item.get_content(), "lxml")

        # Skip nav/toc-only documents — they contain no narratable prose.
        if soup.find("nav"):
            continue

        text = soup.get_text(separator="\n")
        text = "\n".join(ln.strip() for ln in text.splitlines() if ln.strip())
        if len(text) < 40:  # skip cover pages, blank separators, etc.
            continue

        heading = soup.find(["h1", "h2", "h3"])
        if heading and heading.get_text(strip=True):
            title = heading.get_text(strip=True)
        elif soup.title and soup.title.get_text(strip=True):
            title = soup.title.get_text(strip=True)
        else:
            title = f"Chapter {len(chapters) + 1}"

        chapters.append(Chapter(index=len(chapters), title=title, text=text))

    return chapters
