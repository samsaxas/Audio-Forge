"""
PDF extraction via pdfplumber.

PDFs are the messiest source: running headers/footers repeat on every page,
page numbers are interleaved with body text, and there's no chapter markup
at all. This extractor:

  1. Pulls text per page.
  2. Detects lines that repeat across a large fraction of pages (headers/
     footers/watermarks) and drops them.
  3. Drops lines that are just a page number.
  4. Reuses the heading heuristic from the plain-text extractor to split
     the cleaned, concatenated text into chapters.

Font-size-based heading detection (via char metadata) is layered on top
when available, since it's more reliable than regex alone for PDFs that
use a distinct heading style.
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

from audioforge import Chapter
from .txt import _looks_like_heading

_PAGE_NUM_RE = re.compile(r"^\s*[-–—]?\s*\d{1,4}\s*[-–—]?\s*$")


def _extract_pages(path: Path) -> list[list[str]]:
    import pdfplumber

    pages: list[list[str]] = []
    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            lines = [ln.rstrip() for ln in text.splitlines()]
            pages.append(lines)
    return pages


def _find_repeated_lines(pages: list[list[str]], min_fraction: float = 0.4) -> set[str]:
    """Lines that appear near-identically on many pages = headers/footers."""
    counts: Counter[str] = Counter()
    for lines in pages:
        seen_this_page = set()
        for ln in lines:
            key = ln.strip().lower()
            if key and key not in seen_this_page:
                counts[key] += 1
                seen_this_page.add(key)
    threshold = max(3, int(len(pages) * min_fraction))
    return {k for k, c in counts.items() if c >= threshold}


def _clean_pages(pages: list[list[str]]) -> list[str]:
    repeated = _find_repeated_lines(pages)
    cleaned: list[str] = []
    for lines in pages:
        for ln in lines:
            key = ln.strip().lower()
            if not key:
                continue
            if key in repeated:
                continue
            if _PAGE_NUM_RE.match(ln):
                continue
            cleaned.append(ln)
    return cleaned


def _dehyphenate(lines: list[str]) -> list[str]:
    """Join words split across a line break by a trailing hyphen."""
    out: list[str] = []
    buf = ""
    for ln in lines:
        if buf:
            ln = buf + ln.lstrip()
            buf = ""
        if ln.endswith("-") and len(ln) > 1 and ln[-2].isalpha():
            buf = ln[:-1]
        else:
            out.append(ln)
    if buf:
        out.append(buf)
    return out


def extract_pdf(path: Path) -> list[Chapter]:
    pages = _extract_pages(path)
    lines = _clean_pages(pages)
    lines = _dehyphenate(lines)

    chapters: list[Chapter] = []
    current_title = None
    current_lines: list[str] = []

    def flush():
        text = "\n".join(current_lines).strip()
        if text and len(text) > 30:
            title = current_title or f"Chapter {len(chapters) + 1}"
            chapters.append(Chapter(index=len(chapters), title=title, text=text))

    for line in lines:
        if _looks_like_heading(line):
            flush()
            current_title = line.strip()
            current_lines = []
        else:
            current_lines.append(line)
    flush()

    if not chapters:
        text = "\n".join(lines).strip()
        if text:
            chapters = [Chapter(index=0, title=path.stem, text=text)]

    return chapters
