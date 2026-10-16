"""
Plain-text extraction.

Plain .txt files have no structural markup, so chapter boundaries are
detected heuristically by looking for lines that look like headings, e.g.:

    Chapter 1
    CHAPTER ONE: THE BEGINNING
    Prologue
    1.

If nothing heading-like is found, the whole file becomes a single chapter
named after the filename — the LLM-based cleaner can optionally re-split
this later (see cleaner.detect_chapters_llm).
"""
from __future__ import annotations

import re
from pathlib import Path

from audioforge import Chapter

_HEADING_RE = re.compile(
    r"""^\s*(
        (chapter|part|book|section|prologue|epilogue|interlude)\b.{0,60}
        |
        [0-9]{1,3}\s*[.:)-]?\s*$
        |
        [IVXLCDM]{1,8}\s*[.:)-]?\s*$
    )\s*$""",
    re.IGNORECASE | re.VERBOSE,
)


def _looks_like_heading(line: str) -> bool:
    line = line.strip()
    if not line or len(line) > 80:
        return False
    return bool(_HEADING_RE.match(line))


def extract_txt(path: Path) -> list[Chapter]:
    raw = path.read_text(encoding="utf-8", errors="replace")
    lines = raw.splitlines()

    chapters: list[Chapter] = []
    current_title = None
    current_lines: list[str] = []
    any_heading_found = False
    before_first_heading = True
    # Minimum length for pre-heading text (title page, byline, epigraph) to be
    # worth keeping as its own "Front Matter" chapter, rather than discarded.
    _FRONT_MATTER_MIN_CHARS = 150

    def flush():
        text = "\n".join(current_lines).strip()
        if not text:
            return
        if before_first_heading:
            if len(text) >= _FRONT_MATTER_MIN_CHARS:
                chapters.append(Chapter(index=len(chapters), title="Front Matter", text=text))
            # else: just a title/byline line — not worth narrating as a "chapter".
        else:
            title = current_title or f"Chapter {len(chapters) + 1}"
            chapters.append(Chapter(index=len(chapters), title=title, text=text))

    for line in lines:
        if _looks_like_heading(line):
            any_heading_found = True
            flush()
            before_first_heading = False
            current_title = line.strip()
            current_lines = []
        else:
            current_lines.append(line)
    flush()

    # No heading was ever detected -> the whole file is one chapter, named
    # after the file rather than defaulted to "Chapter 1".
    if not any_heading_found and len(chapters) <= 1:
        text = raw.strip()
        chapters = [Chapter(index=0, title=path.stem, text=text)] if text else []

    return chapters
