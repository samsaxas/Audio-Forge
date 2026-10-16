"""
Format-specific text extraction. Every extractor returns a List[Chapter].

Dispatch is by file extension via `extract(path)`.
"""
from __future__ import annotations

from pathlib import Path

from audioforge import Chapter
from .txt import extract_txt
from .epub import extract_epub
from .pdf import extract_pdf

_DISPATCH = {
    ".txt": extract_txt,
    ".md": extract_txt,
    ".epub": extract_epub,
    ".pdf": extract_pdf,
}


def extract(path: str) -> list[Chapter]:
    """Extract chapters from a .txt/.md, .epub, or .pdf file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    suffix = p.suffix.lower()
    handler = _DISPATCH.get(suffix)
    if handler is None:
        raise ValueError(
            f"Unsupported input format '{suffix}'. "
            f"Supported: {', '.join(sorted(_DISPATCH))}"
        )

    chapters = handler(p)
    if not chapters:
        raise ValueError(f"No extractable text found in {path}")
    return chapters
