"""
Text cleanup, in two layers:

1. `regex_clean` — always runs. Cheap, deterministic whitespace/typography
   fixes. No dependencies beyond the stdlib.
2. `llm_clean` — optional. Calls a local Ollama model to fix things regex
   can't: broken OCR words, awkward mid-sentence line breaks that survived
   dehyphenation, stray running-header fragments, and (for TTS specifically)
   expanding things that sound wrong when read aloud verbatim, e.g. "Dr."
   -> "Doctor", "e.g." -> "for example", "$1,200" -> "one thousand two
   hundred dollars".

Both operate on a single Chapter's text at a time and are called from
pipeline.py before chunking. `llm_clean` chunks its own input into
LLM-context-sized pieces internally so it works on chapters of any length.
"""
from __future__ import annotations

import logging
import re

import requests

logger = logging.getLogger(__name__)

_WHITESPACE_RE = re.compile(r"[ \t]+")
_BLANK_LINES_RE = re.compile(r"\n{3,}")
_SMART_QUOTES = {
    "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
    "\u2013": "-", "\u2014": "-", "\u2026": "...",
}


def regex_clean(text: str) -> str:
    for src, dst in _SMART_QUOTES.items():
        text = text.replace(src, dst)
    text = _WHITESPACE_RE.sub(" ", text)
    text = _BLANK_LINES_RE.sub("\n\n", text)
    lines = [ln.strip() for ln in text.splitlines()]
    return "\n".join(lines).strip()


_LLM_CLEAN_PROMPT = """You are cleaning up text that was extracted from a book so it can be \
read aloud by a text-to-speech engine. Fix ONLY the following, and change \
nothing else:

- Broken words caused by OCR or bad line-wrapping (e.g. "wa lking" -> "walking")
- Stray running headers/page artifacts that don't belong in the prose
- Obvious mid-sentence line-break garbage

Do NOT summarize, rephrase, censor, translate, or shorten the text. Do NOT \
add commentary. Return ONLY the cleaned text, nothing else — no preamble, \
no markdown fences.

TEXT:
{text}"""


def _ollama_generate(host: str, model: str, prompt: str, timeout: int = 120) -> str:
    resp = requests.post(
        f"{host.rstrip('/')}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False, "options": {"temperature": 0.1}},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json().get("response", "").strip()


def llm_clean(
    text: str,
    *,
    model: str = "llama3.2",
    host: str = "http://localhost:11434",
    max_piece_chars: int = 3000,
) -> str:
    """Run text through a local Ollama model for deeper cleanup.

    Falls back to the input unchanged (with a warning) if Ollama isn't
    reachable — a local LLM being offline should never crash a conversion.
    """
    pieces = [text[i:i + max_piece_chars] for i in range(0, len(text), max_piece_chars)] or [text]
    cleaned_pieces = []
    for piece in pieces:
        try:
            cleaned = _ollama_generate(host, model, _LLM_CLEAN_PROMPT.format(text=piece))
            # Guard against a degenerate/empty LLM response wiping out real content.
            cleaned_pieces.append(cleaned if len(cleaned) > 0.5 * len(piece) else piece)
        except requests.RequestException as exc:
            logger.warning(
                "Ollama cleanup unavailable (%s) — falling back to regex-cleaned text only. "
                "Is `ollama serve` running?", exc
            )
            cleaned_pieces.append(piece)
    return "\n".join(cleaned_pieces)


_CHAPTER_SPLIT_PROMPT = """You are given plain text extracted from a book that has NO chapter \
markings. Identify natural chapter/section break points and return ONLY a \
JSON array of the 0-indexed character offsets where each new chapter \
begins (the first offset should be 0). Return nothing but the JSON array, \
e.g. [0, 4231, 9876]. If you cannot find clear breaks, return [0].

TEXT:
{text}"""


def detect_chapters_llm(
    text: str,
    *,
    model: str = "llama3.2",
    host: str = "http://localhost:11434",
    max_chars: int = 12000,
) -> list[int]:
    """Ask a local LLM for likely chapter-break offsets in unstructured text.

    Only practical on short-to-medium texts (one LLM context window). For
    a full book, prefer per-chapter cleanup and keep chapter *detection* to
    the format-specific extractors, which is why this is opt-in and capped.
    """
    import json

    sample = text[:max_chars]
    try:
        raw = _ollama_generate(host, model, _CHAPTER_SPLIT_PROMPT.format(text=sample))
        raw = raw.strip().strip("`")
        offsets = json.loads(raw)
        offsets = sorted({int(o) for o in offsets if isinstance(o, (int, float)) and 0 <= o < len(sample)})
        return offsets or [0]
    except (requests.RequestException, ValueError, TypeError) as exc:
        logger.warning("LLM chapter detection failed (%s) — treating text as one chapter.", exc)
        return [0]
