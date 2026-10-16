"""
Splits chapter text into TTS-sized chunks without ever cutting a sentence
in half — Kokoro (like most neural TTS) produces noticeably worse prosody
if you feed it a fragment, and cutting mid-sentence can audibly clip words.

Strategy: split into sentences with a lightweight regex (no NLTK/spaCy
download needed — this keeps AudioForge fully offline out of the box),
then greedily pack sentences into chunks up to `max_chars`. A single
sentence longer than `max_chars` (rare, but happens with long dialogue or
lists) is hard-split on clause boundaries (commas/semicolons) as a
fallback, never mid-word.
"""
from __future__ import annotations

import re

from audioforge import Chapter, TextChunk

_SENTENCE_RE = re.compile(
    r"""
    (?<=[.!?])      # split after sentence-ending punctuation
    ["')\]]*        # allow a trailing quote/paren
    \s+
    (?=[A-Z0-9"'(\[])  # only if followed by what looks like a new sentence
    """,
    re.VERBOSE,
)

_CLAUSE_RE = re.compile(r"(?<=[,;:])\s+")

# Matches lines like `Marcus: "Get down!"` or `ELENA — I told you.`
_SPEAKER_RE = re.compile(r"^\s*([A-Z][A-Za-z .'-]{1,30})\s*[:\u2014-]\s*(.+)$")


def split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    # Split paragraph-by-paragraph first so we never merge across a
    # paragraph break, then sentence-split within each paragraph.
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    sentences: list[str] = []
    for para in paragraphs:
        sentences.extend(s.strip() for s in _SENTENCE_RE.split(para) if s.strip())
    return sentences


def _hard_split(sentence: str, max_chars: int) -> list[str]:
    """Fallback for a single sentence longer than max_chars."""
    if len(sentence) <= max_chars:
        return [sentence]
    parts = [p.strip() for p in _CLAUSE_RE.split(sentence) if p.strip()]
    if len(parts) == 1:
        # No punctuation to split on at all — break on whitespace as a last resort.
        words = sentence.split()
        parts, buf = [], ""
        for w in words:
            if len(buf) + len(w) + 1 > max_chars:
                parts.append(buf.strip())
                buf = w
            else:
                buf = f"{buf} {w}".strip()
        if buf:
            parts.append(buf)
        return parts

    out: list[str] = []
    buf = ""
    for part in parts:
        if len(buf) + len(part) + 1 > max_chars:
            if buf:
                out.append(buf.strip())
            buf = part
        else:
            buf = f"{buf} {part}".strip()
    if buf:
        out.append(buf.strip())
    return out


def detect_speaker_voice(sentence: str, speaker_voices: dict[str, str]) -> tuple[str, str | None]:
    """If a sentence starts with `Name:`/`Name —`, strip the tag and return
    (clean_text, voice) using the configured speaker->voice map. Falls back
    to (original_text, None) for narration or unknown speakers."""
    m = _SPEAKER_RE.match(sentence)
    if not m:
        return sentence, None
    name, rest = m.group(1).strip(), m.group(2).strip()
    voice = speaker_voices.get(name)
    return (rest, voice) if voice else (sentence, None)


def chunk_chapter(
    chapter: Chapter,
    *,
    max_chars: int = 400,
    speaker_voices: dict[str, str] | None = None,
    default_voice: str | None = None,
) -> list[TextChunk]:
    """Pack a chapter's sentences into TextChunks <= max_chars each.

    Consecutive sentences are merged into one chunk as long as they share
    the same resolved voice and the combined length stays under the limit
    — this avoids a flood of tiny, choppy audio clips.
    """
    speaker_voices = speaker_voices or {}
    sentences = split_sentences(chapter.text)

    resolved: list[tuple[str, str | None]] = []
    for s in sentences:
        clean, voice = detect_speaker_voice(s, speaker_voices)
        for piece in _hard_split(clean, max_chars):
            resolved.append((piece, voice or default_voice))

    chunks: list[TextChunk] = []
    buf_text = ""
    buf_voice: str | None = None

    def flush():
        nonlocal buf_text, buf_voice
        if buf_text.strip():
            chunks.append(TextChunk(
                chapter_index=chapter.index,
                chunk_index=len(chunks),
                text=buf_text.strip(),
                voice=buf_voice,
            ))
        buf_text, buf_voice = "", None

    for text, voice in resolved:
        same_voice = (buf_voice is None) or (voice == buf_voice)
        fits = len(buf_text) + len(text) + 1 <= max_chars
        if buf_text and same_voice and fits:
            buf_text = f"{buf_text} {text}"
        else:
            flush()
            buf_text, buf_voice = text, voice
    flush()

    return chunks
