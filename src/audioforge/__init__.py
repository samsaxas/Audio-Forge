"""
AudioForge — local-first text-to-audiobook converter.

Pipeline: extract -> clean -> chunk -> synthesize (Kokoro) -> stitch -> mp3/m4b
"""
from __future__ import annotations

from dataclasses import dataclass, field

__version__ = "1.0.0"


@dataclass
class Chapter:
    """A single chapter of source text, before chunking."""
    index: int
    title: str
    text: str


@dataclass
class TextChunk:
    """A sentence-safe slice of a chapter, ready for TTS."""
    chapter_index: int
    chunk_index: int
    text: str
    voice: str | None = None  # per-chunk voice override (speaker detection)


@dataclass
class SynthesizedChunk:
    """A chunk after TTS, with the path to its cached wav file."""
    chunk: TextChunk
    wav_path: str
    duration_s: float


@dataclass
class ChapterAudio:
    """A finished chapter: concatenated audio + timing info for chapter markers."""
    index: int
    title: str
    wav_path: str
    start_s: float
    duration_s: float


@dataclass
class RunConfig:
    """All user-facing options for a single conversion run."""
    input_path: str
    output_dir: str
    models_dir: str = "models"
    default_voice: str = "af_sarah"
    speed: float = 1.0
    lang: str = "en-us"
    clean: bool = False
    ollama_model: str = "llama3.2"
    ollama_host: str = "http://localhost:11434"
    workers: int = 2
    max_chunk_chars: int = 400
    silence_between_chunks_ms: int = 220
    silence_between_chapters_ms: int = 900
    chapter_voices: dict[str, str] = field(default_factory=dict)
    speaker_voices: dict[str, str] = field(default_factory=dict)
    make_m4b: bool = False
    cache_dir: str | None = None  # defaults to <output_dir>/.cache
