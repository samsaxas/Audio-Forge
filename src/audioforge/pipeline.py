"""
Orchestrates the full extract -> clean -> chunk -> synthesize -> stitch
pipeline. This is the module the CLI calls; every stage below is also
independently importable/testable.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

from tqdm import tqdm

from audioforge import Chapter, ChapterAudio, RunConfig, TextChunk
from audioforge import extractors
from audioforge.chunker import chunk_chapter
from audioforge.cleaner import llm_clean, regex_clean
from audioforge.stitcher import (
    build_chapter_audio,
    combine_chapters,
    export_chapters_json,
    export_m4b,
    export_mp3_with_chapters,
)
from audioforge.tts_engine import KokoroTTSEngine

logger = logging.getLogger(__name__)

# Rough spoken-word rate at speed=1.0; used only for the pre-synthesis estimate.
_WORDS_PER_MINUTE = 165


def load_and_clean(cfg: RunConfig) -> list[Chapter]:
    logger.info("Extracting text from %s", cfg.input_path)
    chapters = extractors.extract(cfg.input_path)
    logger.info("Found %d chapter(s)", len(chapters))

    cleaned: list[Chapter] = []
    for ch in tqdm(chapters, desc="Cleaning text", unit="chapter"):
        text = regex_clean(ch.text)
        if cfg.clean:
            text = llm_clean(text, model=cfg.ollama_model, host=cfg.ollama_host)
        cleaned.append(Chapter(index=ch.index, title=ch.title, text=text))
    return cleaned


def build_chunks(chapters: list[Chapter], cfg: RunConfig) -> list[TextChunk]:
    all_chunks: list[TextChunk] = []
    for ch in chapters:
        voice = cfg.chapter_voices.get(ch.title, cfg.default_voice)
        chunks = chunk_chapter(
            ch,
            max_chars=cfg.max_chunk_chars,
            speaker_voices=cfg.speaker_voices,
            default_voice=voice,
        )
        all_chunks.extend(chunks)
    return all_chunks


def estimate(cfg: RunConfig) -> dict:
    """Dry run: extract + clean (regex only, no LLM/TTS) and report stats,
    without touching Kokoro or Ollama."""
    chapters = extractors.extract(cfg.input_path)
    chapters = [Chapter(c.index, c.title, regex_clean(c.text)) for c in chapters]
    chunks = build_chunks(chapters, cfg)

    total_words = sum(len(c.text.split()) for c in chapters)
    est_minutes = total_words / _WORDS_PER_MINUTE / max(cfg.speed, 0.1)

    return {
        "chapters": len(chapters),
        "chapter_titles": [c.title for c in chapters],
        "total_words": total_words,
        "total_chunks": len(chunks),
        "estimated_audio_minutes": round(est_minutes, 1),
        "estimated_audio_hms": _format_hms(est_minutes * 60),
    }


def _format_hms(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}"


def run(cfg: RunConfig) -> dict:
    """Execute the full pipeline. Returns a summary dict with output paths."""
    t0 = time.time()
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = cfg.cache_dir or str(out_dir / ".cache")
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    chapters = load_and_clean(cfg)
    engine = KokoroTTSEngine(
        cfg.models_dir, cache_dir,
        default_voice=cfg.default_voice, speed=cfg.speed, lang=cfg.lang,
    )

    book_title = Path(cfg.input_path).stem
    chapter_segments = []

    for ch in chapters:
        voice = cfg.chapter_voices.get(ch.title, cfg.default_voice)
        chunks = chunk_chapter(
            ch, max_chars=cfg.max_chunk_chars,
            speaker_voices=cfg.speaker_voices, default_voice=voice,
        )
        if not chunks:
            logger.warning("Chapter '%s' produced no chunks — skipping.", ch.title)
            continue

        with tqdm(total=len(chunks), desc=f"Synthesizing: {ch.title[:30]}", unit="chunk") as bar:
            synthesized = engine.synthesize_all(
                chunks, workers=cfg.workers, progress_cb=lambda n: bar.update(n)
            )

        chapter_wav = str(Path(cache_dir) / f"chapter_{ch.index:04d}.wav")
        segment = build_chapter_audio(
            ch.index, ch.title, synthesized,
            out_wav_path=chapter_wav,
            silence_between_chunks_ms=cfg.silence_between_chunks_ms,
        )
        chapter_segments.append((ch.index, ch.title, segment))
        logger.info("Chapter '%s' done (%.1fs of audio).", ch.title, len(segment) / 1000.0)

    if not chapter_segments:
        raise RuntimeError("No chapters produced any audio — nothing to stitch.")

    master, markers = combine_chapters(
        chapter_segments, silence_between_chapters_ms=cfg.silence_between_chapters_ms
    )

    mp3_path = str(out_dir / f"{book_title}.mp3")
    json_path = str(out_dir / f"{book_title}.chapters.json")
    export_mp3_with_chapters(master, markers, mp3_path, title=book_title)
    export_chapters_json(markers, json_path)

    m4b_path = None
    if cfg.make_m4b:
        master_wav = str(Path(cache_dir) / "_master.wav")
        master.export(master_wav, format="wav")
        m4b_path = str(out_dir / f"{book_title}.m4b")
        export_m4b(master_wav, markers, m4b_path, title=book_title)

    elapsed = time.time() - t0
    logger.info("Done in %.1fs. Output: %s", elapsed, mp3_path)

    return {
        "mp3": mp3_path,
        "m4b": m4b_path,
        "chapters_json": json_path,
        "chapters": len(chapter_segments),
        "duration_s": len(master) / 1000.0,
        "elapsed_s": round(elapsed, 1),
    }
