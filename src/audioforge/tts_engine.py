"""
Wraps kokoro-onnx and adds the two things a bare Kokoro call doesn't give
you for free: content-hash caching (so re-runs after a crash are nearly
instant) and a worker pool for throughput.

Caching design: each chunk's cache key is a hash of
(text, voice, speed, lang) — NOT chapter/chunk index. This means editing
one paragraph of a book invalidates only that paragraph's cache entry,
and identical repeated text (e.g. "he said.") across a book is only ever
synthesized once.
"""
from __future__ import annotations

import hashlib
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import soundfile as sf

from audioforge import SynthesizedChunk, TextChunk

logger = logging.getLogger(__name__)


def _cache_key(text: str, voice: str, speed: float, lang: str) -> str:
    h = hashlib.sha256(f"{voice}|{speed}|{lang}|{text}".encode("utf-8")).hexdigest()
    return h[:24]


class KokoroTTSEngine:
    """Thin, cached wrapper around kokoro_onnx.Kokoro."""

    def __init__(
        self,
        models_dir: str,
        cache_dir: str,
        *,
        default_voice: str = "af_sarah",
        speed: float = 1.0,
        lang: str = "en-us",
        model_filename: str = "kokoro-v1.0.onnx",
        voices_filename: str = "voices-v1.0.bin",
    ):
        from kokoro_onnx import Kokoro  # imported lazily so `estimate`/tests don't need it installed

        model_path = Path(models_dir) / model_filename
        voices_path = Path(models_dir) / voices_filename
        if not model_path.exists() or not voices_path.exists():
            raise FileNotFoundError(
                f"Kokoro model files not found in {models_dir}. "
                f"Expected {model_path.name} and {voices_path.name} — see README for download links."
            )

        self.kokoro = Kokoro(str(model_path), str(voices_path))
        self.default_voice = default_voice
        self.default_speed = speed
        self.lang = lang
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def available_voices(self) -> list[str]:
        try:
            return sorted(self.kokoro.get_voices())
        except AttributeError:
            return sorted(getattr(self.kokoro, "voices", {}).keys())

    def _synth_one(self, chunk: TextChunk, retries: int = 3) -> SynthesizedChunk:
        voice = chunk.voice or self.default_voice
        key = _cache_key(chunk.text, voice, self.default_speed, self.lang)
        wav_path = self.cache_dir / f"{key}.wav"

        if wav_path.exists():
            info = sf.info(str(wav_path))
            return SynthesizedChunk(chunk=chunk, wav_path=str(wav_path), duration_s=info.duration)

        last_exc = None
        for attempt in range(1, retries + 1):
            try:
                samples, sample_rate = self.kokoro.create(
                    chunk.text, voice=voice, speed=self.default_speed, lang=self.lang
                )
                samples = np.asarray(samples, dtype=np.float32)
                sf.write(str(wav_path), samples, sample_rate)
                duration = len(samples) / sample_rate
                return SynthesizedChunk(chunk=chunk, wav_path=str(wav_path), duration_s=duration)
            except Exception as exc:  # kokoro/onnxruntime can raise several exception types
                last_exc = exc
                wait = 0.5 * attempt
                logger.warning(
                    "TTS failed for chapter %d chunk %d (attempt %d/%d): %s — retrying in %.1fs",
                    chunk.chapter_index, chunk.chunk_index, attempt, retries, exc, wait,
                )
                time.sleep(wait)
        raise RuntimeError(
            f"TTS permanently failed for chapter {chunk.chapter_index} chunk {chunk.chunk_index} "
            f"after {retries} attempts: {last_exc}"
        )

    def synthesize_all(
        self,
        chunks: list[TextChunk],
        *,
        workers: int = 2,
        progress_cb=None,
    ) -> list[SynthesizedChunk]:
        """Synthesize every chunk, using a thread pool for throughput.

        Order of the returned list matches the input order regardless of
        completion order — callers rely on this for stitching.
        """
        results: list[SynthesizedChunk | None] = [None] * len(chunks)

        if workers <= 1:
            for i, chunk in enumerate(chunks):
                results[i] = self._synth_one(chunk)
                if progress_cb:
                    progress_cb(1)
            return results  # type: ignore[return-value]

        with ThreadPoolExecutor(max_workers=workers) as pool:
            future_to_idx = {pool.submit(self._synth_one, c): i for i, c in enumerate(chunks)}
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                results[idx] = future.result()  # re-raises on failure, aborting the run loudly
                if progress_cb:
                    progress_cb(1)

        return results  # type: ignore[return-value]
