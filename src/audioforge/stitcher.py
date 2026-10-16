"""
Takes per-chunk wav files and produces the final audiobook:

  1. Concatenate chunks within a chapter (with a short silence gap between
     them) -> one wav per chapter.
  2. Concatenate chapters (with a longer silence gap) -> one master wav,
     tracking each chapter's start offset as we go.
  3. Export to MP3, then embed ID3v2 `CHAP`/`CTOC` frames so podcast/
     audiobook apps show a real chapter list.
  4. Optionally also export an M4B (the native audiobook container) via
     ffmpeg chapter metadata, which has better player support for
     chapters + remembers playback position.
"""
from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path

from mutagen.id3 import ID3, CHAP, CTOC, TIT2, CTOCFlags, ID3NoHeaderError
from pydub import AudioSegment

from audioforge import ChapterAudio, SynthesizedChunk

logger = logging.getLogger(__name__)


def build_chapter_audio(
    chapter_index: int,
    chapter_title: str,
    synthesized_chunks: list[SynthesizedChunk],
    *,
    out_wav_path: str,
    silence_between_chunks_ms: int = 220,
) -> AudioSegment:
    """Concatenate a chapter's chunks (in order) into a single AudioSegment,
    also writing it to disk so a crash mid-book doesn't lose finished chapters."""
    if not synthesized_chunks:
        raise ValueError(f"Chapter {chapter_index} ('{chapter_title}') has no synthesized audio.")

    gap = AudioSegment.silent(duration=silence_between_chunks_ms)
    segment = AudioSegment.empty()
    for i, sc in enumerate(synthesized_chunks):
        audio = AudioSegment.from_wav(sc.wav_path)
        segment += audio
        if i < len(synthesized_chunks) - 1:
            segment += gap

    Path(out_wav_path).parent.mkdir(parents=True, exist_ok=True)
    segment.export(out_wav_path, format="wav")
    return segment


def combine_chapters(
    chapter_segments: list[tuple[int, str, AudioSegment]],
    *,
    silence_between_chapters_ms: int = 900,
) -> tuple[AudioSegment, list[ChapterAudio]]:
    """Concatenate chapter AudioSegments into the master track, returning
    both the audio and a ChapterAudio list with start/duration timing for
    chapter-marker embedding."""
    gap = AudioSegment.silent(duration=silence_between_chapters_ms)
    master = AudioSegment.empty()
    markers: list[ChapterAudio] = []

    cursor_ms = 0
    for i, (idx, title, seg) in enumerate(chapter_segments):
        master += seg
        markers.append(ChapterAudio(
            index=idx, title=title, wav_path="",
            start_s=cursor_ms / 1000.0, duration_s=len(seg) / 1000.0,
        ))
        cursor_ms += len(seg)
        if i < len(chapter_segments) - 1:
            master += gap
            cursor_ms += silence_between_chapters_ms

    return master, markers


def export_mp3_with_chapters(
    master: AudioSegment,
    markers: list[ChapterAudio],
    out_mp3_path: str,
    *,
    title: str = "Audiobook",
    bitrate: str = "128k",
) -> None:
    """Export to MP3 and embed ID3v2 CHAP/CTOC chapter frames."""
    Path(out_mp3_path).parent.mkdir(parents=True, exist_ok=True)
    master.export(out_mp3_path, format="mp3", bitrate=bitrate, tags={"title": title, "album": title})

    try:
        tags = ID3(out_mp3_path)
    except ID3NoHeaderError:
        tags = ID3()

    child_ids = []
    for i, m in enumerate(markers):
        chap_id = f"chp{i}"
        child_ids.append(chap_id)
        start_ms = int(m.start_s * 1000)
        end_ms = int((m.start_s + m.duration_s) * 1000)
        tags.add(CHAP(
            element_id=chap_id,
            start_time=start_ms,
            end_time=end_ms,
            start_offset=0xFFFFFFFF,
            end_offset=0xFFFFFFFF,
            sub_frames=[TIT2(encoding=3, text=[m.title])],
        ))

    tags.add(CTOC(
        element_id="toc",
        flags=CTOCFlags.TOP_LEVEL | CTOCFlags.ORDERED,
        child_element_ids=child_ids,
        sub_frames=[TIT2(encoding=3, text=[title])],
    ))
    tags.save(out_mp3_path, v2_version=3)
    logger.info("Wrote %s with %d chapter markers.", out_mp3_path, len(markers))


def export_chapters_json(markers: list[ChapterAudio], out_json_path: str) -> None:
    data = [
        {"index": m.index, "title": m.title, "start_s": round(m.start_s, 2), "duration_s": round(m.duration_s, 2)}
        for m in markers
    ]
    Path(out_json_path).write_text(json.dumps(data, indent=2), encoding="utf-8")


def export_m4b(
    wav_path: str,
    markers: list[ChapterAudio],
    out_m4b_path: str,
    *,
    title: str = "Audiobook",
) -> None:
    """Export the master wav to M4B (AAC in an MP4 container) with native
    chapter metadata, via ffmpeg. Requires ffmpeg on PATH."""
    Path(out_m4b_path).parent.mkdir(parents=True, exist_ok=True)
    metadata_path = Path(out_m4b_path).with_suffix(".ffmeta.txt")

    lines = [";FFMETADATA1", f"title={title}"]
    for m in markers:
        start_ms = int(m.start_s * 1000)
        end_ms = int((m.start_s + m.duration_s) * 1000)
        lines += [
            "[CHAPTER]", "TIMEBASE=1/1000", f"START={start_ms}", f"END={end_ms}",
            f"title={m.title}",
        ]
    metadata_path.write_text("\n".join(lines), encoding="utf-8")

    cmd = [
        "ffmpeg", "-y",
        "-i", wav_path,
        "-i", str(metadata_path), "-map_metadata", "1",
        "-c:a", "aac", "-b:a", "64k",
        "-f", "ipod",
        out_m4b_path,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info("Wrote %s", out_m4b_path)
    except FileNotFoundError:
        logger.warning("ffmpeg not found on PATH — skipping M4B export. MP3 output is unaffected.")
    except subprocess.CalledProcessError as exc:
        logger.warning("ffmpeg failed for M4B export: %s", exc.stderr[-500:] if exc.stderr else exc)
    finally:
        metadata_path.unlink(missing_ok=True)
