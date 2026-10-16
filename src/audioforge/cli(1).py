"""
`audioforge` command-line interface.

    audioforge convert book.epub --models-dir models/ --voice af_sarah
    audioforge convert --config myconfig.yaml
    audioforge estimate book.pdf
    audioforge voices --models-dir models/
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

from audioforge import RunConfig
from audioforge import pipeline


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _load_config_file(path: str) -> dict:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return data


def _cfg_from_args(args: argparse.Namespace) -> RunConfig:
    file_cfg = _load_config_file(args.config) if getattr(args, "config", None) else {}

    def pick(key, default=None):
        cli_val = getattr(args, key, None)
        if cli_val is not None:
            return cli_val
        return file_cfg.get(key, default)

    input_path = pick("input")
    if not input_path:
        raise SystemExit("No input file given (pass a path, or `input:` in --config).")

    return RunConfig(
        input_path=input_path,
        output_dir=pick("output", "out"),
        models_dir=pick("models_dir", "models"),
        default_voice=pick("voice", "af_sarah"),
        speed=float(pick("speed", 1.0)),
        lang=pick("lang", "en-us"),
        clean=bool(pick("clean", False)),
        ollama_model=pick("ollama_model", "llama3.2"),
        ollama_host=pick("ollama_host", "http://localhost:11434"),
        workers=int(pick("workers", 2)),
        max_chunk_chars=int(pick("max_chunk_chars", 400)),
        silence_between_chunks_ms=int(pick("silence_between_chunks_ms", 220)),
        silence_between_chapters_ms=int(pick("silence_between_chapters_ms", 900)),
        chapter_voices=file_cfg.get("chapter_voices", {}),
        speaker_voices=file_cfg.get("speaker_voices", {}),
        make_m4b=bool(pick("m4b", False)),
    )


def cmd_convert(args: argparse.Namespace) -> int:
    cfg = _cfg_from_args(args)
    result = pipeline.run(cfg)
    print("\n✅ Audiobook ready:")
    print(f"   MP3:      {result['mp3']}")
    if result.get("m4b"):
        print(f"   M4B:      {result['m4b']}")
    print(f"   Chapters: {result['chapters']} ({result['chapters_json']})")
    print(f"   Duration: {pipeline._format_hms(result['duration_s'])}")
    print(f"   Took:     {result['elapsed_s']}s")
    return 0


def cmd_estimate(args: argparse.Namespace) -> int:
    cfg = _cfg_from_args(args)
    stats = pipeline.estimate(cfg)
    print(f"\n📖 {cfg.input_path}")
    print(f"   Chapters:        {stats['chapters']}")
    for t in stats["chapter_titles"][:15]:
        print(f"     - {t}")
    if len(stats["chapter_titles"]) > 15:
        print(f"     ... and {len(stats['chapter_titles']) - 15} more")
    print(f"   Total words:     {stats['total_words']:,}")
    print(f"   TTS chunks:      {stats['total_chunks']:,}")
    print(f"   Estimated audio: {stats['estimated_audio_hms']}  (~{stats['estimated_audio_minutes']} min)")
    return 0


def cmd_voices(args: argparse.Namespace) -> int:
    from audioforge.tts_engine import KokoroTTSEngine

    engine = KokoroTTSEngine(args.models_dir, cache_dir=".voices_cache_tmp")
    for v in engine.available_voices():
        print(v)
    return 0


def cmd_sample(args: argparse.Namespace) -> int:
    voices = [v.strip() for v in args.voices.split(",")] if args.voices else pipeline.DEFAULT_SAMPLE_VOICES

    if args.text:
        text = args.text
    elif args.input:
        text = pipeline.sample_text_from_input(args.input, max_chars=args.max_chars)
    else:
        text = (
            "The old lighthouse stood at the edge of the world, its light sweeping "
            "slowly across the water. Every night, it promised the same thing: "
            "you are not lost, not yet."
        )

    print(f"Sample text ({len(text)} chars):\n  \"{text}\"\n")
    print(f"Synthesizing {len(voices)} voice(s): {', '.join(voices)}\n")

    results = pipeline.sample_voices(
        models_dir=args.models_dir,
        voices=voices,
        text=text,
        output_dir=args.output,
        speed=args.speed or 1.0,
        lang=args.lang or "en-us",
    )

    print("\n✅ Samples ready — listen and compare:")
    for voice, info in results.items():
        print(f"   {voice:12s} {info['path']}  ({info['duration_s']}s)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="audioforge", description="Local text-to-audiobook converter.")
    p.add_argument("-v", "--verbose", action="store_true", help="Debug logging.")
    sub = p.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("input", nargs="?", help="Path to .txt/.epub/.pdf (or use --config).")
    common.add_argument("--config", help="YAML config file (overridden by any explicit flags).")
    common.add_argument("--output", help="Output directory (default: out/)")
    common.add_argument("--models-dir", dest="models_dir", help="Directory with Kokoro model files.")
    common.add_argument("--voice", help="Default Kokoro voice (default: af_sarah).")
    common.add_argument("--speed", type=float, help="Playback speed multiplier (default: 1.0).")
    common.add_argument("--lang", help="Kokoro language code (default: en-us).")
    common.add_argument("--clean", action="store_true", default=None, help="Enable Ollama-based text cleanup.")
    common.add_argument("--ollama-model", dest="ollama_model", help="Ollama model for cleanup (default: llama3.2).")
    common.add_argument("--ollama-host", dest="ollama_host", help="Ollama host (default: http://localhost:11434).")
    common.add_argument("--workers", type=int, help="Parallel TTS workers (default: 2).")
    common.add_argument("--max-chunk-chars", dest="max_chunk_chars", type=int, help="Max chars per TTS chunk.")
    common.add_argument("--m4b", action="store_true", default=None, help="Also export an M4B audiobook.")

    p_convert = sub.add_parser("convert", parents=[common], help="Convert a book to an audiobook.")
    p_convert.set_defaults(func=cmd_convert)

    p_estimate = sub.add_parser("estimate", parents=[common], help="Preview chapter/word/duration stats.")
    p_estimate.set_defaults(func=cmd_estimate)

    p_voices = sub.add_parser("voices", help="List available Kokoro voices.")
    p_voices.add_argument("--models-dir", dest="models_dir", default="models", help="Directory with Kokoro model files.")
    p_voices.set_defaults(func=cmd_voices)

    p_sample = sub.add_parser(
        "sample", help="Synthesize the same short passage in several voices, to compare before a full run."
    )
    p_sample.add_argument("--models-dir", dest="models_dir", default="models", help="Directory with Kokoro model files.")
    p_sample.add_argument(
        "--voices",
        help="Comma-separated Kokoro voice names to compare (default: 5-voice starter set). "
             "Run `audioforge voices` first to see everything available.",
    )
    p_sample.add_argument("--input", help="Book file to pull a real sample passage from (recommended).")
    p_sample.add_argument("--text", help="Exact custom text to use instead of --input.")
    p_sample.add_argument("--max-chars", dest="max_chars", type=int, default=600,
                           help="Max length of the auto-extracted sample from --input (default: 600).")
    p_sample.add_argument("--speed", type=float, help="Playback speed multiplier (default: 1.0).")
    p_sample.add_argument("--lang", help="Kokoro language code (default: en-us).")
    p_sample.add_argument("--output", default="out/voice_samples", help="Output directory (default: out/voice_samples).")
    p_sample.set_defaults(func=cmd_sample)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)
    try:
        return args.func(args)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        logging.getLogger("audioforge").error(str(exc))
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted — re-run the same command to resume from cache.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
