# AudioForge 🔊📖
### A local-first Text → Audiobook converter powered by **Kokoro TTS** + **Ollama**

AudioForge turns a `.txt`, `.epub`, or `.pdf` into a fully chaptered MP3 (or M4B)
audiobook — entirely on your own machine. No cloud APIs, no per-character billing,
no data leaving your laptop.

```
book.epub ──▶ extract ──▶ clean (Ollama, optional) ──▶ chunk ──▶ Kokoro TTS ──▶ stitch ──▶ audiobook.mp3
```

## Why this exists

Most "text to audiobook" scripts you find online are single-file hacks that
re-synthesize the whole book every time you run them, don't handle chapters,
and fall over on a 400-page PDF. AudioForge is built like production software:

- **Resumable** — every synthesized sentence chunk is cached on disk by content
  hash. Kill the process at 80% and re-run — it picks up exactly where it left off.
  You will never pay the synthesis cost for the same text twice.
- **Real chapters** — output is a single MP3 with embedded **ID3 chapter markers**
  (`CHAP`/`CTOC` frames), so apps like Smart AudioBook Player, VLC, or Plex show a
  proper chapter list. An M4B (the real audiobook container format) is optionally
  produced too.
- **Multi-format ingestion** — plain text, EPUB (via `ebooklib`), and PDF (via
  `pdfplumber`) all funnel through the same pipeline, with format-specific chapter
  detection.
- **Local LLM cleanup (Ollama, optional)** — PDFs are messy: hyphenated line
  breaks, running headers, page numbers, OCR noise. An optional pass through a
  local Ollama model cleans this up and can even detect chapter boundaries in
  unstructured `.txt` files. Nothing is sent anywhere; it's a call to
  `http://localhost:11434`.
- **Multi-voice narration** — assign different Kokoro voices per chapter, or
  auto-detect `Name: "dialogue"` patterns and narrate them in different voices.
- **Parallel synthesis** — a worker pool drives multiple Kokoro inferences
  concurrently (CPU-bound, so this matters a lot on multi-core machines).
- **Dry-run estimator** — `audioforge estimate book.epub` tells you the word
  count, chunk count, and expected audio duration before you spend any compute.

## Requirements

| Dependency | Why | Install |
|---|---|---|
| Python 3.10+ | runtime | — |
| [Ollama](https://ollama.com) | optional text cleanup / chapter detection | `curl -fsSL https://ollama.com/install.sh \| sh` then `ollama pull llama3.2` |
| `ffmpeg` | audio muxing, mp3/m4b export | `apt install ffmpeg` / `brew install ffmpeg` |
| `espeak-ng` | Kokoro's phonemizer backend | `apt install espeak-ng` / `brew install espeak` |
| Kokoro ONNX model + voices file | the TTS engine itself | see below |

### Getting the Kokoro model

AudioForge uses [`kokoro-onnx`](https://github.com/thewh1teagle/kokoro-onnx), a
lightweight ONNX runtime build of Kokoro (no PyTorch/GPU required, runs fine on
CPU). Download the two model files it needs once:

```bash
mkdir -p models
curl -L -o models/kokoro-v1.0.onnx \
  https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
curl -L -o models/voices-v1.0.bin \
  https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
```

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Quick start

```bash
# 1. See what you're about to synthesize (no synthesis happens)
audioforge estimate mybook.epub

# 2. List available Kokoro voices
audioforge voices --models-dir models/

# 3. Convert, with local LLM cleanup for messy PDFs
audioforge convert mybook.pdf \
    --models-dir models/ \
    --voice af_sarah \
    --speed 1.05 \
    --clean \
    --ollama-model llama3.2 \
    --workers 4 \
    --output out/mybook

# Interrupted? Just run the exact same command again — cached chunks are skipped.
```

Output:
```
out/mybook/
├── mybook.mp3        # final chaptered audiobook (ID3 CHAP/CTOC markers)
├── mybook.m4b         # optional, real audiobook container (--m4b)
├── mybook.chapters.json
└── .cache/            # per-chunk wav cache, safe to delete after a successful run
```

## Configuration file (multi-voice narration)

Instead of flags, you can drive a run from a YAML file — this is where
per-chapter or per-speaker voices go:

```yaml
input: mybook.epub
output: out/mybook
models_dir: models/
default_voice: af_sarah
speed: 1.0
clean: true
ollama_model: llama3.2
chapter_voices:
  "Chapter 12": am_adam        # a narrator switch for one chapter
speaker_voices:                 # auto-detected "Name: ..." dialogue lines
  "Marcus": am_adam
  "Elena": af_bella
silence_between_chunks_ms: 220
silence_between_chapters_ms: 900
```

```bash
audioforge convert --config myconfig.yaml
```

## Architecture

```
src/audioforge/
├── extractors/          # txt.py, epub.py, pdf.py -> List[Chapter(title, text)]
├── cleaner.py           # regex cleanup + optional Ollama pass
├── chunker.py           # sentence-safe splitting to Kokoro's ~ token/char limit
├── tts_engine.py        # Kokoro wrapper, content-hash caching, retries
├── stitcher.py          # wav concat, silence gaps, mp3/m4b export, ID3 chapters
├── pipeline.py          # orchestrates the above stages end to end
└── cli.py               # `audioforge` command (convert / estimate / voices)
```

Each stage is a plain, independently testable Python module — no framework
magic. `pipeline.py` is ~150 lines and reads top-to-bottom like a recipe.

## Testing

```bash
pytest tests/ -v
```

Chunker and cleaner tests run with no external dependencies (no Kokoro/Ollama
needed). TTS/stitcher tests are marked `@pytest.mark.integration` and only run
if a model directory and `ffmpeg` are present.

## FAQ

**Does this need internet access?** No, once Ollama and the Kokoro model files
are on disk, the whole pipeline runs offline.

**Why MP3 with ID3 chapters instead of M4B by default?** MP3 has universal
playback support; M4B is the "correct" audiobook format (variable bitrate,
native chapter support, bookmark position) but not every player handles it.
Pass `--m4b` to get both.

**Can I use a different local LLM?** Yes — anything served by Ollama.
`--ollama-model mistral`, `--ollama-model qwen2.5:14b`, etc.

**What if I don't have/want Ollama?** Omit `--clean`. Regex-based cleanup
(dehyphenation, whitespace, page-number stripping) still runs — it's the LLM
pass that's optional.
