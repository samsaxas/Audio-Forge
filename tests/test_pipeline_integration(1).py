"""
End-to-end tests against the real Kokoro model + ffmpeg. These are opt-in:
they auto-skip unless a `models/` directory with the Kokoro files is
present, so `pytest` works out of the box with zero downloads.

    KOKORO_MODELS_DIR=models pytest tests/ -m integration -v
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

MODELS_DIR = os.environ.get("KOKORO_MODELS_DIR", "models")
_HAS_MODELS = (Path(MODELS_DIR) / "kokoro-v1.0.onnx").exists()
_HAS_FFMPEG = shutil.which("ffmpeg") is not None

pytestmark = pytest.mark.integration


@pytest.mark.skipif(not _HAS_MODELS, reason="Kokoro model files not present; see README to download them.")
def test_full_pipeline_produces_mp3_with_chapters(tmp_path: Path):
    from audioforge import RunConfig
    from audioforge import pipeline

    cfg = RunConfig(
        input_path="examples/sample_book.txt",
        output_dir=str(tmp_path / "out"),
        models_dir=MODELS_DIR,
        workers=2,
    )
    result = pipeline.run(cfg)

    assert Path(result["mp3"]).exists()
    assert result["chapters"] == 2
    assert Path(result["chapters_json"]).exists()


@pytest.mark.skipif(not (_HAS_MODELS and _HAS_FFMPEG), reason="Needs Kokoro models + ffmpeg.")
def test_full_pipeline_can_also_produce_m4b(tmp_path: Path):
    from audioforge import RunConfig
    from audioforge import pipeline

    cfg = RunConfig(
        input_path="examples/sample_book.txt",
        output_dir=str(tmp_path / "out"),
        models_dir=MODELS_DIR,
        workers=2,
        make_m4b=True,
    )
    result = pipeline.run(cfg)
    assert result["m4b"] and Path(result["m4b"]).exists()
