from pathlib import Path

from audioforge import pipeline


def test_sample_text_from_input_has_no_double_periods(tmp_path: Path):
    f = tmp_path / "book.txt"
    f.write_text(
        "Chapter 1\n"
        "The storm arrived before dawn. She stood at the edge of the rocks. "
        "Her father had warned her about nights like this.\n",
        encoding="utf-8",
    )
    snippet = pipeline.sample_text_from_input(str(f), max_chars=300)
    assert ".." not in snippet
    assert snippet.startswith("The storm arrived before dawn.")


def test_sample_text_from_input_respects_max_chars_but_keeps_one_full_sentence(tmp_path: Path):
    f = tmp_path / "book.txt"
    f.write_text(
        "Chapter 1\n"
        "This is a moderately long first sentence that goes on for a while. "
        "This is a second sentence.\n",
        encoding="utf-8",
    )
    snippet = pipeline.sample_text_from_input(str(f), max_chars=20)
    # Even though the first sentence alone exceeds max_chars, we keep it whole
    # rather than truncating mid-sentence (bad for TTS comparison) or returning nothing.
    assert snippet.startswith("This is a moderately long first sentence")
    assert "second sentence" not in snippet


def test_sample_text_from_input_stops_once_max_chars_reached(tmp_path: Path):
    f = tmp_path / "book.txt"
    f.write_text(
        "Chapter 1\n"
        "Short one. Short two. Short three. Short four. Short five.\n",
        encoding="utf-8",
    )
    snippet = pipeline.sample_text_from_input(str(f), max_chars=25)
    assert "Short one." in snippet
    assert "Short five." not in snippet
