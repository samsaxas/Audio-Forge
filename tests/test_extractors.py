from pathlib import Path

from audioforge.extractors.txt import extract_txt
from audioforge.extractors.pdf import _clean_pages, _dehyphenate, _find_repeated_lines


def test_extract_txt_splits_on_chapter_headings(tmp_path: Path):
    content = (
        "Chapter 1\n"
        "First chapter body text goes here.\n"
        "\n"
        "Chapter 2\n"
        "Second chapter body text goes here.\n"
    )
    f = tmp_path / "book.txt"
    f.write_text(content, encoding="utf-8")

    chapters = extract_txt(f)
    assert len(chapters) == 2
    assert chapters[0].title == "Chapter 1"
    assert "First chapter" in chapters[0].text
    assert chapters[1].title == "Chapter 2"
    assert "Second chapter" in chapters[1].text


def test_extract_txt_falls_back_to_single_chapter_without_headings(tmp_path: Path):
    f = tmp_path / "plain.txt"
    f.write_text("Just some prose with no headings at all, spanning one paragraph.", encoding="utf-8")
    chapters = extract_txt(f)
    assert len(chapters) == 1
    assert chapters[0].title == "plain"


def test_find_repeated_lines_detects_running_header():
    pages = [
        ["MY BOOK TITLE", "Page one content.", "12"],
        ["MY BOOK TITLE", "Page two content.", "13"],
        ["MY BOOK TITLE", "Page three content.", "14"],
        ["MY BOOK TITLE", "Page four content.", "15"],
    ]
    repeated = _find_repeated_lines(pages, min_fraction=0.5)
    assert "my book title" in repeated


def test_clean_pages_removes_header_and_page_numbers():
    pages = [
        ["HEADER", "Real content one.", "1"],
        ["HEADER", "Real content two.", "2"],
        ["HEADER", "Real content three.", "3"],
    ]
    cleaned = _clean_pages(pages)
    assert "HEADER" not in cleaned
    assert "1" not in cleaned
    assert "Real content one." in cleaned


def test_dehyphenate_joins_split_words():
    lines = ["This is a hyphen-", "ated word that should join."]
    result = _dehyphenate(lines)
    assert result == ["This is a hyphenated word that should join."]
