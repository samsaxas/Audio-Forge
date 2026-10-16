from audioforge.cleaner import regex_clean


def test_regex_clean_normalizes_smart_quotes():
    text = "\u201cHello\u201d \u2014 she said \u2018quietly\u2019."
    cleaned = regex_clean(text)
    assert cleaned == '"Hello" - she said \'quietly\'.'


def test_regex_clean_collapses_whitespace_and_blank_lines():
    text = "Line one.   \n\n\n\nLine  two."
    cleaned = regex_clean(text)
    assert "\n\n\n" not in cleaned
    assert "Line one." in cleaned and "Line two." in cleaned


def test_regex_clean_strips_trailing_line_whitespace():
    text = "  padded line  \nanother  "
    cleaned = regex_clean(text)
    for line in cleaned.splitlines():
        assert line == line.strip()
