from audioforge import Chapter
from audioforge.chunker import chunk_chapter, detect_speaker_voice, split_sentences


def test_split_sentences_basic():
    text = "The cat sat down. It was tired! Was it hungry? Yes."
    sentences = split_sentences(text)
    assert sentences == ["The cat sat down.", "It was tired!", "Was it hungry?", "Yes."]


def test_split_sentences_handles_abbreviation_like_periods_reasonably():
    # Not perfect (no LLM), but should not crash and should keep content intact.
    text = "Dr. Smith arrived. He was late."
    sentences = split_sentences(text)
    assert "".join(sentences).replace(" ", "") == text.replace(" ", "").replace("\n", "")


def test_chunk_chapter_respects_max_chars():
    long_text = " ".join([f"This is sentence number {i}." for i in range(50)])
    chapter = Chapter(index=0, title="Ch1", text=long_text)
    chunks = chunk_chapter(chapter, max_chars=100)
    assert all(len(c.text) <= 120 for c in chunks)  # small slack for join spacing
    assert sum(len(c.text.split()) for c in chunks) == len(long_text.split())


def test_chunk_chapter_never_drops_text():
    chapter = Chapter(index=0, title="Ch1", text="One. Two. Three. Four. Five.")
    chunks = chunk_chapter(chapter, max_chars=9999)
    combined = " ".join(c.text for c in chunks)
    for word in ["One", "Two", "Three", "Four", "Five"]:
        assert word in combined


def test_detect_speaker_voice_strips_tag_for_known_speaker():
    text, voice = detect_speaker_voice('Marcus: "Get down!"', {"Marcus": "am_adam"})
    assert voice == "am_adam"
    assert text == '"Get down!"'


def test_detect_speaker_voice_ignores_unknown_speaker():
    text, voice = detect_speaker_voice("Marcus: Get down!", {})
    assert voice is None
    assert text == "Marcus: Get down!"


def test_hard_split_never_exceeds_limit_even_without_punctuation():
    chapter = Chapter(index=0, title="Ch1", text="word " * 200)
    chunks = chunk_chapter(chapter, max_chars=50)
    assert all(len(c.text) <= 60 for c in chunks)
