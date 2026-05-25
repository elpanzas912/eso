import os
import tempfile
from pathlib import Path

from subtitle_loader import _clean_text, _parse_srt, srt_to_words


def test_clean_text():
    assert _clean_text("Hello, World!") == "hello world"
    assert _clean_text("<b>Test</b>") == "test"
    assert _clean_text("[music] Hello") == "hello"
    assert _clean_text("") == ""


def test_parse_srt():
    content = """1
00:00:01,000 --> 00:00:03,500
Hello world

2
00:00:04,000 --> 00:00:06,000
Second line
"""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".srt", delete=False, encoding="utf-8"
    ) as f:
        f.write(content)
        path = Path(f.name)
    try:
        entries = _parse_srt(path)
        assert len(entries) == 2
        assert entries[0]["text"] == "Hello world"
        assert entries[0]["start"] == 1.0
        assert entries[0]["end"] == 3.5
        assert entries[1]["text"] == "Second line"
        assert entries[1]["start"] == 4.0
        assert entries[1]["end"] == 6.0
    finally:
        os.remove(path)


def test_srt_to_words():
    content = """1
00:00:01,000 --> 00:00:03,000
Hello world

2
00:00:04,000 --> 00:00:06,000
Foo bar baz
"""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".srt", delete=False, encoding="utf-8"
    ) as f:
        f.write(content)
        path = Path(f.name)
    try:
        words = srt_to_words(path)
        assert len(words) == 5
        assert words[0]["word"] == "Hello"
        assert words[0]["start"] == 1.0
        assert words[0]["end"] == 2.0
        assert words[1]["word"] == "world"
        assert words[1]["start"] == 2.0
        assert words[1]["end"] == 3.0
        assert words[2]["word"] == "Foo"
        assert words[2]["start"] == 4.0
        assert words[2]["end"] == 4.667
        assert words[4]["word"] == "baz"
        assert words[4]["end"] == 6.0
    finally:
        os.remove(path)
