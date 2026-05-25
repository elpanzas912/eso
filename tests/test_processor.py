import sys
from unittest.mock import MagicMock, patch
from pathlib import Path

sys.modules["whisper"] = MagicMock()
sys.modules["torch"] = MagicMock()

import rapidfuzz  # noqa: E402

rapidfuzz.fuzz.ratio = lambda a, b: 100 if a == b else 0

from processor import find_matching_segments, _cache_key, probe_video  # noqa: E402


def test_find_matching_segments_with_srt_time_range():
    long_words = [
        {"word": "hello", "start": 0.0, "end": 0.5},
        {"word": "world", "start": 0.5, "end": 1.0},
        {"word": "foo", "start": 10.0, "end": 10.5},
        {"word": "bar", "start": 10.5, "end": 11.0},
    ]
    short_words = [
        {"word": "foo", "start": 0.0, "end": 0.5},
        {"word": "bar", "start": 0.5, "end": 1.0},
    ]

    # With srt_time_range that covers the second half, should find the match
    segments = find_matching_segments(
        short_words,
        long_words,
        threshold=50,
        srt_time_range=(10.0, 11.0),
        srt_margin=1.0,
    )
    assert len(segments) == 1
    seg = segments[0]
    # The bug caused long_start to be off by long_start_idx words.
    # With the fix, it should correctly point to ~10.0
    assert seg["start"] >= 9.0
    assert seg["end"] <= 12.0


def test_cache_key_consistency():
    p = Path("/tmp/fake.mp4")
    key1 = _cache_key(p, "base", 0)
    key2 = _cache_key(p, "base", 0)
    assert key1 == key2
    key3 = _cache_key(p, "small", 0)
    assert key1 != key3


def test_probe_video_timeout():
    with patch("processor.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout='{"streams": []}')
        probe_video(Path("/tmp/fake.mp4"))
        _args, kwargs = mock_run.call_args
        assert kwargs.get("timeout") == 60
