"""
debug_utils.py - Shared logging helpers for ClipFinder
"""

from __future__ import annotations

import json
import logging
from typing import Any


_LOGGER_ROOT = "clipfinder"
_DEFAULT_LEVEL = logging.INFO


class SafeStreamHandler(logging.StreamHandler):
    def handleError(self, record):
        pass


def configure_logging() -> logging.Logger:
    logger = logging.getLogger(_LOGGER_ROOT)
    if logger.handlers:
        return logger

    handler = SafeStreamHandler()
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(handler)
    logger.setLevel(_DEFAULT_LEVEL)
    logger.propagate = False
    return logger


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    if not name:
        return logging.getLogger(_LOGGER_ROOT)
    if name.startswith(_LOGGER_ROOT):
        return logging.getLogger(name)
    return logging.getLogger(f"{_LOGGER_ROOT}.{name}")


def preview(value: Any, limit: int = 280) -> str:
    try:
        rendered = json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        rendered = repr(value)
    if len(rendered) <= limit:
        return rendered
    return f"{rendered[: limit - 3]}..."


def summarize_words(words: list[dict], sample: int = 5) -> dict:
    if not words:
        return {"count": 0}
    head = [w.get("word", "") for w in words[:sample]]
    tail = [w.get("word", "") for w in words[-sample:]]
    return {
        "count": len(words),
        "start": words[0].get("start"),
        "end": words[-1].get("end"),
        "head": head,
        "tail": tail,
    }


def summarize_segments(segments: list[dict], sample: int = 3) -> dict:
    if not segments:
        return {"count": 0}
    compact = []
    for seg in segments[:sample]:
        compact.append(
            {
                "start": seg.get("start"),
                "end": seg.get("end"),
                "short_start": seg.get("short_start"),
                "short_end": seg.get("short_end"),
                "confidence": seg.get("confidence"),
                "text": str(seg.get("text", ""))[:80],
            }
        )
    return {"count": len(segments), "sample": compact}
