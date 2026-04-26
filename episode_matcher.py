"""
episode_matcher.py - Detect which episode a short clip belongs to
Uses SRT subtitles + rapidfuzz for fast matching
"""

import json
from pathlib import Path
from typing import Optional

from rapidfuzz import fuzz

from subtitle_loader import (
    load_series_registry,
    save_series_registry,
    scan_episodes,
    get_episode_subtitle_text,
)
from processor import WORK_DIR


CACHE_DIR = WORK_DIR / "_srt_cache"


def _get_episode_text(srt_path: Optional[str]) -> str:
    if not srt_path:
        return ""
    try:
        return get_episode_subtitle_text(srt_path)
    except Exception:
        return ""


def _fingerprint_short(words: list[dict]) -> str:
    return " ".join(
        w["word"].lower().strip() for w in words if w.get("word", "").strip()
    )


def detect_episode(
    short_words: list[dict],
    series_name: str,
    language: str = "en",
    top_n: int = 3,
    min_score: int = 30,
) -> list[dict]:
    registry = load_series_registry()
    series = None
    for s in registry:
        if s["name"].lower() == series_name.lower():
            series = s
            break

    if not series:
        raise ValueError(f"Serie '{series_name}' no encontrada en el registro")

    episodes = scan_episodes(series, language=language)
    if not episodes:
        raise ValueError(f"No se encontraron episodios para '{series_name}'")

    short_text = _fingerprint_short(short_words)
    if len(short_text) < 20:
        raise ValueError("Transcripción del clip demasiado corta")

    results = []
    short_len = len(short_text)

    for ep in episodes:
        ep_text = _get_episode_text(ep.get("subtitle_path"))
        if not ep_text or len(ep_text) < 50:
            continue

        score = fuzz.token_set_ratio(short_text, ep_text)

        if score >= min_score:
            partial_score = fuzz.partial_ratio(short_text, ep_text)
            combined = (score + partial_score) / 2

            results.append(
                {
                    "id": ep["id"],
                    "season": ep["season"],
                    "episode": ep["episode"],
                    "title": ep["title"],
                    "score": round(combined, 1),
                    "token_score": score,
                    "partial_score": partial_score,
                    "video_path": ep.get("video_path"),
                }
            )

    results.sort(key=lambda x: x["score"], reverse=True)

    for i, r in enumerate(results):
        if i == 0:
            r["confidence"] = (
                "high"
                if r["score"] >= 70
                else ("medium" if r["score"] >= 50 else "low")
            )
        elif i == 1:
            r["confidence"] = "medium" if r["score"] >= 50 else "low"
        else:
            r["confidence"] = "low"

    return results[:top_n]


def add_series(
    name: str, subtitles_path: str, videos_path: str, language: str = "en"
) -> dict:
    registry = load_series_registry()

    for s in registry:
        if s["name"].lower() == name.lower():
            s["subtitles_path"] = subtitles_path
            s["videos_path"] = videos_path
            s["language"] = language
            save_series_registry(registry)
            return s

    series = {
        "name": name,
        "subtitles_path": subtitles_path,
        "videos_path": videos_path,
        "language": language,
    }
    registry.append(series)
    save_series_registry(registry)
    return series


def list_series() -> list[dict]:
    registry = load_series_registry()
    result = []
    for s in registry:
        subs_path = Path(s.get("subtitles_path", ""))
        vids_path = Path(s.get("videos_path", ""))
        ep_count = 0
        if subs_path.exists():
            for season_dir in subs_path.glob("Season *"):
                for ep_dir in season_dir.iterdir():
                    if ep_dir.is_dir():
                        ep_count += 1
        result.append(
            {
                "name": s["name"],
                "language": s.get("language", "en"),
                "subtitles_path": s["subtitles_path"],
                "videos_path": s["videos_path"],
                "episode_count": ep_count,
            }
        )
    return result
