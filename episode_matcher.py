"""
episode_matcher.py - Detect which episode a short clip belongs to
Uses SRT subtitles + rapidfuzz for fast matching
"""

import json
from pathlib import Path
from typing import Optional

from rapidfuzz import fuzz

from debug_utils import get_logger, preview
from subtitle_loader import (
    load_series_registry,
    save_series_registry,
    scan_episodes,
    get_episode_subtitle_text,
)
from processor import WORK_DIR


CACHE_DIR = WORK_DIR / "_srt_cache"
logger = get_logger("episode_matcher")


def _get_episode_text(srt_path: Optional[str]) -> str:
    if not srt_path:
        logger.warning("Episodio sin ruta de subtitulo")
        return ""
    try:
        text = get_episode_subtitle_text(srt_path)
        logger.info(
            "Texto de episodio cargado | srt=%s | chars=%s",
            srt_path,
            len(text),
        )
        return text
    except Exception as exc:
        logger.exception(
            "Error cargando texto de episodio | srt=%s | error=%s",
            srt_path,
            exc,
        )
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
    logger.info(
        "Deteccion de episodio iniciada | serie=%s | language=%s | top_n=%s | min_score=%s | registry_count=%s",
        series_name,
        language,
        top_n,
        min_score,
        len(registry),
    )
    series = None
    for s in registry:
        if s["name"].lower() == series_name.lower():
            series = s
            break

    if not series:
        logger.error("Serie no encontrada | serie=%s", series_name)
        raise ValueError(f"Serie '{series_name}' no encontrada en el registro")

    episodes = scan_episodes(series, language=language)
    if not episodes:
        logger.error("No se encontraron episodios | serie=%s", series_name)
        raise ValueError(f"No se encontraron episodios para '{series_name}'")

    short_text = _fingerprint_short(short_words)
    if len(short_text) < 20:
        logger.error(
            "Transcripcion de clip demasiado corta | serie=%s | chars=%s | preview=%s",
            series_name,
            len(short_text),
            short_text,
        )
        raise ValueError("Transcripción del clip demasiado corta")

    results = []
    short_len = len(short_text)

    logger.info(
        "Comparando clip contra episodios | serie=%s | episodes=%s | short_chars=%s | preview=%s",
        series_name,
        len(episodes),
        short_len,
        short_text[:180],
    )

    for i, ep in enumerate(episodes):
        ep_text = _get_episode_text(ep.get("subtitle_path"))
        if not ep_text or len(ep_text) < 50:
            logger.warning(
                "Episodio descartado por subtitulos insuficientes | episode=%s | subtitle=%s | chars=%s",
                ep.get("id"),
                ep.get("subtitle_path"),
                len(ep_text),
            )
            continue

        if i > 0 and i % 50 == 0:
            logger.info(
                "Progreso deteccion | comparados=%s/%s | serie=%s",
                i,
                len(episodes),
                series_name,
            )

        score = fuzz.token_set_ratio(short_text, ep_text)
        logger.info(
            "Score token_set_ratio | episode=%s | score=%s | subtitle=%s",
            ep.get("id"),
            score,
            ep.get("subtitle_path"),
        )

        if score >= min_score:
            partial_score = fuzz.partial_ratio(short_text, ep_text)
            combined = (score + partial_score) / 2
            logger.info(
                "Episodio candidato | episode=%s | token_score=%s | partial_score=%s | combined=%s",
                ep.get("id"),
                score,
                partial_score,
                round(combined, 1),
            )

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
        else:
            logger.info(
                "Episodio descartado por score | episode=%s | score=%s | threshold=%s",
                ep.get("id"),
                score,
                min_score,
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

    logger.info(
        "Deteccion de episodio finalizada | serie=%s | resultados=%s",
        series_name,
        preview(results[:top_n], 1200),
    )
    return results[:top_n]


def add_series(
    name: str, subtitles_path: str, videos_path: str, language: str = "en"
) -> dict:
    registry = load_series_registry()
    logger.info(
        "Agregar/actualizar serie | name=%s | subtitles=%s | videos=%s | language=%s",
        name,
        subtitles_path,
        videos_path,
        language,
    )

    for s in registry:
        if s["name"].lower() == name.lower():
            s["subtitles_path"] = subtitles_path
            s["videos_path"] = videos_path
            s["language"] = language
            save_series_registry(registry)
            logger.info("Serie actualizada | payload=%s", preview(s, 400))
            return s

    series = {
        "name": name,
        "subtitles_path": subtitles_path,
        "videos_path": videos_path,
        "language": language,
    }
    registry.append(series)
    save_series_registry(registry)
    logger.info("Serie agregada | payload=%s", preview(series, 400))
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
    logger.info("Series listadas | count=%s | payload=%s", len(result), preview(result, 1000))
    return result
