"""
subtitle_loader.py - Parse SRT subtitle files and extract episode text
"""

import re
import json
import hashlib
from pathlib import Path
from typing import Optional

from debug_utils import get_logger, preview


SERIES_REGISTRY = Path(__file__).resolve().parent / "series_registry.json"
CACHE_DIR = Path(__file__).resolve().parent / "work" / "_srt_cache"
logger = get_logger("subtitle_loader")


def _clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\{[^}]+\}", "", text)
    text = re.sub(r"[\(\[][\s\S]*?[\)\]]", "", text)
    import string

    text = text.translate(str.maketrans("", "", string.punctuation))
    text = " ".join(text.split())
    return text.strip()


def _parse_srt(srt_path: Path) -> list[dict]:
    logger.info("Parseando SRT | path=%s", srt_path)
    with open(srt_path, "r", encoding="utf-8-sig") as f:
        content = f.read()

    blocks = re.split(r"\n\s*\n", content.strip())
    entries = []

    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) < 3:
            continue

        time_match = re.match(
            r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})[,.](\d{3})",
            lines[1].strip(),
        )
        if not time_match:
            continue

        g = time_match.groups()
        start = int(g[0]) * 3600 + int(g[1]) * 60 + int(g[2]) + int(g[3]) / 1000
        end = int(g[4]) * 3600 + int(g[5]) * 60 + int(g[6]) + int(g[7]) / 1000
        text = " ".join(lines[2:]).strip()

        if text:
            entries.append(
                {
                    "start": round(start, 3),
                    "end": round(end, 3),
                    "text": text,
                }
            )

    logger.info(
        "SRT parseado | path=%s | blocks=%s | entries=%s",
        srt_path,
        len(blocks),
        len(entries),
    )
    return entries


def _srt_cache_key(srt_path: Path) -> str:
    key_str = str(srt_path.resolve())
    if srt_path.exists():
        stat = srt_path.stat()
        key_str += f"|{stat.st_size}|{stat.st_mtime}"
    return hashlib.md5(key_str.encode()).hexdigest()[:16]


def load_subtitles(srt_path: str) -> list[dict]:
    srt_path = Path(srt_path)
    if not srt_path.exists():
        logger.warning("Subtitulo inexistente | path=%s", srt_path)
        return []

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_key = _srt_cache_key(srt_path)
    cache_file = CACHE_DIR / f"{cache_key}.json"

    if cache_file.exists():
        logger.info(
            "Subtitulos cache hit | srt=%s | cache=%s",
            srt_path,
            cache_file,
        )
        with open(cache_file, "r", encoding="utf-8") as f:
            entries = json.load(f)
        logger.info(
            "Subtitulos desde cache | srt=%s | entries=%s",
            srt_path,
            len(entries),
        )
        return entries

    entries = _parse_srt(srt_path)

    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False)
    logger.info(
        "Subtitulos cache miss -> guardados | srt=%s | cache=%s | entries=%s",
        srt_path,
        cache_file,
        len(entries),
    )

    return entries


def get_episode_subtitle_text(srt_path: str) -> str:
    entries = load_subtitles(srt_path)
    text = " ".join(_clean_text(e["text"]) for e in entries if _clean_text(e["text"]))
    logger.info(
        "Texto de subtitulo consolidado | srt=%s | entries=%s | chars=%s",
        srt_path,
        len(entries),
        len(text),
    )
    return text


def load_series_registry() -> list[dict]:
    if SERIES_REGISTRY.exists():
        with open(SERIES_REGISTRY, "r", encoding="utf-8") as f:
            registry = json.load(f)
        logger.info(
            "Registro de series cargado | path=%s | count=%s | preview=%s",
            SERIES_REGISTRY,
            len(registry),
            preview(registry, 800),
        )
        return registry
    logger.warning("Registro de series no existe | path=%s", SERIES_REGISTRY)
    return []


def save_series_registry(registry: list[dict]):
    SERIES_REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    with open(SERIES_REGISTRY, "w", encoding="utf-8") as f:
        json.dump(registry, f, ensure_ascii=False, indent=2)
    logger.info(
        "Registro de series guardado | path=%s | count=%s | preview=%s",
        SERIES_REGISTRY,
        len(registry),
        preview(registry, 800),
    )


def find_subtitle_for_episode(
    series: dict, season: int, episode: int, language: str = "en"
) -> Optional[str]:
    subs_path = Path(series.get("subtitles_path", ""))
    if not subs_path.exists():
        logger.warning(
            "Ruta de subtitulos inexistente | serie=%s | path=%s",
            series.get("name"),
            subs_path,
        )
        return None

    season_dirs = list(subs_path.glob(f"Season {season:02d}"))
    if not season_dirs:
        season_dirs = list(subs_path.glob(f"*{season:02d}*"))
    if not season_dirs:
        logger.warning(
            "No se encontro carpeta de temporada | serie=%s | season=%s | language=%s",
            series.get("name"),
            season,
            language,
        )
        return None

    season_dir = season_dirs[0]

    pattern = f"*S{season:02d}E{episode:02d}*"
    episode_dirs = list(season_dir.glob(pattern))

    if not episode_dirs:
        logger.warning(
            "No se encontro carpeta de episodio | serie=%s | season=%s | episode=%s | pattern=%s",
            series.get("name"),
            season,
            episode,
            pattern,
        )
        return None

    episode_dir = episode_dirs[0]
    if not episode_dir.is_dir():
        episode_dir = episode_dir.parent

    lang_patterns = [f".{language}.srt", f".{language}.*.srt"]
    if "-" not in language:
        lang_patterns.append(f".{language}-{language}.srt")

    for lp in lang_patterns:
        matches = list(episode_dir.glob(f"*{lp}"))
        if matches:
            logger.info(
                "Subtitulo encontrado por patron de idioma | serie=%s | episode=%s | match=%s",
                series.get("name"),
                f"S{season:02d}E{episode:02d}",
                matches[0],
            )
            return str(matches[0])

    all_srts = list(episode_dir.glob("*.srt"))
    for srt in all_srts:
        if f".{language}" in srt.name:
            logger.info(
                "Subtitulo encontrado por fallback parcial | serie=%s | episode=%s | match=%s",
                series.get("name"),
                f"S{season:02d}E{episode:02d}",
                srt,
            )
            return str(srt)

    if all_srts:
        logger.warning(
            "Subtitulo encontrado solo por fallback final | serie=%s | episode=%s | match=%s",
            series.get("name"),
            f"S{season:02d}E{episode:02d}",
            all_srts[0],
        )
        return str(all_srts[0])

    logger.warning(
        "No se encontro subtitulo para episodio | serie=%s | episode=%s | language=%s",
        series.get("name"),
        f"S{season:02d}E{episode:02d}",
        language,
    )
    return None


def scan_episodes(series: dict, language: str = "en") -> list[dict]:
    subs_path = Path(series.get("subtitles_path", ""))
    vids_path = Path(series.get("videos_path", ""))

    if not subs_path.exists():
        logger.warning(
            "No se pueden escanear episodios; subtitulos inexistentes | serie=%s | path=%s",
            series.get("name"),
            subs_path,
        )
        return []

    episodes = []
    season_dirs = sorted(subs_path.glob("Season *"))
    logger.info(
        "Escaneando episodios | serie=%s | language=%s | subs_path=%s | videos_path=%s | season_dirs=%s",
        series.get("name"),
        language,
        subs_path,
        vids_path,
        len(season_dirs),
    )

    for season_dir in season_dirs:
        season_match = re.search(r"Season\s+(\d+)", season_dir.name)
        if not season_match:
            logger.warning("Carpeta de temporada ignorada | path=%s", season_dir)
            continue
        season_num = int(season_match.group(1))

        for ep_dir in sorted(season_dir.iterdir()):
            if not ep_dir.is_dir():
                continue

            ep_match = re.search(r"S(\d{2})E(\d{2})", ep_dir.name)
            if not ep_match:
                logger.warning("Carpeta de episodio ignorada | path=%s", ep_dir)
                continue

            ep_num = int(ep_match.group(2))
            episode_id = f"S{season_num:02d}E{ep_num:02d}"

            srt_path = find_subtitle_for_episode(series, season_num, ep_num, language)

            video_path = None
            if vids_path.exists():
                vid_season = vids_path / f"Season {season_num:02d}"
                if not vid_season.exists():
                    vid_season = vids_path / season_dir.name
                if vid_season.exists():
                    vid_pattern = f"*S{season_num:02d}E{ep_num:02d}*"
                    vid_matches = list(vid_season.glob(vid_pattern))
                    vid_matches = [
                        v
                        for v in vid_matches
                        if v.suffix.lower() in (".mkv", ".mp4", ".avi")
                    ]
                    if vid_matches:
                        video_path = str(vid_matches[0])

            episodes.append(
                {
                    "id": episode_id,
                    "season": season_num,
                    "episode": ep_num,
                    "title": ep_dir.name.split("E" + f"{ep_num:02d}" + " ", 1)[-1]
                    if " " in ep_dir.name
                    else ep_dir.name,
                    "subtitle_path": srt_path,
                    "video_path": video_path,
                }
            )
            logger.info(
                "Episodio detectado | serie=%s | episode=%s | subtitle=%s | video=%s",
                series.get("name"),
                episode_id,
                srt_path,
                video_path,
            )

    logger.info(
        "Escaneo de episodios completado | serie=%s | total=%s",
        series.get("name"),
        len(episodes),
    )
    return episodes
