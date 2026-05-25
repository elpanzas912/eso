import json
import re
import shutil
from pathlib import Path

SRC = Path("E:/Blue Bloods/Transcription")
REGISTRY = Path("E:/Gemini/eso/series_registry.json")


def clean_title(title: str) -> str:
    title = title.strip()
    title = title.rstrip(".")
    return title


def main():
    episodes = {}

    for srt_file in SRC.rglob("*.srt"):
        rel = srt_file.relative_to(SRC)
        m = re.search(
            r"(\d+)x(\d+)\s*-\s*(.+?)(?:\.(?:HDTV|720p|1080p|WEB|WEB-DL|BluRay)\b)",
            srt_file.name,
            re.IGNORECASE,
        )
        if not m:
            print(f"SKIP (no match): {rel}")
            continue

        season = int(m.group(1))
        episode = int(m.group(2))
        title = clean_title(m.group(3))

        key = (season, episode)
        if key not in episodes:
            episodes[key] = []
        episodes[key].append((srt_file, title))

    print(f"Found {len(episodes)} unique episodes")

    for (season, episode), items in sorted(episodes.items()):
        items.sort(key=lambda x: x[0].stat().st_size, reverse=True)
        src_file, title = items[0]

        season_str = f"Season {season:02d}"
        ep_id = f"S{season:02d}E{episode:02d}"
        folder_name = f"Blue Bloods {ep_id} {title}"

        dst_dir = SRC / season_str / folder_name
        dst_dir.mkdir(parents=True, exist_ok=True)

        dst_file = dst_dir / f"Blue Bloods {ep_id} {title}.en.srt"

        print(f"{ep_id}: {src_file.name} -> {dst_file}")
        shutil.copy2(src_file, dst_file)

    registry = []
    if REGISTRY.exists():
        with open(REGISTRY, "r", encoding="utf-8") as f:
            registry = json.load(f)

    registry = [s for s in registry if s["name"].lower() != "blue bloods"]
    registry.append(
        {
            "name": "Blue Bloods",
            "subtitles_path": str(SRC),
            "videos_path": "G:\\Blue Bloods",
            "language": "en",
        }
    )

    with open(REGISTRY, "w", encoding="utf-8") as f:
        json.dump(registry, f, ensure_ascii=False, indent=2)

    print("Done! Registry updated.")


if __name__ == "__main__":
    main()
