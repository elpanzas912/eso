"""
processor.py - Core logic for ClipFinder
Handles: yt-dlp download, Whisper transcription, fuzzy segment matching
"""

import subprocess
import json
import os
import shutil
import hashlib
import mimetypes
from pathlib import Path

try:
    import torch

    _CUDA_AVAILABLE = torch.cuda.is_available()
    _GPU_NAME = torch.cuda.get_device_name(0) if _CUDA_AVAILABLE else None
except ImportError:
    torch = None
    _CUDA_AVAILABLE = False
    _GPU_NAME = None

if _CUDA_AVAILABLE:
    print(f"[ClipFinder] GPU detectada: {_GPU_NAME}")
    print(
        f"[ClipFinder] VRAM: {torch.cuda.get_device_properties(0).total_memory // (1024**3)} GB"
    )
else:
    print(
        "[ClipFinder] GPU no detectada — usando CPU (instala PyTorch con CUDA para acelerar)"
    )

PROJECT_ROOT = Path(__file__).resolve().parent
WORK_DIR = PROJECT_ROOT / "work"
MODELS_DIR = WORK_DIR / "_whisper_models"
YT_CACHE_DIR = WORK_DIR / "_yt_cache"


def _find_ffmpeg_dir():
    candidates = [
        os.path.join(
            os.path.expandvars(r"%LOCALAPPDATA%"),
            "Microsoft",
            "WinGet",
            "Packages",
            "Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe",
            "ffmpeg-7.1-full_build",
            "bin",
        ),
    ]
    for c in candidates:
        ffmpeg_path = os.path.join(c, "ffmpeg.exe")
        if os.path.isfile(ffmpeg_path):
            return c
    return None


_FFMPEG_DIR = _find_ffmpeg_dir()
if _FFMPEG_DIR:
    os.environ["PATH"] = _FFMPEG_DIR + os.pathsep + os.environ.get("PATH", "")


def _find_ffmpeg():
    if _FFMPEG_DIR:
        return os.path.join(_FFMPEG_DIR, "ffmpeg.exe")
    return shutil.which("ffmpeg") or "ffmpeg"


def _find_ffprobe():
    if _FFMPEG_DIR:
        return os.path.join(_FFMPEG_DIR, "ffprobe.exe")
    return shutil.which("ffprobe") or "ffprobe"


FFMPEG = _find_ffmpeg()
FFPROBE = _find_ffprobe()


def probe_video(video_path: Path) -> dict:
    cmd = [
        FFPROBE,
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_streams",
        "-show_format",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe error: {result.stderr}")

    info = json.loads(result.stdout)

    audio_streams = []
    for s in info.get("streams", []):
        if s.get("codec_type") == "audio":
            audio_streams.append(
                {
                    "index": s.get("index", 0),
                    "language": s.get("tags", {}).get("language", "und"),
                    "title": s.get("tags", {}).get("title", ""),
                    "codec": s.get("codec_name", ""),
                    "channels": s.get("channels", 0),
                }
            )

    english_idx = 0
    for i, s in enumerate(audio_streams):
        if s["language"] in ("eng", "en"):
            english_idx = i
            break

    lines = []
    for i, s in enumerate(audio_streams):
        title_part = f" · {s['title']}" if s["title"] else ""
        lines.append(
            f"  Pista #{i}: {s['codec']} · {s['language']} · {s['channels']}ch{title_part}"
        )

    return {
        "streams": audio_streams,
        "english_index": english_idx,
        "description": "\n".join(lines)
        if lines
        else "No se encontraron pistas de audio",
    }


def extract_audio_track(video_path: Path, audio_index: int, output_path: Path) -> Path:
    cmd = [
        FFMPEG,
        "-y",
        "-i",
        str(video_path),
        "-map",
        f"0:a:{audio_index}",
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg audio extraction error: {result.stderr}")
    return output_path


def remux_for_browser(
    video_path: Path, output_path: Path, audio_index: int = 0
) -> Path:
    print(
        f"[ClipFinder] Remuxing {video_path.name} -> {output_path.name} (audio track #{audio_index})"
    )

    # Try 1: copy everything (instant if audio is already AAC/MP3 compatible)
    cmd_copy = [
        FFMPEG,
        "-y",
        "-i",
        str(video_path),
        "-map",
        "0:v:0",
        "-map",
        f"0:a:{audio_index}",
        "-c:v",
        "copy",
        "-c:a",
        "copy",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    print("[ClipFinder] Try 1: copy (instant)...")
    result = subprocess.run(cmd_copy, capture_output=True, text=True)
    if result.returncode == 0:
        print("[ClipFinder] Remux copy succeeded")
        return output_path

    # Try 2: copy video, transcode audio to AAC (only audio re-encoded)
    cmd_aac = [
        FFMPEG,
        "-y",
        "-i",
        str(video_path),
        "-map",
        "0:v:0",
        "-map",
        f"0:a:{audio_index}",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    print("[ClipFinder] Try 2: copy video + AAC audio...")
    result = subprocess.run(cmd_aac, capture_output=True, text=True)
    if result.returncode == 0:
        print("[ClipFinder] Remux AAC succeeded")
        return output_path

    # Try 3: full re-encode (slow, last resort)
    if output_path.exists():
        os.remove(output_path)
    cmd_reencode = [
        FFMPEG,
        "-y",
        "-i",
        str(video_path),
        "-map",
        "0:v:0",
        "-map",
        f"0:a:{audio_index}",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    print("[ClipFinder] Try 3: full re-encode (slow)...")
    result = subprocess.run(cmd_reencode, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg remux error: {result.stderr}")
    print("[ClipFinder] Remux re-encode succeeded")
    return output_path


def download_video(url: str, output_path: Path, progress_cb=None) -> Path:
    """Download video using yt-dlp. Returns actual path (yt-dlp may change extension)."""
    import yt_dlp

    class ProgressHook:
        def __call__(self, d):
            if progress_cb and d["status"] == "downloading":
                pct = d.get("_percent_str", "?")
                progress_cb(f"Descargando... {pct}")

    ydl_opts = {
        "outtmpl": str(output_path.with_suffix(".%(ext)s")),
        "format": "bv*+ba/b",
        "merge_output_format": "mp4",
        "progress_hooks": [ProgressHook()],
        "quiet": True,
        "no_warnings": True,
        "cachedir": str(YT_CACHE_DIR),
    }

    ffmpeg_dir = os.path.dirname(FFMPEG)
    if ffmpeg_dir:
        ydl_opts["ffmpeg_location"] = ffmpeg_dir

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        ext = info.get("ext", "mp4")

    actual = output_path.with_suffix("." + ext)
    if not actual.exists():
        actual = output_path.with_suffix(".mp4")
    if not actual.exists():
        candidates = list(output_path.parent.glob(output_path.stem + ".*"))
        if candidates:
            actual = candidates[0]
    return actual


def _cache_key(video_path: Path, model_size: str, audio_index: int) -> str:
    key_str = str(video_path.resolve()) + f"|{model_size}|{audio_index}"
    if video_path.exists():
        stat = video_path.stat()
        key_str += f"|{stat.st_size}|{stat.st_mtime}"
    return hashlib.md5(key_str.encode()).hexdigest()[:16]


def transcribe_video(
    video_path: Path, model_size: str = "base", progress_cb=None, audio_index: int = 0
) -> list[dict]:
    """
    Transcribe video using OpenAI Whisper with word-level timestamps.
    Returns list of: {word, start, end}
    Caches results to avoid re-transcribing on repeated runs.
    """
    cache_key = _cache_key(video_path, model_size, audio_index)
    cache_dir = WORK_DIR / "_cache"
    cache_file = cache_dir / f"{cache_key}.json"

    if cache_file.exists():
        if progress_cb:
            progress_cb("Usando transcripcion cacheada...")
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)

    import whisper

    if progress_cb:
        progress_cb(
            f"Cargando modelo Whisper ({model_size}) en {_GPU_NAME or 'CPU'}..."
        )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if _CUDA_AVAILABLE else "cpu"
    model = whisper.load_model(model_size, download_root=str(MODELS_DIR), device=device)

    actual_path = video_path
    if audio_index > 0:
        if progress_cb:
            progress_cb(f"Extrayendo pista de audio #{audio_index}...")
        audio_extract_path = video_path.parent / f"_audio_track_{audio_index}.wav"
        actual_path = extract_audio_track(video_path, audio_index, audio_extract_path)

    if progress_cb:
        progress_cb(f"Transcribiendo {Path(actual_path).name}...")

    fp16 = device == "cuda"

    words = []
    try:
        result = model.transcribe(
            str(actual_path),
            word_timestamps=True,
            verbose=False,
            language=None,
            fp16=fp16,
        )
        for segment in result.get("segments", []):
            for word_info in segment.get("words", []):
                word = word_info.get("word", "").strip()
                if word:
                    words.append(
                        {
                            "word": word,
                            "start": round(word_info["start"], 3),
                            "end": round(word_info["end"], 3),
                        }
                    )
    except TypeError:
        print(
            "[ClipFinder] word_timestamps failed, falling back to segment-level timestamps"
        )
        result = model.transcribe(
            str(actual_path),
            word_timestamps=False,
            verbose=False,
            language=None,
            fp16=fp16,
        )
        for segment in result.get("segments", []):
            seg_start = segment.get("start", 0)
            seg_end = segment.get("end", 0)
            seg_text = segment.get("text", "").strip()
            if not seg_text or seg_end <= seg_start:
                continue
            seg_words = seg_text.split()
            duration = seg_end - seg_start
            word_dur = duration / len(seg_words) if seg_words else 0
            for i, w in enumerate(seg_words):
                words.append(
                    {
                        "word": w,
                        "start": round(seg_start + i * word_dur, 3),
                        "end": round(seg_start + (i + 1) * word_dur, 3),
                    }
                )

    if not words:
        raise RuntimeError(f"No se pudo transcribir {Path(actual_path).name}")

    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(words, f, ensure_ascii=False)

    return words


def find_matching_segments(
    short_words: list[dict],
    long_words: list[dict],
    window: int = 8,
    threshold: int = 72,
    merge_gap: float = 3.0,
) -> list[dict]:
    """
    Find where segments of the short clip appear in the long video.

    Algorithm:
    1. Slide a window across the short clip words
    2. For each window, search the long video for the best fuzzy text match
    3. Extend matched region word-by-word while similarity holds
    4. Merge overlapping/close segments

    Returns list of: {start, end, text, short_start, short_end, confidence}
    """
    from rapidfuzz import fuzz

    def words_text(words_slice):
        return " ".join(w["word"].lower() for w in words_slice)

    n_short = len(short_words)
    n_long = len(long_words)
    segments = []
    i = 0

    while i < n_short:
        chunk_end = min(i + window, n_short)
        chunk = short_words[i:chunk_end]
        chunk_text = words_text(chunk)

        best_score = 0
        best_j = -1

        # Search long video for best match of this chunk
        for j in range(n_long - len(chunk) + 1):
            long_chunk = long_words[j : j + len(chunk)]
            score = fuzz.ratio(chunk_text, words_text(long_chunk))
            if score > best_score:
                best_score = score
                best_j = j

        if best_score >= threshold and best_j >= 0:
            # Extend match forward as long as words keep matching
            match_len = len(chunk)
            while i + match_len < n_short and best_j + match_len < n_long:
                sw = short_words[i + match_len]["word"].lower()
                lw = long_words[best_j + match_len]["word"].lower()
                if fuzz.ratio(sw, lw) >= 60:
                    match_len += 1
                else:
                    break

            seg_short_start = short_words[i]["start"]
            seg_short_end = short_words[i + match_len - 1]["end"]
            seg_long_start = long_words[best_j]["start"]
            seg_long_end = long_words[best_j + match_len - 1]["end"]
            seg_text = " ".join(w["word"] for w in short_words[i : i + match_len])

            segments.append(
                {
                    "start": seg_long_start,
                    "end": seg_long_end,
                    "text": seg_text,
                    "short_start": seg_short_start,
                    "short_end": seg_short_end,
                    "confidence": round(best_score),
                }
            )
            i += match_len
        else:
            i += 1

    # Merge segments that are too close together
    if not segments:
        return []

    merged = [segments[0].copy()]
    for seg in segments[1:]:
        prev = merged[-1]
        if seg["start"] - prev["end"] <= merge_gap:
            prev["end"] = max(prev["end"], seg["end"])
            prev["short_end"] = max(prev["short_end"], seg["short_end"])
            prev["text"] += " " + seg["text"]
            prev["confidence"] = (prev["confidence"] + seg["confidence"]) // 2
        else:
            merged.append(seg.copy())

    return merged


def export_clip(
    long_video_path: str,
    segments: list[dict],
    output_path: str,
    pad_before: float = 0,
    pad_after: float = 0,
    audio_index: int = 0,
) -> str:
    """
    Export segments from long video as clean MP4.
    Applies padding, merges overlapping segments, re-encodes for clean cuts.
    Maps the selected audio track from the original video.
    """
    if not segments:
        raise RuntimeError("No hay segmentos para exportar")

    video_duration = _get_video_duration(long_video_path)

    padded = []
    for seg in segments:
        start = max(0, seg["start"] - pad_before)
        end = min(video_duration, seg["end"] + pad_after)
        padded.append({"start": start, "end": end})

    padded.sort(key=lambda s: s["start"])

    merged = [padded[0].copy()]
    for seg in padded[1:]:
        prev = merged[-1]
        if seg["start"] <= prev["end"]:
            prev["end"] = max(prev["end"], seg["end"])
        else:
            merged.append(seg.copy())

    print(
        f"[ClipFinder] Export: {len(segments)} segments -> {len(merged)} after padding({pad_before}s before, {pad_after}s after) + merge"
    )

    work_dir = Path(output_path).parent
    work_dir.mkdir(parents=True, exist_ok=True)
    segment_files = []

    for i, seg in enumerate(merged):
        seg_path = str(work_dir / f"_seg_{i}.mp4")
        cmd = [
            FFMPEG,
            "-y",
            "-ss",
            str(seg["start"]),
            "-to",
            str(seg["end"]),
            "-i",
            long_video_path,
            "-map",
            "0:v:0",
            "-map",
            f"0:a:{audio_index}",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "18",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-movflags",
            "+faststart",
            "-fflags",
            "+genpts",
            seg_path,
        ]
        print(
            f"[ClipFinder] Encoding segment {i + 1}/{len(merged)}: {seg['start']:.1f}s - {seg['end']:.1f}s"
        )
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            for sf in segment_files:
                try:
                    os.remove(sf)
                except:
                    pass
            raise RuntimeError(f"ffmpeg error on segment {i}: {result.stderr}")
        segment_files.append(seg_path)

    if len(segment_files) == 1:
        shutil.copy(segment_files[0], output_path)
    else:
        concat_list = str(work_dir / "_concat.txt")
        with open(concat_list, "w") as f:
            for sf in segment_files:
                safe_path = os.path.abspath(sf).replace("\\", "/")
                f.write(f"file '{safe_path}'\n")

        cmd = [
            FFMPEG,
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            concat_list,
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            output_path,
        ]
        print(f"[ClipFinder] Concatenating {len(segment_files)} segments...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg concat error: {result.stderr}")

    for sf in segment_files:
        try:
            os.remove(sf)
        except:
            pass
    try:
        os.remove(str(work_dir / "_concat.txt"))
    except:
        pass

    return output_path


def _get_video_duration(video_path: str) -> float:
    cmd = [
        FFPROBE,
        "-v",
        "quiet",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0 and result.stdout.strip():
        try:
            return float(result.stdout.strip())
        except ValueError:
            pass
    return float("inf")
