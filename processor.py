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

from debug_utils import get_logger, preview, summarize_segments, summarize_words

try:
    import torch

    _CUDA_AVAILABLE = torch.cuda.is_available()
    _GPU_NAME = torch.cuda.get_device_name(0) if _CUDA_AVAILABLE else None
except ImportError:
    torch = None
    _CUDA_AVAILABLE = False
    _GPU_NAME = None

logger = get_logger("processor")

if _CUDA_AVAILABLE:
    logger.info("GPU detectada: %s", _GPU_NAME)
    logger.info(
        "VRAM detectada: %s GB",
        torch.cuda.get_device_properties(0).total_memory // (1024**3),
    )
else:
    logger.warning(
        "GPU no detectada; usando CPU (instala PyTorch con CUDA para acelerar)"
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
logger.info(
    "Herramientas multimedia resueltas | ffmpeg=%s | ffprobe=%s | ffmpeg_dir=%s",
    FFMPEG,
    FFPROBE,
    _FFMPEG_DIR or "PATH",
)


def _format_cmd(cmd: list[str]) -> str:
    return subprocess.list2cmdline([str(part) for part in cmd])


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
    logger.info("Probe iniciado | path=%s | cmd=%s", video_path, _format_cmd(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(
            "Probe fallo | path=%s | returncode=%s | stderr=%s",
            video_path,
            result.returncode,
            preview(result.stderr, 1200),
        )
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

    payload = {
        "streams": audio_streams,
        "english_index": english_idx,
        "description": "\n".join(lines)
        if lines
        else "No se encontraron pistas de audio",
    }
    logger.info("Probe completado | path=%s | payload=%s", video_path, preview(payload, 1200))
    return payload


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
    logger.info(
        "Extrayendo pista de audio | video=%s | audio_index=%s | output=%s | cmd=%s",
        video_path,
        audio_index,
        output_path,
        _format_cmd(cmd),
    )
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(
            "Extraccion de audio fallo | video=%s | audio_index=%s | stderr=%s",
            video_path,
            audio_index,
            preview(result.stderr, 1200),
        )
        raise RuntimeError(f"ffmpeg audio extraction error: {result.stderr}")
    logger.info(
        "Extraccion de audio completada | video=%s | audio_index=%s | output=%s",
        video_path,
        audio_index,
        output_path,
    )
    return output_path


def remux_for_browser(
    video_path: Path, output_path: Path, audio_index: int = 0
) -> Path:
    logger.info(
        "Remux iniciado | input=%s | output=%s | audio_index=%s",
        video_path,
        output_path,
        audio_index,
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
    logger.info("Remux intento 1 | copy directo | cmd=%s", _format_cmd(cmd_copy))
    result = subprocess.run(cmd_copy, capture_output=True, text=True)
    if result.returncode == 0:
        logger.info("Remux intento 1 exitoso | output=%s", output_path)
        return output_path
    logger.warning(
        "Remux intento 1 fallo | stderr=%s", preview(result.stderr, 1200)
    )

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
    logger.info("Remux intento 2 | copy video + AAC | cmd=%s", _format_cmd(cmd_aac))
    result = subprocess.run(cmd_aac, capture_output=True, text=True)
    if result.returncode == 0:
        logger.info("Remux intento 2 exitoso | output=%s", output_path)
        return output_path
    logger.warning(
        "Remux intento 2 fallo | stderr=%s", preview(result.stderr, 1200)
    )

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
    logger.info(
        "Remux intento 3 | reencode completo | cmd=%s", _format_cmd(cmd_reencode)
    )
    result = subprocess.run(cmd_reencode, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(
            "Remux intento 3 fallo | stderr=%s", preview(result.stderr, 1200)
        )
        raise RuntimeError(f"ffmpeg remux error: {result.stderr}")
    logger.info("Remux intento 3 exitoso | output=%s", output_path)
    return output_path


def download_video(url: str, output_path: Path, progress_cb=None) -> Path:
    """Download video using yt-dlp. Returns actual path (yt-dlp may change extension)."""
    import yt_dlp

    class ProgressHook:
        def __call__(self, d):
            if progress_cb and d["status"] == "downloading":
                pct = d.get("_percent_str", "?")
                progress_cb(f"Descargando... {pct}")
            logger.info(
                "yt-dlp progreso | status=%s | downloaded=%s | total=%s | percent=%s | eta=%s",
                d.get("status"),
                d.get("downloaded_bytes"),
                d.get("total_bytes") or d.get("total_bytes_estimate"),
                d.get("_percent_str"),
                d.get("eta"),
            )

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

    logger.info(
        "Descarga iniciada | url=%s | output_template=%s | ydl_opts=%s",
        url,
        output_path.with_suffix(".%(ext)s"),
        preview(ydl_opts, 1200),
    )
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        ext = info.get("ext", "mp4")
        logger.info(
            "yt-dlp metadata | url=%s | ext=%s | title=%s | duration=%s | extractor=%s",
            url,
            ext,
            info.get("title"),
            info.get("duration"),
            info.get("extractor"),
        )

    actual = output_path.with_suffix("." + ext)
    if not actual.exists():
        actual = output_path.with_suffix(".mp4")
    if not actual.exists():
        candidates = list(output_path.parent.glob(output_path.stem + ".*"))
        if candidates:
            actual = candidates[0]
            logger.info(
                "Descarga resolvio archivo por candidatos | stem=%s | elegido=%s | candidatos=%s",
                output_path.stem,
                actual,
                [str(candidate) for candidate in candidates],
            )
    logger.info("Descarga finalizada | url=%s | output=%s", url, actual)
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
        logger.info(
            "Transcripcion cache hit | video=%s | model=%s | audio_index=%s | cache=%s",
            video_path,
            model_size,
            audio_index,
            cache_file,
        )
        with open(cache_file, "r", encoding="utf-8") as f:
            cached_words = json.load(f)
        logger.info(
            "Transcripcion cache cargada | resumen=%s",
            preview(summarize_words(cached_words), 800),
        )
        return cached_words

    import whisper

    if progress_cb:
        progress_cb(
            f"Cargando modelo Whisper ({model_size}) en {_GPU_NAME or 'CPU'}..."
        )
    logger.info(
        "Transcripcion iniciada | video=%s | model=%s | audio_index=%s | device=%s | cache=%s",
        video_path,
        model_size,
        audio_index,
        "cuda" if _CUDA_AVAILABLE else "cpu",
        cache_file,
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if _CUDA_AVAILABLE else "cpu"
    model = whisper.load_model(model_size, download_root=str(MODELS_DIR), device=device)
    logger.info(
        "Modelo Whisper cargado | model=%s | device=%s | models_dir=%s",
        model_size,
        device,
        MODELS_DIR,
    )

    actual_path = video_path
    if audio_index > 0:
        if progress_cb:
            progress_cb(f"Extrayendo pista de audio #{audio_index}...")
        audio_extract_path = video_path.parent / f"_audio_track_{audio_index}.wav"
        actual_path = extract_audio_track(video_path, audio_index, audio_extract_path)
        logger.info(
            "Transcripcion usara audio extraido | source=%s | extracted=%s",
            video_path,
            actual_path,
        )

    if progress_cb:
        progress_cb(f"Transcribiendo {Path(actual_path).name}...")

    fp16 = device == "cuda"

    words = []
    try:
        logger.info(
            "Whisper transcribe | path=%s | word_timestamps=%s | fp16=%s",
            actual_path,
            True,
            fp16,
        )
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
        logger.warning(
            "word_timestamps fallo; usando fallback por segmentos | path=%s",
            actual_path,
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
        logger.error(
            "Transcripcion vacia | video=%s | actual_path=%s | model=%s | audio_index=%s",
            video_path,
            actual_path,
            model_size,
            audio_index,
        )
        raise RuntimeError(f"No se pudo transcribir {Path(actual_path).name}")

    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(words, f, ensure_ascii=False)
    logger.info(
        "Transcripcion completada | video=%s | actual_path=%s | resumen=%s | cache=%s",
        video_path,
        actual_path,
        preview(summarize_words(words), 800),
        cache_file,
    )

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
    logger.info(
        "Matching iniciado | short=%s | long=%s | window=%s | threshold=%s | merge_gap=%s",
        preview(summarize_words(short_words), 800),
        preview(summarize_words(long_words), 800),
        window,
        threshold,
        merge_gap,
    )

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

        logger.info(
            "Matching chunk evaluado | short_idx=%s | chunk_end=%s | chunk_text=%s | best_score=%s | best_long_idx=%s",
            i,
            chunk_end,
            chunk_text[:120],
            best_score,
            best_j,
        )

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
            logger.info(
                "Segmento aceptado | short_idx=%s | long_idx=%s | match_len=%s | segment=%s",
                i,
                best_j,
                match_len,
                preview(segments[-1], 600),
            )
            i += match_len
        else:
            logger.info(
                "Chunk descartado | short_idx=%s | best_score=%s | threshold=%s",
                i,
                best_score,
                threshold,
            )
            i += 1

    # Merge segments that are too close together
    if not segments:
        logger.info("Matching finalizado sin segmentos")
        return []

    merged = [segments[0].copy()]
    for seg in segments[1:]:
        prev = merged[-1]
        if seg["start"] - prev["end"] <= merge_gap:
            logger.info(
                "Mergeando segmentos | prev=%s | current=%s",
                preview(prev, 400),
                preview(seg, 400),
            )
            prev["end"] = max(prev["end"], seg["end"])
            prev["short_end"] = max(prev["short_end"], seg["short_end"])
            prev["text"] += " " + seg["text"]
            prev["confidence"] = (prev["confidence"] + seg["confidence"]) // 2
        else:
            merged.append(seg.copy())

    logger.info(
        "Matching finalizado | raw=%s | merged=%s",
        preview(summarize_segments(segments), 800),
        preview(summarize_segments(merged), 800),
    )
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
    logger.info(
        "Export iniciado | video=%s | audio_index=%s | pad_before=%s | pad_after=%s | input_segments=%s | duration=%s",
        long_video_path,
        audio_index,
        pad_before,
        pad_after,
        preview(summarize_segments(segments), 800),
        video_duration,
    )

    padded = []
    for seg in segments:
        start = max(0, seg["start"] - pad_before)
        end = min(video_duration, seg["end"] + pad_after)
        padded.append({"start": start, "end": end})
        logger.info(
            "Segmento con padding | original=%s | padded=%s",
            preview(seg, 300),
            preview(padded[-1], 200),
        )

    padded.sort(key=lambda s: s["start"])

    merged = [padded[0].copy()]
    for seg in padded[1:]:
        prev = merged[-1]
        if seg["start"] <= prev["end"]:
            prev["end"] = max(prev["end"], seg["end"])
        else:
            merged.append(seg.copy())

    logger.info(
        "Export segmentos preparados | padded=%s | merged=%s",
        preview(padded, 1200),
        preview(merged, 1200),
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
        logger.info(
            "Encoding segmento | idx=%s/%s | start=%.3f | end=%.3f | output=%s | cmd=%s",
            i + 1,
            len(merged),
            seg["start"],
            seg["end"],
            seg_path,
            _format_cmd(cmd),
        )
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            for sf in segment_files:
                try:
                    os.remove(sf)
                except:
                    pass
            logger.error(
                "Encoding segmento fallo | idx=%s | stderr=%s",
                i,
                preview(result.stderr, 1500),
            )
            raise RuntimeError(f"ffmpeg error on segment {i}: {result.stderr}")
        segment_files.append(seg_path)
        logger.info("Encoding segmento completado | idx=%s | file=%s", i, seg_path)

    if len(segment_files) == 1:
        shutil.copy(segment_files[0], output_path)
        logger.info(
            "Export copio segmento unico | source=%s | output=%s",
            segment_files[0],
            output_path,
        )
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
        logger.info(
            "Concatenando segmentos | count=%s | concat_list=%s | cmd=%s",
            len(segment_files),
            concat_list,
            _format_cmd(cmd),
        )
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(
                "Concat fallo | stderr=%s", preview(result.stderr, 1500)
            )
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

    logger.info("Export completado | output=%s", output_path)
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
    logger.info(
        "Consultando duracion de video | path=%s | cmd=%s",
        video_path,
        _format_cmd(cmd),
    )
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0 and result.stdout.strip():
        try:
            duration = float(result.stdout.strip())
            logger.info("Duracion detectada | path=%s | duration=%s", video_path, duration)
            return duration
        except ValueError:
            logger.warning(
                "No se pudo parsear duracion | path=%s | stdout=%s",
                video_path,
                preview(result.stdout, 200),
            )
    else:
        logger.warning(
            "Consulta de duracion fallo | path=%s | returncode=%s | stderr=%s",
            video_path,
            result.returncode,
            preview(result.stderr, 800),
        )
    return float("inf")
