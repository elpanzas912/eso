"""
processor.py - Core logic for ClipFinder
Handles: yt-dlp download, Whisper transcription, fuzzy segment matching
"""

import subprocess
import json
import os
import shutil
import hashlib
import math
import wave
import threading
from array import array
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

_MODEL_CACHE: dict = {}
_model_lock = threading.Lock()
_transcribe_lock = threading.Lock()

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
WAVEFORM_CACHE_DIR = WORK_DIR / "_waveform_cache"

# TTLs for automatic cleanup (seconds)
JOB_DIR_TTL = 24 * 60 * 60  # 24 hours
WAVEFORM_CACHE_TTL = 7 * 24 * 60 * 60  # 7 days
TRANSCRIPTION_CACHE_TTL = 30 * 24 * 60 * 60  # 30 days
CLEANUP_INTERVAL = 6 * 60 * 60  # run cleanup every 6 hours


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


def _waveform_cache_key(video_path: Path, audio_index: int, bins: int) -> str:
    key_str = str(video_path.resolve()) + f"|waveform|{audio_index}|{bins}"
    if video_path.exists():
        stat = video_path.stat()
        key_str += f"|{stat.st_size}|{stat.st_mtime}"
    return hashlib.md5(key_str.encode()).hexdigest()[:20]


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
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
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
    logger.info(
        "Probe completado | path=%s | payload=%s", video_path, preview(payload, 1200)
    )
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
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
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


def get_waveform_peaks(
    video_path: Path,
    audio_index: int = 0,
    bins: int = 4096,
) -> dict:
    if bins <= 0:
        raise RuntimeError(f"Cantidad de bins inválida: {bins}")

    import threading

    if not hasattr(get_waveform_peaks, "_locks"):
        get_waveform_peaks._locks = {}
    if not hasattr(get_waveform_peaks, "_locks_lock"):
        get_waveform_peaks._locks_lock = threading.Lock()

    WAVEFORM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_key = _waveform_cache_key(video_path, audio_index, bins)
    cache_file = WAVEFORM_CACHE_DIR / f"{cache_key}.json"
    wav_file = WAVEFORM_CACHE_DIR / f"{cache_key}.wav"

    if cache_file.exists():
        logger.info(
            "Waveform cache hit | video=%s | audio_index=%s | bins=%s | cache=%s",
            video_path,
            audio_index,
            bins,
            cache_file,
        )
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)

    logger.info(
        "Generando waveform real | video=%s | audio_index=%s | bins=%s | wav=%s",
        video_path,
        audio_index,
        bins,
        wav_file,
    )
    cache_lock = get_waveform_peaks._locks_lock
    with cache_lock:
        lock = get_waveform_peaks._locks.setdefault(cache_key, threading.Lock())
    with lock:
        if not wav_file.exists():
            extract_audio_track(video_path, audio_index, wav_file)

    with wave.open(str(wav_file), "rb") as wav:
        frame_rate = wav.getframerate()
        total_frames = wav.getnframes()
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()

        if sample_width != 2:
            raise RuntimeError(
                f"Waveform solo soporta PCM de 16-bit. sample_width={sample_width}"
            )

        frames_per_bin = max(1, math.ceil(total_frames / bins))
        peaks = []

        for _ in range(bins):
            raw_frames = wav.readframes(frames_per_bin)
            if not raw_frames:
                break

            samples = array("h")
            samples.frombytes(raw_frames)
            if channels > 1:
                samples = array("h", samples[::channels])

            if not samples:
                peaks.append(0.0)
                continue

            peak = max(abs(sample) for sample in samples) / 32768.0
            peaks.append(round(min(1.0, peak), 4))

    if len(peaks) < bins:
        peaks.extend([0.0] * (bins - len(peaks)))

    duration = round(total_frames / frame_rate, 3) if frame_rate else 0
    payload = {
        "audio_index": audio_index,
        "bins": bins,
        "duration": duration,
        "channels": channels,
        "sample_rate": frame_rate,
        "peaks": peaks,
    }

    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)

    logger.info(
        "Waveform generado | video=%s | audio_index=%s | duration=%s | peaks=%s | cache=%s",
        video_path,
        audio_index,
        duration,
        len(peaks),
        cache_file,
    )
    return payload


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
    result = subprocess.run(cmd_copy, capture_output=True, text=True, timeout=300)
    if result.returncode == 0:
        logger.info("Remux intento 1 exitoso | output=%s", output_path)
        return output_path
    logger.warning("Remux intento 1 fallo | stderr=%s", preview(result.stderr, 1200))

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
    result = subprocess.run(cmd_aac, capture_output=True, text=True, timeout=300)
    if result.returncode == 0:
        logger.info("Remux intento 2 exitoso | output=%s", output_path)
        return output_path
    logger.warning("Remux intento 2 fallo | stderr=%s", preview(result.stderr, 1200))

    # Try 3: full re-encode (slow, last resort)
    if output_path.exists():
        os.remove(output_path)
    cmd_reencode = [
        FFMPEG,
        "-y",
        "-nostats",
        "-loglevel",
        "error",
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
    result = subprocess.run(cmd_reencode, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        logger.error("Remux intento 3 fallo | stderr=%s", preview(result.stderr, 1200))
        raise RuntimeError(f"ffmpeg remux error: {result.stderr}")
    logger.info("Remux intento 3 exitoso | output=%s", output_path)
    return output_path


_short_cache: dict[str, Path] = {}
_short_cache_lock = threading.Lock()
_SHORT_CACHE_MAX_SIZE = 50


def download_video(url: str, output_path: Path, progress_cb=None) -> Path:
    """Download video using yt-dlp. Returns actual path (yt-dlp may change extension)."""
    import yt_dlp

    cache_key = hashlib.md5(url.encode()).hexdigest()[:16]
    with _short_cache_lock:
        cached = _short_cache.get(cache_key)
    if cached is not None and cached.exists():
        logger.info(
            "Descarga cache hit (short global) | url=%s | cache_key=%s | path=%s",
            url,
            cache_key,
            cached,
        )
        if progress_cb:
            progress_cb("Usando clip cacheado...")
        return cached

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
        candidates = [
            p
            for p in output_path.parent.glob(output_path.stem + ".*")
            if p.suffix.lower() in (".mp4", ".mkv", ".webm", ".m4v")
        ]
        if candidates:
            actual = candidates[0]
            logger.info(
                "Descarga resolvio archivo por candidatos | stem=%s | elegido=%s | candidatos=%s",
                output_path.stem,
                actual,
                [str(candidate) for candidate in candidates],
            )
    logger.info("Descarga finalizada | url=%s | output=%s", url, actual)
    with _short_cache_lock:
        _short_cache[cache_key] = actual
        if len(_short_cache) > _SHORT_CACHE_MAX_SIZE:
            oldest_key = next(iter(_short_cache))
            del _short_cache[oldest_key]
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

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if _CUDA_AVAILABLE else "cpu"
    model_cache_key = f"{model_size}|{device}"
    with _model_lock:
        model = _MODEL_CACHE.get(model_cache_key)
    if model is None:
        if progress_cb:
            progress_cb(
                f"Cargando modelo Whisper ({model_size}) en {_GPU_NAME or 'CPU'}..."
            )
        logger.info(
            "Transcripcion iniciada | video=%s | model=%s | audio_index=%s | device=%s | cache=%s",
            video_path,
            model_size,
            audio_index,
            device,
            cache_file,
        )
        model = whisper.load_model(
            model_size, download_root=str(MODELS_DIR), device=device
        )
        with _model_lock:
            _MODEL_CACHE[model_cache_key] = model
        logger.info(
            "Modelo Whisper cargado | model=%s | device=%s | models_dir=%s",
            model_size,
            device,
            MODELS_DIR,
        )
    else:
        if progress_cb:
            progress_cb(f"Usando modelo Whisper ({model_size}) cacheado...")
        logger.info(
            "Transcripcion iniciada | video=%s | model=%s | audio_index=%s | device=%s | cache=%s | cached_model=true",
            video_path,
            model_size,
            audio_index,
            device,
            cache_file,
        )

    extracted_wav = None
    actual_path = video_path
    if audio_index > 0:
        if progress_cb:
            progress_cb(f"Extrayendo pista de audio #{audio_index}...")
        audio_extract_path = video_path.parent / f"_audio_track_{audio_index}.wav"
        actual_path = extract_audio_track(video_path, audio_index, audio_extract_path)
        extracted_wav = actual_path
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
        with _transcribe_lock:
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
        with _transcribe_lock:
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

    if extracted_wav is not None:
        try:
            os.remove(str(extracted_wav))
            logger.info("WAV temporal eliminado | path=%s", extracted_wav)
        except Exception:
            pass
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
    step: int = 4,
    srt_time_range: tuple[float, float] | None = None,
    srt_margin: float = 120.0,
) -> list[dict]:
    """
    Find where segments of the short clip appear in the long video.
    Handles non-chronological clips (scenes appearing in any order).

    Algorithm:
    1. Slide overlapping windows across the short clip
    2. For each window, find the best fuzzy match in the long video
    3. Cluster hits by proximity in the long video (ignores short clip order)
    4. Each cluster becomes a segment
    5. Merge segments close in the long video timeline

    If srt_time_range is provided, limits search to that time window +/- margin
    to dramatically reduce search space.

    Returns list of: {start, end, text, short_start, short_end, confidence}
    """
    from rapidfuzz import fuzz

    def words_text(words_slice):
        return " ".join(w["word"].lower() for w in words_slice)

    n_short = len(short_words)
    n_long = len(long_words)

    if n_short == 0 or n_long == 0:
        logger.info("Matching descartado | short=%s | long=%s", n_short, n_long)
        return []

    long_start_idx = 0
    long_end_idx = n_long
    if srt_time_range is not None:
        start_t, end_t = srt_time_range
        search_start = max(0, start_t - srt_margin)
        search_end = end_t + srt_margin
        long_start_idx = 0
        long_end_idx = n_long
        for idx in range(n_long):
            if long_words[idx]["start"] >= search_start:
                long_start_idx = idx
                break
        for idx in range(long_start_idx, n_long):
            if long_words[idx]["start"] > search_end:
                long_end_idx = idx
                break
        trimmed_long = long_words[long_start_idx:long_end_idx]
        logger.info(
            "Matching con rango SRT | time_window=%.1f-%.1f | margin=%.1f | original_words=%s | trimmed_words=%s | original_start=%.1f | trimmed_start=%.1f",
            start_t,
            end_t,
            srt_margin,
            n_long,
            len(trimmed_long),
            long_words[0]["start"] if long_words else 0,
            trimmed_long[0]["start"] if trimmed_long else 0,
        )
    else:
        trimmed_long = long_words
        logger.info(
            "Matching sin rango SRT | words=%s",
            n_long,
        )

    effective_window = min(window, n_short)
    min_chunk = max(3, effective_window // 2)

    logger.info(
        "Matching iniciado | short=%s | long=%s | window=%s | step=%s | threshold=%s | merge_gap=%s",
        preview(summarize_words(short_words), 800),
        preview(summarize_words(trimmed_long), 800),
        effective_window,
        step,
        threshold,
        merge_gap,
    )

    hits = []
    i = 0
    while i < n_short:
        chunk_end = min(i + effective_window, n_short)
        chunk_len = chunk_end - i
        if chunk_len < min_chunk and i > 0:
            break

        chunk = short_words[i:chunk_end]
        chunk_text = words_text(chunk)

        best_score = 0
        best_j = -1

        trimmed_limit = len(trimmed_long) - chunk_len + 1
        for j in range(max(0, min(trimmed_limit, n_long - chunk_len + 1))):
            if j >= len(trimmed_long) - chunk_len + 1:
                break
            jj = j
            long_chunk = trimmed_long[jj : jj + chunk_len]
            score = fuzz.ratio(chunk_text, words_text(long_chunk))
            if score > best_score:
                best_score = score
                best_j = j

        logger.info(
            "Matching ventana | short=%s-%s | len=%s | best_score=%s | best_long=%s | text=%s",
            i,
            chunk_end - 1,
            chunk_len,
            best_score,
            best_j,
            chunk_text[:120],
        )

        if best_score >= threshold and best_j >= 0:
            hits.append(
                {
                    "short_i": i,
                    "short_j": chunk_end - 1,
                    "long_i": best_j + long_start_idx,
                    "long_j": best_j + chunk_len - 1 + long_start_idx,
                    "score": best_score,
                }
            )

        i += step

    if not hits:
        logger.info("Matching finalizado sin hits")
        return []

    logger.info("Hits encontrados | total=%s", len(hits))

    hits.sort(key=lambda h: h["long_i"])

    cluster_gap = effective_window * 3
    clusters = [[hits[0]]]
    for hit in hits[1:]:
        cluster = clusters[-1]
        cluster_long_max = max(h["long_j"] for h in cluster)
        if hit["long_i"] <= cluster_long_max + cluster_gap:
            cluster.append(hit)
        else:
            clusters.append([hit])

    logger.info("Clusters formados | total=%s", len(clusters))

    segments = []
    for cluster in clusters:
        short_start_time = min(short_words[h["short_i"]]["start"] for h in cluster)
        short_end_time = max(short_words[h["short_j"]]["end"] for h in cluster)
        long_start_time = min(long_words[h["long_i"]]["start"] for h in cluster)
        long_end_time = max(long_words[h["long_j"]]["end"] for h in cluster)

        short_indices = sorted(
            set(idx for h in cluster for idx in range(h["short_i"], h["short_j"] + 1))
        )
        text = " ".join(short_words[idx]["word"] for idx in short_indices)
        avg_score = sum(h["score"] for h in cluster) / len(cluster)

        segments.append(
            {
                "start": round(long_start_time, 3),
                "end": round(long_end_time, 3),
                "text": text,
                "short_start": round(short_start_time, 3),
                "short_end": round(short_end_time, 3),
                "confidence": round(avg_score),
            }
        )

    if not segments:
        logger.info("Matching finalizado sin segmentos tras clustering")
        return []

    segments.sort(key=lambda s: s["short_start"])

    logger.info(
        "Matching finalizado | hits=%s | clusters=%s | segments=%s",
        len(hits),
        len(clusters),
        len(segments),
    )
    return segments


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

    logger.info(
        "Export segmentos preparados | count=%s | segments=%s",
        len(padded),
        preview(padded, 1200),
    )

    work_dir = Path(output_path).parent
    work_dir.mkdir(parents=True, exist_ok=True)
    segment_files = []

    for i, seg in enumerate(padded):
        seg_path = str(work_dir / f"_seg_{i}.mp4")
        cmd = [
            FFMPEG,
            "-y",
            "-nostats",
            "-loglevel",
            "error",
            "-fflags",
            "+genpts",
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
            seg_path,
        ]
        logger.info(
            "Encoding segmento | idx=%s/%s | start=%.3f | end=%.3f | output=%s | cmd=%s",
            i + 1,
            len(padded),
            seg["start"],
            seg["end"],
            seg_path,
            _format_cmd(cmd),
        )
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            for sf in segment_files:
                try:
                    os.remove(sf)
                except Exception:
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
            "-nostats",
            "-loglevel",
            "error",
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
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            logger.error("Concat fallo | stderr=%s", preview(result.stderr, 1500))
            raise RuntimeError(f"ffmpeg concat error: {result.stderr}")

    for sf in segment_files:
        try:
            os.remove(sf)
        except Exception:
            pass
    try:
        os.remove(str(work_dir / "_concat.txt"))
    except Exception:
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
            logger.info(
                "Duracion detectada | path=%s | duration=%s", video_path, duration
            )
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
    raise RuntimeError(f"No se pudo determinar la duración del video: {video_path}")
