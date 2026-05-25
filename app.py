"""
app.py - ClipFinder Flask backend
"""

import atexit
import mimetypes
import os
import signal
import shutil
import uuid
import json
import threading
import time
from pathlib import Path
from flask import Flask, jsonify, request, send_file, render_template, abort, g

from debug_utils import configure_logging, get_logger, preview, summarize_segments

configure_logging()
import processor  # noqa: E402
from processor import WORK_DIR  # noqa: E402
import episode_matcher  # noqa: E402
import subtitle_loader  # noqa: E402
from subtitle_loader import load_series_registry  # noqa: E402

RESIDUOS_DIR = Path(__file__).resolve().parent / "residuos para ia"
RESIDUOS_DIR.mkdir(parents=True, exist_ok=True)

OUTPUTS_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
logger = get_logger("app")

WORK_DIR.mkdir(parents=True, exist_ok=True)

MAX_CONCURRENT_JOBS = 2
JOBS_FILE = WORK_DIR / "_jobs.json"
JOBS_CLEANUP_TTL = 24 * 60 * 60  # 24 hours

jobs: dict[str, dict] = {}
jobs_lock = threading.RLock()
_shutdown_flag = threading.Event()


def _active_job_count() -> int:
    with jobs_lock:
        return sum(
            1
            for j in jobs.values()
            if j.get("status") not in ("done", "error", "not_found")
        )


def _persist_jobs():
    try:
        with jobs_lock:
            serializable = {}
            for jid, job in jobs.items():
                serializable[jid] = {
                    k: v
                    for k, v in job.items()
                    if k not in ("_worker_thread",)
                    and isinstance(v, (str, int, float, bool, list, dict, type(None)))
                }
        JOBS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(JOBS_FILE, "w", encoding="utf-8") as f:
            json.dump(serializable, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def _load_jobs():
    if not JOBS_FILE.exists():
        return
    try:
        with open(JOBS_FILE, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        now = time.time()
        restored = 0
        with jobs_lock:
            for jid, job in loaded.items():
                created = job.pop("_created_at", None)
                if created and (now - created) > JOBS_CLEANUP_TTL:
                    continue
                job["status"] = "done"
                job["progress"] = "(restaurado de sesion anterior)"
                jobs[jid] = job
                restored += 1
        if restored:
            logger.info("Jobs restaurados de disco | count=%s", restored)
    except Exception:
        logger.warning("No se pudieron cargar jobs desde %s", JOBS_FILE)


def _cleanup_atexit():
    logger.info("Ejecutando cleanup atexit")
    _shutdown_flag.set()
    with jobs_lock:
        for jid, job in list(jobs.items()):
            if job.get("status") not in ("done", "error"):
                job["status"] = "error"
                job["progress"] = "Servidor cerrado durante el proceso"
        _persist_jobs()
    _KEEP_DIRS = {
        "_whisper_models",
        "_cache",
        "_srt_cache",
        "_waveform_cache",
        "_yt_cache",
    }
    for d in WORK_DIR.iterdir():
        if d.is_dir() and d.name not in _KEEP_DIRS:
            try:
                shutil.rmtree(d, ignore_errors=True)
            except Exception:
                pass
    logger.info("Cleanup completado")


atexit.register(_cleanup_atexit)


def _signal_handler(signum, frame):
    logger.info("Recibida senal %s, iniciando shutdown", signum)
    _cleanup_atexit()
    os._exit(0)


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)

_load_jobs()


def _job_snapshot(job: dict | None) -> dict:
    if not job:
        return {}
    snapshot = {}
    for key in (
        "status",
        "progress",
        "error",
        "audio_index",
        "long_path",
        "original_path",
    ):
        if key in job:
            snapshot[key] = job.get(key)
    if "segments" in job:
        snapshot["segments"] = summarize_segments(job.get("segments", []))
    if "detection_results" in job:
        snapshot["detection_results"] = {
            "count": len(job.get("detection_results", [])),
            "top": (job.get("detection_results") or [None])[0],
        }
    if "streams" in job:
        snapshot["streams"] = len(job.get("streams", []))
    return snapshot


def _resolve_playback_video(job_id: str, job: dict, audio_index: int) -> str:
    streams = job.get("streams") or []
    default_audio_index = int(job.get("audio_index", 0))
    playback_path = job.get("long_path")
    source_path = job.get("original_path") or playback_path

    if source_path is None:
        raise RuntimeError("El job no tiene una fuente de video disponible")

    if streams and (audio_index < 0 or audio_index >= len(streams)):
        raise RuntimeError(
            f"Pista de audio inválida: {audio_index}. Disponibles: 0-{len(streams) - 1}"
        )

    if (
        audio_index == default_audio_index
        and playback_path
        and (str(playback_path) != str(source_path) or len(streams) <= 1)
    ):
        logger.info(
            "Playback reutiliza video principal del job | job_id=%s | audio_index=%s | path=%s",
            job_id,
            audio_index,
            playback_path,
        )
        return playback_path

    job_dir = WORK_DIR / job_id
    job_dir.mkdir(exist_ok=True)
    alt_playback_path = job_dir / f"playback_audio_{audio_index}.mp4"

    if alt_playback_path.exists():
        logger.info(
            "Playback reutiliza remux cacheado | job_id=%s | audio_index=%s | path=%s",
            job_id,
            audio_index,
            alt_playback_path,
        )
        return str(alt_playback_path)

    logger.info(
        "Playback generando remux alternativo | job_id=%s | audio_index=%s | source=%s | output=%s",
        job_id,
        audio_index,
        source_path,
        alt_playback_path,
    )
    remuxed_path = processor.remux_for_browser(
        Path(source_path), alt_playback_path, audio_index=audio_index
    )
    logger.info(
        "Playback remux alternativo listo | job_id=%s | audio_index=%s | path=%s",
        job_id,
        audio_index,
        remuxed_path,
    )
    return str(remuxed_path)


def _request_debug_payload():
    return {
        "query": request.args.to_dict(flat=False),
        "json": request.get_json(silent=True),
        "content_type": request.content_type,
        "content_length": request.content_length,
        "remote_addr": request.remote_addr,
    }


def update_job(job_id: str, **kwargs):
    with jobs_lock:
        before = dict(jobs.get(job_id, {}))
        jobs[job_id].update(kwargs)
        after = dict(jobs[job_id])
    logger.info(
        "Job actualizado | job_id=%s | changes=%s | before=%s | after=%s",
        job_id,
        preview(kwargs, 1000),
        preview(_job_snapshot(before), 1000),
        preview(_job_snapshot(after), 1000),
    )
    if after.get("status") in ("done", "error"):
        _persist_jobs()


@app.before_request
def log_request_start():
    g._request_started_at = time.monotonic()
    logger.info(
        "HTTP inicio | method=%s | path=%s | payload=%s",
        request.method,
        request.full_path.rstrip("?"),
        preview(_request_debug_payload(), 1200),
    )


@app.after_request
def log_request_end(response):
    started_at = getattr(g, "_request_started_at", None)
    duration_ms = (
        round((time.monotonic() - started_at) * 1000, 1) if started_at else None
    )
    logger.info(
        "HTTP fin | method=%s | path=%s | status=%s | duration_ms=%s | response_length=%s",
        request.method,
        request.full_path.rstrip("?"),
        response.status_code,
        duration_ms,
        response.calculate_content_length(),
    )
    return response


@app.teardown_request
def log_request_teardown(exc):
    if exc is None:
        return
    logger.error(
        "HTTP excepcion | method=%s | path=%s | error=%s",
        request.method,
        request.full_path.rstrip("?"),
        exc,
        exc_info=(type(exc), exc, exc.__traceback__),
    )


@app.route("/")
def index():
    logger.info("Render detect.html en /")
    return (
        render_template("detect.html"),
        200,
        {"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


@app.route("/example")
def example():
    logger.info("Render example.html en /example")
    return (
        render_template("example.html"),
        200,
        {"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


@app.route("/detect")
def detect():
    logger.info("Render detect.html en /detect")
    return (
        render_template("detect.html"),
        200,
        {"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


@app.route("/editor")
def editor():
    logger.info("Render index.html en /editor")
    return (
        render_template("index.html"),
        200,
        {"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


@app.route("/timeline-lab")
def timeline_lab():
    logger.info("Render timeline_lab.html en /timeline-lab")
    return (
        render_template("timeline_lab.html"),
        200,
        {"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


@app.route("/api/series", methods=["GET"])
def list_series():
    series = episode_matcher.list_series()
    for s in series:
        subs_path = Path(s.get("subtitles_path", ""))
        vids_path = Path(s.get("videos_path", ""))
        s["online"] = subs_path.exists() or vids_path.exists()
    logger.info(
        "API series GET | count=%s | payload=%s", len(series), preview(series, 1000)
    )
    return jsonify(series)


@app.route("/api/series", methods=["POST"])
def add_series():
    data = request.get_json()
    name = (data or {}).get("name", "").strip()
    subs_path = (data or {}).get("subtitles_path", "").strip()
    vids_path = (data or {}).get("videos_path", "").strip()
    language = (data or {}).get("language", "en").strip()
    logger.info(
        "API series POST | name=%s | subs=%s | vids=%s | language=%s",
        name,
        subs_path,
        vids_path,
        language,
    )

    if not name or not subs_path:
        return jsonify({"error": "Nombre y ruta de subtítulos son obligatorios"}), 400

    if not Path(subs_path).exists():
        return jsonify({"error": f"Ruta de subtítulos no encontrada: {subs_path}"}), 400

    try:
        series = episode_matcher.add_series(name, subs_path, vids_path, language)
        logger.info(
            "Serie agregada/actualizada via API | payload=%s", preview(series, 600)
        )
        return jsonify(series)
    except Exception as e:
        logger.exception("Error agregando serie | payload=%s", preview(data, 800))
        return jsonify({"error": str(e)}), 500


_pending = {}
_pending_lock = threading.Lock()
_HANDOFF_TTL = 600


@app.route("/api/handoff", methods=["POST"])
def handoff_save():
    data = request.get_json() or {}
    key = uuid.uuid4().hex[:10]
    with _pending_lock:
        _pending[key] = {"data": data, "created_at": time.time()}
        # cleanup expired entries
        now = time.time()
        expired = [
            k for k, v in _pending.items() if now - v["created_at"] > _HANDOFF_TTL
        ]
        for k in expired:
            del _pending[k]
        pending_count = len(_pending)
    logger.info(
        "Handoff guardado | key=%s | pending=%s | payload=%s",
        key,
        pending_count,
        preview(data, 1200),
    )
    return jsonify({"key": key})


@app.route("/api/handoff/<key>", methods=["GET"])
def handoff_load(key):
    with _pending_lock:
        item = _pending.pop(key, None)
        pending_count = len(_pending)
    if item is None or time.time() - item["created_at"] > _HANDOFF_TTL:
        logger.warning(
            "Handoff inexistente/expirado | key=%s | pending=%s", key, pending_count
        )
        return jsonify({"error": "expirado"}), 404
    data = item["data"]
    logger.info(
        "Handoff recuperado | key=%s | pending_restante=%s | payload=%s",
        key,
        pending_count,
        preview(data, 1200),
    )
    return jsonify(data)


@app.route("/api/detect", methods=["POST"])
def detect_episode():
    data = request.get_json()
    short_url = (data or {}).get("short_url", "").strip()
    series_name = (data or {}).get("series_name", "").strip()
    language = (data or {}).get("language", "en").strip()
    model = (data or {}).get("model", "base").strip()
    logger.info(
        "Solicitud de deteccion | short_url=%s | series=%s | language=%s | model=%s",
        short_url,
        series_name,
        language,
        model,
    )

    if not short_url or not series_name:
        return jsonify(
            {"error": "URL del clip y nombre de serie son obligatorios"}
        ), 400

    job_id = uuid.uuid4().hex[:8]
    with jobs_lock:
        if _active_job_count() >= MAX_CONCURRENT_JOBS:
            return jsonify(
                {
                    "error": f"Demasiados jobs activos (max {MAX_CONCURRENT_JOBS}). Espera que terminen los actuales."
                }
            ), 429
        jobs[job_id] = {
            "status": "detecting",
            "progress": "Descargando clip corto...",
            "_created_at": time.time(),
        }
    logger.info("Job de deteccion creado | job_id=%s", job_id)

    thread = threading.Thread(
        target=_run_detection,
        args=(job_id, short_url, series_name, language, model),
    )
    thread.daemon = True
    thread.start()
    logger.info(
        "Thread de deteccion iniciada | job_id=%s | thread=%s", job_id, thread.name
    )

    return jsonify({"job_id": job_id})


@app.route("/api/probe", methods=["POST"])
def probe():
    data = request.get_json()
    path = (data or {}).get("path", "").strip()
    logger.info("Solicitud probe | path=%s", path)
    if not path:
        return jsonify({"error": "Se requiere una ruta de archivo"}), 400

    path_obj = Path(path)
    if not path_obj.exists():
        return jsonify({"error": f"Archivo no encontrado: {path}"}), 400

    try:
        result = processor.probe_video(path_obj)
        logger.info(
            "Probe via API completado | path=%s | payload=%s",
            path_obj,
            preview(result, 1200),
        )
        return jsonify(result)
    except Exception as e:
        logger.exception("Probe via API fallo | path=%s", path_obj)
        return jsonify({"error": str(e)}), 500


@app.route("/api/editor_session", methods=["POST"])
def editor_session():
    data = request.get_json()
    raw_path = (data or {}).get("long_path", "").strip()
    try:
        audio_index = int((data or {}).get("audio_index", 0))
    except (TypeError, ValueError):
        return jsonify({"error": "audio_index debe ser un número entero"}), 400

    if not raw_path:
        return jsonify({"error": "Se requiere la ruta del video"}), 400

    source_path = Path(raw_path)
    if not source_path.exists() or not source_path.is_file():
        return jsonify({"error": f"No existe el archivo: {source_path}"}), 404

    job_id = str(uuid.uuid4())[:8]
    job_dir = WORK_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    try:
        probe_result = processor.probe_video(source_path)
        streams = probe_result.get("streams", [])
        if streams and (audio_index < 0 or audio_index >= len(streams)):
            return (
                jsonify(
                    {
                        "error": f"Pista de audio inválida: {audio_index}. Disponibles: 0-{len(streams) - 1}"
                    }
                ),
                400,
            )

        playback_path = source_path
        if source_path.suffix.lower() not in (".mp4", ".m4v"):
            playback_path = processor.remux_for_browser(
                source_path, job_dir / "long.mp4", audio_index=audio_index
            )

        segments = (data or {}).get("segments")
        if not isinstance(segments, list):
            segments = []

        with jobs_lock:
            jobs[job_id] = {
                "status": "done",
                "progress": "Editor listo",
                "segments": segments,
                "error": None,
                "audio_index": audio_index,
                "long_path": str(playback_path),
                "original_path": str(source_path),
                "streams": streams,
                "_created_at": time.time(),
                "editor_only": True,
            }
        _persist_jobs()

        logger.info(
            "Sesion de editor creada | job_id=%s | source=%s | playback=%s | audio_index=%s | streams=%s",
            job_id,
            source_path,
            playback_path,
            audio_index,
            len(streams),
        )
        return jsonify(
            {
                "job_id": job_id,
                "status": "done",
                "progress": "Editor listo",
                "segments": segments,
                "audio_index": audio_index,
                "long_path": str(playback_path),
                "original_path": str(source_path),
                "streams": streams,
            }
        )
    except Exception as e:
        logger.exception("Sesion de editor fallo | path=%s", source_path)
        return jsonify({"error": str(e)}), 500


@app.route("/api/process", methods=["POST"])
def process():
    data = request.get_json()
    long_url = (data or {}).get("long_url", "").strip()
    short_url = (data or {}).get("short_url", "").strip()
    long_path = (data or {}).get("long_path", "").strip()
    long_source = (data or {}).get("long_source", "url")
    audio_index = (data or {}).get("audio_index", 0)
    try:
        audio_index = int(audio_index)
    except (TypeError, ValueError):
        return jsonify({"error": "audio_index debe ser un número entero"}), 400
    episode_id = (data or {}).get("episode_id", "")
    series_name = (data or {}).get("series_name", "")
    model_size = (data or {}).get("model", "base")
    use_original_transcription = (data or {}).get("use_original_transcription", False)
    logger.info(
        "Solicitud de proceso | long_source=%s | long_url=%s | long_path=%s | short_url=%s | audio_index=%s | model=%s | use_original=%s",
        long_source,
        long_url,
        long_path,
        short_url,
        audio_index,
        model_size,
        use_original_transcription,
    )

    if not short_url:
        return jsonify({"error": "Se requiere la URL del clip corto"}), 400

    if long_source == "local":
        if not long_path:
            return jsonify({"error": "Se requiere la ruta del video largo"}), 400
    else:
        if not long_url:
            return jsonify({"error": "Se requiere la URL del video largo"}), 400

    job_id = str(uuid.uuid4())[:8]
    with jobs_lock:
        if _active_job_count() >= MAX_CONCURRENT_JOBS:
            return jsonify(
                {
                    "error": f"Demasiados jobs activos (max {MAX_CONCURRENT_JOBS}). Espera que terminen los actuales."
                }
            ), 429
        jobs[job_id] = {
            "status": "queued",
            "progress": "En cola...",
            "segments": [],
            "error": None,
            "audio_index": audio_index,
            "_created_at": time.time(),
            "episode_id": episode_id,
            "series_name": series_name,
            "use_original_transcription": use_original_transcription,
        }
    logger.info("Job de proceso creado | job_id=%s", job_id)

    thread = threading.Thread(
        target=_run_job,
        args=(
            job_id,
            long_url,
            short_url,
            model_size,
            long_source,
            long_path,
            audio_index,
            use_original_transcription,
        ),
        daemon=True,
    )
    thread.start()
    logger.info(
        "Thread de proceso iniciada | job_id=%s | thread=%s", job_id, thread.name
    )

    return jsonify({"job_id": job_id})


@app.route("/api/status/<job_id>")
def status(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        logger.warning("Status consultado para job inexistente | job_id=%s", job_id)
        return jsonify({"status": "not_found"}), 404
    logger.info(
        "Status consultado | job_id=%s | snapshot=%s",
        job_id,
        preview(_job_snapshot(job), 1200),
    )
    return jsonify(job)


@app.route("/api/video/<job_id>")
def video(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
    if not job or "long_path" not in job:
        logger.warning("Video solicitado sin job/path valido | job_id=%s", job_id)
        abort(404)

    audio_index_raw = request.args.get("audio_index", job.get("audio_index", 0))
    try:
        audio_index = int(audio_index_raw)
    except (TypeError, ValueError):
        logger.warning(
            "Video solicitado con audio_index invalido | job_id=%s | raw=%s",
            job_id,
            audio_index_raw,
        )
        return jsonify({"error": f"audio_index inválido: {audio_index_raw}"}), 400

    try:
        video_path = _resolve_playback_video(job_id, job, audio_index)
    except RuntimeError as exc:
        logger.exception(
            "No se pudo resolver playback para video | job_id=%s | audio_index=%s",
            job_id,
            audio_index,
        )
        return jsonify({"error": str(exc)}), 400

    mime_type, _ = mimetypes.guess_type(video_path)
    if not mime_type:
        mime_type = "video/mp4"
    logger.info(
        "Sirviendo video | job_id=%s | audio_index=%s | path=%s | mime=%s",
        job_id,
        audio_index,
        video_path,
        mime_type,
    )

    return send_file(video_path, mimetype=mime_type, conditional=True)


@app.route("/api/waveform/<job_id>")
def waveform(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
    if not job or "long_path" not in job:
        logger.warning("Waveform solicitado sin job/path valido | job_id=%s", job_id)
        return jsonify({"error": "Job no encontrado"}), 404

    audio_index_raw = request.args.get("audio_index", job.get("audio_index", 0))
    bins_raw = request.args.get("bins", 4096)
    try:
        audio_index = int(audio_index_raw)
        bins = int(bins_raw)
    except (TypeError, ValueError):
        logger.warning(
            "Waveform solicitado con parametros invalidos | job_id=%s | audio_index=%s | bins=%s",
            job_id,
            audio_index_raw,
            bins_raw,
        )
        return (
            jsonify(
                {
                    "error": f"Parámetros inválidos: audio_index={audio_index_raw}, bins={bins_raw}"
                }
            ),
            400,
        )

    source_path = job.get("original_path") or job.get("long_path")
    try:
        payload = processor.get_waveform_peaks(
            Path(source_path), audio_index=audio_index, bins=bins
        )
    except RuntimeError as exc:
        logger.exception(
            "Waveform fallo | job_id=%s | audio_index=%s | bins=%s | source=%s",
            job_id,
            audio_index,
            bins,
            source_path,
        )
        return jsonify({"error": str(exc)}), 400

    logger.info(
        "Waveform listo | job_id=%s | audio_index=%s | bins=%s | peaks=%s",
        job_id,
        audio_index,
        bins,
        len(payload.get("peaks", [])),
    )
    return jsonify(payload)


@app.route("/api/export", methods=["POST"])
def export():
    data = request.get_json()
    job_id = data.get("job_id")
    segments = data.get("segments", [])
    try:
        pad_before = float(data.get("pad_before", 0))
        pad_after = float(data.get("pad_after", 0))
        audio_index = int(data.get("audio_index", 0))
    except (TypeError, ValueError):
        return jsonify(
            {"error": "pad_before, pad_after y audio_index deben ser números válidos"}
        ), 400
    logger.info(
        "Solicitud de export | job_id=%s | pad_before=%s | pad_after=%s | audio_index=%s | segments=%s",
        job_id,
        pad_before,
        pad_after,
        audio_index,
        preview(summarize_segments(segments), 1200),
    )

    with jobs_lock:
        job = jobs.get(job_id)
    if not job or "long_path" not in job:
        logger.warning("Export fallido por job invalido | job_id=%s", job_id)
        return jsonify({"error": "Job no encontrado"}), 404

    output_path = str(OUTPUTS_DIR / f"{job_id}_export.mp4")
    source_path = job.get("original_path", job["long_path"])

    try:
        processor.export_clip(
            source_path,
            segments,
            output_path,
            pad_before=pad_before,
            pad_after=pad_after,
            audio_index=audio_index,
        )
    except RuntimeError as e:
        logger.exception(
            "Export fallo | job_id=%s | source=%s | output=%s",
            job_id,
            source_path,
            output_path,
        )
        return jsonify({"error": str(e)}), 500

    logger.info(
        "Export completado | job_id=%s | source=%s | output=%s",
        job_id,
        source_path,
        output_path,
    )
    return send_file(
        output_path,
        as_attachment=True,
        download_name="clip_exportado.mp4",
        mimetype="video/mp4",
    )


def _run_job(
    job_id: str,
    long_url: str,
    short_url: str,
    model_size: str,
    long_source: str = "url",
    long_path_raw: str = "",
    audio_index: int = 0,
    use_original_transcription: bool = False,
):
    job_dir = WORK_DIR / job_id
    job_dir.mkdir(exist_ok=True)
    logger.info(
        "Worker de proceso iniciado | job_id=%s | long_url=%s | short_url=%s | model=%s | long_source=%s | long_path_raw=%s | audio_index=%s | job_dir=%s",
        job_id,
        long_url,
        short_url,
        model_size,
        long_source,
        long_path_raw,
        audio_index,
        job_dir,
    )

    def progress(msg: str):
        logger.info("Worker progreso | job_id=%s | msg=%s", job_id, msg)
        update_job(job_id, progress=msg)

    try:
        long_path = None
        original_path = None
        transcription_source_path = None

        if long_source == "local":
            update_job(job_id, status="probing", progress="Analizando video local...")
            local_file = Path(long_path_raw)
            logger.info(
                "Proceso usara archivo local | job_id=%s | path=%s", job_id, local_file
            )
            probe_result = processor.probe_video(local_file)
            english_index = probe_result.get("english_index", 0)
            if audio_index == 0 and english_index != 0:
                logger.info(
                    "Override audio_index para ingles | job_id=%s | old=%s | new=%s",
                    job_id,
                    audio_index,
                    english_index,
                )
                audio_index = english_index
            update_job(
                job_id, streams=probe_result.get("streams", []), audio_index=audio_index
            )
            logger.info(
                "Probe local completado | job_id=%s | result=%s",
                job_id,
                preview(probe_result, 1200),
            )

            original_path = str(local_file)
            suffix = local_file.suffix.lower()
            if suffix not in (".mp4", ".m4v"):
                update_job(
                    job_id,
                    progress="Preparando video para reproduccion (puede tardar)...",
                )
                long_path = str(
                    processor.remux_for_browser(
                        local_file, job_dir / "long.mp4", audio_index=audio_index
                    )
                )
            else:
                long_path = str(local_file)
            transcription_source_path = local_file
            logger.info(
                "Video largo resuelto desde local | job_id=%s | playback_path=%s | original=%s",
                job_id,
                long_path,
                original_path,
            )
            update_job(job_id, long_path=long_path, original_path=original_path)
        else:
            update_job(
                job_id, status="downloading", progress="Descargando video largo..."
            )
            long_path = processor.download_video(
                long_url, job_dir / "long", progress_cb=progress
            )
            logger.info(
                "Video largo descargado | job_id=%s | url=%s | path=%s",
                job_id,
                long_url,
                long_path,
            )

            probe_result = processor.probe_video(long_path)
            english_index = probe_result.get("english_index", 0)
            if audio_index == 0 and english_index != 0:
                logger.info(
                    "Override audio_index para ingles (remoto) | job_id=%s | old=%s | new=%s",
                    job_id,
                    audio_index,
                    english_index,
                )
                audio_index = english_index
            update_job(
                job_id, streams=probe_result.get("streams", []), audio_index=audio_index
            )
            logger.info(
                "Probe remoto completado | job_id=%s | result=%s",
                job_id,
                preview(probe_result, 1200),
            )

            original_path = str(long_path)
            suffix = long_path.suffix.lower()
            if suffix not in (".mp4", ".m4v"):
                remuxed = job_dir / "long.mp4"
                update_job(
                    job_id,
                    progress="Preparando video para reproduccion (puede tardar)...",
                )
                original_path_remux = str(long_path)
                long_path = str(
                    processor.remux_for_browser(
                        long_path, remuxed, audio_index=audio_index
                    )
                )
                logger.info(
                    "Remux completado para reproduccion | job_id=%s | remuxed=%s | original_download=%s",
                    job_id,
                    long_path,
                    original_path_remux,
                )

            transcription_source_path = Path(original_path)
            update_job(job_id, long_path=long_path, original_path=original_path)

        update_job(job_id, progress="Descargando clip corto...")
        short_path = processor.download_video(
            short_url, job_dir / "short", progress_cb=progress
        )
        logger.info(
            "Clip corto descargado | job_id=%s | url=%s | path=%s",
            job_id,
            short_url,
            short_path,
        )

        long_words = []
        if use_original_transcription:
            with jobs_lock:
                job_data = jobs.get(job_id, {})
            ep_id = job_data.get("episode_id", "")
            series = job_data.get("series_name", "")
            if ep_id and series:
                srt_path = episode_matcher.find_srt_for_episode(series, ep_id)
                if srt_path and srt_path.exists():
                    long_words = subtitle_loader.srt_to_words(srt_path)
                    logger.info(
                        "Transcripcion original SRT usada | job_id=%s | srt=%s | words=%s",
                        job_id,
                        srt_path,
                        len(long_words),
                    )
                else:
                    logger.warning(
                        "SRT no encontrado para transcripcion original | job_id=%s | series=%s | ep=%s | fallback=whisper",
                        job_id,
                        series,
                        ep_id,
                    )
                    use_original_transcription = False
            else:
                logger.warning(
                    "episode_id o series_name faltan para transcripcion original | job_id=%s | fallback=whisper",
                    job_id,
                )
                use_original_transcription = False

        if not use_original_transcription:
            update_job(
                job_id,
                status="transcribing",
                progress="Transcribiendo video largo (puede tardar varios minutos)...",
            )
            if transcription_source_path is None:
                transcription_source_path = (
                    Path(original_path) if original_path else long_path
                )
            logger.info(
                "Fuente para transcripcion larga resuelta | job_id=%s | transcription_source=%s | playback_source=%s | audio_index=%s",
                job_id,
                transcription_source_path,
                long_path,
                audio_index,
            )
            long_words = processor.transcribe_video(
                transcription_source_path,
                model_size,
                progress_cb=progress,
                audio_index=audio_index,
            )
            logger.info(
                "Transcripcion larga lista | job_id=%s | resumen=%s",
                job_id,
                preview(processor.summarize_words(long_words), 1000),
            )

        update_job(
            job_id, status="transcribing", progress="Transcribiendo clip corto..."
        )
        short_words = processor.transcribe_video(
            short_path, model_size, progress_cb=progress
        )
        logger.info(
            "Transcripcion corta lista | job_id=%s | resumen=%s",
            job_id,
            preview(processor.summarize_words(short_words), 1000),
        )

        update_job(
            job_id,
            status="matching",
            progress="Buscando segmentos en el video largo...",
        )
        segments = processor.find_matching_segments(short_words, long_words)
        logger.info(
            "Matching completado | job_id=%s | resumen=%s",
            job_id,
            preview(summarize_segments(segments), 1200),
        )

        # ── Guardar residuos para IA ──
        try:
            residuo = {
                "job_id": job_id,
                "model": model_size,
                "audio_index": audio_index,
                "long_source": long_source,
                "long_path": str(long_path),
                "short_url": short_url,
                "episode_transcript": " ".join(w["word"] for w in long_words),
                "episode_words": long_words,
                "clip_transcript": " ".join(w["word"] for w in short_words),
                "clip_words": short_words,
                "matches": [
                    {
                        "clip_text": s.get("text", ""),
                        "episode_text": s.get("text", ""),
                        "episode_start": s.get("start"),
                        "episode_end": s.get("end"),
                        "clip_start": s.get("short_start"),
                        "clip_end": s.get("short_end"),
                        "confidence": s.get("confidence"),
                    }
                    for s in segments
                ],
            }
            residuo_file = RESIDUOS_DIR / f"{job_id}.jsonl"
            with open(residuo_file, "w", encoding="utf-8") as f:
                json.dump(residuo, f, ensure_ascii=False)
            logger.info("Residuo guardado | job_id=%s | file=%s", job_id, residuo_file)
        except Exception as e:
            logger.exception(
                "Error guardando residuo | job_id=%s | error=%s", job_id, e
            )

        if not segments:
            update_job(
                job_id,
                status="done",
                segments=[],
                progress='⚠️ No se encontraron segmentos. Probá con modelo "small" o "medium".',
            )
            return

        update_job(
            job_id,
            status="done",
            segments=segments,
            progress=f"✅ ¡Listo! Se encontraron {len(segments)} segmento(s).",
        )
        logger.info("Worker de proceso completado | job_id=%s", job_id)

    except Exception as e:
        logger.exception("Worker de proceso fallo | job_id=%s | error=%s", job_id, e)
        update_job(job_id, status="error", error=str(e), progress=f"❌ Error: {e}")


def _run_detection(job_id, short_url, series_name, language, model_size):
    job_dir = WORK_DIR / job_id
    job_dir.mkdir(exist_ok=True)
    logger.info(
        "Worker de deteccion iniciado | job_id=%s | short_url=%s | series=%s | language=%s | model=%s | job_dir=%s",
        job_id,
        short_url,
        series_name,
        language,
        model_size,
        job_dir,
    )

    def progress(msg):
        logger.info("Worker detect progreso | job_id=%s | msg=%s", job_id, msg)
        update_job(job_id, progress=msg)

    try:
        update_job(job_id, status="downloading", progress="Descargando clip corto...")
        short_path = processor.download_video(
            short_url, job_dir / "short", progress_cb=progress
        )
        logger.info(
            "Clip corto descargado para deteccion | job_id=%s | path=%s",
            job_id,
            short_path,
        )

        update_job(
            job_id, status="transcribing", progress="Transcribiendo clip corto..."
        )
        short_words = processor.transcribe_video(
            short_path, model_size, progress_cb=progress
        )
        logger.info(
            "Transcripcion de deteccion lista | job_id=%s | resumen=%s",
            job_id,
            preview(processor.summarize_words(short_words), 1000),
        )

        update_job(
            job_id,
            status="matching",
            progress=f"Buscando en episodios de '{series_name}'...",
        )
        results = episode_matcher.detect_episode(
            short_words, series_name, language=language, top_n=3
        )
        logger.info(
            "Resultados de deteccion | job_id=%s | payload=%s",
            job_id,
            preview(results, 1200),
        )

        if not results:
            update_job(
                job_id,
                status="done",
                detection_results=[],
                progress="⚠️ No se encontró ningún episodio que coincida.",
            )
            return

        top = results[0]
        update_job(
            job_id,
            status="done",
            detection_results=results,
            progress=f"✅ Detectado: {top['id']} — {top['title']} ({top['score']}%)",
        )
        logger.info(
            "Worker de deteccion completado | job_id=%s | top=%s",
            job_id,
            preview(top, 600),
        )

    except Exception as e:
        logger.exception("Worker de deteccion fallo | job_id=%s | error=%s", job_id, e)
        update_job(job_id, status="error", error=str(e), progress=f"❌ Error: {e}")


@app.route("/api/process_batch", methods=["POST"])
def process_batch():
    data = request.get_json()
    short_urls = (data or {}).get("short_urls", [])
    long_path = (data or {}).get("long_path", "").strip()
    model_size = (data or {}).get("model", "base")
    audio_index = (data or {}).get("audio_index", 0)
    episode_id = (data or {}).get("episode_id", "")
    series_name = (data or {}).get("series_name", "")

    if not short_urls or not isinstance(short_urls, list):
        return jsonify({"error": "Se requiere una lista de short_urls"}), 400

    if not long_path:
        return jsonify({"error": "Se requiere la ruta del video largo"}), 400

    job_id = str(uuid.uuid4())[:8]
    with jobs_lock:
        if _active_job_count() >= MAX_CONCURRENT_JOBS:
            return jsonify(
                {
                    "error": f"Demasiados jobs activos (max {MAX_CONCURRENT_JOBS}). Espera."
                }
            ), 429
        jobs[job_id] = {
            "status": "queued",
            "progress": f"Procesando {len(short_urls)} clips en batch...",
            "segments": [],
            "error": None,
            "audio_index": audio_index,
            "_created_at": time.time(),
            "episode_id": episode_id,
            "series_name": series_name,
            "batch_urls": len(short_urls),
        }

    thread = threading.Thread(
        target=_run_batch,
        args=(
            job_id,
            short_urls,
            long_path,
            model_size,
            audio_index,
            episode_id,
            series_name,
        ),
        daemon=True,
    )
    thread.start()
    logger.info("Batch job creado | job_id=%s | urls=%s", job_id, len(short_urls))
    return jsonify({"job_id": job_id})


@app.route("/api/preprocess_series", methods=["POST"])
def preprocess_series():
    data = request.get_json()
    series_name = (data or {}).get("series_name", "").strip()
    model_size = (data or {}).get("model", "base")

    if not series_name:
        return jsonify({"error": "Se requiere el nombre de la serie"}), 400

    registry = load_series_registry()
    series = None
    for s in registry:
        if s["name"].lower() == series_name.lower():
            series = s
            break
    if not series:
        return jsonify({"error": f"Serie '{series_name}' no encontrada"}), 404

    from subtitle_loader import scan_episodes

    episodes = scan_episodes(series)
    episodes_with_video = [e for e in episodes if e.get("video_path")]

    if not episodes_with_video:
        return jsonify({"error": "No hay episodios con video disponible"}), 400

    job_id = str(uuid.uuid4())[:8]
    with jobs_lock:
        if _active_job_count() >= MAX_CONCURRENT_JOBS:
            return jsonify(
                {
                    "error": f"Demasiados jobs activos (max {MAX_CONCURRENT_JOBS}). Espera."
                }
            ), 429
        jobs[job_id] = {
            "status": "preprocessing",
            "progress": f"Pre-procesando {len(episodes_with_video)} episodios de '{series_name}'...",
            "total_episodes": len(episodes_with_video),
            "completed_episodes": 0,
            "error": None,
            "_created_at": time.time(),
        }

    thread = threading.Thread(
        target=_run_preprocess,
        args=(job_id, episodes_with_video, model_size),
        daemon=True,
    )
    thread.start()
    logger.info(
        "Preprocess job creado | job_id=%s | episodes=%s",
        job_id,
        len(episodes_with_video),
    )
    return jsonify({"job_id": job_id, "total_episodes": len(episodes_with_video)})


def _run_batch(
    job_id, short_urls, long_path, model_size, audio_index, episode_id, series_name
):
    job_dir = WORK_DIR / job_id
    job_dir.mkdir(exist_ok=True)
    logger.info("Worker batch iniciado | job_id=%s | urls=%s", job_id, len(short_urls))

    def progress(msg):
        logger.info("Worker batch progreso | job_id=%s | msg=%s", job_id, msg)
        update_job(job_id, progress=msg)

    try:
        local_file = Path(long_path)
        probe_result = processor.probe_video(local_file)
        english_index = probe_result.get("english_index", 0)
        if audio_index == 0 and english_index != 0:
            audio_index = english_index
        update_job(
            job_id, streams=probe_result.get("streams", []), audio_index=audio_index
        )

        suffix = local_file.suffix.lower()
        if suffix not in (".mp4", ".m4v"):
            progress(f"Preparando video largo ({suffix} -> mp4)...")
            long_playback = str(
                processor.remux_for_browser(
                    local_file, job_dir / "long.mp4", audio_index=audio_index
                )
            )
        else:
            long_playback = str(local_file)
        update_job(job_id, long_path=long_playback, original_path=str(local_file))

        progress("Transcribiendo video largo (se usa una sola vez)...")
        long_words = processor.transcribe_video(
            local_file, model_size, progress_cb=None, audio_index=audio_index
        )
        progress(
            f"Video largo transcrito ({len(long_words)} palabras). Procesando clips..."
        )

        all_segments = {}
        for idx, url in enumerate(short_urls):
            progress(f"Clip {idx + 1}/{len(short_urls)}: descargando...")
            update_job(job_id, status="downloading")
            short_path = processor.download_video(
                url, job_dir / f"short_{idx}", progress_cb=None
            )
            short_words = processor.transcribe_video(
                short_path, model_size, progress_cb=None
            )

            srt_hint = None
            try:
                if episode_id and series_name:
                    from subtitle_loader import (
                        load_series_registry,
                        scan_episodes,
                        load_subtitles,
                    )

                    registry = load_series_registry()
                    for series in registry:
                        if series["name"].lower() == (series_name or "").lower():
                            episodes = scan_episodes(series)
                            for ep in episodes:
                                if ep["id"] == episode_id and ep.get("subtitle_path"):
                                    entries = load_subtitles(ep["subtitle_path"])
                                    if entries:
                                        srt_hint = (
                                            entries[0]["start"],
                                            entries[-1]["end"],
                                        )
                                    break
                            break
            except Exception:
                pass

            progress(f"Clip {idx + 1}/{len(short_urls)}: buscando coincidencias...")
            update_job(job_id, status="matching")
            segments = processor.find_matching_segments(
                short_words, long_words, srt_time_range=srt_hint
            )
            all_segments[url] = segments

        total = sum(len(v) for v in all_segments.values())
        flat_segments = []
        for segs in all_segments.values():
            flat_segments.extend(segs)

        try:
            residuo = {
                "job_id": job_id,
                "type": "batch",
                "model": model_size,
                "long_path": str(long_playback),
                "short_urls": short_urls,
                "matches": {
                    url: [
                        {
                            "start": s.get("start"),
                            "end": s.get("end"),
                            "confidence": s.get("confidence"),
                        }
                        for s in segs
                    ]
                    for url, segs in all_segments.items()
                },
            }
            residuo_file = RESIDUOS_DIR / f"{job_id}.json"
            with open(residuo_file, "w", encoding="utf-8") as f:
                json.dump(residuo, f, ensure_ascii=False)
        except Exception:
            pass

        update_job(
            job_id,
            status="done",
            segments=flat_segments,
            progress=f"\u2705 Batch completo: {total} segmentos en {len(short_urls)} clip(s)",
            batch_results=all_segments,
        )
    except Exception as e:
        logger.exception("Worker batch fallo | job_id=%s | error=%s", job_id, e)
        update_job(job_id, status="error", error=str(e), progress=f"\u274c Error: {e}")


def _run_preprocess(job_id, episodes, model_size):
    logger.info(
        "Worker preprocess iniciado | job_id=%s | episodes=%s", job_id, len(episodes)
    )

    try:
        for idx, ep in enumerate(episodes):
            video_path = Path(ep["video_path"])
            if not video_path.exists():
                logger.warning(
                    "Preprocess video no encontrado | episode=%s | path=%s",
                    ep["id"],
                    ep["video_path"],
                )
                update_job(
                    job_id,
                    completed_episodes=idx,
                    progress=f"\u26a0\ufe0f {ep['id']}: video no encontrado, saltando...",
                )
                continue

            update_job(
                job_id,
                completed_episodes=idx,
                progress=f"Transcribiendo {ep['id']} ({idx + 1}/{len(episodes)})...",
                status="transcribing",
            )

            try:
                processor.transcribe_video(video_path, model_size, progress_cb=None)
                update_job(
                    job_id,
                    progress=f"\u2705 {ep['id']} transcrito ({idx + 1}/{len(episodes)})",
                )
            except Exception as e:
                logger.warning(
                    "Preprocess fallo para episodio | episode=%s | error=%s",
                    ep["id"],
                    e,
                )
                update_job(
                    job_id,
                    progress=f"\u26a0\ufe0f {ep['id']}: error de transcripcion, saltando...",
                )
                continue

        update_job(
            job_id,
            status="done",
            completed_episodes=len(episodes),
            progress=f"\u2705 Pre-procesamiento completo: {len(episodes)} episodios procesados",
        )
    except Exception as e:
        logger.exception("Worker preprocess fallo | job_id=%s | error=%s", job_id, e)
        update_job(job_id, status="error", error=str(e), progress=f"\u274c Error: {e}")


if __name__ == "__main__":
    logger.info("Arrancando Flask | port=5050 | threaded=True | use_reloader=False")
    app.run(debug=False, port=5050, threaded=True, use_reloader=False)
