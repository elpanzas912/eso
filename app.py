"""
app.py - ClipFinder Flask backend
"""

import mimetypes
import os
import uuid
import json
import threading
from pathlib import Path
from flask import Flask, jsonify, request, send_file, render_template, abort

import processor
from processor import WORK_DIR
import episode_matcher
from subtitle_loader import load_series_registry, save_series_registry

RESIDUOS_DIR = Path(__file__).resolve().parent / "residuos para ia"
RESIDUOS_DIR.mkdir(parents=True, exist_ok=True)

OUTPUTS_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)

WORK_DIR.mkdir(parents=True, exist_ok=True)

jobs: dict[str, dict] = {}
jobs_lock = threading.Lock()


def update_job(job_id: str, **kwargs):
    with jobs_lock:
        jobs[job_id].update(kwargs)


@app.route("/")
def index():
    return render_template("detect.html")


@app.route("/example")
def example():
    return render_template("example.html")


@app.route("/detect")
def detect():
    return render_template("detect.html")


@app.route("/editor")
def editor():
    return render_template("index.html")


@app.route("/api/series", methods=["GET"])
def list_series():
    return jsonify(episode_matcher.list_series())


@app.route("/api/series", methods=["POST"])
def add_series():
    data = request.get_json()
    name = (data or {}).get("name", "").strip()
    subs_path = (data or {}).get("subtitles_path", "").strip()
    vids_path = (data or {}).get("videos_path", "").strip()
    language = (data or {}).get("language", "en").strip()

    if not name or not subs_path:
        return jsonify({"error": "Nombre y ruta de subtítulos son obligatorios"}), 400

    if not Path(subs_path).exists():
        return jsonify({"error": f"Ruta de subtítulos no encontrada: {subs_path}"}), 400

    try:
        series = episode_matcher.add_series(name, subs_path, vids_path, language)
        return jsonify(series)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/detect", methods=["POST"])
def detect_episode():
    data = request.get_json()
    short_url = (data or {}).get("short_url", "").strip()
    series_name = (data or {}).get("series_name", "").strip()
    language = (data or {}).get("language", "en").strip()
    model = (data or {}).get("model", "base").strip()

    if not short_url or not series_name:
        return jsonify(
            {"error": "URL del clip y nombre de serie son obligatorios"}
        ), 400

    job_id = uuid.uuid4().hex[:8]
    with jobs_lock:
        jobs[job_id] = {"status": "detecting", "progress": "Descargando clip corto..."}

    thread = threading.Thread(
        target=_run_detection,
        args=(job_id, short_url, series_name, language, model),
    )
    thread.daemon = True
    thread.start()

    return jsonify({"job_id": job_id})


@app.route("/api/probe", methods=["POST"])
def probe():
    data = request.get_json()
    path = (data or {}).get("path", "").strip()
    if not path:
        return jsonify({"error": "Se requiere una ruta de archivo"}), 400

    path_obj = Path(path)
    if not path_obj.exists():
        return jsonify({"error": f"Archivo no encontrado: {path}"}), 400

    try:
        result = processor.probe_video(path_obj)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/process", methods=["POST"])
def process():
    data = request.get_json()
    long_url = (data or {}).get("long_url", "").strip()
    short_url = (data or {}).get("short_url", "").strip()
    long_path = (data or {}).get("long_path", "").strip()
    long_source = (data or {}).get("long_source", "url")
    audio_index = (data or {}).get("audio_index", 0)
    model_size = (data or {}).get("model", "base")

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
        jobs[job_id] = {
            "status": "queued",
            "progress": "En cola...",
            "segments": [],
            "error": None,
            "audio_index": audio_index,
        }

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
        ),
        daemon=True,
    )
    thread.start()

    return jsonify({"job_id": job_id})


@app.route("/api/status/<job_id>")
def status(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        return jsonify({"status": "not_found"}), 404
    return jsonify(job)


@app.route("/api/video/<job_id>")
def video(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
    if not job or "long_path" not in job:
        abort(404)

    video_path = job["long_path"]
    mime_type, _ = mimetypes.guess_type(video_path)
    if not mime_type:
        mime_type = "video/mp4"

    return send_file(video_path, mimetype=mime_type, conditional=True)


@app.route("/api/export", methods=["POST"])
def export():
    data = request.get_json()
    job_id = data.get("job_id")
    segments = data.get("segments", [])
    pad_before = float(data.get("pad_before", 0))
    pad_after = float(data.get("pad_after", 0))
    audio_index = int(data.get("audio_index", 0))

    with jobs_lock:
        job = jobs.get(job_id)
    if not job or "long_path" not in job:
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
        return jsonify({"error": str(e)}), 500

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
):
    job_dir = WORK_DIR / job_id
    job_dir.mkdir(exist_ok=True)

    def progress(msg: str):
        print(f"[ClipFinder] Job {job_id}: {msg}")
        update_job(job_id, progress=msg)

    try:
        long_path = None
        original_path = None

        if long_source == "local":
            update_job(job_id, status="probing", progress="Analizando video local...")
            local_file = Path(long_path_raw)
            probe_result = processor.probe_video(local_file)
            update_job(
                job_id, streams=probe_result.get("streams", []), audio_index=audio_index
            )

            original_path = str(local_file)
            suffix = local_file.suffix.lower()
            if suffix not in (".mp4", ".m4v"):
                update_job(
                    job_id,
                    progress="Preparando video para reproduccion (puede tardar)...",
                )
                long_path = processor.remux_for_browser(
                    local_file, job_dir / "long.mp4", audio_index=audio_index
                )
            else:
                long_path = local_file
            update_job(job_id, long_path=str(long_path), original_path=original_path)
        else:
            update_job(
                job_id, status="downloading", progress="Descargando video largo..."
            )
            long_path = processor.download_video(
                long_url, job_dir / "long", progress_cb=progress
            )

            probe_result = processor.probe_video(long_path)
            update_job(
                job_id, streams=probe_result.get("streams", []), audio_index=audio_index
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
                long_path = processor.remux_for_browser(
                    long_path, remuxed, audio_index=audio_index
                )

            update_job(job_id, long_path=str(long_path), original_path=original_path)

        update_job(job_id, progress="Descargando clip corto...")
        short_path = processor.download_video(
            short_url, job_dir / "short", progress_cb=progress
        )

        update_job(
            job_id,
            status="transcribing",
            progress="Transcribiendo video largo (puede tardar varios minutos)...",
        )
        long_words = processor.transcribe_video(
            long_path, model_size, progress_cb=progress, audio_index=audio_index
        )

        update_job(job_id, progress="Transcribiendo clip corto...")
        short_words = processor.transcribe_video(
            short_path, model_size, progress_cb=progress
        )

        update_job(
            job_id,
            status="matching",
            progress="Buscando segmentos en el video largo...",
        )
        segments = processor.find_matching_segments(short_words, long_words)

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
            print(f"[ClipFinder] Residuo guardado: {residuo_file}")
        except Exception as e:
            print(f"[ClipFinder] Error guardando residuo: {e}")

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

    except Exception as e:
        import traceback

        print(f"[ClipFinder] Job {job_id} ERROR: {e}")
        print(traceback.format_exc())
        update_job(job_id, status="error", error=str(e), progress=f"❌ Error: {e}")


if __name__ == "__main__":
    app.run(debug=False, port=5050, threaded=True, use_reloader=False)


def _run_detection(job_id, short_url, series_name, language, model_size):
    job_dir = WORK_DIR / job_id
    job_dir.mkdir(exist_ok=True)

    def progress(msg):
        print(f"[ClipFinder] Detect {job_id}: {msg}")
        update_job(job_id, progress=msg)

    try:
        update_job(job_id, status="detecting", progress="Descargando clip corto...")
        short_path = processor.download_video(
            short_url, job_dir / "short", progress_cb=progress
        )

        update_job(job_id, progress="Transcribiendo clip corto...")
        short_words = processor.transcribe_video(
            short_path, model_size, progress_cb=progress
        )

        update_job(job_id, progress=f"Buscando en episodios de '{series_name}'...")
        results = episode_matcher.detect_episode(
            short_words, series_name, language=language, top_n=3
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

    except Exception as e:
        import traceback

        print(f"[ClipFinder] Detect {job_id} ERROR: {e}")
        print(traceback.format_exc())
        update_job(job_id, status="error", error=str(e), progress=f"❌ Error: {e}")
