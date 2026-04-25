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

app = Flask(__name__)

WORK_DIR = Path("work")
WORK_DIR.mkdir(exist_ok=True)

jobs: dict[str, dict] = {}
jobs_lock = threading.Lock()


def update_job(job_id: str, **kwargs):
    with jobs_lock:
        jobs[job_id].update(kwargs)


@app.route("/")
def index():
    return render_template("index.html")


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

    with jobs_lock:
        job = jobs.get(job_id)
    if not job or "long_path" not in job:
        return jsonify({"error": "Job no encontrado"}), 404

    output_path = str(WORK_DIR / job_id / "export.mp4")

    try:
        processor.export_clip(job["long_path"], segments, output_path)
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
        update_job(job_id, progress=msg)

    try:
        long_path = None

        if long_source == "local":
            update_job(job_id, status="probing", progress="Analizando video local...")
            local_file = Path(long_path_raw)
            probe_result = processor.probe_video(local_file)
            update_job(
                job_id, streams=probe_result.get("streams", []), audio_index=audio_index
            )

            suffix = local_file.suffix.lower()
            if suffix not in (".mp4", ".m4v"):
                update_job(job_id, progress="Preparando video para reproduccion...")
                long_path = processor.remux_for_browser(
                    local_file, job_dir / "long.mp4"
                )
            else:
                long_path = local_file
            update_job(job_id, long_path=str(long_path))
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

            suffix = long_path.suffix.lower()
            if suffix not in (".mp4", ".m4v"):
                remuxed = job_dir / "long.mp4"
                update_job(job_id, progress="Preparando video para reproduccion...")
                long_path = processor.remux_for_browser(long_path, remuxed)

            update_job(job_id, long_path=str(long_path))

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

        update_job(job_id, status="error", error=str(e), progress=f"❌ Error: {e}")
        print(traceback.format_exc())


if __name__ == "__main__":
    app.run(debug=False, port=5050, threaded=True, use_reloader=False)
