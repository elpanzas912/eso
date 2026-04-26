# ClipFinder

Herramienta para detectar de qué episodio de una serie proviene un clip corto (YouTube Short, TikTok, Reel) y encontrar el timestamp exacto dentro del episodio completo.

## Flujo

1. **Detección de episodio** (`/`): Ingresás la URL del clip corto y elegís la serie. El sistema transcribe el clip con Whisper (GPU/CUDA), compara el texto contra los subtítulos (.srt) de todos los episodios usando rapidfuzz, y devuelve los 3 episodios más probables con su porcentaje de coincidencia.

2. **Editor de segmentos** (`/editor`): Carga el episodio detectado, transcribe ambos (clip corto + episodio completo) con Whisper, busca los segmentos del episodio que coinciden con el clip, y los muestra en una timeline interactiva para previsualizar y exportar.

3. **Exportación**: Los segmentos seleccionados se exportan como un video recortado a la carpeta `outputs/`.

## Stack

- **Backend**: Flask (Python) — `app.py`
- **Transcripción**: OpenAI Whisper con GPU (NVIDIA RTX 3090, CUDA/FP16) — `processor.py`
- **Matching de episodios**: rapidfuzz contra subtítulos SRT — `episode_matcher.py`
- **Matching de segmentos**: rapidfuzz contra transcripción Whisper — `processor.py`
- **Carga de subtítulos**: Parser SRT con caché en disco — `subtitle_loader.py`
- **Frontend**: HTML/CSS/JS vanilla con estética dark — `templates/detect.html`, `templates/index.html`

## Estructura

```
├── app.py                  # Flask backend, rutas, jobs
├── processor.py            # Whisper, matching de segmentos, export
├── episode_matcher.py      # Detección de episodio via subtítulos
├── subtitle_loader.py      # Parser SRT, caché, escaneo de directorios
├── series_registry.json    # Series configuradas (nombre, rutas subs/videos)
├── requirements.txt
├── templates/
│   ├── detect.html         # Página de detección de episodios
│   └── index.html          # Editor de segmentos
├── work/                   # Cache, modelos Whisper, descargas temporales
└── outputs/                # Videos exportados
```

## Configuración de series

Las series se configuran en `series_registry.json` o desde la UI. Cada serie tiene:
- `name`: Nombre de la serie
- `subtitles_path`: Carpeta raíz con subtítulos (estructura: `Season XX/Carpeta episodio/archivo.en.srt`)
- `videos_path`: Carpeta raíz con videos (estructura: `Season XX/archivo.mkv`)

## Problema actual

### El auto-start del editor no funciona

Cuando el usuario detecta un episodio en `/` y hace click en "Ir al editor", los datos del episodio detectado (ruta del video, URL del clip, modelo) deben llegar al editor en `/editor` para que se procese automáticamente sin intervención manual.

**Lo que debería pasar:**
1. En `detect.html`, click "Ir al editor" → `POST /api/handoff` guarda datos en el servidor → navega a `/editor?from=CLAVE`
2. En `index.html`, el IIFE `autoStart()` detecta `?from=CLAVE` → `GET /api/handoff/CLAVE` → llena campos → ejecuta `probeFile()` → ejecuta `startProcess()`

**Lo que pasa:**
- El `POST /api/handoff` funciona (devuelve key)
- La navegación a `/editor?from=CLAVE` funciona (HTTP 200 OK)
- Pero el navegador **nunca hace** el `GET /api/handoff/CLAVE` para recuperar los datos
- El `index.html` tiene el código correcto, el servidor sirve la versión correcta con headers no-cache
- El IIFE `autoStart()` debería ejecutarse pero aparentemente no lo hace, o sale antes de hacer el fetch
- No se ve ningún toast en pantalla ni error en consola

**Hipótesis posibles:**
- Puede ser un error de JS antes del IIFE que impide que se ejecute
- Puede ser que el navegador cachee agresivamente el JS inline dentro del HTML
- Puede ser que `__pycache__` cachee una versión vieja de la plantilla

**Próximo paso para debuggear:**
- Abrir DevTools (F12) → Console al cargar `/editor?from=...` y verificar si aparece `[AUTO-START] fromKey:`
- Si no aparece, hay un error de JS antes de esa línea que frena la ejecución
- Si aparece pero no sigue, ver en qué línea se detiene

## Requisitos

- Python 3.10+
- ffmpeg instalado en el PATH (`ffmpeg -version` debe funcionar)
- NVIDIA GPU con CUDA (para aceleración Whisper)
- ~2 GB RAM mínimo (más si usás modelo `medium`)

## Instalación

```bash
pip install -r requirements.txt
```

## Uso

```bash
python app.py
# Abre http://localhost:5050 en el navegador
```

## Modelos Whisper

| Modelo | Velocidad | Precisión | RAM  |
|--------|-----------|-----------|------|
| tiny   | muy rápido | baja | ~1 GB |
| base   | rápido | media | ~1 GB |
| small  | moderado | buena | ~2 GB |
| medium | lento | alta | ~5 GB |

## Controles del editor

| Tecla | Acción |
|-------|--------|
| `Space` | Play/Pause |
| `←` / `→` | Avanzar 1 segundo |
| `Shift+←` / `Shift+→` | Avanzar 10 segundos |
| `A` | Agregar segmento en posición actual |
| `Delete` | Eliminar segmento seleccionado |
