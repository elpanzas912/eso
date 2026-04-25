# ClipFinder 🎬

Editor de segmentos con detección automática por transcripción.
Encuentra en un video largo los fragmentos que componen un clip corto.

## Cómo funciona

1. **Descarga** ambos videos con yt-dlp (acepta YouTube, YouTube Shorts, etc.)
2. **Transcribe** ambos con Whisper (timestamps a nivel de palabra)
3. **Matchea** el texto del clip corto contra el video largo con fuzzy matching
4. **Editor visual** tipo LosslessCut para ajustar y exportar sin re-encodear

## Requisitos

- Python 3.10+
- ffmpeg instalado en el PATH (`ffmpeg -version` debe funcionar)
- ~2 GB RAM mínimo (más si usás modelo `medium`)

## Instalación

```bash
# 1. Crear entorno virtual (recomendado)
python -m venv venv
source venv/bin/activate       # Linux/Mac
# venv\Scripts\activate        # Windows

# 2. Instalar dependencias
pip install -r requirements.txt

# En Mac con Apple Silicon (M1/M2/M3):
pip install openai-whisper --extra-index-url https://download.pytorch.org/whl/cpu
```

## Uso

```bash
python app.py
# Abre http://localhost:5050 en el navegador
```

Ingresá:
- **URL del video largo** — el episodio completo (40 min)
- **URL del clip corto** — el YouTube Short (1 min)
- **Modelo Whisper** — `base` es el punto de equilibrio (recomendado para empezar)

## Modelos Whisper

| Modelo | Velocidad | Precisión | RAM  |
|--------|-----------|-----------|------|
| tiny   | ⚡⚡⚡ muy rápido | ★★☆ | ~1 GB |
| base   | ⚡⚡ rápido | ★★★ | ~1 GB |
| small  | ⚡ moderado | ★★★★ | ~2 GB |
| medium | 🐢 lento | ★★★★★ | ~5 GB |

Para un video de 40 min con modelo `base` esperá ~5-15 min de transcripción
(depende de tu CPU/GPU).

## Controles del editor

| Tecla | Acción |
|-------|--------|
| `Space` | Play/Pause |
| `←` / `→` | Avanzar 1 segundo |
| `Shift+←` / `Shift+→` | Avanzar 10 segundos |
| `A` | Agregar segmento en posición actual |
| `Delete` | Eliminar segmento seleccionado |

- **Click** en el timeline → salta a esa posición
- **Drag** en un segmento → moverlo
- **Drag en los bordes** → ajustar inicio/fin
- **Exportar** → genera MP4 lossless con `ffmpeg -c copy`

## Tips

- Si no encuentra segmentos, probá un modelo más grande (`small` o `medium`)
- Funciona mejor con contenido en el mismo idioma en ambos videos
- Los segmentos se pueden ajustar manualmente si la detección no es exacta
- Podés agregar segmentos manuales con el botón `+ Seg aquí`
