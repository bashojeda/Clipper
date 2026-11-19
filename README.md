# Clipper AI

Herramienta local de línea de comandos para convertir videos largos en clips verticales cortos con subtítulos. Este repositorio comienza con la Fase 0: una prueba de concepto, sin API web ni frontend.

## Requisitos

- Python 3.11 o superior. Las dependencias principales se comprobaron con Python 3.14 en Windows.
- FFmpeg disponible en `PATH` para renderizar los clips. `faster-whisper` decodifica el audio usando PyAV, pero el renderizado final sí invoca FFmpeg.
- Una clave de API de Anthropic para seleccionar los momentos destacados.

## Preparar el entorno en Windows

Desde PowerShell, en la carpeta del proyecto:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Si la política de PowerShell impide activar el entorno, se puede invocar directamente `.\.venv\Scripts\python.exe` en lugar de `python`.

Copia `.env.example` como `.env`; agrega tu clave y un identificador de modelo Anthropic habilitado para tu cuenta. No subas `.env` a Git. Si la terminal no encuentra FFmpeg por `PATH`, también puedes asignar su ruta completa a `FFMPEG_PATH`.

## Ejecutar la CLI

```powershell
clipper .\ruta\al\video.mp4 --output .\outputs --language es
```

La primera ejecución descarga el modelo de faster-whisper y puede tardar. La transcripción de audio se hace localmente; el texto transcrito y sus tiempos se envían a Anthropic para seleccionar los momentos. Los clips se generan en vertical 9:16 y se guardan en la carpeta indicada. Para consultar opciones:

```powershell
clipper --help
```

## Pruebas

```powershell
python -m pytest -q
```

## Estructura

- `src/clipper/`: paquete de la aplicación.
- `tests/`: pruebas automatizadas.
- `pyproject.toml`: metadatos y dependencias.

La aplicación tiene una CLI y no incluye todavía API web ni frontend.
