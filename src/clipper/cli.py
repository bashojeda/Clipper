"""Command-line entry point for the Phase 0 video clipping workflow."""

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from anthropic import AnthropicError
from dotenv import load_dotenv

from clipper.moments import select_best_moments
from clipper.render import render_clips
from clipper.transcribe import transcribe_video


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the clipper command."""
    parser = argparse.ArgumentParser(
        prog="clipper",
        description="Genera tres clips verticales subtitulados a partir de un video.",
    )
    parser.add_argument("video", type=Path, help="Ruta del video de entrada")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("outputs"),
        help="Carpeta donde se guardarán los clips (por defecto: outputs)",
    )
    parser.add_argument(
        "--model-size",
        default="small",
        help="Tamaño del modelo faster-whisper (por defecto: small)",
    )
    parser.add_argument(
        "--language",
        help="Código de idioma opcional, por ejemplo es o en",
    )
    parser.add_argument(
        "--anthropic-model",
        default=os.getenv("ANTHROPIC_MODEL"),
        help="Modelo de Anthropic; también se puede configurar con ANTHROPIC_MODEL",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run transcription, moment selection, and clip rendering."""
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.video.is_file():
        parser.error(f"no se encontró el video: {args.video}")
    if not os.getenv("ANTHROPIC_API_KEY"):
        parser.error("configura ANTHROPIC_API_KEY en el archivo .env")
    if not args.anthropic_model:
        parser.error("configura ANTHROPIC_MODEL en el archivo .env")

    try:
        print("[1/3] Transcribiendo audio con faster-whisper...", file=sys.stderr)
        words = transcribe_video(
            args.video,
            model_size=args.model_size,
            language=args.language,
        )
        print(f"      {len(words)} palabras transcritas.", file=sys.stderr)

        print("[2/3] Seleccionando los mejores momentos con Anthropic...", file=sys.stderr)
        selected_moments = select_best_moments(
            words,
            model=args.anthropic_model,
        )

        print("[3/3] Renderizando clips verticales con FFmpeg...", file=sys.stderr)
        clip_paths = render_clips(
            args.video,
            args.output,
            selected_moments,
            words,
        )
    except (AnthropicError, OSError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print("\nClips exportados:")
    for clip_path in clip_paths:
        print(f"- {clip_path}")
    return 0
