"""Render selected moments as vertical clips with burned-in subtitles."""

import html
import os
import shutil
import subprocess
from pathlib import Path
from collections.abc import Sequence

from clipper.moments import ClipMoment
from clipper.transcribe import WordTimestamp


def render_clips(
    video_path: Path,
    output_dir: Path,
    moments: Sequence[ClipMoment],
    words: Sequence[WordTimestamp],
) -> list[Path]:
    """Cut three moments, convert them to 9:16, and burn in subtitles."""
    if not video_path.is_file():
        raise FileNotFoundError(f"No se encontró el video: {video_path}")
    if len(moments) != 3:
        raise ValueError("Se requieren exactamente tres momentos para exportar")

    ffmpeg_path = shutil.which(os.getenv("FFMPEG_PATH") or "ffmpeg")
    if ffmpeg_path is None:
        raise FileNotFoundError(
            "No se encontró FFmpeg. Agrégalo a PATH o configura FFMPEG_PATH en .env"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    rendered_paths: list[Path] = []

    for index, moment in enumerate(moments, start=1):
        duration = moment.end_seconds - moment.start_seconds
        subtitle_path = output_dir / f"clip_{index:02d}.srt"
        output_path = output_dir / f"clip_{index:02d}.mp4"
        subtitle_cues = _build_subtitle_cues(words, moment)
        if not subtitle_cues:
            raise ValueError(f"No hay palabras para subtitular el clip {index}")

        _write_subtitles(subtitle_path, subtitle_cues)
        video_filter = (
            "scale=1080:1920:force_original_aspect_ratio=increase:flags=lanczos,"
            "crop=1080:1920,setsar=1,"
            f"subtitles={subtitle_path.name}:"
            "force_style='FontName=Arial,FontSize=18,PrimaryColour=&H00FFFFFF,"
            "OutlineColour=&H00000000,Outline=2,Shadow=1,Alignment=2,MarginV=100'"
        )
        command = [
            ffmpeg_path,
            "-y",
            "-i",
            str(video_path.resolve()),
            "-ss",
            f"{moment.start_seconds:.3f}",
            "-t",
            f"{duration:.3f}",
            "-map",
            "0:v:0",
            "-map",
            "0:a?",
            "-vf",
            video_filter,
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-movflags",
            "+faststart",
            str(output_path.resolve()),
        ]

        try:
            subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                cwd=output_dir,
            )
        except subprocess.CalledProcessError as error:
            details = (error.stderr or "").strip()[-1000:]
            raise RuntimeError(
                f"FFmpeg no pudo renderizar {output_path.name}: {details}"
            ) from error

        subtitle_path.unlink(missing_ok=True)
        rendered_paths.append(output_path)

    return rendered_paths


def _build_subtitle_cues(
    words: Sequence[WordTimestamp],
    moment: ClipMoment,
) -> list[tuple[float, float, str]]:
    """Group nearby words into short, readable subtitle cues."""
    cues: list[tuple[float, float, str]] = []
    cue_words: list[str] = []
    cue_start = 0.0
    cue_end = 0.0

    def flush_cue() -> None:
        if cue_words:
            cues.append((cue_start, cue_end, " ".join(cue_words)))
            cue_words.clear()

    for word in words:
        if word.end_seconds <= moment.start_seconds:
            continue
        if word.start_seconds >= moment.end_seconds:
            break

        word_start = max(0.0, word.start_seconds - moment.start_seconds)
        word_end = min(moment.end_seconds, word.end_seconds) - moment.start_seconds
        if word_end <= word_start:
            continue

        if cue_words and (
            len(cue_words) >= 5 or word_start - cue_start >= 3.0
        ):
            flush_cue()

        if not cue_words:
            cue_start = word_start
        cue_end = word_end
        cue_words.append(html.escape(word.text.strip(), quote=False))

    flush_cue()
    return cues


def _write_subtitles(
    subtitle_path: Path,
    cues: Sequence[tuple[float, float, str]],
) -> None:
    """Write subtitle cues in SubRip (SRT) format."""
    entries = []
    for index, (start, end, text) in enumerate(cues, start=1):
        entries.append(
            f"{index}\n"
            f"{_format_srt_timestamp(start)} --> {_format_srt_timestamp(end)}\n"
            f"{text}"
        )
    subtitle_path.write_text("\n\n".join(entries) + "\n", encoding="utf-8")


def _format_srt_timestamp(seconds: float) -> str:
    """Format nonnegative seconds as an SRT timestamp."""
    milliseconds = max(0, round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02}:{minutes:02}:{whole_seconds:02},{milliseconds:03}"
