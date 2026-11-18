"""Transcribe a video and return timestamps for each recognized word."""

from dataclasses import dataclass
from pathlib import Path

from faster_whisper import WhisperModel


@dataclass(frozen=True)
class WordTimestamp:
    """A recognized word and its start and end times in seconds."""

    text: str
    start_seconds: float
    end_seconds: float


def transcribe_video(
    video_path: Path,
    model_size: str = "small",
    language: str | None = None,
) -> list[WordTimestamp]:
    """Transcribe the audio track and return word-level timestamps.

    The model runs on the CPU with int8 quantization so it works without a
    supported NVIDIA GPU. faster-whisper downloads the selected model the first
    time it is used.
    """
    if not video_path.is_file():
        raise FileNotFoundError(f"No se encontró el video: {video_path}")

    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, _ = model.transcribe(
        str(video_path),
        word_timestamps=True,
        vad_filter=True,
        language=language,
    )

    words: list[WordTimestamp] = []
    for segment in segments:
        if segment.words is None:
            continue

        for word in segment.words:
            text = word.word.strip()
            if not text or word.start is None or word.end is None:
                continue

            words.append(
                WordTimestamp(
                    text=text,
                    start_seconds=word.start,
                    end_seconds=word.end,
                )
            )

    if not words:
        raise ValueError(f"No se detectó voz en el video: {video_path}")

    return words
