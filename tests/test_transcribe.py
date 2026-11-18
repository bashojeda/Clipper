"""Tests for word-level video transcription."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from clipper import transcribe


class FakeWhisperModel:
    """A small stand-in that avoids downloading or running a Whisper model."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs

    def transcribe(
        self,
        video_path: str,
        **kwargs: Any,
    ) -> tuple[list[SimpleNamespace], None]:
        self.video_path = video_path
        self.transcribe_kwargs = kwargs
        return [
            SimpleNamespace(
                words=[
                    SimpleNamespace(word=" Hello", start=0.1, end=0.5),
                    SimpleNamespace(word=" ", start=0.5, end=0.6),
                    SimpleNamespace(word="world!", start=0.6, end=1.0),
                    SimpleNamespace(word="missing-time", start=None, end=1.2),
                ]
            )
        ], None


def test_transcribe_video_returns_word_timestamps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video_path = tmp_path / "sample.mp4"
    video_path.touch()
    model_instance: FakeWhisperModel | None = None

    def create_model(*args: Any, **kwargs: Any) -> FakeWhisperModel:
        nonlocal model_instance
        model_instance = FakeWhisperModel(*args, **kwargs)
        return model_instance

    monkeypatch.setattr(transcribe, "WhisperModel", create_model)

    words = transcribe.transcribe_video(video_path, language="es")

    assert [(word.text, word.start_seconds, word.end_seconds) for word in words] == [
        ("Hello", 0.1, 0.5),
        ("world!", 0.6, 1.0),
    ]
    assert model_instance is not None
    assert model_instance.args == ("small",)
    assert model_instance.kwargs == {"device": "cpu", "compute_type": "int8"}
    assert model_instance.transcribe_kwargs == {
        "word_timestamps": True,
        "vad_filter": True,
        "language": "es",
    }


def test_transcribe_video_requires_an_existing_file(tmp_path: Path) -> None:
    missing_video = tmp_path / "missing.mp4"

    with pytest.raises(FileNotFoundError, match="No se encontró el video"):
        transcribe.transcribe_video(missing_video)


def test_transcribe_video_fails_when_no_words_are_detected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video_path = tmp_path / "silent.mp4"
    video_path.touch()

    class SilentWhisperModel(FakeWhisperModel):
        def transcribe(
            self,
            video_path: str,
            **kwargs: Any,
        ) -> tuple[list[SimpleNamespace], None]:
            return [SimpleNamespace(words=[])], None

    monkeypatch.setattr(transcribe, "WhisperModel", SilentWhisperModel)

    with pytest.raises(ValueError, match="No se detectó voz"):
        transcribe.transcribe_video(video_path)
