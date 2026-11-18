"""Tests for Anthropic moment selection and Pydantic validation."""

import json
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

from clipper import moments
from clipper.transcribe import WordTimestamp


class FakeMessages:
    def __init__(self, response_text: str) -> None:
        self.response_text = response_text
        self.request: dict[str, Any] | None = None

    def create(self, **kwargs: Any) -> SimpleNamespace:
        self.request = kwargs
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=self.response_text)]
        )


class FakeAnthropic:
    def __init__(self, response_text: str) -> None:
        self.messages = FakeMessages(response_text)


def _clip(start: float, end: float, title: str, score: int) -> dict[str, Any]:
    return {
        "start_seconds": start,
        "end_seconds": end,
        "title": title,
        "score": score,
        "reason": "Momento claro y atractivo.",
    }


def test_select_best_moments_validates_and_sorts_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response_text = json.dumps(
        {
            "clips": [
                _clip(10, 35, "Idea inicial", 72),
                _clip(60, 90, "Gran revelación", 96),
                _clip(100, 145, "Cierre divertido", 83),
            ]
        }
    )
    client = FakeAnthropic(response_text)
    monkeypatch.setattr(moments, "Anthropic", lambda: client)
    words = [
        WordTimestamp(text="inicio", start_seconds=0, end_seconds=0.5),
        WordTimestamp(text="final", start_seconds=150, end_seconds=151),
    ]

    clips = moments.select_best_moments(words, model="claude-test")

    assert [clip.title for clip in clips] == [
        "Gran revelación",
        "Cierre divertido",
        "Idea inicial",
    ]
    assert client.messages.request is not None
    assert client.messages.request["model"] == "claude-test"


def test_select_best_moments_rejects_invalid_clip_duration() -> None:
    with pytest.raises(ValidationError, match="20 y 60"):
        moments.ClipMoment(
            start_seconds=10,
            end_seconds=25,
            title="Demasiado corto",
            score=50,
            reason="No cumple la duración mínima.",
        )


def test_select_best_moments_rejects_time_after_video(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response_text = json.dumps(
        {
            "clips": [
                _clip(10, 35, "Uno", 90),
                _clip(40, 65, "Dos", 80),
                _clip(70, 95, "Tres", 70),
            ]
        }
    )
    monkeypatch.setattr(moments, "Anthropic", lambda: FakeAnthropic(response_text))
    words = [WordTimestamp(text="final", start_seconds=0, end_seconds=80)]

    with pytest.raises(ValueError, match="fuera del video"):
        moments.select_best_moments(words, model="claude-test")


def test_select_best_moments_requires_model_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ANTHROPIC_MODEL", raising=False)
    words = [WordTimestamp(text="hola", start_seconds=0, end_seconds=0.5)]

    with pytest.raises(ValueError, match="ANTHROPIC_MODEL"):
        moments.select_best_moments(words)
