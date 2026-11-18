"""Select and validate short video moments using Anthropic."""

import os
from collections.abc import Sequence
from typing import Self

from anthropic import Anthropic
from pydantic import BaseModel, Field, model_validator

from clipper.transcribe import WordTimestamp


class ClipMoment(BaseModel):
    """A proposed clip with its transcript time range and ranking."""

    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(gt=0)
    title: str = Field(min_length=1, max_length=80)
    score: int = Field(ge=0, le=100)
    reason: str = Field(min_length=1, max_length=300)

    @model_validator(mode="after")
    def validate_duration(self) -> Self:
        duration = self.end_seconds - self.start_seconds
        if not 20 <= duration <= 60:
            raise ValueError("La duración del clip debe estar entre 20 y 60 segundos")
        return self


class ClipSelection(BaseModel):
    """The three top-ranked moments returned by the language model."""

    clips: list[ClipMoment] = Field(min_length=3, max_length=3)


def select_best_moments(
    words: Sequence[WordTimestamp],
    model: str | None = None,
) -> list[ClipMoment]:
    """Ask Anthropic to rank exactly three moments from a word-timed transcript."""
    if not words:
        raise ValueError("La transcripción está vacía")

    model_name = model or os.getenv("ANTHROPIC_MODEL")
    if not model_name:
        raise ValueError("Configura ANTHROPIC_MODEL en el archivo .env")

    transcript = "\n".join(
        f"{word.start_seconds:.2f}-{word.end_seconds:.2f} {word.text}"
        for word in words
    )
    client = Anthropic()
    response = client.messages.create(
        model=model_name,
        max_tokens=4096,
        system=(
            "Elige momentos atractivos, claros y autocontenidos para clips cortos. "
            "Usa solo el contenido de la transcripción y sus marcas de tiempo. "
            "Devuelve exactamente tres clips distintos, cada uno de 20 a 60 segundos. "
            "Responde únicamente con un objeto JSON válido, sin bloques Markdown."
        ),
        messages=[
            {
                "role": "user",
                "content": (
                    "Selecciona los tres mejores momentos de esta transcripción. "
                    "start_seconds y end_seconds deben estar dentro del tiempo del video; "
                    "score es un entero de 0 a 100. Devuelve este formato: "
                    '{"clips":[{"start_seconds":0,"end_seconds":30,"title":"...",'
                    '"score":90,"reason":"..."}]}.\n\n'
                    f"TRANSCRIPCIÓN:\n{transcript}"
                ),
            }
        ],
    )

    response_text = "\n".join(
        getattr(block, "text", "")
        for block in response.content
        if getattr(block, "type", None) == "text"
    ).strip()
    if not response_text:
        raise ValueError("Anthropic devolvió una respuesta vacía")

    selection = ClipSelection.model_validate_json(response_text)
    video_end_seconds = max(word.end_seconds for word in words)
    for clip in selection.clips:
        if clip.end_seconds > video_end_seconds:
            raise ValueError("Anthropic seleccionó un tiempo fuera del video")

    return sorted(selection.clips, key=lambda clip: clip.score, reverse=True)
