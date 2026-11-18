"""Tests for subtitle generation and FFmpeg command construction."""

from pathlib import Path
from typing import Any

import pytest

from clipper import render
from clipper.moments import ClipMoment
from clipper.transcribe import WordTimestamp


def test_format_srt_timestamp() -> None:
    assert render._format_srt_timestamp(3661.234) == "01:01:01,234"
    assert render._format_srt_timestamp(-2) == "00:00:00,000"


def test_build_subtitle_cues_groups_words_and_clips_to_moment() -> None:
    moment = ClipMoment(
        start_seconds=10,
        end_seconds=35,
        title="Prueba",
        score=90,
        reason="Momento claro.",
    )
    words = [
        WordTimestamp(text="antes", start_seconds=9, end_seconds=9.5),
        WordTimestamp(text="Hola", start_seconds=10, end_seconds=10.4),
        WordTimestamp(text="mundo!", start_seconds=10.5, end_seconds=11),
        WordTimestamp(text="después", start_seconds=35, end_seconds=35.5),
    ]

    cues = render._build_subtitle_cues(words, moment)

    assert cues == [(0.0, 1, "Hola mundo!")]


def test_render_clips_calls_ffmpeg_and_removes_temporary_subtitles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video_path = tmp_path / "source.mp4"
    video_path.touch()
    output_dir = tmp_path / "clips"
    moments = [
        ClipMoment(
            start_seconds=start,
            end_seconds=start + 25,
            title=f"Clip {index}",
            score=90 - index,
            reason="Momento de prueba.",
        )
        for index, start in enumerate((0, 25, 50), start=1)
    ]
    words = [
        WordTimestamp(text="hola", start_seconds=1, end_seconds=1.5),
        WordTimestamp(text="mundo", start_seconds=26, end_seconds=26.5),
        WordTimestamp(text="clip", start_seconds=51, end_seconds=51.5),
    ]
    commands: list[list[str]] = []

    monkeypatch.setattr(render.shutil, "which", lambda _: "ffmpeg")

    def fake_run(
        command: list[str],
        *,
        check: bool,
        capture_output: bool,
        text: bool,
        cwd: Path,
    ) -> None:
        assert check is True
        assert capture_output is True
        assert text is True
        commands.append(command)
        assert (cwd / f"clip_{len(commands):02d}.srt").is_file()
        Path(command[-1]).touch()

    monkeypatch.setattr(render.subprocess, "run", fake_run)

    rendered = render.render_clips(video_path, output_dir, moments, words)

    assert rendered == [
        output_dir / "clip_01.mp4",
        output_dir / "clip_02.mp4",
        output_dir / "clip_03.mp4",
    ]
    assert len(commands) == 3
    video_filter = commands[0][commands[0].index("-vf") + 1]
    assert "scale=1080:1920:force_original_aspect_ratio=increase:flags=lanczos" in video_filter
    assert "subtitles=clip_01.srt" in video_filter
    assert not list(output_dir.glob("*.srt"))


def test_render_clips_reports_missing_ffmpeg(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video_path = tmp_path / "source.mp4"
    video_path.touch()
    moments = [
        ClipMoment(
            start_seconds=start,
            end_seconds=start + 20,
            title=f"Clip {index}",
            score=90 - index,
            reason="Momento de prueba.",
        )
        for index, start in enumerate((0, 20, 40), start=1)
    ]
    monkeypatch.setattr(render.shutil, "which", lambda _: None)

    with pytest.raises(FileNotFoundError, match="FFmpeg"):
        render.render_clips(video_path, tmp_path / "clips", moments, [])
