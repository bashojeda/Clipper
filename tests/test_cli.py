"""Tests for the Phase 0 command-line workflow."""

from pathlib import Path

import pytest
from anthropic import APIConnectionError

from clipper import cli
from clipper.moments import ClipMoment
from clipper.transcribe import WordTimestamp


def test_main_runs_all_stages_and_reports_exported_clips(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    video_path = tmp_path / "episode.mp4"
    video_path.touch()
    output_dir = tmp_path / "rendered"
    words = [WordTimestamp(text="hola", start_seconds=0, end_seconds=0.5)]
    selected = [
        ClipMoment(
            start_seconds=start,
            end_seconds=start + 20,
            title=f"Clip {index}",
            score=90 - index,
            reason="Momento destacado.",
        )
        for index, start in enumerate((0, 20, 40), start=1)
    ]
    exported = [output_dir / f"clip_{index:02d}.mp4" for index in range(1, 4)]
    calls: list[str] = []

    monkeypatch.setattr(cli, "load_dotenv", lambda: None)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(
        cli,
        "transcribe_video",
        lambda *args, **kwargs: calls.append("transcribe") or words,
    )
    monkeypatch.setattr(
        cli,
        "select_best_moments",
        lambda *args, **kwargs: calls.append("select") or selected,
    )
    monkeypatch.setattr(
        cli,
        "render_clips",
        lambda *args, **kwargs: calls.append("render") or exported,
    )

    exit_code = cli.main(
        [str(video_path), "--output", str(output_dir), "--anthropic-model", "claude-test"]
    )

    assert exit_code == 0
    assert calls == ["transcribe", "select", "render"]
    assert "clip_03.mp4" in capsys.readouterr().out


def test_main_fails_early_without_anthropic_api_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video_path = tmp_path / "episode.mp4"
    video_path.touch()
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with pytest.raises(SystemExit) as error:
        cli.main([str(video_path), "--anthropic-model", "claude-test"])

    assert error.value.code == 2


def test_main_reports_anthropic_connection_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    video_path = tmp_path / "episode.mp4"
    video_path.touch()
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(
        cli,
        "transcribe_video",
        lambda *args, **kwargs: [
            WordTimestamp(text="hola", start_seconds=0, end_seconds=0.5)
        ],
    )
    monkeypatch.setattr(
        cli,
        "select_best_moments",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            APIConnectionError(request=None)
        ),
    )

    exit_code = cli.main([str(video_path), "--anthropic-model", "claude-test"])

    assert exit_code == 1
    assert "Error:" in capsys.readouterr().err
