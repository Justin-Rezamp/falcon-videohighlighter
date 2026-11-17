"""Falcon proxy / phone export commands, the lite motion score, and the CLI."""
from __future__ import annotations

import io
import json

import numpy as np
import pytest

from modules.falcon import cli, lite_motion, media


def test_proxy_reason():
    assert "HEVC" in media.proxy_reason({"codec": "hevc", "width": 1920, "height": 1080})
    assert "larger than 1080p" in media.proxy_reason({"codec": "h264", "width": 3840,
                                                      "height": 2160})
    assert media.proxy_reason({"codec": "h264", "width": 1920, "height": 1080}) == ""


def test_make_proxy_writes_part_then_renames_and_reuses(tmp_path):
    src = tmp_path / "game.mp4"
    src.write_bytes(b"x" * 10)
    calls = []

    def fake_run(args, check):
        calls.append(args)
        with open(args[-1], "wb") as handle:
            handle.write(b"proxy")

    out = media.make_proxy(str(src), str(tmp_path / "cache"), run=fake_run, log_fn=lambda m: None)
    assert out.endswith(".mp4") and ".part" not in out
    assert calls[0][-1].endswith(".part.mp4") and "-an" in calls[0]
    media.make_proxy(str(src), str(tmp_path / "cache"), run=fake_run, log_fn=lambda m: None)
    assert len(calls) == 1


@pytest.mark.parametrize("preset", media.PHONE_PRESETS)
def test_phone_commands_never_overwrite_by_default(preset):
    args = media.phone_command("in.mp4", "out.mp4", preset)
    assert "-n" in args and "-y" not in args
    assert media.LOUDNORM in args
    assert args[-1] == "out.mp4"
    assert ("-filter_complex" in args) == (preset == "vertical")


def test_vertical_and_letterbox_are_1080x1920():
    assert "1080:1920" in media.phone_filter("vertical")
    assert "pad=1080:1920" in media.phone_filter("letterbox")
    with pytest.raises(ValueError):
        media.phone_filter("square")


def test_concat_loudnorm_reencodes_audio_only():
    args = media.concat_command("list.txt", "reel.mp4", loudnorm=True)
    assert args[args.index("-c:v") + 1] == "copy" and media.LOUDNORM in args


def test_lite_motion_ignores_a_pan():
    rng = np.random.default_rng(0)
    big = rng.integers(0, 255, (120, 400)).astype(np.float32)
    soft = sum(np.roll(np.roll(big, i, 0), j, 1) for i in range(4) for j in range(4)) / 16
    frames = [soft[10:90, 10 + 4 * k:170 + 4 * k] for k in range(10)]   # steady pan
    raw = lite_motion.motion_per_second(frames, 5, 2, compensate=False)
    comp = lite_motion.motion_per_second(frames, 5, 2, compensate=True)
    assert raw[0] > 1.0 and comp[0] < raw[0] * 0.1


def test_ffmpeg_gray_frames_reads_whole_frames_only():
    payload = bytes(range(6)) * 2 + b"\x01"     # two 2x3 frames and a stray byte

    class FakeProc:
        def __init__(self, cmd, stdout):
            self.cmd = cmd
            self.stdout = io.BytesIO(payload)

        def wait(self):
            return 0

    frames = list(lite_motion.ffmpeg_gray_frames("v.mp4", 5, 3, 2, popen=FakeProc))
    assert len(frames) == 2 and frames[0].shape == (2, 3)


def _game(tmp_path, clicks=(101, 156, 537, 551)):
    video = tmp_path / "Your_Video_File.mp4"
    video.write_bytes(b"0" * 64)
    data = tmp_path / ".data"
    data.mkdir()
    marks = [{"type": "start", "time": 0}] + [{"type": "click", "time": t} for t in clicks]
    (data / "VID_X_01_01.mark").write_text(json.dumps(marks))
    (data / "VID_X_01_01.vdata").write_text(json.dumps({"size": 64, "duration": 3488.17}))
    return video


def test_build_plan_marks_mode_from_renamed_video(tmp_path):
    plan = cli.build_plan(str(_game(tmp_path)), duration=3488.17)
    assert plan["mark_file"].endswith("VID_X_01_01.mark")
    assert [c["filename"] for c in plan["clips"]] == [
        "clip01_mark_1-41.mp4", "clip02_mark_2-36.mp4", "clip03_mark_8-57_merged.mp4"]
    assert plan["pairing_warnings"] == []
    text = cli.describe(plan)
    assert "parent taps" in text and "goal" not in text.lower()


def test_build_plan_flags_a_pairing_mismatch(tmp_path):
    plan = cli.build_plan(str(_game(tmp_path)), duration=1000.0)
    assert any("camera recorded" in w for w in plan["pairing_warnings"])


def test_build_plan_hybrid_with_given_score(tmp_path):
    score = np.zeros(3489)
    score[3::50] = 1
    score[97] = 30.0
    plan = cli.build_plan(str(_game(tmp_path, clicks=(101,))), duration=3488.17,
                          mode="hybrid", score=score)
    assert plan["clips"][0]["shift"] == [-4.0]
    assert plan["notes"] and plan["notes"][0].startswith("Inferred:")


def test_cli_plan_writes_json_and_cut_print_only_runs_nothing(tmp_path, capsys, monkeypatch):
    video = _game(tmp_path)
    out = tmp_path / "plan.json"
    assert cli.main(["plan", str(video), "--duration", "3488.17", "--out", str(out)]) == 0
    plan = json.loads(out.read_text())
    assert len(plan["clips"]) == 3

    def no_run(*_a, **_k):
        raise AssertionError("print-only must not run ffmpeg")

    monkeypatch.setattr(cli.subprocess, "run", no_run)
    assert cli.main(["cut", str(out), "--out-dir", str(tmp_path / "clips"),
                     "--print-only"]) == 0
    assert capsys.readouterr().out.count("ffmpeg ") == 3


def test_cli_join_refuses_to_overwrite(tmp_path, capsys):
    reel = tmp_path / "highlights.mp4"
    reel.write_bytes(b"old")
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"video": "v.mp4", "clips": []}))
    assert cli.main(["join", str(plan), "--output", str(reel)]) == 2
    assert reel.read_bytes() == b"old"


def test_cli_missing_mark_is_a_clean_error(tmp_path, capsys):
    video = tmp_path / "a.mp4"
    video.write_bytes(b"0")
    assert cli.main(["plan", str(video), "--duration", "10"]) == 1
    assert "No .mark found" in capsys.readouterr().err
