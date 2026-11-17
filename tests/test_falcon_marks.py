"""XbotGo Falcon .mark files and the marks-only plan (modules/falcon/marks.py).

The plan must match the Falcon Highlights runner exactly: 10 s either side of
each click, windows that overlap or sit within 2 s merged, clips numbered in
time order and named clip01_mark_9-09.mp4 (+ _merged).
"""
from __future__ import annotations

import json

import pytest

from modules.falcon import marks as fm


def _mark_file(path, clicks, start=True):
    data = ([{"type": "start", "time": 0}] if start else [])
    data += [{"type": "click", "time": t} for t in clicks]
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_only_clicks_count_and_start_is_ignored():
    ms = fm.parse_marks([{"type": "start", "time": 0}, {"type": "click", "time": 549},
                         {"type": "click", "time": 12.5}])
    assert ms.clicks == [12.5, 549.0]
    assert ms.ignored == []


def test_bad_clicks_are_reported_never_moved():
    ms = fm.parse_marks([{"type": "click", "time": "x"}, {"type": "click", "time": -3},
                         {"type": "click", "time": 9000}, "junk",
                         {"type": "click", "time": 60}], duration=3600)
    assert ms.clicks == [60.0]
    assert len(ms.ignored) == 4
    assert any("after the video ends" in s for s in ms.ignored)


def test_not_a_list_is_an_error():
    with pytest.raises(fm.MarkFileError):
        fm.parse_marks({"type": "click"})


def test_invalid_json_is_an_error(tmp_path):
    bad = tmp_path / "x.mark"
    bad.write_text("[{", encoding="utf-8")
    with pytest.raises(fm.MarkFileError):
        fm.read_marks(str(bad))


def test_runner_rules_pad_merge_and_names():
    clips = fm.plan_from_marks([549, 537, 101, 3000], duration=3488)
    assert [c.id for c in clips] == ["clip01", "clip02", "clip03"]
    assert (clips[0].start, clips[0].end) == (91.0, 111.0)
    # 537 and 551 windows overlap -> one merged clip named after the first tap
    assert clips[1].marks == [537.0, 549.0]
    assert (clips[1].start, clips[1].end) == (527.0, 559.0)
    assert clips[1].filename == "clip02_mark_8-57_merged.mp4"
    assert clips[0].filename == "clip01_mark_1-41.mp4"
    assert clips[2].filename == "clip03_mark_50-00.mp4"


def test_windows_within_two_seconds_merge_and_three_do_not():
    two = fm.plan_from_marks([100, 122], duration=1000)   # 110 vs 112: gap 2
    three = fm.plan_from_marks([100, 123], duration=1000)  # 110 vs 113: gap 3
    assert len(two) == 1 and two[0].merged
    assert len(three) == 2


def test_windows_are_clamped_to_the_video():
    clips = fm.plan_from_marks([3, 995], duration=1000)
    assert clips[0].start == 0.0 and clips[-1].end == 1000.0


def test_notes_say_parent_tap_never_goal():
    clips = fm.plan_from_marks([60, 61], duration=200)
    text = " ".join(c.note for c in clips).lower()
    assert "parent tap" in text and "goal" not in text


def test_slug_and_clock():
    assert fm.slug_clock(549) == "9-09"
    assert fm.slug_clock(3725) == "1-02-05"
    assert fm.fmt_clock(549.5) == "9:09.5"


def test_sibling_mark_found(tmp_path):
    video = tmp_path / "game.mp4"
    video.write_bytes(b"0" * 10)
    _mark_file(tmp_path / "game.mark", [5])
    assert fm.find_mark_file(str(video)) == str(tmp_path / "game.mark")


def test_mark_in_data_folder_found_case_insensitively(tmp_path):
    video = tmp_path / "VID_1.mp4"
    video.write_bytes(b"0")
    (tmp_path / ".data").mkdir()
    _mark_file(tmp_path / ".data" / "vid_1.MARK", [5])
    assert fm.find_mark_file(str(video)).endswith("vid_1.MARK")


def test_renamed_video_pairs_through_vdata_size(tmp_path):
    video = tmp_path / "Your_Video_File.mp4"
    video.write_bytes(b"0" * 1234)
    data = tmp_path / ".data"
    data.mkdir()
    _mark_file(data / "VID_20260425_133314_01_01.mark", [101])
    (data / "VID_20260425_133314_01_01.vdata").write_text(json.dumps(
        {"size": 1234, "duration": 60.0, "fps": 27.87, "width": 3840, "height": 2160}))
    _mark_file(data / "VID_OTHER.mark", [7])
    (data / "VID_OTHER.vdata").write_text(json.dumps({"size": 999}))
    assert fm.find_mark_file(str(video)).endswith("VID_20260425_133314_01_01.mark")


def test_explicit_path_wins_and_must_exist(tmp_path):
    video = tmp_path / "a.mp4"
    video.write_bytes(b"0")
    _mark_file(tmp_path / "a.mark", [1])
    other = _mark_file(tmp_path / "other.mark", [2])
    assert fm.find_mark_file(str(video), str(other)) == str(other)
    with pytest.raises(fm.MarkFileError):
        fm.find_mark_file(str(video), str(tmp_path / "missing.mark"))


def test_no_mark_is_none(tmp_path):
    video = tmp_path / "a.mp4"
    video.write_bytes(b"0")
    assert fm.find_mark_file(str(video)) is None


def test_vdata_pairing_checks(tmp_path):
    vd = tmp_path / "v.vdata"
    vd.write_text(json.dumps({"duration": 3488.17, "fps": 27.87, "size": 100,
                              "width": 3840, "height": 2160, "file_name": "VID.mp4"}))
    info = fm.read_vdata(str(vd))
    assert info["width"] == 3840 and info["file_name"] == "VID.mp4"
    assert fm.check_pairing(info, duration=3489.0, size=100, width=3840, height=2160) == []
    problems = fm.check_pairing(info, duration=2000.0, size=5, width=1920, height=1080)
    assert len(problems) == 3
