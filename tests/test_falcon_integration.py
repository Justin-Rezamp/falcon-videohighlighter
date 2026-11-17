"""Falcon wiring: pipeline hooks, preset, cache key, report label, sidecar route."""
from __future__ import annotations

import asyncio
import json

import numpy as np

from modules.falcon import pipeline_hooks as hooks
from modules.falcon import preset


def _video_with_marks(tmp_path, clicks):
    video = tmp_path / "game.mp4"
    video.write_bytes(b"0" * 8)
    (tmp_path / "game.mark").write_text(json.dumps(
        [{"type": "start", "time": 0}] + [{"type": "click", "time": t} for t in clicks]))
    return str(video)


def test_taps_load_only_when_asked(tmp_path):
    video = _video_with_marks(tmp_path, [30, 90])
    assert hooks.load_taps({}, video, 100, log_fn=lambda m: None) == []
    assert hooks.load_taps({"falcon_marks": True}, video, 100,
                           log_fn=lambda m: None) == [30.0, 90.0]


def test_taps_follow_a_time_range_trim(tmp_path):
    video = _video_with_marks(tmp_path, [30, 90])
    said = []
    taps = hooks.load_taps({"falcon_marks": True}, video, 40, offset=60, log_fn=said.append)
    assert taps == [30.0]
    assert any("outside the analysed range" in m for m in said)


def test_marks_mode_replaces_segments_and_off_mode_does_not(tmp_path):
    cfg = {"falcon_mode": "marks"}
    segs, clips, notes = hooks.plan_segments(cfg, [30.0, 40.0], np.zeros(101), 100)
    assert segs == [(20.0, 50.0)] and clips[0].merged and notes == []
    assert hooks.plan_segments({}, [30.0], np.zeros(101), 100) is None
    assert hooks.plan_segments({"falcon_mode": "bogus"}, [30.0], np.zeros(101), 100) is None


def test_analysis_source_is_the_original_unless_asked(tmp_path):
    assert hooks.analysis_source({}, "v.mp4", str(tmp_path)) == "v.mp4"
    # A probe failure never fails the run: it analyses the original instead.
    said = []
    assert hooks.analysis_source({"analysis_proxy": "auto"}, str(tmp_path / "none.mp4"),
                                 str(tmp_path), log_fn=said.append) == str(tmp_path / "none.mp4")
    assert said and "skipped" in said[0]


def test_shipped_preset_flattens_to_run_settings():
    data = preset.load_shipped()
    flat = preset.gui_settings(data)
    assert flat["falcon_mode"] == "marks"
    assert flat["camera_compensation"] is True
    assert flat["parent_tap_points"] == 10
    assert data["scoring"]["scene_points"] == 0
    assert preset.gui_settings({}) == {} and preset.gui_settings({"falcon": "x"}) == {}


def test_preset_install_never_clobbers(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("{}")
    path = preset.install(str(cfg))
    assert path.endswith("falcon-soccer.yaml")
    try:
        preset.install(str(cfg))
    except FileExistsError:
        pass
    else:
        raise AssertionError("second install must refuse")


def test_camera_compensation_changes_the_analysis_cache_key_only_when_on():
    from modules.media.video_cache import build_analysis_cache_params
    plain = build_analysis_cache_params({}, {}, 5, 100)
    on = build_analysis_cache_params({"camera_compensation": True}, {}, 5, 100)
    assert "camera_compensation" not in plain and on["camera_compensation"] is True


def test_report_names_the_signal_parent_tap():
    from modules.report.highlight_report import BOOST_SIGNALS, SIGNAL_LABELS
    labels = dict(SIGNAL_LABELS)
    assert labels["parent_tap"] == "Parent tap"
    assert "parent_tap" in BOOST_SIGNALS


def test_sidecar_falcon_plan(tmp_path, monkeypatch):
    import sidecar.server as server
    from modules.falcon import media

    video = _video_with_marks(tmp_path, [101, 156])
    monkeypatch.setattr(media, "probe_source", lambda p: {
        "duration": 600.0, "size": 8, "width": 3840, "height": 2160})
    res = asyncio.run(server.falcon_plan(path=video))
    assert res["ok"] and len(res["plan"]["clips"]) == 2
    missing = asyncio.run(server.falcon_plan(path=str(tmp_path / "nope.mp4")))
    assert not missing["ok"]


def test_sidecar_run_takes_falcon_defaults_but_request_wins(tmp_path, monkeypatch):
    import sidecar.server as server
    cfg = tmp_path / "config.yaml"
    cfg.write_text("falcon:\n  mode: hybrid\n  camera_compensation: true\n")
    monkeypatch.setattr("modules.system.app_paths.config_path", lambda *a: str(cfg))
    got = server._falcon_defaults()
    assert got == {"falcon_mode": "hybrid", "camera_compensation": True}
    video = tmp_path / "v.mp4"
    video.write_bytes(b"0")
    seen = {}
    monkeypatch.setattr(server.manager, "start_job",
                        lambda job, loop: seen.update(job) or "run1")
    res = asyncio.run(server.start_run(server.RunRequest(
        video_paths=[str(video)], config={"falcon_mode": "marks"})))
    assert res["ok"]
    assert seen["config"]["falcon_mode"] == "marks"
    assert seen["config"]["camera_compensation"] is True
