"""The few places ``pipeline.run_highlighter`` asks about Falcon footage.

Kept here so the pipeline itself only gains a handful of calls. Settings, all
read from the run's ``gui_config`` (the Falcon preset sets them):

``falcon_marks``         look for a ``.mark`` file for the video (bool)
``falcon_mark_path``     explicit ``.mark`` path; implies ``falcon_marks``
``parent_tap_points``    weight of the parent-tap signal in the score (0 = off)
``falcon_mode``          ``""`` (taps only boost the score), ``"marks"``
                         (one clip per tap, runner rules) or ``"hybrid"``
                         (one clip per tap, slid toward a detector peak)
``falcon_pad_before`` / ``falcon_pad_after`` / ``falcon_merge_gap``
``camera_compensation``  subtract the camera's own pan from motion (bool)
``analysis_proxy``       ``"auto"`` / True / False — analyse a 360p copy of
                         4K or HEVC footage
"""
from __future__ import annotations

import os

from modules.falcon import marks as fm

MODES = ("", "marks", "hybrid")


def wants_marks(gui_config: dict) -> bool:
    return bool(gui_config.get("falcon_marks") or gui_config.get("falcon_mark_path")
                or gui_config.get("falcon_mode"))


def load_taps(gui_config: dict, video_path: str, video_duration: float, *,
              offset: float = 0.0, log_fn=print) -> list[float]:
    """Tap times in the analysed video's own clock, or ``[]``.

    ``offset`` is where the analysed range starts in the original recording
    (the time-range trim), so taps line up with a trimmed copy. Taps outside
    the analysed range are dropped and said so — never moved.
    """
    if not wants_marks(gui_config):
        return []
    try:
        path = fm.find_mark_file(video_path, gui_config.get("falcon_mark_path") or None)
    except fm.MarkFileError as exc:
        log_fn(f"⚠️ Parent taps: {exc}")
        return []
    if not path:
        log_fn("ℹ️ Parent taps: no .mark file found beside the video or in .data")
        return []
    try:
        markset = fm.read_marks(path)
    except fm.MarkFileError as exc:
        log_fn(f"⚠️ Parent taps: {exc}")
        return []
    vdata = fm.vdata_for_mark(path)
    if vdata and not offset:
        try:
            for problem in fm.check_pairing(fm.read_vdata(vdata), duration=video_duration):
                log_fn(f"⚠️ Parent taps: check the pairing — {problem}")
        except fm.MarkFileError:
            pass
    taps = [t - offset for t in markset.clicks]
    kept = [t for t in taps if 0 <= t <= video_duration]
    if len(kept) < len(taps):
        log_fn(f"ℹ️ Parent taps: {len(taps) - len(kept)} outside the analysed range left out")
    for note in markset.ignored:
        log_fn(f"ℹ️ Parent taps: ignored {note}")
    log_fn(f"👆 Parent taps: {len(kept)} from {path}")
    return kept


def mode(gui_config: dict) -> str:
    value = str(gui_config.get("falcon_mode") or "").strip().lower()
    return value if value in MODES else ""


def plan_segments(gui_config: dict, taps, detector_score, video_duration: float):
    """``(segments, clips, notes)`` for marks / hybrid mode; ``None`` otherwise."""
    which = mode(gui_config)
    if not which or not taps:
        return None
    before = float(gui_config.get("falcon_pad_before", fm.DEFAULT_PAD))
    after = float(gui_config.get("falcon_pad_after", fm.DEFAULT_PAD))
    gap = float(gui_config.get("falcon_merge_gap", fm.DEFAULT_MERGE_GAP))
    if which == "hybrid":
        from modules.falcon.hybrid import hybrid_plan
        clips, notes = hybrid_plan(taps, detector_score, video_duration,
                                   before=before, after=after, merge_gap=gap)
    else:
        clips = fm.plan_from_marks(taps, video_duration, before=before, after=after,
                                   merge_gap=gap)
        notes = []
    return [(float(c.start), float(c.end)) for c in clips], clips, notes


def analysis_source(gui_config: dict, video_path: str, cache_dir: str, *, log_fn=print) -> str:
    """The file the motion pass should read: a proxy for 4K/HEVC when asked."""
    want = gui_config.get("analysis_proxy", False)
    if not want:
        return video_path
    try:
        from modules.falcon.media import make_proxy, probe_source, proxy_reason
        if want != "always" and want is not True:
            reason = proxy_reason(probe_source(video_path))
            if not reason:
                return video_path
            log_fn(f"ℹ️ Analysis proxy: {reason}")
        return make_proxy(video_path, os.path.join(cache_dir, "falcon-proxies"), log_fn=log_fn)
    except Exception as exc:  # a proxy is an optimisation, never a reason to fail
        log_fn(f"⚠️ Analysis proxy skipped, analysing the original: {exc}")
        return video_path
