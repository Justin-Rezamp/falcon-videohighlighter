"""Parent taps as a scoring signal, and the hybrid tap + detector plan.

Two ways the taps meet the rest of the app:

* :func:`tap_signal` turns taps into a per-second score array, the same shape
  as every other signal in ``pipeline.run_highlighter`` (``scene_score``,
  ``audio_score``…). It is added to the total like they are, drawn on the
  timeline as **PARENT TAPS**, and broken out in the report as
  "Parent tap". A parent usually taps *after* the moment, so the boost covers
  the seconds before a tap more than the seconds after.

* :func:`hybrid_plan` keeps the marks-only plan's one-clip-per-tap shape but
  lets the detectors move each window: when the detector score peaks clearly
  within ``max_earlier`` seconds before or ``max_later`` seconds after a tap,
  the window slides toward that peak. The tap itself is a fact; the slide is an
  inference, and every slide is reported as one in ``notes``.

A tap is never treated as proof of a goal.
"""
from __future__ import annotations

import numpy as np

from modules.falcon.marks import (DEFAULT_MERGE_GAP, DEFAULT_PAD, TAP_LABEL,
                                  FalconClip, MarkSet, fmt_clock, merge_windows,
                                  number_clips)

MAX_EARLIER = 6.0
MAX_LATER = 3.0
DEFAULT_TAP_POINTS = 10.0
#: A detector peak must reach this percentile of the game's own score to pull a
#: window: "clear action", not "the busiest second of some jogging".
PEAK_PERCENTILE = 75.0


def _clicks(marks) -> list[float]:
    if isinstance(marks, MarkSet):
        return list(marks.clicks)
    return sorted({float(m) for m in (marks or [])})


def tap_signal(marks, video_duration: float, points: float = DEFAULT_TAP_POINTS,
               *, earlier: float = MAX_EARLIER, later: float = MAX_LATER) -> np.ndarray:
    """Per-second ``parent_tap`` score, ``int(video_duration) + 1`` long.

    Full ``points`` on the tap's own second, falling linearly to zero
    ``earlier`` seconds before and ``later`` seconds after it. Overlapping taps
    keep the larger value rather than adding, so a parent tapping twice does not
    outweigh every detector at once.
    """
    n = int(video_duration) + 1
    out = np.zeros(n, dtype=float)
    if points <= 0:
        return out
    for t in _clicks(marks):
        if t < 0 or t > video_duration:
            continue
        lo = max(0, int(np.floor(t - earlier)))
        hi = min(n - 1, int(np.ceil(t + later)))
        for sec in range(lo, hi + 1):
            gap = sec - t
            span = earlier if gap < 0 else later
            weight = 1.0 if span <= 0 else max(0.0, 1.0 - abs(gap) / span)
            out[sec] = max(out[sec], points * weight)
    return out


def hybrid_plan(marks, score, video_duration: float, *, before: float = DEFAULT_PAD,
                after: float = DEFAULT_PAD, merge_gap: float = DEFAULT_MERGE_GAP,
                max_earlier: float = MAX_EARLIER, max_later: float = MAX_LATER,
                peak_percentile: float = PEAK_PERCENTILE):
    """One window per tap, slid toward a clear detector peak nearby.

    ``score`` is the per-second detector total *without* the tap signal (the
    slide must come from other evidence). Returns ``(clips, notes)``; every
    note describes an inference, never a fact about the game.
    """
    clicks = _clicks(marks)
    score = np.asarray(score if score is not None else [], dtype=float)
    notes: list[str] = []
    n = score.size
    usable = n > 0 and float(score.max()) > 0
    if not usable and clicks:
        notes.append("No detector signal to compare against; windows stay centred "
                     "on each parent tap (marks-only).")
    bar = float(np.percentile(score, peak_percentile)) if usable else 0.0

    windows = []
    shift_of: dict[float, float] = {}
    for t in clicks:
        shift = 0.0
        if usable:
            lo = max(0, int(np.floor(t - max_earlier)))
            hi = min(n - 1, int(np.ceil(t + max_later)))
            seg = score[lo:hi + 1]
            if seg.size:
                p = lo + int(np.argmax(seg))
                if score[p] > 0 and score[p] >= bar and score[p] > float(np.median(seg)):
                    shift = float(np.clip(p - t, -max_earlier, max_later))
        # The tap must always stay inside its own clip.
        shift = float(np.clip(shift, -after + 0.5, before - 0.5))
        shift_of[t] = shift
        windows.append((max(0.0, t + shift - before),
                        min(video_duration, t + shift + after), [t]))

    clips = []
    for start, end, ms in merge_windows(windows, merge_gap):
        lo, hi = int(start), max(int(start) + 1, int(np.ceil(end)))
        mean = float(score[lo:hi].mean()) if usable and lo < n else 0.0
        clips.append(FalconClip(
            id="", kind="mark", start=round(start, 3), end=round(end, 3), marks=ms,
            score=round(mean, 4), shift=[round(shift_of[m], 2) for m in ms],
            note=" + ".join(f"{TAP_LABEL} {fmt_clock(m)}" for m in ms)))
        for m in ms:
            s = shift_of[m]
            if abs(s) >= 0.5:
                notes.append(
                    f"Inferred: {TAP_LABEL} at {fmt_clock(m)} — the detectors peak "
                    f"{abs(s):.0f}s {'before' if s < 0 else 'after'} it, so its window "
                    f"slid that way. The tap time itself is unchanged.")
    return number_clips(clips), notes
