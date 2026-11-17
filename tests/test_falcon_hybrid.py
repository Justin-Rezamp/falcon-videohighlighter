"""Parent taps as a signal, and the hybrid tap + detector plan."""
from __future__ import annotations

import numpy as np

from modules.falcon.hybrid import MAX_EARLIER, MAX_LATER, hybrid_plan, tap_signal


def test_tap_signal_peaks_on_the_tap_and_leans_earlier():
    sig = tap_signal([100], 200, points=10)
    assert len(sig) == 201
    assert sig[100] == 10
    assert sig[97] > 0 and sig[102] > 0
    assert sig[100 - int(MAX_EARLIER)] == 0 and sig[100 + int(MAX_LATER)] == 0
    assert sig[90] == 0 and sig[110] == 0


def test_tap_signal_off_when_points_zero_and_taps_do_not_stack():
    assert not tap_signal([5], 20, points=0).any()
    assert tap_signal([10, 10.4], 20, points=10).max() == 10


def test_hybrid_slides_toward_a_clear_earlier_peak_and_says_it_is_inferred():
    score = np.zeros(301)
    score[::7] = 1.0             # background
    score[196] = 20.0            # clear action 4 s before the tap
    clips, notes = hybrid_plan([200], score, 300)
    assert clips[0].shift == [-4.0]
    assert (clips[0].start, clips[0].end) == (186.0, 206.0)
    assert clips[0].marks == [200.0]          # the tap time itself is unchanged
    assert any(n.startswith("Inferred:") and "4s before" in n for n in notes)
    assert all("goal" not in n.lower() for n in notes)


def test_hybrid_never_looks_beyond_six_earlier_or_three_later():
    score = np.zeros(301)
    score[190] = 50.0   # 10 s early: out of reach
    score[205] = 50.0   # 5 s late: out of reach
    score[1] = 1.0
    clips, notes = hybrid_plan([200], score, 300)
    assert clips[0].shift == [0.0] and notes == []


def test_hybrid_without_signal_falls_back_to_marks_only():
    clips, notes = hybrid_plan([50, 51], np.zeros(100), 100)
    assert len(clips) == 1 and clips[0].merged
    assert (clips[0].start, clips[0].end) == (40.0, 61.0)
    assert notes and "marks-only" in notes[0]
