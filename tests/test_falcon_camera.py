"""Camera-motion (auto-pan) compensation, numpy only (modules/falcon/camera.py)."""
from __future__ import annotations

import numpy as np

from modules.falcon import camera


def _field(seed=0, shape=(300, 600)):
    rng = np.random.default_rng(seed)
    img = rng.normal(100, 30, shape)
    out = np.zeros_like(img)
    for dy in range(5):                       # soften like grass at low res
        for dx in range(5):
            out += np.roll(np.roll(img, dy, 0), dx, 1) / 25.0
    return out.astype(np.float32)


def _pan(big, y, x, h=200, w=400):
    return big[y:y + h, x:x + w]


def test_estimates_a_pan():
    big = _field()
    prev, curr = _pan(big, 50, 80), _pan(big, 47, 68)   # content moves +3 down, +12 right
    assert camera.estimate_shift(prev, curr) == (3, 12)


def test_pure_pan_leaves_no_residual_but_a_raw_diff_does():
    big = _field(1)
    prev, curr = _pan(big, 40, 100), _pan(big, 40, 90)
    raw = camera.residual_energy(prev, curr, compensate=False)
    comp = camera.residual_energy(prev, curr, compensate=True)
    assert raw > 1.0 and comp < raw * 0.05


def test_a_player_moving_during_a_pan_survives_compensation():
    big = _field(2)
    prev = _pan(big, 40, 100).copy()
    curr = _pan(big, 40, 90).copy()
    curr[80:110, 200:215] = 250.0              # a bright shirt that was not there
    comp = camera.residual_energy(prev, curr)
    still = camera.residual_energy(_pan(big, 40, 100), _pan(big, 40, 90))
    assert comp > still + 0.5


def test_align_pair_marks_the_uncovered_strip_invalid():
    prev = np.arange(20, dtype=np.float32).reshape(4, 5) + 1
    aligned, valid = camera.align_pair(prev, prev, 1, -2)
    assert valid[0].sum() == 0 and valid[:, 3:].sum() == 0
    assert aligned[1, 0] == prev[0, 2]


def test_works_on_a_leading_batch_axis():
    prev = np.ones((2, 6, 6), dtype=np.float32)
    aligned, valid = camera.align_pair(prev, prev, 0, 2)
    assert aligned.shape == prev.shape and valid[:, :, :2].sum() == 0


def test_huge_or_mismatched_shifts_are_ignored():
    a = np.zeros((10, 10), dtype=np.float32)
    assert camera.estimate_shift(a, np.zeros((10, 12), dtype=np.float32)) == (0, 0)
    assert camera.estimate_shift(np.zeros((4, 4)), np.zeros((4, 4))) == (0, 0)


def test_no_shift_is_plain_difference():
    a = np.full((4, 4), 3.0)
    b = np.full((4, 4), 5.0)
    assert np.array_equal(camera.compensated_diff(a, b, 0, 0), np.full((4, 4), 2.0))
