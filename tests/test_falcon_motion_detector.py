"""The real motion detector with camera compensation on a synthetic panning shot.

Needs real torch and OpenCV (the suite's shims are MagicMocks), so it skips in
the light dev install and runs where the full stack is present.
"""
from __future__ import annotations

import importlib
import importlib.util
import sys
from unittest.mock import MagicMock

import numpy as np
import pytest

_REAL = ("torch", "torch.nn", "torch.nn.functional", "cv2", "tqdm")
_DETECTOR = "modules.segments.motion_scene_detect_optimized"


@pytest.fixture
def real_stack():
    """Borrow the real torch/OpenCV/tqdm for one test, then hand the shims back.

    The rest of the suite is written against conftest's MagicMocks and must keep
    getting them, so the detector module is imported fresh here and dropped after.
    """
    shims = {name: sys.modules.pop(name, None) for name in _REAL}
    old_detector = sys.modules.pop(_DETECTOR, None)
    try:
        for name in ("torch", "cv2", "tqdm"):
            if importlib.util.find_spec(name) is None:
                pytest.skip(f"needs the real {name}")
        mods = {name: importlib.import_module(name) for name in ("torch", "cv2")}
        if any(isinstance(m, MagicMock) for m in mods.values()):
            pytest.skip("needs real torch and OpenCV")
        yield mods["cv2"], importlib.import_module(_DETECTOR)
    finally:
        for name in list(sys.modules):
            if name == _DETECTOR or name.split(".")[0] in ("torch", "cv2", "tqdm"):
                sys.modules.pop(name, None)
        for name, shim in shims.items():
            if shim is not None:
                sys.modules[name] = shim
        if old_detector is not None:
            sys.modules[_DETECTOR] = old_detector


def _write_pan(cv2, path, seconds=6, fps=15, player=False):
    rng = np.random.default_rng(0)
    big = rng.integers(40, 200, (400, 1400, 3)).astype(np.uint8)
    big = cv2.GaussianBlur(big, (7, 7), 0)
    w, h = 480, 270
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for i in range(seconds * fps):
        x = 20 + 6 * i                        # steady auto-pan
        frame = big[60:60 + h, x:x + w].copy()
        if player:
            px = 100 + (i * 9) % 260
            cv2.rectangle(frame, (px, 120), (px + 30, 180), (255, 255, 255), -1)
        out.write(frame)
    out.release()


def _energy(detector, path, compensate):
    details = {}
    detector.detect_scenes_motion_optimized(str(path), device="cpu", frame_skip=2, debug=False,
                                   details=details, camera_compensation=compensate)
    curve = details.get("motion_curve") or {}
    values = curve.get("values") if isinstance(curve, dict) else curve
    return float(np.mean(values or [0.0])), details


def test_pan_is_not_motion_but_a_player_is(tmp_path, real_stack):
    cv2, detector = real_stack
    pan = tmp_path / "pan.mp4"
    _write_pan(cv2, pan)
    raw, _ = _energy(detector, pan, False)
    comp, details = _energy(detector, pan, True)
    assert raw > 0.5
    assert comp < raw * 0.25
    assert details["camera_pan"]["moving_share"] > 0.8

    player = tmp_path / "player.mp4"
    _write_pan(cv2, player, player=True)
    with_player, _ = _energy(detector, player, True)
    assert with_player > comp * 2
