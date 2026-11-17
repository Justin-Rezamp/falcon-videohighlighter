"""A torch-free, OpenCV-free motion score for Falcon recordings.

The full motion detector needs torch and OpenCV. This needs numpy and an
ffmpeg binary: ffmpeg decodes a small greyscale copy straight into a pipe
(``-vf fps=…,scale=…,format=gray``), and each pair of frames is compared after
camera-motion compensation (:mod:`modules.falcon.camera`). The result is one
value per second — how much moved that the camera did not explain — which is
what the hybrid planner slides tap windows toward.

It exists so the Falcon workflow (marks + motion) runs on machines where the
ML stack does not install, such as Windows on ARM.
"""
from __future__ import annotations

import subprocess

import numpy as np

from modules.falcon.camera import compensated_diff, estimate_shift

NOISE_FLOOR = 8.0


def motion_per_second(frames, sample_fps: float, duration: float, *,
                      compensate: bool = True) -> np.ndarray:
    """Per-second mean residual energy from an iterable of 2-D uint8/float frames."""
    n = int(duration) + 1
    total = np.zeros(n)
    count = np.zeros(n)
    prev = None
    for i, frame in enumerate(frames):
        cur = np.asarray(frame, dtype=np.float32)
        if prev is not None:
            dy, dx = estimate_shift(prev, cur) if compensate else (0, 0)
            diff = compensated_diff(prev, cur, dy, dx)
            sec = min(n - 1, int(i / sample_fps))
            total[sec] += float(np.clip(diff - NOISE_FLOOR, 0, None).mean())
            count[sec] += 1
        prev = cur
    return np.divide(total, count, out=np.zeros(n), where=count > 0)


def ffmpeg_gray_frames(video: str, sample_fps: float = 5.0, width: int = 320,
                       height: int = 180, popen=subprocess.Popen):
    """Yield ``height x width`` uint8 greyscale frames decoded by ffmpeg."""
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-i", video, "-an",
           "-vf", f"fps={sample_fps},scale={width}:{height}:flags=area,format=gray",
           "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1"]
    size = width * height
    proc = popen(cmd, stdout=subprocess.PIPE)
    try:
        while True:
            buf = proc.stdout.read(size)
            if not buf or len(buf) < size:
                break
            yield np.frombuffer(buf, dtype=np.uint8).reshape(height, width)
    finally:
        proc.stdout.close()
        proc.wait()


def motion_score(video: str, duration: float, *, sample_fps: float = 5.0,
                 compensate: bool = True) -> np.ndarray:
    """Camera-compensated motion, one value per second of ``video``."""
    return motion_per_second(ffmpeg_gray_frames(video, sample_fps), sample_fps,
                             duration, compensate=compensate)
