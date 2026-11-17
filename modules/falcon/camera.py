"""Camera-motion (auto-pan) compensation for the motion detector.

The motion detector in ``modules/segments/motion_scene_detect_optimized.py``
measures how much the picture changed between two sampled frames. On a tripod
that is the same question as how much happened. On an auto-tracking camera such
as the XbotGo Falcon it is not: the camera pans and tilts all the time to
follow play, so the whole frame changes and the raw difference mostly measures
the camera.

The fix is the classic one from video stabilisation: estimate the *global*
shift between the two frames — the part every pixel shares — move the previous
frame by it, and difference only the area both frames still see. What is left
is the motion the camera did not explain: players, the ball, a goalmouth
scramble.

Translation comes from phase correlation on a small block-averaged copy of the
frame (exact for a pure shift and robust to auto-exposure drift). numpy only;
no OpenCV and no torch, so it runs on any build — including Windows on ARM.
:func:`align_pair` is written against the operations numpy arrays and torch
tensors share (slicing, ``* 0``, ``abs``), so the detector can apply the shift
to the tensors it already holds on whatever device they live on.

Ported from the falcon-hypecut project (Apache-2.0, same authors).
"""
from __future__ import annotations

import numpy as np

#: Width the frame is block-averaged down to before the FFT. A pan of a few
#: pixels at this size is tens of pixels at analysis size; finer than this buys
#: nothing for deciding "the camera moved", and the FFT stays ~1 ms.
ESTIMATE_WIDTH = 160
#: Largest shift believed, as a fraction of the frame. A Falcon pan between two
#: samples a sixth of a second apart is well under this; a bigger answer is a
#: cut or a failed estimate, and is treated as "no camera motion".
MAX_SHIFT_FRACTION = 0.25
#: Low-pass width on the cross-power spectrum, in cycles per pixel.
_LOWPASS_SIGMA = 0.15


def _block_mean(img: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return img
    h, w = img.shape[:2]
    h2, w2 = h // factor, w // factor
    if h2 < 4 or w2 < 4:
        return img
    trimmed = img[:h2 * factor, :w2 * factor]
    return trimmed.reshape(h2, factor, w2, factor).mean(axis=(1, 3))


def _subpixel(left: float, centre: float, right: float) -> float:
    denom = left - 2.0 * centre + right
    if abs(denom) < 1e-12:
        return 0.0
    return float(np.clip(0.5 * (left - right) / denom, -0.5, 0.5))


def phase_correlate(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Shift ``(dy, dx)`` with ``b(y, x) ≈ a(y - dy, x - dx)``.

    Half-whitened cross-power spectrum with a gentle low-pass: plain phase
    correlation hands codec noise on grass the same vote as field lines.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    h, w = a.shape
    win = np.outer(np.hanning(h), np.hanning(w))
    fa = np.fft.fft2((a - a.mean()) * win)
    fb = np.fft.fft2((b - b.mean()) * win)
    cross = fb * np.conj(fa)
    cross = cross / np.sqrt(np.maximum(np.abs(cross), 1e-9))
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    cross = cross * np.exp(-(fy ** 2 + fx ** 2) / (2 * _LOWPASS_SIGMA ** 2))
    corr = np.real(np.fft.ifft2(cross))
    py, px = np.unravel_index(int(np.argmax(corr)), corr.shape)
    oy = _subpixel(corr[(py - 1) % h, px], corr[py, px], corr[(py + 1) % h, px])
    ox = _subpixel(corr[py, (px - 1) % w], corr[py, px], corr[py, (px + 1) % w])
    dy = (py - h if py > h // 2 else py) + oy
    dx = (px - w if px > w // 2 else px) + ox
    return float(dy), float(dx)


def estimate_shift(prev: np.ndarray, curr: np.ndarray,
                   estimate_width: int = ESTIMATE_WIDTH) -> tuple[int, int]:
    """Whole-pixel global shift ``(dy, dx)`` taking ``prev`` onto ``curr``.

    Both are 2-D greyscale arrays at analysis size. ``(0, 0)`` when the frames
    disagree in shape, are too small, or the estimate exceeds
    :data:`MAX_SHIFT_FRACTION` of the frame (a cut, not a pan).
    """
    prev = np.asarray(prev, dtype=np.float32)
    curr = np.asarray(curr, dtype=np.float32)
    if prev.ndim != 2 or prev.shape != curr.shape or min(prev.shape) < 8:
        return 0, 0
    factor = max(1, int(prev.shape[1] // max(16, estimate_width)))
    small_prev = _block_mean(prev, factor)
    small_curr = _block_mean(curr, factor)
    dy, dx = phase_correlate(small_prev, small_curr)
    dy, dx = int(round(dy * factor)), int(round(dx * factor))
    h, w = prev.shape
    if abs(dy) > h * MAX_SHIFT_FRACTION or abs(dx) > w * MAX_SHIFT_FRACTION:
        return 0, 0
    return dy, dx


def _span(shift: int, size: int) -> tuple[slice, slice]:
    """(destination, source) slices moving content ``shift`` px along an axis."""
    if shift >= 0:
        return slice(shift, size), slice(0, size - shift)
    return slice(0, size + shift), slice(-shift, size)


def align_pair(prev, curr, dy: int, dx: int):
    """``(aligned_prev, valid)`` — ``prev`` moved by ``(dy, dx)`` onto ``curr``.

    ``valid`` is 1 where both frames see the same scene and 0 on the strip the
    pan uncovered. Works on the last two axes of numpy arrays or torch tensors
    of any leading shape, on any device: only slicing, ``* 0`` and ``+ 1``.
    """
    h, w = prev.shape[-2], prev.shape[-1]
    dy, dx = int(dy), int(dx)
    aligned = prev * 0
    valid = prev * 0
    if abs(dy) >= h or abs(dx) >= w:
        return aligned, valid
    ys_d, ys_s = _span(dy, h)
    xs_d, xs_s = _span(dx, w)
    aligned[..., ys_d, xs_d] = prev[..., ys_s, xs_s]
    valid[..., ys_d, xs_d] = 1
    return aligned, valid


def compensated_diff(prev, curr, dy: int, dx: int):
    """``|curr - aligned prev|`` with the uncovered strip zeroed.

    Zero rather than left as raw difference: a strip the camera just panned
    onto is new scenery, not movement.
    """
    if not dy and not dx:
        return abs(curr - prev)
    aligned, valid = align_pair(prev, curr, dy, dx)
    return abs(curr - aligned) * valid


def residual_energy(prev: np.ndarray, curr: np.ndarray, noise_floor: float = 8.0,
                    compensate: bool = True) -> float:
    """Mean per-pixel change beyond ``noise_floor``, with or without compensation.

    A small numpy reference for the detector's energy figure — what the tests
    and the CLI's quick checks use.
    """
    prev = np.asarray(prev, dtype=np.float32)
    curr = np.asarray(curr, dtype=np.float32)
    dy, dx = estimate_shift(prev, curr) if compensate else (0, 0)
    diff = compensated_diff(prev, curr, dy, dx)
    return float(np.clip(diff - noise_floor, 0, None).mean())
