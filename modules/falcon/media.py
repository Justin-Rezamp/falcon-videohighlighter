"""Analysis proxy and phone-friendly exports for Falcon recordings.

A Falcon game is 4K HEVC at ~50 Mbit/s: an hour is over 20 GB, and decoding
HEVC at that size is the slowest part of any analysis — on a laptop without a
hardware HEVC decoder it can take longer than the game. Every analysis in this
app works on small greyscale frames anyway, so :func:`make_proxy` writes a
640x360 H.264 copy (cached by file identity) that the motion pass reads
instead. Timestamps are unchanged, and the reel is always cut from the
original.

The export helpers build plain ffmpeg argument lists, so the CLI can show a
parent the exact command before anything runs:

* ``1080p``     — at most 1920x1080, H.264 High/AAC, faststart: plays on any phone
* ``vertical``  — 1080x1920, the full picture over a blurred fill of itself
* ``letterbox`` — 1080x1920, the full picture between black bars

and any of them can carry single-pass EBU R128 loudness normalisation
(``-16 LUFS``, the usual target for phones and the web).

Ported from the falcon-hypecut project (Apache-2.0, same authors).
"""
from __future__ import annotations

import hashlib
import os
import subprocess

#: Single-pass EBU R128 to -16 LUFS.
LOUDNORM = "loudnorm=I=-16:TP=-1.5:LRA=11"
PROXY_HEIGHT = 360
PHONE_PRESETS = ("1080p", "vertical", "letterbox")

_H264_OUT = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "21",
             "-profile:v", "high", "-level", "4.1", "-pix_fmt", "yuv420p"]
_AAC_OUT = ["-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-ac", "2"]


def probe_source(path: str) -> dict:
    """``{width, height, codec, duration, fps, has_audio, size}`` for a recording."""
    from modules.media.ffmpeg_tools import probe

    data = probe(path)
    streams = data.get("streams", []) or []
    video = next((s for s in streams if s.get("codec_type") == "video"), {}) or {}
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    fmt = data.get("format", {}) or {}

    def num(value, default=0.0):
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    rate = str(video.get("avg_frame_rate") or video.get("r_frame_rate") or "0/0")
    top, _, bottom = rate.partition("/")
    fps = num(top) / num(bottom, 1.0) if num(bottom, 0.0) else num(top)
    return {
        "width": int(num(video.get("width"))),
        "height": int(num(video.get("height"))),
        "codec": str(video.get("codec_name") or "").lower(),
        "duration": num(fmt.get("duration") or video.get("duration")),
        "fps": fps,
        "has_audio": audio is not None,
        "size": int(num(fmt.get("size"))) or (os.path.getsize(path)
                                              if os.path.isfile(path) else 0),
    }


def proxy_reason(info: dict) -> str:
    """Why this source should be analysed through a proxy, or ``""``."""
    reasons = []
    if info.get("codec") in {"hevc", "h265"}:
        reasons.append("HEVC (H.265) is slow to decode")
    if int(info.get("height") or 0) > 1080 or int(info.get("width") or 0) > 1920:
        reasons.append(f"{info.get('width')}x{info.get('height')} is larger than 1080p")
    return " and ".join(reasons)


def proxy_path(src: str, cache_dir: str) -> str:
    """Where :func:`make_proxy` puts (or finds) the proxy for ``src``."""
    st = os.stat(src)
    key = hashlib.sha1(f"{os.path.abspath(src)}|{st.st_size}|{int(st.st_mtime)}"
                       .encode("utf-8")).hexdigest()[:12]
    stem = os.path.splitext(os.path.basename(src))[0]
    return os.path.join(cache_dir, f"{stem}.proxy-{key}.mp4")


def proxy_command(src: str, dest: str, height: int = PROXY_HEIGHT) -> list[str]:
    """Video-only, fast-decoding H.264 copy for analysis. Never shared."""
    return ["ffmpeg", "-v", "error", "-nostdin", "-y", "-i", src,
            "-map", "0:v:0", "-vf", f"scale=-2:{int(height)}:flags=fast_bilinear",
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "30",
            "-pix_fmt", "yuv420p", "-an", dest]


def make_proxy(src: str, cache_dir: str, *, run=subprocess.run, log_fn=print) -> str:
    """Make (or reuse) the analysis proxy and return its path.

    Written to a ``.part`` file and renamed, so a cancelled run never leaves a
    half proxy that a later run would trust.
    """
    os.makedirs(cache_dir, exist_ok=True)
    dest = proxy_path(src, cache_dir)
    if os.path.isfile(dest) and os.path.getsize(dest) > 0:
        return dest
    tmp = dest[:-4] + ".part.mp4"
    log_fn(f"🎞️ Making a {PROXY_HEIGHT}p analysis copy (the reel is still cut "
           f"from the original)…")
    run(proxy_command(src, tmp), check=True)
    os.replace(tmp, dest)
    return dest


def cut_command(src: str, start: float, end: float, dest: str, *,
                copy: bool = False, crf: int = 20) -> list[str]:
    """One clip. Re-encoded by default: frame-accurate cuts and phone-safe H.264."""
    dur = f"{max(0.0, end - start):.3f}"
    head = ["ffmpeg", "-v", "error", "-nostdin", "-n"]
    if copy:
        return head + ["-ss", f"{start:.3f}", "-i", src, "-t", dur,
                       "-map", "0:v:0", "-map", "0:a:0?", "-c", "copy",
                       "-avoid_negative_ts", "make_zero", "-movflags", "+faststart", dest]
    return head + ["-ss", f"{start:.3f}", "-i", src, "-t", dur,
                   "-map", "0:v:0", "-map", "0:a:0?",
                   "-c:v", "libx264", "-preset", "veryfast", "-crf", str(int(crf)),
                   "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
                   "-ar", "48000", "-ac", "2", "-movflags", "+faststart", dest]


def concat_command(list_file: str, dest: str, *, overwrite: bool = False,
                   loudnorm: bool = False) -> list[str]:
    """Join clips listed in an ffmpeg concat file. ``-n`` unless overwriting."""
    args = ["ffmpeg", "-v", "error", "-nostdin", "-y" if overwrite else "-n",
            "-f", "concat", "-safe", "0", "-i", list_file,
            "-map", "0:v:0", "-map", "0:a:0?"]
    if loudnorm:
        args += ["-c:v", "copy", "-af", LOUDNORM, *_AAC_OUT]
    else:
        args += ["-c", "copy"]
    return args + ["-movflags", "+faststart", dest]


def phone_filter(preset: str) -> str:
    """The ``-vf`` / ``-filter_complex`` graph for a phone preset."""
    if preset == "1080p":
        return ("scale=w='min(1920,iw)':h='min(1080,ih)':force_original_aspect_ratio=decrease,"
                "scale=trunc(iw/2)*2:trunc(ih/2)*2")
    if preset == "letterbox":
        return ("scale=1080:1920:force_original_aspect_ratio=decrease,"
                "pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black,setsar=1")
    if preset == "vertical":
        return ("[0:v]split=2[bg][fg];"
                "[bg]scale=1080:1920:force_original_aspect_ratio=increase,"
                "crop=1080:1920,boxblur=20:2[bgb];"
                "[fg]scale=1080:-2[fgs];"
                "[bgb][fgs]overlay=(W-w)/2:(H-h)/2,setsar=1[v]")
    raise ValueError(f"unknown phone preset {preset!r}; pick one of {', '.join(PHONE_PRESETS)}")


def phone_command(src: str, dest: str, preset: str = "1080p", *,
                  loudnorm: bool = True, overwrite: bool = False) -> list[str]:
    """Re-encode a finished reel for phones. Never overwrites unless told to."""
    graph = phone_filter(preset)
    args = ["ffmpeg", "-v", "error", "-nostdin", "-y" if overwrite else "-n", "-i", src]
    if preset == "vertical":
        args += ["-filter_complex", graph, "-map", "[v]", "-map", "0:a:0?"]
    else:
        args += ["-map", "0:v:0", "-map", "0:a:0?", "-vf", graph]
    args += _H264_OUT + _AAC_OUT
    if loudnorm:
        args += ["-af", LOUDNORM]
    return args + ["-movflags", "+faststart", dest]
