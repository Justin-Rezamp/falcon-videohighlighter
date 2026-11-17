"""XbotGo Falcon ``.mark`` files, and the plain mark-driven cut plan.

During a game a parent taps a button in the XbotGo app; the app exports a
``.mark`` file for the recording. It is a JSON list of small objects::

    [{"type": "start", "time": 0}, {"type": "click", "time": 549}, ...]

``time`` is seconds from the start of the recording. Only ``click`` entries
mean anything — the ``start`` entry at 0 is the recording starting — and a
click is *not* proof that anything in particular happened: it is a parent
saying "something worth seeing happened near here". Everything below treats it
that way and labels it a *parent tap*. Nothing here ever invents a mark time;
marks outside the video are reported and dropped, never moved.

The Falcon keeps ``.mark``, ``.vdata`` and ``.track`` files in a hidden
``.data`` folder beside the recordings, named after the camera's own file name
(``VID_20260425_133314_01_01.mark``). Recordings are often renamed once they
are copied off the card, so :func:`find_mark_file` also pairs a video with a
``.vdata`` whose recorded byte size matches, and an explicit path always wins.

:func:`plan_from_marks` reproduces the Falcon Highlights runner rules exactly:

* pad ``before`` / ``after`` seconds around each click (default 10 / 10),
* merge windows that overlap or sit within ``merge_gap`` seconds (default 2),
* number clips ``clip01``…, named ``clip01_mark_9-09.mp4`` after the first
  mark they contain, with ``_merged`` appended when a clip holds several.

Ported from the falcon-hypecut project (Apache-2.0, same authors).
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, field
from typing import Any

DEFAULT_PAD = 10.0
DEFAULT_MERGE_GAP = 2.0

#: What a mark is called anywhere a person reads it. Never "goal".
TAP_LABEL = "parent tap"


class MarkFileError(ValueError):
    """The .mark file could not be read as a Falcon mark list."""


@dataclass
class MarkSet:
    """The click marks of one recording, plus what was ignored and why."""

    clicks: list[float]
    source: str = ""
    ignored: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.clicks)


@dataclass
class FalconClip:
    """One clip of a Falcon plan.

    ``kind`` is ``"mark"`` when the clip exists because of one or more parent
    taps. ``shift`` holds, per mark, how far the hybrid planner slid the window
    toward a detector peak — an inference, reported as one; 0 in marks-only.
    """

    id: str
    kind: str
    start: float
    end: float
    marks: list[float] = field(default_factory=list)
    filename: str = ""
    keep: bool = True
    score: float = 0.0
    shift: list[float] = field(default_factory=list)
    note: str = ""

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    @property
    def merged(self) -> bool:
        return len(self.marks) > 1

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["start"] = round(self.start, 3)
        data["end"] = round(self.end, 3)
        data["duration"] = round(self.duration, 3)
        data["merged"] = self.merged
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FalconClip":
        known = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})


# ------------------------------------------------------------------- parsing


def parse_marks(data: Any, *, duration: float | None = None) -> MarkSet:
    """Click times (seconds, sorted, de-duplicated) from parsed ``.mark`` JSON.

    Entries that are not clicks are skipped silently (``start`` is expected);
    clicks without a usable time, negative times, and — when ``duration`` is
    given — times past the end of the video are skipped *and reported* in
    ``ignored``, because a parent will want to know a tap went missing.
    """
    if isinstance(data, dict) and isinstance(data.get("marks"), list):
        data = data["marks"]
    if not isinstance(data, list):
        raise MarkFileError("A .mark file should be a JSON list of {type, time} objects")

    clicks: list[float] = []
    ignored: list[str] = []
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            ignored.append(f"entry {index}: not an object")
            continue
        if str(item.get("type", "")).strip().lower() != "click":
            continue
        raw = item.get("time")
        try:
            t = float(raw)
        except (TypeError, ValueError):
            ignored.append(f"entry {index}: click without a usable time ({raw!r})")
            continue
        if not math.isfinite(t) or t < 0:
            ignored.append(f"entry {index}: click at invalid time {raw!r}")
            continue
        if duration is not None and t > duration:
            ignored.append(f"entry {index}: click at {fmt_clock(t)} is after the "
                           f"video ends ({fmt_clock(duration)})")
            continue
        clicks.append(t)
    return MarkSet(clicks=sorted(set(clicks)), ignored=ignored)


def _read_json(path: str) -> Any:
    try:
        with open(path, encoding="utf-8-sig") as handle:
            text = handle.read()
    except OSError as exc:
        raise MarkFileError(f"Could not read {path}: {exc}") from exc
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise MarkFileError(f"{os.path.basename(path)} is not valid JSON "
                            f"({exc.msg} at line {exc.lineno})") from exc


def read_marks(path: str, *, duration: float | None = None) -> MarkSet:
    """Read and parse a ``.mark`` file. See :func:`parse_marks`."""
    marks = parse_marks(_read_json(str(path)), duration=duration)
    marks.source = str(path)
    return marks


def read_vdata(path: str) -> dict:
    """The camera's ``.vdata`` sidecar (duration, fps, size, width, height…).

    Returns only the fields this package uses, as numbers where they are
    numbers; an unreadable file raises :class:`MarkFileError`.
    """
    data = _read_json(str(path))
    if not isinstance(data, dict):
        raise MarkFileError(f"{os.path.basename(path)} is not a .vdata object")
    out: dict[str, Any] = {"source": str(path)}
    for key in ("duration", "fps", "size", "width", "height"):
        try:
            out[key] = float(data[key]) if key in ("duration", "fps") else int(data[key])
        except (KeyError, TypeError, ValueError):
            pass
    if isinstance(data.get("file_name"), str):
        out["file_name"] = data["file_name"]
    return out


def _data_dirs(video: str) -> list[str]:
    folder = os.path.dirname(os.path.abspath(video))
    return [folder, os.path.join(folder, ".data")]


def _find_named(folder: str, names: list[str]) -> str | None:
    for name in names:
        candidate = os.path.join(folder, name)
        if os.path.isfile(candidate):
            return candidate
    if os.path.isdir(folder):
        wanted = {n.lower() for n in names}
        for entry in sorted(os.listdir(folder)):
            full = os.path.join(folder, entry)
            if entry.lower() in wanted and os.path.isfile(full):
                return full
    return None


def find_mark_file(video: str, explicit: str | None = None) -> str | None:
    """The ``.mark`` file that belongs to ``video``, or None.

    In order: ``explicit`` when given (it must exist); ``<stem>.mark`` or
    ``<name>.mark`` next to the video or in a ``.data`` folder beside it
    (case-insensitive); and finally a ``.mark`` in either place whose
    ``.vdata`` records the same byte size as the video — which is how a
    renamed copy (``Your_Video_File.mp4``) finds
    ``.data/VID_20260425_133314_01_01.mark``.
    """
    if explicit:
        if not os.path.isfile(explicit):
            raise MarkFileError(f"Mark file not found: {explicit}")
        return explicit
    stem, _ext = os.path.splitext(os.path.basename(video))
    names = [stem + ".mark", os.path.basename(video) + ".mark"]
    for folder in _data_dirs(video):
        hit = _find_named(folder, names)
        if hit:
            return hit
    vdata = find_vdata_by_size(video)
    if vdata:
        mark = os.path.splitext(vdata)[0] + ".mark"
        if os.path.isfile(mark):
            return mark
    return None


def find_vdata_by_size(video: str) -> str | None:
    """A ``.vdata`` beside the video (or in ``.data``) recording its exact size."""
    try:
        size = os.path.getsize(video)
    except OSError:
        return None
    for folder in _data_dirs(video):
        if not os.path.isdir(folder):
            continue
        for entry in sorted(os.listdir(folder)):
            if not entry.lower().endswith(".vdata"):
                continue
            try:
                if read_vdata(os.path.join(folder, entry)).get("size") == size:
                    return os.path.join(folder, entry)
            except MarkFileError:
                continue
    return None


def vdata_for_mark(mark_path: str) -> str | None:
    """The ``.vdata`` written alongside a ``.mark``, if the camera left one."""
    candidate = os.path.splitext(mark_path)[0] + ".vdata"
    return candidate if os.path.isfile(candidate) else None


def check_pairing(vdata: dict, *, duration: float | None = None,
                  size: int | None = None, width: int | None = None,
                  height: int | None = None) -> list[str]:
    """Plain-English reasons a video and a ``.vdata`` may not belong together.

    Empty when everything that can be compared agrees. Duration is allowed
    two seconds of slack (container vs. camera rounding).
    """
    problems = []
    if duration and vdata.get("duration") and abs(duration - vdata["duration"]) > 2.0:
        problems.append(f"video runs {fmt_clock(duration)} but the camera recorded "
                        f"{fmt_clock(vdata['duration'])}")
    if size and vdata.get("size") and size != vdata["size"]:
        problems.append(f"video is {size:,} bytes but the camera recorded "
                        f"{vdata['size']:,}")
    if width and height and vdata.get("width") and vdata.get("height"):
        if (width, height) != (vdata["width"], vdata["height"]):
            problems.append(f"video is {width}x{height} but the camera recorded "
                            f"{vdata['width']}x{vdata['height']}")
    return problems


# ------------------------------------------------------------------ planning


def merge_windows(windows, merge_gap: float = DEFAULT_MERGE_GAP):
    """Merge ``(start, end, marks)`` windows that overlap or sit within ``merge_gap``.

    Same rule as the runner: a window joins the previous one when it starts at
    or before ``previous end + merge_gap``.
    """
    merged: list[list[Any]] = []
    for start, end, marks in sorted(windows, key=lambda w: (w[0], w[1])):
        if merged and start <= merged[-1][1] + merge_gap:
            merged[-1][1] = max(merged[-1][1], end)
            merged[-1][2].extend(marks)
        else:
            merged.append([start, end, list(marks)])
    return [(float(s), float(e), sorted(m)) for s, e, m in merged]


def plan_from_marks(marks, duration: float, *, before: float = DEFAULT_PAD,
                    after: float = DEFAULT_PAD,
                    merge_gap: float = DEFAULT_MERGE_GAP) -> list[FalconClip]:
    """Marks-only plan: the runner's pad / merge / naming rules, exactly."""
    clicks = marks.clicks if isinstance(marks, MarkSet) else sorted({float(m) for m in marks})
    windows = [(max(0.0, t - before), min(duration, t + after), [t]) for t in clicks]
    clips = [
        FalconClip(id="", kind="mark", start=round(s, 3), end=round(e, 3), marks=m,
                   shift=[0.0] * len(m),
                   note=" + ".join(f"{TAP_LABEL} {fmt_clock(x)}" for x in m))
        for s, e, m in merge_windows(windows, merge_gap)
    ]
    return number_clips(clips)


def number_clips(clips: list[FalconClip]) -> list[FalconClip]:
    """Give clips their IDs and filenames, in time order.

    ``clip01_mark_9-09.mp4`` (``_merged`` when it holds several marks).
    """
    ordered = sorted(clips, key=lambda c: (c.start, c.end))
    for i, clip in enumerate(ordered, start=1):
        clip.id = f"clip{i:02d}"
        anchor = clip.marks[0] if clip.marks else clip.start
        name = f"{clip.id}_{clip.kind}_{slug_clock(anchor)}"
        if clip.merged:
            name += "_merged"
        clip.filename = name + ".mp4"
    return ordered


# ------------------------------------------------------------------ clocks


def slug_clock(seconds: float) -> str:
    """``549`` -> ``9-09``; ``3725`` -> ``1-02-05``. Safe in a filename."""
    total = int(round(max(0.0, float(seconds))))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h}-{m:02d}-{s:02d}" if h else f"{m}-{s:02d}"


def fmt_clock(seconds: float) -> str:
    """``549.5`` -> ``9:09.5`` — for people, not filenames."""
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    body = f"{s:05.2f}".rstrip("0").rstrip(".")
    if len(body.split(".")[0]) < 2:
        body = "0" + body
    return f"{h}:{m:02d}:{body}" if h else f"{m}:{body}"
