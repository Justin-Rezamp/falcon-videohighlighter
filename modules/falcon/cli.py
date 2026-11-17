"""``python -m modules.falcon`` — Falcon plans, cuts and phone exports from a shell.

    plan   VIDEO [--mark FILE] [--mode marks|hybrid] [--out plan.json]
    cut    PLAN  [--out-dir DIR] [--copy]
    join   PLAN  --output reel.mp4 [--overwrite] [--loudnorm]
    phone  REEL  --preset 1080p|vertical|letterbox --output FILE [--overwrite]
    proxy  VIDEO [--cache-dir DIR]
    preset [--install [--overwrite]]

Every command that writes video takes ``--print-only`` to show the exact
ffmpeg lines without running them. Nothing here deletes or modifies the source
recording or its ``.mark``; ``join`` and ``phone`` refuse to overwrite an
existing file unless ``--overwrite`` is given.

Facts and inferences stay apart in the output: tap times come from the
``.mark`` file; any window slide in hybrid mode is printed as an inference. A
tap is a "parent tap", never a goal.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys

from modules.falcon import marks as fm
from modules.falcon import media


def _quote(args: list[str]) -> str:
    return " ".join(shlex.quote(a) for a in args)


def _video_facts(video: str, duration: float | None) -> dict:
    if duration:
        size = os.path.getsize(video) if os.path.isfile(video) else 0
        return {"duration": float(duration), "size": size}
    return media.probe_source(video)


def build_plan(video: str, *, mark: str | None = None, mode: str = "marks",
               before: float = fm.DEFAULT_PAD, after: float = fm.DEFAULT_PAD,
               merge_gap: float = fm.DEFAULT_MERGE_GAP, duration: float | None = None,
               score=None) -> dict:
    """The plan as a JSON-ready dict. ``score`` overrides the hybrid detector."""
    mark_path = fm.find_mark_file(video, mark)
    if not mark_path:
        raise fm.MarkFileError(
            f"No .mark found for {os.path.basename(video)} (looked beside it, in .data, "
            "and for a .vdata with the same size). Pass --mark.")
    facts = _video_facts(video, duration)
    dur = float(facts.get("duration") or 0)
    markset = fm.read_marks(mark_path, duration=dur or None)
    warnings: list[str] = []
    vdata_path = fm.vdata_for_mark(mark_path)
    if vdata_path:
        try:
            vdata = fm.read_vdata(vdata_path)
            warnings = fm.check_pairing(vdata, duration=dur or None,
                                        size=facts.get("size") or None,
                                        width=facts.get("width"), height=facts.get("height"))
            if not dur and vdata.get("duration"):
                dur = float(vdata["duration"])
        except fm.MarkFileError as exc:
            warnings.append(str(exc))
    if not dur:
        raise fm.MarkFileError("Could not tell how long the video is; pass --duration.")

    notes: list[str] = []
    if mode == "hybrid":
        from modules.falcon.hybrid import hybrid_plan
        if score is None:
            from modules.falcon.lite_motion import motion_score
            score = motion_score(video, dur)
        clips, notes = hybrid_plan(markset, score, dur, before=before, after=after,
                                   merge_gap=merge_gap)
    elif mode == "marks":
        clips = fm.plan_from_marks(markset, dur, before=before, after=after,
                                   merge_gap=merge_gap)
    else:
        raise ValueError(f"unknown mode {mode!r}")
    return {
        "video": os.path.abspath(video),
        "mark_file": os.path.abspath(mark_path),
        "vdata_file": os.path.abspath(vdata_path) if vdata_path else None,
        "mode": mode, "duration": round(dur, 3),
        "before": before, "after": after, "merge_gap": merge_gap,
        "taps": markset.clicks, "ignored": markset.ignored,
        "pairing_warnings": warnings, "notes": notes,
        "clips": [c.to_dict() for c in clips],
    }


def describe(plan: dict) -> str:
    clips = [fm.FalconClip.from_dict(c) for c in plan["clips"]]
    kept = [c for c in clips if c.keep]
    lines = [
        f"Facts: {len(plan['taps'])} {fm.TAP_LABEL}s in {os.path.basename(plan['mark_file'])}",
        f"Plan ({plan['mode']}): {len(clips)} clips, "
        f"{sum(c.merged for c in clips)} merged, reel {fm.fmt_clock(sum(c.duration for c in kept))}",
    ]
    for c in clips:
        flag = "" if c.keep else "  [dropped]"
        lines.append(f"  {c.id}  {fm.fmt_clock(c.start)}–{fm.fmt_clock(c.end)}  "
                     f"{c.filename}{flag}")
    for w in plan.get("pairing_warnings") or []:
        lines.append(f"Check pairing: {w}")
    for w in plan.get("ignored") or []:
        lines.append(f"Ignored: {w}")
    for n in plan.get("notes") or []:
        lines.append(n)
    return "\n".join(lines)


def _load(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _run(args: list[str], print_only: bool) -> None:
    print(_quote(args))
    if not print_only:
        subprocess.run(args, check=True)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m modules.falcon", description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("plan", help="plan clips from a .mark file")
    sp.add_argument("video")
    sp.add_argument("--mark")
    sp.add_argument("--mode", choices=("marks", "hybrid"), default="marks")
    sp.add_argument("--before", type=float, default=fm.DEFAULT_PAD)
    sp.add_argument("--after", type=float, default=fm.DEFAULT_PAD)
    sp.add_argument("--merge-gap", type=float, default=fm.DEFAULT_MERGE_GAP)
    sp.add_argument("--duration", type=float, help="skip probing (seconds)")
    sp.add_argument("--out", help="write the plan JSON here")

    sc = sub.add_parser("cut", help="cut the kept clips of a plan")
    sc.add_argument("plan")
    sc.add_argument("--out-dir", default="clips")
    sc.add_argument("--copy", action="store_true", help="stream copy (fast, keyframe-rough)")
    sc.add_argument("--print-only", action="store_true")

    sj = sub.add_parser("join", help="join cut clips into one reel")
    sj.add_argument("plan")
    sj.add_argument("--clips-dir", default="clips")
    sj.add_argument("--output", required=True)
    sj.add_argument("--overwrite", action="store_true")
    sj.add_argument("--loudnorm", action="store_true")
    sj.add_argument("--print-only", action="store_true")

    sph = sub.add_parser("phone", help="phone-friendly copy of a reel")
    sph.add_argument("reel")
    sph.add_argument("--preset", choices=media.PHONE_PRESETS, default="1080p")
    sph.add_argument("--output", required=True)
    sph.add_argument("--no-loudnorm", action="store_true")
    sph.add_argument("--overwrite", action="store_true")
    sph.add_argument("--print-only", action="store_true")

    spr = sub.add_parser("preset", help="install the falcon-soccer preset")
    spr.add_argument("--install", action="store_true")
    spr.add_argument("--overwrite", action="store_true")

    spx = sub.add_parser("proxy", help="make the low-res analysis copy")
    spx.add_argument("video")
    spx.add_argument("--cache-dir", default=os.path.join("cache", "falcon-proxies"))
    spx.add_argument("--print-only", action="store_true")

    a = p.parse_args(argv)
    try:
        if a.cmd == "plan":
            plan = build_plan(a.video, mark=a.mark, mode=a.mode, before=a.before,
                              after=a.after, merge_gap=a.merge_gap, duration=a.duration)
            print(describe(plan))
            if a.out:
                with open(a.out, "w", encoding="utf-8") as handle:
                    json.dump(plan, handle, indent=2)
                print(f"Plan written to {a.out}")
        elif a.cmd == "cut":
            plan = _load(a.plan)
            os.makedirs(a.out_dir, exist_ok=True)
            for c in plan["clips"]:
                if not c.get("keep", True):
                    continue
                dest = os.path.join(a.out_dir, c["filename"])
                if os.path.exists(dest):
                    print(f"exists, skipped: {dest}")
                    continue
                _run(media.cut_command(plan["video"], c["start"], c["end"], dest,
                                       copy=a.copy), a.print_only)
        elif a.cmd == "join":
            plan = _load(a.plan)
            if os.path.exists(a.output) and not a.overwrite:
                print(f"{a.output} already exists; pass --overwrite to replace it.")
                return 2
            listing = os.path.splitext(a.output)[0] + ".concat.txt"
            parts = [os.path.abspath(os.path.join(a.clips_dir, c["filename"]))
                     for c in plan["clips"] if c.get("keep", True)]
            body = "".join("file '" + x.replace("\\", "/").replace("'", "'\\''") + "'\n"
                           for x in parts)
            if not a.print_only:
                with open(listing, "w", encoding="utf-8") as handle:
                    handle.write(body)
            try:
                _run(media.concat_command(listing, a.output, overwrite=a.overwrite,
                                          loudnorm=a.loudnorm), a.print_only)
            finally:
                if not a.print_only and os.path.exists(listing):
                    os.remove(listing)
        elif a.cmd == "phone":
            if os.path.exists(a.output) and not a.overwrite:
                print(f"{a.output} already exists; pass --overwrite to replace it.")
                return 2
            _run(media.phone_command(a.reel, a.output, a.preset,
                                     loudnorm=not a.no_loudnorm, overwrite=a.overwrite),
                 a.print_only)
        elif a.cmd == "preset":
            from modules.falcon import preset
            if not a.install:
                print(open(preset.PRESET_FILE, encoding="utf-8").read())
            else:
                try:
                    print(f"Installed {preset.install(overwrite=a.overwrite)}")
                except FileExistsError as exc:
                    print(f"{exc}; pass --overwrite to replace it.")
                    return 2
        elif a.cmd == "proxy":
            info = media.probe_source(a.video)
            reason = media.proxy_reason(info)
            print(f"Proxy {'recommended: ' + reason if reason else 'not needed'}")
            if a.print_only:
                print(_quote(media.proxy_command(a.video, media.proxy_path(a.video, a.cache_dir))))
            else:
                print(media.make_proxy(a.video, a.cache_dir))
    except (fm.MarkFileError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0
