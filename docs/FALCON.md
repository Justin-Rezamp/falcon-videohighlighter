# XbotGo Falcon footage

This fork adds support for recordings from the XbotGo Falcon, an
auto-tracking camera used for youth soccer. Three things make Falcon footage
different, and each has a fix:

| Falcon footage | What goes wrong | What this fork does |
|---|---|---|
| A parent taps a button when something happens; the app saves the taps as a `.mark` file | Nothing in the app knew about them | Taps become a **parent tap** signal on the timeline and in the report, and can drive the cut directly |
| The camera pans by itself to follow play | The whole frame changes, so the pan is scored as action | Motion is measured after removing the camera's own pan (`camera_compensation`) |
| 4K HEVC at ~50 Mbit/s (an hour is 20+ GB) | Decoding is the slowest step | Motion can be analysed through a 360p proxy (`analysis_proxy`); the reel is still cut from the original |

A tap only says *a parent pressed the button near here*. It is never treated
or labelled as a goal.

## The .mark file

A JSON list. Only `click` entries count; the `start` entry at 0 is ignored.
`time` is seconds from the start of the recording.

```json
[{"type": "start", "time": 0}, {"type": "click", "time": 549}, ...]
```

The Falcon keeps `.mark`, `.vdata` and `.track` files in a hidden `.data`
folder named after its own file (`VID_20260425_133314_01_01.mark`). The app
finds the mark, in order:

1. the path you give (`falcon:` `mark_path`, `--mark`),
2. `<video name>.mark` beside the video or in `.data` (any case),
3. a `.mark` whose `.vdata` records the same byte size as the video, so a
   renamed copy such as `Your_Video_File.mp4` still pairs.

When a `.vdata` is present its duration, size and resolution are checked
against the video and any mismatch is reported.

## Clip plans

* **marks** (the Falcon Highlights runner rules): 10 s before and after each
  tap; windows that overlap or sit within 2 s merge; clips are numbered
  `clip01`… in time order and named `clip01_mark_9-09.mp4`, with `_merged`
  when a clip holds several taps.
* **hybrid**: the same windows, but each may slide up to 6 s earlier or 3 s
  later toward a clear detector peak (camera-compensated motion, audio…). The
  tap time never changes; each slide is reported as an *inference*.
* **off** (`mode: ''`): taps only add `parent_tap_points` to the normal scorer.

## In the app

Install the preset, then pick **falcon-soccer** under Presets:

```
python -m modules.falcon preset --install
```

It sets the `falcon:` section of config.yaml:

```yaml
falcon:
  marks: true
  mark_path: ''
  parent_tap_points: 10
  mode: marks          # marks | hybrid | ''
  pad_before: 10
  pad_after: 10
  merge_gap: 2
  camera_compensation: true
  analysis_proxy: auto # auto | true | false
```

The web sidecar reads the same section for `/run`, and offers
`GET /falcon/plan?path=…` (read-only). The MCP server has a `falcon_plan`
tool.

## Command line (no ML stack needed)

```
python -m modules.falcon plan  game.mp4 [--mark FILE] [--mode marks|hybrid] --out plan.json
python -m modules.falcon cut   plan.json --out-dir clips [--print-only]
python -m modules.falcon join  plan.json --clips-dir clips --output highlights.mp4 [--loudnorm]
python -m modules.falcon phone highlights.mp4 --preset 1080p|vertical|letterbox --output phone.mp4
python -m modules.falcon proxy game.mp4 [--print-only]
```

Edit `plan.json` (`"keep": false`, or change `start`/`end`) between `plan`
and `cut`. `--print-only` shows the exact ffmpeg lines without running them.
`join` and `phone` never overwrite an existing file unless `--overwrite` is
given, and nothing touches the source video or its `.mark`.

The command line needs only numpy and an ffmpeg binary
(`requirements-falcon-lite.txt`), so it runs where the full app cannot be
installed — including Windows on ARM64, where torch (PyPI), OpenCV and
OpenVINO have no wheels.
