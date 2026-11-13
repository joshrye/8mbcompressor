#!/usr/bin/env python3
"""Compress a video to fit under a target file size (two-pass x264 + AAC).

Usage:
    python main.py <input_file> [-o OUTPUT]

The target size gives a total bitrate budget (video + audio). The first rung of
this ladder that leaves enough bitrate for the video is used:

  1. 1080p, source audio untouched.
  2. 720p, audio at the source rate, then 128, 96 and (favor = "video" only)
     min_audio_kbps.
  3. 480p, audio at the last-resort rate (stage4_* settings), then min_audio_kbps.

Sources are never upscaled, and a source without audio skips every audio step.
If the result is still over the target, it is re-encoded at a lower bitrate.

Optional settings live in compress_config.toml (current directory first, then
next to this script). Requires Python 3.11+, ffmpeg and ffprobe.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import tomllib
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import NamedTuple

CONFIG_FILE = "compress_config.toml"

# Resolution ladder: (short side in pixels, minimum video kbps worth using).
TIERS = ((1080, 1870), (720, 870))
LAST_RESORT_HEIGHT = 480
MIN_USABLE_VIDEO_KBPS = 100  # floor for the last resort and for size retries

AUDIO_STEPS_KBPS = (128, 96)  # tried at 720p, between the source rate and the floor
MIN_AUDIO_KBPS = 8            # lowest bitrate we will ask the encoder for
MAX_AUDIO_FLOOR_KBPS = 128    # highest allowed value of min_audio_kbps

# 1 MB is budgeted as 8192 kbit, ~2.4% under the real 8388.6, as headroom for
# container overhead and rate-control drift.
BUDGET_KBIT_PER_MB = 8192
BYTES_PER_MB = 1024 * 1024

MAX_SIZE_RETRIES = 2
SIZE_RETRY_MARGIN = 0.97  # aim this far below the limit when retrying


class CompressionError(Exception):
    """A problem the user can act on (bad input, bad config, ffmpeg failure)."""


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

@dataclass
class Config:
    target_mb: float = 7.5
    min_audio_kbps: int = 48          # lowest audio rate the script may drop to
    stage4_min_audio_kbps: int = 96   # last-resort audio rate when favor = "video"
    stage4_max_audio_kbps: int = 128  # last-resort audio rate when favor = "audio"
    favor: str = "video"              # "video": cut audio before resolution; "audio": the reverse

    def __post_init__(self) -> None:
        if self.target_mb <= 0:
            raise CompressionError(f"'target_mb' must be positive, got: {self.target_mb}")
        if self.favor not in ("video", "audio"):
            raise CompressionError(f"'favor' must be 'video' or 'audio', got: {self.favor!r}")
        if min(self.stage4_min_audio_kbps, self.stage4_max_audio_kbps) < MIN_AUDIO_KBPS:
            raise CompressionError(f"stage4 audio bitrates must be at least {MIN_AUDIO_KBPS}kbps")
        self.min_audio_kbps = max(MIN_AUDIO_KBPS, min(MAX_AUDIO_FLOOR_KBPS, self.min_audio_kbps))


def load_config() -> Config:
    for directory in (Path.cwd(), Path(__file__).resolve().parent):
        path = directory / CONFIG_FILE
        if path.is_file():
            break
    else:
        return Config()

    try:
        raw = tomllib.loads(path.read_text())
        if unknown := raw.keys() - Config.__dataclass_fields__.keys():
            raise ValueError(f"unknown keys: {', '.join(sorted(unknown))}")
        return Config(**raw)
    except (OSError, tomllib.TOMLDecodeError, TypeError, ValueError) as exc:
        raise CompressionError(f"Bad config {path}: {exc}") from exc


# --------------------------------------------------------------------------- #
# Probing
# --------------------------------------------------------------------------- #

def run(cmd: list[str]) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as exc:
        tail = "\n".join(exc.stderr.strip().splitlines()[-10:])
        raise CompressionError(f"{cmd[0]} failed:\n{tail}") from exc


def probe(path: Path, *args: str) -> dict:
    return json.loads(run(["ffprobe", "-v", "error", *args, "-of", "json", str(path)]).stdout)


@dataclass(frozen=True)
class Source:
    duration: float    # seconds
    short_side: int    # the "p" in 1080p, after display rotation
    is_vertical: bool
    audio_tracks: int  # 0 when there is no audio
    audio_kbps: int    # bitrate of the largest track (tracks get mixed); 0 if none


def probe_source(path: Path) -> Source:
    info = probe(path, "-select_streams", "v:0", "-show_streams", "-show_format")
    if not info.get("streams"):
        raise CompressionError(f"No video stream found in {path}")
    stream = info["streams"][0]

    width, height = stream["width"], stream["height"]
    rotation = next((s["rotation"] for s in stream.get("side_data_list", []) if "rotation" in s), 0)
    if round(float(rotation)) % 180 == 90:  # ffmpeg autorotates, so use the displayed size
        width, height = height, width

    duration = float(info["format"].get("duration") or stream.get("duration") or 0)
    if duration <= 0:
        raise CompressionError("Could not determine video duration")

    packets = probe(path, "-select_streams", "a", "-show_entries", "packet=stream_index,size")
    bytes_per_track = Counter()
    for packet in packets.get("packets", []):
        bytes_per_track[packet["stream_index"]] += int(packet["size"])
    audio_kbps = 0
    if bytes_per_track:
        # Floored so a tiny but real track is never mistaken for "no audio".
        audio_kbps = max(MIN_AUDIO_KBPS, round(max(bytes_per_track.values()) * 8 / duration / 1000))

    return Source(duration, min(width, height), height > width, len(bytes_per_track), audio_kbps)


# --------------------------------------------------------------------------- #
# Planning
# --------------------------------------------------------------------------- #

class Rung(NamedTuple):
    height: int
    audio_kbps: int
    min_video_kbps: int


@dataclass(frozen=True)
class Plan:
    video_kbps: int
    audio_kbps: int
    height: int | None  # target short side; None keeps the source size


def audio_options(cfg: Config, src: Source) -> list[int]:
    """Audio bitrates to try, best first. Starts with the source rate (0 if no audio)."""
    if not src.audio_tracks:
        return [0]
    steps = (*AUDIO_STEPS_KBPS, *([cfg.min_audio_kbps] if cfg.favor == "video" else []))
    lower = {k for k in steps if cfg.min_audio_kbps <= k < src.audio_kbps}
    return sorted({src.audio_kbps, *lower}, reverse=True)


def ladder(cfg: Config, src: Source) -> list[Rung]:
    """Every (resolution, audio) combination worth trying, best first."""
    audio = audio_options(cfg, src)
    rungs = []
    for height, min_video in TIERS:
        if src.short_side < height:
            continue
        # At 1080p keep the source audio: give up resolution before audio.
        for kbps in audio[:1] if height == TIERS[0][0] else audio:
            rungs.append(Rung(height, kbps, min_video))

    preferred = cfg.stage4_min_audio_kbps if cfg.favor == "video" else cfg.stage4_max_audio_kbps
    last_audio = min(preferred, audio[0])
    for kbps in (last_audio, min(cfg.min_audio_kbps, last_audio)):
        rungs.append(Rung(LAST_RESORT_HEIGHT, kbps, MIN_USABLE_VIDEO_KBPS))
    return rungs


def make_plan(cfg: Config, src: Source, budget_kbps: float) -> Plan:
    for rung in ladder(cfg, src):
        video = int(budget_kbps - rung.audio_kbps)
        if video >= rung.min_video_kbps:
            return Plan(video, rung.audio_kbps, rung.height if src.short_side > rung.height else None)
    raise CompressionError(
        f"Only {max(video, 0)}kbps of video bitrate fits in the target size; "
        "increase 'target_mb' or use a shorter clip."
    )


# --------------------------------------------------------------------------- #
# Encoding
# --------------------------------------------------------------------------- #

def video_filter(src: Source, height: int | None) -> str:
    if height is None:
        return "scale=trunc(iw/2)*2:trunc(ih/2)*2"  # yuv420p needs even dimensions
    # "-2" keeps the aspect ratio and rounds the other side to an even number.
    return f"scale={height}:-2" if src.is_vertical else f"scale=-2:{height}"


def audio_args(src: Source, kbps: int) -> list[str]:
    """One AAC track; several source tracks are mixed together."""
    if not kbps:
        return ["-an"]
    encode = ["-c:a", "aac", "-b:a", f"{kbps}k"]
    if src.audio_tracks == 1:
        return ["-map", "0:a:0", *encode]
    inputs = "".join(f"[0:a:{i}]" for i in range(src.audio_tracks))
    # normalize=0 keeps each track at full volume; the limiter stops the sum clipping.
    mix = f"{inputs}amix=inputs={src.audio_tracks}:duration=longest:normalize=0,alimiter=limit=0.95[aout]"
    return ["-filter_complex", mix, "-map", "[aout]", *encode]


def encode(input_file: Path, output_file: Path, src: Source, plan: Plan, workdir: Path) -> None:
    print(f"  -> {plan.height or src.short_side}p, video {plan.video_kbps}k, audio {plan.audio_kbps}kbps")
    # Only the first video stream is mapped; subtitles and the like stay out of the budget.
    common = [
        "ffmpeg", "-y", "-nostdin", "-hide_banner", "-loglevel", "error",
        "-i", str(input_file), "-map", "0:v:0", "-vf", video_filter(src, plan.height),
        "-c:v", "libx264", "-b:v", f"{plan.video_kbps}k", "-pix_fmt", "yuv420p",
        "-passlogfile", str(workdir / "pass"),
    ]
    run([*common, "-pass", "1", "-an", "-f", "null", os.devnull])
    run([*common, "-pass", "2", *audio_args(src, plan.audio_kbps),
         "-movflags", "+faststart", str(output_file)])


def lowered(plan: Plan, size: int, limit: float) -> Plan:
    """The plan with a video bitrate that should bring a file of `size` bytes under `limit`."""
    total_kbps = (plan.video_kbps + plan.audio_kbps) * limit / size * SIZE_RETRY_MARGIN
    video = int(total_kbps) - plan.audio_kbps
    if video < MIN_USABLE_VIDEO_KBPS:
        raise CompressionError("Could not reach the target size without destroying video quality")
    return replace(plan, video_kbps=video)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def compress(input_file: Path, output_file: Path, cfg: Config) -> None:
    if not input_file.is_file():
        raise CompressionError(f"File not found: {input_file}")
    if output_file.resolve() == input_file.resolve():
        raise CompressionError("Output path must differ from the input path")

    src = probe_source(input_file)
    budget_kbps = cfg.target_mb * BUDGET_KBIT_PER_MB / src.duration
    plan = make_plan(cfg, src, budget_kbps)
    print(f"Source: {src.short_side}p {'vertical' if src.is_vertical else 'landscape'}, "
          f"{src.duration:.1f}s, audio {f'{src.audio_kbps}kbps' if src.audio_kbps else 'none'} | "
          f"budget {int(budget_kbps)}kbps | favor {cfg.favor}")

    limit = cfg.target_mb * BYTES_PER_MB
    output_file.parent.mkdir(parents=True, exist_ok=True)
    # Work in a scratch dir beside the output, so a failed run never clobbers an
    # existing output and the final os.replace stays on one filesystem.
    with tempfile.TemporaryDirectory(prefix=".compress_", dir=output_file.parent) as scratch:
        workdir = Path(scratch)
        partial = workdir / output_file.name
        for attempt in range(MAX_SIZE_RETRIES + 1):
            encode(input_file, partial, src, plan, workdir)
            size = partial.stat().st_size
            if size <= limit or attempt == MAX_SIZE_RETRIES:
                break
            print(f"  ! {size / BYTES_PER_MB:.2f} MB is over the {limit / BYTES_PER_MB:g} MB target, "
                  "retrying lower")
            plan = lowered(plan, size, limit)
        os.replace(partial, output_file)

    print(f"\n✓ Done -> {output_file} ({size / BYTES_PER_MB:.2f} MB)")
    if size > limit:
        print(f"Warning: output is still over the {limit / BYTES_PER_MB:g} MB target", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compress a video to a target file size.")
    parser.add_argument("input_file", type=Path)
    parser.add_argument("-o", "--output", type=Path, default=Path("output.mp4"),
                        help="output path (default: output.mp4)")
    args = parser.parse_args(argv)

    try:
        if missing := [t for t in ("ffmpeg", "ffprobe") if not shutil.which(t)]:
            raise CompressionError(f"Required tool(s) not found on PATH: {', '.join(missing)}")
        compress(args.input_file, args.output, load_config())
    except CompressionError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nCancelled.", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
