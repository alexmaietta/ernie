# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import colorsys
import json
import shutil
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ernie_bench.analysis import summarize_run

BACKGROUND = (24, 24, 24)
TEXT = (235, 235, 235)
MIN_TILE = 256
VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".gif"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def point(scene: str, workload: dict) -> tuple:
    w = workload
    return (scene, w["worlds"], w["cameras_per_world"], w["width"], w["height"], tuple(w["outputs"]), w["tier"], w["spp"])


@dataclass
class Captured:
    backend: str
    scene: str
    workload: dict
    directory: Path
    steps: list[int]
    images: list[int]
    outputs: list[str]

    @property
    def point(self) -> tuple:
        return point(self.scene, self.workload)

    @property
    def label(self) -> str:
        return self.directory.name

    def sequence(self, output: str, image: int) -> np.ndarray:
        slot = self.images.index(image)
        return np.stack([np.load(self.directory / f"step-{step:05d}.npz")[output][slot] for step in self.steps])


def captures(run: Path) -> list[Captured]:
    manifest = json.loads((Path(run) / "manifest.json").read_text())
    found = []
    for outcome in manifest["outcomes"]:
        if outcome.get("status") == "ok" and outcome.get("captures"):
            directory = Path(outcome["captures"])
            index = json.loads((directory / "index.json").read_text())
            found.append(Captured(outcome["backend"], outcome["scene"], outcome["workload"], directory, index["steps"], index["images"], index["outputs"]))
    return found


def _srgb(linear: np.ndarray) -> np.ndarray:
    linear = np.clip(linear, 0.0, 1.0)
    encoded = np.where(linear <= 0.0031308, 12.92 * linear, 1.055 * np.power(linear, 1 / 2.4) - 0.055)
    return np.round(encoded * 255).astype(np.uint8)


def _palette(ids: np.ndarray) -> np.ndarray:
    colors = np.zeros((*ids.shape, 3), np.uint8)
    for value in np.unique(ids):
        if value > 0:
            colors[ids == value] = [round(255 * c) for c in colorsys.hsv_to_rgb((value * 0.618034) % 1.0, 0.65, 0.95)]
    return colors


def displayable(output: str, frames: np.ndarray, exposure: float | None = None) -> np.ndarray:
    if output == "rgb":
        luminance = frames[..., :3] @ np.array([0.2126, 0.7152, 0.0722])
        bright = np.percentile(luminance, 99) if luminance.size else 0.0
        scale = exposure if exposure is not None else (0.9 / bright if bright > 0 else 1.0)
        return _srgb(frames[..., :3] * scale)
    if output == "depth":
        depth = frames[..., 0].astype(np.float64)
        valid = np.isfinite(depth) & (depth > 0)
        gray = np.zeros(depth.shape)
        if valid.any():
            near, far = np.percentile(depth[valid], [1, 99])
            gray[valid] = 1 - np.clip((depth[valid] - near) / (far - near if far > near else 1.0), 0, 1)
            gray[valid] = 0.15 + 0.85 * gray[valid]
        return np.repeat(np.round(gray * 255).astype(np.uint8)[..., None], 3, axis=-1)
    if output == "segmentation":
        return _palette(frames[..., 0])
    raise ValueError(f"no display mapping for {output!r}")


def upscale(image: np.ndarray, minimum: int = MIN_TILE) -> np.ndarray:
    factor = max(1, minimum // max(image.shape[:2]))
    return image.repeat(factor, axis=0).repeat(factor, axis=1) if factor > 1 else image


def fit_height(image: np.ndarray, height: int) -> np.ndarray:
    if image.shape[0] == height:
        return image
    width = max(1, round(image.shape[1] * height / image.shape[0]))
    return np.asarray(Image.fromarray(image).resize((width, height), Image.NEAREST))


def _text_bar(text: str, width: int, size: int) -> np.ndarray:
    height = round(size * 1.7)
    bar = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(bar)
    font = ImageFont.load_default(size=size)
    while size > 8 and draw.textlength(text, font=font) > width - 8:
        size -= 1
        font = ImageFont.load_default(size=size)
    draw.text((width / 2, height / 2), text, fill=TEXT, font=font, anchor="mm")
    return np.asarray(bar)


def labelled(image: np.ndarray, label: str) -> np.ndarray:
    return np.concatenate([_text_bar(label, image.shape[1], max(12, image.shape[1] // 22)), image])


def grid(tiles: list[np.ndarray], labels: list[str] | None = None, title: str | None = None, columns: int | None = None, gap: int = 6) -> np.ndarray:
    if labels:
        tiles = [labelled(tile, label) for tile, label in zip(tiles, labels)]
    columns = columns or len(tiles)
    rows = -(-len(tiles) // columns)
    height = max(tile.shape[0] for tile in tiles)
    width = max(tile.shape[1] for tile in tiles)
    canvas = np.full((rows * height + (rows + 1) * gap, columns * width + (columns + 1) * gap, 3), BACKGROUND, np.uint8)
    for index, tile in enumerate(tiles):
        row, column = divmod(index, columns)
        top = gap + row * (height + gap) + (height - tile.shape[0]) // 2
        left = gap + column * (width + gap) + (width - tile.shape[1]) // 2
        canvas[top : top + tile.shape[0], left : left + tile.shape[1]] = tile
    if title:
        canvas = np.concatenate([_text_bar(title, canvas.shape[1], max(14, min(28, canvas.shape[1] // 40))), canvas])
    return canvas


def save_image(path: Path, image: np.ndarray) -> Path:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path)
    return Path(path)


def read_image(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def ffmpeg() -> str:
    try:
        import imageio_ffmpeg
    except ImportError:
        found = shutil.which("ffmpeg")
        if not found:
            raise RuntimeError("movies need ffmpeg: pip install 'ernie-bench[media]'") from None
        return found
    return imageio_ffmpeg.get_ffmpeg_exe()


def write_movie(path: Path, frames: Iterable[np.ndarray], fps: float = 30) -> Path:
    frames = iter(frames)
    first = next(frames)
    height, width = first.shape[:2]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg(), "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
        str(path),
    ]
    with subprocess.Popen(command, stdin=subprocess.PIPE) as encoder:
        for frame in chain([first], frames):
            if frame.shape[:2] != (height, width):
                raise ValueError(f"movie frames must all be {width}x{height}, got {frame.shape[1]}x{frame.shape[0]}")
            encoder.stdin.write(np.ascontiguousarray(frame, np.uint8).tobytes())
        encoder.stdin.close()
        if encoder.wait():
            raise RuntimeError(f"ffmpeg failed writing {path}")
    return Path(path)


def read_movie(path: Path) -> Iterator[np.ndarray]:
    import imageio_ffmpeg

    reader = imageio_ffmpeg.read_frames(str(path))
    width, height = next(reader)["size"]
    for frame in reader:
        yield np.frombuffer(frame, np.uint8).reshape(height, width, 3)


def read_media(path: Path) -> list[np.ndarray]:
    path = Path(path)
    if path.is_dir():
        return [read_image(child) for child in sorted(path.iterdir()) if child.suffix.lower() in IMAGE_SUFFIXES]
    if path.suffix.lower() in VIDEO_SUFFIXES:
        return list(read_movie(path))
    return [read_image(path)]


def montage(sequences: list[list[np.ndarray]], labels: list[str] | None = None, title: str | None = None, columns: int | None = None) -> list[np.ndarray]:
    height = max(upscale(sequence[0]).shape[0] for sequence in sequences)
    length = max(len(sequence) for sequence in sequences)
    return [
        grid([fit_height(sequence[min(step, len(sequence) - 1)], height) for sequence in sequences], labels, title, columns)
        for step in range(length)
    ]


def _point_title(captured: Captured, output: str, image: int) -> str:
    w = captured.workload
    return f"{captured.scene} · {w['worlds']} worlds x {w['cameras_per_world']} cams · {w['width']}x{w['height']} · {w['tier']} · {output} · image {image}"


def captured_frames(run: Path, exposure: float | None) -> Iterator[tuple[Captured, str, int, np.ndarray]]:
    for captured in captures(run):
        for output in captured.outputs:
            for image in captured.images:
                yield captured, output, image, displayable(output, captured.sequence(output, image), exposure)


def export_images(run: Path, exposure: float | None = None) -> list[Path]:
    return [
        save_image(Path(run) / "media" / "images" / captured.label / output / f"image-{image:05d}" / f"step-{step:05d}.png", upscale(frame))
        for captured, output, image, frames in captured_frames(run, exposure)
        for step, frame in zip(captured.steps, frames)
    ]


def export_movies(run: Path, fps: float = 30, exposure: float | None = None) -> list[Path]:
    return [
        write_movie(Path(run) / "media" / "movies" / f"{captured.label}-{output}-image-{image:05d}.mp4", (upscale(frame) for frame in frames), fps)
        for captured, output, image, frames in captured_frames(run, exposure)
    ]


def export_panels(
    runs: list[Path],
    outputs: list[str] | None = None,
    images: list[int] | None = None,
    movie: bool = False,
    fps: float = 30,
    exposure: float | None = None,
) -> list[Path]:
    groups = defaultdict(list)
    medians = {}
    for run in runs:
        for captured in captures(run):
            groups[captured.point].append(captured)
        for record in summarize_run(run)[1]:
            medians[(record["backend"], *point(record["scene"], record | {"outputs": record["outputs"].split(",")}))] = record.get("median_seconds")
    destination = Path(runs[0]) / "media" / "panels"
    written = []
    for key, members in groups.items():
        shared_outputs = [o for o in members[0].outputs if all(o in m.outputs for m in members) and (not outputs or o in outputs)]
        shared_images = [i for i in members[0].images if all(i in m.images for m in members) and (images is None or i in images)]
        steps = sorted(set.intersection(*(set(m.steps) for m in members)))
        for output in shared_outputs:
            for image in shared_images:
                sequences, labels = [], []
                for member in members:
                    frames = displayable(output, member.sequence(output, image), exposure)
                    sequences.append([upscale(frames[member.steps.index(step)]) for step in steps])
                    median = medians.get((member.backend, *key))
                    labels.append(f"{member.backend} · {median * 1e3:.3g} ms/frame" if median else member.backend)
                title = _point_title(members[0], output, image)
                panels = montage(sequences, labels, title)
                stem = f"{key[0]}-w{key[1]}-c{key[2]}-{key[3]}x{key[4]}-{key[6]}-spp{key[7]}-{output}-image-{image:05d}"
                if movie and len(steps) > 1:
                    written.append(write_movie(destination / f"{stem}.mp4", panels, fps))
                else:
                    written.extend(save_image(destination / f"{stem}-step-{step:05d}.png", panel) for step, panel in zip(steps, panels))
    return written
