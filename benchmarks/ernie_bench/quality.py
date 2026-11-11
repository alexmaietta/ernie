# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import json
from collections import defaultdict
from functools import cache
from pathlib import Path

import numpy as np

from ernie_bench import reference as analytic
from ernie_bench.backends.base import hits, srgb_from_linear
from ernie_bench.datasets import build_scene
from ernie_bench.media import Captured, captures
from ernie_bench.config import SceneConfig

GAUSSIAN = np.exp(-0.5 * (np.arange(-5, 6) / 1.5) ** 2)
GAUSSIAN /= GAUSSIAN.sum()


def blur(image: np.ndarray) -> np.ndarray:
    padded = np.pad(image, 5, mode="reflect")
    rows = sum(weight * padded[:, offset : offset + image.shape[1]] for offset, weight in enumerate(GAUSSIAN))
    return sum(weight * rows[offset : offset + image.shape[0]] for offset, weight in enumerate(GAUSSIAN))


def ssim(test: np.ndarray, expected: np.ndarray) -> float:
    luminance = np.array([0.2126, 0.7152, 0.0722])
    x, y = srgb_from_linear(test) @ luminance, srgb_from_linear(expected) @ luminance
    mean_x, mean_y = blur(x), blur(y)
    variance_x = blur(x * x) - mean_x**2
    variance_y = blur(y * y) - mean_y**2
    covariance = blur(x * y) - mean_x * mean_y
    c1, c2 = 0.01**2, 0.03**2
    index = ((2 * mean_x * mean_y + c1) * (2 * covariance + c2)) / ((mean_x**2 + mean_y**2 + c1) * (variance_x + variance_y + c2))
    return float(index.mean())


def flip(test: np.ndarray, expected: np.ndarray) -> float | None:
    try:
        import flip_evaluator
    except ImportError:
        return None
    _, mean, _ = flip_evaluator.evaluate(
        np.ascontiguousarray(srgb_from_linear(expected), np.float32), np.ascontiguousarray(srgb_from_linear(test), np.float32), "LDR"
    )
    return float(mean)


@cache
def lpips_network():
    try:
        import lpips
    except ImportError:
        return None
    return lpips.LPIPS(net="alex", verbose=False).eval()


def perceptual(test: np.ndarray, expected: np.ndarray) -> float | None:
    network = lpips_network()
    if network is None:
        return None
    import torch

    images = [torch.from_numpy(np.ascontiguousarray(srgb_from_linear(image).transpose(2, 0, 1)[None] * 2 - 1, np.float32)) for image in (test, expected)]
    with torch.no_grad():
        return float(network(*images).mean())


def color_metrics(test: np.ndarray, expected: np.ndarray) -> dict[str, float | None]:
    error = test.astype(np.float64) - expected
    mse = float(np.mean(error**2))
    return {
        "psnr": float(10 * np.log10(1 / mse)) if mse > 0 else float("inf"),
        "ssim": ssim(test, expected),
        "flip": flip(test, expected),
        "lpips": perceptual(test, expected),
        "mean_absolute_error": float(np.mean(np.abs(error))),
    }


def depth_metrics(test: np.ndarray, expected: np.ndarray) -> dict[str, float]:
    test, expected = test[..., 0].astype(np.float64), expected[..., 0].astype(np.float64)
    test_hit, expected_hit = hits(test), hits(expected)
    both = test_hit & expected_hit
    return {
        "hit_agreement": float(np.mean(test_hit == expected_hit)),
        "absolute_relative": float(np.mean(np.abs(test[both] - expected[both]) / expected[both])) if both.any() else float("inf"),
        "rmse": float(np.sqrt(np.mean((test[both] - expected[both]) ** 2))) if both.any() else float("inf"),
    }


def segmentation_metrics(test: np.ndarray, expected: np.ndarray) -> dict[str, float]:
    test, expected = test[..., 0], expected[..., 0]
    unions = []
    for label in np.unique(expected[expected > 0]):
        intersection = np.sum((test == label) & (expected == label))
        union = np.sum((test == label) | (expected == label))
        unions.append(intersection / union)
    return {"mean_iou": float(np.mean(unions)) if unions else float("nan")}


METRICS = {"rgb": color_metrics, "depth": depth_metrics, "segmentation": segmentation_metrics}


def geometry_only(rgb: np.ndarray, depth: np.ndarray) -> np.ndarray:
    return np.where(hits(depth)[..., :1], rgb, 0.0)


def analytic_frames(manifest: dict, member: Captured, image: int) -> dict[str, np.ndarray]:
    specs = {spec["name"]: spec for spec in manifest["suite"]["scenes"]}
    workload = member.workload
    scene = build_scene(SceneConfig(**specs[member.scene]), workload["worlds"], workload["cameras_per_world"])
    tier = workload["tier"] if "rgb" in member.outputs else None
    rendered = [analytic.render(scene, scene.state(step * workload["dt"]), workload["width"], workload["height"], [image], tier) for step in member.steps]
    depth, classes, rgb = zip(*rendered)
    frames = {"depth": np.concatenate(depth)[..., None], "segmentation": np.concatenate(classes)[..., None]}
    if rgb[0] is not None:
        frames["rgb"] = np.concatenate(rgb)
    return frames


def evaluate(runs: list[Path], reference: str = "analytic") -> list[dict]:
    members = defaultdict(list)
    manifests = {}
    for run in runs:
        manifests[Path(run)] = json.loads((Path(run) / "manifest.json").read_text())
        for captured in captures(run):
            members[captured.point].append((Path(run), captured))
    rows = []
    for point, found in members.items():
        expected_member = next((captured for _, captured in reversed(found) if captured.backend == reference), None)
        for run, member in found:
            if member.backend == reference:
                continue
            for image in member.images:
                analytic_images = analytic_frames(manifests[run], member, image) if expected_member is None and reference == "analytic" else {}
                for output in member.outputs:
                    if expected_member is not None:
                        if output not in expected_member.outputs or image not in expected_member.images:
                            continue
                        expected = expected_member.sequence(output, image)
                        silhouette = expected_member.sequence("depth", image) if "depth" in expected_member.outputs else None
                    elif reference == "analytic":
                        expected = analytic_images.get(output)
                        silhouette = analytic_images["depth"]
                    else:
                        expected = silhouette = None
                    if expected is None:
                        continue
                    test = member.sequence(output, image)
                    if output == "rgb" and silhouette is not None:
                        test, expected = geometry_only(test, silhouette), geometry_only(expected, silhouette)
                    for step, rendered, truth in zip(member.steps, test, expected):
                        rows.append(
                            {"backend": member.backend, "scene": point[0], "worlds": point[1], "width": point[3], "height": point[4],
                             "tier": point[6], "spp": point[7], "output": output, "image": image, "step": step, "reference": reference}
                            | METRICS[output](rendered, truth)
                        )
    return rows


GROUP = ("backend", "scene", "worlds", "width", "height", "tier", "spp", "output")


def summarize(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[name] for name in GROUP)].append(row)
    summary = []
    for key, members in sorted(groups.items()):
        names = [name for name in members[0] if name not in (*GROUP, "image", "step", "reference")]
        values = {name: [m[name] for m in members if m[name] is not None and np.isfinite(m[name])] for name in names}
        summary.append(
            dict(zip(GROUP, key))
            | {"samples": len(members)}
            | {name: float(np.mean(found)) if found else None for name, found in values.items()}
        )
    return summary
