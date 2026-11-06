# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import numpy as np

from ernie_bench.backends.base import hits
from ernie_bench.scene import Scene, State

CHUNK = 1 << 20
SHADED = ("unlit", "direct")


def camera_rays(camera: np.ndarray, fov_y: float, width: int, height: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    half = np.tan(np.radians(fov_y) / 2)
    columns = (np.arange(width) + 0.5) / width * 2 - 1
    rows = 1 - (np.arange(height) + 0.5) / height * 2
    x, y = np.meshgrid(columns * half * width / height, rows * half)
    local = np.stack([x, y, -np.ones_like(x)], axis=-1).reshape(-1, 3)
    local /= np.linalg.norm(local, axis=1, keepdims=True)
    directions = local @ camera[:3, :3].T.astype(np.float64)
    return np.broadcast_to(camera[:3, 3].astype(np.float64), directions.shape), directions, -local[:, 2]


def world_triangles(scene: Scene, transforms: np.ndarray, world: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    triangles, classes, albedo = [], [], []
    for index in scene.members[world]:
        mesh = scene.meshes[scene.instances.mesh[index]]
        corners = mesh.positions[mesh.indices].astype(np.float64)
        matrix = transforms[index].astype(np.float64)
        triangles.append(corners @ matrix[:3, :3].T + matrix[:3, 3])
        classes.append(np.full(len(mesh.indices), scene.instances.semantic[index] + 1, np.int32))
        albedo.append(np.tile(scene.colors[index], (len(mesh.indices), 1)))
    return np.concatenate(triangles), np.concatenate(classes), np.concatenate(albedo)


def intersect(origins: np.ndarray, directions: np.ndarray, triangles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    nearest = np.full(len(origins), np.inf)
    hit = np.full(len(origins), -1, np.int64)
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    edge1, edge2 = b - a, c - a
    step = max(1, CHUNK // max(1, len(triangles)))
    for start in range(0, len(origins), step):
        o = origins[start : start + step, None, :]
        d = directions[start : start + step, None, :]
        p = np.cross(d, edge2)
        determinant = np.einsum("rtk,tk->rt", p, edge1)
        valid = np.abs(determinant) > 1e-12
        inverse = np.where(valid, 1 / np.where(valid, determinant, 1), 0)
        offset = o - a
        u = np.einsum("rtk,rtk->rt", offset, p) * inverse
        q = np.cross(offset, edge1)
        v = np.einsum("rk,rtk->rt", d[:, 0], q) * inverse
        t = np.einsum("tk,rtk->rt", edge2, q) * inverse
        valid &= (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 0)
        t = np.where(valid, t, np.inf)
        closest = np.argmin(t, axis=1)
        nearest[start : start + step] = t[np.arange(len(t)), closest]
        hit[start : start + step] = np.where(np.isfinite(nearest[start : start + step]), closest, -1)
    return nearest, hit


def shade(scene: Scene, triangles: np.ndarray, albedo: np.ndarray, hit: np.ndarray, points: np.ndarray, directions: np.ndarray, tier: str) -> np.ndarray:
    color = albedo[hit]
    if tier == "unlit" or not scene.lights:
        return color
    a, b, c = triangles[hit, 0], triangles[hit, 1], triangles[hit, 2]
    normals = np.cross(b - a, c - a)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    normals *= np.where(np.einsum("ij,ij->i", normals, directions) > 0, -1.0, 1.0)[:, None]
    light = scene.lights[0]
    towards = -np.asarray(light.direction, np.float64) / np.linalg.norm(light.direction)
    cosine = np.maximum(normals @ towards, 0)
    origins = points + normals * 1e-4 * np.maximum(1, np.linalg.norm(points, axis=1, keepdims=True))
    blocked, _ = intersect(origins, np.broadcast_to(towards, origins.shape), triangles)
    lit = cosine * np.isinf(blocked) * light.irradiance / np.pi
    return color * (scene.ambient + lit)[:, None] * np.asarray(light.color)


def render(scene: Scene, state: State, width: int, height: int, images: list[int], tier: str | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    transforms = scene.transforms(state)
    depth = np.full((len(images), height, width), np.inf, np.float32)
    classes = np.zeros((len(images), height, width), np.int32)
    rgb = np.zeros((len(images), height, width, 3), np.float32) if tier in SHADED else None
    for slot, image in enumerate(images):
        triangles, labels, albedo = world_triangles(scene, transforms, scene.cameras.world[image])
        origins, directions, forward = camera_rays(state.camera[image], scene.cameras.fov_y[image], width, height)
        distance, hit = intersect(origins, directions, triangles)
        planar = distance * forward
        visible = np.isfinite(distance) & (planar >= scene.cameras.near[image]) & (planar <= scene.cameras.far[image])
        depth[slot] = np.where(visible, distance, np.inf).reshape(height, width)
        classes[slot] = np.where(visible, labels[np.maximum(hit, 0)], 0).reshape(height, width)
        if rgb is not None and visible.any():
            pixels = np.zeros((width * height, 3))
            points = origins[visible] + directions[visible] * distance[visible, None]
            pixels[visible] = shade(scene, triangles, albedo, hit[visible], points, directions[visible], tier)
            rgb[slot] = pixels.reshape(height, width, 3)
    return depth, classes, rgb


def agreement(
    reference_depth: np.ndarray,
    reference_classes: np.ndarray,
    frames: dict[str, np.ndarray],
    tolerance: float = 0.01,
    reference_rgb: np.ndarray | None = None,
    color_tolerance: float = 0.05,
) -> dict[str, float]:
    measured = {}
    if "rgb" in frames and reference_rgb is not None:
        lit = np.isfinite(reference_depth)
        difference = np.abs(frames["rgb"][lit].astype(np.float64) - reference_rgb[lit])
        measured["rgb_agreement"] = float(np.mean(difference.max(axis=-1) <= color_tolerance)) if lit.any() else 0.0
        measured["rgb_median_error"] = float(np.median(difference)) if lit.any() else float("inf")
    if "depth" in frames:
        depth = frames["depth"][..., 0].astype(np.float64)
        rendered_hit = hits(depth)
        reference_hit = np.isfinite(reference_depth)
        both = rendered_hit & reference_hit
        measured["hit_agreement"] = float(np.mean(rendered_hit == reference_hit))
        close = np.abs(depth[both] - reference_depth[both]) <= tolerance * reference_depth[both]
        measured["depth_agreement"] = float(np.mean(close)) if both.any() else 0.0
        measured["depth_median_error"] = float(np.median(np.abs(depth[both] - reference_depth[both]) / reference_depth[both])) if both.any() else float("inf")
    if "segmentation" in frames:
        measured["segmentation_agreement"] = float(np.mean(frames["segmentation"][..., 0] == reference_classes))
    return measured
