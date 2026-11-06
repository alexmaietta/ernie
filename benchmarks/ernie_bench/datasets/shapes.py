# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import numpy as np

from ernie_bench.scene import Mesh


def _convex(faces: list[np.ndarray]) -> Mesh:
    center = np.concatenate(faces).mean(axis=0)
    positions, normals, indices = [], [], []
    for face in faces:
        face = np.asarray(face, np.float32)
        normal = np.cross(face[1] - face[0], face[2] - face[0])
        if np.dot(normal, face.mean(axis=0) - center) < 0:
            face, normal = face[::-1], -normal
        base = len(positions)
        positions.extend(face)
        normals.extend([normal / np.linalg.norm(normal)] * len(face))
        indices.extend((base, base + i, base + i + 1) for i in range(1, len(face) - 1))
    return Mesh(
        np.array(positions, np.float32),
        np.array(normals, np.float32),
        np.array(indices, np.uint32),
    )


def plane() -> Mesh:
    return Mesh(
        np.array([[-0.5, -0.5, 0], [0.5, -0.5, 0], [0.5, 0.5, 0], [-0.5, 0.5, 0]], np.float32),
        np.tile(np.array([0, 0, 1], np.float32), (4, 1)),
        np.array([[0, 1, 2], [0, 2, 3]], np.uint32),
    )


def box() -> Mesh:
    c = np.array([[x, y, z] for z in (0, 1) for y in (-0.5, 0.5) for x in (-0.5, 0.5)])
    quads = [[0, 1, 3, 2], [4, 5, 7, 6], [0, 1, 5, 4], [2, 3, 7, 6], [0, 2, 6, 4], [1, 3, 7, 5]]
    return _convex([c[quad] for quad in quads])


def ramp() -> Mesh:
    b0, b1, b2, b3 = [-0.5, -0.5, 0], [0.5, -0.5, 0], [0.5, 0.5, 0], [-0.5, 0.5, 0]
    t1, t2 = [0.5, -0.5, 1], [0.5, 0.5, 1]
    faces = [[b0, b1, b2, b3], [b1, b2, t2, t1], [b0, t1, t2, b3], [b0, b1, t1], [b3, b2, t2]]
    return _convex([np.array(face) for face in faces])


def cylinder(segments: int = 16) -> Mesh:
    angle = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    ring = np.column_stack([0.5 * np.cos(angle), 0.5 * np.sin(angle)])
    bottom = np.column_stack([ring, np.zeros(segments)])
    top = np.column_stack([ring, np.ones(segments)])
    sides = [
        np.array([bottom[i], bottom[(i + 1) % segments], top[(i + 1) % segments], top[i]])
        for i in range(segments)
    ]
    return _convex([bottom, top, *sides])
