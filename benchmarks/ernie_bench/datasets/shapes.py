# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import numpy as np
import trimesh

from ernie_bench.scene import Mesh

LIFT = trimesh.transformations.translation_matrix((0.0, 0.0, 0.5))


def _faceted(mesh: trimesh.Trimesh) -> Mesh:
    return Mesh(
        mesh.triangles.reshape(-1, 3).astype(np.float32),
        np.repeat(mesh.face_normals, 3, axis=0).astype(np.float32),
        np.arange(3 * len(mesh.faces), dtype=np.uint32).reshape(-1, 3),
    )


def plane() -> Mesh:
    return _faceted(trimesh.Trimesh([[-0.5, -0.5, 0], [0.5, -0.5, 0], [0.5, 0.5, 0], [-0.5, 0.5, 0]], [[0, 1, 2], [0, 2, 3]]))


def box() -> Mesh:
    return _faceted(trimesh.creation.box(extents=(1.0, 1.0, 1.0), transform=LIFT))


def ramp() -> Mesh:
    corners = [[-0.5, -0.5, 0], [0.5, -0.5, 0], [0.5, 0.5, 0], [-0.5, 0.5, 0], [0.5, -0.5, 1], [0.5, 0.5, 1]]
    faces = [[0, 2, 1], [0, 3, 2], [1, 2, 5], [1, 5, 4], [0, 4, 5], [0, 5, 3], [0, 1, 4], [3, 5, 2]]
    return _faceted(trimesh.Trimesh(corners, faces))


def cylinder(segments: int = 16) -> Mesh:
    return _faceted(trimesh.creation.cylinder(radius=0.5, height=1.0, sections=segments, transform=LIFT))
