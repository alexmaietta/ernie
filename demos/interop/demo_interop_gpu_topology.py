# SPDX-License-Identifier: BSD-3-Clause
"""Metaballs polygonised by Warp every frame straight into the
renderer's memory: their points and triangles change number as the
blobs merge and split, within a capacity, with no copy at all."""

from typing import no_type_check

import numpy as np
import warp as wp

import ernie
import ernie.demo
from ernie import tokens

BLOBS = ernie.Path("/World/Blobs")
CELLS = wp.constant(48)
LOW = wp.constant(wp.vec3(-1.2, -1.2, 0.0))
SPACING = wp.constant(2.4 / 48)
MAX_POINTS = wp.constant(1 << 16)
MAX_TRIANGLES = wp.constant(1 << 17)
ISO = wp.constant(1.0)

# Each cell as six tetrahedra along its diagonal, one per order of the
# axes (Kuhn's): every edge joins a grid point to one above it.
ORDERS = [
    (0, 1, 2),
    (0, 2, 1),
    (1, 0, 2),
    (1, 2, 0),
    (2, 0, 1),
    (2, 1, 0),
]


@wp.func
@no_type_check
def field(p: wp.vec3, blobs: wp.array[wp.vec4], count: int):
    """Sum of each blob's r^2 / d^2; the surface is where it is ISO."""
    f = wp.float32(0.0)
    for i in range(count):
        b = blobs[i]
        d = p - wp.vec3(b[0], b[1], b[2])
        f += b[3] * b[3] / (wp.dot(d, d) + 1.0e-6)
    return f


@wp.func
@no_type_check
def outward(p: wp.vec3, blobs: wp.array[wp.vec4], count: int):
    """The surface's outward normal at `p`: down the field's slope."""
    g = wp.vec3(0.0)
    for i in range(count):
        b = blobs[i]
        d = p - wp.vec3(b[0], b[1], b[2])
        q = wp.dot(d, d) + 1.0e-6
        g += 2.0 * b[3] * b[3] / (q * q) * d
    return wp.normalize(g)


@wp.func
@no_type_check
def grid_index(p: wp.vec3i):
    n = CELLS + 1
    return (p[0] * n + p[1]) * n + p[2]


@wp.func
@no_type_check
def grid_point(p: wp.vec3i):
    return LOW + SPACING * wp.vec3(
        float(p[0]), float(p[1]), float(p[2])
    )


@wp.kernel
@no_type_check
def sample(
    values: wp.array[float], blobs: wp.array[wp.vec4], count: int
):
    """The field at every grid point."""
    tid = wp.tid()
    n = CELLS + 1
    p = wp.vec3i(tid // (n * n), (tid // n) % n, tid % n)
    values[tid] = field(grid_point(p), blobs, count) - ISO


@wp.kernel
@no_type_check
def cut_edges(
    values: wp.array[float],
    blobs: wp.array[wp.vec4],
    count: int,
    edges: wp.array[int],
    points: wp.array[wp.vec3],
    normals: wp.array[wp.vec3],
    counts: wp.array[wp.int32],
):
    """A point where the surface cuts each grid edge, shared by every
    tetrahedron around it."""
    tid = wp.tid()
    n = CELLS + 1
    g = tid // 7
    d = (tid % 7) + 1
    a = wp.vec3i(g // (n * n), (g // n) % n, g % n)
    b = a + wp.vec3i(d & 1, (d >> 1) & 1, (d >> 2) & 1)
    edges[tid] = -1
    if b[0] > CELLS or b[1] > CELLS or b[2] > CELLS:
        return
    fa = values[g]
    fb = values[grid_index(b)]
    if (fa > 0.0) == (fb > 0.0):
        return
    v = wp.atomic_add(counts, 0, 1)
    if v >= MAX_POINTS:
        return
    p = wp.lerp(grid_point(a), grid_point(b), fa / (fa - fb))
    points[v] = p
    normals[v] = outward(p, blobs, count)
    edges[tid] = v


@wp.func
@no_type_check
def edge(edges: wp.array[int], p: wp.vec3i, q: wp.vec3i):
    """The point on the edge from `p` to `q`, either way round."""
    a = p
    b = q
    if p[0] + p[1] + p[2] > q[0] + q[1] + q[2]:
        a = q
        b = p
    d = b - a
    return edges[grid_index(a) * 7 + d[0] + 2 * d[1] + 4 * d[2] - 1]


@wp.func
@no_type_check
def emit(
    i: int,
    j: int,
    k: int,
    points: wp.array[wp.vec3],
    normals: wp.array[wp.vec3],
    indices: wp.array[wp.int32],
    counts: wp.array[wp.int32],
):
    """A triangle, turned to face out."""
    if i < 0 or j < 0 or k < 0:
        return
    t = wp.atomic_add(counts, 1, 1)
    if t >= MAX_TRIANGLES:
        return
    face = wp.cross(points[j] - points[i], points[k] - points[i])
    second = j
    third = k
    if wp.dot(face, normals[i] + normals[j] + normals[k]) < 0.0:
        second = k
        third = j
    indices[3 * t] = i
    indices[3 * t + 1] = second
    indices[3 * t + 2] = third


@wp.kernel
@no_type_check
def triangulate(
    values: wp.array[float],
    edges: wp.array[int],
    orders: wp.array[wp.vec3i],
    points: wp.array[wp.vec3],
    normals: wp.array[wp.vec3],
    indices: wp.array[wp.int32],
    counts: wp.array[wp.int32],
):
    """Each tetrahedron's piece of the surface: none, a triangle, or a
    quad as two (marching tetrahedra)."""
    tid = wp.tid()
    cell = tid // 6
    order = orders[tid % 6]
    c = wp.vec3i(
        cell // (CELLS * CELLS), (cell // CELLS) % CELLS, cell % CELLS
    )
    v0 = c
    v1 = v0
    v1[order[0]] = v1[order[0]] + 1
    v2 = v1
    v2[order[1]] = v2[order[1]] + 1
    v3 = c + wp.vec3i(1, 1, 1)
    mask = wp.int32(0)
    if values[grid_index(v0)] > 0.0:
        mask |= 1
    if values[grid_index(v1)] > 0.0:
        mask |= 2
    if values[grid_index(v2)] > 0.0:
        mask |= 4
    if values[grid_index(v3)] > 0.0:
        mask |= 8
    if mask == 0 or mask == 15:
        return
    # The corners inside first.
    a = v0
    b = v1
    e = v2
    f = v3
    if mask == 1 or mask == 14:
        a = v0
        b = v1
        e = v2
        f = v3
    elif mask == 2 or mask == 13:
        a = v1
        b = v0
        e = v2
        f = v3
    elif mask == 4 or mask == 11:
        a = v2
        b = v0
        e = v1
        f = v3
    elif mask == 8 or mask == 7:
        a = v3
        b = v0
        e = v1
        f = v2
    elif mask == 3 or mask == 12:
        a = v0
        b = v1
        e = v2
        f = v3
    elif mask == 5 or mask == 10:
        a = v0
        b = v2
        e = v1
        f = v3
    else:  # 6 or 9
        a = v1
        b = v2
        e = v0
        f = v3
    lone = (
        mask == 1
        or mask == 2
        or mask == 4
        or mask == 8
        or mask == 14
        or mask == 13
        or mask == 11
        or mask == 7
    )
    if lone:
        emit(
            edge(edges, a, b),
            edge(edges, a, e),
            edge(edges, a, f),
            points,
            normals,
            indices,
            counts,
        )
    else:
        ae = edge(edges, a, e)
        af = edge(edges, a, f)
        bf = edge(edges, b, f)
        be = edge(edges, b, e)
        emit(ae, af, bf, points, normals, indices, counts)
        emit(ae, bf, be, points, normals, indices, counts)


@wp.kernel
@no_type_check
def clamp_counts(counts: wp.array[wp.int32]):
    """The points and triangle indices in use, within the capacity."""
    counts[0] = wp.min(counts[0], MAX_POINTS)
    counts[1] = 3 * wp.min(counts[1], MAX_TRIANGLES)


def blob_centres(time, count):
    """Blobs drifting on Lissajous paths, meeting and parting."""
    i = np.arange(count)
    phase = time * (0.5 + 0.13 * i)
    return np.stack(
        [
            0.65 * np.sin(phase + 1.7 * i),
            0.65 * np.cos(1.3 * phase + 0.9 * i),
            0.75 + 0.4 * np.sin(0.8 * phase + 2.3 * i),
            np.full(count, 0.28),
        ],
        axis=1,
    ).astype(np.float32)


class Demo(ernie.demo.Demo):
    num_frames = 60

    def __init__(self, viewer, args):
        super().__init__(viewer, args)
        wp.init()
        if not wp.is_cuda_available():
            raise RuntimeError("This demo needs CUDA interop")
        self.device = "cuda:0"
        world = self.scene.define("/World", ernie.Xform)
        world.define("Ground", ernie.GroundPlane, checker=True)
        world.define("Sun", ernie.DistantLight)
        self.scene.define(
            "/World/Looks/Goo", ernie.Material, albedo=(0.1, 0.4, 0.8)
        )
        world.define(
            "Blobs", ernie.Mesh, material_binding="/World/Looks/Goo"
        )
        # GPU arrays at their capacity: the renderer then holds the
        # points and triangles and lends them out to write in place.
        with wp.ScopedDevice(self.device):
            self.normals = wp.zeros(MAX_POINTS, dtype=wp.vec3)
            self.counts = wp.zeros(2, dtype=wp.int32)
            self.values = wp.zeros((CELLS + 1) ** 3, dtype=float)
            self.edges = wp.zeros(7 * (CELLS + 1) ** 3, dtype=int)
            self.orders = wp.array(ORDERS, dtype=wp.vec3i)
            self.blobs = wp.zeros(8, dtype=wp.vec4)
            self.scene.write(
                BLOBS,
                tokens.POINTS,
                wp.zeros((MAX_POINTS, 3), dtype=float),
            )
            self.scene.write(
                BLOBS,
                tokens.FACE_VERTEX_INDICES,
                wp.zeros(3 * MAX_TRIANGLES, dtype=wp.int32),
            )
            self.scene.write(BLOBS, tokens.NORMALS, self.normals)
            self.scene.write(BLOBS, tokens.ACTIVE_COUNTS, self.counts)
        world.define("Camera", ernie.Camera, focal_length=30.0).set(
            tokens.TRANSFORM,
            ernie.look_at_transform(
                (3.2, -3.2, 2.4), (0.0, 0.0, 0.6)
            ),
        )
        self.scene.define(
            "/Render/Settings",
            ernie.RenderSettings,
            camera=ernie.Path("/World/Camera"),
            resolution=(960, 540),
        )
        if viewer is not None:
            viewer.active_camera = "/World/Camera"
        self.count = 5
        self.speed = 1.0
        self.history = []
        self.widgets.slider(
            "Blobs",
            1,
            8,
            self.count,
            on_change=lambda value: setattr(
                self, "count", int(value)
            ),
        )
        self.widgets.slider(
            "Speed",
            0.0,
            3.0,
            self.speed,
            on_change=lambda value: setattr(self, "speed", value),
        )

    def step(self):
        stream = wp.get_stream(self.device)
        self.blobs.assign(
            blob_centres(self.speed * self.time, self.count)
        )
        # The renderer's own points and triangles, written in place.
        points = wp.from_dlpack(
            self.map(BLOBS, tokens.POINTS, stream=stream),
            dtype=wp.vec3,
        )
        indices = wp.from_dlpack(
            self.map(BLOBS, tokens.FACE_VERTEX_INDICES, stream=stream)
        )
        self.counts.zero_()
        launch = dict(device=self.device, stream=stream)
        wp.launch(
            sample,
            dim=(CELLS + 1) ** 3,
            inputs=[self.values, self.blobs, self.count],
            **launch,
        )
        wp.launch(
            cut_edges,
            dim=7 * (CELLS + 1) ** 3,
            inputs=[
                self.values,
                self.blobs,
                self.count,
                self.edges,
                points,
                self.normals,
                self.counts,
            ],
            **launch,
        )
        wp.launch(
            triangulate,
            dim=6 * CELLS**3,
            inputs=[
                self.values,
                self.edges,
                self.orders,
                points,
                self.normals,
                indices,
                self.counts,
            ],
            **launch,
        )
        wp.launch(clamp_counts, dim=1, inputs=[self.counts], **launch)
        self.scene.unmap(BLOBS, tokens.POINTS, stream=stream)
        self.scene.unmap(
            BLOBS, tokens.FACE_VERTEX_INDICES, stream=stream
        )
        # How many of each are in use, read on the GPU.
        self.scene.write(
            BLOBS, tokens.NORMALS, self.normals, stream=stream
        )
        self.scene.write(
            BLOBS, tokens.ACTIVE_COUNTS, self.counts, stream=stream
        )
        if self.args.test:
            self.history.append(tuple(self.counts.numpy()))

    def test_final(self):
        used, indices = self.history[-1]
        triangles = indices // 3
        print(f"{used} points and {triangles} triangles in use")
        assert 0 < used < MAX_POINTS and 0 < triangles < MAX_TRIANGLES
        assert len(set(self.history)) > 1, (
            "the topology should change"
        )
        mapped = self.map(BLOBS, tokens.FACE_VERTEX_INDICES)
        held = np.asarray(self.renderer.cuda_download(mapped))
        self.scene.unmap(BLOBS, tokens.FACE_VERTEX_INDICES)
        assert held[: 3 * triangles].max() < used
        image = self.render_image().astype(np.int32)
        blue = image[..., 2] - np.maximum(
            image[..., 0], image[..., 1]
        )
        covered = (blue > 30).mean()
        print(f"the blobs cover {100 * covered:.1f}% of the image")
        assert covered > 0.03, "the blobs should be drawn"


if __name__ == "__main__":
    ernie.demo.main(Demo)
