# SPDX-License-Identifier: BSD-3-Clause
"""A flag waving on its pole, its mesh deformed every frame by a Warp kernel on the GPU.

    PYTHONPATH=build/python python demos/python/demo_gpu_deforming_flag.py
        [--viewer null --output flag.png]

Needs Warp (`uv pip install warp-lang`) and CUDA interop. A Warp kernel writes the flag's points
and vertex colours into Warp arrays each frame, and `scene.write` hands Ernie those arrays on the
kernel's stream: they are copied device to device into the renderer's own buffers after the
kernel, and nothing passes through host memory. The flag is authored with normals, so Ernie
recomputes them on the GPU from the moved points (area-weighted around each vertex), and its
`extent` bounds every pose, so Ernie need not read the points back to bound them either.
"""

import numpy as np
import warp as wp

import ernie
import ernie.demo
from ernie import tokens

COLUMNS = 160  # vertices along the flag, from the pole
ROWS = 96  # vertices up the flag
WIDTH = 4.0  # metres
HEIGHT = 2.4
FOOT = 2.0  # height of the flag's lower edge


@wp.kernel
def wave(points: wp.array(dtype=wp.vec3), colors: wp.array(dtype=wp.vec3), time: float):
    """The flag at `time`: travelling waves growing from the pole, and its stripes shaded by
    them."""
    tid = wp.tid()
    i = tid % COLUMNS
    j = tid // COLUMNS
    u = float(i) / float(COLUMNS - 1)  # 0 at the pole
    v = float(j) / float(ROWS - 1)  # 0 at the foot
    phase = 7.0 * u - 4.0 * time + 0.9 * v
    gust = 0.75 + 0.25 * wp.sin(0.7 * time)
    amplitude = 0.8 * u * gust
    y = amplitude * (wp.sin(phase) + 0.35 * wp.sin(2.3 * phase + 1.7 * v + 1.3 * time))
    # Folds pull the free edge in and let it sag a little.
    x = u * WIDTH * (1.0 - 0.05 * u * gust * wp.cos(phase) * wp.cos(phase))
    z = FOOT + v * HEIGHT - 0.12 * u * u * (1.0 - gust)
    points[tid] = wp.vec3(x, y, z)
    # Three stripes, lighter on the crests.
    stripe = wp.vec3(0.05, 0.12, 0.6)
    if v > 0.333:
        stripe = wp.vec3(0.85, 0.85, 0.8)
    if v > 0.667:
        stripe = wp.vec3(0.7, 0.05, 0.04)
    colors[tid] = stripe * (0.8 + 0.2 * wp.sin(phase))


def flag_triangles():
    """Two triangles per grid quad, counter-clockwise seen from -y."""
    i, j = np.meshgrid(np.arange(COLUMNS - 1), np.arange(ROWS - 1))
    a = (j * COLUMNS + i).ravel()
    b, c, d = a + 1, a + COLUMNS + 1, a + COLUMNS
    return np.stack([a, b, c, a, c, d], axis=1).reshape(-1).astype(np.int32)


class Demo(ernie.demo.Demo):
    num_frames = 60

    def __init__(self, viewer, args):
        super().__init__(viewer, args)
        wp.init()
        self.device = "cuda:0"
        count = COLUMNS * ROWS
        self.points = wp.zeros(count, dtype=wp.vec3, device=self.device)
        self.colors = wp.zeros(count, dtype=wp.vec3, device=self.device)
        self.stream = wp.get_stream(self.device)
        wp.launch(wave, dim=count, inputs=[self.points, self.colors, 0.0], device=self.device)
        self.first_points = self.points.numpy()

        scene = self.scene
        world = scene.define("/World", ernie.Xform)
        world.define("Ground", ernie.GroundPlane, checker=True, checker_size=1.0)
        world.define("Pole", ernie.Cylinder, radius=0.05, height=FOOT + HEIGHT + 0.2,
                     translate=(0.0, 0.0, 0.5 * (FOOT + HEIGHT + 0.2)),
                     display_color=(0.7, 0.7, 0.7))
        indices = flag_triangles()
        self.flag = world.define(
            "Flag", ernie.Mesh, face_vertex_counts=np.full(indices.size // 3, 3, np.int32),
            face_vertex_indices=indices,
            normals=np.tile(np.array([0.0, -1.0, 0.0], np.float32), (count, 1)))
        # Every pose the kernel can give, so bounds never need the points on the host.
        self.flag.set(tokens.EXTENT,
                      [-0.1, -1.2, FOOT - 0.3, WIDTH + 0.1, 1.2, FOOT + HEIGHT + 0.1])
        # The kernel's arrays, on the GPU: the points, and the vertex colours as displayColor.
        self.flag.write(tokens.POINTS, self.points).write(tokens.DISPLAY_COLOR, self.colors)
        world.define("Sun", ernie.DistantLight).set(
            tokens.TRANSFORM, ernie.distant_light_transform((-0.5, -0.8, 0.6)))
        camera = world.define("Camera", ernie.Camera, focal_length=30.0).set(
            tokens.TRANSFORM, ernie.look_at_transform((8.5, -4.5, 4.2), (1.6, 0.0, 2.9)))
        scene.define("/Render/Settings", ernie.RenderSettings, camera=camera)

        if viewer is not None:
            viewer.active_camera = camera.path
            if args.output:
                display = viewer.viewport_display
                display.lights = display.grid = display.cameras = False
                viewer.viewport_display = display

    def step(self):
        wp.launch(wave, dim=COLUMNS * ROWS, inputs=[self.points, self.colors, self.time],
                  device=self.device)
        # Ernie copies the arrays after the kernel, on its stream; nothing waits on the host.
        self.scene.write(self.flag.path, tokens.POINTS, self.points, stream=self.stream)
        self.scene.write(self.flag.path, tokens.DISPLAY_COLOR, self.colors, stream=self.stream)

    def test_final(self):
        moved = np.abs(self.points.numpy() - self.first_points).max()
        assert self.time > 0.0 and moved > 0.05, f"the flag should wave (moved {moved:.3f} m)"
        image = self.render_image()[..., :3]
        assert image.std() > 1.0, "the flag, pole and ground should not render flat"
        # The flag's stripes: blue, white and red somewhere in the frame.
        assert ((image[..., 2] > image[..., 0] + 40)).any(), "the flag's blue stripe is missing"
        assert ((image[..., 0] > image[..., 2] + 40)).any(), "the flag's red stripe is missing"


if __name__ == "__main__":
    ernie.demo.main(Demo)
