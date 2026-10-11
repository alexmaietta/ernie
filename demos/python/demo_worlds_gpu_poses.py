# SPDX-License-Identifier: BSD-3-Clause
"""Steps a batch of cameras whose poses live in CUDA memory, as a reinforcement learning loop
whose simulator keeps its state on the GPU, and times a step with the poses kept on the GPU
against downloading them first.

    PYTHONPATH=build/python python demos/python/demo_worlds_gpu_poses.py [--views 256]
        [--steps 50] [--viewer null --output poses.png]

Poses written as a contiguous float32 CUDA array (any DLPack array: PyTorch, JAX, Warp, CuPy;
here `Renderer.cuda_upload` stands in for the simulator's) are copied device to device and each
view's matrices are computed on the GPU, so a step never waits for the host to read them: the CPU
prepares the frame while the simulator's queued GPU work still runs, which downloading the poses
would have to wait for. Stepping with `step_async` and mapping the outputs to CUDA without
waiting lets the CPU prepare the next step while the GPU renders this one.
Setting the camera's `poses_device` (`ernie:poses:device`) to "host" downloads them instead,
which lets CPU stages see exact poses (see `tokens.POSES_DEVICE`).

The demo runs five rounds of `--steps` frames: a warm-up, then poses on the GPU and downloaded,
without and then after queued physics, reporting each pair of rounds (without a window the demo
runs them once: five times `--steps` frames). `--output` writes the
first 16 views of the last step as a contact sheet.
"""

import time

import numpy as np

import ernie
import ernie.demo
from ernie import tokens

SIZE = 64

#: The rounds, each `--steps` frames: (where the poses are read, whether physics is queued).
ROUNDS = ((tokens.GPU, False), (tokens.GPU, False), (tokens.HOST, False), (tokens.GPU, True),
          (tokens.HOST, True))


def build_scene(scene, views):
    world = scene.define("/World", ernie.Xform)
    world.define("Ground", ernie.GroundPlane)
    world.define("Sun", ernie.DistantLight).set(
        tokens.TRANSFORM, ernie.distant_light_transform((0.4, -0.6, 0.7))
    )
    world.define("Box", ernie.Cube, size=2.0, translate=(0.0, 0.0, 1.0))
    camera = world.define("Camera", ernie.Camera, focal_length=18.0, poses=orbit(views, 0.0))
    scene.define("/Render/Batch", ernie.RenderProduct, camera=camera, resolution=(SIZE, SIZE))
    # A camera of one view, for the viewer and for --viewer null.
    overview = world.define("Overview", ernie.Camera, focal_length=24.0).set(
        tokens.TRANSFORM, ernie.look_at_transform((14.0, -10.0, 8.0), (0.0, 0.0, 1.0)))
    scene.define("/Render/Settings", ernie.RenderSettings, camera=overview)
    return camera, overview


def orbit(views, angle):
    """One camera per view around the box, each at its own heading, turned by `angle`."""
    poses = []
    for v in range(views):
        heading = angle + 2.0 * np.pi * v / views
        eye = (10.0 * np.cos(heading), 10.0 * np.sin(heading), 6.0)
        poses.append(ernie.look_at_transform(eye, (0.0, 0.0, 1.0), up=(0.0, 0.0, 1.0)))
    return np.asarray(poses, np.float32)


def contact_sheet(images, columns=4):
    rows = (len(images) + columns - 1) // columns
    sheet = np.zeros((rows * SIZE, columns * SIZE, 4), np.uint8)
    for i, image in enumerate(images):
        r, c = divmod(i, columns)
        sheet[r * SIZE : (r + 1) * SIZE, c * SIZE : (c + 1) * SIZE] = image
    return sheet


class Demo(ernie.demo.Demo):

    @classmethod
    def create_parser(cls):
        parser = super().create_parser()
        parser.add_argument("--views", type=int, default=256, help="cameras in the batch")
        parser.add_argument("--steps", type=int, default=50, help="steps in each timed round")
        return parser

    def __init__(self, viewer, args):
        super().__init__(viewer, args)
        renderer = self.renderer
        if not renderer.cuda_available:
            raise RuntimeError("This demo needs CUDA interop")
        self.views, self.steps = args.views, args.steps
        self.num_frames = len(ROUNDS) * self.steps
        self.camera, overview = build_scene(self.scene, self.views)
        # The simulator's poses for every step, already on the GPU, and a stand-in for its
        # physics: a large device-to-device copy queued on the same stream before each step's
        # poses.
        self.on_gpu = [renderer.cuda_upload(orbit(self.views, 0.05 * s))
                       for s in range(self.steps)]
        self.state = [renderer.cuda_upload(np.zeros(64 << 20, np.float32)) for _ in range(2)]
        self.colors = None
        self.start = None
        self.times = {}  # ms per step by (poses device, physics)
        if viewer is not None:
            viewer.active_camera = overview.path

    def step(self):
        index = self.frame % self.steps
        device, physics = ROUNDS[(self.frame // self.steps) % len(ROUNDS)]
        if index == 0:
            self.camera.poses_device = device
            self.start = time.perf_counter()
        if physics:
            self.renderer.cuda_copy(self.state[0], self.state[1])
        self.camera.write(tokens.POSES, self.on_gpu[index])
        # Submit and take the outputs without waiting: the CUDA mapping orders on the GPU, and
        # the next step is prepared while this one renders.
        step = self.renderer.step_async(self.scene, render_products=["/Render/Batch"])
        product = step.outputs()["/Render/Batch"]
        self.colors = product.frames[0].render_vars["/Render/Batch/LdrColor"].map(
            device=ernie.Device.CUDA)
        if index == self.steps - 1:
            self.finish_round(device, physics)

    def finish_round(self, device, physics):
        warm_up = self.frame // self.steps % len(ROUNDS) == 0
        if warm_up:
            return
        self.times[device, physics] = (time.perf_counter() - self.start) / self.steps * 1e3
        if device == tokens.HOST:
            gpu, host = self.times[tokens.GPU, physics], self.times[tokens.HOST, physics]
            print(
                f"{self.views} views{', after queued physics' if physics else ''}: "
                f"{gpu:.2f} ms per step with poses on the GPU, {host:.2f} ms downloading them"
            )

    def render(self):
        # Without a viewer the batch step is the render; the overview renders only on demand.
        if self.viewer is not None:
            self.viewer.step()

    def images(self):
        """The last step's views, uint8 (views, SIZE, SIZE, 4), waiting for the render."""
        return np.asarray(self.renderer.cuda_download(self.colors))

    def save_output(self, path):
        ernie.demo.write_png(path, contact_sheet(self.images()[:16]))

    def test_final(self):
        images = self.images()[..., :3]
        assert images.shape == (self.views, SIZE, SIZE, 3)
        assert images.std() > 1.0, "the views should not render flat"
        if self.views > 1:
            # Views from opposite sides of the box see it differently.
            opposite = self.views // 2
            assert np.abs(images[0].astype(int) - images[opposite]).max() > 16, \
                f"views 0 and {opposite} should differ"
        if self.frame >= len(ROUNDS) * self.steps:
            assert len(self.times) == len(ROUNDS) - 1, "every round should be timed"
            assert all(ms > 0.0 for ms in self.times.values())
        overview = self.render_image()[..., :3]
        assert overview.std() > 1.0, "the overview should not render flat"


if __name__ == "__main__":
    ernie.demo.main(Demo)
