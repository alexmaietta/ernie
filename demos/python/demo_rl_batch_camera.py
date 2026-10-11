# SPDX-License-Identifier: BSD-3-Clause
"""Benchmarks batched cameras for reinforcement learning: one environment ("world") per view,
each a clone of the first, with camera poses that change every step on the GPU, and reports the
time per step and views rendered per second.

    PYTHONPATH=build/python python demos/python/demo_rl_batch_camera.py [--views 1024]
        [--bands 3] [--minimal texturedDiffuse] [--viewer null] [--sweep] [--output views.png]

Each environment holds a few boxes and spheres over a shared ground under a sun with shadows. The
views orbit their environment, a step's poses arrive as a CUDA array, and the observations stay
on the GPU (`BatchCamera.step`), the way a training loop consumes them with torch.from_dlpack.
Spectral bands (`tokens.SPECTRAL_BANDS`) set most of the shading and sensor cost; RGB
observations need few. `--minimal` draws with minimal shading (`tokens.MINIMAL_MODE`) instead:
one forward pass straight into the observations, with no spectral shading or sensor filters;
"none" renders black colour and distances alone.

Each frame is one batch step, timed after a few steps' warm-up; `--sweep` first times 256, 1024
and 4096 views at 3 and 16 bands, each in a scene of its own. The environments are clones at the
origin, so the viewer shows them as one. `--output` writes 16 views of the last step, spread round
the orbit, as a contact sheet.
"""

import time

import numpy as np

import ernie
import ernie.demo
from ernie import tokens
from ernie.batch import BatchCamera

SIZE = 64
STEPS = 60  # distinct poses, cycled
WARMUP = 5  # steps before timing: uploads, pipelines, allocations


def build_scene(scene, worlds, bands):
    scene.define("/World/Ground", ernie.GroundPlane)
    scene.define("/World/Sun", ernie.DistantLight).set(
        tokens.TRANSFORM, ernie.distant_light_transform((0.4, -0.6, 0.7))
    )
    env0 = scene.define("/Envs/Env0", ernie.Xform)
    rng = np.random.default_rng(7)
    for i in range(6):
        shape = ernie.Cube if i % 2 == 0 else ernie.Sphere
        env0.define(f"Thing{i}", shape, display_color=tuple(rng.uniform(0.2, 1.0, 3)),
                    translate=(*rng.uniform(-3.0, 3.0, 2), 0.5))
    for world in range(worlds):
        env = env0 if world == 0 else ernie.Xform(scene.clone("/Envs/Env0", f"/Envs/Env{world}"))
        env.world = world
    scene.define("/World/Camera", ernie.Camera, focal_length=18.0, spectral_bands=bands)
    return scene


def orbit(views, angle):
    """One pose per view, circling its environment from its own heading."""
    poses = np.empty((views, 4, 4), np.float32)
    for v in range(views):
        heading = angle + 2.0 * np.pi * v / views
        eye = (8.0 * np.cos(heading), 8.0 * np.sin(heading), 5.0)
        poses[v] = ernie.look_at_transform(eye, (0.0, 0.0, 0.5), up=(0.0, 0.0, 1.0))
    return poses


MINIMAL_MODES = ("off", "none", "constantDiffuse", "texturedDiffuse")


def batch_camera(renderer, scene, views, minimal):
    """The batch camera, with minimal shading unless `minimal` is "off"; minimal "none" renders
    distances besides its black colour."""
    render_vars = ("LdrColor", "DistanceToCamera") if minimal == "none" else ("LdrColor",)
    return BatchCamera(renderer, scene, "/World/Camera", views, resolution=(SIZE, SIZE),
                       render_vars=render_vars, minimal=None if minimal == "off" else minimal)


def benchmark(renderer, views, bands, minimal="off"):
    """Seconds per step of `views` views at `bands` bands, in a scene of their own."""
    scene = build_scene(ernie.Scene(), views, bands)
    camera = batch_camera(renderer, scene, views, minimal)
    camera.set_worlds(np.arange(views, dtype=np.int32))
    poses = [renderer.cuda_upload(orbit(views, 0.03 * s)) for s in range(STEPS)]
    for p in poses[:WARMUP]:
        camera.step(p)
    start = time.perf_counter()
    observations = None
    for p in poses:
        observations = camera.step(p)
    renderer.cuda_download(observations["LdrColor"])  # the last step has finished
    return (time.perf_counter() - start) / STEPS


def contact_sheet(images, columns=4):
    rows = (len(images) + columns - 1) // columns
    sheet = np.zeros((rows * SIZE, columns * SIZE, 4), np.uint8)
    for i, image in enumerate(images):
        r, c = divmod(i, columns)
        sheet[r * SIZE : (r + 1) * SIZE, c * SIZE : (c + 1) * SIZE] = image
    return sheet


class Demo(ernie.demo.Demo):
    num_frames = WARMUP + STEPS

    @classmethod
    def create_parser(cls):
        parser = super().create_parser()
        parser.add_argument("--views", type=int, default=1024, help="environments, one view each")
        parser.add_argument("--bands", type=int, default=3, help="spectral bands per view")
        parser.add_argument("--minimal", choices=MINIMAL_MODES, default="off",
                            help="minimal shading (ernie:minimal:mode) for throughput")
        parser.add_argument("--sweep", action="store_true",
                            help="first time 256, 1024 and 4096 views at 3 and 16 bands, then "
                                 "with each minimal shading mode")
        return parser

    def __init__(self, viewer, args):
        super().__init__(viewer, args)
        renderer = self.renderer
        if not renderer.cuda_available:
            raise RuntimeError("This demo needs CUDA interop")
        print(f"{renderer.device_name}, {SIZE} x {SIZE} LdrColor observations")
        if args.sweep:
            for bands in (3, 16):
                for views in (256, 1024, 4096):
                    self.print_rate(views, bands, benchmark(renderer, views, bands))
            # Minimal shading has no spectral bands to speak of: one row per mode.
            for minimal in MINIMAL_MODES[1:]:
                for views in (256, 1024, 4096):
                    self.print_rate(views, 3, benchmark(renderer, views, 3, minimal),
                                    f", minimal {minimal}")
        self.views, self.bands, self.minimal = args.views, args.bands, args.minimal
        build_scene(self.scene, self.views, self.bands)
        self.camera = batch_camera(renderer, self.scene, self.views, self.minimal)
        self.camera.set_worlds(np.arange(self.views, dtype=np.int32))
        self.poses = [renderer.cuda_upload(orbit(self.views, 0.03 * s)) for s in range(STEPS)]
        self.observations = None
        self.timer_start = None
        self.timed_steps = 0

    def step(self):
        if self.frame == WARMUP:
            self.images()  # the warm-up has finished
            self.timer_start = time.perf_counter()
            self.timed_steps = 0
        self.observations = self.camera.step(self.poses[self.frame % STEPS])
        if self.timer_start is not None:
            self.timed_steps += 1

    def render(self):
        # Without a viewer the batch step is the render.
        if self.viewer is not None:
            self.viewer.step()

    def images(self):
        """The last step's views, uint8 (views, SIZE, SIZE, 4), waiting for the render."""
        return np.asarray(self.renderer.cuda_download(self.observations["LdrColor"]))

    def print_rate(self, views, bands, per_step, note=""):
        print(f"{views:6d} views, {bands:2d} bands: {per_step * 1e3:7.2f} ms per step, "
              f"{views / per_step:10,.0f} views per second{note}")

    def report(self):
        """Prints the time per step since the warm-up, once the last step has finished; before
        the output and the test, so that their downloads are not counted."""
        if self.observations is None or not self.timed_steps:
            return
        self.images()  # the last step has finished
        per_step = (time.perf_counter() - self.timer_start) / self.timed_steps
        note = "" if self.minimal == "off" else f", minimal {self.minimal}"
        note += "" if self.viewer is None else " (with the viewer drawing)"
        self.print_rate(self.views, self.bands, per_step, note)
        self.timed_steps = 0

    def save_output(self, path):
        self.report()
        # 16 views spread round the orbit; neighbouring views differ by only a few degrees.
        ernie.demo.write_png(path, contact_sheet(self.images()[:: max(self.views // 16, 1)][:16]))

    def close(self):
        self.report()

    def test_final(self):
        self.report()
        images = self.images()[..., :3]
        assert images.shape == (self.views, SIZE, SIZE, 3)
        if self.minimal == "none":
            # Black by design: the distances show the scene.
            assert (images == 0).all(), "minimal none renders no colour"
            images = np.asarray(
                self.renderer.cuda_download(self.observations["DistanceToCamera"]))
            # Tenths of a metre, as the colours' scale for the checks below.
            images = np.where(np.isfinite(images), np.minimum(10.0 * images, 255.0), 0.0)
            images = np.repeat(images[..., None], 3, axis=-1)
        assert images.std() > 1.0, "the views should not render flat"
        # Views from opposite headings see their environment from opposite sides.
        if self.views > 1:
            opposite = self.views // 2
            assert np.abs(images[0].astype(int) - images[opposite]).max() > 16, \
                "views from opposite headings should differ"


if __name__ == "__main__":
    ernie.demo.main(Demo)
