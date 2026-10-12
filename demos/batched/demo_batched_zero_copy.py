# SPDX-License-Identifier: BSD-3-Clause
"""Times a batched camera's observations mapped to CUDA in place
against copied at each map (ernie:outputs:inPlace), for a few sets of
render vars, and checks both give the same observations."""

import time

import numpy as np

import ernie
import ernie.demo
from ernie import tokens
from ernie.batch import BatchCamera

SIZE = 64
STEPS = 30  # timed steps
WARMUP = 5  # steps before timing: uploads, pipelines, allocations
BANDS = 3

# The render vars timed, from one small one to the spectrum.
VAR_SETS = [
    ("LdrColor",),
    ("LdrColor", "DistanceToCamera", "Normal", "PrimId"),
    ("LdrColor", "SpectralRadiance"),
]


def build_scene(scene, worlds):
    """A cube and a sphere in each world, under the sun."""
    scene.define("/World/Ground", ernie.GroundPlane)
    scene.define("/World/Sun", ernie.DistantLight).set(
        tokens.TRANSFORM,
        ernie.distant_light_transform((0.4, -0.6, 0.7)),
    )
    env0 = scene.define("/Envs/Env0", ernie.Xform)
    env0.define(
        "Cube",
        ernie.Cube,
        display_color=(0.8, 0.3, 0.2),
        translate=(-1.0, 0.0, 0.5),
    )
    env0.define(
        "Sphere",
        ernie.Sphere,
        display_color=(0.2, 0.5, 0.8),
        translate=(1.2, 0.5, 0.5),
    )
    for world in range(worlds):
        env = (
            env0
            if world == 0
            else ernie.Xform(
                scene.clone("/Envs/Env0", f"/Envs/Env{world}")
            )
        )
        env.world = world
    scene.define(
        "/World/Camera",
        ernie.Camera,
        focal_length=18.0,
        spectral_bands=BANDS,
        sensor_noise=False,
    )
    return scene


def orbit(views, angle):
    """One pose per view, circling its world from its own heading."""
    poses = np.empty((views, 4, 4), np.float32)
    for v in range(views):
        heading = angle + 2.0 * np.pi * v / views
        eye = (6.0 * np.cos(heading), 6.0 * np.sin(heading), 4.0)
        poses[v] = ernie.look_at_transform(
            eye, (0.0, 0.0, 0.5), up=(0.0, 0.0, 1.0)
        )
    return poses


class Batch:
    """A batch camera of `views` views over a scene of their worlds
    (`scene`, else one of its own), its outputs mapped in place or
    copied."""

    def __init__(
        self, renderer, views, render_vars, in_place, scene=None
    ):
        self.renderer = renderer
        self.scene = build_scene(
            scene if scene is not None else ernie.Scene(), views
        )
        self.camera = BatchCamera(
            renderer,
            self.scene,
            "/World/Camera",
            views,
            resolution=(SIZE, SIZE),
            render_vars=render_vars,
        )
        self.scene.set(
            self.camera.product, tokens.OUTPUTS_IN_PLACE, in_place
        )
        self.camera.set_worlds(np.arange(views, dtype=np.int32))
        self.poses = [
            renderer.cuda_upload(orbit(views, 0.05 * s))
            for s in range(WARMUP + STEPS)
        ]

    def step(self, s):
        return self.camera.step(self.poses[s % len(self.poses)])

    def time(self):
        """Milliseconds per step until its observations are mapped,
        and per step with the render finished."""
        for s in range(WARMUP):
            self.step(s)
        mapped = 0.0
        start = time.perf_counter()
        for s in range(STEPS):
            begun = time.perf_counter()
            observations = self.step(WARMUP + s)
            mapped += time.perf_counter() - begun
        # The last step has finished.
        self.renderer.cuda_download(observations["LdrColor"])
        total = time.perf_counter() - start
        return 1e3 * mapped / STEPS, 1e3 * total / STEPS


def contact_sheet(images, columns=4):
    rows = (len(images) + columns - 1) // columns
    sheet = np.zeros((rows * SIZE, columns * SIZE, 4), np.uint8)
    for i, image in enumerate(images):
        r, c = divmod(i, columns)
        sheet[
            r * SIZE : (r + 1) * SIZE, c * SIZE : (c + 1) * SIZE
        ] = image
    return sheet


class Demo(ernie.demo.Demo):
    num_frames = 1

    @classmethod
    def create_parser(cls):
        parser = super().create_parser()
        parser.add_argument(
            "--views",
            type=int,
            default=1024,
            help="environments, one view each",
        )
        return parser

    def __init__(self, viewer, args):
        super().__init__(viewer, args)
        renderer = self.renderer
        if not renderer.cuda_available:
            raise RuntimeError("This demo needs CUDA interop")
        self.views = args.views
        print(
            f"{renderer.device_name}, {self.views} views of "
            f"{SIZE} x {SIZE}, {BANDS} bands, ms per step"
        )
        print(
            f"{'render vars':52} {'copied':>15} {'in place':>15}  "
            "(until mapped / rendered)"
        )
        for render_vars in VAR_SETS:
            copied = Batch(renderer, self.views, render_vars, False)
            in_place = Batch(renderer, self.views, render_vars, True)
            times = [copied.time(), in_place.time()]
            columns = "".join(
                f" {until:6.2f} / {whole:6.2f}"
                for until, whole in times
            )
            print(f"{', '.join(render_vars):52}{columns}")
        # The views of the last set, for the output and the test: the
        # same step mapped in place and copied.
        render_vars = VAR_SETS[-1]
        self.in_place = Batch(
            renderer, self.views, render_vars, True, self.scene
        )
        self.copied = Batch(renderer, self.views, render_vars, False)

    def step(self):
        pass

    def render(self):
        # Without a viewer the timed steps were the renders.
        if self.viewer is not None:
            self.viewer.step()

    def observations(self, batch):
        """Its first pose's observations on the host, after a step
        that leaves its outputs in place."""
        batch.step(0)
        observations = batch.step(0)
        return {
            name: np.asarray(self.renderer.cuda_download(array))
            for name, array in observations.items()
        }

    def save_output(self, path):
        images = self.observations(self.in_place)["LdrColor"]
        # 16 views spread round the orbit.
        ernie.demo.write_png(
            path,
            contact_sheet(images[:: max(self.views // 16, 1)][:16]),
        )

    def test_final(self):
        in_place = self.observations(self.in_place)
        copied = self.observations(self.copied)
        for name, array in copied.items():
            assert np.array_equal(in_place[name], array), (
                f"{name} in place differs from copied"
            )
        images = in_place["LdrColor"][..., :3]
        assert images.std() > 1.0, "the views should not render flat"


if __name__ == "__main__":
    ernie.demo.main(Demo)
