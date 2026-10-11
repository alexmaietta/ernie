# SPDX-License-Identifier: BSD-3-Clause
"""Robot arms in thousands of worlds, posed on the GPU: a Warp kernel computes every arm's forward
kinematics into link positions and quaternions, and Ernie renders one view per world from those
arrays without the poses ever coming to the host, as a reinforcement learning loop whose
simulator keeps its state on the GPU.

    PYTHONPATH=build/python python demos/python/demo_rl_gpu_robot_arms.py [--worlds 1024]
        [--viewer null --output arms.png]

Each world ("environment") holds a three-link arm, the worlds laid out on a grid over a shared
ground. The links are Xform prims whose transform ops (`xformOp:translate`, `xformOp:orient`) are
written as rows of two CUDA arrays for all links at once (`Scene.write` of many paths); the boxes
under them are posed by the GPU between their world's transform and their own
(`tokens.XFORM_EXTENT` bounds each link so nothing is copied to the host). The cameras orbit
their worlds, their poses also computed by a kernel as positions and quaternions (`ernie:poses`
as (views, 7)). Each step renders every view through a BatchCamera; the viewer shows the grid of
worlds, and `--output` writes the first 16 worlds' last views as a contact sheet. Needs Warp
(`pip install warp-lang`).
"""

import math
import time

import numpy as np
import warp as wp

import ernie
import ernie.demo
from ernie import tokens
from ernie.batch import BatchCamera

SIZE = 64
LINKS = 3
LENGTHS = (1.2, 1.0, 0.6)
SPACING = 8.0  # metres between neighbouring worlds on the grid
WARMUP = 5  # steps before timing: pipelines, allocations


@wp.kernel
def arm_poses(
    t: float,
    lengths: wp.vec3,
    positions: wp.array(dtype=wp.vec3),
    orientations: wp.array(dtype=wp.quat),
):
    """Forward kinematics of each world's arm: a yaw at the base, then three pitch joints."""
    world = wp.tid()
    phase = float(world) * 0.37
    up = wp.vec3(0.0, 0.0, 1.0)
    side = wp.vec3(0.0, 1.0, 0.0)
    frame = wp.transform(wp.vec3(0.0, 0.0, 0.6), wp.quat_from_axis_angle(up, 0.8 * t + phase))
    pitch0 = -0.7 - 0.4 * wp.sin(1.3 * t + phase)
    pitch1 = 1.1 + 0.5 * wp.sin(1.9 * t + 2.0 * phase)
    pitch2 = 0.7 * wp.sin(2.7 * t + phase)

    frame = wp.transform_multiply(frame, wp.transform(wp.vec3(), wp.quat_from_axis_angle(side, pitch0)))
    positions[world * 3] = wp.transform_get_translation(frame)
    orientations[world * 3] = wp.transform_get_rotation(frame)

    step = wp.transform(wp.vec3(lengths[0], 0.0, 0.0), wp.quat_from_axis_angle(side, pitch1))
    frame = wp.transform_multiply(frame, step)
    positions[world * 3 + 1] = wp.transform_get_translation(frame)
    orientations[world * 3 + 1] = wp.transform_get_rotation(frame)

    step = wp.transform(wp.vec3(lengths[1], 0.0, 0.0), wp.quat_from_axis_angle(side, pitch2))
    frame = wp.transform_multiply(frame, step)
    positions[world * 3 + 2] = wp.transform_get_translation(frame)
    orientations[world * 3 + 2] = wp.transform_get_rotation(frame)


@wp.kernel
def camera_poses(t: float, views: int, columns: int, spacing: float,
                 poses: wp.array2d(dtype=float)):
    """A view per world circling it at its place on the grid: position, then the quaternion
    (x, y, z, w) of a camera looking down its -z at the arm, +y up."""
    view = wp.tid()
    centre = wp.vec3(spacing * float(view % columns), spacing * float(view // columns), 0.0)
    heading = 0.2 * t + 6.2831853 * float(view) / float(views)
    eye = centre + wp.vec3(5.0 * wp.cos(heading), 5.0 * wp.sin(heading), 3.0)
    forward = wp.normalize(centre + wp.vec3(0.0, 0.0, 1.2) - eye)
    right = wp.normalize(wp.cross(forward, wp.vec3(0.0, 0.0, 1.0)))
    up = wp.cross(right, forward)
    rotation = wp.mat33(
        right[0], up[0], -forward[0], right[1], up[1], -forward[1], right[2], up[2], -forward[2]
    )
    q = wp.quat_from_matrix(rotation)
    poses[view, 0] = eye[0]
    poses[view, 1] = eye[1]
    poses[view, 2] = eye[2]
    poses[view, 3] = q[0]
    poses[view, 4] = q[1]
    poses[view, 5] = q[2]
    poses[view, 6] = q[3]


def build_scene(scene, worlds, columns):
    scene.define("/World/Ground", ernie.GroundPlane)
    scene.define("/World/Sun", ernie.DistantLight).set(
        tokens.TRANSFORM, ernie.distant_light_transform((0.4, -0.6, 0.7))
    )
    env0 = scene.define("/Envs/Env0", ernie.Xform)
    env0.define("Base", ernie.Cylinder, radius=0.35, height=0.6, translate=(0.0, 0.0, 0.3),
                display_color=(0.3, 0.3, 0.35))
    colors = [(0.9, 0.5, 0.1), (0.2, 0.6, 0.9), (0.9, 0.9, 0.9)]
    for k in range(LINKS):
        # Each link is posed in its world's frame (not its parent link's), as simulators report
        # bodies, so the links are siblings.
        link = env0.define(f"Link{k}", ernie.Xform)
        link.set(tokens.XFORM_OP_ORDER, [tokens.TRANSLATE, tokens.ORIENT])
        link.set(tokens.XFORM_EXTENT, [-3.5, -3.5, -1.0, 3.5, 3.5, 4.0])
        link.define("Shape", ernie.Cube, translate=(LENGTHS[k] / 2, 0.0, 0.0),
                    scale=(LENGTHS[k], 0.18, 0.18), display_color=colors[k])
    links = []
    for world in range(worlds):
        env = env0 if world == 0 else ernie.Xform(scene.clone("/Envs/Env0", f"/Envs/Env{world}"))
        env.world = world
        row, column = divmod(world, columns)
        env.set(translate=(SPACING * column, SPACING * row, 0.0))
        links += [f"{env.path}/Link{k}" for k in range(LINKS)]
    scene.define("/World/Camera", ernie.Camera, focal_length=18.0, spectral_bands=3)
    # An overview of the grid for the viewer and for --viewer null; the batch views are the
    # BatchCamera's, through /World/Camera.
    rows = (worlds + columns - 1) // columns
    middle = (0.5 * SPACING * (columns - 1), 0.5 * SPACING * (rows - 1), 0.0)
    reach = SPACING * max(columns, rows)
    overview = scene.define("/World/Overview", ernie.Camera, focal_length=24.0).set(
        tokens.TRANSFORM, ernie.look_at_transform(
            (middle[0] - 0.15 * reach, middle[1] - 0.9 * reach, 0.6 * reach + 4.0), middle))
    scene.define("/Render/Settings", ernie.RenderSettings, camera=overview)
    # Looked up once and kept, as every step writes all of them.
    return ernie.PrimSet(links), overview


def contact_sheet(images, columns=4):
    rows = (len(images) + columns - 1) // columns
    sheet = np.zeros((rows * SIZE, columns * SIZE, 4), np.uint8)
    for i, image in enumerate(images):
        r, c = divmod(i, columns)
        sheet[r * SIZE : (r + 1) * SIZE, c * SIZE : (c + 1) * SIZE] = image
    return sheet


class Demo(ernie.demo.Demo):
    num_frames = 120

    @classmethod
    def create_parser(cls):
        parser = super().create_parser()
        parser.add_argument("--worlds", type=int, default=1024, help="worlds, one view each")
        return parser

    def __init__(self, viewer, args):
        super().__init__(viewer, args)
        if not self.renderer.cuda_available:
            raise RuntimeError("This demo needs CUDA interop")
        wp.init()
        self.device = wp.get_device("cuda:0")
        self.stream = wp.get_stream(self.device)
        self.worlds = args.worlds
        self.columns = math.ceil(math.sqrt(self.worlds))

        self.links, overview = build_scene(self.scene, self.worlds, self.columns)
        self.positions = wp.zeros(len(self.links), dtype=wp.vec3, device=self.device)
        self.orientations = wp.zeros(len(self.links), dtype=wp.quat, device=self.device)
        self.poses = wp.zeros((self.worlds, 7), dtype=float, device=self.device)
        self.camera = BatchCamera(self.renderer, self.scene, "/World/Camera", self.worlds,
                                  resolution=(SIZE, SIZE))
        self.camera.set_worlds(np.arange(self.worlds, dtype=np.int32))
        self.observations = None
        self.timer_start = None
        self.timed_steps = 0
        if viewer is not None:
            viewer.active_camera = overview.path

    def step(self):
        t = self.time
        wp.launch(arm_poses, self.worlds,
                  [t, wp.vec3(*LENGTHS), self.positions, self.orientations], device=self.device)
        wp.launch(camera_poses, self.worlds, [t, self.worlds, self.columns, SPACING, self.poses],
                  device=self.device)
        # The kernels' arrays as they are, on the stream they were written on.
        stream = self.stream.cuda_stream
        self.scene.write(self.links, tokens.TRANSLATE, self.positions, stream=stream)
        self.scene.write(self.links, tokens.ORIENT, self.orientations, stream=stream)
        self.observations = self.camera.step(self.poses, stream=stream)
        if self.frame + 1 == WARMUP:
            self.renderer.cuda_download(self.observations["LdrColor"])  # wait for the warm-up
            self.timer_start = time.perf_counter()
        elif self.timer_start is not None:
            self.timed_steps += 1

    def render(self):
        # Without a viewer the batch step is the render; the overview renders only on demand.
        if self.viewer is not None:
            self.viewer.step()

    def images(self):
        """The last step's views, uint8 (worlds, SIZE, SIZE, 4), waiting for the render."""
        return np.asarray(self.renderer.cuda_download(self.observations["LdrColor"]))

    def save_output(self, path):
        self.report()
        ernie.demo.write_png(path, contact_sheet(self.images()[:16]))

    def close(self):
        self.report()

    def report(self):
        """Prints the time per step since the warm-up, once the last step has finished; before
        the output and the test, so that their renders are not counted."""
        if self.observations is None or not self.timed_steps:
            return
        self.images()  # the last step finished
        per_step = (time.perf_counter() - self.timer_start) / self.timed_steps
        where = "" if self.viewer is None else ", with the viewer drawing"
        print(
            f"{self.worlds} worlds of {LINKS}-link arms ({len(self.links)} links posed on the "
            f"GPU): {per_step * 1e3:.2f} ms per step{where}, "
            f"{self.worlds / per_step:,.0f} views per second"
        )
        self.timed_steps = 0

    def test_final(self):
        self.report()
        images = self.images()[..., :3]
        assert images.shape == (self.worlds, SIZE, SIZE, 3)
        assert images.std() > 1.0, "the views should not render flat"
        # Every arm is posed and seen differently.
        if self.worlds > 1:
            assert np.abs(images[0].astype(int) - images[1]).max() > 16, \
                "worlds 0 and 1 should differ"
        # The arms' colours: each view sees its own arm (orange, blue or white links).
        arms = (images[..., 0] > images[..., 2] + 60) | (images[..., 2] > images[..., 0] + 60)
        seen = arms.reshape(self.worlds, -1).any(axis=1).mean()
        assert seen > 0.9, f"only {seen:.0%} of the views see their arm"
        overview = self.render_image()[..., :3]
        assert overview.std() > 1.0, "the overview of the grid should not render flat"


if __name__ == "__main__":
    ernie.demo.main(Demo)
