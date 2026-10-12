# SPDX-License-Identifier: BSD-3-Clause
"""A Warp simulation of bouncing bodies drawn over with debug overlays
read straight from its GPU arrays every frame, with no host work per
step: trails from a ring buffer, velocity arrows from the velocities,
frames and bounds posed under the bodies' transforms, contacts as many
as an atomic counter says, swept capsules, and numbers formatted on
the GPU."""

from typing import no_type_check

import numpy as np
import warp as wp

import ernie
import ernie.demo
from ernie import tokens

BODIES = ernie.Path("/World/Bodies")
RADIUS = wp.constant(0.25)
ARENA = wp.constant(6.0)  # half the arena's width
GRAVITY = wp.constant(9.81)
RESTITUTION = wp.constant(0.7)
TRAIL = 48  # points per trail
MAX_CONTACTS = 4096
LABELLED = 8  # bodies whose speeds are printed


@wp.kernel
@no_type_check
def integrate(
    x: wp.array[wp.vec3],
    v: wp.array[wp.vec3],
    q: wp.array[wp.quat],
    w: wp.array[wp.vec3],
    dt: float,
    previous: wp.array[wp.vec3],
):
    """Gravity, then the arena's floor and walls, bouncing."""
    i = wp.tid()
    previous[i] = x[i]
    vi = v[i] + wp.vec3(0.0, 0.0, -GRAVITY) * dt
    xi = x[i] + vi * dt
    if xi[2] < RADIUS:
        xi = wp.vec3(xi[0], xi[1], RADIUS)
        vi = wp.vec3(vi[0] * 0.98, vi[1] * 0.98, -vi[2] * RESTITUTION)
    for axis in range(2):
        if wp.abs(xi[axis]) > ARENA - RADIUS:
            xi[axis] = wp.sign(xi[axis]) * (ARENA - RADIUS)
            vi[axis] = -vi[axis] * RESTITUTION
    x[i] = xi
    v[i] = vi
    # Spin about the angular velocity.
    wi = w[i]
    angle = wp.length(wi) * dt
    if angle > 0.0:
        turn = wp.quat_from_axis_angle(wp.normalize(wi), angle)
        q[i] = wp.normalize(turn * q[i])


@wp.kernel
@no_type_check
def collide(
    x: wp.array[wp.vec3],
    v: wp.array[wp.vec3],
    contact_points: wp.array[wp.vec3],
    contact_normals: wp.array[wp.vec3],
    contact_count: wp.array[wp.int32],
):
    """Spheres pushed apart and bounced off each other; each touching
    pair's contact appended once, by its lower index."""
    i = wp.tid()
    xi = x[i]
    for j in range(x.shape[0]):
        if j == i:
            continue
        d = xi - x[j]
        dist = wp.length(d)
        if dist < 2.0 * RADIUS and dist > 1.0e-6:
            n = d / dist
            approach = wp.dot(v[i] - v[j], n)
            if approach < 0.0:
                v[i] = v[i] - (1.0 + RESTITUTION) * 0.5 * approach * n
            xi = xi + n * (2.0 * RADIUS - dist) * 0.5
            if i < j:
                slot = wp.atomic_add(contact_count, 0, 1)
                if slot < contact_points.shape[0]:
                    contact_points[slot] = x[j] + n * RADIUS
                    contact_normals[slot] = n
    x[i] = xi


@wp.kernel
@no_type_check
def publish(
    x: wp.array[wp.vec3],
    v: wp.array[wp.vec3],
    q: wp.array[wp.quat],
    previous: wp.array[wp.vec3],
    head: int,
    points: wp.array[wp.vec3],
    body_q: wp.array[wp.transform],
    trails: wp.array2d[wp.vec3],
    heads: wp.array[wp.int32],
    speeds: wp.array[float],
    swept_centres: wp.array[wp.vec3],
    swept_halves: wp.array[float],
    swept_turns: wp.array[wp.quat],
):
    """What the renderer reads: the bodies' points and transforms,
    trails, speeds and swept capsules."""
    i = wp.tid()
    points[i] = x[i]
    body_q[i] = wp.transform(x[i], q[i])
    trails[i, head] = x[i]
    heads[i] = head
    speeds[i] = wp.length(v[i])
    # The capsule this step swept, along its local z.
    step = x[i] - previous[i]
    swept_centres[i] = 0.5 * (x[i] + previous[i])
    swept_halves[i] = 0.5 * wp.length(step)
    swept_turns[i] = wp.quat_between_vectors(
        wp.vec3(0.0, 0.0, 1.0),
        wp.normalize(step + wp.vec3(0.0, 0.0, 1.0e-6)),
    )


@wp.kernel
@no_type_check
def tally(
    time: float,
    contact_count: wp.array[wp.int32],
    speeds: wp.array[float],
    stats: wp.array[float],
):
    """The HUD's numbers: time, contacts and mean speed."""
    stats[0] = time
    stats[1] = float(contact_count[0])
    total = wp.float32(0.0)
    for i in range(speeds.shape[0]):
        total += speeds[i]
    stats[2] = total / float(speeds.shape[0])


class Demo(ernie.demo.Demo):
    num_frames = 180

    @classmethod
    def create_parser(cls):
        parser = super().create_parser()
        parser.add_argument(
            "--bodies", type=int, default=256, help="bodies simulated"
        )
        return parser

    def __init__(self, viewer, args):
        super().__init__(viewer, args)
        wp.init()
        if (
            not wp.is_cuda_available()
            or not self.renderer.cuda_available
        ):
            raise RuntimeError("This demo needs CUDA interop")
        self.device = "cuda:0"
        n = self.bodies = args.bodies
        scene = self.scene
        world = scene.define("/World", ernie.Xform)
        world.define(
            "Ground",
            ernie.GroundPlane,
            checker=True,
            checker_size=1.0,
        )
        world.define("Sun", ernie.DistantLight).set(
            tokens.TRANSFORM,
            ernie.distant_light_transform((0.4, -0.6, 0.7)),
        )
        world.define("Camera", ernie.Camera, focal_length=20.0).set(
            tokens.TRANSFORM,
            ernie.look_at_transform(
                (9.0, -13.0, 9.0), (0.0, 0.0, 1.0)
            ),
        )
        scene.define(
            "/Render/Settings",
            ernie.RenderSettings,
            camera=ernie.Path("/World/Camera"),
            resolution=(960, 540),
            overlays="*",
        )
        if viewer is not None:
            viewer.active_camera = "/World/Camera"
        rng = np.random.default_rng(3)
        start = np.stack(
            [
                rng.uniform(-5.0, 5.0, n),
                rng.uniform(-5.0, 5.0, n),
                rng.uniform(1.0, 6.0, n),
            ],
            axis=1,
        ).astype(np.float32)
        with wp.ScopedDevice(self.device):
            self.x = wp.array(start, dtype=wp.vec3)
            self.v = wp.array(
                rng.normal(0.0, 3.0, (n, 3)).astype(np.float32),
                dtype=wp.vec3,
            )
            self.q = wp.array(
                np.tile(np.float32([0, 0, 0, 1]), (n, 1)),
                dtype=wp.quat,
            )
            self.w = wp.array(
                rng.normal(0.0, 3.0, (n, 3)).astype(np.float32),
                dtype=wp.vec3,
            )
            self.previous = wp.array(start, dtype=wp.vec3)
            # The arrays the overlays read, as the simulation writes them.
            self.body_q = wp.zeros(n, dtype=wp.transform)
            nan = np.full((n, TRAIL, 3), np.nan, np.float32)
            self.trails = wp.array(nan, dtype=wp.vec3)
            self.heads = wp.zeros(n, dtype=wp.int32)
            self.speeds = wp.zeros(n, dtype=float)
            self.swept_centres = wp.zeros(n, dtype=wp.vec3)
            self.swept_halves = wp.zeros(n, dtype=float)
            self.swept_turns = wp.zeros(n, dtype=wp.quat)
            self.contact_points = wp.zeros(
                MAX_CONTACTS, dtype=wp.vec3
            )
            self.contact_normals = wp.zeros(
                MAX_CONTACTS, dtype=wp.vec3
            )
            self.contact_count = wp.zeros(1, dtype=wp.int32)
            self.stats = wp.zeros(3, dtype=float)
            scene.define(
                str(BODIES),
                ernie.Points,
                widths=2.0 * RADIUS,
                display_color=(0.9, 0.55, 0.25),
            )
            scene.write(
                BODIES, tokens.POINTS, wp.zeros((n, 3), dtype=float)
            )
        self.add_overlays()
        self.peak_contacts = 0

    def add_overlays(self):
        """Every layer once: from here on the GPU arrays alone change."""
        scene = self.scene
        stream = wp.get_stream(self.device)
        arrays = dict(stream=stream)
        zeros = wp.zeros(
            (self.bodies, 3), dtype=float, device=self.device
        )
        # Trails, fading from each body back through its ring.
        scene.overlay("trails").trails(
            self.trails,
            heads=self.heads,
            color=(0.3, 0.7, 1.0),
            width=2,
            **arrays,
        )
        # Velocities, the simulation's own vectors.
        scene.overlay("velocities").arrows(
            self.x,
            self.v,
            directions=True,
            scale=0.15,
            color=(0.3, 1.0, 0.4),
            width=2,
            **arrays,
        )
        # Frames and oriented bounds posed under the bodies' transforms.
        scene.overlay("frames").frames(
            zeros, length=0.45, poses=self.body_q, width=2, **arrays
        )
        scene.overlay("bounds").boxes(
            zeros,
            wp.full(
                (self.bodies, 3),
                0.25,
                dtype=float,
                device=self.device,
            ),
            poses=self.body_q,
            color=(0.3, 0.9, 1.0, 0.5),
            width=1,
            depth="xray",
            **arrays,
        )
        # Contacts, as many as the counter says, over everything.
        contacts = scene.overlay("contacts")
        contacts.points(
            self.contact_points,
            shape="ring",
            size=10,
            count=self.contact_count,
            color=(1.0, 0.9, 0.2),
            depth="always",
            **arrays,
        )
        contacts.arrows(
            self.contact_points,
            self.contact_normals,
            directions=True,
            scale=0.5,
            count=self.contact_count,
            color=(1.0, 0.5, 0.1),
            width=2,
            depth="always",
            **arrays,
        )
        # The capsules the bodies swept this step.
        scene.overlay("swept").capsules(
            self.swept_centres,
            wp.full(
                self.bodies, 0.25, dtype=float, device=self.device
            ),
            self.swept_halves,
            orientations=self.swept_turns,
            color=(1.0, 0.4, 0.8, 0.35),
            width=1,
            **arrays,
        )
        # The first bodies' speeds, under their transforms.
        above = wp.array(
            np.tile(np.float32([0.0, 0.0, 0.5]), (LABELLED, 1)),
            dtype=float,
            device=self.device,
        )
        scene.overlay("labels").numbers(
            self.speeds[:LABELLED],
            points=above,
            suffix=" m/s",
            decimals=1,
            poses=self.body_q,
            pose_indices=wp.array(
                np.arange(LABELLED, dtype=np.int32),
                device=self.device,
            ),
            size=11,
            depth="always",
            **arrays,
        )
        # The HUD: a title from the host once, numbers from the GPU.
        hud = scene.overlay("hud")
        hud.text(
            f"{self.bodies} bodies, every overlay read from Warp arrays",
            offset=(12, 12),
            size=14,
        )
        for row, (prefix, suffix, decimals) in enumerate(
            [
                ("t = ", " s", 2),
                ("contacts ", "", 0),
                ("mean speed ", " m/s", 2),
            ]
        ):
            hud.numbers(
                self.stats[row : row + 1],
                prefix=prefix,
                suffix=suffix,
                decimals=decimals,
                screen=True,
                offset=(12, 36 + 20 * row),
                size=14,
                **arrays,
            )

    def step(self):
        stream = wp.get_stream(self.device)
        launch = dict(device=self.device, stream=stream)
        n = self.bodies
        self.contact_count.zero_()
        wp.launch(
            integrate,
            dim=n,
            inputs=[
                self.x,
                self.v,
                self.q,
                self.w,
                self.frame_dt,
                self.previous,
            ],
            **launch,
        )
        wp.launch(
            collide,
            dim=n,
            inputs=[
                self.x,
                self.v,
                self.contact_points,
                self.contact_normals,
                self.contact_count,
            ],
            **launch,
        )
        # The bodies' points written in the renderer's own memory.
        points = wp.from_dlpack(
            self.map(BODIES, tokens.POINTS, stream=stream),
            dtype=wp.vec3,
        )
        wp.launch(
            publish,
            dim=n,
            inputs=[
                self.x,
                self.v,
                self.q,
                self.previous,
                self.frame % TRAIL,
                points,
                self.body_q,
                self.trails,
                self.heads,
                self.speeds,
                self.swept_centres,
                self.swept_halves,
                self.swept_turns,
            ],
            **launch,
        )
        self.scene.unmap(BODIES, tokens.POINTS, stream=stream)
        wp.launch(
            tally,
            dim=1,
            inputs=[
                self.time,
                self.contact_count,
                self.speeds,
                self.stats,
            ],
            **launch,
        )
        if self.args.test:
            self.peak_contacts = max(
                self.peak_contacts, int(self.contact_count.numpy()[0])
            )

    def test_final(self):
        assert self.peak_contacts > 0, "bodies should touch"
        assert self.peak_contacts <= MAX_CONTACTS
        for layer in (
            "trails",
            "velocities",
            "frames",
            "contacts",
            "hud",
        ):
            assert not self.scene.overlay(layer).empty, layer
        heads = self.heads.numpy()
        assert (heads == (self.frame - 1) % TRAIL).all(), (
            "the trails' heads should follow the steps"
        )
        image = self.render_image().astype(np.int32)
        assert image[..., :3].std() > 10.0, (
            "the image should not be flat"
        )


if __name__ == "__main__":
    ernie.demo.main(Demo)
