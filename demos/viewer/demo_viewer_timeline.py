# SPDX-License-Identifier: BSD-3-Clause
"""A moon round a planet and a swinging pendulum posed from the stage's
time code, so the timeline plays, pauses, scrubs, steps and loops them
(Space plays; the Capture panel records the loop)."""

import math

import ernie
import ernie.demo
from ernie import tokens

FPS = 24.0
LOOP = 10.0  # s, the timeline's range; every period divides it
ORBIT = (4.0, LOOP)  # m and s, the planet round the origin
MOON = (1.2, LOOP / 4.0)  # m and s, the moon round the planet
PIVOT = (-7.0, 0.0, 6.0)  # m, where the pendulum hangs
ARM = 3.5  # m
SWING = (40.0, LOOP / 2.0)  # degrees and s


def pose(t):
    """Where everything is `t` seconds into the loop: the planet's and
    moon's centres and the pendulum's angle in degrees."""
    a = 2.0 * math.pi * t / ORBIT[1]
    planet = (ORBIT[0] * math.cos(a), ORBIT[0] * math.sin(a), 1.0)
    b = 2.0 * math.pi * t / MOON[1]
    moon = (
        planet[0] + MOON[0] * math.cos(b),
        planet[1] + MOON[0] * math.sin(b),
        planet[2],
    )
    swing = SWING[0] * math.cos(2.0 * math.pi * t / SWING[1])
    return planet, moon, swing


class Demo(ernie.demo.Demo):
    num_frames = 240

    def __init__(self, viewer, args):
        super().__init__(viewer, args)
        scene = self.scene
        # Ten seconds at 24 frames a second: the timeline's range.
        stage = ernie.Stage(scene)
        stage.time_codes_per_second = FPS
        stage.frames_per_second = FPS
        stage.start_time_code = 0.0
        stage.end_time_code = LOOP * FPS
        stage.time_code = 0.0
        world = scene.define("/World", ernie.Xform)
        world.define("Ground", ernie.GroundPlane, checker=True)
        world.define("Sun", ernie.DistantLight).set(
            tokens.TRANSFORM,
            ernie.distant_light_transform((0.3, -0.5, 0.8)),
        )
        world.define("Camera", ernie.Camera, focal_length=20.0).set(
            tokens.TRANSFORM,
            ernie.look_at_transform(
                (-2.0, -17.0, 8.0), (-2.0, 0.0, 2.0)
            ),
        )
        self.planet = world.define(
            "Planet",
            ernie.Sphere,
            radius=0.8,
            display_color=(0.2, 0.4, 0.9),
        )
        self.moon = world.define(
            "Moon",
            ernie.Sphere,
            radius=0.3,
            display_color=(0.8, 0.8, 0.75),
        )
        self.rod = world.define(
            "Rod", ernie.Cylinder, radius=0.05, height=ARM
        )
        self.bob = world.define(
            "Bob",
            ernie.Sphere,
            radius=0.5,
            display_color=(0.9, 0.5, 0.1),
        )
        self.place(0.0)
        if viewer is not None:
            # The timeline drives the poses: after it moves on, before
            # the frame renders.
            viewer.on_frame(
                lambda viewer, info: self.place(info.time_code / FPS)
            )
            viewer.looping = True
            viewer.play()

    def place(self, t):
        planet, moon, swing = pose(t)
        self.planet.set(translate=planet)
        self.moon.set(translate=moon)
        theta = math.radians(swing)
        down = (math.sin(theta), 0.0, -math.cos(theta))
        self.bob.set(
            translate=tuple(p + ARM * d for p, d in zip(PIVOT, down))
        )
        # The rod, an upright cylinder, tilted along the arm.
        self.rod.set(
            translate=tuple(
                p + 0.5 * ARM * d for p, d in zip(PIVOT, down)
            ),
            rotate=(0.0, -swing, 0.0),
        )

    def step(self):
        if self.viewer is not None:
            return
        # Without a viewer, the demo plays the timeline itself.
        stage = ernie.Stage(self.scene)
        code = stage.time_code + self.frame_dt * FPS
        stage.time_code = code % (LOOP * FPS)
        self.place(stage.time_code / FPS)

    def test_post_step(self):
        t = ernie.Stage(self.scene).time_code / FPS
        planet, moon, _ = pose(t)
        at = self.scene.world_transform(self.planet.path)
        assert all(
            abs(at[i][3] - planet[i]) < 1e-9 for i in range(3)
        ), "the planet is where the time code puts it"
        assert abs(math.dist(planet, moon) - MOON[0]) < 1e-9

    def test_final(self):
        code = ernie.Stage(self.scene).time_code
        assert 0.0 <= code < LOOP * FPS, (
            "the time code stays in range"
        )
        # The loop is seamless: its end poses as its start.
        start, end = pose(0.0), pose(LOOP)
        for a, b in zip(start[:2], end[:2]):
            assert math.dist(a, b) < 1e-9
        assert abs(start[2] - end[2]) < 1e-9


if __name__ == "__main__":
    ernie.demo.main(Demo)
