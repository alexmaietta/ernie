# SPDX-License-Identifier: BSD-3-Clause
"""Bouncing balls under a heads-up display: screen text and an energy
bar redrawn each step, labels posed under each ball that follow it
unchanged, and each ball's height printed on the GPU from an array."""

import numpy as np

import ernie
import ernie.demo
from ernie import tokens

BALLS = 5
GRAVITY = 9.81
RADIUS = 0.5
RESTITUTION = 0.85
WIDTH, HEIGHT = 640, 360
BAR_WIDTH = 240.0
BAR_COLOR = (0.25, 0.85, 0.35, 1.0)


class Demo(ernie.demo.Demo):
    num_frames = 300

    def __init__(self, viewer, args):
        super().__init__(viewer, args)
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
        camera = world.define(
            "Camera", ernie.Camera, focal_length=24.0
        ).set(
            tokens.TRANSFORM,
            ernie.look_at_transform(
                (0.0, -14.0, 5.0), (0.0, 0.0, 2.0)
            ),
        )
        scene.define(
            "/Render/Settings",
            ernie.RenderSettings,
            camera=camera,
            resolution=(WIDTH, HEIGHT),
            overlays="*",
        )
        # Labels and rings in each ball's space: added once, they
        # follow the balls with no redraw.
        tracked = scene.overlay("tracked")
        self.balls = []
        for i in range(BALLS):
            ball = world.define(
                f"Ball{i}",
                ernie.Sphere,
                radius=RADIUS,
                display_color=(0.9, 0.3 + 0.12 * i, 0.2),
            )
            path = str(ball.path)
            tracked.label(
                (0.0, 0.0, RADIUS),
                f"ball {i}",
                anchor=path,
                depth="always",
            )
            tracked.marker(
                (0.0, 0.0, 0.0),
                shape="ring",
                size=28,
                color=(1, 1, 1, 0.6),
                anchor=path,
                depth="always",
            )
            self.balls.append(ball)
        self.heights = 1.0 + 0.8 * np.arange(BALLS)
        self.velocities = np.zeros(BALLS)
        # Each ball's height, formatted on the GPU from an array
        # written in place each step.
        self.values = np.zeros(BALLS, np.float32)
        below = np.zeros((BALLS, 3), np.float32)
        below[:, 2] = -2.5 * RADIUS
        tracked.numbers(
            self.values,
            points=below,
            prefix="h ",
            suffix=" m",
            decimals=2,
            anchors=[str(ball.path) for ball in self.balls],
            color=(0.6, 0.9, 1.0),
            depth="always",
        )
        self.initial_energy = None
        self.place()

    def step(self):
        self.velocities -= GRAVITY * self.frame_dt
        self.heights += self.velocities * self.frame_dt
        landed = self.heights < 0.0
        self.heights[landed] = 0.0
        self.velocities[landed] *= -RESTITUTION
        self.place()

    def place(self):
        """Poses the balls, writes their heights for the GPU's text
        and redraws the screen text."""
        for i, ball in enumerate(self.balls):
            ball.set(
                translate=(
                    -4.0 + 2.0 * i,
                    0.0,
                    self.heights[i] + RADIUS,
                )
            )
        self.values[:] = self.heights
        # Host arrays written in place: read again once touched.
        self.scene.overlay("tracked").touch()
        energy = float(
            np.sum(GRAVITY * self.heights + 0.5 * self.velocities**2)
        )
        if self.initial_energy is None:
            self.initial_energy = energy
        hud = self.scene.overlay("hud").clear()
        hud.text(
            f"t = {self.frame * self.frame_dt:.2f} s\nstep {self.frame}",
            offset=(12, 12),
            size=16,
        )
        hud.text(
            f"{BALLS} balls", at=(1, 0), offset=(-12, 12), size=16
        )
        # The energy left, as a bar over a dark track.
        share = min(max(energy / self.initial_energy, 0.0), 1.0)
        bar = dict(
            at=(0.5, 1), offset=(-BAR_WIDTH / 2, -16), align=(0, 1)
        )
        hud.rect((BAR_WIDTH, 14), color=(0, 0, 0, 0.6), **bar)
        hud.rect((BAR_WIDTH * share, 14), color=BAR_COLOR, **bar)
        hud.text(
            "energy",
            at=(0.5, 1),
            offset=(0, -34),
            align=(0.5, 1),
            size=12,
            background=(0, 0, 0, 0),
        )

    def test_final(self):
        assert not self.scene.overlay("hud").empty, (
            "the HUD holds text"
        )
        image = self.image
        if image is None or np.asarray(image).size == 0:
            return
        image = np.asarray(image)
        # The energy bar's fill, left of the bottom centre.
        texel = image[
            HEIGHT - 16 - 7, int(WIDTH / 2 - BAR_WIDTH / 2 + 4)
        ]
        assert texel[1] > 150 and texel[0] < 120 and texel[2] < 140, (
            "the energy bar should be green at its left end"
        )


if __name__ == "__main__":
    ernie.demo.main(Demo)
