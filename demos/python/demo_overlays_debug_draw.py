# SPDX-License-Identifier: BSD-3-Clause
"""Debug drawing with overlays: a ball bouncing among boxes, drawn over with its trajectory,
velocity, contacts, bounding boxes, frames and labels, redrawn every frame. The Overlays panel
shows or clears each layer.

    PYTHONPATH=build/python python demos/python/demo_overlays_debug_draw.py
    PYTHONPATH=build/python python demos/python/demo_overlays_debug_draw.py --headless \
        --output overlays.png

Overlays draw only in the viewer: `--headless` renders it without a window and saves its
viewport, overlays and all; `--viewer null` renders the scene's camera without them.
"""

import math

import numpy as np

import ernie
import ernie.demo
from ernie import tokens

GRAVITY = 9.81
RADIUS = 0.5


class Ball:
    """A ball bouncing on the ground and circling the origin, with its recent path."""

    def __init__(self):
        self.height = 3.0
        self.climb = 0.0
        self.angle = 0.0
        self.path = []
        self.contacts = []

    def step(self, dt):
        self.angle += 0.6 * dt
        self.climb -= GRAVITY * dt
        self.height += self.climb * dt
        if self.height < RADIUS:
            self.height = RADIUS
            self.climb = abs(self.climb) * 0.92 + 0.3
            self.contacts = (self.contacts + [self.position() - (0, 0, RADIUS)])[-8:]
        self.path = (self.path + [self.position()])[-240:]

    def position(self):
        return np.array([5.0 * math.cos(self.angle), 5.0 * math.sin(self.angle), self.height])

    def velocity(self):
        return np.array([-3.0 * math.sin(self.angle), 3.0 * math.cos(self.angle), self.climb])


def draw(scene, ball):
    p = ball.position()
    paths = scene.overlay("trajectory").clear()
    if len(ball.path) > 1:
        paths.polyline(np.array(ball.path), color=(1.0, 0.85, 0.2), width=2, depth="xray")
    paths.arrow(p, p + 0.5 * ball.velocity(), color=(1.0, 0.3, 0.3), width=3, depth="always")

    contacts = scene.overlay("contacts").clear()
    if ball.contacts:
        contacts.markers(np.array(ball.contacts), shape="cross", size=12, color=(1, 0.2, 0.2))
        contacts.circle(ball.contacts[-1], (0, 0, 1), 0.8, color=(1, 0.2, 0.2), width=2)

    bounds = scene.overlay("bounds").clear()
    for i in range(2):
        transform = scene.world_transform(f"/World/Box{i}")
        bounds.box((-1, -1, -1), (1, 1, 1), transform=transform, color=(0.3, 1, 0.5), width=1.5)
        bounds.axes(transform, 1.5, width=2, depth="always")
    bounds.sphere(p, RADIUS * 1.05, color=(0.3, 1, 0.5), width=1.5)

    labels = scene.overlay("labels").clear()
    labels.label(p + (0, 0, RADIUS), f"ball  z={p[2]:.2f} m  vz={ball.climb:+.2f} m/s", size=14)
    labels.label((0, 0, 0), "origin", size=12, color=(0.8, 0.8, 0.8))
    labels.marker((0, 0, 0), shape="ring", size=10, depth="always")


class Demo(ernie.demo.Demo):
    num_frames = 90

    def __init__(self, viewer, args):
        super().__init__(viewer, args)
        world = self.scene.define("/World", ernie.Xform)
        world.define("Ground", ernie.GroundPlane, checker=True, checker_size=1.0)
        world.define("Sun", ernie.DistantLight).set(
            tokens.TRANSFORM, ernie.distant_light_transform((0.4, -0.5, 0.8)))
        for i, x in enumerate((-4.0, 4.0)):
            world.define(f"Box{i}", ernie.Cube, size=2.0, translate=(x, 2.0, 1.0),
                         display_color=(0.75, 0.45, 0.3))
        self.sphere = world.define("Ball", ernie.Sphere, radius=RADIUS,
                                   display_color=(0.25, 0.55, 0.9))
        # The viewer frames the scene with its own camera; this one renders it without a viewer.
        world.define("Camera", ernie.Camera, focal_length=24.0).set(
            tokens.TRANSFORM, ernie.look_at_transform((0.0, -16.0, 8.0), (0.0, 0.0, 1.0)))
        self.ball = Ball()
        self.lowest = self.ball.height
        self.sphere.set(translate=self.ball.position())

    def step(self):
        self.ball.step(self.frame_dt)
        self.lowest = min(self.lowest, self.ball.height)
        self.sphere.set(translate=self.ball.position())
        draw(self.scene, self.ball)

    def test_final(self):
        assert self.lowest >= RADIUS - 1e-6, f"the ball sank to z={self.lowest:.3f}"
        assert self.ball.contacts, "the ball should have bounced"
        assert len(self.ball.path) == min(self.frame, 240)
        image = self.render_image()[..., :3]
        assert image.std() > 1.0, "the ground, boxes and ball should not render flat"


if __name__ == "__main__":
    ernie.demo.main(Demo)
