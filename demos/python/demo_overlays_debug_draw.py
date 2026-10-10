# SPDX-License-Identifier: BSD-3-Clause
"""Debug drawing with overlays: a ball bouncing among boxes, drawn over with its trajectory,
velocity, contacts, bounding boxes, frames and labels, redrawn every frame. The Overlays panel
shows or clears each layer.

    PYTHONPATH=build/python python demos/python/demo_overlays_debug_draw.py

Pass --capture PATH to render a few frames without a window and save the last.
"""

import argparse
import math

import numpy as np

import ernie
from ernie import tokens
from ernie.app import Viewer

GRAVITY = 9.81
RADIUS = 0.5


def translation(x, y, z):
    m = np.eye(4)
    m[:3, 3] = (x, y, z)
    return m


def build_scene():
    scene = ernie.Scene()
    scene.define("/World/Ground", tokens.GROUND_PLANE).set(tokens.CHECKER, True).set(
        tokens.CHECKER_SIZE, 1.0
    )
    scene.define("/World/Sun", tokens.DISTANT_LIGHT).set(
        tokens.TRANSFORM, ernie.distant_light_transform((0.4, -0.5, 0.8))
    )
    for i, x in enumerate((-4.0, 4.0)):
        scene.define(f"/World/Box{i}", tokens.CUBE).set(tokens.SIZE, 2.0).set(
            tokens.TRANSFORM, translation(x, 2.0, 1.0)
        ).set(tokens.DISPLAY_COLOR, (0.75, 0.45, 0.3))
    scene.define("/World/Ball", tokens.SPHERE).set(tokens.GEOM_RADIUS, RADIUS).set(
        tokens.DISPLAY_COLOR, (0.25, 0.55, 0.9)
    )
    return scene


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


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--capture", help="render without a window and save the image here")
    args = parser.parse_args()

    scene = build_scene()
    ball = Ball()
    viewer = Viewer(scene, title="Ernie overlays", hidden=args.capture is not None)

    def update(v, info):
        ball.step(min(info.delta, 1.0 / 30.0) if args.capture is None else 1.0 / 60.0)
        v.scene.set("/World/Ball", tokens.TRANSFORM, translation(*ball.position()))
        draw(v.scene, ball)

    viewer.on_frame(update)
    if args.capture is None:
        viewer.run()
        return
    for _ in range(90):
        viewer.step()
    viewer.capture(args.capture)
    viewer.step()


if __name__ == "__main__":
    main()
