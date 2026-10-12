# SPDX-License-Identifier: BSD-3-Clause
"""A drone camera's live image shown where it was seen from: on its
frustum a few metres out, with the frustum's edges, and as a picture in
the picture, rendered each step from its own settings."""

import math

import numpy as np

import ernie
import ernie.demo
from ernie import tokens

BILLBOARD = 3.0  # metres in front of the drone
DRONE_IMAGE = (320, 180)


class Demo(ernie.demo.Demo):
    num_frames = 120

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
        # What the drone looks at: a few shapes, one bouncing.
        world.define(
            "Red",
            ernie.Cube,
            display_color=(0.85, 0.2, 0.15),
            translate=(-1.5, 0, 0.5),
        )
        world.define(
            "Blue",
            ernie.Cube,
            display_color=(0.2, 0.35, 0.9),
            translate=(1.5, 1, 0.5),
        )
        self.ball = world.define(
            "Ball",
            ernie.Sphere,
            radius=0.5,
            display_color=(0.95, 0.8, 0.2),
        )
        # The view over it all, and the drone's own camera and settings,
        # which draw no overlays.
        camera = world.define(
            "Camera", ernie.Camera, focal_length=22.0
        ).set(
            tokens.TRANSFORM,
            ernie.look_at_transform(
                (-9.0, -14.0, 9.0), (0.0, 0.0, 1.5)
            ),
        )
        scene.define(
            "/Render/Settings",
            ernie.RenderSettings,
            camera=camera,
            resolution=(960, 540),
            overlays="*",
        )
        self.drone = world.define(
            "Drone",
            ernie.Camera,
            focal_length=18.0,
            horizontal_aperture=20.955,
            vertical_aperture=20.955
            * DRONE_IMAGE[1]
            / DRONE_IMAGE[0],
        )
        scene.define(
            "/Render/Drone",
            ernie.RenderSettings,
            camera=self.drone,
            resolution=DRONE_IMAGE,
        )
        if viewer is not None:
            viewer.active_camera = "/World/Camera"
        self.drone_image = None
        self.place()

    def step(self):
        self.place()

    def place(self):
        """Flies the drone, renders what it sees and shows it."""
        scene = self.scene
        t = self.time
        self.ball.set(
            translate=(0.0, -1.0, 0.5 + 1.5 * abs(math.sin(2.5 * t)))
        )
        angle = 0.4 * t
        eye = (6.0 * math.cos(angle), 6.0 * math.sin(angle), 3.5)
        self.drone.set(
            tokens.TRANSFORM,
            ernie.look_at_transform(eye, (0.0, 0.0, 0.5)),
        )
        self.drone_image = np.asarray(
            self.renderer.render(scene, ernie.Path("/Render/Drone"))
        )
        # Its image where it was seen from, the frustum's edges to it,
        # and the same image in the top right corner.
        drone = scene.overlay("drone")
        drone.clear()
        corners = ernie.camera_image_corners(
            scene, "/World/Drone", BILLBOARD
        )
        edge = dict(color=(1.0, 0.85, 0.3), width=2.0)
        for k in range(4):
            drone.line(eye, corners[k], **edge)
            drone.line(corners[k], corners[(k + 1) % 4], **edge)
        drone.image(
            corners, self.drone_image, color=(1.0, 1.0, 1.0, 0.95)
        )
        drone.rect(
            (DRONE_IMAGE[0] + 6.0, DRONE_IMAGE[1] + 6.0),
            at=(1, 0),
            offset=(-9.0, 9.0),
            color=edge["color"],
        )
        drone.screen_image(
            self.drone_image,
            at=(1, 0),
            offset=(-12, 12),
            size=DRONE_IMAGE,
        )
        drone.text(
            "drone camera",
            at=(1, 0),
            offset=(-12, 24 + DRONE_IMAGE[1]),
            size=13,
        )

    def test_final(self):
        image = self.drone_image[..., :3].astype(float)
        assert image.std() > 5.0, "the drone should see the scene"
        assert self.scene.overlay("drone").batch_count == 2, (
            "its image on its frustum and in the corner"
        )


if __name__ == "__main__":
    ernie.demo.main(Demo)
