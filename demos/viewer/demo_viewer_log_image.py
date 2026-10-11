# SPDX-License-Identifier: BSD-3-Clause
"""An onboard camera circling a few objects, its colour (kept on the
GPU) and depth logged each step as live images (log_image)."""

import math

import numpy as np

import ernie
import ernie.demo
from ernie import tokens

SIZE = (160, 120)  # the onboard camera's columns and rows
NEAR, FAR = 2.0, 14.0  # metres the depth image spans, bright to dark
PRODUCT = "/Render/Onboard"

# Things around the origin: (shape, colour, x, y).
THINGS = [
    (ernie.Cube, (0.85, 0.3, 0.25), 2.0, 0.0),
    (ernie.Sphere, (0.25, 0.6, 0.9), -1.5, 1.5),
    (ernie.Cube, (0.95, 0.8, 0.2), -1.0, -2.0),
    (ernie.Sphere, (0.4, 0.85, 0.4), 0.5, 2.5),
    (ernie.Cube, (0.7, 0.4, 0.9), -3.0, -0.5),
]


class Demo(ernie.demo.Demo):
    num_frames = 60

    def __init__(self, viewer, args):
        super().__init__(viewer, args)
        world = self.scene.define("/World", ernie.Xform)
        overview = world.define(
            "Camera", ernie.Camera, focal_length=20.0
        ).set(
            tokens.TRANSFORM,
            ernie.look_at_transform(
                (0.0, -11.0, 7.0), (0.0, 0.0, 0.5)
            ),
        )
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
        for i, (shape, colour, x, y) in enumerate(THINGS):
            world.define(
                f"Thing{i}",
                shape,
                display_color=colour,
                translate=(x, y, 0.5),
            )
        self.onboard = world.define(
            "Onboard", ernie.Camera, focal_length=14.0
        )
        # The onboard camera's product: its colour and its distances.
        product = self.scene.define(
            PRODUCT,
            ernie.RenderProduct,
            camera=self.onboard,
            resolution=SIZE,
        )
        ldr = product.define(
            "Vars/LdrColor", ernie.RenderVar, source_name="LdrColor"
        )
        depth = product.define(
            "Vars/DistanceToCamera",
            ernie.RenderVar,
            source_name="DistanceToCamera",
        )
        product.set(tokens.ORDERED_VARS, [ldr.path, depth.path])
        self.scene.define(
            "/Render/Settings",
            ernie.RenderSettings,
            camera=overview,
            resolution=(640, 360),
        )
        self.speed = 0.4
        self.widgets.slider(
            "Orbit speed",
            0.0,
            2.0,
            self.speed,
            on_change=lambda value: setattr(self, "speed", value),
        )

    def step(self):
        # The onboard camera circles the things, looking in.
        angle = self.speed * self.time
        eye = (6.0 * math.cos(angle), 6.0 * math.sin(angle), 1.5)
        self.onboard.set(
            tokens.TRANSFORM,
            ernie.look_at_transform(eye, (0.0, 0.0, 0.5)),
        )
        outputs = self.renderer.step(
            self.scene, render_products=[PRODUCT]
        )
        frame = outputs[PRODUCT].frames[0]
        # Colour: on the GPU with CUDA interop, never reaching the
        # host.
        device = (
            ernie.Device.CUDA
            if self.renderer.cuda_available
            else ernie.Device.CPU
        )
        self.log_image(
            "onboard/colour",
            frame.render_vars[PRODUCT + "/Vars/LdrColor"].map(
                device=device
            ),
        )
        # Depth: on the host, scaled to grey.
        metres = np.asarray(
            frame.render_vars[
                PRODUCT + "/Vars/DistanceToCamera"
            ].map()
        )
        metres = np.where(np.isfinite(metres), metres, FAR)
        grey = 1.0 - np.clip((metres - NEAR) / (FAR - NEAR), 0.0, 1.0)
        self.log_image("onboard/depth", grey.astype(np.float32))

    def test_final(self):
        depth = self.logged_image("onboard/depth")
        assert depth is not None and depth.shape == (SIZE[1], SIZE[0])
        assert depth.max() > 0.3 and depth.min() < 0.1, (
            "near things bright, the sky dark"
        )
        logged = self.logged_image("onboard/colour")
        assert logged is not None
        colour = np.asarray(
            self.renderer.cuda_download(logged)
            if self.renderer.cuda_available
            else logged
        )
        assert colour.shape == (SIZE[1], SIZE[0], 4)
        assert colour[..., :3].std() > 5.0, (
            "the onboard colour image should not be flat"
        )


if __name__ == "__main__":
    ernie.demo.main(Demo)
