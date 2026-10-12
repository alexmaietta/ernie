# SPDX-License-Identifier: BSD-3-Clause
"""The Earth and its atmosphere rising over the Moon's horizon: gibbous
over the sunlit surface, and a crescent ringed by its lit air."""

import math

import numpy as np

import ernie
import ernie.demo
from ernie import tokens
from ernie.demo import side_by_side, srgb_to_linear, write_png

MOON_RADIUS = 1737.4e3
EARTH_RADIUS = 6371.0088e3
EARTH_DISTANCE = 384400e3

# The Sun's direction (degrees, azimuth from the Earth's) and an exposure
# (f-number, seconds at ISO 100).
VIEWS = [
    ("Earthrise", 110.0, 25.0, 11.0, 1.0 / 250.0),
    ("Crescent Earth", 20.0, -6.0, 4.0, 1.0 / 15.0),
]


def towards(azimuth, elevation):
    a = math.radians(azimuth)
    e = math.radians(elevation)
    return np.array(
        [
            math.cos(e) * math.cos(a),
            math.cos(e) * math.sin(a),
            math.sin(e),
        ]
    )


def rows_linear_rgb(image, top, bottom):
    """The mean linear colour of rows [top, bottom) of a uint8 image."""
    rgb = srgb_to_linear(np.asarray(image)[top:bottom, :, :3] / 255.0)
    return rgb.reshape(-1, 3).mean(axis=0)


class Demo(ernie.demo.Demo):
    num_frames = 1

    def __init__(self, viewer, args):
        super().__init__(viewer, args)
        world = self.scene.define("/World", ernie.Xform)
        # The Moon: a smooth planet Terrain about the world's origin, its
        # regolith's normal albedo.
        world.define(
            "Moon",
            ernie.Terrain,
            heights=np.zeros((6, 65, 65), dtype=np.float32),
            planet_radius=MOON_RADIUS,
            material_binding="/World/Looks/Regolith",
        )
        self.scene.define(
            "/World/Looks/Regolith",
            ernie.Material,
            albedo=(0.12, 0.12, 0.12),
        )
        # The Earth 2 degrees over the horizon to +x: an icosphere with
        # its air centred on it.
        eye = np.array([0.0, 0.0, MOON_RADIUS + 2.0])
        earth = world.define(
            "Earth",
            ernie.Xform,
            translate=tuple(eye + EARTH_DISTANCE * towards(0.0, 2.0)),
        )
        earth.define(
            "Globe",
            ernie.Sphere,
            radius=EARTH_RADIUS,
            tessellation=tokens.ICOSPHERE,
            tessellation_level=6,
            material_binding="/World/Looks/Earth",
        )
        self.scene.define(
            "/World/Looks/Earth",
            ernie.Material,
            albedo=(0.25, 0.28, 0.35),
        )
        earth.define(
            "Air",
            ernie.Atmosphere,
            preset=tokens.EARTH,
            anchor=tokens.ANCHOR_CENTRE,
        )
        # No shadows at these distances.
        self.sun = world.define(
            "Sun",
            ernie.DistantLight,
            irradiance=1361.0,
            physical_illuminant=tokens.SUN,
            angle=0.53,
            shadow_enable=False,
        )
        # Two metres up through a telephoto lens, its far plane infinite.
        self.camera = world.define(
            "Camera",
            ernie.Camera,
            focal_length=150.0,
            clipping_range=(0.5, math.inf),
            sensor_noise=False,
            exposure_iso=100.0,
        )
        self.camera.set(
            tokens.TRANSFORM,
            ernie.look_at_transform(
                tuple(eye),
                tuple(eye + towards(0.0, 1.4)),
                (0.0, 0.0, 1.0),
            ),
        )
        self.scene.define(
            "/Render/Settings",
            ernie.RenderSettings,
            camera=self.camera,
            resolution=(480, 320),
            ambient=0.0,
        )
        self.show(0)

        self.widgets.combo(
            "View",
            [name for name, *_ in VIEWS],
            0,
            on_change=self.show,
        )
        if viewer is not None:
            viewer.active_camera = self.camera.path

    def show(self, index):
        self.view = index
        _, azimuth, elevation, f_stop, exposure_time = VIEWS[index]
        self.camera.exposure_f_stop = f_stop
        self.camera.exposure_time = exposure_time
        self.sun.set(
            tokens.TRANSFORM,
            ernie.distant_light_transform(
                tuple(towards(azimuth, elevation))
            ),
        )

    def render_all(self):
        """The scene from each view in turn, RGBA uint8; leaves the
        view as it was."""
        current = self.view
        images = []
        for index in range(len(VIEWS)):
            self.show(index)
            images.append(np.array(self.renderer.render(self.scene)))
        self.show(current)
        return images

    def save_output(self, path):
        """The views side by side."""
        write_png(path, side_by_side(self.render_all()))

    def test_final(self):
        images = self.render_all()
        rows = images[0].shape[0]
        for (name, *_), image in zip(VIEWS, images):
            sky = rows_linear_rgb(image, 0, rows // 8)
            print(f"{name}: sky {sky}")
            assert sky.sum() < 0.01, "the Moon's sky is black"
        ground = rows_linear_rgb(images[0], rows - rows // 8, rows)
        earth = rows_linear_rgb(images[0], rows // 4, rows // 2)
        print(f"Earthrise: ground {ground}, Earth {earth}")
        assert ground[1] > 0.05, "the sunlit Moon shows"
        assert earth[2] > earth[0], "the Earth shows blue"
        # The crescent: the Earth's lit air beside its night side.
        crescent = rows_linear_rgb(images[1], rows // 4, rows // 2)
        assert crescent.sum() > 0.002, "the crescent shows"


if __name__ == "__main__":
    ernie.demo.main(Demo)
