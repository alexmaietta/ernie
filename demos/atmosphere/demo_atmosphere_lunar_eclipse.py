# SPDX-License-Identifier: BSD-3-Clause
"""A lunar eclipse from the ground: the Moon, a sphere of Lommel-Seeliger
regolith, in the Earth's shadow, lit red by sunlight the atmosphere
refracts into it, darker with clouds."""

import math

import numpy as np

import ernie
import ernie.demo
from ernie import tokens
from ernie.demo import side_by_side, srgb_to_linear, write_png

EARTH_RADIUS = 6371.0088e3
MOON_DISTANCE = 384400e3
MOON_ELEVATION = 40.0  # degrees

# The Moon's offset from the shadow's axis (degrees, from the Earth's
# centre), the clouds' tops (m), and an exposure: f-number and seconds
# at ISO 100. The umbra is some 0.7 degrees in radius at the Moon, the
# penumbra 1.2.
VIEWS = [
    ("Full Moon", 3.0, 0.0, 11.0, 1.0 / 125.0),
    ("Partial: the umbra's edge", 0.75, 0.0, 11.0, 1.0 / 30.0),
    ("Totality, clear air", 0.2, 0.0, 2.8, 1.0 / 2.0),
    ("Totality, clouds to 10 km", 0.2, 10e3, 2.8, 1.0 / 2.0),
]


def middle_linear_rgb(image, half):
    """The mean linear colour of a square of `half` pixels either side
    of the middle of a uint8 image."""
    rows, cols = image.shape[:2]
    square = np.asarray(image)[
        rows // 2 - half : rows // 2 + half,
        cols // 2 - half : cols // 2 + half,
        :3,
    ]
    return srgb_to_linear(square / 255.0).reshape(-1, 3).mean(axis=0)


class Demo(ernie.demo.Demo):
    num_frames = 1

    def __init__(self, viewer, args):
        super().__init__(viewer, args)
        world = self.scene.define("/World", ernie.Xform)
        self.air = world.define(
            "Air", ernie.Atmosphere, preset=tokens.EARTH
        )
        self.sun = world.define(
            "Sun",
            ernie.DistantLight,
            irradiance=1361.0,
            physical_illuminant=tokens.SUN,
            angle=0.53,
        )
        # Normal albedo 0.12: Lommel-Seeliger's single-scattering
        # albedo 8 times it.
        self.scene.define(
            "/World/Looks/Regolith",
            ernie.Material,
            brdf="lommel_seeliger",
            albedo=(8.0 * 0.12,) * 3,
        )
        self.moon = world.define(
            "Moon",
            ernie.Sphere,
            radius=1737.4e3,
            material_binding="/World/Looks/Regolith",
        )
        # 1200 mm: the Moon fills half the frame's height.
        self.camera = world.define(
            "Camera",
            ernie.Camera,
            focal_length=1200.0,
            sensor_noise=False,
            exposure_iso=100.0,
            clipping_range=(1.0, math.inf),
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
        self.widgets.slider(
            "Offset from the axis",
            0.0,
            2.0,
            VIEWS[0][1],
            on_change=self.sun_at,
        )
        self.widgets.slider(
            "Cloud tops (km)",
            0.0,
            15.0,
            0.0,
            on_change=lambda km: setattr(
                self.air, "cloud_top", km * 1e3
            ),
        )
        if viewer is not None:
            viewer.active_camera = self.camera.path

    def show(self, index):
        self.view = index
        _, offset, cloud_top, f_stop, exposure_time = VIEWS[index]
        self.air.cloud_top = cloud_top
        # The Moon 40 degrees up from the Earth's centre; the camera on
        # the ground aims at it.
        e = math.radians(MOON_ELEVATION)
        moon = (
            MOON_DISTANCE * math.cos(e),
            0.0,
            -EARTH_RADIUS + MOON_DISTANCE * math.sin(e),
        )
        self.moon.set_xform(translate=moon)
        eye = (0.0, 0.0, 2.0)
        towards = np.subtract(moon, eye)
        self.camera.exposure_f_stop = f_stop
        self.camera.exposure_time = exposure_time
        self.camera.set(
            tokens.TRANSFORM,
            ernie.look_at_transform(
                eye,
                tuple(np.add(eye, towards / np.linalg.norm(towards))),
                (0.0, 0.0, 1.0),
            ),
        )
        self.sun_at(offset)

    def sun_at(self, degrees):
        """The sun opposite the Moon from the Earth's centre, `degrees`
        off the line, so the Moon lies that far from the shadow's
        axis."""
        e = math.radians(MOON_ELEVATION + degrees)
        self.sun.set(
            tokens.TRANSFORM,
            ernie.distant_light_transform(
                (-math.cos(e), 0.0, -math.sin(e))
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
        """The four views side by side."""
        write_png(path, side_by_side(self.render_all()))

    def test_final(self):
        middle = [
            middle_linear_rgb(image, 10)
            for image in self.render_all()
        ]
        for (name, *_), m in zip(VIEWS, middle):
            print(
                f"{name}: the Moon's middle {m[0]:.4f} {m[1]:.4f} {m[2]:.4f}"
            )
        assert (
            middle[0][0] > 0.05 and middle[0][2] > 0.6 * middle[0][0]
        ), "the full Moon should be bright and grey"
        assert middle[2][0] > 3.0 * middle[2][2], (
            "the eclipsed Moon should be red"
        )
        # The same exposure: the clouds darken the umbra.
        assert middle[3][0] < 0.75 * middle[2][0], (
            "clouds should darken the eclipse"
        )


if __name__ == "__main__":
    ernie.demo.main(Demo)
