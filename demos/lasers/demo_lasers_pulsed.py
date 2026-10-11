# SPDX-License-Identifier: BSD-3-Clause
"""A pulsed line laser seen by three cameras: a long exposure sees its
average, short ones catch a pulse or miss it."""

import numpy as np

import ernie
import ernie.demo
from ernie import tokens
from ernie.demo import side_by_side, write_png

RATE = 1000.0  # pulses per second
WIDTH = 20e-6  # s
SHORT = 50e-6  # the short exposures, s
FRAME = 1.0 / 60.0  # s
# The pulses start 40 us before each whole millisecond, so a short
# exposure ending on one holds a pulse.
DELAY = -40e-6

# The cameras: name, product, exposure, f-number and ISO (the short
# ones opened up and turned up to make up for their exposure), and
# where their exposure ends in frames from the frame's time: in step
# with the pulses, or 0.01 frame (167 us) early, between them.
VIEWS = [
    ("1/60 s: the average", "/Render/Long", FRAME, 4.0, 400.0, 0.0),
    (
        "50 us in step: a pulse",
        "/Render/InStep",
        SHORT,
        1.4,
        16000.0,
        0.0,
    ),
    (
        "50 us between pulses",
        "/Render/Between",
        SHORT,
        1.4,
        16000.0,
        -0.01,
    ),
]

RESOLUTION = (320, 200)


def red(image, x, y):
    """The red channel (0 to 1) at the fractions `x` across and `y`
    down the image."""
    rows, cols = image.shape[:2]
    return (
        image[
            min(int(y * rows), rows - 1),
            min(int(x * cols), cols - 1),
            0,
        ]
        / 255.0
    )


class Demo(ernie.demo.Demo):
    num_frames = 60
    fixed_dt = FRAME

    def __init__(self, viewer, args):
        super().__init__(viewer, args)
        scene = self.scene
        world = scene.define("/World", ernie.Xform)
        world.define(
            "Floor",
            ernie.GroundPlane,
            checker=True,
            checker_color0=(0.4,) * 3,
            checker_color1=(0.3,) * 3,
        )
        world.define(
            "Wall",
            ernie.Cube,
            size=1.0,
            display_color=(0.7,) * 3,
            translate=(0.0, 0.05, 1.0),
            scale=(4.0, 0.1, 2.0),
        )
        world.define(
            "Lamp",
            ernie.SphereLight,
            radius=0.1,
            photometric_power=2000.0,
        ).set(
            tokens.TRANSFORM,
            ernie.look_at_transform(
                (0.0, -1.5, 2.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)
            ),
        )
        # 50 mW average, fanned into a flat 40 degree line along the
        # wall.
        world.define(
            "Laser",
            ernie.LaserLight,
            wavelength=650.0,
            power=0.05,
            waist_radius=(0.001, 0.001),
            pattern="line",
            fan_angle=40.0,
            pulse_rate=RATE,
            pulse_width=WIDTH,
            pulse_delay=DELAY,
        ).set(
            tokens.TRANSFORM,
            ernie.look_at_transform(
                (0.0, -1.5, 1.0), (0.0, 0.0, 1.0)
            ),
        )
        self.cameras = []
        for c, (_, product, exposure, f_stop, iso, end) in enumerate(
            VIEWS
        ):
            camera = world.define(
                f"Camera{c}",
                ernie.Camera,
                focal_length=24.0,
                sensor_noise=False,
                exposure_time=exposure,
                exposure_f_stop=f_stop,
                exposure_iso=iso,
                shutter_open=end,
                shutter_close=end,
            ).set(
                tokens.TRANSFORM,
                ernie.look_at_transform(
                    (0.6, -3.0, 1.2), (0.0, 0.0, 1.0)
                ),
            )
            self.cameras.append(camera)
            ldr = scene.define(f"{product}/LdrColor", ernie.RenderVar)
            ldr.source_name = "LdrColor"
            scene.define(
                product,
                ernie.RenderProduct,
                camera=camera,
                resolution=RESOLUTION,
                ordered_vars=[ldr.path],
            )
        scene.define(
            "/Render/Settings",
            ernie.RenderSettings,
            camera=self.cameras[1],
            resolution=RESOLUTION,
            ambient=0.0,
        )

        def look_through(index):
            if self.viewer is not None:
                self.viewer.active_camera = self.cameras[index].path

        self.widgets.combo(
            "Camera",
            [name for name, *_ in VIEWS],
            1,
            on_change=look_through,
        )
        if viewer is not None:
            viewer.active_camera = self.cameras[1].path

    def render_at(self, time):
        """Every camera's LdrColor for the frame from `time - FRAME` to
        `time` on the sensor clock, uint8 (rows, cols, 4)."""
        self.renderer.reset_time(time - FRAME)
        products = [product for _, product, *_ in VIEWS]
        outputs = self.renderer.step(
            self.scene, render_products=products, delta_time=FRAME
        )
        return [
            np.array(
                outputs[product]
                .frames[0]
                .render_vars[f"{product}/LdrColor"]
                .map()
            )
            for product in products
        ]

    def save_output(self, path):
        """The three cameras' frames ending at 0.1 s, side by side."""
        write_png(path, side_by_side(self.render_at(0.1)))

    def test_final(self):
        images = self.render_at(0.1)
        # The cameras share a pose: the line lies where the long
        # exposure sees it, reddest down the middle column.
        rows = images[0].shape[0]
        row = (
            int(np.argmax(images[0][:, images[0].shape[1] // 2, 0]))
            + 0.5
        ) / rows
        line = [red(image, 0.5, row) for image in images]
        wall = [red(image, 0.5, 0.3) for image in images]
        for (name, *_), at, above in zip(VIEWS, line, wall):
            print(f"{name}: line {at:.3f}, wall above {above:.3f}")
        assert line[0] > wall[0] + 0.2, (
            "the long exposure should see the line's average"
        )
        assert line[1] > wall[1] + 0.2, (
            "the exposure in step should catch a pulse"
        )
        assert line[2] < wall[2] + 0.05, (
            "the exposure between pulses should see no line"
        )


if __name__ == "__main__":
    ernie.demo.main(Demo)
