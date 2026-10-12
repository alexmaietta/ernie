# SPDX-License-Identifier: BSD-3-Clause
"""A flash fired by the camera's exposure at night: in sync it lights
the frame, a late trigger loses part of it, a later one misses it; a
rolling shutter catches it in a band of rows, and a xenon flash's
jittery trigger makes frames flicker.

References:
- ON Semiconductor, "Rolling Shutter vs. Global Shutter" (AND9858),
  2018: a flash during a rolling readout lights only the rows exposing
  then.
"""

import numpy as np

import ernie
import ernie.app
import ernie.demo
from ernie import tokens
from ernie.demo import side_by_side, write_png

RESOLUTION = (240, 180)
FRAME = 1.0 / 30.0  # s
EXPOSURE = 2e-3  # s, each row's
FLASH = 1e-3  # s, the flash's width
READOUT = 15e-3  # s, a rolling shutter's top row to bottom row

# name: trigger delay (s), rolling readout (s), decay (s), jitter (s)
CASES = {
    "In sync": (0.5e-3, 0.0, 0.0, 0.0),
    "Late trigger": (1.5e-3, 0.0, 0.0, 0.0),
    "Missed": (2.5e-3, 0.0, 0.0, 0.0),
    "Rolling shutter": (8e-3, READOUT, 0.0, 0.0),
    "Xenon, jittery trigger": (0.5e-3, 0.0, 0.3e-3, 1e-3),
}


class Demo(ernie.demo.Demo):
    num_frames = 1
    fixed_dt = FRAME

    def __init__(self, viewer, args):
        super().__init__(viewer, args)
        scene = self.scene
        world = scene.define("/World", ernie.Xform)
        world.define(
            "Floor",
            ernie.Cube,
            size=1.0,
            display_color=(0.5,) * 3,
            translate=(0.0, 0.0, -0.05),
            scale=(8.0, 8.0, 0.1),
        )
        world.define(
            "Wall",
            ernie.Cube,
            size=1.0,
            display_color=(0.6,) * 3,
            translate=(0.0, 2.0, 1.5),
            scale=(8.0, 0.1, 3.0),
        )
        for name, x, colour in (
            ("Red", -0.8, (0.7, 0.1, 0.1)),
            ("Green", 0.0, (0.1, 0.6, 0.1)),
            ("Blue", 0.8, (0.1, 0.2, 0.7)),
        ):
            world.define(
                name,
                ernie.Cube,
                size=0.5,
                display_color=colour,
                translate=(x, 0.5, 0.25),
            )
        # An LED flash beside the lens: its intensity is its level
        # while it fires.
        self.flash = world.define(
            "Flash",
            ernie.SphereLight,
            radius=0.02,
            intensity=2e7,
            pulse_width=FLASH,
            pulse_trigger="exposure",
            translate=(0.1, -3.0, 1.3),
        )
        self.camera = world.define(
            "Camera",
            ernie.Camera,
            focal_length=24.0,
            sensor_noise=False,
            exposure_time=EXPOSURE,
            exposure_f_stop=2.8,
            exposure_iso=400.0,
            shutter_samples=64,
        ).set(
            tokens.TRANSFORM,
            ernie.look_at_transform(
                (0.0, -3.0, 1.2), (0.0, 1.0, 0.8), (0.0, 0.0, 1.0)
            ),
        )
        product = scene.define("/Render/Camera", ernie.RenderProduct)
        radiance = scene.define(
            "/Render/Camera/SpectralRadiance", ernie.RenderVar
        )
        radiance.source_name = "SpectralRadiance"
        product.set(
            camera=self.camera,
            resolution=RESOLUTION,
            ordered_vars=[radiance.path],
        )
        scene.define(
            "/Render/Settings",
            ernie.RenderSettings,
            camera=self.camera,
            resolution=RESOLUTION,
            ambient=0.0,
        )
        self.images = {}
        self.show(0)
        self.widgets.combo(
            "Case", list(CASES), 0, on_change=self.show
        )
        self.widgets.slider(
            "Trigger delay (ms)",
            -2.0,
            20.0,
            CASES["In sync"][0] * 1e3,
            on_change=self.delay,
        )
        self.widgets.slider(
            "Trigger jitter (ms)",
            0.0,
            3.0,
            0.0,
            on_change=self.jitter,
        )
        if viewer is not None:
            viewer.active_camera = self.camera.path
            display = viewer.viewport_display
            display.lights = False
            display.grid = False
            viewer.viewport_display = display

    def show(self, index):
        """Sets up case `index`: the trigger's delay and jitter, the
        flash's decay, and a global or rolling shutter, each row
        exposed for EXPOSURE."""
        delay, readout, decay, jitter = list(CASES.values())[index]
        self.flash.set(
            pulse_delay=delay, pulse_decay=decay, pulse_jitter=jitter
        )
        if readout > 0.0:
            # Rows read out over READOUT, each open for EXPOSURE
            # before the frame's time.
            self.camera.set(
                shutter_open=-EXPOSURE / FRAME,
                shutter_close=0.0,
                shutter_rolling=readout / FRAME,
            )
        else:
            self.camera.set(
                shutter_open=0.0,
                shutter_close=0.0,
                shutter_rolling=0.0,
            )

    def delay(self, ms):
        self.flash.set(pulse_delay=ms * 1e-3)

    def jitter(self, ms):
        self.flash.set(pulse_jitter=ms * 1e-3)

    def radiance(self, name):
        """Case `name`'s radiance over the bands, rows by columns; each
        renders once."""
        if name in self.images:
            return self.images[name]
        self.show(list(CASES).index(name))
        # A frame of the sensor clock, which the rolling readout and
        # the flash run through.
        outputs = self.renderer.step(
            self.scene,
            render_products=["/Render/Camera"],
            delta_time=FRAME,
        )
        bands = np.array(
            outputs["/Render/Camera"]
            .frames[0]
            .render_vars["/Render/Camera/SpectralRadiance"]
            .map()
        ).astype(np.float64)
        self.images[name] = bands.sum(axis=0)
        return self.images[name]

    def render_all(self):
        """Each case as uint8 grey, as bright as the frame in sync."""
        peak = np.percentile(self.radiance("In sync"), 99.0)
        return [
            np.repeat(
                (255.0 * np.clip(self.radiance(name) / peak, 0, 1))[
                    ..., None
                ],
                3,
                axis=2,
            ).astype(np.uint8)
            for name in CASES
        ]

    def save_output(self, path):
        write_png(path, side_by_side(self.render_all()))

    def test_final(self):
        sync = self.radiance("In sync").mean()
        late = self.radiance("Late trigger").mean() / sync
        missed = self.radiance("Missed").mean() / sync
        print(
            f"late trigger {late:.2f} of the light in sync (0.50), "
            f"missed {missed:.2f}"
        )
        assert abs(late - 0.5) < 0.01, (
            "half the flash after the close"
        )
        assert missed == 0.0, "the flash after the exposure"
        # The rows exposing while it fires, 8 to 9 ms after the top
        # row opens, against the frame in sync (all of the flash): at
        # least half of it in those opening 6.5 to 8.5 ms in.
        rows = self.radiance("Rolling shutter").mean(
            axis=1
        ) / np.maximum(self.radiance("In sync").mean(axis=1), 1e-30)
        lit = np.flatnonzero(rows > 0.5) / RESOLUTION[1]
        print(
            f"rolling shutter: rows {lit.min():.2f} to {lit.max():.2f} "
            "of the way down lit (0.43 to 0.57)"
        )
        assert (
            abs(lit.min() - 0.43) < 0.03
            and abs(lit.max() - 0.57) < 0.03
        )
        assert rows[: RESOLUTION[1] // 4].max() == 0.0, "the top dark"


if __name__ == "__main__":
    ernie.demo.main(Demo)
