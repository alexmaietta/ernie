# SPDX-License-Identifier: BSD-3-Clause
"""A Warp kernel drawing a laser projector's pattern straight into the
renderer's memory every frame, with no copy at all (Renderer.map)."""

import math
from typing import no_type_check

import numpy as np
import warp as wp

import ernie
import ernie.demo
from ernie import tokens

PROJECTOR = ernie.Path("/World/Projector")
MODES = ["fringes", "spiral", "rings"]
ROWS = 128
COLS = 256


@wp.kernel
@no_type_check
def pattern(
    image: wp.array2d[float], mode: int, time: float, period: float
):
    """Moving fringes, a turning spiral or expanding rings, `period`
    of the image's width apart."""
    y, x = wp.tid()
    rows = float(image.shape[0])
    cols = float(image.shape[1])
    u = (float(x) + 0.5) / cols * 2.0 - 1.0
    v = ((float(y) + 0.5) / rows * 2.0 - 1.0) * rows / cols
    r = wp.sqrt(u * u + v * v)
    phase = 2.0 * wp.pi * (u / period - time)
    if mode == 1:
        phase = 3.0 * wp.atan2(v, u) + 2.0 * wp.pi * (
            r / period - time
        )
    elif mode == 2:
        phase = 2.0 * wp.pi * (r / period - time)
    image[y, x] = 0.5 + 0.5 * wp.cos(phase)


def fringes(time, period):
    """The kernel's fringes on the host, to check it by."""
    u = (np.arange(COLS) + 0.5) / COLS * 2.0 - 1.0
    row = 0.5 + 0.5 * np.cos(2.0 * math.pi * (u / period - time))
    return np.tile(row, (ROWS, 1)).astype(np.float32)


class Demo(ernie.demo.Demo):
    num_frames = 30

    def __init__(self, viewer, args):
        super().__init__(viewer, args)
        wp.init()
        if not wp.is_cuda_available():
            raise RuntimeError("This demo needs CUDA interop")
        self.device = "cuda:0"
        world = self.scene.define("/World", ernie.Xform)
        world.define(
            "Floor",
            ernie.GroundPlane,
            checker=True,
            checker_color0=(0.6,) * 3,
            checker_color1=(0.5,) * 3,
        )
        world.define(
            "Wall",
            ernie.Cube,
            size=1.0,
            display_color=(0.7,) * 3,
            translate=(0.0, 1.5, 1.0),
            scale=(4.0, 0.1, 2.0),
        )
        world.define(
            "Ball",
            ernie.Sphere,
            radius=0.3,
            display_color=(0.8,) * 3,
            translate=(0.3, 0.4, 0.3),
        )
        world.define(
            "Lamp",
            ernie.SphereLight,
            radius=0.1,
            photometric_power=300.0,
        ).set(
            tokens.TRANSFORM,
            ernie.look_at_transform(
                (-1.5, -1.0, 2.5), (0.0, 0.0, 0.0)
            ),
        )
        # The pattern starts as a GPU array, which the renderer then
        # holds and lends out to write in place.
        self.initial = wp.zeros(
            (ROWS, COLS), dtype=float, device=self.device
        )
        world.define(
            "Projector",
            ernie.LaserLight,
            wavelength=532.0,
            power=0.03,
            waist_radius=(0.001, 0.001),
            pattern="image",
            pattern_image_array=self.initial,
            pattern_field_of_view=(60.0, 32.0),
        ).set(
            tokens.TRANSFORM,
            ernie.look_at_transform(
                (0.0, -1.6, 2.2), (0.0, 0.4, 0.2)
            ),
        )
        world.define(
            "Camera",
            ernie.Camera,
            focal_length=20.0,
            sensor_noise=False,
            exposure_iso=800.0,
            exposure_f_stop=2.0,
            exposure_time=1.0 / 30.0,
        ).set(
            tokens.TRANSFORM,
            ernie.look_at_transform(
                (1.6, -2.6, 1.7), (0.0, 0.4, 0.3)
            ),
        )
        self.scene.define(
            "/Render/Settings",
            ernie.RenderSettings,
            camera=ernie.Path("/World/Camera"),
            resolution=(448, 300),
            ambient=0.0,
        )
        if viewer is not None:
            viewer.active_camera = "/World/Camera"
        self.mode = 0
        self.period = 0.2
        self.drawn = None
        self.widgets.combo(
            "Pattern",
            MODES,
            0,
            on_change=lambda index: setattr(self, "mode", index),
        )
        self.widgets.slider(
            "Period",
            0.05,
            0.5,
            self.period,
            on_change=lambda value: setattr(self, "period", value),
        )

    def step(self):
        stream = wp.get_stream(self.device)
        # The renderer's own memory, written in place: no copy.
        mapped = wp.from_dlpack(
            self.map(
                PROJECTOR,
                tokens.LASER_PATTERN_IMAGE_ARRAY,
                stream=stream,
            )
        )
        wp.launch(
            pattern,
            dim=(ROWS, COLS),
            inputs=[mapped, self.mode, self.time, self.period],
            device=self.device,
            stream=stream,
        )
        self.scene.unmap(
            PROJECTOR, tokens.LASER_PATTERN_IMAGE_ARRAY, stream=stream
        )
        self.drawn = (self.mode, self.time, self.period)

    def test_final(self):
        assert self.drawn is not None
        mode, time, period = self.drawn
        assert mode == 0
        mapped = self.map(PROJECTOR, tokens.LASER_PATTERN_IMAGE_ARRAY)
        held = np.asarray(self.renderer.cuda_download(mapped))
        self.scene.unmap(PROJECTOR, tokens.LASER_PATTERN_IMAGE_ARRAY)
        np.testing.assert_allclose(
            held, fringes(time, period), atol=1e-4
        )
        image = self.render_image().astype(np.int32)
        green = image[..., 1] - np.maximum(
            image[..., 0], image[..., 2]
        )
        lit = (green > 20).mean()
        print(f"fringes light {100 * lit:.1f}% of the image")
        assert lit > 0.05, "the fringes should light the scene"


if __name__ == "__main__":
    ernie.demo.main(Demo)
