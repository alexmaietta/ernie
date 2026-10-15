# SPDX-License-Identifier: BSD-3-Clause
"""A gallery of the built-in sensor effects: one scene through a
camera and filter chain per effect, each image logged in the viewer and
laid out in a grid by --output."""

import math

import numpy as np

import ernie
import ernie.demo
from ernie import tokens
from ernie.demo import write_png

COLS = 384
ROWS = 240
GRID_COLUMNS = 3
GAP = 4
PITCH_MM = 0.01  # 10 um pixels
EXPOSURE_TIME = 0.00035
EYE = np.array([0.0, -5.5, 2.2])
TARGET = np.array([0.0, 1.0, 1.2])

# The cameras the effects look through; the moving ones pan between
# frames.
RIGS = [
    "Clean",
    "LowLight",
    "Short",
    "Defocus",
    "Distorted",
    "Fisheye",
    "Vignetted",
    "Chromatic",
    "Rgb",
    "Rolling",
    "Blurred",
    "Moving",
    "MovingRgb",
]
COLOUR_RIGS = {"Rgb", "Chromatic", "MovingRgb"}
MOVING_RIGS = {"Rolling", "Blurred", "Moving", "MovingRgb"}


def step(kind, inputs=None, arrays=None):
    return (kind, inputs or {}, arrays or {})


QE = step("qe")
ADC = step("adc")
CIE = step("cie")
EXPOSURE = step("exposure")


def four_bit_uppers():
    """16 codes of square-root companding from 8 bits: each code's last
    input DN."""
    uppers = [255.0] * 256
    for c in range(16):
        uppers[c] = float(
            min(math.ceil((c + 1) * (c + 1)) - 1.0, 255.0)
        )
    return uppers


def red_filter():
    """A long-pass filter's transmission, 360 to 830 nm every 10 nm:
    red light only."""
    return [
        1.0 if 360.0 + 10.0 * i >= 600.0 else 0.0 for i in range(48)
    ]


def effects():
    """(id, rig, var, steps, display gain on the 8-bit numbers)."""
    smear_time = 0.1 * EXPOSURE_TIME
    demosaic = [
        QE,
        step("bayer"),
        ADC,
        step("demosaic", {"method": 0.0}),
    ]
    sepia = {
        "m00": 0.393,
        "m01": 0.769,
        "m02": 0.189,
        "m10": 0.349,
        "m11": 0.686,
        "m12": 0.168,
        "m20": 0.272,
        "m21": 0.534,
        "m22": 0.131,
    }
    return [
        # The detector.
        ("clean", "Clean", "RawDN", [QE, ADC], 1.0),
        ("noise", "LowLight", "RawDN", [QE, step("noise"), ADC], 1.0),
        (
            "dark_current",
            "LowLight",
            "RawDN",
            [
                QE,
                step(
                    "darkCurrent",
                    {
                        "temperature": 60.0,
                        "referenceRate": 2e3,
                        "nonuniformity": 0.3,
                    },
                ),
                step("noise"),
                ADC,
            ],
            1.0,
        ),
        (
            "defect_pixels",
            "Clean",
            "RawDN",
            [
                QE,
                step(
                    "defectPixels",
                    {"dead": 3e-3, "hot": 3e-3, "hotRate": 5e6},
                ),
                ADC,
            ],
            1.0,
        ),
        (
            "radiation",
            "Clean",
            "RawDN",
            [
                QE,
                step("radiation", {"flux": 2e6, "thickness": 40.0}),
                ADC,
            ],
            1.0,
        ),
        (
            "detector_pattern",
            "Clean",
            "RawDN",
            [
                QE,
                step(
                    "detectorPattern", {"gain": 0.08, "offset": 400.0}
                ),
                ADC,
            ],
            1.0,
        ),
        (
            "readout_taps",
            "Clean",
            "RawDN",
            [
                QE,
                ADC,
                step("readoutTaps", {"offset1": 12.0, "gain1": 1.1}),
            ],
            1.0,
        ),
        (
            "smear",
            "Clean",
            "RawDN",
            [
                QE,
                step(
                    "smear",
                    {
                        "transferTime": smear_time,
                        "resetTime": 0.25 * smear_time,
                    },
                ),
                ADC,
            ],
            1.0,
        ),
        (
            "blooming",
            "Clean",
            "RawDN",
            [QE, step("blooming"), ADC],
            1.0,
        ),
        (
            "lag",
            "Moving",
            "RawDN",
            [QE, step("lag", {"fraction": 0.4}), ADC],
            1.0,
        ),
        (
            "tdi",
            "Short",
            "RawDN",
            [QE, step("tdi", {"stages": 4.0, "drift": 1.5}), ADC],
            1.0,
        ),
        # The optics.
        (
            "flare",
            "Clean",
            "RawDN",
            [QE, step("flare", {"strength": 2e-3}), ADC],
            1.0,
        ),
        (
            "gaussian_psf",
            "Clean",
            "RawDN",
            [
                QE,
                step(
                    "gaussianPsf",
                    arrays={
                        "weights": [0.8, 0.15, 0.05, 0, 0, 0, 0, 0],
                        "sigmas": [0.0, 3.0, 24.0, 0, 0, 0, 0, 0],
                    },
                ),
                ADC,
            ],
            1.0,
        ),
        (
            "airy_psf",
            "Clean",
            "RawDN",
            [step("psf", {"profile": 1.0, "blur": 0.7}), QE, ADC],
            1.0,
        ),
        (
            "defocus",
            "Defocus",
            "RawDN",
            [step("defocus"), QE, ADC],
            1.0,
        ),
        (
            "stray_light",
            "Clean",
            "RawDN",
            [step("strayLight", {"strength": 3e-3}), QE, ADC],
            1.0,
        ),
        ("distortion", "Distorted", "RawDN", [QE, ADC], 1.0),
        ("fisheye", "Fisheye", "RawDN", [QE, ADC], 1.0),
        ("vignetting", "Vignetted", "RawDN", [QE, ADC], 1.0),
        ("chromatic", "Chromatic", "LdrColor", [CIE, EXPOSURE], 1.0),
        (
            "red_filter",
            "Clean",
            "RawDN",
            [
                step(
                    "spectralTransmission",
                    arrays={"curve": red_filter()},
                ),
                QE,
                ADC,
            ],
            2.0,
        ),
        # The shutter.
        ("rolling_shutter", "Rolling", "RawDN", [QE, ADC], 1.0),
        ("motion_blur", "Blurred", "RawDN", [QE, ADC], 1.0),
        # The digital numbers.
        (
            "comms_errors",
            "Clean",
            "RawDN",
            [
                QE,
                ADC,
                step(
                    "commsErrors",
                    {
                        "packetPixels": float(COLS),
                        "packetLoss": 0.03,
                        "slips": 0.05,
                        "bitErrorRate": 2e-3,
                    },
                ),
            ],
            1.0,
        ),
        (
            "compand_sqrt",
            "Clean",
            "RawDN",
            [
                QE,
                ADC,
                step(
                    "compand", {"inputBits": 8.0, "outputBits": 6.0}
                ),
            ],
            4.0,
        ),
        (
            "compand_4bit",
            "Clean",
            "RawDN",
            [
                QE,
                ADC,
                step(
                    "compandLookup",
                    {"codes": 16.0},
                    {"uppers": four_bit_uppers()},
                ),
            ],
            17.0,
        ),
        # Colour.
        (
            "measured_qe",
            "Rgb",
            "RawDN",
            [step("measuredQe"), ADC],
            1.8,
        ),
        ("bayer_mosaic", "Rgb", "aov:mosaic", demosaic, 1.8),
        ("demosaic", "Rgb", "RawDN", demosaic, 1.8),
        (
            "white_balance",
            "Rgb",
            "LdrColor",
            [
                CIE,
                step("whiteBalance", {"red": 1.5, "blue": 0.6}),
                EXPOSURE,
            ],
            1.0,
        ),
        (
            "color_matrix",
            "Rgb",
            "LdrColor",
            [CIE, step("colorMatrix", sepia), EXPOSURE],
            1.0,
        ),
        (
            "tone_curve",
            "Rgb",
            "LdrColor",
            [
                CIE,
                EXPOSURE,
                step(
                    "toneCurve",
                    arrays={
                        "curve": [
                            0.0,
                            0.02,
                            0.08,
                            0.3,
                            0.7,
                            0.92,
                            0.98,
                            1.0,
                        ]
                    },
                ),
            ],
            1.0,
        ),
        (
            "glare",
            "Rgb",
            "LdrColor",
            [CIE, step("glare", {"strength": 0.1}), EXPOSURE],
            1.0,
        ),
        (
            "temporal_average",
            "MovingRgb",
            "LdrColor",
            [CIE, step("temporalAverage", {"alpha": 0.5}), EXPOSURE],
            1.0,
        ),
        (
            "jpeg",
            "Rgb",
            "LdrColor",
            [CIE, EXPOSURE, step("jpeg", {"quality": 8.0})],
            1.0,
        ),
    ]


def camera_of(rig):
    return f"/World/Cameras/{rig}"


def product_of(effect_id):
    return f"/Render/Effects/{effect_id}"


def var_of(effect_id, var):
    """The var's prim under its product: its source name without the
    AOV's colon."""
    return f"{product_of(effect_id)}/{var.replace(':', '_')}"


def pan_transform(step_index):
    """The moving cameras' pose on step `step_index`: panned about the
    vertical."""
    yaw = 0.14 * math.sin(0.8 * step_index)
    look = TARGET - EYE
    turned = (
        look[0] * math.cos(yaw) - look[1] * math.sin(yaw),
        look[0] * math.sin(yaw) + look[1] * math.cos(yaw),
        look[2],
    )
    return ernie.look_at_transform(
        tuple(EYE), tuple(EYE + np.array(turned)), (0.0, 0.0, 1.0)
    )


def tile(var, rig, var_name, scale):
    """An 8-bit RGBA image of a product's var: a mono sensor's numbers
    as grey, an RGB sensor's balanced to a grey world."""
    values = (
        np.asarray(var).reshape(ROWS * COLS, -1).astype(np.float64)
    )
    colour = rig in COLOUR_RIGS
    rgb = (
        values[:, :3]
        if colour
        else np.repeat(values[:, :1], 3, axis=1)
    )
    gain = np.ones(3)
    if colour and var_name != "LdrColor":
        mean = rgb.sum(axis=0)
        gain = mean[1] / np.maximum(mean, 1.0)
    pixels = np.full((ROWS * COLS, 4), 255, dtype=np.uint8)
    pixels[:, :3] = np.clip(rgb * scale * gain, 0.0, 255.0).astype(
        np.uint8
    )
    return pixels.reshape(ROWS, COLS, 4)


class Demo(ernie.demo.Demo):
    num_frames = 2  # the moving cameras' last frame and this

    def __init__(self, viewer, args):
        super().__init__(viewer, args)
        self.effects = effects()
        self.tiles = []
        self.steps = 0
        self.cameras = {}
        scene = self.scene
        world = scene.define("/World", ernie.Xform)
        world.define(
            "Ground",
            ernie.GroundPlane,
            checker=True,
            checker_size=1.0,
            checker_color0=(0.08,) * 3,
            checker_color1=(0.35,) * 3,
        )
        colours = [
            (0.8, 0.3, 0.2),
            (0.3, 0.6, 0.85),
            (0.85, 0.8, 0.3),
        ]
        for i, colour in enumerate(colours):
            world.define(
                f"Ball{i}",
                ernie.Sphere,
                radius=0.5,
                display_color=colour,
                translate=(1.4 * (i - 1), 0.5 * i, 0.5),
            )
        # A small, very bright beacon, far beyond the full well.
        glow = world.define(
            "Glow", ernie.Material, emissive_color=(2e7,) * 3
        )
        world.define(
            "Beacon",
            ernie.Sphere,
            radius=0.06,
            material_binding=glow.path,
            translate=(0.5, 3.0, 2.6),
        )
        # The Sun to the side, outside the field, for stray light.
        sun = np.array([-1.0, 0.5, 0.6])
        world.define(
            "Sun",
            ernie.DistantLight,
            irradiance=1000.0,
            physical_illuminant=tokens.SUN,
            angle=0.53,
        ).set(
            tokens.TRANSFORM,
            ernie.distant_light_transform(
                tuple(sun / np.linalg.norm(sun))
            ),
        )
        world.define("Cameras", ernie.Xform)
        for rig in RIGS:
            self.define_camera(rig)
        for effect_id, rig, var, steps, _ in self.effects:
            self.define_product(effect_id, rig, var, steps)
        scene.define(
            "/Render/Settings",
            ernie.RenderSettings,
            camera=camera_of("Rgb"),
            resolution=(COLS * 3, ROWS * 3),
            ambient=0.0,
        )
        if viewer is not None:
            viewer.active_camera = camera_of("Rgb")

    def define_camera(self, rig):
        low = rig == "LowLight"
        if low:
            seconds = EXPOSURE_TIME / 16.0
        elif rig == "Short":
            seconds = EXPOSURE_TIME / 4.0
        else:
            seconds = EXPOSURE_TIME
        colour = rig in COLOUR_RIGS
        camera = self.scene.define(
            camera_of(rig),
            ernie.Camera,
            focal_length=3.2,
            horizontal_aperture=COLS * PITCH_MM,
            vertical_aperture=ROWS * PITCH_MM,
            spectral_bands=8,
            sensor_type="rgb" if colour else "mono",
            sensor_adc_bits=8,
            sensor_noise=low,
            exposure_f_stop=8.0,
            exposure_time=seconds,
            exposure_iso=1600.0 if low else 100.0,
            exposure=1.5 if colour else 0.0,
        ).set(
            tokens.TRANSFORM,
            ernie.look_at_transform(
                tuple(EYE), tuple(TARGET), (0.0, 0.0, 1.0)
            ),
        )
        self.cameras[rig] = camera
        if rig == "Defocus":
            camera.f_stop = 1.4
            camera.focus_distance = 0.25
        elif rig == "Distorted":
            camera.lens_distortion = [-0.25, 0.05]
        elif rig == "Fisheye":
            camera.lens_projection = tokens.FISHEYE
        elif rig == "Vignetted":
            camera.lens_vignetting = True
        elif rig == "Chromatic":
            camera.lens_chromatic_scale = [
                400.0,
                1.03,
                550.0,
                1.0,
                700.0,
                0.97,
            ]
        elif rig == "Rolling":
            camera.shutter_rolling = 1.0
        elif rig == "Blurred":
            camera.shutter_open = -1.0
            camera.shutter_close = 0.0
            camera.shutter_samples = 16

    def define_product(self, effect_id, rig, var, steps):
        product = product_of(effect_id)
        self.scene.define(
            var_of(effect_id, var), ernie.RenderVar, source_name=var
        )
        filters = []
        for i, (kind, inputs, arrays) in enumerate(steps):
            prim = self.scene.define(
                f"{product}/Chain/Step{i}",
                ernie.SensorFilter,
                filter_type=kind,
            )
            for name, value in inputs.items():
                prim.set(f"inputs:{name}", float(value))
            for name, values in arrays.items():
                prim.set(f"inputs:{name}", [float(v) for v in values])
            filters.append(ernie.Path(prim.path))
        self.scene.define(
            product,
            ernie.RenderProduct,
            camera=camera_of(rig),
            resolution=(COLS, ROWS),
        ).set(
            tokens.ORDERED_VARS, [ernie.Path(var_of(effect_id, var))]
        ).set(tokens.FILTERS, filters)

    def step(self):
        """Pans the moving cameras and renders every effect, logging each
        image (new noise, hits and errors every frame)."""
        for rig in RIGS:
            if rig in MOVING_RIGS:
                self.cameras[rig].set(
                    tokens.TRANSFORM, pan_transform(self.steps)
                )
        self.steps += 1
        products = [product_of(e[0]) for e in self.effects]
        outputs = self.renderer.step(
            self.scene, render_products=products
        )
        self.tiles = []
        for effect_id, rig, var, _, scale in self.effects:
            values = (
                outputs[product_of(effect_id)]
                .frames[0]
                .render_vars[var_of(effect_id, var)]
                .map()
            )
            self.tiles.append(tile(values, rig, var, scale))
            self.log_image(f"effects/{effect_id}", self.tiles[-1])

    def grid(self):
        rows = (len(self.tiles) + GRID_COLUMNS - 1) // GRID_COLUMNS
        width = GRID_COLUMNS * COLS + (GRID_COLUMNS - 1) * GAP
        height = rows * ROWS + (rows - 1) * GAP
        image = np.zeros((height, width, 4), dtype=np.uint8)
        for t, pixels in enumerate(self.tiles):
            x0 = t % GRID_COLUMNS * (COLS + GAP)
            y0 = t // GRID_COLUMNS * (ROWS + GAP)
            image[y0 : y0 + ROWS, x0 : x0 + COLS] = pixels
        image[..., 3] = 255
        return image

    def save_output(self, path):
        """Every effect's image, three to a row in the table's order."""
        if not self.tiles:
            self.step()
        write_png(path, self.grid())

    def test_final(self):
        grey = {
            effect[0]: pixels[..., 0].astype(int)
            for effect, pixels in zip(self.effects, self.tiles)
        }
        clean = grey["clean"]
        mean = clean.mean()
        print(f"clean mean {mean:.1f} DN")
        assert 25.0 < mean < 140.0, (
            "the clean image should be well exposed"
        )
        for effect_id, image in grey.items():
            if effect_id == "clean":
                continue
            changed = int((np.abs(image - clean) > 2).sum())
            print(f"{effect_id}: {changed} pixels changed")
            assert changed > 20, (
                f"{effect_id} should change the image"
            )
        # The beacon's column: smear streaks it, above and below.
        column = int(np.argmax(clean)) % COLS
        assert (
            grey["smear"][:, column].mean()
            > clean[:, column].mean() + 5.0
        ), "smear should streak the beacon's column"

        # Vignetting darkens the bottom corners, which see the ground.
        def corners(image):
            return float(
                image[-20:, :20].sum() + image[-20:, -20:].sum()
            )

        assert corners(grey["vignetting"]) < 0.8 * corners(clean), (
            "vignetting should darken the corners"
        )
        assert len(np.unique(grey["compand_4bit"])) <= 16, (
            "4-bit companding should leave 16 codes at most"
        )
        lost = (grey["comms_errors"] == 0) & (clean > 0)
        run = longest = 0
        for value in lost.ravel():
            run = run + 1 if value else 0
            longest = max(longest, run)
        assert longest >= COLS // 2, (
            "a lost packet should blank much of a line"
        )


if __name__ == "__main__":
    ernie.demo.main(Demo)
