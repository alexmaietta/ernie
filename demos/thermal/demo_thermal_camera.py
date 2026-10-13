# SPDX-License-Identifier: BSD-3-Clause
"""Measured finishes at one temperature and black spheres from 270 to
370 K, through a visible camera, an 8 to 14 um camera and the
Temperature var: white paint glows as black paint does, while bare
aluminium mirrors the cold sky.

References:
- S. K. Meerdink et al., "The ECOSTRESS spectral library version 1.0",
  Remote Sensing of Environment 230:111196, 2019.
- M. Vollmer and K.-P. Moellmann, "Infrared Thermal Imaging", 2nd ed.,
  Wiley-VCH, 2018.
"""

import numpy as np

import ernie
import ernie.demo
from ernie import tokens
from ernie.demo import heatmap, side_by_side, write_png

# The back row, each at BLOCK_TEMPERATURE: all but bare aluminium emit
# nearly as a black body does at 8 to 14 um, weathered copper's oxide
# among them.
FINISHES = [
    "black_paint",
    "olive_paint",
    "white_coating",
    "pine_wood",
    "brick",
    "copper",
    "aluminium",
]
# The front row's black-painted spheres, in kelvin.
RAMP = [270.0, 290.0, 310.0, 330.0, 350.0, 370.0]
BLOCK_TEMPERATURE = 330.0
GROUND_TEMPERATURE = 295.0
SKY_TEMPERATURE = 230.0  # a clear sky's, 8 to 14 um

RESOLUTION = (400, 300)
FOCAL_LENGTH = 24.0  # mm
APERTURE = 20.955  # mm
EYE = np.array([0.0, -5.5, 2.6])
TARGET = np.array([0.0, 0.2, 1.2])  # the sky above the horizon

VISIBLE_PRODUCT = "/Render/Visible"
THERMAL_PRODUCT = "/Render/Thermal"


def finish_centre(i):
    return np.array([0.7 * (i - 3.0), 0.8, 0.25])


def ramp_centre(i):
    return np.array([0.7 * (i - 2.5), -0.6, 0.25])


def project(point):
    """Where the cameras see `point`, as (column, row)."""
    forward = TARGET - EYE
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    d = point - EYE
    pixels = FOCAL_LENGTH / APERTURE * RESOLUTION[0]
    depth = d @ forward
    return (
        0.5 * RESOLUTION[0] + pixels * (d @ right) / depth,
        0.5 * RESOLUTION[1] - pixels * (d @ up) / depth,
    )


def sample(images, point, kelvin):
    """Mean brightness (0 to 1) of the visible and thermal pixels
    around `point` that read `kelvin`, and their count."""
    visible, thermal, temperature = images[:3]
    col, row = (int(v) for v in project(point))
    window = np.s_[row - 4 : row + 5, col - 4 : col + 5]
    mask = np.abs(temperature[window] - kelvin) <= 0.01
    count = int(mask.sum())
    if count == 0:
        return 0.0, 0.0, 0
    return (
        float(visible[window][mask, :3].mean()) / 255.0,
        float(thermal[window][mask, :3].mean()) / 255.0,
        count,
    )


class Demo(ernie.demo.Demo):
    num_frames = 1

    def __init__(self, viewer, args):
        super().__init__(viewer, args)
        scene = self.scene
        world = scene.define("/World", ernie.Xform)
        looks = world.define("Looks", ernie.Xform)
        # Concrete, whose emissivity is its own and not the default
        # material's.
        concrete = looks.define(
            "Concrete", ernie.Material, reflectance_library="concrete"
        )
        world.define(
            "Ground",
            ernie.GroundPlane,
            checker=True,
            checker_color0=(1.0,) * 3,
            checker_color1=(0.8,) * 3,
            material_binding=concrete.path,
        )
        self.finishes = []
        for i, finish in enumerate(FINISHES):
            look = looks.define(
                finish,
                ernie.Material,
                reflectance_library=finish,
                thermal_temperature=BLOCK_TEMPERATURE,
            )
            self.finishes.append(look)
            world.define(
                f"Block{i}",
                ernie.Cube,
                size=0.5,
                material_binding=look.path,
                translate=tuple(finish_centre(i)),
            )
        for i, kelvin in enumerate(RAMP):
            look = looks.define(
                f"Sphere{i}",
                ernie.Material,
                reflectance_library="black_paint",
                thermal_temperature=kelvin,
            )
            world.define(
                f"Sphere{i}",
                ernie.Sphere,
                radius=0.25,
                material_binding=look.path,
                translate=tuple(ramp_centre(i)),
            )
        # The Sun 40 degrees up behind the camera.
        direction = np.array([-0.4, -0.8, 0.75])
        world.define(
            "Sun",
            ernie.DistantLight,
            irradiance=1000.0,
            physical_illuminant=tokens.SUN,
            angle=0.53,
        ).set(
            tokens.TRANSFORM,
            ernie.distant_light_transform(
                tuple(direction / np.linalg.norm(direction))
            ),
        )
        # A blue sky, scattered sunlight falling as 1/lambda^4 to none
        # past 1.1 um; the thermal camera sees its background
        # temperature.
        world.define(
            "Sky",
            ernie.DomeLight,
            spectrum=[
                380.0, 0.22, 450.0, 0.17, 500.0, 0.13,
                550.0, 0.10, 600.0, 0.075, 700.0, 0.04,
                850.0, 0.017, 1100.0, 0.0, 20000.0, 0.0,
            ],
        )  # fmt: skip

        # Visible light by the sunny 16 rule; the thermal camera a
        # microbolometer over 8 to 14 um, its brightness temperature in
        # the ironbow palette from 265 to 375 K.
        self.visible = self.define_camera(
            "Visible", VISIBLE_PRODUCT, (380.0, 720.0), 17, ["LdrColor"]
        )
        self.visible.exposure_f_stop = 16.0
        self.visible.exposure_time = 0.01
        self.visible.exposure_iso = 100.0
        chain = []
        for name, kind, inputs in (
            ("Bolometer", "bolometer", {}),
            (
                "Display",
                "thermalDisplay",
                {"automatic": 0.0, "low": 265.0, "high": 375.0, "palette": 2.0},
            ),
            ("Exposure", "exposure", {}),
        ):
            prim = scene.define(
                f"/Render/ThermalChain/{name}",
                ernie.SensorFilter,
                filter_type=kind,
            )
            for input_name, value in inputs.items():
                prim.set(f"inputs:{input_name}", float(value))
            chain.append(ernie.Path(prim.path))
        self.thermal = self.define_camera(
            "Thermal",
            THERMAL_PRODUCT,
            (8000.0, 14000.0),
            6,
            ["LdrColor", "Temperature", "BrightnessTemperature"],
            chain,
        )
        scene.define(
            "/Render/Settings",
            ernie.RenderSettings,
            camera=self.visible,
            resolution=RESOLUTION,
            ambient=0.0,
            thermal_enable=True,
            thermal_default_temperature=GROUND_TEMPERATURE,
            thermal_background_temperature=SKY_TEMPERATURE,
        )

        def heat(kelvin):
            for look in self.finishes:
                look.thermal_temperature = kelvin

        self.widgets.combo(
            "View",
            [
                "Visible",
                "Thermal camera",
                "Temperature",
                "Brightness temperature",
            ],
            1,
            on_change=self.show,
        )
        self.widgets.slider(
            "Blocks (K)", 250.0, 400.0, BLOCK_TEMPERATURE, on_change=heat
        )
        self.show(1)
        if viewer is not None:
            display = viewer.viewport_display
            display.lights = False
            viewer.viewport_display = display

    def define_camera(
        self, name, product, spectral, bands, variables, filters=()
    ):
        camera = self.scene.define(
            f"/World/{name}",
            ernie.Camera,
            focal_length=FOCAL_LENGTH,
            horizontal_aperture=APERTURE,
            vertical_aperture=APERTURE * RESOLUTION[1] / RESOLUTION[0],
            sensor_noise=False,
            spectral_range=spectral,
            spectral_bands=bands,
        ).set(
            tokens.TRANSFORM,
            ernie.look_at_transform(
                tuple(EYE), tuple(TARGET), (0.0, 0.0, 1.0)
            ),
        )
        ordered = []
        for var in variables:
            render_var = self.scene.define(
                f"{product}/{var}", ernie.RenderVar
            )
            render_var.source_name = var
            ordered.append(render_var.path)
        prim = self.scene.define(
            product,
            ernie.RenderProduct,
            camera=camera,
            resolution=RESOLUTION,
            ordered_vars=ordered,
        )
        if filters:
            prim.set(tokens.FILTERS, list(filters))
        return camera

    def show(self, index):
        if self.viewer is None:
            return
        camera = self.visible if index == 0 else self.thermal
        self.viewer.active_camera = camera.path
        self.viewer.view_var = {
            2: "Temperature",
            3: "BrightnessTemperature",
        }.get(index, "LdrColor")

    def render_all(self):
        """The visible and thermal LdrColor, uint8 (rows, cols, 4), and
        the Temperature and BrightnessTemperature vars in kelvin (rows,
        cols)."""
        outputs = self.renderer.step(
            self.scene, render_products=[VISIBLE_PRODUCT, THERMAL_PRODUCT]
        )

        def var(product, name):
            frame = outputs[product].frames[0]
            return np.array(frame.render_vars[f"{product}/{name}"].map())

        return (
            var(VISIBLE_PRODUCT, "LdrColor"),
            var(THERMAL_PRODUCT, "LdrColor"),
            var(THERMAL_PRODUCT, "Temperature").reshape(
                RESOLUTION[1], RESOLUTION[0]
            ),
            var(THERMAL_PRODUCT, "BrightnessTemperature").reshape(
                RESOLUTION[1], RESOLUTION[0]
            ),
        )

    def save_output(self, path):
        """The visible and thermal camera's images, and false-colour
        Temperature and BrightnessTemperature, side by side."""
        visible, thermal, temperature, brightness = self.render_all()
        write_png(
            path,
            side_by_side(
                [
                    visible[..., :3],
                    thermal[..., :3],
                    heatmap(temperature, RAMP[0], RAMP[-1]),
                    heatmap(brightness, RAMP[0], RAMP[-1]),
                ]
            ),
        )

    def test_final(self):
        images = self.render_all()
        finish = []
        for i, name in enumerate(FINISHES):
            sampled = sample(images, finish_centre(i), BLOCK_TEMPERATURE)
            print(
                f"{name}: visible {sampled[0]:.3f} "
                f"thermal {sampled[1]:.3f}"
            )
            assert sampled[2] > 0, (
                f"{name} should read its temperature in the "
                "Temperature var"
            )
            finish.append(sampled)
        black = finish[0]
        white = finish[2]
        copper = finish[5]
        aluminium = finish[6]
        assert white[0] > 3.0 * black[0], (
            "white paint should be the brighter in the visible"
        )
        assert abs(white[1] - black[1]) < 0.1 * black[1], (
            "white and black paint should glow alike at 8 to 14 um"
        )
        assert abs(copper[1] - black[1]) < 0.1 * black[1], (
            "weathered copper's oxide should glow as paint does"
        )
        assert aluminium[1] < 0.6 * black[1], (
            "bare aluminium should look cold: it mirrors the sky"
        )
        ground = sample(images, np.array([1.0, -1.0, 0.0]), GROUND_TEMPERATURE)
        assert ground[2] > 0, (
            "the ground should read the default temperature"
        )
        assert aluminium[1] < ground[1], (
            "hot aluminium should look colder than the cooler ground"
        )
        spheres = [
            sample(images, ramp_centre(i), kelvin)
            for i, kelvin in enumerate(RAMP)
        ]
        for kelvin, sphere in zip(RAMP, spheres):
            print(f"sphere at {kelvin:.0f} K: thermal {sphere[1]:.3f}")
            assert sphere[2] > 0
        assert all(
            a[1] <= b[1] for a, b in zip(spheres, spheres[1:])
        ), "the spheres should brighten with temperature"
        assert spheres[-1][1] > 1.5 * spheres[0][1], (
            "370 K should far outshine 270 K"
        )
        visible, thermal, temperature, brightness = images
        assert temperature[0, 0] == 0.0, "the sky should have no temperature"
        # What the camera reads, for an emissivity of 1: black paint
        # near its 330 K, bare aluminium far colder.
        def reads(point):
            col, row = (int(v) for v in project(point))
            return float(brightness[row, col])

        paint_reads = reads(finish_centre(0))
        aluminium_reads = reads(finish_centre(6))
        print(
            f"brightness temperature: black paint {paint_reads:.1f} K, "
            f"bare aluminium {aluminium_reads:.1f} K"
        )
        assert abs(paint_reads - BLOCK_TEMPERATURE) < 6.0, (
            "black paint should read near its temperature"
        )
        assert aluminium_reads < 300.0, (
            "bare aluminium should read far colder than it is"
        )
        assert thermal[0, :, :3].mean() / 255.0 < ground[1], (
            "the sky should be the coldest in the thermal image"
        )


if __name__ == "__main__":
    ernie.demo.main(Demo)
