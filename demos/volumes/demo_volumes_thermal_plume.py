# SPDX-License-Identifier: BSD-3-Clause
"""An engine firing on a test stand through visible, 3 to 5 um and
Temperature views: its exhaust a soot and temperature field glowing by
Planck's law, its chamber and bell hot metal at their own temperatures.

References:
- M. Vollmer and K.-P. Moellmann, "Infrared Thermal Imaging", 2nd ed.,
  Wiley-VCH, 2018.
- F. S. Simmons, "Rocket Exhaust Plume Phenomenology", Aerospace Press,
  2000 (plume radiance and shock diamonds).
"""

import numpy as np

import ernie
import ernie.demo
from ernie import tokens
from ernie.demo import heatmap, side_by_side, write_png

# The exhaust's grid: 0.125 m voxels from the nozzle's exit at the origin
# of x, along +x, about the engine's axis 1.5 m up.
NX, NY, NZ = 96, 32, 32
VOXEL = 0.125
GRID_ORIGIN = np.array([0.0, -2.0, -0.5])
AXIS_HEIGHT = 1.5
AMBIENT = 295.0  # the air, the ground's too
PEAK_RISE = 1700.0  # above the air, at the nozzle
CHAMBER_KELVIN = 700.0
BELL_KELVIN = 900.0

RESOLUTION = (480, 270)
FOCAL_LENGTH = 28.0  # mm
APERTURE = 20.955  # mm
EYE = np.array([4.0, -20.0, 3.0])
TARGET = np.array([4.0, 0.0, 1.3])

VISIBLE_PRODUCT = "/Render/Visible"
INFRARED_PRODUCT = "/Render/Infrared"


def exhaust(s, radial):
    """The exhaust at points `s` metres downstream and `radial` off its
    axis: its soot (relative density) and its temperature rise in kelvin,
    which falls away downstream through bright shock diamonds."""
    width = 0.28 + 0.07 * s
    core = np.exp(-(radial * radial) / (width * width))
    fade = np.clip((11.5 - s) / 2.0, 0.0, 1.0)
    soot = core * (0.25 + 0.6 * (1.0 - np.exp(-s / 3.0))) / (1.0 + 0.1 * s) * fade
    diamonds = 1.0 + 0.35 * np.exp(-s / 2.5) * np.cos(7.0 * s)
    rise = PEAK_RISE * np.exp(-s / 4.0) * core * diamonds
    return soot, rise * fade


def pixel(point):
    """The pixel (column, row) the cameras see `point` in."""
    forward = TARGET - EYE
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    d = np.asarray(point) - EYE
    pixels = FOCAL_LENGTH / APERTURE * RESOLUTION[0]
    depth = d @ forward
    return (
        int(0.5 * RESOLUTION[0] + pixels * (d @ right) / depth),
        int(0.5 * RESOLUTION[1] - pixels * (d @ up) / depth),
    )


def brightness(image, point):
    """The mean brightness (0 to 1) of the 3 x 3 pixels about where the
    cameras see `point`."""
    col, row = pixel(point)
    window = image[row - 1 : row + 2, col - 1 : col + 2, :3]
    return float(window.astype(np.float64).mean()) / 255.0


class Demo(ernie.demo.Demo):
    num_frames = 1

    def __init__(self, viewer, args):
        super().__init__(viewer, args)
        scene = self.scene
        world = scene.define("/World", ernie.Xform)
        looks = world.define("Looks", ernie.Xform)
        concrete = looks.define(
            "Concrete", ernie.Material, reflectance_library="concrete"
        )
        steel = looks.define(
            "Steel", ernie.Material, reflectance_library="galvanized_steel"
        )
        world.define(
            "Ground", ernie.GroundPlane, material_binding=concrete.path
        )
        # The stand, and the engine on it: chamber and bell at their own
        # temperatures.
        world.define(
            "Stand",
            ernie.Cube,
            size=1.0,
            material_binding=concrete.path,
            translate=(-1.6, 0.0, 0.55),
            scale=(2.4, 1.6, 1.1),
        )
        world.define(
            "Chamber",
            ernie.Cylinder,
            axis="X",
            radius=0.3,
            height=1.2,
            material_binding=steel.path,
            thermal_temperature=CHAMBER_KELVIN,
            translate=(-1.5, 0.0, AXIS_HEIGHT),
        )
        # The bell's apex towards the chamber, its mouth at the origin.
        world.define(
            "Bell",
            ernie.Cone,
            axis="X",
            radius=0.42,
            height=0.9,
            material_binding=steel.path,
            thermal_temperature=BELL_KELVIN,
            translate=(-0.45, 0.0, AXIS_HEIGHT),
            rotate=(0.0, 0.0, 180.0),
        )

        # The exhaust: soot whose absorption glows at its temperature.
        i, j, k = np.meshgrid(
            np.arange(NX), np.arange(NY), np.arange(NZ), indexing="ij"
        )
        x = GRID_ORIGIN[0] + VOXEL * i
        y = GRID_ORIGIN[1] + VOXEL * j
        z = GRID_ORIGIN[2] + VOXEL * k
        soot, rise = exhaust(x, np.hypot(y, z - AXIS_HEIGHT))
        held = soot > 1e-4
        plume = world.define(
            "Plume", ernie.Volume, material_binding="/World/Looks/Exhaust"
        )
        for name, values in (("Density", soot), ("Temperature", rise)):
            plume.define(
                name,
                ernie.DenseField,
                voxel_size=VOXEL,
                values=np.where(held, values, 0.0).astype(np.float32),
                translate=tuple(GRID_ORIGIN),
            )
        plume.set_field("density", plume.path.child("Density"))
        plume.set_field("temperature", plume.path.child("Temperature"))
        self.exhaust = looks.define(
            "Exhaust",
            ernie.Material,
            density=4.0,
            volume_albedo=(0.3, 0.3, 0.3),
            temperature=AMBIENT,
            temperature_scale=1.0,
        )

        direction = np.array([-0.3, -0.7, 0.8])
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
        # A blue sky, scattered sunlight to none past 1.1 um.
        world.define(
            "Sky",
            ernie.DomeLight,
            spectrum=[
                380.0, 0.22, 450.0, 0.17, 500.0, 0.13,
                550.0, 0.10, 600.0, 0.075, 700.0, 0.04,
                850.0, 0.017, 1100.0, 0.0, 20000.0, 0.0,
            ],
        )  # fmt: skip

        # Visible light by the sunny 16 rule; a 3 to 5 um camera of flat
        # response exposed for the exhaust, the cool ground dark.
        self.visible = self.define_camera(
            "Visible", VISIBLE_PRODUCT, (380.0, 720.0), 17, ["LdrColor"]
        )
        self.visible.exposure_f_stop = 16.0
        self.visible.exposure_time = 0.01
        self.visible.exposure_iso = 100.0
        self.infrared = self.define_camera(
            "Infrared",
            INFRARED_PRODUCT,
            (3000.0, 5000.0),
            8,
            ["LdrColor", "Temperature"],
        )
        self.infrared.sensor_type = "mono"
        self.infrared.sensor_qe_mono = [3000.0, 0.8, 5000.0, 0.8]
        self.infrared.sensor_full_well = 1e12
        self.infrared.exposure_f_stop = 4.0
        self.infrared.exposure_time = 1e-6
        scene.define(
            "/Render/Settings",
            ernie.RenderSettings,
            camera=self.visible,
            resolution=RESOLUTION,
            ambient=0.0,
            thermal_enable=True,
            thermal_default_temperature=AMBIENT,
            thermal_background_temperature=230.0,
        )

        def rise_to(kelvin):
            self.exhaust.temperature_scale = kelvin / PEAK_RISE

        self.widgets.combo(
            "View",
            ["Visible", "3-5 um", "Temperature"],
            1,
            on_change=self.show,
        )
        self.widgets.slider(
            "Exhaust rise (K)", 200.0, 3000.0, PEAK_RISE, on_change=rise_to
        )
        self.show(1)

    def define_camera(self, name, product, spectral, bands, variables):
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
        self.scene.define(
            product,
            ernie.RenderProduct,
            camera=camera,
            resolution=RESOLUTION,
            ordered_vars=ordered,
        )
        return camera

    def show(self, index):
        if self.viewer is None:
            return
        camera = self.visible if index == 0 else self.infrared
        self.viewer.active_camera = camera.path
        self.viewer.view_var = "Temperature" if index == 2 else "LdrColor"

    def render_all(self):
        """The visible and 3 to 5 um LdrColor, uint8 (rows, cols, 4), and
        the Temperature var in kelvin (rows, cols)."""
        outputs = self.renderer.step(
            self.scene, render_products=[VISIBLE_PRODUCT, INFRARED_PRODUCT]
        )

        def var(product, name):
            frame = outputs[product].frames[0]
            return np.array(frame.render_vars[f"{product}/{name}"].map())

        return (
            var(VISIBLE_PRODUCT, "LdrColor"),
            var(INFRARED_PRODUCT, "LdrColor"),
            var(INFRARED_PRODUCT, "Temperature").reshape(
                RESOLUTION[1], RESOLUTION[0]
            ),
        )

    def save_output(self, path):
        """The visible, 3 to 5 um and false-colour Temperature images side
        by side."""
        visible, infrared, temperature = self.render_all()
        write_png(
            path,
            side_by_side(
                [
                    visible[..., :3],
                    infrared[..., :3],
                    heatmap(temperature, 250.0, 1300.0),
                ]
            ),
        )

    def test_final(self):
        visible, infrared, temperature = self.render_all()

        def kelvin_at(point):
            col, row = pixel(point)
            return float(temperature[row, col])

        # At the first shock diamond, 2 pi / 7 m out.
        nozzle = kelvin_at((0.9, 0.0, AXIS_HEIGHT))
        tail = kelvin_at((9.0, 0.0, AXIS_HEIGHT))
        ground = kelvin_at((4.0, -4.0, 0.0))
        chamber = kelvin_at((-1.5, 0.0, AXIS_HEIGHT + 0.2))
        print(
            f"Temperature: nozzle {nozzle:.0f} K, tail {tail:.0f} K, "
            f"ground {ground:.0f} K, chamber {chamber:.0f} K"
        )
        # A mean through the cooler sheath about the core, weighted by
        # what each part holds back.
        assert nozzle > 800.0, (
            "the exhaust by the nozzle should read over 800 K"
        )
        assert AMBIENT < tail < nozzle - 300.0, (
            "the exhaust should cool downstream"
        )
        assert abs(ground - AMBIENT) < 0.5, (
            "the ground should read the air's temperature"
        )
        assert abs(chamber - CHAMBER_KELVIN) < 0.5, (
            "the chamber should read its own temperature"
        )
        plume_ir = brightness(infrared, (1.0, 0.0, AXIS_HEIGHT))
        ground_ir = brightness(infrared, (4.0, -4.0, 0.0))
        plume_visible = brightness(visible, (1.0, 0.0, AXIS_HEIGHT))
        ground_visible = brightness(visible, (4.0, -4.0, 0.0))
        print(
            f"3-5 um: exhaust {plume_ir:.2f} ground {ground_ir:.2f}; "
            f"visible: exhaust {plume_visible:.2f} "
            f"ground {ground_visible:.2f}"
        )
        assert plume_ir > ground_ir + 0.3, (
            "the exhaust should outshine the ground at 3 to 5 um"
        )
        assert plume_visible > ground_visible, (
            "the exhaust should glow in the visible"
        )


if __name__ == "__main__":
    ernie.demo.main(Demo)
