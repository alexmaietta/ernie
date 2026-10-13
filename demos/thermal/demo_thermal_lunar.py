# SPDX-License-Identifier: BSD-3-Clause
"""Temperatures computed as the scene renders: a small Moon whose
surface follows a diurnal table baked by a regolith model (hot at
noon, the afternoon warmer than the morning, the night near 100 K), and
a spacecraft whose panels balance the sunlight they absorb against
what they radiate at once, with no solver running. Move the Sun and
both follow; the bare aluminium panel is the hottest yet looks the
coldest to the 8 to 14 um camera, mirroring cold space.

References:
- P. O. Hayne et al., "Global regolith thermophysical properties of
  the Moon from the Diviner Lunar Radiometer Experiment", JGR Planets
  122(12), 2017.
- D. G. Gilmore (ed.), "Spacecraft Thermal Control Handbook", 2nd ed.,
  The Aerospace Press, 2002.
"""

import numpy as np

import ernie
import ernie.demo
from ernie import tokens
from ernie.demo import heatmap, side_by_side, write_png

RADIUS = 10.0  # m: the Moon, small enough to share the frame
CRAFT = np.array([-5.0, -26.0, 4.0])  # the spacecraft's body centre
PANELS = {"White": (-2.2, "white_coating"), "Aluminium": (2.2, "aluminium")}
INTERNAL = 150.0  # W/m^2 the body's electronics give off
SPACE = 3.0  # K
SUN_ELEVATION = 0.15  # z of the Sun's direction, before normalizing
AZIMUTH = 20.0  # degrees: the Sun right of the camera, its start

RESOLUTION = (400, 300)
FOCAL_LENGTH = 24.0  # mm
APERTURE = 20.955  # mm
EYE = np.array([0.0, -45.0, 0.0])
TARGET = np.array([0.0, 0.0, 0.0])

VISIBLE_PRODUCT = "/Render/Visible"
THERMAL_PRODUCT = "/Render/Thermal"


def sun_direction(azimuth):
    """Towards the Sun at `azimuth` degrees from behind the camera,
    positive to the right: 0 is a full Moon."""
    a = np.radians(azimuth)
    direction = np.array([np.sin(a), -np.cos(a), SUN_ELEVATION])
    return direction / np.linalg.norm(direction)


def panel_points(offset, facing):
    """A 2 x 1 m panel beside the body, `offset` along x, turned to
    face `facing`."""
    centre = CRAFT + np.array([offset, 0.0, 0.0])
    u = np.cross([0.0, 0.0, 1.0], facing)
    u /= np.linalg.norm(u)
    v = np.cross(facing, u)
    return np.array(
        [centre + a * 0.75 * u + b * 0.5 * v
         for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    )  # fmt: skip


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
        int(0.5 * RESOLUTION[0] + pixels * (d @ right) / depth),
        int(0.5 * RESOLUTION[1] - pixels * (d @ up) / depth),
    )


class Demo(ernie.demo.Demo):
    num_frames = 1

    def __init__(self, viewer, args):
        super().__init__(viewer, args)
        scene = self.scene
        world = scene.define("/World", ernie.Xform)
        looks = world.define("Looks", ernie.Xform)
        # The Moon's diurnal table about its pole, the sphere's +z.
        regolith = looks.define(
            "Regolith",
            ernie.Material,
            albedo=(0.12, 0.12, 0.12),
            thermal_model="diurnal",
        )
        world.define(
            "Moon", ernie.Sphere, radius=RADIUS, material_binding=regolith.path
        )
        # The body black and warmed by its electronics, through the
        # material's thermal terminal; the panels open behind, each with
        # its finish's absorptivity and emissivity.
        body = looks.define(
            "Body", ernie.Material, reflectance_library="black_paint"
        )
        thermal = body.define(
            "Thermal",
            ernie.Shader,
            info_id=tokens.ERNIE_THERMAL,
            thermal_emissivity=0.85,
            thermal_model="balance",
            thermal_model_params=[-1.0, -1.0, SPACE, INTERNAL, 1.0],
        )
        scene.connect(body.path, tokens.THERMAL_OUTPUT, thermal.path, "outputs:out")
        world.define(
            "Body",
            ernie.Cube,
            size=1.0,
            material_binding=body.path,
            translate=tuple(CRAFT),
        )
        self.panels = {}
        for name, (offset, finish) in PANELS.items():
            look = looks.define(
                name,
                ernie.Material,
                reflectance_library=finish,
                thermal_model="balance",
                thermal_model_params=[-1.0, -1.0, SPACE, 0.0, 2.0],
            )
            self.panels[name] = world.define(
                name,
                ernie.Mesh,
                points=panel_points(offset, sun_direction(0.0)),
                face_vertex_counts=np.array([4]),
                face_vertex_indices=np.arange(4),
                material_binding=look.path,
            )
        self.sun = world.define(
            "Sun",
            ernie.DistantLight,
            irradiance=1361.0,
            physical_illuminant=tokens.SUN,
        )
        self.azimuth = 0.0
        self.place_sun(AZIMUTH)

        # Visible light for sunlit regolith; the thermal camera a
        # microbolometer over 8 to 14 um, ironbow over the range the last
        # frame showed (80 to 400 K before one), so the night side reads.
        self.visible = self.define_camera(
            "Visible", VISIBLE_PRODUCT, (380.0, 720.0), 17, ["LdrColor"]
        )
        self.visible.exposure_f_stop = 16.0
        self.visible.exposure_time = 0.004
        self.visible.exposure_iso = 100.0
        chain = []
        for name, kind, inputs in (
            ("Bolometer", "bolometer", {}),
            (
                "Display",
                "thermalDisplay",
                {"automatic": 1.0, "low": 80.0, "high": 400.0, "palette": 2.0},
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
            ["LdrColor", "Temperature"],
            chain,
        )
        scene.define(
            "/Render/Settings",
            ernie.RenderSettings,
            camera=self.thermal,
            resolution=RESOLUTION,
            ambient=0.0,
            thermal_enable=True,
            thermal_background_temperature=SPACE,
        )
        self.widgets.combo(
            "View",
            ["Visible", "Thermal camera", "Temperature"],
            1,
            on_change=self.show,
        )
        self.widgets.slider(
            "Sun azimuth (deg)", -80.0, 80.0, AZIMUTH, on_change=self.place_sun
        )
        self.show(1)

    def place_sun(self, azimuth):
        """Moves the Sun, the panels turning to face it."""
        self.azimuth = azimuth
        direction = sun_direction(azimuth)
        self.sun.set(
            tokens.TRANSFORM, ernie.distant_light_transform(tuple(direction))
        )
        for name, (offset, _) in PANELS.items():
            self.panels[name].points = panel_points(offset, direction)

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
        self.viewer.view_var = "Temperature" if index == 2 else "LdrColor"

    def render_all(self):
        """The visible and thermal LdrColor, uint8 (rows, cols, 4), and
        the Temperature var in kelvin (rows, cols)."""
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
        )

    def save_output(self, path):
        """The visible and thermal cameras' images and false-colour
        Temperature, side by side."""
        visible, thermal, kelvin = self.render_all()
        write_png(
            path,
            side_by_side(
                [visible[..., :3], thermal[..., :3], heatmap(kelvin, 80.0, 400.0)]
            ),
        )

    def test_final(self):
        visible, thermal, kelvin = self.render_all()

        def surface(longitude):
            """The Temperature var on the Moon's equator `longitude`
            degrees right of the sub-camera point."""
            a = np.radians(longitude)
            col, row = project(RADIUS * np.array([np.sin(a), -np.cos(a), 0.0]))
            return float(kelvin[row, col])

        noon = surface(self.azimuth)
        equator = kelvin[project(np.zeros(3))[1]]
        night = float(equator[equator > 0.0].min())
        print(f"sun at {self.azimuth:.0f} deg: noon {noon:.1f} K, "
              f"night side {night:.1f} K")  # fmt: skip
        assert abs(noon - 390.0) < 10.0, "the subsolar point should be near 390 K"
        assert night < 130.0, "the night side should be near 100 K"
        # The same sun 45 degrees either side of noon: afternoon (east of
        # the subsolar point, right) warmer than morning.
        afternoon = surface(self.azimuth + 45.0)
        morning = surface(self.azimuth - 45.0)
        print(f"morning {morning:.1f} K, afternoon {afternoon:.1f} K")
        assert afternoon > morning + 0.5, "the afternoon should be warmer"
        # The panels: aluminium absorbs little but radiates less still,
        # hot; white paint radiates well, cold; yet the camera sees the
        # aluminium darker, its low emissivity mirroring space.
        reads = {}
        for name, (offset, _) in PANELS.items():
            col, row = project(CRAFT + np.array([offset, 0.0, 0.0]))
            reads[name] = (float(kelvin[row, col]),
                           float(thermal[row, col, :3].mean()))  # fmt: skip
            print(f"{name} panel: {reads[name][0]:.1f} K, "
                  f"thermal {reads[name][1]:.0f}")  # fmt: skip
        assert reads["Aluminium"][0] > reads["White"][0] + 100.0, (
            "bare aluminium should run far hotter than white paint"
        )
        assert reads["Aluminium"][1] < reads["White"][1], (
            "yet it should look the colder at 8 to 14 um"
        )
        col, row = project(CRAFT)
        body = float(kelvin[row, col])
        print(f"body {body:.1f} K")
        assert reads["White"][0] < body < reads["Aluminium"][0], (
            "the body should sit between the panels"
        )


if __name__ == "__main__":
    ernie.demo.main(Demo)
