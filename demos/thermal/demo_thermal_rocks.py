# SPDX-License-Identifier: BSD-3-Clause
"""A patch of lunar terrain whose every pixel has its own albedo and
thermal inertia: a diurnal table baked here by the regolith model over
both, read at each pixel's albedo map and rock abundance map. By day
the dark mare runs hotter than the bright highland; at night the rocky
ejecta of fresh craters hold the day's heat and glow against fines
cooled to about 100 K, as Diviner's night-time maps show.

References:
- J. L. Bandfield et al., "Lunar surface rock abundance and regolith
  fines temperatures derived from LRO Diviner Radiometer data", JGR
  Planets 116(E12), 2011.
- P. O. Hayne et al., "Global regolith thermophysical properties of
  the Moon from the Diviner Lunar Radiometer Experiment", JGR Planets
  122(12), 2017.
"""

import numpy as np

import ernie
import ernie.demo
from ernie import tokens
from ernie.demo import heatmap, side_by_side, write_png

HALF = 100.0  # m: the patch spans -HALF to HALF in x (east) and y (north)
GRID = 128  # quads along a side
MAP = 128  # texels along a side of the albedo and rock maps
# Craters: centre x, y (m), radius (m) and freshness (0 old, 1 fresh).
CRATERS = [
    (-40.0, 30.0, 22.0, 1.0),
    (45.0, -35.0, 16.0, 0.9),
    (30.0, 50.0, 12.0, 0.2),
    (-55.0, -50.0, 18.0, 0.1),
    (5.0, -5.0, 9.0, 0.7),
]
FINES = 50.0  # J m^-2 K^-1 s^-1/2: regolith fines' thermal inertia
ROCK = 1500.0  # bare rock's
INERTIAS = [25.0, 50.0, 100.0, 200.0, 400.0, 800.0, 1600.0]
ALBEDOS = [0.05, 0.1, 0.15, 0.25]
START_HOUR = 16.0  # local solar time: lit; the night is the slider's

RESOLUTION = (400, 300)
FOCAL_LENGTH = 24.0  # mm
APERTURE = 20.955  # mm
EYE = np.array([0.0, -230.0, 140.0])
TARGET = np.array([0.0, 0.0, -10.0])

VISIBLE_PRODUCT = "/Render/Visible"
THERMAL_PRODUCT = "/Render/Thermal"


def height(x, y):
    """The terrain's height (m): each crater a bowl with a raised rim."""
    z = np.zeros_like(x)
    for cx, cy, radius, _ in CRATERS:
        r = np.hypot(x - cx, y - cy) / radius
        inside = 0.2 * radius * (r * r - 1.0) + 0.04 * radius
        outside = 0.04 * radius / np.maximum(r, 1.0) ** 3
        z += np.where(r < 1.0, inside, outside)
    return z


def maps():
    """The albedo and thermal inertia maps, float32 (MAP, MAP), row 0 at
    the north edge: dark mare to the west, bright highland to the east,
    fresh craters' rays brighter and their ejecta rocky."""
    centres = -HALF + (np.arange(MAP) + 0.5) * 2.0 * HALF / MAP
    x, y = np.meshgrid(centres, centres[::-1])
    albedo = 0.115 + 0.035 * np.tanh(x / 25.0)
    rock = np.zeros_like(x)
    for cx, cy, radius, fresh in CRATERS:
        r = np.maximum(np.hypot(x - cx, y - cy) / radius, 0.5)
        albedo += 0.08 * fresh * np.where(r > 1.0, 1.0 / (r * r), 1.0)
        rock = np.maximum(rock, fresh * np.minimum(1.0, 1.0 / r**3))
    inertia = FINES + (ROCK - FINES) * rock
    return albedo.astype(np.float32), inertia.astype(np.float32)


def terrain():
    """The patch's points, face counts, indices and texture coordinates
    (its smooth normals the renderer's)."""
    s = np.linspace(-HALF, HALF, GRID + 1)
    x, y = np.meshgrid(s, s)
    points = np.stack([x, y, height(x, y)], -1).reshape(-1, 3)
    st = np.stack(
        [(x + HALF) / (2.0 * HALF), (y + HALF) / (2.0 * HALF)], -1
    )
    i, j = np.meshgrid(
        np.arange(GRID), np.arange(GRID), indexing="ij"
    )
    first = (i * (GRID + 1) + j).ravel()
    indices = np.stack(
        [first, first + 1, first + GRID + 2, first + GRID + 1], -1
    ).ravel()
    return (
        points,
        np.full(GRID * GRID, 4),
        indices,
        st.reshape(-1, 2).astype(np.float32),
    )


def sun_direction(hour):
    """Towards the Sun at local solar time `hour` on the equator, the
    pole north (+y): rising in the east (+x)."""
    a = (hour / 24.0 - 0.5) * 2.0 * np.pi
    return np.array([-np.sin(a), 0.0, np.cos(a)])


def project(x, y):
    """Where the cameras see the ground at (x, y), as (column, row)."""
    point = np.array([x, y, float(height(np.array(x), np.array(y)))])
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
        # The table over every inertia and albedo the maps hold.
        table = ernie.DiurnalTable.bake(
            ernie.RegolithProperties(),
            latitudes=7,
            times=48,
            inertias=INERTIAS,
            albedos=ALBEDOS,
        )
        albedo, inertia = maps()
        look = world.define("Looks", ernie.Xform).define(
            "Regolith",
            ernie.Material,
            thermal_model="diurnal",
            thermal_model_params=[0.0, 1.0, 0.0],
            thermal_model_data=table.model_data(),
        )
        look.write(tokens.THERMAL_INERTIA_TEXTURE, inertia)
        surface = look.define(
            "Surface",
            ernie.Shader,
            info_id=tokens.USD_PREVIEW_SURFACE,
            roughness=0.9,
        )
        scene.connect(look.path, tokens.SURFACE_OUTPUT, surface.path,
                      tokens.SURFACE_OUTPUT)  # fmt: skip
        albedo_map = look.define(
            "Albedo",
            ernie.Shader,
            info_id=tokens.USD_UV_TEXTURE,
            source_color_space="raw",
        )
        albedo_map.write(
            tokens.SHADER_TEXTURE, np.repeat(albedo[..., None], 3, -1)
        )
        scene.connect(surface.path, tokens.DIFFUSE_COLOR, albedo_map.path,
                      "outputs:rgb")  # fmt: skip
        points, counts, indices, st = terrain()
        world.define(
            "Terrain",
            ernie.Mesh,
            points=points,
            face_vertex_counts=counts,
            face_vertex_indices=indices,
            material_binding=look.path,
        ).write("primvars:st", st)
        self.sun = world.define(
            "Sun",
            ernie.DistantLight,
            irradiance=1361.0,
            physical_illuminant=tokens.SUN,
        )
        self.place_sun(START_HOUR)

        # Visible light by day; a microbolometer over 8 to 14 um, ironbow
        # over the range the last frame showed (80 to 400 K before one).
        self.visible = self.define_camera(
            "Visible",
            VISIBLE_PRODUCT,
            (380.0, 720.0),
            17,
            ["LdrColor"],
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
                {
                    "automatic": 1.0,
                    "low": 80.0,
                    "high": 400.0,
                    "palette": 2.0,
                },
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
            thermal_background_temperature=3.0,
        )
        self.widgets.combo(
            "View",
            ["Visible", "Thermal camera", "Temperature"],
            1,
            on_change=self.show,
        )
        self.widgets.slider(
            "Local time (h)",
            0.0,
            24.0,
            START_HOUR,
            on_change=self.place_sun,
        )
        self.show(1)

    def place_sun(self, hour):
        self.hour = hour
        self.sun.set(
            tokens.TRANSFORM,
            ernie.distant_light_transform(tuple(sun_direction(hour))),
        )

    def define_camera(
        self, name, product, spectral, bands, variables, filters=()
    ):
        camera = self.scene.define(
            f"/World/{name}",
            ernie.Camera,
            focal_length=FOCAL_LENGTH,
            horizontal_aperture=APERTURE,
            vertical_aperture=APERTURE
            * RESOLUTION[1]
            / RESOLUTION[0],
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
        self.viewer.view_var = (
            "Temperature" if index == 2 else "LdrColor"
        )

    def render_all(self):
        """The visible and thermal LdrColor, uint8 (rows, cols, 4), and
        the Temperature var in kelvin (rows, cols)."""
        outputs = self.renderer.step(
            self.scene,
            render_products=[VISIBLE_PRODUCT, THERMAL_PRODUCT],
        )

        def var(product, name):
            frame = outputs[product].frames[0]
            return np.array(
                frame.render_vars[f"{product}/{name}"].map()
            )

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
                [
                    visible[..., :3],
                    thermal[..., :3],
                    heatmap(kelvin, 80.0, 400.0),
                ]
            ),
        )

    def test_final(self):
        def kelvin_at(kelvin, x, y):
            col, row = project(x, y)
            return float(kelvin[row, col])

        # 10 pm: the fresh crater's rocky ejecta against old fines (a
        # frame first for the display's gain to take the night's range).
        self.place_sun(22.0)
        self.render_all()
        _, thermal, night = self.render_all()
        rocky = kelvin_at(night, -40.0, 30.0 - 22.0 * 1.15)
        fines = kelvin_at(night, -60.0, -85.0)
        print(f"{self.hour:.0f} h: rocky ejecta {rocky:.1f} K, "
              f"fines {fines:.1f} K")  # fmt: skip
        assert fines < 130.0, "fines should be near 100 K at night"
        assert rocky > fines + 40.0, (
            "rocks should hold the day's heat"
        )
        # The camera sees them: textured ground emits by its own albedo.
        col, row = project(-40.0, 30.0 - 22.0 * 1.15)
        glow = float(thermal[row, col, :3].mean())
        col, row = project(-60.0, -85.0)
        dark = float(thermal[row, col, :3].mean())
        print(f"thermal camera: rocks {glow:.0f}, fines {dark:.0f}")
        assert glow > dark + 40.0, (
            "the rocks should glow to the camera"
        )
        # Noon: the dark mare (west) hotter than the bright highland.
        self.place_sun(12.0)
        _, _, noon = self.render_all()
        mare = kelvin_at(noon, -80.0, -10.0)
        highland = kelvin_at(noon, 80.0, 10.0)
        print(f"noon: mare {mare:.1f} K, highland {highland:.1f} K")
        assert 360.0 < mare < 400.0, "the mare should be near 390 K"
        assert mare > highland + 4.0, "darker ground should be hotter"
        self.place_sun(START_HOUR)


if __name__ == "__main__":
    ernie.demo.main(Demo)
