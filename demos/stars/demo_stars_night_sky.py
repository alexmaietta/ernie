# SPDX-License-Identifier: BSD-3-Clause
"""Orion over Kitt Peak on a January night through the Earth's air: the
stars coming out at dusk, high at night, dimmed and reddened setting,
and without the air. The Bright Star Catalogue once fetched; offline,
Orion's brightest among made-up stars."""

import math
import os

import numpy as np

import ernie
import ernie.demo
from ernie import tokens
from ernie.demo import (
    AsyncDownload,
    cache_dir,
    can_download,
    write_png,
)

WIDTH, HEIGHT = 720, 480
LATITUDE, LONGITUDE = 31.96, -111.6  # Kitt Peak, degrees
DATE = (2026, 1, 16)  # the evening of the 15th, local time
ORION = (83.8, -1.2)  # right ascension and declination, degrees

BSC_URL = (
    "https://vizier.cds.unistra.fr/viz-bin/asu-tsv?-out.max=unlimited"
    "&-source=V/50/catalog&-out=_RAJ2000,_DEJ2000,Vmag,B-V"
)

# Orion's brightest stars and neighbours (J2000 degrees, V, B-V).
BRIGHT_STARS = [
    (88.7929, 7.4071, 0.42, 1.85),  # Betelgeuse
    (78.6345, -8.2016, 0.13, -0.03),  # Rigel
    (81.2828, 6.3497, 1.64, -0.22),  # Bellatrix
    (84.0534, -1.2019, 1.69, -0.18),  # Alnilam
    (85.1897, -1.9426, 1.77, -0.21),  # Alnitak
    (83.0017, -0.2991, 2.23, -0.22),  # Mintaka
    (86.9391, -9.6696, 2.09, -0.17),  # Saiph
    (83.7845, 9.9342, 3.39, -0.16),  # Meissa
    (83.8583, -5.9099, 2.77, -0.24),  # Hatysa
    (101.2872, -16.7161, -1.46, 0.00),  # Sirius
    (114.8255, 5.2250, 0.34, 0.42),  # Procyon
    (68.9802, 16.5093, 0.87, 1.54),  # Aldebaran
    (79.1723, 45.9980, 0.08, 0.80),  # Capella
    (113.6494, 31.8883, 1.58, 0.03),  # Castor
    (116.3290, 28.0262, 1.14, 1.00),  # Pollux
]

# Each view: name, hours UT, the sun's elevation, exposure seconds, ISO
# and whether the air shows.
VIEWS = [
    ("Civil dusk, Orion rising", 1.0, -4.0, 1.0 / 15.0, 100.0, True),
    ("Nautical dusk", 1.5, -10.0, 2.0, 400.0, True),
    ("Night, Orion high", 4.5, -45.0, 15.0, 1600.0, True),
    ("Orion setting", 10.5, -45.0, 15.0, 1600.0, True),
    ("Without the air", 4.5, -45.0, 15.0, 1600.0, False),
]


def offline_stars():
    """Orion's brightest among made-up stars to V 6.5."""
    made_up = ernie.synthetic_stars(9000, seed=3)
    return np.concatenate(
        [
            np.asarray(BRIGHT_STARS, dtype=np.float32),
            np.stack(
                [
                    np.asarray(made_up[key], dtype=np.float32)
                    for key in (
                        "right_ascension",
                        "declination",
                        "magnitude",
                        "color_index",
                    )
                ],
                axis=1,
            ),
        ]
    )


def read_vizier(path):
    """A VizieR table of right ascension, declination, V and B-V, rows
    without a magnitude left out."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                continue
            fields = line.split("\t")
            try:
                ra, dec, v = (float(x) for x in fields[:3])
            except ValueError:
                continue  # headers, units and blank magnitudes
            try:
                bv = float(fields[3])
            except (IndexError, ValueError):
                bv = 0.65
            rows.append((ra, dec, v, bv))
    return np.asarray(rows, dtype=np.float32)


def towards(azimuth, elevation):
    """A unit vector at `azimuth` (from north, east positive) and
    `elevation`, degrees: x east, y north, z up."""
    a, e = math.radians(azimuth), math.radians(elevation)
    return (
        math.cos(e) * math.sin(a),
        math.cos(e) * math.cos(a),
        math.sin(e),
    )


def star_peaks(image, threshold=0.02):
    """The stars seen: pixels no dimmer than their eight neighbours
    and brighter by `threshold` (linear) than any three pixels away."""
    lum = ernie.demo.srgb_to_linear(image[..., :3] / 255.0) @ [
        0.2126,
        0.7152,
        0.0722,
    ]
    rows, cols = lum.shape
    inner = lum[3:-3, 3:-3]
    peak = np.ones(inner.shape, bool)
    ring = np.zeros(inner.shape)
    for dy in range(-3, 4):
        for dx in range(-3, 4):
            shifted = lum[
                3 + dy : rows - 3 + dy, 3 + dx : cols - 3 + dx
            ]
            if max(abs(dy), abs(dx)) == 1:
                peak &= inner >= shifted
            elif max(abs(dy), abs(dx)) == 3:
                ring = np.maximum(ring, shifted)
    return np.argwhere(peak & (inner > ring + threshold)) + 3


class Demo(ernie.demo.Demo):
    num_frames = 1

    def __init__(self, viewer, args):
        super().__init__(viewer, args)
        world = self.scene.define("/World", ernie.Xform)
        ernie.demo.define_planet_surface(
            self.scene,
            "/World/Earth",
            6371.0088e3,
            (0.15, 0.13, 0.11),
            "/World/Looks/Ground",
        )
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
        self.stars = world.define("Stars", ernie.StarField)
        self.set_stars(offline_stars())
        self.camera = world.define(
            "Camera",
            ernie.Camera,
            focal_length=24.0,
            horizontal_aperture=36.0,
            vertical_aperture=24.0,
            exposure_f_stop=2.8,
            sensor_noise=False,
        )
        # A little blur and glare, so bright stars read as brighter.
        filters = []
        for i, (kind, inputs) in enumerate(
            [
                ("psf", {"blur": 0.9, "maxRadius": 4.0}),
                ("cie", {}),
                ("glare", {"strength": 0.05}),
                ("exposure", {}),
            ]
        ):
            prim = self.scene.define(
                f"/Render/Product/Chain/Step{i}",
                ernie.SensorFilter,
                filter_type=kind,
            )
            for name, value in inputs.items():
                prim.set(f"inputs:{name}", float(value))
            filters.append(ernie.Path(prim.path))
        self.scene.define(
            "/Render/Product",
            ernie.RenderProduct,
            camera=self.camera,
            resolution=(WIDTH, HEIGHT),
        ).set(tokens.FILTERS, filters)
        self.scene.define(
            "/Render/Settings",
            ernie.RenderSettings,
            camera=self.camera,
            resolution=(WIDTH, HEIGHT),
            products=[ernie.Path("/Render/Product")],
            ambient=0.0,
        )
        self.hours = VIEWS[0][1]
        self.show(0)

        self.widgets.combo(
            "View", [v[0] for v in VIEWS], 0, on_change=self.show
        )
        self.widgets.slider(
            "Hours (UT)", 0.0, 14.0, self.hours, on_change=self.sky_at
        )
        self.widgets.slider(
            "Sun elevation",
            -45.0,
            5.0,
            VIEWS[0][2],
            on_change=self.sun_at,
        )
        self.widgets.checkbox(
            "Air", True, on_change=lambda on: self.show_air(on)
        )
        if viewer is not None:
            viewer.active_camera = self.camera.path

        # The Bright Star Catalogue when cached; else fetched in the
        # background, but not for the checks, which run offline.
        self.catalogue = os.path.join(
            cache_dir(), "stars", "bsc5.tsv"
        )
        self.download = None
        if not args.test:
            if os.path.exists(self.catalogue):
                self.load()
            elif can_download():
                self.download = AsyncDownload(BSC_URL, self.catalogue)

    def set_stars(self, table):
        self.stars.right_ascension = table[:, 0].copy()
        self.stars.declination = table[:, 1].copy()
        self.stars.magnitude = table[:, 2].copy()
        self.stars.color_index = table[:, 3].copy()

    def report(self, text, error=False):
        if self.viewer is not None:
            self.viewer.log(text, "error" if error else "info")
        else:
            print(text)

    def load(self):
        try:
            table = read_vizier(self.catalogue)
            self.set_stars(table)
            self.report(
                f"Loaded {len(table)} Bright Star Catalogue stars"
            )
        except (OSError, ValueError) as error:
            self.report(str(error), True)

    def show_air(self, on):
        self.air.visibility = (
            tokens.INHERITED if on else tokens.INVISIBLE
        )

    def sun_at(self, elevation):
        self.sun.set(
            tokens.TRANSFORM,
            ernie.distant_light_transform(towards(245.0, elevation)),
        )

    def sky_at(self, hours):
        """The sky at `hours` UT, the camera following Orion but kept
        high enough to show the horizon."""
        self.hours = hours
        sky = ernie.celestial_to_local(
            LATITUDE, LONGITUDE, ernie.julian_date(*DATE, hours)
        )
        self.stars.set(tokens.TRANSFORM, sky)
        ra, dec = (math.radians(a) for a in ORION)
        d = np.asarray(sky)[:3, :3] @ [
            math.cos(dec) * math.cos(ra),
            math.cos(dec) * math.sin(ra),
            math.sin(dec),
        ]
        azimuth = math.degrees(math.atan2(d[0], d[1]))
        elevation = max(math.degrees(math.asin(d[2])), 22.0)
        eye = (0.0, 0.0, 2.0)
        self.camera.set(
            tokens.TRANSFORM,
            ernie.look_at_transform(
                eye,
                tuple(np.add(eye, towards(azimuth, elevation))),
                (0.0, 0.0, 1.0),
            ),
        )

    def show(self, index):
        self.view = index
        _, hours, sun, seconds, iso, air = VIEWS[index]
        self.camera.exposure_time = seconds
        self.camera.exposure_iso = iso
        self.show_air(air)
        self.sun_at(sun)
        self.sky_at(hours)

    def step(self):
        if self.download is not None:
            for text, error in self.download.take_messages():
                self.report(text, error)
            if self.download.done:
                if self.download.succeeded:
                    self.load()
                self.download = None

    def render_all(self):
        """Every view, uint8 RGBA; leaves the view as it was."""
        current = self.view
        images = []
        for index in range(len(VIEWS)):
            self.show(index)
            images.append(np.array(self.renderer.render(self.scene)))
        self.show(current)
        return images

    def save_output(self, path):
        """The views side by side."""
        write_png(path, ernie.demo.side_by_side(self.render_all()))

    def test_final(self):
        images = self.render_all()
        stars = [len(star_peaks(image)) for image in images]
        for (name, *_), count in zip(VIEWS, stars):
            print(f"{name}: {count} stars")
        assert stars[0] < stars[1] < stars[2], (
            "the stars come out at dusk"
        )

        # Each view's stars: summed light and colour above the sky.
        def starlight(image):
            linear = ernie.demo.srgb_to_linear(image[..., :3] / 255.0)
            sky = np.median(linear.reshape(-1, 3), axis=0)
            rows, cols = star_peaks(image).T
            return (linear[rows, cols] - sky).clip(0).sum(axis=0)

        high, low, bare = (starlight(images[i]) for i in (2, 3, 4))
        print(
            f"starlight high {high}, setting {low}, without air {bare}"
        )
        assert bare.sum() > high.sum(), "the air dims the stars"
        assert low[0] / low[2] > high[0] / high[2], (
            "stars low in the sky are redder"
        )


if __name__ == "__main__":
    ernie.demo.main(Demo)
