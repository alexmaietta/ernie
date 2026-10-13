# SPDX-License-Identifier: BSD-3-Clause
"""The whole Moon from LRO's data: LOLA's heights paged in from a
latitude-longitude file as the view needs them, and LROC's colour mosaic
baked a band at a time into the texture cache and wrapped by latitude and
longitude. The panel picks each dataset's resolution, fetching it once into
the cache; a low sun shows the relief."""

import math
import os
import tempfile

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

MOON_RADIUS = 1737400.0

# NASA SVS's CGI Moon Kit (record 4720): LOLA heights in km and LROC WAC
# colour, both from 180 W; and LOLA's 128 pixel a degree grid from the PDS
# Geosciences Node, half metres about the Moon's radius.
KIT = "https://svs.gsfc.nasa.gov/vis/a000000/a004700/a004720/"
LOLA = (
    "https://pds-geosciences.wustl.edu/lro/lro-l-lola-3-rdr-v1/"
    "lrolol_1xxx/data/lola_gdr/cylindrical/img/"
)

# Each dataset: its label, its files' URLs (the first is the one read; a
# PDS label's image follows it), and its values' unit and reference.
HEIGHTS = [
    ("LOLA 16 ppd (66 MB)", [KIT + "ldem_16.tif"], 1000.0, 0.0),
    ("LOLA 64 ppd (1.1 GB)", [KIT + "ldem_64.tif"], 1000.0, 0.0),
    (
        "LOLA 128 ppd (2.1 GB)",
        [LOLA + "ldem_128.lbl", LOLA + "ldem_128.img"],
        1.0,
        MOON_RADIUS,
    ),
]
COLOURS = [
    ("LROC 8k (51 MB)", [KIT + "lroc_color_poles_8k.tif"]),
    ("LROC 16k (187 MB)", [KIT + "lroc_color_poles_16k.tif"]),
    ("LROC 27k (518 MB)", [KIT + "lroc_color_poles.tif"]),
]

# The Moon's material and its shaders under it.
LOOK = ernie.Path("/World/Looks/Regolith")


def direction(latitude, longitude):
    """The unit vector at `latitude` and `longitude` (degrees, east of
    +x), z north."""
    phi = math.radians(latitude)
    lam = math.radians(longitude)
    return np.array(
        [
            math.cos(phi) * math.cos(lam),
            math.cos(phi) * math.sin(lam),
            math.sin(phi),
        ]
    )


def cached_path(url):
    """Where dataset file `url` is cached."""
    return os.path.join(cache_dir(), "moon", os.path.basename(url))


def stand_in_moon():
    """A stand-in Moon for the checks, which run offline: heights in a PDS
    image as LOLA's 128 ppd grid holds them (a little finer than a face of
    1024 quads, so a depth is paged), rolling with a 40 km crater at 43 S
    11 W; and colour by quarter of longitude, west to east red, green, blue
    and yellow."""
    folder = os.path.join(tempfile.gettempdir(), "ernie_demo_moon")
    os.makedirs(folder, exist_ok=True)
    cols, rows = 4608, 2304
    latitude = np.radians(
        90.0 - (np.arange(rows) + 0.5) * 180.0 / rows
    )
    longitude = np.radians((np.arange(cols) + 0.5) * 360.0 / cols)
    phi = latitude[:, None]
    lam = longitude[None, :]
    d = np.stack(
        [
            np.cos(phi) * np.cos(lam),
            np.cos(phi) * np.sin(lam),
            np.sin(phi) * np.ones_like(lam),
        ],
        axis=-1,
    )
    # A bowl 2 km deep within a rim 500 m high.
    crater = direction(-43.3, -11.2)
    r = np.arccos(np.minimum(1.0, d @ crater)) * MOON_RADIUS / 20000.0
    bowl = np.where(
        r < 1.0,
        -2000.0 * (1.0 - r * r),
        500.0 * np.exp(-4.0 * (r - 1.0)),
    )
    metres = 1500.0 * np.sin(3.0 * phi) * np.cos(2.0 * lam) + bowl
    samples = np.round(2.0 * metres).astype("<i2")
    samples.tofile(os.path.join(folder, "stand_in.img"))
    heights = os.path.join(folder, "stand_in.lbl")
    with open(heights, "w") as f:
        f.write(
            'PDS_VERSION_ID = PDS3\n^IMAGE = "stand_in.img"\n'
            "OBJECT = IMAGE\n  LINES = 2304\n  LINE_SAMPLES = 4608\n"
            "  SAMPLE_TYPE = LSB_INTEGER\n  SAMPLE_BITS = 16\n"
            "  SCALING_FACTOR = 0.5\n  OFFSET = 1737400\n"
            "END_OBJECT = IMAGE\nOBJECT = IMAGE_MAP_PROJECTION\n"
            '  MAP_PROJECTION_TYPE = "SIMPLE CYLINDRICAL"\n'
            "  MAP_RESOLUTION = 12.8\n  CENTER_LATITUDE = 0\n"
            "  MAXIMUM_LATITUDE = 90\n  WESTERNMOST_LONGITUDE = 0\n"
            "END_OBJECT = IMAGE_MAP_PROJECTION\nEND\n"
        )

    width, height = 2048, 1024
    quarters = np.array(
        [[200, 40, 40], [40, 200, 40], [40, 40, 200], [200, 200, 40]],
        dtype=np.uint8,
    )
    rgb = np.broadcast_to(
        quarters[np.arange(width) * 4 // width][None, :, :],
        (height, width, 3),
    )
    image = np.concatenate(
        [rgb, np.full((height, width, 1), 255, dtype=np.uint8)],
        axis=2,
    )
    colour = os.path.join(folder, "stand_in.png")
    write_png(colour, image)
    return heights, colour


class Fetch:
    """A dataset's files fetched one after another into the cache, with a
    progress bar while they come; `on_done` once all are there."""

    def __init__(self, widgets, urls, label, on_done, report):
        self.widgets = widgets
        self.urls = [
            u for u in urls if not os.path.exists(cached_path(u))
        ]
        self.next = 0
        self.download = None
        self.on_done = on_done
        self.report = report
        self.progress = widgets.progress(label)

    def close(self):
        """Stops the fetch, keeping its part to resume."""
        if self.download is not None:
            self.download.cancel()
        self.widgets.remove(self.progress)

    def step(self):
        """Moves the fetch on; True once it has ended."""
        if self.download is None and self.next < len(self.urls):
            url = self.urls[self.next]
            self.download = AsyncDownload(url, cached_path(url))
        if self.download is None:
            self.on_done()
            return True
        for text, error in self.download.take_messages():
            self.report(text, error)
        self.widgets.set(self.progress, self.download.fraction)
        self.widgets.set(self.progress, self.download.status)
        if not self.download.done:
            return False
        succeeded = self.download.succeeded
        self.download = None
        self.next += 1
        return not succeeded


class Demo(ernie.demo.Demo):
    num_frames = 3

    def __init__(self, viewer, args):
        super().__init__(viewer, args)
        scene = self.scene
        scene.define("/World", ernie.Xform)
        # A smooth Moon until its heights come.
        self.moon = scene.define(
            "/World/Moon",
            ernie.Terrain,
            heights=np.zeros((6, 65, 65), dtype=np.float32),
            planet_radius=MOON_RADIUS,
            material_binding="/World/Looks/Regolith",
        )
        scene.define("/World/Looks/Regolith", ernie.Material)
        scene.define(
            "/World/Looks/Regolith/Surface",
            ernie.Shader,
            info_id=tokens.USD_PREVIEW_SURFACE,
            diffuse_color=(0.12, 0.12, 0.12),
            roughness=1.0,
        )
        scene.connect(
            LOOK,
            tokens.SURFACE_OUTPUT,
            LOOK.child("Surface"),
            tokens.SURFACE_OUTPUT,
        )
        self.sun = scene.define(
            "/World/Sun",
            ernie.DistantLight,
            irradiance=1361.0,
            physical_illuminant=tokens.SUN,
            angle=0.53,
        )
        self.camera = scene.define(
            "/World/Camera",
            ernie.Camera,
            focal_length=24.0,
            clipping_range=(10.0, math.inf),
            sensor_noise=False,
            exposure_iso=100.0,
            exposure_f_stop=11.0,
            exposure_time=1.0 / 125.0,
        )
        scene.define(
            "/Render/Settings",
            ernie.RenderSettings,
            camera=self.camera,
            resolution=(1280, 720),
            ambient=0.0,
        )

        self.altitude = 60.0  # km
        self.latitude = -40.5
        self.longitude = -11.2
        self.heading = 180.0
        self.tilt = 55.0
        self.sun_elevation = 8.0
        self.sun_azimuth = 90.0
        self.fetches = {}
        if args.test:
            heights, colour = stand_in_moon()
            self.use_heights(heights, 1.0, MOON_RADIUS)
            self.use_colour(colour)
        elif can_download() or all(
            os.path.exists(cached_path(u)) for u in HEIGHTS[0][1]
        ):
            self.select_heights(0)
            self.select_colour(0)
        else:
            self.report(
                "The Moon's data is not cached and cannot be fetched",
                True,
            )

        widgets = self.widgets
        widgets.combo(
            "Heights",
            [label for label, *_ in HEIGHTS],
            0,
            on_change=self.select_heights,
        )
        widgets.combo(
            "Colour",
            [label for label, _ in COLOURS],
            0,
            on_change=self.select_colour,
        )
        widgets.slider(
            "Exaggeration",
            1.0,
            10.0,
            1.0,
            on_change=lambda scale: self.moon.set(
                tokens.EXAGGERATION, scale
            ),
        )
        for label, low, high, name in [
            ("Altitude (km)", 1.0, 5000.0, "altitude"),
            ("Latitude", -89.0, 89.0, "latitude"),
            ("Longitude", -180.0, 180.0, "longitude"),
            ("Heading", 0.0, 360.0, "heading"),
            ("Tilt", 0.0, 85.0, "tilt"),
            ("Sun elevation", 0.0, 90.0, "sun_elevation"),
            ("Sun azimuth", 0.0, 360.0, "sun_azimuth"),
        ]:
            widgets.slider(
                label,
                low,
                high,
                getattr(self, name),
                on_change=lambda value, name=name: self.place(
                    **{name: value}
                ),
            )
        self.place()
        if viewer is not None:
            viewer.active_camera = self.camera.path

    def step(self):
        for kind, fetch in list(self.fetches.items()):
            if fetch.step():
                fetch.close()
                del self.fetches[kind]

    def test_final(self):
        # From 5000 km over the equator straight down: the stand-in's blue
        # quarter east of the meridian, its green one west.
        def centre(longitude):
            self.place(
                altitude=5000.0,
                latitude=0.0,
                longitude=longitude,
                tilt=0.0,
                sun_elevation=90.0,
            )
            image = np.asarray(self.renderer.render(self.scene))
            rows, cols = image.shape[:2]
            return image[rows // 2, cols // 2, :3].astype(np.float64)

        blue = centre(45.0)
        green = centre(-45.0)
        print(f"Centre over 45 E: {blue}; 45 W: {green}")
        assert blue[2] > blue[0] and blue[2] > blue[1], (
            "the blue quarter shows east of the meridian"
        )
        assert green[1] > green[0] and green[1] > green[2], (
            "the green quarter shows west of it"
        )

        # The crater from 60 km under a sun 8 degrees up in the east: its
        # eastern wall in shadow, the plain around it lit.
        self.place(
            altitude=60.0,
            latitude=-40.5,
            longitude=-11.2,
            heading=180.0,
            tilt=55.0,
            sun_elevation=8.0,
            sun_azimuth=90.0,
        )
        image = np.asarray(self.renderer.render(self.scene))
        g = image[:, :, 1].astype(np.int64)
        shadow = float(np.mean(g < 16))
        light = float(np.mean(g > 40))
        print(
            f"Crater view: {100.0 * shadow:.1f}% in shadow, "
            f"{100.0 * light:.1f}% lit"
        )
        assert shadow > 0.005, (
            "the crater's eastern wall is in shadow"
        )
        assert light > 0.5, "the plain around it is lit"

    def select_heights(self, index):
        """Reads heights dataset `index`, fetching it first if need be."""
        _, urls, unit, reference = HEIGHTS[index]
        self.fetch(
            "heights",
            urls,
            "Heights download",
            lambda: self.use_heights(
                cached_path(urls[0]), unit, reference
            ),
        )

    def select_colour(self, index):
        """Wraps colour dataset `index` over the Moon, fetching it first
        if need be."""
        _, urls = COLOURS[index]
        self.fetch(
            "colour",
            urls,
            "Colour download",
            lambda: self.use_colour(cached_path(urls[0])),
        )

    def fetch(self, kind, urls, label, on_done):
        """Starts fetching `urls`, in place of the last fetch of
        `kind`."""
        if kind in self.fetches:
            self.fetches.pop(kind).close()
        self.fetches[kind] = Fetch(
            self.widgets, urls, label, on_done, self.report
        )

    def use_heights(self, path, unit, reference):
        self.moon.set(tokens.HEIGHTS_FILE, str(path))
        self.moon.set(tokens.HEIGHTS_UNIT, unit)
        self.moon.set(tokens.HEIGHTS_REFERENCE, reference)

    def use_colour(self, path):
        """The Moon's colour from `path`, in latitude and longitude, its
        display stretch scaled to about the regolith's albedo."""
        scene = self.scene
        colour = scene.define(
            "/World/Looks/Regolith/Colour",
            ernie.Shader,
            info_id=tokens.USD_UV_TEXTURE,
            source_color_space="sRGB",
            file=str(path),
        )
        colour.set(tokens.SHADER_SCALE, (0.3, 0.3, 0.3, 1.0))
        scene.connect(
            LOOK.child("Surface"),
            tokens.DIFFUSE_COLOR,
            LOOK.child("Colour"),
            "outputs:rgb",
        )

    def place(self, **values):
        """The camera `altitude` km over `latitude` and `longitude`,
        facing `heading` (clockwise from north) `tilt` degrees up from
        straight down; the sun at `sun_elevation` and `sun_azimuth`
        there."""
        for name, value in values.items():
            setattr(self, name, value)
        up = direction(self.latitude, self.longitude)
        east = np.cross([0.0, 0.0, 1.0], up)
        east /= np.linalg.norm(east)
        north = np.cross(up, east)

        def towards(azimuth, elevation):
            a = math.radians(azimuth)
            e = math.radians(elevation)
            return (
                math.cos(e)
                * (math.cos(a) * north + math.sin(a) * east)
                + math.sin(e) * up
            )

        eye = up * (MOON_RADIUS + self.altitude * 1e3)
        forward = towards(self.heading, self.tilt - 90.0)
        self.camera.set(
            tokens.TRANSFORM,
            ernie.look_at_transform(
                tuple(eye), tuple(eye + forward), tuple(up)
            ),
        )
        self.sun.set(
            tokens.TRANSFORM,
            ernie.distant_light_transform(
                tuple(towards(self.sun_azimuth, self.sun_elevation))
            ),
        )

    def report(self, text, error=False):
        """A message in the viewer's log, else printed."""
        if self.viewer is not None:
            self.viewer.log(text, "error" if error else "info")
        else:
            print(text)


if __name__ == "__main__":
    ernie.demo.main(Demo)
