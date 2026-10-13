# SPDX-License-Identifier: BSD-3-Clause
"""The whole of Mars from MGS's laser altimeter: MOLA's heights paged in
from a latitude-longitude file (four of them joined at 64 pixels a degree)
over the ellipsoid, under its dusty air; its colour plain or Viking's global
mosaic, baked a band at a time into the texture cache. The panel picks each
dataset, fetching it once into the cache; a low sun shows the relief."""

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

# The IAU's ellipsoid of Mars: equatorial and polar radii.
EQUATOR = 3396190.0
POLE = 3376200.0
# The Sun's irradiance at Mars's mean distance, 1.5237 au.
IRRADIANCE = 1361.0 / (1.5237 * 1.5237)

# MGS MOLA's Mission Experiment Gridded Data Records (PDS Geosciences Node):
# heights in metres above the areoid, big-endian, from 0 E; and USGS's
# Viking Orbiter global colour mosaic, 925 m a pixel, from 180 W.
MEGDR = (
    "https://pds-geosciences.wustl.edu/mgs/mgs-m-mola-5-megdr-l3-v1/"
    "mgsl_300x/"
)
VIKING = (
    "https://planetarymaps.usgs.gov/mosaic/"
    "Mars_Viking_ClrMosaic_global_925m.tif"
)


def megdr(folder, name):
    """MEGDR file `name`'s label and image in folder `folder`."""
    return [
        f"{MEGDR}{folder}/{name}.lbl",
        f"{MEGDR}{folder}/{name}.img",
    ]


# Each dataset: its label, its files' URLs, and the file read (the first's,
# else a `.mosaic` of its rows written beside them); none for no files.
HEIGHTS = [
    (
        "MOLA 16 ppd (33 MB)",
        megdr("meg016", "megt90n000eb"),
        None,
        [],
    ),
    (
        "MOLA 32 ppd (133 MB)",
        megdr("meg032", "megt90n000fb"),
        None,
        [],
    ),
    (
        "MOLA 64 ppd (531 MB)",
        megdr("meg064", "megt90n000gb")
        + megdr("meg064", "megt90n180gb")
        + megdr("meg064", "megt00n000gb")
        + megdr("meg064", "megt00n180gb"),
        "megt_64.mosaic",
        [
            "megt90n000gb.lbl megt90n180gb.lbl",
            "megt00n000gb.lbl megt00n180gb.lbl",
        ],
    ),
]
COLOURS = [("Plain", []), ("Viking colour (798 MB)", [VIKING])]

# Olympus Mons from 300 km up and 400 km south, the sun low in the east.
DEFAULT_VIEW = {
    "altitude": 300.0,
    "latitude": 12.0,
    "longitude": -133.8,
    "heading": 0.0,
    "tilt": 55.0,
    "sun_elevation": 10.0,
    "sun_azimuth": 90.0,
}

# Mars's material and its shaders under it.
LOOK = ernie.Path("/World/Looks/Soil")


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


def cached_path(name):
    """Where file `name` (a URL's last part) is cached."""
    return os.path.join(cache_dir(), "mars", os.path.basename(name))


def stand_in_mars():
    """A stand-in Mars for the checks, which run offline: heights in a
    big-endian PDS image as MOLA's grids hold them (a little finer than a
    face of 1024 quads, so a depth is paged), with a 25 km high shield
    volcano 300 km across at 18.65 N 133.8 W; and colour by quarter of
    longitude, west to east red, green, blue and yellow."""
    folder = os.path.join(tempfile.gettempdir(), "ernie_demo_mars")
    os.makedirs(folder, exist_ok=True)
    cols, rows = 4608, 2304
    latitude = 90.0 - (np.arange(rows) + 0.5) * 180.0 / rows
    longitude = (np.arange(cols) + 0.5) * 360.0 / cols
    phi = np.radians(latitude)[:, None]
    lam = np.radians(longitude)[None, :]
    d = np.stack(
        [
            np.cos(phi) * np.cos(lam),
            np.cos(phi) * np.sin(lam),
            np.sin(phi) * np.ones_like(lam),
        ],
        axis=-1,
    )
    summit = direction(18.65, -133.8)
    r = np.arccos(np.minimum(1.0, d @ summit)) * EQUATOR / 150000.0
    metres = 2000.0 * np.sin(2.0 * phi) + np.where(
        r < 1.0, 25000.0 * (1.0 - r), 0.0
    )
    np.round(metres).astype(">i2").tofile(
        os.path.join(folder, "stand_in.img")
    )
    heights = os.path.join(folder, "stand_in.lbl")
    with open(heights, "w") as f:
        f.write(
            'PDS_VERSION_ID = PDS3\n^IMAGE = "stand_in.img"\n'
            "OBJECT = IMAGE\n  LINES = 2304\n  LINE_SAMPLES = 4608\n"
            "  SAMPLE_TYPE = MSB_INTEGER\n  SAMPLE_BITS = 16\n"
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
        self.widgets.set(
            self.progress,
            f"{self.next + 1} of {len(self.urls)}: "
            f"{self.download.status}",
        )
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
        # A smooth Mars until its heights come.
        self.mars = scene.define(
            "/World/Mars",
            ernie.Terrain,
            heights=np.zeros((6, 65, 65), dtype=np.float32),
            planet_radii=(EQUATOR, EQUATOR, POLE),
            material_binding="/World/Looks/Soil",
        )
        # Reddish soil, albedo about Mars's mean 0.25.
        scene.define("/World/Looks/Soil", ernie.Material)
        scene.define(
            "/World/Looks/Soil/Surface",
            ernie.Shader,
            info_id=tokens.USD_PREVIEW_SURFACE,
            diffuse_color=(0.33, 0.22, 0.14),
            roughness=1.0,
        )
        scene.connect(
            LOOK,
            tokens.SURFACE_OUTPUT,
            LOOK.child("Surface"),
            tokens.SURFACE_OUTPUT,
        )
        # Carbon dioxide and dust (feldspar with hematite).
        scene.define(
            "/World/Air",
            ernie.Atmosphere,
            preset=tokens.MARS,
            anchor=tokens.ANCHOR_CENTRE,
            dust_optical_depth=0.5,
            ground_albedo=0.25,
        )
        self.sun = scene.define(
            "/World/Sun",
            ernie.DistantLight,
            irradiance=IRRADIANCE,
            physical_illuminant=tokens.SUN,
            angle=0.35,
        )
        self.camera = scene.define(
            "/World/Camera",
            ernie.Camera,
            focal_length=24.0,
            clipping_range=(10.0, math.inf),
            sensor_noise=False,
            exposure_iso=100.0,
            exposure_f_stop=8.0,
            exposure_time=1.0 / 125.0,
        )
        scene.define(
            "/Render/Settings",
            ernie.RenderSettings,
            camera=self.camera,
            resolution=(1280, 720),
            ambient=0.0,
        )

        self.altitude = DEFAULT_VIEW["altitude"]  # km
        self.latitude = DEFAULT_VIEW["latitude"]
        self.longitude = DEFAULT_VIEW["longitude"]
        self.heading = DEFAULT_VIEW["heading"]
        self.tilt = DEFAULT_VIEW["tilt"]
        self.sun_elevation = DEFAULT_VIEW["sun_elevation"]
        self.sun_azimuth = DEFAULT_VIEW["sun_azimuth"]
        self.fetches = {}
        if args.test:
            heights, colour = stand_in_mars()
            self.use_heights(heights)
            self.use_colour(colour)
        elif can_download() or os.path.exists(
            self.read_path(HEIGHTS[0])
        ):
            self.select_heights(0)
        else:
            self.report(
                "Mars's heights are not cached and cannot be fetched",
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
            on_change=lambda scale: self.mars.set(
                tokens.EXAGGERATION, scale
            ),
        )
        for label, low, high, name in [
            ("Altitude (km)", 1.0, 10000.0, "altitude"),
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
        # From 8000 km over the equator straight down: the stand-in's blue
        # quarter east of the meridian, its green one west.
        def centre(longitude):
            self.place(
                altitude=8000.0,
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

        # The volcano from 400 km south under a sun 10 degrees up in the
        # east: its western flank in shadow, the plains lit.
        self.place(**DEFAULT_VIEW)
        image = np.asarray(self.renderer.render(self.scene))
        g = image[:, :, 1].astype(np.int64)
        shadow = float(np.mean(g < 16))
        light = float(np.mean(g > 40))
        print(
            f"Volcano view: {100.0 * shadow:.1f}% in shadow, "
            f"{100.0 * light:.1f}% lit"
        )
        assert shadow > 0.005, (
            "the volcano's western flank is in shadow"
        )
        assert light > 0.4, "the plains are lit"

    @staticmethod
    def read_path(dataset):
        """The file a dataset is read from."""
        _, urls, mosaic, _ = dataset
        return cached_path(mosaic if mosaic is not None else urls[0])

    def select_heights(self, index):
        """Reads heights dataset `index`, fetching it first if need be."""
        dataset = HEIGHTS[index]
        _, urls, mosaic, rows = dataset
        path = self.read_path(dataset)

        def done():
            if mosaic is not None:
                with open(path, "w") as f:
                    f.write("".join(row + "\n" for row in rows))
            self.use_heights(path)

        self.fetch("heights", urls, "Heights download", done)

    def select_colour(self, index):
        """Colours Mars plain, or with colour dataset `index`, fetching it
        first if need be."""
        _, urls = COLOURS[index]
        if not urls:
            if "colour" in self.fetches:
                self.fetches.pop("colour").close()
            self.scene.clear(
                LOOK.child("Surface"),
                tokens.DIFFUSE_COLOR + ".connect",
            )
            return
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

    def use_heights(self, path):
        self.mars.set(tokens.HEIGHTS_FILE, str(path))
        self.mars.set(tokens.HEIGHTS_UNIT, 1.0)
        self.mars.set(tokens.HEIGHTS_REFERENCE, 0.0)

    def use_colour(self, path):
        """Mars's colour from `path`, in latitude and longitude, its
        display stretch scaled to about the soil's albedo."""
        scene = self.scene
        colour = scene.define(
            "/World/Looks/Soil/Colour",
            ernie.Shader,
            info_id=tokens.USD_UV_TEXTURE,
            source_color_space="sRGB",
            file=str(path),
        )
        colour.set(tokens.SHADER_SCALE, (0.6, 0.6, 0.6, 1.0))
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

        eye = up * (EQUATOR + self.altitude * 1e3)
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
