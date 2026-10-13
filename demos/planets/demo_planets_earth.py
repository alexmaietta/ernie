# SPDX-License-Identifier: BSD-3-Clause
"""The whole Earth: an elevation model's heights paged in over the WGS84
ellipsoid under its air, coloured by NASA's Blue Marble, one image or its
eight 21600 x 21600 tiles joined (86400 x 43200, past a GPU image's side),
baked a band at a time to BC7 on the GPU and streamed as a virtual texture.
The panel picks each dataset, fetching it once into the cache; a low sun
shows the relief."""

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

# WGS84's ellipsoid: equatorial and polar radii.
EQUATOR = 6378137.0
POLE = 6356752.314245

# NASA Earth Observatory's Blue Marble Next Generation (August 2004, with
# topography and bathymetry) and its GEBCO land relief (0 to 6400 m in 8
# bits); NOAA NCEI's ETOPO 2022 (land and sea floor in metres).
RECORD = "https://eoimages.gsfc.nasa.gov/images/imagerecords/73000/"
ETOPO = "https://www.ngdc.noaa.gov/mgg/global/relief/ETOPO2022/data/"
TILES = ["A1", "B1", "C1", "D1", "A2", "B2", "C2", "D2"]


def tile_file(name):
    """Blue Marble tile `name`'s file: A to D west to east from 180 W, 1
    the northern row."""
    return f"world.topo.bathy.200408.3x21600x21600.{name}.jpg"


# Each dataset: its label, its files' URLs, the file read (the first's,
# else a `.mosaic` of its rows written beside them), and its heights' unit.
HEIGHTS: list[tuple[str, list[str], str | None, list[str], float]] = [
    (
        "GEBCO land, 8-bit (18 MB)",
        [RECORD + "73934/gebco_08_rev_elev_21600x10800.png"],
        None,
        [],
        6400.0 / 255.0,
    ),
    (
        'ETOPO 2022 60" (466 MB)',
        [
            ETOPO + "60s/60s_surface_elev_gtif/"
            "ETOPO_2022_v1_60s_N90W180_surface.tif"
        ],
        None,
        [],
        1.0,
    ),
    (
        'ETOPO 2022 30" (1.6 GB)',
        [
            ETOPO + "30s/30s_surface_elev_gtif/"
            "ETOPO_2022_v1_30s_N90W180_surface.tif"
        ],
        None,
        [],
        1.0,
    ),
]
COLOURS: list[tuple[str, list[str], str | None, list[str], float]] = [
    (
        "Blue Marble 21600 (27 MB)",
        [RECORD + "73776/world.topo.bathy.200408.3x21600x10800.jpg"],
        None,
        [],
        1.0,
    ),
    (
        "Blue Marble 8 tiles, 86400 (420 MB)",
        [RECORD + "73776/" + tile_file(t) for t in TILES],
        "world.topo.bathy.200408.3x21600x21600.mosaic",
        [
            " ".join(tile_file(t) for t in TILES[:4]),
            " ".join(tile_file(t) for t in TILES[4:]),
        ],
        1.0,
    ),
]

# The Himalaya from 150 km up and 250 km south of Everest, the sun low in
# the east.
DEFAULT_VIEW = {
    "altitude": 150.0,
    "latitude": 25.8,
    "longitude": 87.0,
    "heading": 0.0,
    "tilt": 60.0,
    "sun_elevation": 12.0,
    "sun_azimuth": 100.0,
}

# The Earth's material and its shaders under it.
LOOK = ernie.Path("/World/Looks/Earth")


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
    """Where file `name` (a URL's last part) is cached: Blue Marble's
    beside the earlier demo's."""
    file = os.path.basename(name)
    folder = (
        "blue_marble" if file.startswith("world.topo") else "earth"
    )
    return os.path.join(cache_dir(), folder, file)


def read_path(dataset):
    """The file a dataset is read from."""
    _, urls, mosaic, _, _ = dataset
    return cached_path(mosaic if mosaic is not None else urls[0])


def stand_in_earth():
    """A stand-in Earth for the checks, which run offline: heights in a
    PDS image (a little finer than a face of 1024 quads, so a depth is
    paged) with a mountain 8 km high and 60 km across at 28 N 87 E; and
    colour as Blue Marble's tiles, eight joined (a `.mosaic`), each quarter
    of longitude its own colour, west to east red, green, blue and
    yellow."""
    folder = os.path.join(tempfile.gettempdir(), "ernie_demo_earth")
    os.makedirs(folder, exist_ok=True)
    cols, rows = 4608, 2304
    latitude = 90.0 - (np.arange(rows) + 0.5) * 180.0 / rows
    longitude = -180.0 + (np.arange(cols) + 0.5) * 360.0 / cols
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
    summit = direction(28.0, 87.0)
    r = np.arccos(np.minimum(1.0, d @ summit)) * EQUATOR / 30000.0
    metres = np.where(r < 1.0, 8000.0 * (1.0 - r), 0.0)
    np.round(metres).astype("<i2").tofile(
        os.path.join(folder, "stand_in.img")
    )
    heights = os.path.join(folder, "stand_in.lbl")
    with open(heights, "w") as f:
        f.write(
            'PDS_VERSION_ID = PDS3\n^IMAGE = "stand_in.img"\n'
            "OBJECT = IMAGE\n  LINES = 2304\n  LINE_SAMPLES = 4608\n"
            "  SAMPLE_TYPE = LSB_INTEGER\n  SAMPLE_BITS = 16\n"
            "END_OBJECT = IMAGE\nOBJECT = IMAGE_MAP_PROJECTION\n"
            '  MAP_PROJECTION_TYPE = "SIMPLE CYLINDRICAL"\n'
            "  MAP_RESOLUTION = 12.8\n  CENTER_LATITUDE = 0\n"
            "  MAXIMUM_LATITUDE = 90\n  WESTERNMOST_LONGITUDE = -180\n"
            "END_OBJECT = IMAGE_MAP_PROJECTION\nEND\n"
        )

    side = 1050
    quarters = [
        (200, 40, 40),
        (40, 200, 40),
        (40, 40, 200),
        (200, 200, 40),
    ]
    names = []
    for row in range(2):
        line = []
        for column in range(4):
            tile = np.empty((side, side, 4), dtype=np.uint8)
            tile[:, :, :3] = quarters[column]
            tile[:, :, 3] = 255
            name = f"tile{row}{column}.png"
            write_png(os.path.join(folder, name), tile)
            line.append(name)
        names.append(" ".join(line))
    mosaic = os.path.join(folder, "stand_in.mosaic")
    with open(mosaic, "w") as f:
        f.write("\n".join(names) + "\n")
    return heights, mosaic


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
        # A smooth Earth until its heights come.
        self.earth = scene.define(
            "/World/Earth",
            ernie.Terrain,
            heights=np.zeros((6, 65, 65), dtype=np.float32),
            planet_radii=(EQUATOR, EQUATOR, POLE),
            material_binding="/World/Looks/Earth",
        )
        # Ocean blue until its colour comes.
        scene.define("/World/Looks/Earth", ernie.Material)
        scene.define(
            "/World/Looks/Earth/Surface",
            ernie.Shader,
            info_id=tokens.USD_PREVIEW_SURFACE,
            diffuse_color=(0.02, 0.05, 0.12),
            roughness=1.0,
        )
        scene.connect(
            LOOK,
            tokens.SURFACE_OUTPUT,
            LOOK.child("Surface"),
            tokens.SURFACE_OUTPUT,
        )
        scene.define(
            "/World/Air",
            ernie.Atmosphere,
            preset=tokens.EARTH,
            anchor=tokens.ANCHOR_CENTRE,
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
            exposure_f_stop=8.0,
            exposure_time=1.0 / 500.0,
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
            heights, colour = stand_in_earth()
            self.use_heights(heights, 1.0)
            self.use_colour(colour)
        elif can_download() or os.path.exists(read_path(HEIGHTS[0])):
            self.select_heights(0)
            self.select_colour(0)
        else:
            self.report(
                "The Earth's data is not cached and cannot be fetched",
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
            [label for label, *_ in COLOURS],
            0,
            on_change=self.select_colour,
        )
        widgets.slider(
            "Exaggeration",
            1.0,
            10.0,
            1.0,
            on_change=lambda scale: self.earth.set(
                tokens.EXAGGERATION, scale
            ),
        )
        for label, low, high, name in [
            ("Altitude (km)", 1.0, 40000.0, "altitude"),
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
            # The Sun's gizmo would sit at the Earth's centre.
            display = viewer.viewport_display
            display.lights = False
            viewer.viewport_display = display

    def step(self):
        for kind, fetch in list(self.fetches.items()):
            if fetch.step():
                fetch.close()
                del self.fetches[kind]

    def test_final(self):
        # From 20 000 km over the equator straight down: the stand-in's
        # blue quarter east of the meridian, its green one west.
        def centre(longitude):
            self.place(
                altitude=20000.0,
                latitude=0.0,
                longitude=longitude,
                heading=0.0,
                tilt=0.0,
                sun_elevation=90.0,
                sun_azimuth=0.0,
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

        # The mountain from 40 km up and 50 km south, the sun low in the
        # east: its western slopes in shadow, darker than the plains
        # around it by more than the air's light fills in.
        self.place(
            altitude=40.0,
            latitude=27.55,
            longitude=87.0,
            heading=0.0,
            tilt=55.0,
            sun_elevation=12.0,
            sun_azimuth=100.0,
        )
        image = np.asarray(self.renderer.render(self.scene))
        brightness = np.sort(
            image[:, :, :3].astype(np.int64).sum(axis=2), axis=None
        )
        median = float(brightness[brightness.size // 2])
        darkest = float(brightness[brightness.size // 200])
        print(
            f"Mountain view: darkest {darkest:.0f}, median {median:.0f}"
        )
        assert median > 60.0, "the plains are lit"
        assert darkest < 0.7 * median, (
            "the mountain's western slopes are in shadow"
        )

    def select_heights(self, index):
        """Reads heights dataset `index`, fetching it first if need be."""
        dataset = HEIGHTS[index]
        _, urls, _, _, unit = dataset
        self.fetch(
            "heights",
            urls,
            "Heights download",
            lambda: self.use_heights(read_path(dataset), unit),
        )

    def select_colour(self, index):
        """Colours the Earth with colour dataset `index`, fetching it
        first if need be."""
        dataset = COLOURS[index]
        _, urls, mosaic, rows, _ = dataset
        path = read_path(dataset)

        def done():
            if mosaic is not None:
                with open(path, "w") as f:
                    f.write("".join(row + "\n" for row in rows))
            self.use_colour(path)

        self.fetch("colour", urls, "Colour download", done)

    def fetch(self, kind, urls, label, on_done):
        """Starts fetching `urls`, in place of the last fetch of
        `kind`."""
        if kind in self.fetches:
            self.fetches.pop(kind).close()
        self.fetches[kind] = Fetch(
            self.widgets, urls, label, on_done, self.report
        )

    def use_heights(self, path, unit):
        self.earth.set(tokens.HEIGHTS_FILE, str(path))
        self.earth.set(tokens.HEIGHTS_UNIT, unit)
        self.earth.set(tokens.HEIGHTS_REFERENCE, 0.0)

    def use_colour(self, path):
        """The Earth's colour from `path`, in latitude and longitude."""
        scene = self.scene
        scene.define(
            "/World/Looks/Earth/Colour",
            ernie.Shader,
            info_id=tokens.USD_UV_TEXTURE,
            source_color_space="sRGB",
            file=str(path),
        )
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
