# SPDX-License-Identifier: BSD-3-Clause
"""NASA's Blue Marble on a globe: one 21600 x 10800 image of the Earth, or
its eight 21600 x 21600 tiles joined (86400 x 43200, past a GPU image's
side). Fetched once into the cache, baked a band at a time to BC7 on the GPU
off the frame (a grey checker meanwhile) and streamed as a virtual texture,
its tiles sharpening as the camera comes down."""

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

EARTH_RADIUS = 6371.0088e3

# Blue Marble Next Generation, August 2004, with topography and bathymetry
# (NASA Earth Observatory, public domain): one image, or eight tiles of 90
# degrees square, A to D west to east from 180 W, 1 the northern row.
RECORD = (
    "https://eoimages.gsfc.nasa.gov/images/imagerecords/73000/73776/"
)
IMAGE = "world.topo.bathy.200408.3x21600x10800.jpg"
TILES = ["A1", "B1", "C1", "D1", "A2", "B2", "C2", "D2"]


def tile_file(name):
    """Tile `name`'s file."""
    return f"world.topo.bathy.200408.3x21600x21600.{name}.jpg"


LOADING = ["auto", "async", "complete"]

# The globe's material and its shaders under it.
LOOK = ernie.Path("/World/Looks/Earth")


def at(latitude, longitude, radius):
    """The point at `latitude` and `longitude` (degrees, east of +x) on a
    sphere of radius `radius` about the origin, z north."""
    phi = math.radians(latitude)
    lam = math.radians(longitude)
    return (
        radius * math.cos(phi) * math.cos(lam),
        radius * math.cos(phi) * math.sin(lam),
        radius * math.sin(phi),
    )


def stand_in_earth():
    """A stand-in Earth for the checks, which run offline: eight tiles
    joined as the real ones are (a `.mosaic`), wider than a virtual
    texture's side, each quarter of longitude its own colour, west to east
    red, green, blue and yellow, darker towards the poles."""
    cols, rows = 8400, 4200
    side = cols // 4
    quarters = np.array(
        [[200, 40, 40], [40, 200, 40], [40, 40, 200], [200, 200, 40]],
        dtype=np.float64,
    )
    shade = 1.0 - 0.5 * np.abs(np.arange(rows) / rows - 0.5)
    colour = quarters[np.arange(cols) * 4 // cols]
    rgb = (shade[:, None, None] * colour[None, :, :]).astype(np.uint8)
    image = np.concatenate(
        [rgb, np.full((rows, cols, 1), 255, dtype=np.uint8)], axis=2
    )
    folder = os.path.join(
        tempfile.gettempdir(), "ernie_demo_blue_marble"
    )
    os.makedirs(folder, exist_ok=True)
    rows_of_names = []
    for row in range(2):
        names = []
        for column in range(4):
            name = f"tile{row}{column}.png"
            write_png(
                os.path.join(folder, name),
                image[
                    row * side : (row + 1) * side,
                    column * side : (column + 1) * side,
                ],
            )
            names.append(name)
        rows_of_names.append(" ".join(names))
    path = os.path.join(folder, "stand_in.mosaic")
    with open(path, "w") as f:
        f.write("\n".join(rows_of_names) + "\n")
    return path


class Fetch:
    """A dataset's files fetched one after another, with a progress bar
    while they come; `on_done` once all are there."""

    def __init__(self, widgets, files, on_done, report):
        self.widgets = widgets
        self.files = [
            (u, p) for u, p in files if not os.path.exists(p)
        ]
        self.next = 0
        self.download = None
        self.on_done = on_done
        self.report = report
        self.progress = widgets.progress("Download")

    def close(self):
        """Stops the fetch, keeping its part to resume."""
        if self.download is not None:
            self.download.cancel()
        self.widgets.remove(self.progress)

    def step(self):
        """Moves the fetch on; True once it has ended."""
        if self.download is None and self.next < len(self.files):
            self.download = AsyncDownload(*self.files[self.next])
        if self.download is None:
            self.on_done()
            return True
        for text, error in self.download.take_messages():
            self.report(text, error)
        self.widgets.set(self.progress, self.download.fraction)
        self.widgets.set(
            self.progress,
            f"{self.next + 1} of {len(self.files)}: "
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
        # The globe: a cube sphere with texture coordinates in latitude
        # and longitude, ocean blue until its image comes.
        scene.define(
            "/World/Earth",
            ernie.Sphere,
            radius=EARTH_RADIUS,
            tessellation=tokens.CUBE_SPHERE,
            tessellation_level=192,
            material_binding="/World/Looks/Earth",
        )
        scene.define("/World/Looks/Earth", ernie.Material)
        scene.define(
            "/World/Looks/Earth/Surface",
            ernie.Shader,
            info_id=tokens.USD_PREVIEW_SURFACE,
            diffuse_color=(0.02, 0.05, 0.12),
            roughness=0.8,
        )
        scene.connect(
            LOOK,
            tokens.SURFACE_OUTPUT,
            LOOK.child("Surface"),
            tokens.SURFACE_OUTPUT,
        )
        scene.define(
            "/World/Looks/Earth/St",
            ernie.Shader,
            info_id=tokens.USD_PRIMVAR_READER_FLOAT2,
            varname="st",
        )
        scene.define(
            "/World/Air",
            ernie.Atmosphere,
            preset=tokens.EARTH,
            anchor=tokens.ANCHOR_CENTRE,
        )
        sun = scene.define(
            "/World/Sun",
            ernie.DistantLight,
            irradiance=1361.0,
            physical_illuminant=tokens.SUN,
            angle=0.53,
            shadow_enable=False,
        )
        sun.set(
            tokens.TRANSFORM,
            ernie.distant_light_transform(at(10.0, 30.0, 1.0)),
        )
        self.camera = scene.define(
            "/World/Camera",
            ernie.Camera,
            focal_length=24.0,
            clipping_range=(1000.0, math.inf),
            sensor_noise=False,
            exposure_iso=100.0,
            exposure_f_stop=8.0,
            exposure_time=1.0 / 500.0,
        )
        self.settings = scene.define(
            "/Render/Settings",
            ernie.RenderSettings,
            camera=self.camera,
            resolution=(1280, 720),
            ambient=0.0,
        )

        # The image: the stand-in for the checks; else the cached Blue
        # Marble, fetched in the background the first time.
        self.altitude = 12000.0  # km
        self.latitude = 20.0
        self.longitude = 10.0
        self.fetch = None
        if args.test:
            self.use_image(stand_in_earth())
        else:
            self.select_image(0)

        widgets = self.widgets
        widgets.combo(
            "Image",
            [
                "21600 x 10800 (27 MB)",
                "8 tiles, 86400 x 43200 (420 MB)",
            ],
            0,
            on_change=self.select_image,
        )
        widgets.slider(
            "Altitude (km)",
            200.0,
            40000.0,
            self.altitude,
            on_change=lambda km: self.place(altitude=km),
        )
        widgets.slider(
            "Latitude",
            -80.0,
            80.0,
            self.latitude,
            on_change=lambda d: self.place(latitude=d),
        )
        widgets.slider(
            "Longitude",
            -180.0,
            180.0,
            self.longitude,
            on_change=lambda d: self.place(longitude=d),
        )
        widgets.combo(
            "Texture loading",
            LOADING,
            0,
            on_change=lambda i: setattr(
                self.settings, "texture_loading", LOADING[i]
            ),
        )
        # Magenta while loading; green to red by how coarse its tiles are.
        widgets.checkbox(
            "Texture loading view",
            False,
            on_change=lambda on: setattr(
                self.settings, "texture_loading_debug", on
            ),
        )
        # Back to a first load: the GPU's textures go, and with the
        # cache's copies decoding and compression start over too.
        widgets.button("Evict textures", lambda: self.evict(False))
        widgets.button(
            "Evict textures and cache", lambda: self.evict(True)
        )
        self.place()
        if viewer is not None:
            viewer.active_camera = self.camera.path
            # The Sun's gizmo would sit at the Earth's centre.
            display = viewer.viewport_display
            display.lights = False
            viewer.viewport_display = display

    def step(self):
        # The download's news in the log, and the image once fetched.
        if self.fetch is not None and self.fetch.step():
            self.fetch.close()
            self.fetch = None

    def select_image(self, index):
        """Textures the globe with image `index` (0 the single image, 1
        the eight tiles), fetching it first if need be."""
        folder = os.path.join(cache_dir(), "blue_marble")
        if index == 0:
            files = [(RECORD + IMAGE, os.path.join(folder, IMAGE))]
            image = os.path.join(folder, IMAGE)
        else:
            files = [
                (
                    RECORD + tile_file(t),
                    os.path.join(folder, tile_file(t)),
                )
                for t in TILES
            ]
            image = os.path.join(
                folder, "world.topo.bathy.200408.3x21600x21600.mosaic"
            )
        if self.fetch is not None:
            self.fetch.close()
            self.fetch = None
        if not can_download() and not all(
            os.path.exists(p) for _, p in files
        ):
            self.report(
                "Blue Marble is not cached and cannot be fetched",
                True,
            )
            return

        def done():
            # The tiles joined, two rows of four.
            if index == 1:
                with open(image, "w") as f:
                    for row in range(2):
                        names = TILES[4 * row : 4 * row + 4]
                        f.write(" ".join(tile_file(t) for t in names))
                        f.write("\n")
            self.use_image(image)

        self.fetch = Fetch(self.widgets, files, done, self.report)

    def test_final(self):
        # Over the blue and the green quarters of the stand-in, from
        # 20 000 km: the globe shows that quarter's colour.
        def centre(longitude):
            self.place(
                altitude=20000.0, latitude=0.0, longitude=longitude
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

    def use_image(self, path):
        """Textures the globe with `path`, read as sRGB colour."""
        scene = self.scene
        scene.define(
            "/World/Looks/Earth/Image",
            ernie.Shader,
            info_id=tokens.USD_UV_TEXTURE,
            source_color_space="sRGB",
            file=str(path),
        )
        scene.connect(
            LOOK.child("Image"),
            tokens.ST,
            LOOK.child("St"),
            "outputs:result",
        )
        scene.connect(
            LOOK.child("Surface"),
            tokens.DIFFUSE_COLOR,
            LOOK.child("Image"),
            "outputs:rgb",
        )

    def place(self, altitude=None, latitude=None, longitude=None):
        """The camera `altitude` km over `latitude` and `longitude`,
        looking down at the globe's centre, north up."""
        if altitude is not None:
            self.altitude = altitude
        if latitude is not None:
            self.latitude = latitude
        if longitude is not None:
            self.longitude = longitude
        eye = at(
            self.latitude,
            self.longitude,
            EARTH_RADIUS + self.altitude * 1e3,
        )
        self.camera.set(
            tokens.TRANSFORM,
            ernie.look_at_transform(
                eye, (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)
            ),
        )

    def evict(self, cache):
        """Drops the textures of the viewport's renderer, or the
        demo's own without a viewer."""
        if self.viewer is not None:
            self.viewer.evict_textures(cache)
        else:
            self.renderer.evict_textures(cache)
        self.report(
            "Textures and their cache evicted"
            if cache
            else "Textures evicted"
        )

    def report(self, text, error=False):
        """A message in the viewer's log, else printed."""
        if self.viewer is not None:
            self.viewer.log(text, "error" if error else "info")
        else:
            print(text)


if __name__ == "__main__":
    ernie.demo.main(Demo)
