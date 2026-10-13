# SPDX-License-Identifier: BSD-3-Clause
"""Hubble's panorama of the Andromeda Galaxy on a plane: up to 40000 x 12788
pixels (half a gigapixel), fetched once into the cache, baked a band at a
time to BC7 on the GPU off the frame and streamed as a virtual texture as
the camera zooms in, only the tiles in view resident. Image: NASA, ESA, J.
Dalcanton, B. F. Williams, L. C. Johnson (University of Washington), the
PHAT team and R. Gendler (ESA/Hubble heic1502a, CC BY 4.0)."""

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

# The plane: as wide as the panorama is long, in metres, and as tall as its
# sides' ratio.
WIDTH = 32.0
HEIGHT = WIDTH * 3197.0 / 10000.0
# The camera's distance with the whole width in view (a 24 mm lens on a
# 20.955 mm wide aperture).
WHOLE = 0.5 * WIDTH * 24.0 / (0.5 * 20.955) * 1.05

# ESA/Hubble's publication TIFFs of heic1502a, 10000 and 40000 pixels wide:
# label, URL and cached file name.
PICTURES = [
    (
        "10000 x 3197 (114 MB)",
        "https://esahubble.org/media/archives/images/publicationtiff10k/"
        "heic1502a.tif",
        "heic1502a_10k.tif",
    ),
    (
        "40000 x 12788 (1.7 GB)",
        "https://esahubble.org/media/archives/images/publicationtiff40k/"
        "heic1502a.tif",
        "heic1502a_40k.tif",
    ),
]

# The plane's material and its shaders under it.
LOOK = ernie.Path("/World/Looks/Print")


def stand_in():
    """A stand-in for the checks, which run offline: 9000 x 2880 pixels
    (wider than a virtual texture's side, so it streams as the real ones
    do), red on the left half and blue on the right, a green square 32
    pixels across at three quarters of the way along."""
    cols, rows = 9000, 2880
    image = np.full((rows, cols, 4), 20, dtype=np.uint8)
    image[:, :, 3] = 255
    image[:, : cols // 2, 0] = 220
    image[:, cols // 2 :, 2] = 220
    col = np.arange(cols)[None, :]
    row = np.arange(rows)[:, None]
    square = (np.abs(col - cols * 3 // 4) < 16) & (
        np.abs(row - rows // 2) < 16
    )
    image[square] = (20, 220, 20, 255)
    path = os.path.join(
        tempfile.gettempdir(), "ernie_demo_andromeda_stand_in.png"
    )
    write_png(path, image)
    return path


class Demo(ernie.demo.Demo):
    num_frames = 3

    def __init__(self, viewer, args):
        super().__init__(viewer, args)
        scene = self.scene
        scene.define("/World", ernie.Xform)
        # The plane facing +z, texture coordinates (0, 0) to (1, 1) from
        # its lower left corner.
        x, y = 0.5 * WIDTH, 0.5 * HEIGHT
        scene.define(
            "/World/Plane",
            ernie.Mesh,
            points=np.array(
                [
                    [-x, -y, 0.0],
                    [x, -y, 0.0],
                    [x, y, 0.0],
                    [-x, y, 0.0],
                ]
            ),
            face_vertex_counts=np.array([4]),
            face_vertex_indices=np.arange(4),
            material_binding="/World/Looks/Print",
        ).write(
            "primvars:st",
            np.array(
                [[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32
            ),
        )
        scene.define("/World/Looks/Print", ernie.Material)
        scene.define(
            "/World/Looks/Print/Surface",
            ernie.Shader,
            info_id=tokens.USD_PREVIEW_SURFACE,
            diffuse_color=(0.05, 0.05, 0.05),
            roughness=1.0,
        )
        scene.connect(
            LOOK,
            tokens.SURFACE_OUTPUT,
            LOOK.child("Surface"),
            tokens.SURFACE_OUTPUT,
        )
        scene.define(
            "/World/Looks/Print/St",
            ernie.Shader,
            info_id=tokens.USD_PRIMVAR_READER_FLOAT2,
            varname="st",
        )
        # Lit straight on.
        light = scene.define(
            "/World/Light",
            ernie.DistantLight,
            irradiance=1361.0,
            shadow_enable=False,
        )
        light.set(
            tokens.TRANSFORM,
            ernie.distant_light_transform((0.0, 0.0, 1.0)),
        )
        self.camera = scene.define(
            "/World/Camera",
            ernie.Camera,
            focal_length=24.0,
            clipping_range=(1e-4, math.inf),
            sensor_noise=False,
            exposure_iso=100.0,
            exposure_f_stop=16.0,
            exposure_time=1.0 / 250.0,
        )
        scene.define(
            "/Render/Settings",
            ernie.RenderSettings,
            camera=self.camera,
            resolution=(1280, 720),
            ambient=0.0,
        )

        self.zoom = 0.0  # stops
        self.across = 0.5
        self.up = 0.5
        self.download = None
        self.downloading = ""
        self.progress = 0
        if args.test:
            self.use_image(stand_in())
        else:
            self.select_picture(0)
        widgets = self.widgets
        widgets.combo(
            "Image",
            [label for label, *_ in PICTURES],
            0,
            on_change=self.select_picture,
        )
        # Zoom in powers of two; pans as fractions of the image.
        for label, high, name in [
            ("Zoom (stops)", 10.0, "zoom"),
            ("Across", 1.0, "across"),
            ("Up", 1.0, "up"),
        ]:
            widgets.slider(
                label,
                0.0,
                high,
                getattr(self, name),
                on_change=lambda value, name=name: self.place(
                    **{name: value}
                ),
            )
        self.place()
        if viewer is not None:
            viewer.active_camera = self.camera.path
            display = viewer.viewport_display
            display.lights = False
            viewer.viewport_display = display

    def step(self):
        if self.download is None:
            return
        for text, error in self.download.take_messages():
            self.report(text, error)
        self.widgets.set(self.progress, self.download.fraction)
        self.widgets.set(self.progress, self.download.status)
        if self.download.done:
            if self.download.succeeded:
                self.use_image(self.downloading)
            self.download = None
            self.widgets.remove(self.progress)

    def test_final(self):
        def at(u):
            image = np.asarray(self.renderer.render(self.scene))
            rows, cols = image.shape[:2]
            return image[rows // 2, int(u * cols), :3].astype(
                np.float64
            )

        # The whole stand-in: red on the left, blue on the right.
        self.place(zoom=0.0, across=0.5, up=0.5)
        left = at(0.25)
        right = at(0.75)
        # Zoomed 8 stops onto the green square, 32 of 9000 pixels: only
        # its fine tiles show it.
        self.place(zoom=8.0, across=0.75)
        square = at(0.5)
        print(f"Left {left}; right {right}; zoomed {square}")
        assert left[0] > 2.0 * left[2], "the left half is red"
        assert right[2] > 2.0 * right[0], "the right half is blue"
        assert square[1] > 2.0 * max(square[0], square[2]), (
            "zoomed in, the green square fills the view"
        )

    def select_picture(self, index):
        """Shows picture `index`, fetching it first if need be."""
        _, url, name = PICTURES[index]
        path = os.path.join(cache_dir(), "andromeda", name)
        if self.download is not None:
            self.download.cancel()
            self.download = None
            self.widgets.remove(self.progress)
        if os.path.exists(path):
            self.use_image(path)
        elif can_download():
            self.downloading = path
            self.download = AsyncDownload(url, path)
            self.progress = self.widgets.progress("Download")
        else:
            self.report(
                "The panorama is not cached and cannot be fetched",
                True,
            )

    def use_image(self, path):
        """Prints the plane with `path`, read as sRGB colour."""
        scene = self.scene
        scene.define(
            "/World/Looks/Print/Image",
            ernie.Shader,
            info_id=tokens.USD_UV_TEXTURE,
            source_color_space="sRGB",
            wrap_s="clamp",
            wrap_t="clamp",
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

    def place(self, zoom=None, across=None, up=None):
        """The camera `zoom` stops in from the whole plane, over the point
        `across` and `up` of the way along and up it."""
        if zoom is not None:
            self.zoom = zoom
        if across is not None:
            self.across = across
        if up is not None:
            self.up = up
        target = (
            (self.across - 0.5) * WIDTH,
            (self.up - 0.5) * HEIGHT,
            0.0,
        )
        eye = (target[0], target[1], WHOLE / 2.0**self.zoom)
        self.camera.set(
            tokens.TRANSFORM,
            ernie.look_at_transform(eye, target, (0.0, 1.0, 0.0)),
        )

    def report(self, text, error=False):
        """A message in the viewer's log, else printed."""
        if self.viewer is not None:
            self.viewer.log(text, "error" if error else "info")
        else:
            print(text)


if __name__ == "__main__":
    ernie.demo.main(Demo)
