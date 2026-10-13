# SPDX-License-Identifier: BSD-3-Clause
"""Pixar's Point Instanced Med City, imported with Scene.import_usd: one
UsdGeomPointInstancer placing 40,000 buildings of eight prototypes over
Mediterranean hills, in centimetres (USD's fallback), under the sun. The
asset (0.5 MB from openusd.org, under Pixar's license in its
LICENSE.txt) is fetched once into the cache."""

import math
import os
import shutil
import zipfile

import numpy as np

import ernie
import ernie.demo
from ernie import tokens
from ernie.demo import (
    cache_dir,
    can_download,
    side_by_side,
    write_png,
)

URL = "https://openusd.org/files/PointInstancedMedCity.zip"
# Each view: name, eye and target, metres.
VIEWS = [
    ("Over the hills", (0.9, -2.4, 1.4), (0.9, 0.15, 0.05)),
    ("Rooftops", (0.55, -0.45, 0.42), (1.0, 0.2, 0.15)),
    ("Along the hillside", (-0.35, 0.15, 0.3), (0.9, 0.2, 0.12)),
    ("Overhead", (0.89, 0.15, 3.2), (0.89, 0.16, 0.0)),
]
CITY = ernie.Path("/World/City")
BUILDINGS = "/MediterraneanHills/Buildings"
INSTANCES = 40000


def asset_directory():
    return os.path.join(
        cache_dir(), "usd", "med_city", "PointInstancedMedCity"
    )


def fetch_asset(report):
    """Downloads and unpacks the asset into the cache (zipfile checks
    each file's CRC-32), through a directory renamed when complete."""
    root = os.path.join(cache_dir(), "usd")
    archive = os.path.join(root, "PointInstancedMedCity.zip")
    report(f"Fetching {URL}")
    ernie.demo.download(URL, archive, report)
    partial = os.path.join(root, "med_city.part")
    shutil.rmtree(partial, ignore_errors=True)
    with zipfile.ZipFile(archive) as z:
        z.extractall(partial)
    os.replace(partial, os.path.join(root, "med_city"))
    os.remove(archive)


class Demo(ernie.demo.Demo):
    num_frames = 1

    def __init__(self, viewer, args):
        super().__init__(viewer, args)
        scene = self.scene
        world = scene.define("/World", ernie.Xform)
        self.sun = world.define("Sun", ernie.DistantLight, angle=0.53)
        self.azimuth = -60.0
        self.elevation = 35.0
        self.place_sun()
        self.camera = world.define(
            "Camera",
            ernie.Camera,
            focal_length=24.0,
            clipping_range=(0.002, 20.0),
            sensor_noise=False,
            exposure_mode=tokens.AUTO,
        )
        # A little blue fill for the shaded sides.
        scene.define(
            "/Render/Settings",
            ernie.RenderSettings,
            camera=self.camera,
            resolution=(960, 600),
            ambient=0.08,
            ambient_color=(0.55, 0.7, 1.0),
        )
        self.result: ernie.UsdImport | None = None
        self.view = 0

        available, why = ernie.usd_import_available()
        if not available:
            self.report(why, True)
            return
        if not os.path.isdir(asset_directory()) and can_download():
            try:
                fetch_asset(self.report)
            except (
                OSError,
                RuntimeError,
                zipfile.BadZipFile,
            ) as error:
                self.report(str(error), True)
        if not os.path.isdir(asset_directory()):
            self.report(
                "The Med City is not cached and cannot be fetched",
                True,
            )
            return
        self.result = scene.import_usd(
            CITY,
            os.path.join(
                asset_directory(), "PointInstancedMedCity.usd"
            ),
        )
        for warning in self.result.warnings:
            self.report(warning, True)
        self.report(
            f"Imported the Med City: {len(self.result.prims)} prims"
        )
        self.show(0)

        self.widgets.combo(
            "View",
            [name for name, *_ in VIEWS],
            0,
            on_change=self.show,
        )
        self.widgets.slider(
            "Sun azimuth (deg)",
            -180.0,
            180.0,
            self.azimuth,
            on_change=self.turn_sun,
        )
        self.widgets.slider(
            "Sun elevation (deg)",
            2.0,
            85.0,
            self.elevation,
            on_change=self.raise_sun,
        )
        if viewer is not None:
            viewer.active_camera = self.camera.path

    def report(self, text, error=False):
        if self.viewer is not None:
            self.viewer.log(text, "error" if error else "info")
        else:
            print(text)

    def place_sun(self):
        """Points the sun from its azimuth and elevation."""
        a = math.radians(self.azimuth)
        e = math.radians(self.elevation)
        toward = (
            math.cos(e) * math.cos(a),
            math.cos(e) * math.sin(a),
            math.sin(e),
        )
        self.sun.set(
            tokens.TRANSFORM,
            ernie.look_at_transform(
                toward, (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)
            ),
        )

    def turn_sun(self, degrees):
        self.azimuth = degrees
        self.place_sun()

    def raise_sun(self, degrees):
        self.elevation = degrees
        self.place_sun()

    def show(self, index):
        self.view = index
        _, eye, target = VIEWS[index]
        self.camera.set(
            tokens.TRANSFORM,
            ernie.look_at_transform(eye, target, (0.0, 0.0, 1.0)),
        )

    def render_all(self):
        """The scene from each view, RGBA uint8; leaves the view as it
        was."""
        current = self.view
        images = []
        for index in range(len(VIEWS)):
            self.show(index)
            images.append(np.array(self.renderer.render(self.scene)))
        self.show(current)
        return images

    def save_output(self, path):
        """The views side by side."""
        write_png(path, side_by_side(self.render_all()))

    def test_final(self):
        if self.result is None:
            print("Skipped: no USD importer or no Med City")
            return
        # One instancer of every building, its prototypes shared.
        buildings = self.result.prims[BUILDINGS]
        assert self.scene.type_of(buildings) == "PointInstancer"
        positions = np.asarray(
            self.scene.read(buildings, "positions")
        ).reshape(-1, 3)
        indices = np.asarray(
            self.scene.read(buildings, "protoIndices")
        )
        meshes = sum(
            1
            for path in self.result.prims.values()
            if self.scene.has(path)
            and self.scene.type_of(path) == "Mesh"
        )
        print(
            f"{len(positions)} instances of {indices.max() + 1} prototypes, "
            f"{meshes} meshes"
        )
        assert len(positions) == INSTANCES, (
            "every building should be placed"
        )
        assert meshes == indices.max() + 1, (
            "prototypes should be shared"
        )
        # Centimetres into metres: the hills about 2 m across.
        world = np.array(
            self.scene.world_transform(buildings)
        ).reshape(4, 4)
        placed = positions @ world[:3, :3].T + world[:3, 3]
        span = placed.max(0) - placed.min(0)
        print(f"city spans {span[0]:.2f} x {span[1]:.2f} m")
        assert 1.5 < span[0] < 3.0, "the city should be in metres"
        means = [image[..., :3].mean() for image in self.render_all()]
        print(
            "mean of each view: "
            + ", ".join(f"{m:.1f}" for m in means)
        )
        assert min(means) > 20.0, (
            "every view should show the lit city"
        )


if __name__ == "__main__":
    ernie.demo.main(Demo)
