# SPDX-License-Identifier: BSD-3-Clause
"""Pixar's Kitchen Set, the OpenUSD reference scene, imported with
Scene.import_usd: hundreds of assets composed through references and
payloads, plain or with USD's native instancing, in centimetres, under
two ceiling lights. The set (2.7 MB from openusd.org, under Pixar's
license in its LICENSE.txt) is fetched once into the cache."""

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

URL = "https://openusd.org/files/Kitchen_set.zip"
# Each view: name, eye and target (metres), or None from overhead with
# the ceiling cut away.
VIEWS = [
    ("Into the kitchen", (3.3, -2.3, 1.7), (0.0, 0.4, 0.8)),
    ("Stove and fridge", (0.0, -1.6, 1.4), (2.5, 1.0, 0.9)),
    ("Breakfast table", (-1.0, 0.8, 1.5), (2.0, -1.0, 0.8)),
    ("Cutaway", None, None),
]
KITCHEN = ernie.Path("/World/Kitchen")
CEILING = "/Kitchen_set/Arch_grp/Kitchen_1/Geom/Ceiling"
CEILING_HEIGHT = 2.54  # metres
CUTAWAY_HEIGHT = 9.0  # metres, the overhead camera's
LIGHTS = [
    (-0.3, -0.6),
    (2.0, -0.6),
]  # ceiling lights' x and y, metres


def set_directory():
    return os.path.join(
        cache_dir(), "usd", "kitchen_set", "Kitchen_set"
    )


def fetch_set(report):
    """Downloads and unpacks the set into the cache (zipfile checks each
    file's CRC-32), through a directory renamed when complete."""
    root = os.path.join(cache_dir(), "usd")
    archive = os.path.join(root, "Kitchen_set.zip")
    report(f"Fetching {URL}")
    ernie.demo.download(URL, archive, report)
    partial = os.path.join(root, "kitchen_set.part")
    shutil.rmtree(partial, ignore_errors=True)
    with zipfile.ZipFile(archive) as z:
        z.extractall(partial)
    os.replace(partial, os.path.join(root, "kitchen_set"))
    os.remove(archive)


class Demo(ernie.demo.Demo):
    num_frames = 1

    def __init__(self, viewer, args):
        super().__init__(viewer, args)
        scene = self.scene
        world = scene.define("/World", ernie.Xform)
        # Two lights just under the ceiling, facing down.
        for i, (x, y) in enumerate(LIGHTS):
            world.define(
                f"Light{i}",
                ernie.RectLight,
                width=1.2,
                height=0.6,
                photometric_power=4000.0,
                shadow_enable=True,
            ).set(
                tokens.TRANSFORM,
                ernie.look_at_transform(
                    (x, y, CEILING_HEIGHT - 0.09),
                    (x, y, 0.0),
                    up=(0.0, 1.0, 0.0),
                ),
            )
        self.camera = world.define(
            "Camera",
            ernie.Camera,
            focal_length=18.0,
            clipping_range=(0.02, 50.0),
            sensor_noise=False,
            exposure_mode=tokens.AUTO,
        )
        scene.define(
            "/Render/Settings",
            ernie.RenderSettings,
            camera=self.camera,
            resolution=(960, 600),
            ambient=0.0,
        )
        self.instanced = False
        self.subdivided = True
        self.result: ernie.UsdImport | None = None
        # Each imported mesh's subdivision scheme, to restore.
        self.schemes: dict[ernie.Path, str] = {}
        self.view = 0

        available, why = ernie.usd_import_available()
        if not available:
            self.report(why, True)
            return
        if not os.path.isdir(set_directory()) and can_download():
            try:
                fetch_set(self.report)
            except (
                OSError,
                RuntimeError,
                zipfile.BadZipFile,
            ) as error:
                self.report(str(error), True)
        if not os.path.isdir(set_directory()):
            self.report(
                "The Kitchen Set is not cached and cannot be fetched",
                True,
            )
            return
        self.load()
        self.show(0)

        self.widgets.combo(
            "View",
            [name for name, *_ in VIEWS],
            0,
            on_change=self.show,
        )
        # The instanced set's native instances as instancers of shared
        # prototypes, or the plain set's copies.
        self.widgets.checkbox(
            "Instancers", False, on_change=self.instancing
        )
        # The set's meshes are Catmull-Clark surfaces (USD's fallback),
        # refined by OpenSubdiv, or their control meshes drawn as they are;
        # 2^level segments along each control edge.
        self.widgets.checkbox(
            "Subdivision surfaces",
            True,
            on_change=self.subdivision_surfaces,
        )
        self.widgets.combo(
            "Subdivision level",
            ["0", "1", "2", "3"],
            1,
            on_change=self.subdivision,
        )
        self.widgets.slider(
            "Light power (lm)",
            500.0,
            20000.0,
            4000.0,
            on_change=self.power,
        )
        if viewer is not None:
            viewer.active_camera = self.camera.path

    def report(self, text, error=False):
        if self.viewer is not None:
            self.viewer.log(text, "error" if error else "info")
        else:
            print(text)

    def load(self):
        """Imports the set, natively instanced or not."""
        if self.scene.has(KITCHEN):
            self.scene.remove(KITCHEN)
        name = (
            "Kitchen_set_instanced.usd"
            if self.instanced
            else "Kitchen_set.usd"
        )
        self.result = self.scene.import_usd(
            KITCHEN, os.path.join(set_directory(), name)
        )
        for warning in self.result.warnings:
            self.report(warning, True)
        self.report(
            f"Imported {name}: {len(self.result.prims)} prims"
        )
        self.schemes = {
            path: ernie.Mesh.at(self.scene, path).subdivision_scheme
            for path in self.result.prims.values()
            if self.scene.has(path)
            and self.scene.type_of(path) == "Mesh"
        }
        self.subdivision_surfaces(self.subdivided)

    def instancing(self, on):
        self.instanced = on
        self.load()
        self.show(self.view)

    def subdivision_surfaces(self, on):
        self.subdivided = on
        for path, scheme in self.schemes.items():
            mesh = ernie.Mesh.at(self.scene, path)
            mesh.subdivision_scheme = scheme if on else "none"

    def subdivision(self, level):
        ernie.Stage.at(
            self.scene, "/"
        ).subdivision_refine_level = level

    def power(self, lumens):
        for i in range(len(LIGHTS)):
            light = ernie.RectLight.at(self.scene, f"/World/Light{i}")
            light.photometric_power = lumens

    def show(self, index):
        self.view = index
        _, eye, target = VIEWS[index]
        if eye is None:
            # The cutaway clips the ceiling with the near plane, whether
            # the ceiling is drawn as itself or as instances.
            transform = ernie.look_at_transform(
                (0.7, -0.3, CUTAWAY_HEIGHT),
                (0.7, -0.29, 0.0),
                (0.0, 1.0, 0.0),
            )
            self.camera.clipping_range = (
                CUTAWAY_HEIGHT - CEILING_HEIGHT + 0.05,
                50.0,
            )
        else:
            transform = ernie.look_at_transform(
                eye, target, (0.0, 0.0, 1.0)
            )
            self.camera.clipping_range = (0.02, 50.0)
        self.camera.set(tokens.TRANSFORM, transform)

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
            print("Skipped: no USD importer or no Kitchen Set")
            return
        # Centimetres into metres: the ceiling at 2.54 m.
        ceiling = self.result.prims[CEILING + "/pPlane5"]
        points = np.asarray(self.scene.read(ceiling, "points"))
        world = np.array(self.scene.world_transform(ceiling)).reshape(
            4, 4
        )
        height = (world @ [*points.reshape(-1, 3)[0], 1.0])[2]
        print(f"ceiling at {height:.3f} m")
        assert abs(height - CEILING_HEIGHT) < 0.01, (
            "the set should be in metres"
        )

        # Native instances become instancers sharing their prototypes, so
        # the instanced set imports fewer meshes than the plain one.
        def count(kind: str) -> int:
            assert self.result is not None
            return sum(
                1
                for path in self.result.prims.values()
                if self.scene.has(path)
                and self.scene.type_of(path) == kind
            )

        plain = count("Mesh")
        self.instancing(True)
        print(
            f"meshes: {plain} plain, {count('Mesh')} instanced, in "
            f"{count('PointInstancer')} instancers"
        )
        assert count("PointInstancer") > 0, (
            "instances should be instanced"
        )
        assert count("Mesh") < plain, "prototypes should be shared"
        means = [image[..., :3].mean() for image in self.render_all()]
        print(
            "mean of each view: "
            + ", ".join(f"{m:.1f}" for m in means)
        )
        assert min(means) > 5.0, "every view should be lit"
        # Without subdivision surfaces, the control meshes as they are.
        self.subdivision_surfaces(False)
        assert all(
            ernie.Mesh.at(self.scene, path).subdivision_scheme
            == "none"
            for path in self.schemes
        ), "every mesh should draw its control mesh"
        cage = self.render_all()[0][..., :3].mean()
        print(f"control meshes: mean {cage:.1f}")
        assert cage > 5.0, "the control meshes should be lit"


if __name__ == "__main__":
    ernie.demo.main(Demo)
