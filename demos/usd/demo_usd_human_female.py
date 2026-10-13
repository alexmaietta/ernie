# SPDX-License-Identifier: BSD-3-Clause
"""Pixar's HumanFemale from the UsdSkelExamples, imported with
Scene.import_usd: a skinned character whose skeleton, animation and
blend shapes Ernie poses on the GPU at the timeline's time code, walking
or idling, at three levels of detail. The examples (38 MB from
openusd.org, under Pixar's license in HumanFemale/LICENSE.txt) are
fetched once into the cache."""

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

URL = "https://openusd.org/files/UsdSkelExamples.zip"
CLIPS = [
    ("Walk", "HumanFemale.walk.usd"),
    ("Keep alive", "HumanFemale.keepAlive.usd"),
]
LODS = ["full", "medium", "low"]  # The asset's geometricLOD variants.
HUMAN = ernie.Path("/World/Human")
GROUP = "/HumanFemale_Group"  # The prim with the variant sets.


def asset_directory():
    return os.path.join(
        cache_dir(),
        "usd",
        "skel_examples",
        "UsdSkelExamples",
        "HumanFemale",
    )


def fetch_examples(report):
    """Downloads and unpacks the examples into the cache (zipfile checks
    each file's CRC-32), through a directory renamed when complete."""
    root = os.path.join(cache_dir(), "usd")
    archive = os.path.join(root, "UsdSkelExamples.zip")
    report(f"Fetching {URL}")
    ernie.demo.download(URL, archive, report)
    partial = os.path.join(root, "skel_examples.part")
    shutil.rmtree(partial, ignore_errors=True)
    with zipfile.ZipFile(archive) as z:
        z.extractall(partial)
    os.replace(partial, os.path.join(root, "skel_examples"))
    os.remove(archive)


class Demo(ernie.demo.Demo):
    num_frames = 1

    def __init__(self, viewer, args):
        super().__init__(viewer, args)
        scene = self.scene
        world = scene.define("/World", ernie.Xform)
        world.define("Sun", ernie.DistantLight, angle=0.53).set(
            tokens.TRANSFORM,
            ernie.look_at_transform(
                (1.0, -2.0, 2.0), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)
            ),
        )
        scene.define(
            "/Looks/Floor", ernie.Material, albedo=(0.6, 0.6, 0.6)
        )
        world.define(
            "Floor",
            ernie.GroundPlane,
            height=-0.04,
            checker=True,
            checker_size=0.5,
            material_binding="/Looks/Floor",
        )
        self.camera = world.define(
            "Camera",
            ernie.Camera,
            focal_length=24.0,
            clipping_range=(0.01, 50.0),
            sensor_noise=False,
            exposure_mode=tokens.AUTO,
        )
        self.camera.set(
            tokens.TRANSFORM,
            ernie.look_at_transform(
                (2.8, -1.9, 0.9), (0.0, -0.3, 0.65), (0.0, 0.0, 1.0)
            ),
        )
        scene.define(
            "/Render/Settings",
            ernie.RenderSettings,
            camera=self.camera,
            resolution=(600, 800),
            ambient=0.15,
        )
        self.clip = 0
        self.lod = 0
        self.method = tokens.CLASSIC_LINEAR
        self.result: ernie.UsdImport | None = None

        available, why = ernie.usd_import_available()
        if not available:
            self.report(why, True)
            return
        if not os.path.isdir(asset_directory()) and can_download():
            try:
                fetch_examples(self.report)
            except (
                OSError,
                RuntimeError,
                zipfile.BadZipFile,
            ) as error:
                self.report(str(error), True)
        if not os.path.isdir(asset_directory()):
            self.report(
                "The UsdSkelExamples are not cached and cannot be fetched",
                True,
            )
            return
        self.load()

        self.widgets.combo(
            "Clip",
            [name for name, _ in CLIPS],
            0,
            on_change=self.choose_clip,
        )
        self.widgets.combo(
            "Level of detail", LODS, 0, on_change=self.choose_lod
        )
        # Dual quaternions keep volume where joints twist (wrists, elbows).
        self.widgets.combo(
            "Skinning",
            ["Linear", "Dual quaternion"],
            0,
            on_change=self.choose_method,
        )
        self.widgets.combo(
            "Subdivision level",
            ["0", "1", "2"],
            1,
            on_change=lambda level: setattr(
                ernie.Stage.at(self.scene, "/"),
                "subdivision_refine_level",
                level,
            ),
        )
        if viewer is not None:
            viewer.active_camera = self.camera.path
            viewer.play()

    def report(self, text, error=False):
        if self.viewer is not None:
            self.viewer.log(text, "error" if error else "info")
        else:
            print(text)

    def load(self):
        """Imports the clip at the level of detail, its range the
        timeline's."""
        if self.scene.has(HUMAN):
            self.scene.remove(HUMAN)
        name = CLIPS[self.clip][1]
        self.result = self.scene.import_usd(
            HUMAN,
            os.path.join(asset_directory(), name),
            variants={GROUP: {"geometricLOD": LODS[self.lod]}},
        )
        for warning in self.result.warnings:
            self.report(warning, True)
        stage = ernie.Stage.at(self.scene, "/")
        if self.result.time_code_range is not None:
            stage.start_time_code, stage.end_time_code = (
                self.result.time_code_range
            )
            stage.time_codes_per_second = (
                self.result.time_codes_per_second
            )
            stage.frames_per_second = self.result.frames_per_second
            stage.time_code = stage.start_time_code
        self.choose_method(
            1 if self.method == tokens.DUAL_QUATERNION else 0
        )
        self.report(
            f"Imported {name} ({LODS[self.lod]}): {len(self.skinned())} skinned meshes"
        )

    def skinned(self):
        assert self.result is not None
        return [
            path
            for path in self.result.prims.values()
            if self.scene.has(path)
            and self.scene.type_of(path) == "Mesh"
            and self.scene.get(path, tokens.SKEL_SKELETON) is not None
        ]

    def choose_clip(self, index):
        self.clip = index
        self.load()

    def choose_lod(self, index):
        self.lod = index
        self.load()

    def choose_method(self, index):
        self.method = (
            tokens.DUAL_QUATERNION
            if index == 1
            else tokens.CLASSIC_LINEAR
        )
        for path in self.skinned():
            self.scene.set(
                path, tokens.SKEL_SKINNING_METHOD, self.method
            )

    def render_at(self, times):
        """The scene at each time code, RGBA uint8; leaves the time as it
        was."""
        stage = ernie.Stage.at(self.scene, "/")
        current = stage.time_code
        images = []
        for t in times:
            stage.time_code = float(t)
            images.append(np.array(self.renderer.render(self.scene)))
        stage.time_code = current
        return images

    def save_output(self, path):
        """Four moments of the clip side by side."""
        stage = ernie.Stage.at(self.scene, "/")
        times = np.linspace(
            stage.start_time_code, stage.end_time_code, 4
        )
        write_png(path, side_by_side(self.render_at(times)))

    def test_final(self):
        if self.result is None:
            print("Skipped: no USD importer or no UsdSkelExamples")
            return

        def count(kind):
            assert self.result is not None
            return sum(
                1
                for path in self.result.prims.values()
                if self.scene.has(path)
                and self.scene.type_of(path) == kind
            )

        stage = ernie.Stage.at(self.scene, "/")
        print(
            f"{len(self.skinned())} skinned meshes, {count('Skeleton')} skeleton, "
            f"{count('SkelAnimation')} animation, {count('BlendShape')} blend shapes, "
            f"time codes {stage.start_time_code} to {stage.end_time_code}"
        )
        assert count("Skeleton") == 1 and count("SkelAnimation") == 1
        assert len(self.skinned()) == 89, (
            "every mesh of the full level should be skinned"
        )
        assert (stage.start_time_code, stage.end_time_code) == (
            101.0,
            129.0,
        )
        # She moves: the walk's start and middle differ, both lit.
        start, middle = self.render_at([101.0, 115.0])
        moved = (
            np.abs(start.astype(int) - middle.astype(int)).max(-1)
            > 16
        ).mean()
        print(
            f"pixels changed mid-walk: {moved:.3f}; means {start[..., :3].mean():.1f}, "
            f"{middle[..., :3].mean():.1f}"
        )
        assert moved > 0.01, "the walk should move her"
        assert (
            min(start[..., :3].mean(), middle[..., :3].mean()) > 5.0
        )
        # Dual quaternions pose her too.
        self.choose_method(1)
        assert self.render_at([115.0])[0][..., :3].mean() > 5.0
        # The idle clip's own range; the low level's fewer meshes.
        self.choose_clip(1)
        self.choose_lod(2)
        print(
            f"keep alive, low: {len(self.skinned())} meshes, to {stage.end_time_code}"
        )
        assert stage.end_time_code == 250.0
        assert len(self.skinned()) == 22


if __name__ == "__main__":
    ernie.demo.main(Demo)
