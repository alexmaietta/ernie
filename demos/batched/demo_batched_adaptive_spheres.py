# SPDX-License-Identifier: BSD-3-Clause
"""Four worlds in one place, each an Earth-sized adaptive sphere seen by
one view of a batched camera from its own altitude: every sphere is refined
only towards its own world's view, so each horizon lies where the true
sphere's does. LOD levels are tinted."""

import math

import numpy as np

import ernie
import ernie.demo
from ernie import tokens
from ernie.demo import write_png

EARTH = 6371000.0
SIZE = 256
FOCAL = 24.0
CAMERA = ernie.Path("/World/Camera")
PRODUCT = ernie.Path("/Render/Batch")

# Each world's view: its altitude, metres, and its sphere's colour.
WORLDS = [
    (100000.0, (0.30, 0.45, 0.80)),
    (10000.0, (0.35, 0.60, 0.30)),
    (1000.0, (0.75, 0.60, 0.35)),
    (100.0, (0.70, 0.70, 0.70)),
]


def pose_of(w):
    """The view of world `w`: over the north pole at its altitude, looking
    level along +x."""
    eye = (0.0, 0.0, EARTH + WORLDS[w][0])
    return ernie.look_at_transform(
        eye, (1.0, 0.0, eye[2]), (0.0, 0.0, 1.0)
    )


def horizon_row(w):
    """The image row of the true sphere's horizon from world `w`'s view:
    acos(R / (R + h)) below the centre."""
    dip = math.acos(EARTH / (EARTH + WORLDS[w][0]))
    half_height = 0.5 * 15.2908 / FOCAL
    return 0.5 * SIZE * (1.0 + math.tan(dip) / half_height)


class Demo(ernie.demo.Demo):
    num_frames = 1

    def __init__(self, viewer, args):
        super().__init__(viewer, args)
        scene = self.scene
        scene.define("/World", ernie.Xform)
        scene.define(
            "/World/Sun", ernie.DistantLight, intensity=1.0
        ).set(
            tokens.TRANSFORM,
            ernie.distant_light_transform((0.3, -0.4, 0.85)),
        )
        for w, (_, colour) in enumerate(WORLDS):
            env = ernie.Path(f"/Envs/Env{w}")
            scene.define(env, ernie.Xform, world=w)
            look = env.child("Look")
            scene.define(look, ernie.Material)
            scene.define(
                look.child("Surface"),
                ernie.Shader,
                info_id=tokens.USD_PREVIEW_SURFACE,
                diffuse_color=colour,
            )
            scene.connect(
                look,
                tokens.SURFACE_OUTPUT,
                look.child("Surface"),
                tokens.SURFACE_OUTPUT,
            )
            scene.define(
                env.child("Planet"),
                ernie.Sphere,
                radius=EARTH,
                tessellation=tokens.ADAPTIVE_SPHERE,
                material_binding=look,
            )
        views = len(WORLDS)
        camera = scene.define(
            CAMERA,
            ernie.Camera,
            focal_length=FOCAL,
            clipping_range=(1.0, math.inf),
        )
        scene.write(
            CAMERA,
            tokens.POSES,
            np.array([pose_of(w) for w in range(views)], np.float32),
        )
        scene.write(
            CAMERA, tokens.WORLDS, np.arange(views, dtype=np.int32)
        )
        ldr = scene.define(
            PRODUCT.child("LdrColor"),
            ernie.RenderVar,
            source_name="LdrColor",
        )
        scene.define(
            PRODUCT,
            ernie.RenderProduct,
            camera=camera,
            resolution=(SIZE, SIZE),
            ordered_vars=[ldr],
        )
        # The viewer looks down on the worlds, all in one place.
        overview = scene.define(
            "/World/Overview",
            ernie.Camera,
            focal_length=24.0,
            clipping_range=(1.0, math.inf),
        )
        overview.set(
            tokens.TRANSFORM,
            ernie.look_at_transform(
                (0.0, -2.0e7, 1.0e7), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)
            ),
        )
        scene.define(
            "/Render/Settings",
            ernie.RenderSettings,
            camera=overview,
            resolution=(1280, 720),
            lod_show_levels=True,
        )
        if viewer is not None:
            viewer.active_camera = overview.path

    def views(self):
        """Every view's LdrColor, uint8 (views, SIZE, SIZE, 4)."""
        outputs = self.renderer.step(
            self.scene, render_products=[PRODUCT]
        )
        return np.array(
            outputs[str(PRODUCT)]
            .frames[0]
            .render_vars[f"{PRODUCT}/LdrColor"]
            .map()
        )

    def save_output(self, path):
        """The four views side by side."""
        write_png(path, np.concatenate(list(self.views()), axis=1))

    def test_final(self):
        # Down each view's middle column, the first row that is not sky is
        # the true sphere's horizon.
        images = self.views()
        for w, (altitude, _) in enumerate(WORLDS):
            column = (
                images[w, :, SIZE // 2, :3]
                .astype(np.int64)
                .sum(axis=1)
            )
            ground = np.nonzero(column > 0)[0]
            row = int(ground[0]) if len(ground) else SIZE
            print(
                f"world {w}, {altitude:.0f} m up: horizon at row {row}, "
                f"the sphere's {horizon_row(w):.1f}"
            )
            assert abs(row - horizon_row(w)) < 1.5, (
                "each world's horizon is its true sphere's"
            )


if __name__ == "__main__":
    ernie.demo.main(Demo)
