# SPDX-License-Identifier: BSD-3-Clause
"""OpenVDB's sample volumes side by side, read from their OpenVDB files
(.vdb) as they are: Ernie reads each with its built-in OpenVDB and walks
the grids on the GPU as NanoVDB. Smoke and a second smoke column scatter
the sun; fire, and the explosion when switched on, glow as black bodies
at their temperature grids. The samples (60 MB, the explosion 76 MB
more) are fetched once from openvdb.org into the cache."""

import os

import numpy as np

import ernie
import ernie.demo
from ernie import tokens
from ernie.demo import cache_dir, can_download, write_png

URL = (
    "https://media.githubusercontent.com/media/AcademySoftwareFoundation/"
    "openvdb-website/master/download/models/{}.vdb"
)
SAMPLES = ernie.Path("/World/Samples")

# Each sample: its file, its active bounds in its own units (Y up), the
# height it stands in metres, where it stands (x, y), and its material.
SAMPLES_TABLE = [
    (
        "smoke",
        ((-33.6, -12.1, -26.0), (18.8, 93.1, 26.9)),
        4.0,
        (-4.5, 0.0),
        dict(
            density=2.5, volume_albedo=(0.9, 0.9, 0.9), anisotropy=0.3
        ),
    ),
    (
        "fire",
        ((-19.1, -8.2, -18.7), (19.9, 80.9, 18.4)),
        3.0,
        (0.0, 0.0),
        # Its temperature grid peaks near 45: 40 K a unit.
        dict(
            density=5.0,
            volume_albedo=(0.3, 0.3, 0.3),
            temperature_field="temperature",
            temperature_scale=40.0,
        ),
    ),
    (
        "smoke2",
        ((-6.9, -1.2, -7.9), (12.1, 59.7, 9.8)),
        5.0,
        (4.5, 0.0),
        dict(
            density=30.0,
            volume_albedo=(0.95, 0.95, 0.95),
            anisotropy=0.3,
        ),
    ),
    (
        "explosion",
        ((-85.5, -35.5, -108.5), (114.5, 235.5, 120.5)),
        6.0,
        (0.0, 9.0),
        # Its temperature grid peaks near 0.9: 3000 K a unit.
        dict(
            density=0.8,
            volume_albedo=(0.4, 0.4, 0.4),
            temperature_field="temperature",
            temperature_scale=3000.0,
        ),
    ),
]
OPTIONAL = {"explosion"}  # 76 MB, fetched when switched on; behind.


def sample_file(name):
    return os.path.join(cache_dir(), "openvdb", name + ".vdb")


class Demo(ernie.demo.Demo):
    num_frames = 1

    def __init__(self, viewer, args):
        super().__init__(viewer, args)
        scene = self.scene
        world = scene.define("/World", ernie.Xform)
        scene.define(SAMPLES, ernie.Xform)
        world.define(
            "Sun",
            ernie.DistantLight,
            irradiance=1361.0,
            physical_illuminant=tokens.SUN,
            shadow_enable=True,
        ).set(
            tokens.TRANSFORM,
            ernie.distant_light_transform((-0.6, -1.0, 0.7)),
        )
        scene.define(
            "/World/Looks/Ground",
            ernie.Material,
            albedo=(0.06, 0.06, 0.06),
        )
        world.define(
            "Ground",
            ernie.GroundPlane,
            checker=True,
            checker_size=1.0,
            material_binding="/World/Looks/Ground",
        )
        self.camera = world.define(
            "Camera",
            ernie.Camera,
            focal_length=28.0,
            sensor_noise=False,
            # Daylight exposure: auto exposure would meter on the fire.
            exposure_iso=100.0,
            exposure_f_stop=8.0,
            exposure_time=1.0 / 250.0,
        ).set(
            tokens.TRANSFORM,
            ernie.look_at_transform(
                (0.5, -11.0, 2.4), (0.5, 0.0, 2.1), (0.0, 0.0, 1.0)
            ),
        )
        scene.define(
            "/Render/Settings",
            ernie.RenderSettings,
            camera=self.camera,
            resolution=(960, 400),
            ambient=0.1,
        )
        self.shown = []
        for name, *_ in SAMPLES_TABLE:
            if name not in OPTIONAL:
                self.show(name)

        self.widgets.checkbox(
            "Explosion (76 MB)", False, on_change=self.set_explosion
        )
        if viewer is not None:
            viewer.active_camera = self.camera.path

    def report(self, text, error=False):
        if self.viewer is not None:
            self.viewer.log(text, "error" if error else "info")
        else:
            print(text)

    def show(self, name):
        """Fetches the sample once and stands it on the ground."""
        file = sample_file(name)
        if not os.path.isfile(file) and can_download():
            try:
                self.report(f"Fetching {URL.format(name)}")
                ernie.demo.download(
                    URL.format(name), file, self.report
                )
            except (OSError, RuntimeError) as error:
                self.report(str(error), True)
        if not os.path.isfile(file):
            self.report(
                f"{name}.vdb is not cached and cannot be fetched",
                True,
            )
            return
        _, (low, high), height, (x, y), material = next(
            entry for entry in SAMPLES_TABLE if entry[0] == name
        )
        scale = height / (high[1] - low[1])
        look = f"/World/Looks/{name}"
        self.scene.define(look, ernie.Material, **material)
        # Y up to Z up (a quarter turn about x takes y to z and z to -y),
        # centred on x, its foot on the ground.
        volume = self.scene.define(
            SAMPLES.child(name),
            ernie.Volume,
            material_binding=look,
            rotate=(90.0, 0.0, 0.0),
            scale=(scale, scale, scale),
            translate=(
                x - 0.5 * (low[0] + high[0]) * scale,
                y + 0.5 * (low[2] + high[2]) * scale,
                -low[1] * scale,
            ),
        )
        fields = ["density"] + (
            ["temperature"] if "temperature_field" in material else []
        )
        for field in fields:
            path = volume.define(
                field.capitalize(),
                ernie.OpenVDBAsset,
                file_path=file,
                field_name=field,
            ).path
            volume.set_field(field, path)
        self.shown.append(name)

    def set_explosion(self, on):
        if on:
            self.show("explosion")
        else:
            self.scene.remove(SAMPLES.child("explosion"))
            self.shown.remove("explosion")

    def render(self):
        """The scene, RGB as floats."""
        return np.array(self.renderer.render(self.scene))[
            ..., :3
        ].astype(float)

    def save_output(self, path):
        write_png(path, np.array(self.renderer.render(self.scene)))

    def test_final(self):
        if not self.shown:
            print("Skipped: no OpenVDB samples")
            return
        assert ernie.openvdb_available(), "Ernie should read OpenVDB"
        scene = self.scene
        self.render()
        full = self.render()
        for name in self.shown:
            path = SAMPLES.child(name)
            scene.set(path, tokens.VISIBILITY, tokens.INVISIBLE)
            without = self.render()
            scene.set(path, tokens.VISIBILITY, tokens.INHERITED)
            covers = (np.abs(full - without).max(-1) > 16).mean()
            print(f"{name}: covers {covers:.3f} of the view")
            assert covers > 0.005, f"{name} should show"
        # In the dark the fire still glows at its temperature.
        scene.set("/World/Sun", tokens.VISIBILITY, tokens.INVISIBLE)
        scene.set("/Render/Settings", tokens.AMBIENT, 0.0)
        dark = self.render()
        scene.set(
            SAMPLES.child("fire"), tokens.VISIBILITY, tokens.INVISIBLE
        )
        unlit = self.render()
        print(
            f"in the dark: mean {dark.mean():.2f} with fire, {unlit.mean():.2f} without"
        )
        assert dark.mean() > unlit.mean() + 0.05, (
            "the fire should glow"
        )


if __name__ == "__main__":
    ernie.demo.main(Demo)
