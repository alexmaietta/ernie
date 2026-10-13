# SPDX-License-Identifier: BSD-3-Clause
"""Laser speckle in a room, through a phone-sized camera: an expanded
green beam floods a table and a red pointer marks the wall. Rough plaster,
paper and wood sparkle with full-contrast grain, brushed metal with
partial grain, and a polished ball with none; the grain grows as the
lens stops down and boils as the camera moves.

References:
- J. W. Goodman, "Speckle Phenomena in Optics", Roberts and Company,
  2007 (grain size about lambda N on the sensor, contrast 1 / sqrt(M),
  partially developed speckle from smooth surfaces).
"""

import math

import numpy as np

import ernie
import ernie.app
import ernie.demo
from ernie import tokens
from ernie.demo import write_png

RESOLUTION = (480, 360)
PIXEL = 2e-3  # mm, a phone camera's pixel
FOCAL = 1.0  # mm
F_NUMBER = 4.0
EYE = (0.7, -0.5, 1.35)  # m, the camera
TARGET = (0.0, 1.5, 0.9)
# Each surface: its look's albedo and rms height (m), which makes its
# speckle fully developed (rough), partial or absent (polished).
LOOKS = {
    "Plaster": ((0.8, 0.78, 0.74), 1e-5),
    "Wood": ((0.45, 0.3, 0.18), 1e-5),
    "Paper": ((0.85, 0.85, 0.8), 1e-5),
    "Brushed": ((0.6, 0.6, 0.62), 2.5e-8),
    "Polished": ((0.9, 0.9, 0.9), 1e-9),
}


class Demo(ernie.demo.Demo):
    num_frames = 120

    def __init__(self, viewer, args):
        super().__init__(viewer, args)
        scene = self.scene
        world = scene.define("/World", ernie.Xform)
        looks = scene.define("/World/Looks", ernie.Xform)
        for name, (albedo, height) in LOOKS.items():
            looks.define(
                name,
                ernie.Material,
                albedo=albedo,
                speckle_roughness=height,
            )
        world.define(
            "Floor",
            ernie.GroundPlane,
            material_binding="/World/Looks/Wood",
        )
        world.define(
            "Wall",
            ernie.Cube,
            size=1.0,
            translate=(0.0, 2.5, 1.5),
            scale=(6.0, 0.1, 3.0),
            material_binding="/World/Looks/Plaster",
        )
        world.define(
            "Table",
            ernie.Cube,
            size=1.0,
            translate=(0.0, 1.5, 0.725),
            scale=(1.6, 0.8, 0.05),
            material_binding="/World/Looks/Wood",
        )
        world.define(
            "Box",
            ernie.Cube,
            size=0.25,
            translate=(-0.5, 1.5, 0.875),
            material_binding="/World/Looks/Paper",
        )
        world.define(
            "Can",
            ernie.Cylinder,
            radius=0.1,
            height=0.3,
            translate=(0.0, 1.5, 0.9),
            material_binding="/World/Looks/Brushed",
        )
        world.define(
            "Ball",
            ernie.Sphere,
            radius=0.12,
            translate=(0.5, 1.5, 0.87),
            material_binding="/World/Looks/Polished",
        )
        # A green beam expanded over the table's objects and a red
        # pointer's dot on the wall, both keeping their polarization.
        self.lasers = [
            world.define(
                "Beam",
                ernie.LaserLight,
                wavelength=532.0,
                linewidth=0.0,
                power=0.25,
                waist_radius=(0.45, 0.25),
                polarization_degree=1.0,
                speckle=True,
            ).set(
                tokens.TRANSFORM,
                ernie.look_at_transform(
                    (-0.2, 0.3, 1.6), (0.0, 1.5, 0.9), (0.0, 0.0, 1.0)
                ),
            ),
            world.define(
                "Pointer",
                ernie.LaserLight,
                wavelength=650.0,
                linewidth=0.0,
                power=0.005,
                waist_radius=(0.02, 0.02),
                polarization_degree=1.0,
                speckle=True,
            ).set(
                tokens.TRANSFORM,
                ernie.look_at_transform(
                    (0.3, 0.2, 1.2),
                    (-0.6, 2.45, 1.45),
                    (0.0, 0.0, 1.0),
                ),
            ),
        ]
        # A dim lamp over the table, so the room reads around the lasers.
        self.lamp = world.define(
            "Lamp",
            ernie.SphereLight,
            radius=0.1,
            photometric_power=600.0,
            enable_color_temperature=True,
            color_temperature=3000.0,
            translate=(0.0, 1.0, 2.6),
        )
        self.camera = world.define(
            "Camera",
            ernie.Camera,
            focal_length=FOCAL,
            horizontal_aperture=PIXEL * RESOLUTION[0],
            vertical_aperture=PIXEL * RESOLUTION[1],
            exposure_iso=800.0,
            sensor_noise=False,
            clipping_range=(0.02, 100.0),
        )
        product = scene.define("/Render/Camera", ernie.RenderProduct)
        variables = []
        for name in ("SpectralRadiance", "PrimId"):
            var = scene.define(
                f"/Render/Camera/{name}", ernie.RenderVar
            )
            var.source_name = name
            variables.append(var.path)
        product.set(
            camera=self.camera,
            resolution=RESOLUTION,
            ordered_vars=variables,
        )
        scene.define(
            "/Render/Settings",
            ernie.RenderSettings,
            camera=self.camera,
            resolution=RESOLUTION,
            ambient=0.0,
        )
        self.orbit = 0.0  # radians a second the camera circles
        self.angle = 0.0
        self.look()
        self.stop_down(F_NUMBER)
        self.widgets.slider(
            "f-number", 1.8, 16.0, F_NUMBER, on_change=self.stop_down
        )
        self.widgets.checkbox("Speckle", True, on_change=self.speckle)
        self.widgets.slider(
            "Camera orbit (deg/s)",
            0.0,
            5.0,
            0.0,
            on_change=self.circle,
        )
        if viewer is not None:
            viewer.active_camera = self.camera.path
            viewer.lighting = ernie.app.Lighting.STAGE
            display = viewer.viewport_display
            display.lights = False
            display.grid = False
            viewer.viewport_display = display

    def stop_down(self, n):
        """Sets the lens to f-number `n`, the exposure time keeping the
        image as bright."""
        self.camera.set(exposure_f_stop=n, exposure_time=2e-3 * n * n)

    def speckle(self, on):
        for laser in self.lasers:
            laser.speckle = on

    def circle(self, degrees):
        self.orbit = math.radians(degrees)

    def look(self):
        """The camera, turned `angle` about the table's centre."""
        c, s = math.cos(self.angle), math.sin(self.angle)
        x, y = EYE[0] - TARGET[0], EYE[1] - TARGET[1]
        eye = (
            TARGET[0] + c * x - s * y,
            TARGET[1] + s * x + c * y,
            EYE[2],
        )
        self.camera.set(
            tokens.TRANSFORM,
            ernie.look_at_transform(eye, TARGET, (0.0, 0.0, 1.0)),
        )

    def step(self):
        if self.orbit > 0.0:
            self.angle += self.orbit * self.frame_dt
            self.look()

    def capture(self, speckle=True):
        """The lasers' radiance summed over the bands (the lamp off),
        with or without speckle, and the prims' paths per pixel."""
        self.speckle(speckle)
        self.lamp.set(tokens.VISIBILITY, tokens.INVISIBLE)
        outputs = self.renderer.step(
            self.scene, render_products=["/Render/Camera"]
        )
        self.lamp.set(tokens.VISIBILITY, tokens.INHERITED)
        self.speckle(True)
        frame = outputs["/Render/Camera"].frames[0].render_vars
        radiance = np.array(
            frame["/Render/Camera/SpectralRadiance"].map()
        ).astype(np.float64)
        ids = np.array(frame["/Render/Camera/PrimId"].map())
        ids = ids.reshape(RESOLUTION[1], RESOLUTION[0], -1)[..., 0]
        return radiance.sum(axis=0), ids

    def contrasts(self):
        """Each lit surface's speckle contrast (deviation over mean of
        the light with speckle over without), over its brightly lit
        pixels."""
        speckled, ids = self.capture()
        plain, _ = self.capture(speckle=False)
        ratio = speckled / np.maximum(plain, 1e-30)
        paths = {
            str(self.renderer.prim_path(int(i))): i
            for i in np.unique(ids)
        }
        result = {}
        for prim in ("Wall", "Box", "Can", "Ball", "Table"):
            if f"/World/{prim}" not in paths:
                continue
            on = ids == paths[f"/World/{prim}"]
            # Its pixels the lasers light brightly, as it sees them.
            mask = on & (plain > 0.2 * plain[on].max())
            if mask.sum() >= 20:
                values = ratio[mask]
                result[prim] = float(values.std() / values.mean())
        return result

    def save_output(self, path):
        write_png(path, self.render_image())

    def test_final(self):
        contrasts = self.contrasts()
        print(
            "speckle contrast: "
            + ", ".join(f"{k} {v:.2f}" for k, v in contrasts.items())
        )
        for rough in ("Wall", "Box"):
            assert contrasts[rough] > 0.3, f"{rough} speckles fully"
        assert contrasts["Ball"] < 0.1, "the polished ball does not"
        assert (
            contrasts["Ball"] < contrasts["Can"] < contrasts["Box"]
        ), "the brushed can speckles in part"


if __name__ == "__main__":
    ernie.demo.main(Demo)
