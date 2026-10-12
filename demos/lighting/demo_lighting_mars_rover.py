# SPDX-License-Identifier: BSD-3-Clause
"""A rover driving at night on Mars: its LED headlights and mast
floodlight light the rocks ahead and show as beams in the dusty air,
the dust scattering their light forwards, towards the chase camera
behind; a dust storm thickens them.

References:
- M. T. Lemmon et al., "Dust aerosol, clouds, and the atmospheric
  optical depth record over 5 Mars years of the Mars Exploration Rover
  mission", Icarus 251:96-111, 2015 (the dust's optical depth, about
  0.3 to 0.9 in clear seasons and several in storms).
"""

import numpy as np

import ernie
import ernie.app
import ernie.demo
from ernie import tokens
from ernie.demo import write_png

RESOLUTION = (320, 200)
DUST = 0.6  # the dust's vertical optical depth at 880 nm
SPEED = 0.1  # m/s, about a rover's top speed
# Rocks ahead of the rover: place (m) and radius.
ROCKS = [
    ((-2.0, 8.0), 0.5),
    ((1.5, 12.0), 0.8),
    ((-0.5, 17.0), 0.4),
    ((3.0, 20.0), 1.1),
    ((-3.5, 24.0), 0.9),
    ((0.8, 28.0), 0.6),
]


class Demo(ernie.demo.Demo):
    num_frames = 120

    def __init__(self, viewer, args):
        super().__init__(viewer, args)
        scene = self.scene
        world = scene.define("/World", ernie.Xform)
        looks = scene.define("/World/Looks", ernie.Xform)
        # Mars's soil and rocks, reddish, about its mean albedo 0.25.
        looks.define(
            "Soil", ernie.Material, albedo=(0.33, 0.22, 0.14)
        )
        looks.define("Rock", ernie.Material, albedo=(0.3, 0.2, 0.14))
        looks.define("Body", ernie.Material, albedo=(0.7,) * 3)
        looks.define("Wheel", ernie.Material, albedo=(0.15,) * 3)
        world.define(
            "Ground",
            ernie.GroundPlane,
            material_binding="/World/Looks/Soil",
        )
        self.air = world.define(
            "Air",
            ernie.Atmosphere,
            preset=tokens.MARS,
            dust_optical_depth=DUST,
        )
        for i, ((x, y), radius) in enumerate(ROCKS):
            world.define(
                f"Rock{i}",
                ernie.Sphere,
                radius=radius,
                translate=(x, y, 0.4 * radius),
                material_binding="/World/Looks/Rock",
            )
        # The rover, which drives along +y: its body, six wheels and
        # mast, and its lights, all carried by its transform.
        self.rover = world.define("Rover", ernie.Xform)
        self.rover.define(
            "Body",
            ernie.Cube,
            size=1.0,
            translate=(0.0, 0.0, 0.9),
            scale=(1.6, 2.4, 0.5),
            material_binding="/World/Looks/Body",
        )
        for i in range(6):
            side = -1.0 if i % 2 == 0 else 1.0
            self.rover.define(
                f"Wheel{i}",
                ernie.Cylinder,
                radius=0.26,
                height=0.25,
                axis="X",
                translate=(side * 0.95, -0.9 + 0.9 * (i // 2), 0.26),
                material_binding="/World/Looks/Wheel",
            )
        self.rover.define(
            "Mast",
            ernie.Cylinder,
            radius=0.05,
            height=1.0,
            translate=(0.5, 0.9, 1.6),
            material_binding="/World/Looks/Wheel",
        )
        self.lamps = []
        # Two LED headlights, cool white, a little apart and dipped.
        for name, x in (("HeadLeft", -0.55), ("HeadRight", 0.55)):
            lamp = self.rover.define(
                name,
                ernie.SphereLight,
                radius=0.04,
                shaping_cone_angle=22.0,
                shaping_cone_softness=0.4,
                photometric_power=2500.0,
                enable_color_temperature=True,
                color_temperature=6000.0,
            ).set(
                tokens.TRANSFORM,
                ernie.look_at_transform(
                    (x, 1.22, 0.9),
                    (x * 2.0, 12.0, 0.0),
                    (0.0, 0.0, 1.0),
                ),
            )
            self.lamps.append(lamp)
        # A floodlight on the mast, wider and looking down ahead.
        self.lamps.append(
            self.rover.define(
                "Flood",
                ernie.SphereLight,
                radius=0.06,
                shaping_cone_angle=45.0,
                shaping_cone_softness=0.5,
                photometric_power=3000.0,
                enable_color_temperature=True,
                color_temperature=5000.0,
            ).set(
                tokens.TRANSFORM,
                ernie.look_at_transform(
                    (0.5, 0.9, 2.15), (0.5, 6.0, 0.0), (0.0, 0.0, 1.0)
                ),
            )
        )
        self.y = 0.0
        self.speed = SPEED
        self.drive(0.0)
        # A chase camera behind and above, a long exposure for the
        # night.
        self.camera = world.define(
            "Camera",
            ernie.Camera,
            focal_length=16.0,
            sensor_noise=False,
            exposure_time=0.25,
            exposure_f_stop=2.8,
            exposure_iso=400.0,
            clipping_range=(0.05, 1e4),
        )
        product = scene.define("/Render/Camera", ernie.RenderProduct)
        var = scene.define(
            "/Render/Camera/SpectralRadiance", ernie.RenderVar
        )
        var.source_name = "SpectralRadiance"
        product.set(
            camera=self.camera,
            resolution=RESOLUTION,
            ordered_vars=[var.path],
        )
        scene.define(
            "/Render/Settings",
            ernie.RenderSettings,
            camera=self.camera,
            resolution=RESOLUTION,
            ambient=0.0,
        )
        self.follow()
        self.widgets.slider(
            "Dust optical depth", 0.0, 5.0, DUST, on_change=self.dust
        )
        self.widgets.slider(
            "Speed (m/s)", 0.0, 1.0, SPEED, on_change=self.drive_at
        )
        if viewer is not None:
            viewer.active_camera = self.camera.path
            viewer.lighting = ernie.app.Lighting.STAGE
            display = viewer.viewport_display
            display.lights = False
            display.grid = False
            viewer.viewport_display = display

    def drive(self, y):
        self.y = y
        self.rover.set(translate=(0.0, y, 0.0))

    def follow(self):
        """The chase camera behind and above the rover."""
        self.camera.set(
            tokens.TRANSFORM,
            ernie.look_at_transform(
                (-4.0, self.y - 9.0, 3.5),
                (0.0, self.y + 9.0, 0.5),
                (0.0, 0.0, 1.0),
            ),
        )

    def dust(self, depth):
        self.air.dust_optical_depth = depth

    def drive_at(self, speed):
        self.speed = speed

    def step(self):
        self.drive(self.y + self.speed * self.frame_dt)
        self.follow()

    def capture(self, lit=True):
        """The radiance summed over the bands, with the lights on or
        off."""
        for lamp in self.lamps:
            lamp.set(
                tokens.VISIBILITY,
                tokens.INHERITED if lit else tokens.INVISIBLE,
            )
        outputs = self.renderer.step(
            self.scene, render_products=["/Render/Camera"]
        )
        frame = outputs["/Render/Camera"].frames[0].render_vars
        radiance = np.array(
            frame["/Render/Camera/SpectralRadiance"].map()
        ).astype(np.float64)
        return radiance.sum(axis=0)

    def save_output(self, path):
        write_png(path, self.render_image())

    def test_final(self):
        lit = self.capture()
        self.dust(4.0 * DUST)
        storm = self.capture()
        self.air.set(tokens.VISIBILITY, tokens.INVISIBLE)
        airless = self.capture()
        self.air.set(tokens.VISIBILITY, tokens.INHERITED)
        self.dust(DUST)
        dark = self.capture(lit=False)
        # The beams: what the dusty air adds over the ground and rocks.
        beams = lit - airless
        shown = (beams > 0.05 * np.percentile(beams, 99.9)).mean()
        ratio = (storm - airless).sum() / beams.sum()
        print(
            f"the beams light {100 * shown:.1f}% of the frame; a dust "
            f"storm four times as thick scatters {ratio:.2f} times as "
            f"much; dark, the brightest pixel is {dark.max():.3g}"
        )
        assert shown > 0.05, "the beams show in the dust"
        assert ratio > 1.5, "the storm thickens them"
        assert dark.max() == 0.0, "nothing else lights the night"


if __name__ == "__main__":
    ernie.demo.main(Demo)
