# SPDX-License-Identifier: BSD-3-Clause
"""A lighthouse at night: its two beams sweep round through the hazy
air, seen from the shore by the light the air scatters out of them,
the lantern's iron bars cutting dark shafts through each; the lantern
glass glows all round, the keeper's cottage windows throw their light
out into the haze and the pier's lamps pour cones down onto it.

References:
- C. Kulla and M. Fajardo, "Importance Sampling Techniques for Path
  Tracing in Participating Media", Computer Graphics Forum
  31(4):1519-1528, 2012 (the beams' light along each pixel's ray,
  sampled equiangularly from the lamp).
"""

import math

import numpy as np

import ernie
import ernie.app
import ernie.demo
from ernie import tokens
from ernie.demo import write_png

RESOLUTION = (320, 200)
LAMP = (0.0, 0.0, 24.0)  # m, the lantern
HAZE = 0.5  # the haze's vertical optical depth at 550 nm
PERIOD = 10.0  # s, a turn of the beacon
# A lighthouse's beam: about 10^6 candela in a cone a few degrees
# across, from a halogen lamp behind a Fresnel lens.
LUMENS = 2e4
CONE = 3.0  # degrees, the beam's half angle
COTTAGE = (22.0, -28.0)  # m, the keeper's cottage
# The pier's lamps, along the shore in front of the tower.
PIER = [(-48.0 + 12.0 * i, -48.0) for i in range(4)]


def beam(angle):
    """The beacon's transform turned `angle` radians about the
    tower."""
    target = (
        LAMP[0] + math.cos(angle),
        LAMP[1] + math.sin(angle),
        LAMP[2] + 0.02,
    )
    return ernie.look_at_transform(LAMP, target, (0.0, 0.0, 1.0))


class Demo(ernie.demo.Demo):
    num_frames = 120

    def __init__(self, viewer, args):
        super().__init__(viewer, args)
        scene = self.scene
        world = scene.define("/World", ernie.Xform)
        looks = scene.define("/World/Looks", ernie.Xform)
        looks.define("Shore", ernie.Material, albedo=(0.08,) * 3)
        looks.define("Tower", ernie.Material, albedo=(0.8,) * 3)
        looks.define("Iron", ernie.Material, albedo=(0.05,) * 3)
        world.define(
            "Shore",
            ernie.GroundPlane,
            material_binding="/World/Looks/Shore",
        )
        self.air = world.define(
            "Air", ernie.Atmosphere, haze_optical_depth=HAZE
        )
        world.define(
            "Tower",
            ernie.Cylinder,
            radius=2.5,
            height=22.0,
            translate=(0.0, 0.0, 11.0),
            material_binding="/World/Looks/Tower",
        )
        # The lantern's bars, which cut shafts through the beams.
        for i in range(8):
            a = 2.0 * math.pi * (i + 0.5) / 8.0
            world.define(
                f"Bar{i}",
                ernie.Cylinder,
                radius=0.1,
                height=3.0,
                translate=(
                    1.4 * math.cos(a),
                    1.4 * math.sin(a),
                    24.0,
                ),
                material_binding="/World/Looks/Iron",
            )
        # A turn of the beacon is the timeline's range, which loops.
        stage = ernie.Stage(scene)
        stage.start_time_code = 0.0
        stage.end_time_code = PERIOD * stage.time_codes_per_second
        stage.time_code = 0.0
        self.angle = 0.0
        self.speed = 2.0 * math.pi / PERIOD
        self.phase = 0.0  # the beam's angle at time 0
        self.lamps = []
        for name in ("Beam", "Back"):
            self.lamps.append(
                world.define(
                    name,
                    ernie.SphereLight,
                    radius=0.3,
                    shaping_cone_angle=CONE,
                    shaping_cone_softness=0.3,
                    photometric_power=LUMENS,
                    enable_color_temperature=True,
                    color_temperature=3200.0,
                )
            )
        # The lantern glass's own glow, all round (a cylinder light, its
        # axis turned upright), which lights the tower's top and
        # haloes the lamp.
        self.lamps.append(
            world.define(
                "Glow",
                ernie.CylinderLight,
                radius=1.2,
                length=2.5,
                photometric_power=3000.0,
                enable_color_temperature=True,
                color_temperature=3200.0,
                translate=LAMP,
                rotate=(0.0, -90.0, 0.0),
            )
        )
        # The keeper's cottage, its two lit windows (rect lights)
        # facing the shore.
        looks.define("Cottage", ernie.Material, albedo=(0.5,) * 3)
        world.define(
            "Cottage",
            ernie.Cube,
            size=1.0,
            translate=(COTTAGE[0], COTTAGE[1], 2.5),
            scale=(8.0, 5.0, 5.0),
            material_binding="/World/Looks/Cottage",
        )
        for i, dx in enumerate((-2.0, 2.0)):
            window = (COTTAGE[0] + dx, COTTAGE[1] - 2.55, 2.5)
            self.lamps.append(
                world.define(
                    f"Window{i}",
                    ernie.RectLight,
                    width=1.0,
                    height=1.2,
                    photometric_power=1200.0,
                    enable_color_temperature=True,
                    color_temperature=2700.0,
                ).set(
                    tokens.TRANSFORM,
                    ernie.look_at_transform(
                        window,
                        (window[0], window[1] - 1.0, window[2]),
                        (0.0, 0.0, 1.0),
                    ),
                )
            )
        # The pier's lamps (disk lights) on their posts, facing down.
        for i, (x, y) in enumerate(PIER):
            world.define(
                f"Post{i}",
                ernie.Cylinder,
                radius=0.08,
                height=4.0,
                translate=(x, y, 2.0),
                material_binding="/World/Looks/Iron",
            )
            self.lamps.append(
                world.define(
                    f"PierLamp{i}",
                    ernie.DiskLight,
                    radius=0.2,
                    photometric_power=1500.0,
                    enable_color_temperature=True,
                    color_temperature=4000.0,
                ).set(
                    tokens.TRANSFORM,
                    ernie.look_at_transform(
                        (x, y, 4.1), (x, y, 0.0), (0.0, 1.0, 0.0)
                    ),
                )
            )
        self.turn(0.0)
        # A long exposure, as for the night sky.
        self.camera = world.define(
            "Camera",
            ernie.Camera,
            focal_length=18.0,
            sensor_noise=False,
            exposure_time=0.5,
            exposure_f_stop=2.0,
            exposure_iso=3200.0,
            clipping_range=(0.05, 1e5),
        ).set(
            tokens.TRANSFORM,
            ernie.look_at_transform(
                (-70.0, -100.0, 2.0),
                (0.0, 0.0, 26.0),
                (0.0, 0.0, 1.0),
            ),
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
        self.widgets.slider(
            "Haze optical depth",
            0.0,
            3.0,
            HAZE,
            on_change=self.thicken,
        )
        self.widgets.slider(
            "Turns per minute",
            0.0,
            30.0,
            60.0 / PERIOD,
            on_change=self.spin,
        )
        if viewer is not None:
            viewer.active_camera = self.camera.path
            viewer.lighting = ernie.app.Lighting.STAGE
            display = viewer.viewport_display
            display.lights = False
            display.grid = False
            viewer.viewport_display = display
            # The timeline turns the beacon: play, scrub or step it.
            viewer.on_frame(
                lambda viewer, info: self.turn_at(self.time())
            )
            viewer.play()

    def time(self):
        """Seconds into the timeline."""
        return ernie.Stage(self.scene).time

    def turn_at(self, t):
        """The beacon as it is `t` seconds into the timeline."""
        self.turn(self.phase + self.speed * t)

    def turn(self, angle):
        self.angle = angle
        self.lamps[0].set(tokens.TRANSFORM, beam(angle))
        self.lamps[1].set(tokens.TRANSFORM, beam(angle + math.pi))

    def thicken(self, depth):
        self.air.haze_optical_depth = depth

    def spin(self, turns):
        # From where the beam points now, not a jump.
        self.speed = 2.0 * math.pi * turns / 60.0
        self.phase = self.angle - self.speed * self.time()

    def step(self):
        if self.viewer is not None:
            return
        # Without a viewer, the demo plays the timeline itself.
        stage = ernie.Stage(self.scene)
        length = stage.end_time_code - stage.start_time_code
        stage.time_code = (
            stage.time_code
            + self.frame_dt * stage.time_codes_per_second
        ) % length
        self.turn_at(stage.time)

    def capture(self, lit=True):
        """The radiance summed over the bands, with the beacon lit or
        dark."""
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
        # The beam swinging past the camera's side, a quarter turn on.
        self.turn(math.radians(200.0))
        side = self.capture()
        self.turn(math.radians(290.0))
        turned = self.capture()
        self.thicken(2.0 * HAZE)
        hazier = self.capture()
        self.thicken(HAZE)
        dark = self.capture(lit=False)
        # The beams in the sky above the horizon, the frame's top half.
        sky = slice(0, RESOLUTION[1] // 2)
        lit = (side[sky] > 0.02 * side[sky].max()).mean()
        moved = np.abs(turned - side)[sky].sum() / side[sky].sum()
        ratio = hazier[sky].sum() / turned[sky].sum()
        print(
            f"the beams light {100 * lit:.1f}% of the sky; a quarter "
            f"turn changes {100 * moved:.0f}% of its light; twice "
            f"the haze scatters {ratio:.2f} times as much; dark, the "
            f"brightest pixel is {dark.max():.3g}"
        )
        assert lit > 0.02, "the beams show in the air"
        assert moved > 0.5, "they sweep round"
        assert ratio > 1.2, "haze shows them more"
        assert dark.max() == 0.0, "nothing else lights the night"


if __name__ == "__main__":
    ernie.demo.main(Demo)
