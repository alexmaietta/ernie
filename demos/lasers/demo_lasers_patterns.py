# SPDX-License-Identifier: BSD-3-Clause
"""A 532 nm projector over a table through each of its patterns, one
of them written in Slang at run time."""

from pathlib import Path

import numpy as np

import ernie
import ernie.demo
from ernie import tokens
from ernie.demo import cache_dir, side_by_side, write_png

PATTERNS = ["spot", "line", "grid", "dots", "fringes", "rings"]

# Concentric rings, as a diffractive axicon or a ring light makes: one
# ring every `period` of tan off the axis, each a thin line as wide as
# the pixel's footprint and the beam, any scale (Ernie normalises it
# over its extent).
RINGS_SLANG = """
import light_pattern;

public struct Rings : ILightPattern {
    public float emission(LightPatternInput input, LightPatternParams p) {
        let period = p.get(0);
        let r = length(input.t);
        let off = r - period * round(r / period);
        let width = max(length(input.footprint), 2.0 * max(input.beam.x, input.beam.y));
        return r > 0.5 * period && abs(off) < 0.5 * width ? 1.0 / width : 0.0;
    }
}
"""


def add_rings(target):
    """Registers the rings with `target` (a Renderer or Viewer)."""
    if not target.has_light_pattern("rings"):
        target.add_light_pattern(
            "rings",
            "Rings",
            RINGS_SLANG,
            params=[
                {
                    "name": "period",
                    "default": 0.05,
                    "description": "tan between rings",
                }
            ],
            extent=0.4,
            name="Rings",
            description="Concentric rings every `period` of tan.",
        )


def fringes(periods):
    """A PNG of vertical sinusoidal fringes, `periods` across 256 x 16
    texels, written to the demos' cache; returns its path."""
    cols = (np.arange(256) + 0.5) / 256
    row = np.rint(
        255 * (0.5 + 0.5 * np.cos(2 * np.pi * periods * cols))
    )
    image = np.repeat(
        np.tile(row.astype(np.uint8), (16, 1))[:, :, None], 3, axis=2
    )
    path = Path(cache_dir()) / "laser_fringes.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    write_png(path, image)
    return path


def lit_by(image, dark):
    """What the laser lights in `image` against `dark`, the scene
    without it: pixels whose green rose by more than 25, as the
    fraction of the image and its extent's width over its height."""
    lit = image[:, :, 1].astype(int) - dark[:, :, 1].astype(int) > 25
    if not lit.any():
        return 0.0, 0.0
    rows, cols = np.nonzero(lit)
    aspect = (cols.max() - cols.min() + 1) / (
        rows.max() - rows.min() + 1
    )
    return float(lit.mean()), float(aspect)


class Demo(ernie.demo.Demo):
    num_frames = 1

    def __init__(self, viewer, args):
        super().__init__(viewer, args)
        self.fringes = fringes(12)
        # The custom pattern, with whichever renderers draw the scene.
        add_rings(self.renderer)
        if viewer is not None:
            add_rings(viewer)
        world = self.scene.define("/World", ernie.Xform)
        world.define(
            "Floor",
            ernie.GroundPlane,
            checker=True,
            checker_color0=(0.6,) * 3,
            checker_color1=(0.5,) * 3,
        )
        for name, centre, size, grey in (
            ("Wall", (0.0, 1.5, 1.0), (4.0, 0.1, 2.0), 0.7),
            ("Block", (-0.5, 0.3, 0.2), (0.4, 0.4, 0.4), 0.75),
        ):
            world.define(
                name,
                ernie.Cube,
                size=1.0,
                display_color=(grey,) * 3,
                translate=centre,
                scale=size,
            )
        world.define(
            "Ball",
            ernie.Sphere,
            radius=0.25,
            display_color=(0.8,) * 3,
            translate=(0.45, 0.2, 0.25),
        )
        # A dim room light.
        world.define(
            "Lamp",
            ernie.SphereLight,
            radius=0.1,
            photometric_power=300.0,
        ).set(
            tokens.TRANSFORM,
            ernie.look_at_transform(
                (-1.5, -1.0, 2.5), (0.0, 0.0, 0.0)
            ),
        )
        # The projector 2 m up and back, looking down onto the table:
        # 30 mW at 532 nm, 1 mm waist.
        self.laser = world.define(
            "Projector",
            ernie.LaserLight,
            wavelength=532.0,
            power=0.03,
            waist_radius=(0.001, 0.001),
        ).set(
            tokens.TRANSFORM,
            ernie.look_at_transform(
                (0.0, -1.6, 2.2), (0.0, 0.4, 0.2)
            ),
        )
        # Exposed for the patterns' tens of lux (30 mW of green is
        # about 18 lumens), the same with the laser on or off.
        camera = world.define(
            "Camera",
            ernie.Camera,
            focal_length=20.0,
            sensor_noise=False,
            exposure_iso=800.0,
            exposure_f_stop=2.0,
            exposure_time=1.0 / 30.0,
        ).set(
            tokens.TRANSFORM,
            ernie.look_at_transform(
                (1.6, -2.6, 1.7), (0.0, 0.4, 0.3)
            ),
        )
        self.scene.define(
            "/Render/Settings",
            ernie.RenderSettings,
            camera=camera,
            resolution=(448, 300),
            ambient=0.0,
        )
        self.show(0)
        self.widgets.combo(
            "Pattern", PATTERNS, 0, on_change=self.show
        )
        if viewer is not None:
            viewer.active_camera = camera.path

    def show(self, index):
        self.index = index
        laser = self.laser
        laser.pattern = PATTERNS[index]
        laser.zero_order = 0.0
        field = (50.0, 40.0)
        if index == 1:
            # A flat 50 degree line from a Powell lens.
            laser.fan_angle = 50.0
            laser.line_profile = "flat"
        elif index == 2:
            # A 9 x 9 diffractive grid, 3 degrees apart.
            laser.set(
                grid_columns=9,
                grid_rows=9,
                grid_pitch=(3.0, 3.0),
                efficiency=0.85,
                zero_order=0.05,
            )
        elif index == 3:
            # 40 x 30 dots in 3 x 3 tiles, as a depth camera's.
            laser.set(
                dot_columns=40,
                dot_rows=30,
                dot_tile_columns=3,
                dot_tile_rows=3,
                pattern_field_of_view=field,
                efficiency=0.8,
            )
        elif index == 4:
            # Fringes from an image, as a phase-shift projector's.
            laser.set(
                pattern="image",
                pattern_image=str(self.fringes),
                pattern_field_of_view=field,
            )
        elif index == 5:
            # Rings, a pattern registered at run time.
            laser.pattern_params = [0.05]

    def render_all(self):
        """Every pattern's image in turn, RGBA uint8; leaves the pattern
        as it was."""
        current = self.index
        images = []
        for index in range(len(PATTERNS)):
            self.show(index)
            images.append(np.array(self.renderer.render(self.scene)))
        self.show(current)
        return images

    def save_output(self, path):
        """Every pattern side by side."""
        write_png(path, side_by_side(self.render_all()))

    def test_final(self):
        self.laser.power = 0.0
        dark = np.array(self.renderer.render(self.scene))
        self.laser.power = 0.03
        lit, aspect = zip(
            *(lit_by(image, dark) for image in self.render_all())
        )
        for name, fraction, ratio in zip(PATTERNS, lit, aspect):
            print(
                f"{name}: {100 * fraction:.2f}% of the image lit, "
                f"{ratio:.1f} wide to tall"
            )
            assert fraction > 0.0, (
                f"the {name} pattern should light the scene"
            )
        assert lit[0] < 0.01, "the bare beam should light a spot"
        for name, fraction in zip(PATTERNS[1:], lit[1:]):
            assert fraction > lit[0], (
                f"the {name} pattern should light more than the spot"
            )
        assert aspect[1] > 2.0, "the line should be wider than tall"
        assert lit[4] > lit[2], (
            "the fringes should light more than the grid"
        )


if __name__ == "__main__":
    ernie.demo.main(Demo)
