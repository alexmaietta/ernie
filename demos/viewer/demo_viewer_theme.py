# SPDX-License-Identifier: BSD-3-Clause
"""The viewer restyled while it runs: a light, a dark and a
high-contrast theme, then the accent, background and text colours,
text size, corner rounding and spacing each changed live. The Demos
panel puts the viewer's own theme back when another demo starts.

References:
- W3C, "Web Content Accessibility Guidelines (WCAG) 2.1", 2018,
  success criteria 1.4.3 and 1.4.6 (contrast ratios).
"""

import ernie
import ernie.demo
from ernie import tokens
from ernie.app import Theme

# Each preset's colours (sRGB, alpha last) and metrics, over the
# viewer's light default.
DARK = {
    "background": (0.13, 0.14, 0.15, 1.0),
    "panel": (0.17, 0.18, 0.19, 1.0),
    "header": (0.22, 0.23, 0.25, 1.0),
    "field": (0.10, 0.11, 0.12, 1.0),
    "field_hovered": (0.14, 0.15, 0.17, 1.0),
    "button": (0.24, 0.26, 0.28, 1.0),
    "button_lit": (0.25, 0.38, 0.55, 1.0),
    "separator": (0.28, 0.30, 0.32, 1.0),
    "border": (0.33, 0.35, 0.37, 1.0),
    "text": (0.90, 0.91, 0.92, 1.0),
    "text_muted": (0.62, 0.64, 0.66, 1.0),
    "selection": (0.22, 0.33, 0.48, 1.0),
    "accent": (0.38, 0.62, 0.95, 1.0),
    "overlay": (0.17, 0.18, 0.19, 0.9),
    "overlay_border": (0.33, 0.35, 0.37, 1.0),
    "check_box": (0.10, 0.11, 0.12, 1.0),
    "check_mark": (0.90, 0.91, 0.92, 1.0),
    "rounding": 4.0,
    "pill_rounding": 8.0,
}
HIGH_CONTRAST = {
    "background": (1.0, 1.0, 1.0, 1.0),
    "panel": (1.0, 1.0, 1.0, 1.0),
    "header": (0.85, 0.85, 0.85, 1.0),
    "field": (1.0, 1.0, 1.0, 1.0),
    "field_hovered": (0.92, 0.92, 0.92, 1.0),
    "button": (0.90, 0.90, 0.90, 1.0),
    "button_lit": (0.70, 0.80, 1.0, 1.0),
    "separator": (0.0, 0.0, 0.0, 1.0),
    "border": (0.0, 0.0, 0.0, 1.0),
    "text": (0.0, 0.0, 0.0, 1.0),
    "text_muted": (0.25, 0.25, 0.25, 1.0),
    "selection": (0.75, 0.85, 1.0, 1.0),
    "accent": (0.0, 0.25, 0.75, 1.0),
    "overlay": (1.0, 1.0, 1.0, 0.95),
    "overlay_border": (0.0, 0.0, 0.0, 1.0),
    "check_box": (1.0, 1.0, 1.0, 1.0),
    "check_mark": (0.0, 0.0, 0.0, 1.0),
    "frame_border": 2.0,
    "font_size": 16.0,
}
PRESETS = {"Light": {}, "Dark": DARK, "High contrast": HIGH_CONTRAST}
# The contrast each preset's text needs against its panels: WCAG's AA
# for body text, AAA for the high-contrast theme, and 3 for muted
# labels.
CONTRAST = {"Light": 4.5, "Dark": 4.5, "High contrast": 7.0}
MUTED_CONTRAST = 3.0


def themed(preset):
    """The viewer's default theme with `preset`'s fields."""
    theme = Theme()
    for name, value in PRESETS[preset].items():
        setattr(theme, name, value)
    return theme


def luminance(rgb):
    """WCAG's relative luminance of an sRGB colour."""

    def linear(c):
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = (linear(c) for c in rgb[:3])
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(a, b):
    """WCAG's contrast ratio of two sRGB colours, 1 to 21."""
    high, low = sorted((luminance(a), luminance(b)), reverse=True)
    return (high + 0.05) / (low + 0.05)


class Demo(ernie.demo.Demo):
    num_frames = 1

    def __init__(self, viewer, args):
        super().__init__(viewer, args)
        world = self.scene.define("/World", ernie.Xform)
        world.define("Ground", ernie.GroundPlane, checker=True)
        for i, colour in enumerate(
            [(0.8, 0.3, 0.2), (0.2, 0.6, 0.3), (0.2, 0.4, 0.8)]
        ):
            world.define(
                f"Ball{i}",
                ernie.Sphere,
                radius=0.6,
                translate=(1.6 * (i - 1), 0.0, 0.6),
                display_color=colour,
            )
        world.define("Sun", ernie.DistantLight).set(
            tokens.TRANSFORM,
            ernie.distant_light_transform((0.4, -0.6, 0.7)),
        )
        world.define("Camera", ernie.Camera, focal_length=24.0).set(
            tokens.TRANSFORM,
            ernie.look_at_transform((0.0, -7.0, 3.0), (0.0, 0.0, 0.6)),
        )
        self.preset = "Light"
        self.theme = themed(self.preset)
        w = self.widgets
        presets = list(PRESETS)
        w.combo(
            "Preset",
            presets,
            0,
            on_change=lambda index: self.choose(presets[index]),
        )
        w.separator()
        self.accent = w.color(
            "Accent", self.theme.accent[:3],
            on_change=lambda rgb: self.paint("accent", rgb),
        )  # fmt: skip
        self.background = w.color(
            "Background", self.theme.panel[:3],
            on_change=lambda rgb: self.paint("panel", rgb),
        )  # fmt: skip
        self.text = w.color(
            "Text", self.theme.text[:3],
            on_change=lambda rgb: self.paint("text", rgb),
        )  # fmt: skip
        self.font = w.slider(
            "Text size", 10.0, 22.0, self.theme.font_size,
            on_change=lambda size: self.restyle("font_size", size),
        )  # fmt: skip
        self.rounding = w.slider(
            "Rounding", 0.0, 12.0, self.theme.rounding,
            on_change=lambda radius: self.restyle("rounding", radius),
        )  # fmt: skip
        self.pill = w.slider(
            "Toolbar rounding", 0.0, 16.0, self.theme.pill_rounding,
            on_change=lambda radius: self.restyle("pill_rounding", radius),
        )  # fmt: skip
        self.spacing = w.slider(
            "Spacing", 2.0, 12.0, self.theme.item_spacing[0],
            on_change=lambda gap: self.restyle("item_spacing", (gap, gap)),
        )  # fmt: skip
        self.contrast = w.text("")
        self.apply()

    def choose(self, preset):
        """Starts again from `preset`, its controls showing it."""
        self.preset = preset
        self.theme = themed(preset)
        w = self.widgets
        w.set(self.accent, self.theme.accent[:3])
        w.set(self.background, self.theme.panel[:3])
        w.set(self.text, self.theme.text[:3])
        w.set(self.font, self.theme.font_size)
        w.set(self.rounding, self.theme.rounding)
        w.set(self.pill, self.theme.pill_rounding)
        w.set(self.spacing, self.theme.item_spacing[0])
        self.apply()

    def paint(self, name, rgb):
        alpha = getattr(self.theme, name)[3]
        setattr(self.theme, name, (*rgb, alpha))
        # The background colours the docks behind the panels too.
        if name == "panel":
            self.theme.background = (*rgb, alpha)
        self.apply()

    def restyle(self, name, value):
        setattr(self.theme, name, value)
        self.apply()

    def apply(self):
        """Shows the theme in the viewer, and its text's contrast."""
        ratio = contrast(self.theme.text, self.theme.panel)
        self.widgets.set(self.contrast, f"Text contrast {ratio:.1f}:1")
        if self.viewer is not None:
            self.viewer.theme = self.theme

    def test_final(self):
        for preset in PRESETS:
            theme = themed(preset)
            text = contrast(theme.text, theme.panel)
            muted = contrast(theme.text_muted, theme.panel)
            print(f"{preset}: text {text:.1f}:1, muted {muted:.1f}:1")
            assert text >= CONTRAST[preset], f"{preset}'s text should read"
            assert muted >= MUTED_CONTRAST, f"{preset}'s labels should read"
        # Changing the accent keeps the rest of the theme (single
        # precision, as the viewer stores it).
        def near(a, b):
            return all(abs(x - y) < 1e-6 for x, y in zip(a, b))

        self.choose("Dark")
        self.paint("accent", (0.9, 0.5, 0.1))
        assert near(self.theme.accent, (0.9, 0.5, 0.1, 1.0))
        assert near(self.theme.panel, DARK["panel"])


if __name__ == "__main__":
    ernie.demo.main(Demo)
