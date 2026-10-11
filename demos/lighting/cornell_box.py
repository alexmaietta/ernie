# SPDX-License-Identifier: BSD-3-Clause
"""The Cornell box from its measured geometry, wall reflectance
spectra and light spectrum: direct light only, as yet."""

import numpy as np

import ernie
import ernie.demo
from ernie import tokens

# The Cornell Program of Computer Graphics' measurements, in its
# coordinates (millimetres, y up) scaled to metres.
#
# References:
# - Cornell University Program of Computer Graphics, "Cornell Box
#   Data", 2005.
#   http://www.graphics.cornell.edu/online/box/data.html
# - C. M. Goral, K. E. Torrance, D. P. Greenberg and B. Battaile,
#   "Modeling the Interaction of Light Between Diffuse Surfaces",
#   SIGGRAPH 1984. https://doi.org/10.1145/964965.808601
MM = 0.001

# Reflectance at each wavelength (nm): white, green and red.
REFLECTANCE = [
    (400, 0.343, 0.092, 0.040),
    (404, 0.445, 0.096, 0.046),
    (408, 0.551, 0.098, 0.048),
    (412, 0.624, 0.097, 0.053),
    (416, 0.665, 0.098, 0.049),
    (420, 0.687, 0.095, 0.050),
    (424, 0.708, 0.095, 0.053),
    (428, 0.723, 0.097, 0.055),
    (432, 0.715, 0.095, 0.057),
    (436, 0.710, 0.094, 0.056),
    (440, 0.745, 0.097, 0.059),
    (444, 0.758, 0.098, 0.057),
    (448, 0.739, 0.096, 0.061),
    (452, 0.767, 0.101, 0.061),
    (456, 0.777, 0.103, 0.060),
    (460, 0.765, 0.104, 0.062),
    (464, 0.751, 0.107, 0.062),
    (468, 0.745, 0.109, 0.062),
    (472, 0.748, 0.112, 0.061),
    (476, 0.729, 0.115, 0.062),
    (480, 0.745, 0.125, 0.060),
    (484, 0.757, 0.140, 0.059),
    (488, 0.753, 0.160, 0.057),
    (492, 0.750, 0.187, 0.058),
    (496, 0.746, 0.229, 0.058),
    (500, 0.747, 0.285, 0.058),
    (504, 0.735, 0.343, 0.056),
    (508, 0.732, 0.390, 0.055),
    (512, 0.739, 0.435, 0.056),
    (516, 0.734, 0.464, 0.059),
    (520, 0.725, 0.472, 0.057),
    (524, 0.721, 0.476, 0.055),
    (528, 0.733, 0.481, 0.059),
    (532, 0.725, 0.462, 0.059),
    (536, 0.732, 0.447, 0.058),
    (540, 0.743, 0.441, 0.059),
    (544, 0.744, 0.426, 0.061),
    (548, 0.748, 0.406, 0.061),
    (552, 0.728, 0.373, 0.063),
    (556, 0.716, 0.347, 0.063),
    (560, 0.733, 0.337, 0.067),
    (564, 0.726, 0.314, 0.068),
    (568, 0.713, 0.285, 0.072),
    (572, 0.740, 0.277, 0.080),
    (576, 0.754, 0.266, 0.090),
    (580, 0.764, 0.250, 0.099),
    (584, 0.752, 0.230, 0.124),
    (588, 0.736, 0.207, 0.154),
    (592, 0.734, 0.186, 0.192),
    (596, 0.741, 0.171, 0.255),
    (600, 0.740, 0.160, 0.287),
    (604, 0.732, 0.148, 0.349),
    (608, 0.745, 0.141, 0.402),
    (612, 0.755, 0.136, 0.443),
    (616, 0.751, 0.130, 0.487),
    (620, 0.744, 0.126, 0.513),
    (624, 0.731, 0.123, 0.558),
    (628, 0.733, 0.121, 0.584),
    (632, 0.744, 0.122, 0.620),
    (636, 0.731, 0.119, 0.606),
    (640, 0.712, 0.114, 0.609),
    (644, 0.708, 0.115, 0.651),
    (648, 0.729, 0.117, 0.612),
    (652, 0.730, 0.117, 0.610),
    (656, 0.727, 0.118, 0.650),
    (660, 0.707, 0.120, 0.638),
    (664, 0.703, 0.122, 0.627),
    (668, 0.729, 0.128, 0.620),
    (672, 0.750, 0.132, 0.630),
    (676, 0.760, 0.139, 0.628),
    (680, 0.751, 0.144, 0.642),
    (684, 0.739, 0.146, 0.639),
    (688, 0.724, 0.150, 0.657),
    (692, 0.730, 0.152, 0.639),
    (696, 0.740, 0.157, 0.635),
    (700, 0.737, 0.159, 0.642),
]

# The light's emission spectrum at each wavelength (nm), relative.
EMISSION = [(400, 0.0), (500, 8.0), (600, 15.6), (700, 18.4)]

# Each surface's quadrilaterals, in millimetres, and its reflectance
# (a column of REFLECTANCE).
SURFACES = {
    "Floor": (
        "white",
        [
            [
                (552.8, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 559.2),
                (549.6, 0.0, 559.2),
            ]
        ],
    ),
    "Ceiling": (
        "white",
        [
            [
                (556.0, 548.8, 0.0),
                (556.0, 548.8, 559.2),
                (0.0, 548.8, 559.2),
                (0.0, 548.8, 0.0),
            ]
        ],
    ),
    "BackWall": (
        "white",
        [
            [
                (549.6, 0.0, 559.2),
                (0.0, 0.0, 559.2),
                (0.0, 548.8, 559.2),
                (556.0, 548.8, 559.2),
            ]
        ],
    ),
    "RightWall": (
        "green",
        [
            [
                (0.0, 0.0, 559.2),
                (0.0, 0.0, 0.0),
                (0.0, 548.8, 0.0),
                (0.0, 548.8, 559.2),
            ]
        ],
    ),
    "LeftWall": (
        "red",
        [
            [
                (552.8, 0.0, 0.0),
                (549.6, 0.0, 559.2),
                (556.0, 548.8, 559.2),
                (556.0, 548.8, 0.0),
            ]
        ],
    ),
    "ShortBlock": (
        "white",
        [
            [
                (130.0, 165.0, 65.0),
                (82.0, 165.0, 225.0),
                (240.0, 165.0, 272.0),
                (290.0, 165.0, 114.0),
            ],
            [
                (290.0, 0.0, 114.0),
                (290.0, 165.0, 114.0),
                (240.0, 165.0, 272.0),
                (240.0, 0.0, 272.0),
            ],
            [
                (130.0, 0.0, 65.0),
                (130.0, 165.0, 65.0),
                (290.0, 165.0, 114.0),
                (290.0, 0.0, 114.0),
            ],
            [
                (82.0, 0.0, 225.0),
                (82.0, 165.0, 225.0),
                (130.0, 165.0, 65.0),
                (130.0, 0.0, 65.0),
            ],
            [
                (240.0, 0.0, 272.0),
                (240.0, 165.0, 272.0),
                (82.0, 165.0, 225.0),
                (82.0, 0.0, 225.0),
            ],
        ],
    ),
    "TallBlock": (
        "white",
        [
            [
                (423.0, 330.0, 247.0),
                (265.0, 330.0, 296.0),
                (314.0, 330.0, 456.0),
                (472.0, 330.0, 406.0),
            ],
            [
                (423.0, 0.0, 247.0),
                (423.0, 330.0, 247.0),
                (472.0, 330.0, 406.0),
                (472.0, 0.0, 406.0),
            ],
            [
                (472.0, 0.0, 406.0),
                (472.0, 330.0, 406.0),
                (314.0, 330.0, 456.0),
                (314.0, 0.0, 456.0),
            ],
            [
                (314.0, 0.0, 456.0),
                (314.0, 330.0, 456.0),
                (265.0, 330.0, 296.0),
                (265.0, 0.0, 296.0),
            ],
            [
                (265.0, 0.0, 296.0),
                (265.0, 330.0, 296.0),
                (423.0, 330.0, 247.0),
                (423.0, 0.0, 247.0),
            ],
        ],
    ),
}

# The light: a 130 x 105 mm opening in the ceiling, centred over
# (278, 279.5), a hair below it so the two do not overlap.
LIGHT_CENTRE = (278.0, 548.7, 279.5)
LIGHT_SIZE = (130.0, 105.0)
LUMENS = 300.0
# The opening as the camera sees it: a panel just above the light,
# glowing at the luminance of a Lambertian emitter of its size and
# flux, L = flux / (pi area), in cd/m^2.
PANEL_LUMINANCE = LUMENS / (
    np.pi * LIGHT_SIZE[0] * LIGHT_SIZE[1] * MM**2
)


def pairs(column):
    """(nm, value) pairs of REFLECTANCE's column `column`, flat."""
    return [v for row in REFLECTANCE for v in (row[0], row[column])]


class Demo(ernie.demo.Demo):
    num_frames = 1

    def __init__(self, viewer, args):
        super().__init__(viewer, args)
        world = self.scene.define("/World", ernie.Xform)
        looks = self.scene.define("/Looks", ernie.Scope)
        materials = {
            name: looks.define(
                name.capitalize(),
                ernie.Material,
                reflectance=pairs(column),
            )
            for column, name in enumerate(
                ("white", "green", "red"), start=1
            )
        }
        for name, (look, quads) in SURFACES.items():
            points = np.array(quads, dtype=np.float64).reshape(-1, 3)
            world.define(
                name,
                ernie.Mesh,
                points=points * MM,
                face_vertex_counts=np.full(len(quads), 4),
                face_vertex_indices=np.arange(4 * len(quads)),
                material_binding=materials[look],
            )
        centre = np.array(LIGHT_CENTRE) * MM
        glow = looks.define(
            "Glow",
            ernie.Material,
            emissive_color=(PANEL_LUMINANCE,) * 3,
        )
        x0, x1 = 213.0, 343.0
        z0, z1 = 227.0, 332.0
        y = 548.75
        world.define(
            "Panel",
            ernie.Mesh,
            points=np.array(
                [(x1, y, z0), (x1, y, z1), (x0, y, z1), (x0, y, z0)]
            )
            * MM,
            face_vertex_counts=np.array([4]),
            face_vertex_indices=np.arange(4),
            material_binding=glow,
        )
        light = world.define(
            "Light",
            ernie.RectLight,
            width=LIGHT_SIZE[0] * MM,
            height=LIGHT_SIZE[1] * MM,
            physical_illuminant=tokens.CUSTOM,
            physical_custom_illuminant=[
                v for pair in EMISSION for v in pair
            ],
            photometric_power=LUMENS,
            shadow_enable=True,
        )
        # Facing down, its width along x and height along z.
        light.set(
            tokens.TRANSFORM,
            ernie.look_at_transform(
                tuple(centre),
                tuple(centre - (0.0, 1.0, 0.0)),
                up=(0.0, 0.0, 1.0),
            ),
        )
        # The published camera: at (278, 273, -800) looking along +z,
        # 35 mm on a 25 mm square film.
        camera = world.define(
            "Camera",
            ernie.Camera,
            focal_length=35.0,
            horizontal_aperture=25.0,
            vertical_aperture=25.0,
            # The box's front is 0.8 m away.
            clipping_range=(0.01, 10.0),
            sensor_noise=False,
            exposure_time=1.0 / 30.0,
            exposure_iso=200.0,
            exposure_f_stop=2.8,
        ).set(
            tokens.TRANSFORM,
            ernie.look_at_transform(
                (0.278, 0.273, -0.8),
                (0.278, 0.273, 0.0),
                up=(0.0, 1.0, 0.0),
            ),
        )
        self.scene.define(
            "/Render/Settings",
            ernie.RenderSettings,
            camera=camera,
            resolution=(512, 512),
            ambient=0.0,
        )
        if viewer is not None:
            viewer.active_camera = camera.path

    def test_final(self):
        image = self.render_image()[..., :3].astype(np.float64)
        rows, cols, _ = image.shape
        middle = slice(rows // 3, 2 * rows // 3)
        # The red wall fills the image's left edge, the green its
        # right.
        left = image[middle, cols // 32 : cols // 10].mean(
            axis=(0, 1)
        )
        right = image[middle, -cols // 10 : -cols // 32].mean(
            axis=(0, 1)
        )
        assert left[0] > 1.5 * max(left[1], left[2]), (
            f"the left wall should be red, not {left}"
        )
        assert right[1] > 1.2 * max(right[0], right[2]), (
            f"the right wall should be green, not {right}"
        )
        assert image.std() > 10.0, "the box should not render flat"


if __name__ == "__main__":
    ernie.demo.main(Demo)
