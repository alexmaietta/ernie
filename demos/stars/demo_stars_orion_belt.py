# SPDX-License-Identifier: BSD-3-Clause
"""A zoom from Orion onto Alnitak in its belt, then a tour of how optics
spread a star: diffraction, seeing, defocus, colour, glare, ghosts and
blooming. Hipparcos's stars once fetched; offline, Orion's brightest
among made-up ones."""

import math
import os

import numpy as np

import ernie
import ernie.demo
from ernie import tokens
from ernie.demo import (
    AsyncDownload,
    cache_dir,
    can_download,
    write_png,
)

WIDTH, HEIGHT = 960, 640
PITCH_MM = 0.002  # 2 um pixels
WIDE, BELT_FOCAL, LONG = 3.5, 30.0, 6000.0  # millimetres
# A 167 mm aperture: f/12 up to 2000 mm, then a 3x Barlow to f/36.
APERTURE = 2000.0 / 12.0
ZOOM_SECONDS = 10.0
EFFECT_SECONDS = 3.0
# Seconds keeping a star's peak, its light over its Airy disc's area,
# (f / N)^2 / N^2, as at 2000 mm f/12.
EXPOSURE = 0.002

ORION = (83.5, 0.1)  # right ascension and declination, degrees
BELT = (84.1, -1.15)
ALNITAK = (85.1897, -1.9426)

HIPPARCOS_URL = (
    "https://vizier.cds.unistra.fr/viz-bin/asu-tsv?-out.max=unlimited"
    "&-source=I/239/hip_main&-out=RAICRS,DEICRS,Vmag,B-V"
)

# Orion's brightest stars (J2000 degrees, V, B-V), for running offline.
ORION_STARS = [
    (88.7929, 7.4071, 0.42, 1.85),  # Betelgeuse
    (78.6345, -8.2016, 0.13, -0.03),  # Rigel
    (81.2828, 6.3497, 1.64, -0.22),  # Bellatrix
    (84.0534, -1.2019, 1.69, -0.18),  # Alnilam
    (85.1897, -1.9426, 1.77, -0.21),  # Alnitak
    (83.0017, -0.2991, 2.23, -0.22),  # Mintaka
    (86.9391, -9.6696, 2.09, -0.17),  # Saiph
    (83.7845, 9.9342, 3.39, -0.16),  # Meissa
    (83.8583, -5.9099, 2.77, -0.24),  # Hatysa
    (81.1192, -2.3971, 3.36, -0.17),  # Eta Orionis
    (84.6865, -2.6000, 3.81, -0.24),  # Sigma Orionis
    (83.8186, -5.3897, 5.13, 0.02),  # Theta1 Orionis C
]

# An achromat's secondary spectrum: its focal shift, a fraction of f, by
# wavelength (C and F focus together, violet focuses long).
ACHROMAT = [
    (nm, 3.5e-4 * ((nm - 570.0) / 100.0) ** 2)
    for nm in range(380, 790, 30)
]
# Lateral colour: the image's scale by wavelength.
LATERAL = [400.0, 1.008, 550.0, 1.0, 700.0, 0.995]


def offline_stars():
    """Orion's brightest among made-up stars to V 9."""
    made_up = ernie.synthetic_stars(40000, seed=7, faintest=9.0)
    real = np.asarray(ORION_STARS, dtype=np.float32)
    return np.concatenate(
        [
            real,
            np.stack(
                [
                    np.asarray(made_up[key], dtype=np.float32)
                    for key in (
                        "right_ascension",
                        "declination",
                        "magnitude",
                        "color_index",
                    )
                ],
                axis=1,
            ),
        ]
    )


def read_vizier(path):
    """A VizieR table of right ascension, declination, V and B-V, rows
    without a magnitude left out."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                continue
            fields = line.split("\t")
            try:
                ra, dec, v = (float(x) for x in fields[:3])
            except ValueError:
                continue  # headers, units and blank magnitudes
            try:
                bv = float(fields[3])
            except (IndexError, ValueError):
                bv = 0.65
            rows.append((ra, dec, v, bv))
    return np.asarray(rows, dtype=np.float32)


def towards(ra, dec):
    a, d = math.radians(ra), math.radians(dec)
    return np.array(
        [
            math.cos(d) * math.cos(a),
            math.cos(d) * math.sin(a),
            math.sin(d),
        ]
    )


# Ghosts of the saturated stars alone.
FLARE = ("flare", {"strength": 0.1, "threshold": 15000.0})


def seeing_sigma(arcseconds, focal):
    """Seeing's Gaussian sigma in pixels at focal length `focal` mm."""
    return arcseconds / 2.3548 / 206265.0 * focal / PITCH_MM


def effects():
    """Each effect: name, focal length, target, filter chain (type,
    inputs), the var shown and the camera's changes."""
    psf = ("psf", {"profile": 1.0, "maxRadius": 24.0})
    photo: list[tuple[str, dict[str, float]]] = [
        ("cie", {}),
        ("exposure", {}),
    ]
    return [
        ("Points, no optics", LONG, ALNITAK, photo, "LdrColor", {}),
        (
            "Gaussian fit to the Airy disc",
            LONG,
            ALNITAK,
            [("psf", {"profile": 0.0, "maxRadius": 24.0}), *photo],
            "LdrColor",
            {},
        ),
        (
            "Airy pattern",
            LONG,
            ALNITAK,
            [psf, *photo],
            "LdrColor",
            {},
        ),
        (
            "Airy pattern, 33% central obstruction",
            LONG,
            ALNITAK,
            [
                (
                    "psf",
                    {
                        "profile": 1.0,
                        "obstruction": 0.33,
                        "maxRadius": 24.0,
                    },
                ),
                *photo,
            ],
            "LdrColor",
            {},
        ),
        (
            "Seeing, 1 arcsecond",
            LONG,
            ALNITAK,
            [
                (
                    "psf",
                    {
                        "profile": 0.0,
                        "blur": seeing_sigma(1.0, LONG),
                        "maxRadius": 24.0,
                    },
                ),
                *photo,
            ],
            "LdrColor",
            {},
        ),
        (
            "Defocused",
            LONG,
            ALNITAK,
            [psf, ("defocus", {"maxRadius": 16.0}), *photo],
            "LdrColor",
            {"focus_distance": 20000.0},
        ),
        (
            "Axial colour, an achromat",
            1000.0,
            ALNITAK,
            [psf, ("defocus", {"maxRadius": 32.0}), *photo],
            "LdrColor",
            {"achromat": True, "stops": 3.0},
        ),
        ("The belt", BELT_FOCAL, BELT, [psf, *photo], "LdrColor", {}),
        (
            "Lateral colour",
            BELT_FOCAL,
            BELT,
            [psf, *photo],
            "LdrColor",
            {"lateral": True},
        ),
        (
            "Glare",
            BELT_FOCAL,
            BELT,
            [
                psf,
                ("cie", {}),
                ("glare", {"strength": 0.3}),
                ("exposure", {}),
            ],
            "LdrColor",
            {"stops": 4.0},
        ),
        (
            "Lens flare ghosts",
            BELT_FOCAL,
            BELT,
            [psf, ("qe", {}), FLARE, ("adc", {})],
            "RawDN",
            {"stops": 10.0},
        ),
        (
            "Blooming, a CCD",
            BELT_FOCAL,
            BELT,
            [psf, ("qe", {}), ("blooming", {}), ("adc", {})],
            "RawDN",
            {"stops": 10.0},
        ),
    ]


EFFECTS = effects()


class Demo(ernie.demo.Demo):
    num_frames = 120

    def __init__(self, viewer, args):
        super().__init__(viewer, args)
        # The whole tour over a run without a window.
        self.fixed_dt = (
            ZOOM_SECONDS + EFFECT_SECONDS * len(EFFECTS)
        ) / self.num_frames
        world = self.scene.define("/World", ernie.Xform)
        self.stars = world.define("Stars", ernie.StarField)
        self.set_stars(offline_stars())
        self.camera = world.define(
            "Camera",
            ernie.Camera,
            horizontal_aperture=WIDTH * PITCH_MM,
            vertical_aperture=HEIGHT * PITCH_MM,
            exposure_iso=100.0,
            spectral_bands=16,
            sensor_noise=False,
            sensor_type="rgb",
            sensor_adc_bits=8,
        )
        self.product = self.scene.define(
            "/Render/Product",
            ernie.RenderProduct,
            camera=self.camera,
            resolution=(WIDTH, HEIGHT),
        )
        for var in ("LdrColor", "RawDN"):
            self.scene.define(
                f"/Render/Product/{var}",
                ernie.RenderVar,
                source_name=var,
            )
        self.scene.define(
            "/Render/Settings",
            ernie.RenderSettings,
            camera=self.camera,
            resolution=(WIDTH, HEIGHT),
            products=[ernie.Path("/Render/Product")],
        )
        self.chain_length = 0
        self.shown = None
        self.tour = True
        self.zoom(0.0)

        names = [effect[0] for effect in EFFECTS]
        self.widgets.checkbox("Tour", True, on_change=self.set_tour)
        self.widgets.combo("Effect", names, 0, on_change=self.pick)
        if viewer is not None:
            viewer.active_camera = self.camera.path

        # Hipparcos when cached; else fetched in the background, but
        # not for the checks, which run offline.
        self.catalogue = os.path.join(
            cache_dir(), "stars", "hipparcos.tsv"
        )
        self.download = None
        if not args.test:
            if os.path.exists(self.catalogue):
                self.load()
            elif can_download():
                self.download = AsyncDownload(
                    HIPPARCOS_URL, self.catalogue
                )

    def set_stars(self, table):
        self.stars.right_ascension = table[:, 0].copy()
        self.stars.declination = table[:, 1].copy()
        self.stars.magnitude = table[:, 2].copy()
        self.stars.color_index = table[:, 3].copy()

    def report(self, text, error=False):
        if self.viewer is not None:
            self.viewer.log(text, "error" if error else "info")
        else:
            print(text)

    def load(self):
        try:
            table = read_vizier(self.catalogue)
            self.set_stars(table)
            self.report(f"Loaded {len(table)} Hipparcos stars")
        except (OSError, ValueError) as error:
            self.report(str(error), True)

    def set_tour(self, on):
        self.tour = on

    def pick(self, index):
        self.tour = False
        self.apply(index)

    def aim(self, focal, ra_dec):
        n = max(12.0, focal / APERTURE)
        self.camera.focal_length = focal
        self.camera.f_stop = n
        self.camera.exposure_f_stop = n
        self.camera.exposure_time = (
            EXPOSURE * (n * n / focal / 0.072) ** 2
        )
        self.camera.set(
            tokens.TRANSFORM,
            ernie.look_at_transform(
                (0.0, 0.0, 0.0),
                tuple(towards(*ra_dec)),
                (0.0, 0.0, 1.0),
            ),
        )

    def chain(self, steps, var):
        """The product's filters and the var shown; a raw chain keeps
        LdrColor too, for frames rendered as photographs."""
        for i in range(self.chain_length):
            self.scene.remove(
                ernie.Path(f"/Render/Product/Chain/Step{i}")
            )
        if var == "RawDN":
            steps = [*steps, ("cie", {}), ("exposure", {})]
        self.var = var
        filters = []
        for i, (kind, inputs) in enumerate(steps):
            prim = self.scene.define(
                f"/Render/Product/Chain/Step{i}",
                ernie.SensorFilter,
                filter_type=kind,
            )
            for name, value in inputs.items():
                prim.set(f"inputs:{name}", float(value))
            filters.append(ernie.Path(prim.path))
        self.chain_length = len(steps)
        self.product.set(tokens.FILTERS, filters).set(
            tokens.ORDERED_VARS,
            [
                ernie.Path(f"/Render/Product/{v}")
                for v in sorted({"LdrColor", var})
            ],
        )
        if self.viewer is not None:
            self.viewer.view_var = var

    def lens(self, focal, achromat=False, lateral=False, focus=0.0):
        self.camera.focus_distance = focus
        self.camera.lens_chromatic_focus = (
            [v for nm, shift in ACHROMAT for v in (nm, shift * focal)]
            if achromat
            else [550.0, 0.0]
        )
        self.camera.lens_chromatic_scale = (
            LATERAL if lateral else [550.0, 1.0]
        )

    def zoom(self, u):
        """The zoom at `u` of the way: focal length growing
        exponentially, the aim moving from Orion to the belt to
        Alnitak."""
        if self.shown != "zoom":
            self.chain(
                [
                    ("psf", {"profile": 1.0}),
                    ("cie", {}),
                    ("exposure", {}),
                ],
                "LdrColor",
            )
            self.lens(WIDE)
            self.shown = "zoom"
        e = u * u * (3.0 - 2.0 * u)
        focal = WIDE * (LONG / WIDE) ** e
        k = math.log(focal / WIDE) / math.log(BELT_FOCAL / WIDE)
        if k <= 1.0:
            a, b, w = ORION, BELT, k
        else:
            a, b = BELT, ALNITAK
            w = min(
                math.log(focal / BELT_FOCAL)
                / math.log(400.0 / BELT_FOCAL),
                1.0,
            )
        d = (1.0 - w) * towards(*a) + w * towards(*b)
        ra = math.degrees(math.atan2(d[1], d[0])) % 360.0
        dec = math.degrees(math.asin(d[2] / np.linalg.norm(d)))
        self.aim(focal, (ra, dec))

    def apply(self, index):
        """Effect `index` (`EFFECTS`)."""
        _, focal, target, steps, var, change = EFFECTS[index]
        self.aim(focal, target)
        self.camera.exposure_time *= 2.0 ** change.get("stops", 0.0)
        self.lens(
            focal,
            change.get("achromat", False),
            change.get("lateral", False),
            change.get("focus_distance", 0.0),
        )
        self.chain(steps, var)
        self.shown = index

    def step(self):
        if self.download is not None:
            for text, error in self.download.take_messages():
                self.report(text, error)
            if self.download.done:
                if self.download.succeeded:
                    self.load()
                self.download = None
        if not self.tour:
            return
        t = self.time % (ZOOM_SECONDS + EFFECT_SECONDS * len(EFFECTS))
        if t < ZOOM_SECONDS:
            self.zoom(t / ZOOM_SECONDS)
        else:
            index = int((t - ZOOM_SECONDS) / EFFECT_SECONDS)
            if index != self.shown:
                self.apply(index)

    def render_shown(self):
        """What the product shows now as uint8 RGBA: LdrColor, or the
        RGB sensor's 8-bit numbers."""
        outputs = self.renderer.step(
            self.scene, render_products=["/Render/Product"]
        )
        values = np.asarray(
            outputs["/Render/Product"]
            .frames[0]
            .render_vars[f"/Render/Product/{self.var}"]
            .map()
        )
        if values.dtype == np.uint8:
            return np.array(values)
        image = np.full((HEIGHT, WIDTH, 4), 255, np.uint8)
        image[..., :3] = np.clip(
            values.reshape(HEIGHT, WIDTH, -1)[..., :3], 0, 255
        )
        return image

    def render_effects(self):
        """Every effect's image; leaves the demo as it was."""
        shown = self.shown
        images = []
        for index in range(len(EFFECTS)):
            self.apply(index)
            images.append(self.render_shown())
            self.log_image(f"effects/{index}", images[-1])
        if isinstance(shown, int):
            self.apply(shown)
        return images

    def save_output(self, path):
        """The effects in a grid, three to a row."""
        images = self.render_effects()
        rows = [
            ernie.demo.side_by_side(images[i : i + 3])
            for i in range(0, len(images), 3)
        ]
        gap = np.zeros((4, rows[0].shape[1], 4), np.uint8)
        gap[..., 3] = 255
        write_png(
            path,
            np.concatenate([x for r in rows for x in (r, gap)][:-1]),
        )

    def test_final(self):
        images = self.render_effects()
        luminance = [
            ernie.demo.srgb_to_linear(i[..., :3] / 255.0)
            @ [0.2126, 0.7152, 0.0722]
            if EFFECTS[n][4] == "LdrColor"
            else i[..., :3].astype(float).mean(axis=2) / 255.0
            for n, i in enumerate(images)
        ]

        # Each pixel's distance from the image's centre.
        y, x = np.mgrid[:HEIGHT, :WIDTH]
        r = np.hypot(x - WIDTH / 2 + 0.5, y - HEIGHT / 2 + 0.5)

        def half_light_radius(index):
            """The radius around the image's centre holding half its
            light within 60 pixels."""
            near = r < 60
            order = np.argsort(r[near])
            light = np.cumsum(luminance[index][near][order])
            return r[near][order][
                np.searchsorted(light, 0.5 * light[-1])
            ]

        radii = [half_light_radius(i) for i in range(6)]
        for i in range(6):
            print(
                f"{EFFECTS[i][0]}: half the light within {radii[i]:.1f} px"
            )
        assert radii[0] < 1.5, "a bare star lands in a pixel or two"
        assert radii[1] > radii[0] and radii[2] > radii[0], (
            "diffraction spreads the star"
        )
        # Beyond the first dark ring, 1.22 lambda N at 550 nm.
        ring = 1.22 * 0.55e-3 * LONG / APERTURE / PITCH_MM
        rings = [
            luminance[i][(r > ring) & (r < 60)].sum()
            / luminance[i][r < 60].sum()
            for i in (2, 3)
        ]
        print(
            f"light in the rings: clear {rings[0]:.3f}, obstructed {rings[1]:.3f}"
        )
        assert rings[1] > rings[0], (
            "an obstruction moves light into the rings"
        )
        assert radii[4] > 1.5 * radii[2], (
            "seeing blurs beyond diffraction"
        )
        assert radii[5] > 1.5 * radii[2], "defocus spreads the star"

        # Axial colour: a halo bluer than the core.
        linear = ernie.demo.srgb_to_linear(images[6][..., :3] / 255.0)
        core = linear[r < 2].mean(axis=0)
        halo = linear[(r > 5) & (r < 14)].mean(axis=0)
        print(f"axial colour: core {core}, halo {halo}")
        assert halo[2] / max(halo[1], 1e-6) > core[2] / max(
            core[1], 1e-6
        ), "an achromat's halo is bluer than its core"

        # Lateral colour: off-axis stars' red and blue apart, radially.
        def colour_split(image):
            linear = ernie.demo.srgb_to_linear(image[..., :3] / 255.0)
            far = np.where(r > 150, linear.sum(axis=2), 0.0)
            cy, cx = np.unravel_index(far.argmax(), far.shape)
            window = linear[cy - 8 : cy + 9, cx - 8 : cx + 9]
            wy, wx = np.mgrid[-8:9, -8:9]

            def centroid(c):
                weight = window[..., c]
                return np.array(
                    [(wx * weight).sum(), (wy * weight).sum()]
                ) / max(weight.sum(), 1e-9)

            return np.linalg.norm(centroid(0) - centroid(2))

        plain, lateral = (
            colour_split(images[7]),
            colour_split(images[8]),
        )
        print(
            f"red to blue off axis: {plain:.2f} px, lateral colour {lateral:.2f} px"
        )
        assert lateral > plain + 0.5, (
            "lateral colour splits red from blue"
        )

        lit = [int((luminance[i] > 0.01).sum()) for i in (7, 9)]
        print(f"lit pixels: belt {lit[0]}, glare {lit[1]}")
        assert lit[1] > 2 * lit[0], "glare spreads a halo"

        # Ghosts: light on the dark sky the same exposure lacks without
        # them.
        self.apply(10)
        self.scene.set(
            ernie.Path("/Render/Product/Chain/Step2"),
            "inputs:strength",
            0.0,
        )
        clean = self.render_shown()[..., :3].astype(float)
        self.apply(10)  # the ghosts back
        dark = luminance[7] < 0.002
        ghosts = float(
            (images[10][..., :3].astype(float) - clean)[dark]
            .clip(0)
            .sum()
        )
        print(f"ghost DN on the dark sky: {ghosts:.0f}")
        assert ghosts > 1000.0, "flare puts ghosts on the dark sky"

        # Blooming: the brightest star streaks down its column.
        bloom = luminance[11]
        cy, cx = np.unravel_index(bloom.argmax(), bloom.shape)
        tall = int((bloom[:, cx] > 0.5).sum())
        wide = int((bloom[cy, :] > 0.5).sum())
        print(f"blooming: {tall} px tall, {wide} px wide")
        assert tall > 3 * wide, "blooming streaks along columns"


if __name__ == "__main__":
    ernie.demo.main(Demo)
