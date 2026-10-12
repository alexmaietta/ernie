# SPDX-License-Identifier: BSD-3-Clause
"""Orion from space through four star catalogues: made-up stars, the
Yale Bright Star Catalogue (to V 6.5), Hipparcos (to about V 12) and
Tycho-2 around Orion (to VT 11), fetched from VizieR. Offline, made-up
stars to each one's depth around Orion's brightest."""

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
ORION = (83.8, -1.2)  # right ascension and declination, degrees
VIZIER = "https://vizier.cds.unistra.fr/viz-bin/asu-tsv?-out.max=unlimited&"

# Each catalogue: name, cached file, url, and offline made-up stars'
# count and faintest magnitude. Tycho-2 rows give BT and VT.
CATALOGUES = [
    ("Made-up stars", None, None, (9110, 6.5)),
    (
        "Yale Bright Star Catalogue",
        "bsc5.tsv",
        VIZIER
        + "-source=V/50/catalog&-out=_RAJ2000,_DEJ2000,Vmag,B-V",
        (9110, 6.5),
    ),
    (
        "Hipparcos",
        "hipparcos.tsv",
        VIZIER + "-source=I/239/hip_main&-out=RAICRS,DEICRS,Vmag,B-V",
        (118000, 9.5),
    ),
    (
        "Tycho-2 around Orion",
        "tycho2_orion.tsv",
        VIZIER
        + "-source=I/259/tyc2&-out=RAmdeg,DEmdeg,BTmag,VTmag"
        + "&-c=83.8,-1.2&-c.rd=25&VTmag=%3C11",
        (400000, 11.0),
    ),
]

# Orion's brightest stars and neighbours (J2000 degrees, V, B-V).
BRIGHT_STARS = [
    (88.7929, 7.4071, 0.42, 1.85),  # Betelgeuse
    (78.6345, -8.2016, 0.13, -0.03),  # Rigel
    (81.2828, 6.3497, 1.64, -0.22),  # Bellatrix
    (84.0534, -1.2019, 1.69, -0.18),  # Alnilam
    (85.1897, -1.9426, 1.77, -0.21),  # Alnitak
    (83.0017, -0.2991, 2.23, -0.22),  # Mintaka
    (86.9391, -9.6696, 2.09, -0.17),  # Saiph
    (83.7845, 9.9342, 3.39, -0.16),  # Meissa
    (83.8583, -5.9099, 2.77, -0.24),  # Hatysa
    (101.2872, -16.7161, -1.46, 0.00),  # Sirius
    (114.8255, 5.2250, 0.34, 0.42),  # Procyon
    (68.9802, 16.5093, 0.87, 1.54),  # Aldebaran
]


def made_up(count, faintest, seed):
    stars = ernie.synthetic_stars(count, seed=seed, faintest=faintest)
    return np.stack(
        [
            np.asarray(stars[key], dtype=np.float32)
            for key in (
                "right_ascension",
                "declination",
                "magnitude",
                "color_index",
            )
        ],
        axis=1,
    )


def read_vizier(path):
    """A VizieR table of right ascension, declination, V and B-V, rows
    without a magnitude left out. Tycho-2's BT and VT become V = VT -
    0.090 (BT - VT) and B - V = 0.850 (BT - VT) (ESA SP-1200, 1997,
    volume 1, section 1.3)."""
    rows = []
    tycho = False
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            tycho = tycho or "BTmag" in fields
            try:
                ra, dec = float(fields[0]), float(fields[1])
                a, b = (
                    float(x) if x.strip() else math.nan
                    for x in fields[2:4]
                )
            except (ValueError, IndexError):
                continue  # headers and units
            if tycho:
                bt, vt = a, b
                a = vt if math.isnan(bt) else vt - 0.090 * (bt - vt)
                b = 0.65 if math.isnan(bt) else 0.850 * (bt - vt)
            if not math.isnan(a):
                rows.append(
                    (ra, dec, a, 0.65 if math.isnan(b) else b)
                )
    return np.asarray(rows, dtype=np.float32)


def star_peaks(image, threshold=0.003):
    """The stars seen: pixels no dimmer than their eight neighbours
    and brighter by `threshold` (linear) than any three pixels away."""
    lum = ernie.demo.srgb_to_linear(image[..., :3] / 255.0) @ [
        0.2126,
        0.7152,
        0.0722,
    ]
    rows, cols = lum.shape
    inner = lum[3:-3, 3:-3]
    peak = np.ones(inner.shape, bool)
    ring = np.zeros(inner.shape)
    for dy in range(-3, 4):
        for dx in range(-3, 4):
            shifted = lum[
                3 + dy : rows - 3 + dy, 3 + dx : cols - 3 + dx
            ]
            if max(abs(dy), abs(dx)) == 1:
                peak &= inner >= shifted
            elif max(abs(dy), abs(dx)) == 3:
                ring = np.maximum(ring, shifted)
    return np.argwhere(peak & (inner > ring + threshold)) + 3


class Demo(ernie.demo.Demo):
    num_frames = 1

    def __init__(self, viewer, args):
        super().__init__(viewer, args)
        world = self.scene.define("/World", ernie.Xform)
        bright = np.asarray(BRIGHT_STARS, dtype=np.float32)
        self.fields = []
        for i, (_, _, _, (count, faintest)) in enumerate(CATALOGUES):
            field = world.define(f"Catalogue{i}", ernie.StarField)
            stars = made_up(count, faintest, seed=i + 1)
            self.set_stars(
                field, np.concatenate([bright, stars]) if i else stars
            )
            self.fields.append(field)
        ra, dec = (math.radians(a) for a in ORION)
        self.camera = world.define(
            "Camera",
            ernie.Camera,
            focal_length=50.0,
            horizontal_aperture=36.0,
            vertical_aperture=24.0,
            exposure_f_stop=2.0,
            exposure_time=8.0,
            exposure_iso=800.0,
            sensor_noise=False,
        ).set(
            tokens.TRANSFORM,
            ernie.look_at_transform(
                (0.0, 0.0, 0.0),
                (
                    math.cos(dec) * math.cos(ra),
                    math.cos(dec) * math.sin(ra),
                    math.sin(dec),
                ),
                (0.0, 0.0, 1.0),
            ),
        )
        filters = []
        for i, (kind, inputs) in enumerate(
            [
                ("psf", {"blur": 0.8, "maxRadius": 4.0}),
                ("cie", {}),
                ("glare", {"strength": 0.05}),
                ("exposure", {}),
            ]
        ):
            prim = self.scene.define(
                f"/Render/Product/Chain/Step{i}",
                ernie.SensorFilter,
                filter_type=kind,
            )
            for name, value in inputs.items():
                prim.set(f"inputs:{name}", float(value))
            filters.append(ernie.Path(prim.path))
        self.scene.define(
            "/Render/Product",
            ernie.RenderProduct,
            camera=self.camera,
            resolution=(WIDTH, HEIGHT),
        ).set(tokens.FILTERS, filters)
        self.scene.define(
            "/Render/Settings",
            ernie.RenderSettings,
            camera=self.camera,
            resolution=(WIDTH, HEIGHT),
            products=[ernie.Path("/Render/Product")],
        )
        self.show(1)

        self.widgets.combo(
            "Catalogue",
            [c[0] for c in CATALOGUES],
            1,
            on_change=self.show,
        )
        self.widgets.slider(
            "Magnitude limit", 0.0, 12.0, 12.0, on_change=self.limit
        )
        self.widgets.slider(
            "Exposure (stops)", -6.0, 4.0, 0.0, on_change=self.expose
        )
        if viewer is not None:
            viewer.active_camera = self.camera.path

        # The catalogues when cached; else fetched in the background,
        # but not for the checks, which run offline.
        self.downloads = {}
        if not args.test:
            for i, (_, file, url, _) in enumerate(CATALOGUES):
                if file is None:
                    continue
                path = os.path.join(cache_dir(), "stars", file)
                if os.path.exists(path):
                    self.load(i, path)
                elif can_download():
                    self.downloads[i] = (
                        AsyncDownload(url, path),
                        path,
                    )

    def set_stars(self, field, table):
        field.right_ascension = table[:, 0].copy()
        field.declination = table[:, 1].copy()
        field.magnitude = table[:, 2].copy()
        field.color_index = table[:, 3].copy()

    def report(self, text, error=False):
        if self.viewer is not None:
            self.viewer.log(text, "error" if error else "info")
        else:
            print(text)

    def load(self, index, path):
        try:
            table = read_vizier(path)
            self.set_stars(self.fields[index], table)
            self.report(
                f"Loaded {len(table)} stars of {CATALOGUES[index][0]}"
            )
        except (OSError, ValueError) as error:
            self.report(str(error), True)

    def show(self, index):
        """Catalogue `index` alone: the first visible star field
        shows."""
        self.shown = index
        for i, field in enumerate(self.fields):
            field.visibility = (
                tokens.INHERITED if i == index else tokens.INVISIBLE
            )

    def limit(self, magnitude):
        for field in self.fields:
            field.magnitude_limit = magnitude

    def expose(self, stops):
        self.camera.exposure = stops

    def step(self):
        for index, (download, path) in list(self.downloads.items()):
            for text, error in download.take_messages():
                self.report(text, error)
            if download.done:
                if download.succeeded:
                    self.load(index, path)
                del self.downloads[index]

    def render_all(self):
        """Every catalogue's view; leaves the catalogue shown."""
        shown = self.shown
        images = []
        for index in range(len(CATALOGUES)):
            self.show(index)
            images.append(np.array(self.renderer.render(self.scene)))
        self.show(shown)
        return images

    def save_output(self, path):
        """The catalogues two by two."""
        images = self.render_all()
        gap = np.zeros((4, 2 * WIDTH + 4, 4), np.uint8)
        gap[..., 3] = 255
        write_png(
            path,
            np.concatenate(
                [
                    ernie.demo.side_by_side(images[:2]),
                    gap,
                    ernie.demo.side_by_side(images[2:]),
                ]
            ),
        )

    def test_final(self):
        counts = [
            len(star_peaks(image)) for image in self.render_all()
        ]
        for (name, *_), count in zip(CATALOGUES, counts):
            print(f"{name}: {count} stars")
        assert counts[2] > 1.3 * counts[1], "Hipparcos goes deeper"
        assert counts[3] > counts[1], "Tycho-2 goes deeper"
        self.show(2)
        self.limit(4.0)
        bright = len(
            star_peaks(np.array(self.renderer.render(self.scene)))
        )
        self.limit(12.0)
        print(f"Hipparcos to V 4: {bright} stars")
        assert 5 < bright < counts[1] / 2, (
            "the limit leaves the faint out"
        )


if __name__ == "__main__":
    ernie.demo.main(Demo)
