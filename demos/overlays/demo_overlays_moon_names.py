# SPDX-License-Identifier: BSD-3-Clause
"""The IAU's names of lunar features (USGS Gazetteer of Planetary
Nomenclature) as labels on the Moon, chosen each frame by a level of
detail: facing the camera, near enough, large enough on screen, the
largest first, at most so many. Offline, a few well-known features."""

import math
import os
import struct
import zipfile

import numpy as np

import ernie
import ernie.demo
from ernie import tokens
from ernie.demo import AsyncDownload, cache_dir, can_download

RADIUS = 1737400.0  # metres
WIDTH, HEIGHT = 960, 540
FOCAL, APERTURE = 35.0, 36.0  # mm
URL = (
    "https://asc-planetarynames-data.s3.us-west-2.amazonaws.com/"
    "MOON_nomenclature_center_pts.zip"
)

# Well-known features, for running offline: name, latitude, east
# longitude, diameter (km), type.
FALLBACK = [
    ("Oceanus Procellarum", 20.67, 303.32, 2592.2, "Oceanus"),
    ("Mare Imbrium", 34.72, 345.09, 1145.5, "Mare"),
    ("Mare Tranquillitatis", 8.35, 30.83, 875.7, "Mare"),
    ("Mare Serenitatis", 27.29, 18.36, 674.3, "Mare"),
    ("Mare Crisium", 16.18, 59.10, 555.9, "Mare"),
    ("Mare Nectaris", -15.19, 34.60, 339.4, "Mare"),
    ("Clavius", -58.62, 345.27, 230.8, "Crater"),
    ("Plato", 51.62, 350.62, 100.7, "Crater"),
    ("Copernicus", 9.62, 339.92, 96.1, "Crater"),
    ("Tycho", -43.30, 348.78, 85.3, "Crater"),
    ("Aristarchus", 23.73, 312.51, 40.0, "Crater"),
    ("Kepler", 8.12, 321.99, 29.5, "Crater"),
]


def read_nomenclature(path):
    """The Gazetteer's features from the shapefile's attribute table
    (.dbf) in the zip."""
    with zipfile.ZipFile(path) as archive:
        name = next(
            n for n in archive.namelist() if n.endswith(".dbf")
        )
        data = archive.read(name)
    records, header, length = struct.unpack("<IHH", data[4:12])
    fields = {}
    offset = 1  # past the deletion flag
    for at in range(32, header - 1, 32):
        field = data[at : at + 11].split(b"\0")[0].decode()
        fields[field] = (offset, data[at + 16])
        offset += data[at + 16]

    def text(base, field):
        start, size = fields[field]
        return (
            data[base + start : base + start + size]
            .decode("utf-8", "replace")
            .rstrip(" ")
        )

    features = []
    for r in range(records):
        base = header + r * length
        features.append(
            (
                text(base, "clean_name"),
                float(text(base, "center_lat")),
                float(text(base, "center_lon")),
                float(text(base, "diameter")),
                text(base, "type"),
            )
        )
    return features


def direction(lat, lon):
    """The unit vector out of the Moon at a feature: x at 0 east, z
    north."""
    phi, lam = math.radians(lat), math.radians(lon)
    return np.array(
        [
            math.cos(phi) * math.cos(lam),
            math.cos(phi) * math.sin(lam),
            math.sin(phi),
        ]
    )


def type_color(kind):
    """A feature type's colour: seas, craters, mountains, lines,
    others."""
    if kind.startswith(
        ("Mare", "Oceanus", "Lacus", "Sinus", "Palus")
    ):
        return (0.55, 0.75, 1.0, 1.0)
    if kind.startswith("Crater"):
        return (1.0, 1.0, 1.0, 1.0)
    if kind.startswith(("Mons", "Promontorium")):
        return (1.0, 0.7, 0.35, 1.0)
    if kind.startswith(
        ("Rima", "Rupes", "Dorsum", "Vallis", "Catena")
    ):
        return (0.55, 1.0, 0.55, 1.0)
    if kind.startswith("Satellite"):
        return (0.75, 0.75, 0.75, 1.0)
    return (1.0, 0.55, 0.9, 1.0)


class Demo(ernie.demo.Demo):
    num_frames = 240

    def __init__(self, viewer, args):
        super().__init__(viewer, args)
        scene = self.scene
        world = scene.define("/World", ernie.Xform)
        world.define(
            "Moon",
            ernie.Sphere,
            radius=RADIUS,
            display_color=(0.55, 0.55, 0.55),
        )
        world.define("Sun", ernie.DistantLight).set(
            tokens.TRANSFORM,
            ernie.distant_light_transform((0.6, -0.5, 0.3)),
        )
        self.camera = world.define(
            "Camera",
            ernie.Camera,
            focal_length=FOCAL,
            horizontal_aperture=APERTURE,
            vertical_aperture=APERTURE * HEIGHT / WIDTH,
            clipping_range=(1000.0, 1.0e8),
        )
        scene.define(
            "/Render/Settings",
            ernie.RenderSettings,
            camera=self.camera,
            resolution=(WIDTH, HEIGHT),
            overlays="*",
        )
        # In the viewer the camera is free, starting 2500 km over
        # Oceanus Procellarum, near enough for names; the checks
        # follow a descent.
        self.free = viewer is not None and not args.test
        if viewer is not None:
            self.pose(0.0)
            viewer.active_camera = "/World/Camera"
            if self.free:
                viewer.look_at(
                    tuple((RADIUS + 2.5e6) * direction(10.0, 315.0)),
                    (0.0, 0.0, 0.0),
                )
            # The Sun's gizmo would sit at the Moon's centre.
            display = viewer.viewport_display
            display.lights = False
            viewer.viewport_display = display
        # The Gazetteer when cached; else a few features while it is
        # fetched in the background, but not for the checks, which run
        # offline.
        self.zip = os.path.join(
            cache_dir(), "moon", "MOON_nomenclature_center_pts.zip"
        )
        self.features = FALLBACK
        self.download = None
        if os.path.exists(self.zip) and not args.test:
            self.load()
        elif not args.test and can_download():
            self.download = AsyncDownload(URL, self.zip)
        # The level of detail's options.
        self.max_labels = 60
        self.min_pixels = 12.0  # a feature's diameter on screen
        self.max_distance = 6000.0  # km from the camera
        self.satellites = False  # lettered satellite craters
        self.outlines = True
        widgets = self.widgets
        widgets.slider(
            "Max labels",
            1,
            300,
            self.max_labels,
            on_change=lambda v: setattr(self, "max_labels", int(v)),
        )
        widgets.slider(
            "Min pixels",
            1,
            100,
            self.min_pixels,
            on_change=lambda v: setattr(self, "min_pixels", v),
        )
        widgets.slider(
            "Max distance (km)",
            100,
            10000,
            self.max_distance,
            on_change=lambda v: setattr(self, "max_distance", v),
        )
        widgets.checkbox(
            "Satellite craters",
            self.satellites,
            on_change=lambda on: setattr(self, "satellites", on),
        )
        widgets.checkbox(
            "Outlines",
            self.outlines,
            on_change=lambda on: setattr(self, "outlines", on),
        )
        self.shown = []
        self.smallest = []  # km, per frame
        self.place()

    def step(self):
        # The download's news in the viewer's log, and its names once
        # fetched.
        if self.download is not None:
            for text, error in self.download.take_messages():
                self.report(text, error)
            if self.download.done:
                if self.download.succeeded:
                    self.load()
                self.download = None
        self.place()

    def report(self, text, error=False):
        """A message in the viewer's log, else printed."""
        if self.viewer is not None:
            self.viewer.log(text, "error" if error else "info")
        else:
            print(text)

    def load(self):
        """The Gazetteer's names in place of the few built in."""
        try:
            self.features = read_nomenclature(self.zip)
            self.report(
                f"Loaded {len(self.features)} lunar feature names"
            )
        except (OSError, ValueError, KeyError) as error:
            self.report(str(error), True)

    def pose(self, t):
        """The camera at `t` of its descent: from four radii out,
        looking at the Moon, to a low orbit past Copernicus, looking
        back down at Kepler."""
        distance = RADIUS * (4.0 - 2.8 * t)
        lat = 10.0 + 5.0 * t
        lon = 300.0 + 35.0 * t
        eye = distance * direction(lat, lon)
        target = t * RADIUS * direction(lat - 5.0, lon - 13.0)
        self.camera.set(
            tokens.TRANSFORM,
            ernie.look_at_transform(
                tuple(eye), tuple(target), up=(0.0, 0.0, 1.0)
            ),
        )

    def place(self):
        """Moves the camera down its path, unless free, and labels
        what it sees."""
        scene = self.scene
        if not self.free:
            self.pose(min(self.frame / 239.0, 1.0))
        pose = np.array(scene.world_transform("/World/Camera"))
        eye = pose[:3, 3]
        distance = float(np.linalg.norm(eye))
        # The camera's axes (right, up, forward), to keep to what it
        # sees.
        right, up, forward = pose[:3, 0], pose[:3, 1], -pose[:3, 2]
        # Each candidate's size on screen, largest first.
        focal = WIDTH * FOCAL / APERTURE  # pixels
        candidates = []
        for i, (_name, lat, lon, diameter, kind) in enumerate(
            self.features
        ):
            if not self.satellites and kind.startswith("Satellite"):
                continue
            n = direction(lat, lon)
            to_eye = eye - RADIUS * n
            distance_to = float(np.linalg.norm(to_eye))
            # Facing the camera, above its horizon, near enough.
            if float(n @ to_eye) < 0.1 * distance_to or (
                distance_to > 1000.0 * self.max_distance
            ):
                continue
            # In view.
            d = -to_eye
            depth = float(d @ forward)
            if (
                depth <= 0.0
                or abs(float(d @ right)) * focal > 0.5 * WIDTH * depth
                or abs(float(d @ up)) * focal > 0.5 * HEIGHT * depth
            ):
                continue
            pixels = focal * 1000.0 * diameter / distance_to
            if pixels >= self.min_pixels:
                candidates.append((pixels, i))
        candidates.sort(key=lambda c: (-c[0], c[1]))
        candidates = candidates[: int(self.max_labels)]
        names = scene.overlay("names")
        names.clear()
        outlines = scene.overlay("outlines")
        outlines.clear()
        for pixels, i in candidates:
            name, lat, lon, diameter, kind = self.features[i]
            n = direction(lat, lon)
            color = type_color(kind)
            names.label(
                tuple(RADIUS * n),
                name,
                color=color,
                size=min(max(10.0 + pixels / 40.0, 10.0), 18.0),
                depth="always",
                # A dark halo keeps names legible over bright ground.
                background=(0, 0, 0, 0.8),
                halo=1.5,
            )
            if self.outlines:
                # The feature's rim traced on the surface.
                angle = 0.5 * diameter * 1000.0 / RADIUS
                helper = (
                    np.array([0.0, 0.0, 1.0])
                    if abs(n[2]) < 0.9
                    else np.array([1.0, 0.0, 0.0])
                )
                u = np.cross(helper, n)
                u /= np.linalg.norm(u)
                v = np.cross(n, u)
                a = 2.0 * np.pi * np.arange(48) / 48.0
                rim = (
                    1.0005
                    * RADIUS
                    * (
                        math.cos(angle) * n
                        + math.sin(angle)
                        * (
                            np.cos(a)[:, None] * u
                            + np.sin(a)[:, None] * v
                        )
                    )
                )
                outlines.polyline(
                    rim,
                    closed=True,
                    color=(*color[:3], 0.6),
                    width=1.5,
                )
        self.shown.append(len(candidates))
        self.smallest.append(
            min(
                (self.features[i][3] for _, i in candidates),
                default=math.inf,
            )
        )
        # The HUD: what the level of detail shows, and its options.
        hud = scene.overlay("hud")
        hud.clear()
        hud.text(
            "Lunar nomenclature (IAU, USGS)\n"
            f"{len(candidates)} of {len(self.features)} names, "
            f"altitude {(distance - RADIUS) / 1000.0:.0f} km\n"
            f"max {int(self.max_labels)} labels, min {self.min_pixels:.0f} px, "
            f"max {self.max_distance:.0f} km"
            + (", satellites" if self.satellites else ""),
            at=(1, 1),
            offset=(-12, -12),
            size=13,
        )

    def test_final(self):
        assert all(0 < c <= self.max_labels for c in self.shown), (
            "labels each frame, no more than allowed"
        )
        assert self.smallest[-1] < self.smallest[0], (
            "smaller features named as the camera comes down"
        )


if __name__ == "__main__":
    ernie.demo.main(Demo)
