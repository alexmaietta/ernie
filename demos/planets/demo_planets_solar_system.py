# SPDX-License-Identifier: BSD-3-Clause
"""The Sun, its planets and the Moon at their true sizes and distances on a
chosen date (no motion), lit by the Sun as a sphere light, their orbits
drawn as overlays split finely where they pass the eye. Each body wears a
latitude-longitude map streamed as a virtual texture: NASA's where one is
small enough, else Solar System Scope's (CC BY 4.0, from NASA's data); the
Sun's own generated, emitted through its material. The panel picks the
date and a body to go to.

References:
- E. M. Standish, "Keplerian Elements for Approximate Positions of the
  Major Planets", JPL Solar System Dynamics (1800 to 2050).
- B. A. Archinal et al., "Report of the IAU Working Group on Cartographic
  Coordinates and Rotational Elements: 2015", Celestial Mechanics and
  Dynamical Astronomy 130:22, 2018."""

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

DEGREE = math.pi / 180.0
AU = 1.495978707e11
# The ecliptic's obliquity at J2000, which turns the IAU's equatorial poles
# into the ecliptic frame the scene is in (z its north).
OBLIQUITY = 23.43928 * DEGREE
# The Sun's mean radiance as luminance: its disc's 1.6e9 cd/m^2.
SUN_LUMINANCE = 1.6e9
SCOPE = "https://www.solarsystemscope.com/textures/download/"

# Each body: its name, equatorial and polar radii (m), its mean orbital
# elements at J2000 and their rates per Julian century (Standish: semi-major
# axis in au, eccentricity, inclination, mean longitude, longitudes of
# perihelion and of the ascending node, degrees; none for the Sun and the
# Moon), its north pole's right ascension and declination and prime
# meridian's angle (degrees, and degrees a day) as the IAU gives them, its
# map (Solar System Scope's by file name, else a URL; none for the Sun's),
# the folder it is cached in, and a colour for it and its orbit.
BODIES = [
    ("Sun", 695.7e6, 695.7e6, None, (286.13, 63.87, 84.176, 14.1844),
     "", "solar_system", (1.0, 0.8, 0.5)),
    ("Mercury", 2440.5e3, 2438.3e3,
     ((0.38709927, 0.20563593, 7.00497902, 252.25032350, 77.45779628,
       48.33076593),
      (0.00000037, 0.00001906, -0.00594749, 149472.67411175, 0.16047689,
       -0.12534081)),
     (281.0103, 61.4155, 329.5988, 6.1385108), "8k_mercury.jpg",
     "solar_system", (0.6, 0.6, 0.6)),
    ("Venus", 6051.8e3, 6051.8e3,
     ((0.72333566, 0.00677672, 3.39467605, 181.97909950, 131.60246718,
       76.67984255),
      (0.00000390, -0.00004107, -0.00078890, 58517.81538729, 0.00268329,
       -0.27769418)),
     (272.76, 67.16, 160.20, -1.4813688), "4k_venus_atmosphere.jpg",
     "solar_system", (0.9, 0.8, 0.6)),
    ("Earth", 6378.137e3, 6356.752e3,
     ((1.00000261, 0.01671123, -0.00001531, 100.46457166, 102.93768193,
       0.0),
      (0.00000562, -0.00004392, -0.01294668, 35999.37244981, 0.32327364,
       0.0)),
     (0.0, 90.0, 190.147, 360.9856235),
     "https://eoimages.gsfc.nasa.gov/images/imagerecords/73000/73776/"
     "world.topo.bathy.200408.3x21600x10800.jpg",
     "blue_marble", (0.2, 0.4, 0.9)),
    ("Moon", 1738.1e3, 1736.0e3, None,
     (269.9949, 66.5392, 38.3213, 13.17635815),
     "https://svs.gsfc.nasa.gov/vis/a000000/a004700/a004720/"
     "lroc_color_poles_8k.tif",
     "moon", (0.7, 0.7, 0.7)),
    ("Mars", 3396.2e3, 3376.2e3,
     ((1.52371034, 0.09339410, 1.84969142, -4.55343205, -23.94362959,
       49.55953891),
      (0.00001847, 0.00007882, -0.00813131, 19140.30268499, 0.44441088,
       -0.29257343)),
     (317.269202, 54.432516, 176.049863, 350.891982443297), "8k_mars.jpg",
     "solar_system", (0.8, 0.4, 0.2)),
    ("Jupiter", 71492e3, 66854e3,
     ((5.20288700, 0.04838624, 1.30439695, 34.39644051, 14.72847983,
       100.47390909),
      (-0.00011607, -0.00013253, -0.00183714, 3034.74612775, 0.21252668,
       0.20469106)),
     (268.056595, 64.495303, 284.95, 870.5360000),
     "https://assets.science.nasa.gov/content/dam/science/psd/"
     "photojournal/pia/pia07/pia07782/PIA07782.tif",
     "solar_system", (0.8, 0.7, 0.5)),
    ("Saturn", 60268e3, 54364e3,
     ((9.53667594, 0.05386179, 2.48599187, 49.95424423, 92.59887831,
       113.66242448),
      (-0.00125060, -0.00050991, 0.00193609, 1222.49362201, -0.41897216,
       -0.28867794)),
     (40.589, 83.537, 38.90, 810.7939024), "8k_saturn.jpg",
     "solar_system", (0.9, 0.8, 0.6)),
    ("Uranus", 25559e3, 24973e3,
     ((19.18916464, 0.04725744, 0.77263783, 313.23810451, 170.95427630,
       74.01692503),
      (-0.00196176, -0.00004397, -0.00242939, 428.48202785, 0.40805281,
       0.04240589)),
     (257.311, -15.175, 203.81, -501.1600928), "2k_uranus.jpg",
     "solar_system", (0.6, 0.8, 0.9)),
    ("Neptune", 24764e3, 24341e3,
     ((30.06992276, 0.00859048, 1.77004347, -55.12002969, 44.96476227,
       131.78422574),
      (0.00026291, 0.00005105, 0.00035372, 218.45945325, -0.32241464,
       -0.00508664)),
     (299.36, 43.46, 249.978, 541.1397757), "2k_neptune.jpg",
     "solar_system", (0.3, 0.4, 0.9)),
]  # fmt: skip

# Saturn's rings: their radii, and a map of their colour and opacity from
# the inside outwards.
RING_INNER = 74500e3
RING_OUTER = 140220e3
RING_MAP = "8k_saturn_ring_alpha.png"

SUN, EARTH, MOON, SATURN = 0, 3, 4, 7


def url_of(name):
    """A map's URL: Solar System Scope's by file name, else as given."""
    return name if name.startswith("https://") else SCOPE + name


def cached_path(url, folder):
    """Where map `url` is cached, in `folder` (the other demos' for their
    bodies, shared)."""
    return os.path.join(cache_dir(), folder, os.path.basename(url))


def ecliptic_of(x, y, omega, i, node):
    """Point (x, y) of an orbit's plane, its periapsis along x, in the
    ecliptic frame: turned by the argument of periapsis `omega`, the
    inclination `i` and the node's longitude `node` (radians)."""
    co, so = math.cos(omega), math.sin(omega)
    cn, sn = math.cos(node), math.sin(node)
    ci, si = math.cos(i), math.sin(i)
    return np.array(
        [
            (co * cn - so * sn * ci) * x
            + (-so * cn - co * sn * ci) * y,
            (co * sn + so * cn * ci) * x
            + (-so * sn + co * cn * ci) * y,
            (so * si) * x + (co * si) * y,
        ]
    )


class Orbit:
    """An orbit at `centuries` past J2000: where it lies at eccentric
    anomaly `e` (radians), and the body's own eccentric anomaly then."""

    def __init__(self, elements, centuries):
        at, rate = elements
        v = [at[k] + rate[k] * centuries for k in range(6)]
        self.a = v[0] * AU
        self.e = v[1]
        self.i = v[2] * DEGREE
        self.node = v[5] * DEGREE
        self.omega = (v[4] - v[5]) * DEGREE
        # Kepler's equation, M = E - e sin E, by Newton's method.
        m = math.remainder((v[3] - v[4]) * DEGREE, 2.0 * math.pi)
        anomaly = m + self.e * math.sin(m)
        for _ in range(8):
            anomaly -= (anomaly - self.e * math.sin(anomaly) - m) / (
                1.0 - self.e * math.cos(anomaly)
            )
        self.anomaly = anomaly

    def at(self, eccentric):
        return ecliptic_of(
            self.a * (math.cos(eccentric) - self.e),
            self.a * math.sqrt(1.0 - self.e**2) * math.sin(eccentric),
            self.omega,
            self.i,
            self.node,
        )


class MoonOrbit:
    """The Moon's place about the Earth `days` past J2000: a mean circular
    orbit 384 400 km out, 5.145 degrees to the ecliptic, its node
    regressing; and that orbit at argument of latitude `u`."""

    def __init__(self, days):
        self.node = (125.08 - 0.052954 * days) * DEGREE
        self.latitude = (
            218.316 + 13.176396 * days - 125.08 + 0.052954 * days
        ) * DEGREE

    def at(self, u):
        return ecliptic_of(
            384400e3 * math.cos(u),
            384400e3 * math.sin(u),
            0.0,
            5.145 * DEGREE,
            self.node,
        )


def frame_of(body, days):
    """Body `body`'s axes `days` past J2000 as columns: x towards its prime
    meridian, z its north pole, turned from the IAU's equatorial frame into
    the ecliptic's."""
    ra, dec, w0, w_rate = body[4]
    ra *= DEGREE
    dec *= DEGREE
    pole = np.array(
        [math.cos(dec) * math.cos(ra), math.cos(dec) * math.sin(ra),
         math.sin(dec)]
    )  # fmt: skip
    # The ascending node of its equator on the Earth's.
    node = np.array([-math.sin(ra), math.cos(ra), 0.0])
    w = math.remainder((w0 + w_rate * days) * DEGREE, 2.0 * math.pi)
    x = math.cos(w) * node + math.sin(w) * np.cross(pole, node)
    c, s = math.cos(OBLIQUITY), math.sin(OBLIQUITY)
    to_ecliptic = np.array([[1, 0, 0], [0, c, s], [0, -s, c]])
    return np.stack(
        [to_ecliptic @ x, to_ecliptic @ np.cross(pole, x),
         to_ecliptic @ pole],
        axis=1,
    )  # fmt: skip


def sun_map():
    """The Sun's surface, made here: granules over a brighter background
    and a few dark spots, latitude by longitude."""
    path = os.path.join(cache_dir(), "solar_system", "sun_map.png")
    if os.path.exists(path):
        return path
    cols, rows = 2048, 1024
    random = np.random.default_rng(1)
    value = np.zeros((rows, cols))
    # Granulation: noise of a few octaves, each cell a random level blended
    # across.
    for octave in range(4):
        cells = 64 << octave
        lattice = random.random((cells // 2 + 1, cells + 1))
        y = np.arange(rows) * (cells // 2) / rows
        x = np.arange(cols) * cells / cols
        y0 = y.astype(np.int64)[:, None]
        x0 = x.astype(np.int64)[None, :]
        fy = (y - y0[:, 0])[:, None]
        fx = (x - x0[0])[None, :]
        value += 2.0**-octave * (
            (1 - fx) * (1 - fy) * lattice[y0, x0]
            + fx * (1 - fy) * lattice[y0, x0 + 1]
            + (1 - fx) * fy * lattice[y0 + 1, x0]
            + fx * fy * lattice[y0 + 1, x0 + 1]
        )
    level = 0.75 + 0.2 * (value / 1.875 - 0.5)
    # Sunspots in the active latitudes.
    col = np.arange(cols)[None, :]
    row = np.arange(rows)[:, None]
    for _ in range(6):
        sx = random.random() * cols
        side = -1 if random.random() < 0.5 else 1
        sy = rows * (0.5 + side * (0.06 + 0.1 * random.random()))
        size = 6.0 + 10.0 * random.random()
        d2 = (col - sx) ** 2 + (row - sy) ** 2
        level = level * (1.0 - 0.7 * np.exp(-d2 / size**2))
    image = np.empty((rows, cols, 4), dtype=np.uint8)
    for c, scale in enumerate((1.0, 0.8, 0.45)):
        image[:, :, c] = np.clip(
            level * scale * 255.0, 0, 255
        ).astype(np.uint8)
    image[:, :, 3] = 255
    os.makedirs(os.path.dirname(path), exist_ok=True)
    write_png(path, image)
    return path


def orbit_points(at, eye):
    """Points along an orbit's closed curve `at(t)`, t from 0 to 2 pi,
    split finely where it passes `eye`: until a piece's middle strays from
    its chord by under a ten-thousandth of its distance from the eye, so the
    line meets the body there however close the eye is."""
    start = 256
    points = [at(0.0)]

    def refine(t0, t1, p0, p1, depth):
        tm = 0.5 * (t0 + t1)
        pm = at(tm)
        sag = np.linalg.norm(pm - 0.5 * (p0 + p1))
        if depth < 40 and sag > 1e-4 * np.linalg.norm(pm - eye):
            refine(t0, tm, p0, pm, depth + 1)
            refine(tm, t1, pm, p1, depth + 1)
            return
        points.append(p1)

    previous = points[0]
    for k in range(1, start + 1):
        t = 2.0 * math.pi * k / start
        following = at(t)
        refine(
            2.0 * math.pi * (k - 1) / start, t, previous, following, 0
        )
        previous = following
    points.pop()  # The first again.
    return [tuple(p) for p in points]


class Demo(ernie.demo.Demo):
    num_frames = 3

    def __init__(self, viewer, args):
        super().__init__(viewer, args)
        scene = self.scene
        scene.define("/World", ernie.Xform)
        for i, body in enumerate(BODIES):
            name, equator, _, _, _, _, _, colour = body
            look = ernie.Path(f"/World/Looks/{name}")
            scene.define(
                f"/World/{name}",
                ernie.Sphere,
                radius=equator,
                tessellation=tokens.CUBE_SPHERE,
                tessellation_level=64 if i == SUN else 128,
                material_binding=look,
            )
            scene.define(look, ernie.Material)
            scene.define(
                look.child("Surface"),
                ernie.Shader,
                info_id=tokens.USD_PREVIEW_SURFACE,
                diffuse_color=(0.0, 0.0, 0.0) if i == SUN else colour,
                emissive_color=tuple(
                    SUN_LUMINANCE * c for c in colour
                )
                if i == SUN
                else (0.0, 0.0, 0.0),
                roughness=1.0,
            )
            scene.connect(
                look,
                tokens.SURFACE_OUTPUT,
                look.child("Surface"),
                tokens.SURFACE_OUTPUT,
            )
            scene.define(
                look.child("St"),
                ernie.Shader,
                info_id=tokens.USD_PRIMVAR_READER_FLOAT2,
                varname="st",
            )
        self.define_rings()
        # The Sun's light: a sphere of its radius and radiance.
        scene.define(
            "/World/SunLight",
            ernie.SphereLight,
            radius=BODIES[SUN][1],
            intensity=SUN_LUMINANCE,
            shadow_enable=False,
        )
        self.camera = scene.define(
            "/World/Camera",
            ernie.Camera,
            focal_length=24.0,
            clipping_range=(1000.0, math.inf),
            sensor_noise=False,
            exposure_iso=100.0,
            exposure_f_stop=8.0,
            exposure_time=1.0 / 500.0,
        )
        scene.define(
            "/Render/Settings",
            ernie.RenderSettings,
            camera=self.camera,
            resolution=(1280, 720),
            ambient=0.0,
            overlays="*",
        )

        self.year = 2026.8
        self.stops = 0.0
        self.sunlight = 1.0
        self.download = None
        self.downloading = ""
        self.progress = 0
        self.drawn_from = np.full(3, np.inf)
        # Maps: the Sun's made here, the rest fetched in turn, each worn
        # as it comes.
        self.use_map(SUN, sun_map())
        self.maps = []
        if not args.test and can_download():
            self.maps = [i for i, b in enumerate(BODIES) if b[5]]
            self.maps.append(len(BODIES))  # The rings'.
            self.progress = self.widgets.progress("Maps")

        widgets = self.widgets
        widgets.slider(
            "Year", 1900.0, 2100.0, self.year, on_change=self.set_year
        )
        widgets.combo(
            "Go to",
            [b[0] for b in BODIES]
            + ["Inner planets", "Whole system"],
            EARTH,
            on_change=self.go_to,
        )
        # Stops past the exposure for the sunlight where the camera goes.
        widgets.slider(
            "Exposure (stops)",
            -10.0,
            10.0,
            0.0,
            on_change=self.set_stops,
        )
        widgets.checkbox(
            "Orbits",
            True,
            on_change=lambda on: setattr(
                self.scene.overlay("orbits"), "visible", on
            ),
        )
        self.place()
        if viewer is not None:
            viewer.active_camera = self.camera.path
            display = viewer.viewport_display
            display.lights = False
            viewer.viewport_display = display
        self.go_to(EARTH)

    def step(self):
        self.step_maps()
        # Orbits drawn again when the eye has moved.
        eye = self.eye()
        if np.linalg.norm(
            eye - self.drawn_from
        ) > 1e-3 * self.nearest(eye):
            self.draw_orbits()

    def test_final(self):
        def centre():
            image = np.asarray(self.renderer.render(self.scene))
            rows, cols = image.shape[:2]
            return image[rows // 2, cols // 2, :3].astype(np.float64)

        # The Earth from the side facing the Sun: lit, not black.
        self.go_to(EARTH)
        earth = centre()
        # The Sun, exposed for its own light: its surface's map shows, not
        # one white.
        self.go_to(SUN)
        sun = np.asarray(self.renderer.render(self.scene))
        rows, cols = sun.shape[:2]
        middle = sun[
            rows // 2 - 20 : rows // 2 + 20,
            cols // 2 - 20 : cols // 2 + 20,
            0,
        ].astype(np.int64)
        lowest, highest = int(middle.min()), int(middle.max())
        # The whole system: orbits drawn across the black.
        self.go_to(len(BODIES) + 1)
        whole = np.asarray(self.renderer.render(self.scene))
        drawn = int(
            np.sum(whole[:, :, :3].astype(np.int64).sum(axis=2) > 60)
        )
        print(
            f"Earth {earth}; Sun {lowest} to {highest}; whole system "
            f"{drawn} pixels drawn"
        )
        assert earth.sum() > 60.0, (
            "the Earth's day side is lit by the Sun"
        )
        assert highest > 40 and highest - lowest > 10, (
            "the Sun's surface varies over its map"
        )
        assert drawn > 1000, "the orbits are drawn"

    def define_rings(self):
        """Saturn's rings: a flat annulus in its equator, its texture
        coordinates' s running outwards, its map's alpha its opacity."""
        segments = 512
        angle = 2.0 * math.pi * np.arange(segments + 1) / segments
        radii = np.array([RING_INNER, RING_OUTER])
        points = np.stack(
            [
                radii[None, :] * np.cos(angle)[:, None],
                radii[None, :] * np.sin(angle)[:, None],
                np.zeros((segments + 1, 2)),
            ],
            axis=-1,
        ).reshape(-1, 3)
        st = np.stack(
            [
                np.tile([0.0, 1.0], segments + 1),
                np.repeat(np.arange(segments + 1) / segments, 2),
            ],
            axis=1,
        ).astype(np.float32)
        # Both windings, seen from either side.
        k = 2 * np.arange(segments)
        indices = np.stack(
            [k, k + 1, k + 3, k + 2, k, k + 2, k + 3, k + 1], axis=1
        ).reshape(-1)
        self.scene.define(
            "/World/Rings",
            ernie.Mesh,
            points=points,
            face_vertex_counts=np.full(2 * segments, 4),
            face_vertex_indices=indices,
            material_binding="/World/Looks/Rings",
        ).write("primvars:st", st)
        look = ernie.Path("/World/Looks/Rings")
        self.scene.define(look, ernie.Material)
        self.scene.define(
            look.child("Surface"),
            ernie.Shader,
            info_id=tokens.USD_PREVIEW_SURFACE,
            diffuse_color=(0.6, 0.55, 0.45),
            roughness=1.0,
        )
        self.scene.connect(
            look,
            tokens.SURFACE_OUTPUT,
            look.child("Surface"),
            tokens.SURFACE_OUTPUT,
        )
        self.scene.define(
            look.child("St"),
            ernie.Shader,
            info_id=tokens.USD_PRIMVAR_READER_FLOAT2,
            varname="st",
        )

    def step_maps(self):
        """Fetches the next map, wearing each as it comes."""
        if self.download is None:
            while self.maps:
                i = self.maps[0]
                if i < len(BODIES):
                    url = url_of(BODIES[i][5])
                    path = cached_path(url, BODIES[i][6])
                else:
                    url = url_of(RING_MAP)
                    path = cached_path(url, "solar_system")
                if os.path.exists(path):
                    self.use_map(i, path)
                    self.maps.pop(0)
                    continue
                self.download = AsyncDownload(url, path)
                self.downloading = path
                break
            if self.download is None:
                if self.progress:
                    self.widgets.remove(self.progress)
                    self.progress = 0
                return
        for text, error in self.download.take_messages():
            self.report(text, error)
        self.widgets.set(self.progress, self.download.fraction)
        self.widgets.set(
            self.progress,
            f"{len(self.maps)} left: {self.download.status}",
        )
        if self.download.done:
            if self.download.succeeded:
                self.use_map(self.maps[0], self.downloading)
            self.download = None
            self.maps.pop(0)

    def use_map(self, i, path):
        """Body `i` wears map `path` (the rings' for `len(BODIES)`); the
        Sun's emitted."""
        scene = self.scene
        name = BODIES[i][0] if i < len(BODIES) else "Rings"
        look = ernie.Path(f"/World/Looks/{name}")
        texture = scene.define(
            look.child("Map"),
            ernie.Shader,
            info_id=tokens.USD_UV_TEXTURE,
            source_color_space="sRGB",
            file=str(path),
        )
        scene.connect(
            look.child("Map"),
            tokens.ST,
            look.child("St"),
            "outputs:result",
        )
        surface = look.child("Surface")
        if i == SUN:
            texture.set(
                tokens.SHADER_SCALE, (SUN_LUMINANCE,) * 3 + (1.0,)
            )
            scene.connect(
                surface,
                tokens.EMISSIVE_COLOR,
                look.child("Map"),
                "outputs:rgb",
            )
            return
        scene.connect(
            surface,
            tokens.DIFFUSE_COLOR,
            look.child("Map"),
            "outputs:rgb",
        )
        if i == len(BODIES):
            scene.connect(
                surface,
                tokens.OPACITY,
                look.child("Map"),
                "outputs:a",
            )
            scene.set(surface, tokens.OPACITY_THRESHOLD, 0.2)

    def days(self):
        """Days past J2000 at `year`."""
        return (self.year - 2000.0) * 365.25

    def position(self, i):
        """Body `i`'s centre on the date."""
        if i == SUN:
            return np.zeros(3)
        if i == MOON:
            moon = MoonOrbit(self.days())
            return self.position(EARTH) + moon.at(moon.latitude)
        orbit = Orbit(BODIES[i][3], self.days() / 36525.0)
        return orbit.at(orbit.anomaly)

    def set_year(self, year):
        self.year = year
        self.place()

    def place(self):
        """Every body at its place and turn on the date, and the orbits."""
        for i, body in enumerate(BODIES):
            axes = frame_of(body, self.days())
            transform = np.eye(4)
            transform[:3, :3] = axes
            transform[:3, 3] = self.position(i)
            rings = transform.copy()
            transform[:3, 2] *= body[2] / body[1]
            self.scene.set(
                ernie.Path(f"/World/{body[0]}"),
                tokens.TRANSFORM,
                transform,
            )
            if i == SATURN:
                self.scene.set(
                    ernie.Path("/World/Rings"),
                    tokens.TRANSFORM,
                    rings,
                )
        self.draw_orbits()

    def draw_orbits(self):
        """The orbits about the Sun, and the Moon's about the Earth, and a
        label beside each body."""
        overlay = self.scene.overlay("orbits")
        overlay.clear()
        eye = self.eye()
        self.drawn_from = eye
        for i, body in enumerate(BODIES):
            colour = (*body[7], 0.7)
            if body[3] is not None:
                orbit = Orbit(body[3], self.days() / 36525.0)
                overlay.polyline(
                    orbit_points(orbit.at, eye),
                    color=colour,
                    width=1.5,
                    closed=True,
                )
            elif i == MOON:
                moon = MoonOrbit(self.days())
                earth = self.position(EARTH)
                overlay.polyline(
                    orbit_points(
                        lambda t, earth=earth, moon=moon: (
                            earth + moon.at(t)
                        ),
                        eye,
                    ),
                    color=colour,
                    width=1.5,
                    closed=True,
                )
            overlay.label(
                tuple(self.position(i)),
                body[0],
                color=(1.0, 1.0, 1.0, 1.0),
                size=13.0,
                depth="always",
            )

    def go_to(self, i):
        """Goes to body `i` (its day side, four radii out), or for the two
        after the bodies the inner planets or the whole system from
        above."""
        target = np.zeros(3)
        if i < len(BODIES):
            target = self.position(i)
            radius = RING_OUTER if i == SATURN else BODIES[i][1]
            # Between the body and the Sun, a little above the ecliptic.
            sunward = (
                np.array([1.0, 0.0, 0.0])
                if i == SUN
                else -target / np.linalg.norm(target)
            )
            side = np.cross([0.0, 0.0, 1.0], sunward)
            side /= np.linalg.norm(side)
            offset = sunward + 0.6 * side + np.array([0.0, 0.0, 0.3])
            eye = target + 4.0 * radius * offset / np.linalg.norm(
                offset
            )
        else:
            reach = 2.0 * AU if i == len(BODIES) else 35.0 * AU
            eye = np.array([0.0, -0.6 * reach, reach])
        if self.viewer is not None:
            self.viewer.look_at(tuple(eye), tuple(target))
        else:
            self.camera.set(
                tokens.TRANSFORM,
                ernie.look_at_transform(
                    tuple(eye), tuple(target), (0.0, 0.0, 1.0)
                ),
            )
        # Exposed for the sunlight there: it falls as the square of the
        # distance from the Sun; the Sun itself 16 stops darker than the
        # Earth's daylight.
        if i == SUN:
            self.sunlight = 2.0**16.0
        elif i < len(BODIES):
            self.sunlight = (np.linalg.norm(target) / AU) ** -2.0
        else:
            self.sunlight = 1.0
        self.expose()
        self.draw_orbits()

    def set_stops(self, stops):
        self.stops = stops
        self.expose()

    def expose(self):
        """The camera's exposure: the Earth's daylight's (f/8, 1/500 s at
        ISO 100) for `sunlight` times as much light, `stops` more."""
        self.camera.set(
            tokens.EXPOSURE_TIME,
            2.0**self.stops / (500.0 * self.sunlight),
        )

    def eye(self):
        """The camera's place."""
        transform = self.camera.get(tokens.TRANSFORM)
        if transform is None:
            return np.zeros(3)
        return np.asarray(transform)[:3, 3].astype(np.float64)

    def nearest(self, eye):
        """The distance from `eye` to the nearest body's surface, at least a
        kilometre."""
        return max(
            min(
                np.linalg.norm(eye - self.position(i)) - b[1]
                for i, b in enumerate(BODIES)
            ),
            1000.0,
        )

    def report(self, text, error=False):
        """A message in the viewer's log, else printed."""
        if self.viewer is not None:
            self.viewer.log(text, "error" if error else "info")
        else:
            print(text)


if __name__ == "__main__":
    ernie.demo.main(Demo)
