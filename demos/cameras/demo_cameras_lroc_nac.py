# SPDX-License-Identifier: BSD-3-Clause
"""LROC's Narrow Angle Camera line-scanning the Moon's south pole at a
low Sun, posed from a real ISD on LOLA's terrain (fetched on the first
run, 3 MB).

References:
- M. S. Robinson et al., "Lunar Reconnaissance Orbiter Camera (LROC)
  Instrument Overview", Space Science Reviews 150:81-124, 2010: Table 2
  (the NACs) and Appendix A (their companding).
- USGS Astrogeology, usgscsm's test data (toughLroNacLineScan.json, an
  ISD ALE wrote for a NAC image) and ALE,
  https://github.com/DOI-USGS/usgscsm and
  https://github.com/DOI-USGS/ale
- D. E. Smith et al., "The Lunar Orbiter Laser Altimeter Investigation
  on the Lunar Reconnaissance Orbiter Mission", Space Science Reviews
  150:209-241, 2010 (LOLA, whose polar LDEM is the terrain).
- G. Kopp and J. L. Lean, "A new, lower value of total solar
  irradiance", Geophysical Research Letters 38:L01706, 2011 (1361
  W/m^2 at 1 au).
"""

import math
import os

import numpy as np

import ernie
import ernie.demo
from ernie import tokens
from ernie.demo import (
    cache_dir,
    can_download,
    fetch_ranges,
    write_png,
)

# The image lines rendered: 8 strips of 64 about the image's middle,
# where it passes the pole.
FIRST_LINE = 3840
STRIP_LINES = 64
STRIPS = 8

# The terrain about the middle line's ground: cells of 10 m, 12 km
# across.
CELLS = 1201
CELL_SIZE = 10.0

# The Sun's radius and the astronomical unit, metres; the total solar
# irradiance at 1 au, W/m^2 (Kopp and Lean 2011).
SUN_RADIUS = 6.957e8
AU = 1.495978707e11
SOLAR_CONSTANT = 1361.0

# NAC-L's published values (Robinson et al. 2010, Table 2; section 3.1.2
# for the KLI-5001G's QE): focal ratio, gain, detector noise and full
# well. The ISD sets the geometry and line time.
NAC_F_NUMBER = 3.577
NAC_ELECTRONS_PER_DN = 90.5
NAC_READ_NOISE = 101.0
NAC_FULL_WELL = 334000.0


def nac_compand(x):
    """The NAC's nominal 12- to 8-bit table (Appendix A, code 0): five
    linear segments."""
    if x < 32:
        return math.floor(x / 2)
    if x < 136:
        return math.floor(x / 4 + 8)
    if x < 543:
        return math.floor(x / 8 + 25)
    if x < 2207:
        return math.floor(x / 16 + 59)
    return math.floor(x / 32 + 128)


# ----------------------------------------------------------------- LOLA

# LOLA's 10 m polar LDEM south of 85 S at the PDS Geosciences Node: 30336
# x 30336 16-bit samples, the pole at line and sample 15167.5.
LDEM_URL = (
    "https://pds-geosciences.wustl.edu/lro/lro-l-lola-3-rdr-v1/"
    "lrolol_1xxx/data/lola_gdr/polar/img/ldem_85s_10m.img"
)
LDEM_SIZE = 30336
LDEM_OFFSET = 15167.5
LDEM_SCALE = 10.0
MOON_RADIUS = 1737400.0


def ldem_label():
    """The cached window's label."""
    return os.path.join(
        cache_dir(), "lola", "ldem_85s_10m_lroc_nac.lbl"
    )


def ldem_grid(latitude, longitude):
    """The polar grid's fractional (line, sample) of a latitude and
    east longitude, as `ernie.PolarStereographicDem.to_grid`: polar
    distance rho = 2 R tan(45 + lat / 2), x = rho sin(lon) and y = rho
    cos(lon) (Snyder 1987, eq. 21-33 and 21-34)."""
    rho = (
        2.0
        * MOON_RADIUS
        * math.tan(0.25 * math.pi + 0.5 * math.radians(latitude))
    )
    x = rho * math.sin(math.radians(longitude))
    y = rho * math.cos(math.radians(longitude))
    return (
        LDEM_OFFSET - y / LDEM_SCALE,
        LDEM_OFFSET + x / LDEM_SCALE,
    )


def fetch_ldem_window(latitude, longitude, half):
    """Fetches the window of the polar LDEM `half` pixels about a
    latitude and longitude, 256 lines a request, each line's samples in
    the window alone (HTTP range requests; the product is 1.8 GB), and
    writes it to the cache as an image and a PDS3 label of its own,
    which `ernie.read_polar_stereographic_dem` reads as it would the
    whole product. As demo_cameras_lroc_nac.cpp's
    `FetchLdemWindow`."""
    centre = ldem_grid(latitude, longitude)
    first_line = round(centre[0]) - half
    first_sample = round(centre[1]) - half
    size = 2 * half + 1
    print(f"Fetching {size} lines of {size} samples from {LDEM_URL}")
    rows = bytearray()
    row_bytes = 2 * LDEM_SIZE
    for start in range(0, size, 256):
        ranges = [
            (
                line * row_bytes + 2 * first_sample,
                line * row_bytes + 2 * (first_sample + size) - 1,
            )
            for line in range(
                first_line + start,
                first_line + min(start + 256, size),
            )
        ]
        for data in fetch_ranges(LDEM_URL, ranges):
            rows += data
    label = ldem_label()
    directory = os.path.dirname(label)
    os.makedirs(directory, exist_ok=True)
    with open(
        os.path.join(directory, "ldem_85s_10m_lroc_nac.img"), "wb"
    ) as f:
        f.write(rows)
    with open(label + ".part", "w") as f:
        f.write(
            f"""PDS_VERSION_ID = PDS3
/* A window of LOLA's LDEM_85S_10M (LRO-L-LOLA-4-GDR-V1.0), fetched
   by demos/cameras/demo_cameras_lroc_nac from the PDS Geosciences Node. */
RECORD_TYPE = FIXED_LENGTH
RECORD_BYTES = {2 * size}
FILE_RECORDS = {size}
^IMAGE = "LDEM_85S_10M_LROC_NAC.IMG"
OBJECT = IMAGE
  LINES = {size}
  LINE_SAMPLES = {size}
  SAMPLE_TYPE = LSB_INTEGER
  SAMPLE_BITS = 16
  UNIT = METER
  SCALING_FACTOR = 0.5
  OFFSET = 1737400.
END_OBJECT = IMAGE
OBJECT = IMAGE_MAP_PROJECTION
  MAP_PROJECTION_TYPE = "POLAR STEREOGRAPHIC"
  A_AXIS_RADIUS = 1737.4 <km>
  CENTER_LATITUDE = -90 <deg>
  CENTER_LONGITUDE = 0 <deg>
  MAP_SCALE = 10 <m/pix>
  LINE_PROJECTION_OFFSET = {LDEM_OFFSET - first_line:g} <pix>
  SAMPLE_PROJECTION_OFFSET = {LDEM_OFFSET - first_sample:g} <pix>
END_OBJECT = IMAGE_MAP_PROJECTION
END
"""
        )
    os.replace(label + ".part", label)
    print(f"Wrote {label} ({len(rows) / 1e6:.1f} MB)")


# ------------------------------------------------------------- the demo


def data_file(name):
    """Data file `name` the demos ship (LRO's ISDs): in
    $ERNIE_DEMO_DATA, else the package's ernie/demos/data, else (run
    from the source tree) its ISD fixtures, as C++ `demos::DataFile`."""
    here = os.path.dirname(os.path.abspath(__file__))
    for directory in (
        os.environ.get("ERNIE_DEMO_DATA"),
        os.path.join(here, "..", "data"),
        os.path.join(
            here, "..", "..", "ernie", "test", "fixtures", "isd"
        ),
    ):
        if directory and os.path.exists(
            os.path.join(directory, name)
        ):
            return os.path.join(directory, name)
    raise FileNotFoundError(
        f"The demos' data file {name} is in neither $ERNIE_DEMO_DATA "
        "nor the package"
    )


def latitude_longitude(point):
    """Latitude and east longitude (degrees) of body-fixed `point`."""
    d = np.asarray(point) / np.linalg.norm(point)
    return (
        math.degrees(math.asin(max(-1.0, min(1.0, d[2])))),
        math.degrees(math.atan2(d[1], d[0])),
    )


def transformed(matrix, point):
    """4 x 4 `matrix` applied to a 3-vector point."""
    return (matrix @ np.append(point, 1.0))[:3]


class Demo(ernie.demo.Demo):
    num_frames = 1

    def __init__(self, viewer, args):
        super().__init__(viewer, args)
        scene = self.scene
        self.nac = ernie.IsdLineScanner.from_file(
            data_file("toughLroNacLineScan.json")
        )
        scene.define("/World", ernie.Xform)
        # The middle line's ground, on the reference sphere.
        middle = FIRST_LINE + 0.5 * STRIP_LINES * STRIPS
        ground = np.asarray(
            self.nac.image_to_ground(
                middle, 0.5 * self.nac.samples, 0.0
            )
        )
        site = latitude_longitude(ground)
        # LOLA's terrain there, else the bare sphere, in a frame there:
        # x east, y north, z up. Fetched once, but not for the checks,
        # which run offline.
        label = ldem_label()
        if (
            not os.path.exists(label)
            and not args.test
            and can_download()
        ):
            try:
                fetch_ldem_window(
                    site[0],
                    site[1],
                    (CELLS - 1) // 2 * int(CELL_SIZE / LDEM_SCALE)
                    + 40,
                )
            except (OSError, RuntimeError) as error:
                print(error)
        if os.path.exists(label):
            self.dem = ernie.read_polar_stereographic_dem(label)
        else:
            print(
                f"No LOLA terrain in {os.path.dirname(label)}; the "
                "bare sphere stands in."
            )
            self.dem = ernie.PolarStereographicDem(MOON_RADIUS)
        terrain = ernie.local_terrain_of(
            self.dem, site[0], site[1], CELLS, CELL_SIZE
        )
        self.body_to_local = np.asarray(terrain["body_to_local"])
        # The regolith's normal albedo, about 0.1, flat (its reddening
        # left out).
        scene.define(
            "/World/Looks/Regolith",
            ernie.Material,
            reflectance=[250.0, 0.12, 800.0, 0.12],
        )
        scene.define(
            "/World/Terrain",
            ernie.Terrain,
            heights=terrain["heights"],
            origin=terrain["origin"],
            cell_size=terrain["cell_size"],
            material_binding="/World/Looks/Regolith",
        )

        # The Sun when the middle line was taken.
        taken = self.nac.image_time(middle)
        to_sun = np.asarray(self.nac.sun_position(taken)) - ground
        distance = float(np.linalg.norm(to_sun))
        local = self.body_to_local[:3, :3] @ (to_sun / distance)
        print(
            f"{-site[0]:.3f} S, {site[1]:.2f} E; the Sun "
            f"{distance / AU:.4f} au away, "
            f"{math.degrees(math.asin(local[2])):.2f} degrees up"
        )
        scene.define(
            "/World/Sun",
            ernie.DistantLight,
            irradiance=SOLAR_CONSTANT * (AU / distance) ** 2,
            physical_illuminant=tokens.SUN,
            angle=math.degrees(
                2.0 * math.asin(SUN_RADIUS / distance)
            ),
        ).set(tokens.TRANSFORM, ernie.distant_light_transform(local))

        self.define_nac()
        # The viewer's own view: the site from 3 km up to the
        # south-west of it, the Sun low behind.
        view = scene.define(
            "/World/View",
            ernie.Camera,
            focal_length=24.0,
            clipping_range=(10.0, 1e5),
            exposure_time=1.0 / 250.0,
        )
        view.set(
            tokens.TRANSFORM,
            ernie.look_at_transform(
                (-4e3, -4e3, 3e3), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)
            ),
        )
        scene.define(
            "/Render/Settings",
            ernie.RenderSettings,
            camera=view,
            resolution=(480, 320),
            ambient=0.0,
        )
        if viewer is not None:
            viewer.active_camera = view.path

    def save_output(self, path):
        """The NAC's 8-bit image of the lines, as it sends it."""
        write_png(path, self.strip_image()[..., :3])

    def test_final(self):
        self.camera().sensor_noise = False
        self.check_strips_see_the_ground_where_the_isd_puts_it()
        self.check_companding()
        self.camera().sensor_noise = True

    # -------------------------------------------------------- helpers

    def camera(self):
        return ernie.Camera.at(self.scene, "/World/Nac")

    def define_nac(self):
        """The NAC as published, its geometry the ISD's: the
        KLI-5001G's QE (> 60% over 400 to 760 nm, section 3.1.2), gain,
        noise and full well, a 12-bit ADC, the even and odd channels,
        the 12- to 8-bit table and photographic outputs to show."""
        self.scene.define(
            "/World/Nac",
            ernie.Camera,
            sensor_type="mono",
            sensor_qe_mono=[
                390.0,
                0.0,
                400.0,
                0.6,
                760.0,
                0.6,
                770.0,
                0.0,
            ],
            exposure_f_stop=NAC_F_NUMBER,
            exposure_iso=100.0,  # the gain is the ADC's
            sensor_electrons_per_dn=NAC_ELECTRONS_PER_DN,
            sensor_read_noise=NAC_READ_NOISE,
            sensor_full_well=NAC_FULL_WELL,
            sensor_adc_bits=12,
            sensor_black_level=0.0,
            sensor_noise=True,
        )
        self.use_strip(0, False)
        paths = []
        for var in ("RawDN", "Electrons", "PrimId", "LdrColor"):
            self.scene.define(
                f"/Render/Nac/{var}", ernie.RenderVar, source_name=var
            )
            paths.append(ernie.Path(f"/Render/Nac/{var}"))
        types = [
            "qe",
            "noise",
            "adc",
            "readoutTaps",
            "compandTable",
            "cie",
            "exposure",
        ]
        filters = [
            self.scene.define(
                f"/Render/Nac/Chain/Step{i}",
                ernie.SensorFilter,
                filter_type=kind,
            )
            for i, kind in enumerate(types)
        ]
        filters[3].set("inputs:taps", 2.0)
        self.scene.define(
            "/Render/Nac",
            ernie.RenderProduct,
            camera="/World/Nac",
            resolution=(self.nac.samples, STRIP_LINES),
        ).set(tokens.ORDERED_VARS, paths).set(
            tokens.FILTERS, [ernie.Path(f.path) for f in filters]
        )

    def use_strip(self, strip, at_end):
        """Camera /World/Nac as strip `strip` of the lines, posed at its
        end (or its start) in the local frame."""
        self.nac.author_strip(
            self.scene,
            "/World/Nac",
            FIRST_LINE + strip * STRIP_LINES,
            STRIP_LINES,
            at_end=at_end,
        )
        camera = self.camera()
        pose = np.asarray(camera.get(tokens.TRANSFORM))
        camera.set(tokens.TRANSFORM, self.body_to_local @ pose)

    def render_strip(self, strip):
        """Strip `strip` rendered: the camera stepped at its start, then
        at its end, the rows its lines; its vars by name."""
        self.use_strip(strip, False)
        self.renderer.step(
            self.scene, render_products=["/Render/Nac"]
        )
        self.use_strip(strip, True)
        frame = self.renderer.step(
            self.scene, render_products=["/Render/Nac"]
        )["/Render/Nac"].frames[0]
        return {
            name: np.asarray(
                frame.render_vars[f"/Render/Nac/{name}"].map()
            )
            for name in ("RawDN", "Electrons", "PrimId")
        }

    def strip_image(self):
        """The 8-bit image of the lines, uint8 (lines, samples, 4)
        grey."""
        samples = self.nac.samples
        pixels = np.full(
            (STRIP_LINES * STRIPS, samples, 4), 255, np.uint8
        )
        for k in range(STRIPS):
            # A mono sensor's numbers are each pixel's first channel.
            raw = self.render_strip(k)["RawDN"].reshape(
                STRIP_LINES, samples, -1
            )[..., 0]
            rows = slice(k * STRIP_LINES, (k + 1) * STRIP_LINES)
            pixels[rows, :, :3] = raw[..., None].astype(np.uint8)
        return pixels

    # --------------------------------------------------------- checks

    def check_strips_see_the_ground_where_the_isd_puts_it(self):
        """Balls on the ground (lifted 20 m up the pixels' rays, clear
        of the terrain's cells) where the ISD's model puts three pixels'
        rays on LOLA's terrain show on those pixels of their strips: the
        ISD's geometry, as Ernie draws it in the local frame."""
        samples = self.nac.samples
        pixels = [
            (FIRST_LINE + 100, 300),
            (FIRST_LINE + 256, 1266),
            (FIRST_LINE + 400, 2200),
        ]
        for i, (line, sample) in enumerate(pixels):
            at_line, s = line + 0.5, sample + 0.5
            h = 0.0
            for _ in range(4):
                ground = np.asarray(
                    self.nac.image_to_ground(at_line, s, h)
                )
                h = self.dem.height_at(*latitude_longitude(ground))
            eye = np.asarray(
                self.nac.position(self.nac.image_time(at_line))
            )
            towards = eye - ground
            lifted = ground + 20.0 * towards / np.linalg.norm(towards)
            ball = f"/World/Marker{i}"
            # About two lines and 40 samples across.
            self.scene.define(
                ball,
                ernie.Sphere,
                radius=15.0,
                translate=tuple(
                    transformed(self.body_to_local, lifted)
                ),
            )
            strip = (line - FIRST_LINE) // STRIP_LINES
            ids = self.render_strip(strip)["PrimId"].reshape(
                STRIP_LINES, samples, -1
            )[..., 0]
            hits = [
                (r + 0.5, c + 0.5)
                for r, c in zip(*np.nonzero(ids))
                if str(self.renderer.prim_path(int(ids[r, c])))
                == ball
            ]
            self.scene.remove(ernie.Path(ball))
            assert hits, "each marker should show in its strip"
            centre = np.mean(hits, axis=0)
            row = at_line - FIRST_LINE - strip * STRIP_LINES
            print(
                f"marker {i}: ({centre[0]:.2f}, {centre[1]:.2f}) for "
                f"({row:.2f}, {s:.2f})"
            )
            assert (
                abs(centre[0] - row) < 1.0
                and abs(centre[1] - s) < 1.5
            ), "a marker should land on its pixel"

    def check_companding(self):
        """Every pixel's 8-bit number is the table's of its 12-bit one,
        its electrons over the gain (either side of a half DN, which the
        GPU's floats may round either way); some of the ground is
        lit."""
        out = self.render_strip(STRIPS // 2)
        electrons = np.minimum(
            out["Electrons"].ravel(), NAC_FULL_WELL
        )
        raw = out["RawDN"].ravel()

        def code(e, scale):
            twelve = math.floor(
                e * scale / NAC_ELECTRONS_PER_DN + 0.5
            )
            return nac_compand(min(max(twelve, 0), 4095))

        for e, dn in zip(electrons.tolist(), raw.tolist()):
            assert dn in (code(e, 1 - 1e-6), code(e, 1 + 1e-6)), (
                "the NAC should send its table's 8-bit numbers"
            )
        lit = int(np.count_nonzero(electrons > 1000.0))
        print(f"NAC: {lit} lit pixels of {raw.size}")
        assert lit > 0, "the low Sun should light some of the ground"


if __name__ == "__main__":
    ernie.demo.main(Demo)
