# SPDX-License-Identifier: BSD-3-Clause
"""LROC's Wide Angle Camera over Tycho as it took a real image: each
framelet posed by the image's ISD (ALE's from LRO's SPICE kernels), on
LOLA's terrain, under the Sun where the ISD puts it, through the WAC's
604 nm band and readout as published, its 11-bit numbers companded to
the 8 it sends; and the Narrow Angle Camera's readout from the same
orbit.

LOLA's terrain about Tycho is fetched into the cache on the first run
(27 MB; not under --test); without it the ground is the bare reference
sphere.

References:
- M. S. Robinson et al., "Lunar Reconnaissance Orbiter Camera (LROC)
  Instrument Overview", Space Science Reviews 150:81-124, 2010: Table 2
  (the NACs), Table 3 (the WAC), Table 4 (the WAC's bandpasses and
  responsivities) and Appendix A (the NAC's companding).
- LROC EDR/CDR Data Product Software Interface Specification, version
  1.18, 2010: section 3.3 (LRO:LOOKUP_CONVERSION_TABLE) and Appendix B
  (the WAC's companding table).
- USGS Astrogeology, usgscsm's test data (lroWacPushFrame.json, an ISD
  ALE wrote for a WAC image) and ALE,
  https://github.com/DOI-USGS/usgscsm and
  https://github.com/DOI-USGS/ale
- D. E. Smith et al., "The Lunar Orbiter Laser Altimeter Investigation
  on the Lunar Reconnaissance Orbiter Mission", Space Science Reviews
  150:209-241, 2010 (LOLA, whose LDEM is the terrain).
- G. A. Neumann, "LRO LOLA RDR and Derived Products Software Interface
  Specification", version 2.42, 2010 (LDEM labels and grids).
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

# Tycho's centre, planetocentric latitude and east longitude.
TYCHO_LATITUDE = -43.31
TYCHO_LONGITUDE = 348.78

# The terrain about it: cells of 100 m, 153.6 km across.
CELLS = 1537
CELL_SIZE = 100.0

# The image lines rendered, centred on Tycho.
LINES = 560

# The Sun's radius and the astronomical unit, metres; the total solar
# irradiance at 1 au, W/m^2 (Kopp and Lean 2011).
SUN_RADIUS = 6.957e8
AU = 1.495978707e11
SOLAR_CONSTANT = 1361.0

# ------------------------------------------------------------------ NAC

# NAC-L's published values (Robinson et al. 2010, Table 2; section
# 3.1.2 for the KLI-5001G's QE): its 700 mm telescope and 7 um pixels,
# focal ratio, gain, detector noise and full well.
NAC_FOCAL_MM = 700.0
NAC_PIXEL_UM = 7.0
NAC_F_NUMBER = 3.577
NAC_ELECTRONS_PER_DN = 90.5
NAC_READ_NOISE = 101.0
NAC_FULL_WELL = 334000.0
NAC_SIZE = 128


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


# ------------------------------------------------------------------ WAC

# Each WAC band's published values: Table 4's band name, effective
# wavelength, full width at half maximum and responsivity coefficient
# ((W/m^3/sr) / (DN/s)), and Table 3's focal ratio and pixel (9 um, 4 x
# 4 summed in the UV).
WAC_BANDS = [
    (320, 321.0, 32.0, 865.0, 5.65, 36.0),
    (360, 360.0, 15.0, 918.0, 5.65, 36.0),
    (415, 415.0, 36.0, 3727.0, 5.052, 9.0),
    (565, 566.0, 20.0, 3310.0, 5.052, 9.0),
    (605, 604.0, 20.0, 3377.0, 5.052, 9.0),
    (645, 643.0, 23.0, 3273.0, 5.052, 9.0),
    (690, 689.0, 39.0, 2612.0, 5.052, 9.0),
]

# The ISD's band: 604 nm.
ISD_BAND = 4

# The UV bands' framelets: 16 detector rows, read out 4 x 4 summed
# (Robinson et al. 2010, section 3.2).
UV_ROWS = 16

# The KAI-1001's gain, read noise and full well (Table 3).
WAC_ELECTRONS_PER_DN = 25.9
WAC_READ_NOISE = 66.0
WAC_FULL_WELL = 46100.0

# The WAC's 11- to 8-bit table, each 8-bit code's last 11-bit input DN:
# the LRO:LOOKUP_CONVERSION_TABLE of its EDR labels (codes 3 and 6,
# which no input reaches, repeat the bound before them), the same as
# the SIS's Appendix B table of all 2048 inputs.
WAC_COMPAND = [
    1, 2, 3, 3, 4, 5, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 17, 18, 19, 21,
    23, 24, 26, 28, 30, 32, 33, 36, 38, 40, 42, 44, 47, 49, 51, 54, 56,
    59, 62, 64, 67, 70, 73, 76, 79, 82, 85, 88, 92, 95, 98, 102, 105,
    109, 113, 116, 120, 124, 128, 132, 136, 140, 144, 148, 152, 156, 161,
    165, 170, 174, 179, 183, 188, 193, 198, 203, 208, 213, 218, 223, 228,
    233, 238, 244, 249, 255, 260, 266, 271, 277, 283, 289, 295, 301, 307,
    313, 319, 325, 331, 337, 344, 350, 357, 363, 370, 376, 383, 390, 397,
    404, 411, 418, 425, 432, 439, 446, 454, 461, 468, 476, 483, 491, 499,
    506, 514, 522, 530, 538, 546, 554, 562, 570, 579, 587, 595, 604, 612,
    621, 630, 638, 647, 656, 665, 674, 683, 692, 701, 710, 719, 728, 738,
    747, 756, 766, 776, 785, 795, 805, 814, 824, 834, 844, 854, 864, 874,
    885, 895, 905, 916, 926, 937, 947, 958, 969, 979, 990, 1001, 1012,
    1023, 1034, 1045, 1056, 1068, 1079, 1090, 1102, 1113, 1125, 1136,
    1148, 1160, 1171, 1183, 1195, 1207, 1219, 1231, 1243, 1255, 1268,
    1280, 1292, 1305, 1317, 1330, 1342, 1355, 1368, 1380, 1393, 1406,
    1419, 1432, 1445, 1458, 1472, 1485, 1498, 1512, 1525, 1538, 1552,
    1566, 1579, 1593, 1607, 1621, 1635, 1648, 1663, 1677, 1691, 1705,
    1719, 1734, 1748, 1762, 1777, 1791, 1806, 1821, 1835, 1850, 1865,
    1880, 1895, 1910, 1925, 1940, 1955, 1971, 1986, 2001, 2017, 2032,
    2047,
]  # fmt: skip


def wac_compand(x):
    """The 8-bit code of 11-bit `x` through `WAC_COMPAND`: the first
    code whose bound reaches it."""
    return min(int(np.searchsorted(WAC_COMPAND, x, side="left")), 255)


def effective_qe(band):
    """The band's effective quantum efficiency: with Ernie's camera
    equation it turns a flat spectral radiance L (W/m^3/sr) into Table
    4's L / RC DN per second, gain h c / (RC FWHM pi / (4 N^2) A
    lambda), the optics', filter's and fill factor's losses all in
    it."""
    _, effective, fwhm, responsivity, f_number, pixel = band
    planck = 6.62607015e-34  # J s
    light = 299792458.0  # m / s
    area = (pixel * 1e-6) ** 2
    optics = math.pi / (4.0 * f_number * f_number)
    return (
        WAC_ELECTRONS_PER_DN
        * planck
        * light
        / (
            responsivity
            * fwhm
            * 1e-9
            * optics
            * area
            * effective
            * 1e-9
        )
    )


# ----------------------------------------------------------------- LOLA

# LOLA's LDEM_512 tiles at the PDS Geosciences Node: 45 degrees of
# latitude by 90 of longitude, each 23040 lines of 46080 16-bit samples,
# 512 to the degree.
LDEM_ROOT = (
    "https://pds-geosciences.wustl.edu/lro/lro-l-lola-3-rdr-v1/"
    "lrolol_1xxx/data/lola_gdr/cylindrical/img/"
)
TILE_SAMPLES = 46080
PIXELS_PER_DEGREE = 512

# The window of LDEM_512 about Tycho and the WAC's image over it:
# latitudes 40 to 47 S, longitudes 345 to 352.5 E.
WINDOW_NORTH = -40.0
WINDOW_SOUTH = -47.0
WINDOW_WEST = 345.0
WINDOW_EAST = 352.5


def ldem_label():
    """The cached window's label."""
    return os.path.join(cache_dir(), "lola", "ldem_512_lroc.lbl")


def fetch_ldem_window():
    """Fetches the window from LDEM_512's two tiles south of the
    equator between 270 and 360 E, 256 lines a request, each line's
    samples in the window alone (HTTP range requests; a tile is 2.1
    GB), and writes it to the cache as an image and a PDS3 label of its
    own, which `ernie.read_cylindrical_dem` reads as it would the whole
    tile. As demos/cameras/lroc.cpp's `FetchLdemWindow`."""
    first = round((WINDOW_WEST - 270.0) * PIXELS_PER_DEGREE)
    samples = round((WINDOW_EAST - WINDOW_WEST) * PIXELS_PER_DEGREE)
    rows = bytearray()
    lines = 0
    for tile_north in (0, -45):
        top = max(min(WINDOW_NORTH, tile_north), tile_north - 45.0)
        bottom = max(WINDOW_SOUTH, tile_north - 45.0)
        if top <= bottom:
            continue
        name = f"ldem_512_{45 - tile_north:02d}s_{-tile_north:02d}s_270_360"
        url = LDEM_ROOT + name + ".img"
        line0 = round((tile_north - top) * PIXELS_PER_DEGREE)
        count = round((top - bottom) * PIXELS_PER_DEGREE)
        print(
            f"Fetching {count} lines of {samples} samples from {url}"
        )
        row_bytes = 2 * TILE_SAMPLES
        for start in range(line0, line0 + count, 256):
            batch = range(start, min(start + 256, line0 + count))
            ranges = [
                (
                    line * row_bytes + 2 * first,
                    line * row_bytes + 2 * (first + samples) - 1,
                )
                for line in batch
            ]
            for data in fetch_ranges(url, ranges):
                rows += data
            lines += len(batch)
    label = ldem_label()
    directory = os.path.dirname(label)
    os.makedirs(directory, exist_ok=True)
    with open(
        os.path.join(directory, "ldem_512_lroc.img"), "wb"
    ) as f:
        f.write(rows)
    with open(label + ".part", "w") as f:
        f.write(
            f"""PDS_VERSION_ID = PDS3
/* A window of LOLA's LDEM_512 (LRO-L-LOLA-4-GDR-V1.0), fetched by
   demos/cameras/lroc from the PDS Geosciences Node. */
RECORD_TYPE = FIXED_LENGTH
RECORD_BYTES = {2 * samples}
FILE_RECORDS = {lines}
^IMAGE = "LDEM_512_LROC.IMG"
OBJECT = IMAGE
  LINES = {lines}
  LINE_SAMPLES = {samples}
  SAMPLE_TYPE = LSB_INTEGER
  SAMPLE_BITS = 16
  UNIT = METER
  SCALING_FACTOR = 0.5
  OFFSET = 1737400.
END_OBJECT = IMAGE
OBJECT = IMAGE_MAP_PROJECTION
  MAP_PROJECTION_TYPE = "SIMPLE CYLINDRICAL"
  MAP_RESOLUTION = {PIXELS_PER_DEGREE} <pix/deg>
  A_AXIS_RADIUS = 1737.4 <km>
  MAXIMUM_LATITUDE = {WINDOW_NORTH:g} <deg>
  MINIMUM_LATITUDE = {WINDOW_NORTH - lines / PIXELS_PER_DEGREE:g} <deg>
  WESTERNMOST_LONGITUDE = {WINDOW_WEST:g} <deg>
  EASTERNMOST_LONGITUDE = {WINDOW_WEST + samples / PIXELS_PER_DEGREE:g} <deg>
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


def body_direction(latitude, longitude):
    """The unit body-fixed direction of a planetocentric latitude and
    east longitude, degrees."""
    lat, lon = math.radians(latitude), math.radians(longitude)
    return np.array(
        [
            math.cos(lat) * math.cos(lon),
            math.cos(lat) * math.sin(lon),
            math.sin(lat),
        ]
    )


def transformed(matrix, point):
    """4 x 4 `matrix` applied to a 3-vector point."""
    return (matrix @ np.append(point, 1.0))[:3]


class Demo(ernie.demo.Demo):
    num_frames = 1

    def __init__(self, viewer, args):
        super().__init__(viewer, args)
        scene = self.scene
        self.wac = ernie.IsdPushFrame.from_file(
            data_file("lroWacPushFrame.json")
        )
        scene.define("/World", ernie.Xform)
        # LOLA's terrain about Tycho, else the bare sphere, in a frame
        # there: x east, y north, z up. Fetched once, but not for the
        # checks, which run offline.
        label = ldem_label()
        if (
            not os.path.exists(label)
            and not args.test
            and can_download()
        ):
            try:
                fetch_ldem_window()
            except (OSError, RuntimeError) as error:
                print(error)
        if os.path.exists(label):
            self.dem = ernie.read_cylindrical_dem(label)
        else:
            print(
                f"No LOLA terrain in {os.path.dirname(label)}; the "
                "bare sphere stands in."
            )
            self.dem = ernie.CylindricalDem(1737400.0)
        terrain = ernie.local_terrain_of(
            self.dem,
            TYCHO_LATITUDE,
            TYCHO_LONGITUDE,
            CELLS,
            CELL_SIZE,
        )
        self.body_to_local = np.asarray(terrain["body_to_local"])
        # The mare and highland regolith's normal albedo, about 0.1, as
        # a flat spectrum (its reddening left out): an RGB grey's
        # sigmoid spectrum dips in the UV.
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

        # The image line over Tycho, and the Sun when it was taken.
        tycho = self.dem.reference_radius * body_direction(
            TYCHO_LATITUDE, TYCHO_LONGITUDE
        )
        line, _ = self.wac.ground_to_image(tycho)
        self.first_line = max(0, int(line) - LINES // 2)
        taken = self.wac.image_time(line)
        to_sun = np.asarray(self.wac.sun_position(taken)) - tycho
        distance = float(np.linalg.norm(to_sun))
        local = self.body_to_local[:3, :3] @ (to_sun / distance)
        print(
            f"Tycho at image line {line:.0f}; the Sun "
            f"{distance / AU:.4f} au away, "
            f"{math.degrees(math.asin(local[2])):.1f} degrees up"
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

        # The WAC, a framelet at a time, and the NAC from LRO where the
        # WAC saw Tycho, looking down at it.
        scene.define("/World/Wac", ernie.Camera)
        self.define_product(
            "/Render/Wac",
            "/World/Wac",
            (self.wac.samples, self.wac.framelet_lines),
            False,
        )
        self.use_wac_band(ISD_BAND)
        lro = transformed(
            self.body_to_local, np.asarray(self.wac.position(taken))
        )
        scene.define("/World/Nac", ernie.Camera).set(
            tokens.TRANSFORM,
            ernie.look_at_transform(
                tuple(lro), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)
            ),
        )
        self.define_product(
            "/Render/Nac", "/World/Nac", (NAC_SIZE, NAC_SIZE), True
        )
        self.use_nac()
        # The viewer's own view: Tycho from the south-west, 120 km off
        # and 60 km up.
        view = scene.define(
            "/World/View",
            ernie.Camera,
            focal_length=35.0,
            clipping_range=(100.0, 1e6),
            exposure_time=1.0 / 500.0,
        )
        view.set(
            tokens.TRANSFORM,
            ernie.look_at_transform(
                (-85e3, -85e3, 60e3),
                (0.0, 0.0, -2e3),
                (0.0, 0.0, 1.0),
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
        """The WAC's 8-bit image over Tycho, as it sends it."""
        write_png(path, self.strip()[..., :3])

    def test_final(self):
        self.camera("/World/Wac").sensor_noise = False
        self.camera("/World/Nac").sensor_noise = False
        self.check_framelets_see_the_ground_where_the_isd_puts_it()
        self.check_nac_companding()
        self.check_wac_companding()
        self.check_wac_responsivities()
        self.camera("/World/Wac").sensor_noise = True
        self.camera("/World/Nac").sensor_noise = True

    # -------------------------------------------------------- helpers

    def camera(self, path):
        return ernie.Camera.at(self.scene, path)

    def define_product(
        self, product, camera, resolution, photographic
    ):
        """RenderProduct `product` of `camera` at `resolution`, holding
        RawDN, Electrons and PrimId, and LdrColor when
        `photographic`."""
        names = ["RawDN", "Electrons", "PrimId"]
        if photographic:
            names.append("LdrColor")
        paths = []
        for var in names:
            path = f"{product}/{var}"
            self.scene.define(path, ernie.RenderVar, source_name=var)
            paths.append(ernie.Path(path))
        self.scene.define(
            product,
            ernie.RenderProduct,
            camera=camera,
            resolution=resolution,
        ).set(tokens.ORDERED_VARS, paths)

    def chain(self, product, types):
        """The sensor filter chain of `types` under `product`, as
        SensorFilter prims Step0, Step1, ...; returns them."""
        filters = []
        for i, kind in enumerate(types):
            path = f"{product}/Chain/Step{i}"
            if self.scene.has(ernie.Path(path)):
                self.scene.remove(ernie.Path(path))
            filters.append(
                self.scene.define(
                    path, ernie.SensorFilter, filter_type=kind
                )
            )
        ernie.RenderProduct.at(self.scene, product).set(
            tokens.FILTERS, [ernie.Path(f.path) for f in filters]
        )
        return filters

    def use_nac(self):
        """The NAC's sensor and readout: electrons, their noise, the
        12-bit ADC, the even and odd channels (sections 3.1.2 and
        4.1.1), the 12- to 8-bit table (compandTable's defaults,
        Appendix A's code 0), and photographic outputs to show."""
        # Section 3.1.2: the KLI-5001G's average QE through the 400-760
        # nm response is > 60%.
        self.camera("/World/Nac").set(
            focal_length=NAC_FOCAL_MM,
            horizontal_aperture=1e-3 * NAC_PIXEL_UM * NAC_SIZE,
            vertical_aperture=1e-3 * NAC_PIXEL_UM * NAC_SIZE,
            clipping_range=(1e3, 1e6),
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
            exposure_time=0.002,
            exposure_iso=100.0,  # the gain is the ADC's
            sensor_electrons_per_dn=NAC_ELECTRONS_PER_DN,
            sensor_read_noise=NAC_READ_NOISE,
            sensor_full_well=NAC_FULL_WELL,
            sensor_adc_bits=12,
            sensor_black_level=0.0,
            sensor_noise=True,
        )
        filters = self.chain(
            "/Render/Nac",
            [
                "qe",
                "noise",
                "adc",
                "readoutTaps",
                "compandTable",
                "cie",
                "exposure",
            ],
        )
        filters[3].set("inputs:taps", 2.0)

    def use_wac_band(self, index):
        """Camera /World/Wac's sensor and readout through WAC band
        `index` (its geometry is each framelet's, from the ISD): the
        passband Table 4's FWHM about the effective wavelength, its
        edges ramped over 2 nm about the half maxima so its area is the
        FWHM's, at the band's effective QE (the lens's transmission in
        it); then electrons, their noise, the 11-bit ADC and the stored
        table to 8 bits."""
        band = WAC_BANDS[index]
        qe = effective_qe(band)
        low = band[1] - 0.5 * band[2]
        high = band[1] + 0.5 * band[2]
        self.camera("/World/Wac").set(
            sensor_type="mono",
            sensor_qe_mono=[
                low - 1.0,
                0.0,
                low + 1.0,
                qe,
                high - 1.0,
                qe,
                high + 1.0,
                0.0,
            ],  # fmt: skip
            spectral_range=(low - 1.0, high + 1.0),
            spectral_bands=4,
            lens_transmission=1.0,
            exposure_f_stop=band[4],
            exposure_iso=100.0,  # the gain is the ADC's
            sensor_electrons_per_dn=WAC_ELECTRONS_PER_DN,
            sensor_read_noise=WAC_READ_NOISE,
            sensor_full_well=WAC_FULL_WELL,
            sensor_adc_bits=11,
            sensor_black_level=0.0,
            sensor_noise=True,
        )
        filters = self.chain(
            "/Render/Wac", ["qe", "noise", "adc", "compandLookup"]
        )
        filters[3].set("inputs:codes", 256.0)
        filters[3].set(
            "inputs:uppers", [float(u) for u in WAC_COMPAND]
        )

    def use_framelet(self, framelet):
        """Camera /World/Wac as framelet `framelet`, posed in the local
        frame; returns (first_line, rows_reversed, body-fixed eye)."""
        _, _, first_line, reversed_rows, _ = self.wac.author_framelet(
            self.scene, "/World/Wac", framelet
        )
        camera = self.camera("/World/Wac")
        pose = np.asarray(camera.get(tokens.TRANSFORM))
        camera.set(tokens.TRANSFORM, self.body_to_local @ pose)
        return first_line, reversed_rows, pose[:3, 3]

    def framelet_at(self, line):
        """The framelet holding image line `line`, in image order."""
        return line // self.wac.framelet_lines

    def render_vars(self, product):
        """Product `product`'s vars by name, rendered now."""
        frame = self.renderer.step(
            self.scene, render_products=[product]
        )[product].frames[0]
        return {
            name: np.asarray(
                frame.render_vars[f"{product}/{name}"].map()
            )
            for name in ("RawDN", "Electrons", "PrimId")
        }

    def strip(self):
        """The WAC's 8-bit image of lines [first_line, first_line +
        LINES), uint8 (rows, samples, 4) grey: each framelet rendered
        and stacked where the ISD says, its rows flipped if they run
        up."""
        samples = self.wac.samples
        height = self.wac.framelet_lines
        pixels = np.zeros((LINES, samples, 4), np.uint8)
        pixels[..., 3] = 255
        for f in range(
            self.framelet_at(self.first_line),
            self.framelet_at(self.first_line + LINES - 1) + 1,
        ):
            first_line, reversed_rows, _ = self.use_framelet(f)
            # A mono sensor's numbers are each pixel's first channel.
            raw = self.render_vars("/Render/Wac")["RawDN"].reshape(
                height, samples, -1
            )[..., 0]
            for r in range(height):
                line = first_line + r - self.first_line
                if not 0 <= line < LINES:
                    continue
                row = height - 1 - r if reversed_rows else r
                pixels[line, :, :3] = raw[row, :, None].astype(
                    np.uint8
                )
        return pixels

    # --------------------------------------------------------- checks

    def check_framelets_see_the_ground_where_the_isd_puts_it(self):
        """Small balls on the ground along three pixels' rays (lifted
        200 m up them, clear of the terrain's cells) show on those
        pixels of their framelets: the ISD's geometry, as Ernie draws
        it in the local frame."""
        pixels = [
            (self.first_line + 100, 200),
            (self.first_line + LINES // 2, 512),
            (self.first_line + LINES - 100, 820),
        ]
        height = self.wac.framelet_lines
        samples = self.wac.samples
        for i, (line, sample) in enumerate(pixels):
            at_line, s = line + 0.5, sample + 0.5
            # On the terrain: the ray's point at the DEM's height there,
            # settled in a few rounds.
            h = 0.0
            for _ in range(4):
                ground = np.asarray(
                    self.wac.image_to_ground(at_line, s, h)
                )
                d = ground / np.linalg.norm(ground)
                h = self.dem.height_at(
                    math.degrees(math.asin(d[2])),
                    math.degrees(math.atan2(d[1], d[0])),
                )
            first_line, reversed_rows, eye = self.use_framelet(
                self.framelet_at(line)
            )
            at = transformed(self.body_to_local, ground)
            towards = transformed(self.body_to_local, eye) - at
            up = at + 200.0 * towards / np.linalg.norm(towards)
            ball = f"/World/Marker{i}"
            # About 0.8 of a 75 m pixel across.
            self.scene.define(
                ball, ernie.Sphere, radius=30.0, translate=tuple(up)
            )
            ids = self.render_vars("/Render/Wac")["PrimId"].reshape(
                height, samples, -1
            )[..., 0]
            hits = [
                (r, c)
                for r, c in zip(*np.nonzero(ids))
                if str(self.renderer.prim_path(int(ids[r, c])))
                == ball
            ]
            self.scene.remove(ernie.Path(ball))
            assert hits, "each marker should show in its framelet"
            rows = [
                (height - 1 - r if reversed_rows else r) + 0.5
                for r, _ in hits
            ]
            centre = (
                np.mean(rows),
                np.mean([c + 0.5 for _, c in hits]),
            )
            row = at_line - first_line
            print(
                f"marker {i}: ({centre[0]:.2f}, {centre[1]:.2f}) for "
                f"({row:.2f}, {s:.2f})"
            )
            assert (
                abs(centre[0] - row) < 1.5
                and abs(centre[1] - s) < 1.5
            ), "a marker should land on its pixel"

    def check_nac_companding(self):
        """Every NAC pixel's 8-bit number is the table's of its 12-bit
        one, its electrons over the gain."""
        out = self.render_vars("/Render/Nac")
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
            # Either side of a half DN, which the GPU's floats may
            # round either way.
            assert dn in (code(e, 1 - 1e-6), code(e, 1 + 1e-6)), (
                "the NAC should send its table's 8-bit numbers"
            )
        lit = int(np.count_nonzero(electrons > 1000.0))
        print(f"NAC: {lit} lit pixels")
        assert lit > 100, "the NAC should see sunlit ground"

    def check_wac_companding(self):
        """Every WAC pixel's 8-bit number, from black to saturation, is
        the stored table's of its 11-bit one."""
        self.use_framelet(
            self.framelet_at(self.first_line + LINES // 2)
        )
        camera = self.camera("/World/Wac")
        exposure_time = camera.exposure_time
        seen = set()

        def code(e, scale):
            eleven = math.floor(
                e * scale / WAC_ELECTRONS_PER_DN + 0.5
            )
            return wac_compand(min(max(eleven, 0), 2047))

        exposure = 1e-5
        while exposure < 4.0:
            camera.exposure_time = exposure
            out = self.render_vars("/Render/Wac")
            electrons = np.minimum(
                out["Electrons"].ravel(), WAC_FULL_WELL
            )
            for e, dn in zip(
                electrons.tolist(), out["RawDN"].ravel().tolist()
            ):
                assert dn in (code(e, 1 - 1e-6), code(e, 1 + 1e-6)), (
                    "the WAC should send its stored table's codes"
                )
                seen.add(dn)
            exposure *= 1.5
        camera.exposure_time = exposure_time
        print(f"WAC: {len(seen)} codes seen")
        assert len(seen) > 30, (
            "the sweep should reach codes along the table"
        )
        # The full well fills first, at 1780 DN of the 11-bit 2047.
        saturated = wac_compand(
            math.floor(WAC_FULL_WELL / WAC_ELECTRONS_PER_DN + 0.5)
        )
        assert 0 in seen and saturated in seen, (
            "the sweep should run from black to the full well"
        )

    def check_wac_responsivities(self):
        """Under a flat spectrum, each band's DN per second over a
        framelet is the same radiance over its responsivity
        coefficient: rate times RC alike across the seven, Table 4's
        values as Ernie renders them. The UV bands render as the camera
        reads them, 4 x 4 summed into 36 um pixels (over the visible
        framelet's geometry: the UV optics' own differ, but a pixel's
        rate depends only on its area and the f-number)."""
        sun = ernie.DistantLight.at(self.scene, "/World/Sun")
        sun.set(
            physical_illuminant=tokens.CUSTOM,
            physical_custom_illuminant=[250.0, 1.0, 800.0, 1.0],
        )
        framelet = self.framelet_at(self.first_line + LINES // 2)
        exposure = 0.005
        samples = self.wac.samples
        product = ernie.RenderProduct.at(self.scene, "/Render/Wac")
        scaled = []
        for b, band in enumerate(WAC_BANDS):
            self.use_framelet(framelet)
            self.use_wac_band(b)
            camera = self.camera("/World/Wac")
            camera.set(sensor_noise=False, exposure_time=exposure)
            # The UV bands' framelets: 16 detector rows read 4 x 4
            # summed, 36 um pixels.
            summed = round(band[5] / WAC_BANDS[ISD_BAND][5])
            if summed > 1:
                camera.vertical_aperture = (
                    1e-3 * WAC_BANDS[ISD_BAND][5] * UV_ROWS
                )
                product.resolution = (
                    samples // summed,
                    UV_ROWS // summed,
                )
            electrons = self.render_vars("/Render/Wac")["Electrons"]
            rate = (
                float(np.mean(electrons, dtype=np.float64))
                / exposure
                / WAC_ELECTRONS_PER_DN
            )
            scaled.append(rate * band[3])
            product.resolution = (samples, self.wac.framelet_lines)
            print(
                f"WAC {band[0]} nm: {rate:.1f} DN/s, effective QE "
                f"{effective_qe(band):.4f}"
            )
        sun.physical_illuminant = tokens.SUN
        self.use_framelet(framelet)
        self.use_wac_band(ISD_BAND)
        self.camera("/World/Wac").sensor_noise = False
        for value in scaled:
            assert abs(value / scaled[0] - 1.0) < 0.01, (
                "the WAC's bands should read Table 4's responsivities"
            )


if __name__ == "__main__":
    ernie.demo.main(Demo)
