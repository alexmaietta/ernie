# SPDX-License-Identifier: BSD-3-Clause
"""LROC's Narrow Angle Camera scanning Tycho's central peak as image
M1282557272RE's ISD has it, on LROC's 2 m stereo terrain (fetched on
the first run).

References:
- M. S. Robinson et al., "Lunar Reconnaissance Orbiter Camera (LROC)
  Instrument Overview", Space Science Reviews 150:81-124, 2010: Table 2
  (the NACs) and Appendix A (their companding).
- USGS Astrogeology, ISIS and ALE (the ISD, from the image's EDR with
  SPICE from the USGS web service), https://github.com/DOI-USGS/ISIS3
  and https://github.com/DOI-USGS/ale
- M. R. Henriksen et al., "Extracting accurate and precise topography
  from LROC narrow angle camera stereo observations", Icarus
  283:122-137, 2017 (the NAC DTMs; NAC_DTM_TYCHOPK).
- G. Kopp and J. L. Lean, "A new, lower value of total solar
  irradiance", Geophysical Research Letters 38:L01706, 2011 (1361
  W/m^2 at 1 au).
"""

import math
import os
import struct

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

# The image lines rendered: 8 strips of 256 near the image's southern
# end, beside the peak.
FIRST_LINE = 22528
STRIP_LINES = 256
STRIPS = 8

# The terrain about the middle line's ground: cells of 2 m, the DTM's
# own, 3.2 km across.
CELLS = 1601
CELL_SIZE = 2.0
MOON_RADIUS = 1737400.0

# The Sun's radius and the astronomical unit, metres; the total solar
# irradiance at 1 au, W/m^2 (Kopp and Lean 2011).
SUN_RADIUS = 6.957e8
AU = 1.495978707e11
SOLAR_CONSTANT = 1361.0

# The NACs' focal ratio, gain, noise, full well and 7 um pixels
# (Robinson et al. 2010, Table 2).
NAC_F_NUMBER = 3.577
NAC_ELECTRONS_PER_DN = 90.5
NAC_READ_NOISE = 101.0
NAC_FULL_WELL = 334000.0
NAC_PIXEL_UM = 7.0

# NAC-R's responsivity, (DN/ms) / (W/m^2/sr/um), and nonlinearity limit,
# DN (ISIS lronaccal; Humm et al. 2016).
NAC_RESPONSIVITY = 27.746
NAC_NONLINEAR_LIMIT = 600.0


def nac_qe():
    """The flat QE over 400 to 760 nm that gives that responsivity
    under a flat spectrum, the optics' losses in it."""
    planck = 6.62607015e-34
    light = 299792458.0
    area = 1e-6 * NAC_PIXEL_UM * 1e-6 * NAC_PIXEL_UM
    optics = math.pi / (4.0 * NAC_F_NUMBER * NAC_F_NUMBER)
    return (
        1e3
        * NAC_RESPONSIVITY
        * NAC_ELECTRONS_PER_DN
        * planck
        * light
        / (1e6 * 360e-9 * optics * area * 580e-9)
    )


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


# ------------------------------------------------------------------ DTM

# LROC's NAC DTM mosaic of the peak: a GeoTIFF of 32-bit reals, one row
# a strip, as its PDS3 label gives it.
DTM_URL = (
    "https://pds.mcp.nasa.gov/data/store/img/lunar_reconnaissance_"
    "orbiter/pds4/lroc/lro-l-lroc-5-rdr/LROLRC_2001/DATA/SDP/NAC_DTM/"
    "TYCHOPK/NAC_DTM_TYCHOPK.TIF"
)
DTM_LINES = 30892
DTM_SAMPLES = 12240
DTM_PIXELS_PER_DEGREE = 15161.675212075
DTM_PARALLEL = -43.3
DTM_NORTH = -42.3668223
DTM_WEST = 348.0633983


def dtm_label():
    """The cached window's label."""
    return os.path.join(
        cache_dir(), "lroc", "nac_dtm_tychopk_nac.lbl"
    )


def fetch_bytes(url, first, count):
    """The bytes of `url` from `first` for `count`."""
    return fetch_ranges(url, [(first, first + count - 1)])[0]


def row_offsets(url, first_row, rows):
    """The byte offset of every row of a little-endian TIFF of one row a
    strip (its StripOffsets, tag 273)."""
    header = fetch_bytes(url, 0, 8)
    if header[:2] != b"II":
        raise RuntimeError(f"{url} is not a little-endian TIFF")
    ifd = struct.unpack("<I", header[4:8])[0]
    entries = struct.unpack("<H", fetch_bytes(url, ifd, 2))[0]
    table = fetch_bytes(url, ifd + 2, 12 * entries)
    for e in range(entries):
        tag, kind, _, at = struct.unpack(
            "<HHII", table[12 * e : 12 * e + 12]
        )
        if tag != 273:
            continue
        size = 8 if kind == 16 else 4
        data = fetch_bytes(url, at + size * first_row, size * rows)
        return list(
            struct.unpack(f"<{rows}{'Q' if size == 8 else 'I'}", data)
        )
    raise RuntimeError(f"{url} has no strip offsets")


def fetch_dtm_window(latitude, longitude, half):
    """Fetches the DTM's window `half` metres about a point into the
    cache, as an image and a PDS3 label `ernie.read_cylindrical_dem`
    reads."""
    along = DTM_PIXELS_PER_DEGREE
    across = along * math.cos(math.radians(DTM_PARALLEL))
    degrees = math.degrees(half / MOON_RADIUS)
    first_row = math.floor((DTM_NORTH - latitude - degrees) * along)
    rows = math.ceil(2.0 * degrees * along)
    spread = degrees / math.cos(math.radians(latitude))
    east = math.fmod(longitude - DTM_WEST + 720.0, 360.0)
    first_column = math.floor((east - spread) * across)
    columns = math.ceil(2.0 * spread * across)
    if (
        first_row + rows > DTM_LINES
        or first_column + columns > DTM_SAMPLES
    ):
        raise RuntimeError("The window leaves the DTM")
    print(f"Fetching {rows} rows of {columns} samples from {DTM_URL}")
    offsets = row_offsets(DTM_URL, first_row, rows)
    # Each run of contiguous rows in one request (the server serves one
    # range a request).
    row_bytes = 4 * DTM_SAMPLES
    data = bytearray()
    start = 0
    while start < rows:
        end = start + 1
        while (
            end < rows
            and offsets[end] == offsets[end - 1] + row_bytes
        ):
            end += 1
        run = fetch_bytes(
            DTM_URL, offsets[start], (end - start) * row_bytes
        )
        for r in range(end - start):
            at = r * row_bytes + 4 * first_column
            data += run[at : at + 4 * columns]
        start = end
    label = dtm_label()
    directory = os.path.dirname(label)
    os.makedirs(directory, exist_ok=True)
    with open(
        os.path.join(directory, "nac_dtm_tychopk_nac.img"), "wb"
    ) as f:
        f.write(data)
    with open(label + ".part", "w") as f:
        f.write(
            f"""PDS_VERSION_ID = PDS3
/* A window of LROC's NAC_DTM_TYCHOPK (LRO-L-LROC-5-RDR-V1.0), fetched
   by demos/cameras/demo_cameras_lroc_nac from the PDS. */
RECORD_TYPE = FIXED_LENGTH
RECORD_BYTES = {4 * columns}
FILE_RECORDS = {rows}
^IMAGE = "NAC_DTM_TYCHOPK_NAC.IMG"
OBJECT = IMAGE
  LINES = {rows}
  LINE_SAMPLES = {columns}
  SAMPLE_TYPE = PC_REAL
  SAMPLE_BITS = 32
  UNIT = METER
END_OBJECT = IMAGE
OBJECT = IMAGE_MAP_PROJECTION
  MAP_PROJECTION_TYPE = "EQUIRECTANGULAR"
  A_AXIS_RADIUS = 1737.4 <KM>
  CENTER_LATITUDE = {DTM_PARALLEL:.12g} <DEG>
  MAP_RESOLUTION = {along:.12g} <PIX/DEG>
  MAXIMUM_LATITUDE = {DTM_NORTH - first_row / along:.12g} <DEG>
  WESTERNMOST_LONGITUDE = {DTM_WEST + first_column / across:.12g} <DEG>
END_OBJECT = IMAGE_MAP_PROJECTION
END
"""
        )
    os.replace(label + ".part", label)
    print(f"Wrote {label} ({len(data) / 1e6:.1f} MB)")


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
        # NAC-R's per-column readout (lroNacRCalibration.csv, from
        # ISIS's calibration data).
        table = np.loadtxt(
            data_file("lroNacRCalibration.csv"),
            delimiter=",",
            skiprows=2,
        )
        self.columns = {
            name: table[:, i].astype(np.float32).astype(np.float64)
            for i, name in enumerate(
                ("gain", "dark", "offset", "a", "b", "c")
            )
        }
        self.nac = ernie.IsdLineScanner.from_file(
            data_file("lroNacM1282557272RE.json")
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
        # The NAC DTM there inside LOLA's terrain (the lroc demo's cached
        # window), else the bare sphere, in a frame there: x east, y
        # north, z up. Fetched once, but not for the checks, which run
        # offline.
        half = 0.5 * (CELLS - 1) * CELL_SIZE + 200.0
        label = dtm_label()
        if (
            not os.path.exists(label)
            and not args.test
            and can_download()
        ):
            try:
                fetch_dtm_window(site[0], site[1], half)
            except (OSError, RuntimeError) as error:
                print(error)
        if os.path.exists(label):
            self.dtm = ernie.read_cylindrical_dem(label)
        else:
            print(
                f"No NAC DTM in {os.path.dirname(label)}; LOLA's terrain "
                "or the bare sphere stands in."
            )
            self.dtm = ernie.CylindricalDem(MOON_RADIUS)
        lola = os.path.join(cache_dir(), "lola", "ldem_512_lroc.lbl")
        self.lola = (
            ernie.read_cylindrical_dem(lola)
            if os.path.exists(lola)
            else ernie.CylindricalDem(MOON_RADIUS)
        )
        terrain = ernie.local_terrain_of(
            self.dtm,
            site[0],
            site[1],
            CELLS,
            CELL_SIZE,
            fallback=self.lola,
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
            f"{-site[0]:.3f} S, {math.fmod(site[1] + 360.0, 360.0):.2f} E; "
            f"the Sun {distance / AU:.4f} au away, "
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

        self.define_nac()
        # The viewer's own view: the flank from 1.5 km up to its
        # south-west.
        view = scene.define(
            "/World/View",
            ernie.Camera,
            focal_length=24.0,
            clipping_range=(10.0, 1e5),
            # Sunlit regolith's exposure: the Looney 11 rule, f/11 at
            # 1/ISO seconds.
            exposure_f_stop=11.0,
            exposure_iso=100.0,
            exposure_time=1.0 / 100.0,
        )
        view.set(
            tokens.TRANSFORM,
            ernie.look_at_transform(
                (-1.8e3, -1.8e3, 1.5e3),
                (0.0, 0.0, 0.0),
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
        """The NAC's 8-bit image of the lines, as it sends it."""
        write_png(path, self.strip_image()[..., :3])

    def test_final(self):
        self.camera().sensor_noise = False
        self.check_strips_see_the_ground_where_the_isd_puts_it()
        self.check_readout()
        self.check_responsivity()
        self.camera().sensor_noise = True

    # -------------------------------------------------------- helpers

    def camera(self):
        return ernie.Camera.at(self.scene, "/World/Nac")

    def define_nac(self):
        """The NAC's sensor and readout as published; the ISD sets its
        geometry and line time."""
        self.scene.define(
            "/World/Nac",
            ernie.Camera,
            sensor_type="mono",
            sensor_qe_mono=[
                395.0,
                0.0,
                405.0,
                nac_qe(),
                755.0,
                nac_qe(),
            ]
            + [765.0, 0.0],
            spectral_range=(390.0, 770.0),
            spectral_bands=8,
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
        for var in (
            "RawDN",
            "Electrons",
            "PrimId",
            "LdrColor",
            "SpectralRadiance",
        ):
            self.scene.define(
                f"/Render/Nac/{var}", ernie.RenderVar, source_name=var
            )
            paths.append(ernie.Path(f"/Render/Nac/{var}"))
        types = [
            "qe",
            "noise",
            "adc",
            "readoutTable",
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
        for name in ("gain", "dark", "offset", "a", "b", "c"):
            filters[3].set(
                f"inputs:{name}", self.columns[name].tolist()
            )
        filters[3].set("inputs:limit", NAC_NONLINEAR_LIMIT)
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
            for name in (
                "RawDN",
                "Electrons",
                "PrimId",
                "SpectralRadiance",
            )
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

    def height_at(self, point):
        """The terrain's height (m) at body-fixed `point`'s latitude and
        longitude: the DTM's, else LOLA's."""
        at = latitude_longitude(point)
        h = self.dtm.height_at(*at)
        return self.lola.height_at(*at) if math.isnan(h) else h

    def check_strips_see_the_ground_where_the_isd_puts_it(self):
        """Balls where the ISD puts three pixels' rays on the terrain
        show on those pixels."""
        samples = self.nac.samples
        pixels = [
            (FIRST_LINE + 100, 600),
            (FIRST_LINE + 1100, 2532),
            (FIRST_LINE + 1900, 4400),
        ]
        for i, (line, sample) in enumerate(pixels):
            at_line, s = line + 0.5, sample + 0.5
            h = 0.0
            for _ in range(6):
                ground = np.asarray(
                    self.nac.image_to_ground(at_line, s, h)
                )
                h = self.height_at(ground)
            eye = np.asarray(
                self.nac.position(self.nac.image_time(at_line))
            )
            towards = eye - ground
            lifted = ground + 10.0 * towards / np.linalg.norm(towards)
            ball = f"/World/Marker{i}"
            # About ten pixels across.
            self.scene.define(
                ball,
                ernie.Sphere,
                radius=3.0,
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
                and abs(centre[1] - s) < 1.0
            ), "a marker should land on its pixel"

    def measured(self, electrons):
        """The 12-bit numbers a strip's columns read for `electrons`
        (lines, samples): flatfield, nonlinearity (bisection), offset
        and dark."""
        k = self.columns
        gain = np.where(k["gain"] > 0.0, k["gain"], 1.0)
        lin = (
            np.floor(
                np.minimum(electrons, NAC_FULL_WELL)
                / NAC_ELECTRONS_PER_DN
                + 0.5
            )
            * gain
        )

        def h(y):
            return 1.0 / np.maximum(
                k["a"] * np.power(k["b"], y) + k["c"], 1e-6
            )

        lo, hi = lin, lin + h(lin)
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            below = mid - h(mid) < lin
            lo, hi = (
                np.where(below, mid, lo),
                np.where(below, hi, mid),
            )
        y = 0.5 * (lo + hi)
        return (
            np.where(y < NAC_NONLINEAR_LIMIT, y, lin)
            - k["offset"]
            + k["dark"]
        )

    def check_readout(self):
        """Every 8-bit number is the companding table's of its column's
        readout, and the daylit ground sits within the ADC's range."""
        out = self.render_strip(STRIPS // 2)
        samples = self.nac.samples
        electrons = out["Electrons"].reshape(
            STRIP_LINES, samples, -1
        )[..., 0]
        raw = out["RawDN"].reshape(STRIP_LINES, samples, -1)[..., 0]
        twelve = np.clip(
            np.floor(self.measured(electrons) + 0.5), 0, 4095
        )
        expected = np.vectorize(nac_compand)(twelve)
        off = int(np.count_nonzero(np.abs(raw - expected) > 1.0))
        within = int(np.count_nonzero((raw > 0) & (raw < 255)))
        print(
            f"NAC: {within} of {raw.size} numbers within the range, "
            f"{off} off the readout"
        )
        assert off < raw.size // 1000, (
            "the NAC should send its columns' readout, companded"
        )
        assert within > raw.size // 2, (
            "the daylit ground should be exposed within the range"
        )

    def check_responsivity(self):
        """Under a flat spectrum the electrons, as DN per ms, are 27.746
        times the radiance: NAC-R's responsivity."""
        sun = ernie.DistantLight.at(self.scene, "/World/Sun")
        sun.set(
            physical_illuminant=tokens.CUSTOM,
            physical_custom_illuminant=[250.0, 1.0, 800.0, 1.0],
        )
        out = self.render_strip(STRIPS // 2)
        sun.physical_illuminant = tokens.SUN
        pixels = STRIP_LINES * self.nac.samples
        electrons = out["Electrons"].reshape(pixels, -1)[:, 0]
        radiance = out["SpectralRadiance"].reshape(-1, pixels)
        ms = 1e3 * self.camera().exposure_time
        dn = (
            float(np.sum(electrons, dtype=np.float64))
            / NAC_ELECTRONS_PER_DN
            / ms
        )
        # W/m^2/sr/nm to /um.
        light = (
            1e3
            * float(np.sum(radiance, dtype=np.float64))
            / radiance.shape[0]
        )
        print(f"NAC: {dn / light:.3f} (DN/ms) / (W/m^2/sr/um)")
        assert abs(dn / light / NAC_RESPONSIVITY - 1.0) < 0.01, (
            "the NAC should read its published responsivity"
        )


if __name__ == "__main__":
    ernie.demo.main(Demo)
