# SPDX-License-Identifier: BSD-3-Clause
"""Perseverance's Navcam, Hazcam and Mastcam-Z stereo pairs on a patch
of Mars, as published, each binned to 320 x 240.

References:
- J. N. Maki et al., "The Mars 2020 Engineering Cameras and Microphone
  on the Perseverance Rover: A Next-Generation Imaging System for Mars
  Exploration", Space Science Reviews 216:137, 2020,
  https://doi.org/10.1007/s11214-020-00765-9: Table 2 and the sections
  on the detector and flight software.
- J. F. Bell III et al., "The Mars 2020 Perseverance Rover Mast Camera
  Zoom (Mastcam-Z) Multispectral, Stereoscopic Imaging Investigation",
  Space Science Reviews 217:24, 2021,
  https://doi.org/10.1007/s11214-020-00755-x: Tables 3 and 5 and the
  focal plane section.
- J. F. Bell III et al., "Mineralogic and compositional properties of
  Martian soil and dust: Results from Mars Pathfinder", Journal of
  Geophysical Research 105(E1):1721-1755, 2000 (soil reflectance, dark
  in the blue and about 0.3 in the red and near infrared).
- J. Kannala and S. S. Brandt, "A Generic Camera Model and Calibration
  Method for Conventional, Wide-Angle, and Fish-Eye Lenses", IEEE TPAMI
  28(8):1335-1340, 2006 (the f-theta lens's k1).
"""

import math

import numpy as np

import ernie
import ernie.demo
from ernie import tokens
from ernie.demo import mean_linear_rgb, side_by_side, write_png

# The Sun's irradiance at Mars's mean distance, 1.5237 au, and its
# angular diameter there, degrees.
MARS_IRRADIANCE = 1361.0 / 1.5237**2
SUN_ANGLE = 0.35

# Every product's size: the detectors binned.
WIDTH = 320
HEIGHT = 240

EYES = ("Left", "Right")

# ------------------------------------------------- engineering cameras

# The Navcams' and Hazcams' CMV-20000 (Maki et al. 2020, Table 2): 5120
# x 3840 pixels of 6.4 um, 0.24 DN per electron, 8 e- dark noise, a
# 15000 e- full well and a 12-bit ADC.
CMV_PIXEL_UM = 6.4
CMV_WIDTH = 5120
CMV_HEIGHT = 3840
CMV_ELECTRONS_PER_DN = 1.0 / 0.24
CMV_DARK_NOISE = 8.0
CMV_FULL_WELL = 15000.0

# Each engineering camera's optics (Table 2): name, focal length (mm),
# f-number, best focus (m), stereo baseline (m) and fields of view
# (horizontal, vertical, diagonal; degrees).
NAVCAM = ("Navcam", 19.1, 12.0, 3.5, 0.424, (96.0, 73.0, 120.0))
HAZCAM = ("Hazcam", 14.0, 12.0, 0.9, 0.248, (136.0, 102.0, 172.0))
# The rear Hazcams: the front pair's optics, 93.4 cm apart.
REAR_HAZCAM = ("RearHazcam", 14.0, 12.0, 0.9, 0.934, HAZCAM[5])


def cmv_reaches():
    """The half widths the fields reach on the detector, mm: across,
    down and to the corner."""
    half_width = 0.5e-3 * CMV_PIXEL_UM * CMV_WIDTH
    half_height = 0.5e-3 * CMV_PIXEL_UM * CMV_HEIGHT
    return (
        half_width,
        half_height,
        math.hypot(half_width, half_height),
    )


def f_theta_k1(optics):
    """The f-theta lens's Kannala-Brandt k1 for the published fields:
    the least squares of theta (1 + k1 theta^2) = r / f over the three,
    theta each field's half angle and r the half width it reaches. The
    paper gives the calibrated model only as CAHVORE, without its
    coefficients."""
    numerator = 0.0
    denominator = 0.0
    for field, reach in zip(optics[5], cmv_reaches()):
        theta = math.radians(0.5 * field)
        r = reach / optics[1]
        cube = theta**3
        numerator += cube * (r - theta)
        denominator += cube * cube
    return numerator / denominator


def f_theta_angle(r, k1):
    """The angle off axis (radians) that lands `r` focal lengths out
    through the f-theta lens: theta (1 + k1 theta^2) = r, by Newton."""
    theta = r
    for _ in range(20):
        f = theta * (1.0 + k1 * theta * theta) - r
        theta -= f / (1.0 + 3.0 * k1 * theta * theta)
    return theta


# ----------------------------------------------------------- Mastcam-Z

# The KAI-2020CM (Bell et al. 2021, Table 5): 1600 x 1200 pixels of 7.4
# um, 15.6 e-/DN and an 11-bit ADC; each camera's full well and read
# noise.
KAI_PIXEL_UM = 7.4
KAI_WIDTH = 1600
KAI_HEIGHT = 1200
KAI_ELECTRONS_PER_DN = 15.6
KAI_FULL_WELL = (21826.0, 21818.0)
KAI_READ_NOISE = (22.0, 21.0)

# The zoom (26 mm at f/6.7 to 110 mm at f/9.5), the cameras' 24.4 cm
# baseline and each one's 1.15 degree toe-in (2.3 between them), and the
# detector's ~40% peak QE through its Bayer microlenses.
SHORTEST_FOCAL_MM = 26.0
LONGEST_FOCAL_MM = 110.0
WIDEST_F_NUMBER = 6.7
NARROWEST_F_NUMBER = 9.5
MASTCAM_BASELINE_M = 0.244
TOE_IN_DEG = 1.15
PEAK_QE = 0.40


def f_number_at(focal_mm):
    """The f-number at focal length `focal_mm`, linearly between the
    zoom's ends (the paper gives only those)."""
    t = (focal_mm - SHORTEST_FOCAL_MM) / (
        LONGEST_FOCAL_MM - SHORTEST_FOCAL_MM
    )
    return WIDEST_F_NUMBER + t * (
        NARROWEST_F_NUMBER - WIDEST_F_NUMBER
    )


# Each camera's filter wheel (Table 3): effective wavelength and half
# width at half maximum (nm) and neutral density; 0 the Bayer RGB's
# near-IR cut-off (by its green channel), 1 to 6 narrow bands, 7 the
# solar filter; the left's visible, the right's reaching 1022 nm.
FILTERS = (
    (
        (544.0, 41.0, 0.0),
        (800.0, 9.0, 0.0),
        (754.0, 10.0, 0.0),
        (677.0, 11.0, 0.0),
        (605.0, 9.0, 0.0),
        (528.0, 11.0, 0.0),
        (442.0, 12.0, 0.0),
        (590.0, 88.0, 6.0),
    ),
    (
        (544.0, 42.0, 0.0),
        (800.0, 9.0, 0.0),
        (866.0, 10.0, 0.0),
        (910.0, 12.0, 0.0),
        (939.0, 12.0, 0.0),
        (978.0, 10.0, 0.0),
        (1022.0, 19.0, 0.0),
        (880.0, 10.0, 5.0),
    ),
)

# Filter 0's red, green and blue Bayer channels (Table 3).
BAYER = (
    ((630.0, 43.0), (544.0, 41.0), (480.0, 46.0)),
    ((631.0, 43.0), (544.0, 42.0), (480.0, 46.0)),
)

# A generic front-illuminated silicon response (Ernie's mono sensor's),
# for the detector behind the narrow bands: the paper publishes no
# curve.
SILICON_NM = (350, 400, 500, 600, 700, 800, 900, 1000, 1100)
SILICON_QE = (0.15, 0.35, 0.55, 0.62, 0.55, 0.40, 0.22, 0.06, 0.0)


def silicon(nm):
    return float(np.interp(nm, SILICON_NM, SILICON_QE))


# ----------------------------------------------------------- the scene

# Mars soil's reflectance, roughly (Bell et al. 2000): dark in the blue,
# rising to about 0.3 by 750 nm and flat into the near infrared; and
# darker rock.
SOIL = [350.0, 0.05, 450.0, 0.08, 550.0, 0.14, 650.0, 0.25]
SOIL += [750.0, 0.30, 1100.0, 0.30]
ROCK = [350.0, 0.06, 550.0, 0.09, 750.0, 0.14, 1100.0, 0.15]

# Rocks on the ground ahead of the rover: x, y and radius, m.
ROCKS = (
    (-1.2, 3.0, 0.25),
    (0.9, 4.5, 0.4),
    (-2.5, 7.0, 0.6),
    (2.0, 9.0, 0.5),
    (0.3, 1.6, 0.12),
    (-0.6, 2.2, 0.18),
)

# The rigs, each looking down its -z with +y up, a heading (degrees
# clockwise from north, +y) and a tilt down: the mast's Navcams and
# Mastcam-Z about 2 m up looking 15 degrees down ahead (north), the
# front Hazcams 0.7 m up and 30 degrees down ahead, the rear ones behind
# the rover looking back. Path, eye, degrees down and heading.
NAVCAM_RIG = ("/World/Rover/Navcam", (0.0, 0.0, 2.0), 15.0, 0.0)
HAZCAM_RIG = ("/World/Rover/Hazcam", (0.0, 0.6, 0.7), 30.0, 0.0)
REAR_HAZCAM_RIG = (
    "/World/Rover/RearHazcam",
    (0.0, -0.9, 0.7),
    30.0,
    180.0,
)
MASTCAM_RIG = ("/World/Rover/MastcamZ", (0.3, 0.0, 2.0), 15.0, 0.0)

# The cameras the viewer can show: name and camera (a pair's left eye).
VIEWS = (
    ("Overview", "/World/View"),
    ("Navcam", "/World/Rover/Navcam/Left"),
    ("Front Hazcam", "/World/Rover/Hazcam/Left"),
    ("Rear Hazcam", "/World/Rover/RearHazcam/Left"),
    ("Mastcam-Z", "/World/Rover/MastcamZ/Left"),
)


def filter_name(e, number):
    """Filter `number` of eye `e`'s wheel, as Table 3 names it: L0 to
    L7, R0 to R7."""
    effective, _, density = FILTERS[e][number]
    name = f"{'L' if e == 0 else 'R'}{number} "
    if number == 0:
        return name + "Bayer RGB"
    name += f"{int(effective)} nm"
    if density > 0.0:
        name += f", solar (ND {int(density)})"
    return name


def rig_transform(rig):
    _, eye, down_deg, heading_deg = rig
    down = math.radians(down_deg)
    heading = math.radians(heading_deg)
    forward = np.array(
        [
            math.sin(heading) * math.cos(down),
            math.cos(heading) * math.cos(down),
            -math.sin(down),
        ]
    )
    return np.asarray(
        ernie.look_at_transform(
            eye, tuple(np.asarray(eye) + forward), (0.0, 0.0, 1.0)
        )
    )


def eye_pose(x, yaw):
    """Rig-local pose of an eye `x` along the rig's x, turned `yaw`
    (radians) about its +y."""
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            [c, 0.0, s, x],
            [0.0, 1.0, 0.0, 0.0],
            [-s, 0.0, c, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


class Demo(ernie.demo.Demo):
    num_frames = 1

    def __init__(self, viewer, args):
        super().__init__(viewer, args)
        self.focal_mm = SHORTEST_FOCAL_MM
        self.filters = [0, 0]
        scene = self.scene
        scene.define("/World", ernie.Xform)
        scene.define(
            "/World/Air",
            ernie.Atmosphere,
            preset=tokens.MARS,
            dust_optical_depth=0.5,
            ground_albedo=0.25,
        )
        # The Sun's measured spectrum at Mars's distance, 40 degrees up
        # in the south-east.
        up, azimuth = math.radians(40.0), math.radians(135.0)
        scene.define(
            "/World/Sun",
            ernie.DistantLight,
            irradiance=MARS_IRRADIANCE,
            physical_illuminant=tokens.SUN,
            angle=SUN_ANGLE,
        ).set(
            tokens.TRANSFORM,
            ernie.distant_light_transform(
                (
                    math.sin(azimuth) * math.cos(up),
                    math.cos(azimuth) * math.cos(up),
                    math.sin(up),
                )
            ),
        )
        scene.define(
            "/World/Looks/Soil", ernie.Material, reflectance=SOIL
        )
        scene.define(
            "/World/Looks/Rock", ernie.Material, reflectance=ROCK
        )
        scene.define(
            "/World/Ground",
            ernie.GroundPlane,
            material_binding="/World/Looks/Soil",
        )
        for i, (x, y, radius) in enumerate(ROCKS):
            scene.define(
                f"/World/Rock{i}",
                ernie.Sphere,
                radius=radius,
                material_binding="/World/Looks/Rock",
                translate=(x, y, 0.5 * radius),
            )

        scene.define("/World/Rover", ernie.Xform)
        self.define_engineering_pair(NAVCAM_RIG, NAVCAM)
        self.define_engineering_pair(HAZCAM_RIG, HAZCAM)
        self.define_engineering_pair(REAR_HAZCAM_RIG, REAR_HAZCAM)
        self.define_mastcam_pair()

        # The viewer's own view: the rover's patch from behind.
        view = scene.define(
            "/World/View",
            ernie.Camera,
            focal_length=18.0,
            sensor_noise=False,
            exposure_time=1.0 / 250.0,
        )
        view.set(
            tokens.TRANSFORM,
            ernie.look_at_transform(
                (1.5, -4.0, 2.5), (0.0, 4.0, 0.0), (0.0, 0.0, 1.0)
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

        # The camera the viewer shows (a pair's left eye), Mastcam-Z's
        # zoom and each eye's filter wheel.
        def show(index):
            if viewer is not None:
                viewer.active_camera = VIEWS[index][1]

        def zoom(focal):
            self.focal_mm = focal
            for e in range(2):
                self.use_mastcam_filter(
                    self.mastcam_camera(e), e, self.filters[e]
                )

        self.widgets.combo(
            "Camera", [name for name, _ in VIEWS], 0, on_change=show
        )
        self.widgets.slider(
            "Mastcam-Z focal length (mm)",
            SHORTEST_FOCAL_MM,
            LONGEST_FOCAL_MM,
            SHORTEST_FOCAL_MM,
            on_change=zoom,
        )

        def wheel(e):
            def turn(index):
                self.set_mastcam_filter(e, index)

            return turn

        for e in range(2):
            self.widgets.combo(
                f"Mastcam-Z {'left' if e == 0 else 'right'} filter",
                [filter_name(e, number) for number in range(8)],
                0,
                on_change=wheel(e),
            )

    def save_output(self, path):
        """The left Navcam, Hazcam and Mastcam-Z images and the right
        Navcam's, side by side, as the rover sends them."""
        products = [
            "/Render/NavcamLeft",
            "/Render/NavcamRight",
            "/Render/HazcamLeft",
            "/Render/MastcamZLeft",
        ]
        outputs = self.render_vars(products, "LdrColor")
        images = [outputs[p] for p in products]
        write_png(path, side_by_side(images)[..., :3])

    def test_final(self):
        for rig in (
            NAVCAM_RIG,
            HAZCAM_RIG,
            REAR_HAZCAM_RIG,
            MASTCAM_RIG,
        ):
            for eye in EYES:
                self.camera(f"{rig[0]}/{eye}").sensor_noise = False
        self.check_fields_of_view(NAVCAM, NAVCAM_RIG, 1.0)
        self.check_fields_of_view(HAZCAM, HAZCAM_RIG, 1.5)
        self.check_stereo(NAVCAM, NAVCAM_RIG, 5.0, 0.1)
        self.check_stereo(HAZCAM, HAZCAM_RIG, 1.0, 0.05)
        self.check_stereo(REAR_HAZCAM, REAR_HAZCAM_RIG, 1.0, 0.05)
        self.check_mastcam_rig()
        self.check_soil_is_red()
        self.check_near_infrared()
        self.check_every_filter_and_zoom()

    # -------------------------------------------------------- helpers

    def camera(self, path):
        return ernie.Camera.at(self.scene, path)

    def render_vars(self, products, var):
        """Var `var` of each of `products`, rendered now, by
        product."""
        outputs = self.renderer.step(
            self.scene, render_products=products
        )
        return {
            p: np.asarray(
                outputs[p].frames[0].render_vars[f"{p}/{var}"].map()
            )
            for p in products
        }

    def define_product(
        self, product, camera, types, photographic=True
    ):
        """Product `product` of `camera` at the binned size, with its
        readout chain of SensorFilter prims of `types` under it."""
        names = ["RawDN", "Electrons", "PrimId"]
        if photographic:
            names.append("LdrColor")
        paths = []
        for var in names:
            self.scene.define(
                f"{product}/{var}", ernie.RenderVar, source_name=var
            )
            paths.append(ernie.Path(f"{product}/{var}"))
        if self.scene.has(ernie.Path(f"{product}/Chain")):
            self.scene.remove(ernie.Path(f"{product}/Chain"))
        filters = [
            self.scene.define(
                f"{product}/Chain/Step{i}",
                ernie.SensorFilter,
                filter_type=kind,
            )
            for i, kind in enumerate(types)
        ]
        self.scene.define(
            product,
            ernie.RenderProduct,
            camera=camera,
            resolution=(WIDTH, HEIGHT),
        ).set(tokens.ORDERED_VARS, paths).set(
            tokens.FILTERS, [ernie.Path(f.path) for f in filters]
        )
        return filters

    def define_engineering_pair(self, rig, optics):
        """A Navcam or Hazcam pair (Maki et al. 2020, Table 2) on its
        rig, the baseline apart along the rig's x, parallel: the
        f-theta lens over the full frame, focused at its best focus, on
        the CMV-20000; its readout qe, bayer (RGGB; the paper does not
        give the layout), noise, adc, demosaic (RawDN the 12-bit
        numbers), then cie, exposure and jpeg at quality 85 for the
        8-bit images the rover sends (the paper gives the quality as
        selectable). The detector's QE is published only as 64.5% at
        550 nm, monochrome, so the channels keep Ernie's generic RGB
        curves."""
        name, focal, f_number, focus, baseline, _ = optics
        self.scene.define(rig[0], ernie.Xform).set(
            tokens.TRANSFORM, rig_transform(rig)
        )
        k1 = f_theta_k1(optics)
        for e, eye in enumerate(EYES):
            camera = self.scene.define(
                f"{rig[0]}/{eye}",
                ernie.Camera,
                lens_projection=tokens.FISHEYE,
                lens_fisheye=[k1, 0.0, 0.0, 0.0],
                focal_length=focal,
                horizontal_aperture=1e-3 * CMV_PIXEL_UM * CMV_WIDTH,
                vertical_aperture=1e-3 * CMV_PIXEL_UM * CMV_HEIGHT,
                clipping_range=(0.05, 1e4),
                f_stop=f_number,
                focus_distance=focus,
                exposure_f_stop=f_number,
                exposure_time=4e-5,
                exposure_mode="auto",  # the 8-bit images
                exposure_iso=100.0,  # the gain is the ADC's
                sensor_type="rgb",
                sensor_electrons_per_dn=CMV_ELECTRONS_PER_DN,
                sensor_read_noise=CMV_DARK_NOISE,
                sensor_full_well=CMV_FULL_WELL,
                sensor_adc_bits=12,
                sensor_black_level=0.0,
                sensor_noise=True,
            )
            camera.set(
                tokens.TRANSFORM,
                eye_pose((-0.5 if e == 0 else 0.5) * baseline, 0.0),
            )
            filters = self.define_product(
                f"/Render/{name}{eye}",
                camera.path,
                [
                    "qe",
                    "bayer",
                    "noise",
                    "adc",
                    "demosaic",
                    "cie",
                    "exposure",
                    "jpeg",
                ],
            )
            filters[7].set("inputs:quality", 85.0)

    def define_mastcam_pair(self):
        """Mastcam-Z's pair (Bell et al. 2021, Table 5) on its rig at
        the widest zoom, 24.4 cm apart, each toed in 1.15 degrees,
        through filter 0."""
        self.scene.define(MASTCAM_RIG[0], ernie.Xform).set(
            tokens.TRANSFORM, rig_transform(MASTCAM_RIG)
        )
        for e, eye in enumerate(EYES):
            camera = self.scene.define(
                f"{MASTCAM_RIG[0]}/{eye}", ernie.Camera
            )
            # The left turns towards +x, the right towards -x.
            camera.set(
                tokens.TRANSFORM,
                eye_pose(
                    (-0.5 if e == 0 else 0.5) * MASTCAM_BASELINE_M,
                    (-1.0 if e == 0 else 1.0)
                    * math.radians(TOE_IN_DEG),
                ),
            )
            self.set_mastcam_filter(e, 0)

    def mastcam_camera(self, e):
        return f"{MASTCAM_RIG[0]}/{EYES[e]}"

    def set_mastcam_filter(self, e, number):
        """Mastcam-Z eye `e` through filter `number`, with its readout:
        qe, bayer, noise, adc and demosaic for filter 0's colour, qe,
        noise and adc for a narrow band; then cie and exposure for the
        image to show."""
        self.filters[e] = number
        self.use_mastcam_filter(self.mastcam_camera(e), e, number)
        self.define_product(
            f"/Render/MastcamZ{EYES[e]}",
            self.mastcam_camera(e),
            [
                "qe",
                "bayer",
                "noise",
                "adc",
                "demosaic",
                "cie",
                "exposure",
            ]
            if number == 0
            else ["qe", "noise", "adc", "cie", "exposure"],
        )

    def use_mastcam_filter(self, path, e, number):
        """Mastcam-Z camera `path` (eye `e`) through filter `number` at
        the zoom's focal length: filter 0 an RGB sensor whose channels are
        Gaussians at Table 3's Bayer effective wavelengths and half
        widths peaking at ~40%; 1 to 7 one channel, the filter's
        passband (2 x HWHM about its effective wavelength, ramped over
        2 nm about the half maxima, times 10^-ND) on silicon scaled to
        the ~40% peak, the spectral range narrowed to it."""
        camera = self.camera(path)
        camera.set(
            focal_length=self.focal_mm,
            horizontal_aperture=1e-3 * KAI_PIXEL_UM * KAI_WIDTH,
            vertical_aperture=1e-3 * KAI_PIXEL_UM * KAI_HEIGHT,
            clipping_range=(0.05, 1e4),
            exposure_f_stop=f_number_at(self.focal_mm),
            exposure_time=2e-5,
            exposure_mode="auto",  # the 8-bit images
            exposure_iso=100.0,  # the gain is the ADC's
            sensor_electrons_per_dn=KAI_ELECTRONS_PER_DN,
            sensor_read_noise=KAI_READ_NOISE[e],
            sensor_full_well=KAI_FULL_WELL[e],
            sensor_adc_bits=11,
            sensor_black_level=0.0,
            sensor_noise=True,
        )
        if number == 0:
            curves = []
            for effective, hwhm in BAYER[e]:
                sigma = hwhm / math.sqrt(2.0 * math.log(2.0))
                curve = []
                for w in range(380, 781, 5):
                    x = (w - effective) / sigma
                    curve += [
                        float(w),
                        PEAK_QE * math.exp(-0.5 * x * x),
                    ]
                curves.append(curve)
            camera.set(
                sensor_type="rgb",
                sensor_qe_r=curves[0],
                sensor_qe_g=curves[1],
                sensor_qe_b=curves[2],
                spectral_range=(380.0, 780.0),
                spectral_bands=16,
            )
            return
        effective, hwhm, density = FILTERS[e][number]
        transmission = 10.0**-density
        peak = max(SILICON_QE)
        low, high = effective - hwhm, effective + hwhm
        curve = []

        def add(w, in_band):
            curve.extend(
                [
                    w,
                    in_band
                    * transmission
                    * PEAK_QE
                    * silicon(w)
                    / peak,
                ]
            )

        add(low - 1.0, 0.0)
        step = max(1.0, (high - low - 2.0) / 8.0)
        nm = low + 1.0
        while nm <= high - 1.0 + 1e-9:
            add(nm, 1.0)
            nm += step
        add(high + 1.0, 0.0)
        camera.set(
            sensor_type="mono",
            sensor_qe_mono=curve,
            spectral_range=(low - 1.0, high + 1.0),
            spectral_bands=4,
        )

    # --------------------------------------------------------- checks

    def check_fields_of_view(self, optics, rig, tolerance):
        """The f-theta lens sees the published fields of view: each
        field's half width on the detector over the focal length,
        through the authored k1, within `tolerance` degrees."""
        k1 = float(self.camera(f"{rig[0]}/Left").lens_fisheye[0])
        for i, (field, reach) in enumerate(
            zip(optics[5], cmv_reaches())
        ):
            degrees = 2.0 * math.degrees(
                f_theta_angle(reach / optics[1], k1)
            )
            print(
                f"{optics[0]} field {i}: {degrees:.1f} degrees for {field:.0f}"
            )
            assert abs(degrees - field) < tolerance, (
                "the lens should see the published fields of view"
            )

    def check_stereo(self, optics, rig, depth, radius):
        """A ball of `radius` on the rig's axis `depth` metres out sits
        further right in the left eye's image than the right's, by the
        baseline's disparity through the f-theta lens: 2 f theta (1 +
        k1 theta^2) pixels, theta = atan(baseline / 2 depth). The
        numbers stay within the 12-bit ADC's range, most exposed within
        it."""
        name, focal, _, _, baseline, _ = optics
        target = (
            rig_transform(rig) @ np.array([0.0, 0.0, -depth, 1.0])
        )[:3]
        ball = "/World/StereoTarget"
        self.scene.define(
            ball, ernie.Sphere, radius=radius, translate=tuple(target)
        )
        products = [f"/Render/{name}Left", f"/Render/{name}Right"]
        ids = self.render_vars(products, "PrimId")
        raws = self.render_vars(products, "RawDN")
        columns = []
        for product in products:
            image = ids[product].reshape(HEIGHT, WIDTH, -1)[..., 0]
            hits = [
                col + 0.5
                for row, col in zip(*np.nonzero(image))
                if str(self.renderer.prim_path(int(image[row, col])))
                == ball
            ]
            assert hits, "each eye should see the target"
            columns.append(float(np.mean(hits)))
            raw = raws[product].ravel()
            assert raw.max() <= 4095, "the numbers should be 12-bit"
            between = int(np.count_nonzero((raw > 0) & (raw < 4095)))
            assert between > raw.size // 2, (
                "the image should be exposed within the ADC's range"
            )
        self.scene.remove(ernie.Path(ball))
        k1 = f_theta_k1(optics)
        pixels = focal / (1e-3 * CMV_PIXEL_UM * CMV_WIDTH / WIDTH)
        theta = math.atan(0.5 * baseline / depth)
        expected = 2.0 * pixels * theta * (1.0 + k1 * theta * theta)
        disparity = columns[0] - columns[1]
        print(
            f"{name} disparity at {depth:.1f} m: {disparity:.2f} pixels "
            f"for {expected:.2f}"
        )
        assert abs(disparity - expected) < 1.0, (
            "the pair should see the baseline's disparity"
        )

    def check_mastcam_rig(self):
        """Mastcam-Z's eyes 24.4 cm apart, looking 2.3 degrees apart,
        each turned in."""
        poses = [
            np.asarray(
                self.camera(f"{MASTCAM_RIG[0]}/{eye}").get(
                    tokens.TRANSFORM
                )
            )
            for eye in EYES
        ]
        baseline = float(
            np.linalg.norm(poses[1][:3, 3] - poses[0][:3, 3])
        )
        looks = [-pose[:3, 2] for pose in poses]
        between = math.degrees(
            math.acos(min(1.0, float(np.dot(looks[0], looks[1]))))
        )
        print(
            f"Mastcam-Z: {baseline:.3f} m apart, {between:.2f} degrees "
            "between"
        )
        assert (
            abs(baseline - MASTCAM_BASELINE_M) < 1e-9
            and abs(between - 2.0 * TOE_IN_DEG) < 1e-6
            and looks[0][0] > 0.0
            and looks[1][0] < 0.0
        ), "Mastcam-Z's eyes should be toed in as published"

    def check_soil_is_red(self):
        """Mastcam-Z's colour image of the ground: redder than it is
        blue, as Mars's soil."""
        product = "/Render/MastcamZLeft"
        rgb = mean_linear_rgb(
            self.render_vars([product], "LdrColor")[product]
        )
        print(
            f"Mastcam-Z: soil {rgb[0]:.3f} {rgb[1]:.3f} {rgb[2]:.3f}"
        )
        assert rgb[0] > 1.5 * rgb[2], "the soil should look red"

    def check_near_infrared(self):
        """Through the right camera's R1 (800 +/- 9 nm) and R6 (1022 +/-
        19 nm) on a camera of their own at the right eye: both see the
        lit ground, R6 the dimmer, silicon nearly blind at 1 um despite
        its band twice as wide."""
        camera = f"{MASTCAM_RIG[0]}/Narrow"
        product = "/Render/MastcamZNarrow"
        self.scene.define(camera, ernie.Camera).set(
            tokens.TRANSFORM,
            self.camera(f"{MASTCAM_RIG[0]}/Right").get(
                tokens.TRANSFORM
            ),
        )
        self.define_product(
            product,
            camera,
            ["qe", "noise", "adc"],
            photographic=False,
        )
        means = []
        for number in (1, 6):
            self.use_mastcam_filter(camera, 1, number)
            self.camera(camera).set(
                sensor_noise=False, exposure_time=1e-3
            )
            electrons = self.render_vars([product], "Electrons")[
                product
            ]
            assert electrons.max() < KAI_FULL_WELL[1], (
                "the bands should stay below the full well"
            )
            means.append(float(np.mean(electrons, dtype=np.float64)))
        self.scene.remove(ernie.Path(product))
        self.scene.remove(ernie.Path(camera))
        print(
            f"Mastcam-Z R1 {means[0]:.0f}, R6 {means[1]:.0f} electrons"
        )
        assert 100.0 < means[0] and 0.0 < means[1] < 0.5 * means[0], (
            "R6 should see less than half of R1's light"
        )

    def check_every_filter_and_zoom(self):
        """Every filter of both wheels renders an image to show and, but
        for the solar filters, collects light; at the long end of the
        zoom the camera closes to f/9.5. Back to filter 0 at 26 mm."""
        for e, eye in enumerate(EYES):
            product = f"/Render/MastcamZ{eye}"
            for number in range(1, 8):
                self.set_mastcam_filter(e, number)
                self.camera(
                    self.mastcam_camera(e)
                ).sensor_noise = False
                ldr = self.render_vars([product], "LdrColor")[product]
                electrons = self.render_vars([product], "Electrons")[
                    product
                ]
                assert ldr.size > 0 and (
                    number == 7 or float(electrons.sum()) > 0.0
                ), "every filter should render"
            self.set_mastcam_filter(e, 0)
            self.camera(self.mastcam_camera(e)).sensor_noise = False
        self.focal_mm = LONGEST_FOCAL_MM
        self.use_mastcam_filter(self.mastcam_camera(0), 0, 0)
        left = self.camera(self.mastcam_camera(0))
        print(
            f"Mastcam-Z at {left.focal_length:.0f} mm: "
            f"f/{left.exposure_f_stop:.1f}"
        )
        assert (
            abs(left.exposure_f_stop - NARROWEST_F_NUMBER) < 1e-12
        ), "the zoom's long end should be f/9.5"
        self.focal_mm = SHORTEST_FOCAL_MM
        self.use_mastcam_filter(self.mastcam_camera(0), 0, 0)
        left.sensor_noise = False


if __name__ == "__main__":
    ernie.demo.main(Demo)
