# SPDX-License-Identifier: BSD-3-Clause
"""Perseverance's Navcam, Hazcam and Mastcam-Z stereo pairs on a patch of
Mars, their detectors as their datasheets give them.

References:
- J. N. Maki et al., "The Mars 2020 Engineering Cameras and Microphone
  on the Perseverance Rover", Space Science Reviews 216:137, 2020:
  Tables 2 and 3 and the detector section.
- J. F. Bell III et al., "The Mars 2020 Perseverance Rover Mast Camera
  Zoom (Mastcam-Z) Multispectral, Stereoscopic Imaging Investigation",
  Space Science Reviews 217:24, 2021: Tables 3 and 5.
- ams OSRAM, "CMV20000 20MP CMOS Machine Vision Image Sensor", datasheet
  DS000440 v3-00, 2026: Table 5 and Figure 4.
- onsemi, "KAI-2020 Image Sensor", datasheet KAI-2020/D: Table 7 and
  Figure 12 (Gen1 colour).
- J. F. Bell III et al., "Mineralogic and compositional properties of
  Martian soil and dust: Results from Mars Pathfinder", JGR
  105(E1):1721-1755, 2000 (soil reflectance, roughly).
- J. Kannala and S. S. Brandt, "A Generic Camera Model and Calibration
  Method for Conventional, Wide-Angle, and Fish-Eye Lenses", IEEE TPAMI
  28(8):1335-1340, 2006.
"""

import math

import numpy as np

import ernie
import ernie.demo
from ernie import tokens
from ernie.demo import mean_linear_rgb, side_by_side, write_png

# The Sun's irradiance and angular diameter at Mars's mean distance.
MARS_IRRADIANCE = 1361.0 / 1.5237**2
SUN_ANGLE = 0.35

EYES = ("Left", "Right")


def curve(values, first, step, nm):
    """`values` at `nm`, linear between wavelengths `first` + k `step`,
    zero outside them."""
    x = (nm - first) / step
    if x < 0.0 or x > len(values) - 1:
        return 0.0
    i = min(int(x), len(values) - 2)
    return values[i] + (x - i) * (values[i + 1] - values[i])


def pass_band(nm, low, high):
    """A trapezoid passband over [low, high], ramped over 20 nm."""
    return min(max(min(nm - low, high - nm) / 20.0 + 0.5, 0.0), 1.0)


def sampled(first, last, f):
    """(nm, value) pairs of `f` every 5 nm over [first, last]."""
    pairs = []
    nm = first
    while nm <= last + 1e-9:
        pairs += [nm, f(nm)]
        nm += 5.0
    return pairs


# -------------------------------------------------- engineering cameras

# The CMV20000 (DS000440, Table 5): 6.4 um pixels, 0.25 DN/e-, 8 e-
# noise, 15000 e- full well, 125 e-/s dark current, 10 e-/s DSNU, 1%
# PRNU.
CMV_PIXEL_UM = 6.4
CMV_WIDTH = 5120
CMV_HEIGHT = 3840

# Figure 4's Bayer QE, read every 25 nm from 350 nm.
CMV_BLUE = (
    .05, .06, .20, .27, .32, .35, .27, .10, .05, .04, .03, .03, .03, .04,
    .05, .05, .05, .05, .19, .21, .21, .16, .13, .10, .06, .04, .03, .02,
    .015, .01, .005,
)  # fmt: skip
CMV_GREEN = (
    .04, .05, .05, .06, .07, .10, .25, .42, .44, .30, .12, .06, .05, .08,
    .13, .13, .17, .15, .20, .21, .21, .16, .13, .10, .06, .04, .03, .02,
    .015, .01, .005,
)  # fmt: skip
CMV_RED = (
    .05, .08, .07, .03, .02, .02, .02, .02, .03, .03, .44, .40, .38, .37,
    .34, .32, .32, .25, .27, .22, .20, .16, .13, .10, .06, .04, .03, .02,
    .015, .01, .005,
)  # fmt: skip

# Maki's 8 x 8 panchromatic mode (640 x 480): each pixel the mean of 64
# behind a Bayer cell's mean response and the UV and IR blocking filter
# (its edges unpublished; 400 and 700 nm assumed).
ECAM_SUM = 8
ECAM_WIDTH = CMV_WIDTH // ECAM_SUM
ECAM_HEIGHT = CMV_HEIGHT // ECAM_SUM
ECAM_EXPOSURE = 1e-3


def ecam_qe(nm):
    cell = 0.25 * (
        curve(CMV_RED, 350.0, 25.0, nm)
        + 2.0 * curve(CMV_GREEN, 350.0, 25.0, nm)
        + curve(CMV_BLUE, 350.0, 25.0, nm)
    )
    return cell * pass_band(nm, 400.0, 700.0)


# An engineering camera's optics (Maki Table 2): name, focal length
# (mm), f-number, best focus (m), baseline (m) and fields of view
# (degrees).
NAVCAM = ("Navcam", 19.1, 12.0, 3.5, 0.424, (96.0, 73.0, 120.0))
HAZCAM = ("Hazcam", 14.0, 12.0, 0.9, 0.248, (136.0, 102.0, 172.0))
REAR_HAZCAM = ("RearHazcam", 14.0, 12.0, 0.9, 0.934, HAZCAM[5])


def cmv_reaches():
    """The detector's half width, half height and half diagonal, mm."""
    half_width = 0.5e-3 * CMV_PIXEL_UM * CMV_WIDTH
    half_height = 0.5e-3 * CMV_PIXEL_UM * CMV_HEIGHT
    return (
        half_width,
        half_height,
        math.hypot(half_width, half_height),
    )


def f_theta_k1(optics):
    """The f-theta lens's k1 fitted to the published fields."""
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
    """The angle off axis landing `r` focal lengths out, by Newton."""
    theta = r
    for _ in range(20):
        f = theta * (1.0 + k1 * theta * theta) - r
        theta -= f / (1.0 + 3.0 * k1 * theta * theta)
    return theta


# ------------------------------------------------------------ Mastcam-Z

# The KAI-2020CM (Bell Table 5; datasheet Table 7): 1600 x 1200 pixels
# of 7.4 um, 15.6 e-/DN, 11 bits, 40 e-/s dark current and 2.5% rms
# PRNU.
KAI_PIXEL_UM = 7.4
KAI_WIDTH = 1600
KAI_HEIGHT = 1200
KAI_ELECTRONS_PER_DN = 15.6
KAI_FULL_WELL = (21826.0, 21818.0)
KAI_READ_NOISE = (22.0, 21.0)

# Exposures through the colour filter and the narrow bands.
MASTCAM_EXPOSURE = 2e-3
NARROW_EXPOSURE = 2e-2

# Figure 12's Gen1 colour QE, read every 20 nm from 380 nm.
KAI_BLUE = (
    .02, .20, .33, .40, .46, .40, .25, .08, .02, .01, .005, .005, .005,
    .005, .005, .01, .01, .01, .01, .01, .03, .06, .08, .07, .06, .045,
    .035, .03, .025, .02, .015, .012, .01, .008, .006, .004, .002,
)  # fmt: skip
KAI_GREEN = (
    .01, .01, .015, .02, .03, .05, .12, .25, .41, .33, .12, .02, .01,
    .01, .01, .02, .03, .04, .04, .05, .06, .07, .08, .07, .06, .045,
    .035, .03, .025, .02, .015, .012, .01, .008, .006, .004, .002,
)  # fmt: skip
KAI_RED = (
    .01, .01, .01, .01, .005, .005, .005, .005, .005, .01, .05, .25, .33,
    .30, .27, .25, .22, .19, .16, .14, .12, .10, .085, .07, .06, .045,
    .035, .03, .025, .02, .015, .012, .01, .008, .006, .004, .002,
)  # fmt: skip

# Zoom, baseline and toe-in (Bell Table 5).
SHORTEST_FOCAL_MM = 26.0
LONGEST_FOCAL_MM = 110.0
WIDEST_F_NUMBER = 6.7
NARROWEST_F_NUMBER = 9.5
MASTCAM_BASELINE_M = 0.244
TOE_IN_DEG = 1.15


def f_number_at(focal_mm):
    t = (focal_mm - SHORTEST_FOCAL_MM) / (
        LONGEST_FOCAL_MM - SHORTEST_FOCAL_MM
    )
    return WIDEST_F_NUMBER + t * (
        NARROWEST_F_NUMBER - WIDEST_F_NUMBER
    )


# Each filter's effective wavelength, half width at half maximum (nm)
# and neutral density (Bell Table 3).
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

# Filter 0's infrared cut-off: the red channel's upper half maximum in
# Table 3, 631 + 43 nm.
IR_CUT_NM = 674.0

# ------------------------------------------------------------ the scene

# Mars soil and darker rock's reflectance, roughly (Bell 2000).
SOIL = [350.0, 0.05, 450.0, 0.08, 550.0, 0.14, 650.0, 0.25]
SOIL += [750.0, 0.30, 1100.0, 0.30]
ROCK = [350.0, 0.06, 550.0, 0.09, 750.0, 0.14, 1100.0, 0.15]

# Rocks ahead of the rover: x, y and radius, m.
ROCKS = (
    (-1.2, 3.0, 0.25),
    (0.9, 4.5, 0.4),
    (-2.5, 7.0, 0.6),
    (2.0, 9.0, 0.5),
    (0.3, 1.6, 0.12),
    (-0.6, 2.2, 0.18),
)

# Stereo rigs: path, eye, tilt down and heading (degrees from north).
NAVCAM_RIG = ("/World/Rover/Navcam", (0.0, 0.0, 2.0), 15.0, 0.0)
HAZCAM_RIG = ("/World/Rover/Hazcam", (0.0, 0.6, 0.7), 30.0, 0.0)
REAR_HAZCAM_RIG = (
    "/World/Rover/RearHazcam",
    (0.0, -0.9, 0.7),
    30.0,
    180.0,
)
MASTCAM_RIG = ("/World/Rover/MastcamZ", (0.3, 0.0, 2.0), 15.0, 0.0)

# The cameras the viewer can show.
VIEWS = (
    ("Overview", "/World/View"),
    ("Navcam", "/World/Rover/Navcam/Left"),
    ("Front Hazcam", "/World/Rover/Hazcam/Left"),
    ("Rear Hazcam", "/World/Rover/RearHazcam/Left"),
    ("Mastcam-Z", "/World/Rover/MastcamZ/Left"),
)


def filter_name(e, number):
    """Filter `number` of eye `e`'s wheel as Table 3 names it."""
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
    """An eye `x` along its rig, turned `yaw` radians about the rig's
    +y."""
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            [c, 0.0, s, x],
            [0.0, 1.0, 0.0, 0.0],
            [-s, 0.0, c, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def centre(image, width, height):
    """The central `width` x `height` of an image."""
    top = (image.shape[0] - height) // 2
    left = (image.shape[1] - width) // 2
    return image[top : top + height, left : left + width]


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
        # The Sun, 40 degrees up in the south-east.
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

        # The viewer's own view of the rover's patch.
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
        """The left Navcam, the right Navcam, the left Hazcam and the
        middle of the left Mastcam-Z, as the rover sends them."""
        products = [
            "/Render/NavcamLeft",
            "/Render/NavcamRight",
            "/Render/HazcamLeft",
            "/Render/MastcamZLeft",
        ]
        outputs = self.render_vars(products, "LdrColor")
        images = [outputs[p] for p in products]
        images[-1] = centre(images[-1], ECAM_WIDTH, ECAM_HEIGHT)
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
        """Var `var` of each of `products`, rendered now."""
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
        self, product, camera, width, height, types, photographic=True
    ):
        """Product `product` of `camera`, with a readout chain of
        `types`."""
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
            resolution=(width, height),
        ).set(tokens.ORDERED_VARS, paths).set(
            tokens.FILTERS, [ernie.Path(f.path) for f in filters]
        )
        return filters

    def define_engineering_pair(self, rig, optics):
        """A Navcam or Hazcam pair in the 8 x 8 mode: the f-theta lens on
        the CMV20000, its sums of 64 pixels scaled as means, sent as
        JPEG."""
        name, focal, f_number, focus, baseline, _ = optics
        self.scene.define(rig[0], ernie.Xform).set(
            tokens.TRANSFORM, rig_transform(rig)
        )
        k1 = f_theta_k1(optics)
        n = ECAM_SUM * ECAM_SUM
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
                exposure_time=ECAM_EXPOSURE,
                exposure_mode="auto",  # the 8-bit images
                exposure_iso=100.0,  # the gain is the ADC's
                sensor_type="mono",
                sensor_qe_mono=sampled(350.0, 1100.0, ecam_qe),
                spectral_range=(380.0, 720.0),
                spectral_bands=16,
                sensor_electrons_per_dn=4.0 * n,
                sensor_read_noise=8.0 * ECAM_SUM,
                sensor_full_well=15000.0 * n,
                sensor_dark_current=125.0 * n,
                sensor_dsnu=10.0 * ECAM_EXPOSURE * ECAM_SUM,
                sensor_prnu=0.01 / ECAM_SUM,
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
                ECAM_WIDTH,
                ECAM_HEIGHT,
                ["qe", "noise", "adc", "cie", "exposure", "jpeg"],
            )
            filters[5].set("inputs:quality", 85.0)

    def define_mastcam_pair(self):
        """Mastcam-Z's pair at the widest zoom, toed in, through filter
        0."""
        self.scene.define(MASTCAM_RIG[0], ernie.Xform).set(
            tokens.TRANSFORM, rig_transform(MASTCAM_RIG)
        )
        for e in range(2):
            camera = self.scene.define(
                self.mastcam_camera(e), ernie.Camera
            )
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
        """Mastcam-Z eye `e` through filter `number`, with its
        readout."""
        self.filters[e] = number
        self.use_mastcam_filter(self.mastcam_camera(e), e, number)
        self.define_product(
            f"/Render/MastcamZ{EYES[e]}",
            self.mastcam_camera(e),
            KAI_WIDTH,
            KAI_HEIGHT,
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
        """Mastcam-Z camera `path` through filter `number`: Gen1 colour
        under the IR cut-off for filter 0, else the passband on the
        Bayer cell's mean response."""
        camera = self.camera(path)
        camera.set(
            focal_length=self.focal_mm,
            horizontal_aperture=1e-3 * KAI_PIXEL_UM * KAI_WIDTH,
            vertical_aperture=1e-3 * KAI_PIXEL_UM * KAI_HEIGHT,
            clipping_range=(0.05, 1e4),
            exposure_f_stop=f_number_at(self.focal_mm),
            exposure_time=MASTCAM_EXPOSURE
            if number == 0
            else NARROW_EXPOSURE,
            exposure_mode="auto",  # the 8-bit images
            exposure_iso=100.0,  # the gain is the ADC's
            sensor_electrons_per_dn=KAI_ELECTRONS_PER_DN,
            sensor_read_noise=KAI_READ_NOISE[e],
            sensor_full_well=KAI_FULL_WELL[e],
            sensor_dark_current=40.0,
            sensor_dsnu=0.0,
            sensor_prnu=0.025,
            sensor_adc_bits=11,
            sensor_black_level=0.0,
            sensor_noise=True,
        )

        def kai(values):
            def qe(nm):
                return curve(values, 380.0, 20.0, nm) * pass_band(
                    nm, 330.0, IR_CUT_NM
                )

            return qe

        if number == 0:
            camera.set(
                sensor_type="rgb",
                sensor_qe_r=sampled(380.0, 780.0, kai(KAI_RED)),
                sensor_qe_g=sampled(380.0, 780.0, kai(KAI_GREEN)),
                sensor_qe_b=sampled(380.0, 780.0, kai(KAI_BLUE)),
                spectral_range=(380.0, 780.0),
                spectral_bands=16,
            )
            return
        effective, hwhm, density = FILTERS[e][number]
        transmission = 10.0**-density
        low, high = effective - hwhm, effective + hwhm

        def qe(nm):
            # The passband's edges ramped over 2 nm about its half
            # maxima.
            in_band = min(
                max(min(nm - low, high - nm) / 2.0 + 0.5, 0.0), 1.0
            )
            cell = 0.25 * (
                curve(KAI_RED, 380.0, 20.0, nm)
                + 2.0 * curve(KAI_GREEN, 380.0, 20.0, nm)
                + curve(KAI_BLUE, 380.0, 20.0, nm)
            )
            return in_band * transmission * cell

        pairs = []
        nm = low - 1.0
        while nm <= high + 1.0 + 1e-9:
            pairs += [nm, qe(nm)]
            nm += 0.5
        camera.set(
            sensor_type="mono",
            sensor_qe_mono=pairs,
            spectral_range=(low - 1.0, high + 1.0),
            spectral_bands=4,
        )

    # --------------------------------------------------------- checks

    def check_fields_of_view(self, optics, rig, tolerance):
        """The f-theta lens sees the published fields of view."""
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
        """A ball `depth` m out on the rig's axis shows the baseline's
        disparity through the f-theta lens, the numbers 12-bit."""
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
            image = ids[product].reshape(ECAM_HEIGHT, ECAM_WIDTH, -1)[
                ..., 0
            ]
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
            assert between > raw.size // 8, (
                "the image should be exposed within the ADC's range"
            )
        self.scene.remove(ernie.Path(ball))
        k1 = f_theta_k1(optics)
        pixels = focal / (
            1e-3 * CMV_PIXEL_UM * CMV_WIDTH / ECAM_WIDTH
        )
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
        """Mastcam-Z's eyes are 24.4 cm apart and toed in 2.3 degrees."""
        poses = [
            np.asarray(
                self.camera(self.mastcam_camera(e)).get(
                    tokens.TRANSFORM
                )
            )
            for e in range(2)
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
        """Mastcam-Z sees Mars's soil redder than it is blue."""
        product = "/Render/MastcamZLeft"
        rgb = mean_linear_rgb(
            self.render_vars([product], "LdrColor")[product]
        )
        print(
            f"Mastcam-Z: soil {rgb[0]:.3f} {rgb[1]:.3f} {rgb[2]:.3f}"
        )
        assert rgb[0] > 1.5 * rgb[2], "the soil should look red"

    def check_near_infrared(self):
        """R6 (1022 nm) sees under half of R1's (800 nm) light: silicon
        is nearly blind at 1 um."""
        camera = f"{MASTCAM_RIG[0]}/Narrow"
        product = "/Render/MastcamZNarrow"
        self.scene.define(camera, ernie.Camera).set(
            tokens.TRANSFORM,
            self.camera(self.mastcam_camera(1)).get(tokens.TRANSFORM),
        )
        self.define_product(
            product,
            camera,
            KAI_WIDTH,
            KAI_HEIGHT,
            ["qe", "noise", "adc"],
            photographic=False,
        )
        means = []
        for number in (1, 6):
            self.use_mastcam_filter(camera, 1, number)
            self.camera(camera).sensor_noise = False
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
        """Every filter renders, and the zoom's long end is f/9.5."""
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
