# SPDX-License-Identifier: BSD-3-Clause
import math

import numpy as np
import pytest

import ernie
from ernie import tokens


def test_navcam_pair_is_authored_on_its_rig():
    scene = ernie.Scene()
    for product in ("/Render/Left", "/Render/Right"):
        scene.define(product, ernie.RenderProduct)
    ernie.author_mars2020_navcam_pair(scene, "/Rover/Navcam", "/Render/Left", "/Render/Right",
                                      pattern="bggr", jpeg_quality=70.0)
    left = np.array(scene.get("/Rover/Navcam/Left", tokens.TRANSFORM))
    right = np.array(scene.get("/Rover/Navcam/Right", tokens.TRANSFORM))
    # Maki et al. 2020, Table 2: a 42.4 cm stereo baseline.
    assert math.isclose(right[0, 3] - left[0, 3], 0.424)
    assert math.isclose(scene.get("/Rover/Navcam/Left", tokens.FOCAL_LENGTH), 19.1)
    with pytest.raises(ValueError):
        ernie.author_mars2020_navcam(scene, "/Rover/Navcam/Left", "/Render/Left", pattern="rgb")


def test_hazcam_pairs_front_and_rear():
    scene = ernie.Scene()
    for product in ("/Render/Left", "/Render/Right"):
        scene.define(product, ernie.RenderProduct)
    # Maki et al. 2020, Table 2: 24.8 cm (front) and 93.4 cm (rear) baselines, 14.0 mm lenses.
    for which, baseline in (("front", 0.248), ("rear", 0.934)):
        ernie.author_mars2020_hazcam_pair(scene, "/Rover/Hazcam", "/Render/Left", "/Render/Right",
                                          which=which)
        left = np.array(scene.get("/Rover/Hazcam/Left", tokens.TRANSFORM))
        right = np.array(scene.get("/Rover/Hazcam/Right", tokens.TRANSFORM))
        assert math.isclose(right[0, 3] - left[0, 3], baseline)
    assert math.isclose(scene.get("/Rover/Hazcam/Left", tokens.FOCAL_LENGTH), 14.0)
    with pytest.raises(ValueError):
        ernie.author_mars2020_hazcam_pair(scene, "/Rover/Hazcam", "/Render/Left", "/Render/Right",
                                          which="side")


def test_mastcamz_pair_reaches_the_near_infrared():
    scene = ernie.Scene()
    for product in ("/Render/Left", "/Render/Right"):
        scene.define(product, ernie.RenderProduct)
    ernie.author_mastcamz_pair(scene, "/Rover/MastcamZ", "/Render/Left", "/Render/Right",
                               left_filter=6, right_filter=6, focal_length=63.0)
    # Bell et al. 2021, Table 3: L6 is 442 +/- 12 nm, R6 1022 +/- 19 nm.
    left_range = scene.get("/Rover/MastcamZ/Left", tokens.SPECTRAL_RANGE)
    right_range = scene.get("/Rover/MastcamZ/Right", tokens.SPECTRAL_RANGE)
    assert math.isclose(left_range[0], 429.0) and math.isclose(left_range[1], 455.0)
    assert math.isclose(right_range[0], 1002.0) and math.isclose(right_range[1], 1042.0)
    with pytest.raises(ValueError):
        ernie.author_mastcamz(scene, "/Rover/MastcamZ/Left", "/Render/Left", filter=9)


def test_prim_paths_are_strs_paths_prims_or_typed_prims():
    scene = ernie.Scene()
    camera = scene.define("/World/Camera", ernie.Camera, focal_length=50.0)
    product = scene.define("/Render/Product", ernie.RenderProduct)
    ernie.author_lroc_nac(scene, camera, product)
    assert camera.sensor_adc_bits == 12
    ernie.author_lroc_wac(scene, camera.prim, ernie.Path("/Render/Product"), band=415)
    assert camera.sensor_adc_bits == 11
    ernie.author_singlet_chromatic(scene, camera)
    assert scene.get("/World/Camera", tokens.LENS_CHROMATIC_FOCUS) is not None
    rig = scene.define("/Rover/Navcam", ernie.Xform)
    left = scene.define("/Render/Left", ernie.RenderProduct)
    ernie.author_mars2020_navcam_pair(scene, rig, left, left.prim)
    assert ernie.Camera.at(rig, "Right").focal_length == 19.1
    with pytest.raises(TypeError, match="incompatible function arguments"):
        ernie.author_lroc_nac(scene, 3, product)
    with pytest.raises(ValueError, match="not absolute"):
        ernie.author_lroc_nac(scene, "World/Camera", product)


def test_singlet_chromatic_follows_its_glass():
    scene = ernie.Scene()
    scene.define("/Camera", ernie.Camera, focal_length=50.0)
    # SCHOTT N-BK7 by default: the C line focuses about f / V = 0.78 mm further than the F line.
    ernie.author_singlet_chromatic(scene, "/Camera", lateral=0.5)
    focus = dict(zip(*[iter(scene.get("/Camera", tokens.LENS_CHROMATIC_FOCUS))] * 2))
    scale = dict(zip(*[iter(scene.get("/Camera", tokens.LENS_CHROMATIC_SCALE))] * 2))
    assert 0.6 < focus[660.0] - focus[490.0] < 0.8
    assert scale[700.0] > 1.0 > scale[400.0]
    with pytest.raises(ValueError):
        ernie.author_singlet_chromatic(scene, "/Camera", nd=0.5)
