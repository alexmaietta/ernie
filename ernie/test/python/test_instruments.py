import math

import numpy as np
import pytest

import ernie
from ernie import tokens


def test_navcam_pair_is_authored_on_its_rig():
    scene = ernie.Scene()
    for product in ("/Render/Left", "/Render/Right"):
        scene.define(product, tokens.RENDER_PRODUCT)
    ernie.author_mars2020_navcam_pair(scene, "/Rover/Navcam", "/Render/Left", "/Render/Right",
                                      pattern="bggr", jpeg_quality=70.0)
    left = np.array(scene.get("/Rover/Navcam/Left", tokens.TRANSFORM))
    right = np.array(scene.get("/Rover/Navcam/Right", tokens.TRANSFORM))
    # Maki et al. 2020, Table 2: a 42.4 cm stereo baseline.
    assert math.isclose(right[0, 3] - left[0, 3], 0.424)
    assert math.isclose(scene.get("/Rover/Navcam/Left", tokens.FOCAL_LENGTH), 19.1)
    with pytest.raises(ValueError):
        ernie.author_mars2020_navcam(scene, "/Rover/Navcam/Left", "/Render/Left", pattern="rgb")


def test_mastcamz_pair_reaches_the_near_infrared():
    scene = ernie.Scene()
    for product in ("/Render/Left", "/Render/Right"):
        scene.define(product, tokens.RENDER_PRODUCT)
    ernie.author_mastcamz_pair(scene, "/Rover/MastcamZ", "/Render/Left", "/Render/Right",
                               left_filter=6, right_filter=6, focal_length=63.0)
    # Bell et al. 2021, Table 3: L6 is 442 +/- 12 nm, R6 1022 +/- 19 nm.
    left_range = scene.get("/Rover/MastcamZ/Left", tokens.SPECTRAL_RANGE)
    right_range = scene.get("/Rover/MastcamZ/Right", tokens.SPECTRAL_RANGE)
    assert math.isclose(left_range[0], 429.0) and math.isclose(left_range[1], 455.0)
    assert math.isclose(right_range[0], 1002.0) and math.isclose(right_range[1], 1042.0)
    with pytest.raises(ValueError):
        ernie.author_mastcamz(scene, "/Rover/MastcamZ/Left", "/Render/Left", filter=9)
