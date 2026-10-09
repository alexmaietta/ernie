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
