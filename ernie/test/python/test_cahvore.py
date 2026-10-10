import math
from pathlib import Path

import numpy as np
import pytest

import ernie
from ernie import tokens

MODELS = Path(__file__).resolve().parents[1] / "fixtures" / "cahvore" / "m2020_camera_models.json"


def test_mars2020_models_round_trip_and_author_cameras():
    models = {name: (frame, model) for name, frame, model in ernie.read_cahvore_models(str(MODELS))}
    frame, navl = models["NAVL"]
    assert frame == "Body_RSM_EL" and navl.kind == "cahvore"
    origin, direction = navl.image_to_ray(100.0, 3000.0)
    point = np.array(origin) + 4.0 * np.array(direction)
    x, y = navl.point_to_image(point)
    assert math.isclose(x, 100.0, abs_tol=1e-6) and math.isclose(y, 3000.0, abs_tol=1e-6)

    scene = ernie.Scene()
    width, height, rows_reversed, residual = navl.author_camera(scene, "/Rover/RSM/NAVL", 6.4)
    assert (width, height, rows_reversed) == (5120, 3840, False)
    assert residual < 5.0
    assert scene.get("/Rover/RSM/NAVL", tokens.LENS_PROJECTION) == tokens.FISHEYE
    assert math.isclose(scene.get("/Rover/RSM/NAVL", tokens.FOCAL_LENGTH), 19.1, abs_tol=0.5)
    # The model's own rays every 8 pixels, corners included.
    assert scene.get("/Rover/RSM/NAVL", tokens.LENS_RAYS_STEP) == 8
    assert np.asarray(scene.read("/Rover/RSM/NAVL", tokens.LENS_RAYS)).shape == (481, 641, 3)
    with pytest.raises(ValueError):
        models["SuperCam"][1].fit(6.4)
