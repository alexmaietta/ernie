import math

import numpy as np

import ernie
from ernie import tokens


def test_astronomy_helpers():
    assert ernie.julian_date(2000, 1, 1, 12.0) == 2451545.0
    assert 0.0 <= ernie.greenwich_sidereal_time(2460000.5) < 360.0
    assert ernie.star_illuminance(0.0) == 2.54e-6
    assert abs(ernie.color_index_to_temperature(0.65) - 5778.0) < 5.0
    # The celestial pole stands at the site's latitude, due north.
    sky = np.asarray(ernie.celestial_to_local(52.0, 0.0, 2460000.5))
    pole = sky[:3, :3] @ np.array([0.0, 0.0, 1.0])
    assert abs(math.degrees(math.asin(pole[2])) - 52.0) < 1e-9
    assert abs(pole[0]) < 1e-12
    stars = ernie.synthetic_stars(100, seed=3)
    assert stars == ernie.synthetic_stars(100, seed=3), "the same for the same seed"
    assert len(stars["magnitude"]) == 100 and max(stars["magnitude"]) <= 6.5


def test_stars_and_the_dome_show_behind_the_scene(renderer):
    scene = ernie.Scene()
    scene.define("/World/Camera", tokens.CAMERA).set(
        tokens.TRANSFORM, ernie.look_at_transform((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    ).set(tokens.SENSOR_NOISE, False).set(tokens.EXPOSURE_TIME, 1e6)
    scene.define("/Render/Settings", tokens.RENDER_SETTINGS).set(
        tokens.CAMERA_REL, ernie.Path("/World/Camera")
    ).set(tokens.RESOLUTION, (32, 32))
    # One bright star straight ahead: right ascension 0 on the equator is +x.
    scene.define("/World/Stars", tokens.STAR_FIELD).write(
        tokens.STARS_RIGHT_ASCENSION, np.array([0.0], np.float32)
    ).write(tokens.STARS_DECLINATION, np.array([0.0], np.float32)).write(
        tokens.STARS_MAGNITUDE, np.array([0.0], np.float32)
    )
    image = np.asarray(renderer.render(scene))
    assert image[15:17, 15:17, :3].max() > 200, "the star at the centre"
    assert image[2, 2, :3].max() < 60, "dark sky elsewhere"

    scene.define("/World/Sky", tokens.DOME_LIGHT).set(tokens.INTENSITY, 1e-6)
    lit = np.asarray(renderer.render(scene))
    assert lit[2, 2, :3].min() > image[2, 2, :3].max(), "the dome fills the background"
    scene.set("/World/Sky", tokens.VISIBLE_IN_PRIMARY_RAY, False)
    assert np.asarray(renderer.render(scene))[2, 2, :3].max() == image[2, 2, :3].max()
    assert renderer.take_validation_messages() == []
