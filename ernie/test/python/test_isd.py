import math
from pathlib import Path

import ernie
from ernie import tokens

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "isd"


def test_line_scanner_round_trips_and_authors_strips():
    nac = ernie.IsdLineScanner.from_file(str(FIXTURES / "toughLroNacLineScan.json"))
    assert (nac.lines, nac.samples) == (8192, 2532)
    ground = nac.image_to_ground(4096.0, 1266.0, 100.0)
    assert math.isclose(math.dist(ground, (0, 0, 0)), 1737.4e3 + 100.0, abs_tol=1e-3)
    line, sample = nac.ground_to_image(ground)
    assert math.isclose(line, 4096.0, abs_tol=1e-4)
    assert math.isclose(sample, 1266.0, abs_tol=1e-4)

    first, last, start, end = nac.strip(64, 64)
    # Ephemeris seconds near 3e8 resolve to about 6e-8.
    assert math.isclose(end - start, 64 * 0.0118595288, abs_tol=1e-6)
    scene = ernie.Scene()
    assert nac.author_strip(scene, "/World/Camera", 64, 64, at_end=False) == (2532, 64)
    path = "/World/Camera"
    assert scene.get(path, tokens.CAMERA_SCAN) == tokens.PUSHBROOM
    assert (scene.get(path, tokens.TRANSFORM) == first).all()
    assert math.isclose(scene.get(path, tokens.EXPOSURE_TIME), 0.0118595288, rel_tol=1e-6), (
        "each line exposed for its line time"
    )
    scene.set(path, tokens.EXPOSURE_ISO, 400.0)
    nac.author_strip(scene, path, 64, 64)
    assert (scene.get(path, tokens.TRANSFORM) == last).all()
    assert scene.get(path, tokens.EXPOSURE_ISO) == 400.0, "the ISO stays as authored"


def test_push_frame_round_trips_and_authors_framelets():
    wac = ernie.IsdPushFrame.from_file(str(FIXTURES / "lroWacPushFrame.json"))
    assert (wac.lines, wac.samples, wac.framelet_lines, wac.framelets) == (5096, 1024, 14, 364)
    ground = wac.image_to_ground(2555.0, 512.0)
    line, sample = wac.ground_to_image(ground)
    assert math.isclose(line, 2555.0, abs_tol=1e-4)
    assert math.isclose(sample, 512.0, abs_tol=1e-4)

    scene = ernie.Scene()
    path = "/World/Camera"
    samples, lines, first, _, time = wac.author_framelet(scene, path, 182)
    assert (samples, lines, first) == (1024, 14, 182 * 14)
    assert time == wac.image_time(first + 0.5)
    assert scene.get(path, tokens.CAMERA_SCAN) == tokens.FRAME_SCAN
    assert math.isclose(scene.get(path, tokens.EXPOSURE_TIME), 0.039)
