# SPDX-License-Identifier: BSD-3-Clause
import numpy as np
import pytest

import ernie
from ernie import tokens


def test_camera_class_describes_the_camera():
    camera = ernie.find_prim_type("Camera")
    assert ernie.Camera.prim_type.name == "Camera" and not camera.abstract
    assert camera.base.name == "Imageable"
    focal = camera.find("focal_length")
    assert focal is camera.find(tokens.FOCAL_LENGTH)
    assert (focal.name, focal.default, focal.group) == ("focalLength", 50.0, "Lens")
    assert camera.find("projection").options == ["perspective", "orthographic"]
    assert camera.find("exposure_f_stop").name == tokens.EXPOSURE_FSTOP
    assert camera.find("sensor_electrons_per_dn").name == tokens.SENSOR_GAIN
    assert camera.find("shutter_samples").kind == ernie.AttributeKind.INT
    assert camera.find("poses").kind == ernie.AttributeKind.TENSOR
    names = [a.py_name for a in ernie.Camera.attributes()]
    assert names[:2] == ["visibility", "world"] and len(names) == len(set(names))


def test_camera_defines_with_keywords_and_resolves():
    scene = ernie.Scene()
    camera = scene.define(
        "/World/Camera",
        ernie.Camera,
        focal_length=35,
        clipping_range=(0.1, 100.0),
        sensor_type="mono",
        sensor_qe_mono=[400.0, 0.5, 700.0, 0.5],
        spectral_bands=8,
        translate=(0, 0, 5),
    )
    assert isinstance(camera, ernie.Camera) and isinstance(camera, ernie.Imageable)
    assert camera.focal_length == 35.0 and isinstance(camera.focal_length, float)
    assert scene.get("/World/Camera", tokens.FOCAL_LENGTH) == 35.0
    assert camera.clipping_range == (0.1, 100.0)
    assert camera.sensor_qe_mono == [400.0, 0.5, 700.0, 0.5]
    assert camera.spectral_bands == 8 and isinstance(camera.spectral_bands, int)
    # Unset attributes read as the camera's defaults.
    assert camera.vertical_aperture == pytest.approx(15.2908)
    assert camera.exposure_iso == 100.0 and camera.shutter_samples == 8
    assert camera.scan == "frame" and camera.poses_device == "gpu"
    assert camera.lens_distortion == [] and camera.poses is None
    camera.poses = np.tile(np.eye(4, dtype=np.float32), (2, 1, 1))
    assert camera.poses.shape == (2, 4, 4)
    with pytest.raises(TypeError, match="did you mean 'focal_length'"):
        scene.define("/Other", ernie.Camera, focal_lenght=20.0)
