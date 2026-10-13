# SPDX-License-Identifier: BSD-3-Clause
import pytest

import ernie
from ernie import tokens


def test_render_types_derive_from_render_settings_base():
    base = ernie.find_prim_type("RenderSettingsBase")
    assert base.abstract and base.base.name == "Imageable"
    assert ernie.find_prim_type("RenderSettings").base is base
    assert ernie.find_prim_type("RenderProduct").base is base
    assert issubclass(ernie.RenderSettings, ernie.RenderSettingsBase)
    assert issubclass(ernie.RenderProduct, ernie.RenderSettingsBase)
    assert ernie.find_prim_type("RenderVar").base.name == "Imageable"
    assert (
        ernie.find_prim_type("SensorFilter").find("filter_type").name
        == tokens.FILTER_TYPE
    )
    with pytest.raises(TypeError, match="abstract"):
        ernie.Scene().define("/Base", ernie.RenderSettingsBase)


def test_render_settings_read_fallbacks_and_define_with_keywords():
    scene = ernie.Scene()
    scene.define("/Cam", tokens.CAMERA)
    settings = scene.define(
        "/Render/Settings",
        ernie.RenderSettings,
        camera="/Cam",
        resolution=(640, 480),
        shadow_cascades=2,
        culling="cpu",
    )
    assert isinstance(settings, ernie.RenderSettings)
    assert settings.camera == ernie.Path(
        "/Cam"
    ) and settings.cameras == [ernie.Path("/Cam")]
    assert scene.get(
        "/Render/Settings", tokens.CAMERA_REL
    ) == ernie.Path("/Cam")
    assert settings.resolution == (640.0, 480.0)
    assert (
        settings.shadow_cascades == 2
        and scene.get("/Render/Settings", tokens.SHADOW_CASCADES) == 2
    )
    assert settings.culling == "cpu"
    # Unset, the renderer's defaults.
    assert settings.ambient == 0.0
    assert settings.occlusion == "off" and settings.wireframe == "off"
    assert (
        settings.dome_light_mode == "approximate"
        and settings.dome_directions == 16
    )
    assert settings.meter is False
    assert (
        settings.lod_mesh_error == 1.0 and settings.lod_enable is True
    )
    assert (
        settings.minimal_mode == "off"
        and settings.minimal_constant_color == (1.0, 1.0, 1.0)
    )
    assert (
        ernie.RenderSettings.shadow_resolution.doc
        == "Texels along each cascade's side."
    )


def test_products_vars_and_filters():
    scene = ernie.Scene()
    product = scene.define(
        "/Render/Product",
        ernie.RenderProduct,
        overlays="*",
        minimal_mode="constantDiffuse",
    )
    assert product.resolution == (2048.0, 1080.0)
    assert scene.get("/Render/Product", tokens.OVERLAYS) == "*"
    assert product.minimal_mode == "constantDiffuse"
    var = scene.define(
        "/Render/Vars/Depth",
        ernie.RenderVar,
        source_name="DistanceToCamera",
    )
    assert var.source_name == "DistanceToCamera"
    assert (
        ernie.RenderVar.define(
            scene, "/Render/Vars/Color"
        ).source_name
        == "LdrColor"
    )
    assert "MotionVector" in ernie.RenderVar.source_name.options
    qe = scene.define(
        "/Render/Qe", ernie.SensorFilter, filter_type="qe"
    )
    assert (
        qe.filter_type == "qe"
        and scene.get("/Render/Qe", tokens.FILTER_TYPE) == "qe"
    )
    with pytest.raises(ValueError):
        qe.filter_type = 3


def test_relationship_lists_take_one_target_or_several():
    scene = ernie.Scene()
    cams = [
        scene.define(f"/Cams/{name}", ernie.Camera)
        for name in ("A", "B")
    ]
    var = scene.define("/Render/Product/Color", ernie.RenderVar)
    assert (
        ernie.RenderProduct.camera.kind
        == ernie.AttributeKind.RELATIONSHIPS
    )
    assert ernie.RenderProduct.ordered_vars.default == []
    # One target, as a str, Path, Prim or typed prim, is stored as a
    # Path: `camera` reads it, and `cameras` the list of one.
    product = scene.define(
        "/Render/Product", ernie.RenderProduct, camera=cams[0]
    )
    assert scene.get(
        "/Render/Product", tokens.CAMERA_REL
    ) == ernie.Path("/Cams/A")
    assert product.camera == ernie.Path("/Cams/A")
    assert product.cameras == [ernie.Path("/Cams/A")]
    # A batch of cameras: a list of any of them, mixed; `camera` reads
    # the first.
    product.cameras = [cams[0], "/Cams/B"]
    assert product.cameras == [
        ernie.Path("/Cams/A"),
        ernie.Path("/Cams/B"),
    ]
    assert product.camera == ernie.Path("/Cams/A")
    assert scene.define(
        "/Render/Batch", ernie.RenderProduct, cameras=cams
    ).cameras == [cam.path for cam in cams]
    assert ernie.RenderProduct.cameras is ernie.RenderProduct.camera
    unset = scene.define("/Render/Unset", ernie.RenderProduct)
    assert unset.camera is None and unset.cameras == []
    assert scene.get("/Render/Product", tokens.CAMERA_REL) == [
        ernie.Path("/Cams/A"),
        ernie.Path("/Cams/B"),
    ]
    product.set(ordered_vars=[var], filters=["/Render/Qe"])
    assert product.ordered_vars == [var.path] and product.filters == [
        ernie.Path("/Render/Qe")
    ]
    # Empty, set apart from unset.
    settings = scene.define(
        "/Render/Settings",
        ernie.RenderSettings,
        products=[product.prim],
    )
    assert settings.products == [ernie.Path("/Render/Product")]
    lights = ernie.RenderSettings.lights.name
    assert lights == "ernie:lights"
    assert (
        settings.lights == []
        and scene.get("/Render/Settings", lights) is None
    )
    settings.lights = []
    assert (
        settings.lights == []
        and scene.get("/Render/Settings", lights) == []
    )
    with pytest.raises(
        ValueError, match="a prim path or a list of them"
    ):
        settings.lights = 3.0
