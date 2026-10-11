# SPDX-License-Identifier: BSD-3-Clause
import numpy as np
import pytest

import ernie
from ernie import tokens


def test_material_and_shader_are_registered():
    material = ernie.find_prim_type("Material")
    shader = ernie.find_prim_type("Shader")
    assert material.base.name == "Imageable" and not material.abstract
    assert shader.base.name == "Imageable" and not shader.abstract
    assert material.find("brdf") is material.find(tokens.BRDF)
    assert (
        material.find("detail_directx_normals").name
        == tokens.DETAIL_DIRECTX_NORMALS
    )
    assert (
        material.find("albedo_map").kind == ernie.AttributeKind.TENSOR
    )
    assert shader.find("info_id").options == [
        "UsdPreviewSurface",
        "UsdUVTexture",
        "UsdPrimvarReader_float2",
        "UsdTransform2d",
    ]
    assert shader.find("source_color_space").options == [
        "auto",
        "raw",
        "sRGB",
    ]
    assert ernie.Material.albedo.group == "Surface"


def test_material_defaults_and_keyword_define():
    scene = ernie.Scene()
    rock = scene.define(
        "/Looks/Rock",
        ernie.Material,
        brdf="hapke",
        albedo=(0.3, 0.3, 0.3),
    )
    assert isinstance(rock, ernie.Material) and isinstance(
        rock, ernie.Imageable
    )
    assert rock.brdf == "hapke" and rock.albedo == (0.3, 0.3, 0.3)
    assert scene.get("/Looks/Rock", tokens.BRDF) == "hapke"
    assert rock.albedo_map_enable is True
    assert (
        rock.detail_tile_size == 4.0
        and rock.detail_hex_sharpness == 7.0
    )
    assert (
        rock.detail_hex_tiling is True
        and rock.detail_directx_normals is False
    )
    assert rock.emissive_color == (0.0, 0.0, 0.0)
    rock.detail_tile_size = 2
    assert rock.detail_tile_size == 2.0 and isinstance(
        rock.detail_tile_size, float
    )
    with pytest.raises(ValueError):
        rock.albedo_map_enable = "yes"


def test_shader_network_through_classes():
    scene = ernie.Scene()
    material = ernie.Material.define(scene, "/Looks/Wood")
    surface = material.prim.define(
        "Surface",
        ernie.Shader,
        info_id=tokens.USD_PREVIEW_SURFACE,
        roughness=0.3,
    )
    texture = ernie.Shader.define(
        scene,
        "/Looks/Wood/Tex",
        info_id=tokens.USD_UV_TEXTURE,
        wrap_s="clamp",
    )
    assert (
        surface.diffuse_color == (0.18, 0.18, 0.18)
        and surface.roughness == 0.3
    )
    assert surface.ior == 1.5 and surface.clearcoat_roughness == 0.01
    assert surface.use_specular_workflow is False
    assert (
        texture.wrap_s == "clamp" and texture.wrap_t == "useMetadata"
    )
    assert (
        texture.source_color_space == "auto"
        and texture.varname == "st"
    )
    texture.texture = np.zeros((2, 2, 3), np.float32)
    assert texture.texture.shape == (2, 2, 3)
    with pytest.raises(TypeError, match="not a Material"):
        ernie.Material.at(scene, "/Looks/Wood/Surface")


def test_texture_bias_is_a_four_vector():
    scene = ernie.Scene()
    texture = ernie.Shader.define(
        scene, "/Looks/Tex", info_id=tokens.USD_UV_TEXTURE
    )
    assert ernie.Shader.bias.kind == ernie.AttributeKind.VEC4
    assert texture.bias == (0.0, 0.0, 0.0, 0.0)
    texture.bias = (-1, -1, -1, 0)
    assert texture.bias == (-1.0, -1.0, -1.0, 0.0)
    assert scene.get("/Looks/Tex", tokens.SHADER_BIAS) == (
        -1.0,
        -1.0,
        -1.0,
        0.0,
    )
    with pytest.raises(ValueError, match="4-vector"):
        texture.bias = (1.0, 2.0, 3.0)
