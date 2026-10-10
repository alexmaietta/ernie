# SPDX-License-Identifier: BSD-3-Clause
import struct
import zlib

import numpy as np

import ernie
from ernie import tokens


def write_png(path, rgb):
    """Writes a uint8 (rows, cols, 3) array as an 8-bit RGB PNG, with the standard library."""
    rows, cols, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[row].tobytes() for row in range(rows))

    def chunk(kind, data):
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )

    header = struct.pack(">IIBBBBB", cols, rows, 8, 2, 0, 0, 0)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


def preview_surface(scene, material):
    """A Material whose surface is a UsdPreviewSurface with a UsdUVTexture on diffuseColor,
    reading primvar st."""
    looks = scene.define(material, ernie.Material)
    surface = looks.define("Surface", ernie.Shader, info_id=tokens.USD_PREVIEW_SURFACE)
    texture = looks.define("Tex", ernie.Shader, info_id=tokens.USD_UV_TEXTURE)
    reader = looks.define(
        "St", ernie.Shader, info_id=tokens.USD_PRIMVAR_READER_FLOAT2, varname="st"
    )
    scene.connect(looks.path, tokens.SURFACE_OUTPUT, surface.path, tokens.SURFACE_OUTPUT)
    scene.connect(surface.path, tokens.DIFFUSE_COLOR, texture.path, "outputs:rgb")
    scene.connect(texture.path, tokens.ST, reader.path, "outputs:result")
    return texture


def test_connections_are_usd_property_paths():
    scene = ernie.Scene()
    preview_surface(scene, "/Looks/M")
    assert scene.get("/Looks/M/Surface", "inputs:diffuseColor.connect") == "/Looks/M/Tex.outputs:rgb"
    assert scene.connection("/Looks/M/Surface", tokens.DIFFUSE_COLOR) == (
        "/Looks/M/Tex",
        "outputs:rgb",
    )
    assert scene.connection("/Looks/M/Surface", tokens.NORMAL_INPUT) is None


def test_meshes_show_an_image_file_through_their_texture_coordinates(renderer, tmp_path):
    # Red on the image's left half, blue on its right, on a quad seen from above.
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    image[:, :2] = (255, 0, 0)
    image[:, 2:] = (0, 0, 255)
    write_png(tmp_path / "halves.png", image)

    scene = ernie.Scene()
    scene.define("/World/Camera", ernie.Camera, focal_length=18.0).set(
        tokens.TRANSFORM, ernie.look_at_transform((0, 0, 60), (0, 0, 0), up=(0, 1, 0))
    )
    # A dim sun, so nothing clips, on a matte material, so no highlight greys it.
    scene.define("/World/Sun", ernie.DistantLight, intensity=0.3)
    scene.define("/Render/Settings", ernie.RenderSettings, resolution=(48, 48))
    half = 12.0
    points = np.array([[-half, -half, 0], [half, -half, 0], [half, half, 0], [-half, half, 0]])
    scene.define(
        "/World/Quad",
        ernie.Mesh,
        points=points,
        face_vertex_counts=np.array([4]),
        face_vertex_indices=np.arange(4),
        material_binding="/Looks/M",
    ).write("primvars:st", np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32))
    texture = preview_surface(scene, "/Looks/M")
    ernie.Material.at(scene, "/Looks/M").brdf = "lambert"
    texture.file = str(tmp_path / "halves.png")

    pixels = renderer.render(scene).astype(np.int32)
    # About 6 m either side of the middle (1.45 m per pixel), well inside each half, where the
    # filtering blends only texels of one colour: the primaries come back pure.
    left = pixels[24, 20, :3]
    right = pixels[24, 28, :3]
    assert left[0] > 100 and max(left[1], left[2]) < 0.1 * left[0], f"red on the left {left}"
    assert right[2] > 100 and max(right[0], right[1]) < 0.1 * right[2], f"blue on the right {right}"
    assert renderer.take_validation_messages() == []


def test_preview_surfaces_shade_metals_and_dielectrics_differently(renderer):
    assert "preview_surface" in renderer.brdfs
    scene = ernie.Scene()
    scene.define("/World/Camera", ernie.Camera, focal_length=18.0).set(
        tokens.TRANSFORM, ernie.look_at_transform((0, 0, 60), (0, 0, 0), up=(0, 1, 0))
    )
    scene.define("/World/Sun", ernie.DistantLight).set(
        tokens.TRANSFORM, ernie.distant_light_transform((0.3, 0.0, 1.0))
    )
    scene.define("/Render/Settings", ernie.RenderSettings, resolution=(48, 48))
    half = 12.0
    points = np.array([[-half, -half, 0], [half, -half, 0], [half, half, 0], [-half, half, 0]])
    material = scene.define("/Looks/M", ernie.Material)
    scene.define(
        "/World/Quad",
        ernie.Mesh,
        points=points,
        face_vertex_counts=np.array([4]),
        face_vertex_indices=np.arange(4),
        material_binding=material,
    )
    surface = material.define(
        "Surface",
        ernie.Shader,
        info_id=tokens.USD_PREVIEW_SURFACE,
        diffuse_color=(0.6, 0.6, 0.6),
        roughness=0.3,
    )
    scene.connect(material.path, tokens.SURFACE_OUTPUT, surface.path, tokens.SURFACE_OUTPUT)

    dielectric = renderer.render(scene).astype(np.int32)
    surface.metallic = 1.0
    metal = renderer.render(scene).astype(np.int32)
    assert np.abs(metal - dielectric)[16:32, 16:32, :3].mean() > 5, "metallic changes the shading"
    assert renderer.take_validation_messages() == []
