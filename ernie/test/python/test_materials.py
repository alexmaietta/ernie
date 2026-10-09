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
    scene.define(material, tokens.MATERIAL)
    surface = f"{material}/Surface"
    texture = f"{material}/Tex"
    reader = f"{material}/St"
    scene.define(surface, tokens.SHADER).set(tokens.INFO_ID, tokens.USD_PREVIEW_SURFACE)
    scene.define(texture, tokens.SHADER).set(tokens.INFO_ID, tokens.USD_UV_TEXTURE)
    scene.define(reader, tokens.SHADER).set(tokens.INFO_ID, tokens.USD_PRIMVAR_READER_FLOAT2).set(
        tokens.VARNAME, "st"
    )
    scene.connect(material, tokens.SURFACE_OUTPUT, surface, tokens.SURFACE_OUTPUT)
    scene.connect(surface, tokens.DIFFUSE_COLOR, texture, "outputs:rgb")
    scene.connect(texture, tokens.ST, reader, "outputs:result")
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
    scene.define("/World/Camera", tokens.CAMERA).set(
        tokens.TRANSFORM, ernie.look_at_transform((0, 0, 60), (0, 0, 0), up=(0, 1, 0))
    ).set(tokens.FOCAL_LENGTH, 18.0)
    # A dim sun, so nothing clips, on a matte material, so no highlight greys it.
    scene.define("/World/Sun", tokens.DISTANT_LIGHT).set(tokens.INTENSITY, 0.3)
    scene.define("/Render/Settings", tokens.RENDER_SETTINGS).set(tokens.RESOLUTION, (48, 48))
    half = 12.0
    points = np.array([[-half, -half, 0], [half, -half, 0], [half, half, 0], [-half, half, 0]])
    scene.define("/World/Quad", tokens.MESH).write(tokens.POINTS, points).write(
        tokens.FACE_VERTEX_COUNTS, np.array([4])
    ).write(tokens.FACE_VERTEX_INDICES, np.arange(4)).write(
        "primvars:st", np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
    ).set(tokens.MATERIAL_BINDING, ernie.Path("/Looks/M"))
    texture = preview_surface(scene, "/Looks/M")
    scene.set("/Looks/M", tokens.BRDF, "lambert")
    scene.set(texture, tokens.FILE, str(tmp_path / "halves.png"))

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
    scene.define("/World/Camera", tokens.CAMERA).set(
        tokens.TRANSFORM, ernie.look_at_transform((0, 0, 60), (0, 0, 0), up=(0, 1, 0))
    ).set(tokens.FOCAL_LENGTH, 18.0)
    scene.define("/World/Sun", tokens.DISTANT_LIGHT).set(
        tokens.TRANSFORM, ernie.distant_light_transform((0.3, 0.0, 1.0))
    )
    scene.define("/Render/Settings", tokens.RENDER_SETTINGS).set(tokens.RESOLUTION, (48, 48))
    half = 12.0
    points = np.array([[-half, -half, 0], [half, -half, 0], [half, half, 0], [-half, half, 0]])
    scene.define("/World/Quad", tokens.MESH).write(tokens.POINTS, points).write(
        tokens.FACE_VERTEX_COUNTS, np.array([4])
    ).write(tokens.FACE_VERTEX_INDICES, np.arange(4)).set(
        tokens.MATERIAL_BINDING, ernie.Path("/Looks/M")
    )
    scene.define("/Looks/M", tokens.MATERIAL)
    scene.define("/Looks/M/Surface", tokens.SHADER).set(
        tokens.INFO_ID, tokens.USD_PREVIEW_SURFACE
    ).set(tokens.DIFFUSE_COLOR, (0.6, 0.6, 0.6)).set(tokens.ROUGHNESS, 0.3)
    scene.connect("/Looks/M", tokens.SURFACE_OUTPUT, "/Looks/M/Surface", tokens.SURFACE_OUTPUT)

    dielectric = renderer.render(scene).astype(np.int32)
    scene.set("/Looks/M/Surface", tokens.METALLIC, 1.0)
    metal = renderer.render(scene).astype(np.int32)
    assert np.abs(metal - dielectric)[16:32, 16:32, :3].mean() > 5, "metallic changes the shading"
    assert renderer.take_validation_messages() == []
