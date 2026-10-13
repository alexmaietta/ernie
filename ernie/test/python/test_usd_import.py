# SPDX-License-Identifier: BSD-3-Clause
"""Scene.import_usd: the default prim as the root, units, variants,
material networks, and the UsdImport it returns. Skipped without the
USD importer plugin (-DERNIE_USD=ON)."""

import numpy as np
import pytest

import ernie

AVAILABLE, WHY = ernie.usd_import_available()
pytestmark = pytest.mark.skipif(not AVAILABLE, reason=WHY)

# A rover in centimetres, Y up: a triangle mesh, a ball whose colour a
# variant picks, and a UsdPreviewSurface bound to the mesh.
ROVER = """#usda 1.0
(
    defaultPrim = "Rover"
    upAxis = "Y"
    metersPerUnit = 0.01
)

def Xform "Rover" (
    variants = {
        string paint = "red"
    }
    prepend variantSets = "paint"
)
{
    double3 xformOp:translate = (0, 100, 0)
    uniform token[] xformOpOrder = ["xformOp:translate"]

    def Mesh "Deck" (
        prepend apiSchemas = ["MaterialBindingAPI"]
    )
    {
        point3f[] points = [(0, 0, 0), (100, 0, 0), (0, 0, 100)]
        int[] faceVertexCounts = [3]
        int[] faceVertexIndices = [0, 1, 2]
        rel material:binding = </Rover/Looks/Paint>
    }

    def Sphere "Ball"
    {
        double radius = 10
    }

    def Scope "Looks"
    {
        def Material "Paint"
        {
            token outputs:surface.connect = </Rover/Looks/Paint/Surface.outputs:surface>
            def Shader "Surface"
            {
                uniform token info:id = "UsdPreviewSurface"
                color3f inputs:diffuseColor = (0.2, 0.4, 0.6)
                token outputs:surface
            }
        }
    }

    variantSet "paint" = {
        "blue" {
            over "Ball"
            {
                color3f[] primvars:displayColor = [(0, 0, 1)]
            }
        }
        "red" {
            over "Ball"
            {
                color3f[] primvars:displayColor = [(1, 0, 0)]
            }
        }
    }
}
"""


@pytest.fixture
def rover(tmp_path):
    path = tmp_path / "rover.usda"
    path.write_text(ROVER)
    return path


def test_default_prim_becomes_the_root_in_scene_units(rover):
    scene = ernie.Scene()
    scene.define("/World", ernie.Xform)
    result = scene.import_usd("/World/rover", rover)
    assert isinstance(result, ernie.UsdImport)
    assert result.root.path == "/World/rover"
    assert result.root.type == "Xform"
    assert result.prims["/Rover/Deck"] == "/World/rover/Deck"
    assert result.warnings == []
    # 100 cm up Y is 1 m up Z.
    world = np.array(scene.world_transform("/World/rover")).reshape(
        4, 4
    )
    origin = world @ [0.0, 0.0, 0.0, 1.0]
    np.testing.assert_allclose(origin[:3], [0.0, 0.0, 1.0], atol=1e-9)


def test_variants_and_materials(rover):
    scene = ernie.Scene()
    scene.define("/World", ernie.Xform)
    scene.import_usd("/World/red", rover)
    scene.import_usd(
        "/World/blue", rover, variants={"/Rover": {"paint": "blue"}}
    )
    red = np.asarray(
        scene.get_prim("/World/red/Ball").read(
            "primvars:displayColor"
        )
    )
    blue = np.asarray(
        scene.get_prim("/World/blue/Ball").read(
            "primvars:displayColor"
        )
    )
    np.testing.assert_allclose(red.reshape(-1), [1.0, 0.0, 0.0])
    np.testing.assert_allclose(blue.reshape(-1), [0.0, 0.0, 1.0])
    deck = scene.get_prim("/World/red/Deck")
    assert (
        str(deck.get("material:binding")) == "/World/red/Looks/Paint"
    )
    surface = scene.get_prim("/World/red/Looks/Paint/Surface")
    assert surface.get("info:id") == "UsdPreviewSurface"


def test_missing_files_raise(tmp_path):
    scene = ernie.Scene()
    with pytest.raises(RuntimeError):
        scene.import_usd("/Missing", tmp_path / "missing.usda")
