# SPDX-License-Identifier: BSD-3-Clause
import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / "envs" / "ovrtx" / ".venv" / "bin" / "python"
SCRIPT = """
import json, sys
from pxr import Usd
from ernie_bench.backends.ovrtx import author_layer
from ernie_bench.config import SceneConfig, Workload
from ernie_bench.datasets import build_scene

tier, tiled = sys.argv[1], sys.argv[2] == "tiled"
workload = Workload(worlds=2, cameras_per_world=3, width=16, height=8, outputs=("rgb", "depth", "segmentation"), tier=tier)
scene = build_scene(SceneConfig("hide_and_seek", "hide_and_seek"), 2, 3)
layer, moving, cameras, products = author_layer(scene, workload, tiled)
stage = Usd.Stage.Open(layer)
product = stage.GetPrimAtPath(products[0])
instance = stage.GetPrimAtPath("/World/W00001/I00000")
print(json.dumps({
    "up": stage.GetMetadata("upAxis"),
    "cameras": cameras,
    "products": products,
    "moving": len(moving) == len(scene.motion.instance) and all(stage.GetPrimAtPath(path).IsA(Usd.Typed) for path in moving),
    "targets": [str(path) for path in product.GetRelationship("camera").GetTargets()],
    "vars": [str(path) for path in product.GetRelationship("orderedVars").GetTargets()],
    "resolution": list(product.GetAttribute("resolution").Get()),
    "settings": {attribute.GetName(): attribute.Get() for attribute in product.GetAuthoredAttributes() if attribute.GetName().startswith("omni:rtx")},
    "partition": stage.GetPrimAtPath("/World/W00001").GetAttribute("primvars:omni:scenePartition").Get(),
    "camera_partition": stage.GetPrimAtPath(cameras[-1]).GetAttribute("omni:scenePartition").Get(),
    "points": len(instance.GetAttribute("points").Get()),
    "schemas": list(instance.GetPrimTypeInfo().GetAppliedAPISchemas()),
    "lights": [str(prim.GetPath()) for prim in stage.Traverse() if prim.GetTypeName() == "DistantLight"],
}))
"""


def author(tier: str, mode: str) -> dict:
    if not PYTHON.exists():
        pytest.skip("no envs/ovrtx")
    result = subprocess.run([str(PYTHON), "-c", SCRIPT, tier, mode], capture_output=True, text=True, timeout=120, check=True)
    return json.loads(result.stdout)


def test_tiled_layer_holds_every_camera_in_one_product():
    stage = author("direct", "tiled")
    assert stage["up"] == "Z"
    assert stage["products"] == ["/Render/Tiled"] and stage["targets"] == stage["cameras"]
    assert len(stage["cameras"]) == 6 and stage["cameras"][-1] == "/World/W00001/C002"
    assert stage["resolution"] == [3 * 16, 2 * 8]
    assert stage["vars"] == [f"/Render/Tiled/{name}" for name in ("rgb", "depth", "segmentation", "ids")]
    assert stage["settings"] == {"omni:rtx:rendermode": "MinimalRendering", "omni:rtx:minimal:mode": 2, "omni:rtx:minimal:castShadows": True}
    assert stage["partition"] == stage["camera_partition"] == "W00001"
    assert stage["points"] > 0 and stage["schemas"] == ["SemanticsAPI:class"] and stage["moving"]
    assert stage["lights"]


def test_per_camera_products_carry_path_tracing_settings():
    stage = author("pathtraced", "products")
    assert stage["products"] == [f"/Render/P{index:05d}" for index in range(6)]
    assert stage["targets"] == stage["cameras"][:1] and stage["resolution"] == [16, 8]
    assert stage["settings"] == {"omni:rtx:rendermode": "PathTracing", "omni:rtx:pt:limits:maxBounces": 4, "omni:rtx:pt:denoising:enabled": False, "omni:rtx:pt:samplesPerPixel": 1}
