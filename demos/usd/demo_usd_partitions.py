# SPDX-License-Identifier: BSD-3-Clause
"""Four cloned environments as Omniverse lays them out for a tiled
camera, imported with Scene.import_usd: overlapping at the origin, kept
apart by RTX scene partitions (`omni:scenePartition`), which become
Ernie's worlds, and lit by Omniverse's photometric lights. The
environments' cameras render as one batch, each seeing its own."""

import os

import numpy as np

import ernie
import ernie.demo
from ernie import tokens
from ernie.demo import cache_dir, side_by_side, write_png

ENVIRONMENTS = 4
WIDTH, HEIGHT = 320, 240
ROOT = ernie.Path("/World/Lab")
PRODUCT = ernie.Path("/Render/Environments")
# Each environment's block colour, as its dominant channels.
COLOURS = ["red", "green", "blue", "yellow"]

# Each environment a table, a block of its own colour in the same
# spot as the others', and a lamp of its own power and colour; the
# floor and the dome are every environment's.
LAYER = """#usda 1.0
(
    defaultPrim = "Lab"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "Lab"
{
    def Cube "Floor"
    {
        double size = 1
        color3f[] primvars:displayColor = [(0.5, 0.5, 0.5)]
        double3 xformOp:translate = (0, 0, -0.05)
        float3 xformOp:scale = (20, 20, 0.1)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
    }

    def DomeLight "Sky" (
        prepend apiSchemas = ["PhotometricDomeLightAPI"]
    )
    {
        float photometric:illuminance = 400
    }

    def Xform "Envs"
    {
        def Xform "Env_0"
        {
            token omni:scenePartition = "env_0"
            def Cube "Table"
            {
                double size = 1
                color3f[] primvars:displayColor = [(0.45, 0.3, 0.18)]
                double3 xformOp:translate = (0, 0, 0.35)
                float3 xformOp:scale = (1.2, 0.8, 0.7)
                uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
            }
            def Cube "Block"
            {
                double size = 0.3
                color3f[] primvars:displayColor = [(0.8, 0.1, 0.1)]
                double3 xformOp:translate = (0, 0, 0.85)
                uniform token[] xformOpOrder = ["xformOp:translate"]
            }
            def SphereLight "Lamp" (
                prepend apiSchemas = ["PhotometricAreaLightAPI"]
            )
            {
                float photometric:power = 1600
                token physical:illuminant = "blackbody"
                float inputs:colorTemperature = 2700
                float inputs:radius = 0.05
                double3 xformOp:translate = (-0.5, -0.4, 1.6)
                uniform token[] xformOpOrder = ["xformOp:translate"]
            }
            def Xform "Rig"
            {
                float xformOp:rotateZ = 0
                uniform token[] xformOpOrder = ["xformOp:rotateZ"]
                def Camera "Camera"
                {
                    float focalLength = 24
                    double3 xformOp:translate = (0, -2.6, 1.5)
                    float3 xformOp:rotateXYZ = (75, 0, 0)
                    uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ"]
                }
            }
        }
        def Xform "Env_1"
        {
            token omni:scenePartition = "env_1"
            def Cube "Table"
            {
                double size = 1
                color3f[] primvars:displayColor = [(0.45, 0.3, 0.18)]
                double3 xformOp:translate = (0, 0, 0.35)
                float3 xformOp:scale = (1.2, 0.8, 0.7)
                uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
            }
            def Cube "Block"
            {
                double size = 0.36
                color3f[] primvars:displayColor = [(0.1, 0.7, 0.15)]
                double3 xformOp:translate = (0, 0, 0.88)
                uniform token[] xformOpOrder = ["xformOp:translate"]
            }
            def SphereLight "Lamp" (
                prepend apiSchemas = ["PhotometricAreaLightAPI"]
            )
            {
                float photometric:power = 3200
                token physical:illuminant = "blackbody"
                float inputs:colorTemperature = 4000
                float inputs:radius = 0.05
                double3 xformOp:translate = (0.5, -0.4, 1.6)
                uniform token[] xformOpOrder = ["xformOp:translate"]
            }
            def Xform "Rig"
            {
                float xformOp:rotateZ = 30
                uniform token[] xformOpOrder = ["xformOp:rotateZ"]
                def Camera "Camera"
                {
                    float focalLength = 24
                    double3 xformOp:translate = (0, -2.6, 1.5)
                    float3 xformOp:rotateXYZ = (75, 0, 0)
                    uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ"]
                }
            }
        }
        def Xform "Env_2"
        {
            token omni:scenePartition = "env_2"
            def Cube "Table"
            {
                double size = 1
                color3f[] primvars:displayColor = [(0.45, 0.3, 0.18)]
                double3 xformOp:translate = (0, 0, 0.35)
                float3 xformOp:scale = (1.2, 0.8, 0.7)
                uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
            }
            def Cube "Block"
            {
                double size = 0.42
                color3f[] primvars:displayColor = [(0.12, 0.2, 0.8)]
                double3 xformOp:translate = (0, 0, 0.91)
                uniform token[] xformOpOrder = ["xformOp:translate"]
            }
            def SphereLight "Lamp" (
                prepend apiSchemas = ["PhotometricAreaLightAPI"]
            )
            {
                float photometric:power = 6400
                token physical:illuminant = "blackbody"
                float inputs:colorTemperature = 5500
                float inputs:radius = 0.05
                double3 xformOp:translate = (0, 0.5, 1.8)
                uniform token[] xformOpOrder = ["xformOp:translate"]
            }
            def Xform "Rig"
            {
                float xformOp:rotateZ = 60
                uniform token[] xformOpOrder = ["xformOp:rotateZ"]
                def Camera "Camera"
                {
                    float focalLength = 24
                    double3 xformOp:translate = (0, -2.6, 1.5)
                    float3 xformOp:rotateXYZ = (75, 0, 0)
                    uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ"]
                }
            }
        }
        def Xform "Env_3"
        {
            token omni:scenePartition = "env_3"
            def Cube "Table"
            {
                double size = 1
                color3f[] primvars:displayColor = [(0.45, 0.3, 0.18)]
                double3 xformOp:translate = (0, 0, 0.35)
                float3 xformOp:scale = (1.2, 0.8, 0.7)
                uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
            }
            def Cube "Block"
            {
                double size = 0.48
                color3f[] primvars:displayColor = [(0.8, 0.7, 0.1)]
                double3 xformOp:translate = (0, 0, 0.94)
                uniform token[] xformOpOrder = ["xformOp:translate"]
            }
            def SphereLight "Lamp" (
                prepend apiSchemas = ["PhotometricAreaLightAPI"]
            )
            {
                float photometric:power = 12800
                token physical:illuminant = "blackbody"
                float inputs:colorTemperature = 6500
                float inputs:radius = 0.05
                double3 xformOp:translate = (0, -0.6, 1.8)
                uniform token[] xformOpOrder = ["xformOp:translate"]
            }
            def Xform "Rig"
            {
                float xformOp:rotateZ = 90
                uniform token[] xformOpOrder = ["xformOp:rotateZ"]
                def Camera "Camera"
                {
                    float focalLength = 24
                    double3 xformOp:translate = (0, -2.6, 1.5)
                    float3 xformOp:rotateXYZ = (75, 0, 0)
                    uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ"]
                }
            }
        }
    }

    # Pebbles on the tables, two in each environment's partition.
    def PointInstancer "Pebbles"
    {
        point3f[] positions = [(-0.45, 0.25, 0.73), (0.45, 0.25, 0.73), (-0.45, -0.25, 0.73), (0.45, -0.25, 0.73), (-0.25, 0.3, 0.73), (0.25, 0.3, 0.73), (-0.25, -0.3, 0.73), (0.25, -0.3, 0.73)]
        int[] protoIndices = [0, 0, 0, 0, 0, 0, 0, 0]
        rel prototypes = </Lab/Pebbles/Pebble>
        token[] primvars:omni:scenePartition = ["env_0", "env_0", "env_1", "env_1", "env_2", "env_2", "env_3", "env_3"] (
            interpolation = "vertex"
        )
        def Sphere "Pebble"
        {
            double radius = 0.06
            color3f[] primvars:displayColor = [(0.85, 0.85, 0.8)]
        }
    }

    # Sees every partition: what the environments would look like
    # without them.
    def Camera "Overview"
    {
        float focalLength = 24
        double3 xformOp:translate = (0, -2.6, 1.5)
        float3 xformOp:rotateXYZ = (75, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ"]
    }
}
"""


def world_of(scene, path):
    """A prim's world: its own or its nearest ancestor's, -1 none."""
    while path != ernie.Path("/"):
        world = scene.get(path, tokens.WORLD)
        if world is not None:
            return int(world)
        path = path.parent
    return -1


class Demo(ernie.demo.Demo):
    num_frames = 1

    def __init__(self, viewer, args):
        super().__init__(viewer, args)
        self.result: ernie.UsdImport | None = None
        self.scene.define("/World", ernie.Xform)
        available, why = ernie.usd_import_available()
        if not available:
            self.report(why, True)
            return
        self.load()
        names = [f"env_{i}" for i in range(ENVIRONMENTS)]
        self.widgets.combo(
            "Camera",
            names + ["Overview (every partition)"],
            0,
            on_change=self.look,
        )
        self.look(0)

    def report(self, text, error=False):
        if self.viewer is not None:
            self.viewer.log(text, "error" if error else "info")
        else:
            print(text)

    def load(self):
        """Writes the layer to the cache and imports it, with a product
        of the environments' cameras."""
        scene = self.scene
        file = os.path.join(cache_dir(), "usd", "partitions.usda")
        os.makedirs(os.path.dirname(file), exist_ok=True)
        with open(file, "w") as f:
            f.write(LAYER)
        self.result = scene.import_usd(ROOT, file)
        for warning in self.result.warnings:
            self.report(warning, True)
        cameras = [self.camera_of(i) for i in range(ENVIRONMENTS)]
        # A batch meters no exposure: an indoor one, about EV 8.
        for camera in cameras + [self.prim("/Lab/Overview")]:
            scene.set(camera, tokens.EXPOSURE_FSTOP, 2.8)
            scene.set(camera, tokens.EXPOSURE_TIME, 1.0 / 30.0)
            scene.set(camera, tokens.EXPOSURE_ISO, 100.0)
            scene.set(camera, tokens.SENSOR_NOISE, False)
        var = scene.define(PRODUCT.child("LdrColor"), ernie.RenderVar)
        var.source_name = "LdrColor"
        scene.define(
            PRODUCT,
            ernie.RenderProduct,
            camera=cameras,
            resolution=(WIDTH, HEIGHT),
            ordered_vars=[var.path],
        )
        scene.define(
            "/Render/Settings",
            ernie.RenderSettings,
            camera=self.prim("/Lab/Overview"),
            resolution=(WIDTH, HEIGHT),
        )
        self.report(
            f"Imported {os.path.basename(file)} with {ENVIRONMENTS} partitions"
        )

    def look(self, index):
        """Looks through environment `index`'s camera, or the overview's."""
        if self.viewer is not None:
            self.viewer.active_camera = (
                self.camera_of(index)
                if index < ENVIRONMENTS
                else self.prim("/Lab/Overview")
            )

    def render_views(self):
        """The environments' views, one batch, as RGBA uint8 images."""
        outputs = self.renderer.step(
            self.scene, render_products=[str(PRODUCT)]
        )
        frame = outputs[str(PRODUCT)].frames[0].render_vars
        batch = np.array(frame[str(PRODUCT.child("LdrColor"))].map())
        return [batch[i] for i in range(ENVIRONMENTS)]

    @staticmethod
    def env(i):
        return f"/Lab/Envs/Env_{i}"

    def camera_of(self, i):
        return self.prim(self.env(i) + "/Rig/Camera")

    def prim(self, usd_path):
        """The Ernie prim the lab's USD prim became."""
        assert self.result is not None
        return self.result.prims[usd_path]

    def save_output(self, path):
        """The four environments' views, then the overview's."""
        if self.result is None:
            return
        images = self.render_views()
        images.append(np.array(self.renderer.render(self.scene)))
        write_png(path, side_by_side(images))

    def test_final(self):
        if self.result is None:
            print("Skipped: no USD importer")
            return
        scene = self.scene
        assert not self.result.warnings, self.result.warnings

        # Each environment is a world, and its camera and lamp in it.
        names = scene.get(ROOT, "ernie:usd:scenePartitions")
        assert (
            isinstance(names, list) and len(names) == ENVIRONMENTS
        ), names
        for i in range(ENVIRONMENTS):
            world = names.index(f"env_{i}")
            assert world_of(scene, self.prim(self.env(i))) == world
            assert world_of(scene, self.camera_of(i)) == world
            lamp = self.prim(self.env(i) + "/Lamp")
            assert (
                scene.get(lamp, tokens.PHOTOMETRIC_POWER) is not None
            )
            assert (
                scene.get(lamp, tokens.PHYSICAL_ILLUMINANT)
                == "blackbody"
            )
        assert (
            scene.get(
                self.prim("/Lab/Sky"), tokens.PHOTOMETRIC_ILLUMINANCE
            )
            == 400.0
        )
        assert (
            scene.read(self.prim("/Lab/Pebbles"), tokens.WORLDS)
            is not None
        )

        # Every view sees its own block in the middle, though the
        # blocks share the spot.
        for i, view in enumerate(self.render_views()):
            r, g, b = (
                int(c) for c in view[HEIGHT // 2, WIDTH // 2, :3]
            )
            print(f"env_{i} centre ({r}, {g}, {b})")
            seen = {
                "red": r > g + 30 and r > b + 30,
                "green": g > r + 30 and g > b + 30,
                "blue": b > r + 30 and b > g + 30,
                "yellow": r > b + 30 and g > b + 30,
            }[COLOURS[i]]
            assert seen, (
                f"env_{i}'s camera should see its own block alone"
            )


if __name__ == "__main__":
    ernie.demo.main(Demo)
