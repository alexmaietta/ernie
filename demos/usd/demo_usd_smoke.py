# SPDX-License-Identifier: BSD-3-Clause
"""OpenVDB's smoke sample in a USD shot, imported with Scene.import_usd:
a UsdVol Volume reading the OpenVDB file through Ernie's built-in
OpenVDB, an ember light linked to light the smoke alone with its
intensity sampled over time, a sun whose shadows leave out a post, and
the shot's camera and render settings. The shot is written in code
beside the sample (2.6 MB from openvdb.org, fetched once into the
cache), as .usda or packed with it into a .usdz."""

import os
import struct
import zlib

import numpy as np

import ernie
import ernie.demo
from ernie import tokens
from ernie.demo import (
    cache_dir,
    can_download,
    side_by_side,
    write_png,
)

URL = (
    "https://media.githubusercontent.com/media/AcademySoftwareFoundation/"
    "openvdb-website/master/download/models/smoke.vdb"
)
SHOT = ernie.Path("/World/Shot")
FORMATS = ["usda", "usdz"]

# The smoke spans about -34 to 19 by -12 to 93 by -26 to 27 of the
# sample's units, Y up; at 5 cm a unit it stands 5 m tall.
LAYER = """#usda 1.0
(
    defaultPrim = "Shot"
    upAxis = "Y"
    metersPerUnit = 0.05
    startTimeCode = 1
    endTimeCode = 48
    timeCodesPerSecond = 24
)

def Xform "Shot"
{
    def Volume "Smoke"
    {
        rel field:density = </Shot/Smoke/Density>

        def OpenVDBAsset "Density"
        {
            asset filePath = @smoke.vdb@
            token fieldName = "density"
        }
    }

    def Cube "Ground"
    {
        double size = 1
        color3f[] primvars:displayColor = [(0.45, 0.42, 0.38)]
        double3 xformOp:translate = (0, -14, 0)
        float3 xformOp:scale = (400, 2, 400)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
    }

    def Cube "Post"
    {
        double size = 1
        color3f[] primvars:displayColor = [(0.3, 0.32, 0.36)]
        double3 xformOp:translate = (48, 17, 10)
        float3 xformOp:scale = (8, 62, 8)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
    }

    def DistantLight "Sun" (
        prepend apiSchemas = ["CollectionAPI:shadowLink"]
    )
    {
        float inputs:intensity = 6
        float inputs:angle = 0.53
        rel collection:shadowLink:excludes = </Shot/Post>
        float3 xformOp:rotateXYZ = (-35, 40, 0)
        uniform token[] xformOpOrder = ["xformOp:rotateXYZ"]
    }

    def SphereLight "Ember" (
        prepend apiSchemas = ["CollectionAPI:lightLink"]
    )
    {
        float inputs:intensity.timeSamples = {
            1: 60,
            12: 220,
            24: 80,
            36: 260,
            48: 60,
        }
        color3f inputs:color = (1, 0.45, 0.15)
        float inputs:radius = 4
        uniform bool collection:lightLink:includeRoot = 0
        rel collection:lightLink:includes = </Shot/Smoke>
        double3 xformOp:translate = (-8, 4, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }
}

def Camera "ShotCam"
{
    float focalLength = 24
    float2 clippingRange = (1, 5000)
    double3 xformOp:translate = (10, 45, 190)
    float3 xformOp:rotateXYZ = (-4, 8, 0)
    uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ"]
}

def Scope "Render"
{
    def RenderSettings "Settings"
    {
        rel camera = </ShotCam>
        uniform int2 resolution = (960, 540)
    }
}
"""


def write_usdz(path, files):
    """Writes `files` ((name, bytes), the layer first) as a .usdz: an
    uncompressed zip whose files start on 64-byte boundaries, padded in
    each entry's extra field, as USD requires."""
    out = bytearray()
    directory = bytearray()
    for name, data in files:
        encoded = name.encode()
        crc = zlib.crc32(data)
        offset = len(out)
        pad = (64 - (offset + 30 + len(encoded) + 4) % 64) % 64
        extra = struct.pack("<HH", 0x1986, pad) + bytes(pad)
        out += struct.pack(
            "<IHHHHHIIIHH",
            0x04034B50,
            20,
            0,
            0,
            0,
            0,
            crc,
            len(data),
            len(data),
            len(encoded),
            len(extra),
        )
        out += encoded + extra + data
        directory += struct.pack(
            "<IHHHHHHIIIHHHHHII",
            0x02014B50,
            20,
            20,
            0,
            0,
            0,
            0,
            crc,
            len(data),
            len(data),
            len(encoded),
            0,
            0,
            0,
            0,
            0,
            offset,
        )
        directory += encoded
    start = len(out)
    out += directory
    out += struct.pack(
        "<IHHHHIIH",
        0x06054B50,
        0,
        0,
        len(files),
        len(files),
        len(directory),
        start,
        0,
    )
    partial = path + ".part"
    with open(partial, "wb") as f:
        f.write(out)
    os.replace(partial, path)


def asset_directory():
    return os.path.join(cache_dir(), "usd", "openvdb")


class Demo(ernie.demo.Demo):
    num_frames = 1

    def __init__(self, viewer, args):
        super().__init__(viewer, args)
        self.format = 0
        self.density = 1.0
        self.result: ernie.UsdImport | None = None
        scene = self.scene
        scene.define("/World", ernie.Xform)
        # USD has no standard volume shader, so the smoke takes Ernie's.
        scene.define(
            "/Looks/Smoke",
            ernie.Material,
            density=self.density,
            volume_albedo=(0.9, 0.9, 0.9),
            phase_function="henyeyGreenstein",
            anisotropy=0.3,
        )

        available, why = ernie.usd_import_available()
        if not available:
            self.report(why, True)
            return
        sample = os.path.join(asset_directory(), "smoke.vdb")
        if not os.path.isfile(sample) and can_download():
            try:
                self.report(f"Fetching {URL}")
                ernie.demo.download(URL, sample, self.report)
            except (OSError, RuntimeError) as error:
                self.report(str(error), True)
        if not os.path.isfile(sample):
            self.report(
                "OpenVDB's smoke sample is not cached and cannot be fetched",
                True,
            )
            return
        self.write_shot(sample)
        self.load()

        self.widgets.combo(
            "Format",
            ["USDA", "USDZ (packed)"],
            0,
            on_change=self.choose,
        )
        self.widgets.slider(
            "Density",
            0.1,
            4.0,
            self.density,
            on_change=self.set_density,
        )
        if viewer is not None:
            viewer.play()

    def report(self, text, error=False):
        if self.viewer is not None:
            self.viewer.log(text, "error" if error else "info")
        else:
            print(text)

    def write_shot(self, sample):
        """The shot's layer beside the sample, and both in a .usdz,
        written when they change (Ernie unpacks a package once per
        version)."""
        directory = asset_directory()
        layer = os.path.join(directory, "smoke_shot.usda")
        package = os.path.join(directory, "smoke_shot.usdz")
        if os.path.isfile(layer) and os.path.isfile(package):
            with open(layer) as f:
                if f.read() == LAYER:
                    return
        with open(sample, "rb") as f:
            vdb = f.read()
        write_usdz(
            package,
            [("smoke_shot.usda", LAYER.encode()), ("smoke.vdb", vdb)],
        )
        with open(layer, "w") as f:
            f.write(LAYER)

    def load(self):
        """Imports the shot in the chosen format, its range the
        timeline's and its camera the viewer's."""
        if self.scene.has(SHOT):
            self.scene.remove(SHOT)
        file = os.path.join(
            asset_directory(), "smoke_shot." + FORMATS[self.format]
        )
        self.result = self.scene.import_usd(SHOT, file)
        for warning in self.result.warnings:
            self.report(warning, True)
        stage = ernie.Stage.at(self.scene, "/")
        if self.result.time_code_range is not None:
            stage.start_time_code, stage.end_time_code = (
                self.result.time_code_range
            )
            stage.time_codes_per_second = (
                self.result.time_codes_per_second
            )
            stage.time_code = stage.start_time_code
        self.scene.set(
            self.prim("/Shot/Smoke"),
            tokens.MATERIAL_BINDING,
            ernie.Path("/Looks/Smoke"),
        )
        camera = self.prim("/ShotCam")
        self.scene.set(camera, tokens.EXPOSURE_MODE, tokens.AUTO)
        self.scene.set(camera, tokens.SENSOR_NOISE, False)
        if self.viewer is not None:
            self.viewer.active_camera = camera
        self.report(f"Imported {os.path.basename(file)}")

    def prim(self, usd_path):
        """The Ernie prim the shot's USD prim became."""
        assert self.result is not None
        return self.result.prims[usd_path]

    def choose(self, index):
        self.format = index
        self.load()

    def set_density(self, density):
        self.density = density
        self.scene.set("/Looks/Smoke", tokens.VOLUME_DENSITY, density)

    def render_at(self, times):
        """The scene at each time code, RGBA uint8."""
        stage = ernie.Stage.at(self.scene, "/")
        images = []
        for t in times:
            stage.time_code = float(t)
            images.append(np.array(self.renderer.render(self.scene)))
        return images

    def save_output(self, path):
        """The ember dim, then bright."""
        write_png(path, side_by_side(self.render_at([1.0, 36.0])))

    def test_final(self):
        if self.result is None:
            print("Skipped: no USD importer or no smoke sample")
            return
        scene = self.scene
        smoke = self.prim("/Shot/Smoke")
        density = self.prim("/Shot/Smoke/Density")
        assert scene.type_of(smoke) == "Volume"
        assert scene.type_of(density) == "OpenVDBAsset"
        assert str(scene.get(density, tokens.FILE_PATH)).endswith(
            "smoke.vdb"
        )
        assert scene.get(
            self.prim("/Shot/Ember"), "collection:lightLink:includes"
        ) == [smoke]
        settings = self.prim("/Render/Settings")
        assert scene.get(settings, tokens.CAMERA_REL) == [
            self.prim("/ShotCam")
        ]

        # The smoke shows, and the ember brightens it as it flares.
        dim, bright = (
            image[..., :3].astype(float)
            for image in self.render_at([1.0, 36.0])
        )
        scene.set(smoke, tokens.VISIBILITY, tokens.INVISIBLE)
        (bare,) = (
            image[..., :3].astype(float)
            for image in self.render_at([1.0])
        )
        scene.set(smoke, tokens.VISIBILITY, tokens.INHERITED)
        shown = (np.abs(dim - bare).max(-1) > 16).mean()
        print(
            f"smoke covers {shown:.3f} of the view; means {bare.mean():.1f} bare, "
            f"{dim.mean():.1f} dim, {bright.mean():.1f} bright"
        )
        assert shown > 0.05, "the smoke should show"
        assert bright.mean() > dim.mean(), "the ember should flare"

        # Packed: the sample comes out of the .usdz and renders the same
        # (exposure settled on each).
        before = self.render_at([1.0, 1.0])[-1][..., :3].astype(float)
        self.choose(1)
        assert not self.result.warnings, self.result.warnings
        packed = str(
            scene.get(
                self.prim("/Shot/Smoke/Density"), tokens.FILE_PATH
            )
        )
        print(f"packed sample read from {packed}")
        assert (
            packed.endswith("smoke.vdb") and "usd_packages" in packed
        )
        again = self.render_at([1.0, 1.0])[-1][..., :3].astype(float)
        assert np.abs(again - before).mean() < 1.0


if __name__ == "__main__":
    ernie.demo.main(Demo)
