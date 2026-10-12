# SPDX-License-Identifier: BSD-3-Clause
import ernie
from ernie import tokens


def test_volume_names_its_fields():
    scene = ernie.Scene()
    density = scene.define(
        "/World/Plume/Density",
        ernie.OpenVDBAsset,
        file_path="plume.nvdb",
        field_name="density",
    )
    temperature = scene.define(
        "/World/Plume/Temperature",
        ernie.OpenVDBAsset,
        file_path="plume.nvdb",
        field_name="temperature",
    )
    volume = scene.define("/World/Plume", ernie.Volume)
    assert isinstance(volume, ernie.Gprim)
    assert isinstance(density, ernie.FieldAsset)
    assert volume.field_names() == []
    assert volume.get_field("density") is None

    volume.set_field("density", density).set_field(
        "temperature", "/World/Plume/Temperature"
    )
    assert volume.get_field("density") == density.path
    assert volume.get_field("temperature") == temperature.path
    assert volume.field_names() == ["density", "temperature"]
    assert density.field_index == 0
    assert (
        scene.get("/World/Plume/Density", tokens.FILE_PATH)
        == "plume.nvdb"
    )
    assert ernie.OpenVDBAsset.file_path.is_asset


def test_nanovdb_field_holds_an_array():
    import numpy as np

    scene = ernie.Scene()
    grids = scene.define(
        "/World/Plume/Grids", ernie.NanoVDBField, field_name="density"
    )
    assert isinstance(grids, ernie.FieldBase)
    assert grids.buffer is None
    scene.write(
        "/World/Plume/Grids",
        tokens.NANOVDB_BUFFER,
        np.zeros(64, dtype=np.uint8),
    )
    assert grids.buffer.shape == (64,)
    assert grids.field_name == "density"


def test_material_volume_inputs():
    scene = ernie.Scene()
    smoke = scene.define(
        "/Looks/Smoke",
        ernie.Material,
        density=20,
        volume_albedo=(0.9, 0.9, 0.9),
        temperature_scale=1500,
    )
    assert smoke.density == 20.0
    assert smoke.volume_albedo == (0.9, 0.9, 0.9)
    assert smoke.density_field == "density"
    assert smoke.temperature_field == "temperature"
    assert smoke.blackbody == 1.0
    assert scene.get("/Looks/Smoke", tokens.VOLUME_DENSITY) == 20.0
    assert ernie.Material.anisotropy.group == "Volume"
