# SPDX-License-Identifier: BSD-3-Clause
from pathlib import Path

import pytest

from ernie_bench.config import Suite, Workload, load_suite

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_sweep_expands_cartesian_product_sorted_by_worlds():
    suite = Suite(
        "s",
        ("null",),
        (),
        Workload(),
        sweep={"resolution": [[32, 32], [64, 48]], "worlds": [16, 1]},
    )
    points = suite.points()
    assert [(p.worlds, p.width, p.height) for p in points] == [
        (1, 32, 32),
        (1, 64, 48),
        (16, 32, 32),
        (16, 64, 48),
    ]


def test_unknown_sweep_axis_is_rejected():
    with pytest.raises(ValueError, match="sweep axes"):
        Suite("s", ("null",), (), Workload(), sweep={"bogus": [1]})


def test_workload_validates_tier_and_outputs():
    assert Workload(outputs=["rgb", "depth"]).outputs == ("rgb", "depth")
    with pytest.raises(ValueError, match="tier"):
        Workload(tier="raster")
    with pytest.raises(ValueError, match="outputs"):
        Workload(outputs=("normals",))


@pytest.mark.parametrize("path", sorted((CONFIGS / "suites").glob("*.toml")), ids=lambda p: p.stem)
def test_shipped_suites_load(path):
    suite = load_suite(path)
    assert suite.scenes
    assert suite.points()
