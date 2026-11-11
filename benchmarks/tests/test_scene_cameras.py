# SPDX-License-Identifier: BSD-3-Clause
from dataclasses import replace

from ernie_bench.config import SceneConfig, Suite, Workload


def test_scene_camera_count_overrides_the_sweep():
    suite = Suite("s", ("null",), (), Workload(), sweep={"worlds": [4, 8], "cameras_per_world": [1, 2]})
    fixed = SceneConfig("h", "hide_and_seek", cameras_per_world=6)
    assert [(w.worlds, w.cameras_per_world) for w in suite.scene_points(fixed)] == [(4, 6), (8, 6)]
    assert len(suite.scene_points(replace(fixed, cameras_per_world=None))) == 4
