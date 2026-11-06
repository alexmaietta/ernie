# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

from ernie_bench.datasets import hide_and_seek
from ernie_bench.scene import Scene
from ernie_bench.config import SceneSpec

DATASETS = {
    "hide_and_seek": hide_and_seek.build,
}


def build_scene(spec: SceneSpec, worlds: int, cameras_per_world: int) -> Scene:
    if spec.dataset not in DATASETS:
        raise ValueError(f"unknown dataset {spec.dataset!r}, expected one of {sorted(DATASETS)}")
    return DATASETS[spec.dataset](spec, worlds, cameras_per_world)
