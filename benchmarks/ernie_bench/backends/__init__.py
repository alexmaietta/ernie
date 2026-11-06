# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import importlib

from ernie_bench.backends.base import Backend

BACKENDS = {
    "null": ("ernie_bench.backends.null:NullBackend", {}),
    "newton": ("ernie_bench.backends.newton:NewtonBackend", {}),
    "ovrtx": ("ernie_bench.backends.ovrtx:OvrtxBackend", {}),
    "ovrtx_tiled": ("ernie_bench.backends.ovrtx:OvrtxBackend", {"tiled": True}),
    "mitsuba": ("ernie_bench.backends.mitsuba:MitsubaBackend", {}),
    "mitsuba_cpu": ("ernie_bench.backends.mitsuba:MitsubaBackend", {"variant": "llvm_ad_rgb"}),
    "madrona_rt": ("ernie_bench.backends.madrona:MadronaBackend", {"raytrace": True}),
    "madrona_rast": ("ernie_bench.backends.madrona:MadronaBackend", {"raytrace": False}),
}
for api in (None, "vulkan", "cuda"):
    for mode in (None, "hardware", "software"):
        BACKENDS["_".join(["ernie", *filter(None, (api, mode))])] = (
            "ernie_bench.backends.ernie:ErnieBackend",
            {"api": api and api.upper(), "traversal": (mode or "auto").upper(), "sampling": (mode or "auto").upper()},
        )


def known(name: str) -> bool:
    return name in BACKENDS or ":" in name


def create(name: str) -> Backend:
    path, options = BACKENDS.get(name, (name, {}))
    module, _, attribute = path.partition(":")
    return getattr(importlib.import_module(module), attribute)(**options)
