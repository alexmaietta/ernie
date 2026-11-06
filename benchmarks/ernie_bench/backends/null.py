# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import numpy as np

from ernie_bench import __version__
from ernie_bench.backends.base import OUTPUT_FORMATS, BackendInfo, Capabilities
from ernie_bench.scene import Scene, State
from ernie_bench.config import OUTPUTS, TIERS, Workload


class NullBackend:
    def info(self) -> BackendInfo:
        return BackendInfo("null", __version__, "none")

    def capabilities(self) -> Capabilities:
        return Capabilities(frozenset(TIERS), frozenset(OUTPUTS))

    def load(self, scene: Scene, workload: Workload) -> None:
        self._workload = workload

    def update(self, state: State) -> None:
        pass

    def render(self) -> None:
        pass

    def sync(self) -> None:
        pass

    def readback(self) -> dict[str, np.ndarray]:
        w = self._workload
        return {
            output: np.zeros((w.image_count, w.height, w.width, OUTPUT_FORMATS[output][0]), OUTPUT_FORMATS[output][1])
            for output in w.outputs
        }

    def stream(self) -> int | None:
        return None

    def close(self) -> None:
        pass
