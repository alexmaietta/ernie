# SPDX-License-Identifier: BSD-3-Clause
import os
import sys
import time
from time import perf_counter_ns

import numpy as np

from ernie_bench.backends.null import NullBackend


def spin(seconds):
    end = perf_counter_ns() + int(seconds * 1e9)
    while perf_counter_ns() < end:
        pass


class Spinner(NullBackend):
    def render(self):
        spin(0.002)


class Crasher(NullBackend):
    def render(self):
        os.abort()


class Hanger(NullBackend):
    def render(self):
        time.sleep(3600)


class Chatty(NullBackend):
    def render(self):
        print("chatty stdout", flush=True)
        print("chatty stderr", file=sys.stderr, flush=True)


class SlowStart(NullBackend):
    def __init__(self):
        time.sleep(0.3)


class Raiser(NullBackend):
    def load(self, scene, workload):
        raise ValueError("bad scene")


def paint(workload, seconds):
    count, height, width = workload.image_count, workload.height, workload.width
    index = np.arange(count, dtype=np.float32)[:, None, None]
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    shape = (count, height, width)
    rgb = np.stack([np.broadcast_to(x / width, shape), np.broadcast_to(y / height, shape), np.broadcast_to((index / count + seconds) % 1, shape)], axis=-1)
    depth = np.broadcast_to(1 + index + seconds + x / width, shape)[..., None]
    segmentation = np.broadcast_to(index.astype(np.int32) + 1 + (x // 4).astype(np.int32), shape)[..., None]
    return {"rgb": rgb.astype(np.float32), "depth": depth.astype(np.float32), "segmentation": segmentation.astype(np.int32)}


class Painter(NullBackend):
    def update(self, state):
        self.time = state.time

    def readback(self):
        return paint(self._workload, self.time)


class Squashed(Painter):
    def readback(self):
        return {output: frames[:, ::2] for output, frames in super().readback().items()}


class CacheReporter(NullBackend):
    def info(self):
        info = super().info()
        names = ("XDG_CACHE_HOME", "WARP_CACHE_PATH", "__GL_SHADER_DISK_CACHE_PATH", "ERNIE_BENCH_CACHE")
        info.details.update({name: os.environ.get(name) for name in names})
        return info


class Litterer(NullBackend):
    def load(self, scene, workload):
        super().load(scene, workload)
        with open("litter.txt", "w") as litter:
            litter.write("renderers sometimes drop files in their working directory\n")


class PainterCopy(Painter):
    pass
