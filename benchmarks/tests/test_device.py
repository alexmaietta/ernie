# SPDX-License-Identifier: BSD-3-Clause
import ctypes

import pytest
from conftest import ScriptedBackend, measure, values

from ernie_bench.system import cuda

try:
    cuda.driver()
    cuda.PrimaryContext().close()
except cuda.CudaUnavailable:
    pytest.skip("needs a CUDA driver and device", allow_module_level=True)

FILL_BYTES = 64 << 20
SUBMIT_SECONDS = 0.004


class StreamBackend(ScriptedBackend):
    def __init__(self, blocking=False, **costs):
        super().__init__(**costs)
        self.blocking = blocking
        self.cuda = cuda.Stream()
        self.buffer = ctypes.c_uint64()
        with cuda.current(self.cuda.context.handle):
            cuda.call("cuMemAlloc_v2", ctypes.byref(self.buffer), FILL_BYTES)

    def render(self):
        super().render()
        with cuda.current(self.cuda.context.handle):
            cuda.call("cuMemsetD8Async", self.buffer, 1, FILL_BYTES, self.cuda.handle)
        if self.blocking:
            self.cuda.synchronize()

    def sync(self):
        self.cuda.synchronize()

    def stream(self):
        return self.cuda.handle.value

    def close(self):
        with cuda.current(self.cuda.context.handle):
            cuda.call("cuMemFree_v2", self.buffer)
        self.cuda.close()


def test_device_time_excludes_cpu_time_spent_submitting(scene, monkeypatch):
    samples = measure(StreamBackend(render=SUBMIT_SECONDS), scene, monkeypatch=monkeypatch)
    device = values(samples, "steady", "render_device_seconds")
    wall = values(samples, "steady", "render_seconds")
    assert len(device) == len(wall)
    assert all(0 < gpu < seconds - SUBMIT_SECONDS / 2 for gpu, seconds in zip(device, wall))


def test_render_that_waits_on_its_stream_falls_back_to_a_span(scene, monkeypatch):
    samples = measure(StreamBackend(blocking=True), scene, monkeypatch=monkeypatch)
    assert list(values(samples, "warmup", "device_timer_blocked")) == [1.0]
    assert values(samples, "steady", "render_device_seconds").size == 0
    assert len(values(samples, "steady", "render_device_span_seconds")) == len(values(samples, "steady", "render_seconds"))
