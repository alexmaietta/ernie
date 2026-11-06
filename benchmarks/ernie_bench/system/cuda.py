# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import ctypes
import threading
from contextlib import contextmanager
from functools import cache

HOST_ALLOC_PORTABLE = 0x01
HOST_ALLOC_DEVICE_MAP = 0x02
STREAM_NON_BLOCKING = 0x01
WAIT_VALUE_GEQ = 0x0
GATE_TIMEOUT_SECONDS = 2.0


class CudaUnavailable(Exception):
    pass


@cache
def driver() -> ctypes.CDLL:
    try:
        library = ctypes.CDLL("libcuda.so.1")
    except OSError as error:
        raise CudaUnavailable(f"no CUDA driver: {error}") from None
    result = library.cuInit(0)
    if result:
        raise CudaUnavailable(f"cuInit failed with {result}")
    return library


def call(name: str, *arguments) -> None:
    library = driver()
    result = getattr(library, name)(*arguments)
    if result:
        text = ctypes.c_char_p()
        library.cuGetErrorName(result, ctypes.byref(text))
        raise CudaUnavailable(f"{name} failed with {(text.value or b'?').decode()}")


def symbol(*names: str):
    for name in names:
        if hasattr(driver(), name):
            return name
    raise CudaUnavailable(f"the CUDA driver has none of {names}")


@contextmanager
def current(context: ctypes.c_void_p):
    call("cuCtxPushCurrent_v2", context)
    try:
        yield
    finally:
        call("cuCtxPopCurrent_v2", ctypes.byref(ctypes.c_void_p()))


class PrimaryContext:
    def __init__(self, ordinal: int = 0):
        self.device = ctypes.c_int()
        call("cuDeviceGet", ctypes.byref(self.device), ordinal)
        self.handle = ctypes.c_void_p()
        call("cuDevicePrimaryCtxRetain", ctypes.byref(self.handle), self.device)

    def close(self) -> None:
        if self.handle:
            call(symbol("cuDevicePrimaryCtxRelease_v2", "cuDevicePrimaryCtxRelease"), self.device)
            self.handle = ctypes.c_void_p()


class Stream:
    def __init__(self, ordinal: int = 0):
        self.context = PrimaryContext(ordinal)
        self.handle = ctypes.c_void_p()
        with current(self.context.handle):
            call("cuStreamCreate", ctypes.byref(self.handle), STREAM_NON_BLOCKING)

    def synchronize(self) -> None:
        call("cuStreamSynchronize", self.handle)

    def close(self) -> None:
        if self.handle:
            call("cuStreamSynchronize", self.handle)
            call("cuStreamDestroy_v2", self.handle)
            self.handle = ctypes.c_void_p()
        self.context.close()


class Timer:
    def __init__(self, stream: int):
        self.stream = ctypes.c_void_p(stream)
        self.context = ctypes.c_void_p()
        call("cuStreamGetCtx", self.stream, ctypes.byref(self.context))
        self.start, self.stop = ctypes.c_void_p(), ctypes.c_void_p()
        self.host = ctypes.c_void_p()
        self.device = ctypes.c_uint64()
        self.wait = symbol("cuStreamWaitValue32_v2", "cuStreamWaitValue32")
        self.gate = 0
        self.gated = True
        self.blocked = False
        self.watchdog = None
        with current(self.context):
            call("cuEventCreate", ctypes.byref(self.start), 0)
            call("cuEventCreate", ctypes.byref(self.stop), 0)
            call("cuMemHostAlloc", ctypes.byref(self.host), 4, HOST_ALLOC_PORTABLE | HOST_ALLOC_DEVICE_MAP)
            call("cuMemHostGetDevicePointer_v2", ctypes.byref(self.device), self.host, 0)
        self.flag = ctypes.c_uint32.from_address(self.host.value)
        self.flag.value = 0

    def begin(self, watched: bool = False) -> None:
        self.gate += 1
        with current(self.context):
            if self.gated:
                call(self.wait, self.stream, self.device, ctypes.c_uint32(self.gate), WAIT_VALUE_GEQ)
            call("cuEventRecord", self.start, self.stream)
        self.watchdog = threading.Timer(GATE_TIMEOUT_SECONDS, self.release) if watched and self.gated else None
        if self.watchdog:
            self.watchdog.start()

    def release(self) -> None:
        if self.flag.value < self.gate:
            self.blocked = True
            self.gated = False
            self.flag.value = self.gate

    def end(self) -> None:
        with current(self.context):
            call("cuEventRecord", self.stop, self.stream)
        if self.watchdog:
            self.watchdog.cancel()
        self.flag.value = self.gate

    def seconds(self) -> float:
        call("cuEventSynchronize", self.stop)
        milliseconds = ctypes.c_float()
        call("cuEventElapsedTime", ctypes.byref(milliseconds), self.start, self.stop)
        return milliseconds.value / 1e3

    def close(self) -> None:
        if self.watchdog:
            self.watchdog.cancel()
        self.flag.value = self.gate
        with current(self.context):
            call("cuStreamSynchronize", self.stream)
            call("cuEventDestroy_v2", self.start)
            call("cuEventDestroy_v2", self.stop)
            call("cuMemFreeHost", self.host)


@contextmanager
def profiled(ordinal: int = 0):
    context = PrimaryContext(ordinal)
    try:
        with current(context.handle):
            call("cuProfilerStart")
        yield
    finally:
        with current(context.handle):
            call("cuProfilerStop")
        context.close()
